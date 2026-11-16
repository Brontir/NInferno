#include "product/choice/choice_replay.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>
#include <map>
#include <memory>
#include <stdexcept>

namespace ninferno::product::choice {
namespace {

using json = nlohmann::ordered_json;

constexpr std::array kKvStorages = {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                    KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                    KvCacheStorage::Fp8KeyNvfp4Value};
constexpr std::array kStateStorages = {RecurrentStateStorage::FP32, RecurrentStateStorage::BF16,
                                       RecurrentStateStorage::FP16};

template <class E>
int code(E e) {
    return static_cast<int>(e);
}

// An unsigned count from a record: a non-negative integer within T, else the record is malformed
// (nlohmann would wrap -1 to T's maximum).
template <class T>
T unsigned_of(const json& j) {
    if (!j.is_number_integer() || (j.is_number_integer() && !j.is_number_unsigned() &&
                                   j.get<std::int64_t>() < 0)) {
        throw std::invalid_argument("a non-negative integer was expected: " + j.dump());
    }
    const std::uint64_t v = j.get<std::uint64_t>();
    if (v > std::numeric_limits<T>::max()) {
        throw std::invalid_argument("an integer out of range: " + j.dump());
    }
    return static_cast<T>(v);
}

// A record's list or table: nlohmann iterates a null as empty and a scalar as itself, so a
// corrupted field would replay as no entries, or one, instead of being refused.
const json& array_at(const json& j, const char* key) {
    const json& v = j.at(key);
    if (!v.is_array()) { throw std::invalid_argument(std::string("a list was expected: ") + key); }
    return v;
}
const json& object_at(const json& j, const char* key) {
    const json& v = j.at(key);
    if (!v.is_object()) { throw std::invalid_argument(std::string("a table was expected: ") + key); }
    return v;
}

// An enumeration's value from a record: one of its `count` values, else the record is malformed.
// (Checked before narrowing: nlohmann would turn 4294967296 into 0.)
template <class E>
E enum_of(const json& j, int count) {
    const auto v = unsigned_of<std::uint64_t>(j);
    if (v >= static_cast<std::uint64_t>(count)) {
        throw std::invalid_argument("an enumeration value out of range: " + j.dump());
    }
    return static_cast<E>(v);
}

// A field's value from a record, of the field's own type and within its range, else the record is
// malformed: nlohmann converts a number to a bool and wraps or narrows integers unchecked.
template <class T>
T value_of(const json& j) {
    if constexpr (std::is_same_v<T, bool>) {
        if (!j.is_boolean()) { throw std::invalid_argument("a boolean was expected: " + j.dump()); }
        return j.get<bool>();
    } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
        return unsigned_of<T>(j);
    } else if constexpr (std::is_integral_v<T>) {
        if (!j.is_number_integer() ||
            (j.is_number_unsigned() &&
             j.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))) {
            throw std::invalid_argument("an integer was expected: " + j.dump());
        }
        const std::int64_t v = j.get<std::int64_t>();
        if (v < std::numeric_limits<T>::min() || v > std::numeric_limits<T>::max()) {
            throw std::invalid_argument("an integer out of range: " + j.dump());
        }
        return static_cast<T>(v);
    } else if constexpr (std::is_floating_point_v<T>) {
        if (!j.is_number()) { throw std::invalid_argument("a number was expected: " + j.dump()); }
        return j.get<T>();
    } else {
        return j.get<T>();
    }
}

template <class T>
void read(T& field, const json& j) {
    field = value_of<T>(j);
}

// A candidate's key in the own-context table: its axes in a fixed order.
std::string candidate_key(const Candidate& c) {
    return std::to_string(code(c.kv)) + "," + std::to_string(code(c.ring)) + "," +
           std::to_string(code(c.state)) + "," + std::to_string(code(c.activations)) + "," +
           std::to_string(c.chunk) + "," + std::to_string(c.drafter);
}

json candidate_json(const Candidate& c) {
    return json{{"kv", code(c.kv)},           {"ring", code(c.ring)},   {"state", code(c.state)},
                {"activations", code(c.activations)}, {"chunk", c.chunk}, {"drafter", c.drafter}};
}

template <class Fn, class Storages>
json storage_table(const Fn& fn, const Storages& storages) {
    if (!fn) { return nullptr; }
    json out = json::object();
    for (const auto s : storages) { out[std::to_string(code(s))] = fn(s); }
    return out;
}

// A storage table's key: a storage code as table_json writes it (decimal, no sign, no leading
// zero), else the record is malformed (std::stoi would read "0junk" as 0).
int storage_key(const std::string& key) {
    const bool canonical = !key.empty() && key.size() <= 3 &&
                           std::all_of(key.begin(), key.end(),
                                       [](char c) { return c >= '0' && c <= '9'; }) &&
                           (key.size() == 1 || key.front() != '0');
    if (!canonical) { throw std::invalid_argument("a storage key that is no storage code: " + key); }
    return std::stoi(key);
}

// Its keys are the codes of `storages`, the ones storage_table writes: any other is malformed.
template <class E, std::size_t N>
std::function<std::uint64_t(E)> table_fn(const json& table, const char* what,
                                         const std::array<E, N>& storages) {
    if (table.is_null()) { return {}; }
    if (!table.is_object()) { throw std::invalid_argument(std::string("a ") + what + " table that is no object"); }
    auto values = std::make_shared<std::map<int, std::uint64_t>>();
    for (const auto& [k, v] : table.items()) {
        const int key = storage_key(k);
        if (std::none_of(storages.begin(), storages.end(), [&](E s) { return code(s) == key; })) {
            throw std::invalid_argument(std::string("a ") + what + " table key of no storage: " + k);
        }
        (*values)[key] = unsigned_of<std::uint64_t>(v);
    }
    return [values, what](E e) -> std::uint64_t {
        const auto it = values->find(code(e));
        if (it == values->end()) {
            throw ReplayIncomplete(std::string("the record lacks the ") + what + " of storage " +
                                   std::to_string(code(e)));
        }
        return it->second;
    };
}

json linear_json(const LinearShape& l) {
    return {{"n", l.n},           {"k", l.k},         {"per_token", l.per_token},
            {"expert", l.expert}, {"gated", l.gated}, {"format", l.format}};
}

LinearShape linear_of(const json& j) {
    LinearShape l;
    read(l.n, j.at("n"));
    read(l.k, j.at("k"));
    read(l.per_token, j.at("per_token"));
    read(l.expert, j.at("expert"));
    read(l.gated, j.at("gated"));
    read(l.format, j.at("format"));
    return l;
}

// The full arguments, not a linear's index: selected_kernel can fill in a missing format from
// the quantization, and the callback's contract includes every LinearShape field (#931).
std::string prefill_key(const LinearShape& l, std::uint32_t tokens, ActivationMode activations) {
    if (!std::isfinite(l.n) || !std::isfinite(l.k) || !std::isfinite(l.per_token)) {
        throw std::invalid_argument(
            "a prefill route query must have finite dimensions and frequency");
    }
    return json{{"linear", linear_json(l)}, {"tokens", tokens}, {"activations", code(activations)}}
        .dump();
}

std::string moe_key(std::uint32_t rows, ActivationMode activations) {
    return json{{"rows", rows}, {"activations", code(activations)}}.dump();
}

json moe_answer(const std::vector<MoeStepRoute>& routes) {
    json out = json::array();
    for (const auto& r : routes) {
        if (!std::isfinite(r.share)) {
            throw std::invalid_argument("a MoE route share must be finite");
        }
        out.push_back({{"gate_up", code(r.gate_up)}, {"down", code(r.down)}, {"share", r.share}});
    }
    return out;
}

using RouteAnswers = std::map<std::string, json>;

void record_answer(RouteAnswers& answers, const std::string& key, const json& answer) {
    const auto [it, inserted] = answers.emplace(key, answer);
    if (!inserted && it->second != answer) {
        throw std::invalid_argument("a route query changed its answer while recording: " + key);
    }
}

json routes_json(bool present, const RouteAnswers& answers) {
    if (!present) { return nullptr; }
    json out = json::array();
    for (const auto& [key, answer] : answers) {
        out.push_back({{"query", json::parse(key)}, {"answer", answer}});
    }
    return out;
}

// null: no callback. []: a callback which was never asked. Unknown/empty answers are entries,
// not absent callbacks. No lookup may silently fall back to today's selector or assumed route.
decltype(ModelFacts::prefill_routes) prefill_fn(const json& table) {
    if (table.is_null()) { return {}; }
    if (!table.is_array()) { throw std::invalid_argument("prefill_routes must be a list or null"); }
    auto answers = std::make_shared<std::map<std::string, PrefillKernel>>();
    for (const auto& entry : table) {
        const auto& q     = object_at(entry, "query");
        const auto key    = prefill_key(linear_of(object_at(q, "linear")),
                                        unsigned_of<std::uint32_t>(q.at("tokens")),
                                        enum_of<ActivationMode>(q.at("activations"), 3));
        const auto answer = enum_of<PrefillKernel>(entry.at("answer"), 5);
        if (!answers->emplace(key, answer).second) {
            throw std::invalid_argument("duplicate prefill route query: " + key);
        }
    }
    return [answers](const LinearShape& l, std::uint32_t tokens, ActivationMode activations) {
        const auto key = prefill_key(l, tokens, activations);
        const auto it  = answers->find(key);
        if (it == answers->end()) {
            throw ReplayIncomplete("the record lacks the prefill route: " + key);
        }
        return it->second;
    };
}

decltype(ModelFacts::moe_routes) moe_fn(const json& table) {
    if (table.is_null()) { return {}; }
    if (!table.is_array()) { throw std::invalid_argument("moe_routes must be a list or null"); }
    auto answers = std::make_shared<std::map<std::string, std::vector<MoeStepRoute>>>();
    for (const auto& entry : table) {
        const auto& q  = object_at(entry, "query");
        const auto key = moe_key(unsigned_of<std::uint32_t>(q.at("rows")),
                                 enum_of<ActivationMode>(q.at("activations"), 3));
        std::vector<MoeStepRoute> answer;
        for (const auto& r : array_at(entry, "answer")) {
            const auto share = value_of<double>(r.at("share"));
            if (!std::isfinite(share)) {
                throw std::invalid_argument("a MoE route share must be finite");
            }
            answer.push_back({enum_of<ExpertKernel>(r.at("gate_up"), 5),
                              enum_of<ExpertKernel>(r.at("down"), 5), share});
        }
        if (!answers->emplace(key, std::move(answer)).second) {
            throw std::invalid_argument("duplicate MoE route query: " + key);
        }
    }
    return [answers](std::uint32_t rows, ActivationMode activations) {
        const auto key = moe_key(rows, activations);
        const auto it  = answers->find(key);
        if (it == answers->end()) {
            throw ReplayIncomplete("the record lacks the MoE route: " + key);
        }
        return it->second;
    };
}

json model_json(const ModelFacts& m) {
    json linears = json::array();
    for (const LinearShape& l : m.linears) { linears.push_back(linear_json(l)); }
    json drafters = json::array();
    for (const DrafterOption& d : m.drafters) {
        json load = json::array();
        for (const auto& g : d.load) {
            load.push_back({{"streams", g.streams}, {"speedup", g.speedup}, {"depth", g.depth},
                            {"bounded", g.bounded}, {"measured", g.measured}});
        }
        json drafter = {{"label", d.label},       {"drafter", d.drafter},
                        {"speedup", d.speedup},   {"plain_from_streams", d.plain_from_streams},
                        {"basis", d.basis},       {"measured", d.measured},
                        {"load", std::move(load)}};
        // Written only when priced (#303): a record without it replays as before.
        if (!d.joint.empty()) {
            json joint = json::array();
            for (const auto& j : d.joint) {
                joint.push_back({{"streams", j.streams}, {"scale", j.scale}});
            }
            drafter["joint"] = std::move(joint);
        }
        drafters.push_back(std::move(drafter));
    }
    return json{
        {"family", m.family},
        {"params", m.params},
        {"active_params", m.active_params},
        {"experts", m.experts},
        {"experts_per_token", m.experts_per_token},
        {"routed_expert_params", m.routed_expert_params},
        {"input_table_params", m.input_table_params},
        {"input_table_bytes", m.input_table_bytes},
        {"layers", m.layers},
        {"hidden", m.hidden},
        {"prompt_tail_calls", m.prompt_tail_calls},
        {"kv_bytes_per_token", storage_table(m.kv_bytes_per_token, kKvStorages)},
        {"ring_bytes_per_token", storage_table(m.ring_bytes_per_token, kKvStorages)},
        {"state_bytes", storage_table(m.state_bytes, kStateStorages)},
        {"window", m.window},
        {"full_attention_width", m.full_attention_width},
        {"sliding_attention_width", m.sliding_attention_width},
        {"full_layers", m.full_layers},
        {"full_head_dim", m.full_head_dim},
        {"full_kv_heads", m.full_kv_heads},
        {"sliding_layers", m.sliding_layers},
        {"sliding_head_dim", m.sliding_head_dim},
        {"sliding_kv_heads", m.sliding_kv_heads},
        {"full_attending_layers", m.full_attending_layers},
        {"sliding_attending_layers", m.sliding_attending_layers},
        {"recurrent", m.recurrent},
        {"fp8_ring", m.fp8_ring},
        {"linears", std::move(linears)},
        {"recurrent_state_elements", m.recurrent_state_elements},
        {"drafters", std::move(drafters)},
        {"default_drafter", m.default_drafter},
        {"moe_route_unqueried", m.moe_route_unqueried},
    };
}

ModelFacts model_of(const json& j) {
    ModelFacts m;
    read(m.family, j.at("family"));
    read(m.params, j.at("params"));
    read(m.active_params, j.at("active_params"));
    read(m.experts, j.at("experts"));
    read(m.experts_per_token, j.at("experts_per_token"));
    read(m.routed_expert_params, j.at("routed_expert_params"));
    read(m.input_table_params, j.at("input_table_params"));
    read(m.input_table_bytes, j.at("input_table_bytes"));
    read(m.layers, j.at("layers"));
    read(m.hidden, j.at("hidden"));
    read(m.prompt_tail_calls, j.at("prompt_tail_calls"));
    m.kv_bytes_per_token       = table_fn(j.at("kv_bytes_per_token"), "KV bytes", kKvStorages);
    m.ring_bytes_per_token     = table_fn(j.at("ring_bytes_per_token"), "ring bytes", kKvStorages);
    m.state_bytes              = table_fn(j.at("state_bytes"), "state bytes", kStateStorages);
    read(m.window, j.at("window"));
    read(m.full_attention_width, j.at("full_attention_width"));
    read(m.sliding_attention_width, j.at("sliding_attention_width"));
    read(m.full_layers, j.at("full_layers"));
    read(m.full_head_dim, j.at("full_head_dim"));
    read(m.full_kv_heads, j.at("full_kv_heads"));
    read(m.sliding_layers, j.at("sliding_layers"));
    read(m.sliding_head_dim, j.at("sliding_head_dim"));
    read(m.sliding_kv_heads, j.at("sliding_kv_heads"));
    read(m.full_attending_layers, j.at("full_attending_layers"));
    read(m.sliding_attending_layers, j.at("sliding_attending_layers"));
    read(m.recurrent, j.at("recurrent"));
    read(m.fp8_ring, j.at("fp8_ring"));
    for (const auto& l : array_at(j, "linears")) { m.linears.push_back(linear_of(l)); }
    read(m.recurrent_state_elements, j.at("recurrent_state_elements"));
    for (const auto& d : array_at(j, "drafters")) {
        DrafterOption o;
        read(o.label, d.at("label"));
        read(o.drafter, d.at("drafter"));
        read(o.speedup, d.at("speedup"));
        read(o.plain_from_streams, d.at("plain_from_streams"));
        read(o.basis, d.at("basis"));
        read(o.measured, d.at("measured"));
        for (const auto& g : array_at(d, "load")) {
            DrafterOption::LoadGain gain;
            read(gain.streams, g.at("streams"));
            read(gain.speedup, g.at("speedup"));
            read(gain.depth, g.at("depth"));
            read(gain.bounded, g.at("bounded"));
            read(gain.measured, g.at("measured"));
            o.load.push_back(gain);
        }
        if (d.contains("joint")) {
            for (const auto& j : array_at(d, "joint")) {
                DrafterOption::JointScale scale;
                read(scale.streams, j.at("streams"));
                read(scale.scale, j.at("scale"));
                // Rejected, not sorted: price_source_at reads the table in ascending order of
                // streams (a stored record is written ascending), and a scale is a positive ratio.
                // A joint verify is of two streams or more: one request's verify is the base the
                // scale is taken against, so an entry below two streams is not a price.
                if (!(scale.scale > 0) || scale.streams < 2 ||
                    (!o.joint.empty() && o.joint.back().streams >= scale.streams)) {
                    throw std::invalid_argument(
                        "a joint verify scale must be positive, of two streams or more, streams "
                        "ascending: " +
                        j.dump());
                }
                o.joint.push_back(scale);
            }
        }
        m.drafters.push_back(std::move(o));
    }
    read(m.default_drafter, j.at("default_drafter"));
    read(m.moe_route_unqueried, j.at("moe_route_unqueried"));
    m.prefill_routes = prefill_fn(j.at("prefill_routes"));
    m.moe_routes     = moe_fn(j.at("moe_routes"));
    return m;
}

json card_json(const CardFacts& c) {
    return json{{"major", c.major},       {"minor", c.minor},
                {"sms", c.sms},           {"clock_hz", c.clock_hz},
                {"dram_bytes_per_s", c.dram_bytes_per_s}, {"l2_bytes", c.l2_bytes},
                {"fp4", c.fp4}};
}

CardFacts card_of(const json& j) {
    CardFacts c;
    read(c.major, j.at("major"));
    read(c.minor, j.at("minor"));
    read(c.sms, j.at("sms"));
    read(c.clock_hz, j.at("clock_hz"));
    read(c.dram_bytes_per_s, j.at("dram_bytes_per_s"));
    read(c.l2_bytes, j.at("l2_bytes"));
    read(c.fp4, j.at("fp4"));
    return c;
}

json quant_json(const QuantOption& q) {
    return json{{"label", q.label},
                {"files", q.files},
                {"bits_per_weight", q.bits_per_weight},
                {"weight_bytes", q.weight_bytes},
                {"routed_expert_bytes", q.routed_expert_bytes},
                {"supported", q.supported},
                {"int8_route", q.int8_route},
                {"nvfp4", q.nvfp4},
                {"weights", code(q.weights)},
                {"expert_banks", code(q.expert_banks)},
                // null: none given, expert_banks stands under int8 too
                {"expert_banks_int8",
                 q.expert_banks_int8 ? json(code(*q.expert_banks_int8)) : json(nullptr)},
                {"expert_banks_assumed", q.expert_banks_assumed},
                {"rest_weights", code(q.rest_weights)}};
}

QuantOption quant_of(const json& j) {
    QuantOption q;
    read(q.label, j.at("label"));
    read(q.files, j.at("files"));
    read(q.bits_per_weight, j.at("bits_per_weight"));
    read(q.weight_bytes, j.at("weight_bytes"));
    read(q.routed_expert_bytes, j.at("routed_expert_bytes"));
    read(q.supported, j.at("supported"));
    read(q.int8_route, j.at("int8_route"));
    read(q.nvfp4, j.at("nvfp4"));
    q.weights             = enum_of<WeightClass>(j.at("weights"), 4);
    q.expert_banks        = enum_of<ExpertBanks>(j.at("expert_banks"), 4);
    if (const json& int8 = j.at("expert_banks_int8"); !int8.is_null()) {
        q.expert_banks_int8 = enum_of<ExpertBanks>(int8, 4);
    }
    read(q.expert_banks_assumed, j.at("expert_banks_assumed"));
    q.rest_weights        = enum_of<WeightClass>(j.at("rest_weights"), 4);
    return q;
}

template <class T>
json optional_json(const std::optional<T>& v) {
    if (!v) { return nullptr; }
    if constexpr (std::is_enum_v<T>) {
        return code(*v);
    } else {
        return *v;
    }
}

template <class T>
std::optional<T> optional_of(const json& j) {
    if (j.is_null()) { return std::nullopt; }
    if constexpr (std::is_same_v<T, KvCacheStorage>) {
        return enum_of<T>(j, static_cast<int>(kKvStorages.size()));
    } else if constexpr (std::is_same_v<T, RecurrentStateStorage>) {
        return enum_of<T>(j, static_cast<int>(kStateStorages.size()));
    } else if constexpr (std::is_enum_v<T>) {
        return enum_of<T>(j, 3); // ActivationMode
    } else {
        return value_of<T>(j);
    }
}

json pins_json(const Pins& p) {
    return json{{"kv", optional_json(p.kv)},
                {"ring", optional_json(p.ring)},
                {"state", optional_json(p.state)},
                {"activations", optional_json(p.activations)},
                {"chunk", optional_json(p.chunk)},
                {"drafter", optional_json(p.drafter)}};
}

Pins pins_of(const json& j) {
    Pins p;
    p.kv          = optional_of<KvCacheStorage>(j.at("kv"));
    p.ring        = optional_of<KvCacheStorage>(j.at("ring"));
    p.state       = optional_of<RecurrentStateStorage>(j.at("state"));
    p.activations = optional_of<ActivationMode>(j.at("activations"));
    p.chunk       = optional_of<std::uint32_t>(j.at("chunk"));
    p.drafter     = optional_of<std::size_t>(j.at("drafter"));
    return p;
}

json allowed_json(const Allowed& a) {
    json kv = json::array(), activations = json::array();
    for (const auto s : a.kv) { kv.push_back(code(s)); }
    for (const auto s : a.activations) { activations.push_back(code(s)); }
    return json{{"kv", std::move(kv)}, {"activations", std::move(activations)},
                {"automatic_kv", code(a.automatic_kv)}};
}

Allowed allowed_of(const json& j) {
    Allowed a;
    for (const auto& s : array_at(j, "kv")) { a.kv.push_back(enum_of<KvCacheStorage>(s, static_cast<int>(kKvStorages.size()))); }
    for (const auto& s : array_at(j, "activations")) {
        a.activations.push_back(enum_of<ActivationMode>(s, 3));
    }
    a.automatic_kv = enum_of<KvCacheStorage>(j.at("automatic_kv"), static_cast<int>(kKvStorages.size()));
    return a;
}

json task_json(const Task& t) {
    return json{{"contexts", t.contexts},
                {"concurrency", t.concurrency},
                {"prompt", t.prompt},
                {"answer", t.answer},
                {"accuracy_weight", t.accuracy_weight}};
}

Task task_of(const json& j) {
    Task t;
    t.contexts.clear();
    for (const auto& c : array_at(j, "contexts")) { t.contexts.push_back(unsigned_of<std::uint32_t>(c)); }
    t.concurrency     = unsigned_of<std::uint32_t>(j.at("concurrency"));
    t.prompt          = unsigned_of<std::uint32_t>(j.at("prompt"));
    t.answer          = unsigned_of<std::uint32_t>(j.at("answer"));
    read(t.accuracy_weight, j.at("accuracy_weight"));
    return t;
}

json gates_json(const Gates& g) {
    return json{{"accuracy", optional_json(g.accuracy)},
                {"moe_chunk", g.moe_chunk},
                {"typed_resolution", g.typed_resolution}};
}

Gates gates_of(const json& j) {
    Gates g;
    g.accuracy  = optional_of<bool>(j.at("accuracy"));
    read(g.moe_chunk, j.at("moe_chunk"));
    read(g.typed_resolution, j.at("typed_resolution"));
    return g;
}

} // namespace

ChoiceRecord record_choice(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                           const Allowed& allowed, const Pins& pins, const Task& task,
                           const OwnContextFn& own_context, const Gates& gates) {
    json own = json::object();
    const OwnContextFn recording = [&](const Candidate& c) {
        const std::uint32_t tokens = own_context ? own_context(c) : 0;
        own[candidate_key(c)]      = tokens;
        return tokens;
    };
    ChoiceRecord out;
    ModelFacts recorded = model;
    RouteAnswers prefill, moe;
    if (model.prefill_routes) {
        recorded.prefill_routes = [&](const LinearShape& l, std::uint32_t tokens,
                                      ActivationMode activations) {
            const auto answer = model.prefill_routes(l, tokens, activations);
            record_answer(prefill, prefill_key(l, tokens, activations), code(answer));
            return answer;
        };
    }
    if (model.moe_routes) {
        recorded.moe_routes = [&](std::uint32_t rows, ActivationMode activations) {
            const auto answer = model.moe_routes(rows, activations);
            record_answer(moe, moe_key(rows, activations), moe_answer(answer));
            return answer;
        };
    }
    out.choice       = choose(recorded, card, quant, allowed, pins, task, recording, gates);
    json model_input = model_json(model);
    model_input["prefill_routes"] = routes_json(bool(model.prefill_routes), prefill);
    model_input["moe_routes"]     = routes_json(bool(model.moe_routes), moe);
    out.inputs = json{{"schema", kChoiceReplaySchema},    {"model", std::move(model_input)},
                      {"card", card_json(card)},          {"quant", quant_json(quant)},
                      {"allowed", allowed_json(allowed)}, {"pins", pins_json(pins)},
                      {"task", task_json(task)},          {"gates", gates_json(gates)},
                      {"own_context", std::move(own)}};
    return out;
}

Choice replay_choice(const json& inputs) {
    const auto schema = inputs.is_object() ? inputs.find("schema") : inputs.end();
    if (!inputs.is_object() || schema == inputs.end() || !schema->is_string() ||
        schema->get<std::string>() != kChoiceReplaySchema) {
        const std::string found =
            inputs.is_object() && schema != inputs.end() && schema->is_string()
                ? schema->get<std::string>()
                : "none";
        throw std::invalid_argument("a choice replay record of schema " +
                                    std::string(kChoiceReplaySchema) + " was expected, got " +
                                    found +
                                    ": records of an older schema are not replayed, "
                                    "record the choice again");
    }
    try {
        const ModelFacts model = model_of(inputs.at("model"));
        auto own = std::make_shared<std::map<std::string, std::uint32_t>>();
        for (const auto& [k, v] : object_at(inputs, "own_context").items()) {
            (*own)[k] = unsigned_of<std::uint32_t>(v);
        }
        const OwnContextFn own_context = [own](const Candidate& c) -> std::uint32_t {
            const auto it = own->find(candidate_key(c));
            if (it == own->end()) {
                throw ReplayIncomplete("the record lacks the own context of candidate " +
                                       candidate_key(c));
            }
            return it->second;
        };
        return choose(model, card_of(inputs.at("card")), quant_of(inputs.at("quant")),
                      allowed_of(inputs.at("allowed")), pins_of(inputs.at("pins")),
                      task_of(inputs.at("task")), own_context, gates_of(inputs.at("gates")));
    } catch (const nlohmann::json::exception& e) {
        throw std::invalid_argument(std::string("a malformed choice replay record: ") + e.what());
    } catch (const std::out_of_range& e) {
        throw std::invalid_argument(std::string("a malformed choice replay record: ") + e.what());
    }
}

json choice_summary(const Choice& c) {
    json scored = json::array();
    for (const Scored& s : c.scored) {
        scored.push_back({{"candidate", candidate_json(s.candidate)},
                          {"own_context", s.own_context},
                          {"fits", s.fits},
                          {"worth", s.worth},
                          {"seconds", s.seconds},
                          {"decision_seconds", s.decision_seconds},
                          {"decision_basis", std::string(speed_basis_name(s.decision_basis))},
                          {"weighed_at", s.weighed_at},
                          {"kld", s.accuracy.kld},
                          {"variant", s.variant},
                          {"drafter", s.drafter}});
    }
    json kv_not_decided = json::array();
    for (const KvCacheStorage kv : c.kv_not_decided) {
        kv_not_decided.push_back(static_cast<int>(kv));
    }
    return json{{"scored", std::move(scored)},
                {"chosen", c.chosen ? json(*c.chosen) : json(nullptr)},
                {"context", c.context},
                {"by_worth", c.by_worth},
                {"activations_by_worth", c.activations_by_worth},
                {"kv_not_decided", kv_not_decided},
                {"chunk_by_speed", c.chunk_by_speed},
                {"chunk_basis", std::string(chunk_basis_name(c.chunk_basis))},
                {"drafter_by_worth", c.drafter_by_worth},
                {"resolution", c.resolution},
                {"resolution_basis", std::string(resolution_basis_name(c.resolution_basis))},
                {"resolution_decides", c.resolution_decides},
                {"resolution_floor", c.resolution_floor},
                {"why", c.why},
                {"why_code", c.why_code},
                {"gate_code", c.gate_code}};
}

} // namespace ninferno::product::choice
