#include "product/choice/choice_replay.h"

#include <array>
#include <cstdint>
#include <limits>
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

// An enumeration's value from a record: one of its `count` values, else the record is malformed.
template <class E>
E enum_of(const json& j, int count) {
    const int v = j.get<int>();
    if (v < 0 || v >= count) {
        throw std::invalid_argument("an enumeration value out of range: " + std::to_string(v));
    }
    return static_cast<E>(v);
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

template <class E>
std::function<std::uint64_t(E)> table_fn(const json& table, const char* what) {
    if (table.is_null()) { return {}; }
    auto values = std::make_shared<std::map<int, std::uint64_t>>();
    for (const auto& [k, v] : table.items()) { (*values)[std::stoi(k)] = unsigned_of<std::uint64_t>(v); }
    return [values, what](E e) -> std::uint64_t {
        const auto it = values->find(code(e));
        if (it == values->end()) {
            throw ReplayIncomplete(std::string("the record lacks the ") + what + " of storage " +
                                   std::to_string(code(e)));
        }
        return it->second;
    };
}

json model_json(const ModelFacts& m) {
    json linears = json::array();
    for (const LinearShape& l : m.linears) {
        linears.push_back({{"n", l.n}, {"k", l.k}, {"per_token", l.per_token}, {"expert", l.expert},
                           {"gated", l.gated}, {"format", l.format}});
    }
    json drafters = json::array();
    for (const DrafterOption& d : m.drafters) {
        json load = json::array();
        for (const auto& g : d.load) {
            load.push_back({{"streams", g.streams}, {"speedup", g.speedup}, {"depth", g.depth},
                            {"bounded", g.bounded}, {"measured", g.measured}});
        }
        drafters.push_back({{"label", d.label}, {"drafter", d.drafter}, {"speedup", d.speedup},
                            {"plain_from_streams", d.plain_from_streams}, {"basis", d.basis},
                            {"measured", d.measured}, {"load", std::move(load)}});
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
    };
}

ModelFacts model_of(const json& j) {
    ModelFacts m;
    m.family                   = j.at("family").get<std::string>();
    m.params                   = j.at("params");
    m.active_params            = j.at("active_params");
    m.experts                  = j.at("experts");
    m.experts_per_token        = j.at("experts_per_token");
    m.routed_expert_params     = j.at("routed_expert_params");
    m.input_table_params       = j.at("input_table_params");
    m.input_table_bytes        = j.at("input_table_bytes");
    m.layers                   = j.at("layers");
    m.hidden                   = j.at("hidden");
    m.prompt_tail_calls        = j.at("prompt_tail_calls");
    m.kv_bytes_per_token       = table_fn<KvCacheStorage>(j.at("kv_bytes_per_token"), "KV bytes");
    m.ring_bytes_per_token     = table_fn<KvCacheStorage>(j.at("ring_bytes_per_token"), "ring bytes");
    m.state_bytes              = table_fn<RecurrentStateStorage>(j.at("state_bytes"), "state bytes");
    m.window                   = j.at("window");
    m.full_attention_width     = j.at("full_attention_width");
    m.sliding_attention_width  = j.at("sliding_attention_width");
    m.full_layers              = j.at("full_layers");
    m.full_head_dim            = j.at("full_head_dim");
    m.full_kv_heads            = j.at("full_kv_heads");
    m.sliding_layers           = j.at("sliding_layers");
    m.sliding_head_dim         = j.at("sliding_head_dim");
    m.sliding_kv_heads         = j.at("sliding_kv_heads");
    m.full_attending_layers    = j.at("full_attending_layers");
    m.sliding_attending_layers = j.at("sliding_attending_layers");
    m.recurrent                = j.at("recurrent");
    m.fp8_ring                 = j.at("fp8_ring");
    for (const auto& l : j.at("linears")) {
        m.linears.push_back({l.at("n"), l.at("k"), l.at("per_token"), l.at("expert"), l.at("gated"),
                             l.at("format").get<std::string>()});
    }
    m.recurrent_state_elements = j.at("recurrent_state_elements");
    for (const auto& d : j.at("drafters")) {
        DrafterOption o;
        o.label              = d.at("label").get<std::string>();
        o.drafter            = d.at("drafter").get<std::string>();
        o.speedup            = d.at("speedup");
        o.plain_from_streams = d.at("plain_from_streams");
        o.basis              = d.at("basis").get<std::string>();
        o.measured           = d.at("measured");
        for (const auto& g : d.at("load")) {
            o.load.push_back({g.at("streams"), g.at("speedup"), g.at("depth"), g.at("bounded"),
                              g.at("measured")});
        }
        m.drafters.push_back(std::move(o));
    }
    m.default_drafter = j.at("default_drafter");
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
    c.major            = j.at("major");
    c.minor            = j.at("minor");
    c.sms              = j.at("sms");
    c.clock_hz         = j.at("clock_hz");
    c.dram_bytes_per_s = j.at("dram_bytes_per_s");
    c.l2_bytes         = j.at("l2_bytes");
    c.fp4              = j.at("fp4");
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
                {"rest_weights", code(q.rest_weights)}};
}

QuantOption quant_of(const json& j) {
    QuantOption q;
    q.label               = j.at("label").get<std::string>();
    q.files               = j.at("files").get<std::string>();
    q.bits_per_weight     = j.at("bits_per_weight");
    q.weight_bytes        = j.at("weight_bytes");
    q.routed_expert_bytes = j.at("routed_expert_bytes");
    q.supported           = j.at("supported");
    q.int8_route          = j.at("int8_route");
    q.nvfp4               = j.at("nvfp4");
    q.weights             = enum_of<WeightClass>(j.at("weights"), 4);
    q.expert_banks        = enum_of<ExpertBanks>(j.at("expert_banks"), 4);
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
        return j.get<T>();
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
    for (const auto& s : j.at("kv")) { a.kv.push_back(enum_of<KvCacheStorage>(s, static_cast<int>(kKvStorages.size()))); }
    for (const auto& s : j.at("activations")) {
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
    for (const auto& c : j.at("contexts")) { t.contexts.push_back(unsigned_of<std::uint32_t>(c)); }
    t.concurrency     = unsigned_of<std::uint32_t>(j.at("concurrency"));
    t.prompt          = unsigned_of<std::uint32_t>(j.at("prompt"));
    t.answer          = unsigned_of<std::uint32_t>(j.at("answer"));
    t.accuracy_weight = j.at("accuracy_weight");
    return t;
}

json gates_json(const Gates& g) {
    return json{{"accuracy", optional_json(g.accuracy)}, {"moe_chunk", g.moe_chunk}};
}

Gates gates_of(const json& j) {
    Gates g;
    g.accuracy  = optional_of<bool>(j.at("accuracy"));
    g.moe_chunk = j.at("moe_chunk");
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
    out.choice = choose(model, card, quant, allowed, pins, task, recording, gates);
    out.inputs = json{{"schema", kChoiceReplaySchema},
                      {"model", model_json(model)},
                      {"card", card_json(card)},
                      {"quant", quant_json(quant)},
                      {"allowed", allowed_json(allowed)},
                      {"pins", pins_json(pins)},
                      {"task", task_json(task)},
                      {"gates", gates_json(gates)},
                      {"own_context", std::move(own)}};
    return out;
}

Choice replay_choice(const json& inputs) {
    const auto schema = inputs.is_object() ? inputs.find("schema") : inputs.end();
    if (!inputs.is_object() || schema == inputs.end() || !schema->is_string() ||
        schema->get<std::string>() != kChoiceReplaySchema) {
        throw std::invalid_argument("a choice replay record of schema " +
                                    std::string(kChoiceReplaySchema) + " was expected");
    }
    try {
        const ModelFacts model = model_of(inputs.at("model"));
        auto own = std::make_shared<std::map<std::string, std::uint32_t>>();
        for (const auto& [k, v] : inputs.at("own_context").items()) {
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
                          {"kld", s.accuracy.kld},
                          {"variant", s.variant},
                          {"drafter", s.drafter}});
    }
    return json{{"scored", std::move(scored)},
                {"chosen", c.chosen ? json(*c.chosen) : json(nullptr)},
                {"context", c.context},
                {"by_worth", c.by_worth},
                {"chunk_by_speed", c.chunk_by_speed},
                {"drafter_by_worth", c.drafter_by_worth},
                {"resolution", c.resolution},
                {"why", c.why}};
}

} // namespace ninferno::product::choice
