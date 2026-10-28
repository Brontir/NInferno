#include "convert/checks.h"
#include "convert/drafters.h"

#include "core/utf8.h"
#include "core/weight.h"
#include "spec/draft_head.h"
#include "spec/fingerprint.h"
#include "spec/round_cost.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ninferno::convert {

// ---------------------------------------------------------------------------------------------
// config_checks.py

bool present(const OJson* value) { return value && !value->is_null(); }

std::uint64_t positive(const OJson* value, const std::string& name) {
    if (!value || !value->is_number_integer() ||
        (value->is_number_integer() && !value->is_number_unsigned() &&
         value->get<std::int64_t>() <= 0) ||
        (value->is_number_unsigned() && value->get<std::uint64_t>() == 0)) {
        throw std::runtime_error(name + ": expected a positive integer");
    }
    return value->get<std::uint64_t>();
}

// A positive finite value representable as FP32; returns the FP32 value as a Python float.
double f32(const OJson* value, const std::string& name) {
    if (!value || !value->is_number()) {
        throw std::runtime_error(name + ": expected a real value");
    }
    const double x = value->get<double>();
    const auto y   = static_cast<float>(x);
    if (std::isinf(y) && !std::isinf(x)) {
        throw std::runtime_error(name + ": not representable as F32");
    }
    if (!std::isfinite(y) || y <= 0.0F) {
        throw std::runtime_error(name + ": expected positive finite F32");
    }
    return static_cast<double>(y);
}

double f32_value(const OJson* value, double fallback, const std::string& name) {
    if (!value) {
        const OJson f = fallback;
        return f32(&f, name);
    }
    return f32(value, name);
}

// Python `a == b` for JSON scalars: bools are integers, ints and floats compare by value.
bool py_equal(const OJson& a, const OJson& b) {
    const auto numeric   = [](const OJson& v) { return v.is_number() || v.is_boolean(); };
    const auto as_double = [](const OJson& v) {
        return v.is_boolean() ? (v.get<bool>() ? 1.0 : 0.0) : v.get<double>();
    };
    if (numeric(a) && numeric(b)) {
        if (a.is_number_float() || b.is_number_float()) { return as_double(a) == as_double(b); }
        const auto as_int = [](const OJson& v) -> long double {
            if (v.is_boolean()) { return v.get<bool>() ? 1 : 0; }
            if (v.is_number_unsigned()) { return static_cast<long double>(v.get<std::uint64_t>()); }
            return static_cast<long double>(v.get<std::int64_t>());
        };
        return as_int(a) == as_int(b);
    }
    return a == b;
}

void fixed(const OJson& config, std::string_view key, const OJson& value,
           const std::string& label) {
    const auto* actual = member(config, key);
    if (!actual) { return; }
    if (!py_equal(*actual, value) || (value.is_boolean() && !actual->is_boolean())) {
        throw std::runtime_error(label + "." + std::string(key) + ": expected " + value.dump() +
                                 ", got " + actual->dump());
    }
}

// ---------------------------------------------------------------------------------------------
// resources.py

constexpr std::string_view kTextResources[]   = {"tokenizer.json", "tokenizer_config.json",
                                                 "chat_template.jinja", "generation_config.json"};
constexpr std::string_view kVisionResources[] = {"preprocessor_config.json",
                                                 "video_preprocessor_config.json"};

namespace {

// An added token past the embedding table is outside the domain (`beyond`, id -> content): the
// multimodal tokenizer Gemma 3 1B keeps defines <image_soft_token> at 262144 over a 262144-row
// table. No id the model reads or writes names it, and a session refuses the id should text encode
// to it (transformers' embedding raises on it alike). The vocabulary proper must fit the table.
std::pair<std::uint64_t, std::vector<std::int64_t>>
token_domain(const Json& tokenizer, const Json& config, std::uint64_t vocab_size,
             std::map<std::uint64_t, std::string>& beyond) {
    std::vector<const std::string*> ids(vocab_size, nullptr);
    std::vector<signed char> special(vocab_size, -1);
    const auto add_text = [&](const Json* index, const std::string* content,
                              bool added) -> std::optional<std::uint64_t> {
        if (!index || !index->is_number_integer() || !content ||
            (!index->is_number_unsigned() && index->get<std::int64_t>() < 0) ||
            (!added && index->get<std::uint64_t>() >= vocab_size)) {
            throw std::runtime_error("invalid tokenizer entry id=" +
                                     (index ? index->dump() : std::string("None")));
        }
        const auto id = index->get<std::uint64_t>();
        if (id >= vocab_size) {
            const auto [it, fresh] = beyond.emplace(id, *content);
            if (!fresh && it->second != *content) {
                throw std::runtime_error("tokenizer resources disagree about token " +
                                         std::to_string(id));
            }
            return std::nullopt;
        }
        if (ids[id] && *ids[id] != *content) {
            throw std::runtime_error("tokenizer resources disagree about token " +
                                     std::to_string(id));
        }
        ids[id] = content;
        return id;
    };
    const auto add = [&](const Json* index, const Json* content) {
        return add_text(index,
                        content && content->is_string() ? &content->get_ref<const std::string&>()
                                                        : nullptr,
                        true);
    };
    const auto add_special = [&](std::uint64_t index, const Json& token) {
        const auto* flag_value = member(token, "special");
        bool flag              = false;
        if (flag_value) {
            if (!flag_value->is_boolean()) {
                throw std::runtime_error("token " + std::to_string(index) +
                                         ": special must be boolean");
            }
            flag = flag_value->get<bool>();
        }
        if (special[index] >= 0 && special[index] != (flag ? 1 : 0)) {
            throw std::runtime_error("tokenizer resources disagree about token " +
                                     std::to_string(index) + " special flag");
        }
        special[index] = flag ? 1 : 0;
    };
    const auto* model = member(tokenizer, "model");
    const auto* vocab = model ? member(*model, "vocab") : nullptr;
    if (!vocab || !vocab->is_object()) {
        throw std::runtime_error("tokenizer model must provide a vocabulary mapping");
    }
    for (auto it = vocab->begin(); it != vocab->end(); ++it) {
        add_text(&it.value(), &it.key(), false);
    }
    if (const auto* added = member(tokenizer, "added_tokens")) {
        for (const auto& token : *added) {
            if (const auto id = add(&token.at("id"), &token.at("content"))) {
                add_special(*id, token);
            }
        }
    }
    if (const auto* decoder = member(config, "added_tokens_decoder")) {
        for (const auto& [raw_id, token] : decoder->items()) {
            const Json index = static_cast<std::int64_t>(std::stoll(raw_id));
            if (const auto id = add(&index, &token.at("content"))) { add_special(*id, token); }
        }
    }
    std::uint64_t count = 0;
    std::uint64_t last  = 0;
    for (std::uint64_t i = 0; i < vocab_size; ++i) {
        if (ids[i]) {
            ++count;
            last = i;
        }
    }
    if (!count || count != last + 1) {
        throw std::runtime_error("this tokenizer requires a contiguous public token ID domain");
    }
    std::vector<std::int64_t> specials;
    for (std::uint64_t i = 0; i < vocab_size; ++i) {
        if (special[i] == 1) { specials.push_back(static_cast<std::int64_t>(i)); }
    }
    return {count, specials};
}

} // namespace

ResourceFallbacks resource_fallbacks(const Checkpoint& checkpoint) {
    ResourceFallbacks out = checkpoint.rebuilt_resources();
    const auto root       = checkpoint.root();
    // A chat template kept in tokenizer_config.json, the layout before transformers wrote
    // chat_template.jinja (Qwen3, Llama 3.1, ...): its "chat_template" string, or the "default"
    // entry (else the first) of a list of named templates.
    if (!out.contains("chat_template.jinja") && !std::filesystem::exists(root / "chat_template.jinja") &&
        std::filesystem::exists(root / "tokenizer_config.json")) {
        const auto tokenizer = Json::parse(read_file(root / "tokenizer_config.json"));
        if (const auto* chat = member(tokenizer, "chat_template")) {
            if (chat->is_string()) {
                out["chat_template.jinja"] = chat->get<std::string>();
            } else if (chat->is_array() && !chat->empty()) {
                const Json* chosen = &(*chat)[0];
                for (const auto& item : *chat) {
                    if (item.value("name", std::string()) == "default") { chosen = &item; }
                }
                if (chosen->contains("template") && (*chosen)["template"].is_string()) {
                    out["chat_template.jinja"] = (*chosen)["template"].get<std::string>();
                }
            }
        }
    }
    if (out.contains("generation_config.json") ||
        std::filesystem::exists(root / "generation_config.json")) {
        return out;
    }
    // End-of-sequence ids: config.json's (top level or text_config), then tokenizer_config.json's
    // eos_token looked up among its added tokens.
    std::vector<std::int64_t> stops;
    const auto add = [&](const OJson* value) {
        if (!value) { return; }
        const auto push = [&](const OJson& id) {
            if (id.is_number_integer() &&
                std::find(stops.begin(), stops.end(), id.get<std::int64_t>()) == stops.end()) {
                stops.push_back(id.get<std::int64_t>());
            }
        };
        if (value->is_array()) {
            for (const auto& id : *value) { push(id); }
        } else {
            push(*value);
        }
    };
    const auto& config = checkpoint.config();
    add(member(config, "eos_token_id"));
    if (const auto* text = member(config, "text_config")) { add(member(*text, "eos_token_id")); }
    std::optional<std::int64_t> bos, pad;
    if (std::filesystem::exists(root / "tokenizer_config.json")) {
        const auto tokenizer = Json::parse(read_file(root / "tokenizer_config.json"));
        // transformers 5 keeps the added tokens in tokenizer.json only (no added_tokens_decoder).
        std::optional<Json> tokenizer_json;
        const auto lookup = [&](const char* key) -> std::optional<std::int64_t> {
            const auto* name = member(tokenizer, key);
            if (!name || !name->is_string()) { return std::nullopt; }
            if (const auto* decoder = member(tokenizer, "added_tokens_decoder")) {
                for (const auto& [id, token] : decoder->items()) {
                    if (token.value("content", std::string()) == name->get<std::string>()) {
                        return std::stoll(id);
                    }
                }
                return std::nullopt;
            }
            if (!tokenizer_json) {
                tokenizer_json = std::filesystem::exists(root / "tokenizer.json")
                                     ? Json::parse(read_file(root / "tokenizer.json"))
                                     : Json::object();
            }
            if (const auto* added = member(*tokenizer_json, "added_tokens")) {
                for (const auto& token : *added) {
                    if (token.value("content", std::string()) == name->get<std::string>() &&
                        token.contains("id") && token.at("id").is_number_integer()) {
                        return token.at("id").get<std::int64_t>();
                    }
                }
            }
            return std::nullopt;
        };
        if (const auto eos = lookup("eos_token")) {
            const OJson id = *eos;
            add(&id);
        }
        bos = lookup("bos_token");
        pad = lookup("pad_token");
    }
    if (stops.empty()) { return out; }
    OJson generation;
    if (bos) { generation["bos_token_id"] = *bos; }
    generation["eos_token_id"] = stops.size() == 1 ? OJson(stops[0]) : OJson(stops);
    if (pad) { generation["pad_token_id"] = *pad; }
    out["generation_config.json"] = py_dumps_indent2(generation);
    return out;
}

void load_resources(Model& model, const std::filesystem::path& root, std::uint64_t vocab_size,
                    const ResourceOverrides& overrides, const OJson* vision,
                    const ResourceFallbacks& fallbacks) {
    std::vector<std::pair<std::string, std::vector<std::string_view>>> roles = {
        {"text", {std::begin(kTextResources), std::end(kTextResources)}}};
    if (vision) {
        roles.push_back({"vision", {std::begin(kVisionResources), std::end(kVisionResources)}});
    }
    std::set<std::string> unused;
    for (const auto& [role, path] : overrides) {
        bool allowed = false;
        for (const auto& [component, names] : roles) {
            allowed = allowed || std::find(names.begin(), names.end(), role) != names.end();
        }
        if (!allowed) { unused.insert(role); }
    }
    if (!unused.empty()) {
        std::string names;
        for (const auto& name : unused) { names += (names.empty() ? "'" : ", '") + name + "'"; }
        throw std::runtime_error("resource overrides have no selected consumer: [" + names + "]");
    }
    // A GGUF rebuilds its tokenizer files from its own metadata; files of the same names beside it
    // are read in their place only when they are this model's. A directory that keeps several
    // models' files (a downloads folder) can hold another model's tokenizer.json: its vocabulary
    // differs from the GGUF's, and then the rebuilt files win.
    bool foreign_beside = false;
    if (const auto rebuilt = fallbacks.find("tokenizer.json");
        rebuilt != fallbacks.end() && std::filesystem::exists(root / "tokenizer.json") &&
        std::none_of(overrides.begin(), overrides.end(),
                     [](const auto& o) { return o.first == "tokenizer.json"; })) {
        std::string beside, own;
        try {
            beside = spec::vocab_fingerprint_file(root / "tokenizer.json").value;
            own    = spec::vocab_fingerprint_text(rebuilt->second).value;
        } catch (const std::exception&) {}
        if (beside != own) {
            foreign_beside = true;
            model.resource_notes.push_back(
                "tokenizer files in " + root.string() +
                " are another model's (their vocabulary is not the GGUF's): not read; the files "
                "rebuilt from the GGUF are used (--resource names one explicitly)");
        }
    }
    Json tokenizer;
    Json tokenizer_config;
    for (const auto& [component, names] : roles) {
        OJson references = OJson::object();
        for (const auto role : names) {
            std::filesystem::path path = root / role;
            bool overridden            = false;
            for (const auto& [name, override_path] : overrides) {
                if (name == role) {
                    path       = override_path;
                    overridden = true;
                }
            }
            std::string data;
            const auto fallback = fallbacks.find(std::string(role));
            if (!overridden && fallback != fallbacks.end() &&
                (foreign_beside || !std::filesystem::exists(path))) {
                data = fallback->second;
                path = std::string(role) + " (rebuilt)";
                model.resource_notes.push_back(std::string(role) + ": rebuilt, not read from " +
                                               root.string());
            } else {
                data = read_file(path);
            }
            if (data.empty()) { throw std::runtime_error(path.string() + ": resource is empty"); }
            if (!valid_utf8(data)) { throw std::runtime_error(path.string() + ": invalid UTF-8"); }
            if (role.ends_with(".json")) {
                auto value = Json::parse(data);
                if (!value.is_object()) {
                    throw std::runtime_error(path.string() +
                                             ": resource must contain a JSON object");
                }
                if (component == "vision") {
                    for (const auto& [field, config_field] :
                         {std::pair{"patch_size", "patch_size"},
                          std::pair{"temporal_patch_size", "temporal_patch_size"},
                          std::pair{"merge_size", "spatial_merge_size"}}) {
                        const auto* got = member(value, field);
                        if (!got || !got->is_number_integer() ||
                            got->get<std::int64_t>() !=
                                (*vision)[config_field].get<std::int64_t>()) {
                            throw std::runtime_error(path.string() + ": " + field +
                                                     " differs from Vision config");
                        }
                    }
                }
                if (role == "tokenizer.json") { tokenizer = std::move(value); }
                if (role == "tokenizer_config.json") { tokenizer_config = std::move(value); }
            }
            const auto id                 = "resource/" + component + "/" + std::string(role);
            references[std::string(role)] = id;
            model.resources.push_back({id, std::move(data)});
        }
        model.components[component]["resources"] = std::move(references);
    }
    std::map<std::uint64_t, std::string> beyond;
    auto [count, special]   = token_domain(tokenizer, tokenizer_config, vocab_size, beyond);
    for (const auto& [id, content] : beyond) {
        model.resource_notes.push_back("tokenizer: added token " + std::to_string(id) + " '" +
                                       content + "' lies past the embedding table (vocab_size " +
                                       std::to_string(vocab_size) +
                                       "): outside the token domain, a session refuses it");
    }
    model.token_count       = count;
    model.special_token_ids = std::move(special);
}

bool has_model_config(const OJson& config) {
    static const std::set<std::string, std::less<>> bookkeeping = {
        "quantization_config", "_name_or_path", "transformers_version", "torch_dtype", "dtype"};
    for (auto it = config.begin(); it != config.end(); ++it) {
        if (!bookkeeping.contains(it.key())) { return true; }
    }
    return false;
}

void check_source_fields(const std::string& name, const OJson& actual, const OJson& expected,
                         const std::set<std::string>& fields) {
    for (const auto& field : fields) {
        const auto* a    = member(actual, field);
        const auto* e    = member(expected, field);
        const bool equal = (!a && !e) || (a && e && py_equal(*a, *e));
        if (!equal) {
            throw std::runtime_error(name + ": additional source " + field +
                                     " differs; provide an explicit logical source");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// proposal.py select_shortlist / shortlist_ids

std::vector<std::int64_t> select_shortlist(const std::vector<std::int64_t>& counts, std::uint64_t n,
                                           const std::vector<std::int64_t>& force_include) {
    if (!(0 < n && n <= counts.size())) {
        throw std::runtime_error("shortlist size is outside the frequency domain");
    }
    const auto by_count = [&](std::int64_t a, std::int64_t b) {
        return -counts[static_cast<std::size_t>(a)] < -counts[static_cast<std::size_t>(b)];
    };
    std::vector<std::int64_t> order(counts.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), by_count);
    std::set<std::int64_t> forced_set;
    for (const auto id : force_include) {
        if (0 <= id && static_cast<std::uint64_t>(id) < counts.size()) { forced_set.insert(id); }
    }
    std::vector<std::int64_t> forced(forced_set.begin(), forced_set.end());
    if (forced.size() >= n) {
        std::stable_sort(forced.begin(), forced.end(), by_count);
        forced.resize(n);
        return forced;
    }
    const auto wanted = n - forced.size();
    std::vector<std::int64_t> selected;
    for (const auto id : order) {
        if (!forced_set.contains(id)) {
            selected.push_back(id);
            if (selected.size() == wanted) { break; }
        }
    }
    selected.insert(selected.end(), forced.begin(), forced.end());
    std::stable_sort(selected.begin(), selected.end(), by_count);
    return selected;
}

namespace {

// Row 0 of a ranking file (total counts over the vocabulary), cut to the model's token domain.
std::vector<std::int64_t> ranking_totals(const Model& model, const std::filesystem::path& ranking) {
    const auto vocab = model.config()["vocab_size"].get<std::uint64_t>();
    const auto size  = std::filesystem::file_size(ranking);
    if (size == 0 || size % (vocab * 8)) {
        throw std::runtime_error("ranking must contain complete int64 vocabulary rows");
    }
    std::vector<std::int64_t> total(vocab);
    {
        std::ifstream stream(ranking, std::ios::binary);
        stream.read(reinterpret_cast<char*>(total.data()), static_cast<std::streamsize>(vocab * 8));
        if (!stream) { throw std::runtime_error("cannot read " + ranking.string()); }
    }
    total.resize(std::min<std::uint64_t>(model.token_count, total.size()));
    return total;
}

// Bits a stored value takes in `format` (codes, high bits and scales together).
double stored_bits(QType format) {
    if (format == QType::GGML_Q4_K) { return 4.5; }
    if (format == QType::GGML_Q5_K) { return 5.5; }
    if (const auto p = ggml_plane_bytes(format); p.values != 0) {
        return 8.0 * (p.code + p.high + p.scale) / p.values;
    }
    switch (format) {
    case QType::Q4_G64_FP16: return 4.25;
    case QType::Q5_G64_FP16: return 5.25;
    case QType::Q6_G64_FP16: return 6.25;
    case QType::Q8_G32_FP16: return 8.5;
    case QType::FP32:
    case QType::INT32: return 32;
    case QType::NVFP4: return 4.5;
    case QType::MXFP4: return 4.25;
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::FP8_E4M3FN_ROW_F32:
    case QType::FP8_E4M3FN_B128_F32: return 8;
    case QType::U4_G128_ZP_FP16: return 4.25;
    case QType::U4_G64_ZP_FP16: return 4.5;
    case QType::U4_G32_ZP_FP16: return 5;
    case QType::Q3_G128_FP16: return 3.125;
    case QType::Q3_G64_FP16: return 3.25;
    default: return 16;
    }
}

// The bytes a parameter is stored in: its published encoding when it has one (kept as
// published), else the recipe's format.
double stored_bytes(const Model& model, const Recipe& recipe, std::size_t index) {
    const Parameter& p = model.parameters[index];
    double values      = 1;
    for (const auto d : p.shape) { values *= static_cast<double>(d); }
    const auto& selection = recipe.selections[index];
    const SourcePtr& source = selection.source ? selection.source : p.source;
    QType format            = selection.format;
    if (source) {
        if (const auto encoded = source->encoded_format()) { format = *encoded; }
    }
    return values * stored_bits(format) / 8.0;
}

} // namespace

DefaultProposal default_proposal(const Model& model, const Recipe& recipe,
                                 const std::filesystem::path& ranking) {
    DefaultProposal out;
    const auto type = model.config().value("model_type", std::string());
    if (type != "qwen3_5_text" && type != "qwen3_5_moe_text") { return out; }
    const std::string drafter = model.components.contains("dflash2") ? "dflash2"
                                : model.components.contains("dflash") ? "dflash"
                                                                      : "";
    if (drafter.empty() || !model.index.contains("text/output_head")) { return out; }
    // A round as spec/round_cost.h prices it: the target's verify pass over depth + 1 rows (its
    // routed experts at that width), the drafter's block pass, the draft head.
    spec::PassFacts target;
    double drafter_bytes = 0;
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const Parameter& p = model.parameters[i];
        // A parameter bound to another's objects is read once (a tied head is the head's read).
        if (recipe.alias[i] && p.name != "text/output_head") { continue; }
        const double bytes = stored_bytes(model, recipe, i);
        if (p.residency == drafter) {
            drafter_bytes += bytes;
        } else if (p.residency == "text" && p.name != "text/token_embedding") {
            if (p.name.find("/moe/experts/") != std::string::npos) {
                target.expert_bytes += bytes;
            } else {
                target.bytes += bytes;
            }
        }
    }
    if (target.expert_bytes > 0) {
        target.experts      = model.config().value("num_experts", std::int32_t{0});
        target.experts_used = model.config().value("num_experts_per_tok", std::int32_t{0});
    }
    const double head  = stored_bytes(model, recipe, model.at("text/output_head"));
    const double round = target.bytes +
                         target.expert_bytes * spec::moe_read_fraction(target, spec::kDraftHeadDepth + 1) +
                         drafter_bytes + head;
    if (!(round > head && head > 0)) { return out; }
    const double share = head / round;
    const auto choice  = spec::draft_head_rows(ranking_totals(model, ranking), share,
                                               spec::kDraftHeadDepth);
    std::ostringstream note;
    note.precision(3);
    note << "draft head: " << drafter << " round reads the head as " << 100 * share
         << " % of its bytes; ";
    if (choice.rows == 0) {
        note << "no restriction pays, the full head drafts";
    } else {
        note << choice.rows << " ranked rows keep " << 100 * choice.coverage
             << " % of the ranking's tokens (bound at depth " << spec::kDraftHeadDepth << ": "
             << 100 * choice.gain << " % a round)";
    }
    out.rows = choice.rows;
    out.note = note.str();
    return out;
}

std::uint64_t planned_proposal_bytes(double head_bytes, double target_bytes, double drafter_bytes,
                                     std::uint64_t vocab, std::uint64_t hidden) {
    if (vocab == 0 || hidden == 0 || target_bytes <= 0) { return 0; }
    const auto found = resolve_ranking(kDefaultProposalRanking, std::filesystem::current_path());
    if (!found) { return 0; }
    std::vector<std::int64_t> totals(vocab);
    try {
        if (std::filesystem::file_size(*found) % (vocab * 8) != 0) { return 0; }
        std::ifstream stream(*found, std::ios::binary);
        stream.read(reinterpret_cast<char*>(totals.data()), static_cast<std::streamsize>(vocab * 8));
        if (!stream) { return 0; }
    } catch (const std::exception&) { return 0; }
    // A checkpoint's head: BF16 in the round (the share), its gathered rows the recipe's Q4.
    const double values = static_cast<double>(vocab) * static_cast<double>(hidden);
    const double full   = head_bytes > 0 ? head_bytes : values * 2;
    const double rows_bytes =
        head_bytes > 0 ? head_bytes / static_cast<double>(vocab) : hidden * stored_bits(QType::Q4_G64_FP16) / 8;
    // The target's weights hold its head already (a GGUF's tensor table counts output.weight).
    const double round = std::max(target_bytes, full) + drafter_bytes + full;
    const auto choice  = spec::draft_head_rows(totals, full / round, spec::kDraftHeadDepth);
    return static_cast<std::uint64_t>(static_cast<double>(choice.rows) * rows_bytes);
}

std::vector<std::int64_t> shortlist_ids(const Model& model, const std::filesystem::path& ranking,
                                        std::uint64_t rows) {
    return select_shortlist(ranking_totals(model, ranking), rows, model.special_token_ids);
}

std::string declared_architecture(const OJson& config) {
    const auto* architectures = member(config, "architectures");
    if (!architectures || !architectures->is_array() || architectures->size() != 1 ||
        !(*architectures)[0].is_string()) {
        throw std::runtime_error("checkpoint config must declare exactly one architecture");
    }
    return (*architectures)[0].get<std::string>();
}

void add_official_proposal(Model& model, Recipe& recipe, const std::filesystem::path& ranking,
                           const std::string& ranking_text, std::uint64_t rows) {
    if (model.index.contains("proposal/head")) {
        throw std::runtime_error("proposal head already configured");
    }
    const auto ids = shortlist_ids(model, ranking, rows);
    auto head      = gather_source(model.parameters[model.at("text/output_head")].source, ids);
    std::vector<std::string> inputs;
    for (const char* component : {"mtp", "dflash", "dflash2"}) {
        if (model.components.contains(component)) {
            inputs.push_back(std::string(component) + "/final_hidden");
        }
    }
    model.components["text"]["proposal"] = OJson{{"domain", "indexed"}, {"rows", rows}};
    Parameter proposal;
    proposal.name         = "proposal/head";
    proposal.shape        = head->shape();
    proposal.source       = std::move(head);
    proposal.inputs       = std::move(inputs);
    proposal.residency    = "proposal";
    const auto head_index = model.add(std::move(proposal));
    std::vector<std::int32_t> words(ids.begin(), ids.end());
    Parameter token_ids;
    token_ids.name       = "proposal/token_ids";
    token_ids.shape      = {rows};
    token_ids.source     = int32_source(std::move(words), "shortlist(" + ranking_text + ")");
    token_ids.direct     = QType::INT32;
    token_ids.residency  = "proposal";
    const auto ids_index = model.add(std::move(token_ids));
    recipe.add_parameter(model, head_index);
    recipe.add_parameter(model, ids_index);
    recipe.assign(head_index, QType::Q4_G64_FP16, Method::GroupedAbsmax);
}

} // namespace ninferno::convert
