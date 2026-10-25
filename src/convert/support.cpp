#include "convert/support.h"

#include "convert/checks.h"
#include "convert/convert.h"
#include "convert/dense.h"
#include "convert/gguf.h"
#include "convert/gguf_model.h"
#include "convert/pyjson.h"
#include "convert/safetensors.h"
#include "core/ascii.h"
#include "core/sys.h"
#include "spec/registry.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

namespace ninferno::convert {

namespace fs = std::filesystem;

namespace {

// The tasks a generative model is tagged with on the hub ("" unsaid: not a reason to refuse).
bool generative(const std::string& pipeline) {
    static const std::set<std::string> tasks = {"",           "text-generation",
                                                "image-text-to-text", "any-to-any",
                                                "text2text-generation", "image-to-text",
                                                "chat"};
    return tasks.contains(pipeline);
}

// config.json's architecture: architectures[0], else model_type ("" when neither).
std::string architecture_of(const Json& config) {
    if (!config.is_object()) { return {}; }
    if (const auto a = config.find("architectures");
        a != config.end() && a->is_array() && !a->empty() && (*a)[0].is_string()) {
        return (*a)[0].get<std::string>();
    }
    return config.value("model_type", std::string());
}

bool has_chat_template(const Json& tokenizer_config) {
    if (!tokenizer_config.is_object()) { return false; }
    const auto t = tokenizer_config.find("chat_template");
    return t != tokenizer_config.end() &&
           ((t->is_string() && !t->get<std::string>().empty()) ||
            (t->is_array() && !t->empty()));
}

constexpr const char* kGated =
    "the repository is gated: accept its terms on the hub and give a Hugging Face read token "
    "($HF_TOKEN, `hf auth login`, or the UI's settings)";

// The engine families (spec::engine_for_architecture).
constexpr const char* kEngines[] = {"qwen3_5", "gemma4", "nemotron_h", "dense"};

Support refuse(std::string role, std::string reason) {
    return {Verdict::Unsupported, std::move(role), std::move(reason)};
}

} // namespace

std::string serve_refusal(const Support& support) {
    switch (support.verdict) {
    case Verdict::Unsupported:
    case Verdict::NeedsToken: return support.reason;
    case Verdict::Supported:
        return support.role == "model"
                   ? std::string()
                   : "a speculative drafter, which runs beside its model: start the model and "
                     "give it with --drafter";
    case Verdict::Unknown: return {};
    }
    return {};
}

std::string_view verdict_name(Verdict verdict) {
    switch (verdict) {
    case Verdict::Supported: return "supported";
    case Verdict::NeedsToken: return "needs_token";
    case Verdict::Unsupported: return "unsupported";
    case Verdict::Unknown: return "unknown";
    }
    return "unknown";
}

std::string config_refusal(const Json& config) {
    // The converter decides from a checkpoint directory: config.json alone in a fresh one, as the
    // download's preflight reads it.
    std::error_code error;
    fs::path dir = fs::temp_directory_path(error);
    if (error) { return {}; }
    dir /= "ninferno-support-" + std::to_string(sys::pid()) + "-" +
           std::to_string(reinterpret_cast<std::uintptr_t>(&config));
    struct Remove {
        fs::path path;
        ~Remove() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }
    } remove{dir};
    fs::create_directories(dir, error);
    if (error) { return {}; }
    std::ofstream(dir / "config.json") << config.dump();
    try {
        (void)choose_recipe(dir, GgufRequest::Model, false);
        const auto ordered = OJson::parse(config.dump());
        if (is_dense_architecture(declared_architecture(ordered))) {
            (void)dense_text_config(ordered, true);
        }
    } catch (const std::exception& refused) { return refused.what(); }
    return {};
}

Support model_support(const SupportFacts& facts) {
    if (!generative(facts.pipeline)) {
        return refuse("", "not a text generation model (the hub's task: " + facts.pipeline + ")");
    }
    // The weights' format.
    if (ascii_lower(facts.library) == "mlx") {
        return refuse("model", "MLX weights (Apple's format): the converter does not read them");
    }
    if (const auto method = facts.config.is_object() ? unread_quantization(facts.config) : "";
        !method.empty()) {
        return refuse("model", method + " weights: the converter does not read them");
    }
    // MLX's quantization (a "quantization" block, or a quantization_config naming no method), and
    // packed U32 weights any method but the packing ones (GPTQ, AWQ, compressed-tensors,
    // AutoRound) would leave unread.
    const auto quantization = [&]() -> Json {
        if (!facts.config.is_object()) { return Json(); }
        if (const auto q = facts.config.find("quantization_config"); q != facts.config.end()) {
            return *q;
        }
        return facts.config.value("text_config", Json::object())
            .value("quantization_config", Json());
    }();
    const std::string method =
        quantization.is_object() ? ascii_lower(quantization.value("quant_method", std::string()))
                                 : std::string();
    if (facts.config.is_object() && facts.config.contains("quantization") && method.empty()) {
        return refuse("model", "MLX quantized weights (Apple's format): the converter does not "
                               "read them");
    }
    static const std::set<std::string> packing = {"gptq", "awq", "compressed-tensors",
                                                  "auto-round", "intel/auto-round"};
    if (!packing.contains(method) &&
        std::find(facts.weight_dtypes.begin(), facts.weight_dtypes.end(), "U32") !=
            facts.weight_dtypes.end()) {
        return refuse("model", "packed U32 weights without a packing quantization method (MLX's "
                               "packing): the converter does not read them");
    }
    // The architecture, by the converter's own lists.
    std::string role = "model";
    std::string unchecked;
    if (facts.gguf) {
        if (!facts.gguf_error.empty()) {
            return refuse("model", "its GGUF header does not parse: " + facts.gguf_error);
        }
        const auto& arch = facts.gguf_architecture;
        if (arch.empty()) {
            return {Verdict::Unknown, "", "the GGUF architecture is not known yet"};
        }
        if (arch == "clip") { return refuse("projector", "an image projector, not a model"); }
        const auto& known = supported_gguf_architectures();
        if (std::find(known.begin(), known.end(), arch) == known.end()) {
            return refuse("model", "GGUF architecture '" + arch + "' is not one the engine runs");
        }
        if (arch == "dflash" || arch == "gemma4-assistant") { role = "drafter"; }
        if (facts.needs_token) {
            return {Verdict::NeedsToken, role, kGated};
        }
    } else {
        const auto arch = architecture_of(facts.config);
        if (arch.empty()) {
            if (facts.needs_token) {
                return {Verdict::NeedsToken, "", kGated};
            }
            return {Verdict::Unknown, "", "no config.json architecture"};
        }
        const auto& registry = spec::Registry::builtin();
        if (const auto kind = registry.kind_for_architecture(arch)) {
            // A drafter of a kind some engine runs (EAGLE-3, DSpark, Medusa: none does).
            role = "drafter";
            const bool runs = std::any_of(
                std::begin(kEngines), std::end(kEngines),
                [&](const char* engine) { return registry.supported(*kind, engine); });
            if (!runs) {
                return refuse(role, "a " + *kind + " drafter: no engine runs that kind");
            }
        } else if (facts.full_config) {
            // The whole config.json: the converter's own recipe choice and field checks.
            if (const auto why = config_refusal(facts.config); !why.empty()) {
                return refuse(role, why);
            }
        } else if (spec::engine_for_architecture(arch).empty() && !is_dense_architecture(arch)) {
            return refuse("model", "architecture '" + arch + "' is not one the engine runs");
        } else {
            unchecked = "config.json fields not checked (the hub's part of it only)";
        }
        if (facts.needs_token) {
            return {Verdict::NeedsToken, role, kGated};
        }
    }
    // A drafter or a projector takes its model's tokenizer and template.
    if (role == "model" && facts.chat_template == false) {
        return refuse(role, "no chat template: a base model (the engine serves chat)");
    }
    return {Verdict::Supported, role, unchecked};
}

SupportFacts search_support_facts(const Json& item) {
    SupportFacts out;
    out.pipeline = item.value("pipeline_tag", std::string());
    // A gated repository answers its files only with a token.
    if (const auto gated = item.find("gated");
        gated != item.end() && !gated->is_null() && !(gated->is_boolean() && !gated->get<bool>())) {
        out.needs_token = hub_token().value.empty();
    }
    out.library  = item.value("library_name", std::string());
    for (const auto& tag : item.value("tags", Json::array())) {
        if (tag.is_string() && tag.get<std::string>() == "mlx") { out.library = "mlx"; }
    }
    // A GGUF summary decides only without safetensors weights: the fetch takes those first
    // (choose_files), so a GGUF beside them is not what runs.
    const auto safetensors = item.find("safetensors");
    const bool checkpoint  = safetensors != item.end() && safetensors->is_object();
    if (const auto g = item.find("gguf"); !checkpoint && g != item.end() && g->is_object()) {
        out.gguf              = true;
        out.gguf_architecture = g->value("architecture", std::string());
        out.chat_template     = g->contains("chat_template");
    }
    if (const auto c = item.find("config"); c != item.end() && c->is_object()) {
        out.config = *c;
        // The hub's summary of tokenizer_config.json shows a template it holds; one it does not
        // show proves nothing (hub_support_facts reads the file itself).
        if (!out.gguf && has_chat_template(c->value("tokenizer_config", Json::object()))) {
            out.chat_template = true;
        }
    }
    if (const auto s = item.find("safetensors"); s != item.end() && s->is_object()) {
        // Held by name: items() of a temporary outlives it in a range-for.
        const auto parameters = s->value("parameters", Json::object());
        for (const auto& [dtype, count] : parameters.items()) {
            out.weight_dtypes.push_back(dtype);
        }
    }
    // A chat_template.jinja beside the weights (siblings, when listed) is a template too.
    for (const auto& sibling : item.value("siblings", Json::array())) {
        if (sibling.value("rfilename", std::string()) == "chat_template.jinja") {
            out.chat_template = true;
        }
    }
    return out;
}

namespace {

// A GGUF file's facts from its header (the first bytes, read longer until the metadata fits).
void read_gguf_facts(const fs::path& path, SupportFacts& out) {
    out.gguf = true;
    try {
        const std::uint64_t size = fs::file_size(path);
        for (const std::uint64_t want : {8ULL << 20, 64ULL << 20, 512ULL << 20}) {
            const std::uint64_t n = std::min(want, size);
            std::string bytes(static_cast<std::size_t>(n), '\0');
            std::ifstream in(path, std::ios::binary);
            in.read(bytes.data(), static_cast<std::streamsize>(n));
            try {
                const auto file = parse_gguf_head(reinterpret_cast<const std::byte*>(bytes.data()),
                                                  bytes.size(), size, path.string());
                const auto* arch = file.find("general.architecture");
                out.gguf_architecture =
                    arch && arch->is_string() ? arch->get<std::string>() : std::string();
                out.chat_template = file.has("tokenizer.chat_template");
                if (!*out.chat_template && (file.has("general.base_model.0.repo_url") ||
                                            file.has("general.source.huggingface.repository"))) {
                    out.chat_template.reset();
                }
                return;
            } catch (const GgufTruncated&) {
                if (n == size) { throw; }
            }
        }
    } catch (const std::exception& error) { out.gguf_error = error.what(); }
}

} // namespace

SupportFacts local_support_facts(const fs::path& path) {
    SupportFacts out;
    std::error_code ec;
    if (!fs::is_directory(path, ec)) {
        if (ascii_lower(path.extension().string()) == ".gguf") { read_gguf_facts(path, out); }
        return out;
    }
    if (std::ifstream in(path / "config.json"); in) {
        out.config      = Json::parse(in, nullptr, false);
        out.full_config = out.config.is_object();
    }
    bool template_found = fs::exists(path / "chat_template.jinja", ec);
    if (std::ifstream in(path / "tokenizer_config.json"); in) {
        template_found = template_found || has_chat_template(Json::parse(in, nullptr, false));
    }
    out.chat_template = template_found;
    try {
        std::set<std::string> dtypes;
        const Checkpoint weights(path); // held by name: tensors() refers into it
        for (const auto& [name, tensor] : weights.tensors()) { dtypes.insert(tensor.dtype_name); }
        out.weight_dtypes.assign(dtypes.begin(), dtypes.end());
    } catch (const std::exception&) {} // the conversion reads them
    return out;
}

std::optional<SupportFacts> hub_support_facts(const std::string& spec,
                                              const FetchOptions& options) {
    const auto card = fetch_model_card(spec, options);
    if (!card) { return std::nullopt; }
    SupportFacts out = search_support_facts(*card);
    // The model the fetch takes: top-level safetensors weights, else (or with --quant naming one)
    // a GGUF model.
    bool has_config = false, has_gguf = false, has_safetensors = false, has_tokenizer = false;
    for (const auto& sibling : card->value("siblings", Json::array())) {
        const auto name  = sibling.value("rfilename", std::string());
        const auto lower = ascii_lower(name);
        has_config       = has_config || name == "config.json";
        has_tokenizer    = has_tokenizer || name == "tokenizer_config.json";
        has_safetensors  = has_safetensors ||
                          (lower.ends_with(".safetensors") && name.find('/') == std::string::npos);
        has_gguf = has_gguf || (lower.ends_with(".gguf") &&
                                !gguf_auxiliary_file(fs::path(name).filename().string()));
    }
    const auto gated = card->find("gated");
    const bool is_gated =
        gated != card->end() && !(gated->is_boolean() && !gated->get<bool>()) && !gated->is_null();
    // --include naming GGUF files takes them whatever else the repository holds.
    const bool include_gguf =
        std::any_of(options.include.begin(), options.include.end(),
                    [](const std::string& glob) { return ascii_lower(glob).ends_with(".gguf"); });
    if (include_gguf || (has_gguf && (!has_safetensors || !options.quant.empty()))) {
        // The GGUF model the fetch would take: its own header, not the hub's reading of one file;
        // the repository's other weights (their library, their dtypes) are not what runs.
        out.gguf   = true;
        out.config = Json();
        out.library.clear();
        out.weight_dtypes.clear();
        try {
            for (const std::uint64_t bytes : {8ULL << 20, 32ULL << 20, 128ULL << 20}) {
                const auto head = fetch_gguf_head(spec, options, bytes);
                try {
                    const auto file = parse_gguf_head(
                        reinterpret_cast<const std::byte*>(head.bytes.data()), head.bytes.size(),
                        head.size, head.path);
                    const auto* arch = file.find("general.architecture");
                    out.gguf_architecture =
                        arch && arch->is_string() ? arch->get<std::string>() : std::string();
                    // Without a template of its own, one its base model ships comes with the
                    // tokenizer files (acquire.cpp, base_tokenizer): not known before.
                    out.chat_template = file.has("tokenizer.chat_template");
                    if (!*out.chat_template && (file.has("general.base_model.0.repo_url") ||
                                                file.has("general.source.huggingface.repository"))) {
                        out.chat_template.reset();
                    }
                    break;
                } catch (const GgufTruncated&) { continue; }
            }
        } catch (const std::exception& error) {
            if (is_gated) {
                out.needs_token = true;
            } else {
                out.gguf_error = error.what();
            }
        }
        return out;
    }
    // The checkpoint the fetch takes: the hub's summary of a GGUF beside it (its architecture,
    // its template) is not its.
    if (out.gguf) {
        auto checkpoint = *card;
        checkpoint.erase("gguf");
        out = search_support_facts(checkpoint);
    }
    if (has_config) {
        // The whole file for its fields; the template stays the card's reading (config.json
        // does not carry one).
        try {
            out.config      = fetch_config(spec, options);
            out.full_config = true;
        } catch (const std::exception&) {
            if (is_gated) { out.needs_token = true; }
        }
        // The template: the card's summary showed none, so the file decides; a checkpoint
        // without one has none.
        if (!out.chat_template.value_or(false) && !out.needs_token) {
            out.chat_template = false;
            if (has_tokenizer) {
                try {
                    out.chat_template =
                        has_chat_template(fetch_repo_json(spec, options, "tokenizer_config.json"));
                } catch (const std::exception&) { out.chat_template.reset(); }
            }
        }
    }
    return out;
}

} // namespace ninferno::convert
