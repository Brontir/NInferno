#include "ui/app.h"

#include "artifact/reader.h"
#include "convert/drafters.h"
#include "core/ascii.h"
#include "core/device_class.h"
#include "spec/fingerprint.h"
#include "spec/manifest.h"
#include "spec/pairs.h"
#include "spec/profile.h"
#include "spec/registry.h"
#include "convert/fetch.h"
#include "convert/acquire.h"
#include "convert/convert.h"
#include "convert/gguf.h"
#include "convert/gguf_model.h"
#include "convert/cache.h"
#include "convert/support.h"
#include "product/activation_options.h"
#include "product/numeric_options.h"
#include "product/plan/plan_choice.h"
#include "product/plan/plan_device.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_quants.h"
#include "product/plan/plan_speculation.h"
#include "artifact/framing.h"
#include "core/cache_file.h"
#include "core/clock.h"
#include "core/parse.h"
#include "core/sys.h"
#include "core/units.h"
#include "ui/chat_ui.h"
#include "ui/embedded_page.h"

#include <httplib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <regex>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <future>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

namespace ninferno::ui {

namespace fs = std::filesystem;
using Json   = nlohmann::json;

namespace {

void reply_json(httplib::Response& res, const Json& body, int status = 200) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

void reply_error(httplib::Response& res, int status, const std::string& message) {
    reply_json(res, {{"error", message}}, status);
}

int int_param(const httplib::Request& req, const char* name, int fallback);

// What a launch would ask for, from /api/plan's and /api/quants' parameters: ctx (tokens, "auto"
// or absent: automatic), par (1..8), kv, spec, precision, act (auto | exact | fast | int8).
// Throws std::invalid_argument for a value ninferno-serve refuses.
product::LaunchAsk launch_ask_of(const httplib::Request& req) {
    product::LaunchAsk out;
    const std::string ctx = req.get_param_value("ctx");
    if (!ctx.empty() && ctx != "auto") {
        long value = 0;
        try {
            value = std::stol(ctx);
        } catch (const std::exception&) {
            throw std::invalid_argument("ctx must be tokens or auto");
        }
        if (value < 1 || value > (1L << 24)) {
            throw std::invalid_argument("ctx must be in [1, 16777216] tokens or auto");
        }
        out.max_context = static_cast<std::uint32_t>(value);
    }
    const int par = int_param(req, "par", 1);
    if (par < 1 || par > static_cast<int>(kMaximumConcurrency)) {
        throw std::invalid_argument("par must be in [1, 8]");
    }
    out.max_concurrency  = static_cast<std::uint32_t>(par);
    const std::string kv = req.get_param_value("kv");
    if (!kv.empty() && kv != "auto" && kv != "bf16" && kv != "int8" && kv != "fp8" &&
        kv != "nvfp4" && kv != "k8v4") {
        throw std::invalid_argument("kv must be auto, bf16, int8, fp8, nvfp4 or k8v4");
    }
    out.kv_dtype         = kv == "auto" ? std::string() : kv;
    const std::string sp = req.get_param_value("spec");
    if (!sp.empty() && sp != "auto" && sp != "none" && sp != "mtp" && sp != "dflash" &&
        sp != "dflash2") {
        throw std::invalid_argument("spec must be auto, none, mtp, dflash or dflash2");
    }
    out.spec  = sp.empty() ? std::string("auto") : sp;
    out.exact             = req.get_param_value("precision") == "exact";
    const std::string act = req.get_param_value("act");
    if (!act.empty() && act != "auto" && act != "exact" && act != "fast" && act != "int8") {
        throw std::invalid_argument("act must be auto, exact, fast or int8");
    }
    out.activations = act == "auto" ? std::string() : act;
    return out;
}

int int_param(const httplib::Request& req, const char* name, int fallback) {
    if (!req.has_param(name)) { return fallback; }
    try {
        return std::stoi(req.get_param_value(name));
    } catch (const std::exception&) { return fallback; }
}

// What a request to a serve process the UI started carries: the UI passes no key flag, so the
// process requires the key it inherits, the UI's own NINFERNO_API_KEY, when that is set.
httplib::Headers serve_headers() {
    const char* key = std::getenv("NINFERNO_API_KEY");
    if (key == nullptr || *key == '\0') { return {}; }
    return {{"Authorization", std::string("Bearer ") + key}};
}

// The serve process's /v1/models body, or "" when it does not answer 200 in time.
std::string serve_models_body(int port) {
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(std::chrono::seconds(5));
    auto result = client.Get("/v1/models", serve_headers());
    return result && result->status == 200 ? result->body : std::string();
}

// A process id from a route's \d+ match; 0 (no process) when it does not fit.
std::uint64_t process_id(const std::string& digits) {
    return parse_exact<std::uint64_t>(digits).value_or(0);
}

// Streams a chat completion from the serve process on `port` to the page: its SSE as it arrives,
// or, when the server refuses the request (a JSON error and a status other than 200), that error
// as one event and [DONE], as ninferno-serve ends a failed stream, so the page shows the reason.
void stream_chat(httplib::Response& res, int port, std::string body) {
    res.set_chunked_content_provider("text/event-stream", [port, body = std::move(body)](
                                                              std::size_t,
                                                              httplib::DataSink& sink) {
        httplib::Client client("127.0.0.1", port);
        client.set_read_timeout(std::chrono::minutes(10));
        int status = 0;
        std::string refused;
        httplib::Request request;
        request.method  = "POST";
        request.path    = "/v1/chat/completions";
        request.headers = serve_headers();
        request.set_header("Content-Type", "application/json");
        request.body             = body;
        request.response_handler = [&status](const httplib::Response& response) {
            status = response.status;
            return true;
        };
        request.content_receiver = [&](const char* data, std::size_t length, std::size_t,
                                       std::size_t) {
            if (status == 200) { return sink.write(data, length); }
            refused.append(data, length);
            return true;
        };
        const auto result = client.send(request);
        std::string tail;
        if (!result) {
            tail = "data: " +
                   Json{{"error", {{"message", httplib::to_string(result.error())}}}}.dump() +
                   "\n\ndata: [DONE]\n\n";
        } else if (status != 200) {
            const Json error = Json::parse(refused, nullptr, false);
            tail             = "data: " +
                   (error.is_object()
                        ? error.dump()
                        : Json{{"error", {{"message", "HTTP " + std::to_string(status)}}}}.dump()) +
                   "\n\ndata: [DONE]\n\n";
        }
        if (!tail.empty()) { sink.write(tail.data(), tail.size()); }
        sink.done();
        return true;
    });
}

bool is_word(const std::string& text) {
    return !text.empty() && std::all_of(text.begin(), text.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '_' || c == '-' || c == '.';
    });
}

} // namespace

ArtifactFacts read_artifact_facts(const fs::path& artifact) {
    ArtifactFacts out;
    try {
        const artifact::Reader reader(artifact);
        const auto& components = reader.directory().components;
        const auto text        = components.find("text");
        if (text == components.end()) { return out; }
        const auto& config = text->second.config;
        if (const auto it = config.find("architectures");
            it != config.end() && it->is_array() && !it->empty() && (*it)[0].is_string()) {
            out.architecture = (*it)[0].get<std::string>();
        } else {
            out.architecture = config.value("model_type", std::string());
        }
        out.readable = true;
        out.drafter = spec::Registry::builtin().kind_for_architecture(out.architecture).has_value();
        out.vision  = components.contains("vision");
        out.mtp     = components.contains("mtp");
    } catch (const std::exception&) {}
    return out;
}

ArtifactName artifact_name(const std::string& file_name) {
    ArtifactName out;
    std::string base = file_name;
    if (base.ends_with(".ninferno")) { base.resize(base.size() - 9); }
    out.stem = base;
    if (const auto dot = base.rfind('.'); dot != std::string::npos) {
        const auto tail   = base.substr(dot + 1);
        const bool recipe = !tail.empty() &&
                            std::all_of(tail.begin(), tail.end(),
                                        [](unsigned char c) {
                                            return std::islower(c) || std::isdigit(c) || c == '_';
                                        }) &&
                            tail.find('_') != std::string::npos;
        if (recipe) {
            out.recipe = tail;
            out.stem   = base.substr(0, dot);
        }
    }
    out.quant = convert::gguf_quant_label(out.stem + ".gguf");
    // GGUF labels are upper case (Q4_K_M); a lower-case tail ("nvfp4_dflash2_v3" of a native
    // artifact's name) is not one: the format is then looked for by name.
    if (std::none_of(out.quant.begin(), out.quant.end(),
                     [](unsigned char c) { return std::isupper(c); })) {
        out.quant.clear();
    }
    if (out.quant.empty()) {
        const std::string lower = ascii_lower(out.stem);
        for (const char* name : {"nvfp4", "mxfp4", "fp8", "q8", "bf16"}) {
            if (lower.find(name) != std::string::npos) {
                out.quant = ascii_upper(name);
                break;
            }
        }
    }
    return out;
}

std::string params_label(const std::string& name) {
    static const std::regex size(
        R"((?:^|[^0-9A-Za-z.])(\d+(?:\.\d+)?)[bB](?:[-_][aA](\d+(?:\.\d+)?)[bB])?(?![0-9A-Za-z]))");
    std::smatch match;
    if (!std::regex_search(name, match, size)) { return {}; }
    return match[1].str() + "B" + (match[2].matched ? "-A" + match[2].str() + "B" : std::string());
}

// An architecture name reduced for comparison: lower case, letters and digits only, the class
// suffix dropped ("Qwen3_5MoeForCausalLM", "qwen35moe" and model_type "qwen3_5_moe" all qwen35moe).
// The UI's support level of a model (page.html's SUP), from convert::model_support: "full" (the
// engine runs it), "drafter" (runs beside its model), "token" (gated: set HF_TOKEN), "projector"
// (the hub read an image projector, not the model: the model file is read again), "unknown",
// "none" (with the reason).
std::string support_level(const convert::Support& support) {
    switch (support.verdict) {
    case convert::Verdict::Supported: return support.role == "drafter" ? "drafter" : "full";
    case convert::Verdict::NeedsToken: return "token";
    case convert::Verdict::Unknown: return "unknown";
    case convert::Verdict::Unsupported: return support.role == "projector" ? "projector" : "none";
    }
    return "unknown";
}

// The facts an architecture name alone gives (ModelScope's search, a GGUF header, config.json).
convert::SupportFacts facts_of_architecture(const std::string& architecture, bool gguf,
                                            std::string pipeline = {}, std::string library = {}) {
    convert::SupportFacts facts;
    facts.pipeline = std::move(pipeline);
    facts.library  = std::move(library);
    facts.gguf     = gguf;
    if (gguf) {
        facts.gguf_architecture = architecture;
    } else if (!architecture.empty()) {
        facts.config = Json{{"architectures", Json::array({architecture})}};
    }
    return facts;
}

// general.architecture from the first bytes of a GGUF file: the header, then key-value pairs until
// that key (llama.cpp writes it first); "" when it is not within `bytes`.
std::string gguf_architecture(const std::string& bytes) {
    std::size_t at  = 0;
    const auto take = [&](std::size_t n) -> const char* {
        if (at + n > bytes.size()) { return nullptr; }
        const char* p = bytes.data() + at;
        at += n;
        return p;
    };
    const auto u32 = [&](std::uint32_t& v) {
        const char* p = take(4);
        if (p) { std::memcpy(&v, p, 4); }
        return p != nullptr;
    };
    const auto u64 = [&](std::uint64_t& v) {
        const char* p = take(8);
        if (p) { std::memcpy(&v, p, 8); }
        return p != nullptr;
    };
    const auto str = [&](std::string& s) {
        std::uint64_t n = 0;
        if (!u64(n) || n > (1u << 20)) { return false; }
        const char* p = take(static_cast<std::size_t>(n));
        if (p) { s.assign(p, static_cast<std::size_t>(n)); }
        return p != nullptr;
    };
    static const std::size_t scalar[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    std::uint32_t magic = 0, version = 0;
    std::uint64_t tensors = 0, pairs = 0;
    if (bytes.compare(0, 4, "GGUF") != 0) { return ""; }
    if (!u32(magic) || !u32(version) || !u64(tensors) || !u64(pairs)) { return ""; }
    for (std::uint64_t i = 0; i < pairs && i < 64; ++i) {
        std::string key;
        std::uint32_t type = 0;
        if (!str(key) || !u32(type) || type > 12) { return ""; }
        if (type == 8) {
            std::string value;
            if (!str(value)) { return ""; }
            if (key == "general.architecture") { return value; }
        } else if (type == 9) {
            std::uint32_t inner = 0;
            std::uint64_t count = 0;
            if (!u32(inner) || !u64(count) || inner > 12 || inner == 9) { return ""; }
            for (std::uint64_t k = 0; k < count; ++k) {
                std::string skip;
                if (inner == 8 ? !str(skip) : take(scalar[inner]) == nullptr) { return ""; }
            }
        } else if (take(scalar[type]) == nullptr) {
            return "";
        }
    }
    return "";
}

std::string short_architecture(const std::string& architecture) {
    std::string out = architecture;
    for (const char* suffix : {"ForConditionalGeneration", "ForCausalLM", "Model"}) {
        if (out.size() > std::strlen(suffix) && out.ends_with(suffix)) {
            out.resize(out.size() - std::strlen(suffix));
            break;
        }
    }
    return ascii_lower(out);
}

bool deletable_artifact(const fs::path& path, const fs::path& hub_root,
                        const std::vector<fs::path>& model_dirs) {
    std::error_code ec;
    if (path.extension() != ".ninferno" || !fs::is_regular_file(path, ec)) { return false; }
    const auto under = [&](const fs::path& dir) {
        const auto base = fs::weakly_canonical(dir, ec);
        if (ec || base.empty()) { return false; }
        const auto rel = path.lexically_relative(base);
        return !rel.empty() && *rel.begin() != ".." && rel.is_relative();
    };
    const auto rel = path.lexically_relative(fs::weakly_canonical(hub_root, ec));
    if (!rel.empty() && *rel.begin() != ".." && std::distance(rel.begin(), rel.end()) >= 3) {
        auto it             = rel.begin();
        const auto repo_dir = (*it++).string();
        if (repo_dir.rfind("models--", 0) == 0 && (*it).string() == "ninferno") { return true; }
    }
    return std::any_of(model_dirs.begin(), model_dirs.end(), under);
}

std::pair<std::string, std::string> split_model_tag(const std::string& model) {
    // A hub id has exactly one '/', no leading '/' or '.', and the tag after the last ':'.
    const auto colon = model.rfind(':');
    if (colon == std::string::npos || model.empty() || model.front() == '/' ||
        model.front() == '.' || model.find('/') == std::string::npos || model.find('/') > colon) {
        return {model, ""};
    }
    std::error_code ec;
    if (fs::exists(model, ec)) { return {model, ""}; }
    return {model.substr(0, colon), model.substr(colon + 1)};
}

std::string tune_refusal_on(const fs::path& tune, const std::string& artifact_id,
                            const GpuStat* gpu) {
    // The tune itself first: unreadable, another artifact, or a drafter replaced since.
    if (auto why = spec::tune_refusal(tune, artifact_id); !why.empty()) { return why; }
    // The server keys a tune by "<chip>:<sms>sm": boards of one chip differ in SM count (RTX 5090
    // and RTX PRO 6000, RTX 4090 and L40), so on a GPU unknown here or without the count (NVML
    // gave none, or a MIG slice) the class is not judged here: the server names its GPU's class
    // from CUDA and refuses to start with a named tune of another.
    if (gpu == nullptr || gpu->cc_major <= 0 || gpu->sms <= 0) { return ""; }
    const std::string chip = gpu_chip(gpu->pci_device_id, gpu->cc_major, gpu->cc_minor, gpu->sms);
    // The class the server keys a tune by (device_class's id(), its similarity chain's front).
    return spec::tune_refusal(tune, artifact_id, gpu_class_id(chip, gpu->sms));
}

const GpuStat* launch_gpu(const std::vector<GpuStat>& stats, int gpu) {
    if (gpu < 0) { return stats.size() == 1 ? &stats.front() : nullptr; }
    for (const auto& stat : stats) {
        if (stat.index == gpu) { return &stat; }
    }
    return nullptr;
}

LaunchSpec with_tiny_scales_rounded(LaunchSpec spec, int port) {
    if (std::find(spec.argv.begin(), spec.argv.end(), "--round-tiny-scales") == spec.argv.end()) {
        spec.argv.push_back("--round-tiny-scales");
    }
    if (spec.kind == "serve") {
        const auto flag = std::find(spec.argv.begin(), spec.argv.end(), "--port");
        if (flag != spec.argv.end() && flag + 1 != spec.argv.end()) { flag[1] = std::to_string(port); }
        spec.port = port;
    }
    return spec;
}

LaunchSpec make_launch(const UiOptions& options, const LaunchRequest& request, const GpuStat* gpu) {
    auto [model, tag] = split_model_tag(request.model);
    if (model.empty()) { throw std::invalid_argument("no model"); }
    if (model.front() == '-') { throw std::invalid_argument("a model name cannot start with '-'"); }
    // A model is a repository id or a path: a quote or a control character in one is never a
    // model, only a way to end its argument on a command line.
    for (const unsigned char c : model) {
        if (c == '"' || c < 0x20 || c == 0x7f) {
            throw std::invalid_argument("a model name cannot hold quotes or control characters");
        }
    }
    const std::string quant = request.quant.empty() ? tag : request.quant;
    if (!quant.empty() && !is_word(quant)) {
        throw std::invalid_argument("bad quantization: " + quant);
    }
    static const std::set<std::string> specs = {"", "auto", "none", "mtp", "dflash", "dflash2"};
    static const std::set<std::string> precisions = {"", "exact", "balanced", "compact"};
    if (!specs.count(request.spec)) { throw std::invalid_argument("bad spec: " + request.spec); }
    if (!precisions.count(request.precision)) {
        throw std::invalid_argument("bad precision: " + request.precision);
    }
    if (!request.activations.empty() && request.activations != "exact" &&
        request.activations != "fast" && request.activations != "int8") {
        throw std::invalid_argument("bad activations: " + request.activations);
    }
    static const std::set<std::string> kv_dtypes = {"", "bf16", "int8", "fp8", "nvfp4", "k8v4"};
    if (!kv_dtypes.count(request.kv_dtype)) {
        throw std::invalid_argument("bad kv_dtype: " + request.kv_dtype);
    }
    if (!request.hub.empty() && request.hub != "huggingface" && request.hub != "modelscope") {
        throw std::invalid_argument("bad hub: " + request.hub);
    }
    if (request.max_context < 0) { throw std::invalid_argument("bad max_context"); }
    if (request.max_context_cap < 0 || (request.max_context_cap > 0 && request.max_context > 0)) {
        throw std::invalid_argument("bad max_context_cap");
    }
    if (request.max_concurrency < 0 || request.max_concurrency > 64) {
        throw std::invalid_argument("bad max_concurrency");
    }

    LaunchSpec spec;
    spec.kind  = request.prepare_only ? "prepare" : "serve";
    spec.model = quant.empty() ? model : model + ":" + quant;
    spec.gpu   = request.gpu;
    spec.port  = request.prepare_only ? 0 : request.port;
    spec.argv  = {options.serve_binary.string(), model};
    if (!quant.empty()) {
        spec.argv.push_back("--quant");
        spec.argv.push_back(quant);
    }
    if (request.hub == "modelscope") {
        throw std::invalid_argument("ModelScope models are listed only: ninferno-serve fetches from "
                                    "Hugging Face");
    }
    if (request.refetch && !request.prepare_only) {
        throw std::invalid_argument("refetch is a prepare's");
    }
    if (request.keep_replaced && !request.refetch) {
        throw std::invalid_argument("keep_replaced is a refetch's");
    }
    if (!request.recipe.empty() && !is_word(request.recipe)) {
        throw std::invalid_argument("bad recipe: " + request.recipe);
    }
    if (request.prepare_only) {
        spec.argv.push_back("--prepare-only");
        if (request.refetch) { spec.argv.push_back("--refetch"); }
        if (request.keep_replaced) { spec.argv.push_back("--keep-replaced"); }
        if (request.vision) { spec.argv.push_back("--vision"); }
        if (!request.recipe.empty()) {
            spec.argv.push_back("--recipe");
            spec.argv.push_back(request.recipe);
        }
    } else {
        spec.argv.insert(spec.argv.end(),
                         {"--host", "127.0.0.1", "--port", std::to_string(request.port)});
        if (request.max_concurrency > 0) {
            spec.argv.push_back("--max-concurrency");
            spec.argv.push_back(std::to_string(request.max_concurrency));
        }
        if (request.max_context > 0) {
            spec.argv.push_back("--max-context");
            spec.argv.push_back(std::to_string(request.max_context));
        }
        if (request.max_context_cap > 0) {
            spec.argv.push_back("--max-context-cap");
            spec.argv.push_back(std::to_string(request.max_context_cap));
        }
        // Precision: exact is ninferno-serve's --activations exact with a BF16 KV cache (the
        // output does not depend on the load); balanced and compact keep its defaults, which already
        // store the cache in the family's smallest measured format (--kv-dtype auto).
        const bool exact = request.precision == "exact";
        if (!request.kv_dtype.empty() || exact) {
            spec.argv.push_back("--kv-dtype");
            spec.argv.push_back(request.kv_dtype.empty() ? "bf16" : request.kv_dtype);
        }
        // Activations: the exact preset's, else the mode the user chose (16 / 8 bits: exact | fast
        // | int8), else none named: the configuration choice decides as on any start.
        if (exact || !request.activations.empty()) {
            spec.argv.push_back("--activations");
            spec.argv.push_back(exact ? "exact" : request.activations);
        }
        // Speculation: the one the request names, else the published profile's (--spec auto,
        // ninferno-serve's default too), named so that the command line says it.
        spec.argv.push_back("--spec");
        spec.argv.push_back(request.spec.empty() ? "auto" : request.spec);
        // A tune of this artifact (/api/tune writes one beside it) is a measurement of this
        // machine: it runs only when the start asks for it (spec_profile), never by itself, as
        // the server reads one only when named. Asked for, one that does not serve refuses the
        // start: here when NVML names the class of the GPU the request runs on (tune_refusal_on),
        // else by the server, which refuses to start with a named tune that does not serve.
        std::error_code ec;
        const fs::path tune = spec::default_profile_path(model);
        const bool by_path  = fs::path(model).extension() == ".ninferno";
        if (request.spec_profile) {
            if (!by_path) {
                throw std::invalid_argument("spec_profile: a tune is made for a converted model, "
                                            "started by its .ninferno path");
            }
            std::string why;
            try {
                why = tune_refusal_on(tune, spec::artifact_id_hex(model), gpu);
            } catch (const std::exception& e) {
                why = std::string("the model's id is unreadable: ") + e.what();
            }
            if (!why.empty()) {
                throw std::invalid_argument("spec_profile: the tune does not serve here: " + why);
            }
            spec.argv.push_back("--spec-profile");
            spec.argv.push_back(tune.string());
        } else if (by_path && fs::is_regular_file(tune, ec)) {
            spec.notes.push_back("the tune beside the model (" + tune.string() +
                                 ") is not named: a start names it only when it asks for it "
                                 "(POST /api/launch with \"spec_profile\": true; the page's Start "
                                 "does not ask for it yet)");
        }
    }
    if (request.gpu >= 0) {
        spec.env["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
        spec.env["CUDA_VISIBLE_DEVICES"] = std::to_string(request.gpu);
    }
    if (!options.hub_cache.empty()) { spec.env["HF_HUB_CACHE"] = options.hub_cache.string(); }
    return spec;
}

LaunchSpec make_tune(const UiOptions& options, const TuneRequest& request) {
    std::error_code ec;
    if (request.artifact.empty() || request.artifact.front() == '-' ||
        fs::path(request.artifact).extension() != ".ninferno" ||
        !fs::is_regular_file(request.artifact, ec)) {
        throw std::invalid_argument("not a .ninferno artifact: " + request.artifact);
    }
    const fs::path bin = options.serve_binary.parent_path();
    LaunchSpec spec;
    spec.kind  = "tune";
    spec.model = fs::path(request.artifact).filename().string();
    spec.gpu   = request.gpu;
    if (request.engine == "gemma4") {
        spec.argv = {(bin / "ninferno-gemma4").string(), request.artifact, "spec-tune",
                     "--budget-seconds", "120"};
    } else {
        spec.argv = {(bin / "ninferno").string(), "spec-tune", request.artifact, "--budget-seconds",
                     "120"};
    }
    if (request.gpu >= 0) {
        spec.env["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
        spec.env["CUDA_VISIBLE_DEVICES"] = std::to_string(request.gpu);
    }
    return spec;
}

namespace {

std::string utc_now_text() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    sys::gmtime(now, tm);
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%d %H:%M UTC", &tm);
    return text;
}

} // namespace

App::App(UiOptions options, std::unique_ptr<GpuSource> gpus)
    : options_(std::move(options)), gpus_(std::move(gpus)) {
    catalog_        = std::make_shared<const Catalog>(Catalog::builtin());
    catalog_source_ = "built in";
    if (!options_.catalog_url.empty()) {
        // The feed's last copy serves until the feed answers.
        std::error_code ec;
        const auto cached = hub_root() / "ninferno-ui" / "catalog.json";
        if (fs::is_regular_file(cached, ec)) {
            try {
                std::ifstream in(cached);
                catalog_ =
                    std::make_shared<const Catalog>(Catalog::parse(nlohmann::json::parse(in)));
                catalog_source_ = options_.catalog_url + " (cached copy)";
            } catch (const std::exception& error) {
                std::cerr << "ninferno-ui: cached catalog " << cached
                          << " unreadable: " << error.what() << '\n';
            }
        }
        if (!options_.offline) {
            catalog_thread_ = std::thread([this] {
                while (!stopping_) {
                    (void)refresh_catalog();
                    // Every six hours, or at once after a new Hugging Face token (a private
                    // feed answers only with one).
                    for (int i = 0; i < 6 * 3600 * 10 && !stopping_ && !catalog_wake_.exchange(false);
                         ++i) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                }
            });
        }
    }
    health_thread_ = std::thread([this] { watch_health(); });
    if (!options_.offline && update::checks_enabled() &&
        update::is_release_version(options_.version)) {
        update_thread_ = std::thread([this] {
            while (!stopping_) {
                if (auto release = update::check({.current = options_.version})) {
                    std::lock_guard lock(update_mutex_);
                    release_ = std::move(release);
                }
                for (int i = 0; i < 6 * 3600 * 10 && !stopping_; ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        });
    }
}

App::~App() {
    stop();
    if (update_thread_.joinable()) { update_thread_.join(); }
    if (health_thread_.joinable()) { health_thread_.join(); }
    for (auto& stopper : update_stoppers_) { // written by the health thread only, joined above
        if (stopper.joinable()) { stopper.join(); }
    }
    if (prefetch_thread_.joinable()) { prefetch_thread_.join(); }
    if (catalog_thread_.joinable()) { catalog_thread_.join(); }
}

std::shared_ptr<const Catalog> App::catalog() const {
    std::lock_guard lock(catalog_mutex_);
    return catalog_;
}

bool App::refresh_catalog() {
    if (options_.catalog_url.empty()) { return false; }
    try {
        const auto text     = convert::fetch_text(options_.catalog_url, 4U << 20);
        const auto document = nlohmann::json::parse(text);
        if (document.value("schema", 0) != 1) { throw std::runtime_error("not schema 1"); }
        auto parsed = Catalog::parse(document);
        if (parsed.models().empty()) { throw std::runtime_error("no models"); }
        // Kept for the next start (the feed may be unreachable then): written whole, then renamed.
        const auto dir = hub_root() / "ninferno-ui";
        std::error_code ec;
        fs::create_directories(dir, ec);
        try {
            write_file_atomic(dir / "catalog.json", text);
        } catch (const std::exception&) {} // the feed is used all the same
        std::lock_guard lock(catalog_mutex_);
        catalog_        = std::make_shared<const Catalog>(std::move(parsed));
        catalog_source_ = options_.catalog_url + " (read " + utc_now_text() + ")";
        return true;
    } catch (const std::exception& error) {
        std::cerr << "ninferno-ui: catalog feed " << options_.catalog_url << ": " << error.what()
                  << " (the " << (catalog() ? "current" : "built-in") << " catalog stays)\n";
        return false;
    }
}

void App::stop() {
    stopping_ = true;
    if (server_) { server_->stop(); }
}

nlohmann::json App::update_json() {
    nlohmann::json out = {{"current", options_.version}, {"can_install", !options_.updater.empty()}};
    std::lock_guard lock(update_mutex_);
    if (release_ && (release_->newer || release_->required)) {
        out["latest"]   = release_->latest;
        out["newer"]    = release_->newer;
        out["required"] = release_->required;
        out["notes"]    = release_->notes;
    }
    if (update_job_ != 0) {
        ProcessInfo job;
        if (processes_.find(update_job_, job)) {
            out["job"]       = job.id;
            out["job_state"] = job.state;
            out["installed"] = job.state == "exited" && job.exit_code == 0;
            out["last_line"] = job.last_line;
        }
    }
    return out;
}

std::uint64_t App::install_update() {
    if (options_.updater.empty()) {
        throw std::invalid_argument("this installation updates outside the page (ninferno-update)");
    }
    std::lock_guard lock(update_mutex_);
    if (update_job_ != 0) {
        ProcessInfo job;
        if (processes_.find(update_job_, job) && job.state != "exited" && job.state != "failed") {
            return update_job_; // already under way
        }
    }
    LaunchSpec spec;
    spec.kind  = "update";
    spec.model = "NInferno " + (release_ ? release_->latest : std::string("update"));
    spec.argv  = options_.updater;
    spec.argv.emplace_back("--yes");
    update_job_ = processes_.start(std::move(spec));
    return update_job_;
}

void App::restart() {
    // The models stop with the page; the new version starts them again from the page.
    for (const auto& process : processes_.list()) {
        if (process.state != "exited" && process.state != "failed") {
            processes_.stop(process.id, std::chrono::seconds(10));
        }
    }
    restart_requested_ = true;
    stop();
}

SizeTable App::sizes() const {
    std::lock_guard lock(sizes_mutex_);
    return sizes_;
}

void App::prefetch_listings() {
    if (options_.offline) { return; }
    prefetch_thread_ = std::thread([this] {
        std::set<std::string> repos;
        for (const auto& model : catalog()->models()) {
            for (const auto& source : model.sources) { repos.insert(source.repo); }
        }
        for (const auto& repo : repos) {
            if (stopping_) { return; }
            try {
                (void)quants_json(repo, -1);
            } catch (const std::exception& error) {
                std::cerr << "ninferno-ui: listing " << repo << ": " << error.what() << std::endl;
            }
        }
    });
}

// Listings and readings are kept per hub: the same org/name may be another repository there.
std::string hub_key(const std::string& repo, const std::string& hub) {
    return hub == "modelscope" ? "modelscope:" + repo : repo;
}

Json App::listing_of(const std::string& repo, const std::string& hub) {
    // What the hub answers under another token than the one asked with is not kept.
    const auto generation = hub_generation();
    {
        std::lock_guard lock(sizes_mutex_);
        if (const auto it = listings_.find(hub_key(repo, hub)); it != listings_.end()) {
            return it->second;
        }
    }
    if (options_.offline) { throw std::runtime_error("offline: no hub listing"); }
    convert::FetchOptions fetch;
    fetch.hub = hub;
    if (!options_.hub_cache.empty()) { fetch.cache = options_.hub_cache; }
    const auto listing = convert::list_quants(repo, fetch);
    Json list          = Json::array();
    std::lock_guard lock(sizes_mutex_);
    for (const auto& q : listing.quants) {
        list.push_back({{"quant", q.quant},
                        {"format", q.format},
                        {"bytes", q.bytes},
                        {"files", q.paths()},
                        {"bits", q.bits},
                        {"drafter", q.drafter},
                        {"mtp_bytes", q.mtp_bytes},
                        {"weights", spec::weights_identity(q.digests())},
                        {"commit", listing.commit}});
        if (!q.drafter && hub != "modelscope") { sizes_[size_key(repo, q.quant)] = q.bytes; }
    }
    if (generation == hub_generation_) { listings_[hub_key(repo, hub)] = list; }
    return list;
}

Json App::arch_json(const std::string& repo, const std::string& hub) {
    const auto generation = hub_generation();
    {
        std::lock_guard lock(sizes_mutex_);
        if (const auto it = archs_.find(hub_key(repo, hub)); it != archs_.end()) {
            return it->second;
        }
    }
    const auto list = listing_of(repo, hub);
    std::string quant;
    double smallest = 0;
    bool weights = false, native = false;
    for (const auto& q : list) {
        if (q.value("drafter", false)) { continue; }
        if (q.value("format", std::string()) == "ninferno") { native = true; }
        if (q.value("format", std::string()) != "gguf") {
            weights = true;
            continue;
        }
        const double bytes = q.value("bytes", 0.0);
        if (quant.empty() || bytes < smallest) {
            quant    = q.value("quant", std::string());
            smallest = bytes;
        }
    }
    convert::FetchOptions fetch;
    fetch.hub = hub;
    if (!options_.hub_cache.empty()) { fetch.cache = options_.hub_cache; }
    Json out = {{"architecture", ""}, {"source", ""}, {"file", ""}};
    try {
        if (native && hub != "modelscope") {
            // The engine's own artifact: its manifest names the architecture and the container
            // version, which this engine must read.
            const auto manifest = convert::fetch_repo_json(repo, fetch, "artifact-manifest.json");
            const auto artifact = manifest.value("artifact", Json::object());
            out["architecture"] = artifact.value("architecture", std::string());
            out["source"]       = "manifest";
            out["file"]         = "artifact-manifest.json";
            const int version   = artifact.value("container_version", 0);
            const int ours      = std::to_integer<int>(artifact::kEntryMagic[7]);
            if (version != ours) {
                out["support"] = "none";
                out["note"]    = "container v" + std::to_string(version) + ", this engine reads v" +
                              std::to_string(ours);
            }
        } else if (!quant.empty()) {
            fetch.quant         = quant;
            const auto head     = convert::fetch_gguf_head(repo, fetch, 256 << 10);
            out["architecture"] = gguf_architecture(head.bytes);
            out["source"]       = "gguf";
            out["file"]         = head.path;
        } else if (weights && hub != "modelscope") {
            out["architecture"] = product::plan_facts_of_config(convert::fetch_config(repo, fetch),
                                                                0, runtime::WeightBasis::Unknown)
                                      .model.architecture;
            out["source"] = "config";
            out["file"]   = "config.json";
        }
    } catch (const std::exception& error) { out["error"] = error.what(); }
    // Whether the engine runs it: the whole decision from what the hub shows (convert/support.h),
    // else from the architecture read above.
    if (!out.contains("support")) {
        std::optional<convert::Support> support;
        if (hub != "modelscope") {
            try {
                // The model an unqualified fetch takes: safetensors weights when there are
                // any, else the GGUF read above.
                auto ask  = fetch;
                ask.quant = weights ? std::string() : quant;
                if (const auto facts = convert::hub_support_facts(repo, ask)) {
                    support = convert::model_support(*facts);
                }
            } catch (const std::exception&) {}
        }
        if (!support) {
            support = convert::model_support(facts_of_architecture(
                out["architecture"].get<std::string>(), out["source"] == "gguf"));
        }
        out["support"] = support_level(*support);
        if (!support->reason.empty()) { out["note"] = support->reason; }
    }
    std::lock_guard lock(sizes_mutex_);
    if (!out.contains("error") && generation == hub_generation_) { archs_[hub_key(repo, hub)] = out; }
    return out;
}

namespace {

// A GPU as the plan without the weights sees it: the CUDA device a server on it reads
// (totalGlobalMem, the driver model, the watchdog as the display flag: product/plan/plan_device.h),
// found by the NVML entry's PCI bus id; NVML's own numbers only where CUDA does not know the GPU.
std::optional<runtime::PlanDeviceFacts> plan_device(const std::vector<GpuStat>& stats, int gpu) {
    for (const auto& stat : stats) {
        if (stat.index != gpu) { continue; }
        if (auto cuda = product::plan_device_of_bus(stat.bus_id)) { return cuda; }
        return product::plan_device_of_stats(stat.name, stat.memory_total, stat.cc_major,
                                             stat.cc_minor, std::nullopt, stat.pci_device_id,
                                             stat.sms);
    }
    return std::nullopt;
}

// The plans kept at most (each a few hundred KB of JSON): the oldest half goes beyond.
constexpr std::size_t kMaxKeptPlans = 256;

// A listing's rows as the quant rule reads them.
std::vector<product::QuantListing> quant_listing(const Json& list, const Json& config) {
    std::vector<product::QuantListing> out;
    for (const auto& q : list) {
        product::QuantListing row{.quant   = q.value("quant", std::string()),
                                  .format  = q.value("format", std::string()),
                                  .bytes   = q.value("bytes", std::uint64_t{0}),
                                  .bits    = q.value("bits", 0),
                                  .drafter = q.value("drafter", false),
                                  .weights = q.value("weights", std::string())};
        if (row.format == "safetensors" && !row.drafter && config.is_object()) {
            row.converted_bytes = product::checkpoint_converted_bytes(config, row.bytes);
        }
        out.push_back(std::move(row));
    }
    return out;
}

std::string launch_key(const product::LaunchAsk& launch) {
    return std::to_string(launch.max_context) + "|" + std::to_string(launch.max_concurrency) + "|" +
           launch.kv_dtype + "|" + launch.spec + "|" + (launch.exact ? "exact" : "") + "|" +
           launch.activations;
}

} // namespace

std::optional<product::PlanSourceFacts>
App::plan_facts(const std::string& repo, const Json& listing, const std::string& hub) {
    (void)kv_json(repo, listing, hub); // reads the header once, and the facts with it
    std::lock_guard lock(sizes_mutex_);
    if (const auto it = plan_facts_.find(hub_key(repo, hub)); it != plan_facts_.end()) {
        return it->second;
    }
    return std::nullopt;
}

struct App::PlanNowContext {
    std::shared_ptr<product::PlannedSubjects> subjects; // what inputs' subject pointers name
    product::PlanSourceFacts facts;
    product::DryRunInputs inputs;
    std::optional<product::ServedChoice> served;
    product::ChoicePins pins;
};

namespace {

// The free memory of GPU `gpu` now (NVML), nullopt when it does not say.
std::optional<std::uint64_t> free_now_of(const std::vector<GpuStat>& stats, int gpu) {
    for (const auto& stat : stats) {
        if (stat.index == gpu && stat.memory_total > 0) {
            return product::plan_free_now_of_nvml(static_cast<std::uint64_t>(stat.memory_free));
        }
    }
    return std::nullopt;
}

} // namespace

Json App::artifact_plan_json(const fs::path& artifact, int gpu, const product::LaunchAsk& launch) {
    const auto stats  = gpus_->snapshot();
    const auto device = plan_device(stats, gpu);
    if (!device) { throw std::runtime_error("no GPU " + std::to_string(gpu) + " to plan for"); }
    product::DryRunInputs inputs;
    inputs.device = *device;
    inputs.gpu    = product::gpu_key_of(*device);
    for (const auto& stat : stats) {
        if (stat.index != gpu) { continue; }
        if (const auto key = product::gpu_key_of_bus(stat.bus_id)) { inputs.gpu = *key; }
    }
    const product::LaunchPlanOptions options = product::launch_options(launch);
    inputs.options = options.options;
    inputs.ask     = options.ask;
    inputs.pins    = product::ChoicePins{
           .kv = launch.exact || (!launch.kv_dtype.empty() && launch.kv_dtype != "auto"),
           .activations = launch.activations_pinned()};
    const product::ArtifactPlan plan = product::plan_artifact(artifact, inputs);
    Json doc = product::artifact_plan_document(plan, artifact.string());
    // What a start right now would get (a second model may hold the card): the replay says it.
    if (const auto free_now = free_now_of(stats, gpu)) {
        doc["now"] = product::plan_now_json(
            product::plan_now(plan.facts, plan.inputs, plan.served ? &*plan.served : nullptr,
                              plan.inputs.pins, *free_now),
            *free_now);
    }
    return doc;
}

Json App::plan_json(const std::string& repo, const std::string& hub, int gpu,
                    const std::string& quant, const product::LaunchAsk& launch) {
    // The plan of an idle card (cached), and what a start right now would get at the GPU's free
    // memory (product::plan_now, each request: another process's share moves).
    Json doc = plan_json_idle(repo, hub, gpu, quant, launch);
    const auto free_now = free_now_of(gpus_->snapshot(), gpu);
    std::shared_ptr<const PlanNowContext> now;
    {
        std::lock_guard lock(sizes_mutex_);
        if (const auto it = plan_now_.find(plan_key(repo, hub, gpu, quant, launch));
            it != plan_now_.end()) {
            now = it->second;
        }
    }
    if (free_now && now) {
        doc["now"] = product::plan_now_json(
            product::plan_now(now->facts, now->inputs, now->served ? &*now->served : nullptr,
                              now->pins, *free_now),
            *free_now);
    }
    return doc;
}

std::string App::plan_key(const std::string& repo, const std::string& hub, int gpu,
                          const std::string& quant, const product::LaunchAsk& launch) const {
    return hub_key(repo, hub) + "|" + std::to_string(gpu) + "|" + quant + "|" + launch_key(launch) +
           "|" + convert::conversions_stamp(hub_root(), repo);
}

Json App::plan_json_idle(const std::string& repo, const std::string& hub, int gpu,
                         const std::string& quant, const product::LaunchAsk& launch) {
    const auto generation = hub_generation();
    const Json list   = listing_of(repo, hub);
    const auto device = plan_device(gpus_->snapshot(), gpu);
    if (!device) { throw std::runtime_error("no GPU " + std::to_string(gpu) + " to plan for"); }
    const std::string key = plan_key(repo, hub, gpu, quant, launch);
    {
        std::lock_guard lock(sizes_mutex_);
        if (const auto it = plans_.find(key); it != plans_.end()) { return it->second; }
    }
    if (!quant.empty() && std::none_of(list.begin(), list.end(), [&](const Json& q) {
            return !q.value("drafter", false) && q.value("quant", std::string()) == quant;
        })) {
        throw std::invalid_argument("no quant " + quant + " in " + repo);
    }
    const auto facts = plan_facts(repo, list, hub);
    if (!facts) {
        throw std::runtime_error(repo + ": the model's shape could not be read (offline, or no "
                                        "GGUF header or config.json)");
    }
    Json config;
    {
        std::lock_guard lock(sizes_mutex_);
        if (const auto it = configs_.find(hub_key(repo, hub)); it != configs_.end()) {
            config = it->second;
        }
    }
    std::vector<product::QuantListing> listing = quant_listing(list, config);
    const bool gguf                                  = std::any_of(listing.begin(), listing.end(),
                                                                   [](const auto& q) { return !q.drafter && q.format == "gguf"; });
    const product::LaunchPlanOptions options         = product::launch_options(launch);
    product::DryRunInputs inputs;
    inputs.device         = *device;
    // The GPU key a start on that device reads (its reported clocks and DRAM rate), else one
    // from the facts alone: the choice prices the card as the server will.
    inputs.gpu = product::gpu_key_of(*device);
    for (const auto& stat : gpus_->snapshot()) {
        if (stat.index != gpu) { continue; }
        if (const auto key = product::gpu_key_of_bus(stat.bus_id)) { inputs.gpu = *key; }
    }
    inputs.options        = options.options;
    inputs.ask            = options.ask;
    // The launch pins the KV codec and the activations the user chose; the rest is chosen.
    // The exact preset also pins BF16 KV (launch_options resolves it, the launch passes it).
    inputs.pins = product::ChoicePins{.kv          = launch.exact || (!launch.kv_dtype.empty() && launch.kv_dtype != "auto"),
                                      .activations = launch.activations_pinned()};
    // ninferno-ui's launch passes no --with-drafters: the server attaches a GGUF model's separate
    // drafters when the choice drafts with one (choice/DECISIONS 45) and starts a quantization
    // converted here with its drafters from that conversion; product::PlannedSubjects plans both as
    // the server's plan does.
    // The commit the listing was read at: a sibling quantization's measurement counts from it.
    const std::string commit = list.empty() || !list.front().is_object()
                                   ? std::string()
                                   : list.front().value("commit", std::string());
    product::PlannedSubjects::ConversionOf conversion_of;
    if (gguf) {
        conversion_of = [root = hub_root(), repo](const std::string& label) {
            return convert::cached_conversion_with_drafters(root, repo, "", label);
        };
    }
    // The quantizations' output heads, only where the converter would add the default proposal
    // head (product::default_proposal_possible: local facts): 1 MiB windows around the parsed
    // header's tensor table end, read together beside the MTP reads and kept per commit.
    std::future<std::map<std::string, std::uint64_t>> heads_read;
    if (gguf && !options_.offline && facts->model.gguf_data_offset != 0 &&
        product::default_proposal_possible(
            product::PlannedTarget{.repo = repo, .facts = *facts, .gguf = gguf, .commit = commit},
            inputs.ask)) {
        std::vector<std::string> quants;
        for (const auto& q : listing) {
            if (!q.drafter && q.format == "gguf" && (quant.empty() || q.quant == quant)) {
                quants.push_back(q.quant);
            }
        }
        convert::FetchOptions fetch;
        fetch.hub = hub;
        if (!options_.hub_cache.empty()) { fetch.cache = options_.hub_cache; }
        heads_read = std::async(
            std::launch::async,
            [spec = commit.empty() ? repo : repo + "@" + commit, fetch, quants,
             facts = facts->model, tied = facts->shape.tied_embeddings] {
                return convert::gguf_quants_head_bytes(spec, fetch, quants, facts.gguf_data_offset,
                                                       facts.hidden, facts.vocab, tied);
            });
    }
    // Each GGUF quantization's own MTP layers (the facts are one file's: kv_json's), read once.
    product::PlannedSubjects::MtpOf mtp_of;
    if (gguf && !options_.offline) {
        // The table's every quantization, read together; a quantization named, its own alone.
        std::vector<std::string> quants;
        for (const auto& q : listing) {
            if (!q.drafter && q.format == "gguf" && (quant.empty() || q.quant == quant)) {
                quants.push_back(q.quant);
            }
        }
        gguf_mtps(repo, hub, commit, quants); // read once
        mtp_of = [this, repo, hub, commit](const std::string& label) {
            return gguf_mtp(repo, hub, commit, label);
        };
    }
    // Kept with the "now" context (plan_now_): the inputs' subjects point into it.
    const auto subjects_kept = std::make_shared<product::PlannedSubjects>(
        product::PlannedTarget{.repo = repo, .facts = *facts, .gguf = gguf, .commit = commit},
        inputs.gpu, std::move(conversion_of), std::filesystem::path{}, std::move(mtp_of));
    product::PlannedSubjects& subjects = *subjects_kept;
    std::string weights; // the named quantization's (none for the table: each row its own)
    for (const auto& q : listing) {
        if (!q.drafter && q.quant == quant) { weights = q.weights; }
    }
    const product::SpeculationSubject& subject = subjects.of(quant, weights);
    inputs.subject    = &subject;
    inputs.subject_of = subjects.lookup(); // each quantization's own (the lab's pairs are of weights)
    // The drafters a start attaches beside the target, their weights and a DFlash drafter's
    // state: the server's plan reads the same (product::attached_drafter_memory).
    {
        convert::FetchOptions fetch;
        fetch.hub = hub;
        if (!options_.hub_cache.empty()) { fetch.cache = options_.hub_cache; }
        const product::AttachedDrafterMemory drafter = product::attached_drafter_memory(
            subject, listing, facts->model.model_type, facts->model.hidden,
            options_.offline ? nullptr : &fetch);
        inputs.drafter_bytes                = drafter.bytes;
        inputs.proposal_planned       = drafter.proposal_planned;
        inputs.proposal_drafter_bytes = drafter.proposal_drafter_bytes;
        inputs.drafter_state = drafter.state;
    }
    // Each GGUF quantization's own output head where the default proposal head is priced (Codex on
    // #170: the plan reads them, the UI reused one file's), read beside the MTP reads below.
    if (heads_read.valid() && inputs.proposal_planned) {
        const auto heads = heads_read.get();
        for (auto& q : listing) {
            // A head not read (0) keeps what the listing knows.
            if (const auto it = heads.find(q.quant); it != heads.end() && it->second != 0) {
                q.head_bytes = it->second;
            }
        }
    }
    Json doc;
    product::PlanSourceFacts planned_at = *facts; // the weights the choice is made at
    std::string planned_label           = quant;
    std::uint32_t launch_context        = 0; // the ceiling a start of the shown row passes
    if (quant.empty()) {
        const product::QuantPlan plan   = product::plan_quants(*facts, listing, inputs);
        const product::QuantRow* chosen = plan.chosen ? &plan.rows[*plan.chosen] : nullptr;
        product::PlannedSpeculation planned;
        product::PlanSourceFacts at = *facts;
        if (chosen != nullptr) {
            planned = chosen->planned;
            at      = product::with_weights(*facts, chosen->weight_bytes, chosen->basis);
        } else {
            planned = product::plan_model(*facts, inputs);
            // Nothing fits or runs: a pinned mode the family does not run says so.
            product::refuse_unrun_pins(*facts, inputs, planned);
        }
        doc        = product::dry_plan_document(at, repo, "hub", inputs, planned, &plan);
        planned_at = at;
        if (chosen != nullptr) {
            planned_label  = chosen->listing.quant;
            launch_context = chosen->launch_context;
        }
    } else {
        // One quantization: its row of the table (the same states, sizes and codes), and its plan.
        std::vector<product::QuantListing> one;
        for (const auto& q : listing) {
            if (!q.drafter && q.quant == quant) { one.push_back(q); }
        }
        const product::QuantPlan plan = product::plan_quants(*facts, one, inputs);
        const product::QuantRow& row  = plan.rows.front();
        launch_context                = row.launch_context;
        const product::PlanSourceFacts at =
            product::with_weights(*facts, row.weight_bytes, row.basis);
        product::PlannedSpeculation planned =
            row.planned.plan.planned || !row.planned.plan.refused_code.empty()
                ? row.planned
                : product::plan_model(at, product::inputs_for(inputs, row.listing));
        // A pin the model does not run keeps its verdict (kv_not_run, activations_not_run).
        product::refuse_unrun_pins(at, product::inputs_for(inputs, row.listing), planned);
        doc = product::dry_plan_document(at, repo + ":" + quant, "hub", inputs, planned, &plan);
        planned_at = at;
    }
    // The configuration choice at those weights (product/plan/plan_choice.h): what the launch
    // leaves automatic is chosen there, and what every other value would change. The launch pins
    // the KV codec the user chose and --activations exact.
    product::DryRunInputs at_label = inputs;
    for (const auto& q : listing) {
        if (!q.drafter && q.quant == planned_label) { at_label = product::inputs_for(inputs, q); }
    }
    const auto served =
        product::choose_at_weights(planned_at, at_label, inputs.pins, planned_label);
    if (served) { doc["choice"] = product::choice_json(*served); }
    auto now      = std::make_shared<PlanNowContext>();
    now->subjects = subjects_kept;
    now->facts    = planned_at;
    // A start passes the row's ceiling (several streams each get their own context): now is
    // planned under it, as the idle plan is.
    now->inputs   = launch_context != 0 && inputs.options.max_context_automatic
                        ? product::under_context_cap(at_label, launch_context)
                        : at_label;
    now->served   = served;
    now->pins     = inputs.pins;
    std::lock_guard lock(sizes_mutex_);
    if (plans_.size() >= kMaxKeptPlans) {
        plans_.clear();
        plan_now_.clear();
    }
    if (generation == hub_generation_) {
        plans_[key]     = doc;
        plan_now_[key] = std::move(now);
    }
    return doc;
}

Json App::quants_json(const std::string& repo, int gpu, const std::string& hub,
                      const product::LaunchAsk& launch) {
    const Json list = listing_of(repo, hub);
    // Fit for the GPU asked about.
    double memory_gib = 0;
    if (gpu >= 0) {
        for (const auto& stat : gpus_->snapshot()) {
            if (stat.index == gpu) { memory_gib = to_gib(stat.memory_total); }
        }
    }
    Json out = Json::array();
    for (auto entry : list) {
        const double gib = to_gib(entry.value("bytes", 0.0));
        if (memory_gib > 0 && !entry.value("drafter", false)) {
            const double need = gib + kReserveGib;
            entry["fit"] = need <= memory_gib * 0.95 ? "fits" : need <= memory_gib ? "tight" : "no";
        }
        entry["size_gib"] = std::round(gib * 100) / 100;
        // Above 8 bits per weight (BF16, F16) is not offered to run: the engine serves <= 8.
        entry["supported"] = entry.value("drafter", false) || entry.value("bits", 0) <= 8;
        out.push_back(std::move(entry));
    }
    // Models before sidecars, each by size.
    std::stable_sort(out.begin(), out.end(), [](const Json& a, const Json& b) {
        if (a.value("drafter", false) != b.value("drafter", false)) {
            return !a.value("drafter", false);
        }
        return a.value("bytes", 0.0) < b.value("bytes", 0.0);
    });
    // The one to pick: the 4-bit quantization of best quality that fits (the dynamic UD-Q4_K_XL,
    // then Q4_K_M, as llama.cpp's -hf defaults to), else the largest that fits.
    const auto fits = [](const Json& q) {
        return !q.value("drafter", false) && q.value("supported", false) &&
               q.value("fit", "fits") != "no";
    };
    std::optional<std::size_t> pick;
    for (const char* name :
         {"UD-Q4_K_XL", "Q4_K_M", "UD-Q4_K_M", "Q4_K_S", "MXFP4", "IQ4_XS", "Q4_0", ""}) {
        for (std::size_t i = 0; i < out.size() && !pick; ++i) {
            if (fits(out[i]) && out[i].value("quant", std::string()) == name &&
                (*name != '\0' || out[i].value("format", std::string()) == "safetensors")) {
                pick = i;
            }
        }
    }
    for (std::size_t i = out.size(); i-- > 0 && !pick;) {
        if (fits(out[i])) { pick = i; }
    }
    // The largest that fits (product/plan/plan_quants.h, the same rule ninferno-serve applies
    // before it fetches): each row's fit from the plan without the weights at the launch's
    // context x streams on an idle card of this GPU; the name order above stays only where the
    // shape cannot be read.
    Json plan = nullptr;
    if (gpu >= 0) {
        try {
            plan = plan_json_idle(repo, hub, gpu, "", launch);
        } catch (const std::exception&) {}
    }
    if (plan.is_object() && plan.contains("quants")) {
        pick.reset();
        for (auto& entry : out) {
            for (const auto& row : plan["quants"]) {
                if (row.value("quant", std::string()) != entry.value("quant", std::string()) ||
                    entry.value("drafter", false)) {
                    continue;
                }
                const std::string state = row.value("state", std::string("no"));
                entry["fit"]            = state == "own"                            ? "fits"
                                          : state == "shared" || state == "desktop" ? "tight"
                                          : state == "unknown"                      ? "unknown"
                                                                                    : "no";
                entry["planned"]        = {
                    {"state", state},
                    {"fits", row["fits"]},
                    {"fits_shared", row["fits_shared"]},
                    {"largest", row["largest"]},
                    {"own_context", row["own_context"]},
                    {"launch_context", row["launch_context"]},
                    {"bits_per_weight", row["bits_per_weight"]},
                    {"weights_basis", row["weights_basis"]},
                    {"context", row["plan"].is_object() ? row["plan"].value("context", 0) : 0}};
                if (row.value("largest", false)) {
                    pick = static_cast<std::size_t>(&entry - &out[0]);
                }
            }
        }
    }
    if (pick) { out[*pick]["recommended"] = true; }
    auto kv         = kv_json(repo, list, hub);
    const auto arch = kv.value("architecture", std::string());
    // A GGUF repository's architecture is its general.architecture (no weights of another format).
    const bool gguf = std::none_of(list.begin(), list.end(), [](const Json& q) {
        return !q.value("drafter", false) && q.value("format", std::string()) != "gguf";
    });
    // The repository's whole decision (its task, its weights' format, a token), as the search's;
    // the architecture alone where that could not be read.
    const Json decided = arch_json(repo, hub);
    return {
        {"repo", repo},
        {"hub", hub},
        {"architecture", arch},
        {"support", decided.contains("support")
                        ? decided["support"]
                        : Json(support_level(
                              convert::model_support(facts_of_architecture(arch, gguf))))},
        {"note", decided.value("note", std::string())},
        {"quants", out},
        {"rule", plan.is_object() ? plan.value("rule", std::string()) : std::string()},
        {"rule_code", plan.is_object() ? plan.value("rule_code", std::string()) : std::string()},
        {"rule_params",
         plan.is_object() ? plan.value("rule_params", Json::object()) : Json::object()},
        {"pick", plan.is_object() && plan.contains("pick") ? plan["pick"] : Json(nullptr)},
        {"kv", std::move(kv)}};
}

Json App::state_json() {
    Json gpus = Json::array();
    for (const auto& stat : gpus_->snapshot()) { gpus.push_back(stat.to_json()); }
    Json processes = Json::array();
    for (const auto& info : processes_.list()) { processes.push_back(info.to_json()); }
    return {{"gpus", gpus},
            {"gpu_error", gpus_->error()},
            {"processes", processes},
            {"listen", options_.access.listen},
            {"chat_ui", !options_.chat_ui.empty()},
            {"update", update_json()},
            {"exposed", !is_loopback_host(host_of_authority(options_.access.listen))}};
}

Json App::catalog_json(int gpu, const product::LaunchAsk& launch) {
    TargetGpu target;
    for (const auto& stat : gpus_->snapshot()) {
        if (stat.index == gpu) { target = target_of(stat); }
    }
    const auto table = sizes();
    std::map<std::string, Json> listings;
    {
        std::lock_guard lock(sizes_mutex_);
        listings = listings_;
    }
    Json models        = Json::array();
    const auto catalog = this->catalog();
    for (const auto& listed : catalog->models()) {
        // The plan without the weights for a repository whose shape is read already (no network
        // here: the catalog is drawn often).
        // One plan per repository at the user's context x streams: its rows' states, and the
        // rule's pick (product/plan/plan_quants.h) as the quant the catalog adds.
        const auto doc_of = [&](const std::string& repo) -> std::optional<Json> {
            {
                std::lock_guard lock(sizes_mutex_);
                if (!plan_facts_.contains(hub_key(repo, "huggingface"))) { return std::nullopt; }
            }
            try {
                return plan_json_idle(repo, "huggingface", gpu, "", launch);
            } catch (const std::exception&) { return std::nullopt; }
        };
        const auto fits = [&](const std::string& repo, const Json& q) -> std::optional<bool> {
            const auto doc = doc_of(repo);
            if (!doc) { return std::nullopt; }
            for (const auto& row : (*doc)["quants"]) {
                if (row.value("quant", std::string()) != q.value("quant", std::string())) {
                    continue;
                }
                const std::string state = row.value("state", std::string());
                if (state == "unknown") { return std::nullopt; }
                return state == "own" || state == "shared" || state == "desktop";
            }
            return std::nullopt;
        };
        const auto pick = [&](const std::string& repo) -> std::optional<std::string> {
            const auto doc = doc_of(repo);
            if (!doc || !(*doc)["pick"].is_object()) { return std::nullopt; }
            return (*doc)["pick"].value("quant", std::string());
        };
        const auto model = with_fitting_quant(listed, target, listings, fits, pick);
        auto json        = to_json(estimate(*catalog, model, target, table));
        attach_speculation(json, speculation_plan(model.id, target));
        models.push_back(std::move(json));
    }
    // Recommended first; among them measured before estimated (an estimate scaled across cards is
    // the weaker claim); then by the preferred source's speed.
    const auto preferred = [](const Json& m) -> const Json* {
        return m["preferred"].is_null() ? nullptr
                                        : &m["sources"][m["preferred"].get<std::size_t>()];
    };
    std::stable_sort(models.begin(), models.end(), [&](const Json& a, const Json& b) {
        if (a.value("recommended", false) != b.value("recommended", false)) {
            return a.value("recommended", false);
        }
        const Json* pa = preferred(a);
        const Json* pb = preferred(b);
        const bool ma  = pa != nullptr && pa->value("tps_basis", std::string()) == "measured";
        const bool mb  = pb != nullptr && pb->value("tps_basis", std::string()) == "measured";
        if (ma != mb) { return ma; }
        return (pa ? pa->value("tps", 0.0) : 0.0) > (pb ? pb->value("tps", 0.0) : 0.0);
    });
    std::string source;
    {
        std::lock_guard lock(catalog_mutex_);
        source = catalog_source_;
    }
    return {{"gpu", gpu},
            {"gpu_name", target.name},
            {"gpu_chip", target.chip},
            {"models", models},
            {"catalog_source", source}};
}

Json App::moved_json() {
    Json out = Json::array();
    if (options_.offline) { return {{"models", out}, {"offline", true}}; }
    const auto root = hub_root();
    const std::int64_t now = std::time(nullptr);
    std::error_code ec;
    for (const auto& repo_dir : fs::directory_iterator(root, ec)) {
        const auto repo = hub_repo_of_directory(repo_dir.path().filename().string());
        if (repo.empty() || !fs::is_directory(repo_dir.path() / "ninferno", ec)) { continue; }
        // The branches (refs/<name>) and the commit each named when it was fetched.
        for (const auto& ref_file : fs::directory_iterator(repo_dir.path() / "refs", ec)) {
            if (!ref_file.is_regular_file(ec)) { continue; }
            std::string commit;
            std::ifstream(ref_file.path()) >> commit;
            const auto dir = repo_dir.path() / "ninferno" / commit;
            // A conversion of the commit the ref named, its download removed or still here: a
            // start keeps it either way (convert::acquire_model).
            if (commit.size() != 40 || !fs::is_directory(dir, ec)) { continue; }
            const auto ref = ref_file.path().filename().string();
            const auto key = repo + "@" + ref;
            std::string hub_commit;
            {
                std::lock_guard lock(moved_mutex_);
                const auto it = hub_commits_.find(key);
                if (it != hub_commits_.end() && now - it->second.first < 3600) {
                    hub_commit = it->second.second;
                } else {
                    try {
                        convert::FetchOptions fetch;
                        fetch.cache = root;
                        hub_commit  = convert::current_commit(key, fetch).value_or("");
                    } catch (const std::exception&) {} // unchecked: nothing to say
                    hub_commits_[key] = {now, hub_commit};
                }
            }
            if (hub_commit.empty() || hub_commit == commit) { continue; }
            Json artifacts = Json::array();
            for (const auto& entry : fs::directory_iterator(dir, ec)) {
                if (entry.path().extension() == ".ninferno" && !facts(entry.path()).drafter) {
                    artifacts.push_back(entry.path().string());
                }
            }
            out.push_back({{"repo", repo},
                           {"ref", ref},
                           {"commit", commit},
                           {"hub_commit", hub_commit},
                           {"accepted", convert::converted_accepted(dir, ref, hub_commit)},
                           {"artifacts", artifacts}});
        }
    }
    Json updates = Json::array();
    {
        std::lock_guard lock(moved_mutex_);
        for (const auto& [key, update] : updates_) {
            Json percent = Json::array();
            for (const auto& part : update.parts) {
                ProcessInfo info;
                percent.push_back(processes_.find(part.prepare, info) ? info.percent : -1);
            }
            updates.push_back({{"repo", update.repo},
                               {"ref", update.ref},
                               {"commit", update.old_commit},
                               {"hub_commit", update.hub_commit},
                               {"state", update.state},
                               {"error", update.error},
                               {"prepares", percent}});
        }
    }
    return {{"models", out}, {"updates", updates}};
}

Json App::answer_moved(const std::string& repo, const std::string& ref, const std::string& commit,
                       const std::string& action) {
    static const std::regex hex40("[0-9a-f]{40}");
    static const std::regex ref_name("[A-Za-z0-9._-]+");
    if (!convert::is_repository_spec(repo) || repo.find('@') != std::string::npos ||
        !std::regex_match(commit, hex40) || !std::regex_match(ref, ref_name)) {
        throw std::invalid_argument("bad repo, ref or commit");
    }
    const auto dir = hub_root() / hub_repo_directory(repo) / "ninferno" / commit;
    std::string hub_commit;
    {
        std::lock_guard lock(moved_mutex_);
        const auto it = hub_commits_.find(repo + "@" + ref);
        if (it != hub_commits_.end()) { hub_commit = it->second.second; }
    }
    std::error_code ec;
    if (!fs::is_directory(dir, ec) || hub_commit.empty() || hub_commit == commit) {
        throw std::invalid_argument(repo + "@" + ref + " is not known to have moved from " +
                                    commit.substr(0, 12));
    }
    if (action == "accept") {
        convert::accept_converted(dir, ref, hub_commit);
        return {{"accepted", true}};
    }
    if (action != "refetch") { throw std::invalid_argument("action is accept or refetch"); }
    const auto key = repo + "@" + ref;
    std::lock_guard lock(moved_mutex_);
    if (const auto it = updates_.find(key);
        it != updates_.end() && (it->second.state == "downloading" || it->second.state == "switching")) {
        throw std::invalid_argument(key + " is already being updated");
    }
    ModelUpdate update;
    update.repo       = repo;
    update.ref        = ref;
    update.old_commit = commit;
    update.hub_commit = hub_commit;
    update.state      = "downloading";
    Json ids = Json::array();
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (entry.path().extension() != ".ninferno" || facts(entry.path()).drafter) { continue; }
        LaunchRequest request;
        request.model         = ref == "main" ? repo : repo + "@" + ref;
        const auto name       = artifact_name(entry.path().filename().string());
        request.quant         = name.quant;
        request.recipe        = name.recipe; // the replaced conversion's, official or not
        request.prepare_only  = true;
        request.refetch       = true;
        request.keep_replaced = true; // removed here once nothing serves it
        request.vision        = facts(entry.path()).vision; // a Vision conversion stays one
        request.gpu           = -1;
        const auto id = processes_.start(make_launch(options_, request));
        update.parts.push_back({.old_artifact = entry.path(), .new_artifact = {}, .prepare = id});
        ids.push_back(id);
    }
    if (update.parts.empty()) { throw std::invalid_argument("no converted model of " + key); }
    updates_[key] = std::move(update);
    return {{"ids", ids}};
}

namespace {

bool ended(const std::string& state) { return state == "exited" || state == "failed"; }

bool same_file(const std::string& a, const fs::path& b) {
    std::error_code ec;
    return !a.empty() && fs::equivalent(fs::path(a), b, ec) && !ec;
}

// The requests a ninferno-serve on `port` has in flight (its /health); nullopt when unanswered.
std::optional<long> active_requests(int port) {
    if (port <= 0) { return std::nullopt; }
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(std::chrono::milliseconds(300));
    client.set_read_timeout(std::chrono::seconds(1));
    const auto res = client.Get("/health");
    if (!res || res->status != 200) { return std::nullopt; }
    try {
        return Json::parse(res->body).at("active_requests").get<long>();
    } catch (const std::exception&) { return std::nullopt; }
}

} // namespace

namespace {

// `spec` started on `artifact` by its path: the model argument (argv[1]) is the path, the hub's
// --quant goes (the path names one quantization); `fresh` (another artifact than the one `spec`
// ran) also drops --spec-profile, a tune keyed to the old artifact's id that ninferno-serve refuses
// for another.
LaunchSpec on_artifact(LaunchSpec spec, const fs::path& artifact, bool fresh) {
    std::vector<std::string> argv;
    for (std::size_t i = 0; i < spec.argv.size(); ++i) {
        const std::string& arg = spec.argv[i];
        if (i == 1) {
            argv.push_back(artifact.string());
        } else if (arg == "--quant" || (fresh && arg == "--spec-profile")) {
            ++i; // and its value
        } else {
            argv.push_back(arg);
        }
    }
    spec.argv  = std::move(argv);
    spec.model = artifact.string();
    return spec;
}

} // namespace

void App::advance_updates() {
    std::lock_guard lock(moved_mutex_);
    for (auto& [key, update] : updates_) {
        if (update.state == "downloading") {
            // Ended only when every prepare has: a failed one leaves the others to finish, so no
            // retry starts a second converter beside one still writing.
            bool all = true;
            std::string failure;
            for (auto& part : update.parts) {
                ProcessInfo info;
                if (!processes_.find(part.prepare, info)) {
                    if (failure.empty()) {
                        failure = "the download's process is gone; the old conversion stays";
                    }
                    continue;
                }
                if (!ended(info.state)) {
                    all = false;
                } else if (info.state == "failed" || info.exit_code != 0 || info.artifact.empty()) {
                    if (failure.empty()) {
                        failure = "the new commit's download or conversion failed (" +
                                  info.last_line + "); the old conversion stays";
                    }
                } else {
                    part.new_artifact = info.artifact;
                }
            }
            if (!all) { continue; }
            if (!failure.empty()) {
                update.state = "failed";
                update.error = failure;
                continue;
            }
            // The models serving an old conversion, each to restart once on its new one.
            for (const auto& info : processes_.list()) {
                if (info.spec.kind != "serve" || ended(info.state) || info.state == "stopping") {
                    continue;
                }
                for (const auto& part : update.parts) {
                    if (!same_file(info.artifact, part.old_artifact)) { continue; }
                    ModelUpdate::Switch next;
                    next.old_serve = info.id;
                    // Both starts name their conversion by its path: a start by the hub id would
                    // resolve to whatever the ref names now (after the update, the new commit),
                    // so a revert would start the new conversion again.
                    next.back = on_artifact(info.spec, part.old_artifact, false);
                    next.next = on_artifact(info.spec, part.new_artifact, true);
                    update.switches.push_back(std::move(next));
                }
            }
            update.state = "switching";
        }
        if (update.state != "switching") { continue; }
        bool all = true;
        for (auto& sw : update.switches) {
            if (sw.done) { continue; }
            all = false;
            if (sw.serve == 0) {
                ProcessInfo old;
                const bool found = processes_.find(sw.old_serve, old);
                if (!sw.stopping) {
                    // Stopped by the user meanwhile: nothing to restart.
                    if (!found || ended(old.state) || old.state == "stopping") {
                        sw.done = true;
                        continue;
                    }
                    // Restarted when it serves no request (its /health): not in a conversation.
                    if (old.state != "ready" || active_requests(old.spec.port).value_or(1) != 0) {
                        continue;
                    }
                    sw.stopping = true;
                    // Owned, and joined before the pool goes (~App): a stop outlives no App.
                    update_stoppers_.emplace_back([this, id = sw.old_serve] { processes_.stop(id); });
                    continue;
                }
                if (found && !ended(old.state)) { continue; }
                try {
                    sw.serve = processes_.start(sw.next);
                } catch (const std::exception& error) {
                    update.error = sw.next.model + " did not start on the new conversion (" +
                                   error.what() + "); back on the old one";
                    sw.reverted = true;
                    try {
                        sw.serve = processes_.start(sw.back);
                    } catch (const std::exception& again) {
                        update.error += std::string("; it did not start on the old one either (") +
                                        again.what() + ")";
                        sw.done = true;
                    }
                }
                continue;
            }
            ProcessInfo now;
            if (!processes_.find(sw.serve, now) || now.state == "ready") {
                sw.done = true;
                continue;
            }
            if (!ended(now.state)) { continue; }
            if (sw.reverted) {
                // Not again: the old conversion did not come back up either.
                update.error += "; it did not start on the old one either (" + now.last_line + ")";
                sw.done = true;
                continue;
            }
            update.error = sw.next.model + " did not start on the new conversion (" +
                           now.last_line + "); back on the old one";
            sw.reverted = true;
            try {
                sw.serve = processes_.start(sw.back);
            } catch (const std::exception& again) {
                update.error += std::string("; it did not start on the old one either (") +
                                again.what() + ")";
                sw.done = true;
            }
        }
        if (!all) { continue; }
        if (std::any_of(update.switches.begin(), update.switches.end(),
                        [](const ModelUpdate::Switch& sw) { return sw.reverted; })) {
            // The old conversion stays: a model serves it again. The new one stays beside.
            update.state = "failed";
            continue;
        }
        for (const auto& part : update.parts) {
            try {
                (void)delete_artifact(part.old_artifact);
            } catch (const std::exception& error) {
                update.error = "the old conversion " + part.old_artifact.filename().string() +
                               " was kept: " + error.what();
            }
        }
        update.state = "done"; // switched; an old conversion kept is said in error
    }
}

fs::path App::hub_root() const {
    return options_.hub_cache.empty() ? convert::default_hub_cache() : options_.hub_cache;
}

ArtifactFacts App::facts(const fs::path& artifact) {
    std::error_code ec;
    const auto stamp = fs::last_write_time(artifact, ec);
    std::lock_guard lock(kinds_mutex_);
    auto it = kinds_.find(artifact.string());
    if (it == kinds_.end() || it->second.first != stamp) {
        it = kinds_
                 .insert_or_assign(artifact.string(),
                                   std::pair{stamp, read_artifact_facts(artifact)})
                 .first;
    }
    return it->second.second;
}

namespace {

std::uint64_t tree_bytes(const fs::path& dir) {
    std::uint64_t total = 0;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) { return 0; }
    for (auto it = fs::recursive_directory_iterator(
             dir, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) { break; }
        if (it->is_regular_file(ec) && !it->is_symlink(ec)) { total += it->file_size(ec); }
    }
    return total;
}

double gib2(std::uint64_t bytes) { return std::round(to_gib(bytes) * 100) / 100; }

std::string iso_time(fs::file_time_type stamp) {
    // clock_cast: MSVC's file_clock has no to_sys.
    return utc_iso8601(std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        std::chrono::clock_cast<std::chrono::system_clock>(stamp)));
}

} // namespace

Json App::local_json() {
    const fs::path root = hub_root();
    std::set<std::string> running;
    for (const auto& info : processes_.list()) {
        if (info.state == "exited" || info.state == "failed") { continue; }
        running.insert(info.artifact.empty() ? info.spec.model : info.artifact);
    }
    Json out = Json::array();
    std::error_code ec;
    const auto add = [&](const std::string& repo, const std::string& publisher,
                         const fs::path& path, const fs::path& repo_dir) {
        const auto f = facts(path);
        if (!f.readable || f.drafter) { return; }
        const auto name   = artifact_name(path.filename().string());
        std::string title = repo.empty() ? name.stem : repo.substr(repo.find('/') + 1);
        for (const char* suffix : {"-GGUF", "-gguf", "-NVFP4", "-nvfp4", "-MXFP4", "-FP8"}) {
            if (title.size() > std::strlen(suffix) && title.ends_with(suffix)) {
                title.resize(title.size() - std::strlen(suffix));
            }
        }
        const std::uint64_t original = repo_dir.empty() ? 0 : tree_bytes(repo_dir / "blobs");
        out.push_back({{"repo", repo},
                       {"publisher", publisher},
                       {"title", title},
                       {"artifact", path.string()},
                       {"name", path.filename().string()},
                       {"arch", short_architecture(f.architecture)},
                       {"params", params_label(repo.empty() ? name.stem : repo + " " + name.stem)},
                       {"quant", name.quant},
                       {"vision", f.vision},
                       {"mtp", f.mtp || fs::exists(path.string() + ".speculation.json", ec)},
                       {"size_gib", gib2(fs::file_size(path, ec))},
                       {"modified", iso_time(fs::last_write_time(path, ec))},
                       {"original_gib", gib2(original)},
                       {"running", running.contains(path.string())},
                       // A tune made on this machine (spec-tune wrote it): a start names it only
                       // when it asks for it (spec_profile).
                       {"own_profile", fs::exists(spec::default_profile_path(path), ec)}});
    };
    // Converted hub models: <cache>/models--org--name/ninferno/<commit>/*.ninferno.
    if (fs::is_directory(root, ec)) {
        for (const auto& repo_dir : fs::directory_iterator(root, ec)) {
            const auto repo = hub_repo_of_directory(repo_dir.path().filename().string());
            if (repo.empty()) { continue; }
            const auto converted = repo_dir.path() / "ninferno";
            if (!fs::is_directory(converted, ec)) { continue; }
            for (const auto& entry : fs::recursive_directory_iterator(converted, ec)) {
                if (entry.is_regular_file(ec) && entry.path().extension() == ".ninferno") {
                    add(repo, repo.substr(0, repo.find('/')), entry.path(), repo_dir.path());
                }
            }
        }
    }
    // Artifacts in the --models-dir directories, two levels deep.
    for (const auto& dir : options_.model_dirs) {
        for (auto it = fs::recursive_directory_iterator(
                 dir, fs::directory_options::skip_permission_denied, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) { break; }
            if (it.depth() >= 2) { it.disable_recursion_pending(); }
            const auto& path = it->path();
            if (it->is_regular_file(ec) && path.extension() == ".ninferno") {
                add("", path.parent_path().filename().string(), path, fs::path());
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const Json& a, const Json& b) {
        return a.value("modified", std::string()) > b.value("modified", std::string());
    });
    double total = 0, originals = 0;
    for (const auto& m : out) {
        total += m.value("size_gib", 0.0);
        originals += m.value("original_gib", 0.0);
    }
    return {{"root", root.string()},
            {"models", out},
            {"total_gib", std::round(total * 100) / 100},
            {"originals_gib", std::round(originals * 100) / 100},
            {"keep_originals", options_.keep_originals}};
}

fs::path App::local_artifact(const std::string& repo, const std::string& quant) {
    std::error_code ec;
    const auto dir = hub_root() / hub_repo_directory(repo) / "ninferno";
    if (!fs::is_directory(dir, ec)) { return {}; }
    for (const auto& entry : fs::recursive_directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec) || entry.path().extension() != ".ninferno") { continue; }
        const auto f = facts(entry.path());
        if (!f.readable || f.drafter) { continue; }
        if (quant.empty() || artifact_name(entry.path().filename().string()).quant == quant) {
            return entry.path();
        }
    }
    return {};
}

std::uint64_t App::delete_artifact(const fs::path& requested) {
    std::error_code ec;
    const auto path = fs::weakly_canonical(requested, ec);
    if (ec || !deletable_artifact(path, hub_root(), options_.model_dirs)) {
        throw std::invalid_argument("not a converted model ninferno-ui manages: " +
                                    requested.string());
    }
    for (const auto& info : processes_.list()) {
        if (info.state != "exited" && info.state != "failed" &&
            (info.artifact == path.string() || info.spec.model == path.string())) {
            throw std::invalid_argument("stop the model first: it is running");
        }
    }
    std::uint64_t freed = 0;
    // Its speculation manifest names the drafters converted for it (read before the sidecars go).
    std::string manifest;
    if (std::ifstream in(spec::manifest_path(path)); in) {
        manifest.assign(std::istreambuf_iterator<char>(in), {});
    }
    // The file and its sidecars (<artifact>.conversion.json, .speculation.json, ...).
    const auto prefix = path.filename().string();
    for (const auto& entry : fs::directory_iterator(path.parent_path(), ec)) {
        const auto name = entry.path().filename().string();
        if (name == prefix || name.rfind(prefix + ".", 0) == 0) {
            freed += entry.is_regular_file(ec) ? entry.file_size(ec) : 0;
            fs::remove(entry.path(), ec);
        }
    }
    // A hub model with nothing converted left: its repository directory (original included).
    const auto root = fs::weakly_canonical(hub_root(), ec);
    fs::path repo_dir;
    for (auto p = path.parent_path(); !p.empty() && p != p.parent_path(); p = p.parent_path()) {
        if (p.parent_path() == root &&
            !hub_repo_of_directory(p.filename().string()).empty()) {
            repo_dir = p;
            break;
        }
    }
    if (!repo_dir.empty()) {
        // Its commit's directory holding no conversion but the drafters its manifest names:
        // removed whole, as the converter removes a replaced conversion (acquire.cpp). Another
        // quantization of the commit keeps it.
        const auto commit_dir = path.parent_path();
        if (commit_dir.parent_path() == repo_dir / "ninferno") {
            bool others = false;
            for (const auto& entry : fs::directory_iterator(commit_dir, ec)) {
                if (entry.path().extension() == ".ninferno" &&
                    manifest.find(entry.path().filename().string()) == std::string::npos) {
                    others = true;
                }
            }
            if (!others) {
                freed += tree_bytes(commit_dir);
                fs::remove_all(commit_dir, ec);
            }
        }
        bool left = false;
        for (const auto& entry : fs::recursive_directory_iterator(repo_dir / "ninferno", ec)) {
            if (entry.is_regular_file(ec) && entry.path().extension() == ".ninferno" &&
                !facts(entry.path()).drafter) {
                left = true;
            }
        }
        if (!left) {
            freed += tree_bytes(repo_dir);
            fs::remove_all(repo_dir, ec);
        }
    }
    std::lock_guard lock(kinds_mutex_);
    kinds_.erase(path.string());
    return freed;
}

std::uint64_t App::delete_original(const std::string& repo) {
    if (!convert::is_repository_spec(repo) || repo.find('@') != std::string::npos) {
        throw std::invalid_argument("repo must be org/name");
    }
    const auto dir = hub_root() / hub_repo_directory(repo);
    std::error_code ec;
    if (!fs::is_directory(dir / "ninferno", ec)) {
        throw std::invalid_argument(repo + " has no converted model here; its download is kept");
    }
    // A download or a conversion of the repository (a prepare, or a serve not yet loaded) still
    // reads the snapshot.
    for (const auto& info : processes_.list()) {
        if (info.state != "exited" && info.state != "failed" &&
            (info.spec.kind == "prepare" || info.state == "starting") &&
            split_model_tag(info.spec.model).first == repo) {
            throw std::invalid_argument(repo + " is being downloaded or converted");
        }
    }
    std::uint64_t freed = tree_bytes(dir / "blobs");
    // The snapshot links point into blobs/; the converted models in ninferno/ stay.
    for (const char* sub : {"blobs", "snapshots", ".no_exist"}) { fs::remove_all(dir / sub, ec); }
    return freed;
}

// After a conversion (a prepare that finished, a serve that loaded what it converted), the
// downloaded original is not read again: removed unless --keep-originals.
void App::remove_converted_originals() {
    if (options_.keep_originals) { return; }
    for (const auto& info : processes_.list()) {
        const bool done =
            (info.spec.kind == "prepare" && info.state == "exited" && info.exit_code == 0) ||
            (info.spec.kind == "serve" && info.state == "ready");
        if (!done || !info.converted || originals_done_.contains(info.id)) { continue; }
        originals_done_.insert(info.id);
        const auto repo = split_model_tag(info.spec.model).first;
        if (!convert::is_repository_spec(repo)) { continue; }
        try {
            const auto freed = delete_original(repo);
            std::cerr << "ninferno-ui: removed " << repo << "'s downloaded original (" << gib2(freed)
                      << " GiB); its converted model stays\n";
        } catch (const std::exception& error) {
            std::cerr << "ninferno-ui: original of " << repo << " kept: " << error.what() << '\n';
        }
    }
}

Json App::search_json(const std::string& query, const std::string& sort,
                      const std::string& hub_name) {
    const auto generation = hub_generation();
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    const auto key = hub_name + "\n" + sort + "\n" + query;
    {
        std::lock_guard lock(sizes_mutex_);
        if (const auto it = searches_.find(key);
            it != searches_.end() && now - it->second.first < 600) {
            return it->second.second;
        }
    }
    const std::string lower = ascii_lower(query);
    Json picks = Json::array();
    for (const auto& model : catalog()->models()) {
        std::string text = model.name + " " + model.id;
        for (const auto& source : model.sources) { text += " " + source.repo; }
        text = ascii_lower(text);
        if (lower.empty() || text.find(lower) != std::string::npos) {
            Json sources = Json::array();
            for (const auto& source : model.sources) { sources.push_back(source.repo); }
            picks.push_back({{"id", model.id},
                             {"name", model.name},
                             {"kind", model.kind},
                             {"use", model.use},
                             {"repos", sources}});
        }
    }
    Json hub = Json::array();
    std::string error;
    if (!options_.offline && (query.empty() || query.size() >= 2)) {
        try {
            convert::FetchOptions fetch;
            fetch.hub = hub_name;
            for (const auto& m : convert::search_models(query, fetch, 40, sort)) {
                // Hugging Face's answer carries the facts; ModelScope's only the architecture.
                const auto support = convert::model_support(
                    m.card.is_object() ? convert::search_support_facts(m.card)
                                       : facts_of_architecture(m.architecture, m.gguf, m.pipeline,
                                                               m.library));
                hub.push_back(
                    {{"id", m.id},
                     {"downloads", m.downloads},
                     {"likes", m.likes},
                     {"updated", m.updated},
                     {"pipeline", m.pipeline},
                     {"gguf", m.gguf},
                     {"architecture", m.architecture},
                     {"context", m.context},
                     {"library", m.library},
                     {"support", support_level(support)},
                     {"note", support.reason}});
            }
        } catch (const std::exception& e) { error = e.what(); }
    }
    std::string source;
    {
        std::lock_guard lock(catalog_mutex_);
        source = catalog_source_;
    }
    Json out{{"query", query},
             {"picks", picks},
             {"hub", hub},
             {"error", error},
             {"catalog_source", source}};
    std::lock_guard lock(sizes_mutex_);
    if (error.empty() && generation == hub_generation_) { searches_[key] = {now, out}; }
    return out;
}

// The KV cache per token of the repository's models, from the GGUF metadata of its smallest
// quantization (an 8 MiB head by an HTTP range; 32 MiB when the metadata runs longer), else from
// its config.json (safetensors weights).
void App::gguf_mtps(const std::string& repo, const std::string& hub, const std::string& commit,
                    const std::vector<std::string>& quants) {
    const auto key_of = [&](const std::string& quant) {
        return hub_key(repo, hub) + "@" + commit + "|" + quant;
    };
    std::vector<std::string> missing;
    {
        std::lock_guard lock(sizes_mutex_);
        for (const auto& quant : quants) {
            if (!mtp_.contains(key_of(quant))) { missing.push_back(quant); }
        }
    }
    if (missing.empty()) { return; }
    convert::FetchOptions fetch;
    fetch.hub = hub;
    if (!options_.hub_cache.empty()) { fetch.cache = options_.hub_cache; }
    const auto read =
        convert::gguf_quants_carry_mtp(commit.empty() ? repo : repo + "@" + commit, fetch, missing);
    std::lock_guard lock(sizes_mutex_);
    for (const auto& [quant, carries] : read) {
        if (carries) { mtp_[key_of(quant)] = carries; } // a read that failed is tried again
    }
}

std::optional<bool> App::gguf_mtp(const std::string& repo, const std::string& hub,
                                  const std::string& commit, const std::string& quant) {
    const std::string key = hub_key(repo, hub) + "@" + commit + "|" + quant;
    {
        std::lock_guard lock(sizes_mutex_);
        if (const auto it = mtp_.find(key); it != mtp_.end()) { return it->second; }
    }
    convert::FetchOptions fetch;
    fetch.hub   = hub;
    fetch.quant = quant;
    if (!options_.hub_cache.empty()) { fetch.cache = options_.hub_cache; }
    const auto carries = convert::gguf_carries_mtp(commit.empty() ? repo : repo + "@" + commit, fetch);
    std::lock_guard lock(sizes_mutex_);
    if (carries) { mtp_[key] = carries; } // a read that failed is tried again
    return carries;
}

Json App::kv_json(const std::string& repo, const Json& listing, const std::string& hub) {
    const auto generation = hub_generation();
    {
        std::lock_guard lock(sizes_mutex_);
        if (const auto it = kv_.find(hub_key(repo, hub)); it != kv_.end()) { return it->second; }
    }
    std::string quant;
    double smallest = 0;
    for (const auto& q : listing) {
        if (q.value("drafter", false) || q.value("format", std::string()) != "gguf") { continue; }
        const double bytes = q.value("bytes", 0.0);
        if (quant.empty() || bytes < smallest) {
            quant    = q.value("quant", std::string());
            smallest = bytes;
        }
    }
    Json out = kv_summary(nullptr);
    if (options_.offline) { return out; }
    // A read that failed is tried again after a minute (a network blip is not the model's shape).
    {
        std::lock_guard lock(sizes_mutex_);
        if (const auto it = kv_failed_.find(hub_key(repo, hub));
            it != kv_failed_.end() &&
            std::chrono::steady_clock::now() - it->second < std::chrono::minutes(1)) {
            return out;
        }
    }
    const auto remember = [&](const Json& got, bool read) {
        std::lock_guard lock(sizes_mutex_);
        if (generation != hub_generation_) { return; }
        if (read) {
            kv_[hub_key(repo, hub)] = got;
            kv_failed_.erase(hub_key(repo, hub));
        } else {
            kv_failed_[hub_key(repo, hub)] = std::chrono::steady_clock::now();
        }
    };
    convert::FetchOptions fetch;
    fetch.hub = hub;
    if (!options_.hub_cache.empty()) { fetch.cache = options_.hub_cache; }
    if (quant.empty()) {
        // Without a quantization the fetch takes top-level safetensors weights when there are any
        // (convert/fetch.cpp choose_files), else the published artifact: the plan reads the same.
        const auto is_format = [](const char* format) {
            return [format](const Json& q) {
                return !q.value("drafter", false) && q.value("format", std::string()) == format;
            };
        };
        const bool safetensors =
            std::any_of(listing.begin(), listing.end(), is_format("safetensors"));
        const bool native =
            !safetensors && std::any_of(listing.begin(), listing.end(), is_format("ninferno"));
        bool read         = false;
        try {
            {
                // A native artifact keeps no config.json; its manifest names the base model's.
                auto source = repo;
                if (native) {
                    const auto manifest =
                        convert::fetch_repo_json(repo, fetch, "artifact-manifest.json");
                    source = manifest.value("source", Json::object())
                                 .value("base", Json::object())
                                 .value("repo_id", repo);
                }
                const Json config     = convert::fetch_config(source, fetch);
                // The largest listed weights, with their MTP part (a checkpoint's
                // model_mtp.safetensors: loaded only under MTP), as the server's plan counts them.
                std::uint64_t weights = 0, mtp = 0;
                for (const auto& q : listing) {
                    if (!q.value("drafter", false) && q.value("bytes", 0ULL) > weights &&
                        (!safetensors || q.value("format", std::string()) == "safetensors")) {
                        weights = q.value("bytes", 0ULL);
                        mtp     = q.value("mtp_bytes", 0ULL);
                    }
                }
                auto facts =
                    product::plan_facts_of_config(config, weights, runtime::WeightBasis::FileSize);
                facts.model.mtp_bytes   = mtp;
                out                     = kv_summary(&facts);
                facts.shape.base_models = convert::hub_base_models(repo, fetch);
                std::lock_guard lock(sizes_mutex_);
                if (generation == hub_generation_) {
                    plan_facts_[hub_key(repo, hub)] = std::move(facts);
                    configs_[hub_key(repo, hub)]    = config;
                }
                read                         = true;
            }
        } catch (const std::exception&) {}
        remember(out, read);
        return out;
    }
    fetch.quant = quant;
    bool read   = false;
    for (const std::uint64_t bytes : {8ULL << 20, 32ULL << 20}) {
        try {
            const auto head = convert::fetch_gguf_head(repo, fetch, bytes);
            const auto file =
                convert::parse_gguf_head(reinterpret_cast<const std::byte*>(head.bytes.data()),
                                         head.bytes.size(), head.size, head.path);
            auto facts = product::plan_facts_of_gguf(file, static_cast<std::uint64_t>(smallest));
            out        = kv_summary(&facts);
            {
                std::lock_guard lock(sizes_mutex_);
                if (generation == hub_generation_) {
                    plan_facts_[hub_key(repo, hub)] = std::move(facts);
                }
            }
            read = true;
            break;
        } catch (const convert::GgufTruncated&) { continue; } catch (const std::exception&) {
            break;
        }
    }
    remember(out, read);
    return out;
}

void App::watch_health() {
    while (!stopping_) {
        for (const auto& info : processes_.list()) {
            if (info.spec.kind != "serve" || info.state != "starting" || info.spec.port <= 0) {
                continue;
            }
            httplib::Client client("127.0.0.1", info.spec.port);
            client.set_connection_timeout(std::chrono::milliseconds(300));
            client.set_read_timeout(std::chrono::seconds(1));
            if (auto res = client.Get("/health"); res && res->status == 200) {
                processes_.mark_ready(info.id);
            }
        }
        remove_converted_originals();
        advance_updates();
        processes_.forget_ended(std::chrono::hours(6));
        for (int i = 0; i < 10 && !stopping_; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

// What a browser without the access key sees.
constexpr const char* kKeyPage =
    "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
    "<title>NInferno</title><body style='font:16px system-ui;margin:3em auto;max-width:22em'>"
    "<form method=get action=/><p>NInferno: access key</p><input name=key type=password "
    "autofocus required style='width:100%;font:inherit'><p><button>Open</button></p></form>";

constexpr const char* kKeylessProxy =
    "a proxy in front of ninferno-ui needs an access key: start it with NINFERNO_UI_KEY or "
    "--access-key-file F (16-256 of A-Z a-z 0-9 . _ ~ -)";

namespace {

// A query component percent-encoded: everything but A-Z a-z 0-9 - . _ ~ (RFC 3986 unreserved).
std::string url_component(std::string_view text) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char c : text) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' ||
            c == '.' || c == '_' || c == '~') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

} // namespace

void App::install(httplib::Server& server) {
    const AccessPolicy policy = options_.access;
    server.set_pre_routing_handler([policy](const httplib::Request& req, httplib::Response& res) {
        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_header("X-Frame-Options", "DENY");
        res.set_header("Referrer-Policy", "no-referrer");
        res.set_header("Cache-Control", "no-store");
        const bool proxied = req.has_header("Forwarded") || req.has_header("X-Forwarded-For") ||
                             req.has_header("X-Real-IP");
        if (!host_allowed(policy, req.get_header_value("Host"))) {
            // A keyless UI behind a proxy that forwards the public Host: the missing key is the
            // cause, so say that rather than the Host.
            reply_error(res, 403,
                        proxied && policy.key.empty()
                            ? kKeylessProxy
                            : "Host not allowed: ninferno-ui answers localhost only (see --listen)");
            return httplib::Server::HandlerResponse::Handled;
        }
        switch (check_key(policy, req.remote_addr, req.get_header_value("Host"), proxied,
                          req.get_header_value("Cookie"), req.get_header_value("Authorization"),
                          req.get_param_value("key"))) {
        case KeyCheck::pass: break;
        case KeyCheck::pass_set_cookie:
            res.set_header("Set-Cookie",
                           key_cookie(policy.port) + "=" + policy.key + "; Path=/; HttpOnly; SameSite=Lax" +
                               (forwarded_https(req.get_header_value("X-Forwarded-Proto"),
                                                req.get_header_value("Forwarded"))
                                    ? "; Secure"
                                    : ""));
            if (req.method == "GET") { // the key leaves the address bar and the history
                // The path as the browser sent it (req.path is decoded: %09/evil.example would
                // turn into a tab a browser strips, //evil.example), and only plain path bytes.
                const std::string_view path =
                    std::string_view(req.target).substr(0, req.target.find('?'));
                const bool plain = std::all_of(path.begin(), path.end(), [](unsigned char c) {
                    return c > 0x20 && c < 0x7f && c != '\\';
                });
                const bool local = plain && !path.empty() && path[0] == '/' &&
                                   (path.size() < 2 || path[1] != '/');
                // The other query parameters stay (an API GET keeps its arguments).
                std::string query;
                for (const auto& [name, value] : req.params) {
                    if (name == "key") { continue; }
                    query += (query.empty() ? "?" : "&") + url_component(name) + "=" +
                             url_component(value);
                }
                res.set_redirect(local ? std::string(path) + query : std::string("/"), 303);
                return httplib::Server::HandlerResponse::Handled;
            }
            break;
        case KeyCheck::keyless_proxy:
            reply_error(res, 403, kKeylessProxy);
            return httplib::Server::HandlerResponse::Handled;
        case KeyCheck::refused:
            if (req.path != "/") {
                reply_error(res, 401, "access key required");
            } else {
                res.status = 401;
                res.set_content(kKeyPage, "text/html; charset=utf-8");
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        if (req.method != "GET" && req.method != "HEAD") {
            if (!origin_allowed(req.get_header_value("Origin"), req.get_header_value("Host"))) {
                reply_error(res, 403,
                            origin_refusal(req.get_header_value("Origin"),
                                           req.get_header_value("Host"),
                                           proxied || req.has_header("X-Forwarded-Proto")));
                return httplib::Server::HandlerResponse::Handled;
            }
            if (req.get_header_value("Content-Type").rfind("application/json", 0) != 0) {
                reply_error(res, 415, "POST bodies must be application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Security-Policy",
                       "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                       "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                       "connect-src 'self'; frame-src 'self'; frame-ancestors 'none'; "
                       "base-uri 'none'; form-action 'none'");
        res.set_content(std::string(embedded::kPageHtml), "text/html; charset=utf-8");
    });
    server.Get("/api/state", [this](const httplib::Request&, httplib::Response& res) {
        reply_json(res, state_json());
    });
    server.Get("/api/catalog", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            reply_json(res, catalog_json(int_param(req, "gpu", 0), launch_ask_of(req)));
        } catch (const std::invalid_argument& error) { reply_error(res, 400, error.what()); }
    });
    server.Get("/api/quants", [this](const httplib::Request& req, httplib::Response& res) {
        const auto [repo, tag] = split_model_tag(req.get_param_value("repo"));
        if (!convert::is_repository_spec(repo)) {
            return reply_error(res, 400, "repo must be org/name");
        }
        const auto hub = req.get_param_value("hub") == "modelscope" ? "modelscope" : "huggingface";
        try {
            reply_json(res, quants_json(repo, int_param(req, "gpu", -1), hub, launch_ask_of(req)));
        } catch (const std::invalid_argument& error) {
            reply_error(res, 400, error.what());
        } catch (const std::exception& error) { reply_error(res, 502, error.what()); }
    });
    // The plan without the weights (ninferno-dry-plan/1): repo, hub, gpu, quant ("" or absent:
    // every quantization and the largest that fits), ctx (tokens or "auto"), par (streams), kv
    // (auto or a codec), spec (auto, none, mtp, dflash, dflash2), precision (exact).
    server.Get("/api/plan", [this](const httplib::Request& req, httplib::Response& res) {
        // A model converted here (My models: a start again is planned first), by its path: only
        // an artifact this UI lists (/api/local), never any path.
        if (req.has_param("artifact")) {
            try {
                const fs::path artifact = fs::path(req.get_param_value("artifact"));
                const Json local        = local_json();
                const bool listed = std::any_of(
                    local["models"].begin(), local["models"].end(), [&](const Json& m) {
                        return m.value("artifact", std::string()) == req.get_param_value("artifact");
                    });
                if (!listed) { return reply_error(res, 400, "not a model listed here"); }
                reply_json(res, artifact_plan_json(artifact, int_param(req, "gpu", 0),
                                                   launch_ask_of(req)));
            } catch (const std::invalid_argument& error) {
                reply_error(res, 400, error.what());
            } catch (const std::exception& error) { reply_error(res, 502, error.what()); }
            return;
        }
        const auto [repo, tag] = split_model_tag(req.get_param_value("repo"));
        if (!convert::is_repository_spec(repo)) {
            return reply_error(res, 400, "repo must be org/name");
        }
        const auto hub = req.get_param_value("hub") == "modelscope" ? "modelscope" : "huggingface";
        try {
            reply_json(res, plan_json(repo, hub, int_param(req, "gpu", 0),
                                      req.has_param("quant") ? req.get_param_value("quant") : tag,
                                      launch_ask_of(req)));
        } catch (const std::invalid_argument& error) {
            reply_error(res, 400, error.what());
        } catch (const std::exception& error) { reply_error(res, 502, error.what()); }
    });
    server.Get("/api/arch", [this](const httplib::Request& req, httplib::Response& res) {
        const auto repo = req.get_param_value("repo");
        if (!convert::is_repository_spec(repo)) {
            return reply_error(res, 400, "repo must be org/name");
        }
        const auto hub = req.get_param_value("hub") == "modelscope" ? "modelscope" : "huggingface";
        try {
            reply_json(res, arch_json(repo, hub));
        } catch (const std::exception& error) { reply_error(res, 502, error.what()); }
    });
    server.Get("/api/local", [this](const httplib::Request&, httplib::Response& res) {
        reply_json(res, local_json());
    });
    server.Get("/api/log", [this](const httplib::Request& req, httplib::Response& res) {
        const auto id = static_cast<std::uint64_t>(int_param(req, "id", 0));
        const auto n  = static_cast<std::size_t>(std::max(0, int_param(req, "n", 200)));
        reply_json(res, {{"id", id}, {"lines", processes_.log(id, n)}});
    });

    const auto parse_request = [](const httplib::Request& req) {
        const Json body = Json::parse(req.body);
        LaunchRequest out;
        out.model       = body.value("model", std::string());
        out.quant       = body.value("quant", std::string());
        out.gpu         = body.value("gpu", 0);
        out.max_context = body.value("max_context", 0L);
        out.max_context_cap = body.value("max_context_cap", 0L);
        out.max_concurrency = body.value("max_concurrency", 0);
        out.spec        = body.value("spec", std::string());
        out.precision   = body.value("precision", std::string());
        out.activations = body.value("activations", std::string());
        if (out.activations == "auto") { out.activations.clear(); }
        out.kv_dtype        = body.value("kv_dtype", std::string());
        out.hub             = body.value("hub", std::string());
        out.spec_profile    = body.value("spec_profile", false);
        return out;
    };
    // A launch the converter refused for scales below binary16's normal range, again as it was
    // with --round-tiny-scales (the log drawer offers it when the refusal names the flag).
    server.Post("/api/retry_rounded", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto id = Json::parse(req.body).value("id", std::uint64_t{0});
            for (const auto& info : processes_.list()) {
                if (info.id != id) { continue; }
                if (info.state != "exited" && info.state != "failed") {
                    return reply_error(res, 409, "the launch is still running");
                }
                const int port = info.spec.kind == "serve" ? processes_.free_port() : 0;
                const auto next = processes_.start(with_tiny_scales_rounded(info.spec, port));
                // A moved model's update waits on its prepares: it follows the retry.
                {
                    std::lock_guard moved_lock(moved_mutex_);
                    for (auto& [key, update] : updates_) {
                        for (auto& part : update.parts) {
                            if (part.prepare != id) { continue; }
                            part.prepare = next;
                            if (update.state == "failed") {
                                update.state = "downloading";
                                update.error.clear();
                            }
                        }
                    }
                }
                return reply_json(res, {{"id", next}, {"port", port}});
            }
            reply_error(res, 404, "no launch " + std::to_string(id));
        } catch (const Json::exception& error) {
            reply_error(res, 400, std::string("bad request body: ") + error.what());
        } catch (const std::exception& error) { reply_error(res, 500, error.what()); }
    });
    server.Post("/api/launch",
                [this, parse_request](const httplib::Request& req, httplib::Response& res) {
                    try {
                        auto request           = parse_request(req);
                        const auto [repo, tag] = split_model_tag(request.model);
                        if (convert::is_repository_spec(repo)) {
                            const auto quant = request.quant.empty() ? tag : request.quant;
                            if (const auto local = local_artifact(repo, quant); !local.empty()) {
                                request.model = local.string();
                                request.quant.clear();
                            }
                        }
                        // A pin the model does not run (an activation mode its family does not
                        // run, a KV codec it does not take) is refused here, with the plan's words
                        // (product::activation_refusal, kv_refusal), from the facts at hand: the
                        // artifact's, else the repository's read for its plan. The server refuses
                        // it at start as well when neither is known yet. Never substituted.
                        if (!request.activations.empty() || !request.kv_dtype.empty()) {
                            std::optional<product::PlanSourceFacts> facts;
                            const std::string model = split_model_tag(request.model).first;
                            if (fs::path(model).extension() == ".ninferno") {
                                try {
                                    facts = product::plan_facts_of_artifact(model);
                                } catch (const std::exception&) {} // unreadable: the server says

                            } else {
                                std::lock_guard lock(sizes_mutex_);
                                const auto it = plan_facts_.find(hub_key(
                                    model, request.hub.empty() ? "huggingface" : request.hub));
                                if (it != plan_facts_.end()) { facts = it->second; }
                            }
                            if (facts && !request.activations.empty()) {
                                const ActivationMode mode =
                                    product::parse_activation_mode(request.activations);
                                if (const std::string why =
                                        product::activation_refusal(*facts, mode);
                                    !why.empty()) {
                                    throw std::invalid_argument(why);
                                }
                            }
                            // A shape the facts do not name is the engine's to judge at
                            // load (as refuse_unrun_kv_at_start): not refused here.
                            if (facts && facts->model.known() && !request.kv_dtype.empty()) {
                                if (const std::string why = product::kv_refusal(
                                        *facts, product::parse_kv_dtype(request.kv_dtype));
                                    !why.empty()) {
                                    throw std::invalid_argument(why);
                                }
                            }
                        }
                        request.port     = processes_.free_port();
                        const auto stats = gpus_->snapshot();
                        const auto id    = processes_.start(
                            make_launch(options_, request, launch_gpu(stats, request.gpu)));
                        reply_json(res, {{"id", id}, {"port", request.port}});
                    } catch (const std::invalid_argument& error) {
                        reply_error(res, 400, error.what());
                    } catch (const Json::exception& error) {
                        reply_error(res, 400, std::string("bad request body: ") + error.what());
                    } catch (const std::exception& error) { reply_error(res, 500, error.what()); }
                });
    // Speculation tuned on this GPU at the user's request: spec-tune's quick pass writes
    // <artifact>.spec_profile.json. A measurement of this machine steers nothing by itself: a
    // start names it (--spec-profile) only when it asks for it (/api/launch spec_profile).
    server.Post("/api/tune", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            const Json body = Json::parse(req.body);
            TuneRequest request;
            request.artifact = body.value("artifact", std::string());
            request.gpu      = body.value("gpu", 0);
            request.engine   = convert::describe_artifact(request.artifact, "").engine;
            const auto id    = processes_.start(make_tune(options_, request));
            reply_json(res, {{"id", id}});
        } catch (const std::invalid_argument& error) {
            reply_error(res, 400, error.what());
        } catch (const Json::exception& error) {
            reply_error(res, 400, std::string("bad request body: ") + error.what());
        } catch (const std::exception& error) { reply_error(res, 500, error.what()); }
    });
    server.Post("/api/prepare",
                [this, parse_request](const httplib::Request& req, httplib::Response& res) {
                    try {
                        auto request         = parse_request(req);
                        request.prepare_only = true;
                        request.gpu          = -1;
                        auto spec            = make_launch(options_, request);
                        // The same quantization once: a second download would only wait for
                        // the first (convert::download_file's lock).
                        for (const auto& info : processes_.list()) {
                            if (info.spec.kind == "prepare" && info.state != "exited" &&
                                info.state != "failed" && info.spec.model == spec.model) {
                                return reply_error(res, 409,
                                                   spec.model + " is already downloading");
                            }
                        }
                        const auto id = processes_.start(std::move(spec));
                        reply_json(res, {{"id", id}});
                    } catch (const std::invalid_argument& error) {
                        reply_error(res, 400, error.what());
                    } catch (const Json::exception& error) {
                        reply_error(res, 400, std::string("bad request body: ") + error.what());
                    } catch (const std::exception& error) { reply_error(res, 500, error.what()); }
                });
    server.Get("/api/moved", [this](const httplib::Request&, httplib::Response& res) {
        try {
            reply_json(res, moved_json());
        } catch (const std::exception& error) { reply_error(res, 500, error.what()); }
    });
    // A model on disk the user names (a GGUF file, a checkpoint folder, an artifact): whether the
    // engine runs it, by the converter's rule set, before it is converted and started.
    server.Get("/api/disk", [](const httplib::Request& req, httplib::Response& res) {
        const fs::path path = req.get_param_value("path");
        std::error_code ec;
        if (path.empty() || !fs::exists(path, ec)) {
            reply_error(res, 404, "no such file or folder: " + path.string());
            return;
        }
        if (path.extension() == ".ninferno") {
            reply_json(res, {{"support", "full"}, {"kind", "artifact"}});
            return;
        }
        if (!convert::is_convertible(path)) {
            reply_json(res, {{"support", "none"},
                             {"note", "neither a GGUF file nor a Hugging Face checkpoint folder "
                                      "(config.json and safetensors weights)"}});
            return;
        }
        const auto support = convert::model_support(convert::local_support_facts(path));
        reply_json(res, {{"support", support_level(support)},
                         {"note", support.reason},
                         {"kind", fs::is_directory(path, ec) ? "checkpoint" : "gguf"}});
    });
    // The Hugging Face token gated repositories need: whether one is set, where it comes from and
    // whose account it is; set or removed from this machine only, where `hf auth login` keeps it.
    server.Get("/api/hf", [this](const httplib::Request&, httplib::Response& res) {
        const auto token = convert::hub_token();
        Json out{{"token", !token.value.empty()},
                 {"source", token.source},
                 {"path", convert::hub_token_path().string()}};
        if (!token.value.empty() && !options_.offline) {
            try {
                out["account"] = convert::hub_account(token.value, {});
            } catch (const std::exception& error) { out["error"] = error.what(); }
        }
        reply_json(res, out);
    });
    server.Post("/api/hf", [this](const httplib::Request& req, httplib::Response& res) {
        // A proxy (nginx, Caddy) connects over loopback for every browser: a forwarded request is
        // not from this machine.
        const bool proxied = req.has_header("Forwarded") || req.has_header("X-Forwarded-For") ||
                             req.has_header("X-Real-IP");
        if (!is_loopback_host(req.remote_addr) || proxied) {
            reply_error(res, 403, "the Hugging Face token is set on the machine ninferno-ui runs on");
            return;
        }
        try {
            const auto token = Json::parse(req.body).value("token", std::string());
            if (const auto set = convert::hub_token(); set.source == "HF_TOKEN" ||
                                                       set.source == "HUGGING_FACE_HUB_TOKEN") {
                reply_error(res, 409,
                            "the token is set by " + set.source + ": change it there");
                return;
            }
            if (!token.empty() && !options_.offline) {
                (void)convert::hub_account(token, {}); // refused: not saved
            }
            convert::save_hub_token(token);
            // What the hub answered under the old token (gated listings, searches, headers).
            {
                std::lock_guard lock(sizes_mutex_);
                ++hub_generation_; // requests still running under the old token keep nothing
                listings_.clear();
                archs_.clear();
                searches_.clear();
                configs_.clear();
                kv_.clear();
                mtp_.clear();
                kv_failed_.clear();
                plan_facts_.clear();
                plans_.clear();
                plan_now_.clear();
            }
            catalog_wake_ = true;
            const auto now = convert::hub_token();
            reply_json(res, {{"token", !now.value.empty()}, {"source", now.source}});
        } catch (const std::exception& error) { reply_error(res, 400, error.what()); }
    });
    server.Post("/api/moved", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            const Json body = Json::parse(req.body);
            reply_json(res, answer_moved(body.at("repo").get<std::string>(),
                                         body.at("ref").get<std::string>(),
                                         body.at("commit").get<std::string>(),
                                         body.at("action").get<std::string>()));
        } catch (const std::invalid_argument& error) {
            reply_error(res, 400, error.what());
        } catch (const std::exception& error) { reply_error(res, 400, error.what()); }
    });
    server.Post("/api/stop", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto id = Json::parse(req.body).at("id").get<std::uint64_t>();
            // Stopping waits for the drain; answer at once and let the state poll show it.
            std::thread([this, id] { processes_.stop(id); }).detach();
            reply_json(res, {{"id", id}, {"stopping", true}});
        } catch (const std::exception& error) { reply_error(res, 400, error.what()); }
    });
    // {action: "install"}: run the updater (a process like the others, its log in /api/log);
    // {action: "restart"}: stop the models and start the installed version in place of this one.
    server.Post("/api/update", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto action = Json::parse(req.body).at("action").get<std::string>();
            if (action == "install") {
                reply_json(res, {{"job", install_update()}});
            } else if (action == "restart") {
                reply_json(res, {{"restarting", true}});
                std::thread([this] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(300));
                    restart();
                }).detach();
            } else {
                throw std::invalid_argument("action is install or restart");
            }
        } catch (const std::invalid_argument& error) {
            reply_error(res, 400, error.what());
        } catch (const Json::exception& error) {
            reply_error(res, 400, std::string("bad request body: ") + error.what());
        } catch (const std::exception& error) { reply_error(res, 500, error.what()); }
    });
    install_chat(server);
    server.Get("/api/search", [this](const httplib::Request& req, httplib::Response& res) {
        const auto sort = req.get_param_value("sort");
        const auto hub  = req.get_param_value("hub") == "modelscope" ? "modelscope" : "huggingface";
        try {
            reply_json(res, search_json(req.get_param_value("q").substr(0, 120),
                                        sort.empty() ? "downloads" : sort, hub));
        } catch (const std::invalid_argument& error) { reply_error(res, 400, error.what()); }
    });
    server.Post("/api/delete", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            const Json body  = Json::parse(req.body);
            const auto freed = delete_artifact(body.at("artifact").get<std::string>());
            reply_json(res, {{"freed_gib", gib2(freed)}});
        } catch (const std::invalid_argument& error) {
            reply_error(res, 400, error.what());
        } catch (const std::exception& error) { reply_error(res, 400, error.what()); }
    });
    server.Post("/api/delete_original",
                [this](const httplib::Request& req, httplib::Response& res) {
                    try {
                        const Json body  = Json::parse(req.body);
                        const auto freed = delete_original(body.at("repo").get<std::string>());
                        reply_json(res, {{"freed_gib", gib2(freed)}});
                    } catch (const std::exception& error) { reply_error(res, 400, error.what()); }
                });
}

std::string first_model_id(const std::string& models_body) {
    const auto models = Json::parse(models_body, nullptr, false);
    if (models.is_discarded() || !models.contains("data") || !models["data"].is_array()) {
        return {};
    }
    for (const auto& model : models["data"]) {
        if (model.is_object() && model.contains("id") && model["id"].is_string()) {
            return model["id"].get<std::string>();
        }
    }
    return {};
}

std::string with_model(const std::string& chat_body, const std::string& served) {
    auto body = Json::parse(chat_body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        throw std::invalid_argument("the chat request is not a JSON object");
    }
    const auto it = body.find("model");
    if (!served.empty() &&
        (it == body.end() || !it->is_string() || it->get<std::string>().empty())) {
        body["model"] = served;
    }
    return body.dump();
}

void App::install_chat(httplib::Server& server) {
    // The running serve process `id` names, or nullopt (and a 409 written).
    const auto running = [this](const httplib::Request& req,
                                httplib::Response& res) -> std::optional<ProcessInfo> {
        ProcessInfo info;
        const std::uint64_t id = process_id(req.matches[1].str());
        if (!processes_.find(id, info) || info.spec.kind != "serve" || info.state != "ready") {
            reply_error(res, 409, "that model is not running");
            return std::nullopt;
        }
        return info;
    };
    server.Get(R"(/chat/(\d+)/props)", [=](const httplib::Request& req, httplib::Response& res) {
        const auto info = running(req, res);
        if (!info) { return; }
        auto model = chat_model_of(serve_models_body(info->spec.port));
        if (model.id.empty()) { return reply_error(res, 502, "the model's server did not answer"); }
        // A local artifact is launched by its path; a hub model's is in the serve log.
        const std::string artifact = !info->artifact.empty()                 ? info->artifact
                                     : info->spec.model.ends_with(".ninferno") ? info->spec.model
                                                                             : std::string();
        model.vision               = !artifact.empty() && artifact_has_vision(artifact);
        const auto& argv           = info->spec.argv;
        for (std::size_t i = 0; i + 1 < argv.size(); ++i) {
            if (argv[i] == "--max-concurrency") {
                model.slots = std::max(1, std::atoi(argv[i + 1].c_str()));
            }
        }
        reply_json(res, chat_props(model));
    });
    server.Get(R"(/chat/(\d+)/v1/models)",
               [=](const httplib::Request& req, httplib::Response& res) {
                   const auto info = running(req, res);
                   if (!info) { return; }
                   const auto body = serve_models_body(info->spec.port);
                   if (body.empty()) {
                       return reply_error(res, 502, "the model's server did not answer");
                   }
                   res.set_content(body, "application/json");
               });
    // Slot introspection is llama-server's (--slots); the bundle runs without it.
    server.Get(R"(/chat/(\d+)/slots)", [](const httplib::Request&, httplib::Response& res) {
        reply_error(res, 501, "slots are not exposed");
    });
    server.Post(R"(/chat/(\d+)/v1/chat/completions)", [=](const httplib::Request& req,
                                                          httplib::Response& res) {
        const auto info = running(req, res);
        if (!info) { return; }
        // ninferno-serve requires the request's "model" to be the id it serves (/v1/models); the
        // chat window does not know it, so the proxy names it.
        std::string body;
        try {
            body = with_model(req.body, first_model_id(serve_models_body(info->spec.port)));
        } catch (const std::exception& error) { return reply_error(res, 400, error.what()); }
        stream_chat(res, info->spec.port, std::move(body));
    });
    // The bundle's files ("" is its index.html; it routes by the URL's hash, so any other path
    // without a file is an API it probes, answered 404).
    const auto root = options_.chat_ui;
    server.Get(R"(/chat/(\d+)/(.*))", [=](const httplib::Request& req, httplib::Response& res) {
        if (root.empty()) { return reply_error(res, 404, "no chat bundle (see --chat-ui)"); }
        const std::string relative = req.matches[2].str();
        if (!safe_bundle_path(relative)) { return reply_error(res, 400, "bad path"); }
        std::error_code ec;
        auto file = root / relative;
        if (relative.empty()) { file = root / "index.html"; }
        if (!std::filesystem::is_regular_file(file, ec)) {
            return reply_error(res, 404, "not served here: " + relative);
        }
        std::ifstream in(file, std::ios::binary);
        if (!in) { return reply_error(res, 404, "not found"); }
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        res.set_header("Content-Security-Policy",
                       "default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; "
                       "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                       "media-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; "
                       "worker-src 'self' blob:; frame-ancestors 'self'; base-uri 'self'; "
                       "form-action 'none'");
        res.headers.erase("X-Frame-Options");
        res.set_header("X-Frame-Options", "SAMEORIGIN");
        if (relative.starts_with("_app/immutable/")) {
            res.headers.erase("Cache-Control");
            res.set_header("Cache-Control", "public, max-age=31536000, immutable");
        }
        res.set_content(std::move(content), bundle_mime(file));
    });
    server.Get(R"(/chat/(\d+))", [](const httplib::Request& req, httplib::Response& res) {
        res.set_redirect("/chat/" + req.matches[1].str() + "/");
    });
}

bool App::run() {
    server_ = std::make_unique<httplib::Server>();
    install(*server_);
    if (!server_->bind_to_port(options_.access.listen, options_.access.port)) { return false; }
    return server_->listen_after_bind();
}

} // namespace ninferno::ui
