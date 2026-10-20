#include "ui/app.h"

#include "artifact/reader.h"
#include "convert/drafters.h"
#include "core/ascii.h"
#include "spec/registry.h"
#include "convert/fetch.h"
#include "convert/acquire.h"
#include "convert/convert.h"
#include "convert/gguf.h"
#include "convert/gguf_model.h"
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
std::string arch_key(std::string name) {
    for (const char* suffix : {"ForConditionalGeneration", "ForCausalLM", "Model"}) {
        if (name.size() > std::strlen(suffix) && name.ends_with(suffix)) {
            name.resize(name.size() - std::strlen(suffix));
            break;
        }
    }
    std::string out;
    for (const unsigned char c : name) {
        if (std::isalnum(c)) { out += static_cast<char>(std::tolower(c)); }
    }
    return out;
}

// The architectures the engine runs, reduced: map_gguf_model's general.architecture values and
// the checkpoint architectures the converter names (as config.json does).
const std::set<std::string>& engine_arch_keys() {
    static const std::set<std::string> keys = [] {
        std::set<std::string> out;
        for (const auto& name : convert::supported_gguf_architectures()) {
            out.insert(arch_key(name));
        }
        static const std::regex name(
            R"(\b[A-Z][A-Za-z0-9_]*(?:ForCausalLM|ForConditionalGeneration)\b)");
        const auto& text = convert::supported_architectures_text();
        for (std::sregex_iterator it(text.begin(), text.end(), name), end; it != end; ++it) {
            out.insert(arch_key(it->str()));
        }
        return out;
    }();
    return keys;
}

// How far the engine runs a hub model, by the architecture named for its weights (the hub's
// reading, or ours of the weights' own head): "full" (the engine runs exactly it), "family" (a
// variant of a family it runs: a multimodal wrapper, a newer revision; not verified), "drafter"
// (a speculative drafter: runs beside its model), "projector" (the hub read an image projector,
// not the model: unknown until the weights are read), "none" (another family, or a task that is
// not text generation: speech, embeddings, ranking), "unknown" (nothing said).
std::string architecture_support(const std::string& architecture, const std::string& pipeline) {
    static const std::set<std::string> generative = {
        "",           "text-generation",      "image-text-to-text",
        "any-to-any", "text2text-generation", "image-to-text",
        "chat"};
    if (!generative.contains(pipeline)) { return "none"; }
    if (architecture.empty()) { return "unknown"; }
    const auto key = arch_key(architecture);
    if (key == "clip" || key == "mmproj") { return "projector"; }
    if (key == "dflash" || key.find("assistant") != std::string::npos ||
        key.find("draft") != std::string::npos || key.find("dspark") != std::string::npos ||
        key.ends_with("mtp")) {
        return "drafter";
    }
    const auto& keys = engine_arch_keys();
    if (keys.contains(key)) { return "full"; }
    // A family the engine runs, by the longest known name the key starts with (at least five
    // characters: "qwen3" covers Qwen3 variants, not "qwen4").
    for (const auto& known : keys) {
        if (known.size() >= 5 && key.starts_with(known)) { return "family"; }
    }
    return "none";
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

LaunchSpec make_launch(const UiOptions& options, const LaunchRequest& request) {
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
    static const std::set<std::string> kv_dtypes = {"", "bf16", "int8", "fp8", "nvfp4", "k8v4"};
    if (!kv_dtypes.count(request.kv_dtype)) {
        throw std::invalid_argument("bad kv_dtype: " + request.kv_dtype);
    }
    if (!request.hub.empty() && request.hub != "huggingface" && request.hub != "modelscope") {
        throw std::invalid_argument("bad hub: " + request.hub);
    }
    if (request.max_context < 0) { throw std::invalid_argument("bad max_context"); }
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
    if (request.prepare_only) {
        spec.argv.push_back("--prepare-only");
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
        // Precision: exact is ninferno-serve's --activations exact with a BF16 KV cache (the
        // output does not depend on the load); balanced and compact keep its defaults, which already
        // store the cache in the family's smallest measured format (--kv-dtype auto).
        const bool exact = request.precision == "exact";
        if (!request.kv_dtype.empty() || exact) {
            spec.argv.push_back("--kv-dtype");
            spec.argv.push_back(request.kv_dtype.empty() ? "bf16" : request.kv_dtype);
        }
        if (exact) {
            spec.argv.push_back("--activations");
            spec.argv.push_back("exact");
        }
        // Speculation is the product's measured choice unless the request names one (ninferno-serve
        // alone decodes plainly without --spec).
        spec.argv.push_back("--spec");
        spec.argv.push_back(request.spec.empty() ? "auto" : request.spec);
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
                    for (int i = 0; i < 6 * 3600 * 10 && !stopping_; ++i) {
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
    const auto quants = convert::list_quants(repo, fetch);
    Json list         = Json::array();
    std::lock_guard lock(sizes_mutex_);
    for (const auto& q : quants) {
        list.push_back({{"quant", q.quant},
                        {"format", q.format},
                        {"bytes", q.bytes},
                        {"files", q.files},
                        {"bits", q.bits},
                        {"drafter", q.drafter}});
        if (!q.drafter && hub != "modelscope") { sizes_[size_key(repo, q.quant)] = q.bytes; }
    }
    listings_[hub_key(repo, hub)] = list;
    return list;
}

Json App::arch_json(const std::string& repo, const std::string& hub) {
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
            out["architecture"] =
                kv_model_of_config(convert::fetch_config(repo, fetch)).architecture;
            out["source"] = "config";
            out["file"]   = "config.json";
        }
    } catch (const std::exception& error) { out["error"] = error.what(); }
    if (!out.contains("support")) {
        out["support"] = architecture_support(out["architecture"].get<std::string>(), "");
    }
    std::lock_guard lock(sizes_mutex_);
    if (!out.contains("error")) { archs_[hub_key(repo, hub)] = out; }
    return out;
}

Json App::quants_json(const std::string& repo, int gpu, const std::string& hub) {
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
    if (pick) { out[*pick]["recommended"] = true; }
    auto kv          = kv_json(repo, list, hub);
    auto speculation = speculation_json(repo, gpu, kv, list);
    const auto arch  = kv.value("architecture", std::string());
    return {{"repo", repo},
            {"hub", hub},
            {"architecture", arch},
            {"support", architecture_support(arch, "")},
            {"quants", out},
            {"kv", std::move(kv)},
            {"speculation", std::move(speculation)}};
}

Json App::speculation_json(const std::string& repo, int gpu, const Json& kv, const Json& listing) {
    Json out             = {{"known", false}};
    const auto& registry = spec::Registry::builtin();
    std::vector<std::string> names{repo}; // the repository and the base model its card names
    const spec::ModelEntry* entry = registry.by_repo(repo);
    if (entry == nullptr) {
        for (const auto& model : catalog()->models()) {
            for (const auto& source : model.sources) {
                if (source.repo == repo && entry == nullptr) { entry = registry.by_id(model.id); }
            }
        }
    }
    if (entry == nullptr && !options_.offline) {
        std::string base;
        {
            std::lock_guard lock(sizes_mutex_);
            if (const auto it = base_models_.find(repo); it != base_models_.end()) {
                base = it->second;
            }
        }
        if (base.empty()) {
            try {
                const auto card = Json::parse(
                    convert::fetch_text("https://huggingface.co/api/models/" + repo, 4 << 20));
                const auto declared =
                    card.value("cardData", Json::object()).value("base_model", Json());
                base = declared.is_string() ? declared.get<std::string>()
                       : declared.is_array() && !declared.empty() && declared[0].is_string()
                           ? declared[0].get<std::string>()
                           : std::string("-");
            } catch (const std::exception&) { base = "-"; }
            std::lock_guard lock(sizes_mutex_);
            base_models_[repo] = base;
        }
        if (base != "-") {
            names.push_back(base);
            entry = registry.by_repo(base);
            if (entry == nullptr) { entry = registry.by_lineage(base); }
        }
    }
    if (entry == nullptr) { return out; }
    TargetGpu target;
    for (const auto& stat : gpus_->snapshot()) {
        if (stat.index == gpu) { target = target_of(stat); }
    }
    // The profile serve takes: the first class of the GPU's similarity chain that has one, the
    // repository's own (or its base model's) before one for every variant; a class with only
    // other variants' profiles still says what speculation gains there (another quantization of
    // the same model accepts about as many drafts).
    auto chain = target.profile_chain;
    if (std::find(chain.begin(), chain.end(), "*") == chain.end()) { chain.push_back("*"); }
    const Json* profile = nullptr;
    for (const auto& key : chain) {
        int rank = 0; // 3: own, 2: every variant, 1: another variant's
        for (const auto& p : entry->profiles) {
            if (p.value("gpu_class", std::string()) != key) { continue; }
            const auto applies = p.value("applies_to", Json::array());
            const bool own     = std::any_of(names.begin(), names.end(), [&](const std::string& n) {
                return std::find(applies.begin(), applies.end(), n) != applies.end();
            });
            const int here     = own ? 3 : applies.empty() ? 2 : 1;
            if (here > rank) {
                profile = &p;
                rank    = here;
            }
        }
        if (profile != nullptr) {
            out["profile_class"] = key;
            out["profile"]       = key == "*" ? "generic"
                                   : !target.chip.empty() && key.rfind(target.chip, 0) == 0 ? "own"
                                                                                            : "similar";
            break;
        }
    }
    if (profile == nullptr) { return out; }
    // Requests that sample (the chat client's temperature 1.0) take the sampled choices.
    const auto sampled    = profile->find("sampled");
    const Json& choice    = sampled != profile->end() && sampled->contains("default")
                                ? sampled->at("default")
                                : profile->value("default", Json::object());
    const auto drafter    = choice.value("drafter", std::string());
    out["known"]          = true;
    out["config"]         = choice.value("config", std::string());
    out["drafter"]        = drafter;
    out["speedup"]        = choice.value("speedup", 0.0);
    out["greedy_speedup"] = profile->value("default", Json::object()).value("speedup", 0.0);
    out["load"]           = profile->value("load", Json::array());
    const auto name       = spec::split_variant(drafter).first;
    const auto decl       = std::find_if(entry->drafters.begin(), entry->drafters.end(),
                                         [&](const spec::DrafterDecl& d) { return d.name == name; });
    if (decl == entry->drafters.end() || out["config"] == "plain") { return out; }
    out["kind"] = decl->kind;
    // A drafter file in a listing: its GGUF selector, else the kind in its name (mtp-*.gguf).
    const auto drafter_file = [&](const Json& files, const std::string& part) -> double {
        double bytes = 0;
        for (const auto& q : files) {
            if (!q.value("drafter", false)) { continue; }
            const auto label = ascii_lower(q.value("quant", std::string()) + " " +
                                           q.value("files", Json::array()).dump());
            if (label.find(ascii_lower(part)) != std::string::npos &&
                (bytes == 0 || q.value("bytes", 0.0) < bytes)) {
                bytes = q.value("bytes", 0.0);
            }
        }
        return bytes;
    };
    // One full-attention layer of the target: what an MTP or Gemma 4 assistant layer keeps.
    const double one_layer =
        kv.value("full_layers", 0) > 0
            ? kv.value("full_bytes_per_token", 0.0) / kv.value("full_layers", 1)
            : 0;
    double bytes = 0, kv_per_token = 0, window_per_token = 0, window = 0;
    if (!decl->gguf.empty()) {
        bytes = drafter_file(listing, decl->gguf);
    } else if (!decl->repo.empty()) {
        try {
            Json files;
            {
                std::lock_guard lock(sizes_mutex_);
                if (const auto it = listings_.find(decl->repo); it != listings_.end()) {
                    files = it->second;
                }
            }
            if (files.is_null() && !options_.offline) {
                convert::FetchOptions fetch;
                if (!options_.hub_cache.empty()) { fetch.cache = options_.hub_cache; }
                files = Json::array();
                for (const auto& q : convert::list_quants(decl->repo, fetch)) {
                    files.push_back({{"quant", q.quant},
                                     {"format", q.format},
                                     {"bytes", q.bytes},
                                     {"files", q.files},
                                     {"bits", q.bits},
                                     {"drafter", q.drafter}});
                }
                std::lock_guard lock(sizes_mutex_);
                listings_[decl->repo] = files;
            }
            for (const auto& q : files) { bytes = std::max(bytes, q.value("bytes", 0.0)); }
            const auto drafter_kv = kv_json(decl->repo, files);
            kv_per_token          = drafter_kv.value("full_bytes_per_token", 0.0);
            window_per_token      = drafter_kv.value("window_bytes_per_token", 0.0);
            window                = drafter_kv.value("window", 0.0);
        } catch (const std::exception&) {}
    } else {
        bytes = drafter_file(listing, decl->kind == "gemma4_assistant" ? "assistant" : decl->kind);
    }
    if (kv_per_token == 0 && window_per_token == 0 &&
        (decl->kind.find("mtp") != std::string::npos || decl->kind == "gemma4_assistant")) {
        kv_per_token = one_layer;
    }
    // A variant NInferno quantizes from the drafter ("dflash2.q8") is that much smaller.
    const auto variant = spec::split_variant(drafter).second;
    if (variant == "q8") { bytes /= 2; }
    if (variant == "q4" || variant == "nvfp4") { bytes /= 3.5; }
    out["drafter_bytes"]                  = bytes;
    out["drafter_kv_bytes_per_token"]     = kv_per_token;
    out["drafter_window_bytes_per_token"] = window_per_token;
    out["drafter_window"]                 = window;
    return out;
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

Json App::catalog_json(int gpu) {
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
        const auto model = with_fitting_quant(listed, target, listings);
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
                       // Calibrated on this machine (spec-tune wrote it), else a registry profile.
                       {"own_profile", fs::exists(path.string() + ".spec_profile.json", ec)}});
    };
    // Converted hub models: <cache>/models--org--name/ninferno/<commit>/*.ninferno.
    if (fs::is_directory(root, ec)) {
        for (const auto& repo_dir : fs::directory_iterator(root, ec)) {
            const auto repo = convert::hub_repo_of_directory(repo_dir.path().filename().string());
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
    const auto dir = hub_root() / convert::hub_repo_directory(repo) / "ninferno";
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
            !convert::hub_repo_of_directory(p.filename().string()).empty()) {
            repo_dir = p;
            break;
        }
    }
    if (!repo_dir.empty()) {
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
    const auto dir = hub_root() / convert::hub_repo_directory(repo);
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
                     {"weights_format",
                      m.library == "mlx" ? std::string("MLX") : m.unread_quantization},
                     // Apple's MLX weights, a quantization the converter does not read
                     // (bitsandbytes, HQQ): the architecture may be the engine's, the
                     // format is not.
                     {"support", m.library == "mlx" || !m.unread_quantization.empty()
                                     ? std::string("format")
                                     : architecture_support(m.architecture, m.pipeline)}});
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
    if (error.empty()) { searches_[key] = {now, out}; }
    return out;
}

// The KV cache per token of the repository's models, from the GGUF metadata of its smallest
// quantization (an 8 MiB head by an HTTP range; 32 MiB when the metadata runs longer), else from
// its config.json (safetensors weights).
Json App::kv_json(const std::string& repo, const Json& listing, const std::string& hub) {
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
    Json out = to_json(KvModel{});
    if (options_.offline) { return out; }
    convert::FetchOptions fetch;
    fetch.hub = hub;
    if (!options_.hub_cache.empty()) { fetch.cache = options_.hub_cache; }
    if (quant.empty()) {
        const bool native = std::any_of(listing.begin(), listing.end(), [](const Json& q) {
            return q.value("format", std::string()) == "ninferno";
        });
        try {
            if (hub != "modelscope") {
                // A native artifact keeps no config.json; its manifest names the base model's.
                auto source = repo;
                if (native) {
                    const auto manifest =
                        convert::fetch_repo_json(repo, fetch, "artifact-manifest.json");
                    source = manifest.value("source", Json::object())
                                 .value("base", Json::object())
                                 .value("repo_id", repo);
                }
                out = to_json(kv_model_of_config(convert::fetch_config(source, fetch)));
            }
        } catch (const std::exception&) {}
        std::lock_guard lock(sizes_mutex_);
        kv_[hub_key(repo, hub)] = out;
        return out;
    }
    fetch.quant = quant;
    for (const std::uint64_t bytes : {8ULL << 20, 32ULL << 20}) {
        try {
            const auto head = convert::fetch_gguf_head(repo, fetch, bytes);
            const auto file =
                convert::parse_gguf_head(reinterpret_cast<const std::byte*>(head.bytes.data()),
                                         head.bytes.size(), head.size, head.path);
            out = to_json(kv_model_of_gguf(file.metadata));
            break;
        } catch (const convert::GgufTruncated&) { continue; } catch (const std::exception&) {
            break;
        }
    }
    std::lock_guard lock(sizes_mutex_);
    kv_[hub_key(repo, hub)] = out;
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

void App::install(httplib::Server& server) {
    const AccessPolicy policy = options_.access;
    server.set_pre_routing_handler([policy](const httplib::Request& req, httplib::Response& res) {
        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_header("X-Frame-Options", "DENY");
        res.set_header("Referrer-Policy", "no-referrer");
        res.set_header("Cache-Control", "no-store");
        if (!host_allowed(policy, req.get_header_value("Host"))) {
            reply_error(res, 403,
                        "Host not allowed: ninferno-ui answers localhost only (see --listen)");
            return httplib::Server::HandlerResponse::Handled;
        }
        switch (check_key(policy, req.remote_addr, req.get_header_value("Cookie"),
                          req.get_header_value("Authorization"), req.get_param_value("key"))) {
        case KeyCheck::pass: break;
        case KeyCheck::pass_set_cookie:
            res.set_header("Set-Cookie", std::string(kKeyCookie) + "=" + policy.key +
                                             "; Path=/; HttpOnly; SameSite=Lax");
            if (req.method == "GET") { // the key leaves the address bar and the history
                res.set_redirect(req.path, 303);
                return httplib::Server::HandlerResponse::Handled;
            }
            break;
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
                reply_error(res, 403, "cross-origin request refused");
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
        reply_json(res, catalog_json(int_param(req, "gpu", 0)));
    });
    server.Get("/api/quants", [this](const httplib::Request& req, httplib::Response& res) {
        const auto [repo, tag] = split_model_tag(req.get_param_value("repo"));
        if (!convert::is_repository_spec(repo)) {
            return reply_error(res, 400, "repo must be org/name");
        }
        const auto hub = req.get_param_value("hub") == "modelscope" ? "modelscope" : "huggingface";
        try {
            reply_json(res, quants_json(repo, int_param(req, "gpu", -1), hub));
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
        out.max_concurrency = body.value("max_concurrency", 0);
        out.spec        = body.value("spec", std::string());
        out.precision   = body.value("precision", std::string());
        out.kv_dtype        = body.value("kv_dtype", std::string());
        out.hub             = body.value("hub", std::string());
        return out;
    };
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
                        request.port  = processes_.free_port();
                        const auto id = processes_.start(make_launch(options_, request));
                        reply_json(res, {{"id", id}, {"port", request.port}});
                    } catch (const std::invalid_argument& error) {
                        reply_error(res, 400, error.what());
                    } catch (const Json::exception& error) {
                        reply_error(res, 400, std::string("bad request body: ") + error.what());
                    } catch (const std::exception& error) { reply_error(res, 500, error.what()); }
                });
    // Speculation calibration on this GPU: spec-tune's quick pass writes
    // <artifact>.spec_profile.json, which the engines read at their next start in place of a
    // profile borrowed from a similar chip.
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
