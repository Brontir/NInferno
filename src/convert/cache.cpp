#include "convert/cache.h"
#include "convert/checks.h"
#include "convert/drafters.h"

#include "convert/acquire.h"
#include "convert/convert.h"
#include "convert/safetensors.h"

#include "core/cache_file.h"
#include "core/sys.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>

namespace ninferno::convert {

bool is_checkpoint_directory(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(path) || !fs::is_regular_file(path / "config.json")) { return false; }
    if (fs::is_regular_file(path / "model.safetensors.index.json") ||
        fs::is_regular_file(path / "model.safetensors") ||
        fs::is_directory(path / ".ninferno-cache")) {
        return true;
    }
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.path().extension() == ".safetensors") { return true; }
    }
    return false;
}

bool is_convertible(const std::filesystem::path& path) {
    return is_checkpoint_directory(path) || !gguf_model_files(path).empty();
}

int cached_revision(const std::filesystem::path& artifact) {
    std::ifstream in(artifact.string() + ".revision");
    int revision = 0;
    return in >> revision && revision > 0 ? revision : 1;
}

bool unquantized_drafter_attention(const std::filesystem::path& artifact) {
    std::ifstream stream(artifact.string() + ".conversion.json");
    if (!stream) { return false; }
    try {
        const OJson report = OJson::parse(stream);
        const auto methods = report.find("methods");
        if (methods == report.end() || !methods->is_array()) { return false; }
        for (const auto& method : *methods) {
            const auto format = method.value("format", std::string());
            if (format != "bf16" && format != "fp16" && format != "fp32") { continue; }
            for (const auto& name : method.value("parameters", OJson::array())) {
                if (!name.is_string()) { continue; }
                const auto& text = name.get_ref<const std::string&>();
                if ((text.starts_with("dflash/") || text.starts_with("dflash2/")) &&
                    text.ends_with("/attention/query")) {
                    return true;
                }
            }
        }
    } catch (const std::exception&) { return false; }
    return false;
}

bool companion_without_proposal(const std::filesystem::path& artifact) {
    std::ifstream stream(artifact.string() + ".conversion.json");
    if (!stream) { return false; }
    bool companion = false;
    try {
        const OJson report = OJson::parse(stream);
        // Only a Qwen target's (default_proposal), and only where this build finds the ranking.
        const auto& config = report.at("components").at("text").at("config");
        const auto type    = config.value("model_type", std::string());
        if ((type != "qwen3_5_text" && type != "qwen3_5_moe_text") ||
            !resolve_ranking(kDefaultProposalRanking, std::filesystem::current_path())) {
            return false;
        }
        const auto methods = report.find("methods");
        if (methods == report.end() || !methods->is_array()) { return false; }
        for (const auto& method : *methods) {
            for (const auto& name : method.value("parameters", OJson::array())) {
                if (!name.is_string()) { continue; }
                const auto& text = name.get_ref<const std::string&>();
                if (text.starts_with("proposal/")) { return false; }
                companion = companion || text.starts_with("dflash/");
            }
        }
    } catch (const std::exception&) { return false; }
    return companion;
}

namespace {

// Whether an artifact at `revision` is served as it is (cache_current's rule).
bool serves_at(const std::filesystem::path& artifact, int revision) {
    return revision >= kCacheRevision ||
           (revision >= 3 && !unquantized_drafter_attention(artifact) &&
            !companion_without_proposal(artifact));
}

} // namespace

bool cache_serves(const std::filesystem::path& artifact) {
    return serves_at(artifact, cached_revision(artifact));
}

bool cache_current(const std::filesystem::path& artifact) {
    const int revision = cached_revision(artifact);
    if (revision >= kCacheRevision) { return true; }
    if (!serves_at(artifact, revision)) { return false; }
    // Nothing a later revision changed is in it: stamped current, so its report is not read
    // again. The stamp is the small sidecar beside the artifact, replaced atomically; a read-only
    // cache keeps the old stamp (and reads the report next time).
    try {
        stamp_revision(artifact);
    } catch (const std::exception&) {}
    return true;
}

void stamp_revision(const std::filesystem::path& artifact) {
    write_file_atomic(artifact.string() + ".revision", std::to_string(kCacheRevision) + "\n");
}

bool checkpoint_has_vision(const std::filesystem::path& checkpoint) {
    std::ifstream stream(checkpoint / "config.json");
    if (!stream) { return false; }
    try {
        const OJson config = OJson::parse(stream);
        return config.contains("vision_config") && config["vision_config"].is_object();
    } catch (const std::exception&) { return false; }
}

std::string source_stamp(const std::filesystem::path& root,
                         std::vector<std::filesystem::path> files) {
    namespace fs = std::filesystem;
    std::map<std::string, std::string> lines; // by name: one line per file, in name order
    for (const auto& file : files) {
        std::error_code size_error, time_error;
        const auto size = fs::file_size(file, size_error);
        const auto time = fs::last_write_time(file, time_error);
        if (size_error || time_error) { continue; }
        // Named from the model's directory, so another spelling of the same path (relative,
        // absolute, a moved directory) is the same source; a file outside it by its full path.
        const auto relative = file.lexically_relative(root);
        std::error_code canonical_error;
        const auto absolute = fs::weakly_canonical(file, canonical_error);
        const auto text     = !relative.empty() && *relative.begin() != ".."
                                  ? relative.generic_u8string()
                                  : (canonical_error ? file : absolute).generic_u8string();
        const std::string name(text.begin(), text.end());
        lines[name] = std::to_string(size) + '\t' + std::to_string(time.time_since_epoch().count());
    }
    std::string out;
    for (const auto& [name, facts] : lines) { out += name + '\t' + facts + '\n'; }
    return out;
}

void stamp_source(const std::filesystem::path& artifact, const std::string& stamp) {
    write_file_atomic(artifact.string() + ".source", stamp);
}

namespace {

std::map<std::string, std::string> stamp_lines(const std::string& stamp) {
    std::map<std::string, std::string> out;
    std::size_t at = 0;
    while (at < stamp.size()) {
        auto end = stamp.find('\n', at);
        if (end == std::string::npos) { end = stamp.size(); }
        const auto line = stamp.substr(at, end - at);
        const auto tab  = line.find('\t');
        if (tab != std::string::npos) { out[line.substr(0, tab)] = line.substr(tab + 1); }
        at = end + 1;
    }
    return out;
}

} // namespace

bool source_changed(const std::filesystem::path& artifact, const std::string& stamp) {
    std::ifstream in(artifact.string() + ".source", std::ios::binary);
    if (!in) {
        try {
            stamp_source(artifact, stamp);
        } catch (const std::exception&) {}
        return false;
    }
    const auto recorded = stamp_lines(
        std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()});
    // Weights the conversion read that are gone are no change (--consume-source removes them);
    // any other file gone (a tokenizer, a template, the configuration), new, or of another size
    // or time is.
    const auto current = stamp_lines(stamp);
    for (const auto& [name, facts] : current) {
        const auto at = recorded.find(name);
        if (at == recorded.end() || at->second != facts) { return true; }
    }
    for (const auto& [name, facts] : recorded) {
        const bool weights = name.ends_with(".safetensors") || name.ends_with(".gguf");
        if (!weights && !current.contains(name)) { return true; }
    }
    return false;
}

void remove_artifact(const std::filesystem::path& artifact) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto prefix = artifact.filename().string();
    for (const auto& entry : fs::directory_iterator(artifact.parent_path(), ec)) {
        const auto name = entry.path().filename().string();
        if (name == prefix || name.rfind(prefix + ".", 0) == 0) { fs::remove(entry.path(), ec); }
    }
}

namespace {

// Whether the weights a cached conversion came from are still there to convert again: a GGUF file
// is; a checkpoint directory is unless --consume-source took its safetensors.
bool source_present(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    if (fs::is_regular_file(path)) return true;
    if (fs::is_regular_file(path / "model.safetensors.index.json")) return true;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(path, ec)) {
        if (entry.path().extension() == ".safetensors") return true;
    }
    return false;
}

// Whether a checkpoint directory's file is one a conversion reads: weights, configuration,
// tokenizer and chat-template files (not a README, a log or a hidden file).
bool read_by_conversion(const std::filesystem::path& file) {
    const auto name = file.filename().string();
    if (name.starts_with(".")) { return false; }
    const auto ext = file.extension().string();
    return ext == ".safetensors" || ext == ".gguf" || ext == ".json" || ext == ".jinja" ||
           ext == ".model" || ext == ".txt" || ext == ".tiktoken";
}

// The files a conversion of `path` reads: a GGUF model's files and its projector, or a checkpoint
// directory's (read_by_conversion); and each resource the model's directory lacks.
std::vector<std::filesystem::path>
source_files(const std::filesystem::path& path,
             const std::optional<std::filesystem::path>& projector,
             const std::vector<std::pair<std::string, std::filesystem::path>>& resources) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    const bool file = fs::is_regular_file(path);
    if (file) {
        files = gguf_model_files(path);
        if (projector) { files.push_back(*projector); }
    } else {
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(path, ec)) {
            if (entry.is_regular_file(ec) && read_by_conversion(entry.path())) {
                files.push_back(entry.path());
            }
        }
    }
    const auto root = file ? path.parent_path() : path;
    for (const auto& [role, source] : resources) {
        files.push_back(fs::exists(root / role) ? root / role : source);
    }
    return files;
}

// A cached artifact to serve as it is: current and converted from the source as it is now, or
// with nothing left to convert from.
bool serve_cached(const std::filesystem::path& artifact, const std::filesystem::path& source,
                  const std::string& stamp) {
    return std::filesystem::exists(artifact) &&
           (!source_present(source) ||
            (cache_current(artifact) && !source_changed(artifact, stamp)));
}

} // namespace

CachedArtifact
cached_artifact(const std::filesystem::path& path,
                const std::vector<std::pair<std::string, std::filesystem::path>>& resources,
                const std::function<void(const std::string&)>& log, bool consume_source,
                bool vision, bool round_tiny_scales) {
    namespace fs = std::filesystem;
    CachedArtifact out;
    if (!is_convertible(path)) {
        out.path = path;
        return out;
    }
    // Converted as text (and vision) below: the MTP layers are not planned.
    const auto choice = choose_recipe(path, GgufRequest::Model, false);
    out.recipe        = choice.recipe;
    // A GGUF file caches beside itself, named after it (a directory may hold several models'
    // files); a directory caches inside.
    const bool file  = fs::is_regular_file(path);
    const auto cache = (file ? path.parent_path() : path) / ".ninferno-cache";
    std::optional<fs::path> projector;
    if (vision && recipe_has_vision(out.recipe)) {
        if (file) {
            projector = gguf_projector_beside(path);
            vision    = projector.has_value();
        } else {
            vision = checkpoint_has_vision(path);
        }
        if (!vision && log) {
            log(std::string("no Vision encoder converted: ") +
                (file ? "no vision projector (mmproj-*.gguf) beside this GGUF model"
                      : "this checkpoint publishes none NInferno runs"));
        }
    } else {
        vision = false;
    }
    out.path = cache / ((file ? path.stem().string() + "." : std::string()) +
                        (vision ? "vision." : "") + out.recipe + ".ninferno");
    // Taken before the conversion: --consume-source removes the weights it stamps.
    const auto stamp =
        source_stamp(file ? path.parent_path() : path, source_files(path, projector, resources));
    if (serve_cached(out.path, path, stamp)) { return out; }
    fs::create_directories(cache);
    const auto lock_path = cache / ".convert.lock";
    const int lock       = sys::open(lock_path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (lock < 0) { throw std::runtime_error("cannot open " + lock_path.string()); }

    struct Close {
        int fd;

        ~Close() { sys::close(fd); }
    } closer{lock};

    if (sys::lock(lock, true) != 0) {
        throw std::runtime_error("cannot lock " + lock_path.string());
    }
    if (serve_cached(out.path, path, stamp)) { return out; }
    if (fs::exists(out.path)) {
        if (log) {
            log(cache_current(out.path)
                    ? "converting " + path.string() +
                          " again: its files changed since the cached artifact was converted"
                    : "converting " + path.string() +
                          " again: its cached artifact is conversion revision " +
                          std::to_string(cached_revision(out.path)) +
                          ", this NInferno makes revision " + std::to_string(kCacheRevision));
        }
        remove_artifact(out.path);
    }
    // A report without its artifact is what an interrupted conversion leaves behind.
    std::error_code ignored;
    fs::remove(out.path.string() + ".conversion.json", ignored);

    ConvertOptions options;
    options.model          = path;
    options.recipe         = out.recipe;
    options.out            = out.path;
    options.consume_source    = consume_source;
    options.round_tiny_scales = round_tiny_scales;
    options.sources           = choice.sources;
    if (vision) {
        options.components = "text,vision";
        if (projector) { options.sources.emplace_back("vision", *projector); }
    }
    const auto root        = file ? path.parent_path() : path;
    for (const auto& [role, source] : resources) {
        if (!fs::exists(root / role)) { options.resources.emplace_back(role, source); }
    }
    options.log       = log;
    const auto result = convert(options);
    stamp_revision(out.path);
    stamp_source(out.path, stamp);
    out.converted     = true;
    out.seconds       = result.seconds;
    return out;
}

} // namespace ninferno::convert
