#include "convert/cache.h"

#include "convert/acquire.h"
#include "convert/convert.h"
#include "convert/safetensors.h"

#include "core/sys.h"

#include <fstream>
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

void stamp_revision(const std::filesystem::path& artifact) {
    std::ofstream out(artifact.string() + ".revision", std::ios::trunc);
    out << kCacheRevision << "\n";
}

bool checkpoint_has_vision(const std::filesystem::path& checkpoint) {
    std::ifstream stream(checkpoint / "config.json");
    if (!stream) { return false; }
    try {
        const OJson config = OJson::parse(stream);
        return config.contains("vision_config") && config["vision_config"].is_object();
    } catch (const std::exception&) { return false; }
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

// A cached artifact to serve as it is: current, or older with nothing left to convert from.
bool serve_cached(const std::filesystem::path& artifact, const std::filesystem::path& source,
                  const std::string& recipe) {
    return std::filesystem::exists(artifact) &&
           (cached_revision(artifact) >= cache_revision_needed(recipe) || !source_present(source));
}

} // namespace

CachedArtifact
cached_artifact(const std::filesystem::path& path,
                const std::vector<std::pair<std::string, std::filesystem::path>>& resources,
                const std::function<void(const std::string&)>& log, bool consume_source,
                bool vision) {
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
    if (serve_cached(out.path, path, out.recipe)) { return out; }
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
    if (serve_cached(out.path, path, out.recipe)) { return out; }
    if (fs::exists(out.path)) {
        if (log) {
            log("converting " + path.string() + " again: its cached artifact is conversion revision " +
                std::to_string(cached_revision(out.path)) + ", this NInferno makes revision " +
                std::to_string(kCacheRevision));
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
    options.consume_source = consume_source;
    options.sources        = choice.sources;
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
    out.converted     = true;
    out.seconds       = result.seconds;
    return out;
}

} // namespace ninferno::convert
