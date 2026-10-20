#include "ui/chat_ui.h"

#include "artifact/reader.h"

#include <array>
#include <utility>

#ifndef NINFERNO_UI_CHAT_BUILD_DIR
#    define NINFERNO_UI_CHAT_BUILD_DIR ""
#endif

namespace ninferno::ui {

namespace fs = std::filesystem;
using Json   = nlohmann::json;

std::optional<fs::path> find_chat_ui(const fs::path& configured, const fs::path& exe_dir) {
    std::error_code ec;
    const auto has_index = [&](const fs::path& dir) {
        return !dir.empty() && fs::is_regular_file(dir / "index.html", ec);
    };
    if (!configured.empty()) {
        if (!has_index(configured)) {
            throw std::invalid_argument(configured.string() + " holds no index.html (--chat-ui)");
        }
        return configured;
    }
    for (const fs::path& dir :
         {exe_dir / "llama-ui", exe_dir / ".." / "share" / "ninferno" / "llama-ui",
          fs::path(NINFERNO_UI_CHAT_BUILD_DIR)}) {
        if (has_index(dir)) { return fs::weakly_canonical(dir, ec); }
    }
    return std::nullopt;
}

bool safe_bundle_path(const std::string& relative) {
    if (relative.size() > 200 || relative.starts_with('/')) { return false; }
    for (const char c : relative) {
        if (c == '\\' || c == '\0' || c == ':') { return false; }
    }
    std::size_t start = 0;
    while (start <= relative.size()) {
        const auto end     = relative.find('/', start);
        const auto segment = relative.substr(start, end == std::string::npos ? end : end - start);
        if (segment == ".." || segment == ".") { return false; }
        if (end == std::string::npos) { break; }
        start = end + 1;
    }
    return true;
}

std::string bundle_mime(const fs::path& file) {
    static const std::array<std::pair<const char*, const char*>, 13> kTypes{{
        {".html", "text/html; charset=utf-8"},
        {".js", "text/javascript; charset=utf-8"},
        {".css", "text/css; charset=utf-8"},
        {".json", "application/json"},
        {".webmanifest", "application/manifest+json"},
        {".svg", "image/svg+xml"},
        {".png", "image/png"},
        {".ico", "image/x-icon"},
        {".jpg", "image/jpeg"},
        {".woff", "font/woff"},
        {".woff2", "font/woff2"},
        {".wasm", "application/wasm"},
        {".txt", "text/plain; charset=utf-8"},
    }};
    const auto ext = file.extension().string();
    for (const auto& [suffix, type] : kTypes) {
        if (ext == suffix) { return type; }
    }
    return "application/octet-stream";
}

Json chat_props(const ChatModel& model) {
    // The sampling fields stay at llama-server's defaults: ninferno-serve applies the model card's
    // presets to a request that names none, and the bundle sends only what its user changes.
    const Json params{{"n_predict", -1},
                      {"seed", -1},
                      {"temperature", 0.8},
                      {"top_k", 40},
                      {"top_p", 0.95},
                      {"min_p", 0.05},
                      {"max_tokens", -1},
                      {"stream", true},
                      {"stop", Json::array()},
                      {"samplers", Json::array()},
                      {"reasoning_format", "auto"},
                      {"timings_per_token", false},
                      {"lora", Json::array()}};
    return Json{{"default_generation_settings",
                 {{"id", 0},
                  {"id_task", -1},
                  {"n_ctx", model.context},
                  {"speculative", false},
                  {"is_processing", false},
                  {"params", params},
                  {"prompt", ""}}},
                {"total_slots", model.slots < 1 ? 1 : model.slots},
                {"model_path", model.id},
                {"model_alias", model.id},
                {"role", "model"},
                {"modalities", {{"vision", model.vision}, {"audio", false}, {"video", false}}},
                {"chat_template", ""},
                {"bos_token", ""},
                {"eos_token", ""},
                {"build_info", "ninferno"},
                {"ui_settings", Json::object()},
                {"cors_proxy_enabled", false}};
}

ChatModel chat_model_of(const std::string& models_body) {
    ChatModel out;
    const auto models = Json::parse(models_body, nullptr, false);
    if (models.is_discarded() || !models.contains("data") || !models["data"].is_array()) {
        return out;
    }
    for (const auto& model : models["data"]) {
        if (!model.is_object() || !model.contains("id") || !model["id"].is_string()) { continue; }
        out.id      = model["id"].get<std::string>();
        out.context = model.value("max_model_len", 0L);
        break;
    }
    return out;
}

bool artifact_has_vision(const fs::path& artifact) {
    try {
        const artifact::Reader reader(artifact);
        return reader.directory().components.contains("vision");
    } catch (const std::exception&) { return false; }
}

} // namespace ninferno::ui
