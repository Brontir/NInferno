#pragma once

// The chat window of a running model: llama.cpp's web UI (llama-ui, MIT, a prebuilt static bundle
// fetched at configure time and pinned by checksum, see src/ui/CMakeLists.txt) served under
// /chat/<id>/. The bundle calls its server by relative paths (./props, ./v1/chat/completions,
// ./v1/models); ninferno-ui answers /props itself and forwards the rest to that model's
// ninferno-serve, so the bundle is used as published.

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>

namespace ninferno::ui {

// The directory holding the bundle's index.html: `configured` when given (an error when it lacks
// one), else <exe_dir>/llama-ui, <exe_dir>/../share/ninferno/llama-ui, then the build tree's copy;
// nullopt when there is none (the page then has no chat window).
[[nodiscard]] std::optional<std::filesystem::path>
find_chat_ui(const std::filesystem::path& configured, const std::filesystem::path& exe_dir);

// A request path under the bundle that is safe to open: relative, no "..", no backslash or NUL,
// at most 200 characters. "" (the directory itself) is index.html.
[[nodiscard]] bool safe_bundle_path(const std::string& relative);

// The Content-Type for a bundle file by its extension.
[[nodiscard]] std::string bundle_mime(const std::filesystem::path& file);

// What the served model is, for the bundle's ./props (llama-server's shape, the fields it reads):
// the model's id, its context per request, how many requests it runs at once, whether it takes
// images, and the sampling preset a request that names none runs with (/v1/models
// meta.sampling: the thinking preset, the template default of every model that has two; null
// when the server did not say).
struct ChatModel {
    std::string id;
    long context = 0;
    int slots    = 1;
    bool vision  = false;
    nlohmann::json
        sampling; // {temperature, top_k, top_p, min_p, presence_penalty, frequency_penalty}
};

[[nodiscard]] nlohmann::json chat_props(const ChatModel& model);

// The model's id, context and sampling preset from its ninferno-serve /v1/models body (id "" when
// unreadable).
[[nodiscard]] ChatModel chat_model_of(const std::string& models_body);

// Whether a converted artifact takes images (it carries a vision component); false when unreadable.
[[nodiscard]] bool artifact_has_vision(const std::filesystem::path& artifact);

} // namespace ninferno::ui
