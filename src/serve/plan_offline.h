#pragma once

// ninferno-serve's plan without the weights (--plan --no-load) and the quantization a GGUF hub
// repository is fetched at without --quant: the model's facts from what it says about itself
// (a hub listing and one GGUF header, a config.json, a local GGUF, checkpoint or artifact), never
// a download of its weights, planned by product/plan/plan_quants.h with the Engine options the
// server builds (engine_options_of) and the speculation it would resolve.

#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace ninferno::serve {

struct OfflinePlan {
    std::string text;    // the terminal report
    nlohmann::json json; // "ninferno-dry-plan/1"
    bool fits = false;   // the chosen plan fits (exit code 0, else 2)
};

// --plan --no-load. Throws std::runtime_error when the model cannot be read (no listing offline,
// no header) or the device cannot be (no GPU and no --plan-device).
[[nodiscard]] OfflinePlan plan_without_load(const ServeOptions& options);

// Before a GGUF hub repository is downloaded without --quant: the largest quantization that fits
// the asked context x --max-concurrency on an idle card of this GPU's class (the rule logs its
// line through `log`). nullopt when the rule does not apply (not a GGUF repository, the listing or
// the GPU unreadable): the fetch then takes its own default. Throws std::runtime_error when no
// quantization fits, before anything is downloaded.
//
// `launch_context`: under an automatic context with several streams, the own context per stream
// the quantization was chosen for: the start pins --max-context to it (ninferno-ui's launch does
// the same), so the streams never share one pool silently. 0 otherwise.
struct HubQuantChoice {
    std::string quant;
    std::uint32_t launch_context = 0;
};

[[nodiscard]] std::optional<HubQuantChoice>
largest_fitting_hub_quant(const ServeOptions& options,
                          const std::function<void(const std::string&)>& log);

// The axes a start leaves to the engine (--kv-dtype auto, no --activations, no --state-dtype),
// chosen for the artifact at `options.artifact_path` on this GPU by the configuration choice
// (product/plan/plan_choice.h): the KV codec, the activation mode and the recurrent state with
// the best worth that keeps the asked context per stream. Its line goes to `log`. nullopt: every
// axis pinned, no GPU facts, the shape unknown, or nothing fits (the start then plans as before
// and says why).
struct ServedAxes {
    std::optional<KvCacheStorage> kv;
    std::optional<KvCacheStorage> ring;
    std::optional<ActivationMode> activations;
    std::optional<RecurrentStateStorage> state;
};
[[nodiscard]] std::optional<ServedAxes>
choose_served_axes(const ServeOptions& options, const std::function<void(const std::string&)>& log);

} // namespace ninferno::serve
