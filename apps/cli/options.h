#pragma once

#include "core/device_identity.h"
#include "ninferno/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t prefill_chunk  = 1024;
    DeviceSelector device_selector; // --device; main sets `device` from it (select_device)
    int device = kDefaultDevice;

    // --kv-dtype auto (or omitted): the model family chooses (EngineOptions::kv_cache_automatic).
    std::optional<KvCacheStorage> kv_cache;
    KvCacheStorage sliding_kv_cache = KvCacheStorage::BFloat16; // --sliding-kv-dtype
    // --state-dtype (omitted: EngineOptions::recurrent_state_automatic).
    std::optional<RecurrentStateStorage> recurrent_state;
    ActivationMode activations = kDefaultActivationMode; // --activations exact|fast
    float rope_yarn_factor     = 1.0F;                   // --rope-yarn-factor
    SpeculativeOptions speculative;
    // --spec given explicitly (mtp|dflash|dflash2|none); otherwise the artifact's speculation
    // profile, when it came with one, decides (--spec auto says so explicitly).
    bool spec_explicit = false;
    // --spec-profile PATH: a maintainer's own tune in place of the published profile (read only
    // when named).
    std::filesystem::path spec_profile;
    // Watchdog floor in tokens per round (0: the profile's, or off without a profile).
    std::optional<float> acceptance_floor;
    bool enable_vision  = false;
    bool use_cuda_graph = true;

    bool raw_output      = false;
    bool print_token_ids = false;
    std::optional<bool> enable_thinking;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy                 = false;
    DecodePacingOptions pacing;   // --decode-steps, --decode-wait, --cuda-sync
    MaintainerOptions maintainer; // --route-table ... --debug-poison-skip
    // --drafter-registry FILE (spec::Registry::use_file, before the model's profile is read).
    std::filesystem::path drafter_registry;
    product::LogLevel log_level = product::LogLevel::Info;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninferno::cli
