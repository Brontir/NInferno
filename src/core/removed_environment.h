#pragma once

// The NINFERNO_* variables that used to steer a decision and were replaced by options
// (docs/maintainer/environment.md). No program reads them to decide anything; every product binary
// warns once at startup about each one still set, naming its replacement, so a script that still
// sets one learns that it changes nothing.

#include <array>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno {

struct RemovedEnvironmentVariable {
    std::string_view name;
    std::string_view replacement; // empty: removed with no replacement
};

inline constexpr std::array<RemovedEnvironmentVariable, 17> kRemovedEnvironmentVariables{{
    {"NINFERNO_CUDA_SYNC", "--cuda-sync"},
    {"NINFERNO_DECODE_STEPS", "--decode-steps"},
    {"NINFERNO_DECODE_WAIT", "--decode-wait"},
    {"NINFERNO_ROUTE_TABLE", "--route-table"},
    {"NINFERNO_ROUTE_DIR", "--route-dir"},
    {"NINFERNO_DENSE_TUNE",
     "--dense-tune in a maintainer tool (ninferno-perplexity, ninferno-dense, ninferno-gemma4)"},
    {"NINFERNO_DENSE_TUNE_CACHE",
     "--tune-cache in a maintainer tool (ninferno-perplexity, ninferno-dense, ninferno-gemma4)"},
    {"NINFERNO_MOE_TUNE", "--moe-tune in ninferno-perplexity"},
    {"NINFERNO_MOE_TUNE_CACHE", "--tune-cache in ninferno-perplexity"},
    {"NINFERNO_QWEN_PREFILL_CUTS", "--debug-prefill-cuts"},
    {"NINFERNO_DEBUG_POISON", "--debug-poison"},
    {"NINFERNO_DEBUG_POISON_SKIP", "--debug-poison-skip"},
    {"NINFERNO_DRAFTER_REGISTRY", "--drafter-registry"},
    {"NINFERNO_SPEC_CORPUS", "--corpus (ninferno spec-tune, ninferno-gemma4)"},
    {"NINFERNO_RANKING_DIR", "ninferno-convert --ranking FILE"},
    {"NINFERNO_MMQ_VERIFY", ""},
    {"NINFERNO_ARTIFACT_DIRECT_IO", "--artifact-io"},
}};

// One message per removed variable `is_set` reports set, in the table's order.
[[nodiscard]] inline std::vector<std::string>
removed_environment_warnings(const std::function<bool(const char* name)>& is_set) {
    std::vector<std::string> out;
    for (const RemovedEnvironmentVariable& variable : kRemovedEnvironmentVariables) {
        const std::string name(variable.name);
        if (!is_set(name.c_str())) { continue; }
        std::string message = name + " is set but no longer read and changes nothing; ";
        message += variable.replacement.empty()
                       ? std::string("it was removed with no replacement")
                       : "use " + std::string(variable.replacement) + " instead";
        out.push_back(std::move(message));
    }
    return out;
}

// Writes those messages for this process's environment to stderr, once per process.
inline void warn_removed_environment(std::string_view program) {
    static std::once_flag once;
    std::call_once(once, [program] {
        for (const std::string& message : removed_environment_warnings(
                 [](const char* name) { return std::getenv(name) != nullptr; })) {
            std::fprintf(stderr, "%.*s: warning: %s\n", static_cast<int>(program.size()),
                         program.data(), message.c_str());
        }
    });
}

} // namespace ninferno
