#pragma once

// The decode pacing and maintainer flags ninferno, ninferno-serve and spec-tune share
// (EngineOptions::pacing and EngineOptions::maintainer; docs/maintainer/environment.md). No
// environment variable stands in for any of them: a decision changes only with an option given.

#include "core/debug_poison.h"
#include "ninferno/types.h"
#include "product/numeric_options.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::product {

// "a,b,c" as its non-empty items.
[[nodiscard]] inline std::vector<std::string> comma_items(std::string_view text) {
    std::vector<std::string> out;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item  = text.substr(0, comma);
        if (!item.empty()) { out.emplace_back(item); }
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
    return out;
}

// The lab tuners' flags (--dense-tune, --moe-tune and their cache, --tune-cache): they time
// kernels on the machine and store choices per machine, so only the maintainer tools take them
// (kTunerHosts); ninferno and ninferno-serve refuse them
// (docs/maintainer/deterministic-decisions.md: no timing at startup).
[[nodiscard]] inline bool is_tuner_flag(std::string_view arg) noexcept {
    return arg == "--dense-tune" || arg == "--moe-tune" || arg == "--tune-cache";
}

inline constexpr const char* kTunerHosts =
    "ninferno-perplexity (both tuners), ninferno-dense and ninferno-gemma4 (the dense tuner)";

// Whether a program takes the tuner flags.
enum class TunerFlags : std::uint8_t { Refused, Accepted };

// When `arg` is a pacing or maintainer flag, parses its value (value() reads it) into `pacing` or
// `maintainer` and returns true. A tuner flag (is_tuner_flag) where `tuners` refuses it throws.
template <class Value>
bool parse_engine_flag(std::string_view arg, Value&& value, DecodePacingOptions& pacing,
                       MaintainerOptions& maintainer, TunerFlags tuners) {
    if (is_tuner_flag(arg) && tuners == TunerFlags::Refused) {
        throw std::invalid_argument(std::string(arg) +
                                    " is a lab tuner's option, which times kernels on this "
                                    "machine; ninferno and ninferno-serve take none (run it in " +
                                    kTunerHosts + ")");
    }
    if (arg == "--decode-steps") {
        pacing.steps = parse_decode_steps(value());
    } else if (arg == "--decode-wait") {
        pacing.wait = parse_decode_wait(value());
    } else if (arg == "--cuda-sync") {
        pacing.cuda_sync = parse_cuda_sync_schedule(value());
    } else if (arg == "--route-table") {
        maintainer.route_table = value();
    } else if (arg == "--route-dir") {
        maintainer.route_dir = value();
    } else if (arg == "--dense-tune") {
        maintainer.dense_tune = parse_tune_mode(value(), "--dense-tune");
    } else if (arg == "--moe-tune") {
        maintainer.moe_tune = parse_tune_mode(value(), "--moe-tune");
    } else if (arg == "--tune-cache") {
        maintainer.tune_cache = value();
    } else if (arg == "--debug-prefill-cuts") {
        maintainer.debug_prefill_cuts.clear();
        for (const auto& item : comma_items(value())) {
            maintainer.debug_prefill_cuts.push_back(
                parse_u64(item.c_str(), "--debug-prefill-cuts"));
        }
    } else if (arg == "--debug-poison") {
        maintainer.debug_poison = parse_debug_poison(value());
    } else if (arg == "--artifact-io") {
        maintainer.artifact_io = parse_artifact_io(value());
    } else if (arg == "--debug-poison-skip") {
        maintainer.debug_poison_skip = comma_items(value());
    } else if (arg == "--prefill-route") {
        maintainer.prefill_route = parse_prefill_route(value());
    } else if (arg == "--decode-route") {
        maintainer.decode_route = parse_decode_route(value());
    } else if (arg == "--moe-path") {
        maintainer.moe_path = parse_moe_path(value());
    } else {
        return false;
    }
    return true;
}

// The usage lines of those flags, for a program that decodes (no tuners: is_tuner_flag).
inline constexpr const char* kEngineFlagsUsage =
    "       [--decode-steps auto|N] [--decode-wait auto|spin|block]\n"
    "       [--cuda-sync spin|blocking|yield|auto]\n"
    "   maintainers (lab tools and debug aids; docs/maintainer/environment.md):\n"
    "       [--route-table FILE] [--route-dir DIR] [--debug-prefill-cuts P1,P2,...]\n"
    "       [--debug-poison 0|1|nan|BYTE] [--debug-poison-skip kv,ingress,state,weights]\n"
    "       [--artifact-io auto|direct|buffered]\n"
    "       [--prefill-route auto|a16|w4a8|i8|mmq|w4a4] [--decode-route auto|a16|r8|vec|mmq]\n"
    "       [--moe-path auto|fused|generic]\n";

// Without the decode pacing and the decode route pin, and with the tuners, for a program that
// only scores (ninferno-perplexity, a tuner host).
inline constexpr const char* kScoringEngineFlagsUsage =
    "       [--cuda-sync spin|blocking|yield|auto]\n"
    "   maintainers (lab tools and debug aids; docs/maintainer/environment.md):\n"
    "       [--route-table FILE] [--route-dir DIR] [--dense-tune off|on|retune]\n"
    "       [--moe-tune off|on|retune] [--tune-cache DIR] [--debug-prefill-cuts P1,P2,...]\n"
    "       [--debug-poison 0|1|nan|BYTE] [--debug-poison-skip kv,ingress,state,weights]\n"
    "       [--artifact-io auto|direct|buffered]\n"
    "       [--prefill-route auto|a16|w4a8|i8|mmq|w4a4] [--moe-path auto|fused|generic]\n";

// The decode pacing flags, which a scoring program refuses.
[[nodiscard]] inline bool is_decode_pacing_flag(std::string_view arg) noexcept {
    return arg == "--decode-steps" || arg == "--decode-wait";
}

// Every flag that only changes decode rounds (the pacing and --decode-route), which a scoring
// program refuses.
[[nodiscard]] inline bool is_decode_only_flag(std::string_view arg) noexcept {
    return is_decode_pacing_flag(arg) || arg == "--decode-route";
}

} // namespace ninferno::product
