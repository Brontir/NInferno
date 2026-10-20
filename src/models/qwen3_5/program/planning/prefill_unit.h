#pragma once

#include <cstdint>
#include <optional>

namespace ninferno::models::qwen3_5::runtime_support {

// The end a staged prefill unit sizes its calls toward: the next capture frontier inside the
// prompt, else the prompt end. A unit stops on that frontier (its offer is taken between units),
// and the admission plan projects ceil(segment / chunk) service units for every segment between
// frontiers. Calls sized toward the prompt end instead stop a few tokens short of the frontier
// when the family shortens a call before a narrow remainder, and the request runs one unit more
// than its plan: the Engine then fails it and every request in its batch.
[[nodiscard]] constexpr std::uint32_t
staged_prefill_unit_end(std::uint32_t cursor, std::uint32_t prompt_tokens,
                        std::optional<std::uint32_t> capture_frontier) noexcept {
    return capture_frontier && *capture_frontier > cursor && *capture_frontier < prompt_tokens
               ? *capture_frontier
               : prompt_tokens;
}

} // namespace ninferno::models::qwen3_5::runtime_support
