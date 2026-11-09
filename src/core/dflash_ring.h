#pragma once

#include <cmath>
#include <cstdint>
#include <optional>

namespace ninferno::core {

// The context length a DFlash drafter's YaRN rope was scaled from, when it is shorter than the
// positions the drafter addresses (0: none), from its resolved rope (convert::resolved_rope). The
// converter records it (original_max_position_embeddings, only when it is below
// max_position_embeddings). An artifact converted before that is told by YaRN's default attention
// factor, 0.1 ln(factor) + 1, with factor = max / original a whole number; it serves only the
// gpt-oss drafters cached then (1.3466 -> 32, 131072 / 32 = 4096). An attention_factor or mscale
// set explicitly in the checkpoint gives no whole factor there: no length. The runtime
// (speculation::parse_dflash_config) and the plan without weights (plan_drafter_state_of_config)
// read it here, so they agree.
[[nodiscard]] inline std::uint32_t yarn_original_length(std::optional<std::uint64_t> recorded,
                                                        bool yarn, float attention_scaling,
                                                        std::uint32_t max_positions) noexcept {
    if (max_positions == 0) { return 0; }
    if (recorded) { return *recorded < max_positions ? static_cast<std::uint32_t>(*recorded) : 0; }
    if (!yarn || !(attention_scaling > 1.0F)) { return 0; }
    const double factor = std::exp((static_cast<double>(attention_scaling) - 1.0) / 0.1);
    const double whole  = std::round(factor);
    if (whole < 2.0 || std::fabs(factor - whole) > 1e-3 * whole ||
        max_positions % static_cast<std::uint32_t>(whole) != 0) {
        return 0;
    }
    return max_positions / static_cast<std::uint32_t>(whole);
}

// The ring a drafter with no sliding layer runs every layer in (#564), for the bound above: the
// cyclic rings hold 2048 or 4096 rows, at D64 (gpt-oss's) only (a D128 drafter of this shape would
// run on the Qwen Program's int8 local path, never qualified for it). 0: the layers stay paged.
[[nodiscard]] constexpr std::uint32_t dflash_ring_window(std::uint32_t bound,
                                                         std::uint32_t head_dim) noexcept {
    return head_dim == 64 && (bound == 2048 || bound == 4096) ? bound : 0;
}

} // namespace ninferno::core
