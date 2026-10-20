#include "ops/linear/q6/q6_shapes.h"

#include "core/device_class.h"

namespace ninferno::ops::detail {

// Gemma 4 31B tied output head. The SIMT launchers keep decode near the stream rate through
// T=6; from seven tokens the 16-column MMA tile is flat to T=16. Each wider tile is chosen at
// the T where its last column fills, the band edges the sweep over T measured.
Q6Launch select_q6_n262144_k5376(std::int32_t tokens) {
    if (tokens <= 4) return launch_q6_a16_simt_r8_t4;
    if (tokens <= 5) return launch_q6_a16_simt_r8_t5;
    if (tokens <= 6) return launch_q6_a16_simt_r8_t6;
    // Without tensor mma.sync (Volta's build) the 16-column tile runs on FP32 FMAs; the sliced
    // tile's launch takes those widths to the SIMT kernel there.
    if (tokens <= 16 && !build_features().mma_tensor) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 16) return launch_q6_a16_mma_r64_t16_k128;
    if (tokens <= 24) return launch_q6_a16_mma_r64_t24_k128;
    if (tokens <= 32) return launch_q6_a16_mma_r64_t32_k128;
    if (tokens <= 40) return launch_q6_a16_mma_r64_t40_k128;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 56) return launch_q6_a16_mma_r64_t56_k128;
    if (tokens <= 64) return launch_q6_a16_mma_r64_t64_k128;
    if (tokens <= 72) return launch_q6_a16_mma_r64_t72_k128;
    if (tokens <= 80) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q6_a16_mma_r64_t112;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninferno::ops::detail
