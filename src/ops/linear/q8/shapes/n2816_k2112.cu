#include "ops/linear/q8/q8_shapes.h"

// Gemma 4 [2816,2112]. Bands measured on the RTX 5090 with interleaved cold-weight timing over
// T in {1..4096}; each band takes the winner at its upper edge.
namespace ninferno::ops::detail {

Q8Launch select_q8_n2816_k2112(std::int32_t tokens) {
    if (tokens <= 24) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 256) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 512) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 768) return launch_q8_a16_mma_r64_t128;
    return launch_q8_a16_mma_r96_t96;
}

} // namespace ninferno::ops::detail
