#pragma once

// The tensor work of the register MMA's min term (#501), counted from the kernel (mma_step in
// generic_gemm.cuh). A decoder with a min term (GgmlQ45KDecoder: Q4_K, Q5_K; GgmlQ2KDecoder: Q2_K;
// Ggml32Decoder with kWithMin: Q4_1, Q5_1) needs the activation sums of each scale group, and
// where the tile does not add them in FP32 it takes them with an MMA of the group's activations
// against ones (the `sums` branch of mma_step, `mma_operands(... kOnes ...)`, G MMAs per token
// tile n and group). That MMA is shared by the MT row tiles and both sets of a SwiGLU tile, while
// the weight MMAs of the group are Sets x MT x NT x G. So the tensor pipe issues
// (Sets x MT + 1) / (Sets x MT) of the weight MMAs the cost model counts: 2.0 at MT = 1, 1.5 at
// MT = 2 (checked against ncu's exact HMMA counts of 17 launches, Q4_K 4096x3840 at T = 24: m1.n1
// 2.0x, m2.n2 and m2.n1 1.5x, m1.n4 and up 1.0x).
// The FP32 adds (`add` in mma_step) replace it where the tile has four or more token tiles, the
// call asks for them (GemmArgs::add_sums, rmma_add_sums) and the operands are BF16; Volta's build
// has no tensor MMA and always adds.

#include "core/weight.h"

namespace ninferno::ops::detail::generic {

// Whether the register-MMA decoder of the format has a min term (MinTermOf in generic_gemm.cuh).
constexpr bool rmma_has_min_term(QType q) noexcept {
    return q == QType::GGML_Q4_K || q == QType::GGML_Q5_K || q == QType::GGML_Q2_K ||
           q == QType::GGML_Q4_1 || q == QType::GGML_Q5_1;
}

// The tensor work of a tile over the weight MMAs alone: `sets` row sets (2 for SwiGLU), `m_tiles`
// row tiles per warp. `adds_sums`: the tile sums the activations with FP32 adds, not an MMA.
constexpr double rmma_min_term_factor(QType q, int sets, int m_tiles, bool adds_sums) noexcept {
    if (!rmma_has_min_term(q) || adds_sums) return 1.0;
    const double weight_mmas = static_cast<double>(sets) * m_tiles;
    return (weight_mmas + 1.0) / weight_mmas;
}

} // namespace ninferno::ops::detail::generic
