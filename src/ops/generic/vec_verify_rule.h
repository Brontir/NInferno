#pragma once

// Which formats ggml's vector kernel takes by rule at the verify widths (vec_rule,
// generic_gemm.cpp). No CUDA here, so a host test checks it.
//
// With tensor mma.sync the vector kernel takes T = 2..4 of the formats it reads, except Q6_K:
// there the register MMA ties or wins on every card of the X6 route matrices (21.10, best route of
// each family per shape, A8 policy, median vector/register-MMA time at T = 2..4): RTX 5070 Ti 1.00,
// RTX 4090 1.02, RTX 4060 Ti 1.04, RTX 3060 1.09, RTX 3090 1.13, RTX 2080 Ti 1.32, A100 1.35.
// Q4_K and Q5_K stay on the vector kernel (0.78..0.96 there). Without tensor mma.sync (Volta,
// TU116/TU117) the register MMA runs on FP32 FMAs and the vector kernel keeps every format.

#include "core/weight.h"

namespace ninferno::ops::detail::generic {

[[nodiscard]] constexpr bool vec_verify_format(QType qtype, bool tensor_mma) noexcept {
    return !tensor_mma || qtype != QType::GGML_Q6_K;
}

} // namespace ninferno::ops::detail::generic
