#pragma once

#include "core/weight.h"
#include "ninferno/ops/linear.h" // LinearPolicy

#include <cstdint>

namespace ninferno::ops::detail {

// Whether the shape-owned linear table of this format registers the exact [N,K]. A shape that is
// not registered takes the generic path (ops/generic/generic_gemm.h).
[[nodiscard]] bool linear_shape_registered(QType qtype, std::int32_t n, std::int32_t k);

// Whether ops::linear(x, weight, out, policy) of `tokens` columns may instead run joined to the
// call before it in one segmented projection (generic Call::join_previous), as the dense pass's
// side-by-side projections do: the generic path takes it (no shape-owned table) with the register
// MMA, in a format the joined launch was measured on (NVFP4, FP8, Q8_G32), or with the R8 decode
// kernel in any format it reads (the GGUF K-quants, i-quants and legacy blocks). The bytes must
// equal ops::linear's (tests/ops/linear/test_side_by_side_projection.cpp checks every format
// against this answer, so a format added here is checked there too).
[[nodiscard]] bool side_by_side_admits(const Weight& weight, LinearPolicy policy,
                                       std::int32_t tokens);

} // namespace ninferno::ops::detail
