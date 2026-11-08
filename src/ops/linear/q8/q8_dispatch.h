#pragma once

#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ops/linear/q8/q8_launch.h"

#include <cstdint>
#include <optional>
#include <string>

namespace ninferno::ops::detail {

// Whether the shape-owned table registers this exact [N,K] (the fast path).
bool q8_linear_registered(std::int32_t n, std::int32_t k);
Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);
// The launch a registered shape's table selects at `t` (A16, the table's only compute), as
// "n<N>_k<K>.launch<i>": its i-th distinct launch in the order the widths 1, 2, ... first select
// them, so the identity changes exactly where the table's launch does (linear_candidates).
[[nodiscard]] std::string q8_launch_id(std::int32_t n, std::int32_t k, std::int32_t t);
// The CTA tile of a launch of the table (output rows, token columns): every column tile of CTAs
// requests the whole weight again, every row tile the whole activation. nullopt for a launch whose
// name does not give it.
struct Q8LaunchTile {
    std::int32_t rows;
    std::int32_t tokens;
};
[[nodiscard]] std::optional<Q8LaunchTile> q8_launch_tile(Q8Launch launch);

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

} // namespace ninferno::ops::detail
