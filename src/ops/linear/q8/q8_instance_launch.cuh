#pragma once
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

namespace ninferno::ops::detail {
template <class Geometry, int Capacity, class Schedule>
void launch_q8_a16_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Instance = typename Schedule::template with_problem<Geometry::kInputRows, Capacity>;
    launch_q8_a16_sliced_k_mma<Instance>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
        LinearIdentityEpilogue{}, stream);
}
// A registered shape's GEMV for up to Capacity tokens (decode, a drafter's verify batch).
template <class Geometry, int Capacity, class Tiling>
void launch_q8_a16_gemv_shape(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    using Schedule = typename Tiling::template with_problem<Geometry::kInputRows, Capacity>;
    launch_q8_a16_gemv<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
        LinearIdentityEpilogue{}, stream);
}
} // namespace ninferno::ops::detail
