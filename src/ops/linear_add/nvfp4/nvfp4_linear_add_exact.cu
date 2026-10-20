#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "ops/linear/nvfp4/nvfp4_a16_exact.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/common/epilogue.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

// The add is an explicit round-to-nearest add, never contracted with the `accumulator * alpha`
// product in front of it: a token's result must not depend on which GEMM its call selected.
struct Nvfp4AddResidualEpilogue {
    const __nv_bfloat16* residual;
    std::int32_t rows;

    __device__ __forceinline__ float apply(std::int32_t row, std::int32_t token,
                                           float value) const {
        return __fadd_rn(value,
                         __bfloat162float(residual[static_cast<std::int64_t>(token) * rows + row]));
    }
};


template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    auto* output = static_cast<__nv_bfloat16*>(residual.data);
    launch_nvfp4_a16_exact<Geometry>(static_cast<const __nv_bfloat16*>(x.data), weight,
                                     Nvfp4AddResidualEpilogue{output, Geometry::kOutputRows},
                                     LinearBf16Output{output, Geometry::kOutputRows}, x.ne[1],
                                     stream);
}

} // namespace

void nvfp4_linear_add_exact_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    if (weight.n == 5120 && weight.k == 6144) {
        launch<Nvfp4N5120K6144>(x, weight, residual, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 17408) {
        launch<Nvfp4N5120K17408>(x, weight, residual, stream);
        return;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninferno::ops::detail
