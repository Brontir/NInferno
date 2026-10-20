#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "ops/common/math.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_exact.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninferno::ops::detail {
namespace {

// Gate row r and up row r + 17408 arrive together (nvfp4_a16_reg_mma_kernel's paired rows): SiLU on
// the FP32 gate times the FP32 up, rounded once, as the fused small-T A16 route does.
struct Nvfp4A16SwiGluOutput {
    static constexpr std::int32_t kPairOffset = Nvfp4N34816K5120::kOutputRows / 2;

    __nv_bfloat16* out; // [17408, T]

    __device__ __forceinline__ void store_pair(std::int32_t gate_row, std::int32_t token,
                                               float gate, float up) const {
        out[static_cast<std::int64_t>(token) * kPairOffset + gate_row] =
            __float2bfloat16_rn(silu(gate) * up);
    }
};

} // namespace

void nvfp4_linear_swiglu_exact_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    launch_nvfp4_a16_exact<Nvfp4N34816K5120>(
        static_cast<const __nv_bfloat16*>(x.data), weight, LinearIdentityEpilogue{},
        Nvfp4A16SwiGluOutput{static_cast<__nv_bfloat16*>(out.data)}, x.ne[1], stream);
}

} // namespace ninferno::ops::detail
