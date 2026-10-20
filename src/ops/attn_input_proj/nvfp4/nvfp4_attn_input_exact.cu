#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_exact.cuh"

#include <cuda_bf16.h>

namespace ninferno::ops::detail {

void nvfp4_attn_input_exact_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_nvfp4_a16_exact<Nvfp4N14336K5120>(
        static_cast<const __nv_bfloat16*>(x.data), weight, LinearIdentityEpilogue{},
        Nvfp4AttentionInputPlanes{static_cast<__nv_bfloat16*>(q.data),
                                  static_cast<__nv_bfloat16*>(k.data),
                                  static_cast<__nv_bfloat16*>(gate.data),
                                  static_cast<__nv_bfloat16*>(v.data)},
        x.ne[1], stream);
}

} // namespace ninferno::ops::detail
