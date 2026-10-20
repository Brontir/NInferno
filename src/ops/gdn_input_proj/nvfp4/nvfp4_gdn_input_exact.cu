#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_exact.cuh"

#include <cuda_bf16.h>

namespace ninferno::ops::detail {

void nvfp4_gdn_input_exact_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  cudaStream_t stream) {
    launch_nvfp4_a16_exact<Nvfp4N16384K5120>(
        static_cast<const __nv_bfloat16*>(x.data), weight, LinearIdentityEpilogue{},
        Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                            static_cast<__nv_bfloat16*>(z.data)},
        x.ne[1], stream);
}

} // namespace ninferno::ops::detail
