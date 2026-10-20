// Generic MoE grouped GEMM, nvfp4 weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_nvfp4(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                    cudaStream_t stream) {
    launch_moe<generic::Nvfp4Decoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
