// Generic MoE grouped GEMM, mxfp4 weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_mxfp4(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                    cudaStream_t stream) {
    launch_moe<generic::Mxfp4RmmaDecoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
