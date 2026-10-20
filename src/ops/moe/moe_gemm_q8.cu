// Generic MoE grouped GEMM, q8 weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_q8(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                    cudaStream_t stream) {
    launch_moe<generic::Q8Decoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
