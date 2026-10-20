// Generic MoE grouped GEMM, q5 weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_q5(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                    cudaStream_t stream) {
    launch_moe<generic::GroupwiseG64Decoder<5>>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
