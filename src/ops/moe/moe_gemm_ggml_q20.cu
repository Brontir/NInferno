// Generic MoE grouped GEMM, ggml_q2_0 weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_ggml_q20(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                        cudaStream_t stream) {
    launch_moe<generic::GgmlQ20Decoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
