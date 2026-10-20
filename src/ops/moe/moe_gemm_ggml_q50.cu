// Generic MoE grouped GEMM, ggml_q5_0 weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_ggml_q50(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                        cudaStream_t stream) {
    launch_moe<generic::GgmlQ50Decoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
