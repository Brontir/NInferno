// Generic MoE grouped GEMM, ggml_q6_k weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_ggml_q6k(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                        cudaStream_t stream) {
    launch_moe<generic::GgmlQ6KDecoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
