// Generic MoE grouped GEMM, ggml_iq3_xxs banks in ggml_frag_v1: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_ggml_iq3xxs_frag(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                                  cudaStream_t stream) {
    launch_moe<generic::GgmlIq3XxsFragDecoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
