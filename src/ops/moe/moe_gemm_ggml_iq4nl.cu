// Generic MoE grouped GEMM, ggml_iq4_nl weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_ggml_iq4nl(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                        cudaStream_t stream) {
    launch_moe<generic::GgmlIq4NlDecoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
