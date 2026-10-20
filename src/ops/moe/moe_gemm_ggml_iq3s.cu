// Generic MoE grouped GEMM, ggml_iq3_s weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_ggml_iq3s(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                        cudaStream_t stream) {
    launch_moe<generic::GgmlIq3SDecoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
