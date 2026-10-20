// Generic MoE grouped GEMM, fp8_b128 weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_fp8_b128(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                        cudaStream_t stream) {
    launch_moe<generic::Fp8B128Decoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
