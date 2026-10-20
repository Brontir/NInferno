// Generic MoE grouped GEMM, u4_g128_zp_fp16 weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_u4zp(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                      cudaStream_t stream) {
    launch_moe<generic::U4ZpDecoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
