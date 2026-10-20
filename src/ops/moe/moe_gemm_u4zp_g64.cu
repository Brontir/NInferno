// Generic MoE grouped GEMM, u4_g64_zp_fp16 weights: every compiled schedule and stage. K in whole
// 128-value steps takes the quad-run spread decoder, any other K the one-group-per-step decoder.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_u4zp_g64(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                          cudaStream_t stream) {
    if (args.w.k % 128 == 0) {
        launch_moe<generic::U4ZpG64QuadDecoder>(stage, schedule, args, grid, stream);
    } else {
        launch_moe<generic::U4ZpG64Decoder>(stage, schedule, args, grid, stream);
    }
}

} // namespace ninferno::ops::detail::moe
