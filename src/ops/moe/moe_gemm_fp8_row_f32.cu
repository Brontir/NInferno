// Generic MoE grouped GEMM, fp8_row_f32 weights: every compiled schedule and stage.
#include "ops/moe/moe_gemm.cuh"

namespace ninferno::ops::detail::moe {

void launch_gemm_fp8_row_f32(Stage stage, int schedule, const GemmArgs& args, dim3 grid,
                        cudaStream_t stream) {
    launch_moe<generic::Fp8RowF32Decoder>(stage, schedule, args, grid, stream);
}

} // namespace ninferno::ops::detail::moe
