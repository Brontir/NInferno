// Generic register-MMA projection, u4_g64_zp_fp16 weights: every compiled schedule and epilogue.
// K in whole 128-value steps takes the quad-run spread decoder (one 128-value step, 128/64 groups);
// any other K the one-group-per-step decoder.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_u4zp_g64(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    if (args.k % 128 == 0) {
        launch_rmma<U4ZpG64QuadDecoder>(schedule, epilogue, args, stream);
    } else {
        launch_rmma<U4ZpG64Decoder>(schedule, epilogue, args, stream);
    }
}

} // namespace ninferno::ops::detail::generic
