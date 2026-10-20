// Generic register-MMA projection, fp8_e4m3fn_b128_f32 weights: every compiled schedule and
// epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_fp8_b128(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<Fp8B128Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
