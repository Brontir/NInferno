// Generic register-MMA projection, ggml_q2_0 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_ggml_q20(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlQ20Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
