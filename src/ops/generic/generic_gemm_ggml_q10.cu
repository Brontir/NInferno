// Generic register-MMA projection, ggml_q1_0 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_ggml_q10(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlQ10Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
