// Generic register-MMA projection, ggml_tq2_0 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_ggml_tq20(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlTq20Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
