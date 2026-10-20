// Generic register-MMA projection, ggml_q4_0 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_ggml_q40(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlQ40Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
