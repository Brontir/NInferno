// Generic register-MMA projection, ggml_iq1_s weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_ggml_iq1s(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlIq1SDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
