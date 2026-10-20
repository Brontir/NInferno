// Generic register-MMA projection, ggml_iq3_xxs weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_ggml_iq3xxs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlIq3XxsDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
