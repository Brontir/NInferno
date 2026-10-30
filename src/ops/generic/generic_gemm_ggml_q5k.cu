// Generic register-MMA projection, ggml_q5_k weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_ggml_q5k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlQ5KQuadDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
