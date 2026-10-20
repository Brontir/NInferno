// Generic register-MMA projection, q8 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_q8(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<Q8Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
