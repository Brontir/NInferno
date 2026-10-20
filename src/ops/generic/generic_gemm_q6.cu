// Generic register-MMA projection, q6 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_q6(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<GroupwiseG64Decoder<6>>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
