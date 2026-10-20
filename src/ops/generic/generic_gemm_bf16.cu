// Generic register-MMA projection, bf16 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_bf16(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<Bf16Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
