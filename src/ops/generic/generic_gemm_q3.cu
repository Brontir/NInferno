// Generic register-MMA projection, 3-bit groupwise weights (G128, G64): every compiled schedule
// and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

void launch_rmma_q3g128(int schedule, Epilogue epilogue, const GemmArgs& args,
                        cudaStream_t stream) {
    launch_rmma<GroupwiseQ3Decoder<128>>(schedule, epilogue, args, stream);
}

void launch_rmma_q3g64(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<GroupwiseQ3Decoder<64>>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
