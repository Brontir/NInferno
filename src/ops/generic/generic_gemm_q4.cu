// Generic register-MMA projection, q4 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GroupwiseG64Decoder<4>>(rmma_decoder_of(WeightCodec::Q4,
                                                                                0)),
              "generic_rmma_decoders.h: the Q4 row differs from GroupwiseG64Decoder<4>");

void launch_rmma_q4(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<GroupwiseG64Decoder<4>>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
