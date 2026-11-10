// Generic register-MMA projection, q6 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GroupwiseG64Decoder<6>>(rmma_decoder_of(WeightCodec::Q6,
                                                                                0)),
              "generic_rmma_decoders.h: the Q6 row differs from GroupwiseG64Decoder<6>");

void launch_rmma_q6(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<GroupwiseG64Decoder<6>>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
