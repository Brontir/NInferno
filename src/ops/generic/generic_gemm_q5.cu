// Generic register-MMA projection, q5 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GroupwiseG64Decoder<5>>(rmma_decoder_of(WeightCodec::Q5,
                                                                                0)),
              "generic_rmma_decoders.h: the Q5 row differs from GroupwiseG64Decoder<5>");

void launch_rmma_q5(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<GroupwiseG64Decoder<5>>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
