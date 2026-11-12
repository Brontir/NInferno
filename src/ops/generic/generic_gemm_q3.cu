// Generic register-MMA projection, 3-bit groupwise weights (G128, G64): every compiled schedule
// and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(
    rmma_decoder_traits_match<GroupwiseQ3Decoder<128>>(rmma_decoder_of(WeightCodec::Q3G128, 0)),
    "generic_rmma_decoders.h: the Q3G128 row differs from GroupwiseQ3Decoder<128>");
static_assert(rmma_decoder_traits_match<GroupwiseQ3Decoder<64>>(rmma_decoder_of(WeightCodec::Q3G64,
                                                                                0)),
              "generic_rmma_decoders.h: the Q3G64 row differs from GroupwiseQ3Decoder<64>");

void launch_rmma_q3g128(int schedule, Epilogue epilogue, const GemmArgs& args,
                        cudaStream_t stream) {
    launch_rmma<GroupwiseQ3Decoder<128>>(schedule, epilogue, args, stream);
}

void launch_rmma_q3g64(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<GroupwiseQ3Decoder<64>>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
