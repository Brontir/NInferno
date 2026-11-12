// Generic register-MMA projection, mxfp4 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

// The row of the decoder this build launches (Mxfp4RmmaDecoder: the FP4 converter's, or with
// NINFERNO_WITHOUT_NVFP4 the spread one).
static_assert(rmma_decoder_traits_match<Mxfp4RmmaDecoder>(rmma_decoder_of(
                  WeightCodec::Mxfp4, 0, std::is_same_v<Mxfp4RmmaDecoder, Mxfp4SpreadDecoder>)),
              "generic_rmma_decoders.h: the Mxfp4 row differs from Mxfp4RmmaDecoder");

void launch_rmma_mxfp4(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<Mxfp4RmmaDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
