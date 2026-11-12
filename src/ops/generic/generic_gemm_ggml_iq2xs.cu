// Generic register-MMA projection, ggml_iq2_xs weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlIq2XsDecoder>(rmma_decoder_of(WeightCodec::GgmlIq2Xs,
                                                                          0)),
              "generic_rmma_decoders.h: the GgmlIq2Xs row differs from GgmlIq2XsDecoder");

void launch_rmma_ggml_iq2xs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlIq2XsDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
