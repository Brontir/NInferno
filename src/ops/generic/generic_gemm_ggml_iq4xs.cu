// Generic register-MMA projection, ggml_iq4_xs weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlIq4XsDecoder>(rmma_decoder_of(WeightCodec::GgmlIq4Xs,
                                                                          0)),
              "generic_rmma_decoders.h: the GgmlIq4Xs row differs from GgmlIq4XsDecoder");

void launch_rmma_ggml_iq4xs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlIq4XsDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
