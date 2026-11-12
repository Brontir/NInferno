// Generic register-MMA projection, ggml_iq2_xxs weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlIq2XxsDecoder>(rmma_decoder_of(WeightCodec::GgmlIq2Xxs,
                                                                           0)),
              "generic_rmma_decoders.h: the GgmlIq2Xxs row differs from GgmlIq2XxsDecoder");

void launch_rmma_ggml_iq2xxs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlIq2XxsDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
