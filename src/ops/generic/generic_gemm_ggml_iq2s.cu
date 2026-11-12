// Generic register-MMA projection, ggml_iq2_s weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlIq2SDecoder>(rmma_decoder_of(WeightCodec::GgmlIq2S, 0)),
              "generic_rmma_decoders.h: the GgmlIq2S row differs from GgmlIq2SDecoder");

void launch_rmma_ggml_iq2s(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlIq2SDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
