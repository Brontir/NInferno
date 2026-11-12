// Generic register-MMA projection, ggml_iq4_nl weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlIq4NlDecoder>(rmma_decoder_of(WeightCodec::GgmlIq4Nl,
                                                                          0)),
              "generic_rmma_decoders.h: the GgmlIq4Nl row differs from GgmlIq4NlDecoder");

void launch_rmma_ggml_iq4nl(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlIq4NlDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
