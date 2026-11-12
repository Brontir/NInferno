// Generic register-MMA projection, ggml_iq1_s weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlIq1SDecoder>(rmma_decoder_of(WeightCodec::GgmlIq1S, 0)),
              "generic_rmma_decoders.h: the GgmlIq1S row differs from GgmlIq1SDecoder");

void launch_rmma_ggml_iq1s(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlIq1SDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
