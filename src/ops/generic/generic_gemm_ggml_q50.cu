// Generic register-MMA projection, ggml_q5_0 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlQ50Decoder>(rmma_decoder_of(WeightCodec::GgmlQ50, 0)),
              "generic_rmma_decoders.h: the GgmlQ50 row differs from GgmlQ50Decoder");

void launch_rmma_ggml_q50(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlQ50Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
