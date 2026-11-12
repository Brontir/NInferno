// Generic register-MMA projection, ggml_q6_k weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlQ6KDecoder>(rmma_decoder_of(WeightCodec::GgmlQ6K, 0)),
              "generic_rmma_decoders.h: the GgmlQ6K row differs from GgmlQ6KDecoder");

void launch_rmma_ggml_q6k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlQ6KDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
