// Generic register-MMA projection, ggml_q3_k weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlQ3KDecoder>(rmma_decoder_of(WeightCodec::GgmlQ3K, 0)),
              "generic_rmma_decoders.h: the GgmlQ3K row differs from GgmlQ3KDecoder");

void launch_rmma_ggml_q3k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlQ3KDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
