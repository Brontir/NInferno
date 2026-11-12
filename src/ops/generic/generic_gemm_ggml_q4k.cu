// Generic register-MMA projection, ggml_q4_k weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlQ4KQuadDecoder>(rmma_decoder_of(WeightCodec::GgmlQ4K,
                                                                            0)),
              "generic_rmma_decoders.h: the GgmlQ4K row differs from GgmlQ4KQuadDecoder");

void launch_rmma_ggml_q4k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlQ4KQuadDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
