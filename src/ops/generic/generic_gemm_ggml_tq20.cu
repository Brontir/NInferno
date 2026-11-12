// Generic register-MMA projection, ggml_tq2_0 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlTq20Decoder>(rmma_decoder_of(WeightCodec::GgmlTq20, 0)),
              "generic_rmma_decoders.h: the GgmlTq20 row differs from GgmlTq20Decoder");

void launch_rmma_ggml_tq20(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlTq20Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
