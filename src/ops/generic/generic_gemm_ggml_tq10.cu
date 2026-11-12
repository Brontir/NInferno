// Generic register-MMA projection, ggml_tq1_0 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlTq10Decoder>(rmma_decoder_of(WeightCodec::GgmlTq10, 0)),
              "generic_rmma_decoders.h: the GgmlTq10 row differs from GgmlTq10Decoder");

void launch_rmma_ggml_tq10(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlTq10Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
