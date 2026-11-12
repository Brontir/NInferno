// Generic register-MMA projection, ggml_q4_1 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlQ41Decoder>(rmma_decoder_of(WeightCodec::GgmlQ41, 0)),
              "generic_rmma_decoders.h: the GgmlQ41 row differs from GgmlQ41Decoder");

void launch_rmma_ggml_q41(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlQ41Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
