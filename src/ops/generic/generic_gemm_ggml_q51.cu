// Generic register-MMA projection, ggml_q5_1 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<GgmlQ51Decoder>(rmma_decoder_of(WeightCodec::GgmlQ51, 0)),
              "generic_rmma_decoders.h: the GgmlQ51 row differs from GgmlQ51Decoder");

void launch_rmma_ggml_q51(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream) {
    launch_rmma<GgmlQ51Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
