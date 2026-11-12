// Generic register-MMA projection, bf16 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<Bf16Decoder>(rmma_decoder_of(WeightCodec::Bf16, 0)),
              "generic_rmma_decoders.h: the Bf16 row differs from Bf16Decoder");

void launch_rmma_bf16(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<Bf16Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
