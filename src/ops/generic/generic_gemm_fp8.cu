// Generic register-MMA projection, fp8 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<Fp8Decoder>(rmma_decoder_of(WeightCodec::Fp8, 0)),
              "generic_rmma_decoders.h: the Fp8 row differs from Fp8Decoder");

void launch_rmma_fp8(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<Fp8Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
