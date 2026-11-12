// Generic register-MMA projection, u4_g128_zp_fp16 weights: every compiled schedule and epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<U4ZpDecoder>(rmma_decoder_of(WeightCodec::U4Zp, 0)),
              "generic_rmma_decoders.h: the U4Zp row differs from U4ZpDecoder");

void launch_rmma_u4zp(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    launch_rmma<U4ZpDecoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
