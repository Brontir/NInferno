// Generic register-MMA projection, fp8_e4m3fn_row_f32 weights: every compiled schedule and
// epilogue.
#include "ops/generic/generic_gemm_launch.cuh"

namespace ninferno::ops::detail::generic {

static_assert(rmma_decoder_traits_match<Fp8RowF32Decoder>(rmma_decoder_of(WeightCodec::Fp8RowF32,
                                                                          0)),
              "generic_rmma_decoders.h: the Fp8RowF32 row differs from Fp8RowF32Decoder");

void launch_rmma_fp8_row_f32(int schedule, Epilogue epilogue, const GemmArgs& args,
                             cudaStream_t stream) {
    launch_rmma<Fp8RowF32Decoder>(schedule, epilogue, args, stream);
}

} // namespace ninferno::ops::detail::generic
