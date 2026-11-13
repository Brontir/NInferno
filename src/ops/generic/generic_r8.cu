// Generic R8 (W8A8 decode): the codec dispatch, q8_g32, q4/q5/q6_g64, u4_g{32,64,128}_zp and
// q3_g128 / q3_g64 (NINFERNO_R8_BASE_CODECS, compiled_codecs.h).
#include "ops/generic/generic_r8.cuh"

#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_gemm_internal.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

bool launch_r8_base(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                    cudaStream_t stream, W4a8Activation pre) {
    switch (codec) {
        NINFERNO_R8_BASE_CODECS(NINFERNO_R8_LAUNCH_CASE)
    default:
        return false;
    }
}

} // namespace

void launch_r8(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
               cudaStream_t stream, W4a8Activation pre) {
    if (launch_r8_base(schedule, codec, epilogue, args, stream, pre)) return;
    if (launch_r8_iq3(schedule, codec, epilogue, args, stream, pre)) return;
    if (launch_r8_iq2(schedule, codec, epilogue, args, stream, pre)) return;
    if (launch_r8_ggml4(schedule, codec, epilogue, args, stream, pre)) return;
    if (launch_r8_kquant(schedule, codec, epilogue, args, stream, pre)) return;
    throw std::invalid_argument("generic R8: the weight format has no R8 codec");
}

} // namespace ninferno::ops::detail::generic
