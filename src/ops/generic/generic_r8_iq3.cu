// Generic R8 (W8A8 decode) for IQ3_XXS and IQ3_S.
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_r8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_r8_iq3(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream, W4a8Activation pre) {
    switch (codec) {
    case WeightCodec::GgmlIq3Xxs:
        r8_launch_codec<W4a8Iq3Xxs, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlIq3S:
        r8_launch_codec<W4a8Iq3S, true>(schedule, epilogue, args, stream, pre);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
