// Generic R8 (W8A8 decode) for IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M.
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_r8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_r8_iq2(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream, W4a8Activation pre) {
    switch (codec) {
    case WeightCodec::GgmlIq2Xxs:
        r8_launch_codec<W4a8Iq2Xxs, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlIq2Xs:
        r8_launch_codec<W4a8Iq2Xs, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlIq2S:
        r8_launch_codec<W4a8Iq2S, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlIq1S:
        r8_launch_codec<W4a8Iq1S, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlIq1M:
        r8_launch_codec<W4a8Iq1M, true>(schedule, epilogue, args, stream, pre);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
