// Generic W4A8 for the formats whose int8 code is not a bit field: q8_g32 (GGUF Q8_0) and the
// codebook i-quants IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S (signed grid values).
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_w4a8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_w4a8_codebook(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                          W4a8Activation act, cudaStream_t stream) {
    switch (codec) {
    case WeightCodec::Q8:
        w4a8_launch_codec<W4a8Q8>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlIq3Xxs:
        w4a8_launch_codec<W4a8Iq3Xxs>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlIq3S:
        w4a8_launch_codec<W4a8Iq3S>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlIq2Xxs:
        w4a8_launch_codec<W4a8Iq2Xxs>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlIq2Xs:
        w4a8_launch_codec<W4a8Iq2Xs>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlIq2S:
        w4a8_launch_codec<W4a8Iq2S>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlIq1S:
        w4a8_launch_codec<W4a8Iq1S>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlIq1M:
        w4a8_launch_codec<W4a8Iq1M>(schedule, epilogue, args, act, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
