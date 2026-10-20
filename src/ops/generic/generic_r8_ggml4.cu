// Generic R8 (W8A8 decode) for the 4- and 5-bit ggml blocks: Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL
// (32-value blocks) and IQ4_XS (32-value sub-blocks of a superblock, one scale each), MXFP4, and
// NVFP4 on a build without the native NVFP4 kernels.
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_r8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_r8_ggml4(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                     cudaStream_t stream, W4a8Activation pre) {
    switch (codec) {
    case WeightCodec::GgmlQ40:
        r8_launch_codec<W4a8Ggml32<Ggml32::Q4_0>, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlQ41:
        r8_launch_codec<W4a8Ggml32<Ggml32::Q4_1>, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlQ50:
        r8_launch_codec<W4a8Ggml32<Ggml32::Q5_0>, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlQ51:
        r8_launch_codec<W4a8Ggml32<Ggml32::Q5_1>, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlIq4Nl:
        r8_launch_codec<W4a8Ggml32<Ggml32::IQ4_NL>, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlIq4Xs:
        r8_launch_codec<W4a8Iq4Xs, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::Mxfp4:
        r8_launch_codec<W4a8Mxfp4, true>(schedule, epilogue, args, stream, pre);
        return true;
#ifdef NINFERNO_WITHOUT_NVFP4
    case WeightCodec::Nvfp4: // r8_supports admits NVFP4 only without kNativeNvfp4
        r8_launch_codec<W4a8Nvfp4, true>(schedule, epilogue, args, stream, pre);
        return true;
#endif
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
