// Generic W4A8 for the ggml 32-value block formats: Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL,
// and MXFP4 (32-value blocks with an E8M0 exponent); NVFP4 (doubled E2M1 codes) where W4A4 does
// not take the weight.
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_w4a8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_w4a8_ggml32(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                        W4a8Activation act, cudaStream_t stream) {
    switch (codec) {
    case WeightCodec::GgmlQ40:
        w4a8_launch_codec<W4a8Ggml32<Ggml32::Q4_0>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlQ41:
        w4a8_launch_codec<W4a8Ggml32<Ggml32::Q4_1>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlQ50:
        w4a8_launch_codec<W4a8Ggml32<Ggml32::Q5_0>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlQ51:
        w4a8_launch_codec<W4a8Ggml32<Ggml32::Q5_1>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlIq4Nl:
        w4a8_launch_codec<W4a8Ggml32<Ggml32::IQ4_NL>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::Mxfp4:
        w4a8_launch_codec<W4a8Mxfp4>(schedule, epilogue, args, act, stream);
        return true;
    // Every build: on sm_120a the prefill of a weight W4A4 does not take (allows_w4a8).
    case WeightCodec::Nvfp4:
        w4a8_launch_codec<W4a8Nvfp4>(schedule, epilogue, args, act, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
