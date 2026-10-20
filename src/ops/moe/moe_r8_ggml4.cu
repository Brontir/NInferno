// Generic MoE grouped R8 (W8A8 decode): the 4- and 5-bit ggml blocks Q4_0, Q4_1, Q5_0, Q5_1,
// IQ4_NL and IQ4_XS, MXFP4 (gpt-oss), and NVFP4 on a build without the native NVFP4 kernels.
#include "ops/moe/moe_r8.cuh"

namespace ninferno::ops::detail::moe {

bool launch_r8_gemm_ggml4(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                          int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    using generic::Ggml32;
    using generic::W4a8Ggml32;
    if (frag) {
        if (codec != generic::WeightCodec::GgmlQ40) { return false; }
        launch_moe_r8<W4a8Q40Frag>(stage, schedule, tokens, args, grid, stream);
        return true;
    }
    switch (codec) {
    case generic::WeightCodec::GgmlQ40:
        launch_moe_r8<W4a8Ggml32<Ggml32::Q4_0>>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ41:
        launch_moe_r8<W4a8Ggml32<Ggml32::Q4_1>>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ50:
        launch_moe_r8<W4a8Ggml32<Ggml32::Q5_0>>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ51:
        launch_moe_r8<W4a8Ggml32<Ggml32::Q5_1>>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq4Nl:
        launch_moe_r8<W4a8Ggml32<Ggml32::IQ4_NL>>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq4Xs:
        launch_moe_r8<generic::W4a8Iq4Xs>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::Mxfp4:
        launch_moe_r8<generic::W4a8Mxfp4>(stage, schedule, tokens, args, grid, stream);
        return true;
#ifdef NINFERNO_WITHOUT_NVFP4
    case generic::WeightCodec::Nvfp4: // r8_supports admits NVFP4 only without kNativeNvfp4
        launch_moe_r8<generic::W4a8Nvfp4>(stage, schedule, tokens, args, grid, stream);
        return true;
#endif
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::moe
