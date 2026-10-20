// Grouped MoE W4A8 (moe_w4a8.cuh): the ggml 32-value block formats and MXFP4.
#include "ops/moe/moe_w4a8.cuh"

namespace ninferno::ops::detail::moe {

bool launch_moe_w4a8_ggml32(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                            const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream) {
    switch (codec) {
    case generic::WeightCodec::GgmlQ40:
        moe_w4a8_launch_codec<generic::W4a8Ggml32<generic::Ggml32::Q4_0>>(stage, args, act, grid,
                                                                          stream);
        return true;
    case generic::WeightCodec::GgmlQ41:
        moe_w4a8_launch_codec<generic::W4a8Ggml32<generic::Ggml32::Q4_1>>(stage, args, act, grid,
                                                                          stream);
        return true;
    case generic::WeightCodec::GgmlQ50:
        moe_w4a8_launch_codec<generic::W4a8Ggml32<generic::Ggml32::Q5_0>>(stage, args, act, grid,
                                                                          stream);
        return true;
    case generic::WeightCodec::GgmlQ51:
        moe_w4a8_launch_codec<generic::W4a8Ggml32<generic::Ggml32::Q5_1>>(stage, args, act, grid,
                                                                          stream);
        return true;
    case generic::WeightCodec::GgmlIq4Nl:
        moe_w4a8_launch_codec<generic::W4a8Ggml32<generic::Ggml32::IQ4_NL>>(stage, args, act, grid,
                                                                            stream);
        return true;
    case generic::WeightCodec::Mxfp4:
        moe_w4a8_launch_codec<generic::W4a8Mxfp4>(stage, args, act, grid, stream);
        return true;
#ifdef NINFERNO_WITHOUT_NVFP4
    case generic::WeightCodec::Nvfp4: // w4a8_supports admits NVFP4 only without kNativeNvfp4
        moe_w4a8_launch_codec<generic::W4a8Nvfp4>(stage, args, act, grid, stream);
        return true;
#endif
    default:
        return false;
    }
}

// Without native NVFP4 no MX codec has kernels: every parameter goes unused there.
bool launch_moe_w4a8_mx([[maybe_unused]] generic::WeightCodec codec, [[maybe_unused]] Stage stage,
                        [[maybe_unused]] const GemmArgs& args,
                        [[maybe_unused]] const MoeW4a8Activation& act, [[maybe_unused]] dim3 grid,
                        [[maybe_unused]] cudaStream_t stream) {
#ifndef NINFERNO_WITHOUT_NVFP4
    if (codec == generic::WeightCodec::Mxfp4) {
        moe_w4a8_launch_codec<generic::W4a8Mxfp4Mx>(stage, args, act, grid, stream);
        return true;
    }
#endif
    return false;
}

} // namespace ninferno::ops::detail::moe
