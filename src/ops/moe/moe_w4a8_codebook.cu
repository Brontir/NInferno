// Grouped MoE W4A8 (moe_w4a8.cuh): Q8 and the codebook i-quants.
#include "ops/moe/moe_w4a8.cuh"

namespace ninferno::ops::detail::moe {

bool launch_moe_w4a8_codebook(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                              const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream) {
    switch (codec) {
    case generic::WeightCodec::Q8:
        moe_w4a8_launch_codec<generic::W4a8Q8>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq3Xxs:
        moe_w4a8_launch_codec<generic::W4a8Iq3Xxs>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq3S:
        moe_w4a8_launch_codec<generic::W4a8Iq3S>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq2Xxs:
        moe_w4a8_launch_codec<generic::W4a8Iq2Xxs>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq2Xs:
        moe_w4a8_launch_codec<generic::W4a8Iq2Xs>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq2S:
        moe_w4a8_launch_codec<generic::W4a8Iq2S>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq1S:
        moe_w4a8_launch_codec<generic::W4a8Iq1S>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq1M:
        moe_w4a8_launch_codec<generic::W4a8Iq1M>(stage, args, act, grid, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::moe
