// Grouped MoE W4A8 (moe_w4a8.cuh): the 256-value superblock formats (IQ4_XS, Q4_K, Q5_K, Q6_K,
// Q3_K, Q2_K).
#include "ops/moe/moe_w4a8.cuh"

namespace ninferno::ops::detail::moe {

bool launch_moe_w4a8_superblock(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                                const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream) {
    switch (codec) {
    case generic::WeightCodec::GgmlIq4Xs:
        moe_w4a8_launch_codec<generic::W4a8Iq4Xs>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ4K:
        moe_w4a8_launch_codec<generic::W4a8Q45K<4>>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ5K:
        moe_w4a8_launch_codec<generic::W4a8Q45K<5>>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ6K:
        moe_w4a8_launch_codec<generic::W4a8Q6K>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ3K:
        moe_w4a8_launch_codec<generic::W4a8Q3K>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ2K:
        moe_w4a8_launch_codec<generic::W4a8Q2K>(stage, args, act, grid, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::moe
