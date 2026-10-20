// Grouped MoE W4A8 (moe_w4a8.cuh) over ggml_frag_v1 banks (core/ggml_frag.h), the converter's
// default for routed Q4_0, Q4_K, IQ3_XXS and IQ2_S experts: the published codecs' decode over the
// same bytes, gathered from their fragment-order positions (the R8 route's codecs, moe_r8.cuh).
#include "ops/moe/moe_r8.cuh"
#include "ops/moe/moe_w4a8.cuh"

namespace ninferno::ops::detail::moe {

// Q4_0 and Q4_K take the R8 route's fragment-order codecs (moe_r8.cuh W4a8Q40Frag, W4a8Q4KFrag):
// the same Raw over the same bytes.

bool launch_moe_w4a8_frag(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                          const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream) {
    switch (codec) {
    case generic::WeightCodec::GgmlQ40:
        moe_w4a8_launch_codec<W4a8Q40Frag>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ4K:
        moe_w4a8_launch_codec<W4a8Q4KFrag>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq3Xxs:
        moe_w4a8_launch_codec<W4a8Iq3XxsFrag>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq2S:
        moe_w4a8_launch_codec<W4a8Iq2SFrag>(stage, args, act, grid, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::moe
