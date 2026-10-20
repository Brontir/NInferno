// Generic MoE grouped R8 (W8A8 decode): IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M experts.
#include "ops/moe/moe_r8.cuh"

namespace ninferno::ops::detail::moe {

bool launch_r8_gemm_iq2(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                        int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    if (frag) {
        if (codec != generic::WeightCodec::GgmlIq2S) { return false; }
        launch_moe_r8<W4a8Iq2SFrag>(stage, schedule, tokens, args, grid, stream);
        return true;
    }
    switch (codec) {
    case generic::WeightCodec::GgmlIq2Xxs:
        launch_moe_r8<generic::W4a8Iq2Xxs>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq2Xs:
        launch_moe_r8<generic::W4a8Iq2Xs>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq2S:
        launch_moe_r8<generic::W4a8Iq2S>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq1S:
        launch_moe_r8<generic::W4a8Iq1S>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq1M:
        launch_moe_r8<generic::W4a8Iq1M>(stage, schedule, tokens, args, grid, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::moe
