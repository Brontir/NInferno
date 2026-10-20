// Generic MoE grouped R8 (W8A8 decode): IQ3_XXS and IQ3_S experts.
#include "ops/moe/moe_r8.cuh"

namespace ninferno::ops::detail::moe {

bool launch_r8_gemm_iq3(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                        int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    if (frag) {
        if (codec != generic::WeightCodec::GgmlIq3Xxs) { return false; }
        launch_moe_r8<W4a8Iq3XxsFrag>(stage, schedule, tokens, args, grid, stream);
        return true;
    }
    switch (codec) {
    case generic::WeightCodec::GgmlIq3Xxs:
        launch_moe_r8<generic::W4a8Iq3Xxs>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlIq3S:
        launch_moe_r8<generic::W4a8Iq3S>(stage, schedule, tokens, args, grid, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::moe
