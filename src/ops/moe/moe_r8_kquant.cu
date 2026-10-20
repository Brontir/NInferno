// Generic MoE grouped R8 (W8A8 decode): Q3_K, Q4_K, Q5_K and Q6_K experts.
#include "ops/moe/moe_r8.cuh"

namespace ninferno::ops::detail::moe {

bool launch_r8_gemm_kquant(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                           int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    if (frag) {
        if (codec != generic::WeightCodec::GgmlQ4K) { return false; }
        launch_moe_r8<W4a8Q4KFrag>(stage, schedule, tokens, args, grid, stream);
        return true;
    }
    switch (codec) {
    case generic::WeightCodec::GgmlQ4K:
        launch_moe_r8<generic::W4a8Q45K<4>>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ5K:
        launch_moe_r8<generic::W4a8Q45K<5>>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ6K:
        launch_moe_r8<generic::W4a8Q6K>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::GgmlQ3K:
        launch_moe_r8<generic::W4a8Q3K>(stage, schedule, tokens, args, grid, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::moe
