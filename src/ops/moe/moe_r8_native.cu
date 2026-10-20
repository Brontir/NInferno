// Generic MoE grouped R8 (W8A8 decode): Q8 G32, the native groupwise Q4/Q5/Q6 G64 and the u4
// zero-point formats (G32, G64, G128), on the codecs generic R8 and W4A8 read them with.
#include "ops/moe/moe_r8.cuh"

namespace ninferno::ops::detail::moe {

bool launch_r8_gemm_native(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                           int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    if (frag) return false;
    switch (codec) {
    case generic::WeightCodec::Q8:
        launch_moe_r8<generic::W4a8Q8>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::Q4:
        launch_moe_r8<generic::W4a8G64<4>>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::Q5:
        launch_moe_r8<generic::W4a8G64<5>>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::Q6:
        launch_moe_r8<generic::W4a8G64<6>>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::U4Zp:
        launch_moe_r8<generic::W4a8U4Zp>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::U4ZpG32:
        launch_moe_r8<generic::W4a8U4ZpG32>(stage, schedule, tokens, args, grid, stream);
        return true;
    case generic::WeightCodec::U4ZpG64:
        launch_moe_r8<generic::W4a8U4ZpG64>(stage, schedule, tokens, args, grid, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::moe
