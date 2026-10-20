// Grouped MoE W4A8 (moe_w4a8.cuh): the native groupwise formats Q4/Q5/Q6 G64 and the u4
// zero-point formats (G32, G64, G128). w4a8_banks_admit admits every format generic W4A8 reads;
// before these, such an expert bank reached the grouped W4A8 with no tile for it.
#include "ops/moe/moe_w4a8.cuh"

namespace ninferno::ops::detail::moe {

bool launch_moe_w4a8_native(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                            const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream) {
    switch (codec) {
    case generic::WeightCodec::Q4:
        moe_w4a8_launch_codec<generic::W4a8G64<4>>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::Q5:
        moe_w4a8_launch_codec<generic::W4a8G64<5>>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::Q6:
        moe_w4a8_launch_codec<generic::W4a8G64<6>>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::U4Zp:
        moe_w4a8_launch_codec<generic::W4a8U4Zp>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::U4ZpG32:
        moe_w4a8_launch_codec<generic::W4a8U4ZpG32>(stage, args, act, grid, stream);
        return true;
    case generic::WeightCodec::U4ZpG64:
        moe_w4a8_launch_codec<generic::W4a8U4ZpG64>(stage, args, act, grid, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::moe
