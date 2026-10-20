// Generic W4A8 for the formats with a scale per 32 (Q6_K, Q3_K, Q2_K: 16) values inside larger
// blocks: IQ4_XS, Q4_K, Q5_K, Q6_K, Q3_K, Q2_K (256-value superblocks), u4_g128_zp (128-value
// groups), q4/q5/q6_g64 (64-value groups) and q3_g128 / q3_g64 (128- and 64-value groups).
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_w4a8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_w4a8_superblock(int schedule, WeightCodec codec, Epilogue epilogue,
                            const GemmArgs& args, W4a8Activation act, cudaStream_t stream) {
    switch (codec) {
    case WeightCodec::GgmlIq4Xs:
        w4a8_launch_codec<W4a8Iq4Xs>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlQ4K:
        w4a8_launch_codec<W4a8Q45K<4>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlQ5K:
        w4a8_launch_codec<W4a8Q45K<5>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlQ6K:
        w4a8_launch_codec<W4a8Q6K>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlQ3K:
        w4a8_launch_codec<W4a8Q3K>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::GgmlQ2K:
        w4a8_launch_codec<W4a8Q2K>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::U4Zp:
        w4a8_launch_codec<W4a8U4Zp>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::U4ZpG32:
        w4a8_launch_codec<W4a8U4ZpG32>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::U4ZpG64:
        w4a8_launch_codec<W4a8U4ZpG64>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::Q4:
        w4a8_launch_codec<W4a8G64<4>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::Q5:
        w4a8_launch_codec<W4a8G64<5>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::Q6:
        w4a8_launch_codec<W4a8G64<6>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::Q3G128:
        w4a8_launch_codec<W4a8Q3<128>>(schedule, epilogue, args, act, stream);
        return true;
    case WeightCodec::Q3G64:
        w4a8_launch_codec<W4a8Q3<64>>(schedule, epilogue, args, act, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
