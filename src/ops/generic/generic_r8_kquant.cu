// Generic R8 (W8A8 decode) for the K-quants Q4_K and Q5_K (a scale and a min per 32 values, two
// blocks per load), Q6_K (a scale per 16 values, blocks u and u + 2 per load), Q3_K (a scale per
// 16 values) and Q2_K (a scale and a min per 16 values), the last two the four blocks of a half
// superblock per load.
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_r8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_r8_kquant(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                      cudaStream_t stream, W4a8Activation pre) {
    switch (codec) {
    case WeightCodec::GgmlQ4K:
        r8_launch_codec<W4a8Q45K<4>, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlQ5K:
        r8_launch_codec<W4a8Q45K<5>, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlQ6K:
        r8_launch_codec<W4a8Q6K, true>(schedule, epilogue, args, stream, pre);
        return true;
    case WeightCodec::GgmlQ3K:
        r8_launch_codec<W4a8Q3K>(schedule, epilogue, args, stream);
        return true;
    case WeightCodec::GgmlQ2K:
        r8_launch_codec<W4a8Q2K>(schedule, epilogue, args, stream);
        return true;
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
