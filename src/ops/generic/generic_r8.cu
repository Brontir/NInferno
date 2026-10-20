// Generic R8 (W8A8 decode): the codec dispatch, q8_g32, q4/q5/q6_g64, u4_g{32,64,128}_zp and
// q3_g128 / q3_g64.
#include "ops/generic/generic_r8.cuh"

#include "ops/generic/generic_gemm_internal.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {

void launch_r8(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
               cudaStream_t stream, W4a8Activation pre) {
    if (codec == WeightCodec::Q8) {
        r8_launch_codec<W4a8Q8, true>(schedule, epilogue, args, stream, pre);
        return;
    }
    if (codec == WeightCodec::Q4) {
        r8_launch_codec<W4a8G64<4>, true>(schedule, epilogue, args, stream, pre);
        return;
    }
    if (codec == WeightCodec::Q5) {
        r8_launch_codec<W4a8G64<5>, true>(schedule, epilogue, args, stream, pre);
        return;
    }
    if (codec == WeightCodec::Q6) {
        r8_launch_codec<W4a8G64<6>, true>(schedule, epilogue, args, stream, pre);
        return;
    }
    if (codec == WeightCodec::U4Zp) {
        r8_launch_codec<W4a8U4Zp, true>(schedule, epilogue, args, stream, pre);
        return;
    }
    if (codec == WeightCodec::U4ZpG32) {
        r8_launch_codec<W4a8U4ZpG32, true>(schedule, epilogue, args, stream, pre);
        return;
    }
    if (codec == WeightCodec::U4ZpG64) {
        r8_launch_codec<W4a8U4ZpG64, true>(schedule, epilogue, args, stream, pre);
        return;
    }
    if (codec == WeightCodec::Q3G128) {
        r8_launch_codec<W4a8Q3<128>, true>(schedule, epilogue, args, stream, pre);
        return;
    }
    if (codec == WeightCodec::Q3G64) {
        r8_launch_codec<W4a8Q3<64>, true>(schedule, epilogue, args, stream, pre);
        return;
    }
    if (launch_r8_iq3(schedule, codec, epilogue, args, stream, pre)) return;
    if (launch_r8_iq2(schedule, codec, epilogue, args, stream, pre)) return;
    if (launch_r8_ggml4(schedule, codec, epilogue, args, stream, pre)) return;
    if (launch_r8_kquant(schedule, codec, epilogue, args, stream, pre)) return;
    throw std::invalid_argument("generic R8: the weight format has no R8 codec");
}

} // namespace ninferno::ops::detail::generic
