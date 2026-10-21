#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

// Gemma 4 [5376,16384]. Bands measured on the RTX 5090 with interleaved cold-weight timing over
// T in {1..4096}; each band takes the winner at its upper edge.
namespace ninferno::ops::detail {
namespace {
using Geometry = Q8N5376K16384;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
} // namespace

Q8Launch select_q8_n5376_k16384(std::int32_t tokens) {
    if (tokens <= 4)
        return launch_q8_a16_sliced<
            Geometry, 4,
            Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 8)
        return launch_q8_a16_sliced<
            Geometry, 8,
            Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 16)
        return launch_q8_a16_sliced<
            Geometry, 16,
            Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>>;
    if (tokens <= 24)
        return launch_q8_a16_sliced<
            Geometry, 24,
            Q8A16SlicedKMmaSchedule<24, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 32)
        return launch_q8_a16_sliced<
            Geometry, 32,
            Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 48)
        return launch_q8_a16_sliced<
            Geometry, 48,
            Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 64)
        return launch_q8_a16_sliced<
            Geometry, 64,
            Q8A16SlicedKMmaSchedule<64, 4, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>>;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    if (tokens <= 192) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 256) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 384) return launch_q8_a16_mma_r64_t96;
    // The 128-token tile on every board. The 256-token tile (MmaR128x64T256) won at the multiples
    // of 1024 on the RTX 5090 (170 SMs, 2.6-5.7 %) and the RTX 4070 Ti (60 SMs, 2-6 % at 1024 and
    // 2048) but lost on the RTX 5070 Ti (70 SMs, 4-20 %) and the RTX 4090 (128 SMs, 9-30 %), even
    // where both grids fill their waves alike: no device fact tried (whole waves, wave cost, L2 or
    // DRAM bandwidth per SM) orders the four boards, so the choice is the one whose worst case is
    // small (principle 7: about 6 % at most, against 30 % for the wide tile).
    return launch_q8_a16_mma_r64x32_t128;
}

} // namespace ninferno::ops::detail
