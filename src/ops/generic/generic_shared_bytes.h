#pragma once

// The shared memory, in bytes, of the tiled and the staged register-MMA layouts (sizeof of
// TiledShared / StagedShared, generic_tiled.cuh and generic_staged.cuh): what route admission
// (tiled_fits, staged_fits) compares with the build's per-block limit (ops/common/arch_limits.h)
// with no CUDA headers (#620). generic_tiled.cu and generic_staged.cu static_assert every entry
// against the sizeof of their own build, so a table that disagrees stops that build; a tiled layout
// is the same on every architecture but Turing's, whose 64 KiB block lowers the stage counts.
//
// Regenerate: print tiled_shared_bytes<TiledSchedule<...>, Tile>() and sizeof(StagedShared<...>)
// for the lists in generic_schedules.h (NINFERNO_GENERIC_TILED_SCHEDULES / _STAGED_SCHEDULES) with
// -DNINFERNO_BUILD_ARCH=75 and any other architecture.

#include "ops/generic/generic_gemm.h"

#include <array>

namespace ninferno::ops::detail::generic {

// The tiled codecs in the order of tiled_codec_index, by tiled schedule.
inline constexpr std::array<std::array<int, 3>, 11> kTiledSharedBytes{{
    {78368, 53792, 63776}, // U4Zp
    {78880, 54304, 64032}, // U4ZpG32
    {78368, 53792, 63776}, // U4ZpG64
    {81920, 56320, 66560}, // GgmlQ4K
    {94208, 68608, 72704}, // GgmlQ5K
    {78368, 53792, 63776}, // Q4
    {78368, 53792, 63776}, // Q5
    {78368, 53792, 63776}, // Q6
    {78368, 53792, 63776}, // Q3G128
    {78368, 53792, 63776}, // Q3G64
    {91168, 66592, 70176}, // Q8
}};

// The same on Turing (NINFERNO_BUILD_ARCH 75).
inline constexpr std::array<std::array<int, 3>, 11> kTiledSharedBytesTuring{{
    {57888, 53792, 63776}, // U4Zp
    {58400, 54304, 64032}, // U4ZpG32
    {57888, 53792, 63776}, // U4ZpG64
    {61440, 56320, 48128}, // GgmlQ4K
    {69632, 52224, 52224}, // GgmlQ5K
    {57888, 53792, 63776}, // Q4
    {57888, 53792, 63776}, // Q5
    {57888, 53792, 63776}, // Q6
    {57888, 53792, 63776}, // Q3G128
    {57888, 53792, 63776}, // Q3G64
    {66592, 50208, 49696}, // Q8
}};

// The staged schedules, by index.
inline constexpr std::array<int, 13> kStagedSharedBytes{
    {98304, 98304, 98304, 98304, 98304, 65536, 49152, 49152, 49152, 65536, 163840, 147456, 147456}};

// The row of kTiledSharedBytes of a codec; -1 for a codec the tiled schedules do not run.
constexpr int tiled_codec_index(WeightCodec codec) {
    switch (codec) {
    case WeightCodec::U4Zp:
        return 0;
    case WeightCodec::U4ZpG32:
        return 1;
    case WeightCodec::U4ZpG64:
        return 2;
    case WeightCodec::GgmlQ4K:
        return 3;
    case WeightCodec::GgmlQ5K:
        return 4;
    case WeightCodec::Q4:
        return 5;
    case WeightCodec::Q5:
        return 6;
    case WeightCodec::Q6:
        return 7;
    case WeightCodec::Q3G128:
        return 8;
    case WeightCodec::Q3G64:
        return 9;
    case WeightCodec::Q8:
        return 10;
    default:
        return -1;
    }
}

// The shared bytes of the tiled layout of codec index `codec` and schedule `schedule` in this
// build.
constexpr int tiled_shared_table(int codec, int schedule) {
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH == 75
    return kTiledSharedBytesTuring[static_cast<std::size_t>(codec)]
                                  [static_cast<std::size_t>(schedule)];
#else
    return kTiledSharedBytes[static_cast<std::size_t>(codec)][static_cast<std::size_t>(schedule)];
#endif
}

} // namespace ninferno::ops::detail::generic
