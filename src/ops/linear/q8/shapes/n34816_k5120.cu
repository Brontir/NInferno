#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"
#include "core/device_class.h"
#include "core/measured_boards.h"

namespace ninferno::ops::detail {
namespace {
using Geometry = Q8N34816K5120;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C4 =
    Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C8 =
    Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C16 =
    Q8A16SlicedKMmaSchedule<16, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 =
    Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C32 = Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using C40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
// A100 and the GA102 boards (3090, A10) at T <= 8: eight K-warps a tile (ninfer-all's W8):
// MTP-layer gate_up 254 us on the four-K-warp C4 against 154 us there; an Ampere board takes it by
// the nearest of them (measured::measured_gate), the other architectures' boards were not measured.
using C8W8 = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                     Stage::RuntimeActive>;
using C56 =
    Q8A16SlicedKMmaSchedule<56, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
constexpr measured::Gate kAmpereW8[] = {
    {measured::kA100Sxm40, true}, {measured::kRtx3090, true}, {measured::kA10, true}};

} // namespace

Q8Launch select_q8_n34816_k5120(std::int32_t tokens) {
    if (tokens <= 8 && measured::measured_gate(kAmpereW8)) {
        return launch_q8_a16_sliced<Geometry, 8, C8W8>;
    }
    if (tokens <= 4) return launch_q8_a16_sliced<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_mma_r64x16_t48_k128_a1;
    if (tokens <= 56) return launch_q8_a16_sliced<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_a16_mma_r128_t64;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninferno::ops::detail
