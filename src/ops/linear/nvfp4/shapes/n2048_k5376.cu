#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

#include "core/device_class.h"
#include "ops/common/sm_wave_rules.h"

// Gemma 4 31B projection [2048,5376].
namespace ninferno::ops::detail {
namespace {
using Geometry = Nvfp4N2048K5376;
// Decode: one-row warps with eight resident CTAs per SM, 4% under the two-row form in three
// independent interleaved series.
using Gemv = Nvfp4A16GemvSchedule<4, 1, 8, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 8>;
// A16 speculative verify and chunked A16 prefill: BF16 tensor cores on weights dequantized in
// registers, one schedule per token bucket.
using A16T8             = Nvfp4A16RegMmaSchedule<1, 8, 1, 8, 8, 2, 1>;
using A16T16            = Nvfp4A16RegMmaSchedule<1, 8, 1, 16, 8, 2, 1>;
using A16T32            = Nvfp4A16RegMmaSchedule<1, 8, 2, 32, 8, 2, 1>;
using A16T64            = Nvfp4A16RegMmaSchedule<1, 8, 2, 64, 8, 2, 1>;
using T16R64            = Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>;
using T32R128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64R128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using T128R128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;

Nvfp4Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 8) return launch_nvfp4_a16_reg_mma<Geometry, A16T8>;
    if (tokens <= 16) return launch_nvfp4_a16_reg_mma<Geometry, A16T16>;
    if (tokens <= 32) return launch_nvfp4_a16_reg_mma<Geometry, A16T32>;
    if (tokens <= 64) return launch_nvfp4_a16_reg_mma<Geometry, A16T64>;
    throw std::logic_error("nvfp4 A16 chunk exceeds shape capacity");
}

int sm_count() { return device_class().sms; }

// The TMA grid is only 16 row tiles wide, so it stays under one wave and loses to the 128-token
// MMA tile until that tile's grid (16 row tiles by T/128) passes one CTA per SM
// (sm_rules::grid_passes_wave; on the RTX 5090's 170 SMs: T > 1280).
Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T16R64>();
    if (tokens <= 320) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 640) return nvfp4_a4_mma_route<Geometry, T64R128>();
    if (!sm_rules::grid_passes_wave(tokens, sm_count(), 2048 / 128, 128, 1))
        return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_tma_route<Nvfp4GeometryId::N2048K5376>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 12; }
} // namespace

const Nvfp4LinearShape kNvfp4N2048K5376{2048, 5376, launch_nvfp4_a16_chunks_prefill<64, select_a16>,
                                        launch_nvfp4_a4<select_a4>, uses_a4};
} // namespace ninferno::ops::detail
