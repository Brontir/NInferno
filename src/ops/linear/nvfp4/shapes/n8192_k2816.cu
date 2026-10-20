#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

// Gemma 4 26B-A4B projection [8192,2816].
namespace ninferno::ops::detail {
namespace {
using Geometry = Nvfp4N8192K2816;
// Decode (T=1) runs on the eight-token tile too: 5.9% faster than the tuned GEMV.
// A16 speculative verify and chunked A16 prefill: BF16 tensor cores on weights dequantized in
// registers, one schedule per token bucket.
using A16T8             = Nvfp4A16RegMmaSchedule<1, 4, 1, 8, 16, 2, 2>;
using A16T16            = Nvfp4A16RegMmaSchedule<1, 4, 1, 16, 16, 2, 2>;
using A16T32            = Nvfp4A16RegMmaSchedule<1, 4, 1, 32, 8, 2, 2>;
using A16T64            = Nvfp4A16RegMmaSchedule<2, 4, 2, 64, 8, 2, 1>;
using T16R64            = Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>;
using T32R128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T128R128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using T128R128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

Nvfp4Launch select_a16(std::int32_t tokens) {
    if (tokens <= 8) return launch_nvfp4_a16_reg_mma<Geometry, A16T8>;
    if (tokens <= 16) return launch_nvfp4_a16_reg_mma<Geometry, A16T16>;
    if (tokens <= 32) return launch_nvfp4_a16_reg_mma<Geometry, A16T32>;
    if (tokens <= 64) return launch_nvfp4_a16_reg_mma<Geometry, A16T64>;
    throw std::logic_error("nvfp4 A16 chunk exceeds shape capacity");
}

// The TMA grid is 64 row tiles by ceil(T/256) token tiles. At 513..640 it spills 192 CTAs into a
// second wave, where the 128-token MMA tile's 320 CTAs fill two waves instead.
Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens <= 32) return nvfp4_a4_mma_route<Geometry, T16R64>();
    if (tokens <= 160) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 256) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    if (tokens <= 512) return nvfp4_a4_tma_route<Nvfp4GeometryId::N8192K2816>();
    if (tokens <= 640) return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
    return nvfp4_a4_tma_route<Nvfp4GeometryId::N8192K2816>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 8; }
} // namespace

const Nvfp4LinearShape kNvfp4N8192K2816{8192, 2816, launch_nvfp4_a16_chunks_prefill<64, select_a16>,
                                        launch_nvfp4_a4<select_a4>, uses_a4};
} // namespace ninferno::ops::detail
