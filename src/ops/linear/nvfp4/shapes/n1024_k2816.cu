#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

// Gemma 4 26B-A4B projection [1024,2816].
namespace ninferno::ops::detail {
namespace {
using Geometry = Nvfp4N1024K2816;
using Gemv = Nvfp4A16GemvSchedule<4, 1, 8, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 4>;
// A16 speculative verify and chunked A16 prefill: BF16 tensor cores on weights dequantized in
// registers, one schedule per token bucket.
using A16T8             = Nvfp4A16RegMmaSchedule<1, 8, 1, 8, 8, 2, 1>;
using A16T16            = Nvfp4A16RegMmaSchedule<1, 8, 1, 16, 8, 2, 1>;
using A16T32            = Nvfp4A16RegMmaSchedule<1, 4, 1, 32, 16, 2, 2>;
using A16T64            = Nvfp4A16RegMmaSchedule<1, 4, 1, 64, 8, 2, 2>;
using T32R128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T128R128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;

Nvfp4Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 8) return launch_nvfp4_a16_reg_mma<Geometry, A16T8>;
    if (tokens <= 16) return launch_nvfp4_a16_reg_mma<Geometry, A16T16>;
    if (tokens <= 32) return launch_nvfp4_a16_reg_mma<Geometry, A16T32>;
    if (tokens <= 64) return launch_nvfp4_a16_reg_mma<Geometry, A16T64>;
    throw std::logic_error("nvfp4 A16 chunk exceeds shape capacity");
}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens <= 1024) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 2560) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_tma_route<Nvfp4GeometryId::N1024K2816>();
}

// A16 up to the widest decode or verify call (T=16), W4A4 above it. One A16 chunk of up to 32
// tokens is faster (under 11 us, below the W4A4 quantizer plus GEMM), but A16 and W4A4 give a
// token different values: with the frontier at 33, a 17..32-token prefill chunk computed this
// layer's keys in another precision than a wider chunk, and a prompt decoded differently depending
// on where it was cut. The other Gemma 4 projections all switch to W4A4 at or below T=17.
bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }
} // namespace

const Nvfp4LinearShape kNvfp4N1024K2816{1024, 2816, launch_nvfp4_a16_chunks_prefill<64, select_a16>,
                                        launch_nvfp4_a4<select_a4>, uses_a4};
} // namespace ninferno::ops::detail
