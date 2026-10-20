#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

// Gemma 4 26B-A4B projection [2816,2112].
//
// K=2112 is 33 scale tiles: eight whole lane phases of 256 values and a ragged ninth. Every A16
// schedule is therefore the token-packed form, the only one that admits a ragged phase, and the
// W4A4 GEMM runs K tiles of 64 on the cp.async route at every T: the TMA route reads activation
// scales in tiles of 256 values.
namespace ninferno::ops::detail {
namespace {
using Geometry = Nvfp4N2816K2112;
// A16 speculative verify and chunked A16 prefill: BF16 tensor cores on weights dequantized in
// registers, one schedule per token bucket.
using A16T8               = Nvfp4A16RegMmaSchedule<1, 8, 1, 8, 8, 2, 1>;
using A16T16              = Nvfp4A16RegMmaSchedule<1, 4, 1, 16, 8, 2, 2>;
using A16T32              = Nvfp4A16RegMmaSchedule<1, 8, 2, 32, 8, 2, 1>;
using A16T64              = Nvfp4A16RegMmaSchedule<2, 4, 1, 64, 8, 2, 1>;
using T32R64K64           = Nvfp4A4MmaSchedule<32, 64, 64, 2, 4, 4, 2>;
using T32R128K64          = Nvfp4A4MmaSchedule<32, 128, 64, 2, 4, 4, 1>;
using T64R128K64          = Nvfp4A4MmaSchedule<64, 128, 64, 4, 2, 4, 1>;
using T128R128ResidentK64 = Nvfp4A4MmaSchedule<128, 128, 64, 4, 2, 5, 2>;

Nvfp4Launch select_a16(std::int32_t tokens) {
    // Decode too: the GEMV requires whole lane phases of K, which 2112 is not.
    if (tokens <= 8) return launch_nvfp4_a16_reg_mma<Geometry, A16T8>;
    if (tokens <= 16) return launch_nvfp4_a16_reg_mma<Geometry, A16T16>;
    if (tokens <= 32) return launch_nvfp4_a16_reg_mma<Geometry, A16T32>;
    if (tokens <= 64) return launch_nvfp4_a16_reg_mma<Geometry, A16T64>;
    throw std::logic_error("nvfp4 A16 chunk exceeds shape capacity");
}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R64K64>();
    if (tokens <= 224) return nvfp4_a4_mma_route<Geometry, T32R128K64>();
    if (tokens <= 448) return nvfp4_a4_mma_route<Geometry, T64R128K64>();
    return nvfp4_a4_mma_route<Geometry, T128R128ResidentK64>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }
} // namespace

const Nvfp4LinearShape kNvfp4N2816K2112{2816, 2112, launch_nvfp4_a16_chunks_prefill<64, select_a16>,
                                        launch_nvfp4_a4<select_a4>, uses_a4};
} // namespace ninferno::ops::detail
