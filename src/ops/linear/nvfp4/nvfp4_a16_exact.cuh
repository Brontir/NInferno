#pragma once

// NVFP4 A16 under LinearPolicy::A16Only (--activations exact, ninferno/types.h) on the Qwen3.5-27B
// shapes: BF16 tensor cores on weights dequantized in registers (nvfp4_a16_mma.cuh).
//
// Every width runs on schedules of one (WarpsK, Bytes) per geometry, so a token's bytes do not
// depend on the size of its call, its column in it, or where a prompt was cut into chunks: the
// linears add nothing to what batch neighbours, chunking, prefix reuse or a speculative verify
// could change. Calls up to Tokens of the widest schedule run in one token tile; wider calls tile
// the tokens over the grid's y dimension on the widest schedule.

#include "core/device.h"
#include "core/weight.h"
#include "ops/linear/nvfp4/nvfp4_a16_reg_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_tiled.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace ninferno::ops::detail {

// How a schedule lays its work on the GPU without touching the sum order: warps over rows, m16
// tiles per warp, K steps in flight, and the CTAs per SM its registers are bounded for.
template <int WarpsM, int MTiles, int Stages, int MinBlocksPerSm = 1>
struct Nvfp4A16Tiling {
    static constexpr int kWarpsM         = WarpsM;
    static constexpr int kMTiles         = MTiles;
    static constexpr int kStages         = Stages;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
};

// Calls of From <= T < To tokens (To = 0: no upper bound) take nvfp4_a16_tiled_kernel on Schedule.
template <int From, int To, class Schedule>
struct Nvfp4A16TiledBand {
    static_assert(From > 16 && (To == 0 || To > From));
    using TiledSchedule = Schedule;
    static constexpr bool holds(int tokens) { return tokens >= From && (To == 0 || tokens < To); }
};

// A family of A16 MMA schedules: one per token bucket (8, 16, 32 and, when WidestTokens is 64, 64),
// all with the same WarpsK and Bytes, so a family computes every token's bytes identically at every
// width (nvfp4_a16_reg_mma_kernel). The narrow buckets (T <= 16) and the wide ones each have their own
// tiling; calls wider than WidestTokens tile their tokens over the grid on the widest schedule.
// TiledBands (Nvfp4A16TiledBand, WarpsK = 8 and Bytes = 16 only: the order the tiled kernel
// reproduces) hand the calls they hold to nvfp4_a16_tiled_kernel instead; the first band that
// holds a width takes it.
template <int WarpsK, int Bytes, class Narrow, class Wide, int WidestTokens, class... TiledBands>
struct Nvfp4A16ExactFamily {
    static_assert(WidestTokens == 32 || WidestTokens == 64);
    static_assert(sizeof...(TiledBands) == 0 || (WarpsK == 8 && Bytes == 16),
                  "the tiled kernel sums in the order of WarpsK = 8, Bytes = 16");
    static constexpr int kWidestTokens = WidestTokens;
    using Bands                        = std::tuple<TiledBands...>;
    template <int Tokens>
    using TilingAt = std::conditional_t<(Tokens <= 16), Narrow, Wide>;
    template <int Tokens>
    using At = Nvfp4A16RegMmaSchedule<TilingAt<Tokens>::kWarpsM, WarpsK, TilingAt<Tokens>::kMTiles,
                                   Tokens, Bytes, TilingAt<Tokens>::kStages,
                                   TilingAt<Tokens>::kMinBlocksPerSm>;
};

// The family of each Qwen3.5-27B geometry, measured on the RTX 5090 (nvfp4_a16_mma_sweep_bench,
// cold L2): eight warps split K, sixteen code bytes per lane and K step, one m16 tile per warp up
// to 16 tokens and two above, two steps in flight, 32-token tiles. Through T = 32 (decode, MTP
// verify of up to eight requests) it is within 5% of the best MMA-family candidate on every shape,
// at or below the Linear A16 GEMV at T = 1, and 1.4-2.3x (T = 16) and 2.4-3.2x (T = 32) faster
// than the Linear A16 SIMT route.
template <class... TiledBands>
using Nvfp4A16Exact27B = Nvfp4A16ExactFamily<8, 16, Nvfp4A16Tiling<1, 1, 2>,
                                             Nvfp4A16Tiling<1, 2, 2>, 32, TiledBands...>;

// Tiled bands, same order (sweeps on the RTX 5090, cold L2, three passes within 2%,
// against the family alone). MTP verify (17..32 tokens): the tiled kernel with one m16 tile per
// warp, 32 tokens and enough warps that the rows fill one wave (13 warps: 168 CTAs for N=34816,
// 7 warps: 147 for N=16384) takes -16..-18% (N=34816 at T = 32) and -9% (N=16384); N=14336 gains
// no more than 3% and stays on the family. Prefill chunks: four warps of one m16 tile by 32 tokens,
// two CTAs per SM (Wide), -19..-29% at T = 128..1024 on N=14336/16384 and the N=5120 projections;
// N=34816 keeps the one-wave tile up to T = 127 (-6% at T = 64) and takes eight warps of one tile
// by 32 tokens from T = 128 (-12% at 128, -24% at 512).
using Nvfp4A16TiledVerify34816 = Nvfp4A16TiledSchedule<13, 1, 1, 4, 3, 1, 8, 16, 2>;
using Nvfp4A16TiledVerify16384 = Nvfp4A16TiledSchedule<7, 1, 1, 4, 4, 1, 8, 16, 4>;
using Nvfp4A16TiledWide        = Nvfp4A16TiledSchedule<4, 1, 1, 4, 4, 2, 8, 16, 4>;
using Nvfp4A16TiledWide34816   = Nvfp4A16TiledSchedule<8, 1, 1, 4, 4, 1, 8, 16, 2>;

template <class Geometry>
struct Nvfp4A16ExactSchedules;

template <>
struct Nvfp4A16ExactSchedules<Nvfp4N14336K5120>
    : Nvfp4A16Exact27B<Nvfp4A16TiledBand<48, 0, Nvfp4A16TiledWide>> {};
template <>
struct Nvfp4A16ExactSchedules<Nvfp4N16384K5120>
    : Nvfp4A16Exact27B<Nvfp4A16TiledBand<17, 33, Nvfp4A16TiledVerify16384>,
                       Nvfp4A16TiledBand<64, 0, Nvfp4A16TiledWide>> {};
template <>
struct Nvfp4A16ExactSchedules<Nvfp4N34816K5120>
    : Nvfp4A16Exact27B<Nvfp4A16TiledBand<17, 128, Nvfp4A16TiledVerify34816>,
                       Nvfp4A16TiledBand<128, 0, Nvfp4A16TiledWide34816>> {};
template <>
struct Nvfp4A16ExactSchedules<Nvfp4N5120K6144>
    : Nvfp4A16Exact27B<Nvfp4A16TiledBand<128, 0, Nvfp4A16TiledWide>> {};
template <>
struct Nvfp4A16ExactSchedules<Nvfp4N5120K17408>
    : Nvfp4A16Exact27B<Nvfp4A16TiledBand<128, 0, Nvfp4A16TiledWide>> {};

// The first band of `Bands` that holds `tokens` launches the call; false when none does.
template <class Geometry, class Epilogue, class Output, class... Bands>
bool launch_nvfp4_a16_tiled_bands(std::tuple<Bands...>*, const __nv_bfloat16* x,
                                  const Weight& weight, Epilogue epilogue, Output output,
                                  std::int32_t tokens, cudaStream_t stream) {
    bool done = false;
    const auto one = [&](auto band) {
        using Band = decltype(band);
        if (done || !Band::holds(tokens)) return;
        launch_nvfp4_a16_tiled<Geometry, typename Band::TiledSchedule>(x, weight, epilogue, output,
                                                                       tokens, stream);
        done = true;
    };
    (void)one;
    (one(Bands{}), ...);
    return done;
}

template <class Geometry, class Schedule, class Epilogue, class Output>
void launch_nvfp4_a16_mma_tiles(const __nv_bfloat16* x, const Weight& weight, Epilogue epilogue,
                                Output output, std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kRowsPerCta,
                    (tokens + Schedule::kTokens - 1) / Schedule::kTokens);
    nvfp4_a16_reg_mma_kernel<Geometry, Schedule, Epilogue, Output>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            x, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), 1.0F / weight.weight_scale_divisor,
            epilogue, output, tokens);
    CUDA_CHECK(cudaGetLastError());
}

// `tokens` columns of x ([K, tokens] BF16) on one family: Epilogue and Output as in
// nvfp4_a16_reg_mma_kernel, token indices counted from the call's first column.
template <class Geometry, class Family, class Epilogue, class Output>
void launch_nvfp4_a16_family(const __nv_bfloat16* x, const Weight& weight, Epilogue epilogue,
                             Output output, std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 A16 MMA: T must be positive"); }
    if (weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("nvfp4 A16 MMA: weight does not match the geometry");
    }
    if (launch_nvfp4_a16_tiled_bands<Geometry>(static_cast<typename Family::Bands*>(nullptr), x,
                                               weight, epilogue, output, tokens, stream)) {
        return;
    }
    using T8  = typename Family::template At<8>;
    using T16 = typename Family::template At<16>;
    using T32 = typename Family::template At<32>;
    if (tokens <= 8) {
        launch_nvfp4_a16_mma_tiles<Geometry, T8>(x, weight, epilogue, output, tokens, stream);
    } else if (tokens <= 16) {
        launch_nvfp4_a16_mma_tiles<Geometry, T16>(x, weight, epilogue, output, tokens, stream);
    } else if (tokens <= 32 || Family::kWidestTokens == 32) {
        launch_nvfp4_a16_mma_tiles<Geometry, T32>(x, weight, epilogue, output, tokens, stream);
    } else if constexpr (Family::kWidestTokens == 64) {
        using T64 = typename Family::template At<64>;
        launch_nvfp4_a16_mma_tiles<Geometry, T64>(x, weight, epilogue, output, tokens, stream);
    }
}

// The A16Only projection (ActivationMode::Exact) on a Qwen3.5-27B geometry.
template <class Geometry, class Epilogue, class Output>
void launch_nvfp4_a16_exact(const __nv_bfloat16* x, const Weight& weight, Epilogue epilogue,
                            Output output, std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a16_family<Geometry, Nvfp4A16ExactSchedules<Geometry>>(x, weight, epilogue,
                                                                        output, tokens, stream);
}

} // namespace ninferno::ops::detail
