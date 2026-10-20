#pragma once

// Candidate A16 tensor-core schedules (nvfp4_a16_mma.cuh) on one NVFP4 geometry, cold L2, against
// the Linear A16 shape route (ops::linear, A16Only) at the same T. A family
// (Nvfp4A16ExactFamily) serves T through 8/16/32(/64)-token schedules, tiling wider calls over grid
// rows; every column of every T must equal the family's widest call bit for bit
// (the T invariance nvfp4_a16_exact.cuh relies on), and stay within a rounding-level distance of
// the Linear route's output.

#include "core/device.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ninferno_bench_common.h"
#include "ops/linear/nvfp4/nvfp4_a16_exact.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ninferno::bench::a16_sweep {

struct Options {
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 128};
    int warmup = 5;
    int repeat = 30;
};

static __global__ void fill_random_bytes(std::uint8_t* bytes, std::uint64_t count,
                                         std::uint32_t seed, std::uint8_t base,
                                         std::uint8_t span) {
    for (std::uint64_t i = blockIdx.x * std::uint64_t{blockDim.x} + threadIdx.x; i < count;
         i += std::uint64_t{gridDim.x} * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 0x9e3779b1U ^ seed ^
                          static_cast<std::uint32_t>(i >> 32) * 0x85ebca6bU;
        h ^= h >> 15;
        h *= 0x2c1b3c6dU;
        h ^= h >> 12;
        bytes[i] = span == 0 ? static_cast<std::uint8_t>(h)
                             : static_cast<std::uint8_t>(base + (h >> 8) % span);
    }
}

template <int WarpsM, int MTiles, int Stages, int MinBlocksPerSm = 1>
using Tiling = ops::detail::Nvfp4A16Tiling<WarpsM, MTiles, Stages, MinBlocksPerSm>;

template <class T>
std::string tiling_name() {
    return "wm" + std::to_string(T::kWarpsM) + "mt" + std::to_string(T::kMTiles) + "s" +
           std::to_string(T::kStages) + "b" + std::to_string(T::kMinBlocksPerSm);
}

template <int WarpsM, int WarpsN, int MTiles, int NTiles, int Stages, int MinBlocksPerSm = 1,
          int CodeStages = 2>
using Tiled = ops::detail::Nvfp4A16TiledSchedule<WarpsM, WarpsN, MTiles, NTiles, Stages,
                                                 MinBlocksPerSm, 8, 16, CodeStages>;

template <class T>
std::string tiled_name() {
    if constexpr (std::is_void_v<T>) {
        return "";
    } else {
        return "_tiled" + std::to_string(T::kWarpsM) + "x" + std::to_string(T::kWarpsN) + "mt" +
               std::to_string(T::kMTiles) + "nt" + std::to_string(T::kNTiles) + "s" +
               std::to_string(T::kStages) + "b" + std::to_string(T::kMinBlocksPerSm) + "c" +
               std::to_string(T::kCodeStages);
    }
}

// A candidate: the production family template (nvfp4_a16_exact.cuh) with a printable name,
// wk<WarpsK>_b<Bytes>_<narrow tiling>_<wide tiling>_w<widest token tile>[_tiled..._from<T>].
template <int WarpsK, int Bytes, class Narrow, class Wide, int WidestTokens, class T = void,
          int TiledFrom = 0>
struct Family : ops::detail::Nvfp4A16ExactFamily<WarpsK, Bytes, Narrow, Wide, WidestTokens> {
    using Bands = std::conditional_t<std::is_void_v<T>, std::tuple<>,
                                     std::tuple<ops::detail::Nvfp4A16TiledBand<TiledFrom, 0, T>>>;
    static std::string name() {
        return "wk" + std::to_string(WarpsK) + "_b" + std::to_string(Bytes) + "_" +
               tiling_name<Narrow>() + "_" + tiling_name<Wide>() + "_w" +
               std::to_string(WidestTokens) + tiled_name<T>() +
               (std::is_void_v<T> ? std::string() : "_from" + std::to_string(TiledFrom));
    }
};

// The family shipped for every 27B geometry: the sum order `--activations exact` answers
// are pinned to. A candidate with the same WarpsK and Bytes must reproduce its bytes exactly.
using Baseline = Family<8, 16, Tiling<1, 1, 2>, Tiling<1, 2, 2>, 32>;

template <class Geometry, class F>
void launch_family(const __nv_bfloat16* x, const Weight& weight, __nv_bfloat16* out,
                   std::int32_t tokens, cudaStream_t stream) {
    ops::detail::launch_nvfp4_a16_family<Geometry, F>(
        x, weight, ops::detail::LinearIdentityEpilogue{},
        ops::detail::LinearBf16Output{out, Geometry::kOutputRows}, tokens, stream);
}

inline std::vector<std::uint16_t> download(const DeviceBuffer& buffer, std::size_t values) {
    std::vector<std::uint16_t> host(values);
    CUDA_CHECK(cudaMemcpy(host.data(), buffer.p, values * 2, cudaMemcpyDeviceToHost));
    return host;
}

inline float bf16_value(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, 4);
    return value;
}

struct Context {
    std::int32_t n, k, max_tokens;
    PackedQuantizedWeight weight;
    DeviceBuffer x, out;
    L2FlushBuffer flush;
    cudaStream_t stream = nullptr;
    // The Linear A16 route's output at the widest T, the reference every candidate is compared to.
    std::vector<std::uint16_t> linear_reference;
    // The Baseline family's output at the widest T: the bytes `--activations exact` gives.
    std::vector<std::uint16_t> baseline_reference;
};

template <class Geometry>
Context make_context(const Options& options) {
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    Context c{Geometry::kOutputRows,
              Geometry::kInputRows,
              max_tokens,
              make_nvfp4_weight(Geometry::kOutputRows, Geometry::kInputRows),
              make_bf16(static_cast<std::size_t>(Geometry::kInputRows) * max_tokens, 0x61U),
              DeviceBuffer(static_cast<std::size_t>(Geometry::kOutputRows) * max_tokens * 2),
              L2FlushBuffer(std::size_t{256} << 20)};
    CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
    const std::uint64_t codes = static_cast<std::uint64_t>(c.n) * c.k / 2;
    const std::uint64_t scales = static_cast<std::uint64_t>(c.n) * c.k / 16;
    // Random codes; scales between e4m3 0x30 and 0x3f (0.5 .. 1.875).
    auto* storage = static_cast<std::uint8_t*>(c.weight.storage.p);
    fill_random_bytes<<<1024, 256>>>(storage, codes, 0x1234U, 0, 0);
    fill_random_bytes<<<1024, 256>>>(storage + c.weight.scale_offset, scales, 0x5678U, 0x30, 16);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return c;
}

inline void run_linear(Context& c, std::int32_t tokens, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    Tensor x(c.x.p, DType::BF16, {c.k, tokens});
    Tensor out(c.out.p, DType::BF16, {c.n, tokens});
    ops::linear(x, c.weight.weight, out, ops::LinearPolicy::A16Only, workspace, stream);
}

inline void print_header() {
    std::printf("shape,candidate,T,median_us,min_us,p95_us,eff_gbs,invariant,same_as_baseline,"
                "rel_l2_vs_linear\n");
}

inline double rel_l2(const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& ref,
                     std::size_t values) {
    double diff = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < values; ++i) {
        const double a = bf16_value(got[i]), b = bf16_value(ref[i]);
        diff += (a - b) * (a - b);
        norm += b * b;
    }
    return norm > 0.0 ? std::sqrt(diff / norm) : std::sqrt(diff);
}

template <class Geometry>
void sweep_linear(Context& c, const Options& options, const char* shape) {
    const std::size_t capacity = ops::linear_workspace_capacity_bytes(
        QType::NVFP4, c.n, c.k, ops::LinearPolicy::A16Only, 1, c.max_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    run_linear(c, c.max_tokens, workspace, c.stream);
    CUDA_CHECK(cudaStreamSynchronize(c.stream));
    c.linear_reference = download(c.out, static_cast<std::size_t>(c.n) * c.max_tokens);
    const double bytes = static_cast<double>(c.n) * c.k * (0.5 + 1.0 / 16.0);
    for (const std::int32_t tokens : options.tokens) {
        const ColdTiming t = measure_cold_launch(
            [&](cudaStream_t s) { run_linear(c, tokens, workspace, s); }, c.flush, c.stream,
            options.warmup, options.repeat);
        std::printf("%s,linear_a16,%d,%.2f,%.2f,%.2f,%.0f,-,-,-\n", shape, tokens, t.median_us,
                    t.min_us, t.p95_us, bytes / (t.median_us * 1e3));
    }
}

template <class Geometry, class F>
void sweep_family(Context& c, const Options& options, const char* shape) {
    const auto* x = static_cast<const __nv_bfloat16*>(c.x.p);
    auto* out     = static_cast<__nv_bfloat16*>(c.out.p);
    launch_family<Geometry, F>(x, c.weight.weight, out, c.max_tokens, c.stream);
    CUDA_CHECK(cudaStreamSynchronize(c.stream));
    const std::size_t widest = static_cast<std::size_t>(c.n) * c.max_tokens;
    const std::vector<std::uint16_t> reference = download(c.out, widest);
    const double distance = rel_l2(reference, c.linear_reference, widest);
    const bool same_as_baseline =
        std::memcmp(reference.data(), c.baseline_reference.data(), widest * sizeof(std::uint16_t)) ==
        0;
    const double bytes    = static_cast<double>(c.n) * c.k * (0.5 + 1.0 / 16.0);
    for (const std::int32_t tokens : options.tokens) {
        CUDA_CHECK(cudaMemsetAsync(c.out.p, 0xff, c.out.bytes, c.stream));
        launch_family<Geometry, F>(x, c.weight.weight, out, tokens, c.stream);
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
        const std::size_t values        = static_cast<std::size_t>(c.n) * tokens;
        const std::vector<std::uint16_t> got = download(c.out, values);
        const bool invariant =
            std::memcmp(got.data(), reference.data(), values * sizeof(std::uint16_t)) == 0;
        const ColdTiming t = measure_cold_launch(
            [&](cudaStream_t s) { launch_family<Geometry, F>(x, c.weight.weight, out, tokens, s); },
            c.flush, c.stream, options.warmup, options.repeat);
        std::printf("%s,%s,%d,%.2f,%.2f,%.2f,%.0f,%s,%s,%.2e\n", shape, F::name().c_str(), tokens,
                    t.median_us, t.min_us, t.p95_us, bytes / (t.median_us * 1e3),
                    invariant ? "yes" : "NO", same_as_baseline ? "yes" : "NO", distance);
    }
    std::fflush(stdout);
}

template <class Geometry, class... Families>
void sweep(const Options& options, const char* shape) {
    Context c = make_context<Geometry>(options);
    sweep_linear<Geometry>(c, options, shape);
    launch_family<Geometry, Baseline>(static_cast<const __nv_bfloat16*>(c.x.p), c.weight.weight,
                                      static_cast<__nv_bfloat16*>(c.out.p), c.max_tokens,
                                      c.stream);
    CUDA_CHECK(cudaStreamSynchronize(c.stream));
    c.baseline_reference = download(c.out, static_cast<std::size_t>(c.n) * c.max_tokens);
    (sweep_family<Geometry, Families>(c, options, shape), ...);
    CUDA_CHECK(cudaStreamDestroy(c.stream));
}

// The candidate families every geometry is swept over. All keep the Baseline's WarpsK = 8 and
// Bytes = 16 (so its sum order: `--activations exact` answers stay byte for byte) and its narrow
// tiling. The first vary the wide MMA tiling (T > 16: MTP verify of five to eight requests):
// Tiling<WarpsM, MTiles, Stages, MinBlocksPerSm>. The others hand calls of 17 tokens and more to
// the tiled kernel (nvfp4_a16_tiled.cuh): Tiled<WarpsM, WarpsN, MTiles, NTiles, Stages, MinBlocks,
// CodeStages>; the 13-, 7- and 6-warp tiles put the N=34816/16384/14336 rows in one wave at T = 32.
#define NINFERNO_A16_WIDE(...) Family<8, 16, Tiling<1, 1, 2>, Tiling<__VA_ARGS__>, 32>
#define NINFERNO_A16_TILED(...)                                                                    \
    Family<8, 16, Tiling<1, 1, 2>, Tiling<1, 2, 2>, 32, Tiled<__VA_ARGS__>, 17>
#define NINFERNO_A16_SWEEP_FAMILIES                                                                \
    Baseline, NINFERNO_A16_TILED(4, 1, 2, 8, 3), NINFERNO_A16_TILED(8, 1, 1, 4, 4),                \
        NINFERNO_A16_TILED(8, 1, 1, 4, 4, 1, 4), NINFERNO_A16_TILED(13, 1, 1, 4, 3, 1, 2),         \
        NINFERNO_A16_TILED(13, 1, 1, 4, 4, 1, 4), NINFERNO_A16_TILED(7, 1, 1, 4, 4, 1, 4),         \
        NINFERNO_A16_TILED(6, 1, 1, 4, 4, 1, 4), NINFERNO_A16_TILED(6, 1, 2, 4, 4, 1, 3),          \
        NINFERNO_A16_TILED(16, 1, 1, 4, 4, 1, 4), NINFERNO_A16_TILED(4, 1, 1, 4, 4, 2, 4)

void sweep_n14336_k5120(const Options& options);
void sweep_n16384_k5120(const Options& options);
void sweep_n34816_k5120(const Options& options);
void sweep_n5120_k6144(const Options& options);
void sweep_n5120_k17408(const Options& options);

} // namespace ninferno::bench::a16_sweep
