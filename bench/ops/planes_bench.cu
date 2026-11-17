// #616 step 6: the layout-locality test. R8's Q6_K expert banks are stored as three planes (the
// codes, 192 bytes a 256-value block; the high plane of sub-block scales, 16 bytes; the scale plane
// of binary16 d, 2 bytes), so a lane's load of one 32-value group touches three arrays
// (generic_w4a8.cuh W4a8Q6K::load). The interleaved layout keeps a block's bytes together (192 + 16
// + 2, padded to 224 for the 16-byte loads). This bench runs R8's exact per-lane loads over both,
// with no arithmetic past folding the words (work 0), on a buffer far past the L2, and prints each
// layout's time and payload rate (210 bytes a block).
//
//   ./ninferno_planes_bench [--rows 65536] [--k 4096] [--warps 4] [--repeat 20]
//
// Prediction (#616, written before the first run): the planes are >= 15 % slower at work 0 on the
// RTX 4090. If they are not, layout locality is refuted as the cause of R8's Q6_K/Q3_K losses.
// (The first runs, 17.11, used one 32-value group a lane, kb = lane + 32 i, which is not R8's
// mapping, and a memset-only flush; the #979 review found both, and the runs were redone. The
// second runs used all 32 lanes on one row and a fixed planes-then-interleaved order; codex found
// both (R8 puts 16 lanes on a row at K 4096), and the runs were redone with ABBA rounds; the
// fourth runs write the eviction buffer once, not before every sample.)
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        const cudaError_t e = (x);                                                                 \
        if (e != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e));                           \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int kInterleavedBlockBytes = 224; // 192 codes + 16 sub-block scales + 2 d, padded to 16

struct Planes {
    const std::uint8_t* codes;   // 192 bytes a block
    const std::uint8_t* high;    // 16 bytes a block
    const std::uint16_t* scales; // 2 bytes a block
};

__device__ __forceinline__ unsigned fold(const uint4& v) { return v.x ^ v.y ^ v.z ^ v.w; }

// A warp splits into 32 / lanes rows as R8's does (moe_r8.cuh, moe.cpp r8_lanes: the fewest of 8,
// 16, 32 lanes that cover the row's K / 128 Q6_K loads at three a lane; 16 at K 4096, 8 at K
// 2048). A lane owns R8's group g = sl + lanes i (sl its lane in the row group), 128 values,
// and loads it as R8's fetch does (generic_r8.cuh: C::load at kb = 4 g + q for q < kStride = 2):
// W4a8Q6K::load's ql (two 16-byte loads at 64 h + 32 u), qh (two at 128 + 32 h), the sub-block
// scales (8 bytes at 8 h) and d (2 bytes), with u = kb & 3 (0 or 1) and h = (kb >> 2) & 1.
template <bool Interleaved>
__global__ void loads_kernel(Planes p, const std::uint8_t* inter, int rows, int groups_per_row,
                             int lanes, unsigned* out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int sl   = lane & (lanes - 1);
    const int row =
        (static_cast<int>(blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5)) * (32 / lanes) +
        lane / lanes;
    if (row >= rows) return;
    unsigned acc = 0;
    for (int g = sl; g < groups_per_row / 4; g += lanes)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int kb = 4 * g + q;
            const std::int64_t blk =
                static_cast<std::int64_t>(row) * (groups_per_row / 8) + (kb >> 3);
            const int h = (kb >> 2) & 1;
            const int u = kb & 3; // 0 or 1
            const std::uint8_t* cb;
            const std::uint8_t* hb;
            unsigned d;
            if constexpr (Interleaved) {
                const std::uint8_t* b = inter + blk * kInterleavedBlockBytes;
                cb                    = b;
                hb                    = b + 192;
                d                     = *reinterpret_cast<const unsigned short*>(b + 208);
            } else {
                cb = p.codes + blk * 192;
                hb = p.high + blk * 16;
                d  = p.scales[blk];
            }
            const auto* ql = reinterpret_cast<const uint4*>(cb + 64 * h + 32 * u);
            const auto* qh = reinterpret_cast<const uint4*>(cb + 128 + 32 * h);
            const uint4 la = ql[0], lb = ql[1], ha = qh[0], hb4 = qh[1];
            const uint2 sc = *reinterpret_cast<const uint2*>(hb + 8 * h);
            acc ^= fold(la) ^ fold(lb) ^ fold(ha) ^ fold(hb4) ^ sc.x ^ sc.y ^ d;
        }
    if (acc == 0x9e3779b9U) out[row] = acc; // keeps the loads; never true for the fill below
}

// Reads the flush buffer, so the L2 holds clean lines of it and nothing of the layouts: a memset
// alone would leave up to an L2 of dirty lines written back during the timed kernel.
__global__ void read_kernel(const uint4* p, std::size_t n, unsigned* out) {
    unsigned acc = 0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        acc ^= fold(p[i]);
    if (acc == 0x9e3779b9U) out[0] = acc;
}

int arg(int argc, char** argv, const char* name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == name) return std::atoi(argv[i + 1]);
    return fallback;
}

} // namespace

int main(int argc, char** argv) {
    const int rows   = arg(argc, argv, "--rows", 65536);
    const int k      = arg(argc, argv, "--k", 4096);
    const int warps  = arg(argc, argv, "--warps", 4);
    const int repeat = arg(argc, argv, "--repeat", 20);
    if (k % 256 != 0 || rows < 1 || warps < 1 || warps > 32 || repeat < 1) {
        std::fprintf(stderr, "--k must be a multiple of 256; rows, repeat >= 1; warps 1..32\n");
        return 1;
    }
    const int groups          = k / 32;
    const int loads           = k / 128; // R8's Q6_K loads a row (moe.cpp r8_lanes)
    const int lanes           = loads <= 24 ? 8 : loads <= 48 ? 16 : 32;
    const std::int64_t blocks = static_cast<std::int64_t>(rows) * (k / 256);
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "device %s, %d SMs, L2 %d MB; rows %d, k %d, blocks %lld (%.1f MB payload), %d warps "
        "a CTA, %d lanes a row, %d repeats\n",
        prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20, rows, k,
        static_cast<long long>(blocks), blocks * 210.0 / 1e6, warps, lanes, repeat);
    std::uint8_t *codes, *high, *inter;
    std::uint16_t* scales;
    unsigned* out;
    CHECK(cudaMalloc(&codes, blocks * 192));
    CHECK(cudaMalloc(&high, blocks * 16));
    CHECK(cudaMalloc(&scales, blocks * 2));
    CHECK(cudaMalloc(&inter, blocks * kInterleavedBlockBytes));
    CHECK(cudaMalloc(&out, rows * sizeof(unsigned)));
    CHECK(cudaMemset(codes, 0x11, blocks * 192));
    CHECK(cudaMemset(high, 0x22, blocks * 16));
    CHECK(cudaMemset(scales, 0x33, blocks * 2));
    CHECK(cudaMemset(inter, 0x44, blocks * kInterleavedBlockBytes));
    // A buffer of four L2s, read between repeats: every repeat starts cold, with clean lines.
    const std::size_t flush_bytes = std::max<std::size_t>(prop.l2CacheSize, 1 << 20) * 4;
    std::uint8_t* flush;
    CHECK(cudaMalloc(&flush, flush_bytes));
    // Written once: a memset before every sample would leave the buffer's last L2 of lines dirty
    // after the read, to be written back during the timed kernel.
    CHECK(cudaMemset(flush, 0x55, flush_bytes));
    CHECK(cudaDeviceSynchronize());
    const Planes p{codes, high, scales};
    const int rows_per_cta = warps * (32 / lanes);
    const dim3 grid((rows + rows_per_cta - 1) / rows_per_cta), block(32 * warps);
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    const auto run = [&](bool interleaved) {
        std::vector<float> ms;
        for (int r = 0; r < repeat + 2; ++r) {
            read_kernel<<<4 * prop.multiProcessorCount, 256>>>(
                reinterpret_cast<const uint4*>(flush), flush_bytes / 16, out);
            CHECK(cudaEventRecord(a));
            if (interleaved)
                loads_kernel<true><<<grid, block>>>(p, inter, rows, groups, lanes, out);
            else
                loads_kernel<false><<<grid, block>>>(p, inter, rows, groups, lanes, out);
            CHECK(cudaGetLastError());
            CHECK(cudaEventRecord(b));
            CHECK(cudaEventSynchronize(b));
            float t = 0;
            CHECK(cudaEventElapsedTime(&t, a, b));
            if (r >= 2) ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };
    // ABBA rounds (planes first, then interleaved first, ...), so a drift of clocks or heat over
    // the run falls on both layouts alike; the median of four rounds is the mean of the middle two.
    std::vector<float> planes_ms, inter_ms;
    for (int round = 0; round < 4; ++round) {
        const bool planes_first = round % 2 == 0;
        (planes_first ? planes_ms : inter_ms).push_back(run(!planes_first));
        (planes_first ? inter_ms : planes_ms).push_back(run(planes_first));
    }
    std::sort(planes_ms.begin(), planes_ms.end());
    std::sort(inter_ms.begin(), inter_ms.end());
    const double tp = (planes_ms[1] + planes_ms[2]) / 2, ti = (inter_ms[1] + inter_ms[2]) / 2;
    std::printf("planes      %.4f ms  %.1f GB/s payload (rounds %.4f %.4f %.4f %.4f)\n", tp,
                blocks * 210.0 / tp / 1e6, planes_ms[0], planes_ms[1], planes_ms[2], planes_ms[3]);
    std::printf(
        "interleaved %.4f ms  %.1f GB/s payload (rounds %.4f %.4f %.4f %.4f; %.1f MB read)\n", ti,
        blocks * 210.0 / ti / 1e6, inter_ms[0], inter_ms[1], inter_ms[2], inter_ms[3],
        blocks * double(kInterleavedBlockBytes) / 1e6);
    std::printf("planes / interleaved %.3f\n", tp / ti);
    return 0;
}
