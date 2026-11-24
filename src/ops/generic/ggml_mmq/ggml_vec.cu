// ggml_mmq.h vector kernel: products of a ggml block matrix with one to kVecMaxColumns q8_1
// activation columns, the output epilogue in the kernel. After ninfer-all's bridge's vector
// kernel (its src/ops/linear/gguf/ggml_bridge_vec.cuh, Apache-2.0), reading the planes NInferno
// stores.
//
// A warp owns two output rows (a gated product: the gate row and its up row); lane l takes the
// 32-value slices l, l + 32, ... of each. Every slice of a row is decoded once into eight int8x4
// words in value order (word i holds values 4i..4i+3) plus the factors of its format, then dotted
// with that slice of every column (dp4a). The activation is planar: int8 values [columns][K], then
// one (d, sum of x) half2 per 32 values [columns][K / 32], ggml's q8_1 numbers.

#include "ggml_mmq_internal.cuh"

#include "core/device_class.h"
#include "core/kernel_occupancy.h"
#include "core/pdl.cuh"
#include "ops/generic/kernel_facts.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninferno::ops::ggml_mmq {

static_assert(kHalf2Bytes == sizeof(half2)); // vec_activation_bytes (ggml_mmq_host.cpp)
namespace detail {
namespace {

constexpr int kVecWarps = 4;

// The runtime-K plain instance of 4 columns has a twin bounded to 7 resident CTAs per SM
// (__launch_bounds__' minimum), the registers of 3 columns: unbounded it takes 85 registers where 3
// take 72 (sm_86 / sm_89, CUDA 13), which is 5 CTAs of 128 threads on a 64K-register SM instead of
// 7. The grid is sized to the resident CTAs, so the launch takes the twin when that saves a round
// of row groups (480..512 groups on an 82-SM card: 2 rounds against 1), and the plain instance
// otherwise. The rule is deliberately conservative: on a 2080 Ti the twin is 11..22 % faster even
// at shapes it does not take (the plain instance's second round runs 72 CTAs, the twin's 4), but
// on a 4090 the same bound measured 9 % slower at 1024 groups with 2 rounds either way
// (route-tune), and no model here explains both, so only the case with a counted saving is taken.
// The other instances have no twin: a bound makes the compiler spend more registers on every
// instance it is given to (110 for 3 columns), and the unrolled-K and gated ones spill under 72.
constexpr bool kVecHasBoundedTwin(int columns, int ks, bool gated) {
    return columns == 4 && ks == 0 && !gated;
}

constexpr int kVecBoundedBlocks = 7;

thread_local VecInstance g_vec_instance = VecInstance::Auto;

struct Slice {
    int w[8];
    float f[6];
};

// The weight planes of a product with the bytes per block of each (ggml_planes_v1, or whole
// blocks in codes for Q4_K / Q5_K).
struct VecPlanes {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
};

__device__ __forceinline__ std::uint32_t u32_a2(const std::uint8_t* p) {
    const auto* h = reinterpret_cast<const std::uint16_t*>(p);
    return std::uint32_t(h[0]) | (std::uint32_t(h[1]) << 16);
}

__device__ __forceinline__ float half_at(const std::uint8_t* p) {
    return __half2float(*reinterpret_cast<const half*>(p));
}

// Negates the bytes of `g` whose bit is set in the low nibble of `bits`. Every grid this serves has
// no zero byte, so the per-byte two's-complement increment cannot carry.
__device__ __forceinline__ int negate_bytes(std::uint32_t g, std::uint32_t bits) {
    const std::uint32_t ones = ((bits & 0xF) * 0x00204081u) & 0x01010101u;
    return int((g ^ (ones * 0xFFu)) + ones);
}

// ksigns_iq2xs: seven stored sign bits and an eighth that makes the parity even.
__device__ __forceinline__ std::uint32_t ksigns(std::uint32_t v) {
    return v ^ ((__popc(v) & 1) << 7);
}

__device__ __forceinline__ int dot4(const int* w, const int* a, int acc) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = ggml_cuda_dp4a(w[i], a[i], acc);
    return acc;
}

// The sum of 16 int8 activation values.
__device__ __forceinline__ int sum4(const int* a) {
    int acc = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = ggml_cuda_dp4a(0x01010101, a[i], acc);
    return acc;
}

// The 6-bit scale and min of sub-block j of a Q4_K / Q5_K block (ggml's get_scale_min_k4).
__device__ __forceinline__ void scale_min_k4(const std::uint8_t* s, int j, int& sc, int& m) {
    if (j < 4) {
        sc = s[j] & 63;
        m  = s[j + 4] & 63;
    } else {
        sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        m  = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// decode(): slice u (0 .. kBlockValues / 32 - 1) of block `blk` of a row; dot(): its contribution
// to one column, d and sum the column's slice scale and the sum of its unquantized values.
template <ggml_type type>
struct Decoder;

template <>
struct Decoder<GGML_TYPE_IQ4_XS> {
    // codes: qs (128); high: scales_l (4); scales: d, scales_h (4).
    static constexpr int kCode = 128, kHigh = 4, kScale = 4, kTableWords = 0, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t*, Slice& s) {
        const std::uint8_t* sc = p.scales + blk * kScale;
        const std::uint32_t sh = *reinterpret_cast<const std::uint16_t*>(sc + 2);
        const int ls = ((p.high[blk * kHigh + u / 2] >> (4 * (u & 1))) & 0xF) | (((sh >> (2 * u)) & 3) << 4);
        s.f[0]       = half_at(sc) * float(ls - 32);
        const std::uint8_t* qs   = p.codes + blk * kCode + 16 * u;
        const uint2 lo           = *reinterpret_cast<const uint2*>(qs);
        const uint2 hi           = *reinterpret_cast<const uint2*>(qs + 8);
        const std::uint32_t q[4] = {lo.x, lo.y, hi.x, hi.y};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = get_int_from_table_16(int(q[j]), kvalues_iq4nl);
            s.w[j]       = v.x;
            s.w[j + 4]   = v.y;
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_Q4_K> {
    // Whole blocks: d, dmin, scales (12), qs (128).
    static constexpr int kCode = 144, kHigh = 0, kScale = 0, kTableWords = 0, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t*, Slice& s) {
        const std::uint8_t* b = p.codes + blk * kCode;
        const float2 dm       = __half22float2(*reinterpret_cast<const half2*>(b));
        int sc, m;
        scale_min_k4(b + 4, u, sc, m);
        s.f[0]              = dm.x * float(sc);
        s.f[1]              = dm.y * float(m);
        const uint4* qs     = reinterpret_cast<const uint4*>(b + 16 + 32 * (u / 2));
        const uint4 a0      = qs[0];
        const uint4 a1      = qs[1];
        const int shift     = 4 * (u & 1);
        const std::uint32_t v[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) s.w[i] = int((v[i] >> shift) & 0x0F0F0F0Fu);
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0))) - s.f[1] * sum;
    }
};

template <>
struct Decoder<GGML_TYPE_Q5_K> {
    // Whole blocks: d, dmin, scales (12), qh (32), qs (128).
    static constexpr int kCode = 176, kHigh = 0, kScale = 0, kTableWords = 0, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t*, Slice& s) {
        const std::uint8_t* b = p.codes + blk * kCode;
        const float2 dm       = __half22float2(*reinterpret_cast<const half2*>(b));
        int sc, m;
        scale_min_k4(b + 4, u, sc, m);
        s.f[0]          = dm.x * float(sc);
        s.f[1]          = dm.y * float(m);
        const uint4* qh = reinterpret_cast<const uint4*>(b + 16);
        const uint4* ql = reinterpret_cast<const uint4*>(b + 48 + 32 * (u / 2));
        const uint4 h0 = qh[0], h1 = qh[1], l0 = ql[0], l1 = ql[1];
        const std::uint32_t hv[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const std::uint32_t lv[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
        const int shift           = 4 * (u & 1);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s.w[i] = int(((lv[i] >> shift) & 0x0F0F0F0Fu) | (((hv[i] >> u) << 4) & 0x10101010u));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0))) - s.f[1] * sum;
    }
};

template <>
struct Decoder<GGML_TYPE_Q2_K> {
    // codes: qs (64); high: scales (16, a 4-bit scale and min per 16 values); scales: d, dmin.
    // Slice u is the 2-bit field 2 (u % 4) of the half u / 4 of qs.
    static constexpr int kCode = 64, kHigh = 16, kScale = 4, kTableWords = 0, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t*, Slice& s) {
        const int h = u / 4, j = u % 4;
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(p.scales + blk * kScale));
        const std::uint8_t* sc = p.high + blk * kHigh + 8 * h + 2 * j;
        const std::uint32_t lo = sc[0], hi = sc[1];
        s.f[0]                 = dm.x * float(lo & 0xF);
        s.f[1]                 = dm.x * float(hi & 0xF);
        s.f[2]                 = dm.y * float(lo >> 4);
        s.f[3]                 = dm.y * float(hi >> 4);
        const uint4* qs        = reinterpret_cast<const uint4*>(p.codes + blk * kCode + 32 * h);
        const uint4 a0         = qs[0];
        const uint4 a1         = qs[1];
        const std::uint32_t v[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) s.w[i] = int((v[i] >> (2 * j)) & 0x03030303u);
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)) -
                    s.f[2] * float(sum4(a)) - s.f[3] * float(sum4(a + 4)));
    }
};

template <>
struct Decoder<GGML_TYPE_Q6_K> {
    // codes: ql (128), qh (64); high: scales (16); scales: d (2).
    static constexpr int kCode = 192, kHigh = 16, kScale = 2, kTableWords = 0, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t*, Slice& s) {
        const int h = u / 4, q = u % 4;
        const std::uint8_t* b = p.codes + blk * kCode;
        const float d         = half_at(p.scales + blk * kScale);
        const auto* sc        = reinterpret_cast<const std::int8_t*>(p.high + blk * kHigh + 8 * h);
        s.f[0]                = d * float(sc[2 * q]);
        s.f[1]                = d * float(sc[2 * q + 1]);
        const std::uint8_t* ql = b + 64 * h + 32 * (q & 1);
        const std::uint8_t* qh = b + 128 + 32 * h;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t v = ((u32_a2(ql + 4 * i) >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                    (((u32_a2(qh + 4 * i) >> (2 * q)) << 4) & 0x30303030u);
            // v - 32 per byte, v in [0, 64): flip the high bits by whether v is below 32.
            s.w[i] = int(v ^ 0xE0E0E0E0u ^ ((v & 0x20202020u) * 6u));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ3_XXS> {
    // codes: qs (64); high: scales and signs (32); scales: d (2).
    static constexpr int kCode = 64, kHigh = 32, kScale = 2, kTableWords = 256, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return iq3xxs_grid; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const uint2 qs          = *reinterpret_cast<const uint2*>(p.codes + blk * kCode + 8 * u);
        const std::uint32_t q0  = qs.x;
        const std::uint32_t q1  = qs.y;
        const std::uint32_t aux = *reinterpret_cast<const std::uint32_t*>(p.high + blk * kHigh + 4 * u);
        s.f[0]                  = half_at(p.scales + blk * kScale) * (0.5f + float(aux >> 28)) * 0.5f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const std::uint32_t q  = e < 2 ? q0 : q1;
            const std::uint32_t sg = ksigns((aux >> (7 * e)) & 0x7F);
            s.w[2 * e]             = negate_bytes(grid[(q >> (16 * (e & 1))) & 0xFF], sg);
            s.w[2 * e + 1]         = negate_bytes(grid[(q >> (16 * (e & 1) + 8)) & 0xFF], sg >> 4);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ3_S> {
    // codes: qs (64), qh (8), signs (32); high: scales (4); scales: d (2).
    static constexpr int kCode = 104, kHigh = 4, kScale = 2, kTableWords = 512, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return iq3s_grid; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        // The code plane's 104-byte blocks keep qs 8-byte and the signs 4-byte aligned.
        const std::uint8_t* b  = p.codes + blk * kCode;
        const uint2 q          = *reinterpret_cast<const uint2*>(b + 8 * u);
        const std::uint32_t q0 = q.x;
        const std::uint32_t q1 = q.y;
        const std::uint32_t qh = b[64 + u];
        const std::uint32_t sg = *reinterpret_cast<const std::uint32_t*>(b + 72 + 4 * u);
        const int sc           = (p.high[blk * kHigh + u / 2] >> (4 * (u & 1))) & 0xF;
        s.f[0]                 = half_at(p.scales + blk * kScale) * float(1 + 2 * sc);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const std::uint32_t idx =
                (((e < 4 ? q0 : q1) >> (8 * (e & 3))) & 0xFF) | ((qh << (8 - e)) & 0x100);
            s.w[e] = negate_bytes(grid[idx], sg >> (4 * e));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_Q3_K> {
    // codes: hmask (32), qs (64); high: scales (12, 6-bit, ggml's packing); scales: d (2).
    // Slice u is the 2-bit field 2 (u % 4) of the half u / 4 of qs with hmask bit u: v - 4.
    static constexpr int kCode = 96, kHigh = 12, kScale = 2, kTableWords = 0, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static int scale(const std::uint8_t* sc, int i) {
        return int(((sc[i % 8] >> (4 * (i / 8))) & 0xF) | (((sc[8 + i % 4] >> (2 * (i / 4))) & 3) << 4)) -
               32;
    }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t*, Slice& s) {
        const int h = u / 4, j = u % 4;
        const float d          = half_at(p.scales + blk * kScale);
        const std::uint8_t* sc = p.high + blk * kHigh;
        s.f[0]                 = d * float(scale(sc, 8 * h + 2 * j));
        s.f[1]                 = d * float(scale(sc, 8 * h + 2 * j + 1));
        const std::uint8_t* b  = p.codes + blk * kCode;
        const uint4* hm        = reinterpret_cast<const uint4*>(b);
        const uint4* qs        = reinterpret_cast<const uint4*>(b + 32 + 32 * h);
        const uint4 m0 = hm[0], m1 = hm[1], q0 = qs[0], q1 = qs[1];
        const std::uint32_t mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
        const std::uint32_t qv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t v =
                ((qv[i] >> (2 * j)) & 0x03030303u) | (((mv[i] >> u) & 0x01010101u) << 2);
            // v - 4 per byte, v in [0, 8): flip the high bits by whether v is below 4.
            s.w[i] = int(v ^ 0xFCFCFCFCu ^ ((v & 0x04040404u) * 0x3Eu));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_XXS> {
    // codes: qs (64: per slice four grid bytes, then seven sign bits x 4 and the scale); scales: d.
    static constexpr int kCode = 64, kHigh = 0, kScale = 2, kTableWords = 512, kBlockValues = 256;
    __device__ static const std::uint32_t* table() {
        return reinterpret_cast<const std::uint32_t*>(iq2xxs_grid);
    }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const uint2 q = *reinterpret_cast<const uint2*>(p.codes + blk * kCode + 8 * u);
        s.f[0]        = half_at(p.scales + blk * kScale) * (0.5f + float(q.y >> 28)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t at = 2 * ((q.x >> (8 * l)) & 0xFF);
            const std::uint32_t sg = ksigns((q.y >> (7 * l)) & 0x7F);
            s.w[2 * l]             = negate_bytes(grid[at], sg);
            s.w[2 * l + 1]         = negate_bytes(grid[at + 1], sg >> 4);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_XS> {
    // codes: qs (64: four 16-bit words per slice, a 9-bit grid index and seven sign bits);
    // high: scales (8, two 4-bit per slice); scales: d.
    static constexpr int kCode = 64, kHigh = 8, kScale = 2, kTableWords = 1024, kBlockValues = 256;
    __device__ static const std::uint32_t* table() {
        return reinterpret_cast<const std::uint32_t*>(iq2xs_grid);
    }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const uint2 q          = *reinterpret_cast<const uint2*>(p.codes + blk * kCode + 8 * u);
        const float d          = half_at(p.scales + blk * kScale);
        const std::uint32_t sc = p.high[blk * kHigh + u];
        s.f[0]                 = d * (0.5f + float(sc & 0xF)) * 0.25f;
        s.f[1]                 = d * (0.5f + float(sc >> 4)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t w  = ((l < 2 ? q.x : q.y) >> (16 * (l & 1))) & 0xFFFF;
            const std::uint32_t at = 2 * (w & 0x1FF);
            const std::uint32_t sg = ksigns(w >> 9);
            s.w[2 * l]             = negate_bytes(grid[at], sg);
            s.w[2 * l + 1]         = negate_bytes(grid[at + 1], sg >> 4);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_S> {
    // codes: qs (32: grid index bytes), signs (32), qh (8: two index bits per 8 values); high:
    // scales (8, two 4-bit per slice); scales: d. The 72-byte blocks keep 4-byte words aligned.
    static constexpr int kCode = 72, kHigh = 8, kScale = 2, kTableWords = 2048, kBlockValues = 256;
    __device__ static const std::uint32_t* table() {
        return reinterpret_cast<const std::uint32_t*>(iq2s_grid);
    }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const std::uint8_t* b  = p.codes + blk * kCode;
        const std::uint32_t qs = *reinterpret_cast<const std::uint32_t*>(b + 4 * u);
        const std::uint32_t sg = *reinterpret_cast<const std::uint32_t*>(b + 32 + 4 * u);
        const std::uint32_t qh = b[64 + u];
        const float d          = half_at(p.scales + blk * kScale);
        const std::uint32_t sc = p.high[blk * kHigh + u];
        s.f[0]                 = d * (0.5f + float(sc & 0xF)) * 0.25f;
        s.f[1]                 = d * (0.5f + float(sc >> 4)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t at = 2 * (((qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300));
            s.w[2 * l]             = negate_bytes(grid[at], sg >> (8 * l));
            s.w[2 * l + 1]         = negate_bytes(grid[at + 1], sg >> (8 * l + 4));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ1_S> {
    // codes: qs (32: grid index bytes); high: qh (16: per slice three index bits x 4, the scale and
    // the delta's sign); scales: d. The grid holds value + 1 in nibbles, so the slice's value is
    // dl (code + delta - 1), and dl (delta - 1) multiplies the column's sum.
    static constexpr int kCode = 32, kHigh = 16, kScale = 2, kTableWords = 2048, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return iq1s_grid_gpu; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const std::uint32_t qs = *reinterpret_cast<const std::uint32_t*>(p.codes + blk * kCode + 4 * u);
        const std::uint32_t qh = *reinterpret_cast<const std::uint16_t*>(p.high + blk * kHigh + 2 * u);
        s.f[0] = half_at(p.scales + blk * kScale) * float(((qh >> 11) & 0x0E) + 1);
        s.f[1] = -1.0f + IQ1S_DELTA - float(qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t g = grid[((qs >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)];
            s.w[2 * l]            = int(g & 0x0F0F0F0Fu);
            s.w[2 * l + 1]        = int((g >> 4) & 0x0F0F0F0Fu);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return s.f[0] * (d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0))) + s.f[1] * sum);
    }
};

// The 32-value block formats: one slice per block, lane l reading block l of each 32.

// Bit k (k < 4) of `bits` at bit 4 of byte k: the fifth bit of four Q5 codes.
__device__ __forceinline__ std::uint32_t fifth_bits(std::uint32_t bits) {
    return (((bits & 0xF) * 0x00204081u) & 0x01010101u) << 4;
}

template <>
struct Decoder<GGML_TYPE_Q4_0> {
    // codes: qs (16: value j low nibble of byte j, value j + 16 its high nibble); scales: d.
    static constexpr int kCode = 16, kHigh = 0, kScale = 2, kTableWords = 0, kBlockValues = 32;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int,
                                                  const std::uint32_t*, Slice& s) {
        const uint4 q            = *reinterpret_cast<const uint4*>(p.codes + blk * kCode);
        const std::uint32_t v[4] = {q.x, q.y, q.z, q.w};
        s.f[0]                   = half_at(p.scales + blk * kScale);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s.w[j]     = __vsubss4(int(v[j] & 0x0F0F0F0Fu), 0x08080808);
            s.w[j + 4] = __vsubss4(int((v[j] >> 4) & 0x0F0F0F0Fu), 0x08080808);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_Q4_1> {
    // codes: qs (16, as Q4_0); scales: d, m (value d q + m).
    static constexpr int kCode = 16, kHigh = 0, kScale = 4, kTableWords = 0, kBlockValues = 32;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int,
                                                  const std::uint32_t*, Slice& s) {
        const uint4 q            = *reinterpret_cast<const uint4*>(p.codes + blk * kCode);
        const std::uint32_t v[4] = {q.x, q.y, q.z, q.w};
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(p.scales + blk * kScale));
        s.f[0]          = dm.x;
        s.f[1]          = dm.y;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s.w[j]     = int(v[j] & 0x0F0F0F0Fu);
            s.w[j + 4] = int((v[j] >> 4) & 0x0F0F0F0Fu);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0))) + s.f[1] * sum;
    }
};

template <>
struct Decoder<GGML_TYPE_Q5_0> {
    // codes: qs (16, as Q4_0); high: qh (4: bit j the fifth bit of value j); scales: d. v - 16.
    static constexpr int kCode = 16, kHigh = 4, kScale = 2, kTableWords = 0, kBlockValues = 32;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int,
                                                  const std::uint32_t*, Slice& s) {
        const uint4 q            = *reinterpret_cast<const uint4*>(p.codes + blk * kCode);
        const std::uint32_t v[4] = {q.x, q.y, q.z, q.w};
        const std::uint32_t qh   = *reinterpret_cast<const std::uint32_t*>(p.high + blk * kHigh);
        s.f[0]                   = half_at(p.scales + blk * kScale);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s.w[j] = __vsubss4(int((v[j] & 0x0F0F0F0Fu) | fifth_bits(qh >> (4 * j))), 0x10101010);
            s.w[j + 4] = __vsubss4(
                int(((v[j] >> 4) & 0x0F0F0F0Fu) | fifth_bits(qh >> (16 + 4 * j))), 0x10101010);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_Q5_1> {
    // codes: qs (16); high: qh (4); scales: d, m (value d q + m, q the 5-bit code).
    static constexpr int kCode = 16, kHigh = 4, kScale = 4, kTableWords = 0, kBlockValues = 32;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int,
                                                  const std::uint32_t*, Slice& s) {
        const uint4 q            = *reinterpret_cast<const uint4*>(p.codes + blk * kCode);
        const std::uint32_t v[4] = {q.x, q.y, q.z, q.w};
        const std::uint32_t qh   = *reinterpret_cast<const std::uint32_t*>(p.high + blk * kHigh);
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(p.scales + blk * kScale));
        s.f[0]          = dm.x;
        s.f[1]          = dm.y;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s.w[j]     = int((v[j] & 0x0F0F0F0Fu) | fifth_bits(qh >> (4 * j)));
            s.w[j + 4] = int(((v[j] >> 4) & 0x0F0F0F0Fu) | fifth_bits(qh >> (16 + 4 * j)));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0))) + s.f[1] * sum;
    }
};

template <>
struct Decoder<GGML_TYPE_IQ4_NL> {
    // codes: qs (16, as Q4_0, the nibbles through kvalues_iq4nl); scales: d.
    static constexpr int kCode = 16, kHigh = 0, kScale = 2, kTableWords = 0, kBlockValues = 32;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int,
                                                  const std::uint32_t*, Slice& s) {
        const uint4 q            = *reinterpret_cast<const uint4*>(p.codes + blk * kCode);
        const std::uint32_t v[4] = {q.x, q.y, q.z, q.w};
        s.f[0]                   = half_at(p.scales + blk * kScale);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 t = get_int_from_table_16(int(v[j]), kvalues_iq4nl);
            s.w[j]       = t.x;
            s.w[j + 4]   = t.y;
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ1_M> {
    // codes: qs (32: grid index bytes); high: qh (16: per slice two bytes of two 4-bit fields,
    // three index bits and the delta's sign per 8 values); scales: four 16-bit words, per slice two
    // 3-bit scales, d (binary16) in their top nibbles. The grid holds value + 1 in nibbles: each 8
    // values contribute dl (code . x + (delta - 1) sum x) with sum x over the column's codes.
    static constexpr int kCode = 32, kHigh = 16, kScale = 8, kTableWords = 2048, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return iq1s_grid_gpu; }
    __device__ __forceinline__ static void decode(const VecPlanes& p, std::int64_t blk, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const auto* sc = reinterpret_cast<const std::uint16_t*>(p.scales + blk * kScale);
        const auto bits = static_cast<unsigned short>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                                                      ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        const float d          = __half2float(__ushort_as_half(bits));
        const std::uint32_t sb = sc[u / 2] >> (6 * (u % 2));
        s.f[0]                 = d * float(2 * (sb & 7) + 1);
        s.f[1]                 = d * float(2 * ((sb >> 3) & 7) + 1);
        const std::uint32_t qs = *reinterpret_cast<const std::uint32_t*>(p.codes + blk * kCode + 4 * u);
        const std::uint32_t qh = *reinterpret_cast<const std::uint16_t*>(p.high + blk * kHigh + 2 * u);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t h = (qh >> (4 * l)) & 0xF;
            const std::uint32_t g = grid[((qs >> (8 * l)) & 0xFF) | ((h & 7) << 8)];
            s.w[2 * l]            = int(g & 0x0F0F0F0Fu);
            s.w[2 * l + 1]        = int((g >> 4) & 0x0F0F0F0Fu);
            s.f[2 + l]            = -1.0f + ((h & 8) != 0 ? -IQ1M_DELTA : IQ1M_DELTA);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        float half[2] = {0.0f, 0.0f};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int* w = s.w + 2 * l;
            const int* x = a + 2 * l;
            const int code =
                ggml_cuda_dp4a(w[1], x[1], ggml_cuda_dp4a(w[0], x[0], 0));
            const int sum = ggml_cuda_dp4a(0x01010101, x[1], ggml_cuda_dp4a(0x01010101, x[0], 0));
            half[l / 2] += float(code) + s.f[2 + l] * float(sum);
        }
        return d * (s.f[0] * half[0] + s.f[1] * half[1]);
    }
};

struct VecArgs {
    VecPlanes first;  // rows [0, rows), or the gate rows
    VecPlanes second; // gated: the up rows
    int blocks_per_row;
    int rows;
    int k;
    const std::int8_t* qs; // [columns][k]
    const half2* ds;       // [columns][k / 32]
    VecOut out;
};

// GELU's tanh form through the logistic, as the generic route computes it (generic_gelu_tanh): the
// sum 0.5 * (1 + tanh(u)) cancels to 0 for a gate below about -5.
__device__ __forceinline__ float gate_of(const VecOut& o, float g) {
    constexpr float kTwoSqrt2Pi = 1.59576912160573071176F; // 2 * sqrt(2 / pi)
    return o.gelu_gate != 0 ? g / (1.0F + expf(-kTwoSqrt2Pi * (g + 0.044715F * g * g * g)))
                            : g / (1.0F + __expf(-g));
}

__device__ __forceinline__ void store(const VecOut& o, int row, int column, float v) {
    if (o.epilogue == VecOut::Epilogue::StoreF32) {
        o.f32[std::int64_t(column) * o.f32_ld + row] = v;
        return;
    }
    if (o.epilogue == VecOut::Epilogue::GateProduct) {
        v = gate_of(o, o.gate[std::int64_t(column) * o.gate_ld + row]) * v;
    }
    int s = 0;
    while (s + 1 < o.segments && row >= o.segment_end[s]) ++s;
    const int local = row - (s > 0 ? o.segment_end[s - 1] : 0);
    auto* out = static_cast<__nv_bfloat16*>(o.out[s]) + static_cast<std::int64_t>(column) * o.ld[s] + local;
    if (o.epilogue == VecOut::Epilogue::AddResidual) v += __bfloat162float(*out);
    *out = __float2bfloat16(v);
}

// The transposed warp reduction of `V` per-lane values (2 <= V <= 32; #1044). While a lane keeps an
// even number n > 1 of values, the butterfly step of offset o (16, 8, ...) halves them: lanes with
// bit o set keep the upper half and send the lower, the others the reverse, and each adds its
// partner's. After those vec_halvings(V) = H steps a lane keeps N = V >> H values, the block of V
// that the lane's top H bits name (lane & 16, lane & 8, ...: block = lane >> (5 - H)); the
// remaining 5 - H steps sum each of the N across the lanes of the block. No padding: V = 6 spends 3
// + 3 x 4 = 15 shuffles, not the 8 values of a power of two (which spilled on sm_86) nor 6 x 5.
// Lane L returns the whole sum of value block * N + sel, sel = L's low 5 - H bits, when sel < N.
__host__ __device__ constexpr int vec_halvings(int v) {
    int h = 0;
    while (v > 1 && v % 2 == 0 && h < 5) {
        v /= 2;
        ++h;
    }
    return h;
}

template <int V>
__device__ __forceinline__ float vec_transpose_reduce(float (&val)[V], int lane) {
    static_assert(V >= 2 && V <= 32);
    constexpr int kH = vec_halvings(V);
    constexpr int kN = V >> kH;
#pragma unroll
    for (int s = 0, n = V; s < kH; ++s, n >>= 1) {
        const int o      = 16 >> s;
        const bool upper = (lane & o) != 0;
#pragma unroll
        for (int i = 0; i < n / 2; ++i) {
            const float send = upper ? val[i] : val[i + n / 2];
            const float keep = upper ? val[i + n / 2] : val[i];
            val[i]           = keep + __shfl_xor_sync(0xFFFFFFFFu, send, o);
        }
    }
#pragma unroll
    for (int o = 16 >> kH; o > 0; o >>= 1) {
#pragma unroll
        for (int i = 0; i < kN; ++i) val[i] += __shfl_xor_sync(0xFFFFFFFFu, val[i], o);
    }
    const int sel = lane & ((32 >> kH) - 1);
    float mine    = 0.0f;
#pragma unroll
    for (int i = 0; i < kN; ++i) {
        if (sel == i) mine = val[i];
    }
    return mine;
}

// KS: the static K the slice loop unrolls over, or 0 for the runtime K. Gated: the SwiGlu pair (gate
// and up rows, one row per warp), a separate instance as ninfer-all's Fused one: the plain product
// keeps no gated branches or second-plane state in its loop.
template <ggml_type type, int T, int KS, bool Gated, bool Bounded = false>
__launch_bounds__(kVecWarps * 32, Bounded ? kVecBoundedBlocks : 0) __global__
    void vec_kernel(VecArgs p) {
    using D                 = Decoder<type>;
    constexpr int kTable    = D::kTableWords > 0 ? D::kTableWords : 4;
    constexpr int kPerBlock = D::kBlockValues / 32;
    constexpr int kRows     = 2;
    __shared__ __align__(16) std::uint32_t table[kTable];
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += kVecWarps * 32) table[i] = src[i];
        __syncthreads();
    }
    // Launched as a programmatic dependent of the activation quantizer (sm_90+): the CTAs and the
    // codebook are set up under it.
    pdl::wait_for_dependencies();
    constexpr bool gated = Gated;
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int k      = KS != 0 ? KS : p.k;
    const int slices = k / 32;
    const int step   = gated ? 1 : kRows;
    const int sub    = lane % kPerBlock;

    for (int row0 = (blockIdx.x * kVecWarps + warp) * step; row0 < p.rows;
         row0 += gridDim.x * kVecWarps * step) {
        // Block index of each row's first block this lane reads.
        std::int64_t base[kRows];
        const VecPlanes second = gated ? p.second : p.first;
        base[0] = std::int64_t(row0) * p.blocks_per_row + lane / kPerBlock;
        base[1] = std::int64_t(gated ? row0 : min(row0 + 1, p.rows - 1)) * p.blocks_per_row +
                  lane / kPerBlock;
        float acc[kRows][T];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) acc[r][j] = 0.0f;
        }
        const auto slice = [&](int it) {
            const int s = lane + 32 * it;
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + s]);
                d[j]            = ds.x;
                sum[j]          = ds.y;
                const auto* q   = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * k + 32 * s);
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                Slice sl;
                D::decode(r == 0 ? p.first : second, base[r] + std::int64_t(32 / kPerBlock) * it, sub,
                          table, sl);
#pragma unroll
                for (int j = 0; j < T; ++j) acc[r][j] += D::dot(sl, a[j], d[j], sum[j]);
            }
        };
        if constexpr (KS != 0) {
            static_assert(KS % 1024 == 0);
#pragma unroll
            for (int it = 0; it < KS / 1024; ++it) slice(it);
        } else {
            for (int it = 0; lane + 32 * it < slices; ++it) slice(it);
        }
        if constexpr (gated || kRows * T != 8) {
            // T != 4 and the gated pair (#1044): five shuffles a value and 2T predicated stores, as
            // before. The transposed reduction below gave Q3_K at T = 2 104 registers, not 96
            // (sm_86: 4 CTAs a SM, not 5; RTX 3060 Q3_K_M C2 -2.2 %), and IQ4_XS at T = 3 lost (RTX
            // 3090 C3 -3.6 %), and q4k 4096x12288 at T = 3 lost 6-13 % with no register or spill
            // change (unexplained). T = 4 only (2T = 8): T = 8, which Volta and the slow-tensor
            // parts run as two parts of eight, was not timed with it either. The gated pair keeps
            // it too: its IQ4_XS T = 4 instance took a 16 B local frame on sm_86 / 89 and lost end
            // to end (RTX 3090 IQ4_XS C4 -1.24 % with the pair transposed).
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
#pragma unroll
                for (int j = 0; j < T; ++j) {
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], o);
                }
            }
            // Every lane holds the sums: lane j stores column j (row pair: lanes T..2T-1 the
            // second).
#pragma unroll
            for (int j = 0; j < T; ++j) {
                if constexpr (gated) {
                    if (lane == j) store(p.out, row0, j, gate_of(p.out, acc[0][j]) * acc[1][j]);
                } else {
#pragma unroll
                    for (int r = 0; r < kRows; ++r) {
                        if (lane == r * T + j && row0 + r < p.rows) {
                            float v = acc[r][j];
                            if (p.out.relu2 != 0) {
                                v = fmaxf(v, 0.0F);
                                v *= v;
                            }
                            store(p.out, row0 + r, j, v);
                        }
                    }
                }
            }
        } else {
            // The 2T sums of the row pair (index r * T + j) reduced in one transposed butterfly
            // (#1044, vec_transpose_reduce): against 5 shuffles a value, and one store a lane, not
            // 2T inlined predicated ones. The first halving splits the rows: lanes with bit 16 hold
            // r 1.
            constexpr int kValues = kRows * T;
            float val[kValues];
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
#pragma unroll
                for (int j = 0; j < T; ++j) val[r * T + j] = acc[r][j];
            }
            const float mine = vec_transpose_reduce<kValues>(val, lane);
            constexpr int kH = vec_halvings(kValues);
            constexpr int kN = kValues >> kH;
            const int sel    = lane & ((32 >> kH) - 1);
            const int idx    = (lane >> (5 - kH)) * kN + sel;
            const int r      = idx / T;
            const int j      = idx % T;
            const bool owner = sel < kN;
            static_assert(!gated,
                          "the gated pair keeps the per-value butterfly (the branch above)");
            if (owner && row0 + r < p.rows) {
                float v = mine;
                if (p.out.relu2 != 0) {
                    v = fmaxf(v, 0.0F);
                    v *= v;
                }
                store(p.out, row0 + r, j, v);
            }
        }
    }
}

// The few-row int8 tile (#1162, Route{GgmlVec, 1}, sm_80+): one CTA takes 16 weight rows (a gated
// pair: 8 gate rows and the same 8 up rows) and every column of the call (at most 8) as one n8 MMA
// operand; its 4 warps split K by superblock (warp w takes w, w + 4, ...), one m16n8k32 s8 MMA a
// 32-value block (two m16n8k16 for Q6_K: MmaQ6K), then a fixed-order sum of the four warps'
// partials. A lane (g = lane / 4, t = lane % 4) feeds the MMA the values 8t .. 8t + 7 of its block
// in rows g and g + 8 (A) and the same 8 activation codes of column g (B): the MMA's K slots 4t ..
// 4t + 3 and 16 + 4t .. are that lane's two words, so one K permutation serves both operands and
// the int32 sum is the block's dot. The activation is vec's q8_1 plane (the quantizer and workspace
// of the dp4a kernel). Against the dp4a kernel it decodes each weight once for every column instead
// of once a column: an RTX 5070 Ti runs Q4_K 4096 x 12288 at T 3-8 in 0.90 of the best shipped
// route's time (issue #1162). A column's sums depend on its own activation only: its bytes do not
// change with the call's width.
constexpr int kVecMmaWarps = 4;

template <ggml_type type>
struct MmaDecoder;

// Q4_K / Q5_K (whole 144- / 176-byte superblocks): sub-block j is the low (j even) or high
// nibbles of qs bytes 32 (j / 2) .. + 31; Q5_K's fifth bit j is bit j of qh[i].
template <int Bits>
struct MmaQ45K {
    static constexpr int kStride = Bits == 5 ? 176 : 144; // a superblock's bytes
    static constexpr int kQs     = Bits == 5 ? 48 : 16;
    static constexpr bool kK16   = false; // one m16n8k32 a 32-value block

    struct Stage {
        uint4 head[2];
        uint2 qs[2][4]; // row r, pair p: qs bytes 32p + 8t .. + 7
        uint2 qh[2];    // Q5_K: qh bytes 8t .. 8t + 7
    };

    // Superblock blk0 of plane w0 (row r0) and blk1 of w1 (row r1).
    __device__ __forceinline__ static void load(Stage& s, const VecPlanes& w0, std::int64_t blk0,
                                                const VecPlanes& w1, std::int64_t blk1, bool l0,
                                                bool l1, int t) {
        const std::uint8_t* b0 = w0.codes + blk0 * kStride;
        const std::uint8_t* b1 = w1.codes + blk1 * kStride;
        s.head[0]              = l0 ? *reinterpret_cast<const uint4*>(b0) : make_uint4(0, 0, 0, 0);
        s.head[1]              = l1 ? *reinterpret_cast<const uint4*>(b1) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            s.qs[0][p] =
                l0 ? *reinterpret_cast<const uint2*>(b0 + kQs + 32 * p + 8 * t) : make_uint2(0, 0);
            s.qs[1][p] =
                l1 ? *reinterpret_cast<const uint2*>(b1 + kQs + 32 * p + 8 * t) : make_uint2(0, 0);
        }
        if constexpr (Bits == 5) {
            s.qh[0] = l0 ? *reinterpret_cast<const uint2*>(b0 + 16 + 8 * t) : make_uint2(0, 0);
            s.qh[1] = l1 ? *reinterpret_cast<const uint2*>(b1 + 16 + 8 * t) : make_uint2(0, 0);
        }
    }

    // Row r's codes of sub-block j as the lane's two A words, and its scale d * sc_j and min
    // dmin * m_j.
    __device__ __forceinline__ static void block(const Stage& s, int r, int j, unsigned& lo,
                                                 unsigned& hi, float& scale, float& min) {
        const unsigned shift = (j & 1) ? 4U : 0U;
        lo                   = (s.qs[r][j >> 1].x >> shift) & 0x0f0f0f0fU;
        hi                   = (s.qs[r][j >> 1].y >> shift) & 0x0f0f0f0fU;
        if constexpr (Bits == 5) {
            lo |= ((s.qh[r].x >> j) << 4) & 0x10101010U;
            hi |= ((s.qh[r].y >> j) << 4) & 0x10101010U;
        }
        const uint4& h = s.head[r];
        unsigned sc, m;
        if (j < 4) {
            sc = (h.y >> (8 * j)) & 63U;
            m  = (h.z >> (8 * j)) & 63U;
        } else {
            const int i = j - 4;
            sc          = ((h.w >> (8 * i)) & 0x0fU) | (((h.y >> (8 * i)) >> 6 & 3U) << 4);
            m           = ((h.w >> (8 * i + 4)) & 0x0fU) | (((h.z >> (8 * i)) >> 6 & 3U) << 4);
        }
        scale =
            __half2float(__ushort_as_half(static_cast<unsigned short>(h.x & 0xffffU))) * float(sc);
        min = __half2float(__ushort_as_half(static_cast<unsigned short>(h.x >> 16))) * float(m);
    }
};

template <>
struct MmaDecoder<GGML_TYPE_Q4_K> : MmaQ45K<4> {};

template <>
struct MmaDecoder<GGML_TYPE_Q5_K> : MmaQ45K<5> {};

// Q6_K (vec's planes: codes = ql (128) and qh (64), high = the 16 int8 scales, scales = d): a
// 16-value sub-block has its own scale, so a 32-value block is two m16n8k16 MMAs, the low one on
// its values 4t .. 4t + 3 and the high one on 16 + 4t .. 16 + 4t + 3 of the lane, with codes
// q - 32 (signed, no sum term). Block u of the superblock is half h = u / 4, quarter q = u % 4:
// ql bytes 64h + 32 (q & 1) .., nibble q / 2, and qh bytes 32h .., bits 2q (ggml's
// dequantize_row_q6_K), scale 8h + 2q for the low sub-block and 8h + 2q + 1 for the high one.
struct MmaQ6K {
    static constexpr int kCode = 192, kHigh = 16, kScale = 2;
    static constexpr bool kK16 = true;

    struct Stage {
        std::uint32_t ql[2][8]; // row r, 4h + 2c + s: ql bytes 64h + 32c + 16s + 4t .. + 3
        std::uint32_t qh[2][4]; // row r, 2h + s: qh bytes 32h + 16s + 4t .. + 3
        std::uint32_t sc[2][4]; // row r: the 16 scales
        float d[2];
    };

    __device__ __forceinline__ static void load(Stage& s, const VecPlanes& w0, std::int64_t blk0,
                                                const VecPlanes& w1, std::int64_t blk1, bool l0,
                                                bool l1, int t) {
        const VecPlanes* w[2]   = {&w0, &w1};
        const std::int64_t b[2] = {blk0, blk1};
        const bool live[2]      = {l0, l1};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const std::uint8_t* c = w[r]->codes + b[r] * kCode;
            const std::uint8_t* h = w[r]->high + b[r] * kHigh;
            // Whole words: ggml_refusal admits the route only with every plane 16-byte aligned,
            // and each offset here is a multiple of 4 (the scales of 16).
            const auto* cw = reinterpret_cast<const std::uint32_t*>(c);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                s.ql[r][i] = live[r] ? cw[8 * (i >> 1) + 4 * (i & 1) + t] : 0U;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                s.qh[r][i] = live[r] ? cw[32 + 8 * (i >> 1) + 4 * (i & 1) + t] : 0U;
            const uint4 sc = live[r] ? *reinterpret_cast<const uint4*>(h) : make_uint4(0, 0, 0, 0);
            s.sc[r][0] = sc.x, s.sc[r][1] = sc.y, s.sc[r][2] = sc.z, s.sc[r][3] = sc.w;
            s.d[r] = live[r] ? half_at(w[r]->scales + b[r] * kScale) : 0.0F;
        }
    }

    // v - 32 per byte, v in [0, 64), as the dp4a decoder writes it.
    __device__ __forceinline__ static unsigned centred(unsigned v) {
        return v ^ 0xE0E0E0E0U ^ ((v & 0x20202020U) * 6U);
    }

    // Row r's codes of block u as the lane's low and high A words, and the two sub-blocks' scales.
    __device__ __forceinline__ static void block(const Stage& s, int r, int u, unsigned& lo,
                                                 unsigned& hi, int& sc_lo, int& sc_hi) {
        const int h = u >> 2, q = u & 3;
        const int i      = 4 * h + 2 * (q & 1);
        const unsigned n = 4U * unsigned(q >> 1), e = 2U * unsigned(q);
        lo          = centred(((s.ql[r][i] >> n) & 0x0F0F0F0FU) |
                              (((s.qh[r][2 * h] >> e) << 4) & 0x30303030U));
        hi          = centred(((s.ql[r][i + 1] >> n) & 0x0F0F0F0FU) |
                              (((s.qh[r][2 * h + 1] >> e) << 4) & 0x30303030U));
        const int j = 8 * h + 2 * q;
        sc_lo       = int(static_cast<std::int8_t>(s.sc[r][j >> 2] >> (8 * (j & 3))));
        sc_hi       = int(static_cast<std::int8_t>(s.sc[r][(j + 1) >> 2] >> (8 * ((j + 1) & 3))));
    }
};

template <>
struct MmaDecoder<GGML_TYPE_Q6_K> : MmaQ6K {};

__device__ __forceinline__ void vec_mma_s8(int (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                           unsigned a3, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // The host refuses the route below sm_80 (ggml_refusal); never reached.
    (void)a0, (void)a1, (void)a2, (void)a3, (void)b0, (void)b1;
    __trap();
#endif
}

// m16n8k16 s8: a0 row g, a1 row g + 8, b0 column g, each the lane's K slots 4t .. 4t + 3.
__device__ __forceinline__ void vec_mma_s8_k16(int (&c)[4], unsigned a0, unsigned a1, unsigned b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
#else
    (void)a0, (void)a1, (void)b0;
    __trap();
#endif
}

template <ggml_type type, bool Gated>
__launch_bounds__(kVecMmaWarps * 32) __global__ void vec_mma_kernel(VecArgs p, int columns) {
    using D            = MmaDecoder<type>;
    using Stage        = typename D::Stage;
    constexpr int kOut = Gated ? 8 : 16; // output rows a CTA
    pdl::wait_for_dependencies();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    // A row r0 (first plane) and r1 (gated: the up row of r0, else 8 rows on).
    const int r0  = blockIdx.x * kOut + g;
    const int r1  = Gated ? r0 : r0 + 8;
    const bool l0 = r0 < p.rows, l1 = r1 < p.rows;
    const VecPlanes& w1 = Gated ? p.second : p.first;
    const int sbs       = p.blocks_per_row;
    const int blocks    = p.k / 32;
    const int c0 = 2 * t, c1 = 2 * t + 1;
    const bool v0 = c0 < columns, v1 = c1 < columns, vb = g < columns;
    // Column g's activation codes: 8t .. 8t + 7 of each block (one k32 MMA), or 4t .. 4t + 3 and
    // 16 + 4t .. 16 + 4t + 3 (two k16 MMAs).
    const std::int8_t* xg = p.qs + std::int64_t(g) * p.k + (D::kK16 ? 4 : 8) * t;
    float acc[4]          = {0.0F, 0.0F, 0.0F, 0.0F}; // (r0, c0), (r0, c1), (r1, c0), (r1, c1)
    const auto blk        = [&](int row, int sb) { return std::int64_t(row) * sbs + sb; };
    Stage cur, nxt;
    uint2 bc[8], bn[8];
    const auto load = [&](Stage& s, uint2(&b)[8], int sb) {
        D::load(s, p.first, blk(l0 ? r0 : 0, sb), w1, blk(l1 ? r1 : 0, sb), l0, l1, t);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::int8_t* x = xg + 256 * sb + 32 * j;
            if constexpr (D::kK16) {
                b[j] = vb ? make_uint2(*reinterpret_cast<const unsigned*>(x),
                                       *reinterpret_cast<const unsigned*>(x + 16))
                          : make_uint2(0, 0);
            } else {
                b[j] = vb ? *reinterpret_cast<const uint2*>(x) : make_uint2(0, 0);
            }
        }
    };
    int sb = warp;
    if (sb < sbs) load(cur, bc, sb);
#pragma unroll 1
    for (; sb < sbs; sb += kVecMmaWarps) {
        if (sb + kVecMmaWarps < sbs) load(nxt, bn, sb + kVecMmaWarps);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int kb = 8 * sb + j;
            const float2 f0 =
                v0 ? __half22float2(p.ds[std::int64_t(c0) * blocks + kb]) : make_float2(0.0F, 0.0F);
            const float2 f1 =
                v1 ? __half22float2(p.ds[std::int64_t(c1) * blocks + kb]) : make_float2(0.0F, 0.0F);
            if constexpr (D::kK16) {
                // Each sub-block's int32 dot times its int8 scale, summed in int32 (at most
                // 2 x 128 x 16 x 32 x 127 in magnitude, exact in a float), then d x the
                // activation's scale.
                unsigned r0lo, r0hi, r1lo, r1hi;
                int s0lo, s0hi, s1lo, s1hi;
                D::block(cur, 0, j, r0lo, r0hi, s0lo, s0hi);
                D::block(cur, 1, j, r1lo, r1hi, s1lo, s1hi);
                int lo[4] = {0, 0, 0, 0}, hi[4] = {0, 0, 0, 0};
                vec_mma_s8_k16(lo, r0lo, r1lo, bc[j].x);
                vec_mma_s8_k16(hi, r0hi, r1hi, bc[j].y);
                acc[0] = fmaf(cur.d[0] * f0.x, float(s0lo * lo[0] + s0hi * hi[0]), acc[0]);
                acc[1] = fmaf(cur.d[0] * f1.x, float(s0lo * lo[1] + s0hi * hi[1]), acc[1]);
                acc[2] = fmaf(cur.d[1] * f0.x, float(s1lo * lo[2] + s1hi * hi[2]), acc[2]);
                acc[3] = fmaf(cur.d[1] * f1.x, float(s1lo * lo[3] + s1hi * hi[3]), acc[3]);
            } else {
                unsigned a0, a2, a1, a3;
                float s0, m0, s1, m1;
                D::block(cur, 0, j, a0, a2, s0, m0);
                D::block(cur, 1, j, a1, a3, s1, m1);
                int c[4] = {0, 0, 0, 0};
                vec_mma_s8(c, a0, a1, a2, a3, bc[j].x, bc[j].y);
                acc[0] = fmaf(s0 * f0.x, float(c[0]), acc[0]);
                acc[0] = fmaf(-m0, f0.y, acc[0]);
                acc[1] = fmaf(s0 * f1.x, float(c[1]), acc[1]);
                acc[1] = fmaf(-m0, f1.y, acc[1]);
                acc[2] = fmaf(s1 * f0.x, float(c[2]), acc[2]);
                acc[2] = fmaf(-m1, f0.y, acc[2]);
                acc[3] = fmaf(s1 * f1.x, float(c[3]), acc[3]);
                acc[3] = fmaf(-m1, f1.y, acc[3]);
            }
        }
        cur = nxt;
#pragma unroll
        for (int j = 0; j < 8; ++j) bc[j] = bn[j];
    }
    // ((w0 + w1) + w2) + w3.
    __shared__ float part[kVecMmaWarps][32][4];
    if (warp > 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) part[warp][lane][i] = acc[i];
    }
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int v = 1; v < kVecMmaWarps; ++v)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] += part[v][lane][i];
#pragma unroll
    for (int i = 0; i < (Gated ? 2 : 4); ++i) {
        const int col = (i & 1) ? c1 : c0;
        if (col >= columns) continue;
        if constexpr (Gated) {
            if (l0) store(p.out, r0, col, gate_of(p.out, acc[i]) * acc[i + 2]);
        } else {
            const int row = i < 2 ? r0 : r1;
            if (row >= p.rows) continue;
            float v = acc[i];
            if (p.out.relu2 != 0) {
                v = fmaxf(v, 0.0F);
                v *= v;
            }
            store(p.out, row, col, v);
        }
    }
}

template <ggml_type type, bool Gated>
void vec_mma_grid(const VecArgs& args, int columns, cudaStream_t stream) {
    const int per_cta = Gated ? 8 : 16;
    const int blocks  = (args.rows + per_cta - 1) / per_cta;
    if (LaunchShape* described = describe_target()) {
        if (substituted_device_class() != nullptr) {
            // No committed kernel facts yet (#1162): the launch's shape, no resources.
            *described                       = LaunchShape{};
            described->threads               = kVecMmaWarps * 32;
            described->max_threads_per_block = kVecMmaWarps * 32;
            described->ptx_version           = NINFERNO_BUILD_ARCH;
            described->binary_version        = NINFERNO_BUILD_ARCH;
            described->grid_x                = static_cast<unsigned>(blocks);
            described->grid_y                = 1;
            described->grid_z                = 1;
            return;
        }
        describe_launch(*described, reinterpret_cast<const void*>(vec_mma_kernel<type, Gated>),
                        dim3(blocks), dim3(kVecMmaWarps * 32), 0);
        return;
    }
    refuse_substituted_launch();
    check(pdl::launch_dependent({dim3(blocks), dim3(kVecMmaWarps * 32), 0, stream},
                                vec_mma_kernel<type, Gated>, args, columns),
          "vector MMA product launch");
}

template <ggml_type type>
void vec_mma_launch(const VecArgs& args, int columns, cudaStream_t stream) {
    if (columns < 1 || columns > kVecMaxColumns)
        throw std::invalid_argument("ggml vec MMA: 1..8 columns");
    if (args.out.epilogue == VecOut::Epilogue::SwiGlu)
        return vec_mma_grid<type, true>(args, columns, stream);
    if (args.out.epilogue != VecOut::Epilogue::Store &&
        args.out.epilogue != VecOut::Epilogue::AddResidual) {
        throw std::invalid_argument("ggml vec MMA: Store, AddResidual or SwiGlu");
    }
    vec_mma_grid<type, false>(args, columns, stream);
}

// The CTAs one launch fills at once. A bounded twin whose compile spilled (local memory) gives 0:
// it is never worth its extra traffic, whatever it fits. A real device asks its own kernel (this
// build's registers, whatever tree or nvcc made it). A substituted board (ScopedDeviceClass) has
// no kernel of its own here: its instance's committed kernel facts (the "vec" section, #763) for
// its code (kernel_facts::sass_arch), registers and static shared memory through the occupancy
// calculator, times its SMs; never the host GPU's kernels. That holds for every substitute, the
// host's own class included (#1033): a substituted answer is the release's, the same on any host
// and any build, which the selector goldens rely on. A board whose code has no release snapshot
// (sass_arch nullopt: the driver would compile the PTX, as the GB10's sm_121) is priced by threads
// alone, as the register MMA is there: the CTAs its thread and block limits allow, no bounded
// twin. A snapshot without the instance's row refuses by name.
template <ggml_type type, int T, int KS, bool Gated, bool Bounded>
int vec_slots() {
    // Per device: a process may drive GPUs of different chips.
    static PerDevice<int> per_device;
    return per_device.get([](int) {
        if (substituted_device_class() != nullptr) {
            namespace kf                          = ninferno::ops::detail::generic::kernel_facts;
            const std::optional<std::string> arch = kf::sass_arch();
            const kf::VecKernelKey key{static_cast<int>(type), T, KS, Gated, Bounded};
            if (!arch) {
                if (Bounded) return 0;
                const DeviceClass& d = device_class();
                const int by_threads =
                    d.max_threads_per_sm > 0 ? d.max_threads_per_sm / (kVecWarps * 32) : 1;
                const int per_sm = d.max_blocks_per_sm > 0
                                       ? std::min(by_threads, d.max_blocks_per_sm)
                                       : by_threads;
                return std::max(per_sm, 1) * device_facts().sm_count;
            }
            if (const std::optional<kf::KernelFacts> facts = kf::vec_facts(*arch, key)) {
                if (Bounded && facts->spill_bytes() != 0) return 0;
                const int per_sm = resident_ctas_per_sm(
                    device_class(),
                    KernelShape{facts->registers, facts->static_shared_bytes, kVecWarps * 32, 0});
                return std::max(per_sm, 1) * device_facts().sm_count;
            }
            throw std::runtime_error("ggml vector kernel: no kernel facts of vec_kernel<" +
                                     std::to_string(key.type) + ", " + std::to_string(T) + ", " +
                                     std::to_string(KS) + ", " + std::to_string(Gated) + ", " +
                                     std::to_string(Bounded) + "> for the substituted board (sm_" +
                                     *arch + ")");
        }
        const auto kernel = vec_kernel<type, T, KS, Gated, Bounded>;
        if constexpr (Bounded) {
            cudaFuncAttributes attributes{};
            check(cudaFuncGetAttributes(&attributes, kernel), "vector kernel attributes");
            if (attributes.localSizeBytes != 0) return 0;
        }
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kVecWarps * 32, 0),
              "vector occupancy");
        return std::max(blocks, 1) * device_facts().sm_count;
    });
}

template <ggml_type type, int T, int KS, bool Gated, bool Bounded>
void vec_launch_grid(const VecArgs& args, int groups, int slots, cudaStream_t stream) {
    const int blocks = std::min(groups, slots);
    if (LaunchShape* described = describe_target()) {
        if (substituted_device_class() != nullptr) {
            // A substituted board is described as vec_slots prices it (#1033): its instance's
            // committed facts, never this build's kernel on the host GPU (which, with no device,
            // throws). The versions are the build's architecture, as the register MMA's
            // describe_from_facts. With no release snapshot for its code the resources stay 0:
            // the threads-only price has no registers to show.
            namespace kf = ninferno::ops::detail::generic::kernel_facts;
            *described   = LaunchShape{};
            if (const std::optional<std::string> arch = kf::sass_arch()) {
                if (const std::optional<kf::KernelFacts> facts = kf::vec_facts(
                        *arch, kf::VecKernelKey{static_cast<int>(type), T, KS, Gated, Bounded})) {
                    described->registers   = facts->registers;
                    described->local_bytes = facts->spill_bytes();
                    described->static_smem = facts->static_shared_bytes;
                }
            }
            described->threads               = kVecWarps * 32;
            described->max_threads_per_block = kVecWarps * 32;
            described->ptx_version           = NINFERNO_BUILD_ARCH;
            described->binary_version        = NINFERNO_BUILD_ARCH;
            described->grid_x                = static_cast<unsigned>(blocks);
            described->grid_y                = 1;
            described->grid_z                = 1;
            return;
        }
        describe_launch(*described,
                        reinterpret_cast<const void*>(vec_kernel<type, T, KS, Gated, Bounded>),
                        dim3(blocks), dim3(kVecWarps * 32), 0);
        return;
    }
    refuse_substituted_launch();
    check(pdl::launch_dependent({dim3(blocks), dim3(kVecWarps * 32), 0, stream},
                                vec_kernel<type, T, KS, Gated, Bounded>, args),
          "vector product launch");
}

template <ggml_type type, int T, int KS, bool Gated>
void vec_launch_ks(const VecArgs& args, cudaStream_t stream) {
    const int per_block = kVecWarps * (Gated ? 1 : 2);
    const int groups    = (args.rows + per_block - 1) / per_block;
    const int slots     = vec_slots<type, T, KS, Gated, false>();
    if constexpr (kVecHasBoundedTwin(T, KS, Gated)) {
        const int twin_slots = vec_slots<type, T, KS, Gated, true>();
        const auto rounds    = [&](int n) { return (groups + n - 1) / n; };
        // Conservative on purpose: see kVecHasBoundedTwin.
        const bool take = g_vec_instance == VecInstance::Twin ||
                          (g_vec_instance == VecInstance::Auto && twin_slots > 0 &&
                           rounds(twin_slots) < rounds(slots));
        if (take) {
            // Forced onto a twin that spilled: its grid is the plain one's.
            return vec_launch_grid<type, T, KS, Gated, true>(
                args, groups, twin_slots > 0 ? twin_slots : slots, stream);
        }
    }
    vec_launch_grid<type, T, KS, Gated, false>(args, groups, slots, stream);
}

// The K of the 27B and 35B text matrices are unrolled; any other K runs the loop.
// A gated pair reads the block input (K = the hidden size): 5120 unrolled, any other K the loop.
template <ggml_type type, int T>
void vec_launch_k(const VecArgs& args, cudaStream_t stream) {
    if (args.out.epilogue == VecOut::Epilogue::SwiGlu) {
        return args.k == 5120 ? vec_launch_ks<type, T, 5120, true>(args, stream)
                              : vec_launch_ks<type, T, 0, true>(args, stream);
    }
    switch (args.k) {
    case 5120: return vec_launch_ks<type, T, 5120, false>(args, stream);
    case 6144: return vec_launch_ks<type, T, 6144, false>(args, stream);
    case 17408: return vec_launch_ks<type, T, 17408, false>(args, stream);
    default: return vec_launch_ks<type, T, 0, false>(args, stream);
    }
}

template <ggml_type type>
void vec_launch(const VecArgs& args, int columns, cudaStream_t stream) {
    switch (columns) {
    case 1: return vec_launch_k<type, 1>(args, stream);
    case 2: return vec_launch_k<type, 2>(args, stream);
    case 3: return vec_launch_k<type, 3>(args, stream);
    case 4: return vec_launch_k<type, 4>(args, stream);
    case 5: return vec_launch_k<type, 5>(args, stream);
    case 6: return vec_launch_k<type, 6>(args, stream);
    case 7: return vec_launch_k<type, 7>(args, stream);
    case 8: return vec_launch_k<type, 8>(args, stream);
    default: throw std::invalid_argument("ggml vec: 1..8 columns");
    }
}

// One warp per 32 values of a column: amax by shuffles, ggml's q8_1 rounding, and the block's sum
// of its codes (d * sum q, as ggml_mmq.cu's quantizer: the offset term of a format with one must see
// the activation the dot sees), held to binary16's range.
__global__ void vec_quantize_kernel(const __nv_bfloat16* __restrict__ x, int ldx, int k,
                                    std::int8_t* __restrict__ qs, half2* __restrict__ ds) {
    const int column = blockIdx.y;
    const int slice  = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane   = threadIdx.x & 31;
    if (slice * 32 >= k) return;
    // A programmatic dependent of the kernel before it (sm_90+): only the launch overlaps.
    pdl::wait_for_dependencies();
    const std::int64_t at = std::int64_t(column) * k + 32 * slice + lane;
    const float v = __bfloat162float(x[std::int64_t(column) * ldx + 32 * slice + lane]);
    float amax    = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
    }
    // An all-zero group stores zero codes and a zero scale (ggml divides by amax).
    const float d_inv  = amax > 0.0f ? 127.0f / amax : 0.0f;
    const std::int8_t q = static_cast<std::int8_t>(roundf(v * d_inv));
    qs[at]             = q;
    float sum          = static_cast<float>(q);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
    }
    constexpr float kHalfMax = 65504.0f;
    const float d            = amax / 127.0f;
    if (lane == 0) {
        ds[std::int64_t(column) * (k / 32) + slice] =
            make_half2(fminf(d, kHalfMax), fminf(fmaxf(d * sum, -kHalfMax), kHalfMax));
    }
}

} // namespace
} // namespace detail


void vec_quantize_activation(const __nv_bfloat16* x, int ldx, int k, int columns, void* out,
                             cudaStream_t stream) {
    if (k <= 0 || k % 32 != 0 || columns <= 0 || columns > kVecMaxColumns || ldx < k) {
        throw std::invalid_argument("ggml vec activation: K whole 32-value groups, 1..8 columns");
    }
    constexpr int kWarps = 8;
    const dim3 blocks((k / 32 + kWarps - 1) / kWarps, columns);
    auto* qs = static_cast<std::int8_t*>(out);
    auto* ds = reinterpret_cast<half2*>(static_cast<char*>(out) + vec_scales_offset(k, columns));
    detail::check(pdl::launch_dependent({blocks, dim3(kWarps * 32), 0, stream},
                                        detail::vec_quantize_kernel, x, ldx, k, qs, ds),
                  "vector activation launch");
}

ScopedVecInstance::ScopedVecInstance(VecInstance instance) noexcept
    : previous_(detail::g_vec_instance) {
    detail::g_vec_instance = instance;
}

ScopedVecInstance::~ScopedVecInstance() { detail::g_vec_instance = previous_; }

void vec_product(Type type, const Planes& weight, const Planes& up, int blocks_per_row, int rows,
                 int k, const void* activation, int columns, const VecOut& out,
                 cudaStream_t stream) {
    const auto planes = [](const Planes& p) {
        return detail::VecPlanes{static_cast<const std::uint8_t*>(p.codes),
                                 static_cast<const std::uint8_t*>(p.high),
                                 static_cast<const std::uint8_t*>(p.scales)};
    };
    detail::VecArgs args{};
    args.first          = planes(weight);
    args.second         = planes(up);
    args.blocks_per_row = blocks_per_row;
    args.rows           = rows;
    args.k              = k;
    args.qs             = static_cast<const std::int8_t*>(activation);
    args.ds             = reinterpret_cast<const half2*>(static_cast<const char*>(activation) +
                                                         vec_scales_offset(k, columns));
    args.out            = out;
    switch (type) {
    case Type::IQ4_XS: return detail::vec_launch<GGML_TYPE_IQ4_XS>(args, columns, stream);
    case Type::Q4_K: return detail::vec_launch<GGML_TYPE_Q4_K>(args, columns, stream);
    case Type::Q5_K: return detail::vec_launch<GGML_TYPE_Q5_K>(args, columns, stream);
    case Type::Q6_K: return detail::vec_launch<GGML_TYPE_Q6_K>(args, columns, stream);
    case Type::Q2_K: return detail::vec_launch<GGML_TYPE_Q2_K>(args, columns, stream);
    case Type::IQ3_XXS: return detail::vec_launch<GGML_TYPE_IQ3_XXS>(args, columns, stream);
    case Type::IQ3_S: return detail::vec_launch<GGML_TYPE_IQ3_S>(args, columns, stream);
    case Type::Q3_K: return detail::vec_launch<GGML_TYPE_Q3_K>(args, columns, stream);
    case Type::IQ2_XXS: return detail::vec_launch<GGML_TYPE_IQ2_XXS>(args, columns, stream);
    case Type::IQ2_XS: return detail::vec_launch<GGML_TYPE_IQ2_XS>(args, columns, stream);
    case Type::IQ2_S: return detail::vec_launch<GGML_TYPE_IQ2_S>(args, columns, stream);
    case Type::IQ1_S: return detail::vec_launch<GGML_TYPE_IQ1_S>(args, columns, stream);
    case Type::Q4_0: return detail::vec_launch<GGML_TYPE_Q4_0>(args, columns, stream);
    case Type::Q4_1: return detail::vec_launch<GGML_TYPE_Q4_1>(args, columns, stream);
    case Type::Q5_0: return detail::vec_launch<GGML_TYPE_Q5_0>(args, columns, stream);
    case Type::Q5_1: return detail::vec_launch<GGML_TYPE_Q5_1>(args, columns, stream);
    case Type::IQ4_NL: return detail::vec_launch<GGML_TYPE_IQ4_NL>(args, columns, stream);
    case Type::IQ1_M: return detail::vec_launch<GGML_TYPE_IQ1_M>(args, columns, stream);
    default: break;
    }
    throw std::invalid_argument("ggml vec: format without a vector decoder");
}

void vec_mma_product(Type type, const Planes& weight, const Planes& up, int blocks_per_row,
                     int rows, int k, const void* activation, int columns, const VecOut& out,
                     cudaStream_t stream) {
    if (k % 256 != 0) throw std::invalid_argument("ggml vec MMA: K whole 256-value superblocks");
    detail::VecArgs args{};
    // vec's planes (whole blocks in codes for Q4_K / Q5_K; codes, scales and d for Q6_K).
    const auto planes = [](const Planes& p) {
        return detail::VecPlanes{static_cast<const std::uint8_t*>(p.codes),
                                 static_cast<const std::uint8_t*>(p.high),
                                 static_cast<const std::uint8_t*>(p.scales)};
    };
    args.first          = planes(weight);
    args.second         = planes(up);
    args.blocks_per_row = blocks_per_row;
    args.rows           = rows;
    args.k              = k;
    args.qs             = static_cast<const std::int8_t*>(activation);
    args.ds             = reinterpret_cast<const half2*>(static_cast<const char*>(activation) +
                                                         vec_scales_offset(k, columns));
    args.out            = out;
    switch (type) {
    case Type::Q4_K:
        return detail::vec_mma_launch<GGML_TYPE_Q4_K>(args, columns, stream);
    case Type::Q5_K:
        return detail::vec_mma_launch<GGML_TYPE_Q5_K>(args, columns, stream);
    case Type::Q6_K:
        return detail::vec_mma_launch<GGML_TYPE_Q6_K>(args, columns, stream);
    default:
        break;
    }
    throw std::invalid_argument("ggml vec MMA: format without an MMA decoder");
}

} // namespace ninferno::ops::ggml_mmq
