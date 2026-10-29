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

#include "core/pdl.cuh"

#include <algorithm>

namespace ninferno::ops::ggml_mmq {
namespace detail {
namespace {

constexpr int kVecWarps = 4;

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

// A binary16 pair from its 32-bit word (low half first, as in memory), and one binary16 from its
// low 16 bits.
__device__ __forceinline__ float2 half2_bits(std::uint32_t v) {
    return __half22float2(__halves2half2(__ushort_as_half(static_cast<unsigned short>(v & 0xFFFF)),
                                         __ushort_as_half(static_cast<unsigned short>(v >> 16))));
}
__device__ __forceinline__ float half_bits(std::uint32_t v) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(v & 0xFFFF)));
}

// The eight words of two 16-byte loads, in memory order.
__device__ __forceinline__ void unpack8(const uint4& a, const uint4& b, std::uint32_t (&v)[8]) {
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

// Byte i of four words.
__device__ __forceinline__ std::uint32_t byte_of(const uint4& h, int i) {
    const std::uint32_t v[4] = {h.x, h.y, h.z, h.w};
    return (v[i / 4] >> (8 * (i % 4))) & 0xFF;
}

// The 6-bit scale and min of sub-block j of a Q4_K / Q5_K block (ggml's get_scale_min_k4), from
// the block's first 16 bytes: d, dmin, then the 12 scale bytes.
__device__ __forceinline__ void scale_min_k4(const uint4& head, int j, int& sc, int& m) {
    const auto s = [&](int i) { return int(byte_of(head, 4 + i)); };
    if (j < 4) {
        sc = s(j) & 63;
        m  = s(j + 4) & 63;
    } else {
        sc = (s(j + 4) & 0xF) | ((s(j - 4) >> 6) << 4);
        m  = (s(j + 4) >> 4) | ((s(j) >> 6) << 4);
    }
}

// Each format's slice in two halves: load() reads slice u of block `blk` of a row (u in 0 ..
// kBlockValues / 32 - 1) as raw words into registers; convert() turns them into the slice's eight
// int8x4 words in value order and the factors of its format. dot(): the slice's contribution to one
// column, d and sum the column's slice scale and the sum of its unquantized values. The kernel
// issues the next step's load() before this step's convert() and products, so a warp waits on
// memory once a step whatever the compiler's schedule; convert(load()) is the arithmetic of one
// decode on the same bytes.
template <ggml_type type>
struct Decoder;

template <>
struct Decoder<GGML_TYPE_IQ4_XS> {
    // codes: qs (128); high: scales_l (4); scales: d, scales_h (4).
    static constexpr int kCode = 128, kHigh = 4, kScale = 4, kTableWords = 0, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return nullptr; }
    struct Raw {
        std::uint32_t sc; // d, scales_h
        std::uint32_t hb; // this slice's scales_l byte
        uint2 lo, hi;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        const std::uint8_t* qs = p.codes + blk * kCode + 16 * u;
        return Raw{*reinterpret_cast<const std::uint32_t*>(p.scales + blk * kScale),
                   p.high[blk * kHigh + u / 2], *reinterpret_cast<const uint2*>(qs),
                   *reinterpret_cast<const uint2*>(qs + 8)};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int u, const std::uint32_t*,
                                                   Slice& s) {
        const std::uint32_t sh = w.sc >> 16;
        const int ls = ((w.hb >> (4 * (u & 1))) & 0xF) | (((sh >> (2 * u)) & 3) << 4);
        s.f[0]       = half_bits(w.sc) * float(ls - 32);
        const std::uint32_t q[4] = {w.lo.x, w.lo.y, w.hi.x, w.hi.y};
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
    struct Raw {
        uint4 head; // d, dmin, scales
        uint4 q0, q1;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        const std::uint8_t* b = p.codes + blk * kCode;
        const uint4* qs       = reinterpret_cast<const uint4*>(b + 16 + 32 * (u / 2));
        return Raw{*reinterpret_cast<const uint4*>(b), qs[0], qs[1]};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int u, const std::uint32_t*,
                                                   Slice& s) {
        const float2 dm = half2_bits(w.head.x);
        int sc, m;
        scale_min_k4(w.head, u, sc, m);
        s.f[0]                   = dm.x * float(sc);
        s.f[1]                   = dm.y * float(m);
        const int shift          = 4 * (u & 1);
        std::uint32_t v[8];
        unpack8(w.q0, w.q1, v);
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
    struct Raw {
        uint4 head; // d, dmin, scales
        uint4 h0, h1, l0, l1;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        const std::uint8_t* b = p.codes + blk * kCode;
        const uint4* qh       = reinterpret_cast<const uint4*>(b + 16);
        const uint4* ql       = reinterpret_cast<const uint4*>(b + 48 + 32 * (u / 2));
        return Raw{*reinterpret_cast<const uint4*>(b), qh[0], qh[1], ql[0], ql[1]};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int u, const std::uint32_t*,
                                                   Slice& s) {
        const float2 dm = half2_bits(w.head.x);
        int sc, m;
        scale_min_k4(w.head, u, sc, m);
        s.f[0]                    = dm.x * float(sc);
        s.f[1]                    = dm.y * float(m);
        std::uint32_t hv[8];
        unpack8(w.h0, w.h1, hv);
        std::uint32_t lv[8];
        unpack8(w.l0, w.l1, lv);
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
    struct Raw {
        std::uint32_t dm; // d, dmin
        std::uint32_t sc; // this slice's two scale bytes
        uint4 a0, a1;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        const int h = u / 4, j = u % 4;
        const uint4* qs = reinterpret_cast<const uint4*>(p.codes + blk * kCode + 32 * h);
        return Raw{*reinterpret_cast<const std::uint32_t*>(p.scales + blk * kScale),
                   *reinterpret_cast<const std::uint16_t*>(p.high + blk * kHigh + 8 * h + 2 * j),
                   qs[0], qs[1]};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int u, const std::uint32_t*,
                                                   Slice& s) {
        const int j            = u % 4;
        const float2 dm        = half2_bits(w.dm);
        const std::uint32_t lo = w.sc & 0xFF, hi = w.sc >> 8;
        s.f[0]                 = dm.x * float(lo & 0xF);
        s.f[1]                 = dm.x * float(hi & 0xF);
        s.f[2]                 = dm.y * float(lo >> 4);
        s.f[3]                 = dm.y * float(hi >> 4);
        std::uint32_t v[8];
        unpack8(w.a0, w.a1, v);
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
    // codes: ql (128), qh (64); high: scales (16); scales: d (2). The code plane's 192-byte blocks
    // keep ql and qh 16-byte aligned.
    static constexpr int kCode = 192, kHigh = 16, kScale = 2, kTableWords = 0, kBlockValues = 256;
    __device__ static const std::uint32_t* table() { return nullptr; }
    struct Raw {
        std::uint32_t d;  // binary16
        std::uint32_t sc; // this slice's two int8 scales
        uint4 l0, l1, h0, h1;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        const int h = u / 4, q = u % 4;
        const std::uint8_t* b = p.codes + blk * kCode;
        const uint4* ql       = reinterpret_cast<const uint4*>(b + 64 * h + 32 * (q & 1));
        const uint4* qh       = reinterpret_cast<const uint4*>(b + 128 + 32 * h);
        return Raw{*reinterpret_cast<const std::uint16_t*>(p.scales + blk * kScale),
                   *reinterpret_cast<const std::uint16_t*>(p.high + blk * kHigh + 8 * h + 2 * q),
                   ql[0], ql[1], qh[0], qh[1]};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int u, const std::uint32_t*,
                                                   Slice& s) {
        const int q   = u % 4;
        const float d = half_bits(w.d);
        s.f[0]        = d * float(static_cast<std::int8_t>(w.sc & 0xFF));
        s.f[1]        = d * float(static_cast<std::int8_t>(w.sc >> 8));
        std::uint32_t lv[8];
        unpack8(w.l0, w.l1, lv);
        std::uint32_t hv[8];
        unpack8(w.h0, w.h1, hv);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t v = ((lv[i] >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                    (((hv[i] >> (2 * q)) << 4) & 0x30303030u);
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
    struct Raw {
        uint2 qs;
        std::uint32_t aux;
        std::uint32_t d;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        return Raw{*reinterpret_cast<const uint2*>(p.codes + blk * kCode + 8 * u),
                   *reinterpret_cast<const std::uint32_t*>(p.high + blk * kHigh + 4 * u),
                   *reinterpret_cast<const std::uint16_t*>(p.scales + blk * kScale)};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int, const std::uint32_t* grid,
                                                   Slice& s) {
        const std::uint32_t q0 = w.qs.x, q1 = w.qs.y, aux = w.aux;
        s.f[0]                 = half_bits(w.d) * (0.5f + float(aux >> 28)) * 0.5f;
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
    struct Raw {
        uint2 q;
        std::uint32_t qh, sg, sc, d;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        // The code plane's 104-byte blocks keep qs 8-byte and the signs 4-byte aligned.
        const std::uint8_t* b = p.codes + blk * kCode;
        return Raw{*reinterpret_cast<const uint2*>(b + 8 * u), b[64 + u],
                   *reinterpret_cast<const std::uint32_t*>(b + 72 + 4 * u),
                   p.high[blk * kHigh + u / 2],
                   *reinterpret_cast<const std::uint16_t*>(p.scales + blk * kScale)};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int u, const std::uint32_t* grid,
                                                   Slice& s) {
        const int sc = (w.sc >> (4 * (u & 1))) & 0xF;
        s.f[0]       = half_bits(w.d) * float(1 + 2 * sc);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const std::uint32_t idx =
                (((e < 4 ? w.q.x : w.q.y) >> (8 * (e & 3))) & 0xFF) | ((w.qh << (8 - e)) & 0x100);
            s.w[e] = negate_bytes(grid[idx], w.sg >> (4 * e));
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
    struct Raw {
        std::uint32_t d;
        uint4 sc; // the 12 scale bytes (w unused)
        uint4 m0, m1, q0, q1;
    };
    __device__ __forceinline__ static int scale(const uint4& sc, int i) {
        return int(((byte_of(sc, i % 8) >> (4 * (i / 8))) & 0xF) |
                   (((byte_of(sc, 8 + i % 4) >> (2 * (i / 4))) & 3) << 4)) -
               32;
    }
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        const int h           = u / 4;
        const std::uint8_t* b = p.codes + blk * kCode;
        const uint4* hm       = reinterpret_cast<const uint4*>(b);
        const uint4* qs       = reinterpret_cast<const uint4*>(b + 32 + 32 * h);
        // The scale plane's 12-byte blocks keep 4-byte words aligned.
        const auto* sw = reinterpret_cast<const std::uint32_t*>(p.high + blk * kHigh);
        return Raw{*reinterpret_cast<const std::uint16_t*>(p.scales + blk * kScale),
                   uint4{sw[0], sw[1], sw[2], 0}, hm[0], hm[1], qs[0], qs[1]};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int u, const std::uint32_t*,
                                                   Slice& s) {
        const int h = u / 4, j = u % 4;
        const float d             = half_bits(w.d);
        s.f[0]                    = d * float(scale(w.sc, 8 * h + 2 * j));
        s.f[1]                    = d * float(scale(w.sc, 8 * h + 2 * j + 1));
        std::uint32_t mv[8];
        unpack8(w.m0, w.m1, mv);
        std::uint32_t qv[8];
        unpack8(w.q0, w.q1, qv);
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
    struct Raw {
        uint2 q;
        std::uint32_t d;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        return Raw{*reinterpret_cast<const uint2*>(p.codes + blk * kCode + 8 * u),
                   *reinterpret_cast<const std::uint16_t*>(p.scales + blk * kScale)};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int, const std::uint32_t* grid,
                                                   Slice& s) {
        const uint2 q = w.q;
        s.f[0]        = half_bits(w.d) * (0.5f + float(q.y >> 28)) * 0.25f;
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
    struct Raw {
        uint2 q;
        std::uint32_t sc, d;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        return Raw{*reinterpret_cast<const uint2*>(p.codes + blk * kCode + 8 * u),
                   p.high[blk * kHigh + u],
                   *reinterpret_cast<const std::uint16_t*>(p.scales + blk * kScale)};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int, const std::uint32_t* grid,
                                                   Slice& s) {
        const float d = half_bits(w.d);
        s.f[0]        = d * (0.5f + float(w.sc & 0xF)) * 0.25f;
        s.f[1]        = d * (0.5f + float(w.sc >> 4)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t v  = ((l < 2 ? w.q.x : w.q.y) >> (16 * (l & 1))) & 0xFFFF;
            const std::uint32_t at = 2 * (v & 0x1FF);
            const std::uint32_t sg = ksigns(v >> 9);
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
    struct Raw {
        std::uint32_t qs, sg, qh, sc, d;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        const std::uint8_t* b = p.codes + blk * kCode;
        return Raw{*reinterpret_cast<const std::uint32_t*>(b + 4 * u),
                   *reinterpret_cast<const std::uint32_t*>(b + 32 + 4 * u), b[64 + u],
                   p.high[blk * kHigh + u],
                   *reinterpret_cast<const std::uint16_t*>(p.scales + blk * kScale)};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int, const std::uint32_t* grid,
                                                   Slice& s) {
        const float d = half_bits(w.d);
        s.f[0]        = d * (0.5f + float(w.sc & 0xF)) * 0.25f;
        s.f[1]        = d * (0.5f + float(w.sc >> 4)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t at =
                2 * (((w.qs >> (8 * l)) & 0xFF) | ((w.qh << (8 - 2 * l)) & 0x300));
            s.w[2 * l]     = negate_bytes(grid[at], w.sg >> (8 * l));
            s.w[2 * l + 1] = negate_bytes(grid[at + 1], w.sg >> (8 * l + 4));
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
    struct Raw {
        std::uint32_t qs, qh, d;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        return Raw{*reinterpret_cast<const std::uint32_t*>(p.codes + blk * kCode + 4 * u),
                   *reinterpret_cast<const std::uint16_t*>(p.high + blk * kHigh + 2 * u),
                   *reinterpret_cast<const std::uint16_t*>(p.scales + blk * kScale)};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int, const std::uint32_t* grid,
                                                   Slice& s) {
        const std::uint32_t qs = w.qs, qh = w.qh;
        s.f[0] = half_bits(w.d) * float(((qh >> 11) & 0x0E) + 1);
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

// A 32-value block's raw words: its 16 code bytes, its high word (Q5_0 / Q5_1) and its scale word
// (d, or d and m).
struct Raw32 {
    uint4 q;
    std::uint32_t qh, sc;
};

template <int kHighBytes, int kScaleBytes>
__device__ __forceinline__ Raw32 load32(const VecPlanes& p, std::int64_t blk) {
    Raw32 w;
    w.q  = *reinterpret_cast<const uint4*>(p.codes + blk * 16);
    w.qh = kHighBytes > 0 ? *reinterpret_cast<const std::uint32_t*>(p.high + blk * kHighBytes) : 0;
    w.sc = kScaleBytes == 4 ? *reinterpret_cast<const std::uint32_t*>(p.scales + blk * kScaleBytes)
                            : *reinterpret_cast<const std::uint16_t*>(p.scales + blk * kScaleBytes);
    return w;
}

template <>
struct Decoder<GGML_TYPE_Q4_0> {
    // codes: qs (16: value j low nibble of byte j, value j + 16 its high nibble); scales: d.
    static constexpr int kCode = 16, kHigh = 0, kScale = 2, kTableWords = 0, kBlockValues = 32;
    __device__ static const std::uint32_t* table() { return nullptr; }
    using Raw = Raw32;
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int) {
        return load32<kHigh, kScale>(p, blk);
    }
    __device__ __forceinline__ static void convert(const Raw& w, int, const std::uint32_t*,
                                                   Slice& s) {
        const std::uint32_t v[4] = {w.q.x, w.q.y, w.q.z, w.q.w};
        s.f[0]                   = half_bits(w.sc);
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
    using Raw = Raw32;
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int) {
        return load32<kHigh, kScale>(p, blk);
    }
    __device__ __forceinline__ static void convert(const Raw& w, int, const std::uint32_t*,
                                                   Slice& s) {
        const std::uint32_t v[4] = {w.q.x, w.q.y, w.q.z, w.q.w};
        const float2 dm          = half2_bits(w.sc);
        s.f[0]                   = dm.x;
        s.f[1]                   = dm.y;
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
    using Raw = Raw32;
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int) {
        return load32<kHigh, kScale>(p, blk);
    }
    __device__ __forceinline__ static void convert(const Raw& w, int, const std::uint32_t*,
                                                   Slice& s) {
        const std::uint32_t v[4] = {w.q.x, w.q.y, w.q.z, w.q.w};
        const std::uint32_t qh   = w.qh;
        s.f[0]                   = half_bits(w.sc);
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
    using Raw = Raw32;
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int) {
        return load32<kHigh, kScale>(p, blk);
    }
    __device__ __forceinline__ static void convert(const Raw& w, int, const std::uint32_t*,
                                                   Slice& s) {
        const std::uint32_t v[4] = {w.q.x, w.q.y, w.q.z, w.q.w};
        const std::uint32_t qh   = w.qh;
        const float2 dm          = half2_bits(w.sc);
        s.f[0]                   = dm.x;
        s.f[1]                   = dm.y;
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
    using Raw = Raw32;
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int) {
        return load32<kHigh, kScale>(p, blk);
    }
    __device__ __forceinline__ static void convert(const Raw& w, int, const std::uint32_t*,
                                                   Slice& s) {
        const std::uint32_t v[4] = {w.q.x, w.q.y, w.q.z, w.q.w};
        s.f[0]                   = half_bits(w.sc);
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
    struct Raw {
        uint2 sc; // the four 16-bit scale words
        std::uint32_t qs, qh;
    };
    __device__ __forceinline__ static Raw load(const VecPlanes& p, std::int64_t blk, int u) {
        return Raw{*reinterpret_cast<const uint2*>(p.scales + blk * kScale),
                   *reinterpret_cast<const std::uint32_t*>(p.codes + blk * kCode + 4 * u),
                   *reinterpret_cast<const std::uint16_t*>(p.high + blk * kHigh + 2 * u)};
    }
    __device__ __forceinline__ static void convert(const Raw& w, int u, const std::uint32_t* grid,
                                                   Slice& s) {
        const std::uint32_t sc[4] = {w.sc.x & 0xFFFF, w.sc.x >> 16, w.sc.y & 0xFFFF, w.sc.y >> 16};
        const auto bits = static_cast<unsigned short>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                                                      ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        const float d          = __half2float(__ushort_as_half(bits));
        const std::uint32_t sb = sc[u / 2] >> (6 * (u % 2));
        s.f[0]                 = d * float(2 * (sb & 7) + 1);
        s.f[1]                 = d * float(2 * ((sb >> 3) & 7) + 1);
        const std::uint32_t qs = w.qs, qh = w.qh;
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

// KS: the static K the slice loop unrolls over, or 0 for the runtime K. Gated: the SwiGlu pair (gate
// and up rows, one row per warp), a separate instance as ninfer-all's Fused one: the plain product
// keeps no gated branches or second-plane state in its loop.
template <ggml_type type, int T, int KS, bool Gated>
__launch_bounds__(kVecWarps * 32) __global__ void vec_kernel(VecArgs p) {
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
        // One step: a lane's slice of each row (raw words) and of each column. The next step's
        // loads are issued before this step's convert() and products, so a warp waits on memory
        // once a step whatever the compiler schedules (RTX 5070 Ti, Q4_K 4096x4096 T = 2..8: one
        // to five loads before the first product by instance, 14.6..26.1 us).
        struct Step {
            typename D::Raw w[kRows];
            int4 a0[T], a1[T];
            half2 ds[T];
        };
        const auto fetch = [&](int it, Step& st) {
            const int s = lane + 32 * it;
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                st.w[r] = D::load(r == 0 ? p.first : second,
                                  base[r] + std::int64_t(32 / kPerBlock) * it, sub);
            }
#pragma unroll
            for (int j = 0; j < T; ++j) {
                st.ds[j]      = p.ds[std::int64_t(j) * slices + s];
                const auto* q = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * k + 32 * s);
                st.a0[j]      = q[0];
                st.a1[j]      = q[1];
            }
        };
        const auto use = [&](const Step& st) {
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(st.ds[j]);
                d[j]            = ds.x;
                sum[j]          = ds.y;
                const int4 v0 = st.a0[j], v1 = st.a1[j];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                Slice sl;
                D::convert(st.w[r], sub, table, sl);
#pragma unroll
                for (int j = 0; j < T; ++j) acc[r][j] += D::dot(sl, a[j], d[j], sum[j]);
            }
        };
        if constexpr (KS != 0) {
            static_assert(KS % 1024 == 0);
            Step cur;
            fetch(0, cur);
#pragma unroll
            for (int it = 0; it < KS / 1024; ++it) {
                Step next;
                if (it + 1 < KS / 1024) fetch(it + 1, next);
                use(cur);
                cur = next;
            }
        } else {
            // This lane's steps: lane + 32 it < slices.
            const int steps = lane < slices ? (slices - 1 - lane) / 32 + 1 : 0;
            if (steps > 0) {
                Step cur;
                fetch(0, cur);
                for (int it = 0; it < steps; ++it) {
                    Step next;
                    if (it + 1 < steps) fetch(it + 1, next);
                    use(cur);
                    cur = next;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], o);
            }
        }
        // Every lane holds the sums: lane j stores column j (row pair: lanes T..2T-1 the second).
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
    }
}

template <ggml_type type, int T, int KS, bool Gated>
void vec_launch_ks(const VecArgs& args, cudaStream_t stream) {
    // Per device: a process may drive GPUs of different chips.
    static PerDevice<int> per_device;
    const int resident = per_device.get([](int) {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, vec_kernel<type, T, KS, Gated>,
                                                            kVecWarps * 32, 0),
              "vector occupancy");
        return std::max(blocks, 1);
    });
    const int per_block = kVecWarps * (Gated ? 1 : 2);
    const int groups    = (args.rows + per_block - 1) / per_block;
    const int blocks    = std::min(groups, resident * device_facts().sm_count);
    check(pdl::launch_dependent({dim3(blocks), dim3(kVecWarps * 32), 0, stream},
                                vec_kernel<type, T, KS, Gated>, args),
          "vector product launch");
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

bool vec_supported(Type type) {
    switch (type) {
    case Type::IQ4_XS:
    case Type::Q4_K:
    case Type::Q5_K:
    case Type::Q6_K:
    case Type::Q2_K:
    case Type::IQ3_XXS:
    case Type::IQ3_S:
    case Type::Q3_K:
    case Type::IQ2_XXS:
    case Type::IQ2_XS:
    case Type::IQ2_S:
    case Type::IQ1_S:
    case Type::Q4_0:
    case Type::Q4_1:
    case Type::Q5_0:
    case Type::Q5_1:
    case Type::IQ4_NL:
    case Type::IQ1_M: return true;
    default: return false;
    }
}

namespace {
std::size_t vec_scales_offset(int k, int columns) {
    return (std::size_t(k) * columns + 15) / 16 * 16;
}
} // namespace

std::size_t vec_activation_bytes(int k, int columns) {
    return vec_scales_offset(k, columns) + std::size_t(columns) * (k / 32) * sizeof(half2);
}

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

} // namespace ninferno::ops::ggml_mmq
