#pragma once

// Generic tiled A16 GEMM: weights whose codes are exact in BF16 (u4 with zero points, ggml Q4_K
// and Q5_K) against BF16 activations, on BF16 tensor cores through shared memory, for
// prefill-sized T.
//
// Orientation as in generic_w4a4.cuh: the activation is the MMA's A operand (M = tokens) and the
// weight its B operand (N = weight rows), with the same row map (SwiGLU gate and up rows
// interleaved in groups of eight so one thread holds both accumulators of an output). A CTA
// stages, per K tile of 64 values, the activation tile and the weight's raw code bytes with
// cp.async, decodes the codes once per CTA into a BF16 tile of exact integers (and each row's
// group scales and mins, from parameters loaded a tile ahead), and runs m16n8k16 MMAs from shared
// memory into an FP32 partial per scale group. At each group's end the partial takes the row's
// group scale into the accumulator, minus (min-term formats) the row's group min times the
// token's activation sum over the group. K tiles and groups accumulate in order and no schedule
// splits K, so every schedule sums an output element in the same order: a token's bytes do not
// depend on the call's T.
//
// Codecs: u4 zero-point (32-, 64- and 128-value groups), Q4_K and Q5_K (32-value groups), the
// symmetric groupwise q4/q5/q6_g64 and q3_g128/q3_g64 (signed codes, an FP16 scale per group). The
// 16-value-group K-quants (Q6_K, Q3_K, Q2_K) stay on the register MMA: rescaling the partial after
// every k16 step made this kernel slower than it on Qwen3-4B (Q6_K prefill 8960 -> 6881 tok/s, Q2_K
// 7645 -> 6828; a double-buffered variant still reached only 7688 for Q6_K).

#include "ops/common/arch_limits.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/generic/generic_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninferno::ops::detail::generic {

template <int BlockM, int BlockN, int WarpsM, int WarpsN, int Stages>
struct TiledSchedule {
    static_assert(BlockM % (16 * WarpsM) == 0 && BlockN % (8 * WarpsN) == 0);
    static constexpr int kBlockM  = BlockM; // tokens
    static constexpr int kBlockN  = BlockN; // weight rows
    static constexpr int kWarpsM  = WarpsM;
    static constexpr int kWarpsN  = WarpsN;
    static constexpr int kStages  = Stages;
    static constexpr int kThreads = WarpsM * WarpsN * 32;
    static constexpr int kWarpM   = BlockM / WarpsM;
    static constexpr int kWarpN   = BlockN / WarpsN;
    static constexpr int kMmaM    = kWarpM / 16;
    static constexpr int kMmaN    = kWarpN / 8;
};

// ---------------------------------------------------------------------------------------------
// Tile codecs. A K tile of one row is 64 values, staged as kRawChunks 16-byte chunks (chunk())
// and decoded as two items: item s gives values 32s .. 32s + 31 as sixteen BF16 pairs (exact
// integers) and the scale (and min) of each scale group it holds. Params are the row's
// parameters for the tile, read from global memory a tile ahead.
// ---------------------------------------------------------------------------------------------

// Eight words of byte codes (< 128) to sixteen BF16 pairs of code - Bias.
template <int Bias>
__device__ __forceinline__ void bytes_to_pairs(const unsigned (&w)[8], unsigned (&out)[16]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) byte_codes<Bias>(w[i], out[2 * i], out[2 * i + 1]);
}

// Thirty-two bytes of shared memory as eight words.
__device__ __forceinline__ void load_words(const std::uint8_t* p, unsigned (&w)[8]) {
    const uint4 lo = load_vec<uint4>(p);
    const uint4 hi = load_vec<uint4>(p + 16);
    w[0] = lo.x, w[1] = lo.y, w[2] = lo.z, w[3] = lo.w;
    w[4] = hi.x, w[5] = hi.y, w[6] = hi.z, w[7] = hi.w;
}

// u4_g<G>_zp_fp16: 32 code bytes per row and tile (value 2i in the low nibble of byte i); a scale
// group is G values: two tiles (G = 128), one tile (64) or one decode item (32).
template <int G>
struct U4ZpTileG {
    static_assert(G == 32 || G == 64 || G == 128);
    static constexpr int kGroup     = G;
    static constexpr bool kMinTerm  = false;
    static constexpr int kRawChunks = 2;
    static constexpr int kParams    = G == 32 ? 2 : 1; // groups of a tile
    struct Params {
        unsigned offset[kParams]; // BF16 pair of 128 + zero
        float scale[kParams];
    };
    __device__ static __forceinline__ const std::uint8_t* chunk(const GemmArgs& a, int row,
                                                                int k_tile, int c) {
        return a.codes + static_cast<std::int64_t>(row) * (a.k / 2) + k_tile * 32 + c * 16;
    }
    __device__ static __forceinline__ Params params(const GemmArgs& a, int row, int k_tile) {
        const std::int64_t gid =
            static_cast<std::int64_t>(row) * a.groups_per_row + k_tile * 64 / G;
        Params p;
#pragma unroll
        for (int i = 0; i < kParams; ++i) {
            p.offset[i] = U4ZpDecoder::offset_of(a.high[gid + i]);
            p.scale[i]  = half_bits(reinterpret_cast<const unsigned short*>(a.scales)[gid + i]);
        }
        return p;
    }
    __device__ static __forceinline__ void decode(const std::uint8_t* raw, int, int s,
                                                  const Params& p, unsigned (&out)[16],
                                                  float* scale, float*) {
        const int i       = kParams == 2 ? s : 0;
        const uint4 piece = load_vec<uint4>(raw + 16 * s);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned word = uint4_word(piece, w);
            const unsigned even = word & 0x0f0f0f0fU;
            const unsigned odd  = (word >> 4) & 0x0f0f0f0fU;
#pragma unroll
            for (unsigned b = 0; b < 4; ++b)
                out[4 * w + b] =
                    codes02_bf16(__byte_perm(even, odd, b | ((b + 4U) << 8)), p.offset[i]);
        }
        scale[0] = p.scale[i];
    }
};

using U4ZpTile    = U4ZpTileG<128>;
using U4ZpG64Tile = U4ZpTileG<64>;
using U4ZpG32Tile = U4ZpTileG<32>;

// q4/q5/q6_g64 (GroupwiseG64Decoder planes): 32 code bytes per row and tile in u4's nibble order,
// the high-bit plane (Q5: 8 bytes per group, bit v for value v; Q6: 16 bytes, bits 2v, 2v + 1)
// read with the params a tile ahead, an FP16 scale per group. A code is a signed Bits-bit
// integer: c ^ 2^(Bits-1) is the code biased by 2^(Bits-1), which byte_codes takes off exactly.
template <int Bits>
struct GroupwiseG64Tile {
    static_assert(Bits == 4 || Bits == 5 || Bits == 6);
    static constexpr int kGroup     = 64;
    static constexpr bool kMinTerm  = false;
    static constexpr int kRawChunks = 2;
    static constexpr int kBias      = 1 << (Bits - 1);
    static constexpr unsigned kFlip = 0x01010101U * kBias;

    struct Params {
        uint4 high; // Q5: x, y; Q6: x .. w
        float scale;
    };

    __device__ static __forceinline__ const std::uint8_t* chunk(const GemmArgs& a, int row,
                                                                int k_tile, int c) {
        return a.codes + (static_cast<std::int64_t>(row) * a.groups_per_row + k_tile) * 32 + c * 16;
    }

    __device__ static __forceinline__ Params params(const GemmArgs& a, int row, int k_tile) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + k_tile;
        Params p;
        p.high = make_uint4(0, 0, 0, 0);
        if constexpr (Bits == 5) {
            const uint2 h = load_vec<uint2>(a.high + gid * 8);
            p.high.x = h.x, p.high.y = h.y;
        } else if constexpr (Bits == 6) {
            p.high = load_vec<uint4>(a.high + gid * 16);
        }
        p.scale = half_bits(reinterpret_cast<const unsigned short*>(a.scales)[gid]);
        return p;
    }

    __device__ static __forceinline__ void decode(const std::uint8_t* raw, int, int s,
                                                  const Params& p, unsigned (&out)[16],
                                                  float* scale, float*) {
        const uint4 piece = load_vec<uint4>(raw + 16 * s);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned word = uint4_word(piece, w);
            const unsigned even = word & 0x0f0f0f0fU;
            const unsigned odd  = (word >> 4) & 0x0f0f0f0fU;
            unsigned lo         = __byte_perm(even, odd, 0x5140); // values 8w .. 8w + 3
            unsigned hi         = __byte_perm(even, odd, 0x7362); // values 8w + 4 .. 8w + 7
            if constexpr (Bits == 5) {
                const unsigned h = (s ? p.high.y : p.high.x) >> (8 * w);
                lo |= spread1x4(h & 0xfU) << 4;
                hi |= spread1x4((h >> 4) & 0xfU) << 4;
            } else if constexpr (Bits == 6) {
                const unsigned h = uint4_word(p.high, 2 * s + (w >> 1)) >> (16 * (w & 1));
                lo |= spread2x4(h & 0xffU) << 4;
                hi |= spread2x4((h >> 8) & 0xffU) << 4;
            }
            byte_codes<kBias>(lo ^ kFlip, out[4 * w], out[4 * w + 1]);
            byte_codes<kBias>(hi ^ kFlip, out[4 * w + 2], out[4 * w + 3]);
        }
        scale[0] = p.scale;
    }
};

// q3_g128 / q3_g64 (GroupwiseQ3Decoder planes): code j of a group in bits 3j .. 3j + 2 of the
// group's 3 * Group / 8 bytes, a signed 3-bit integer; an FP16 scale per group. A tile's 24 code
// bytes start on an 8-byte boundary, so the stage takes them from the 16-byte boundary at or below
// (offset 0 or 8, in the params) and reads no byte past the tile's last.
template <int Group>
struct GroupwiseQ3Tile {
    static_assert(Group == 64 || Group == 128);
    static constexpr int kGroup      = Group;
    static constexpr bool kMinTerm   = false;
    static constexpr int kRawChunks  = 2;
    static constexpr int kGroupBytes = Group * 3 / 8;

    struct Params {
        int offset; // of the tile's first code byte in the staged 32
        float scale;
    };

    __device__ static __forceinline__ std::int64_t gid(const GemmArgs& a, int row, int k_tile) {
        return static_cast<std::int64_t>(row) * a.groups_per_row + k_tile * 64 / Group;
    }

    __device__ static __forceinline__ std::int64_t first_byte(const GemmArgs& a, int row,
                                                              int k_tile) {
        return gid(a, row, k_tile) * kGroupBytes + (k_tile % (Group / 64)) * 24;
    }

    __device__ static __forceinline__ const std::uint8_t* chunk(const GemmArgs& a, int row,
                                                                int k_tile, int c) {
        return a.codes + (first_byte(a, row, k_tile) & ~std::int64_t{15}) + c * 16;
    }

    // Bytes of chunk c the tile uses: at offset 0 the second chunk holds 8 of them.
    __device__ static __forceinline__ int chunk_bytes(const GemmArgs& a, int row, int k_tile,
                                                      int c) {
        return c == 1 && (first_byte(a, row, k_tile) & 15) == 0 ? 8 : 16;
    }

    __device__ static __forceinline__ Params params(const GemmArgs& a, int row, int k_tile) {
        return {static_cast<int>(first_byte(a, row, k_tile) & 15),
                half_bits(reinterpret_cast<const unsigned short*>(a.scales)[gid(a, row, k_tile)])};
    }

    __device__ static __forceinline__ void decode(const std::uint8_t* raw, int, int s,
                                                  const Params& p, unsigned (&out)[16],
                                                  float* scale, float*) {
        // Item s: values 32s .. 32s + 31, bits 0 .. 95 of the three words from byte 12s.
        const auto* w     = reinterpret_cast<const unsigned*>(raw + p.offset + 12 * s);
        const unsigned w0 = w[0], w1 = w[1], w2 = w[2];
        const unsigned long long b0 = (static_cast<unsigned long long>(w1) << 32) | w0;
        const unsigned long long b1 = (static_cast<unsigned long long>(w2) << 32) | w1;
#pragma unroll
        for (int q = 0; q < 8; ++q) { // values 4q .. 4q + 3: bits 12q .. 12q + 11
            const unsigned x = 12 * q + 12 <= 64
                                   ? static_cast<unsigned>(b0 >> (12 * q)) & 0xfffU
                                   : static_cast<unsigned>(b1 >> (12 * q - 32)) & 0xfffU;
            byte_codes<4>(spread3x4(x) ^ 0x04040404U, out[2 * q], out[2 * q + 1]);
        }
        scale[0] = p.scale;
    }
};

// q8_g32 (Q8Decoder): a row's codes are K signed bytes, an FP16 scale per 32 values; a tile is 64
// code bytes and two groups, decode item s is group s. Exact in BF16 (|code| <= 128). Only Volta
// takes it (volta_tiled_route): elsewhere Q8 keeps its own kernels and the register MMA.
struct Q8Tile {
    static constexpr int kGroup     = 32;
    static constexpr bool kMinTerm  = false;
    static constexpr int kRawChunks = 4;

    struct Params {
        float scale[2];
    };

    __device__ static __forceinline__ const std::uint8_t* chunk(const GemmArgs& a, int row,
                                                                int k_tile, int c) {
        return a.codes + (static_cast<std::int64_t>(row) * a.groups_per_row + 2 * k_tile) * 32 +
               c * 16;
    }

    __device__ static __forceinline__ Params params(const GemmArgs& a, int row, int k_tile) {
        const std::int64_t gid = static_cast<std::int64_t>(row) * a.groups_per_row + 2 * k_tile;
        const auto* sc         = reinterpret_cast<const unsigned short*>(a.scales);
        return {{half_bits(sc[gid]), half_bits(sc[gid + 1])}};
    }

    __device__ static __forceinline__ void decode(const std::uint8_t* raw, int, int s,
                                                  const Params& p, unsigned (&out)[16],
                                                  float* scale, float*) {
        unsigned w[8];
        load_words(raw + 32 * s, w);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const auto b = [&](int j) {
                return static_cast<int>(static_cast<signed char>(w[i] >> (8 * j)));
            };
            out[2 * i]     = int_pair_bf16(b(0), b(1));
            out[2 * i + 1] = int_pair_bf16(b(2), b(3));
        }
        scale[0] = p.scale[s];
    }
};

// ggml K-quants: tile t = k_tile % 4 of a superblock is half h = t / 2, part c = t % 2.
__device__ __forceinline__ std::int64_t kq_block(const GemmArgs& a, int row, int k_tile) {
    return static_cast<std::int64_t>(row) * a.groups_per_row + (k_tile >> 2);
}

// Q4_K / Q5_K: tile t is sub-blocks 2t (item 0: low nibbles of nibble bytes 32t .. 32t + 31) and
// 2t + 1 (item 1: their high nibbles), plus bit 2t + s of the high-bit bytes (Q5_K).
template <int Bits>
struct GgmlQ45KTile {
    static constexpr int kGroup       = 32;
    static constexpr bool kMinTerm    = true;
    static constexpr int kRawChunks   = Bits == 4 ? 2 : 4;
    static constexpr int kBlockBytes  = Bits == 4 ? 144 : 176;
    static constexpr int kNibbleStart = Bits == 4 ? 16 : 48;
    struct Params {
        uint4 head; // d, dmin, the 12 packed scale bytes
    };
    __device__ static __forceinline__ const std::uint8_t* chunk(const GemmArgs& a, int row,
                                                                int k_tile, int c) {
        const std::uint8_t* b = a.codes + kq_block(a, row, k_tile) * kBlockBytes;
        return c < 2 ? b + kNibbleStart + 32 * (k_tile & 3) + 16 * c : b + 16 + 16 * (c - 2);
    }
    __device__ static __forceinline__ Params params(const GemmArgs& a, int row, int k_tile) {
        return {load_vec<uint4>(a.codes + kq_block(a, row, k_tile) * kBlockBytes)};
    }
    __device__ static __forceinline__ void decode(const std::uint8_t* raw, int k_tile, int s,
                                                  const Params& p, unsigned (&out)[16],
                                                  float* scale, float* min) {
        const int j = 2 * (k_tile & 3) + s;
        unsigned w[8];
        load_words(raw, w);
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = (w[i] >> (4 * s)) & 0x0f0f0f0fU;
        if constexpr (Bits == 5) {
            unsigned h[8];
            load_words(raw + 32, h);
#pragma unroll
            for (int i = 0; i < 8; ++i) w[i] |= ((h[i] >> j) & 0x01010101U) << 4;
        }
        bytes_to_pairs<0>(w, out);
        float sc, m;
        kq_scale_min(p.head, j, sc, m);
        scale[0] = half_bits(p.head.x) * sc;
        min[0]   = half_bits(p.head.x >> 16) * m;
    }
};

// ---------------------------------------------------------------------------------------------
// The kernel.
// ---------------------------------------------------------------------------------------------

template <class C>
struct TiledGroups {
    static constexpr int kPerTile = C::kGroup >= 64 ? 1 : 64 / C::kGroup; // groups in a K tile
    static constexpr int kPerItem = kPerTile >= 2 ? kPerTile / 2 : 1;     // groups one item sets
};

template <class S, class C, int Stages>
struct TiledShared {
    alignas(16) __nv_bfloat16 act[Stages][S::kBlockM * 64];
    alignas(16) std::uint8_t raw[Stages][S::kBlockN * C::kRawChunks * 16];
    alignas(16) __nv_bfloat16 weight[S::kBlockN * 64];
    float scale[S::kBlockN][TiledGroups<C>::kPerTile];
    float min[C::kMinTerm ? S::kBlockN : 1][TiledGroups<C>::kPerTile];
    float xsum[C::kMinTerm ? S::kBlockM : 1][4]; // activation sums per 16 K values
};

// The schedule's pipeline depth, lowered until the shared tiles fit one block of this build
// (99 KiB; Turing 64 KiB). A schedule over the limit at two stages does not fit (tiled_fits).
template <class S, class C, int Stages = S::kStages>
struct TiledStages {
    static constexpr int value = sizeof(TiledShared<S, C, Stages>) <= kMaxSharedBytesPerBlock ||
                                         Stages == 2
                                     ? Stages
                                     : TiledStages<S, C, Stages - 1>::value;
};
template <class S, class C>
struct TiledStages<S, C, 1> {
    static constexpr int value = 2;
};

template <class S, class C>
constexpr int tiled_shared_bytes() {
    return static_cast<int>(sizeof(TiledShared<S, C, TiledStages<S, C>::value>));
}

template <class S, class C>
constexpr bool tiled_schedule_fits() {
    return tiled_shared_bytes<S, C>() <= static_cast<int>(kMaxSharedBytesPerBlock);
}

// Sixteen-byte chunk c of a 64-value BF16 row, XOR-swizzled by the row's low bits.
__device__ __forceinline__ int tiled_swizzle(int row, int chunk) { return (chunk ^ (row & 7)) * 8; }

template <class S, bool SwiGlu>
__device__ __forceinline__ int tiled_weight_row(const GemmArgs& a, int block_row, int tile_row,
                                                bool& ok) {
    const int local = w4a4_local_row<SwiGlu, S::kBlockN>(block_row, tile_row);
    ok              = local < a.rows;
    return a.row_begin + (ok ? local : 0) + (SwiGlu && ((tile_row >> 3) & 1) ? a.up_offset : 0);
}

template <class S, class C, int St, bool SwiGlu>
__device__ __forceinline__ void tiled_stage(const GemmArgs& a, TiledShared<S, C, St>& sh, int stage,
                                            int k_tile, int token_begin, int block_row) {
    const auto* x = static_cast<const __nv_bfloat16*>(a.x);
    for (int task = static_cast<int>(threadIdx.x); task < S::kBlockM * 8; task += S::kThreads) {
        const int row   = task >> 3;
        const int chunk = task & 7;
        const int token = token_begin + row;
        const bool ok   = token < a.tokens;
        const auto* src =
            x + static_cast<std::int64_t>(ok ? token : 0) * a.ldx + k_tile * 64 + chunk * 8;
        cp_async_zfill<16, Cache::cg>(&sh.act[stage][row * 64 + tiled_swizzle(row, chunk)], src,
                                      ok ? 16 : 0);
    }
    for (int task = static_cast<int>(threadIdx.x); task < S::kBlockN * C::kRawChunks;
         task += S::kThreads) {
        const int row  = task / C::kRawChunks;
        const int c    = task - row * C::kRawChunks;
        bool ok        = false;
        const int wrow = tiled_weight_row<S, SwiGlu>(a, block_row, row, ok);
        int bytes      = 16;
        if constexpr (requires { C::chunk_bytes(a, wrow, k_tile, c); }) {
            bytes = C::chunk_bytes(a, wrow, k_tile, c);
        }
        cp_async_zfill<16, Cache::cg>(&sh.raw[stage][(row * C::kRawChunks + c) * 16],
                                      C::chunk(a, wrow, k_tile, c), ok ? bytes : 0);
    }
}

// ---------------------------------------------------------------------------------------------
// Volta (sm_70): the tensor cores take mma.m8n8k4 with f16 operands only (ops/common/mma.cuh).
// The CTA stages the same tiles, then turns them into f16 once per K tile: the activation tile in
// place (finite values saturate at 65504, as on Turing) and the decoded codes with their group
// scale (and min) applied, w = code * scale - min in FP32 rounded once to f16. Each warp's four
// quad-pairs take 8 tokens each against the same 8 weight rows, so a lane's accumulators of
// consecutive 8-row groups belong to the same tokens and columns: a SwiGLU gate group and its up
// group (interleaved by eight, w4a4_local_row) meet in one lane. A warp covers kWarpM tokens
// (multiples of 32) by kWarpN rows; K tiles accumulate in order, so a token's bytes do not depend
// on the call's T (the Volta build's own bytes: f16 operands, not the BF16 MMA's).
// ---------------------------------------------------------------------------------------------
#if NINFERNO_HMMA884
// The power of two a weight row's f16 values divide by so that code * scale - min stays inside the
// f16 range (codes are at most 128 in magnitude); 0 wherever a real model's weights stand, which
// leaves every byte as it was. The sums multiply it back in the epilogue.
__device__ __forceinline__ int volta_row_shift(float scale, float min) {
    const float bound = fabsf(scale) * 128.0F + fabsf(min);
    return bound < 32768.0F ? 0 : ilogbf(bound) - 14;
}

template <class S, class C, Epilogue E>
__device__ __forceinline__ void tiled_hmma884(const GemmArgs& a) {
    constexpr bool kSwiGlu = E == Epilogue::SwiGlu;
    constexpr int kStages  = TiledStages<S, C>::value;
    constexpr int kItems   = S::kBlockN * 2 / S::kThreads;
    static_assert(kItems * S::kThreads == S::kBlockN * 2 && kItems >= 1);
    static_assert(S::kWarpM % 32 == 0 && S::kWarpN % 8 == 0);
    constexpr int kVM = S::kWarpM / 32; // groups of the four quad-pairs' 8 tokens
    constexpr int kVN = S::kWarpN / 8;  // 8-row groups
    static_assert(!kSwiGlu || (kVN % 2) == 0);
    using G      = TiledGroups<C>;
    using Shared = TiledShared<S, C, kStages>;
    static_assert(G::kPerItem == 1, "a decode item holds one scale group");

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& sh              = *reinterpret_cast<Shared*>(shared_bytes);
    const int block_row   = static_cast<int>(blockIdx.x);
    const int token_begin = static_cast<int>(blockIdx.y) * S::kBlockM;
    const int k_tiles     = a.k / 64;

#    pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
        if (stage < k_tiles)
            tiled_stage<S, C, kStages, kSwiGlu>(a, sh, stage, stage, token_begin, block_row);
        cp_commit();
    }

    int item_row[kItems], item_wrow[kItems], item_s[kItems];
    bool item_ok[kItems];
    typename C::Params next[kItems];
#    pragma unroll
    for (int i = 0; i < kItems; ++i) {
        const int task = static_cast<int>(threadIdx.x) + i * S::kThreads;
        item_row[i]    = task >> 1;
        item_s[i]      = task & 1;
        item_wrow[i]   = tiled_weight_row<S, kSwiGlu>(a, block_row, item_row[i], item_ok[i]);
        next[i]        = C::params(a, item_wrow[i], 0);
    }

    float acc[kVM][kVN][8] = {};
    // Each weight row's power of two out of the f16 range (volta_row_shift): row_shift[r] in
    // shared memory, item_shift per decode item, acc_shift for the rows of this lane's
    // accumulators (kVN groups, four rows each). It only grows; at 0 nothing changes.
    __shared__ int row_shift[S::kBlockN];
    for (int r = static_cast<int>(threadIdx.x); r < S::kBlockN; r += S::kThreads) row_shift[r] = 0;
    int item_shift[kItems] = {};
    int acc_shift[kVN][4]  = {};
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m       = warp / S::kWarpsN;
    const int warp_n       = warp - warp_m * S::kWarpsN;
    const int quad_pair    = (lane >> 2) & 3;
    const int frag_row     = (lane & 3) + ((lane >> 4) << 2); // A row and B column of the lane
    const auto* weight16   = reinterpret_cast<const __half*>(sh.weight);

#    pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % kStages;
        cp_wait<kStages - 1>();
        __syncthreads();
        // The activation tile to f16, in place.
        for (int task = static_cast<int>(threadIdx.x); task < S::kBlockM * 8; task += S::kThreads) {
            __nv_bfloat16* p = &sh.act[stage][task * 8];
            const uint4 v    = load_vec<uint4>(p);
            store_vec(p, make_uint4(bf16x2_to_f16x2(v.x), bf16x2_to_f16x2(v.y),
                                    bf16x2_to_f16x2(v.z), bf16x2_to_f16x2(v.w)));
        }
        // The weight tile: each code times its group's scale, minus the group's min, as f16.
        bool shifted = false;
#    pragma unroll
        for (int i = 0; i < kItems; ++i) {
            const typename C::Params p = next[i];
            if (k_tile + 1 < k_tiles) next[i] = C::params(a, item_wrow[i], k_tile + 1);
            const int row = item_row[i];
            const int s   = item_s[i];
            unsigned v[16];
            float scale[1], min[1] = {0.0F};
            C::decode(&sh.raw[stage][row * C::kRawChunks * 16], k_tile, s, p, v, scale, min);
            float sc = item_ok[i] ? scale[0] : 0.0F;
            float mn = item_ok[i] && C::kMinTerm ? min[0] : 0.0F;
            // A row's two items sit on neighbouring threads; both keep the larger shift.
            int need = volta_row_shift(sc, mn);
            need     = max(need, __shfl_xor_sync(0xffffffffU, need, 1));
            if (need > item_shift[i]) {
                item_shift[i] = need;
                shifted       = true;
                if (s == 0) row_shift[row] = need;
            }
            if (item_shift[i] > 0) {
                const float f = __int_as_float((127 - item_shift[i]) << 23); // 2^-shift
                sc *= f;
                mn *= f;
            }
            unsigned h[16];
#    pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float lo  = fmaf(__uint_as_float(v[j] << 16), sc, -mn);
                const float hi  = fmaf(__uint_as_float(v[j] & 0xffff0000U), sc, -mn);
                const __half2 w = __floats2half2_rn(lo, hi);
                h[j]            = *reinterpret_cast<const unsigned*>(&w);
            }
#    pragma unroll
            for (int c = 0; c < 4; ++c) {
                store_vec(&sh.weight[row * 64 + tiled_swizzle(row, s * 4 + c)],
                          make_uint4(h[4 * c], h[4 * c + 1], h[4 * c + 2], h[4 * c + 3]));
            }
        }
        // A row whose shift grew: the lanes holding its sums scale them down to match (rare: only
        // group scales past the f16 range shift at all).
        if (__syncthreads_or(shifted)) {
#    pragma unroll
            for (int n = 0; n < kVN; ++n)
#    pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int tile_row = warp_n * S::kWarpN + n * 8 + ((q & 2) << 1) + (lane & 2) + (q & 1);
                    const int e        = row_shift[tile_row];
                    if (e == acc_shift[n][q]) continue;
                    const float f = __int_as_float((127 - (e - acc_shift[n][q])) << 23);
                    acc_shift[n][q] = e;
#    pragma unroll
                    for (int m = 0; m < kVM; ++m)
#    pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            if ((((i & 4) >> 1) | (i & 1)) == q) acc[m][n][i] *= f;
                        }
                }
        }
        const auto* at = reinterpret_cast<const __half*>(sh.act[stage]);
#    pragma unroll
        for (int k4 = 0; k4 < 16; ++k4) {
            const int chunk = k4 >> 1;
            const int sub   = (k4 & 1) * 4;
            uint2 af[kVM], bf[kVN];
#    pragma unroll
            for (int m = 0; m < kVM; ++m) {
                const int row = warp_m * S::kWarpM + m * 32 + quad_pair * 8 + frag_row;
                af[m]         = load_vec<uint2>(at + row * 64 + tiled_swizzle(row, chunk) + sub);
            }
#    pragma unroll
            for (int n = 0; n < kVN; ++n) {
                const int row = warp_n * S::kWarpN + n * 8 + frag_row;
                bf[n] = load_vec<uint2>(weight16 + row * 64 + tiled_swizzle(row, chunk) + sub);
            }
#    pragma unroll
            for (int m = 0; m < kVM; ++m)
#    pragma unroll
                for (int n = 0; n < kVN; ++n)
                    mma_m8n8k4_f16(acc[m][n], af[m].x, af[m].y, bf[n].x, bf[n].y);
        }
        __syncthreads();
        const int next_tile = k_tile + kStages;
        if (next_tile < k_tiles)
            tiled_stage<S, C, kStages, kSwiGlu>(a, sh, stage, next_tile, token_begin, block_row);
        cp_commit();
    }

#    pragma unroll
    for (int m = 0; m < kVM; ++m) {
        const int t0 = token_begin + warp_m * S::kWarpM + m * 32 + quad_pair * 8 + (lane & 1) +
                       ((lane >> 4) << 2);
#    pragma unroll
        for (int n = 0; n < kVN; n += (kSwiGlu ? 2 : 1)) {
#    pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int token    = t0 + (i & 2);
                const int tile_row = warp_n * S::kWarpN + n * 8 + (i & 4) + (lane & 2) + (i & 1);
                const int local    = w4a4_local_row<kSwiGlu, S::kBlockN>(block_row, tile_row);
                if (local >= a.rows || token >= a.tokens) continue;
                const int q = ((i & 4) >> 1) | (i & 1);
                float value = acc[m][n][i];
                if (acc_shift[n][q] > 0) value *= __int_as_float((127 + acc_shift[n][q]) << 23);
                if constexpr (kSwiGlu) {
                    float up = acc[m][n + 1][i];
                    if (acc_shift[n + 1][q] > 0) up *= __int_as_float((127 + acc_shift[n + 1][q]) << 23);
                    value = generic_gate(a, value) * up;
                }
                store_output<E>(a, local, token, value);
            }
        }
    }
}
#endif

template <class S, class C, Epilogue E>
__global__ __launch_bounds__(S::kThreads) void tiled_kernel(GemmArgs a) {
#if NINFERNO_HMMA884
    tiled_hmma884<S, C, E>(a);
#else
    constexpr bool kSwiGlu = E == Epilogue::SwiGlu;
    static_assert(!kSwiGlu || (S::kMmaN % 2) == 0);
    constexpr int kStages = TiledStages<S, C>::value;
    constexpr int kItems  = S::kBlockN * 2 / S::kThreads; // decode items per thread
    static_assert(kItems * S::kThreads == S::kBlockN * 2 && kItems >= 1);
    constexpr int kGroupSteps = C::kGroup / 16; // k16 MMA steps per scale group
    using G                   = TiledGroups<C>;
    using Shared              = TiledShared<S, C, kStages>;

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& sh              = *reinterpret_cast<Shared*>(shared_bytes);
    const int block_row   = static_cast<int>(blockIdx.x);
    const int token_begin = static_cast<int>(blockIdx.y) * S::kBlockM;
    const int k_tiles     = a.k / 64;

#    pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
        if (stage < k_tiles)
            tiled_stage<S, C, kStages, kSwiGlu>(a, sh, stage, stage, token_begin, block_row);
        cp_commit();
    }

    // Decode items: this thread's (row, item) pairs and their parameters, one tile ahead.
    int item_row[kItems], item_wrow[kItems], item_s[kItems];
    bool item_ok[kItems];
    typename C::Params next[kItems];
#    pragma unroll
    for (int i = 0; i < kItems; ++i) {
        const int task = static_cast<int>(threadIdx.x) + i * S::kThreads;
        item_row[i]    = task >> 1;
        item_s[i]      = task & 1;
        item_wrow[i]   = tiled_weight_row<S, kSwiGlu>(a, block_row, item_row[i], item_ok[i]);
        next[i]        = C::params(a, item_wrow[i], 0);
    }

    float acc[S::kMmaM][S::kMmaN][4]  = {};
    float part[S::kMmaM][S::kMmaN][4] = {};
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m       = warp / S::kWarpsN;
    const int warp_n       = warp - warp_m * S::kWarpsN;
    const int a_matrix     = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_chunk      = a_matrix >> 1;
    const int b_row_offset = lane & 7;
    const int b_chunk      = (lane >> 3) & 1;
    const int acc_row      = lane >> 2;
    const int acc_col      = 2 * (lane & 3);

#    pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % kStages;
        cp_wait<kStages - 1>();
        __syncthreads();
        // Decode the tile's weight codes, and the rows' group scales and mins, once for the CTA.
#    pragma unroll
        for (int i = 0; i < kItems; ++i) {
            const typename C::Params p = next[i];
            if (k_tile + 1 < k_tiles) next[i] = C::params(a, item_wrow[i], k_tile + 1);
            const int row = item_row[i];
            const int s   = item_s[i];
            unsigned v[16];
            float scale[G::kPerItem], min[G::kPerItem];
            C::decode(&sh.raw[stage][row * C::kRawChunks * 16], k_tile, s, p, v, scale, min);
#    pragma unroll
            for (int c = 0; c < 4; ++c) {
                store_vec(&sh.weight[row * 64 + tiled_swizzle(row, s * 4 + c)],
                          make_uint4(v[4 * c], v[4 * c + 1], v[4 * c + 2], v[4 * c + 3]));
            }
            if (G::kPerTile >= 2 || s == 0) {
#    pragma unroll
                for (int g = 0; g < G::kPerItem; ++g) {
                    const int gt      = G::kPerTile >= 2 ? s * G::kPerItem + g : 0;
                    sh.scale[row][gt] = item_ok[i] ? scale[g] : 0.0F;
                    if constexpr (C::kMinTerm) sh.min[row][gt] = item_ok[i] ? min[g] : 0.0F;
                }
            }
        }
        if constexpr (C::kMinTerm) {
            // Activation sums per token and 16 K values, in one order for every schedule.
            for (int task = static_cast<int>(threadIdx.x); task < S::kBlockM * 4;
                 task += S::kThreads) {
                const int row = task >> 2;
                const int q   = task & 3;
                float sum     = 0.0F;
#    pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 v =
                        load_vec<uint4>(&sh.act[stage][row * 64 + tiled_swizzle(row, 2 * q + h)]);
#    pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const unsigned word = uint4_word(v, w);
                        sum += __uint_as_float(word << 16);
                        sum += __uint_as_float(word & 0xffff0000U);
                    }
                }
                sh.xsum[row][q] = sum;
            }
        }
        __syncthreads();
#    pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned af[S::kMmaM][4];
            unsigned bf[S::kMmaN][2];
#    pragma unroll
            for (int m = 0; m < S::kMmaM; ++m) {
                const int row = warp_m * S::kWarpM + m * 16 + a_row_offset;
                ldmatrix_x4(
                    af[m][0], af[m][1], af[m][2], af[m][3],
                    smem_addr(&sh.act[stage][row * 64 + tiled_swizzle(row, 2 * ks + a_chunk)]));
            }
#    pragma unroll
            for (int n = 0; n < S::kMmaN; ++n) {
                const int row = warp_n * S::kWarpN + n * 8 + b_row_offset;
                ldmatrix_x2(bf[n][0], bf[n][1],
                            smem_addr(&sh.weight[row * 64 + tiled_swizzle(row, 2 * ks + b_chunk)]));
            }
#    pragma unroll
            for (int m = 0; m < S::kMmaM; ++m)
#    pragma unroll
                for (int n = 0; n < S::kMmaN; ++n)
                    mma_bf16(part[m][n][0], part[m][n][1], part[m][n][2], part[m][n][3], af[m][0],
                             af[m][1], af[m][2], af[m][3], bf[n][0], bf[n][1]);
            // A group ends at its last k16 step, or at K's when the row's last group is partial
            // (q3_g128 at K % 128 == 64).
            if ((k_tile * 4 + ks + 1) % kGroupSteps != 0 && !(ks == 3 && k_tile + 1 == k_tiles)) {
                continue;
            }
            // The group ends: scale the partial into the accumulator (and subtract the min term).
            const int gt = G::kPerTile >= 2 ? ks / kGroupSteps : 0;
            float xs[S::kMmaM][2];
            if constexpr (C::kMinTerm) {
#    pragma unroll
                for (int m = 0; m < S::kMmaM; ++m) {
                    const int t0 = warp_m * S::kWarpM + m * 16 + acc_row;
#    pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float sum = 0.0F;
#    pragma unroll
                        for (int q = 0; q < kGroupSteps; ++q) {
                            sum += sh.xsum[t0 + 8 * h][gt * kGroupSteps + q];
                        }
                        xs[m][h] = sum;
                    }
                }
            }
#    pragma unroll
            for (int n = 0; n < S::kMmaN; ++n) {
                const int col  = warp_n * S::kWarpN + n * 8 + acc_col;
                const float s0 = sh.scale[col][gt];
                const float s1 = sh.scale[col + 1][gt];
#    pragma unroll
                for (int m = 0; m < S::kMmaM; ++m) {
                    acc[m][n][0]  = fmaf(part[m][n][0], s0, acc[m][n][0]);
                    acc[m][n][1]  = fmaf(part[m][n][1], s1, acc[m][n][1]);
                    acc[m][n][2]  = fmaf(part[m][n][2], s0, acc[m][n][2]);
                    acc[m][n][3]  = fmaf(part[m][n][3], s1, acc[m][n][3]);
                    part[m][n][0] = part[m][n][1] = part[m][n][2] = part[m][n][3] = 0.0F;
                }
                if constexpr (C::kMinTerm) {
                    const float m0 = sh.min[col][gt];
                    const float m1 = sh.min[col + 1][gt];
#    pragma unroll
                    for (int m = 0; m < S::kMmaM; ++m) {
                        acc[m][n][0] = fmaf(-xs[m][0], m0, acc[m][n][0]);
                        acc[m][n][1] = fmaf(-xs[m][0], m1, acc[m][n][1]);
                        acc[m][n][2] = fmaf(-xs[m][1], m0, acc[m][n][2]);
                        acc[m][n][3] = fmaf(-xs[m][1], m1, acc[m][n][3]);
                    }
                }
            }
        }
        __syncthreads();
        const int next_tile = k_tile + kStages;
        if (next_tile < k_tiles)
            tiled_stage<S, C, kStages, kSwiGlu>(a, sh, stage, next_tile, token_begin, block_row);
        cp_commit();
    }

#    pragma unroll
    for (int m = 0; m < S::kMmaM; ++m) {
        const int t0 = token_begin + warp_m * S::kWarpM + m * 16 + acc_row;
#    pragma unroll
        for (int n = 0; n < S::kMmaN; n += (kSwiGlu ? 2 : 1)) {
            const int tile_row = warp_n * S::kWarpN + n * 8 + acc_col;
#    pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int local = w4a4_local_row<kSwiGlu, S::kBlockN>(block_row, tile_row + c);
                if (local >= a.rows) continue;
#    pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int token = t0 + h * 8;
                    if (token >= a.tokens) continue;
                    float value = acc[m][n][h * 2 + c];
                    if constexpr (kSwiGlu)
                        value = generic_gate(a, value) * acc[m][n + 1][h * 2 + c];
                    store_output<E>(a, local, token, value);
                }
            }
        }
    }
#endif
}

} // namespace ninferno::ops::detail::generic
