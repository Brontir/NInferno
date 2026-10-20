// Generic I8: Q4 / Q5 / Q6 G64 weights against int8 activations with one scale per 64-value group,
// the 4-bit code fed to the int8 MMA as its unsigned operand.
//
// out[t, n] = sum_g sx[t, g] * sw[n, g] * D[t, n, g], D = sum_{k in g} xq[t, k] * c[n, k], exact in
// int32. The MMA computes sum xq * u with u = c + Z (Z = 8 / 16 / 32: the code's top bit flipped,
// a valid u8; Q5 / Q6 take their high bits from the high plane); the zero point -Z * sum_k
// xq[t, k] (the quantizer's per-group sum) and the bias of the
// float trick sit in the accumulator's initial value, so two k32 MMAs chain one group and
// as_float(D) - 1.5 * 2^23 is the exact integer (|D| < 2^20). Then one FMUL and one FFMA per output
// element and group: two FP32 roundings, K groups ascending, K never split, so a token's bytes do
// not depend on its call's width or tile (route_is_token_invariant); they differ from the W4A8
// tiles' (one activation scale per 32 values).
//
// The quantizer writes each 64-group's int8 values in a k-permutation: position 32j + 16h + 4t + i
// holds value 16t + 8j + 2i + h, so the B fragment of lane (n, t) at k-step j is the low (b0) or
// high (b1) nibbles of word j of the row's eight code bytes at 8t: an AND and a shift of the
// weight's own bytes, no repacking. Tokens are padded to whole 256-token tiles with zero codes,
// scales and sums, so the kernel loads no partial tile.
//
// CTA: WT x WN warps of 64 tokens x 64 weight rows (4 x 8 m16n8k32 tiles), K stage = one group,
// Stages-deep cp.async ring; the weight scales of 8 groups are loaded as one 16-byte run per row
// into a two-chunk ring. 255 registers, no spill (ptxas, sm_86 / 89 / 120a). Measured
// (27B shapes, T 512..4096): RTX 3090 0.49..0.57 of its measured int8 MMA peak
// (140..162 TOP/s against 97 for the best W4A8 tile), RTX 4090 0.53, RTX 5070 Ti 0.51..0.54.
#include "core/per_device.h"
#include "core/device.h"
#include "ops/common/arch_limits.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define NINFERNO_I8_KERNELS 1
#endif

template <int WT, int WN, int Stages, int Bits>
struct I8Tile {
    static constexpr int kT          = 64 * WT; // tokens per CTA
    static constexpr int kN          = 64 * WN; // weight rows per CTA
    static constexpr int kStages     = Stages;
    static constexpr int kThreads    = 32 * WT * WN;
    static constexpr int kHigh       = Bits == 4 ? 0 : Bits == 5 ? 8 : 16; // high-plane bytes a group
    static constexpr int kAStage     = kT * 64;    // int8 activation, one group per token
    static constexpr int kBStage     = kN * 32;    // low nibbles, 32 bytes per row and group
    static constexpr int kHStage     = kN * kHigh; // Q5 / Q6: the high bits of the codes
    static constexpr int kSxStage    = kT * 4;     // fp32 activation scale per token
    static constexpr int kSumStage   = kT * 4;     // int32 activation sum per token
    static constexpr int kHOffset    = kAStage + kBStage;
    static constexpr int kSxOffset   = kHOffset + kHStage;
    static constexpr int kSumOffset  = kSxOffset + kSxStage;
    static constexpr int kStageBytes = kSumOffset + kSumStage;
    static constexpr int kRingBytes  = 2 * kN * 8 * 2; // fp16 weight scales, 2 chunks of 8 groups
    static constexpr int kSmemBytes  = kStages * kStageBytes + kRingBytes;
};

__device__ __forceinline__ unsigned i8_smem(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void i8_cp16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void i8_cp8(unsigned dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void i8_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void i8_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void i8_ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void i8_mma(int (&d)[4], const unsigned (&a)[4], unsigned b0,
                                       unsigned b1) {
#ifdef NINFERNO_I8_KERNELS
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)d; (void)a; (void)b0; (void)b1;
#endif
}

// The high bits of a lane's eight values at a k-step (values 16t + 8j + m, m = 0..7) moved to the
// bits above the nibble of the B fragment's bytes: b0 takes the even m (byte i: m = 2i), b1 the
// odd. Q5: one bit per value (bit m of the plane's byte 2t + j) to bit 4; Q6: two (bits 2m, 2m + 1
// of the plane's 16 bits at 4t + 2j) to bits 4..5.
__device__ __forceinline__ unsigned i8_spread_bits(unsigned four) { // bits 0..3 -> bit 4 of bytes
    return ((four * 0x00204081U) & 0x01010101U) << 4;
}
__device__ __forceinline__ unsigned i8_even_bits(unsigned b) { // bits 0, 2, 4, 6 -> 0..3
    b &= 0x55U;
    b = (b | (b >> 1)) & 0x33U;
    return (b | (b >> 2)) & 0x0fU;
}
__device__ __forceinline__ unsigned i8_spread_pairs(unsigned h) { // pairs at 0, 4, 8, 12 -> bits 4..5 of bytes
    h &= 0x3333U;
    h = (h & 0x0033U) | ((h & 0x3300U) << 8);         // 0, 4, 16, 20
    h = (h & 0x00030003U) | ((h & 0x00300030U) << 4); // 0, 8, 16, 24
    return h << 4;
}

// Position of value v of a 64-group in the activation's k-permutation.
__host__ __device__ constexpr int i8_position(int v) {
    return 32 * ((v >> 3) & 1) + 16 * (v & 1) + 4 * (v >> 4) + ((v >> 1) & 3);
}

// One thread per (token, group) of the padded token range: the group's int8 codes rint(x*127/amax)
// in the k-permutation, the scale amax/127 and the codes' integer sum, [groups][padded tokens].
__global__ __launch_bounds__(256) void i8_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                                          int ldx, int k, int tokens, int padded,
                                                          std::int8_t* __restrict__ codes,
                                                          float* __restrict__ scales,
                                                          int* __restrict__ sums) {
    const int groups         = k / 64;
    const std::int64_t task  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (task >= static_cast<std::int64_t>(padded) * groups) return;
    const int group = static_cast<int>(task / padded);
    const int token = static_cast<int>(task - static_cast<std::int64_t>(group) * padded);
    unsigned words[16] = {};
    float amax         = 0.0F;
    int qsum           = 0;
    if (token < tokens) {
        float v[64];
        const auto* p = reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token) * ldx +
                                                       group * 64);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint4 w       = p[i];
            const unsigned u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[8 * i + 2 * j]     = __uint_as_float(u[j] << 16);
                v[8 * i + 2 * j + 1] = __uint_as_float(u[j] & 0xffff0000U);
            }
        }
#pragma unroll
        for (int i = 0; i < 64; ++i) amax = fmaxf(amax, fabsf(v[i]));
        const float inv = amax > 0.0F ? 127.0F / amax : 0.0F;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            const int q   = max(-127, min(127, __float2int_rn(v[i] * inv)));
            const int pos = i8_position(i);
            words[pos >> 2] |= (static_cast<unsigned>(q) & 0xffU) << (8 * (pos & 3));
            qsum += q;
        }
    }
    auto* dst = reinterpret_cast<uint4*>(codes + static_cast<std::int64_t>(token) * k + group * 64);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        dst[i] = make_uint4(words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3]);
    scales[task] = amax / 127.0F;
    sums[task]   = qsum;
}

template <int WT, int WN, int Stages, int Bits, Epilogue E>
__global__ void __launch_bounds__(32 * WT * WN, 1)
    i8_kernel(GemmArgs a, const std::int8_t* __restrict__ xq, const float* __restrict__ sx_plane,
              const int* __restrict__ sum_plane, int padded) {
#ifdef NINFERNO_I8_KERNELS
    using L                = I8Tile<WT, WN, Stages, Bits>;
    constexpr bool kSwiGlu = E == Epilogue::SwiGlu;
    constexpr int kZero    = 1 << (Bits - 1); // the offset that makes a two's-complement code a u8
    extern __shared__ __align__(128) unsigned char i8_shared[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_t = warp % WT, warp_n = warp / WT;
    const int gid = lane >> 2, tig = lane & 3;
    const int t0        = blockIdx.x * L::kT;
    const int block_row = blockIdx.y;
    const int groups    = a.k / 64;
    unsigned char* ring = i8_shared + L::kStages * L::kStageBytes;
    const auto* scale_plane = reinterpret_cast<const __half*>(a.scales);

    // The parent weight row of tile row r (SwiGLU: gate and up rows interleave by 8, as the W4A8
    // tiles'); rows past the call's end read its last row and are dropped at the store.
    const auto parent_row = [&](int r) {
        const int local = w4a4_local_row<kSwiGlu, L::kN>(block_row, r);
        const int up    = kSwiGlu && ((r >> 3) & 1) ? a.up_offset : 0;
        return static_cast<std::int64_t>(a.row_begin + min(local, a.rows - 1) + up);
    };
    const auto issue = [&](int g) {
        unsigned char* st     = i8_shared + (g % L::kStages) * L::kStageBytes;
        const unsigned a_base = i8_smem(st);
        for (int q = tid; q < L::kT * 4; q += L::kThreads) {
            const int row = q >> 2, c = q & 3;
            i8_cp16(a_base + row * 64 + 16 * (c ^ ((row >> 1) & 3)),
                    xq + static_cast<std::int64_t>(t0 + row) * a.k + g * 64 + 16 * c);
        }
        for (int q = tid; q < L::kN * 2; q += L::kThreads) {
            const int row = q >> 1, half = q & 1;
            i8_cp16(i8_smem(st + L::kAStage) + row * 32 + 16 * half,
                    a.codes + (parent_row(row) * a.groups_per_row + g) * 32 + 16 * half);
        }
        if constexpr (L::kHigh == 8) {
            for (int row = tid; row < L::kN; row += L::kThreads)
                i8_cp8(i8_smem(st + L::kHOffset) + row * 8,
                       a.high + (parent_row(row) * a.groups_per_row + g) * 8);
        } else if constexpr (L::kHigh == 16) {
            for (int row = tid; row < L::kN; row += L::kThreads)
                i8_cp16(i8_smem(st + L::kHOffset) + row * 16,
                        a.high + (parent_row(row) * a.groups_per_row + g) * 16);
        }
        for (int q = tid; q < L::kT / 2; q += L::kThreads) {
            if (q < L::kT / 4) {
                i8_cp16(i8_smem(st + L::kSxOffset) + 16 * q,
                        sx_plane + static_cast<std::int64_t>(g) * padded + t0 + 4 * q);
            } else {
                const int r = q - L::kT / 4;
                i8_cp16(i8_smem(st + L::kSumOffset) + 16 * r,
                        sum_plane + static_cast<std::int64_t>(g) * padded + t0 + 4 * r);
            }
        }
        if (g % 8 == 0) {
            for (int row = tid; row < L::kN; row += L::kThreads)
                i8_cp16(i8_smem(ring + ((g / 8) & 1) * L::kN * 16) + row * 16,
                        scale_plane + parent_row(row) * a.groups_per_row + g);
        }
    };

    float acc[4][8][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[m][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < L::kStages - 1; ++s) {
        if (s < groups) issue(s);
        i8_commit();
    }
#pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        i8_wait<L::kStages - 2>();
        __syncthreads();
        if (g + L::kStages - 1 < groups) issue(g + L::kStages - 1);
        i8_commit();

        const unsigned char* st = i8_shared + (g % L::kStages) * L::kStageBytes;
        const float* sxs = reinterpret_cast<const float*>(st + L::kSxOffset);
        const int* sus   = reinterpret_cast<const int*>(st + L::kSumOffset);
        const __half* sws =
            reinterpret_cast<const __half*>(ring + ((g / 8) & 1) * L::kN * 16) + (g & 7);
        float sx[4][2];
        int init[4][2];
        unsigned af[4][2][4];
        const unsigned a_base = i8_smem(st);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int tok = warp_t * 64 + m * 16 + gid;
            sx[m][0]   = sxs[tok];
            sx[m][1]   = sxs[tok + 8];
            init[m][0] = 0x4B400000 - kZero * sus[tok];
            init[m][1] = 0x4B400000 - kZero * sus[tok + 8];
            const int mat = lane >> 3, r = lane & 7;
            const int row = warp_t * 64 + m * 16 + (mat & 1) * 8 + r;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int c = 2 * j + (mat >> 1);
                i8_ldsm_x4(af[m][j], a_base + row * 64 + 16 * (c ^ ((row >> 1) & 3)));
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int brow = warp_n * 64 + j * 8 + gid;
            const uint2 w = *reinterpret_cast<const uint2*>(st + L::kAStage + brow * 32 + 8 * tig);
            unsigned b00, b01, b10, b11;
            if constexpr (Bits == 4) {
                const unsigned x0 = w.x ^ 0x88888888u, x1 = w.y ^ 0x88888888u;
                b00 = x0 & 0x0F0F0F0Fu, b01 = (x0 >> 4) & 0x0F0F0F0Fu;
                b10 = x1 & 0x0F0F0F0Fu, b11 = (x1 >> 4) & 0x0F0F0F0Fu;
            } else if constexpr (Bits == 5) {
                const unsigned h = *reinterpret_cast<const unsigned short*>(
                    st + L::kHOffset + brow * 8 + 2 * tig);
                b00 = ((w.x & 0x0F0F0F0Fu) | i8_spread_bits(i8_even_bits(h))) ^ 0x10101010u;
                b01 = (((w.x >> 4) & 0x0F0F0F0Fu) | i8_spread_bits(i8_even_bits(h >> 1))) ^ 0x10101010u;
                b10 = ((w.y & 0x0F0F0F0Fu) | i8_spread_bits(i8_even_bits(h >> 8))) ^ 0x10101010u;
                b11 = (((w.y >> 4) & 0x0F0F0F0Fu) | i8_spread_bits(i8_even_bits(h >> 9))) ^ 0x10101010u;
            } else {
                const unsigned h =
                    *reinterpret_cast<const unsigned*>(st + L::kHOffset + brow * 16 + 4 * tig);
                b00 = ((w.x & 0x0F0F0F0Fu) | i8_spread_pairs(h)) ^ 0x20202020u;
                b01 = (((w.x >> 4) & 0x0F0F0F0Fu) | i8_spread_pairs(h >> 2)) ^ 0x20202020u;
                b10 = ((w.y & 0x0F0F0F0Fu) | i8_spread_pairs(h >> 16)) ^ 0x20202020u;
                b11 = (((w.y >> 4) & 0x0F0F0F0Fu) | i8_spread_pairs(h >> 18)) ^ 0x20202020u;
            }
            const int ocol  = warp_n * 64 + j * 8 + 2 * tig;
            const float swa = __half2float(sws[ocol * 8]);
            const float swb = __half2float(sws[(ocol + 1) * 8]);
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                int d[4] = {init[m][0], init[m][0], init[m][1], init[m][1]};
                i8_mma(d, af[m][0], b00, b01);
                i8_mma(d, af[m][1], b10, b11);
                acc[m][j][0] = fmaf(__int_as_float(d[0]) - 12582912.0f, sx[m][0] * swa, acc[m][j][0]);
                acc[m][j][1] = fmaf(__int_as_float(d[1]) - 12582912.0f, sx[m][0] * swb, acc[m][j][1]);
                acc[m][j][2] = fmaf(__int_as_float(d[2]) - 12582912.0f, sx[m][1] * swa, acc[m][j][2]);
                acc[m][j][3] = fmaf(__int_as_float(d[3]) - 12582912.0f, sx[m][1] * swb, acc[m][j][3]);
            }
        }
    }
    i8_wait<0>();
    // C fragment: tokens g, g + 8 of each m tile; tile rows 2t, 2t + 1 of each n tile.
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int tok0 = t0 + warp_t * 64 + m * 16 + gid;
#pragma unroll
        for (int j = 0; j < 8; j += (kSwiGlu ? 2 : 1)) {
            const int tile_row = warp_n * 64 + j * 8 + 2 * tig;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int local = w4a4_local_row<kSwiGlu, L::kN>(block_row, tile_row + c);
                if (local >= a.rows) continue;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int token = tok0 + h * 8;
                    if (token >= a.tokens) continue;
                    float value = acc[m][j][h * 2 + c];
                    if constexpr (kSwiGlu) value = generic_gate(a, value) * acc[m][j + 1][h * 2 + c];
                    store_output<E>(a, local, token, value);
                }
            }
        }
    }
#else
    (void)a; (void)xq; (void)sx_plane; (void)sum_plane; (void)padded;
#endif
}

template <int WT, int WN, int Stages, int Bits, Epilogue E>
void launch_tile(const GemmArgs& args, const std::int8_t* codes, const float* scales,
                 const int* sums, int padded, cudaStream_t stream) {
    using L           = I8Tile<WT, WN, Stages, Bits>;
    constexpr int kRowsPerBlock = E == Epilogue::SwiGlu ? L::kN / 2 : L::kN;
    const auto kernel = i8_kernel<WT, WN, Stages, Bits, E>;
    static ::ninferno::PerDevice<bool> sized;
    ::ninferno::allow_dynamic_smem(sized, kernel, L::kSmemBytes);
    const dim3 grid(static_cast<unsigned>(padded / L::kT),
                    static_cast<unsigned>((args.rows + kRowsPerBlock - 1) / kRowsPerBlock));
    kernel<<<grid, L::kThreads, L::kSmemBytes, stream>>>(args, codes, scales, sums, padded);
    CUDA_CHECK(cudaGetLastError());
}

// A tile takes four pipeline stages, or three where four do not fit a block's shared memory (Q6's
// high plane at 256x128: 100 KiB against 99).
template <int WT, int WN, int Bits>
constexpr int kI8Stages =
    I8Tile<WT, WN, 4, Bits>::kSmemBytes <= static_cast<int>(kMaxSharedBytesPerBlock) ? 4 : 3;

template <int Bits, Epilogue E>
void launch_bits(int schedule, const GemmArgs& args, const std::int8_t* codes, const float* scales,
                 const int* sums, int padded, cudaStream_t stream) {
    switch (schedule) {
    case 0:
        launch_tile<4, 2, kI8Stages<4, 2, Bits>, Bits, E>(args, codes, scales, sums, padded, stream);
        return;
    case 1:
        launch_tile<2, 2, kI8Stages<2, 2, Bits>, Bits, E>(args, codes, scales, sums, padded, stream);
        return;
    case 2:
        launch_tile<2, 1, kI8Stages<2, 1, Bits>, Bits, E>(args, codes, scales, sums, padded, stream);
        return;
    default: break;
    }
    throw std::invalid_argument("generic I8: unknown schedule");
}

template <Epilogue E>
void launch_epilogue(int schedule, int bits, const GemmArgs& args, const std::int8_t* codes,
                     const float* scales, const int* sums, int padded, cudaStream_t stream) {
    switch (bits) {
    case 4: launch_bits<4, E>(schedule, args, codes, scales, sums, padded, stream); return;
    case 5: launch_bits<5, E>(schedule, args, codes, scales, sums, padded, stream); return;
    case 6: launch_bits<6, E>(schedule, args, codes, scales, sums, padded, stream); return;
    default: break;
    }
    throw std::invalid_argument("generic I8: Q4, Q5 or Q6 G64 only");
}

} // namespace

void launch_i8(int schedule, int bits, Epilogue epilogue, const GemmArgs& args,
               std::int8_t* codes, float* scales, int* sums, cudaStream_t stream) {
    const int padded         = i8_padded_tokens(args.tokens);
    const std::int64_t tasks = static_cast<std::int64_t>(padded) * (args.k / 64);
    i8_quantize_kernel<<<static_cast<unsigned>((tasks + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(args.x), args.ldx, args.k, args.tokens, padded, codes,
        scales, sums);
    CUDA_CHECK(cudaGetLastError());
    switch (epilogue) {
    case Epilogue::Store:
        launch_epilogue<Epilogue::Store>(schedule, bits, args, codes, scales, sums, padded, stream);
        return;
    case Epilogue::AddResidual:
        launch_epilogue<Epilogue::AddResidual>(schedule, bits, args, codes, scales, sums, padded, stream);
        return;
    case Epilogue::SwiGlu:
        launch_epilogue<Epilogue::SwiGlu>(schedule, bits, args, codes, scales, sums, padded, stream);
        return;
    default: break;
    }
    throw std::invalid_argument("generic I8: unsupported epilogue");
}

} // namespace ninferno::ops::detail::generic
