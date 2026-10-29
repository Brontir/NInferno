// Volta's decode GEMV for the ggml Q4_K / Q5_K / Q6_K, Q8 G32 and BF16 formats: a warp per output
// row, lanes over a superblock (lane l: its values 8 l .. 8 l + 7), codes decoded straight to FP32
// and summed against FP32 activations. Without tensor mma.sync the register-MMA tiles emulate the
// MMA fragment layout on FP32 FMAs; at T <= 4 this kernel reads the same bytes in row order.
// run_route takes it for a lane-local register-MMA call (V100 Gemma 4 31B Q5_K head: 5.6 -> ~1.6
// ms, decode 25.5 -> 28.4 tok/s).
#include "core/device.h"
#include "core/device_class.h"
#include "ops/generic/generic_gemm.cuh"
#include "ops/generic/generic_gemm_internal.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

// The lane's eight values of a superblock as code bytes, with the FP32 factors of w = ds q - dm.
struct LaneCodes {
    unsigned q0, q1;
    float ds, dm;
};

__device__ __forceinline__ float half_of(unsigned bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// Q4_K / Q5_K: 144 / 176-byte blocks in the codes plane (d, dmin, twelve scale bytes, Q5_K's 32
// high-bit bytes, 128 nibble bytes). Lane l: sub-block l / 4.
template <int Bits>
struct K45 {
    static constexpr int kBlockBytes = Bits == 4 ? 144 : 176;
    uint4 head;
    uint2 qs, qh;

    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int lane) {
        const std::uint8_t* b =
            a.codes + (static_cast<std::int64_t>(row) * a.groups_per_row + sb) * kBlockBytes;
        const int j = lane >> 2, p = 8 * (lane & 3);
        head = *reinterpret_cast<const uint4*>(b);
        qs   = *reinterpret_cast<const uint2*>(b + (Bits == 4 ? 16 : 48) + 32 * (j >> 1) + p);
        qh   = Bits == 5 ? *reinterpret_cast<const uint2*>(b + 16 + p) : make_uint2(0, 0);
    }

    __device__ __forceinline__ LaneCodes codes(int lane) const {
        const int j = lane >> 2, sh = 4 * (j & 1);
        // Sub-block j's 6-bit scale and min (ggml get_scale_min_k4).
        const unsigned s0 = head.y, s4 = head.z, s8 = head.w;
        unsigned sc, mn;
        if (j < 4) {
            sc = (s0 >> (8 * j)) & 63U;
            mn = (s4 >> (8 * j)) & 63U;
        } else {
            const int jj = 8 * (j - 4);
            sc           = ((s8 >> jj) & 0x0fU) | (((s0 >> jj) >> 6 & 3U) << 4);
            mn           = ((s8 >> jj) >> 4 & 0x0fU) | (((s4 >> jj) >> 6 & 3U) << 4);
        }
        LaneCodes c;
        c.q0 = (qs.x >> sh) & 0x0f0f0f0fU;
        c.q1 = (qs.y >> sh) & 0x0f0f0f0fU;
        if constexpr (Bits == 5) {
            c.q0 |= ((qh.x >> j) & 0x01010101U) << 4;
            c.q1 |= ((qh.y >> j) & 0x01010101U) << 4;
        }
        c.ds = half_of(head.x) * static_cast<float>(sc);
        c.dm = half_of(head.x >> 16) * static_cast<float>(mn);
        return c;
    }
};

// Q6_K planes: 128 ql then 64 qh bytes a superblock (codes), 16 int8 scales (high), binary16 d
// (scales). Lane l: half h = l / 16, quarter qq = (l / 4) % 4, values 8 (l % 4) .. + 7 of it.
struct K6 {
    uint2 ql, qh;
    int sc;
    unsigned d;

    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int lane) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + sb;
        const int h = lane >> 4, qq = (lane >> 2) & 3, l0 = 8 * (lane & 3);
        const std::uint8_t* b = a.codes + blk * 192;
        ql                    = *reinterpret_cast<const uint2*>(b + 64 * h + 32 * (qq & 1) + l0);
        qh                    = *reinterpret_cast<const uint2*>(b + 128 + 32 * h + l0);
        sc = static_cast<signed char>(a.high[blk * 16 + 8 * h + 2 * qq + (l0 >> 4)]);
        d  = reinterpret_cast<const unsigned short*>(a.scales)[blk];
    }

    __device__ __forceinline__ LaneCodes codes(int lane) const {
        const int qq = (lane >> 2) & 3, sh = 4 * (qq >> 1);
        LaneCodes c;
        c.q0 = ((ql.x >> sh) & 0x0f0f0f0fU) | (((qh.x >> (2 * qq)) & 0x03030303U) << 4);
        c.q1 = ((ql.y >> sh) & 0x0f0f0f0fU) | (((qh.y >> (2 * qq)) & 0x03030303U) << 4);
        c.ds = half_of(d) * static_cast<float>(sc);
        c.dm = 32.0F * c.ds;
        return c;
    }
};

// Q8 G32: int8 codes in row order (codes), a binary16 scale a group of 32 (scales). Lane l: values
// 8 l .. + 7 of the superblock, group 8 sb + l / 4; codes biased by 128.
struct Q8G32 {
    uint2 q;
    unsigned d;

    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int lane) {
        const std::int64_t g =
            static_cast<std::int64_t>(row) * a.groups_per_row + 8 * sb + (lane >> 2);
        q = *reinterpret_cast<const uint2*>(a.codes + g * 32 + 8 * (lane & 3));
        d = reinterpret_cast<const unsigned short*>(a.scales)[g];
    }

    __device__ __forceinline__ LaneCodes codes(int) const {
        LaneCodes c;
        c.q0 = q.x ^ 0x80808080U;
        c.q1 = q.y ^ 0x80808080U;
        c.ds = half_of(d);
        c.dm = 128.0F * c.ds;
        return c;
    }
};

// BF16 rows (codes, K values a row). Lane l: values 8 l .. + 7 of the 256-value step, zero past K
// (K a multiple of 8, not of 256: gpt-oss's 2880).
struct Bf16Rows {
    uint4 w;

    __device__ __forceinline__ void load(const GemmArgs& a, int row, int sb, int lane) {
        const int k0 = 256 * sb + 8 * lane;
        w            = k0 < a.k ? *reinterpret_cast<const uint4*>(
                           a.codes + (static_cast<std::int64_t>(row) * a.k + k0) * 2)
                                : make_uint4(0, 0, 0, 0);
    }

    // Never called: the kernel takes weights() for this format (kDirectOf).
    __device__ __forceinline__ LaneCodes codes(int) const { return {}; }

    __device__ __forceinline__ void weights(float (&v)[8]) const {
        const unsigned words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[2 * i]     = __uint_as_float(words[i] << 16);
            v[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000U);
        }
    }
};

template <class F>
inline constexpr bool kDirectOf = false;
template <>
inline constexpr bool kDirectOf<Bf16Rows> = true;

// Code byte i of `w` as an FP32 value: 2^23 + q, less 2^23.
__device__ __forceinline__ float code_f32(unsigned w, unsigned sel) {
    return __uint_as_float(__byte_perm(w, 0x4B000000U, sel)) - 8388608.0F;
}

// R rows a warp (R > 1 for a call of many rows: the LM head): the activation's words, their FP32
// values and their sum (the min's factor) are a function of the superblock and the token only,
// so a warp takes them once for its R rows instead of once a row. Each row's sums run in the
// same order either way: the bytes do not depend on R.
template <class Format, Epilogue E, int NT, int R>
__global__ __launch_bounds__(256) void volta_kquant_gemv_kernel(GemmArgs a) {
    constexpr int kSets = E == Epilogue::SwiGlu ? 2 : 1;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int l0 =
        (static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5)) * R;
    if (l0 >= a.rows) return;
    const auto* x = static_cast<const __nv_bfloat16*>(a.x);
    const int sb  = (a.k + 255) >> 8;
    float acc[kSets][R][NT];
#pragma unroll
    for (int s = 0; s < kSets; ++s)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int n = 0; n < NT; ++n) acc[s][r][n] = 0.0F;

#pragma unroll
    for (int s = 0; s < kSets; ++s) {
        const int row0 = a.row_begin + l0 + (s == 1 ? a.up_offset : 0);
        Format cur[R];
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (l0 + r < a.rows) cur[r].load(a, row0 + r, 0, lane);
#pragma unroll 1
        for (int b = 0; b < sb; ++b) {
            Format nxt[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                nxt[r] = cur[r];
                if (b + 1 < sb && l0 + r < a.rows) nxt[r].load(a, row0 + r, b + 1, lane);
            }
            const int k0 = 256 * b + 8 * lane;
            // The lane's eight activation values of each token, and (codes) their sum.
            float xv[NT][8], sx[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                sx[n] = 0.0F;
                // Only a BF16 row ends inside a step (K a multiple of 8); a token past the call's
                // is never read.
                if constexpr (kDirectOf<Format>) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) xv[n][i] = 0.0F;
                    if (n >= a.tokens || k0 >= a.k) continue;
                } else {
                    if (n >= a.tokens) break;
                }
                const uint4 w4 =
                    *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(n) * a.ldx + k0);
                const unsigned xw[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    xv[n][2 * i]     = __uint_as_float(xw[i] << 16);
                    xv[n][2 * i + 1] = __uint_as_float(xw[i] & 0xffff0000U);
                    if constexpr (!kDirectOf<Format>) sx[n] += xv[n][2 * i] + xv[n][2 * i + 1];
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (l0 + r >= a.rows) break;
                if constexpr (kDirectOf<Format>) {
                    float w[8];
                    cur[r].weights(w);
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        if (n >= a.tokens || k0 >= a.k) break;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            acc[s][r][n] = fmaf(w[2 * i], xv[n][2 * i], acc[s][r][n]);
                            acc[s][r][n] = fmaf(w[2 * i + 1], xv[n][2 * i + 1], acc[s][r][n]);
                        }
                    }
                } else {
                    const LaneCodes c = cur[r].codes(lane);
                    float q[8];
                    q[0] = code_f32(c.q0, 0x7540U);
                    q[1] = code_f32(c.q0, 0x7541U);
                    q[2] = code_f32(c.q0, 0x7542U);
                    q[3] = code_f32(c.q0, 0x7543U);
                    q[4] = code_f32(c.q1, 0x7540U);
                    q[5] = code_f32(c.q1, 0x7541U);
                    q[6] = code_f32(c.q1, 0x7542U);
                    q[7] = code_f32(c.q1, 0x7543U);
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        if (n >= a.tokens) break;
                        float sqx = 0.0F;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            sqx = fmaf(q[2 * i], xv[n][2 * i], sqx);
                            sqx = fmaf(q[2 * i + 1], xv[n][2 * i + 1], sqx);
                        }
                        acc[s][r][n] = fmaf(c.ds, sqx, fmaf(-c.dm, sx[n], acc[s][r][n]));
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) cur[r] = nxt[r];
        }
    }
#pragma unroll
    for (int s = 0; s < kSets; ++s)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    acc[s][r][n] += __shfl_xor_sync(0xffffffffU, acc[s][r][n], o);
    if (lane != 0) return;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (l0 + r >= a.rows) break;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            if (n >= a.tokens) break;
            float value = acc[0][r][n];
            if constexpr (E == Epilogue::SwiGlu)
                value = generic_gate(a, value) * acc[kSets - 1][r][n];
            store_output<E>(a, l0 + r, n, value);
        }
    }
}

template <class Format, Epilogue E, int R>
void launch_rows(const GemmArgs& args, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>((args.rows + 8 * R - 1) / (8 * R));
    if (args.tokens == 1) {
        volta_kquant_gemv_kernel<Format, E, 1, R><<<blocks, 256, 0, stream>>>(args);
    } else if (args.tokens == 2) {
        volta_kquant_gemv_kernel<Format, E, 2, R><<<blocks, 256, 0, stream>>>(args);
    } else {
        volta_kquant_gemv_kernel<Format, E, 4, R><<<blocks, 256, 0, stream>>>(args);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Four rows a warp where the grid still holds four CTAs an SM or more (the LM head's 248320 rows:
// 7760 CTAs), else one.
template <class Format, Epilogue E>
void launch_format(const GemmArgs& args, cudaStream_t stream) {
    const auto quad_blocks = static_cast<std::int64_t>((args.rows + 31) / 32);
    if (quad_blocks >= 4LL * device_class().sms) {
        launch_rows<Format, E, 4>(args, stream);
    } else {
        launch_rows<Format, E, 1>(args, stream);
    }
}

template <class Format>
void launch_epilogue(Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    switch (epilogue) {
    case Epilogue::Store:
        launch_format<Format, Epilogue::Store>(args, stream);
        return;
    case Epilogue::AddResidual:
        launch_format<Format, Epilogue::AddResidual>(args, stream);
        return;
    case Epilogue::SwiGlu:
        launch_format<Format, Epilogue::SwiGlu>(args, stream);
        return;
    }
}

} // namespace

bool volta_gemv_takes(WeightCodec codec, const GemmArgs& args) {
    if (args.lane_local == 0 || args.tokens < 1 || args.tokens > 4 || args.join_previous != 0) {
        return false;
    }
    if (codec == WeightCodec::Bf16) return (args.k % 8) == 0;
    return (args.k % 256) == 0 && (codec == WeightCodec::GgmlQ4K || codec == WeightCodec::GgmlQ5K ||
                                   codec == WeightCodec::GgmlQ6K || codec == WeightCodec::Q8);
}

void launch_volta_gemv(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                       cudaStream_t stream) {
    switch (codec) {
    case WeightCodec::GgmlQ4K:
        launch_epilogue<K45<4>>(epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ5K:
        launch_epilogue<K45<5>>(epilogue, args, stream);
        return;
    case WeightCodec::Q8:
        launch_epilogue<Q8G32>(epilogue, args, stream);
        return;
    case WeightCodec::Bf16:
        launch_epilogue<Bf16Rows>(epilogue, args, stream);
        return;
    default:
        launch_epilogue<K6>(epilogue, args, stream);
        return;
    }
}

} // namespace ninferno::ops::detail::generic
