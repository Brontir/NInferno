// Gemma 4 decode partials without native BF16 MMA (gemma4_attention_decode_launch). On Volta the
// split-K kernel's MMA tiles run as shuffle-bound FMA emulation (V100, Gemma 4 26B: 52 us per
// sliding layer at ~300 keys, 175 us per full layer, ~20 % of the decode step). Here a CTA of four
// warps folds its split's keys in FP32 on the SIMT path: Q of the KV head's rows in shared memory,
// a warp per eight keys of a 32-key tile for the scores, a thread per row for the online softmax,
// and every thread D / 128 head dims of every row for P V. The partials (FP32 unnormalized
// accumulators, m * scale_log2, l) and the split boundaries are those of gemma4_attention_kernel,
// so gemma4_attention_combine_kernel merges them unchanged. The sums run in another order than the
// MMA tiles: a token's bytes differ from the tensor-core route's (the invariant profile takes the
// prefill route, not this one).
#include "ops/softmax_attention/gemma4/volta_decode.h"

#include "core/device.h"
#include "ops/common/fp8_e4m3.cuh"
#include "ops/softmax_attention/gemma4/kernel.cuh"

#include <cuda_fp8.h>

#include <algorithm>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

constexpr int kVoltaThreads = 128;
constexpr int kVoltaTile    = 32; // keys a CTA folds per step
// Q of the call's rows a CTA holds (FP32, shared): Gemma 4's full-attention layers (D = 512, G = 16:
// 8192 floats) fold here too; at 4096 they took the split-K kernel's emulated MMA (GV100, Gemma 4
// 12B: 58 us a call). The rows stay at most the 4096-float bound's or 32: this route's FP32 sums
// land about twice as far from an FP64 oracle as the MMA tiles' (FP8 planes, D = 128: rel_l2
// 3.2e-3 against 1.7e-3), and at 64 rows (T = 16, G = 4) its worst element left the codec test's
// bound (1.25). Volta only (96 KB of shared memory an SM): Turing's 64 KB keeps one 34 KB CTA an
// SM, and its split-K kernel converts BF16 for a native MMA instead of emulating it, so it keeps
// the 4096-float bound.
constexpr int kVoltaQFloats  = 8192;
constexpr int kTuringQFloats = 4096;
constexpr int kVoltaWideRows = 32;

// The most rows (T * G) of one KV head the route takes at head width D with `q_floats` of Q.
constexpr int volta_max_rows(int D, int q_floats) {
    const int narrow = kTuringQFloats / D;
    const int wide   = q_floats / D < kVoltaWideRows ? q_floats / D : kVoltaWideRows;
    return narrow > wide ? narrow : wide;
}

__device__ __forceinline__ float fp8_value(unsigned code) { return e4m3_to_float(code); }

// D / 32 values of a row at element offset `at` (lane's dims) as FP32: BF16, or FP8 codes times
// the row scale.
template <int E>
__device__ __forceinline__ void load_dims(const void* plane, const float* scales, std::int64_t row,
                                          int at, int D, float (&out)[E]) {
    if (scales == nullptr) {
        const auto* p = static_cast<const __nv_bfloat16*>(plane) + row * D + at;
        if constexpr (E >= 8) {
#pragma unroll
            for (int i = 0; i < E; i += 8) {
                const uint4 w           = *reinterpret_cast<const uint4*>(p + i);
                const unsigned words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    out[i + 2 * j]     = __uint_as_float(words[j] << 16);
                    out[i + 2 * j + 1] = __uint_as_float(words[j] & 0xffff0000U);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < E; i += 2) {
                const unsigned w = *reinterpret_cast<const unsigned*>(p + i);
                out[i]           = __uint_as_float(w << 16);
                out[i + 1]       = __uint_as_float(w & 0xffff0000U);
            }
        }
    } else {
        const auto* p     = static_cast<const std::uint8_t*>(plane) + row * D + at;
        const float scale = scales[row];
        if constexpr (E >= 4) {
#pragma unroll
            for (int i = 0; i < E; i += 4) {
                const unsigned w = *reinterpret_cast<const unsigned*>(p + i);
                const float2 lo  = __half22float2(e4m3x2_to_half2(w & 0xffffU));
                const float2 hi  = __half22float2(e4m3x2_to_half2(w >> 16));
                out[i]           = scale * lo.x;
                out[i + 1]       = scale * lo.y;
                out[i + 2]       = scale * hi.x;
                out[i + 3]       = scale * hi.y;
            }
        } else {
#pragma unroll
            for (int i = 0; i < E; ++i) out[i] = scale * fp8_value(p[i]);
        }
    }
}

// MaxRows: rows (T * G) of one KV head the instance holds; D * MaxRows <= kVoltaQFloats.
template <int D, int MaxRows>
__global__
__launch_bounds__(kVoltaThreads) void gemma4_volta_decode_kernel(const Gemma4AttentionArgs args,
                                                                 int bc) {
    constexpr int E = D / 32; // a lane's dims of a key (scores)
    // A thread's dims of the output (P V); below D = 128 only the first D threads hold any.
    constexpr int J       = D >= kVoltaThreads ? D / kVoltaThreads : 1;
    constexpr int Warps   = kVoltaThreads / 32;
    constexpr int PerWarp = kVoltaTile / Warps;
    // Keys loaded at once: four keys' scores (at most 32 floats a lane), eight keys' values. Wider
    // batches unrolled against every row cost registers and code (D = 128, 32 rows: 82 -> 128
    // registers with spills, 2288 -> 13224 instructions at 32 values), and Volta fetches the code
    // of a launch cold (generic_r8.cuh, NINFERNO_R8_ONE_SLOT).
    constexpr int KeyBatch   = E >= 16 ? 2 : (PerWarp < 4 ? PerWarp : 4);
    constexpr int ValueBatch = 8;
    static_assert(PerWarp % KeyBatch == 0);
    const float NegInf    = -CUDART_INF_F;

    __shared__ __align__(16) float q_s[MaxRows * D];
    __shared__ float s_s[MaxRows][kVoltaTile];
    __shared__ float alpha_s[MaxRows];
    __shared__ float m_s[MaxRows];
    __shared__ float l_s[MaxRows];

    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & 31;
    const int warp    = tid >> 5;
    const int split   = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y) % args.kv_heads;
    const Gemma4BatchRow row_ =
        gemma4_batch_row(args, static_cast<int>(blockIdx.y) / args.kv_heads, D);
    const int G          = args.q_heads / args.kv_heads;
    const int tokens     = args.tokens;
    const int rows_total = tokens * G;
    const float sl       = args.scale_log2;

    int lo = 0, hi = 0, begin = 0, end = 0, active = 0;
    gemma4_union_range(row_.positions, tokens, args.window, lo, hi);
    gemma4_split_range(lo, hi, args.splits, bc, split, begin, end, active);
    if (split >= active) return;

    const int p0     = row_.positions[0];
    const int p_call = gemma4_call_end(row_.positions, tokens) - 1;

    for (int i = tid; i < rows_total * D; i += kVoltaThreads) {
        const int r = i / D, d = i - r * D;
        const int t = r / G, head = kv_head * G + (r - t * G);
        q_s[i] =
            __bfloat162float(row_.q[(static_cast<std::int64_t>(t) * args.q_heads + head) * D + d]);
    }
    if (tid < MaxRows) {
        m_s[tid] = NegInf;
        l_s[tid] = 0.0F;
    }

    const int binding        = gemma4_kv_binding(args.kv, row_.batch_row);
    const std::int64_t plane = gemma4_kv_plane_offset(args.kv, binding) +
                               static_cast<std::int64_t>(kv_head) * args.capacity * D;
    const bool fp8 = args.k_scale != nullptr;
    const void* k_plane =
        fp8 ? static_cast<const void*>(static_cast<const std::uint8_t*>(args.cache_k) + plane)
            : static_cast<const void*>(static_cast<const __nv_bfloat16*>(args.cache_k) + plane);
    const void* v_plane =
        fp8 ? static_cast<const void*>(static_cast<const std::uint8_t*>(args.cache_v) + plane)
            : static_cast<const void*>(static_cast<const __nv_bfloat16*>(args.cache_v) + plane);
    const float* k_scales = fp8 ? args.k_scale + plane / D : nullptr;
    const float* v_scales = fp8 ? args.v_scale + plane / D : nullptr;

    float acc[MaxRows][J];
#pragma unroll
    for (int r = 0; r < MaxRows; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) acc[r][j] = 0.0F;
    __syncthreads();

    for (int k0 = begin; k0 < end; k0 += kVoltaTile) {
        // Scores: warp w takes keys k0 + w * PerWarp .. + PerWarp - 1, KeyBatch of them loaded
        // before the first is used: a batch waits on memory once, not once a key (one key at a
        // time, a tile's eight scores and 32 values were 40 serial reads: GV100 Gemma 4 12B, 40 us
        // a sliding layer against its bytes' ~8).
#pragma unroll 1
        for (int i0 = 0; i0 < PerWarp; i0 += KeyBatch) {
            float kf[KeyBatch][E];
#pragma unroll
            for (int b = 0; b < KeyBatch; ++b) {
                const int key = k0 + warp * PerWarp + i0 + b;
                if (key < end) {
                    const int slot = gemma4_kv_slot(args.kv, binding, key);
                    load_dims<E>(k_plane, k_scales, slot, lane * E, D, kf[b]);
                } else {
#pragma unroll
                    for (int e = 0; e < E; ++e) kf[b][e] = 0.0F;
                }
            }
#pragma unroll
            for (int b = 0; b < KeyBatch; ++b) {
                const int kk = warp * PerWarp + i0 + b;
#pragma unroll
                for (int r = 0; r < MaxRows; ++r) {
                    if (r >= rows_total) break;
                    const float* qr = q_s + r * D + lane * E;
                    float dot       = 0.0F;
                    if constexpr (E >= 4) {
#pragma unroll
                        for (int e = 0; e < E; e += 4) {
                            const float4 qv = *reinterpret_cast<const float4*>(qr + e);
                            dot             = fmaf(qv.x, kf[b][e], dot);
                            dot             = fmaf(qv.y, kf[b][e + 1], dot);
                            dot             = fmaf(qv.z, kf[b][e + 2], dot);
                            dot             = fmaf(qv.w, kf[b][e + 3], dot);
                        }
                    } else {
#pragma unroll
                        for (int e = 0; e < E; ++e) dot = fmaf(qr[e], kf[b][e], dot);
                    }
                    dot = warp_sum(dot);
                    if (lane == 0) s_s[r][kk] = dot;
                }
            }
        }
        __syncthreads();
        // Online softmax: a thread per row; masked keys take P = 0.
        if (tid < rows_total) {
            const int r     = tid;
            const int p     = min(p0 + r / G, p_call);
            const int vis_l = args.window > 0 ? max(0, p - args.window + 1) : 0;
            const int vis_h = args.block != 0 ? p_call : p;
            float m         = m_s[r];
            float m_tile    = NegInf;
            for (int kk = 0; kk < kVoltaTile; ++kk) {
                const int key = k0 + kk;
                if (key < end && key >= vis_l && key <= vis_h) m_tile = fmaxf(m_tile, s_s[r][kk]);
            }
            const float m_new = fmaxf(m, m_tile);
            float alpha       = 1.0F;
            float l           = l_s[r];
            if (m_new != NegInf) {
                alpha = m == NegInf ? 0.0F : exp2f((m - m_new) * sl);
                l *= alpha;
                for (int kk = 0; kk < kVoltaTile; ++kk) {
                    const int key  = k0 + kk;
                    const float pv = key < end && key >= vis_l && key <= vis_h
                                         ? exp2f((s_s[r][kk] - m_new) * sl)
                                         : 0.0F;
                    s_s[r][kk]     = pv;
                    l += pv;
                }
            } else {
                for (int kk = 0; kk < kVoltaTile; ++kk) s_s[r][kk] = 0.0F;
            }
            m_s[r]     = m_new;
            l_s[r]     = l;
            alpha_s[r] = alpha;
        }
        __syncthreads();
        // P V: thread t takes dims t * J .. + J - 1 of every row.
#pragma unroll
        for (int r = 0; r < MaxRows; ++r) {
            if (r >= rows_total) break;
            const float a = alpha_s[r];
#pragma unroll
            for (int j = 0; j < J; ++j) acc[r][j] *= a;
        }
        const int keys = tid * J < D ? min(kVoltaTile, end - k0) : 0;
        // ValueBatch keys' values loaded before the first is used; the sums keep the key order.
#pragma unroll 1
        for (int k1 = 0; k1 < keys; k1 += ValueBatch) {
            float vf[ValueBatch][J];
#pragma unroll
            for (int b = 0; b < ValueBatch; ++b) {
                if (k1 + b >= keys) {
#pragma unroll
                    for (int j = 0; j < J; ++j) vf[b][j] = 0.0F;
                    continue;
                }
                const int slot = gemma4_kv_slot(args.kv, binding, k0 + k1 + b);
                if (fp8) {
                    const auto* p = static_cast<const std::uint8_t*>(v_plane) +
                                    static_cast<std::int64_t>(slot) * D + tid * J;
                    const float scale = v_scales[slot];
                    unsigned w        = J == 4   ? *reinterpret_cast<const unsigned*>(p)
                                        : J == 2 ? *reinterpret_cast<const unsigned short*>(p)
                                                 : static_cast<unsigned>(*p);
                    if constexpr (J >= 2) {
#pragma unroll
                        for (int j = 0; j < J; j += 2) {
                            const float2 v =
                                __half22float2(e4m3x2_to_half2((w >> (8 * j)) & 0xffffU));
                            vf[b][j]     = scale * v.x;
                            vf[b][j + 1] = scale * v.y;
                        }
                    } else {
                        vf[b][0] = scale * fp8_value(w);
                    }
                } else {
                    const auto* p = static_cast<const __nv_bfloat16*>(v_plane) +
                                    static_cast<std::int64_t>(slot) * D + tid * J;
#pragma unroll
                    for (int j = 0; j < J; ++j) vf[b][j] = __bfloat162float(p[j]);
                }
            }
#pragma unroll
            for (int b = 0; b < ValueBatch; ++b) {
                if (k1 + b >= keys) break;
#pragma unroll
                for (int r = 0; r < MaxRows; ++r) {
                    if (r >= rows_total) break;
                    const float pr = s_s[r][k1 + b];
#pragma unroll
                    for (int j = 0; j < J; ++j) acc[r][j] = fmaf(pr, vf[b][j], acc[r][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MaxRows; ++r) {
        if (r >= rows_total) break;
        const std::int64_t index =
            static_cast<std::int64_t>(split * args.kv_heads + kv_head) * rows_total + r;
        float* dst = row_.partial_acc + index * D + tid * J;
        if (tid * J < D) {
#pragma unroll
            for (int j = 0; j < J; ++j) dst[j] = acc[r][j];
        }
        if (tid == 0) {
            row_.partial_m[index] = m_s[r] == NegInf ? NegInf : m_s[r] * sl;
            row_.partial_l[index] = l_s[r];
        }
    }
}

template <int D>
constexpr int max_rows() {
    return volta_max_rows(D, kVoltaQFloats);
}

} // namespace

bool gemma4_volta_decode_takes(std::int32_t head_dim, std::int32_t rows) {
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) return false;
    if (rows < 1) return false;
    try {
        const auto& d = device_class();
        const int q_floats = d.major == 7 && d.minor < 5 ? kVoltaQFloats : kTuringQFloats;
        if (rows > volta_max_rows(head_dim, q_floats)) return false;
        // Turing's split-K kernel runs its BF16 MMA through f16 conversions: on the RTX 2080 Ti
        // this kernel decodes faster (SmolLM3 117.7 -> 167.5 tok/s, 6K context 70.5 -> 98.6;
        // granite-3.3-8b 65.5 -> 79.5, 8.7K context 20.0 -> 30.2).
        return !d.features.mma_bf16_native;
    } catch (const std::exception&) { return false; }
}

namespace {

using VoltaDecodeKernel = void (*)(Gemma4AttentionArgs, int);

// The instance holding the fewest rows the call has (T * G): its accumulators and Q tile are
// sized by MaxRows, and the registers and shared memory they take bound how many CTAs an SM keeps
// resident (RTX 2080 Ti, D = 128: 32 rows took 128 registers a thread for a call of 4, 72 at 4).
template <int D>
VoltaDecodeKernel kernel_for_rows(std::int32_t rows) {
    if (rows <= 4) return gemma4_volta_decode_kernel<D, 4>;
    if (rows <= 8) return gemma4_volta_decode_kernel<D, 8>;
    if constexpr (max_rows<D>() >= 16) {
        if (rows <= 16) return gemma4_volta_decode_kernel<D, 16>;
    }
    if constexpr (max_rows<D>() >= 32) {
        if (rows <= 32) return gemma4_volta_decode_kernel<D, 32>;
    }
    return gemma4_volta_decode_kernel<D, max_rows<D>()>;
}

VoltaDecodeKernel kernel_for(std::int32_t head_dim, std::int32_t rows) {
    switch (head_dim) {
    case 64: return kernel_for_rows<64>(rows);
    case 128: return kernel_for_rows<128>(rows);
    case 256: return kernel_for_rows<256>(rows);
    case 512: return kernel_for_rows<512>(rows);
    default:
        throw std::invalid_argument("gemma4_attention: the Volta decode route takes D = 64..512");
    }
}

} // namespace

int gemma4_volta_decode_resident_ctas(std::int32_t head_dim, std::int32_t rows) {
    const int per_sm = kernel_resident_ctas_per_sm(
        reinterpret_cast<const void*>(kernel_for(head_dim, rows)), kVoltaThreads, 0);
    return std::max(1, per_sm) * device_class().sms;
}

void gemma4_volta_decode_launch(const Gemma4AttentionArgs& args, std::int32_t head_dim,
                                std::int32_t bc, dim3 grid, cudaStream_t stream) {
    const auto kernel = kernel_for(head_dim, args.tokens * (args.q_heads / args.kv_heads));
    kernel<<<grid, kVoltaThreads, 0, stream>>>(args, bc);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
