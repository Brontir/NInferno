// ninferno::ops - Gemma 4 fused elementwise Ops: Q/K/V preparation, GeGLU, sandwich-norm residual
// and the join of the two feed-forward branches. Contracts in ninferno/ops/gemma4_fused.h.
#include "core/pdl.cuh"
#include "ninferno/ops/gemma4_fused.h"

#include "core/device.h" // CUDA_CHECK
#include "core/device_class.h"
#include "ops/common/dflash_rope.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/gemma4/fp8_row.cuh"
#include "ops/softmax_attention/gemma4/kv_address.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::ops {
namespace {

struct QkvParams {
    const __nv_bfloat16* q_src;
    const __nv_bfloat16* k_src;
    const __nv_bfloat16* v_src;
    std::int64_t q_stride; // elements between tokens
    std::int64_t k_stride;
    std::int64_t v_stride;
    const __nv_bfloat16* q_weight;
    const __nv_bfloat16* k_weight;
    const std::int32_t* positions;
    __nv_bfloat16* q_out;
    __nv_bfloat16* k_out;
    __nv_bfloat16* v_out;
    float eps;
    int q_heads;
    int kv_heads;
    int rotary_pairs;
    bool v_norm; // Gemma normalizes V without a learned scale; a Qwen3-style layer leaves V raw
    // Nonzero: k_out/v_out are a layer's cache planes and token t's rows go where
    // gemma4_kv_append would write them for batch row t / row_tokens (see kv_address.h), or
    // nowhere past the row's stored count.
    // Zero: [D, heads, T].
    std::int64_t kv_capacity;
    detail::Gemma4KvAddress kv;
    int row_tokens;
    // theta^(-2i/D) for the rotary pairs, formed on the host in FP64.
    double inverse_frequency[256];
    // FP8 cache planes (k_scale non-null): codes and FP32 row scales instead of k_out/v_out.
    std::uint8_t* k_codes;
    std::uint8_t* v_codes;
    float* k_scale;
    float* v_scale;
};

constexpr int kQkvThreads = 256;

// One block per token. Lane l of a warp holds the E = D/32 contiguous features [l*E, (l+1)*E) of
// one head row, so the RoPE partner of a feature in lane l < 16 sits at the same register index
// in lane l + 16: the rotation needs one xor-16 shuffle per register and no shared staging.
// E contiguous BF16 features per lane: 16-byte vectors when E is a multiple of 8, one 8-byte
// vector at E = 4 (D = 128).
template <int E>
__device__ __forceinline__ void load_features(const __nv_bfloat16* src, float (&x)[E]) {
    if constexpr (E % 8 == 0) {
#pragma unroll
        for (int v = 0; v < E / 8; ++v) {
            const uint4 raw = *reinterpret_cast<const uint4*>(src + v * 8);
            const auto* h   = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f       = __bfloat1622float2(h[j]);
                x[v * 8 + 2 * j]     = f.x;
                x[v * 8 + 2 * j + 1] = f.y;
            }
        }
    } else {
        static_assert(E == 4, "a lane holds 4 or a multiple of 8 features");
        const uint2 raw = *reinterpret_cast<const uint2*>(src);
        const auto* h   = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float2 f = __bfloat1622float2(h[j]);
            x[2 * j]       = f.x;
            x[2 * j + 1]   = f.y;
        }
    }
}

template <int E>
__device__ __forceinline__ void store_features(__nv_bfloat16* out, const float (&x)[E]) {
    if constexpr (E % 8 == 0) {
#pragma unroll
        for (int v = 0; v < E / 8; ++v) {
            uint4 raw;
            auto* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                h[j] = __floats2bfloat162_rn(x[v * 8 + 2 * j], x[v * 8 + 2 * j + 1]);
            }
            *reinterpret_cast<uint4*>(out + v * 8) = raw;
        }
    } else {
        uint2 raw;
        auto* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
        for (int j = 0; j < 2; ++j) { h[j] = __floats2bfloat162_rn(x[2 * j], x[2 * j + 1]); }
        *reinterpret_cast<uint2*>(out) = raw;
    }
}

template <int D>
__global__ void __launch_bounds__(kQkvThreads) gemma4_qkv_prepare_kernel(QkvParams p) {
    constexpr int E = D / 32;
    __shared__ float cosines[D / 2];
    __shared__ float sines[D / 2];

    const int t      = blockIdx.x;
    const double pos = static_cast<double>(p.positions[t]);
    for (int i = threadIdx.x; i < D / 2; i += kQkvThreads) {
        if (i < p.rotary_pairs) {
            rope_angle_sincos(pos * p.inverse_frequency[i], &sines[i], &cosines[i]);
        } else {
            cosines[i] = 1.0f;
            sines[i]   = 0.0f;
        }
    }
    __syncthreads();
    // Under a programmatic launch (join_previous) the table above was built under the projection
    // that writes the sources; every read of them comes after this wait.
    pdl::wait_for_dependencies();

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int rows  = p.q_heads + 2 * p.kv_heads;
    const bool high = lane >= 16;
    // gridDim.y blocks share one token's rows; each rebuilds the small angle table.
    constexpr int kWarps = kQkvThreads / 32;
    for (int r = blockIdx.y * kWarps + warp; r < rows; r += gridDim.y * kWarps) {
        const __nv_bfloat16* src;
        const __nv_bfloat16* weight;
        __nv_bfloat16* out = nullptr;
        std::uint8_t* codes = nullptr; // FP8 cache: the row's codes and scale
        float* scale        = nullptr;
        bool rotate = true;
        bool store  = true;
        if (r < p.q_heads) {
            src    = p.q_src + t * p.q_stride + static_cast<std::int64_t>(r) * D;
            weight = p.q_weight;
            out    = p.q_out + (static_cast<std::int64_t>(t) * p.q_heads + r) * D;
        } else {
            const bool key = r < p.q_heads + p.kv_heads;
            const int g    = key ? r - p.q_heads : r - p.q_heads - p.kv_heads;
            src            = (key ? p.k_src + t * p.k_stride : p.v_src + t * p.v_stride) +
                  static_cast<std::int64_t>(g) * D;
            weight = key ? p.k_weight : nullptr;
            rotate = key;
            if (p.kv_capacity > 0) {
                const int batch_row = t / p.row_tokens;
                store = detail::gemma4_kv_stores(p.kv, batch_row, t - batch_row * p.row_tokens);
                const int binding = detail::gemma4_kv_binding(p.kv, batch_row);
                const int slot    = detail::gemma4_kv_slot(p.kv, binding, p.positions[t]);
                const std::int64_t row = detail::gemma4_kv_plane_offset(p.kv, binding) / D +
                                         static_cast<std::int64_t>(g) * p.kv_capacity + slot;
                if (p.k_scale != nullptr) {
                    codes = (key ? p.k_codes : p.v_codes) + row * D;
                    scale = (key ? p.k_scale : p.v_scale) + row;
                } else {
                    out = (key ? p.k_out : p.v_out) + row * D;
                }
            } else {
                out =
                    (key ? p.k_out : p.v_out) + (static_cast<std::int64_t>(t) * p.kv_heads + g) * D;
            }
        }

        float x[E];
        load_features<E>(src + lane * E, x);
        // Warp-uniform: every lane of the warp works on the same row.
        if (rotate || p.v_norm) {
            float squares = 0.0f;
#pragma unroll
            for (int j = 0; j < E; ++j) { squares += x[j] * x[j]; }
            squares         = warp_sum(squares);
            const float inv = rsqrtf(squares / D + p.eps);
#pragma unroll
            for (int j = 0; j < E; ++j) {
                x[j] *= inv;
                if (weight != nullptr) { x[j] *= __bfloat162float(weight[lane * E + j]); }
            }
        }
        if (rotate) {
#pragma unroll
            for (int j = 0; j < E; ++j) {
                const float partner = __shfl_xor_sync(kFullWarpMask, x[j], 16);
                const int pair      = (lane & 15) * E + j;
                const float c       = cosines[pair];
                const float s       = sines[pair];
                x[j] = high ? fmaf(x[j], c, partner * s) : fmaf(x[j], c, -partner * s);
            }
        }
        if (store && codes != nullptr) {
            detail::gemma4_fp8_store_row<E>(codes, scale, x, lane);
        } else if (store) {
            store_features<E>(out + lane * E, x);
        }
    }
}

__device__ __forceinline__ float gelu_tanh(float z) {
    constexpr float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float inner      = kAlpha * fmaf(0.044715f * z, z * z, z);
    return 0.5f * z * (1.0f + tanhf(inner));
}

__global__ void gelu_tanh_mul_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                     __nv_bfloat16* __restrict__ out, int width,
                                     std::int64_t pairs) {
    // Two features per thread; width is even, so a pair never straddles a column.
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < pairs; index += stride) {
        const std::int64_t element = 2 * index;
        const std::int64_t column  = element / width;
        const int row              = static_cast<int>(element - column * width);
        const __nv_bfloat16* base  = gate_up + column * 2 * width;
        const float2 gate =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(base + row));
        const float2 up =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(base + width + row));
        *reinterpret_cast<__nv_bfloat162*>(out + element) =
            __floats2bfloat162_rn(gelu_tanh(gate.x) * up.x, gelu_tanh(gate.y) * up.y);
    }
}

// GeGLU over separate gate and multiplier planes; `up` may be a row slice with a padded token
// stride (a layer's rows of the per-layer inputs). Strides are in elements.
__global__ void gelu_tanh_mul2_kernel(const __nv_bfloat16* __restrict__ gate,
                                      std::int64_t gate_stride,
                                      const __nv_bfloat16* __restrict__ up, std::int64_t up_stride,
                                      __nv_bfloat16* __restrict__ out, int width,
                                      std::int64_t pairs) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < pairs; index += stride) {
        const std::int64_t element = 2 * index;
        const std::int64_t column  = element / width;
        const int row              = static_cast<int>(element - column * width);
        const float2 g             = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(gate + column * gate_stride + row));
        const float2 u = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(up + column * up_stride + row));
        *reinterpret_cast<__nv_bfloat162*>(out + element) =
            __floats2bfloat162_rn(gelu_tanh(g.x) * u.x, gelu_tanh(g.y) * u.y);
    }
}

__global__ void scale_bf16_kernel(__nv_bfloat162* __restrict__ x, float scale,
                                  std::int64_t pairs) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < pairs; index += stride) {
        const float2 v = __bfloat1622float2(x[index]);
        x[index]       = __floats2bfloat162_rn(v.x * scale, v.y * scale);
    }
}

// Per-layer embedding inputs: one warp per (layer slice, token), P features of it.
struct PleParams {
    const __nv_bfloat16* projection; // [L*P, T]
    const __nv_bfloat16* table_rows; // [L*P, T]
    const __nv_bfloat16* weight;     // [P]
    __nv_bfloat16* out;              // [L*P, T], may be `projection`
    int width;                       // P
    int slices;                      // L
    std::int64_t tokens;
    float eps, projection_scale, embedding_scale, input_scale;
};

constexpr int kPleThreads = 256;

__device__ __forceinline__ float bf16_round(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

__global__ void __launch_bounds__(kPleThreads) gemma4_ple_inputs_kernel(PleParams p) {
    const int lane            = threadIdx.x & 31;
    const std::int64_t warp   = static_cast<std::int64_t>(blockIdx.x) * (kPleThreads / 32) +
                              (threadIdx.x >> 5);
    const std::int64_t warps  = p.tokens * p.slices;
    if (warp >= warps) { return; }
    const std::int64_t column = warp / p.slices;
    const int slice           = static_cast<int>(warp - column * p.slices);
    const std::int64_t base   = column * p.slices * p.width + static_cast<std::int64_t>(slice) * p.width;
    float sum = 0.0f;
    for (int i = 2 * lane; i < p.width; i += 64) {
        const float2 v = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(p.projection + base + i));
        const float a0 = bf16_round(v.x * p.projection_scale);
        const float a1 = bf16_round(v.y * p.projection_scale);
        sum += a0 * a0 + a1 * a1;
    }
    const float inverse = rsqrtf(warp_sum(sum) / p.width + p.eps);
    for (int i = 2 * lane; i < p.width; i += 64) {
        const float2 v = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(p.projection + base + i));
        const float2 e = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(p.table_rows + base + i));
        const float2 w =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p.weight + i));
        const float n0 = bf16_round(bf16_round(v.x * p.projection_scale) * inverse * w.x);
        const float n1 = bf16_round(bf16_round(v.y * p.projection_scale) * inverse * w.y);
        const float s0 = bf16_round(n0 + bf16_round(e.x * p.embedding_scale));
        const float s1 = bf16_round(n1 + bf16_round(e.y * p.embedding_scale));
        *reinterpret_cast<__nv_bfloat162*>(p.out + base + i) =
            __floats2bfloat162_rn(s0 * p.input_scale, s1 * p.input_scale);
    }
}

struct ResidualNormParams {
    __nv_bfloat16* x;
    const __nv_bfloat16* a;
    const __nv_bfloat16* a_weight; // null: the branch is `a` itself
    const __nv_bfloat16* b;
    const __nv_bfloat16* b_weight;
    const __nv_bfloat16* post_weight;
    const __nv_bfloat16* out_weight1;
    __nv_bfloat16* out1;
    const __nv_bfloat16* out_weight2; // null: one output
    __nv_bfloat16* out2;
    const __nv_bfloat16* out_weight3; // null: no third output
    __nv_bfloat16* out3;
    int width;
    float eps;
    float scale;
};

constexpr int kResidualThreads  = 512;
constexpr int kResidualMaxWidth = 8192;

// One block per column. The branch, then the updated residual, are staged in shared memory as
// FP32 so each of the up to four reductions reads on-chip data; the residual is staged *after*
// its BF16 rounding, so the output norms see exactly what a later RMSNorm of x would read.
template <bool kTwoBranches>
__global__ void __launch_bounds__(kResidualThreads)
    gemma4_residual_norm_kernel(ResidualNormParams p) {
    extern __shared__ float stage[]; // [width]
    __shared__ float scratch[kResidualThreads / 32];
    const int width         = p.width;
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * width;

    float branch_inverse = 0.0f;
    if constexpr (kTwoBranches) {
        float sa = 0.0f, sb = 0.0f;
        for (int i = threadIdx.x; i < width; i += kResidualThreads) {
            const float va = __bfloat162float(p.a[base + i]);
            const float vb = __bfloat162float(p.b[base + i]);
            sa += va * va;
            sb += vb * vb;
        }
        const float ia = rsqrtf(block_sum<kResidualThreads>(sa, scratch) / width + p.eps);
        const float ib = rsqrtf(block_sum<kResidualThreads>(sb, scratch) / width + p.eps);
        float st       = 0.0f;
        for (int i = threadIdx.x; i < width; i += kResidualThreads) {
            const float t = __bfloat162float(p.a[base + i]) * ia * __bfloat162float(p.a_weight[i]) +
                            __bfloat162float(p.b[base + i]) * ib * __bfloat162float(p.b_weight[i]);
            stage[i] = t;
            st += t * t;
        }
        branch_inverse = rsqrtf(block_sum<kResidualThreads>(st, scratch) / width + p.eps);
    } else {
        float st = 0.0f;
        for (int i = threadIdx.x; i < width; i += kResidualThreads) {
            const float t = __bfloat162float(p.a[base + i]);
            stage[i]      = t;
            st += t * t;
        }
        branch_inverse = rsqrtf(block_sum<kResidualThreads>(st, scratch) / width + p.eps);
    }

    float sx = 0.0f;
    for (int i = threadIdx.x; i < width; i += kResidualThreads) {
        const float updated = (__bfloat162float(p.x[base + i]) +
                               stage[i] * branch_inverse * __bfloat162float(p.post_weight[i])) *
                              p.scale;
        const __nv_bfloat16 stored = __float2bfloat16(updated);
        p.x[base + i]              = stored;
        const float rounded        = __bfloat162float(stored);
        stage[i]                   = rounded;
        sx += rounded * rounded;
    }
    if (p.out1 == nullptr) { return; } // update only: nothing reads a normalized x
    const float x_inverse = rsqrtf(block_sum<kResidualThreads>(sx, scratch) / width + p.eps);
    for (int i = threadIdx.x; i < width; i += kResidualThreads) {
        const float normed = stage[i] * x_inverse;
        p.out1[base + i]   = __float2bfloat16(normed * __bfloat162float(p.out_weight1[i]));
        if (p.out_weight2 != nullptr) {
            p.out2[base + i] = __float2bfloat16(normed * __bfloat162float(p.out_weight2[i]));
        }
        if (p.out_weight3 != nullptr) {
            p.out3[base + i] = __float2bfloat16(normed * __bfloat162float(p.out_weight3[i]));
        }
    }
}

// Two partial sums reduced together: one pass of shuffles and one barrier pair for both.
__device__ __forceinline__ float2 block_sum2(float2 value, float2* scratch) {
    constexpr int kWarps = kResidualThreads / 32;
    const int lane       = threadIdx.x & 31;
    const int warp       = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value.x += __shfl_xor_sync(kFullWarpMask, value.x, offset);
        value.y += __shfl_xor_sync(kFullWarpMask, value.y, offset);
    }
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float2 total = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        total.x += scratch[w].x;
        total.y += scratch[w].y;
    }
    __syncthreads();
    return total;
}

__device__ __forceinline__ void unpack8(const uint4& raw, float* out) {
    const auto* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        out[2 * j]     = f.x;
        out[2 * j + 1] = f.y;
    }
}

// Widths that are whole 16-byte vectors (every Gemma 4 width). The one-block-per-column schedule
// above spends most of a decode column on dependent round trips to memory: here every input and
// weight vector a thread will need is requested in one burst into registers, the two branch
// norms share one reduction, and the result never visits shared memory. Same formula, same
// single rounding of x; only the reduction order differs.
constexpr int kResidualVectors = kResidualMaxWidth / 8 / kResidualThreads; // per thread, max

template <bool kTwoBranches>
__global__ void __launch_bounds__(kResidualThreads)
    gemma4_residual_norm_vector_kernel(ResidualNormParams p) {
    // A programmatic dependent (a decode projection of the output) may launch now: it waits for
    // this kernel before it reads, and streams its weights meanwhile.
    if (threadIdx.x == 0) pdl::trigger_dependents();
    __shared__ float2 scratch2[kResidualThreads / 32];
    __shared__ float scratch[kResidualThreads / 32];
    const int vectors       = p.width / 8;
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * p.width;
    const bool second       = p.out_weight2 != nullptr;
    const bool third        = p.out_weight3 != nullptr;
    const auto load         = [](const __nv_bfloat16* src, int v) {
        return *reinterpret_cast<const uint4*>(src + static_cast<std::int64_t>(v) * 8);
    };
    uint4 ra[kResidualVectors], rb[kResidualVectors], rx[kResidualVectors];
    uint4 wa[kResidualVectors], wb[kResidualVectors], wp[kResidualVectors];
    uint4 w1[kResidualVectors], w2[kResidualVectors], w3[kResidualVectors];
    // The weights first: under a programmatic launch (gemma4_residual_norm's join_previous) they
    // load under the previous kernel, and x and the branch, its output, after the wait.
#pragma unroll
    for (int c = 0; c < kResidualVectors; ++c) {
        const int v = threadIdx.x + c * kResidualThreads;
        if (v < vectors) {
            wp[c] = load(p.post_weight, v);
            if (p.out1 != nullptr) { w1[c] = load(p.out_weight1, v); }
            if constexpr (kTwoBranches) {
                wa[c] = load(p.a_weight, v);
                wb[c] = load(p.b_weight, v);
            }
            if (second) { w2[c] = load(p.out_weight2, v); }
            if (third) { w3[c] = load(p.out_weight3, v); }
        }
    }
    pdl::wait_for_dependencies();
#pragma unroll
    for (int c = 0; c < kResidualVectors; ++c) {
        const int v = threadIdx.x + c * kResidualThreads;
        if (v < vectors) {
            ra[c] = load(p.a + base, v);
            rx[c] = load(p.x + base, v);
            if constexpr (kTwoBranches) { rb[c] = load(p.b + base, v); }
        }
    }

    float t[kResidualVectors][8];
    float st = 0.0f;
    if constexpr (kTwoBranches) {
        float2 s = make_float2(0.0f, 0.0f);
        float fb[kResidualVectors][8];
#pragma unroll
        for (int c = 0; c < kResidualVectors; ++c) {
            if (threadIdx.x + c * kResidualThreads < vectors) {
                unpack8(ra[c], t[c]);
                unpack8(rb[c], fb[c]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s.x += t[c][j] * t[c][j];
                    s.y += fb[c][j] * fb[c][j];
                }
            }
        }
        s              = block_sum2(s, scratch2);
        const float ia = rsqrtf(s.x / p.width + p.eps);
        const float ib = rsqrtf(s.y / p.width + p.eps);
#pragma unroll
        for (int c = 0; c < kResidualVectors; ++c) {
            if (threadIdx.x + c * kResidualThreads < vectors) {
                float fa[8], fwb[8];
                unpack8(wa[c], fa);
                unpack8(wb[c], fwb);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    t[c][j] = t[c][j] * ia * fa[j] + fb[c][j] * ib * fwb[j];
                    st += t[c][j] * t[c][j];
                }
            }
        }
    } else {
#pragma unroll
        for (int c = 0; c < kResidualVectors; ++c) {
            if (threadIdx.x + c * kResidualThreads < vectors) {
                unpack8(ra[c], t[c]);
#pragma unroll
                for (int j = 0; j < 8; ++j) { st += t[c][j] * t[c][j]; }
            }
        }
    }
    const float branch_inverse = rsqrtf(block_sum<kResidualThreads>(st, scratch) / p.width + p.eps);

    float sx = 0.0f;
#pragma unroll
    for (int c = 0; c < kResidualVectors; ++c) {
        const int v = threadIdx.x + c * kResidualThreads;
        if (v < vectors) {
            float fx[8], fp[8];
            unpack8(rx[c], fx);
            unpack8(wp[c], fp);
            uint4 stored;
            auto* h = reinterpret_cast<__nv_bfloat162*>(&stored);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float u0 = (fx[2 * j] + t[c][2 * j] * branch_inverse * fp[2 * j]) * p.scale;
                const float u1 =
                    (fx[2 * j + 1] + t[c][2 * j + 1] * branch_inverse * fp[2 * j + 1]) * p.scale;
                h[j]            = __floats2bfloat162_rn(u0, u1);
                const float2 r  = __bfloat1622float2(h[j]);
                t[c][2 * j]     = r.x;
                t[c][2 * j + 1] = r.y;
                sx += r.x * r.x + r.y * r.y;
            }
            *reinterpret_cast<uint4*>(p.x + base + static_cast<std::int64_t>(v) * 8) = stored;
        }
    }
    if (p.out1 == nullptr) { return; } // update only: nothing reads a normalized x
    const float x_inverse = rsqrtf(block_sum<kResidualThreads>(sx, scratch) / p.width + p.eps);
#pragma unroll
    for (int c = 0; c < kResidualVectors; ++c) {
        const int v = threadIdx.x + c * kResidualThreads;
        if (v < vectors) {
            float f1[8];
            unpack8(w1[c], f1);
            uint4 o1;
            auto* h1 = reinterpret_cast<__nv_bfloat162*>(&o1);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                h1[j] = __floats2bfloat162_rn(t[c][2 * j] * x_inverse * f1[2 * j],
                                              t[c][2 * j + 1] * x_inverse * f1[2 * j + 1]);
            }
            *reinterpret_cast<uint4*>(p.out1 + base + static_cast<std::int64_t>(v) * 8) = o1;
            if (second) {
                float f2[8];
                unpack8(w2[c], f2);
                uint4 o2;
                auto* h2 = reinterpret_cast<__nv_bfloat162*>(&o2);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    h2[j] = __floats2bfloat162_rn(t[c][2 * j] * x_inverse * f2[2 * j],
                                                  t[c][2 * j + 1] * x_inverse * f2[2 * j + 1]);
                }
                *reinterpret_cast<uint4*>(p.out2 + base + static_cast<std::int64_t>(v) * 8) = o2;
            }
            if (third) {
                float f3[8];
                unpack8(w3[c], f3);
                uint4 o3;
                auto* h3 = reinterpret_cast<__nv_bfloat162*>(&o3);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    h3[j] = __floats2bfloat162_rn(t[c][2 * j] * x_inverse * f3[2 * j],
                                                  t[c][2 * j + 1] * x_inverse * f3[2 * j + 1]);
                }
                *reinterpret_cast<uint4*>(p.out3 + base + static_cast<std::int64_t>(v) * 8) = o3;
            }
        }
    }
}

// ---- host-side validation ---------------------------------------------------------------------

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_bf16_vector(const Tensor& t, std::int32_t length, const char* op, const char* name) {
    require(t.dtype == DType::BF16 && t.ne[0] == length && t.ne[1] == 1 && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr,
            std::string(op) + ": " + name + " must be contiguous BF16 [" + std::to_string(length) +
                "]");
}

void require_bf16_matrix(const Tensor& t, std::int32_t rows, std::int32_t columns, const char* op,
                         const char* name) {
    require(t.dtype == DType::BF16 && t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr,
            std::string(op) + ": " + name + " must be contiguous BF16 [" + std::to_string(rows) +
                "," + std::to_string(columns) + "]");
}

// A projection-output slice: features contiguous, token stride a multiple of 16 bytes.
std::int64_t source_stride(const Tensor& t, std::int32_t width, std::int32_t tokens,
                           const char* name) {
    require(t.dtype == DType::BF16 && t.ne[0] == width && t.ne[1] == tokens && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.data != nullptr,
            std::string("gemma4_qkv_prepare: ") + name + " must be BF16 [heads*D, T]");
    require(t.nb[0] == 2 && t.nb[1] >= static_cast<std::int64_t>(width) * 2 && t.nb[1] % 16 == 0 &&
                reinterpret_cast<std::uintptr_t>(t.data) % 16 == 0,
            std::string("gemma4_qkv_prepare: ") + name +
                " needs contiguous features and a 16-byte-aligned token stride");
    return t.nb[1] / 2;
}

// Whether the strided extents of two [ne0, ne1] views (first byte to one past the last) meet.
bool extents_overlap(const Tensor& a, const Tensor& b) {
    const auto* a0 = static_cast<const char*>(a.data);
    const auto* b0 = static_cast<const char*>(b.data);
    const auto a1  = a0 + (a.ne[1] - 1) * a.nb[1] + a.ne[0] * a.nb[0];
    const auto b1  = b0 + (b.ne[1] - 1) * b.nb[1] + b.ne[0] * b.nb[0];
    return a0 < b1 && b0 < a1;
}

} // namespace

namespace {

// One prepare call. Absent query rows (q_src == nullptr) or absent key/value rows
// (k_src == nullptr) shrink the kernel's row loop; the rest follows the public contracts.
struct PrepareCall {
    const char* op;
    const Tensor* q_src        = nullptr;
    const Tensor* k_src        = nullptr;
    const Tensor* v_src        = nullptr;
    const Tensor* q_weight     = nullptr;
    const Tensor* k_weight     = nullptr;
    Tensor* q_out              = nullptr;
    Tensor* k_out              = nullptr;
    Tensor* v_out              = nullptr;
    const Gemma4KvLayer* cache = nullptr; // K/V straight into these planes instead of k/v_out
    std::int32_t head_dim      = 0;
    std::int32_t rotary_pairs  = 0;
    float eps                  = 0;
    float theta                = 0;
    bool v_norm                = true;
    bool join_previous         = false; // gemma4_qkv_prepare's join_previous
};

void launch_prepare(const PrepareCall& c, const Tensor& positions, cudaStream_t stream) {
    const std::string op = c.op;
    const auto check     = [&](bool condition, const std::string& what) {
        require(condition, op + ": " + what);
    };
    const std::int32_t d = c.head_dim;
    check(positions.dtype == DType::I32 && positions.ne[1] == 1 && positions.ne[2] == 1 &&
              positions.ne[3] == 1 && positions.is_contiguous() && positions.data != nullptr &&
              positions.ne[0] > 0,
          "positions must be contiguous I32 [T]");
    const std::int32_t tokens = positions.ne[0];
    check(c.q_src != nullptr || c.k_src != nullptr, "nothing to prepare");
    const auto heads_of = [&](const Tensor* t, const char* name) -> std::int32_t {
        if (t == nullptr) { return 0; }
        check(t->ne[0] % d == 0 && t->ne[0] > 0, std::string(name) + " width must be whole heads");
        return t->ne[0] / d;
    };
    const std::int32_t q_heads  = heads_of(c.q_src, "q_src");
    const std::int32_t kv_heads = heads_of(c.k_src, "k_src");
    if (c.k_src != nullptr) {
        check(c.v_src != nullptr && c.v_src->ne[0] == c.k_src->ne[0], "v_src must match k_src");
    }
    check(std::isfinite(c.eps) && c.eps > 0, "eps must be positive");
    check(std::isfinite(c.theta) && c.theta > 0, "theta must be positive");
    check(c.rotary_pairs > 0 && c.rotary_pairs <= d / 2, "rotary_pairs must be in (0, D/2]");
    const auto require_out = [&](const Tensor* t, std::int32_t heads, const char* name) {
        check(t != nullptr && t->dtype == DType::BF16 && t->ne[0] == d && t->ne[1] == heads &&
                  t->ne[2] == tokens && t->ne[3] == 1 && t->is_contiguous() && t->data != nullptr &&
                  reinterpret_cast<std::uintptr_t>(t->data) % 16 == 0,
              std::string(name) + " must be contiguous BF16 [D,heads,T]");
    };
    std::vector<Tensor> inputs, outputs;
    QkvParams p{};
    if (c.q_src != nullptr) {
        p.q_stride = source_stride(*c.q_src, q_heads * d, tokens, "q_src");
        require_bf16_vector(*c.q_weight, d, c.op, "q_weight");
        require_out(c.q_out, q_heads, "q_out");
        p.q_src    = static_cast<const __nv_bfloat16*>(c.q_src->data);
        p.q_weight = static_cast<const __nv_bfloat16*>(c.q_weight->data);
        p.q_out    = static_cast<__nv_bfloat16*>(c.q_out->data);
        inputs.push_back(*c.q_src);
        outputs.push_back(c.q_out->view({d * q_heads, tokens}));
    }
    if (c.k_src != nullptr) {
        p.k_stride = source_stride(*c.k_src, kv_heads * d, tokens, "k_src");
        p.v_stride = source_stride(*c.v_src, kv_heads * d, tokens, "v_src");
        require_bf16_vector(*c.k_weight, d, c.op, "k_weight");
        p.k_src    = static_cast<const __nv_bfloat16*>(c.k_src->data);
        p.v_src    = static_cast<const __nv_bfloat16*>(c.v_src->data);
        p.k_weight = static_cast<const __nv_bfloat16*>(c.k_weight->data);
        inputs.push_back(*c.k_src);
        inputs.push_back(*c.v_src);
        if (c.cache != nullptr) {
            const Tensor& ck = c.cache->k;
            const Tensor& cv = c.cache->v;
            const detail::Gemma4KvBinding binding =
                detail::gemma4_kv_binding(*c.cache, d, kv_heads, c.op);
            for (const Tensor* plane : {&ck, &cv}) {
                check(reinterpret_cast<std::uintptr_t>(plane->data) % 16 == 0,
                      "cache planes must be 16-byte aligned");
            }
            check(tokens % binding.batch == 0, "T must split evenly over the batch rows");
            p.row_tokens = tokens / binding.batch;
            check(p.row_tokens <= binding.address.capacity, "cache capacity must hold T rows");
            p.kv_capacity = binding.address.capacity;
            p.kv          = binding.address;
            if (binding.fp8) {
                p.k_codes = static_cast<std::uint8_t*>(ck.data);
                p.v_codes = static_cast<std::uint8_t*>(cv.data);
                p.k_scale = static_cast<float*>(c.cache->k_scale.data);
                p.v_scale = static_cast<float*>(c.cache->v_scale.data);
            } else {
                p.k_out = static_cast<__nv_bfloat16*>(ck.data);
                p.v_out = static_cast<__nv_bfloat16*>(cv.data);
            }
            const std::int64_t plane_rows =
                static_cast<std::int64_t>(ck.ne[1]) * ck.ne[2] * ck.ne[3];
            check(plane_rows <= std::numeric_limits<std::int32_t>::max(),
                  "cache planes exceed int32 rows");
            outputs.push_back(ck.view({d, static_cast<std::int32_t>(plane_rows)}));
            outputs.push_back(cv.view({d, static_cast<std::int32_t>(plane_rows)}));
            if (binding.fp8) {
                const auto rows = static_cast<std::int32_t>(plane_rows);
                outputs.push_back(c.cache->k_scale.view({1, rows}));
                outputs.push_back(c.cache->v_scale.view({1, rows}));
            }
        } else {
            require_out(c.k_out, kv_heads, "k_out");
            require_out(c.v_out, kv_heads, "v_out");
            p.k_out = static_cast<__nv_bfloat16*>(c.k_out->data);
            p.v_out = static_cast<__nv_bfloat16*>(c.v_out->data);
            outputs.push_back(c.k_out->view({d * kv_heads, tokens}));
            outputs.push_back(c.v_out->view({d * kv_heads, tokens}));
        }
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        for (const auto& in : inputs) {
            check(!extents_overlap(outputs[i], in), "outputs must not overlap inputs");
        }
        for (std::size_t j = i + 1; j < outputs.size(); ++j) {
            check(!extents_overlap(outputs[i], outputs[j]), "outputs must not overlap one another");
        }
    }
    p.positions    = static_cast<const std::int32_t*>(positions.data);
    p.eps          = c.eps;
    p.q_heads      = q_heads;
    p.kv_heads     = kv_heads;
    p.rotary_pairs = c.rotary_pairs;
    check(c.rotary_pairs >= 0 && c.rotary_pairs <= d / 2, "rotary_pairs exceeds head_dim / 2");
    {
        const double log_theta = static_cast<double>(std::log(c.theta));
        for (int i = 0; i < c.rotary_pairs; ++i) {
            p.inverse_frequency[i] = std::exp(-(2.0 * i / d) * log_theta);
        }
    }
    p.v_norm       = c.v_norm;
    // A decode or verify call has a handful of tokens: one block per token would leave the whole
    // row loop on one SM, so the rows are spread over blocks. A prefill chunk already fills the
    // card, and extra blocks would only repeat the angle table.
    const int rows   = q_heads + 2 * kv_heads;
    const int groups = tokens >= 32 ? 1 : (rows + kQkvThreads / 32 - 1) / (kQkvThreads / 32);
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(groups));
    const bool dependent = c.join_previous && device_class().features.pdl;
    const auto launch    = [&](void (*kernel)(QkvParams)) {
        if (dependent) {
            CUDA_CHECK(pdl::launch_dependent({grid, dim3(kQkvThreads), 0, stream}, kernel, p));
        } else {
            kernel<<<grid, kQkvThreads, 0, stream>>>(p);
        }
    };
    switch (d) {
    case 128:
        launch(gemma4_qkv_prepare_kernel<128>);
        break;
    case 256:
        launch(gemma4_qkv_prepare_kernel<256>);
        break;
    case 512:
        launch(gemma4_qkv_prepare_kernel<512>);
        break;
    default:
        check(false, "unsupported head_dim");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gemma4_qkv_prepare(const Tensor& q_src, const Tensor& k_src, const Tensor& v_src,
                        std::int32_t head_dim, const Tensor& q_weight, const Tensor& k_weight,
                        float eps, const Tensor& positions, float theta, std::int32_t rotary_pairs,
                        Tensor& q_out, Tensor& k_out, Tensor& v_out, cudaStream_t stream) {
    require(head_dim == 256 || head_dim == 512, "gemma4_qkv_prepare: head_dim must be 256 or 512");
    PrepareCall c{"gemma4_qkv_prepare"};
    c.q_src        = &q_src;
    c.k_src        = &k_src;
    c.v_src        = &v_src;
    c.q_weight     = &q_weight;
    c.k_weight     = &k_weight;
    c.q_out        = &q_out;
    c.k_out        = &k_out;
    c.v_out        = &v_out;
    c.head_dim     = head_dim;
    c.rotary_pairs = rotary_pairs;
    c.eps          = eps;
    c.theta        = theta;
    launch_prepare(c, positions, stream);
}

void gemma4_qkv_prepare(const Tensor& q_src, const Tensor& k_src, const Tensor& v_src,
                        std::int32_t head_dim, const Tensor& q_weight, const Tensor& k_weight,
                        float eps, const Tensor& positions, float theta, std::int32_t rotary_pairs,
                        Tensor& q_out, const Gemma4KvLayer& cache, cudaStream_t stream,
                        bool join_previous) {
    require(head_dim == 256 || head_dim == 512, "gemma4_qkv_prepare: head_dim must be 256 or 512");
    PrepareCall c{"gemma4_qkv_prepare"};
    c.join_previous = join_previous;
    c.q_src        = &q_src;
    c.k_src        = &k_src;
    c.v_src        = &v_src;
    c.q_weight     = &q_weight;
    c.k_weight     = &k_weight;
    c.q_out        = &q_out;
    c.cache        = &cache;
    c.head_dim     = head_dim;
    c.rotary_pairs = rotary_pairs;
    c.eps          = eps;
    c.theta        = theta;
    launch_prepare(c, positions, stream);
}

void gemma4_q_prepare(const Tensor& q_src, std::int32_t head_dim, const Tensor& q_weight, float eps,
                      const Tensor& positions, float theta, std::int32_t rotary_pairs,
                      Tensor& q_out, cudaStream_t stream) {
    require(head_dim == 256 || head_dim == 512, "gemma4_q_prepare: head_dim must be 256 or 512");
    PrepareCall c{"gemma4_q_prepare"};
    c.q_src        = &q_src;
    c.q_weight     = &q_weight;
    c.q_out        = &q_out;
    c.head_dim     = head_dim;
    c.rotary_pairs = rotary_pairs;
    c.eps          = eps;
    c.theta        = theta;
    launch_prepare(c, positions, stream);
}

void gemma4_dflash_qkv_prepare(const Tensor* q_src, const Tensor& k_src, const Tensor& v_src,
                               const Tensor* q_weight, const Tensor& k_weight, float eps,
                               const Tensor& positions, float theta, Tensor* q_out, Tensor& k_out,
                               Tensor& v_out, cudaStream_t stream) {
    require((q_src == nullptr) == (q_weight == nullptr) && (q_src == nullptr) == (q_out == nullptr),
            "gemma4_dflash_qkv_prepare: q_src, q_weight and q_out come together");
    PrepareCall c{"gemma4_dflash_qkv_prepare"};
    c.q_src        = q_src;
    c.k_src        = &k_src;
    c.v_src        = &v_src;
    c.q_weight     = q_weight;
    c.k_weight     = &k_weight;
    c.q_out        = q_out;
    c.k_out        = &k_out;
    c.v_out        = &v_out;
    c.head_dim     = 128;
    c.rotary_pairs = 64;
    c.eps          = eps;
    c.theta        = theta;
    c.v_norm       = false;
    launch_prepare(c, positions, stream);
}

void gemma4_dflash_qkv_prepare(const Tensor* q_src, const Tensor& k_src, const Tensor& v_src,
                               const Tensor* q_weight, const Tensor& k_weight, float eps,
                               const Tensor& positions, float theta, Tensor* q_out,
                               const Gemma4KvLayer& cache, cudaStream_t stream) {
    require((q_src == nullptr) == (q_weight == nullptr) && (q_src == nullptr) == (q_out == nullptr),
            "gemma4_dflash_qkv_prepare: q_src, q_weight and q_out come together");
    PrepareCall c{"gemma4_dflash_qkv_prepare"};
    c.q_src        = q_src;
    c.k_src        = &k_src;
    c.v_src        = &v_src;
    c.q_weight     = q_weight;
    c.k_weight     = &k_weight;
    c.q_out        = q_out;
    c.cache        = &cache;
    c.head_dim     = 128;
    c.rotary_pairs = 64;
    c.eps          = eps;
    c.theta        = theta;
    c.v_norm       = false;
    launch_prepare(c, positions, stream);
}

void gelu_tanh_mul(const Tensor& gate_up, Tensor& out, cudaStream_t stream) {
    constexpr const char* op  = "gelu_tanh_mul";
    const std::int32_t width  = out.ne[0];
    const std::int32_t tokens = out.ne[1];
    require(width > 0 && width % 2 == 0, "gelu_tanh_mul: output width must be positive and even");
    require_bf16_matrix(out, width, tokens, op, "out");
    require_bf16_matrix(gate_up, 2 * width, tokens, op, "gate_up");
    require(!extents_overlap(gate_up, out), "gelu_tanh_mul: out must not overlap gate_up");
    const std::int64_t pairs = static_cast<std::int64_t>(width) * tokens / 2;
    const int threads        = 256;
    const std::int64_t want  = (pairs + threads - 1) / threads;
    const int blocks         = static_cast<int>(want < 4096 ? want : 4096);
    gelu_tanh_mul_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up.data), static_cast<__nv_bfloat16*>(out.data),
        width, pairs);
    CUDA_CHECK(cudaGetLastError());
}

void gelu_tanh_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    constexpr const char* op  = "gelu_tanh_mul";
    const std::int32_t width  = out.ne[0];
    const std::int32_t tokens = out.ne[1];
    require(width > 0 && width % 2 == 0, "gelu_tanh_mul: output width must be positive and even");
    require_bf16_matrix(out, width, tokens, op, "out");
    for (const Tensor* t : {&gate, &up}) {
        require(t->dtype == DType::BF16 && t->ne[0] == width && t->ne[1] == tokens &&
                    t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == 2 && t->nb[1] % 4 == 0 &&
                    t->nb[1] >= static_cast<std::int64_t>(width) * 2 && t->data != nullptr &&
                    reinterpret_cast<std::uintptr_t>(t->data) % 4 == 0,
                "gelu_tanh_mul: gate and up must be BF16 [I,T] with 4-byte aligned columns");
        require(!extents_overlap(*t, out), "gelu_tanh_mul: out must not overlap its inputs");
    }
    const std::int64_t pairs = static_cast<std::int64_t>(width) * tokens / 2;
    const int threads        = 256;
    const std::int64_t want  = (pairs + threads - 1) / threads;
    const int blocks         = static_cast<int>(want < 4096 ? want : 4096);
    if (blocks == 0) { return; }
    gelu_tanh_mul2_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), gate.nb[1] / 2,
        static_cast<const __nv_bfloat16*>(up.data), up.nb[1] / 2,
        static_cast<__nv_bfloat16*>(out.data), width, pairs);
    CUDA_CHECK(cudaGetLastError());
}

void scale_bf16(Tensor& x, float scale, cudaStream_t stream) {
    const std::int64_t elements = x.numel();
    require(x.dtype == DType::BF16 && x.is_contiguous() && elements % 2 == 0 &&
                reinterpret_cast<std::uintptr_t>(x.data) % 4 == 0,
            "scale_bf16: x must be contiguous 4-byte aligned BF16 with an even element count");
    require(std::isfinite(scale), "scale_bf16: scale must be finite");
    const std::int64_t pairs = elements / 2;
    const int threads        = 256;
    const std::int64_t want  = (pairs + threads - 1) / threads;
    const int blocks         = static_cast<int>(want < 4096 ? want : 4096);
    if (blocks == 0) { return; }
    scale_bf16_kernel<<<blocks, threads, 0, stream>>>(static_cast<__nv_bfloat162*>(x.data), scale,
                                                      pairs);
    CUDA_CHECK(cudaGetLastError());
}

void gemma4_ple_inputs(const Tensor& projection, const Tensor& table_rows, const Tensor& weight,
                       float eps, float projection_scale, float embedding_scale,
                       float input_scale, Tensor& out, cudaStream_t stream) {
    constexpr const char* op  = "gemma4_ple_inputs";
    const std::int32_t width  = weight.ne[0];
    const std::int32_t rows   = out.ne[0];
    const std::int32_t tokens = out.ne[1];
    require(width > 0 && width % 2 == 0 && rows % width == 0,
            "gemma4_ple_inputs: rows must be whole slices of an even width");
    require_bf16_vector(weight, width, op, "weight");
    require_bf16_matrix(out, rows, tokens, op, "out");
    require_bf16_matrix(projection, rows, tokens, op, "projection");
    require_bf16_matrix(table_rows, rows, tokens, op, "table_rows");
    require(!extents_overlap(table_rows, out) && !extents_overlap(table_rows, projection) &&
                (out.data == projection.data || !extents_overlap(out, projection)),
            "gemma4_ple_inputs: out may alias projection only, and table_rows nothing");
    require(std::isfinite(eps) && eps > 0 && std::isfinite(projection_scale) &&
                std::isfinite(embedding_scale) && std::isfinite(input_scale),
            "gemma4_ple_inputs: eps must be positive and the scales finite");
    PleParams p{};
    p.projection       = static_cast<const __nv_bfloat16*>(projection.data);
    p.table_rows       = static_cast<const __nv_bfloat16*>(table_rows.data);
    p.weight           = static_cast<const __nv_bfloat16*>(weight.data);
    p.out              = static_cast<__nv_bfloat16*>(out.data);
    p.width            = width;
    p.slices           = rows / width;
    p.tokens           = tokens;
    p.eps              = eps;
    p.projection_scale = projection_scale;
    p.embedding_scale  = embedding_scale;
    p.input_scale      = input_scale;
    const std::int64_t warps = p.tokens * p.slices;
    const std::int64_t blocks = (warps + kPleThreads / 32 - 1) / (kPleThreads / 32);
    if (blocks == 0) { return; }
    gemma4_ple_inputs_kernel<<<static_cast<unsigned>(blocks), kPleThreads, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

void gemma4_residual_norm(Tensor& x, const ResidualBranch& branch, const Tensor& post_weight,
                          float eps, float scale, std::span<const ResidualNormOutput> outputs,
                          cudaStream_t stream, bool join_previous) {
    constexpr const char* op  = "gemma4_residual_norm";
    const std::int32_t width  = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    require(width > 0 && width <= kResidualMaxWidth,
            "gemma4_residual_norm: width must be in (0, 8192]");
    require(outputs.size() <= 3, "gemma4_residual_norm: at most three outputs");
    require_bf16_matrix(x, width, tokens, op, "x");
    require_bf16_matrix(branch.a, width, tokens, op, "a");
    require_bf16_vector(post_weight, width, op, "post_weight");
    const bool two = branch.b != nullptr;
    require((branch.a_weight != nullptr) == two && (branch.b_weight != nullptr) == two,
            "gemma4_residual_norm: a_weight, b and b_weight are given together or not at all");
    std::vector<const Tensor*> read{&branch.a};
    if (two) {
        require_bf16_matrix(*branch.b, width, tokens, op, "b");
        require_bf16_vector(*branch.a_weight, width, op, "a_weight");
        require_bf16_vector(*branch.b_weight, width, op, "b_weight");
        read.push_back(branch.b);
    }
    std::vector<const Tensor*> written{&x};
    for (const ResidualNormOutput& o : outputs) {
        require_bf16_matrix(o.out, width, tokens, op, "out");
        require_bf16_vector(o.weight, width, op, "out weight");
        written.push_back(&o.out);
    }
    for (std::size_t i = 0; i < written.size(); ++i) {
        for (const Tensor* r : read) {
            require(!extents_overlap(*written[i], *r),
                    "gemma4_residual_norm: outputs overlap inputs");
        }
        for (std::size_t j = i + 1; j < written.size(); ++j) {
            require(!extents_overlap(*written[i], *written[j]),
                    "gemma4_residual_norm: outputs overlap one another");
        }
    }
    require(std::isfinite(eps) && eps > 0 && std::isfinite(scale),
            "gemma4_residual_norm: eps must be positive and scale finite");

    const auto bf16 = [](const Tensor* t) {
        return t != nullptr ? static_cast<__nv_bfloat16*>(t->data) : nullptr;
    };
    const auto output = [&](std::size_t i) { return i < outputs.size() ? &outputs[i] : nullptr; };
    const auto out_weight = [&](std::size_t i) {
        return output(i) != nullptr ? bf16(&output(i)->weight) : nullptr;
    };
    const auto out = [&](std::size_t i) {
        return output(i) != nullptr ? bf16(&output(i)->out) : nullptr;
    };
    ResidualNormParams p{};
    p.x                = bf16(&x);
    p.a                = bf16(&branch.a);
    p.a_weight         = bf16(branch.a_weight);
    p.b                = bf16(branch.b);
    p.b_weight         = bf16(branch.b_weight);
    p.post_weight      = bf16(&post_weight);
    p.out_weight1      = out_weight(0);
    p.out1             = out(0);
    p.out_weight2      = out_weight(1);
    p.out2             = out(1);
    p.out_weight3      = out_weight(2);
    p.out3             = out(2);
    p.width            = width;
    p.eps              = eps;
    p.scale            = scale;
    const auto aligned = [](const void* ptr) {
        return ptr == nullptr || reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0;
    };
    const bool vector_path = width % 8 == 0 && aligned(p.x) && aligned(p.a) &&
                             aligned(p.a_weight) && aligned(p.b) && aligned(p.b_weight) &&
                             aligned(p.post_weight) && aligned(p.out_weight1) && aligned(p.out1) &&
                             aligned(p.out_weight2) && aligned(p.out2) && aligned(p.out_weight3) &&
                             aligned(p.out3);
    if (vector_path) {
        auto* kernel = two ? gemma4_residual_norm_vector_kernel<true>
                           : gemma4_residual_norm_vector_kernel<false>;
        if (join_previous && device_class().features.pdl) {
            CUDA_CHECK(pdl::launch_dependent(
                {dim3(static_cast<unsigned>(tokens)), dim3(kResidualThreads), 0, stream}, kernel,
                p));
            return;
        }
        kernel<<<tokens, kResidualThreads, 0, stream>>>(p);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::size_t smem = static_cast<std::size_t>(width) * sizeof(float);
    if (two) {
        gemma4_residual_norm_kernel<true><<<tokens, kResidualThreads, smem, stream>>>(p);
    } else {
        gemma4_residual_norm_kernel<false><<<tokens, kResidualThreads, smem, stream>>>(p);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops
