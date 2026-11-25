// ninferno::ops - the Inkling 2 family's own Ops (contracts in ninferno/ops/inkling2.h). This
// first form is written for exactness and clarity: one launch per Op, FP32 arithmetic, no
// fusion. The decode-speed forms (fused prepares, a tiled attention) replace launches here only
// after the family's numerics are qualified against llama.cpp.
#include "ninferno/ops/inkling2.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/softmax_attention/gemma4/kv_address.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, o); }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xffffffffU, v, o)); }
    return v;
}

// Sum over a block of blockDim.x (a multiple of 32, <= 1024) threads; every thread gets it.
__device__ float block_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) { scratch[warp] = v; }
    __syncthreads();
    float total = 0;
    for (int w = 0; w < warps; ++w) { total += scratch[w]; }
    return total;
}

__device__ __forceinline__ float bf(const __nv_bfloat16* p, std::int64_t i) {
    return __bfloat162float(p[i]);
}

void need(bool ok, const char* op, const std::string& what) {
    if (!ok) { throw std::invalid_argument(std::string(op) + ": " + what); }
}

void need_bf16(const Tensor& t, const char* op, const char* name) {
    need(t.data != nullptr && t.dtype == DType::BF16 && t.is_contiguous(), op,
         std::string(name) + " must be a contiguous BF16 tensor");
}

void need_f32(const Tensor& t, const char* op, const char* name) {
    need(t.data != nullptr && t.dtype == DType::FP32 && t.is_contiguous(), op,
         std::string(name) + " must be a contiguous FP32 tensor");
}

// ---------------------------------------------------------------------------------------------
// depth mix

struct MixParams {
    const __nv_bfloat16* states[kInkling2MaxMixStates];
    __nv_bfloat16* out[kInkling2MaxMixQueries];
    const float* query;
    int count, queries, hidden;
};

constexpr int kMixThreads = 256;
constexpr int kMixSums    = kInkling2MaxMixStates * (kInkling2MaxMixQueries + 1);

// Two kernels, a block a chunk of kMixThreads channels a token (9 blocks a token at H 2304, where
// one block a token took 9.8 us a call on the RTX 4090: its loop over H waited on the loads one
// after another). The first gathers every sum the weights need over its chunk (each state's sum of
// squares, its dot with every query) into `part`; the second adds the chunks' sums, takes the
// softmax, and writes its chunk of every output.
__global__ void depth_mix_sums_kernel(MixParams p, float* part) {
    constexpr int S = kInkling2MaxMixStates, Q = kInkling2MaxMixQueries;
    constexpr int kWarps = kMixThreads / 32;
    __shared__ float warp_part[kWarps][kMixSums];
    const int chunk = blockIdx.x, chunks = gridDim.x, t = blockIdx.y;
    const int h             = chunk * kMixThreads + threadIdx.x;
    const std::int64_t base = static_cast<std::int64_t>(t) * p.hidden;
    float acc[kMixSums];
#pragma unroll
    for (int i = 0; i < kMixSums; ++i) { acc[i] = 0.0F; }
    if (h < p.hidden) {
        float qv[Q];
#pragma unroll
        for (int r = 0; r < Q; ++r) {
            qv[r] = r < p.queries ? p.query[static_cast<std::int64_t>(r) * p.hidden + h] : 0.0F;
        }
#pragma unroll
        for (int st = 0; st < S; ++st) {
            if (st < p.count) {
                const float x = bf(p.states[st], base + h);
                acc[st]       = x * x;
#pragma unroll
                for (int r = 0; r < Q; ++r) { acc[S + r * S + st] = x * qv[r]; }
            }
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < kMixSums; ++i) {
        const float v = warp_sum(acc[i]);
        if (lane == 0) { warp_part[warp][i] = v; }
    }
    __syncthreads();
    if (threadIdx.x < kMixSums) {
        float sum = 0.0F;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { sum += warp_part[w][threadIdx.x]; }
        part[(static_cast<std::int64_t>(t) * chunks + chunk) * kMixSums + threadIdx.x] = sum;
    }
}

__global__ void depth_mix_out_kernel(MixParams p, const float* part) {
    constexpr int S = kInkling2MaxMixStates, Q = kInkling2MaxMixQueries;
    __shared__ float total[kMixSums];
    __shared__ float weight[Q][S];
    const int chunk = blockIdx.x, chunks = gridDim.x, t = blockIdx.y;
    if (threadIdx.x < kMixSums) {
        float sum = 0.0F;
        for (int c = 0; c < chunks; ++c) {
            sum += part[(static_cast<std::int64_t>(t) * chunks + c) * kMixSums + threadIdx.x];
        }
        total[threadIdx.x] = sum;
    }
    __syncthreads();
    if (threadIdx.x < p.queries) {
        const int r       = threadIdx.x;
        const float scale = rsqrtf(static_cast<float>(p.hidden));
        float inv[S], w[S];
        float m = -INFINITY;
#pragma unroll
        for (int st = 0; st < S; ++st) {
            inv[st] = st < p.count ? rsqrtf(total[st] / p.hidden + 1e-6F) : 0.0F;
            w[st]   = st < p.count ? total[S + r * S + st] * inv[st] * scale : -INFINITY;
            m       = fmaxf(m, w[st]);
        }
        float z = 0.0F;
#pragma unroll
        for (int st = 0; st < S; ++st) { z += st < p.count ? expf(w[st] - m) : 0.0F; }
#pragma unroll
        for (int st = 0; st < S; ++st) {
            weight[r][st] = st < p.count ? expf(w[st] - m) / z * inv[st] : 0.0F;
        }
    }
    __syncthreads();
    const int h = chunk * kMixThreads + threadIdx.x;
    if (h >= p.hidden) { return; }
    const std::int64_t base = static_cast<std::int64_t>(t) * p.hidden;
    float x[S];
#pragma unroll
    for (int st = 0; st < S; ++st) { x[st] = st < p.count ? bf(p.states[st], base + h) : 0.0F; }
    for (int r = 0; r < p.queries; ++r) {
        float a = 0.0F;
#pragma unroll
        for (int st = 0; st < S; ++st) { a += weight[r][st] * x[st]; }
        p.out[r][base + h] = __float2bfloat16(a);
    }
}

// ---------------------------------------------------------------------------------------------
// short convolution

constexpr int kConvThreads = 256;

__global__ void conv_kernel(const __nv_bfloat16* x, const float* w, const __nv_bfloat16* taps,
                            int heads, Inkling2ConvState st, __nv_bfloat16* out, int channels,
                            int tokens, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    if (c >= channels) { return; }
    const int head = taps ? c / (channels / heads) : 0;
    float* state   = st.data + (st.slots ? st.slots[b] : b) * st.slot_stride;
    x += static_cast<std::int64_t>(b) * tokens * channels;
    out += static_cast<std::int64_t>(b) * tokens * channels;
    if (taps) { taps += static_cast<std::int64_t>(b) * tokens * heads * 4; }
    float u0 = state[c * 3 + 0], u1 = state[c * 3 + 1],
          u2       = state[c * 3 + 2]; // u[t-3], u[t-2], u[t-1]
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < tokens; ++t) {
        const std::int64_t i = static_cast<std::int64_t>(t) * channels + c;
        const float xt       = bf(x, i);
        float y              = xt + w0 * u0 + w1 * u1 + w2 * u2 + w3 * xt;
        if (taps) {
            const std::int64_t tb = static_cast<std::int64_t>(t) * heads * 4 + head * 4;
            // lag 0 reads u[t], lag 1 u[t-1], lag 2 u[t-2], lag 3 u[t-3]
            y += bf(taps, tb + 0) * xt + bf(taps, tb + 1) * u2 + bf(taps, tb + 2) * u1 +
                 bf(taps, tb + 3) * u0;
        }
        out[i] = __float2bfloat16(y * scale);
        u0     = u1;
        u1     = u2;
        u2     = xt;
    }
    state[c * 3 + 0] = u0;
    state[c * 3 + 1] = u1;
    state[c * 3 + 2] = u2;
}

// ---------------------------------------------------------------------------------------------
// controller, norm pair

__global__ void control_kernel(const __nv_bfloat16* cur, const __nv_bfloat16* msg,
                               const __nv_bfloat16* gain, const __nv_bfloat16* forget,
                               __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) { return; }
    const float c = bf(cur, i);
    out[i] = __float2bfloat16(c - tanhf(bf(forget, i)) * c + tanhf(bf(gain, i)) * bf(msg, i));
}

// One block a half (cur, message) of a token: the two norms in parallel.
__global__ void norm_pair_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
                                 int hidden) {
    __shared__ float scratch[32];
    const std::int64_t t   = blockIdx.y;
    const int half         = blockIdx.x;
    const __nv_bfloat16* x = (half ? b : a) + t * hidden;
    float sq               = 0;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) { sq += bf(x, h) * bf(x, h); }
    const float inv = rsqrtf(block_sum(sq, scratch) / hidden + 1e-6F);
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        out[t * 2 * hidden + half * hidden + h] = __float2bfloat16(bf(x, h) * inv);
    }
}

// ---------------------------------------------------------------------------------------------
// attention prepare: one warp a head row

struct PrepareParams {
    __nv_bfloat16 *q, *k, *v, *r;
    const float *q_clip, *rel_clip;
    const std::int32_t* positions;
    int q_heads, kv_heads, d, dr;
    float qk_clip, rel_clip_value, eps, alpha, floor;
    bool full;
};

__global__ void prepare_kernel(PrepareParams p) {
    const int t = blockIdx.y, row = blockIdx.x, lane = threadIdx.x;
    const int rows_q = p.q_heads, rows_k = p.kv_heads;
    float tau = 1.0F;
    if (p.full) {
        const float pos = static_cast<float>(p.positions[t]) + 1.0F;
        tau             = 1.0F + p.alpha * logf(fmaxf(1.0F, pos / p.floor));
    }
    __nv_bfloat16* x;
    const float* scale = nullptr;
    int n;
    float clip = 0, post = 1.0F;
    bool rms = false;
    if (row < rows_q) {
        x     = p.q + (static_cast<std::int64_t>(t) * rows_q + row) * p.d;
        scale = p.q_clip + static_cast<std::int64_t>(row) * p.d;
        n = p.d, clip = p.qk_clip, post = tau;
    } else if (row < rows_q + rows_k) {
        x = p.k + (static_cast<std::int64_t>(t) * rows_k + (row - rows_q)) * p.d;
        n = p.d, clip = p.qk_clip;
    } else if (row < rows_q + 2 * rows_k) {
        x = p.v + (static_cast<std::int64_t>(t) * rows_k + (row - rows_q - rows_k)) * p.d;
        n = p.d, rms = true;
    } else {
        const int h = row - rows_q - 2 * rows_k;
        x           = p.r + (static_cast<std::int64_t>(t) * rows_q + h) * p.dr;
        scale       = p.rel_clip + static_cast<std::int64_t>(h) * p.dr;
        n = p.dr, clip = p.rel_clip_value, post = tau;
    }
    float vals[8]; // n <= 256
    float sq = 0;
    for (int j = 0; j < 8; ++j) {
        const int i = lane + 32 * j;
        float value = 0;
        if (i < n) { value = __bfloat162float(x[i]) * (scale ? scale[i] : 1.0F); }
        vals[j] = value;
        sq += value * value;
    }
    sq            = warp_sum(sq);
    const float f = rms ? rsqrtf(sq / n + p.eps) : clip / fmaxf(sqrtf(sq), clip) * post;
    for (int j = 0; j < 8; ++j) {
        const int i = lane + 32 * j;
        if (i < n) { x[i] = __float2bfloat16(vals[j] * f); }
    }
}

// ---------------------------------------------------------------------------------------------
// attention: a block a (KV group, query column, key split); the group's query heads share each
// key and value row; a second kernel joins the splits (flash decoding)

constexpr int kAttnWarps    = 4;
constexpr int kAttnMaxGroup = 4;  // query heads a KV head
constexpr int kAttnMaxSplit = 16; // key splits a (group, column)

struct AttnParams {
    const __nv_bfloat16 *q, *r, *sink, *k, *v;
    const float* bias;
    const std::int32_t* positions;
    __nv_bfloat16* out;
    float* part; // splits > 1: [column, Hq, split] partials of (m, z, acc[D])
    detail::Gemma4KvAddress kv;
    int q_heads, kv_heads, d, dr, extent, window, tokens, splits;
};

constexpr int kAttnPart = 2 + 128; // m, z, acc[128]

__device__ __forceinline__ void load4(const __nv_bfloat16* row, int i, int d, float (&x)[4]) {
    if (i + 3 < d) {
        const uint2 w          = *reinterpret_cast<const uint2*>(row + i);
        const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&w.x);
        const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
        x[0] = __low2float(a), x[1] = __high2float(a), x[2] = __low2float(b),
        x[3] = __high2float(b);
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) { x[j] = i + j < d ? __bfloat162float(row[i + j]) : 0.0F; }
    }
}

// The batch row's keys through its binding (kv_address.h: a ring image or a page table row). Lane
// l keeps dims 4l..4l+3 of every head of the group; each warp walks its split's keys two at a
// time; the block's warps then join, and either write the output (one split) or the partial.
template <int G>
__global__ void attention_kernel(AttnParams p) {
    const int g = blockIdx.x, col = blockIdx.y, split = blockIdx.z, lane = threadIdx.x & 31,
              warp = threadIdx.x >> 5;
    const int b = col / p.tokens, h0 = g * G, i = 4 * lane;
    const int pos  = p.positions[col];
    const int low  = p.window ? max(0, pos - p.window + 1) : 0;
    const int n    = pos - low + 1;
    const int span = (n + p.splits - 1) / p.splits;
    const int k0 = low + split * span, k1 = min(pos + 1, k0 + span);
    const float sc          = rsqrtf(static_cast<float>(p.d));
    const int binding       = detail::gemma4_kv_binding(p.kv, b);
    const std::int64_t base = detail::gemma4_kv_plane_offset(p.kv, binding) +
                              static_cast<std::int64_t>(g) * p.kv.capacity * p.d;
    float qv[G][4], acc[G][4], rv[G], m[G], z[G];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        const std::int64_t qh = static_cast<std::int64_t>(col) * p.q_heads + h0 + hh;
        load4(p.q + qh * p.d, i, p.d, qv[hh]);
#pragma unroll
        for (int j = 0; j < 4; ++j) { qv[hh][j] *= sc, acc[hh][j] = 0.0F; }
        // The bias and the sink join the mask, after the scale (llama.cpp folds them into kq_mask).
        rv[hh] = lane < p.dr ? __bfloat162float(p.r[qh * p.dr + lane]) : 0.0F;
        m[hh] = -INFINITY, z[hh] = 0.0F;
    }
    for (int key = k0 + 2 * warp; key < k1; key += 2 * kAttnWarps) {
        const bool two = key + 1 < k1;
        float kx[2][4], vx[2][4], s[2][G];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int kk = u == 0 || two ? key + u : key;
            const std::int64_t row =
                base + static_cast<std::int64_t>(detail::gemma4_kv_slot(p.kv, binding, kk)) * p.d;
            load4(p.k + row, i, p.d, kx[u]);
            load4(p.v + row, i, p.d, vx[u]);
            const int dist   = pos - kk;
            const float bias = dist < p.extent && lane < p.dr
                                   ? p.bias[static_cast<std::int64_t>(dist) * p.dr + lane]
                                   : 0.0F;
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                float t = rv[hh] * bias;
#pragma unroll
                for (int j = 0; j < 4; ++j) { t += qv[hh][j] * kx[u][j]; }
                s[u][hh] = t;
            }
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
            for (int hh = 0; hh < G; ++hh) { s[u][hh] = warp_sum(s[u][hh]); }
        }
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            const float s1 = two ? s[1][hh] : -INFINITY;
            const float mn = fmaxf(m[hh], fmaxf(s[0][hh], s1));
            const float c = expf(m[hh] - mn), e0 = expf(s[0][hh] - mn),
                        e1 = two ? expf(s1 - mn) : 0.0F;
            z[hh]          = z[hh] * c + e0 + e1;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[hh][j] = acc[hh][j] * c + e0 * vx[0][j] + e1 * vx[1][j];
            }
            m[hh] = mn;
        }
    }
    __shared__ float sm[kAttnWarps][G], sz[kAttnWarps][G], sacc[kAttnWarps][G][128];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        if (lane == 0) { sm[warp][hh] = m[hh], sz[warp][hh] = z[hh]; }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (i + j < p.d) { sacc[warp][hh][i + j] = acc[hh][j]; }
        }
    }
    __syncthreads();
    if (warp >= G) { return; }
    const int hh = warp, h = h0 + hh;
    float mt = -INFINITY;
    for (int w = 0; w < kAttnWarps; ++w) { mt = fmaxf(mt, sm[w][hh]); }
    const std::int64_t qh = static_cast<std::int64_t>(col) * p.q_heads + h;
    if (p.splits == 1) {
        const float sink = __bfloat162float(p.sink[qh]);
        mt               = fmaxf(mt, sink);
    }
    float zt = 0.0F, cw[kAttnWarps];
    for (int w = 0; w < kAttnWarps; ++w) {
        cw[w] = sm[w][hh] == -INFINITY ? 0.0F : expf(sm[w][hh] - mt);
        zt += sz[w][hh] * cw[w];
    }
    if (p.splits == 1) {
        zt += expf(__bfloat162float(p.sink[qh]) - mt);
        __nv_bfloat16* o = p.out + qh * p.d;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (i + j >= p.d) { continue; }
            float sum = 0;
            for (int w = 0; w < kAttnWarps; ++w) { sum += sacc[w][hh][i + j] * cw[w]; }
            o[i + j] = __float2bfloat16(sum / zt);
        }
        return;
    }
    float* part = p.part + (qh * p.splits + split) * kAttnPart;
    if (lane == 0) { part[0] = mt, part[1] = zt; }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (i + j >= p.d) { continue; }
        float sum = 0;
        for (int w = 0; w < kAttnWarps; ++w) { sum += sacc[w][hh][i + j] * cw[w]; }
        part[2 + i + j] = sum;
    }
}

// One warp a (head, column): the splits' partials and the sink.
__global__ void attention_join_kernel(AttnParams p) {
    const int h = blockIdx.x, col = blockIdx.y, lane = threadIdx.x, i = 4 * lane;
    const std::int64_t qh = static_cast<std::int64_t>(col) * p.q_heads + h;
    const float* part     = p.part + qh * p.splits * kAttnPart;
    const float sink      = __bfloat162float(p.sink[qh]);
    float mt              = sink;
    for (int s = 0; s < p.splits; ++s) { mt = fmaxf(mt, part[s * kAttnPart]); }
    float zt = expf(sink - mt), acc[4] = {0, 0, 0, 0};
    for (int s = 0; s < p.splits; ++s) {
        const float ms = part[s * kAttnPart];
        const float c  = ms == -INFINITY ? 0.0F : expf(ms - mt);
        zt += part[s * kAttnPart + 1] * c;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (i + j < p.d) { acc[j] += part[s * kAttnPart + 2 + i + j] * c; }
        }
    }
    __nv_bfloat16* o = p.out + qh * p.d;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (i + j < p.d) { o[i + j] = __float2bfloat16(acc[j] / zt); }
    }
}

// The key splits a call takes: enough blocks to cover the SMs twice, at most kAttnMaxSplit.
// The current device's SMs, asked once a device (a host attribute query: graph-capture safe).
int device_sms() {
    constexpr int kDevices                  = 64;
    static std::atomic<int> cache[kDevices] = {};
    int device                              = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0) { return 1; }
    if (device < kDevices) {
        if (const int known = cache[device].load(std::memory_order_relaxed); known > 0) {
            return known;
        }
    }
    int count = 0;
    if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        count < 1) {
        return 1;
    }
    if (device < kDevices) { cache[device].store(count, std::memory_order_relaxed); }
    return count;
}

int attention_splits(int groups, int columns) {
    const int sms    = device_sms();
    const int blocks = groups * columns;
    return std::clamp((2 * sms + blocks - 1) / blocks, 1, kAttnMaxSplit);
}

} // namespace

std::size_t inkling2_depth_mix_scratch_bytes(std::int32_t hidden, std::int32_t tokens) {
    const std::size_t chunks = (static_cast<std::size_t>(hidden) + kMixThreads - 1) / kMixThreads;
    return chunks * static_cast<std::size_t>(tokens) * kMixSums * sizeof(float);
}

void inkling2_depth_mix(const Inkling2MixStates& states, const Tensor& query, std::int32_t hidden,
                        std::int32_t tokens, const Inkling2MixOutputs& out, const Tensor& scratch,
                        cudaStream_t stream) {
    const char* op = "inkling2_depth_mix";
    need(states.count >= 1 && states.count <= kInkling2MaxMixStates, op, "1..12 states");
    need(out.count >= 1 && out.count <= kInkling2MaxMixQueries, op, "1..4 outputs");
    need_f32(query, op, "query");
    need(query.ne[0] == hidden && query.ne[1] == out.count && hidden % 8 == 0 && tokens >= 1, op,
         "query must be [queries, H] with H % 8 == 0");
    MixParams p{};
    for (int s = 0; s < states.count; ++s) {
        need(states.data[static_cast<std::size_t>(s)] != nullptr, op, "null state");
        p.states[s] = static_cast<const __nv_bfloat16*>(states.data[static_cast<std::size_t>(s)]);
    }
    for (int r = 0; r < out.count; ++r) {
        need(out.data[static_cast<std::size_t>(r)] != nullptr, op, "null output");
        p.out[r] = static_cast<__nv_bfloat16*>(out.data[static_cast<std::size_t>(r)]);
    }
    p.query   = static_cast<const float*>(query.data);
    p.count   = states.count;
    p.queries = out.count;
    p.hidden  = hidden;
    need(scratch.data != nullptr &&
             scratch.bytes() >= inkling2_depth_mix_scratch_bytes(hidden, tokens),
         op, "scratch of inkling2_depth_mix_scratch_bytes");
    const dim3 grid((hidden + kMixThreads - 1) / kMixThreads, tokens);
    float* part = static_cast<float*>(scratch.data);
    depth_mix_sums_kernel<<<grid, kMixThreads, 0, stream>>>(p, part);
    CUDA_CHECK(cudaGetLastError());
    depth_mix_out_kernel<<<grid, kMixThreads, 0, stream>>>(p, part);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_conv(const Tensor& x, const Tensor& weight, const Tensor& taps, std::int32_t heads,
                   const Inkling2ConvState& state, Tensor& out, cudaStream_t stream) {
    const char* op = "inkling2_conv";
    need_bf16(x, op, "x");
    need_bf16(out, op, "out");
    need_f32(weight, op, "weight");
    const int channels = x.ne[0], tokens = x.ne[1], batch = x.ne[2];
    need(state.data != nullptr && (batch == 1 || state.slots || state.slot_stride >= 3 * channels),
         op, "a state plane for every batch row");
    need(weight.ne[0] == 4 && weight.ne[1] == channels && out.ne[0] == channels &&
             out.ne[1] == tokens && out.ne[2] == batch && x.ne[3] == 1,
         op, "weight [C, 4], x and out [C, T, B]");
    const bool dynamic = taps.data != nullptr;
    if (dynamic) {
        need_bf16(taps, op, "taps");
        need(heads >= 1 && channels % heads == 0 && taps.ne[0] == heads * 4 &&
                 taps.ne[1] == tokens && taps.ne[2] == batch,
             op, "taps must be [heads * 4, T, B] with C % heads == 0");
    }
    const float scale = 1.0F / std::sqrt(4.0F * (dynamic ? 2.0F : 1.0F) + 1.0F);
    conv_kernel<<<dim3((channels + kConvThreads - 1) / kConvThreads, batch), kConvThreads, 0,
                  stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(weight.data),
        dynamic ? static_cast<const __nv_bfloat16*>(taps.data) : nullptr, dynamic ? heads : 1,
        state, static_cast<__nv_bfloat16*>(out.data), channels, tokens, scale);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_control(const Tensor& cur, const Tensor& message, const Tensor& gain,
                      const Tensor& forget, Tensor& out, cudaStream_t stream) {
    const char* op = "inkling2_control";
    for (const Tensor* t : {&cur, &message, &gain, &forget, static_cast<const Tensor*>(&out)}) {
        need_bf16(*t, op, "every tensor");
        need(t->numel() == cur.numel(), op, "every tensor has cur's shape");
    }
    const std::int64_t n = cur.numel();
    control_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(cur.data),
        static_cast<const __nv_bfloat16*>(message.data),
        static_cast<const __nv_bfloat16*>(gain.data),
        static_cast<const __nv_bfloat16*>(forget.data), static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_norm_pair(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    const char* op = "inkling2_norm_pair";
    need_bf16(a, op, "a");
    need_bf16(b, op, "b");
    need_bf16(out, op, "out");
    need(a.ne[0] == b.ne[0] && a.ne[1] == b.ne[1] && out.ne[0] == 2 * a.ne[0] &&
             out.ne[1] == a.ne[1],
         op, "a, b [H, T] and out [2H, T]");
    norm_pair_kernel<<<dim3(2, a.ne[1]), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a.data), static_cast<const __nv_bfloat16*>(b.data),
        static_cast<__nv_bfloat16*>(out.data), a.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_attention_prepare(Tensor& q, Tensor& k, Tensor& v, Tensor& r, const Tensor& q_clip,
                                const Tensor& rel_clip, const Tensor& positions,
                                const Inkling2PrepareParams& params, cudaStream_t stream) {
    const char* op = "inkling2_attention_prepare";
    need_bf16(q, op, "q");
    need_bf16(k, op, "k");
    need_bf16(v, op, "v");
    need_bf16(r, op, "r");
    need_f32(q_clip, op, "q_clip");
    need_f32(rel_clip, op, "rel_clip");
    const int d = q.ne[0], hq = q.ne[1], hkv = k.ne[1], dr = r.ne[0], tokens = q.ne[2];
    need(d <= 256 && dr <= 256 && k.ne[0] == d && v.ne[0] == d && v.ne[1] == hkv && r.ne[1] == hq &&
             k.ne[2] == tokens && v.ne[2] == tokens && r.ne[2] == tokens && q_clip.ne[0] == d &&
             q_clip.ne[1] == hq && rel_clip.ne[0] == dr && rel_clip.ne[1] == hq &&
             positions.ne[0] == tokens,
         op, "shapes q [D, Hq, T], k/v [D, Hkv, T], r [Dr, Hq, T], clips [Hq, D] / [Hq, Dr]");
    PrepareParams p{};
    p.q = static_cast<__nv_bfloat16*>(q.data), p.k = static_cast<__nv_bfloat16*>(k.data);
    p.v = static_cast<__nv_bfloat16*>(v.data), p.r = static_cast<__nv_bfloat16*>(r.data);
    p.q_clip    = static_cast<const float*>(q_clip.data),
    p.rel_clip  = static_cast<const float*>(rel_clip.data);
    p.positions = static_cast<const std::int32_t*>(positions.data);
    p.q_heads = hq, p.kv_heads = hkv, p.d = d, p.dr = dr;
    p.qk_clip = params.qk_clip, p.rel_clip_value = params.rel_clip, p.eps = params.eps;
    p.alpha = params.alpha, p.floor = params.floor, p.full = params.full;
    prepare_kernel<<<dim3(2 * hq + 2 * hkv, tokens), 32, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t inkling2_attention_scratch_bytes(std::int32_t q_heads, std::int32_t kv_heads,
                                             std::int32_t columns) {
    const int splits = attention_splits(kv_heads, columns);
    return splits == 1
               ? 0
               : static_cast<std::size_t>(columns) * q_heads * splits * kAttnPart * sizeof(float);
}

void inkling2_attention(const Tensor& q, const Tensor& r, const Tensor& sink, const Tensor& bias,
                        const Gemma4KvLayer& cache, const Tensor& positions, std::int32_t extent,
                        const Tensor& scratch, Tensor& out, cudaStream_t stream) {
    const char* op = "inkling2_attention";
    need_bf16(q, op, "q");
    need_bf16(r, op, "r");
    need_bf16(sink, op, "sink");
    need_bf16(out, op, "out");
    need_f32(bias, op, "bias");
    const int d = q.ne[0], hq = q.ne[1], tokens = q.ne[2], batch = q.ne[3], dr = r.ne[0];
    const auto kv =
        detail::gemma4_kv_binding(cache, d, cache.k.ne[cache.block_tables.data ? 3 : 2], op);
    const int hkv = cache.k.ne[cache.block_tables.data ? 3 : 2];
    need(!kv.fp8, op, "BF16 cache planes");
    need(kv.batch == batch, op, "one binding a batch row");
    need(d <= 128 && d % 4 == 0 && dr <= 32 && hq % hkv == 0 && hq / hkv <= kAttnMaxGroup &&
             r.ne[1] == hq && r.ne[2] == tokens && r.ne[3] == batch && sink.ne[0] == hq &&
             sink.ne[1] == tokens && sink.ne[2] == batch && bias.ne[0] == dr &&
             bias.ne[1] == extent && positions.ne[0] == tokens && positions.ne[1] == batch &&
             out.ne[0] == d && out.ne[1] == hq && out.ne[2] == tokens && out.ne[3] == batch,
         op,
         "D <= 128 (a multiple of 4), Dr <= 32, Hq / Hkv <= 4, q/out [D, Hq, T, B], "
         "r [Dr, Hq, T, B], sink [Hq, T, B], "
         "bias [extent, Dr], positions [T, B]");
    AttnParams p{};
    p.q         = static_cast<const __nv_bfloat16*>(q.data),
    p.r         = static_cast<const __nv_bfloat16*>(r.data);
    p.sink      = static_cast<const __nv_bfloat16*>(sink.data);
    p.k         = static_cast<const __nv_bfloat16*>(cache.k.data),
    p.v         = static_cast<const __nv_bfloat16*>(cache.v.data);
    p.bias      = static_cast<const float*>(bias.data),
    p.positions = static_cast<const std::int32_t*>(positions.data);
    p.out       = static_cast<__nv_bfloat16*>(out.data);
    p.kv        = kv.address;
    p.q_heads = hq, p.kv_heads = hkv, p.d = d, p.dr = dr, p.tokens = tokens;
    p.extent = extent, p.window = cache.window;
    const int columns = tokens * batch;
    p.splits          = attention_splits(hkv, columns);
    if (p.splits > 1) {
        need(scratch.data != nullptr &&
                 scratch.bytes() >= inkling2_attention_scratch_bytes(hq, hkv, columns),
             op, "scratch of inkling2_attention_scratch_bytes");
        p.part = static_cast<float*>(scratch.data);
    }
    const dim3 grid(hkv, columns, p.splits);
    switch (hq / hkv) {
    case 1:
        attention_kernel<1><<<grid, 32 * kAttnWarps, 0, stream>>>(p);
        break;
    case 2:
        attention_kernel<2><<<grid, 32 * kAttnWarps, 0, stream>>>(p);
        break;
    case 3:
        attention_kernel<3><<<grid, 32 * kAttnWarps, 0, stream>>>(p);
        break;
    default:
        attention_kernel<4><<<grid, 32 * kAttnWarps, 0, stream>>>(p);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
    if (p.splits > 1) {
        attention_join_kernel<<<dim3(hq, columns), 32, 0, stream>>>(p);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninferno::ops
