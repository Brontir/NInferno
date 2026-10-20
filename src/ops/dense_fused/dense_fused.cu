// ninferno::ops - elementwise Ops of the dense decoder family. Contracts in
// ninferno/ops/dense_fused.h.
#include "ninferno/ops/dense_fused.h"

#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"
#include "ops/common/dflash_rope.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/gemma4/fp8_row.cuh"
#include "ops/softmax_attention/gemma4/kv_address.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::ops {
namespace {

// E contiguous BF16 features: 16-byte vectors when E is a multiple of 8, one 8-byte vector at 4,
// one 4-byte word at 2 (a 64-wide head).
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
    } else if constexpr (E == 2) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(src));
        x[0]           = f.x;
        x[1]           = f.y;
    } else {
        static_assert(E == 4, "a lane holds 2, 4 or a multiple of 8 features");
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
    } else if constexpr (E == 2) {
        *reinterpret_cast<__nv_bfloat162*>(out) = __floats2bfloat162_rn(x[0], x[1]);
    } else {
        uint2 raw;
        auto* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
        for (int j = 0; j < 2; ++j) { h[j] = __floats2bfloat162_rn(x[2 * j], x[2 * j + 1]); }
        *reinterpret_cast<uint2*>(out) = raw;
    }
}

struct QkvParams {
    const __nv_bfloat16* src[3]; // q, k, v
    std::int64_t stride[3];      // elements between tokens
    const __nv_bfloat16* bias[3];
    const __nv_bfloat16* norm[2]; // q, k
    const float* inv_freq;
    const std::int32_t* positions;
    __nv_bfloat16* q_out;
    __nv_bfloat16* k_cache;
    __nv_bfloat16* v_cache;
    // Without a cache: token t's key and value rows in column t of [D, Hkv, T].
    __nv_bfloat16* k_out;
    __nv_bfloat16* v_out;
    // Where token t's keys and values go: its batch row t / row_tokens's binding (kv_address.h),
    // or nowhere past the row's stored count.
    detail::Gemma4KvAddress kv;
    int row_tokens;
    // FP8 cache planes (k_scale non-null): codes and FP32 row scales instead of k/v_cache.
    std::uint8_t* k_codes;
    std::uint8_t* v_codes;
    float* k_scale;
    float* v_scale;
    float eps;
    float rotary_scale;
    float query_beta;
    int query_period;
    int q_heads;
    int kv_heads;
    int rotary_dim;
    int norm_scope; // QkNorm
    bool norm_unit_offset;
};

// A norm weight as its gain: 1 + w under Gemma's RMSNorm (unit offset), else w itself.
__device__ __forceinline__ float norm_gain(__nv_bfloat16 w, bool unit_offset) {
    const float g = __bfloat162float(w);
    return unit_offset ? 1.0f + g : g;
}

constexpr int kQkvThreads = 256;
constexpr int kQkvWarps   = kQkvThreads / 32;

// Row r of a token: [0, Hq) queries, [Hq, Hq+Hkv) keys, then values.
struct RowRef {
    int kind; // 0 q, 1 k, 2 v
    int head;
};

__device__ __forceinline__ RowRef row_ref(int r, int q_heads, int kv_heads) {
    if (r < q_heads) { return {0, r}; }
    if (r < q_heads + kv_heads) { return {1, r - q_heads}; }
    return {2, r - q_heads - kv_heads};
}

// One block per token (a small T splits a token's rows over gridDim.y blocks, except under a
// projection-wide norm, whose statistic needs every row of the token in one block). Lane l of a
// warp holds the E = D/32 contiguous features [l*E, (l+1)*E) of one row; the rotation reads the
// partner feature from the warp's staged row, so any rotary width works.
template <int D>
__global__ void __launch_bounds__(kQkvThreads) dense_qkv_prepare_kernel(QkvParams p) {
    constexpr int E = D / 32;
    __shared__ float cosines[D / 2];
    __shared__ float sines[D / 2];
    __shared__ float staged[kQkvWarps][D];
    __shared__ float scratch[kQkvWarps];

    // A programmatic dependent (the attention) may launch now: it waits before it reads.
    if (threadIdx.x == 0) pdl::trigger_dependents();
    // Joined to the q/k/v projection where the launcher says so; a no-op otherwise.
    pdl::wait_for_dependencies();
    const int t       = blockIdx.x;
    const int lane    = threadIdx.x & 31;
    const int warp    = threadIdx.x >> 5;
    const int pos     = p.positions[t];
    const int half    = p.rotary_dim / 2;
    const bool rotary = p.inv_freq != nullptr;
    if (rotary) {
        for (int i = threadIdx.x; i < half; i += kQkvThreads) {
            float s, c;
            rope_angle_sincos(static_cast<double>(pos) * static_cast<double>(p.inv_freq[i]), &s,
                              &c);
            cosines[i] = c * p.rotary_scale;
            sines[i]   = s * p.rotary_scale;
        }
    }
    float query_factor = 1.0f;
    if (p.query_period > 0) {
        query_factor = static_cast<float>(
            1.0 + static_cast<double>(p.query_beta) *
                      log(1.0 + floor(static_cast<double>(pos) / p.query_period)));
    }

    const int rows = p.q_heads + 2 * p.kv_heads;
    // A projection-wide norm: the sum of squares of every query and every key feature first.
    float inverse[2] = {0.0f, 0.0f};
    if (p.norm_scope == static_cast<int>(QkNorm::Projection)) {
        float sums[2] = {0.0f, 0.0f};
        for (int r = warp; r < p.q_heads + p.kv_heads; r += kQkvWarps) {
            const RowRef ref = row_ref(r, p.q_heads, p.kv_heads);
            float x[E];
            load_features<E>(p.src[ref.kind] + t * p.stride[ref.kind] +
                                 static_cast<std::int64_t>(ref.head) * D + lane * E,
                             x);
            if (p.bias[ref.kind] != nullptr) {
#pragma unroll
                for (int j = 0; j < E; ++j) {
                    x[j] += __bfloat162float(p.bias[ref.kind][ref.head * D + lane * E + j]);
                }
            }
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < E; ++j) { s += x[j] * x[j]; }
            sums[ref.kind] += s;
        }
        const float q_total = block_sum<kQkvThreads>(sums[0], scratch);
        const float k_total = block_sum<kQkvThreads>(sums[1], scratch);
        inverse[0]          = rsqrtf(q_total / (p.q_heads * D) + p.eps);
        inverse[1]          = rsqrtf(k_total / (p.kv_heads * D) + p.eps);
    } else {
        __syncthreads(); // the angle table
    }

    for (int r = blockIdx.y * kQkvWarps + warp; r < rows; r += gridDim.y * kQkvWarps) {
        const RowRef ref = row_ref(r, p.q_heads, p.kv_heads);
        float x[E];
        load_features<E>(p.src[ref.kind] + t * p.stride[ref.kind] +
                             static_cast<std::int64_t>(ref.head) * D + lane * E,
                         x);
        if (p.bias[ref.kind] != nullptr) {
#pragma unroll
            for (int j = 0; j < E; ++j) {
                x[j] += __bfloat162float(p.bias[ref.kind][ref.head * D + lane * E + j]);
            }
        }
        if (ref.kind < 2) {
            // Warp-uniform: every lane of the warp works on the same row.
            if (p.norm_scope == static_cast<int>(QkNorm::Head)) {
                float squares = 0.0f;
#pragma unroll
                for (int j = 0; j < E; ++j) { squares += x[j] * x[j]; }
                const float inv = rsqrtf(warp_sum(squares) / D + p.eps);
#pragma unroll
                for (int j = 0; j < E; ++j) {
                    x[j] *= inv * norm_gain(p.norm[ref.kind][lane * E + j], p.norm_unit_offset);
                }
            } else if (p.norm_scope == static_cast<int>(QkNorm::Projection)) {
                const float inv = inverse[ref.kind];
#pragma unroll
                for (int j = 0; j < E; ++j) {
                    x[j] *= inv * norm_gain(p.norm[ref.kind][ref.head * D + lane * E + j],
                                            p.norm_unit_offset);
                }
            }
            if (rotary) {
#pragma unroll
                for (int j = 0; j < E; ++j) { staged[warp][lane * E + j] = x[j]; }
                __syncwarp();
#pragma unroll
                for (int j = 0; j < E; ++j) {
                    const int f = lane * E + j;
                    if (f < half) {
                        x[j] = fmaf(x[j], cosines[f], -staged[warp][f + half] * sines[f]);
                    } else if (f < p.rotary_dim) {
                        const int i = f - half;
                        x[j]        = fmaf(x[j], cosines[i], staged[warp][i] * sines[i]);
                    }
                }
                __syncwarp();
            }
        }
        __nv_bfloat16* out;
        bool store = true;
        if (ref.kind == 0) {
#pragma unroll
            for (int j = 0; j < E; ++j) { x[j] *= query_factor; }
            out = p.q_out + (static_cast<std::int64_t>(t) * p.q_heads + ref.head) * D;
        } else if (p.k_out != nullptr) {
            out = (ref.kind == 1 ? p.k_out : p.v_out) +
                  (static_cast<std::int64_t>(t) * p.kv_heads + ref.head) * D;
        } else {
            // The binding's row (a scale plane is its code plane without the D axis).
            const int batch_row    = t / p.row_tokens;
            store = detail::gemma4_kv_stores(p.kv, batch_row, t - batch_row * p.row_tokens);
            const int binding      = detail::gemma4_kv_binding(p.kv, batch_row);
            const std::int64_t row = detail::gemma4_kv_plane_offset(p.kv, binding) / D +
                                     static_cast<std::int64_t>(ref.head) * p.kv.capacity +
                                     detail::gemma4_kv_slot(p.kv, binding, pos);
            if (p.k_scale != nullptr) {
                if (store) {
                    detail::gemma4_fp8_store_row<E>((ref.kind == 1 ? p.k_codes : p.v_codes) +
                                                        row * D,
                                                    (ref.kind == 1 ? p.k_scale : p.v_scale) + row,
                                                    x, lane);
                }
                continue;
            }
            out = (ref.kind == 1 ? p.k_cache : p.v_cache) + row * D;
        }
        if (store) { store_features<E>(out + lane * E, x); }
    }
}

// ---- residual_norm ----------------------------------------------------------------------------

struct ResidualParams {
    __nv_bfloat16* x;
    const __nv_bfloat16* branch;
    const __nv_bfloat16* bias;
    const __nv_bfloat16* post_weight;
    const __nv_bfloat16* out_weight;
    __nv_bfloat16* out;
    int width;
    float eps;      // the output norm's
    float post_eps; // the post-norm's
    float scale;
    bool unit_offset;     // the post-norm's gains 1 + w (Gemma's RMSNorm)
    bool out_unit_offset; // the output norm's
};

constexpr int kResidualThreads  = 512;
constexpr int kResidualMaxWidth = 8192;
constexpr int kResidualVectors  = kResidualMaxWidth / 8 / kResidualThreads;

__device__ __forceinline__ void unpack8(const uint4& raw, float* out) {
    const auto* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        out[2 * j]     = f.x;
        out[2 * j + 1] = f.y;
    }
}

// One block per column; every vector a thread needs is requested in one burst into registers.
template <bool kPost, bool kOut>
__global__ void __launch_bounds__(kResidualThreads) residual_norm_kernel(ResidualParams p) {
    __shared__ float scratch[kResidualThreads / 32];
    // A programmatic dependent (a decode projection of the output) may launch now: it waits for
    // this kernel before it reads, and streams its weights meanwhile.
    if (threadIdx.x == 0) pdl::trigger_dependents();
    // Joined to the branch's projection where the launcher says so; a no-op otherwise.
    pdl::wait_for_dependencies();
    const int vectors       = p.width / 8;
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * p.width;
    const auto load         = [](const __nv_bfloat16* src, int v) {
        return *reinterpret_cast<const uint4*>(src + static_cast<std::int64_t>(v) * 8);
    };
    uint4 rb[kResidualVectors], rx[kResidualVectors], rbias[kResidualVectors];
    uint4 wp[kResidualVectors], wo[kResidualVectors];
#pragma unroll
    for (int c = 0; c < kResidualVectors; ++c) {
        const int v = threadIdx.x + c * kResidualThreads;
        if (v < vectors) {
            rb[c] = load(p.branch + base, v);
            rx[c] = load(p.x + base, v);
            if (p.bias != nullptr) { rbias[c] = load(p.bias, v); }
            if constexpr (kPost) { wp[c] = load(p.post_weight, v); }
            if constexpr (kOut) { wo[c] = load(p.out_weight, v); }
        }
    }
    float b[kResidualVectors][8];
    float sb = 0.0f;
#pragma unroll
    for (int c = 0; c < kResidualVectors; ++c) {
        if (threadIdx.x + c * kResidualThreads < p.width / 8) {
            unpack8(rb[c], b[c]);
            if (p.bias != nullptr) {
                float fb[8];
                unpack8(rbias[c], fb);
#pragma unroll
                for (int j = 0; j < 8; ++j) { b[c][j] += fb[j]; }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) { sb += b[c][j] * b[c][j]; }
        }
    }
    float branch_factor = p.scale;
    if constexpr (kPost) {
        branch_factor *= rsqrtf(block_sum<kResidualThreads>(sb, scratch) / p.width + p.post_eps);
    }
    float sx = 0.0f;
#pragma unroll
    for (int c = 0; c < kResidualVectors; ++c) {
        const int v = threadIdx.x + c * kResidualThreads;
        if (v < vectors) {
            float fx[8], fp[8];
            unpack8(rx[c], fx);
            if constexpr (kPost) {
                unpack8(wp[c], fp);
                if (p.unit_offset) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { fp[j] += 1.0f; }
                }
            }
            uint4 stored;
            auto* h = reinterpret_cast<__nv_bfloat162*>(&stored);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float u0 = b[c][2 * j] * branch_factor;
                float u1 = b[c][2 * j + 1] * branch_factor;
                if constexpr (kPost) {
                    u0 *= fp[2 * j];
                    u1 *= fp[2 * j + 1];
                }
                h[j]           = __floats2bfloat162_rn(fx[2 * j] + u0, fx[2 * j + 1] + u1);
                const float2 r = __bfloat1622float2(h[j]);
                b[c][2 * j]    = r.x;
                b[c][2 * j + 1] = r.y;
                sx += r.x * r.x + r.y * r.y;
            }
            *reinterpret_cast<uint4*>(p.x + base + static_cast<std::int64_t>(v) * 8) = stored;
        }
    }
    if constexpr (kOut) {
        const float inv = rsqrtf(block_sum<kResidualThreads>(sx, scratch) / p.width + p.eps);
#pragma unroll
        for (int c = 0; c < kResidualVectors; ++c) {
            const int v = threadIdx.x + c * kResidualThreads;
            if (v < vectors) {
                float fo[8];
                unpack8(wo[c], fo);
                if (p.out_unit_offset) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { fo[j] += 1.0f; }
                }
                uint4 o;
                auto* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    h[j] = __floats2bfloat162_rn(b[c][2 * j] * inv * fo[2 * j],
                                                 b[c][2 * j + 1] * inv * fo[2 * j + 1]);
                }
                *reinterpret_cast<uint4*>(p.out + base + static_cast<std::int64_t>(v) * 8) = o;
            }
        }
    }
}

// ---- host-side validation ---------------------------------------------------------------------

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument(message); }
}

bool aligned16(const void* ptr) { return reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0; }

void require_vector(const Tensor& t, std::int32_t length, DType dtype, const char* op,
                    const char* name) {
    require(t.dtype == dtype && t.ne[0] == length && t.ne[1] == 1 && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr,
            std::string(op) + ": " + name + " must be a contiguous [" + std::to_string(length) +
                "] vector of its dtype");
}

void require_matrix(const Tensor& t, std::int32_t rows, std::int32_t columns, const char* op,
                    const char* name) {
    require(t.dtype == DType::BF16 && t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr && aligned16(t.data),
            std::string(op) + ": " + name + " must be 16-byte aligned contiguous BF16 [" +
                std::to_string(rows) + "," + std::to_string(columns) + "]");
}

std::int64_t source_stride(const Tensor& t, std::int32_t width, std::int32_t tokens,
                           const char* name) {
    require(t.dtype == DType::BF16 && t.ne[0] == width && t.ne[1] == tokens && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.data != nullptr,
            std::string("dense_qkv_prepare: ") + name + " must be BF16 [heads*D, T]");
    require(t.nb[0] == 2 && t.nb[1] >= static_cast<std::int64_t>(width) * 2 && t.nb[1] % 16 == 0 &&
                aligned16(t.data),
            std::string("dense_qkv_prepare: ") + name +
                " needs contiguous features and a 16-byte-aligned token stride");
    return t.nb[1] / 2;
}

std::size_t span_bytes(const Tensor& t) {
    return static_cast<std::size_t>((t.ne[1] - 1) * t.nb[1] + t.ne[0] * 2);
}

} // namespace

void dense_qkv_prepare(const DenseQkvPrepare& a, cudaStream_t stream) {
    constexpr const char* op = "dense_qkv_prepare";
    require(a.positions.dtype == DType::I32 && a.positions.ne[1] == 1 && a.positions.ne[2] == 1 &&
                a.positions.ne[3] == 1 && a.positions.is_contiguous() &&
                a.positions.data != nullptr && a.positions.ne[0] > 0,
            "dense_qkv_prepare: positions must be contiguous I32 [T]");
    const std::int32_t tokens = a.positions.ne[0];
    const Tensor& ck          = a.cache.k;
    const Tensor& cv          = a.cache.v;
    // Without cache planes the keys and values go to k_out / v_out, columns [D, Hkv, T].
    const bool columns = ck.data == nullptr && cv.data == nullptr;
    // Plane dtypes (BF16, or both FP8_E4M3FN with FP32 row scales) are checked by the binding.
    const bool fp8 = !columns && ck.dtype == DType::FP8_E4M3FN;
    std::int32_t d        = 0;
    std::int32_t kv_heads = 0;
    detail::Gemma4KvBinding binding{};
    std::int32_t row_tokens = tokens;
    if (columns) {
        const Tensor& ko = a.k_out;
        const Tensor& vo = a.v_out;
        d                = ko.ne[0];
        kv_heads         = ko.ne[1];
        require(d == 64 || d == 128 || d == 256,
                "dense_qkv_prepare: head_dim must be 64, 128 or 256");
        for (const Tensor* o : {&ko, &vo}) {
            require(o->dtype == DType::BF16 && o->ne[0] == d && o->ne[1] == kv_heads &&
                        o->ne[2] == tokens && o->ne[3] == 1 && o->is_contiguous() &&
                        o->data != nullptr && aligned16(o->data),
                    "dense_qkv_prepare: k_out and v_out must be contiguous BF16 [D, Hkv, T]");
        }
        require(kv_heads > 0, "dense_qkv_prepare: key/value heads must be positive");
    } else {
        require(a.k_out.data == nullptr && a.v_out.data == nullptr,
                "dense_qkv_prepare: keys and values go to the cache or to k_out/v_out, not both");
        require(ck.is_contiguous() && ck.data != nullptr && aligned16(ck.data) &&
                    cv.is_contiguous() && cv.data != nullptr && aligned16(cv.data),
                "dense_qkv_prepare: cache planes must be 16-byte aligned contiguous BF16 (or both "
                "FP8_E4M3FN)");
        d = ck.ne[0];
        // A page pool is [D, 64, pages, Hkv]; slot storage and ring images are
        // [D, C, Hkv(, images)].
        kv_heads = a.cache.block_tables.data != nullptr ? ck.ne[3] : ck.ne[2];
        require(d == 64 || d == 128 || d == 256,
                "dense_qkv_prepare: head_dim must be 64, 128 or 256");
        binding = detail::gemma4_kv_binding(a.cache, d, kv_heads, op);
        require(tokens % binding.batch == 0,
                "dense_qkv_prepare: T must split evenly over the cache's batch rows");
        row_tokens = tokens / binding.batch;
        require(kv_heads > 0 && row_tokens <= binding.address.capacity,
                "dense_qkv_prepare: cache capacity must hold T rows");
    }
    // A call without queries (q_out and q_src empty) prepares keys and values only.
    const bool queries = a.q_out.data != nullptr || a.q_src.data != nullptr;
    std::int32_t q_heads = 0;
    if (queries) {
        require(a.q_out.dtype == DType::BF16 && a.q_out.ne[0] == d && a.q_out.ne[2] == tokens &&
                    a.q_out.ne[3] == 1 && a.q_out.is_contiguous() && a.q_out.data != nullptr &&
                    aligned16(a.q_out.data),
                "dense_qkv_prepare: q_out must be contiguous BF16 [D, Hq, T]");
        q_heads = a.q_out.ne[1];
        require(q_heads > 0 && q_heads % kv_heads == 0,
                "dense_qkv_prepare: query heads must be a multiple of key/value heads");
    }

    QkvParams p{};
    const Tensor* src[3]  = {&a.q_src, &a.k_src, &a.v_src};
    const Tensor* bias[3] = {&a.q_bias, &a.k_bias, &a.v_bias};
    const std::int32_t widths[3] = {q_heads * d, kv_heads * d, kv_heads * d};
    const char* names[3]         = {"q_src", "k_src", "v_src"};
    const char* bias_names[3]    = {"q_bias", "k_bias", "v_bias"};
    for (int i = 0; i < 3; ++i) {
        if (i == 0 && !queries) {
            require(a.q_bias.data == nullptr, "dense_qkv_prepare: q_bias without queries");
            continue;
        }
        p.stride[i] = source_stride(*src[i], widths[i], tokens, names[i]);
        p.src[i]    = static_cast<const __nv_bfloat16*>(src[i]->data);
        if (bias[i]->data != nullptr) {
            require_vector(*bias[i], widths[i], DType::BF16, op, bias_names[i]);
            p.bias[i] = static_cast<const __nv_bfloat16*>(bias[i]->data);
        }
    }
    p.norm_scope       = static_cast<int>(a.qk_norm);
    p.norm_unit_offset = a.norm_unit_offset;
    if (a.qk_norm != QkNorm::None) {
        require(std::isfinite(a.eps) && a.eps > 0, "dense_qkv_prepare: eps must be positive");
        const bool head = a.qk_norm == QkNorm::Head;
        if (queries) {
            require_vector(a.q_norm, head ? d : widths[0], DType::BF16, op, "q_norm");
        }
        require_vector(a.k_norm, head ? d : widths[1], DType::BF16, op, "k_norm");
        p.norm[0] = static_cast<const __nv_bfloat16*>(a.q_norm.data);
        p.norm[1] = static_cast<const __nv_bfloat16*>(a.k_norm.data);
    }
    if (a.inv_freq.data != nullptr) {
        require(a.rotary_dim > 0 && a.rotary_dim % 2 == 0 && a.rotary_dim <= d,
                "dense_qkv_prepare: rotary_dim must be even and in (0, D]");
        require_vector(a.inv_freq, a.rotary_dim / 2, DType::FP32, op, "inv_freq");
        require(std::isfinite(a.rotary_scale) && a.rotary_scale > 0,
                "dense_qkv_prepare: rotary_scale must be positive");
        p.inv_freq   = static_cast<const float*>(a.inv_freq.data);
        p.rotary_dim = a.rotary_dim;
    }
    require(a.query_period >= 0 && std::isfinite(a.query_beta),
            "dense_qkv_prepare: the query temperature needs a finite beta and period >= 0");
    // Outputs never overlap the sources.
    const std::size_t q_bytes = static_cast<std::size_t>(q_heads) * d * tokens * 2;
    const void* k_dst         = columns ? a.k_out.data : ck.data;
    const void* v_dst         = columns ? a.v_out.data : cv.data;
    const std::size_t plane_bytes =
        columns ? static_cast<std::size_t>(d) * kv_heads * tokens * 2
                : static_cast<std::size_t>(d) * ck.ne[1] * ck.ne[2] * ck.ne[3] * (fp8 ? 1 : 2);
    for (int i = 0; i < 3; ++i) {
        if (src[i]->data == nullptr) { continue; }
        const std::size_t bytes = span_bytes(*src[i]);
        require(!overlaps(a.q_out.data, q_bytes, src[i]->data, bytes) &&
                    !overlaps(k_dst, plane_bytes, src[i]->data, bytes) &&
                    !overlaps(v_dst, plane_bytes, src[i]->data, bytes),
                "dense_qkv_prepare: outputs must not overlap the sources");
    }
    require(!overlaps(k_dst, plane_bytes, v_dst, plane_bytes) &&
                !overlaps(a.q_out.data, q_bytes, k_dst, plane_bytes) &&
                !overlaps(a.q_out.data, q_bytes, v_dst, plane_bytes),
            "dense_qkv_prepare: outputs must not overlap one another");

    p.positions    = static_cast<const std::int32_t*>(a.positions.data);
    p.q_out        = static_cast<__nv_bfloat16*>(a.q_out.data);
    p.kv           = binding.address;
    p.row_tokens   = row_tokens;
    if (columns) {
        p.k_out = static_cast<__nv_bfloat16*>(a.k_out.data);
        p.v_out = static_cast<__nv_bfloat16*>(a.v_out.data);
    } else if (binding.fp8) {
        p.k_codes = static_cast<std::uint8_t*>(ck.data);
        p.v_codes = static_cast<std::uint8_t*>(cv.data);
        p.k_scale = static_cast<float*>(a.cache.k_scale.data);
        p.v_scale = static_cast<float*>(a.cache.v_scale.data);
    } else {
        p.k_cache = static_cast<__nv_bfloat16*>(ck.data);
        p.v_cache = static_cast<__nv_bfloat16*>(cv.data);
    }
    p.eps          = a.eps;
    p.rotary_scale = a.rotary_scale;
    p.query_beta   = a.query_beta;
    p.query_period = a.query_period;
    p.q_heads      = q_heads;
    p.kv_heads     = kv_heads;
    // A decode or verify call has a handful of tokens: its rows are spread over blocks, unless a
    // projection-wide norm needs the whole token in one block.
    const int rows   = q_heads + 2 * kv_heads;
    const int groups = tokens >= 32 || a.qk_norm == QkNorm::Projection
                           ? 1
                           : (rows + kQkvWarps - 1) / kQkvWarps;
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(groups));
    const auto launch = [&](auto kernel) {
        if (a.join_previous) {
            CUDA_CHECK(pdl::launch_dependent({grid, dim3(kQkvThreads), 0, stream}, kernel, p));
        } else {
            kernel<<<grid, kQkvThreads, 0, stream>>>(p);
        }
    };
    if (d == 64) {
        launch(dense_qkv_prepare_kernel<64>);
    } else if (d == 128) {
        launch(dense_qkv_prepare_kernel<128>);
    } else {
        launch(dense_qkv_prepare_kernel<256>);
    }
    CUDA_CHECK(cudaGetLastError());
}

void residual_norm(const ResidualNorm& a, cudaStream_t stream) {
    constexpr const char* op  = "residual_norm";
    const std::int32_t width  = a.x.ne[0];
    const std::int32_t tokens = a.x.ne[1];
    const bool post           = a.post_weight.data != nullptr;
    const bool emit           = a.out.data != nullptr;
    require(width > 0 && width <= kResidualMaxWidth && width % 8 == 0,
            "residual_norm: width must be a multiple of 8 in (0, 8192]");
    require_matrix(a.x, width, tokens, op, "x");
    require_matrix(a.branch, width, tokens, op, "branch");
    require((a.out_weight.data != nullptr) == emit,
            "residual_norm: out_weight and out are given together");
    const std::size_t bytes = static_cast<std::size_t>(width) * tokens * 2;
    require(!overlaps(a.x.data, bytes, a.branch.data, bytes), "residual_norm: x overlaps branch");
    if (emit) {
        require_matrix(a.out, width, tokens, op, "out");
        require_vector(a.out_weight, width, DType::BF16, op, "out_weight");
        require(!overlaps(a.out.data, bytes, a.x.data, bytes) &&
                    !overlaps(a.out.data, bytes, a.branch.data, bytes),
                "residual_norm: out overlaps an input");
    }
    if (a.bias.data != nullptr) { require_vector(a.bias, width, DType::BF16, op, "bias"); }
    if (post) { require_vector(a.post_weight, width, DType::BF16, op, "post_weight"); }
    for (const Tensor* w : {&a.bias, &a.post_weight, &a.out_weight}) {
        require(aligned16(w->data), "residual_norm: vectors must be 16-byte aligned");
    }
    require(std::isfinite(a.scale) && std::isfinite(a.eps) && a.eps > 0,
            "residual_norm: scale must be finite and eps positive");
    require(!a.post_eps || (std::isfinite(*a.post_eps) && *a.post_eps > 0),
            "residual_norm: post_eps must be positive");
    ResidualParams p{};
    p.x               = static_cast<__nv_bfloat16*>(a.x.data);
    p.branch          = static_cast<const __nv_bfloat16*>(a.branch.data);
    p.bias            = static_cast<const __nv_bfloat16*>(a.bias.data);
    p.post_weight     = static_cast<const __nv_bfloat16*>(a.post_weight.data);
    p.out_weight      = static_cast<const __nv_bfloat16*>(a.out_weight.data);
    p.out             = static_cast<__nv_bfloat16*>(a.out.data);
    p.width           = width;
    p.eps             = a.eps;
    p.post_eps        = a.post_eps.value_or(a.eps);
    p.scale           = a.scale;
    p.unit_offset     = a.unit_offset;
    p.out_unit_offset = a.out_unit_offset.value_or(a.unit_offset);
    const auto launch = [&](auto kernel) {
        if (a.join_previous) {
            CUDA_CHECK(pdl::launch_dependent(
                {dim3(static_cast<unsigned>(tokens)), dim3(kResidualThreads), 0, stream}, kernel,
                p));
        } else {
            kernel<<<tokens, kResidualThreads, 0, stream>>>(p);
        }
    };
    if (post && emit) {
        launch(residual_norm_kernel<true, true>);
    } else if (post) {
        launch(residual_norm_kernel<true, false>);
    } else if (emit) {
        launch(residual_norm_kernel<false, true>);
    } else {
        launch(residual_norm_kernel<false, false>);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops
