// The Mamba2 mixer's Ops (include/ninferno/ops/mamba2.h).
//
// The scan runs one CTA per (head, run of p values, row) and walks the row's tokens in order:
// every thread keeps EPT state entries s[q + e*TPP, p] in registers, the TPP = N / EPT threads of
// one p sum their shares of y[p] with a shuffle butterfly. The row's tokens go through shared
// memory a tile at a time, loaded while the previous tile runs: the recurrence over a tile has no
// global load, barrier or shuffle on its path, and the tile's butterflies run side by side after
// it. A token's bytes are the ones a one-token call gives. Decode (L = 1) and prefill share it; a
// chunked SSD prefill is a later route.

#include "ninferno/ops/mamba2.h"

#include "core/device.h"
#include "core/device_class.h"
#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument("mamba2: " + what); }

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float silu(float v) { return v / (1.0F + __expf(-v)); }

// ---------------------------------------------------------------------------------------------
// Convolution: one thread per (channel, row), sequential over the row's tokens.

__global__ void conv_kernel(const __nv_bfloat16* __restrict__ x, std::int64_t ldx,
                            const __nv_bfloat16* __restrict__ w,
                            const __nv_bfloat16* __restrict__ bias,
                            __nv_bfloat16* __restrict__ states,
                            const std::int32_t* __restrict__ slots, int channels, int row_tokens,
                            __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ record) {
    // The scan that follows joins this kernel by PDL and waits for it before it reads anything this
    // kernel writes; letting it launch now puts its state loads under this kernel.
    if (threadIdx.x == 0) pdl::trigger_dependents();
    const int c = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int b = static_cast<int>(blockIdx.y);
    if (c >= channels) return;
    __nv_bfloat16* window = states + static_cast<std::int64_t>(slots[b]) * 3 * channels + c;
    float u0 = bf(window[0]), u1 = bf(window[channels]), u2 = bf(window[2 * channels]);
    const float w0 = bf(w[4 * c]), w1 = bf(w[4 * c + 1]), w2 = bf(w[4 * c + 2]), w3 = bf(w[4 * c + 3]);
    const float b0 = bias != nullptr ? bf(bias[c]) : 0.0F;
    // Launched as a programmatic dependent of the input projection (mamba2_conv's join_previous),
    // which writes x: the window (the previous token's) and the weights are read before the wait.
    // In a plain launch the wait returns at once.
    pdl::wait_for_dependencies();
    for (int j = 0; j < row_tokens; ++j) {
        const std::int64_t col = static_cast<std::int64_t>(b) * row_tokens + j;
        const __nv_bfloat16 raw = x[col * ldx + c];
        if (record != nullptr) record[col * channels + c] = raw;
        const float u3 = bf(raw);
        float acc = b0;
        acc = fmaf(w0, u0, acc);
        acc = fmaf(w1, u1, acc);
        acc = fmaf(w2, u2, acc);
        acc = fmaf(w3, u3, acc);
        out[col * channels + c] = __float2bfloat16_rn(silu(acc));
        u0 = u1;
        u1 = u2;
        u2 = u3;
    }
    if (record != nullptr) return; // verify: the window stays the source's
    window[0]            = __float2bfloat16_rn(u0);
    window[channels]     = __float2bfloat16_rn(u1);
    window[2 * channels] = __float2bfloat16_rn(u2);
}

// ---------------------------------------------------------------------------------------------
// Scan.

// Tokens a scan CTA stages at a time (at most; a short row stages fewer), and the values of B and
// of C one thread carries from a tile's loads to shared memory.
inline constexpr int kScanTile   = 8;
inline constexpr int kScanStaged = 4;

struct ScanArgs {
    const __nv_bfloat16* xbc;
    std::int64_t ldxbc; // elements between columns of xbc
    const __nv_bfloat16* dt;
    std::int64_t lddt;
    const float* dt_bias;
    const float* a_log;
    const float* d;
    void* states; // FP32 or BF16 [N, P, H, S]: the scan kernel's StateT
    const std::int32_t* slots;
    __nv_bfloat16* y;         // null: no output (a fold)
    __nv_bfloat16* dt_record; // null: none
    int heads, head_dim, groups, state, row_tokens;
    int row_stride; // columns between rows' first columns
    int store;      // write the final state back
    int tile;       // tokens staged in shared memory at a time
    float dt_min, dt_max;
    // A fold: rows given by value (source, dest, commit) instead of slots / row_tokens.
    int fold;
    // Launched as a programmatic dependent of the conv: the state loads go out under the conv's
    // tail, and the wait comes before the first read of xbc or dt.
    int pdl;
    Mamba2FoldRow rows[kMamba2FoldMaxRows];
};

// One CTA per (head, run of `rows` p values, row of the batch): a group of tpp = N / EPT lanes owns
// one p, lane q of it the state entries n = q + e * tpp (a warp's loads and stores of the state are
// whole 128-byte lines). The split over p keeps the SMs busy at decode, where one CTA per head
// left most of them idle.
// The recurrent state is stored as FP32, BF16 or FP16 (--state-dtype); the recurrence runs in FP32
// registers either way and rounds once on the store.
__device__ __forceinline__ float load_state(const float* p) { return *p; }
__device__ __forceinline__ float load_state(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float load_state(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void store_state(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_state(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
// FP16 saturates at its largest finite value instead of rounding to infinity; NaN stays NaN.
__device__ __forceinline__ void store_state(__half* p, float v) {
    *p = __float2half_rn(v != v ? v : fminf(fmaxf(v, -65504.0f), 65504.0f));
}

template <int EPT, class StateT>
__global__ void scan_kernel(ScanArgs a) {
    // The gated norm after it may launch now (a programmatic dependent): it loads its gain under
    // this grid's tail and waits before it reads y or z.
    if (threadIdx.x == 0) pdl::trigger_dependents();
    StateT* const states = static_cast<StateT*>(a.states);
    extern __shared__ float shared[];
    const int threads = static_cast<int>(blockDim.x);
    const int tpp     = a.state / EPT;
    const int rows    = threads / tpp;
    float* bs    = shared;                // [tile][N]
    float* cs    = bs + a.tile * a.state; // [tile][N]
    float* xs    = cs + a.tile * a.state; // [tile][rows]
    float* steps = xs + a.tile * rows;    // [tile][2]: dt, exp(dt * A)
    const int h     = static_cast<int>(blockIdx.x);
    const int b     = static_cast<int>(blockIdx.z);
    const int t     = static_cast<int>(threadIdx.x);
    const int q     = t % tpp;
    const int p0    = static_cast<int>(blockIdx.y) * rows;
    const int pr    = t / tpp;
    const int p     = p0 + pr;
    const int g     = h / (a.heads / a.groups);
    const int inner = a.heads * a.head_dim;
    const std::int64_t head_state = static_cast<std::int64_t>(a.state) * a.head_dim;
    const int source = a.fold ? a.rows[b].source_slot : a.slots[b];
    const int dest   = a.fold ? a.rows[b].dest_slot : source;
    const int tokens = a.fold ? a.rows[b].commit : a.row_tokens;
    const std::int64_t offset = static_cast<std::int64_t>(p) * a.state + q;
    const StateT* s = states + (static_cast<std::int64_t>(source) * a.heads + h) * head_state + offset;
    float r[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) r[e] = load_state(s + e * tpp);
    const float A   = -__expf(a.a_log[h]);
    const float D   = a.d[h];
    const float dtb = a.dt_bias[h];
    // ScanArgs::pdl: the state is the previous token's (written two or more kernels ago), the
    // head's constants are weights; xbc and dt are the conv's and the projection's.
    // In a plain launch the wait returns at once.
    pdl::wait_for_dependencies();
    const std::int64_t first = static_cast<std::int64_t>(b) * a.row_stride;
    // A tile's operands as read from global memory, held in registers until they go to shared
    // memory: the next tile's loads are in flight while the CTA runs the current one. The tile
    // is sized so that a thread holds at most kScanStaged values of B and of C, one x and one dt.
    __nv_bfloat16 staged_b[kScanStaged], staged_c[kScanStaged], staged_x{}, staged_dt{};
    const auto fetch = [&](int j0, int n) {
#pragma unroll
        for (int u = 0; u < kScanStaged; ++u) {
            const int i = t + u * threads;
            if (i < n * a.state) {
                const int j = i / a.state;
                const int k = i - j * a.state;
                const __nv_bfloat16* column = a.xbc + (first + j0 + j) * a.ldxbc;
                staged_b[u] = column[inner + g * a.state + k];
                staged_c[u] = column[inner + (a.groups + g) * a.state + k];
            }
        }
        if (t < n * rows) {
            const int j = t / rows;
            staged_x = a.xbc[(first + j0 + j) * a.ldxbc + h * a.head_dim + p0 + (t - j * rows)];
        }
        if (t < n) staged_dt = a.dt[(first + j0 + t) * a.lddt + h];
    };
    if (tokens > 0) fetch(0, min(a.tile, tokens));
    for (int j0 = 0; j0 < tokens; j0 += a.tile) {
        const int n = min(a.tile, tokens - j0);
#pragma unroll
        for (int u = 0; u < kScanStaged; ++u) {
            const int i = t + u * threads;
            if (i < n * a.state) {
                bs[i] = bf(staged_b[u]);
                cs[i] = bf(staged_c[u]);
            }
        }
        if (t < n * rows) xs[t] = bf(staged_x);
        if (t < n) {
            if (a.dt_record != nullptr) a.dt_record[(first + j0 + t) * a.heads + h] = staged_dt;
            const float v = bf(staged_dt) + dtb;
            float dt      = v > 20.0F ? v : log1pf(__expf(v));
            dt            = fmaxf(dt, a.dt_min);
            if (a.dt_max > 0.0F) dt = fminf(dt, a.dt_max);
            steps[2 * t]     = dt;
            steps[2 * t + 1] = __expf(dt * A);
        }
        __syncthreads();
        if (j0 + a.tile < tokens) fetch(j0 + a.tile, min(a.tile, tokens - j0 - a.tile));
        // Only r carries from token to token: every token's share of y waits in a register, and
        // the tile's shuffle trees run after the recurrence, the tokens' shuffles side by side.
        // Each tree is the butterfly a token alone would take, so the bytes do not depend on the
        // tile or on the call's length.
        float acc[kScanTile];
#pragma unroll
        for (int j = 0; j < kScanTile; ++j) {
            acc[j] = 0.0F;
            if (j < n) {
                const float dt = steps[2 * j], decay = steps[2 * j + 1];
                const float dx = dt * xs[j * rows + pr];
                const float* bj = bs + j * a.state;
                const float* cj = cs + j * a.state;
#pragma unroll
                for (int e = 0; e < EPT; ++e) {
                    r[e]   = fmaf(r[e], decay, dx * bj[q + e * tpp]);
                    acc[j] = fmaf(r[e], cj[q + e * tpp], acc[j]);
                }
            }
        }
        // Only the tile's tokens: a decode step's tile of one runs one butterfly, not kScanTile.
        for (int off = tpp >> 1; off > 0; off >>= 1) {
#pragma unroll
            for (int j = 0; j < kScanTile; ++j) {
                if (j < n) acc[j] += __shfl_xor_sync(0xffffffffU, acc[j], off);
            }
        }
        if (q == 0 && a.y != nullptr) {
#pragma unroll
            for (int j = 0; j < kScanTile; ++j) {
                if (j < n) {
                    a.y[(first + j0 + j) * inner + h * a.head_dim + p] =
                        __float2bfloat16_rn(fmaf(D, xs[j * rows + pr], acc[j]));
                }
            }
        }
        __syncthreads();
    }
    if (!a.store) return;
    StateT* out = states + (static_cast<std::int64_t>(dest) * a.heads + h) * head_state + offset;
#pragma unroll
    for (int e = 0; e < EPT; ++e) store_state(out + e * tpp, r[e]);
}

// The conv half of a fold: one thread per (channel, row); dest window = the last 3 of {source
// window, the first `commit` recorded inputs}. Reads everything before it writes (dest may be
// the source).
struct ConvFoldArgs {
    const __nv_bfloat16* record;
    __nv_bfloat16* states;
    int channels, width;
    Mamba2FoldRow rows[kMamba2FoldMaxRows];
};

__global__ void conv_fold_kernel(ConvFoldArgs a) {
    const int c = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int b = static_cast<int>(blockIdx.y);
    if (c >= a.channels) return;
    const Mamba2FoldRow row = a.rows[b];
    const __nv_bfloat16* src =
        a.states + static_cast<std::int64_t>(row.source_slot) * 3 * a.channels + c;
    __nv_bfloat16 u0 = src[0], u1 = src[a.channels], u2 = src[2 * a.channels];
    const __nv_bfloat16* rec = a.record + static_cast<std::int64_t>(b) * a.width * a.channels + c;
    for (int j = 0; j < row.commit; ++j) {
        u0 = u1;
        u1 = u2;
        u2 = rec[static_cast<std::int64_t>(j) * a.channels];
    }
    __nv_bfloat16* dst = a.states + static_cast<std::int64_t>(row.dest_slot) * 3 * a.channels + c;
    dst[0]              = u0;
    dst[a.channels]     = u1;
    dst[2 * a.channels] = u2;
}

// ---------------------------------------------------------------------------------------------
// Gated group norm: one CTA per (group, column).

__global__ void gated_norm_kernel(const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ z,
                                  std::int64_t ldz, const __nv_bfloat16* __restrict__ w, int rows,
                                  int group, float eps, __nv_bfloat16* __restrict__ out) {
    // A programmatic dependent (the mixer's output projection at decode) may launch now: it waits
    // for this kernel before it reads, and streams its weights meanwhile.
    if (threadIdx.x == 0) pdl::trigger_dependents();
    const int gi  = static_cast<int>(blockIdx.x);
    const std::int64_t col = blockIdx.y;
    const int r0  = gi * group;
    float sum = 0.0F;
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = bf(y[col * rows + r0 + i]) * silu(bf(z[col * ldz + r0 + i]));
        sum = fmaf(v, v, sum);
    }
    __shared__ float part[32];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffU, sum, off);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x + 31) / 32 ? part[threadIdx.x] : 0.0F;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffU, v, off);
        if (threadIdx.x == 0) part[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(part[0] / static_cast<float>(group) + eps);
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = bf(y[col * rows + r0 + i]) * silu(bf(z[col * ldz + r0 + i]));
        const float n = __bfloat162float(__float2bfloat16_rn(v * inv));
        out[col * rows + r0 + i] = __float2bfloat16_rn(bf(w[r0 + i]) * n);
    }
}

// The same norm by one warp per (group, column), for a group of at most 32 * 8 * Runs features in
// 16-byte runs: lane l holds runs l, l + 32, ... of y, z and the gain in registers, so each is read
// once (the CTA form reads y and z twice, two features a thread, and reduces through shared
// memory). Launched as the scan's programmatic dependent: the gain (a weight) loads before the
// wait; y and z after it - z is the input projection's, and the projection, the conv and the scan
// each release their dependent early, so only the scan's completion orders it. y and z are not
// const __restrict__: nvcc hoisted their read-only loads above griddepcontrol.wait (Runs = 4).
template <int Runs>
__global__ void __launch_bounds__(32)
    gated_norm_warp_kernel(const __nv_bfloat16* y, const __nv_bfloat16* z, std::int64_t ldz,
                           const __nv_bfloat16* __restrict__ w, int rows, int group, float eps,
                           __nv_bfloat16* __restrict__ out) {
    if (threadIdx.x == 0) pdl::trigger_dependents();
    const int lane         = static_cast<int>(threadIdx.x);
    const int r0           = static_cast<int>(blockIdx.x) * group;
    const std::int64_t col = blockIdx.y;
    uint4 zr[Runs], wr[Runs], yr[Runs];
#pragma unroll
    for (int i = 0; i < Runs; ++i) {
        const int f = (lane + 32 * i) * 8;
        if (f < group) wr[i] = *reinterpret_cast<const uint4*>(w + r0 + f);
    }
    pdl::wait_for_dependencies();
#pragma unroll
    for (int i = 0; i < Runs; ++i) {
        const int f = (lane + 32 * i) * 8;
        if (f < group) {
            yr[i] = *reinterpret_cast<const uint4*>(y + col * rows + r0 + f);
            zr[i] = *reinterpret_cast<const uint4*>(z + col * ldz + r0 + f);
        }
    }
    float v[Runs][8];
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < Runs; ++i) {
        if ((lane + 32 * i) * 8 >= group) continue;
        const auto* yh = reinterpret_cast<const __nv_bfloat16*>(&yr[i]);
        const auto* zh = reinterpret_cast<const __nv_bfloat16*>(&zr[i]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[i][j] = bf(yh[j]) * silu(bf(zh[j]));
            sum     = fmaf(v[i][j], v[i][j], sum);
        }
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffU, sum, off);
    const float inv = rsqrtf(sum / static_cast<float>(group) + eps);
#pragma unroll
    for (int i = 0; i < Runs; ++i) {
        const int f = (lane + 32 * i) * 8;
        if (f >= group) continue;
        const auto* wh = reinterpret_cast<const __nv_bfloat16*>(&wr[i]);
        uint4 packed;
        auto* o = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float n = __bfloat162float(__float2bfloat16_rn(v[i][j] * inv));
            o[j]          = __float2bfloat16_rn(bf(wh[j]) * n);
        }
        *reinterpret_cast<uint4*>(out + col * rows + r0 + f) = packed;
    }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

std::int64_t column_stride(const Tensor& t, const char* name) {
    if (t.dtype != DType::BF16 || t.nb[0] != 2 || t.ne[2] != 1 || t.ne[3] != 1) {
        fail(std::string(name) + " must be BF16 [R, T] with unit row stride");
    }
    if (t.nb[1] % 2 != 0 || t.nb[1] < 2LL * t.ne[0]) fail(std::string(name) + " column stride");
    return t.nb[1] / 2;
}

bool contiguous_bf16(const Tensor& t, std::int32_t rows, std::int32_t cols) {
    return t.dtype == DType::BF16 && t.is_contiguous() && t.ne[0] == rows && t.ne[1] == cols &&
           t.ne[2] == 1 && t.ne[3] == 1;
}

bool fp32_vector(const Tensor& t, std::int32_t n) {
    return t.dtype == DType::FP32 && t.is_contiguous() && t.numel() == n;
}

void check_slots(const Tensor& slots, std::int32_t batch, std::int32_t tokens) {
    if (slots.dtype != DType::I32 || !slots.is_contiguous() || slots.numel() < batch || batch < 1 ||
        tokens < 1 || tokens % batch != 0) {
        fail("slots must be I32 [B] with T a positive multiple of B");
    }
}

} // namespace

void mamba2_conv(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& states,
                 const Tensor& slots, std::int32_t batch, Tensor& out, cudaStream_t stream,
                 Tensor* input_record, bool join_previous) {
    const std::int32_t c = x.ne[0], t = x.ne[1];
    const auto ldx = column_stride(x, "x");
    check_slots(slots, batch, t);
    if (!contiguous_bf16(weight, 4, c)) fail("weight must be contiguous BF16 [4, C]");
    if (bias.data != nullptr && (bias.dtype != DType::BF16 || bias.numel() != c)) {
        fail("bias must be BF16 [C]");
    }
    if (states.dtype != DType::BF16 || !states.is_contiguous() || states.ne[0] != c ||
        states.ne[1] != 3) {
        fail("states must be contiguous BF16 [C, 3, S]");
    }
    if (!contiguous_bf16(out, c, t)) fail("out must be contiguous BF16 [C, T]");
    if (input_record != nullptr && !contiguous_bf16(*input_record, c, t)) {
        fail("input_record must be contiguous BF16 [C, T]");
    }
    const dim3 grid((c + 127) / 128, static_cast<unsigned>(batch));
    const auto* xp  = static_cast<const __nv_bfloat16*>(x.data);
    const auto* wp  = static_cast<const __nv_bfloat16*>(weight.data);
    const auto* bp  = static_cast<const __nv_bfloat16*>(bias.data);
    auto* sp        = static_cast<__nv_bfloat16*>(states.data);
    const auto* slp = static_cast<const std::int32_t*>(slots.data);
    auto* op        = static_cast<__nv_bfloat16*>(out.data);
    auto* rp = input_record != nullptr ? static_cast<__nv_bfloat16*>(input_record->data) : nullptr;
    // join_previous: the kernel before is a projection that released this one (ops::
    // linear_release_next); the conv waits for it before it reads x.
    CUDA_CHECK(pdl::launch({grid, dim3(128), 0, stream},
                           join_previous && device_class().features.pdl, false, conv_kernel, xp,
                           ldx, wp, bp, sp, slp, c, t / batch, op, rp));
}

namespace {

void check_scan_operands(const Mamba2Geometry& g, const Tensor& dt_bias, const Tensor& a_log,
                         const Tensor& d, const Tensor& states) {
    if (g.heads < 1 || g.head_dim < 1 || g.groups < 1 || g.state < 1 || g.heads % g.groups != 0) {
        fail("invalid geometry");
    }
    if (!fp32_vector(dt_bias, g.heads) || !fp32_vector(a_log, g.heads) || !fp32_vector(d, g.heads)) {
        fail("dt_bias, a_log and d must be FP32 [H]");
    }
    if ((states.dtype != DType::FP32 && states.dtype != DType::BF16 &&
         states.dtype != DType::FP16) ||
        !states.is_contiguous() ||
        states.ne[0] != g.state || states.ne[1] != g.head_dim || states.ne[2] != g.heads) {
        fail("states must be contiguous FP32, BF16 or FP16 [N, P, H, S]");
    }
}

void launch_scan_kernel(void (*kernel)(ScanArgs), const ScanArgs& a, dim3 grid, int threads,
                        std::size_t smem, cudaStream_t stream) {
    CUDA_CHECK(pdl::launch({grid, dim3(threads), smem, stream}, a.pdl != 0, false, kernel, a));
}

// Fills the operands every scan shares and launches B rows.
void launch_scan(ScanArgs& a, const Mamba2Geometry& g, const Tensor& dt_bias, const Tensor& a_log,
                 const Tensor& d, Tensor& states, std::int32_t batch, cudaStream_t stream) {
    // EPT: the fewest state entries per thread with at most 8 threads per p; a CTA takes the
    // most p values that divide P within 256 threads, in whole warps (every lane joins the
    // shuffles of its p: tpp divides 32). Eight lanes per p, not 32: a token's butterfly is three
    // shuffles instead of five and a CTA holds four times the p values. On the 5090 at the
    // Nemotron 3.5 geometry (H 64, P 64, N 128, FP32 state) the decode scan went 5.65 -> 4.24 us
    // and a 512-token row 352 -> 203 us; 16 or 32 lanes per p were slower both ways.
    int ept = 1;
    while (ept < g.state && g.state / ept > 8) ept *= 2;
    if (g.state % ept != 0) fail("state size must be a multiple of " + std::to_string(ept));
    const int tpp = g.state / ept;
    if ((tpp & (tpp - 1)) != 0) fail("state size / entries per thread must be a power of two");
    int rows = std::min(g.head_dim, 256 / tpp);
    while (rows > 0 && (g.head_dim % rows != 0 || rows * tpp % 32 != 0)) --rows;
    if (rows == 0) fail("head_dim admits no run of p values in whole warps");
    const int threads = rows * tpp;
    a.dt_bias  = static_cast<const float*>(dt_bias.data);
    a.a_log    = static_cast<const float*>(a_log.data);
    a.d        = static_cast<const float*>(d.data);
    a.states   = states.data;
    a.heads    = g.heads;
    a.head_dim = g.head_dim;
    a.groups   = g.groups;
    a.state    = g.state;
    a.dt_min   = g.dt_min;
    a.dt_max   = g.dt_max;
    const dim3 grid(static_cast<unsigned>(g.heads), static_cast<unsigned>(g.head_dim / rows),
                    static_cast<unsigned>(batch));
    // The tile: up to kScanTile tokens and no more than a row holds (a decode step stages one
    // token); a thread stages at most kScanStaged values of B and of C and one x.
    a.tile = std::max(1, std::min({kScanTile, kScanStaged * threads / g.state, tpp, a.row_tokens}));
    if (a.tile * g.state > kScanStaged * threads) {
        fail("state size " + std::to_string(g.state) + " exceeds a CTA's staged tile");
    }
    const std::size_t smem = sizeof(float) * a.tile * (2 * g.state + rows + 2);
    switch (ept) {
#define NINFERNO_MAMBA2_SCAN(E)                                                                    \
    case E:                                                                                        \
        if (states.dtype == DType::BF16)                                                           \
            launch_scan_kernel(scan_kernel<E, __nv_bfloat16>, a, grid, threads, smem, stream);     \
        else if (states.dtype == DType::FP16)                                                      \
            launch_scan_kernel(scan_kernel<E, __half>, a, grid, threads, smem, stream);            \
        else                                                                                       \
            launch_scan_kernel(scan_kernel<E, float>, a, grid, threads, smem, stream);             \
        break;
        NINFERNO_MAMBA2_SCAN(1)
        NINFERNO_MAMBA2_SCAN(2)
        NINFERNO_MAMBA2_SCAN(4)
        NINFERNO_MAMBA2_SCAN(8)
        NINFERNO_MAMBA2_SCAN(16)
        NINFERNO_MAMBA2_SCAN(32)
        NINFERNO_MAMBA2_SCAN(64)
        NINFERNO_MAMBA2_SCAN(128)
#undef NINFERNO_MAMBA2_SCAN
    default:
        fail("state size " + std::to_string(g.state) + " is not supported");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void mamba2_scan(const Tensor& xbc, const Tensor& dt, const Tensor& dt_bias, const Tensor& a_log,
                 const Tensor& d, Tensor& states, const Tensor& slots, std::int32_t batch,
                 const Mamba2Geometry& g, Tensor& y, cudaStream_t stream, bool store_state,
                 Tensor* dt_record) {
    const std::int32_t inner = g.heads * g.head_dim;
    const std::int32_t t     = xbc.ne[1];
    check_scan_operands(g, dt_bias, a_log, d, states);
    const auto ldxbc = column_stride(xbc, "xbc");
    if (xbc.ne[0] != inner + 2 * g.groups * g.state) fail("xbc must have H*P + 2*G*N rows");
    const auto lddt = column_stride(dt, "dt");
    if (dt.ne[0] != g.heads || dt.ne[1] != t) fail("dt must be [H, T]");
    check_slots(slots, batch, t);
    if (!contiguous_bf16(y, inner, t)) fail("y must be contiguous BF16 [H*P, T]");
    if (dt_record != nullptr && !contiguous_bf16(*dt_record, g.heads, t)) {
        fail("dt_record must be contiguous BF16 [H, T]");
    }
    ScanArgs a{};
    a.xbc        = static_cast<const __nv_bfloat16*>(xbc.data);
    a.ldxbc      = ldxbc;
    a.dt         = static_cast<const __nv_bfloat16*>(dt.data);
    a.lddt       = lddt;
    a.slots      = static_cast<const std::int32_t*>(slots.data);
    a.y          = static_cast<__nv_bfloat16*>(y.data);
    a.dt_record  = dt_record != nullptr ? static_cast<__nv_bfloat16*>(dt_record->data) : nullptr;
    a.row_tokens = t / batch;
    a.row_stride = t / batch;
    a.store      = store_state ? 1 : 0;
    // The conv runs right before (models/nemotron_h/pass.cu); a fold keeps stream order.
    a.pdl = device_class().features.pdl ? 1 : 0;
    launch_scan(a, g, dt_bias, a_log, d, states, batch, stream);
}

void mamba2_verify_fold(const Tensor& input_record, const Tensor& xbc_record,
                        const Tensor& dt_record, std::int32_t width,
                        std::span<const Mamba2FoldRow> rows, const Tensor& dt_bias,
                        const Tensor& a_log, const Tensor& d, Tensor& conv_states, Tensor& states,
                        const Mamba2Geometry& g, cudaStream_t stream) {
    check_scan_operands(g, dt_bias, a_log, d, states);
    const std::int32_t channels = g.heads * g.head_dim + 2 * g.groups * g.state;
    const auto batch            = static_cast<std::int32_t>(rows.size());
    if (batch < 1 || rows.size() > kMamba2FoldMaxRows || width < 1) {
        fail("a fold takes 1 to " + std::to_string(kMamba2FoldMaxRows) + " rows of width >= 1");
    }
    const std::int32_t t = width * batch;
    if (!contiguous_bf16(input_record, channels, t) || !contiguous_bf16(xbc_record, channels, t)) {
        fail("input_record and xbc_record must be contiguous BF16 [H*P + 2*G*N, width * rows]");
    }
    if (!contiguous_bf16(dt_record, g.heads, t)) fail("dt_record must be contiguous BF16 [H, width * rows]");
    if (conv_states.dtype != DType::BF16 || !conv_states.is_contiguous() ||
        conv_states.ne[0] != channels || conv_states.ne[1] != 3) {
        fail("conv_states must be contiguous BF16 [C, 3, S]");
    }
    const std::int64_t conv_slots = conv_states.numel() / (3LL * channels);
    const std::int64_t scan_slots = states.numel() / (static_cast<std::int64_t>(g.state) * g.head_dim * g.heads);
    ConvFoldArgs cf{};
    ScanArgs a{};
    for (std::int32_t b = 0; b < batch; ++b) {
        const Mamba2FoldRow& r = rows[b];
        if (r.commit < 0 || r.commit > width) fail("commit must lie in [0, width]");
        if (r.source_slot < 0 || r.dest_slot < 0 || r.source_slot >= conv_slots ||
            r.dest_slot >= conv_slots || r.source_slot >= scan_slots || r.dest_slot >= scan_slots) {
            fail("fold slot out of range");
        }
        for (std::int32_t o = 0; o < batch; ++o) {
            if (o != b && (rows[o].dest_slot == r.dest_slot || rows[o].source_slot == r.dest_slot)) {
                fail("fold rows' dest slots must be distinct and no other row's source");
            }
        }
        cf.rows[b] = r;
        a.rows[b]  = r;
    }
    cf.record   = static_cast<const __nv_bfloat16*>(input_record.data);
    cf.states   = static_cast<__nv_bfloat16*>(conv_states.data);
    cf.channels = channels;
    cf.width    = width;
    conv_fold_kernel<<<dim3((channels + 127) / 128, static_cast<unsigned>(batch)), 128, 0, stream>>>(cf);
    CUDA_CHECK(cudaGetLastError());
    a.xbc        = static_cast<const __nv_bfloat16*>(xbc_record.data);
    a.ldxbc      = channels;
    a.dt         = static_cast<const __nv_bfloat16*>(dt_record.data);
    a.lddt       = g.heads;
    a.slots      = nullptr;
    a.y          = nullptr;
    a.dt_record  = nullptr;
    a.row_tokens = width;
    a.row_stride = width;
    a.store      = 1;
    a.fold       = 1;
    launch_scan(a, g, dt_bias, a_log, d, states, batch, stream);
}

void mamba2_gated_norm(const Tensor& y, const Tensor& z, const Tensor& weight, std::int32_t group,
                       float eps, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = y.ne[0], t = y.ne[1];
    if (!contiguous_bf16(y, rows, t) || !contiguous_bf16(out, rows, t)) {
        fail("y and out must be contiguous BF16 [R, T]");
    }
    const auto ldz = column_stride(z, "z");
    if (z.ne[0] != rows || z.ne[1] != t) fail("z must be [R, T]");
    if (weight.dtype != DType::BF16 || weight.numel() != rows) fail("weight must be BF16 [R]");
    if (group < 1 || rows % group != 0) fail("R must be a multiple of the group");
    const dim3 grid(static_cast<unsigned>(rows / group), static_cast<unsigned>(t));
    const auto* yp = static_cast<const __nv_bfloat16*>(y.data);
    const auto* zp = static_cast<const __nv_bfloat16*>(z.data);
    const auto* wp = static_cast<const __nv_bfloat16*>(weight.data);
    auto* op       = static_cast<__nv_bfloat16*>(out.data);
    // The warp form: 16-byte runs of every operand (a group of a multiple of 8 features, rows and
    // the z stride multiples of 8, aligned planes) and at most 1024 features a group.
    const int runs = (group + 255) / 256;
    if (group % 8 == 0 && rows % 8 == 0 && ldz % 8 == 0 && runs <= 4 && aligned16(yp) &&
        aligned16(zp) && aligned16(wp) && aligned16(op)) {
        const auto launch = [&](auto kernel) {
            CUDA_CHECK(pdl::launch({grid, dim3(32), 0, stream}, device_class().features.pdl, false,
                                   kernel, yp, zp, ldz, wp, rows, group, eps, op));
        };
        if (runs == 1) {
            launch(gated_norm_warp_kernel<1>);
        } else if (runs == 2) {
            launch(gated_norm_warp_kernel<2>);
        } else {
            launch(gated_norm_warp_kernel<4>);
        }
        return;
    }
    const int threads = group >= 256 ? 256 : ((group + 31) / 32) * 32;
    gated_norm_kernel<<<grid, threads, 0, stream>>>(yp, zp, ldz, wp, rows, group, eps, op);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops
