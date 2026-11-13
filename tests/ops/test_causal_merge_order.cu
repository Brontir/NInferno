// The natural merges' split loop (common/causal_merge.cuh, dense/causal_cache/bf16/merge.cuh,
// #865) against the per-thread loop it replaced, bit for bit. The merges now load the partials
// with every thread of the CTA, in 16-byte stages two ahead, load the splits' maxima and sums
// beside the positions, and a 256-column schedule narrows to 64 columns where entry O's law (the
// most loaded SM's issue and bytes, with the fixed work every narrow CTA repeats) says it is
// faster (causal_merge_shape.h); none of it may change a column's sum. The legacy kernels below are
// master's code at 66c53fca, renamed. Each case runs them at their schedule's chunk and the new
// kernels at 256 and 64 columns (whole rotated rows: 256 only) and requires the same BF16 bits at
// every split count from 1 to 256 (the partition's maximum), with zero-weight splits (no sum, or an
// underflowing maximum) holding NaN partials, garbage in the splits past the live count, several
// tokens and batch rows, masked columns, the output gate and the aligned partitions.

#include "ops/softmax_attention/common/causal_merge.cuh"
#include "ops/softmax_attention/common/causal_merge_shape.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/merge.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/schedule.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/template_launch.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace legacy {
using namespace ninferno::ops;
using namespace ninferno::ops::detail;

template <class Geometry>
__device__ __forceinline__ float
legacy_causal_merge_natural_statistics(const float* partial_m, const float* partial_l, int q_head,
                                       int token, int tokens, int splits, float* weights,
                                       float* warp_sums, float* scalars) {
    static_assert(CausalKvPartition::kMaxSplits <= 256);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const auto index   = causal_stat_index<Geometry>(q_head, token, tid, tokens);
    const float m      = tid < splits ? partial_m[index] : -CUDART_INF_F;
    const float warp_m = warp_max(m);
    if (lane == 0) warp_sums[warp] = warp_m;
    __syncthreads();
    if (warp == 0) {
        const float maximum = warp_max(tid < 8 ? warp_sums[tid] : -CUDART_INF_F);
        if (tid == 0) scalars[0] = maximum;
    }
    __syncthreads();
    const float maximum     = scalars[0];
    const float l           = tid < splits ? partial_l[index] : 0.0f;
    const float weight      = l > 0.0f && maximum > -CUDART_INF_F ? expf(m - maximum) : 0.0f;
    const float denominator = block_reduce_sum<256>(l * weight, warp_sums);
    if (tid == 0) scalars[1] = denominator;
    __syncthreads();
    const float total = scalars[1];
    if (tid < splits) weights[tid] = total > 0.0f ? weight : 0.0f;
    __syncthreads();
    return total;
}

template <class Geometry, int DChunk, bool MultiBatch, bool Masked, bool InverseRotation,
          bool Aligned = false>
__launch_bounds__(256) __global__
    void legacy_causal_natural_merge_kernel(const float* partial_acc, const float* partial_m,
                                            const float* partial_l, const std::int32_t* positions,
                                            const std::int32_t* valid_columns, std::int32_t tokens,
                                            std::int32_t batch_size, CausalKvPartition partition,
                                            __nv_bfloat16* out, const __nv_bfloat16* gate) {
    static_assert(Geometry::kHeadDim == kCausalHeadDim);
    static_assert(DChunk > 0 && DChunk <= kCausalHeadDim);
    static_assert(!InverseRotation || DChunk == kCausalHeadDim);
    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid         = static_cast<int>(threadIdx.x);
    const int split_count = partition.capacity;
    if (q_head >= Geometry::QHeads || token >= tokens) return;
    if constexpr (MultiBatch) {
        if (batch >= batch_size) return;
    }
    if constexpr (MultiBatch) positions += static_cast<std::int64_t>(batch) * tokens;
    // Aligned: each column merges the splits its own position runs
    // (CausalKvPartition::aligned_row), as a decode of that column alone does. Otherwise the
    // call's window sets the partition.
    const int window  = Aligned ? positions[token] + 1 : positions[tokens - 1] + 1;
    int output_column = token;
    if constexpr (MultiBatch) output_column += batch * tokens;
    if constexpr (Masked) {
        const int absolute_column = token;
        if (absolute_column >= valid_columns[batch]) {
            if (tid < DChunk && d_start + tid < 256) {
                const auto index = causal_q_index<Geometry>(q_head, d_start + tid, output_column);
                out[index]       = causal_gated_output(0.0f, gate ? gate + index : nullptr);
            }
            return;
        }
    }


    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * 256 * Geometry::QHeads * tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }
    const int active_splits =
        Aligned ? CausalKvPartition::aligned_active(window) : partition.active(window);
    __shared__ float weights[256], warp_sums[8], scalars[2];
    const float head_l = legacy_causal_merge_natural_statistics<Geometry>(
        partial_m, partial_l, q_head, token, tokens, active_splits, weights, warp_sums, scalars);
    const int d = d_start + tid;
    if (tid >= DChunk || d >= 256) return;
    float numerator = 0.0f;
    for (int split = 0; split < active_splits; ++split) {
        if (weights[split] != 0.0f)
            numerator +=
                partial_acc[causal_partial_index<Geometry>(q_head, d, token, split, tokens)] *
                weights[split];
    }

    const float value = head_l > 0.0F ? numerator / head_l : 0.0F;
    if constexpr (InverseRotation) {
        __shared__ float normalized[kCausalHeadDim];
        normalized[tid] = value;
        __syncthreads();
        if (tid < 32)
            causal_store_inverse_rotated_row<Geometry>(normalized, out, q_head, output_column);
    } else {
        const auto index = causal_q_index<Geometry>(q_head, d, output_column);
        out[index]       = causal_gated_output(value, gate ? gate + index : nullptr);
    }
}

// Merge one query/head's split statistics once per CTA. Published scalars are separate
// from the reduction/weight storage, so later writes cannot race another warp's scalar read.
template <class Geometry, class Schedule>
__device__ __forceinline__ float
legacy_bf16_kv_merge_statistics(const float* partial_m, const float* partial_l, int q_head,
                                int token, int tokens, int splits, float* weights, float* warp_sums,
                                float* scalars) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const auto index   = causal_stat_index<Geometry>(q_head, token, tid, tokens);
    const float m      = tid < splits ? partial_m[index] : -CUDART_INF_F;
    const float warp_m = warp_max(m);
    if (lane == 0) warp_sums[warp] = warp_m;
    __syncthreads();
    if (warp == 0) {
        const float maximum = warp_max(tid < Schedule::kWarps ? warp_sums[tid] : -CUDART_INF_F);
        if (tid == 0) scalars[0] = maximum;
    }
    __syncthreads();
    const float maximum     = scalars[0];
    const float l           = tid < splits ? partial_l[index] : 0.0f;
    const float weight      = l > 0.0f && maximum > -CUDART_INF_F ? exp2_approx(m - maximum) : 0.0f;
    const float denominator = block_reduce_sum<Schedule::kThreads>(l * weight, warp_sums);
    if (tid == 0) scalars[1] = denominator;
    __syncthreads();
    const float total = scalars[1];
    if (tid < splits) weights[tid] = total > 0.0f ? weight : 0.0f;
    __syncthreads();
    return total;
}

template <typename Geometry, class Schedule, bool MultiBatch, bool Masked, bool Aligned = false>
__launch_bounds__(Schedule::kThreads) __global__
    void legacy_bf16_kv_merge_kernel(const float* partial_acc, const float* partial_m,
                                     const float* partial_l, const std::int32_t* positions,
                                     const std::int32_t* valid_columns, std::int32_t tokens,
                                     std::int32_t batch_size, Bf16KvPartition partition,
                                     __nv_bfloat16* out, const __nv_bfloat16* gate) {
    const int split_count = partition.capacity;
    constexpr int DChunk  = Schedule::kDChunk;
    static_assert(DChunk <= Geometry::kHeadDim);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (MultiBatch) { positions += batch * tokens; }
    const int live_columns = Masked ? valid_columns[batch] : tokens;
    int output_column      = token;
    if constexpr (MultiBatch) { output_column += batch * tokens; }
    if constexpr (Masked) {
        const int absolute_column = token;
        if (absolute_column >= live_columns) {
            if (tid < DChunk && d_start + tid < Geometry::kHeadDim) {
                const auto index = causal_q_index<Geometry>(q_head, d_start + tid, output_column);
                out[index]       = causal_gated_output(0.0f, gate ? gate + index : nullptr);
            }
            return;
        }
    }


    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * Geometry::kHeadDim *
                                             Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }

    // Aligned: each column merges the splits its own position runs (Bf16KvPartition::aligned_row),
    // as a decode of that column alone does. Otherwise the call's window sets the partition.
    const int window = Aligned ? positions[token] + 1 : positions[live_columns - 1] + 1;
    const int active_split_count =
        Aligned ? Bf16KvPartition::aligned_live(window).splits : partition.live(window).splits;

    __shared__ float weights[Schedule::kThreads], warp_sums[Schedule::kWarps], scalars[2];
    const float head_l = legacy_bf16_kv_merge_statistics<Geometry, Schedule>(
        partial_m, partial_l, q_head, token, tokens, active_split_count, weights, warp_sums,
        scalars);
    const int d = d_start + tid;
    if (tid >= DChunk || d >= Geometry::kHeadDim) return;
    float numerator = 0.0f;
    for (int split = 0; split < active_split_count; ++split) {
        if (weights[split] != 0.0f)
            numerator +=
                partial_acc[causal_partial_index<Geometry>(q_head, d, token, split, tokens)] *
                weights[split];
    }

    const float value = (head_l > 0.0f) ? numerator / head_l : 0.0f;
    const auto index  = causal_q_index<Geometry>(q_head, d, output_column);
    out[index]        = causal_gated_output(value, gate ? gate + index : nullptr);
}

} // namespace legacy

namespace {

using namespace ninferno::ops::detail;
using G = CausalGeometry<256, 16, 4>;

// The chunk rule (causal_merge_shape.h, entry O's law: issue and bytes of the most loaded SM, the
// fixed work repeated by every narrow CTA). A decode token of 16 query heads at the wave
// partition's SMs / 2 splits keeps 256 columns below 64 SMs (the RTX 3060, 4060 Ti, 4070 Ti) and
// narrows from 64 SMs up; wider grids narrow while the narrow CTAs' repeated work stays under the
// wide chunk's bytes; whole rows and a 64-column schedule keep their chunk.
static_assert(causal_merge_d_chunk(256, false, 16, 1, 14, 28) == 256);
static_assert(causal_merge_d_chunk(256, false, 16, 1, 17, 34) == 256);
static_assert(causal_merge_d_chunk(256, false, 16, 1, 30, 60) == 256);
static_assert(causal_merge_d_chunk(256, false, 16, 1, 31, 63) == 256);
static_assert(causal_merge_d_chunk(256, false, 16, 1, 32, 64) == 64);
static_assert(causal_merge_d_chunk(256, false, 16, 1, 34, 68) == 64);
static_assert(causal_merge_d_chunk(256, false, 16, 1, 41, 82) == 64);
static_assert(causal_merge_d_chunk(256, false, 16, 1, 85, 170) == 64);
static_assert(causal_merge_d_chunk(256, false, 16, 5, 85, 170) == 64);
static_assert(causal_merge_d_chunk(256, false, 16, 6, 85, 170) == 256);
static_assert(causal_merge_d_chunk(256, false, 16, 2, 64, 128) == 64);
static_assert(causal_merge_d_chunk(256, false, 16, 3, 64, 128) == 256);
static_assert(causal_merge_d_chunk(256, false, 16, 2, 41, 82) == 256);
// Bytes past the repeated work: 256 splits on 28 SMs narrow.
static_assert(causal_merge_d_chunk(256, false, 16, 1, 256, 28) == 64);
static_assert(causal_merge_d_chunk(256, true, 16, 1, 85, 170) == 256);
static_assert(causal_merge_d_chunk(64, false, 16, 1, 85, 170) == 64);

// Runs `launch` while device_class() answers `sms` multiprocessors: the production launches take
// their chunk from it, so both chunks run through them on any device.
template <class Launch>
void with_sms(int sms, Launch&& launch) {
    ninferno::DeviceClass d = ninferno::device_class();
    d.sms                   = sms;
    const ninferno::ScopedDeviceClass scoped(d);
    launch();
}

// The SM counts the production launches are run under: a board that narrows a decode token and
// one that keeps 256 columns.
constexpr int kLaunchSms[] = {170, 28};

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return true;
    std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(status));
    return false;
}

struct Random {
    std::uint64_t state;

    float next() { // uniform in [0, 1)
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(state >> 40) / static_cast<float>(1ULL << 24);
    }

    float range(float low, float high) { return low + (high - low) * next(); }
};

struct Case {
    const char* name;
    int splits;   // the live splits of the last column's window (not aligned: the target)
    int capacity; // the partials' split stride (at least every column's live count)
    int width;
    int batch;
    bool gate;
    bool aligned;
    std::vector<std::int32_t> positions; // width x batch
    std::vector<std::int32_t> valid;     // per batch row; empty: no mask
};

// Device buffers of one case, filled once and read by every kernel of it.
struct Buffers {
    float *acc = nullptr, *m = nullptr, *l = nullptr;
    std::int32_t *positions = nullptr, *valid = nullptr;
    __nv_bfloat16 *gate = nullptr, *out = nullptr;
    std::size_t out_count = 0;

    Buffers()                          = default;
    Buffers(const Buffers&)            = delete;
    Buffers& operator=(const Buffers&) = delete;

    ~Buffers() {
        cudaFree(acc);
        cudaFree(m);
        cudaFree(l);
        cudaFree(positions);
        cudaFree(valid);
        cudaFree(gate);
        cudaFree(out);
    }
};

// Every copy, fill and launch of the test runs on this non-blocking stream (main makes it), none
// on the legacy stream (tools/audit/legacy_stream_calls.sh).
cudaStream_t g_stream = nullptr;

template <class T>
bool upload(T** device, const std::vector<T>& host) {
    const std::size_t bytes = host.size() * sizeof(T);
    return check(cudaMalloc(reinterpret_cast<void**>(device), bytes), "cudaMalloc") &&
           check(cudaMemcpyAsync(*device, host.data(), bytes, cudaMemcpyHostToDevice, g_stream),
                 "copy") &&
           check(cudaStreamSynchronize(g_stream), "copy");
}

bool fill(const Case& c, std::uint64_t seed, Buffers& b) {
    const std::size_t columns = static_cast<std::size_t>(c.width) * c.batch;
    const std::size_t stats = static_cast<std::size_t>(G::QHeads) * c.width * c.capacity * c.batch;
    std::vector<float> acc(stats * 256), m(stats), l(stats);
    Random random{seed};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // Layout (causal_tile_io.cuh): the statistics [batch][split][token][head], acc [..][d].
    for (int row = 0; row < c.batch; ++row) {
        for (int split = 0; split < c.capacity; ++split) {
            for (int token = 0; token < c.width; ++token) {
                for (int head = 0; head < G::QHeads; ++head) {
                    const std::size_t s =
                        head +
                        static_cast<std::size_t>(G::QHeads) *
                            (token + static_cast<std::size_t>(c.width) *
                                         (split + static_cast<std::size_t>(c.capacity) * row));
                    const float pick = random.next();
                    // Past the last column's live splits: values the merge must not read.
                    const bool past = !c.aligned && split >= c.splits;
                    bool zero       = false;
                    if (past) {
                        m[s] = 1.0e30F;
                        l[s] = 1.0e30F;
                        zero = true;
                    } else if (pick < 0.08F) { // no keys: no sum
                        m[s] = random.range(-8.0F, 8.0F);
                        l[s] = 0.0F;
                        zero = true;
                    } else if (pick < 0.14F) { // a maximum far under the others: exp underflows
                        m[s] = -400.0F;
                        l[s] = random.range(0.5F, 40.0F);
                        zero = true;
                    } else {
                        m[s] = random.range(-8.0F, 8.0F);
                        l[s] = random.range(0.5F, 40.0F);
                    }
                    for (int d = 0; d < 256; ++d)
                        acc[s * 256 + d] =
                            zero && random.next() < 0.5F ? nan : random.range(-30.0F, 30.0F);
                }
            }
        }
    }
    std::vector<__nv_bfloat16> gate(columns * G::QHeads * 256);
    for (auto& g : gate) g = __float2bfloat16(random.range(-4.0F, 4.0F));
    b.out_count = gate.size();
    return upload(&b.acc, acc) && upload(&b.m, m) && upload(&b.l, l) &&
           upload(&b.positions, c.positions) && (c.valid.empty() || upload(&b.valid, c.valid)) &&
           (!c.gate || upload(&b.gate, gate)) &&
           check(cudaMalloc(reinterpret_cast<void**>(&b.out), b.out_count * 2), "cudaMalloc");
}

int g_failures = 0;
int g_compared = 0;

// Runs one launch into b.out (a pattern first) and returns the output bits.
template <class Launch>
bool run(Buffers& b, Launch&& launch, std::vector<std::uint16_t>& bits) {
    if (!check(cudaMemsetAsync(b.out, 0x7f, b.out_count * 2, g_stream), "cudaMemsetAsync"))
        return false;
    launch();
    if (!check(cudaGetLastError(), "launch") || !check(cudaStreamSynchronize(g_stream), "kernel"))
        return false;
    bits.resize(b.out_count);
    return check(cudaMemcpyAsync(bits.data(), b.out, b.out_count * 2, cudaMemcpyDeviceToHost,
                                 g_stream),
                 "copy") &&
           check(cudaStreamSynchronize(g_stream), "copy");
}

void compare(const Case& c, const char* family, const char* variant,
             const std::vector<std::uint16_t>& want, const std::vector<std::uint16_t>& got) {
    ++g_compared;
    if (got.size() != want.size()) {
        if (g_failures++ < 16)
            std::printf("FAIL %s %s %s splits %d: no output\n", family, c.name, variant, c.splits);
        return;
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (want[i] == got[i]) continue;
        if (g_failures++ < 16)
            std::printf("FAIL %s %s %s splits %d: element %zu 0x%04x, legacy 0x%04x\n", family,
                        c.name, variant, c.splits, i, got[i], want[i]);
        return;
    }
}

CausalKvPartition causal_partition(const Case& c) {
    CausalKvPartition partition;
    partition.capacity  = c.capacity;
    partition.target    = c.aligned ? CausalKvPartition::kMaxSplits : c.splits;
    partition.key_shift = c.aligned ? CausalKvPartition::kAlignedMinKeyShift : 6;
    partition.aligned   = c.aligned;
    return partition;
}

Bf16KvPartition bf16_partition(const Case& c) {
    Bf16KvPartition partition;
    partition.capacity      = c.capacity;
    partition.normal_target = c.splits;
    partition.long_target   = c.splits;
    partition.key_rows      = 32;
    partition.aligned       = c.aligned;
    return partition;
}

struct WideSchedule {
    static constexpr int kDChunk = 256, kThreads = 256;
};

// The causal merge (INT8, FP8, NVFP4, K8V4): legacy at 256 and 64 columns, new at both, and the
// production launch (its chunk from this device's SM count).
template <bool MultiBatch, bool Masked, bool Rotated, bool Aligned>
void causal_case(const Case& c, Buffers& b) {
    const auto partition      = causal_partition(c);
    const int columns         = c.width * c.batch;
    const __nv_bfloat16* gate = Rotated ? nullptr : b.gate;
    std::vector<std::uint16_t> want, got;
    const auto launch = [&]<int DChunk, bool Legacy>() {
        const dim3 grid(G::QHeads, 256 / DChunk, columns);
        if constexpr (Legacy)
            legacy::legacy_causal_natural_merge_kernel<G, DChunk, MultiBatch, Masked, Rotated,
                                                       Aligned><<<grid, 256, 0, g_stream>>>(
                b.acc, b.m, b.l, b.positions, b.valid, c.width, c.batch, partition, b.out, gate);
        else
            causal_natural_merge_kernel<G, DChunk, MultiBatch, Masked, Rotated, Aligned>
                <<<grid, 256, 0, g_stream>>>(b.acc, b.m, b.l, b.positions, b.valid, c.width,
                                             c.batch, partition, b.out, gate);
    };
    if (!run(b, [&] { launch.template operator()<256, true>(); }, want)) {
        ++g_failures;
        return;
    }
    if (!run(b, [&] { launch.template operator()<256, false>(); }, got)) got.clear();
    compare(c, "causal", Rotated ? "rotated 256" : "256", want, got);
    if constexpr (!Rotated) {
        if (!run(b, [&] { launch.template operator()<64, false>(); }, got)) got.clear();
        compare(c, "causal", "64", want, got);
        if (!run(b, [&] { launch.template operator()<64, true>(); }, got)) got.clear();
        compare(c, "causal", "legacy 64", want, got);
    }
    CausalAttentionOperands p{};
    p.positions = b.positions;
    p.out       = b.out;
    p.width     = c.width;
    p.batch     = c.batch;
    p.gate      = gate;
    for (const int sms : kLaunchSms) {
        if (!run(
                b,
                [&] {
                    with_sms(sms, [&] {
                        launch_causal_natural_merge<G, WideSchedule, MultiBatch, Masked, Rotated,
                                                    true>(
                            p, b.valid, partition, CausalPartialView{b.acc, b.m, b.l}, g_stream);
                    });
                },
                got))
            got.clear();
        compare(c, "causal", sms == kLaunchSms[0] ? "launch, 170 SMs" : "launch, 28 SMs", want,
                got);
    }
}

// The BF16 merge: legacy at its schedule's chunk (256), new at 256 and 64, and the production
// launch (its chunk from this device's SM count: the narrow Bf16KvMergeSchedule<64> instance at a
// decode token).
template <bool MultiBatch, bool Masked, bool Aligned>
void bf16_case(const Case& c, Buffers& b) {
    const auto partition = bf16_partition(c);
    const int columns    = c.width * c.batch;
    std::vector<std::uint16_t> want, got;
    const auto launch = [&]<int DChunk, bool Legacy>() {
        using S = Bf16KvMergeSchedule<DChunk>;
        const dim3 grid(G::QHeads, 256 / DChunk, columns);
        if constexpr (Legacy)
            legacy::legacy_bf16_kv_merge_kernel<G, S, MultiBatch, Masked, Aligned>
                <<<grid, S::kThreads, 0, g_stream>>>(b.acc, b.m, b.l, b.positions, b.valid, c.width,
                                                     c.batch, partition, b.out, b.gate);
        else
            bf16_kv_merge_kernel<G, S, MultiBatch, Masked, Aligned>
                <<<grid, S::kThreads, 0, g_stream>>>(b.acc, b.m, b.l, b.positions, b.valid, c.width,
                                                     c.batch, partition, b.out, b.gate);
    };
    if (!run(b, [&] { launch.template operator()<256, true>(); }, want)) {
        ++g_failures;
        return;
    }
    if (!run(b, [&] { launch.template operator()<256, false>(); }, got)) got.clear();
    compare(c, "bf16", "256", want, got);
    if (!run(b, [&] { launch.template operator()<64, false>(); }, got)) got.clear();
    compare(c, "bf16", "64", want, got);
    CausalAttentionOperands p{};
    p.positions = b.positions;
    p.out       = b.out;
    p.width     = c.width;
    p.batch     = c.batch;
    p.gate      = b.gate;
    // The merge reads only the view's valid columns.
    const Bf16KvCacheView<false> cache{nullptr, nullptr, nullptr, b.valid,
                                       nullptr, 0,       256,     G::KVHeads};
    for (const int sms : kLaunchSms) {
        if (!run(
                b,
                [&] {
                    with_sms(sms, [&] {
                        launch_bf16_kv_merge<G, Bf16KvMergeSchedule<256>, MultiBatch, Masked, false,
                                             true>(p, cache, partition,
                                                   CausalPartialView{b.acc, b.m, b.l}, g_stream);
                    });
                },
                got))
            got.clear();
        compare(c, "bf16", sms == kLaunchSms[0] ? "launch, 170 SMs" : "launch, 28 SMs", want, got);
    }
}

// Not aligned: the last column sees splits x 64 keys (the call's window runs `splits`), earlier
// columns and later batch rows a few keys fewer.
std::vector<std::int32_t> window_positions(int splits, int width, int batch) {
    std::vector<std::int32_t> positions;
    for (int row = 0; row < batch; ++row)
        for (int token = 0; token < width; ++token)
            positions.push_back(std::max(0, splits * 64 - width + token - 5 * row));
    return positions;
}

// Aligned: column t sees about splits x 512 - 700 t keys, so the columns run different counts.
std::vector<std::int32_t> aligned_positions(int splits, int width, int batch) {
    std::vector<std::int32_t> positions;
    for (int row = 0; row < batch; ++row)
        for (int token = 0; token < width; ++token)
            positions.push_back(std::max(0, splits * 512 - 1 - 700 * token - 3 * row));
    return positions;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    if (!check(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking), "stream")) return 1;
    const int split_counts[] = {1,  2,  3,  4,  5,  7,  8,   9,   15,  16,  17,  31, 32,
                                33, 41, 63, 64, 65, 85, 127, 128, 129, 200, 255, 256};
    std::uint64_t seed       = 865;
    // The production launches run under 170 and 28 SMs (kLaunchSms): a decode token's chunk there.
    static_assert(causal_merge_d_chunk(256, false, G::QHeads, 1, 85, kLaunchSms[0]) == 64);
    static_assert(causal_merge_d_chunk(256, false, G::QHeads, 1, 14, kLaunchSms[1]) == 256);
    std::printf("device: %d SMs; the production launches run as on 170 and 28 SMs\n",
                ninferno::device_class().sms);
    for (const int splits : split_counts) {
        const int capacity = std::min(256, splits + 3);
        {
            const Case c{
                "decode", splits, capacity, 1, 1, true, false, window_positions(splits, 1, 1), {}};
            Buffers b;
            if (!fill(c, ++seed, b)) return 1;
            causal_case<false, false, false, false>(c, b);
            causal_case<false, false, true, false>(c, b);
            bf16_case<false, false, false>(c, b);
        }
        {
            const Case c{"batch2-width3-masked",         splits, capacity, 3, 2, true, false,
                         window_positions(splits, 3, 2), {3, 2}};
            Buffers b;
            if (!fill(c, ++seed, b)) return 1;
            causal_case<true, true, false, false>(c, b);
            causal_case<true, true, true, false>(c, b);
            bf16_case<true, true, false>(c, b);
        }
        {
            const Case c{"width4-ungated",
                         splits,
                         capacity,
                         4,
                         1,
                         false,
                         false,
                         window_positions(splits, 4, 1),
                         {}};
            Buffers b;
            if (!fill(c, ++seed, b)) return 1;
            causal_case<false, false, false, false>(c, b);
            bf16_case<false, false, false>(c, b);
        }
    }
    // Aligned partitions: at most kAlignedSplits (64) a column.
    const int aligned_counts[] = {1, 2, 3, 7, 16, 17, 33, 43, 63, 64};
    for (const int splits : aligned_counts) {
        {
            const Case c{"aligned-width4",
                         splits,
                         CausalKvPartition::kAlignedSplits,
                         4,
                         1,
                         true,
                         true,
                         aligned_positions(splits, 4, 1),
                         {}};
            Buffers b;
            if (!fill(c, ++seed, b)) return 1;
            causal_case<false, false, false, true>(c, b);
            causal_case<false, false, true, true>(c, b);
            bf16_case<false, false, true>(c, b);
        }
        {
            const Case c{"aligned-batch2-masked",
                         splits,
                         CausalKvPartition::kAlignedSplits,
                         2,
                         2,
                         true,
                         true,
                         aligned_positions(splits, 2, 2),
                         {2, 1}};
            Buffers b;
            if (!fill(c, ++seed, b)) return 1;
            causal_case<true, true, false, true>(c, b);
            bf16_case<true, true, true>(c, b);
        }
    }
    if (g_failures != 0) {
        std::printf("FAIL: %d of %d comparisons\n", g_failures, g_compared);
        return 1;
    }
    std::printf("PASS: %d comparisons, every output bit equal to the legacy merges\n", g_compared);
    return 0;
}
