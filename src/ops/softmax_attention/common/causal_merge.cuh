#pragma once
#include "core/device.h"
#include "core/device_class.h"
#include <stdexcept>
#include "ops/softmax_attention/common/causal_operands.h"

#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_merge_columns.cuh"
#include "ops/softmax_attention/common/causal_merge_shape.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_softmax.cuh"

namespace ninferno::ops::detail {

// The splits' maximum, the weights (published to `weights`, zero for a split that adds nothing)
// and their sum, from this thread's split's stored maximum and sum (loaded beside the positions,
// before the live split count is known; thread `tid` holds split `tid`).
__device__ __forceinline__ float causal_merge_natural_statistics(float stored_m, float stored_l,
                                                                 int splits, float* weights,
                                                                 float* warp_sums, float* scalars) {
    static_assert(CausalKvPartition::kMaxSplits <= 256);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float m      = tid < splits ? stored_m : -CUDART_INF_F;
    const float warp_m = warp_max(m);
    if (lane == 0) warp_sums[warp] = warp_m;
    __syncthreads();
    if (warp == 0) {
        const float maximum = warp_max(tid < 8 ? warp_sums[tid] : -CUDART_INF_F);
        if (tid == 0) scalars[0] = maximum;
    }
    __syncthreads();
    const float maximum     = scalars[0];
    const float l           = tid < splits ? stored_l : 0.0f;
    const float weight      = l > 0.0f && maximum > -CUDART_INF_F ? expf(m - maximum) : 0.0f;
    const float denominator = block_reduce_sum<256>(l * weight, warp_sums);
    if (tid == 0) scalars[1] = denominator;
    __syncthreads();
    const float total = scalars[1];
    if (tid < splits) weights[tid] = total > 0.0f ? weight : 0.0f;
    __syncthreads();
    return total;
}

// One CTA merges DChunk output columns of one query head and token over the live splits. The
// loads that do not depend on each other go out together: the positions, the splits' maxima and
// sums, the gate and the first two stages of partials (CausalMergeColumns), so a call waits on
// about two L2 round trips before its split loop, not five.
template <class Geometry, int DChunk, bool MultiBatch, bool Masked, bool InverseRotation,
          bool Aligned = false>
__launch_bounds__(256) __global__
    void causal_natural_merge_kernel(const float* partial_acc, const float* partial_m,
                                     const float* partial_l, const std::int32_t* positions,
                                     const std::int32_t* valid_columns, std::int32_t tokens,
                                     std::int32_t batch_size, CausalKvPartition partition,
                                     __nv_bfloat16* out, const __nv_bfloat16* gate) {
    static_assert(Geometry::kHeadDim == kCausalHeadDim);
    static_assert(DChunk > 0 && DChunk <= kCausalHeadDim);
    static_assert(!InverseRotation || DChunk == kCausalHeadDim);
    using Columns = CausalMergeColumns<Geometry, DChunk, 256>;
    static_assert(Columns::kStageFloats == kCausalMergeStageFloats); // the chunk law's stage
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
    if constexpr (MultiBatch) {
        positions += static_cast<std::int64_t>(batch) * tokens;
        partial_acc +=
            static_cast<std::int64_t>(batch) * 256 * Geometry::QHeads * tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }
    // Every split the partials hold (the capacity), before the live count is known.
    const auto stat_index = causal_stat_index<Geometry>(q_head, token, tid, tokens);
    const float stored_m  = tid < split_count ? partial_m[stat_index] : -CUDART_INF_F;
    const float stored_l  = tid < split_count ? partial_l[stat_index] : 0.0f;
    // Aligned: each column merges the splits its own position runs
    // (CausalKvPartition::aligned_row), as a decode of that column alone does. Otherwise the
    // call's window sets the partition.
    const int window  = Aligned ? positions[token] + 1 : positions[tokens - 1] + 1;
    int output_column = token;
    if constexpr (MultiBatch) output_column += batch * tokens;
    const int d      = d_start + tid;
    const bool owner = tid < DChunk && d < 256;
    __nv_bfloat16 gate_value{};
    if constexpr (!InverseRotation) {
        if (gate && owner) gate_value = gate[causal_q_index<Geometry>(q_head, d, output_column)];
    }
    if constexpr (Masked) {
        const int absolute_column = token;
        if (absolute_column >= valid_columns[batch]) {
            if (owner) {
                const auto index = causal_q_index<Geometry>(q_head, d, output_column);
                out[index]       = causal_gated_output(0.0f, gate ? gate + index : nullptr);
            }
            return;
        }
    }

    const int active_splits =
        Aligned ? CausalKvPartition::aligned_active(window) : partition.active(window);
    Columns columns(partial_acc, q_head, d_start, token, tokens, active_splits);
    __shared__ float weights[256], warp_sums[8], scalars[2];
    __shared__ __align__(16) float stage[Columns::kStageFloats];
    const float head_l = causal_merge_natural_statistics(stored_m, stored_l, active_splits, weights,
                                                         warp_sums, scalars);
    const float numerator = columns.numerator(weights, stage);
    if (!owner) return;

    const float value = head_l > 0.0F ? numerator / head_l : 0.0F;
    if constexpr (InverseRotation) {
        __shared__ float normalized[kCausalHeadDim];
        normalized[tid] = value;
        __syncthreads();
        if (tid < 32)
            causal_store_inverse_rotated_row<Geometry>(normalized, out, q_head, output_column);
    } else {
        const auto index = causal_q_index<Geometry>(q_head, d, output_column);
        out[index]       = causal_gated_output(value, gate != nullptr, gate_value);
    }
}

// AlignedCapable: the call site may pass an aligned partition (the parallel tiles of the invariant
// route profile); only those sites build the aligned merge.
template <class G, class S, bool MultiBatch, bool Masked, bool InverseRotation,
          bool AlignedCapable = false>
void launch_causal_natural_merge(const CausalAttentionOperands& p,
                                 const std::int32_t* valid_columns, CausalKvPartition partition,
                                 CausalPartialView partial, cudaStream_t stream) {
    static_assert(S::kThreads == 256);
    // The rotated store converts whole rows in a warp; it takes no gate.
    if (InverseRotation && p.gate)
        throw std::invalid_argument("causal merge: a gate needs the unrotated store");
    if (!aligned_to(partial.acc, 16))
        throw std::invalid_argument("causal merge: the partials need 16-byte alignment");
    const int columns = p.width * p.batch;
    // The column chunk by the law of causal_merge_shape.h: the grid, the splits and the SMs.
    const int d_chunk = causal_merge_d_chunk(S::kDChunk, InverseRotation, G::QHeads, columns,
                                             partition.capacity, device_class().sms);
    const auto launch = [&]<bool Aligned, int DChunk>() {
        const dim3 grid(G::QHeads, div_up(G::kHeadDim, DChunk), columns);
        causal_natural_merge_kernel<G, DChunk, MultiBatch, Masked, InverseRotation, Aligned>
            <<<grid, S::kThreads, 0, stream>>>(partial.acc, partial.maximum, partial.sum,
                                               p.positions, valid_columns, p.width, p.batch,
                                               partition, p.out, p.gate);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto shaped = [&]<bool Aligned>() {
        if constexpr (!InverseRotation && S::kDChunk > kCausalMergeNarrowDChunk) {
            if (d_chunk == kCausalMergeNarrowDChunk)
                return launch.template operator()<Aligned, kCausalMergeNarrowDChunk>();
        }
        launch.template operator()<Aligned, S::kDChunk>();
    };
    if constexpr (AlignedCapable) {
        if (partition.aligned) return shaped.template operator()<true>();
    } else {
        if (partition.aligned)
            throw std::invalid_argument(
                "causal merge: aligned splits at a site that never takes them");
    }
    shaped.template operator()<false>();
}

} // namespace ninferno::ops::detail
