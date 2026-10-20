#pragma once

#include "ops/common/score_id_order.cuh"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

inline constexpr int kSparseMoeExperts = 256;
inline constexpr int kSparseMoeTopK    = 8;

struct SparseMoeRankedValue {
    float value;
    int id;
};

// The ranking compares with bitwise operators: `||` and `&&` may lower to a branch, and the router
// runs one warp a token, where nothing covers a data-dependent branch while it resolves. NaN ranks
// as -inf (rank_score): a NaN router logit can neither leave an expert unselected nor let the
// Gemma 4 network's padding (ids past the experts) win.
__device__ __forceinline__ bool sparse_moe_ranked_better(const SparseMoeRankedValue& a,
                                                         const SparseMoeRankedValue& b) {
    const float av = rank_score(a.value);
    const float bv = rank_score(b.value);
    return (av > bv) | ((av == bv) & (a.id < b.id));
}

// Leaves the better of the pair in `a`, through selects rather than a swap under an `if`.
__device__ __forceinline__ void sparse_moe_order_pair(SparseMoeRankedValue& a,
                                                      SparseMoeRankedValue& b) {
    const SparseMoeRankedValue x = a;
    const SparseMoeRankedValue y = b;
    const bool keep              = sparse_moe_ranked_better(x, y);
    a                            = keep ? x : y;
    b                            = keep ? y : x;
}

// Sorts a lane's eight candidates descending: Batcher's odd-even merge network, 19 exchanges in six
// levels, every index fixed at compile time, so no lane takes a data-dependent path. The ranking is
// a total order over distinct ids, so the order equals any other correct sort's.
__device__ __forceinline__ void
sparse_moe_sort_eight_descending(SparseMoeRankedValue (&run)[kSparseMoeTopK]) {
    static_assert(kSparseMoeTopK == 8);
    constexpr int kPairs[19][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6},
                                   {5, 7}, {1, 2}, {5, 6}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
                                   {2, 4}, {3, 5}, {1, 2}, {3, 4}, {5, 6}};
#pragma unroll
    for (int pair = 0; pair < 19; ++pair) {
        sparse_moe_order_pair(run[kPairs[pair][0]], run[kPairs[pair][1]]);
    }
}

// Merges the descending run of each lane with the run of its xor partner and keeps the better
// half. The eight exchanges of a step do not depend on each other, so the warp reaches the
// warp-wide top-8 in five merge steps instead of eight dependent reduction rounds. Ranking is a
// total order over distinct expert ids, so the selected set and its order are the same as the
// ones any other correct selection produces.
__device__ __forceinline__ void
sparse_moe_merge_ranked_runs(SparseMoeRankedValue (&run)[kSparseMoeTopK]) {
#pragma unroll
    for (int partner = 1; partner < 32; partner <<= 1) {
        SparseMoeRankedValue merged[kSparseMoeTopK];
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            const SparseMoeRankedValue mirror = run[kSparseMoeTopK - 1 - rank];
            SparseMoeRankedValue other;
            other.value  = __shfl_xor_sync(kFullWarpMask, mirror.value, partner);
            other.id     = __shfl_xor_sync(kFullWarpMask, mirror.id, partner);
            merged[rank] = sparse_moe_ranked_better(run[rank], other) ? run[rank] : other;
        }
        // The kept half is bitonic; three compare-exchange stages restore descending order.
#pragma unroll
        for (int stride = kSparseMoeTopK / 2; stride > 0; stride >>= 1) {
#pragma unroll
            for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
                if ((rank & stride) != 0) { continue; }
                sparse_moe_order_pair(merged[rank], merged[rank | stride]);
            }
        }
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) { run[rank] = merged[rank]; }
    }
}

__device__ __forceinline__ void sparse_moe_select_top8_warp(const float* scores, int* ids,
                                                            float* alpha, float* shared_scale,
                                                            float* selected_logits) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue local[kSparseMoeTopK];
#pragma unroll
    for (int item = 0; item < kSparseMoeTopK; ++item) {
        const int id = lane + item * 32;
        local[item]  = {scores[id], id};
    }
    sparse_moe_sort_eight_descending(local);
    sparse_moe_merge_ranked_runs(local);

    if (lane == 0) {
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            ids[rank]             = local[rank].id;
            selected_logits[rank] = local[rank].value;
        }
    }
    __syncwarp();

    float exponential = 0.0f;
    if (lane < kSparseMoeTopK) { exponential = expf(selected_logits[lane] - selected_logits[0]); }
    float denominator = warp_reduce_sum(exponential);
    denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
    if (lane < kSparseMoeTopK) { alpha[lane] = exponential / denominator; }
    if (lane == 0) { *shared_scale = sigmoid(scores[kSparseMoeExperts]); }
}

} // namespace ninferno::ops::detail
