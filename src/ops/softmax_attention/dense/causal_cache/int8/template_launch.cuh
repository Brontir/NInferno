#pragma once

#include "core/per_device.h"
#include "core/device.h"
#include "ops/softmax_attention/dense/causal_cache/int8/grouped_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/prompt_fast.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/tiled_mma.cuh"
#include "ops/softmax_attention/common/causal_merge.cuh"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ninferno::ops::detail {

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input,
          bool ParallelQueries = false>
void launch_int8_kv_grouped_mma(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                                Input input, CausalKvPartition partition, CausalPartialView partial,
                                cudaStream_t stream) {
    static_assert(Writable == Input::writes_cache);
    validate_quantized_causal_operands<G>(p, cache);
    if ((!ParallelQueries && p.width != S::kTokenTile) || MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr) || partition.capacity < 1 ||
        partition.target > CausalKvPartition::kMaxSplits || partition.target < 1 ||
        partition.key_shift < 6 || partition.key_shift > 14 ||
        partition.capacity != partition.capacity_for(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("INT8 grouped attention: invalid schedule/partials");
    if constexpr (Input::writes_cache)
        if (!input.k || !input.v) throw std::invalid_argument("INT8 append requires K/V");
    const auto launch = [&]<bool Aligned>() {
        static constexpr auto kernel =
            int8_kv_grouped_mma_kernel<G, S, MultiBatch, Masked, Input, ParallelQueries, Aligned>;
        constexpr int bytes = S::kDynamicArena ? S::kArenaBytes : 0;
        if constexpr (S::kDynamicArena) {
            static ::ninferno::PerDevice<bool> status;
            ::ninferno::allow_dynamic_smem(status, kernel, bytes);
        }
        const dim3 grid(G::KVHeads * (ParallelQueries ? div_up(p.width, S::kTokenTile) : 1),
                        partition.capacity, p.batch);
        kernel<<<grid, S::kThreads, bytes, stream>>>(
            p.q, input, p.positions, cache.keys, cache.values, cache.key_scales,
            cache.value_scales, cache.tables, cache.valid_columns, cache.table_rows,
            cache.table_stride, p.width, p.visible_capacity, partition, p.scale, partial.acc,
            partial.maximum, partial.sum);
        CUDA_CHECK(cudaGetLastError());
    };
    // The invariant route profile launches the parallel tiles over the cache only
    // (CausalKvPartition::aligned_to_positions); only those instances are built aligned.
    if constexpr (ParallelQueries && !Input::writes_cache) {
        if (partition.aligned) return launch.template operator()<true>();
    } else {
        if (partition.aligned)
            throw std::invalid_argument(
                "INT8 grouped attention: aligned splits need the parallel tiles");
    }
    launch.template operator()<false>();
}

template <class G, class S>
void launch_int8_kv_tiled_mma(const CausalAttentionOperands& p, Int8KvReadView cache,
                              cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("INT8 tiled attention requires a complete single query row");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        static constexpr auto kernel = int8_kv_tiled_mma_kernel<G, S, Metadata>;
        static ::ninferno::PerDevice<bool> status;
        ::ninferno::allow_dynamic_smem(status, kernel, S::kSharedBytes);
        const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads);
        kernel<<<grid, S::kThreads, S::kSharedBytes, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.out, p.width);
        CUDA_CHECK(cudaGetLastError());
    };
    if (!cache.table_rows)
        invoke(PagedKVDirectMetadata{cache.tables});
    else if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride});
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride});
}

// Both prompt-fast shapes run one CTA per SM and every CTA of a launch sweeps a similar key range.
// A four-warp CTA, or an eight-warp CTA owning only 64 rows, sweeps in about 0.72 of a full
// eight-warp CTA's time (ninfer-all, RTX 5090 at 64K context) but covers half the rows.
struct Int8PromptFastLayout {
    bool narrow      = false; // four-warp CTAs throughout
    int narrow_slots = 0;     // eight-warp launch: leading 64-row slots run as half-height CTAs
};

// Makespan of an eight-warp launch, in percent of one full CTA's sweep: wide CTAs issue first,
// then the half-height ones, each to the SM that frees first.
inline long long int8_prompt_fast_makespan(int wide, int narrow, int multiprocessors) {
    constexpr int kHalfCostPercent = 72;
    const int full_waves = wide / multiprocessors;
    const int partial    = wide % multiprocessors;
    std::vector<long long> free_at(multiprocessors, 100LL * full_waves);
    for (int sm = 0; sm < partial; ++sm) free_at[sm] += 100;
    long long end = *std::max_element(free_at.begin(), free_at.end());
    for (int i = 0; i < narrow; ++i) {
        const auto sm = std::min_element(free_at.begin(), free_at.end());
        *sm += kHalfCostPercent;
        end = std::max(end, *sm);
    }
    return end;
}

inline Int8PromptFastLayout int8_prompt_fast_layout(int width, int query_heads,
                                                    int multiprocessors) {
    constexpr int kRows = Int8PromptFastShape<8>::kQueryRows;
    constexpr int kHalf = kRows / 2;
    Int8PromptFastLayout best;
    long long best_cost = int8_prompt_fast_makespan(div_up(width, kRows) * query_heads, 0,
                                                    multiprocessors);
    // Half-height slots only help the last wave: past two SM-fulls of them nothing changes.
    const int max_slots =
        std::min(div_up(width, kHalf), 2 * div_up(2 * multiprocessors, query_heads));
    for (int slots = 1; slots <= max_slots; ++slots) {
        const int wide = div_up(std::max(width - slots * kHalf, 0), kRows);
        const long long cost =
            int8_prompt_fast_makespan(wide * query_heads, slots * query_heads, multiprocessors);
        if (cost < best_cost) {
            best_cost         = cost;
            best.narrow_slots = slots;
        }
    }
    const long long narrow_cost =
        72LL * div_up(div_up(width, kHalf) * query_heads, multiprocessors);
    if (narrow_cost <= best_cost) return {true, 0};
    return best;
}

template <class G>
void launch_int8_kv_prompt_fast(const CausalAttentionOperands& p, Int8KvReadView cache,
                                int multiprocessors, cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("INT8 prompt attention requires a complete single query row");
    const auto layout = int8_prompt_fast_layout(p.width, G::QHeads, multiprocessors);
    const auto invoke = [&]<int Warps, class Metadata>(Metadata metadata) {
        using Shape              = Int8PromptFastShape<Warps>;
        static constexpr auto kernel = int8_kv_prompt_fast_kernel<G, Metadata, Warps>;
        static ::ninferno::PerDevice<bool> status;
        ::ninferno::allow_dynamic_smem(status, kernel, Shape::kSharedBytes);
        // Four-warp CTAs never split: their 64 rows are already the half height.
        const int narrow_slots = Warps == 8 ? layout.narrow_slots : 0;
        const int slots        = narrow_slots +
                          div_up(std::max(p.width - narrow_slots * Shape::kQueryRows / 2, 0),
                                 Shape::kQueryRows);
        const dim3 grid(slots, G::QHeads);
        kernel<<<grid, Shape::kThreads, Shape::kSharedBytes, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.out, p.width, narrow_slots);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto shaped = [&]<class Metadata>(Metadata metadata) {
        if (layout.narrow)
            invoke.template operator()<4>(metadata);
        else
            invoke.template operator()<8>(metadata);
    };
    if (!cache.table_rows)
        shaped(PagedKVDirectMetadata{cache.tables});
    else if (cache.valid_columns)
        shaped(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride});
    else
        shaped(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride});
}

} // namespace ninferno::ops::detail
