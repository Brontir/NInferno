#pragma once

#include "core/per_device.h"
#include "core/device.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/grouped_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/slab_mma.cuh"
#include "ops/softmax_attention/common/causal_merge.cuh"
#include <stdexcept>

namespace ninferno::ops::detail {

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input,
          bool ParallelQueries = false>
void launch_nvfp4_kv_grouped_mma(const CausalAttentionOperands& p, Nvfp4KvCacheView<Writable> cache,
                                 Input input, CausalKvPartition partition,
                                 CausalPartialView partial, cudaStream_t stream) {
    static_assert(Writable == Input::writes_cache);
    validate_quantized_causal_operands<G>(p, cache);
    if ((!ParallelQueries && p.width != S::kTokenTile) || MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr) || partition.capacity < 1 ||
        partition.target > CausalKvPartition::kMaxSplits || partition.target < 1 ||
        partition.key_shift < 6 || partition.key_shift > 14 ||
        partition.capacity != partition.capacity_for(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("NVFP4 grouped attention: invalid schedule/partials");
    if constexpr (Input::writes_cache)
        if (!input.k || !input.v) throw std::invalid_argument("NVFP4 append requires K/V");
    const auto run = [&](auto kernel, int threads, int bytes) {
        const dim3 grid(G::KVHeads * (ParallelQueries ? div_up(p.width, S::kTokenTile) : 1),
                        partition.capacity, p.batch);
        kernel<<<grid, threads, bytes, stream>>>(
            p.q, input, p.positions, cache.keys, cache.values, cache.key_scales, cache.value_scales,
            cache.tables, cache.valid_columns, cache.table_rows, cache.table_stride, p.width,
            p.visible_capacity, partition, p.scale, partial.acc, partial.maximum, partial.sum);
        CUDA_CHECK(cudaGetLastError());
    };
    // The slab kernel where the instance's schedule names it (instances.h), except for the
    // aligned launches of the invariant route profile, which keep the grouped kernel's arithmetic.
    const auto launch = [&]<bool Aligned>() {
        if constexpr (S::kSlab && !Aligned) {
            static constexpr auto kernel =
                nvfp4_kv_slab_mma_kernel<G, S, MultiBatch, Masked, Input, ParallelQueries>;
            static ::ninferno::PerDevice<bool> status;
            ::ninferno::allow_dynamic_smem(status, kernel, S::kArenaBytes);
            run(kernel, S::kThreads, S::kArenaBytes);
        } else {
            using Grouped = typename S::Grouped;
            static constexpr auto kernel =
                nvfp4_kv_grouped_mma_kernel<G, Grouped, MultiBatch, Masked, Input, ParallelQueries,
                                            Aligned>;
            constexpr int bytes = Grouped::kDynamicArena ? Grouped::kArenaBytes : 0;
            if constexpr (Grouped::kDynamicArena) {
                static ::ninferno::PerDevice<bool> status;
                ::ninferno::allow_dynamic_smem(status, kernel, bytes);
            }
            run(kernel, Grouped::kThreads, bytes);
        }
    };
    // The invariant route profile launches the parallel tiles over the cache only
    // (CausalKvPartition::aligned_to_positions); only those instances are built aligned.
    if constexpr (ParallelQueries && !Input::writes_cache) {
        if (partition.aligned) return launch.template operator()<true>();
    } else {
        if (partition.aligned)
            throw std::invalid_argument(
                "NVFP4 grouped attention: aligned splits need the parallel tiles");
    }
    launch.template operator()<false>();
}



} // namespace ninferno::ops::detail
