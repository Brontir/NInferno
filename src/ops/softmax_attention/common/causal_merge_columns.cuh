#pragma once

#include "ops/softmax_attention/common/causal_tile_io.cuh"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

// The split loop of the natural merges (causal_merge.cuh, dense/causal_cache/bf16/merge.cuh). The
// thread that owns output column d sums partial_acc[split][d] x weights[split] over the splits in
// split order and skips a split whose weight is zero, exactly as the per-thread loop this replaced
// did (one FFMA a split into one accumulator), so a column's sum keeps its bits. What changes is
// who reads the partials: every thread of the CTA loads 16-byte vectors of the next two stages of
// splits into registers while the owners sum the current stage out of shared memory. The old loop
// kept four 4-byte loads a thread in flight (sm_120a SASS: the x8 body issues two rounds of four),
// one L2 round trip a fourth of the splits; here a stage is 16 KiB a CTA and two are in flight,
// which is more than one SM's fill rate times its L2 latency on every card in gpu_facts, so the
// loop runs at the SM's fill rate (kernel-proposals.md, entry O).
//
// The loads are not gated by the weights: the first two stages go out before the statistics run,
// so their latency hides behind them. They read every split below `splits`, all inside the
// partials' allocation (the partition's capacity); a split with no keys for this row (its weight
// is zero) may hold another call's values, which the owners skip as before. The merges' statistics
// are read the same way, every split below the capacity before the live count is known. Both
// reads past the live splits are discarded, so compute-sanitizer --tool initcheck flags them where
// the attention kernel left those splits unwritten.
template <class Geometry, int DChunk, int Threads>
class CausalMergeColumns {
public:
    static_assert(DChunk % 4 == 0 && Geometry::kHeadDim % DChunk == 0 && DChunk <= Threads);
    static constexpr int kStageFloats = 4096; // 16 KiB of shared memory
    static constexpr int kStageSplits = kStageFloats / DChunk;
    static constexpr int kRowVectors  = DChunk / 4;
    static constexpr int kLoads       = kStageFloats / 4 / Threads;
    static_assert(kLoads >= 1 && kLoads * 4 * Threads == kStageFloats);

    // Issues the loads of the first two stages. `acc` is the token's batch row of the partials,
    // 16-byte aligned; `splits` is the CTA's live split count (uniform over the CTA).
    __device__ __forceinline__ CausalMergeColumns(const float* acc, int q_head, int d_start,
                                                  int token, int tokens, int splits)
        : acc_(acc), q_head_(q_head), d_start_(d_start), token_(token), tokens_(tokens),
          splits_(splits) {
        fetch(first_, 0);
        fetch(second_, kStageSplits);
    }

    // Column d_start + threadIdx.x's weighted sum (meaningful for threadIdx.x < DChunk). Every
    // thread of the CTA calls it, after the weights are published (a barrier); `stage` holds
    // kStageFloats floats of shared memory.
    __device__ __forceinline__ float numerator(const float* weights, float* stage) {
        float sum = 0.0f;
        for (int base = 0; base < splits_; base += 2 * kStageSplits) {
            sum = consume(first_, base, weights, stage, sum);
            if (base + kStageSplits >= splits_) break;
            sum = consume(second_, base + kStageSplits, weights, stage, sum);
        }
        return sum;
    }

private:
    __device__ __forceinline__ void fetch(float4 (&vectors)[kLoads], int base) const {
        const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
        for (int j = 0; j < kLoads; ++j) {
            const int vector = tid + j * Threads;
            const int split  = base + vector / kRowVectors;
            const int d      = d_start_ + (vector % kRowVectors) * 4;
            vectors[j] =
                split < splits_
                    ? *reinterpret_cast<const float4*>(
                          acc_ + causal_partial_index<Geometry>(q_head_, d, token_, split, tokens_))
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Sums the stage at `base` (held in `vectors`) into `sum`, and refills `vectors` with the
    // stage two ahead while the owners sum.
    __device__ __forceinline__ float consume(float4 (&vectors)[kLoads], int base,
                                             const float* weights, float* stage, float sum) {
        const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
        for (int j = 0; j < kLoads; ++j)
            reinterpret_cast<float4*>(stage)[tid + j * Threads] = vectors[j];
        __syncthreads();
        fetch(vectors, base + 2 * kStageSplits);
        if (tid < DChunk) {
            const int count = splits_ - base < kStageSplits ? splits_ - base : kStageSplits;
            for (int s = 0; s < count; ++s) {
                const float weight = weights[base + s];
                if (weight != 0.0f) sum += stage[s * DChunk + tid] * weight;
            }
        }
        __syncthreads();
        return sum;
    }

    const float* acc_;
    int q_head_, d_start_, token_, tokens_, splits_;
    float4 first_[kLoads], second_[kLoads];
};

} // namespace ninferno::ops::detail
