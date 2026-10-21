#pragma once

#include "ops/common/math.h"
#include "ops/softmax_attention/common/causal_partition.h"

#include <cuda_runtime.h>

namespace ninferno::ops::detail {

struct Bf16KvLivePartition {
    int splits;
    int keys_per_split;
};

// Capture fixes capacity; producer and merge derive the same partition from
// the live row length, including ragged batches and Graph replay.
struct Bf16KvPartition {
    int capacity      = 1;
    int normal_target = 1;
    int long_target   = 1;
    int key_rows      = 32;
    // Aligned (the invariant route profile): a token seeing v keys splits them every
    // 2^CausalKvPartition::aligned_key_shift(v) keys from position 0, as
    // CausalKvPartition::aligned_to_positions does. Otherwise the window is spread over the targets.
    bool aligned = false;

    __host__ __device__ Bf16KvLivePartition live(int visible) const {
        if (aligned) {
            const int keys = 1 << CausalKvPartition::aligned_key_shift(visible);
            return {div_up(visible, keys), keys};
        }
        const int target  = visible >= 32768 ? long_target : normal_target;
        const int rounded = round_up(div_up(visible, target), key_rows);
        const int keys    = rounded < 64 ? 64 : rounded;
        return {div_up(visible, keys), keys};
    }

    // Aligned: the splits of the query row at `position` (negative: a row that stores none).
    __host__ __device__ CausalAlignedRow aligned_row(int position) const {
        if (position < 0) return {CausalKvPartition::kAlignedMinKeyShift, 0};
        return {CausalKvPartition::aligned_key_shift(position + 1), live(position + 1).splits};
    }

    // Between key-tile rounding points, the split count grows with length. A
    // complete target*chunk boundary reaches target; otherwise high is maximal.
    int interval_capacity(int low, int high, int target) const {
        if (low > high) return 0;
        const int boundary = high / (target * key_rows) * (target * key_rows);
        if (boundary >= low && boundary >= target * 64) return target;
        return live(high).splits;
    }
};

} // namespace ninferno::ops::detail
