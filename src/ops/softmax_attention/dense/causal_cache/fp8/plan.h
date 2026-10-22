#pragma once

#include "ninferno/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninferno::ops::detail {

enum class Fp8KvFamily { Grouped, ParallelGrouped, Tiled };

struct Fp8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Fp8KvFamily family;
    int query_heads, kv_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
    int launch_columns; // the columns of one kernel launch (causal_launch_columns)
};

// The invariant route profile aligns the splits to each token's own position
// (CausalKvPartition::aligned_to_positions), never to the call's KV row.
Fp8KvCausalPlan make_fp8_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                        CausalAttentionExecutionEnvelope envelope,
                                        int multiprocessor_count);
std::size_t fp8_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width, int max_width,
                                   CausalAttentionExecutionEnvelope envelope, int multiprocessor_count);

} // namespace ninferno::ops::detail
