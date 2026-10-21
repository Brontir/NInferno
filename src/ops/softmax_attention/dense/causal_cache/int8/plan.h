#pragma once

#include "ninferno/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninferno::ops::detail {

enum class Int8KvFamily { Grouped, ParallelGrouped, Tiled };

struct Int8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Int8KvFamily family;
    int query_heads, kv_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
    int launch_columns; // the columns of one kernel launch (causal_launch_columns)
};

// The invariant route profile aligns the splits to each token's own position
// (CausalKvPartition::aligned_to_positions), never to the call's KV row.
Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope, int multiprocessor_count);
std::size_t int8_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                    int max_width, CausalAttentionExecutionEnvelope envelope, int multiprocessor_count);

} // namespace ninferno::ops::detail
