#pragma once

#include "ninferno/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninferno::ops::detail {

enum class Nvfp4KvFamily { Grouped, ParallelGrouped, Tiled };

struct Nvfp4KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Nvfp4KvFamily family;
    int query_heads, kv_heads, width, batch, query_tile;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
    int launch_columns; // the columns of one kernel launch (causal_launch_columns)
};

// The invariant route profile aligns the splits to each token's own position
// (CausalKvPartition::aligned_to_positions), never to the call's KV row.
Nvfp4KvCausalPlan make_nvfp4_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count);
std::size_t nvfp4_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                     int max_width, CausalAttentionExecutionEnvelope envelope, int multiprocessor_count);

} // namespace ninferno::ops::detail
