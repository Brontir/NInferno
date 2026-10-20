#pragma once

#include "ninferno/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninferno::ops::detail {

enum class K8V4KvFamily { Grouped, ParallelGrouped, Tiled };

struct K8V4KvCausalPlan {
    static constexpr int kTokenTile = 8;
    K8V4KvFamily family;
    int query_heads, kv_heads, width, batch, query_tile;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
    int launch_columns; // the columns of one kernel launch (causal_launch_columns)
};

// row_keys: the keys of the call's KV row (paged_kv_row_keys); the invariant route profile aligns
// its splits to it. 0 (a workspace query) plans the splits of any row.
K8V4KvCausalPlan make_k8v4_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope, int row_keys, int multiprocessor_count);
std::size_t k8v4_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                    int max_width, CausalAttentionExecutionEnvelope envelope, int multiprocessor_count);

} // namespace ninferno::ops::detail
