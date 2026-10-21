#pragma once

#include "ninferno/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/instances.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/split_policy.h"

namespace ninferno::ops::detail {

struct Bf16KvCausalPlan {
    Bf16KvInstance instance;
    Bf16KvPartition partition;
    int query_heads;
    int kv_heads;
    int width;
    int batch;
    CausalAttentionExecutionEnvelope envelope;
    int launch_columns; // the columns of one kernel launch (causal_launch_columns)

    bool grouped() const { return bf16_kv_instance_description(instance).grouped; }
};

// The invariant route profile aligns the splits to each token's own position
// (CausalKvPartition::aligned_to_positions), never to the call's KV row.
Bf16KvCausalPlan make_bf16_kv_causal_plan(int query_heads, int kv_heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope, int multiprocessor_count);
std::size_t bf16_kv_workspace_bytes(int query_heads, int kv_heads, int batch, int min_width,
                                    int max_width, CausalAttentionExecutionEnvelope envelope, int multiprocessor_count);

} // namespace ninferno::ops::detail
