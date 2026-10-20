#pragma once

#include "ops/softmax_attention/dense/causal_cache/bf16/operands.h"
#include "ninferno/ops/softmax_attention.h"

namespace ninferno::ops::detail {

// Returns true when gate (optional, laid out like out) was folded into the final store; the
// caller applies it otherwise.
bool bf16_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution,
                              const Tensor* gate);
void bf16_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution);

} // namespace ninferno::ops::detail
