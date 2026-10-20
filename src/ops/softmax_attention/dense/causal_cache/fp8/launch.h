#pragma once

#include "ninferno/ops/softmax_attention.h"

namespace ninferno::ops::detail {

// Returns true when gate (optional, laid out like out) was folded into the final store; the
// caller applies it otherwise.
bool fp8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& valid, const Tensor& rows,
                             float scale, PagedKVBatchLayerView cache,
                             CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                             Tensor& out, DeviceExecutionView execution, const Tensor* gate);

void fp8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                             const PagedKVLayerView& cache,
                             CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                             Tensor& out, DeviceExecutionView execution);

} // namespace ninferno::ops::detail
