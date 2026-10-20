#pragma once

#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/operands.h"

namespace ninferno::ops::detail {
void k8v4_kv_tiled_attention(const CausalAttentionOperands&, K8V4KvReadView, CausalKvPartition,
                             WorkspaceArena&, cudaStream_t);
} // namespace ninferno::ops::detail
