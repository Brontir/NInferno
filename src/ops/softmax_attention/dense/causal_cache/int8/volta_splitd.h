#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "core/layout.h"
#include "ninferno/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail {

// Volta (sm_70) prefill attention over an INT8-G64 cache (volta_splitd.cu): the visible keys are
// dequantized once into contiguous FP16, and FP16 tensor-core flash attention with FP32
// accumulators (the vendored sm70_flash_d256 split-D kernel) runs 1024-query blocks against them.
// It stands in for the tiled INT8 kernel, whose MMA tiles run on FP32 FMAs on that GPU.

// Whether a prefill call of `width` columns takes it: an sm_70 build with the kernel compiled, a
// GPU without tensor mma.sync, one sequence of more than 256 columns, query heads a multiple of
// 2, 4 or 8 KV heads, and an exact envelope (min = max visible keys: the call's positions are the
// last `width` keys). The workspace query asks the same, and a launch whose workspace cannot
// hold it keeps the tiled kernel.
[[nodiscard]] bool int8_volta_splitd_takes(int query_heads, int kv_heads, int width, int batch,
                                           CausalAttentionExecutionEnvelope envelope);

// Workspace of a call of up to max_width columns over up to max_visible_keys keys.
[[nodiscard]] std::size_t int8_volta_splitd_workspace_bytes(int query_heads, int kv_heads,
                                                            int max_width,
                                                            std::uint32_t max_visible_keys);

// q [256, QHeads, T, 1] BF16; the call's K/V are already in the cache; out [256, QHeads, T, 1]
// BF16 without the output gate (the caller applies it).
void int8_volta_splitd(const Tensor& q, const Tensor& rows, float scale,
                       const PagedKVBatchLayerView& cache,
                       CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                       Tensor& out, cudaStream_t stream);

} // namespace ninferno::ops::detail
