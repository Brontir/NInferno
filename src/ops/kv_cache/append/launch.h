#pragma once

#include "ninferno/ops/kv_cache_append.h"

#include <cstdint>

namespace ninferno::ops::detail {

// From this many tokens a two-KV-head full append of a quantized cache (FP8, INT8, NVFP4, K8V4)
// fills pages by 8-token tiles instead of one warp per token and head.
inline constexpr std::int32_t kKVCacheAppendPageTileMinTokens = 128;

void kv_cache_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                            PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_nvfp4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_nvfp4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        PagedKVBatchLayerView cache, cudaStream_t stream);

void kv_cache_append_k8v4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                 PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_k8v4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                       const Tensor& valid_columns, const Tensor& table_rows,
                                       PagedKVBatchLayerView cache, cudaStream_t stream);

void kv_cache_append_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  PagedKVBatchLayerView cache, cudaStream_t stream);

struct KVCacheAppendPrefixPlan {
    std::int32_t tokens;
    std::int32_t min_count;
    std::int32_t max_count;
};

[[nodiscard]] KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope);

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& table_rows,
                                   PagedKVBatchLayerView cache, const KVCacheAppendPrefixPlan& plan,
                                   cudaStream_t stream);
void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& lanes,
                                   CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream);

} // namespace ninferno::ops::detail
