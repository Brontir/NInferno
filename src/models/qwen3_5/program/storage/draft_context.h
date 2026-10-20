#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/cyclic_kv_cache.h"
#include "models/qwen3_5/program/planning/startup.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninferno::models::qwen3_5::detail {

struct DFlashPersistentState {
    // The drafter's sliding layers as rings of the state images; null for a drafter without one
    // (gpt-oss's: full layers only).
    CyclicKVCache* local;
    std::optional<qwen3_5::PagedKVCache> full;
    Tensor prefill_features;
    Tensor prefill_positions;
    Tensor pending_features;

    DFlashPersistentState(DeviceSpan backing, const DFlashPersistentLayout& layout,
                          CyclicKVCache* local_state);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    [[nodiscard]] PagedKVBatchLayerView full_batch_layer(std::uint32_t layer) const;
    void save_rewrite_checkpoint(std::int32_t source_slot, std::int32_t destination_slot,
                                 cudaStream_t stream);
};

} // namespace ninferno::models::qwen3_5::detail
