#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/transfer_work.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninferno::models::qwen3_5 {

struct DFlashLocalStateSpec {
    std::uint32_t layers   = 0;
    std::uint32_t capacity = 0;
    std::int32_t kv_heads  = 0;
    std::int32_t head_dim  = 0;
    bool int8              = false; // Int8Group64 ring (core/cyclic_kv_cache.h)
};

// One group of sliding-window rings: per layer a K and a V plane [head_dim, capacity, kv_heads]
// per slot, position p in row p % capacity. A family whose sliding layers keep their recent keys
// here carries them with the rest of a sequence's state image. BF16 planes by default; fp8: E4M3
// code planes, each row (one position of one head) with an FP32 scale in a K and a V scale plane
// [1, capacity, kv_heads] per slot (the row is scale * code; ops/gemma4_attention.h FP8 storage).
struct KvRingSpec {
    std::uint32_t layers   = 0;
    std::uint32_t capacity = 0;
    std::int32_t kv_heads  = 0;
    std::int32_t head_dim  = 0;
    bool fp8               = false;

    friend bool operator==(const KvRingSpec&, const KvRingSpec&) = default;
};

// linear.slot_count is the pool's slot count; linear.layers == 0 means the family keeps no linear
// attention state.
struct StateImageSpec {
    LinearAttentionStatePoolSpec linear;
    std::int32_t hidden = 0;
    std::optional<DFlashLocalStateSpec> dflash_local;
    std::vector<KvRingSpec> rings;
};

struct KvRingHostLayout {
    LayoutRegion k;
    LayoutRegion v;
    std::size_t layer_bytes = 0;
    // FP8 rings only (empty regions otherwise): the scale planes, scale_layer_bytes per layer.
    LayoutRegion k_scale;
    LayoutRegion v_scale;
    std::size_t scale_layer_bytes = 0;
};

// Per layer of the group, a K and a V plane [head_dim, capacity, kv_heads, slots], and for FP8
// rings a K and a V scale plane [1, capacity, kv_heads, slots].
struct KvRingDeviceLayout {
    KvRingSpec spec;
    std::vector<TensorRegion> k;
    std::vector<TensorRegion> v;
    std::vector<TensorRegion> k_scale;
    std::vector<TensorRegion> v_scale;
};

struct StateImageHostLayout {
    StateImageSpec spec;
    LayoutRegion linear_conv;
    std::size_t linear_conv_layer_bytes = 0;
    LayoutRegion linear_recurrent;
    std::size_t linear_recurrent_layer_bytes = 0;
    LayoutRegion continuation_hidden;
    std::optional<LayoutRegion> dflash_local_k;
    std::optional<LayoutRegion> dflash_local_v;
    std::size_t dflash_local_layer_bytes = 0;
    std::vector<KvRingHostLayout> rings;
    std::size_t image_bytes = 0;
};

struct StateImageDeviceLayout {
    std::optional<LinearAttentionStatePoolLayout> linear;
    TensorRegion continuation_hidden;
    std::optional<CyclicKVCacheLayout> dflash_local;
    std::vector<KvRingDeviceLayout> rings;
    StateImageHostLayout host;
};

[[nodiscard]] TransferWork state_image_transfer_work(const StateImageHostLayout& layout);
[[nodiscard]] TransferWork dflash_local_transfer_work(const StateImageHostLayout& layout);

[[nodiscard]] StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                                  const StateImageSpec& spec);

struct HostStateImageView {
    std::byte* data                    = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

struct HostStateImageConstView {
    const std::byte* data              = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

struct HostStateSlotHandle {
    std::uint32_t index      = 0;
    std::uint32_t generation = 0;
};

/** Fixed-capacity pinned storage for complete physical StateImage payloads; owns no cache policy.
 */
class HostStatePool {
public:
    HostStatePool(StateImageHostLayout layout, std::uint32_t capacity);

    HostStatePool(const HostStatePool&)            = delete;
    HostStatePool& operator=(const HostStatePool&) = delete;
    HostStatePool(HostStatePool&&)                 = delete;
    HostStatePool& operator=(HostStatePool&&)      = delete;

    [[nodiscard]] std::optional<HostStateSlotHandle> allocate() noexcept;
    [[nodiscard]] bool release(HostStateSlotHandle handle) noexcept;

    [[nodiscard]] HostStateImageView writable_view(HostStateSlotHandle handle);
    [[nodiscard]] HostStateImageConstView view(HostStateSlotHandle handle) const;

    [[nodiscard]] std::uint32_t capacity() const noexcept;

    [[nodiscard]] std::uint32_t occupied() const noexcept { return occupied_; }

    [[nodiscard]] const StateImageHostLayout& layout() const noexcept { return layout_; }

private:
    struct Slot {
        std::uint32_t generation = 1;
        bool occupied            = false;
    };

    [[nodiscard]] bool valid(HostStateSlotHandle handle) const noexcept;
    [[nodiscard]] std::byte* slot_data(std::uint32_t index) const noexcept;

    StateImageHostLayout layout_;
    std::optional<PinnedHostBuffer> backing_;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_slots_;
    std::uint32_t free_count_ = 0;
    std::uint32_t occupied_   = 0;
};

/**
 * Caller-backed fixed storage for Qwen3.6 continuation state.
 *
 * Every absolute slot contains common GDN/hidden state and, for a DFlash Program, its local cyclic
 * K/V state. The pool owns neither slot roles nor logical checkpoint identity.
 */
class StateImageDevicePool {
public:
    StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout);

    StateImageDevicePool(const StateImageDevicePool&)            = delete;
    StateImageDevicePool& operator=(const StateImageDevicePool&) = delete;
    StateImageDevicePool(StateImageDevicePool&&)                 = delete;
    StateImageDevicePool& operator=(StateImageDevicePool&&)      = delete;

    [[nodiscard]] std::int32_t slot_count() const noexcept { return slot_count_; }

    [[nodiscard]] Tensor continuation_hidden_slot(std::int32_t slot) const;

    [[nodiscard]] bool has_linear() const noexcept { return linear_.has_value(); }

    // The linear attention state; a pool without it throws std::logic_error.
    [[nodiscard]] LinearAttentionStatePool& linear();
    [[nodiscard]] const LinearAttentionStatePool& linear() const;

    // Ring group `group`, layer `layer`: its K and V planes [head_dim, capacity, kv_heads, slots]
    // (BF16, or E4M3 codes), and for FP8 rings its FP32 scale planes [1, capacity, kv_heads,
    // slots] (empty for BF16 rings).
    struct RingPlanes {
        Tensor k;
        Tensor v;
        Tensor k_scale;
        Tensor v_scale;
    };

    [[nodiscard]] std::size_t ring_groups() const noexcept { return rings_.size(); }

    [[nodiscard]] const KvRingSpec& ring_spec(std::size_t group) const;
    [[nodiscard]] RingPlanes ring_planes(std::size_t group, std::uint32_t layer) const;

    [[nodiscard]] Tensor& continuation_hidden_store() noexcept { return continuation_hidden_; }

    [[nodiscard]] const Tensor& continuation_hidden_store() const noexcept {
        return continuation_hidden_;
    }

    [[nodiscard]] CyclicKVCache* dflash_local() noexcept;
    [[nodiscard]] const CyclicKVCache* dflash_local() const noexcept;

    [[nodiscard]] const StateImageHostLayout& host_layout() const noexcept { return host_layout_; }

    void zero_slot(std::int32_t slot, cudaStream_t stream);
    // Sets every byte of one slot (linear state, continuation hidden, DFlash local KV) to value.
    void fill_slot(std::int32_t slot, unsigned char value, cudaStream_t stream);
    void zero_all(cudaStream_t stream);
    void copy_slot(std::int32_t source, std::int32_t destination, cudaStream_t stream);
    void copy_dflash_local(std::int32_t source, std::int32_t destination,
                           cudaStream_t stream);
    // Copies one slot's rings, every group and layer.
    void copy_rings(std::int32_t source, std::int32_t destination, cudaStream_t stream);
    void copy_to_host(std::int32_t source, HostStateImageView destination,
                      cudaStream_t stream) const;
    void copy_from_host(HostStateImageConstView source, std::int32_t destination,
                        cudaStream_t stream);

private:
    void validate_host_layout(const StateImageHostLayout* layout, const std::byte* data) const;

    struct RingGroup {
        KvRingSpec spec;
        std::vector<Tensor> k;
        std::vector<Tensor> v;
        std::vector<Tensor> k_scale; // FP8 rings only
        std::vector<Tensor> v_scale;

        // Every plane of one layer: K, V, then (FP8) the K and V scales.
        [[nodiscard]] std::vector<const Tensor*> planes(std::size_t layer) const;
    };

    std::optional<LinearAttentionStatePool> linear_;
    std::int32_t slot_count_ = 0;
    Tensor continuation_hidden_;
    std::optional<CyclicKVCache> dflash_local_;
    std::vector<RingGroup> rings_;
    StateImageHostLayout host_layout_;
};

} // namespace ninferno::models::qwen3_5
