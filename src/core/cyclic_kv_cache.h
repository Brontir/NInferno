#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninferno {

/**
 * Fixed cyclic BF16-K/FP16-V storage with absolute-position addressing.
 *
 * A logical absolute position p resides in physical slot p % capacity. The view deliberately
 * carries no mutable frontier: callers supply the live absolute interval to the consuming Op.
 *
 * Int8 storage (int8 == true) keeps K and V as Int8Group64: per lane one packed block of
 * [head_dim codes] x padded x heads, followed by its FP16 group scales [head_dim / 64] x padded x
 * heads. The planes are I8 tensors of shape [head_dim + 2 * head_dim / 64, padded, heads, lanes],
 * so a lane is still one contiguous slice (copies, transfers and memsets slice dimension 3).
 */
struct CyclicKVCacheLayerView {
    Tensor k;
    Tensor v;
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    std::int32_t lane_capacity    = 0;
    bool int8                     = false;
};

// Bytes of one I8 cyclic row (codes plus the row's FP16 G64 scales).
[[nodiscard]] constexpr std::int32_t cyclic_kv_int8_row_bytes(std::int32_t head_dim) noexcept {
    return head_dim + 2 * (head_dim / 64);
}

struct CyclicKVCacheLayout {
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    std::int32_t lane_capacity    = 0;
    bool int8                     = false;
    std::vector<TensorRegion> k;
    std::vector<TensorRegion> v;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] CyclicKVCacheLayout
plan_cyclic_kv_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                     std::int32_t num_kv_heads, std::int32_t head_dim, std::int32_t lane_capacity,
                     bool int8 = false);

class CyclicKVCache {
public:
    CyclicKVCache(DeviceSpan backing, const CyclicKVCacheLayout& layout);

    CyclicKVCache(const CyclicKVCache&)            = delete;
    CyclicKVCache& operator=(const CyclicKVCache&) = delete;
    CyclicKVCache(CyclicKVCache&&)                 = delete;
    CyclicKVCache& operator=(CyclicKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t layer_count() const noexcept;

    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::uint32_t padded_capacity() const noexcept { return padded_capacity_; }

    [[nodiscard]] std::int32_t num_kv_heads() const noexcept { return num_kv_heads_; }

    [[nodiscard]] std::int32_t head_dim() const noexcept { return head_dim_; }

    [[nodiscard]] std::int32_t lane_capacity() const noexcept { return lane_capacity_; }

    [[nodiscard]] bool int8() const noexcept { return int8_; }

    // Elements of one row in the planes' innermost dimension (bytes for int8 storage).
    [[nodiscard]] std::int32_t row_elements() const noexcept {
        return int8_ ? cyclic_kv_int8_row_bytes(head_dim_) : head_dim_;
    }

    [[nodiscard]] CyclicKVCacheLayerView layer_view(std::uint32_t layer) const;

    // Copies one slot's complete fixed state. Source and destination geometry must match.
    void copy_slot_from(const CyclicKVCache& source, std::int32_t source_slot,
                        std::int32_t destination_slot, cudaStream_t stream);

private:
    std::vector<Tensor> k_;
    std::vector<Tensor> v_;
    std::uint32_t capacity_        = 0;
    std::uint32_t padded_capacity_ = 0;
    std::int32_t num_kv_heads_     = 0;
    std::int32_t head_dim_         = 0;
    std::int32_t lane_capacity_    = 0;
    bool int8_                     = false;
};

} // namespace ninferno
