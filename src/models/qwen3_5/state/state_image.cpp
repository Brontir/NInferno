#include "models/qwen3_5/state/state_image.h"

#include "core/checked.h"
#include "core/device.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninferno::models::qwen3_5 {
namespace {

constexpr std::size_t kStateImageAlignment = 256;

std::uint32_t padded_dflash_capacity(std::uint32_t capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("StateImage DFlash capacity must be positive");
    }
    constexpr std::uint64_t alignment = 128;
    const std::uint64_t padded =
        (static_cast<std::uint64_t>(capacity) + alignment - 1U) & ~(alignment - 1U);
    if (padded > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("StateImage DFlash padded capacity exceeds int32");
    }
    return static_cast<std::uint32_t>(padded);
}

bool same_linear_spec(const LinearAttentionStatePoolSpec& left,
                      const LinearAttentionStatePoolSpec& right) noexcept {
    return left.layers == right.layers && left.conv_channels == right.conv_channels &&
           left.conv_width == right.conv_width && left.value_heads == right.value_heads &&
           left.value_head_dim == right.value_head_dim && left.key_head_dim == right.key_head_dim &&
           left.slot_count == right.slot_count && left.conv_dtype == right.conv_dtype &&
           left.recurrent_dtype == right.recurrent_dtype;
}

bool same_dflash_spec(const std::optional<DFlashLocalStateSpec>& left,
                      const std::optional<DFlashLocalStateSpec>& right) noexcept {
    if (left.has_value() != right.has_value()) { return false; }
    if (!left) { return true; }
    return left->layers == right->layers && left->capacity == right->capacity &&
           left->kv_heads == right->kv_heads && left->head_dim == right->head_dim &&
           left->int8 == right->int8;
}

bool same_region(const LayoutRegion& left, const LayoutRegion& right) noexcept;

bool same_rings(const std::vector<KvRingHostLayout>& left,
                const std::vector<KvRingHostLayout>& right) noexcept {
    if (left.size() != right.size()) { return false; }
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (!same_region(left[i].k, right[i].k) || !same_region(left[i].v, right[i].v) ||
            left[i].layer_bytes != right[i].layer_bytes ||
            !same_region(left[i].k_scale, right[i].k_scale) ||
            !same_region(left[i].v_scale, right[i].v_scale) ||
            left[i].scale_layer_bytes != right[i].scale_layer_bytes) {
            return false;
        }
    }
    return true;
}

bool same_region(const LayoutRegion& left, const LayoutRegion& right) noexcept {
    return left.offset == right.offset && left.bytes == right.bytes &&
           left.alignment == right.alignment;
}

bool same_optional_region(const std::optional<LayoutRegion>& left,
                          const std::optional<LayoutRegion>& right) noexcept {
    return left.has_value() == right.has_value() && (!left || same_region(*left, *right));
}

bool same_host_layout(const StateImageHostLayout& left,
                      const StateImageHostLayout& right) noexcept {
    return same_linear_spec(left.spec.linear, right.spec.linear) &&
           left.spec.hidden == right.spec.hidden &&
           same_dflash_spec(left.spec.dflash_local, right.spec.dflash_local) &&
           same_region(left.linear_conv, right.linear_conv) &&
           left.linear_conv_layer_bytes == right.linear_conv_layer_bytes &&
           same_region(left.linear_recurrent, right.linear_recurrent) &&
           left.linear_recurrent_layer_bytes == right.linear_recurrent_layer_bytes &&
           same_region(left.continuation_hidden, right.continuation_hidden) &&
           same_optional_region(left.dflash_local_k, right.dflash_local_k) &&
           same_optional_region(left.dflash_local_v, right.dflash_local_v) &&
           left.dflash_local_layer_bytes == right.dflash_local_layer_bytes &&
           left.spec.rings == right.spec.rings && same_rings(left.rings, right.rings) &&
           left.image_bytes == right.image_bytes;
}

// One slot of a ring layer: its K (or V) plane [head_dim, capacity, kv_heads].
Tensor ring_slot_shape(const KvRingSpec& ring) {
    return Tensor(nullptr, ring.fp8 ? DType::FP8_E4M3FN : DType::BF16,
                  {ring.head_dim, static_cast<std::int32_t>(ring.capacity), ring.kv_heads});
}

// One slot of an FP8 ring layer's K (or V) scale plane [1, capacity, kv_heads].
Tensor ring_scale_slot_shape(const KvRingSpec& ring) {
    return Tensor(nullptr, DType::FP32, {1, static_cast<std::int32_t>(ring.capacity), ring.kv_heads});
}

StateImageHostLayout plan_host_state_image(const StateImageSpec& spec) {
    if (spec.linear.slot_count <= 0 || spec.hidden <= 0 ||
        (spec.linear.layers != 0 &&
         (spec.linear.conv_channels <= 0 || spec.linear.conv_width <= 0 ||
          spec.linear.value_heads <= 0 || spec.linear.value_head_dim <= 0 ||
          spec.linear.key_head_dim <= 0 ||
          (spec.linear.conv_dtype != DType::BF16 && spec.linear.conv_dtype != DType::FP32) ||
          (spec.linear.recurrent_dtype != DType::BF16 &&
           spec.linear.recurrent_dtype != DType::FP16 &&
           spec.linear.recurrent_dtype != DType::FP32)))) {
        throw std::invalid_argument("StateImage host geometry is invalid");
    }
    for (const KvRingSpec& ring : spec.rings) {
        if (ring.layers == 0 || ring.kv_heads <= 0 || ring.head_dim <= 0 || ring.capacity == 0 ||
            ring.capacity > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("StateImage ring geometry is invalid");
        }
    }
    if (spec.dflash_local && (spec.dflash_local->layers == 0 || spec.dflash_local->kv_heads <= 0 ||
                              spec.dflash_local->head_dim <= 0)) {
        throw std::invalid_argument("StateImage host DFlash geometry is invalid");
    }

    LayoutBuilder builder;
    StateImageHostLayout host;
    host.spec = spec;
    if (spec.linear.layers != 0) {
        const Tensor conv_slot(nullptr, spec.linear.conv_dtype,
                               {spec.linear.conv_channels, spec.linear.conv_width});
        const Tensor recurrent_slot(
            nullptr, spec.linear.recurrent_dtype,
            {spec.linear.key_head_dim, spec.linear.value_head_dim, spec.linear.value_heads});
        host.linear_conv_layer_bytes = conv_slot.bytes();
        host.linear_conv = builder.add(checked_mul(host.linear_conv_layer_bytes, spec.linear.layers,
                                                   "StateImage host convolution bytes overflow"),
                                       kStateImageAlignment, "StateImage host convolution");
        host.linear_recurrent_layer_bytes = recurrent_slot.bytes();
        host.linear_recurrent =
            builder.add(checked_mul(host.linear_recurrent_layer_bytes, spec.linear.layers,
                                    "StateImage host recurrent bytes overflow"),
                        kStateImageAlignment, "StateImage host recurrent");
    }
    const Tensor hidden_slot(nullptr, DType::BF16, {spec.hidden});
    host.continuation_hidden = builder.add(hidden_slot.bytes(), kStateImageAlignment,
                                           "StateImage host continuation hidden");
    if (spec.dflash_local) {
        const bool int8 = spec.dflash_local->int8;
        const Tensor local_slot(
            nullptr, int8 ? DType::I8 : DType::BF16,
            {int8 ? cyclic_kv_int8_row_bytes(spec.dflash_local->head_dim)
                  : spec.dflash_local->head_dim,
             static_cast<std::int32_t>(padded_dflash_capacity(spec.dflash_local->capacity)),
             spec.dflash_local->kv_heads});
        host.dflash_local_layer_bytes = local_slot.bytes();
        const std::size_t component_bytes =
            checked_mul(host.dflash_local_layer_bytes, spec.dflash_local->layers,
                        "StateImage host DFlash local bytes overflow");
        host.dflash_local_k =
            builder.add(component_bytes, kStateImageAlignment, "StateImage host DFlash local K");
        host.dflash_local_v =
            builder.add(component_bytes, kStateImageAlignment, "StateImage host DFlash local V");
    }
    for (const KvRingSpec& ring : spec.rings) {
        KvRingHostLayout& out = host.rings.emplace_back();
        out.layer_bytes       = ring_slot_shape(ring).bytes();
        const std::size_t component_bytes =
            checked_mul(out.layer_bytes, ring.layers, "StateImage host ring bytes overflow");
        out.k = builder.add(component_bytes, kStateImageAlignment, "StateImage host ring K");
        out.v = builder.add(component_bytes, kStateImageAlignment, "StateImage host ring V");
        if (ring.fp8) {
            out.scale_layer_bytes = ring_scale_slot_shape(ring).bytes();
            const std::size_t scale_bytes = checked_mul(out.scale_layer_bytes, ring.layers,
                                                        "StateImage host ring scale bytes overflow");
            out.k_scale =
                builder.add(scale_bytes, kStateImageAlignment, "StateImage host ring K scale");
            out.v_scale =
                builder.add(scale_bytes, kStateImageAlignment, "StateImage host ring V scale");
        }
    }
    host.image_bytes = builder.finish(kStateImageAlignment, "StateImage host image");
    return host;
}

std::byte* byte_offset(std::byte* base, std::size_t offset) noexcept { return base + offset; }

const std::byte* byte_offset(const std::byte* base, std::size_t offset) noexcept {
    return base + offset;
}

// A run of ring rows: rows first .. first + count - 1 of every head.
struct RowRun {
    std::size_t first = 0;
    std::size_t count = 0;
};

// The rows of a ring that a later call from `frontier` still reads: its last `window` positions,
// position p in row p % capacity, as one run or two (wrapping). Every row when the ring has no
// window or the frontier is not given.
std::vector<RowRun> live_rows(const KvRingSpec& spec, std::optional<std::uint64_t> frontier) {
    const std::uint64_t capacity = spec.capacity;
    if (!frontier || spec.window == 0 || spec.window >= capacity) {
        return {RowRun{.first = 0, .count = capacity}};
    }
    const std::uint64_t first = *frontier > spec.window ? *frontier - spec.window : 0;
    const std::uint64_t count = *frontier - first;
    if (count == 0) { return {}; }
    const std::uint64_t start = first % capacity;
    if (start + count <= capacity) { return {RowRun{.first = start, .count = count}}; }
    return {RowRun{.first = start, .count = capacity - start},
            RowRun{.first = 0, .count = start + count - capacity}};
}

// Copies the runs of one slot's ring plane [inner, capacity, kv_heads] (contiguous, as the Host
// image holds it): one copy for the whole plane, else a 2D copy a run (its rows of every head).
void copy_ring_rows(void* destination, const void* source, const Tensor& plane,
                    std::uint32_t capacity, const std::vector<RowRun>& runs, cudaMemcpyKind kind,
                    cudaStream_t stream) {
    if (runs.size() == 1 && runs.front().first == 0 && runs.front().count == capacity) {
        CUDA_CHECK(cudaMemcpyAsync(destination, source, plane.bytes(), kind, stream));
        return;
    }
    const auto heads        = static_cast<std::size_t>(plane.ne[2]);
    const std::size_t row   = plane.bytes() / (static_cast<std::size_t>(capacity) * heads);
    const std::size_t pitch = row * capacity;
    auto* const out         = static_cast<std::byte*>(destination);
    const auto* const in    = static_cast<const std::byte*>(source);
    for (const RowRun& run : runs) {
        CUDA_CHECK(cudaMemcpy2DAsync(out + run.first * row, pitch, in + run.first * row, pitch,
                                     run.count * row, heads, kind, stream));
    }
}

void validate_slot(std::int32_t slot, std::int32_t slot_count, const char* label) {
    if (slot < 0 || slot >= slot_count) { throw std::out_of_range(label); }
}

} // namespace

StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                    const StateImageSpec& spec) {
    if (spec.hidden <= 0) {
        throw std::invalid_argument("StateImage hidden width must be positive");
    }

    StateImageDeviceLayout out;
    if (spec.linear.layers != 0) {
        out.linear = plan_linear_attention_state_pool(builder, spec.linear);
    }
    out.continuation_hidden =
        builder.add_tensor(DType::BF16, {spec.hidden, spec.linear.slot_count}, kStateImageAlignment,
                           "StateImage continuation hidden");
    if (spec.dflash_local) {
        const DFlashLocalStateSpec& dflash = *spec.dflash_local;
        out.dflash_local =
            plan_cyclic_kv_cache(builder, dflash.layers, dflash.capacity, dflash.kv_heads,
                                 dflash.head_dim, spec.linear.slot_count, dflash.int8);
    }
    for (const KvRingSpec& ring : spec.rings) {
        KvRingDeviceLayout& group = out.rings.emplace_back();
        group.spec                = ring;
        for (std::uint32_t layer = 0; layer < ring.layers; ++layer) {
            for (auto* plane : {&group.k, &group.v}) {
                plane->push_back(
                    builder.add_tensor(ring.fp8 ? DType::FP8_E4M3FN : DType::BF16,
                                       {ring.head_dim, static_cast<std::int32_t>(ring.capacity),
                                        ring.kv_heads, spec.linear.slot_count},
                                       kStateImageAlignment, "StateImage ring plane"));
            }
            if (!ring.fp8) { continue; }
            for (auto* plane : {&group.k_scale, &group.v_scale}) {
                plane->push_back(builder.add_tensor(
                    DType::FP32,
                    {1, static_cast<std::int32_t>(ring.capacity), ring.kv_heads,
                     spec.linear.slot_count},
                    kStateImageAlignment, "StateImage ring scale plane"));
            }
        }
    }

    out.host = plan_host_state_image(spec);
    return out;
}

TransferWork state_image_transfer_work(const StateImageHostLayout& layout) {
    std::size_t payload = checked_add(layout.linear_conv.bytes, layout.linear_recurrent.bytes,
                                      "StateImage transfer payload overflow");
    payload             = checked_add(payload, layout.continuation_hidden.bytes,
                                      "StateImage transfer payload overflow");
    if (layout.dflash_local_k) {
        if (!layout.spec.dflash_local || !layout.dflash_local_v) {
            throw std::invalid_argument("StateImage DFlash transfer layout is incomplete");
        }
        const std::size_t component_bytes =
            checked_mul(layout.dflash_local_layer_bytes, layout.spec.dflash_local->layers,
                        "StateImage transfer payload overflow");
        payload = checked_add(
            payload, checked_mul(component_bytes, 2U, "StateImage transfer payload overflow"),
            "StateImage transfer payload overflow");
    }
    std::uint64_t operations =
        2ULL * layout.spec.linear.layers + 1ULL +
        (layout.spec.dflash_local ? 2ULL * layout.spec.dflash_local->layers : 0ULL);
    for (std::size_t group = 0; group < layout.rings.size(); ++group) {
        const KvRingHostLayout& ring = layout.rings[group];
        for (const std::size_t bytes : {ring.k.bytes, ring.v.bytes, ring.k_scale.bytes,
                                        ring.v_scale.bytes}) {
            payload = checked_add(payload, bytes, "StateImage transfer payload overflow");
        }
        operations += (layout.spec.rings[group].fp8 ? 4ULL : 2ULL) * layout.spec.rings[group].layers;
    }
    if (operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("StateImage transfer operation count exceeds uint32");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(payload),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

TransferWork dflash_local_transfer_work(const StateImageHostLayout& layout) {
    if (!layout.spec.dflash_local || !layout.dflash_local_k || !layout.dflash_local_v) {
        throw std::invalid_argument("StateImage has no DFlash local component");
    }
    const std::size_t component_bytes =
        checked_mul(layout.dflash_local_layer_bytes, layout.spec.dflash_local->layers,
                    "StateImage DFlash transfer payload overflow");
    const std::uint64_t operations = 2ULL * layout.spec.dflash_local->layers;
    if (component_bytes > std::numeric_limits<std::uint64_t>::max() / 2U ||
        operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("StateImage DFlash transfer work exceeds its representation");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(2U * component_bytes),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

HostStatePool::HostStatePool(StateImageHostLayout layout, std::uint32_t capacity)
    : layout_(std::move(layout)), slots_(capacity), free_slots_(capacity), free_count_(capacity) {
    if (!same_host_layout(layout_, plan_host_state_image(layout_.spec))) {
        throw std::invalid_argument("HostStatePool image layout is invalid");
    }
    const std::size_t bytes =
        checked_mul(layout_.image_bytes, capacity, "HostStatePool backing size overflow");
    if (bytes != 0) { backing_.emplace(bytes); }
    for (std::uint32_t index = 0; index < capacity; ++index) {
        free_slots_[index] = capacity - 1U - index;
    }
}

std::optional<HostStateSlotHandle> HostStatePool::allocate() noexcept {
    if (free_count_ == 0) { return std::nullopt; }
    const std::uint32_t index = free_slots_[--free_count_];
    Slot& slot                = slots_[index];
    slot.occupied             = true;
    ++occupied_;
    return HostStateSlotHandle{.index = index, .generation = slot.generation};
}

bool HostStatePool::release(HostStateSlotHandle handle) noexcept {
    if (!valid(handle)) { return false; }
    Slot& slot    = slots_[handle.index];
    slot.occupied = false;
    if (++slot.generation == 0) { ++slot.generation; }
    free_slots_[free_count_++] = handle.index;
    --occupied_;
    return true;
}

HostStateImageView HostStatePool::writable_view(HostStateSlotHandle handle) {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

HostStateImageConstView HostStatePool::view(HostStateSlotHandle handle) const {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

std::uint32_t HostStatePool::capacity() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
}

bool HostStatePool::valid(HostStateSlotHandle handle) const noexcept {
    return handle.index < slots_.size() && slots_[handle.index].occupied &&
           slots_[handle.index].generation == handle.generation;
}

std::byte* HostStatePool::slot_data(std::uint32_t index) const noexcept {
    return static_cast<std::byte*>(backing_->data()) +
           static_cast<std::size_t>(index) * layout_.image_bytes;
}

StateImageDevicePool::StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout)
    : slot_count_(layout.host.spec.linear.slot_count),
      continuation_hidden_(layout.continuation_hidden.bind(backing)), host_layout_(layout.host) {
    if (layout.linear) { linear_.emplace(backing, *layout.linear); }
    if (layout.linear.has_value() != (host_layout_.spec.linear.layers != 0) ||
        (linear_ && linear_->slot_count() != slot_count_)) {
        throw std::invalid_argument("StateImage linear layout is inconsistent");
    }
    if (continuation_hidden_.dtype != DType::BF16 || !continuation_hidden_.is_contiguous() ||
        continuation_hidden_.ne[0] != host_layout_.spec.hidden ||
        continuation_hidden_.ne[1] != slot_count_) {
        throw std::invalid_argument("StateImage continuation hidden layout is inconsistent");
    }
    if (layout.dflash_local.has_value() != host_layout_.spec.dflash_local.has_value()) {
        throw std::invalid_argument("StateImage DFlash layout is inconsistent");
    }
    StateImageSpec device_spec{.linear =
                                   layout.linear ? layout.linear->spec : host_layout_.spec.linear,
                               .hidden = continuation_hidden_.ne[0]};
    if (layout.dflash_local) {
        device_spec.dflash_local = DFlashLocalStateSpec{
            .layers   = static_cast<std::uint32_t>(layout.dflash_local->k.size()),
            .capacity = layout.dflash_local->capacity,
            .kv_heads = layout.dflash_local->num_kv_heads,
            .head_dim = layout.dflash_local->head_dim,
            .int8     = layout.dflash_local->int8,
        };
    }
    for (const KvRingDeviceLayout& group : layout.rings) {
        device_spec.rings.push_back(group.spec);
        RingGroup& bound = rings_.emplace_back();
        bound.spec       = group.spec;
        const std::size_t scale_layers = group.spec.fp8 ? group.spec.layers : 0;
        if (group.k.size() != group.spec.layers || group.v.size() != group.spec.layers ||
            group.k_scale.size() != scale_layers || group.v_scale.size() != scale_layers) {
            throw std::invalid_argument("StateImage ring layout is inconsistent");
        }
        for (std::uint32_t layer = 0; layer < group.spec.layers; ++layer) {
            bound.k.push_back(group.k[layer].bind(backing));
            bound.v.push_back(group.v[layer].bind(backing));
            if (group.spec.fp8) {
                bound.k_scale.push_back(group.k_scale[layer].bind(backing));
                bound.v_scale.push_back(group.v_scale[layer].bind(backing));
            }
            for (const Tensor* plane : bound.planes(layer)) {
                if (plane->ne[3] != slot_count_) {
                    throw std::invalid_argument("StateImage ring slots do not match the pool");
                }
            }
        }
    }
    if (!same_host_layout(host_layout_, plan_host_state_image(device_spec))) {
        throw std::invalid_argument("StateImage host layout does not match its device components");
    }
    if (layout.dflash_local) {
        if (layout.dflash_local->lane_capacity != slot_count_) {
            throw std::invalid_argument("StateImage components do not share one slot geometry");
        }
        dflash_local_.emplace(backing, *layout.dflash_local);
    }
}

LinearAttentionStatePool& StateImageDevicePool::linear() {
    if (!linear_) { throw std::logic_error("StateImage has no linear attention state"); }
    return *linear_;
}

const LinearAttentionStatePool& StateImageDevicePool::linear() const {
    if (!linear_) { throw std::logic_error("StateImage has no linear attention state"); }
    return *linear_;
}

const KvRingSpec& StateImageDevicePool::ring_spec(std::size_t group) const {
    return rings_.at(group).spec;
}

std::vector<const Tensor*> StateImageDevicePool::RingGroup::planes(std::size_t layer) const {
    std::vector<const Tensor*> out{&k.at(layer), &v.at(layer)};
    if (spec.fp8) {
        out.push_back(&k_scale.at(layer));
        out.push_back(&v_scale.at(layer));
    }
    return out;
}

StateImageDevicePool::RingPlanes StateImageDevicePool::ring_planes(std::size_t group,
                                                                   std::uint32_t layer) const {
    const RingGroup& ring = rings_.at(group);
    RingPlanes out{.k = ring.k.at(layer), .v = ring.v.at(layer)};
    if (ring.spec.fp8) {
        out.k_scale = ring.k_scale.at(layer);
        out.v_scale = ring.v_scale.at(layer);
    }
    return out;
}

void StateImageDevicePool::copy_rings(std::int32_t source, std::int32_t destination,
                                      cudaStream_t stream) {
    validate_slot(source, slot_count(), "StateImage ring copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage ring copy destination is out of range");
    if (source == destination) { return; }
    for (const RingGroup& ring : rings_) {
        for (std::size_t layer = 0; layer < ring.k.size(); ++layer) {
            for (const Tensor* plane : ring.planes(layer)) {
                const Tensor from = plane->slice(3, source, 1);
                const Tensor to   = plane->slice(3, destination, 1);
                CUDA_CHECK(cudaMemcpyAsync(to.data, from.data, to.bytes(), cudaMemcpyDeviceToDevice,
                                           stream));
            }
        }
    }
}

Tensor StateImageDevicePool::continuation_hidden_slot(std::int32_t slot) const {
    validate_slot(slot, slot_count(), "StateImage slot is out of range");
    return continuation_hidden_.slice(1, slot, 1).view({host_layout_.spec.hidden});
}

CyclicKVCache* StateImageDevicePool::dflash_local() noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

const CyclicKVCache* StateImageDevicePool::dflash_local() const noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

void StateImageDevicePool::zero_slot(std::int32_t slot, cudaStream_t stream) {
    fill_slot(slot, 0, stream);
}

void StateImageDevicePool::fill_slot(std::int32_t slot, unsigned char value, cudaStream_t stream) {
    validate_slot(slot, slot_count(), "StateImage fill slot is out of range");
    if (linear_) { linear_->fill_slot(slot, value, stream); }
    for (const RingGroup& ring : rings_) {
        for (std::size_t layer = 0; layer < ring.k.size(); ++layer) {
            for (const Tensor* plane : ring.planes(layer)) {
                const Tensor part = plane->slice(3, slot, 1);
                CUDA_CHECK(cudaMemsetAsync(part.data, value, part.bytes(), stream));
            }
        }
    }
    const Tensor hidden = continuation_hidden_slot(slot);
    CUDA_CHECK(cudaMemsetAsync(hidden.data, value, hidden.bytes(), stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, slot, 1);
            const Tensor v                    = view.v.slice(3, slot, 1);
            CUDA_CHECK(cudaMemsetAsync(k.data, value, k.bytes(), stream));
            CUDA_CHECK(cudaMemsetAsync(v.data, value, v.bytes(), stream));
        }
    }
}

void StateImageDevicePool::zero_all(cudaStream_t stream) {
    if (linear_) { linear_->zero_all(stream); }
    for (const RingGroup& ring : rings_) {
        for (std::size_t layer = 0; layer < ring.k.size(); ++layer) {
            for (const Tensor* plane : ring.planes(layer)) {
                CUDA_CHECK(cudaMemsetAsync(plane->data, 0, plane->bytes(), stream));
            }
        }
    }
    CUDA_CHECK(cudaMemsetAsync(continuation_hidden_.data, 0, continuation_hidden_.bytes(), stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            CUDA_CHECK(cudaMemsetAsync(view.k.data, 0, view.k.bytes(), stream));
            CUDA_CHECK(cudaMemsetAsync(view.v.data, 0, view.v.bytes(), stream));
        }
    }
}

void StateImageDevicePool::copy_slot(std::int32_t source, std::int32_t destination,
                                     cudaStream_t stream) {
    validate_slot(source, slot_count(), "StateImage copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage copy destination is out of range");
    if (source == destination) { return; }
    if (linear_) { linear_->copy_slot(source, destination, stream); }
    copy_rings(source, destination, stream);
    const Tensor source_hidden      = continuation_hidden_slot(source);
    const Tensor destination_hidden = continuation_hidden_slot(destination);
    CUDA_CHECK(cudaMemcpyAsync(destination_hidden.data, source_hidden.data,
                               destination_hidden.bytes(), cudaMemcpyDeviceToDevice, stream));
    if (dflash_local_) {
        dflash_local_->copy_slot_from(*dflash_local_, source, destination, stream);
    }
}

void StateImageDevicePool::copy_dflash_local(std::int32_t source, std::int32_t destination,
                                             cudaStream_t stream) {
    validate_slot(source, slot_count(), "StateImage DFlash copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage DFlash copy destination is out of range");
    if (!dflash_local_) { throw std::logic_error("StateImage has no DFlash local component"); }
    if (source != destination) {
        dflash_local_->copy_slot_from(*dflash_local_, source, destination, stream);
    }
}

void StateImageDevicePool::validate_host_layout(const StateImageHostLayout* layout,
                                                const std::byte* data) const {
    if (layout == nullptr || data == nullptr || !same_host_layout(*layout, host_layout_)) {
        throw std::invalid_argument("Host and device StateImage layouts do not match");
    }
}

void StateImageDevicePool::copy_to_host(std::int32_t source, HostStateImageView destination,
                                        cudaStream_t stream,
                                        std::optional<std::uint64_t> frontier) const {
    validate_slot(source, slot_count(), "StateImage copy-to-host source is out of range");
    validate_host_layout(destination.layout, destination.data);
    const std::uint32_t linear_layers = linear_ ? linear_->layer_count() : 0;
    for (std::uint32_t layer = 0; layer < linear_layers; ++layer) {
        const Tensor conv = linear_->conv_slot(layer, source);
        CUDA_CHECK(cudaMemcpyAsync(
            byte_offset(destination.data, host_layout_.linear_conv.offset +
                                              layer * host_layout_.linear_conv_layer_bytes),
            conv.data, conv.bytes(), cudaMemcpyDeviceToHost, stream));
        const Tensor recurrent = linear_->recurrent_slot(layer, source);
        CUDA_CHECK(cudaMemcpyAsync(
            byte_offset(destination.data, host_layout_.linear_recurrent.offset +
                                              layer * host_layout_.linear_recurrent_layer_bytes),
            recurrent.data, recurrent.bytes(), cudaMemcpyDeviceToHost, stream));
    }
    const Tensor hidden = continuation_hidden_slot(source);
    CUDA_CHECK(
        cudaMemcpyAsync(byte_offset(destination.data, host_layout_.continuation_hidden.offset),
                        hidden.data, hidden.bytes(), cudaMemcpyDeviceToHost, stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, source, 1);
            const Tensor v                    = view.v.slice(3, source, 1);
            CUDA_CHECK(cudaMemcpyAsync(
                byte_offset(destination.data, host_layout_.dflash_local_k->offset +
                                                  layer * host_layout_.dflash_local_layer_bytes),
                k.data, k.bytes(), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                byte_offset(destination.data, host_layout_.dflash_local_v->offset +
                                                  layer * host_layout_.dflash_local_layer_bytes),
                v.data, v.bytes(), cudaMemcpyDeviceToHost, stream));
        }
    }
    for (std::size_t group = 0; group < rings_.size(); ++group) {
        const RingGroup& ring        = rings_[group];
        const KvRingHostLayout& host = host_layout_.rings[group];
        const std::vector<RowRun> runs = live_rows(ring.spec, frontier);
        const std::uint32_t capacity   = ring.spec.capacity;
        for (std::size_t layer = 0; layer < ring.k.size(); ++layer) {
            const Tensor k = ring.k[layer].slice(3, source, 1);
            const Tensor v = ring.v[layer].slice(3, source, 1);
            copy_ring_rows(byte_offset(destination.data, host.k.offset + layer * host.layer_bytes),
                           k.data, k, capacity, runs, cudaMemcpyDeviceToHost, stream);
            copy_ring_rows(byte_offset(destination.data, host.v.offset + layer * host.layer_bytes),
                           v.data, v, capacity, runs, cudaMemcpyDeviceToHost, stream);
            if (!ring.spec.fp8) { continue; }
            const Tensor k_scale = ring.k_scale[layer].slice(3, source, 1);
            const Tensor v_scale = ring.v_scale[layer].slice(3, source, 1);
            copy_ring_rows(
                byte_offset(destination.data, host.k_scale.offset + layer * host.scale_layer_bytes),
                k_scale.data, k_scale, capacity, runs, cudaMemcpyDeviceToHost, stream);
            copy_ring_rows(
                byte_offset(destination.data, host.v_scale.offset + layer * host.scale_layer_bytes),
                v_scale.data, v_scale, capacity, runs, cudaMemcpyDeviceToHost, stream);
        }
    }
}

void StateImageDevicePool::copy_from_host(HostStateImageConstView source, std::int32_t destination,
                                          cudaStream_t stream,
                                          std::optional<std::uint64_t> frontier) {
    validate_slot(destination, slot_count(),
                  "StateImage copy-from-host destination is out of range");
    validate_host_layout(source.layout, source.data);
    const std::uint32_t linear_layers = linear_ ? linear_->layer_count() : 0;
    for (std::uint32_t layer = 0; layer < linear_layers; ++layer) {
        const Tensor conv = linear_->conv_slot(layer, destination);
        CUDA_CHECK(cudaMemcpyAsync(
            conv.data,
            byte_offset(source.data, host_layout_.linear_conv.offset +
                                         layer * host_layout_.linear_conv_layer_bytes),
            conv.bytes(), cudaMemcpyHostToDevice, stream));
        const Tensor recurrent = linear_->recurrent_slot(layer, destination);
        CUDA_CHECK(cudaMemcpyAsync(
            recurrent.data,
            byte_offset(source.data, host_layout_.linear_recurrent.offset +
                                         layer * host_layout_.linear_recurrent_layer_bytes),
            recurrent.bytes(), cudaMemcpyHostToDevice, stream));
    }
    const Tensor hidden = continuation_hidden_slot(destination);
    CUDA_CHECK(cudaMemcpyAsync(hidden.data,
                               byte_offset(source.data, host_layout_.continuation_hidden.offset),
                               hidden.bytes(), cudaMemcpyHostToDevice, stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, destination, 1);
            const Tensor v                    = view.v.slice(3, destination, 1);
            CUDA_CHECK(cudaMemcpyAsync(
                k.data,
                byte_offset(source.data, host_layout_.dflash_local_k->offset +
                                             layer * host_layout_.dflash_local_layer_bytes),
                k.bytes(), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                v.data,
                byte_offset(source.data, host_layout_.dflash_local_v->offset +
                                             layer * host_layout_.dflash_local_layer_bytes),
                v.bytes(), cudaMemcpyHostToDevice, stream));
        }
    }
    for (std::size_t group = 0; group < rings_.size(); ++group) {
        const RingGroup& ring        = rings_[group];
        const KvRingHostLayout& host = host_layout_.rings[group];
        const std::vector<RowRun> runs = live_rows(ring.spec, frontier);
        const std::uint32_t capacity   = ring.spec.capacity;
        for (std::size_t layer = 0; layer < ring.k.size(); ++layer) {
            const Tensor k = ring.k[layer].slice(3, destination, 1);
            const Tensor v = ring.v[layer].slice(3, destination, 1);
            copy_ring_rows(k.data, byte_offset(source.data, host.k.offset + layer * host.layer_bytes),
                           k, capacity, runs, cudaMemcpyHostToDevice, stream);
            copy_ring_rows(v.data, byte_offset(source.data, host.v.offset + layer * host.layer_bytes),
                           v, capacity, runs, cudaMemcpyHostToDevice, stream);
            if (!ring.spec.fp8) { continue; }
            const Tensor k_scale = ring.k_scale[layer].slice(3, destination, 1);
            const Tensor v_scale = ring.v_scale[layer].slice(3, destination, 1);
            copy_ring_rows(
                k_scale.data,
                byte_offset(source.data, host.k_scale.offset + layer * host.scale_layer_bytes),
                k_scale, capacity, runs, cudaMemcpyHostToDevice, stream);
            copy_ring_rows(
                v_scale.data,
                byte_offset(source.data, host.v_scale.offset + layer * host.scale_layer_bytes),
                v_scale, capacity, runs, cudaMemcpyHostToDevice, stream);
        }
    }
}

} // namespace ninferno::models::qwen3_5
