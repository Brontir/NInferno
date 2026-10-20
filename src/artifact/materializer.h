#pragma once

#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/expert_residency.h"
#include "core/weight_view.h"
#include "ninferno/types.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninferno::artifact {

class Reader;

struct DevicePlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
};

struct HostPlacement {
    ObjectHandle object;
    // Already-read resources move into final storage without invalidating their byte views.
    std::vector<std::byte> data;
};

// One MoE layer's routed-expert objects, materialized into one Host group of an ExpertResidency
// (artifact/expert_groups.h) instead of the device arena; offsets are relative to the group.
struct ExpertHostGroup {
    std::string name; // "text/layers/3"
    std::vector<DevicePlacement> objects;
    std::uint64_t bytes = 0;
};

struct MaterializationPlan {
    const Reader* source                = nullptr;
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t prior_read_bytes      = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::vector<DevicePlacement> device_objects;
    std::vector<HostPlacement> host_objects;
    std::vector<ExpertHostGroup> expert_groups;
    // Objects read into pinned host memory the device addresses in place (Residency::Mapped).
    std::vector<ObjectHandle> mapped_objects;
    // Bindings of components the model bound nothing of (Binder::finish).
    std::vector<std::string> unbound_bindings;
};

struct MaterializationStats {
    std::uint64_t file_bytes = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes  = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t retained_host_bytes   = 0;
    std::uint64_t mapped_host_bytes     = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::uint64_t peak_staging_bytes    = 0;
    std::size_t device_object_count     = 0;
    std::size_t host_object_count       = 0;
    std::uint64_t expert_host_bytes     = 0; // routed experts materialized to Host groups
    std::size_t expert_group_count      = 0;
    std::uint64_t expert_device_bytes   = 0; // routed experts left in the device arena
    std::size_t expert_device_layers    = 0;
    std::size_t mapped_object_count     = 0;
    double upload_seconds               = 0;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] bool has_device(ObjectHandle handle) const noexcept;

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

    // The routed experts materialized to Host groups (plan.expert_groups); nullptr when none.
    [[nodiscard]] ExpertResidency* expert_residency() const noexcept { return experts_.get(); }

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&, DeviceContext&,
                                            const StartupObserver*);

    struct ObjectStorage {
        std::optional<WeightParent> device;
        std::optional<WeightParent> host;
        std::vector<std::byte> host_data;
        std::optional<PinnedHostBuffer> mapped;
    };

    std::unique_ptr<DeviceArena> arena_;
    std::unique_ptr<ExpertResidency> experts_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               DeviceContext& device,
                                               const StartupObserver* startup_observer = nullptr);

} // namespace ninferno::artifact
