#include "artifact/expert_groups.h"

#include "core/host_memory.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace ninferno::artifact {
namespace {

// Which layer's experts an object belongs to: unset (no reader seen yet), a layer, or none
// (read by something else, or by two layers).
struct Owner {
    bool seen = false;
    std::optional<std::string> layer;
};

void claim(Owner& owner, const std::optional<std::string>& layer) {
    if (!owner.seen) {
        owner = {true, layer};
    } else if (owner.layer != layer) {
        owner.layer.reset();
    }
}

// Sort key: the "text" component first, then components by name, then the layer index.
std::tuple<bool, std::string, std::uint64_t> order(const std::string& layer) {
    const auto slash  = layer.find('/');
    const auto digits = layer.rfind('/');
    return {layer.substr(0, slash) != "text", layer.substr(0, slash),
            std::stoull(layer.substr(digits + 1))};
}

} // namespace

std::optional<std::string> routed_expert_layer(std::string_view binding) {
    const auto experts = binding.find("/experts/");
    if (experts == std::string_view::npos) { return std::nullopt; }
    const auto layers = binding.rfind("layers/", experts);
    if (layers == std::string_view::npos || (layers != 0 && binding[layers - 1] != '/')) {
        return std::nullopt;
    }
    std::size_t end = layers + 7;
    while (end < binding.size() && binding[end] >= '0' && binding[end] <= '9') { ++end; }
    if (end == layers + 7 || end > experts || binding[end] != '/') { return std::nullopt; }
    return std::string(binding.substr(0, end));
}

std::vector<std::optional<std::string>> routed_expert_object_layers(const Directory& directory) {
    std::vector<Owner> owners(directory.objects.size());
    for (const auto& [name, binding] : directory.bindings) {
        const auto layer = routed_expert_layer(name);
        for (const auto& part : binding.parts) { claim(owners.at(part.object.index), layer); }
    }
    for (const auto& [key, use] : directory.uses) {
        const auto layer = routed_expert_layer(use.parameter);
        for (const auto& [role, binding] : use.auxiliaries) {
            for (const auto& part : binding.parts) { claim(owners.at(part.object.index), layer); }
        }
    }
    std::vector<std::optional<std::string>> out;
    out.reserve(owners.size());
    for (auto& owner : owners) { out.push_back(std::move(owner.layer)); }
    return out;
}

std::vector<ExpertHostGroup> routed_expert_groups(const Reader& reader,
                                                  const MaterializationPlan& plan) {
    const auto layers = routed_expert_object_layers(reader.directory());
    std::map<std::string, ExpertHostGroup> by_layer;
    for (const auto& placement : plan.device_objects) {
        const auto& layer = layers.at(placement.object.index);
        if (!layer) { continue; }
        auto& group = by_layer[*layer];
        group.name  = *layer;
        const auto offset = align_up(group.bytes, placement.alignment, "expert group offset");
        group.objects.push_back({placement.object, offset, placement.bytes, placement.alignment});
        group.bytes = checked_add(offset, placement.bytes, "expert group bytes");
    }
    std::vector<ExpertHostGroup> out;
    for (auto& [layer, group] : by_layer) { out.push_back(std::move(group)); }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return order(a.name) < order(b.name); });
    return out;
}

std::size_t place_expert_groups(MaterializationPlan& plan, const Reader& reader,
                                std::optional<std::size_t> count) {
    auto groups           = routed_expert_groups(reader, plan);
    const std::size_t all = groups.size();
    if (count && all == 0) {
        throw ArtifactError("MoE expert offload of " + std::to_string(*count) +
                            " layers: the model has no routed experts (it is not a MoE model)");
    }
    if (count && (*count == 0 || *count > all)) {
        throw ArtifactError("MoE expert offload of " + std::to_string(*count) +
                            " layers: the model has " + std::to_string(all) +
                            " MoE layers (offload 1 to " + std::to_string(all) + ", or auto)");
    }
    groups.resize(count.value_or(all));
    std::vector<bool> moved(plan.object_count, false);
    for (const auto& group : groups) {
        for (const auto& object : group.objects) { moved.at(object.object.index) = true; }
    }
    std::vector<DevicePlacement> kept;
    std::uint64_t capacity = 0;
    for (auto placement : plan.device_objects) {
        if (moved.at(placement.object.index)) { continue; }
        placement.offset = align_up(capacity, placement.alignment, "device offset");
        capacity         = checked_add(placement.offset, placement.bytes, "device capacity");
        kept.push_back(placement);
    }
    plan.device_objects        = std::move(kept);
    plan.device_capacity_bytes = capacity;
    plan.expert_groups         = std::move(groups);
    return all;
}

void place_expert_groups(MaterializationPlan& plan, const Reader& reader,
                         const MoeOffloadOptions& options) {
    switch (options.mode) {
    case MoeOffloadMode::Off:
        return;
    case MoeOffloadMode::Automatic: {
        // Every layer the host can hold: the leading layers whose experts fit what the host can
        // pin (the Engine keeps the fewest of them there, core/expert_residency.h); the rest stay
        // in the arena.
        const auto groups = routed_expert_groups(reader, plan);
        std::size_t count = groups.size();
        if (const std::optional<HostMemory> memory = read_host_memory(); memory && count != 0) {
            const std::uint64_t limit = pinned_host_limit(*memory);
            std::uint64_t held        = 0;
            count                     = 0;
            while (count < groups.size() && held + groups[count].bytes <= limit) {
                held += groups[count++].bytes;
            }
            if (count == 0) {
                require_pinned_host_fits(groups[0].bytes, *memory,
                                         "MoE expert offload (the experts of one MoE layer, " +
                                             groups[0].name + ")",
                                         "Free Host memory.");
            }
        }
        if (count != 0) { (void)place_expert_groups(plan, reader, std::optional<std::size_t>(count)); }
        return;
    }
    case MoeOffloadMode::Layers:
        (void)place_expert_groups(plan, reader, std::optional<std::size_t>(options.layers));
        return;
    }
}

} // namespace ninferno::artifact
