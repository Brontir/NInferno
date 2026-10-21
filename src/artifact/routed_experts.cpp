#include "artifact/routed_experts.h"

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

} // namespace ninferno::artifact
