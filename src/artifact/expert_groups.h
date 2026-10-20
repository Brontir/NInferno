#pragma once

// The routed experts of an artifact, by MoE layer, for MoE expert offload
// (core/expert_residency.h). A binding names a routed expert when a "/experts/" segment follows
// a "layers/<i>/" one ("text/layers/3/mlp/experts/7/gate_up", "text/layers/3/moe/experts/7/down",
// the stacked "text/layers/3/mlp/experts/gate_up_bias"): the artifact's naming of routed experts in
// every MoE family. An object is a layer's expert object when every binding and Use auxiliary
// that reads it names a routed expert of that one layer; an object shared with anything else
// stays where the plan put it.

#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::artifact {

// "text/layers/3" for a routed-expert binding name, nullopt for any other.
[[nodiscard]] std::optional<std::string> routed_expert_layer(std::string_view binding);

// Per directory object, the layer whose expert object it is (nullopt for any other object).
[[nodiscard]] std::vector<std::optional<std::string>>
routed_expert_object_layers(const Directory& directory);

// The plan's device objects that are routed-expert objects, one group per MoE layer, ordered by
// component (the "text" component first, the others by name) and layer index. Offsets are
// relative to the group, each object aligned as the plan aligned it.
[[nodiscard]] std::vector<ExpertHostGroup> routed_expert_groups(const Reader& reader,
                                                                const MaterializationPlan& plan);

// Moves the first `count` groups of routed_expert_groups (every group when nullopt) out of the
// device arena into plan.expert_groups, re-laying the remaining device objects. Returns the number
// of MoE layers the artifact has. Throws when `count` exceeds it, or is set and the artifact has
// no routed experts.
std::size_t place_expert_groups(MaterializationPlan& plan, const Reader& reader,
                                std::optional<std::size_t> count);

// place_expert_groups for an Engine's option: nothing when Off; when Automatic, the leading MoE
// layers whose experts together fit pinned_host_limit (every layer on a host with the memory;
// nothing for a model without routed experts), refused with the numbers when not even one does;
// the first `layers` when Layers.
void place_expert_groups(MaterializationPlan& plan, const Reader& reader,
                         const MoeOffloadOptions& options);

} // namespace ninferno::artifact
