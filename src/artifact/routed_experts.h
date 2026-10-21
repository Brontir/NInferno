#pragma once

// The routed experts of an artifact, by MoE layer. A binding names a routed expert when a
// "/experts/" segment follows a "layers/<i>/" one ("text/layers/3/mlp/experts/7/gate_up",
// "text/layers/3/moe/experts/7/down", the stacked "text/layers/3/mlp/experts/gate_up_bias"): the
// artifact's naming of routed experts in every MoE family. An object is a layer's expert object
// when every binding and Use auxiliary that reads it names a routed expert of that one layer.

#include "artifact/schema.h"

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

} // namespace ninferno::artifact
