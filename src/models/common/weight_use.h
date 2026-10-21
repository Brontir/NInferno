#pragma once

// Pieces every model family's loader needs, with no family knowledge.

#include "artifact/binder.h"
#include "ninferno/ops/linear.h"
#include "ninferno/types.h"

#include <optional>
#include <string>
#include <string_view>

namespace ninferno::models {

// What an artifact records for one mathematical use of a matrix: the activation policy the
// linear Op may apply to that input, and the activation divisor a W4A4 route calibrated for it.
struct WeightUsePolicy {
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
};

// The policy a linear runs under: the recorded one under ActivationMode::Fast; under Exact every
// quantized linear keeps BF16 activations at every width; under Int8 every one is allowed eight
// bits and none four (ninferno/types.h).
[[nodiscard]] constexpr ops::LinearPolicy effective_policy(ops::LinearPolicy recorded,
                                                          ActivationMode mode) noexcept {
    if (mode == ActivationMode::Exact) { return ops::LinearPolicy::A16Only; }
    if (mode == ActivationMode::Int8) { return ops::LinearPolicy::AllowA8; }
    return recorded;
}

// Reads the Use of parameter `name` fed by `input`. A Use without a recorded policy, an unknown
// auxiliary, or a divisor that is not a positive finite FP32 is an ArtifactError naming
// `name@input`.
[[nodiscard]] WeightUsePolicy read_weight_use(artifact::Binder& binder, const std::string& name,
                                              std::string_view input);

} // namespace ninferno::models
