#pragma once

// Pieces every model family's loader needs, with no family knowledge.

#include "artifact/binder.h"
#include "core/weight_view.h"
#include "models/common/matrix_role.h"
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

// What the activation policy of one matrix's use depends on besides the mode: the recorded
// policy, whether it is a head or an embedding (its logits keep a recorded BF16 under 8 and 4), and
// whether it is an NVFP4 matrix without a calibrated activation divisor (no four-bit route can
// encode its input, so 4 holds it at eight bits, where its W4A8 route is).
struct MatrixUse {
    ops::LinearPolicy recorded = ops::LinearPolicy::A16Only;
    bool logits_or_embedding   = false;
    bool nvfp4_without_divisor = false;
};

// The policy a linear runs under (ninferno/types.h ActivationMode): the recorded one under Auto;
// A16Only under Bits16; under Bits8 AllowA8 and under Bits4 AllowA4 (AllowA8 for NVFP4 without a
// divisor), except a head or embedding recorded A16Only, which keeps it.
[[nodiscard]] constexpr ops::LinearPolicy effective_policy(const MatrixUse& use,
                                                           ActivationMode mode) noexcept {
    switch (mode) {
    case ActivationMode::Auto:
        return use.recorded;
    case ActivationMode::Bits16:
        return ops::LinearPolicy::A16Only;
    case ActivationMode::Bits8:
    case ActivationMode::Bits4:
        break;
    }
    if (use.logits_or_embedding && use.recorded == ops::LinearPolicy::A16Only) {
        return ops::LinearPolicy::A16Only;
    }
    return mode == ActivationMode::Bits4 && !use.nvfp4_without_divisor ? ops::LinearPolicy::AllowA4
                                                                       : ops::LinearPolicy::AllowA8;
}

// The MatrixUse of parameter `name` bound as `view` with the recorded `use`.
[[nodiscard]] MatrixUse matrix_use(std::string_view name, const WeightView& view,
                                   const WeightUsePolicy& use);

// Reads the Use of parameter `name` fed by `input`. A Use without a recorded policy, an unknown
// auxiliary, or a divisor that is not a positive finite FP32 is an ArtifactError naming
// `name@input`.
[[nodiscard]] WeightUsePolicy read_weight_use(artifact::Binder& binder, const std::string& name,
                                              std::string_view input);

} // namespace ninferno::models
