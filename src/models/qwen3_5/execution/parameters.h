#pragma once

#include "models/qwen3_5/execution/parameter_types.h"
#include "models/qwen3_5/model.h"
#include "ninferno/ops/weight_input.h"

#include <optional>

namespace ninferno::models::qwen3_5::execution {

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
// The forms are prepare_parameters' (prepare.h), which a plan without the weights prepares too.
class Parameters : public PreparedParameters {
public:
    explicit Parameters(const Model& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
};

} // namespace ninferno::models::qwen3_5::execution
