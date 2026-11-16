#include "models/qwen3_5/execution/parameters.h"

#include "models/qwen3_5/execution/prepare.h"

namespace ninferno::models::qwen3_5::execution {

Parameters::Parameters(const Model& source)
    : PreparedParameters(prepare_parameters(source, true)), model(source) {}

} // namespace ninferno::models::qwen3_5::execution
