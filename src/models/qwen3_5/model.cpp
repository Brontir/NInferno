#include "models/qwen3_5/model.h"

#include "models/common/weight_use.h"

#include <stdexcept>
#include <utility>

namespace ninferno::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing, DeviceBuffer yarn_frequencies)
    : backing_(std::move(backing)), yarn_frequencies_(std::move(yarn_frequencies)),
      config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), resources_(std::move(resources)),
      info_(std::move(info)) {}

Model::~Model() = default;

ops::WeightInput bound_weight_input(const BoundWeight& parameter, std::size_t use_index,
                                    const LoadOptions& options) {
    const auto& use = parameter.uses.at(use_index);
    // --activations int8 leaves a head or embedding recorded A16Only at BF16 activations (the
    // logits); one recorded for four bits is held at eight like every other linear.
    const bool head = parameter.name.find("head") != std::string::npos ||
                      parameter.name.find("embed") != std::string::npos;
    const auto mode = head && options.activations == ActivationMode::Int8 &&
                              use.policy == ops::LinearPolicy::A16Only
                          ? ActivationMode::Fast
                          : options.activations;
    return {parameter.view, effective_policy(use.policy, mode), use.activation_input_divisor};
}

ops::WeightInput bound_weight_input(const BoundWeight& parameter, const LoadOptions& options) {
    if (parameter.uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return bound_weight_input(parameter, 0, options);
}

ops::WeightInput Model::input(WeightUseId id) const {
    return bound_weight_input(weight(id.parameter), id.use_index, options_);
}

ops::WeightInput Model::input(WeightId id) const {
    return bound_weight_input(weight(id), options_);
}

} // namespace ninferno::models::qwen3_5
