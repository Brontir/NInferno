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

ops::WeightInput Model::input(WeightUseId id) const {
    const auto& parameter = weight(id.parameter);
    const auto& use       = parameter.uses.at(id.use_index);
    // --activations 8|4 leave a head or embedding recorded A16Only at BF16 activations (the
    // logits); one recorded for four bits is held at eight like every other linear under 8.
    const auto policy =
        effective_policy(matrix_use(parameter.name, parameter.view,
                                    {.policy                   = use.policy,
                                     .activation_input_divisor = use.activation_input_divisor}),
                         options_.activations);
    return {parameter.view, policy, use.activation_input_divisor};
}

ops::WeightInput Model::input(WeightId id) const {
    if (weight(id).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0});
}

} // namespace ninferno::models::qwen3_5
