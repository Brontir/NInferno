#include "models/common/weight_use.h"

#include "artifact/views.h"

#include <cmath>

namespace ninferno::models {

static_assert(effective_policy(ops::LinearPolicy::AllowA4, ActivationMode::Exact) ==
              ops::LinearPolicy::A16Only);
static_assert(effective_policy(ops::LinearPolicy::AllowA8, ActivationMode::Exact) ==
              ops::LinearPolicy::A16Only);
static_assert(effective_policy(ops::LinearPolicy::AllowA4, ActivationMode::Fast) ==
              ops::LinearPolicy::AllowA4);
static_assert(effective_policy(ops::LinearPolicy::A16Only, ActivationMode::Int8) ==
              ops::LinearPolicy::AllowA8);
static_assert(effective_policy(ops::LinearPolicy::AllowA4, ActivationMode::Int8) ==
              ops::LinearPolicy::AllowA8);

WeightUsePolicy read_weight_use(artifact::Binder& binder, const std::string& name,
                                std::string_view input) {
    const std::string site = name + "@" + std::string(input);
    const auto& use        = binder.use(name, std::string(input));
    if (!use.activation_policy) {
        throw artifact::ArtifactError(site + ": missing activation policy");
    }
    WeightUsePolicy result;
    switch (*use.activation_policy) {
    case artifact::ActivationPolicy::A16Only:
        result.policy = ops::LinearPolicy::A16Only;
        break;
    case artifact::ActivationPolicy::AllowA8:
        result.policy = ops::LinearPolicy::AllowA8;
        break;
    case artifact::ActivationPolicy::AllowA4:
        result.policy = ops::LinearPolicy::AllowA4;
        break;
    }
    for (const auto& [role, binding] : use.auxiliaries) {
        if (role != "activation_input_divisor") {
            throw artifact::ArtifactError(site + ": unknown auxiliary " + role);
        }
        const auto value = binder.values(binding, QType::FP32).scalar_f32();
        if (!std::isfinite(value) || value <= 0) {
            throw artifact::ArtifactError(site +
                                          ": activation divisor must be positive finite FP32");
        }
        result.activation_input_divisor = value;
    }
    return result;
}

} // namespace ninferno::models
