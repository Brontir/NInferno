#include "models/common/weight_use.h"

#include "artifact/views.h"

#include <cmath>

namespace ninferno::models {

namespace {
using ops::LinearPolicy;
constexpr MatrixUse kA4{.recorded = LinearPolicy::AllowA4};
constexpr MatrixUse kA16{.recorded = LinearPolicy::A16Only};
constexpr MatrixUse kHead{.recorded = LinearPolicy::A16Only, .logits_or_embedding = true};
constexpr MatrixUse kNoDivisor{.recorded = LinearPolicy::A16Only, .nvfp4_without_divisor = true};
} // namespace

static_assert(effective_policy(kA4, ActivationMode::Bits16) == LinearPolicy::A16Only);
static_assert(effective_policy(kA4, ActivationMode::Auto) == LinearPolicy::AllowA4);
static_assert(effective_policy(kA16, ActivationMode::Auto) == LinearPolicy::A16Only);
static_assert(effective_policy(kA16, ActivationMode::Bits8) == LinearPolicy::AllowA8);
static_assert(effective_policy(kA4, ActivationMode::Bits8) == LinearPolicy::AllowA8);
static_assert(effective_policy(kA16, ActivationMode::Bits4) == LinearPolicy::AllowA4);
static_assert(effective_policy(kHead, ActivationMode::Bits8) == LinearPolicy::A16Only);
static_assert(effective_policy(kHead, ActivationMode::Bits4) == LinearPolicy::A16Only);
static_assert(effective_policy(kNoDivisor, ActivationMode::Bits4) == LinearPolicy::AllowA8);

MatrixUse matrix_use(std::string_view name, const WeightView& view, const WeightUsePolicy& use) {
    bool nvfp4 = false;
    for (const auto& part : view.parts) {
        nvfp4 = nvfp4 || (part.parent != nullptr && part.parent->geometry.format == QType::NVFP4);
    }
    return {.recorded              = use.policy,
            .logits_or_embedding   = logits_or_embedding(name),
            .nvfp4_without_divisor = nvfp4 && !use.activation_input_divisor.has_value()};
}

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
