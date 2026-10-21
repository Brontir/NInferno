#include "models/common/loading.h"

#include "artifact/views.h"
#include "core/weight.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninferno::models::loading {

const artifact::Json& open_text_component(const artifact::Reader& reader, const char* who) {
    const auto& directory = reader.directory();
    const auto text       = directory.components.find("text");
    if (text == directory.components.end()) {
        throw artifact::ArtifactError(std::string(who) + ": artifact has no text component");
    }
    return text->second.config;
}

Id Demands::matrix(const std::string& name, artifact::Shape shape, const std::string& input) {
    Pending pending{binder_.parameter(name, std::move(shape), artifact::Residency::Device), {}};
    pending.use = read_weight_use(binder_, name, input);
    items_.push_back(std::move(pending));
    return items_.size() - 1;
}

Id Demands::direct(const std::string& name, artifact::Shape shape,
                   std::optional<QType> format) {
    items_.push_back(
        {binder_.parameter(name, std::move(shape), artifact::Residency::Device, format), {}});
    return items_.size() - 1;
}

Id Demands::mapped(const std::string& name, artifact::Shape shape,
                   std::optional<QType> format) {
    items_.push_back(
        {binder_.parameter(name, std::move(shape), artifact::Residency::Mapped, format), {}});
    return items_.size() - 1;
}

float Demands::scalar(const std::string& name) {
    const auto reference = binder_.parameter(name, {1}, artifact::Residency::Values, QType::FP32);
    return binder_.values(reference.binding, QType::FP32).scalar_f32();
}

artifact::HostValues Demands::values(Id id, QType format) {
    return binder_.values(items_.at(id).reference.binding, format);
}

std::unique_ptr<ArtifactBacking> Demands::materialize(DeviceContext& device) {
    // Every binding of a component this model loads must be bound.
    auto plan = std::move(binder_).finish(artifact::UnusedBindings::RefuseInLoadedComponents);
    auto backing          = std::make_unique<ArtifactBacking>();
    backing->materialized = artifact::materialize(reader_, std::move(plan), device);
    for (const auto& item : items_) {
        backing->views.push_back(artifact::bind_view(item.reference, backing->materialized));
    }
    return backing;
}

ops::WeightInput Prepare::input(Id id) const {
    const auto& use = items_.at(id).use;
    if (!use) {
        throw std::invalid_argument(items_.at(id).reference.name + " has no mathematical use");
    }
    const auto policy = recorded_ ? use->policy : ops::LinearPolicy::A16Only;
    return {views_.at(id), policy, use->activation_input_divisor};
}

ops::WeightInput Prepare::recorded_input(Id id) const {
    const auto& use = items_.at(id).use;
    if (!use) {
        throw std::invalid_argument(items_.at(id).reference.name + " has no mathematical use");
    }
    return {views_.at(id), use->policy, use->activation_input_divisor};
}

Tensor Prepare::tensor(Id id) const {
    const auto& v = views_.at(id);
    if (v.shape.size() != 1) {
        throw std::invalid_argument(items_.at(id).reference.name + ": expected a vector");
    }
    return weight_tensor(v, {static_cast<std::int32_t>(v.shape[0])});
}

Projection Prepare::projection(Id id) const {
    const auto prepared = ops::prepare_linear_weight(input(id));
    return {prepared.weight, prepared.policy};
}

std::optional<Projection> Prepare::packed(std::span<const Id> ids) const {
    std::vector<ops::WeightInput> rows;
    std::optional<float> common;
    for (const Id id : ids) {
        rows.push_back(input(id));
        if (rows.back().activation_input_divisor) {
            common = std::max(common.value_or(0.0F), *rows.back().activation_input_divisor);
        }
    }
    for (auto& row : rows) {
        if (row.activation_input_divisor) { row.activation_input_divisor = common; }
    }
    WeightView joined;
    for (const Id id : ids) {
        joined.parts.insert(joined.parts.end(), view(id).parts.begin(), view(id).parts.end());
    }
    const auto* parent = joined.parts.front().parent;
    std::uint64_t end  = joined.parts.front().begin;
    for (const auto& part : joined.parts) {
        if (part.parent != parent || part.begin != end) { return std::nullopt; }
        end = part.end;
    }
    if (joined.parts.front().begin != 0 || end != parent->geometry.elements) {
        return std::nullopt;
    }
    const auto prepared = ops::prepare_linear_weight(rows);
    return Projection{prepared.weight, prepared.policy};
}

} // namespace ninferno::models::loading
