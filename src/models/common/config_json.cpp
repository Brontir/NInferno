#include "models/common/config_json.h"

#include <cmath>
#include <limits>
#include <vector>

namespace ninferno::models {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t json_u32(const Json& value, std::string_view label, bool positive) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t json_dimension(const Json& config, const char* name) {
    if (!config.contains(name)) { throw ArtifactError(std::string(name) + " is missing"); }
    return json_u32(config.at(name), name);
}

std::int32_t json_dimension_i32(const Json& config, const char* name) {
    const std::uint32_t value = json_dimension(config, name);
    if (value > static_cast<std::uint32_t>(INT32_MAX)) {
        throw ArtifactError(std::string(name) + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

float json_positive_float(const Json& config, const char* name) {
    if (!config.contains(name)) { throw ArtifactError(std::string(name) + " is missing"); }
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

std::vector<std::uint32_t> json_layer_indices(const Json& value, std::uint32_t layers,
                                              std::string_view label) {
    if (!value.is_array() || value.empty()) {
        throw ArtifactError(std::string(label) + " must be nonempty");
    }
    std::vector<std::uint32_t> out;
    std::vector<bool> seen(layers);
    for (const auto& item : value) {
        const auto index = json_u32(item, label, false);
        if (index >= layers || seen[index]) {
            throw ArtifactError(std::string(label) + ": invalid or repeated layer index");
        }
        seen[index] = true;
        out.push_back(index);
    }
    return out;
}

void validate_token_ids(std::span<const std::int32_t> ids, std::uint32_t domain,
                        std::string_view label) {
    std::vector<bool> seen(domain);
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::uint32_t>(id) >= domain ||
            seen[static_cast<std::size_t>(id)]) {
            throw ArtifactError(std::string(label) + " must be unique and inside the domain");
        }
        seen[static_cast<std::size_t>(id)] = true;
    }
}

} // namespace ninferno::models
