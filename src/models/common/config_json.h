#pragma once

// Readers for the fields of a component's JSON config, shared by every model family. Each one
// validates what it returns; every failure is an ArtifactError that names the field.

#include "artifact/schema.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::models {

// A nonnegative integer (positive when `positive`) that fits 32 bits.
[[nodiscard]] std::uint32_t json_u32(const artifact::Json& value, std::string_view label,
                                     bool positive = true);

// A present, positive, 32-bit integer member `name` of `config`.
[[nodiscard]] std::uint32_t json_dimension(const artifact::Json& config, const char* name);

// json_dimension that also fits a signed 32-bit field (the families' int32 sizes): a larger value
// would turn negative there.
[[nodiscard]] std::int32_t json_dimension_i32(const artifact::Json& config, const char* name);

// A present, positive, finite member `name` of `config`, as FP32.
[[nodiscard]] float json_positive_float(const artifact::Json& config, const char* name);

// A nonempty array of distinct layer indices below `layers` (DFlash target taps).
[[nodiscard]] std::vector<std::uint32_t>
json_layer_indices(const artifact::Json& value, std::uint32_t layers, std::string_view label);

// Every id distinct and in [0, domain): a shortlist head's row-to-token table.
void validate_token_ids(std::span<const std::int32_t> ids, std::uint32_t domain,
                        std::string_view label);

} // namespace ninferno::models
