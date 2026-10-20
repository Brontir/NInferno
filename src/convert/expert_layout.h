#pragma once

// The layout a routed-expert bank of a ggml block format is stored in: its published layout
// (ggml_blocks_v1 / ggml_planes_v1) or ggml_frag_v1 (core/ggml_frag.h), the same words with each
// row's code bytes in the order the grouped MoE kernel reads them. Both are lossless; the choice
// is speed only. The converter (planner.cpp) and ninferno-convert --relayout take it from here.

#include "core/weight.h"

#include <cstdint>
#include <string_view>

namespace ninferno::convert {

// --layout / --relayout: auto (ggml_frag_v1 where it measured not slower, frag_measured_faster),
// published (every bank as published), frag (every bank whose format has a map).
enum class ExpertLayout { Auto, Published, Frag };

[[nodiscard]] ExpertLayout parse_expert_layout(std::string_view name);
[[nodiscard]] std::string_view expert_layout_name(ExpertLayout choice) noexcept;

// Whether the grouped MoE kernel on ggml_frag_v1 measured not slower than on the published
// layout for a bank of `format` with K columns, at every call width (the auto rule).
[[nodiscard]] bool frag_measured_faster(QType format, std::uint64_t k) noexcept;

// The layout a routed-expert bank of `format` with K columns is stored in under `choice`; a
// format that is not a ggml block format keeps its default layout (`published`).
[[nodiscard]] QuantLayout expert_bank_layout(QType format, std::uint64_t k, ExpertLayout choice,
                                             QuantLayout published);

} // namespace ninferno::convert
