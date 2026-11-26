#pragma once

// Inkling 2 text decoder (Thinking Machines' Inkling2LoopForConditionalGeneration, read from
// llama.cpp's inkling2_loop GGUF by gguf_model.cpp map_inkling2). The mathematics and the
// parameter names are docs/maintainer/inkling2-model.md's.

#include "convert/checks.h"
#include "convert/model.h"

#include <string>
#include <string_view>

namespace ninferno::convert {

[[nodiscard]] bool is_inkling2_architecture(std::string_view architecture) noexcept;

[[nodiscard]] Model build_inkling2_model(const Checkpoint& base,
                                         const ResourceOverrides& overrides);

[[nodiscard]] bool is_inkling2_recipe(std::string_view name) noexcept;

// inkling2_published: every tensor as published (Q8_0 projections and head, NVFP4 experts with
// their per-expert scales, BF16 sink and tap projections, FP32 vectors and tables).
void apply_inkling2_recipe(std::string_view name, Model& model, Recipe& recipe,
                           const Checkpoint& base);

[[nodiscard]] std::string default_inkling2_recipe(const OJson& config);

} // namespace ninferno::convert
