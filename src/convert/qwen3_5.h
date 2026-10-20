#pragma once

// Qwen3.5 architecture adapter and its official recipes, mirroring tools/convert/qwen3_5.py and
// the Qwen part of official_recipes.py (which stay the source of truth):
//   Qwen3_5ForCausalLM / Qwen3_5ForConditionalGeneration        (dense text)
//   Qwen3_5MoeForCausalLM / Qwen3_5MoeForConditionalGeneration  (MoE text)
// with the optional Vision, MTP, DFlash and DFlash2 components.

#include "convert/checks.h"
#include "convert/model.h"

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::convert {

// Whether `architecture` is one of the Qwen3.5 text architectures above.
[[nodiscard]] bool is_qwen3_5_architecture(std::string_view architecture) noexcept;

// qwen3_5.build_model: `components` as --components lists them; `companions` the dflash and
// dflash2 checkpoints the selected drafter components read.
[[nodiscard]] Model build_qwen3_5_model(const Checkpoint& base,
                                        const std::vector<std::string>& components,
                                        const std::map<std::string, const Checkpoint*>& companions,
                                        const ResourceOverrides& overrides);

[[nodiscard]] bool is_qwen3_5_recipe(std::string_view name) noexcept;
void apply_qwen3_5_recipe(std::string_view name, Model& model, Recipe& recipe,
                          SourceInputs& sources);

} // namespace ninferno::convert
