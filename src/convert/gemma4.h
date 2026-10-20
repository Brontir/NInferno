#pragma once

// Gemma 4 architecture adapters and official recipes, stated explicitly per architecture:
//   Gemma4ForConditionalGeneration / Gemma4ForCausalLM  (text: 26B-A4B MoE and 31B dense)
//   Gemma4UnifiedForConditionalGeneration               (text of the unified 12B)
//   Gemma4AssistantForCausalLM / Gemma4UnifiedAssistantForCausalLM (Google's MTP drafter)
//   DFlashDraftModel                                    (z-lab's block-diffusion drafter)
//   MuseGlimmerAssistantModel, a GGUF DFlash drafter    (the same design, block mask everywhere)
// Each mirrors its Python counterpart (tools/convert/gemma4*.py, official_recipes.py), which
// stays the source of truth.

#include "convert/checks.h"
#include "convert/model.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::convert {

enum class Architecture { Gemma4Text, Gemma4Assistant, Gemma4DFlash };

// The checkpoint's single declared architecture; throws for anything else.
[[nodiscard]] Architecture architecture_of(const OJson& config);

// `vision`: the checkpoint that holds the vision tower (the model's own, or a projector file
// beside its GGUF); null for the text model alone.
[[nodiscard]] Model build_model(const Checkpoint& base, const ResourceOverrides& overrides,
                                const Checkpoint* vision = nullptr);

// gemma4_assistant.restrict_head: keep the `rows` most frequent tokens of the drafter's head.
// `ranking_text` is the path as the report and provenance spell it.
void restrict_head(Model& model, const std::filesystem::path& ranking,
                   const std::string& ranking_text, std::uint64_t rows);

void apply_recipe(std::string_view name, Model& model, Recipe& recipe, const Checkpoint& base);

// The official recipe a checkpoint converts with by default (engine auto-conversion), from its
// config.json alone.
[[nodiscard]] std::string default_recipe(const OJson& config);

} // namespace ninferno::convert
