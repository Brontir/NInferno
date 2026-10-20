#pragma once

// The dense decoder adapter: every Llama-lineage checkpoint the dense family runs, resolved into
// one config (model_type "dense_decoder", src/models/dense/config.h) and one parameter layout.
//
//   LlamaForCausalLM, MistralForCausalLM, Ministral3ForCausalLM, Mistral3ForConditionalGeneration
//   (the text of Ministral 3 / Mistral Small 3.x / Devstral 2 / Magistral), GraniteForCausalLM,
//   Olmo2ForCausalLM, Olmo3ForCausalLM, SeedOssForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM,
//   SmolLM3ForCausalLM
//
// What differs between them is data in the resolved config: norm placement, q/k norm scope,
// biases, attention scale, the muP multipliers (the embedding and logit ones folded into the
// tables here), the Llama 4 query temperature, sliding-window and NoPE layer patterns, and each
// layer's rotary frequency table computed here from the checkpoint's rope type (default, linear,
// llama3, yarn) exactly as transformers computes it. Anything the family cannot represent is
// refused with the field that carries it.
//
// Recipes: dense_encoded (keeps the checkpoint's NVFP4 or FP8 projections, Q8 elsewhere),
// dense_nvfp4 (keeps them too, and rounds every other projection to NVFP4: weight-only, BF16
// activations), dense_fp8 (row-scaled FP8 with FP8 activations allowed), dense_q8 (every projection
// and the vocabulary in Q8), dense_bf16 (unquantized). dense_nvfp4, dense_fp8 and dense_q8 quantize
// an unquantized checkpoint themselves: maintainer recipes (ninferno-convert
// --maintainer-quantize).

#include "convert/checks.h"
#include "convert/model.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::convert {

[[nodiscard]] bool is_dense_architecture(std::string_view architecture) noexcept;

// The checkpoint architectures (config.json "architectures") the dense, MoE and Nemotron-H
// converters read, in their table's order.
[[nodiscard]] std::vector<std::string_view> dense_architectures();

// The resolved dense_decoder config of a checkpoint's config.json (throws naming what it cannot
// represent). checkpoint_has_head answers a tie_word_embeddings the config leaves unset.
[[nodiscard]] OJson dense_text_config(const OJson& source, bool checkpoint_has_head = true);

[[nodiscard]] Model build_dense_model(const Checkpoint& base, const ResourceOverrides& overrides);

// The rotary embedding a checkpoint's config.json declares (rope_parameters, or the legacy
// rope_theta + rope_scaling pair) over `dim` features of every layer, as transformers computes it:
// {"rope_type", "rope_theta", "rotary_dim", "attention_scaling", "inv_freq"}. A DFlash drafter's
// (gpt-oss's YaRN). A rope whose rotation depends on the context (LongRoPE) is refused.
[[nodiscard]] OJson resolved_rope(const OJson& source, std::uint64_t dim,
                                  std::uint64_t max_positions, const std::string& where);

[[nodiscard]] bool is_dense_recipe(std::string_view name) noexcept;
// Applies the recipe with its vocabulary (token embedding and output head) policy: "native"
// keeps them as the checkpoint publishes them (BF16 stays BF16, a GGUF's Q8_0 stays Q8_0), "q8"
// quantizes an unquantized one to q8_g32_fp16; empty takes default_vocabulary(name). Returns the
// policy applied.
std::string apply_dense_recipe(std::string_view name, Model& model, Recipe& recipe,
                               const Checkpoint& base, std::string_view vocabulary = {});

// q8 for the recipes that quantize the model themselves (dense_q8, dense_fp8, dense_nvfp4,
// moe_q8), native for every recipe that runs a checkpoint as published.
[[nodiscard]] std::string default_vocabulary(std::string_view recipe);

// The default recipe: moe_published for a MoE decoder, else dense_encoded for a checkpoint with a
// published quantization (`quantized`: judged by its tensors, or by its quantization_config when
// only config.json is left) and dense_bf16 for one without.
[[nodiscard]] std::string default_dense_recipe(const OJson& config, bool quantized);

} // namespace ninferno::convert
