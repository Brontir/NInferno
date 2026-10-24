#pragma once

// Whether NInferno runs a model, and why not: one decision from what a repository shows before
// its weights are downloaded (the hub's model card, config.json, the GGUF header). The converter
// refuses with it before a download and the plan without weights before it plans; the hub search
// shows it.

#include "convert/fetch.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::convert {

// What a repository shows of a model before its weights are downloaded. Each fact is optional: a
// rule whose fact is unknown is not applied (the search's answer carries fewer than a repository's
// own files).
struct SupportFacts {
    std::string pipeline;     // the hub's pipeline_tag ("" when unsaid)
    std::string library;      // the hub's library_name or tag ("mlx", "transformers", ...)
    bool needs_token = false; // the hub refused the model's files without a token (gated)
    // config.json: the whole file (`full_config`) or the hub's part of it (architectures,
    // model_type, quantization_config, tokenizer_config); null for a GGUF repository.
    nlohmann::json config;
    bool full_config = false;
    std::vector<std::string> weight_dtypes; // safetensors dtypes the hub counted ("BF16", "U32")
    std::optional<bool> chat_template;      // whether a chat template ships (nullopt: unknown)
    bool gguf = false;                      // the model is a GGUF file
    std::string gguf_architecture;          // its general.architecture
    std::string gguf_error;                 // why its header does not parse ("" when it does)
};

enum class Verdict { Supported, NeedsToken, Unsupported, Unknown };

struct Support {
    Verdict verdict = Verdict::Unknown;
    // What the repository holds: "model", "drafter" (runs beside its model, not alone),
    // "projector" (an image projector); "" when unknown.
    std::string role;
    std::string reason; // why it is not supported, or what was not checked ("" when nothing)
};

// The rules, in order: a task other than text generation; weights the converter does not read
// (MLX, bitsandbytes, ...); an architecture no engine runs (the converter's own lists); a gated
// repository without a token; config.json fields the converter does not represent (with the
// whole file); no chat template (a base model: the engine serves chat).
[[nodiscard]] Support model_support(const SupportFacts& facts);

// "supported", "needs_token", "unsupported", "unknown".
[[nodiscard]] std::string_view verdict_name(Verdict verdict);

// The facts of a hub repository (`spec` as fetch_model takes it, options.quant choosing a GGUF
// model): its model card, then config.json or the chosen GGUF model's header. nullopt when the
// hub is not asked (another hub than Hugging Face, $HF_HUB_OFFLINE).
[[nodiscard]] std::optional<SupportFacts> hub_support_facts(const std::string& spec,
                                                            const FetchOptions& options);

// The facts of a model on disk: a checkpoint directory (config.json, its tokenizer files and
// template, its safetensors' dtypes) or a GGUF file (its header).
[[nodiscard]] SupportFacts local_support_facts(const std::filesystem::path& path);

// The facts a hub search answer carries for one model (an /api/models item with config, gguf,
// safetensors, pipeline_tag, library_name and tags expanded).
[[nodiscard]] SupportFacts search_support_facts(const nlohmann::json& item);

// Why the converter would refuse config.json (its recipe choice and, for a dense decoder, the
// family's representation of it), "" when it would not.
[[nodiscard]] std::string config_refusal(const nlohmann::json& config);

} // namespace ninferno::convert
