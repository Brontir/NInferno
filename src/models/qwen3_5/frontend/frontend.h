#pragma once

#include "ninferno/types.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/vision_style.h"
#include "models/registry.h"
#include "runtime/contract/request.h"
#include "structured/vocabulary.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninferno::models::qwen3_5 {

[[nodiscard]] ModelSamplingDefaults default_sampling(Architecture architecture);

// The sampling a checkpoint recommends in its generation_config.json (temperature, top_k, top_p,
// min_p), for both modes; nullopt when it names none of them. A top_k above the sampler's
// candidate domain (kMaximumTopK) keeps the whole domain.
[[nodiscard]] std::optional<ModelSamplingDefaults>
generation_sampling(std::string_view generation_config_json);

// The sampling defaults of a checkpoint outside the Qwen 3.5 family: what its generation_config.json
// recommends (generation_sampling), else transformers' own defaults, which a generation config that
// names no sampling parameter (gpt-oss) leaves in force: temperature 1, top_k 50, top_p 1, min_p 0,
// no penalties. default_sampling is Qwen 3.5's own
// recommendation (a presence penalty of 1.5 without thinking) and says nothing of another model.
[[nodiscard]] ModelSamplingDefaults
checkpoint_sampling(std::string_view generation_config_json);

// What the frontend requires of tokenizer_config.json. The Qwen artifacts pin their prefix
// semantics; a Hugging Face chat template renders its own special tokens, so the prompt is encoded
// without added BOS/EOS whatever the config says.
enum class TokenizerConfigPolicy : std::uint8_t {
    QwenPrefixSemantics,
    ChatTemplateOwnsSpecialTokens,
};

struct FrontendOptions {
    std::filesystem::path chat_template_path;
    Architecture architecture              = Architecture::Qwen3_5;
    // Sampling defaults from the artifact's generation config; unset uses default_sampling.
    std::optional<ModelSamplingDefaults> sampling;
    TokenizerConfigPolicy tokenizer_config = TokenizerConfigPolicy::QwenPrefixSemantics;
    bool vision_enabled                    = true;
    // Whether the artifact has a Vision component at all: a text-only artifact refuses media as
    // "no vision component" rather than as a disabled option.
    bool vision_component                  = true;
    // The prompt layout of the model's Vision encoder (prepared_prompt.h).
    VisionStyle vision_style               = VisionStyle::Qwen;
    std::uint32_t max_context              = 2'048;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
};

struct FrontendResources;
struct PreparedPromptData;
class Frontend;
class FrontendTestAccess;
class PreparedPromptAccess;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class FrontendTestAccess;
    friend class PreparedPromptAccess;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    // Whether a request on this prompt may be preempted and resumed: a text prompt (a media one
    // would keep its encoded payloads alive for the whole answer).
    [[nodiscard]] static bool resumable(const PreparedPrompt& prompt);
    // A copy of a prepared prompt, kept by a request that may be preempted and resumed.
    [[nodiscard]] static PreparedPrompt copy(const PreparedPrompt& prompt);
    // The prompt followed by the tokens the model generated for it, as text: a request preempted
    // mid-answer resumes from it, the cached endpoint covering all but its last token.
    [[nodiscard]] static PreparedPrompt continued(const PreparedPrompt& prompt,
                                                  std::span<const TokenId> generated);
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output            = {},
                        const ThinkingControlOptions& thinking = {},
                        bool quoted_reasoning_close_is_text    = false) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;
    // What the decision fingerprint calls this Frontend's chat template: "artifact", or the
    // --chat-template file named by the bytes it compiled ("<file name> sha256:<hex>").
    [[nodiscard]] const std::string& chat_template_source() const noexcept;
    // Constrained decoding: presented bytes and control-token flags of every token id in
    // [0, token_domain), the default stop tokens and the reasoning-section terminator.
    [[nodiscard]] structured::GrammarVocabulary grammar_vocabulary(std::uint32_t token_domain) const;
    // Structural-tag grammar for this model's tool-call markup (see ToolCallRequirement).
    [[nodiscard]] std::string
    tool_call_structural_tag(std::span<const ToolCallConstraintTool> tools,
                             ToolCallRequirement requirement, std::string_view named_tool,
                             bool parallel_tool_calls) const;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

// The families whose artifacts carry Hugging Face tokenizer files and a chat template (the dense
// decoders, Gemma 4; runtime/engine) serve through the same frontend: the checkpoint's
// tokenizer.json read by text::HfTokenCodec, the chat template owning the special tokens (no
// Qwen tokenizer_config.json semantics), text only. The chat-template parity test
// (tests/serve/test_chat_template_parity.cpp) builds its frontend with these two as well.
// Sets resources.tokenizer and resources.public_token_count (the tokenizer's whole vocabulary).
void attach_hf_text_tokenizer(FrontendResources& resources);
// Sets the tokenizer_config policy and turns vision off.
void apply_hf_text_options(FrontendOptions& options);

} // namespace ninferno::models::qwen3_5
