#pragma once

// The largest quantization that fits (the owner's flow: the user picks a model, may type the
// context, and sees the largest quant that fits that context x streams on the card): every
// quantization of a model planned without its weights (product/plan/plan_speculation.h over
// runtime/engine/plan_dry_run.h), the rule runtime::largest_fitting_quant picks one, and the
// dry run's report as JSON ("ninferno-dry-plan/1", docs/reference/serving.md) and text. One code
// path for ninferno-serve --plan --no-load, the quant a hub GGUF repository is fetched at without
// --quant, and ninferno-ui's /api/plan and /api/quants.

#include "product/plan/plan_facts.h"
#include "product/plan/plan_speculation.h"
#include "runtime/engine/plan_dry_run.h"
#include "product/choice/choice.h"
#include "runtime/engine/quant_rule.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::product {

// One file group of a repository listing (convert::RepoQuant's fields the plan reads).
struct QuantListing {
    std::string quant;  // as --quant takes it ("" for safetensors weights)
    std::string format; // gguf | ninferno | safetensors
    std::uint64_t bytes = 0;
    int bits            = 0; // from the label, 0 when it says none
    bool drafter        = false;
    // A checkpoint converted at start (safetensors): its weights after the start's recipe
    // (checkpoint_converted_bytes), nullopt when not known before the start.
    std::optional<std::uint64_t> converted_bytes;
};

// A safetensors checkpoint's weights after the recipe a start converts it with (convert.cpp's
// default recipe choice): a recipe that keeps the published encodings (dense BF16, a published
// checkpoint) keeps its bytes; Qwen3.5/3.6 MoE BF16 takes qwen3_6_35b_a3b's (routed experts Q4
// gate/up, Q5 down, the rest Q8; declared bits per weight with their group scales). nullopt for a
// recipe whose size the plan does not know before the start.
[[nodiscard]] std::optional<std::uint64_t> checkpoint_converted_bytes(const nlohmann::json& config,
                                                                      std::uint64_t bytes);

// What a dry run plans with: the device, its profile key, the Engine options the server builds,
// the speculation asked for and the subject it is resolved for.
struct DryRunInputs {
    runtime::PlanDeviceFacts device;
    spec::GpuKey gpu;
    EngineOptions options;
    SpeculationAsk ask;
    const SpeculationSubject* subject = nullptr;
    std::uint64_t drafter_bytes       = 0; // a separate drafter's weights, when known
};

// The rule's ask from the inputs: the context (automatic or pinned), the streams, the model's own.
[[nodiscard]] runtime::QuantAsk quant_ask_of(const DryRunInputs& inputs,
                                             std::uint32_t native_context);

// The card's memory as sold: the driver's total rounded up to whole GiB (23.99 GiB: 24).
[[nodiscard]] std::uint32_t card_gb(const runtime::PlanDeviceFacts& device) noexcept;

// A plan's reason as a code and its numbers (docs/reference/serving.md): "" (fits),
// shape_unknown, weights_too_big, smallest_plan_short, survival_only.
struct PlanReasonCode {
    std::string code;
    nlohmann::json params = nlohmann::json::object();
};

[[nodiscard]] PlanReasonCode plan_reason_code(const runtime::DryRunPlan& plan);

// What ninferno-ui's launch asks a server for (LaunchRequest: --max-context, --max-concurrency,
// --kv-dtype, --spec, --activations exact), as ninferno-serve's parser and engine_options_of turn
// those flags into Engine options and a speculation ask (a CPU test checks the two agree).
struct LaunchAsk {
    std::uint32_t max_context     = 0; // 0: automatic
    std::uint32_t max_concurrency = 1;
    std::string kv_dtype;      // "" (auto) or bf16 | int8 | fp8 | nvfp4 | k8v4
    std::string spec = "auto"; // auto | none | mtp | dflash | dflash2
    bool exact       = false;  // --activations exact
};

struct LaunchPlanOptions {
    EngineOptions options;
    SpeculationAsk ask;
};

[[nodiscard]] LaunchPlanOptions launch_options(const LaunchAsk& launch);

struct QuantRow {
    QuantListing listing;
    runtime::QuantCandidate candidate;
    runtime::WeightBasis basis = runtime::WeightBasis::FileSize;
    bool converted             = false; // planned at a checkpoint's converted size
    std::uint64_t weight_bytes = 0;     // what the plan counts (the converted size when converted)
    std::string unsupported_code;       // "" | drafter_file | converted_size_unknown | above_8_bits
                                        // | nvfp4_card
    std::string why_unsupported;        // "" when supported
    PlannedSpeculation planned;         // empty plan when not planned
    runtime::QuantFit fit;
    // What a launch passes as --max-context: under an automatic context with several streams and
    // an own fit the streams would otherwise share, the own context; the asked one when pinned;
    // 0 (automatic) otherwise.
    std::uint32_t launch_context = 0;
    bool largest                 = false; // the rule's pick
    // The choice model's speed at the plan's storages (product/choice/choice.h): one stream of the
    // asked ones decoding at `estimate_depth` tokens in its window (a quarter of the context it
    // holds, at least 512: an empty context flatters decode), the prefill, and the seconds of a
    // typical request (choice::kRequestPrompt + kRequestAnswer tokens). Not set for a row not planned.
    std::optional<choice::Speed> estimate;
    std::uint32_t estimate_depth = 0;
    double request_seconds       = 0;
};

struct QuantPlan {
    std::vector<QuantRow> rows; // the listing's order (drafters and sidecars left out)
    std::optional<std::size_t> chosen;
    std::optional<runtime::QuantPick> pick;
    // "UD-Q6_K_XL: the largest that fits on this RTX 4090 (24 GB) at auto context (49,152 each)
    // x 1 stream" or why none (English, the CLI); `rule_code` and `rule_params` the same as data.
    std::string rule;
    std::string rule_code; // largest_fits | none_fits | none_served
    nlohmann::json rule_params = nlohmann::json::object();
};

// Every listed quantization of a model whose shape `shape` is (any quantization's header: the
// shape does not depend on it), planned as `inputs` ask, and the rule's pick. A row above 8 bits
// per weight, a drafter or sidecar, and a format the engine does not serve are listed unsupported.
[[nodiscard]] QuantPlan plan_quants(const PlanSourceFacts& shape,
                                    const std::vector<QuantListing>& listing,
                                    const DryRunInputs& inputs);

// One model at its own weights.
[[nodiscard]] PlannedSpeculation plan_model(const PlanSourceFacts& facts,
                                            const DryRunInputs& inputs);

// The rule's line ("<quant>: the largest that fits on this <device> (<N> GB) at <ctx> x <C>
// streams, ..."), and the line when nothing fits (what fits instead, from the smallest
// quantization's plan); fills plan.rule_code and plan.rule_params with the same as data.
[[nodiscard]] std::string quant_rule_line(QuantPlan& plan, const DryRunInputs& inputs);

// The longest context each of the inputs' streams holds on its own for a model at its weights
// (0: none): the served plan's when private, else the longest of the declared probe contexts
// (8K .. the model's own) whose plan at that pinned context keeps every stream's own.
[[nodiscard]] std::uint32_t own_context_of(const PlanSourceFacts& facts, const DryRunInputs& inputs,
                                           const PlannedSpeculation& served);

// JSON pieces of "ninferno-dry-plan/1" (docs/reference/serving.md, "The plan without the
// weights").
[[nodiscard]] nlohmann::json plan_json(const runtime::DryRunPlan& plan, bool ladder);
[[nodiscard]] nlohmann::json planned_json(const PlannedSpeculation& planned, bool ladder);
[[nodiscard]] nlohmann::json device_json(const runtime::PlanDeviceFacts& device,
                                         const runtime::DryRunPlan& plan);
[[nodiscard]] nlohmann::json model_json(const PlanSourceFacts& facts, const std::string& source,
                                        const std::string& kind);
[[nodiscard]] nlohmann::json ask_json(const EngineOptions& options, const SpeculationAsk& ask,
                                      bool kv_automatic);
[[nodiscard]] nlohmann::json quant_plan_json(const QuantPlan& plan);

// The whole document: `quants` empty for a single model (its plan alone).
[[nodiscard]] nlohmann::json dry_plan_document(const PlanSourceFacts& facts,
                                               const std::string& source, const std::string& kind,
                                               const DryRunInputs& inputs,
                                               const PlannedSpeculation& planned,
                                               const QuantPlan* quants);

// The same as text for a terminal: the quant table when there is one, the chosen plan's ladder
// with the rung marked, its speculation and the advice.
[[nodiscard]] std::string render_dry_plan_text(const PlanSourceFacts& facts,
                                               const std::string& source,
                                               const DryRunInputs& inputs,
                                               const PlannedSpeculation& planned,
                                               const QuantPlan* quants);

} // namespace ninferno::product
