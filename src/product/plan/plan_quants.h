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
};

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
    std::string why_unsupported; // "" when supported
    PlannedSpeculation planned;  // empty plan when unsupported
    runtime::QuantFit fit;
    bool largest = false; // the rule's pick
};

struct QuantPlan {
    std::vector<QuantRow> rows; // the listing's order (drafters and sidecars left out)
    std::optional<std::size_t> chosen;
    // "UD-Q6_K_XL: the largest that fits 32,768 x 1 on RTX 4090 (24 GiB class)" or why none.
    std::string rule;
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

// The rule's line ("<quant>: the largest that fits <ctx> x <C> on <device>"), and the line when
// nothing fits (what fits instead, from the smallest quantization's plan).
[[nodiscard]] std::string quant_rule_line(const QuantPlan& plan, const DryRunInputs& inputs);

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
