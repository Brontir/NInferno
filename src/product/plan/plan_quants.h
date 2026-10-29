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
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::convert {
struct FetchOptions;
} // namespace ninferno::convert

namespace ninferno::product {

// A GGUF quantization's own tensor table, its header read (plan_facts_of_gguf of an unsplit
// file): the bytes a start of it uploads and its NextN (MTP) layers' share, as the converter moves
// them (a converted artifact kept in a GGUF's place: its tensors and own drafter, as its own plan
// counts them). A row with one prices these, not the listed file size (header and padding).
struct GgufTensorTable {
    std::uint64_t tensor_bytes      = 0;
    std::uint64_t mtp_bytes         = 0;
    std::uint64_t own_drafter_bytes = 0; // a converted artifact's own drafter, inside tensor_bytes
    runtime::WeightBasis basis      = runtime::WeightBasis::GgufTensors; // Artifact for one
    // A GGUF table's own stored routed-expert banks (trunk, MTP layers) and token table: a mixed
    // quantization encodes them at other widths than its dense tensors, so the file's total does
    // not scale another row's (Codex on #202). Unset: scaled from the parsed row's.
    std::optional<std::uint64_t> routed_expert_bytes;
    std::optional<std::uint64_t> mtp_routed_expert_bytes;
    std::optional<std::uint64_t> token_table_bytes;
};
// The table from the facts of a head read (nullopt: a split part or no tensor table).
[[nodiscard]] std::optional<GgufTensorTable> gguf_tensor_table(const PlanSourceFacts& facts);

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
    // Its files' identity (spec::weights_identity of the listing's digests; "" unknown): the key
    // of the lab's speculation pairs.
    std::string weights;
    // Its output head's stored bytes (output.weight, or the token table it is tied to) read from
    // its own GGUF header (read_head_bytes); 0: not read.
    std::uint64_t head_bytes = 0;
    // A GGUF's own tensor table when its header was read (nullopt: the file size stands).
    std::optional<GgufTensorTable> table;
};


// The weights and the state of the drafters a start attaches beside the target (the subject's
// attached_drafters, as the converter's drafter plan takes them): a GGUF drafter file's size from
// the target's own `listing`, a drafter repository's from its listing at the declared revision
// (the hub's default branch may move) as the converter stores it beside a target of
// `target_model_type` (convert::companion_drafter_bytes: a Qwen DFlash companion's projections at
// Q8), a DFlash drafter's state from that revision's config.json.
// One code for every plan of a model not converted yet: ninferno-serve --plan --no-load and its
// start's memory check, ninferno-ui's /api/plan. `fetch` nullptr (offline): only the target's
// listing is read. What is not known stays 0 / nullopt.
struct AttachedDrafterMemory {
    std::uint64_t bytes = 0;
    std::optional<models::DrafterStateShape> state;
    // Whether the target takes the converter's default proposal head (a Qwen target), and its
    // DFlash companion's bytes as converted (0: none): what the head is priced on, per
    // quantization with the row's MTP layers (DryRunInputs::proposal_drafter_bytes).
    bool proposal_planned         = false;
    double proposal_drafter_bytes = 0;
};
[[nodiscard]] AttachedDrafterMemory
attached_drafter_memory(const SpeculationSubject& subject, const std::vector<QuantListing>& listing,
                        std::string_view target_model_type, std::uint32_t target_hidden,
                        const convert::FetchOptions* fetch);

// The memory of a drafter artifact given with --drafter, which drafts in place of the model's
// own: its tensors' bytes, and a DFlash drafter's state from its components' config (beside a
// target of hidden width `target_hidden`). Throws when the artifact cannot be read.
[[nodiscard]] AttachedDrafterMemory
supplied_drafter_memory(const std::filesystem::path& artifact, std::uint32_t target_hidden);

// A safetensors checkpoint's weights after the recipe a start converts it with (convert.cpp's
// default recipe choice): a recipe that keeps the published encodings (dense BF16, a published
// checkpoint) keeps its bytes; Qwen3.5/3.6 MoE BF16 takes qwen3_6_35b_a3b's (routed experts Q4
// gate/up, Q5 down, the rest Q8; declared bits per weight with their group scales). nullopt for a
// recipe whose size the plan does not know before the start.
[[nodiscard]] std::optional<std::uint64_t> checkpoint_converted_bytes(const nlohmann::json& config,
                                                                      std::uint64_t bytes);

// The axes the user pinned with flags (what is not pinned the configuration choice picks:
// product/plan/plan_choice.h), as the command line or the launch gave them, never inferred from the
// resolved options.
struct ChoicePins {
    bool kv          = false; // --kv-dtype other than auto
    bool activations = false; // --activations given
    bool state       = false; // --state-dtype given
    bool ring        = false; // --sliding-kv-dtype given
    bool chunk       = false; // --prefill-chunk given
};

// What a dry run plans with: the device, its profile key, the Engine options the server builds,
// the speculation asked for and the subject it is resolved for, and the axes the user pinned.
struct DryRunInputs {
    runtime::PlanDeviceFacts device;
    spec::GpuKey gpu;
    EngineOptions options;
    SpeculationAsk ask;
    const SpeculationSubject* subject = nullptr;
    // A quantization's own subject (the lab's speculation pairs are of its weights); unset, or
    // nullptr from it: `subject` for every quantization.
    std::function<const SpeculationSubject*(const QuantListing&)> subject_of;
    std::uint64_t drafter_bytes       = 0; // a separate drafter's weights, when known
    // The default proposal head a Qwen conversion with MTP layers or a DFlash companion adds,
    // priced for each quantization (proposal_bytes_at): whether one is planned and the
    // companion's bytes (0: none).
    bool proposal_planned         = false;
    double proposal_drafter_bytes = 0;
    // The row's own head (QuantListing::head_bytes, inputs_for); 0: the facts' head stands.
    std::uint64_t proposal_row_head_bytes = 0;
    // A DFlash drafter's state read from its config before it is downloaded (nullopt: unknown).
    std::optional<models::DrafterStateShape> drafter_state;
    ChoicePins pins;
};

// The inputs of a launch with its context capped at `context` and still automatic (ninferno-serve
// --max-context-cap X: a row's launch_context; the free memory picks the rung under that ceiling).
[[nodiscard]] DryRunInputs under_context_cap(const DryRunInputs& inputs, std::uint32_t context);

// The default proposal head's bytes (convert::planned_proposal_bytes) at `facts`' weights: the
// row's own head where its header was read, else the facts' head as it is (a head's format is
// the file's, not a share of the weights); 0 when the inputs plan none.
// A split Qwen GGUF's head, where its parsed part 0 did not hold it (Codex on #199): the later
// parts' headers read (convert::gguf_quants_head_bytes) for `quant`'s output.weight, else the
// tied token table every part was read without; the head's bytes, the tie, and an untied model's
// embedding (its part 0's token_embd.weight) set in `facts`. Nothing for a whole table or a head
// known.
void complete_split_gguf_head(PlanSourceFacts& facts, const std::string& spec,
                              const convert::FetchOptions& fetch, const std::string& quant);

[[nodiscard]] std::uint64_t proposal_bytes_at(const PlanSourceFacts& facts,
                                              const DryRunInputs& inputs);

// `inputs` for one quantization: its own subject when DryRunInputs::subject_of gives one.
[[nodiscard]] DryRunInputs inputs_for(const DryRunInputs& inputs, const QuantListing& quant);

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
    bool exact       = false;  // the exact preset: --activations exact with a BF16 KV cache
    // --activations: "" (automatic: the configuration choice's) or exact | fast | int8. The exact
    // preset takes exact whatever this says.
    std::string activations;
    // Whether the activations are pinned (the choice keeps them) rather than chosen.
    [[nodiscard]] bool activations_pinned() const noexcept { return exact || !activations.empty(); }
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
    std::optional<std::uint64_t> mtp_bytes; // its own NextN share (a tensor table), else the shape's
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
    // asked ones decoding at `estimate_depth` tokens in its window (a quarter of the asked context,
    // under Auto of its first step min(native, 32K); the same for every row, so the rows compare;
    // at least 512: an empty context flatters decode), the prefill, and the seconds of a
    // typical request (choice::kRequestPrompt + kRequestAnswer tokens). Not set for a row not planned.
    std::optional<choice::Speed> estimate;
    std::uint32_t estimate_depth = 0;
    // The drafter the estimate decodes with ("dflash:7"), "" when it counts none: plain decode,
    // or a drafter the plan serves whose gain no measurement gives (the choice's drafter axis).
    std::string estimate_drafter;
    double request_seconds       = 0;
    // The plan counts a default proposal head whose size its own header did not give (the
    // parsed header's head taken as this row's).
    bool proposal_head_assumed = false;
};

struct QuantPlan {
    std::vector<QuantRow> rows; // the listing's order (drafters and sidecars left out)
    std::optional<std::size_t> chosen;
    std::optional<runtime::QuantPick> pick;
    // "UD-Q6_K_XL: the largest that fits on this RTX 4090 (24 GB) at auto context (49,152 each)
    // x 1 stream" or why none (English, the CLI); `rule_code` and `rule_params` the same as data.
    std::string rule;
    std::string rule_code; // largest_fits | best_worth | none_fits | none_served
    // The worth picked the quantization within the rule's tier (product/choice: a GPU generation
    // that decides and the measured quantization curve); `largest` is then the rule's own pick.
    bool by_worth = false;
    std::optional<std::size_t> largest_index;
    // What the worth weighed against the rule's largest, each at its own choice on the generation's
    // reference card (choice::Scored::decision_seconds, accuracy): the largest's time per request
    // over the pick's, and its KLD less the pick's (the trade lambda priced).
    struct Trade {
        double time_ratio = 0;
        double kld_delta  = 0;
    };
    std::optional<Trade> trade;
    nlohmann::json rule_params = nlohmann::json::object();
};

// Every listed quantization of a model whose shape `shape` is (any quantization's header: the
// shape does not depend on it), planned as `inputs` ask, and the rule's pick. A row above 8 bits
// per weight, a drafter or sidecar, and a format the engine does not serve are listed unsupported.
[[nodiscard]] QuantPlan plan_quants(const PlanSourceFacts& shape,
                                    const std::vector<QuantListing>& listing,
                                    const DryRunInputs& inputs);

// The facts a row is planned at: the shape at the row's bytes and basis, its own NextN share when
// its tensor table was read. One code for every plan of a row (the table, the pick, now).
[[nodiscard]] PlanSourceFacts row_facts(const PlanSourceFacts& shape, const QuantRow& row);

// plan_quants, with the chosen GGUF row priced by its own tensor table: read (`read_table`, the
// caller's header read of that quantization; nullopt when it cannot) when the listing has none,
// and the table planned again with it, until the chosen row has its own (a table can move the
// choice) or its read fails; then the row ranked next above it, still at its file size (header
// and padding counted: a row just over the line may fit by its tensors), is read once: when it
// does not become the pick, the rows above it (larger) are not read. Nothing chosen: the
// lowest-ranked row at its file size is read once. One code for ninferno-serve --plan --no-load and
// ninferno-ui's /api/plan: the bytes a row shows are the ones its start reads.
// The reader returns nullopt when the read fails, an empty table (tensor_bytes 0) when the header
// was read but holds no whole table (a split file's part).
using ReadTensorTable = std::function<std::optional<GgufTensorTable>(const std::string& quant)>;
[[nodiscard]] QuantPlan plan_quants_read(const PlanSourceFacts& shape,
                                         std::vector<QuantListing>& listing,
                                         const DryRunInputs& inputs,
                                         const ReadTensorTable& read_table);

// One model at its own weights.
[[nodiscard]] PlannedSpeculation plan_model(const PlanSourceFacts& facts,
                                            const DryRunInputs& inputs);

// The rule's line ("<quant>: the largest that fits on this <device> (<N> GB) at <ctx> x <C>
// streams, ..."), and the line when nothing fits (what fits instead, from the smallest
// quantization's plan); fills plan.rule_code and plan.rule_params with the same as data.
[[nodiscard]] std::string quant_rule_line(QuantPlan& plan, const DryRunInputs& inputs);

// A plan of a model at given inputs: plan_model, or at a configuration choice's drafter.
using PlanAt = std::function<PlannedSpeculation(const DryRunInputs&)>;

// The longest context each of the inputs' streams holds on its own for a model at its weights
// (0: none): the served plan's when private, else the longest of the declared probe contexts
// (8K .. the model's own) whose plan at that pinned context (`plan_at`, plan_model when empty)
// keeps every stream's own.
[[nodiscard]] std::uint32_t own_context_of(const PlanSourceFacts& facts, const DryRunInputs& inputs,
                                           const PlannedSpeculation& served,
                                           const PlanAt& plan_at = {});

// Whether every stream of `inputs` holds `context` of its own (the plan at that pinned context).
[[nodiscard]] bool holds_own_context(const PlanSourceFacts& facts, const DryRunInputs& inputs,
                                     std::uint32_t context);

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
