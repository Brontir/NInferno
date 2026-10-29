#pragma once

// The speculation a start would serve with, without the weights: the same resolution and the same
// fallback order as ninferno-serve's --spec auto (serve/generation_service.cpp), each step judged
// by the plan without the weights (runtime/engine/plan_dry_run.h) instead of an Engine build.
//
//   the subject     an artifact on disk (product::speculation_subject, as a start reads it), or a
//                   model not converted yet: the drafter manifest the converter would write
//                   (convert::plan_drafters; a GGUF hub model only with --with-drafters, else the
//                   registry entry of its shape over the drafters it carries, its MTP layers);
//   the choice      the profile's default for the server's thinking flag (product::
//                   served_speculation): drafter, depth, every depth the engine captures, the load
//                   table (lanes -> depth) and the depth it bounds a step of C lanes at, the
//                   watchdog floor, where the profile came from (this GPU's, a similar one's, the
//                   "*" fallback, the built-in drafter's default);
//   the fallbacks   what a start tries when the choice does not fit, in its order: the profile's
//                   fit fallback, the deepest shallower depth that fits, the model's own MTP head,
//                   plain decode; and, with an automatic pool, the pool rule (the drafter leaves
//                   fewer whole contexts than C while plain decode holds more: the MTP head when
//                   it keeps four fifths of plain decode's contexts, else plain). Each step names
//                   whether it fits; the first that does is what the start serves.

#include "product/plan/plan_facts.h"
#include "product/speculation/speculation.h"
#include "runtime/engine/plan_dry_run.h"
#include "spec/profile.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::product {

struct QuantListing; // plan_quants.h

// A model not converted yet, as the dry run sees it.
struct PlannedTarget {
    std::string repo;                 // hub id ("" for a local GGUF or checkpoint)
    PlanSourceFacts facts;
    bool gguf          = true;  // converted from a GGUF
    bool with_drafters = false; // --with-drafters (a GGUF's separate drafters are attached)
    // The weights planned (spec::weights_identity of the listing's digests, "" unknown) and their
    // quantization label: the key of the lab's speculation pairs and the target's bit class.
    std::string weights;
    std::string quant;
    // The commit the listing was read at ("" unknown): a sibling quantization's measurement counts
    // only from the same commit (spec::pair_choice).
    std::string commit;
};

// A hub quantization's subject: with `cached`, the conversion kept here that carries drafters
// (convert::cached_conversion_with_drafters), resolved as the start resolves it (its manifest, its
// components, speculation_subject), the drafters the target declares with --with-drafters beside it
// for their memory; without, planned_subject.
[[nodiscard]] SpeculationSubject
hub_quant_subject(const PlannedTarget& target, const spec::GpuKey& gpu,
                  const std::optional<std::filesystem::path>& cached,
                  const std::filesystem::path& spec_profile = {});

// The subject of a planned target: the manifest convert::plan_drafters would give it (each
// drafter a start would attach counted as attached, a component one as embedded), resolved for
// `gpu` with spec::resolve_planned_speculation, its built-in drafter's default when nothing is
// found.
[[nodiscard]] SpeculationSubject planned_subject(const PlannedTarget& target,
                                                 const spec::GpuKey& gpu);

// The speculation subjects of a model not converted yet, one per quantization, as a start resolves
// each (the lab's pairs are of weights, so every row resolves its own). A hub model's separate
// drafters are planned as attached, an axis of the choice: a start attaches them when the choice
// drafts with one (choice/DECISIONS 45); a local GGUF's start attaches none, so none is planned.
// A quantization converted here with its drafters is planned from that conversion, which the start
// serves (hub_quant_subject). The server's plan without weights and ninferno-ui's plan both plan
// through it.
class PlannedSubjects {
public:
    // The conversion kept here with its drafters of a quantization (convert::
    // cached_conversion_with_drafters), nullopt when none; empty for a model not converted from
    // GGUF files.
    using ConversionOf =
        std::function<std::optional<std::filesystem::path>(const std::string& quant)>;
    // Whether a quantization's own file carries MTP layers (convert::gguf_carries_mtp), nullopt
    // when not read: then the model's facts say. A repository's quantizations differ (unsloth's
    // smallest drop them), and the facts are one file's.
    using MtpOf = std::function<std::optional<bool>(const std::string& quant)>;

    PlannedSubjects(PlannedTarget model, spec::GpuKey gpu, ConversionOf conversion_of,
                    std::filesystem::path spec_profile = {}, MtpOf mtp_of = {});

    // The subject of quantization `quant` ("" for the model read without one) whose files are
    // `weights` (spec::weights_identity of the listing's digests).
    [[nodiscard]] const SpeculationSubject& of(const std::string& quant, const std::string& weights);

    // DryRunInputs::subject_of over these subjects (none for a drafter's row); valid while this
    // object lives.
    [[nodiscard]] std::function<const SpeculationSubject*(const QuantListing&)> lookup();

private:
    PlannedTarget model_;
    spec::GpuKey gpu_;
    ConversionOf conversion_of_;
    std::filesystem::path spec_profile_;
    MtpOf mtp_of_;
    std::map<std::string, SpeculationSubject> made_;
};

// What was asked for: --spec auto (`automatic`), else the explicit settings (backend None: --spec
// none). `thinking` the server's thinking default, `per_request` --spec-select per request.
struct SpeculationAsk {
    bool automatic = true;
    SpeculativeOptions given;
    bool thinking    = true;
    bool per_request = false;
    // --spec-select not given: per request where the subject's measurements say it pays
    // (per_request_pays); per_request_of reads it.
    bool per_request_from_profile = false;
    // An explicit --spec without --draft-tokens: the depth the model's profile runs that backend
    // with on this GPU, else product::default_draft_tokens (ServeOptions::draft_tokens_from_model).
    bool depth_from_profile = false;
};

// Whether the converter would add its default proposal head to the target (convert::
// default_proposal): a Qwen 3.5 target with the default ranking found, and its MTP layers under a
// speculation that may draft with MTP (auto, or MTP asked), or a DFlash companion the registry
// declares under one that may draft with DFlash.
// Local facts only: the plan reads the quantizations' heads only when one is planned.
[[nodiscard]] bool default_proposal_possible(const PlannedTarget& target, const SpeculationAsk& ask);

// Whether the plan runs each request at its own thinking flag's choice: --spec-select's, or,
// left to the profile, per_request_pays for `subject` (as GenerationService decides it).
[[nodiscard]] bool per_request_of(const SpeculationAsk& ask, const SpeculationSubject* subject);

struct SpeculationFallbackStep {
    std::string step;  // profile | fit_fallback | shallower | head | plain | pool_head | pool_plain
    std::string label; // "dflash2:7", "plain"
    bool tried  = false; // the start would reach this step
    bool fits   = false; // the plan without the weights fits with it
    bool served = false; // the step a start ends on
    std::string note;
};

struct SpeculationReport {
    std::string ask; // "auto" | "none" | the explicit label
    // What the start would serve with: plain when nothing speculates.
    SpeculativeOptions engine;
    std::string label = "plain";
    std::string drafter, drafter_kind;
    std::string source; // where the profile came from (the resolution's words)
    // own (this GPU's profile) | similar (another GPU's standing in) | generic (the "*" profile) |
    // builtin (the built-in drafter's untuned default) | explicit (--spec) | none
    std::string basis = "none";
    std::vector<std::int32_t> depths;  // the depths the engine captures
    std::vector<spec::LoadBound> load; // the profile's load table (after the stand-in cap)
    std::string load_basis;            // measured | none
    // The depth a step of C lanes runs at under the load table (0: plain), C the asked concurrency.
    std::int32_t depth_at_concurrency = 0;
    std::vector<SpeculationFallbackStep> fallbacks;
    std::vector<std::string> notes;
    // The profile's choice for requests with thinking and for those without (its configuration
    // and gain, 0 when not measured), and whether each request runs its own (per_request_of): what
    // a thinking default changes, shown beside the served speculation. Empty with no profile.
    struct ThinkingVariant {
        std::string label;  // "dflash:15", "plain"
        double speedup = 0; // decode against plain decode's, as measured (0: not measured)
    };
    std::optional<ThinkingVariant> with_thinking, without_thinking;
    bool per_request = false;
};

// Whether the speculation `report` serves drafts with one of `subject`'s separate drafters
// (attached_drafters): its backend, or, each request its own, either thinking flag's; never with
// a drafter file given (--drafter: it drafts in their place). What decides
// whether a start attaches a GGUF hub model's separate drafter when --with-drafters does not.
[[nodiscard]] bool uses_attached_drafter(const SpeculationSubject& subject,
                                         const SpeculationReport& report);

// The plan of a model with the speculation a start ends on, and the report of how it got there.
struct PlannedSpeculation {
    runtime::DryRunPlan plan; // the plan with the served speculation (plain when none)
    SpeculationReport speculation;
};

// The draft head a run of `speculative` drafts with on a target of `model`: the proposal head it
// holds (an artifact's, or the one its conversion adds: PlanModelFacts::proposal_bytes) under a
// drafter it serves and no separate drafter, as the start's model_draft_head sets it; else as
// given. The family planner sizes the restricted head's logits.
[[nodiscard]] ProposalHead planned_proposal_head(const runtime::PlanModelFacts& model,
                                                 const SpeculativeOptions& speculative) noexcept;

// Runs the start's order over the dry run: `options` the Engine options the server builds with no
// speculation chosen yet (its --spec settings in options.speculative for an explicit --spec),
// `facts` the target's (the drafter's weights are added for a separate drafter when known:
// `drafter_bytes`, and a DFlash drafter's state, `drafter_state`, read from its config before it is
// downloaded), `subject` from speculation_subject or planned_subject (nullptr: no profile at all,
// as for an explicit --spec).
[[nodiscard]] PlannedSpeculation
plan_with_speculation(const PlanSourceFacts& facts, const runtime::PlanDeviceFacts& device,
                      const EngineOptions& options, const SpeculationSubject* subject,
                      const SpeculationAsk& ask, std::uint64_t drafter_bytes = 0,
                      const std::optional<models::DrafterStateShape>& drafter_state = std::nullopt);

// The profile key of a device from its facts (a named board's what-if: core/device_class.h
// synthetic_device_class from its PCI id, architecture and SMs, the facts table's rates), as
// spec::gpu_identity builds it for a CUDA device.
[[nodiscard]] spec::GpuKey gpu_key_of(const runtime::PlanDeviceFacts& device);

[[nodiscard]] nlohmann::json to_json(const SpeculationReport& report);

} // namespace ninferno::product
