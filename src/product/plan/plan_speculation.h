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
#include <optional>
#include <string>
#include <vector>

namespace ninferno::product {

// A model not converted yet, as the dry run sees it.
struct PlannedTarget {
    std::string repo;                 // hub id ("" for a local GGUF or checkpoint)
    std::vector<std::string> lineage; // base models (a GGUF's general.base_model)
    PlanSourceFacts facts;
    bool gguf          = true;  // converted from a GGUF
    bool with_drafters = false; // --with-drafters (a GGUF's separate drafters are attached)
};

// The subject of a planned target: the manifest convert::plan_drafters would give it (each
// drafter a start would attach counted as attached, a component one as embedded), resolved for
// `gpu` with spec::resolve_planned_speculation, its built-in drafter's default when nothing is
// found.
[[nodiscard]] SpeculationSubject planned_subject(const PlannedTarget& target,
                                                 const spec::GpuKey& gpu);

// What was asked for: --spec auto (`automatic`), else the explicit settings (backend None: --spec
// none). `thinking` the server's thinking default, `per_request` --spec-select per request.
struct SpeculationAsk {
    bool automatic = true;
    SpeculativeOptions given;
    bool thinking    = true;
    bool per_request = false;
    // An explicit --spec without --draft-tokens: the depth the model's profile runs that backend
    // with on this GPU, else product::default_draft_tokens (ServeOptions::draft_tokens_from_model).
    bool depth_from_profile = false;
};

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
};

// The plan of a model with the speculation a start ends on, and the report of how it got there.
struct PlannedSpeculation {
    runtime::DryRunPlan plan; // the plan with the served speculation (plain when none)
    SpeculationReport speculation;
};

// Runs the start's order over the dry run: `options` the Engine options the server builds with no
// speculation chosen yet (its --spec settings in options.speculative for an explicit --spec),
// `facts` the target's (the drafter's weights are added for a separate drafter when known:
// `drafter_bytes`), `subject` from speculation_subject or planned_subject (nullptr: no profile at
// all, as for an explicit --spec).
[[nodiscard]] PlannedSpeculation
plan_with_speculation(const PlanSourceFacts& facts, const runtime::PlanDeviceFacts& device,
                      const EngineOptions& options, const SpeculationSubject* subject,
                      const SpeculationAsk& ask, std::uint64_t drafter_bytes = 0);

// The profile key of a device from its facts (a named board's what-if: core/device_class.h
// synthetic_device_class from its PCI id, architecture and SMs, the facts table's rates), as
// spec::gpu_identity builds it for a CUDA device.
[[nodiscard]] spec::GpuKey gpu_key_of(const runtime::PlanDeviceFacts& device);

[[nodiscard]] nlohmann::json to_json(const SpeculationReport& report);

} // namespace ninferno::product
