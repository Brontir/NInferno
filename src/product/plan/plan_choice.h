#pragma once

// The configuration choice (product/choice/choice.h) at a model's weights, from the facts the plan
// without the weights reads (plan_facts.h) and its own memory: each candidate's own context per
// stream is the dry run's (own_context_of) at that KV codec, activation mode and state. One code
// path for ninferno-serve's start (the axes the user left automatic), --plan --no-load, and
// ninferno-ui's /api/plan.

#include "product/choice/choice.h"
#include "product/plan/plan_quants.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::product {

// The choice's facts of a model (its shape, the bytes a step reads per storage) and of a card.
[[nodiscard]] choice::ModelFacts choice_model_of(const PlanSourceFacts& facts);
[[nodiscard]] choice::CardFacts choice_card_of(const spec::GpuKey& gpu);

// The axes the user pinned with flags (what is not pinned the choice picks).
struct ChoicePins {
    bool kv          = false; // --kv-dtype other than auto
    bool activations = false; // --activations given
    bool state       = false; // --state-dtype given
    bool ring        = false; // --sliding-kv-dtype given
};

struct ServedChoice {
    std::vector<choice::QuantOption> quants; // the one quantization planned
    choice::Task task;
    choice::Choice choice;
    // The own context per stream the choice was asked for: the pinned context, else Auto's targets
    // in turn (runtime::quant_own_targets: min(native, 32K), then 16K, then 8K).
    std::uint32_t target_context = 0;
    // The depth the quant table's estimates take (a quarter of the asked context or of Auto's
    // first step), and the chosen candidate's decode there.
    std::uint32_t depth         = 0;
    double decode_at_depth      = 0;
};

// The choice at `facts`' own weights with `inputs`' options, the pinned axes kept. nullopt: the
// model's shape is unknown (nothing to plan).
[[nodiscard]] std::optional<ServedChoice> choose_at_weights(const PlanSourceFacts& facts,
                                                            const DryRunInputs& inputs,
                                                            const ChoicePins& pins,
                                                            const std::string& label = "");

// What a start takes from a choice: the axes the user left automatic, each where the model has it
// (a recurrent state, sliding rings). One mapping for ninferno-serve and the CLI.
struct StartAxes {
    std::optional<KvCacheStorage> kv;
    std::optional<KvCacheStorage> ring;
    std::optional<RecurrentStateStorage> state;
    std::optional<ActivationMode> activations;
};
[[nodiscard]] StartAxes start_axes(const ServedChoice& served, const ChoicePins& pins);
// The axes set on a start's Engine options (the KV codec then no longer automatic, the state
// likewise).
void apply_start_axes(EngineOptions& options, const StartAxes& axes);

// The choice for an artifact on disk a process starts on CUDA device `device` with `options` (its
// Engine options as built, the speculation `ask` as given): the artifact's facts, the device's
// own. nullopt with `why` when the device or the artifact cannot be read.
[[nodiscard]] std::optional<ServedChoice>
choose_for_artifact(const std::filesystem::path& artifact, int device, const EngineOptions& options,
                    const SpeculationAsk& ask, const std::string& spec_profile,
                    const ChoicePins& pins, std::string* why);

// The inputs with a candidate's KV codec, ring, state and activation mode pinned.
[[nodiscard]] DryRunInputs inputs_with(const DryRunInputs& inputs, const choice::Candidate& c);

// "choice | KV int8, activations fast, state f16: <why>" (one line, English), and the same as JSON
// (docs/reference/serving.md, "choice"): the chosen candidate, its estimates, and every
// alternative of each axis with what it would change.
[[nodiscard]] std::string choice_line(const ServedChoice& served);
[[nodiscard]] nlohmann::json choice_json(const ServedChoice& served);

} // namespace ninferno::product
