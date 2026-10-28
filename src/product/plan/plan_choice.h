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

// The drafter axis (choice/SPECULATION.md S3): beside plain decode, one option per speculation
// mode the subject's profile chose or measured and its Engine runs. A profile with measurements
// (a lab pair's, a local tune's) chooses each mode again over that mode's measurements alone; one
// without offers its own choice. Each option carries the served speculation a start of it runs
// (served_speculation over the profile of that mode), its gain the profile's choice for the
// server's thinking flag, and the streams from which its load table decodes plainly.
struct DrafterAxis {
    std::vector<choice::DrafterOption> options;
    std::vector<ServedSpeculation> served;
    std::size_t default_drafter = 0; // the subject's own served choice (0: plain)
};
[[nodiscard]] DrafterAxis drafter_axis(const SpeculationSubject& subject, const SpeculationAsk& ask);

struct ServedChoice {
    // The drafter axis's options (their gains, measured or not), beside `drafters`.
    std::vector<choice::DrafterOption> drafter_options;
    choice::QuantOption quant; // the quantization planned
    choice::Task task;
    choice::Choice choice;
    // The depth the quant table's estimates take (a quarter of the asked context or of Auto's
    // first step), and the chosen candidate's decode there.
    std::uint32_t depth         = 0;
    double decode_at_depth      = 0;
    // The drafter axis's speculations (choice::ModelFacts::drafters, in its order): what a start of
    // each would run. Empty without a speculation subject.
    std::vector<ServedSpeculation> drafters;
    // The activation modes the model's family runs (choice::allowed_axes), pinned or not.
    std::vector<ActivationMode> allowed_activations;
};

// How a choice plans each candidate's memory: its exact own context per stream (the longest it
// holds: what a caller shows), only whether it holds each of Auto's steps (one plan a step: enough
// to pick), or not at all (every candidate fits: the best worth the weights could reach, a bound).
enum class OwnContext : std::uint8_t { Exact, Steps, Unbounded };

// The choice at `facts`' own weights with `inputs`' options, the pinned axes kept. nullopt: the
// model's shape is unknown (nothing to plan).
[[nodiscard]] std::optional<ServedChoice> choose_at_weights(const PlanSourceFacts& facts,
                                                            const DryRunInputs& inputs,
                                                            const ChoicePins& pins,
                                                            const std::string& label = "",
                                                            OwnContext memory = OwnContext::Exact);

// What a start takes from a choice: the axes the user left automatic, each where the model has it
// (a recurrent state, sliding rings). One mapping for ninferno-serve and the CLI.
struct StartAxes {
    std::optional<KvCacheStorage> kv;
    std::optional<KvCacheStorage> ring;
    std::optional<RecurrentStateStorage> state;
    std::optional<ActivationMode> activations;
    std::optional<std::uint32_t> chunk; // the widest prefill chunk
    // The speculation mode the choice took (None: plain decode), where it chose a drafter: a
    // --spec auto start serves its profile chosen again for that mode (subject_of_mode).
    std::optional<spec::Mode> drafter;
};
[[nodiscard]] StartAxes start_axes(const ServedChoice& served, const ChoicePins& pins);

// The inputs with the storages a start of the choice runs (StartAxes' KV codec, state, rings and
// activations, the axes the user left automatic; not the chunk or the drafter).
[[nodiscard]] DryRunInputs inputs_at_choice(const DryRunInputs& inputs, const ServedChoice& served,
                                            const ChoicePins& pins);
// The plan a start serves at the configuration choice: `planned` planned again at the storages the
// choice took for the axes left automatic (the KV codec, state, rings and activations a start sets:
// apply_start_axes) where they differ from the ones it was planned at, and, under --spec auto, with
// the choice's drafter (StartAxes::drafter) where that differs, plain decode said so. One code for
// ninferno-serve's plan without the weights, the CLI's and ninferno-ui's.
[[nodiscard]] PlannedSpeculation plan_at_choice(const PlanSourceFacts& facts,
                                                const DryRunInputs& inputs,
                                                const ServedChoice& served, const ChoicePins& pins,
                                                PlannedSpeculation planned);
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
