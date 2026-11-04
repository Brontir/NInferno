#pragma once

// The configuration choice's replay record (#304): everything choose() was given, as canonical
// JSON, so that the decision can be made again from the record alone, with no device, model file,
// clock or measurement, and compared with the one that was made.
//
// What choose() reads through functions is recorded as the values it asked for: the model's KV,
// ring and state bytes per storage (every storage), and the own context of each candidate the
// choice enumerated (record_choice calls choose() once with a recording oracle). A replay that asks
// a value the record lacks is incomplete, never a guess: replay_choice throws ReplayIncomplete.
// The record names its schema (kChoiceReplaySchema); the constants the choice prices with are the
// build's (choice_constants.h), so a replay under another build answers what that build decides.

#include "product/choice/choice.h"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::product::choice {

inline constexpr std::string_view kChoiceReplaySchema = "ninferno-choice-replay/1";

struct ReplayIncomplete : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ChoiceRecord {
    Choice choice;
    nlohmann::ordered_json inputs; // the record: replay_choice(inputs) decides again
};

// choose() with what it was given recorded.
[[nodiscard]] ChoiceRecord record_choice(const ModelFacts& model, const CardFacts& card,
                                         const QuantOption& quant, const Allowed& allowed,
                                         const Pins& pins, const Task& task,
                                         const OwnContextFn& own_context, const Gates& gates = {});

// choose() again from a record. Throws ReplayIncomplete for a value the record lacks, and
// std::invalid_argument for another schema or a malformed record.
[[nodiscard]] Choice replay_choice(const nlohmann::ordered_json& inputs);

// A choice as canonical JSON: every scored candidate (its axes, own context, fit, worth, seconds,
// drafter), the chosen one, the context, the flags, the resolution and why. Two decisions are the
// same where these are.
[[nodiscard]] nlohmann::ordered_json choice_summary(const Choice& choice);

} // namespace ninferno::product::choice
