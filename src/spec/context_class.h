#pragma once

// Context classes of speculation acceptance: how a drafter's acceptance falls as the prompt grows.
// A request is classed by its prompt length at admission (the bounds are lab constants):
//
//   Short  < 1024 tokens      Mid  < 8192      Long  < 24576      Huge  the rest
//
// The registry's top-level "context_classes" object carries, per drafter kind, the factor of each
// class: its acceptance (accepted drafts per round) over the Short class's, so Short is 1.0:
//
//   "context_classes": {"bounds": [1024, 8192, 24576],
//                       "stamp": {"engine": kSpecEngineVersion, "measured_on": [GPU classes],
//                                 "source": free text},
//                       "factors": {"dflash": [1.0, f_mid, f_long, f_huge or null], ...}}
//
// A factor is known only when it was measured on at least two distinct cards by this engine
// version, for these bounds; otherwise factor() says unknown and why. Unknown is never 1.0.

#include "spec/profile.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::spec {

enum class ContextClass : std::uint8_t { Short, Mid, Long, Huge };

inline constexpr std::size_t kContextClasses = 4;
// The upper bounds (exclusive) of Short, Mid and Long, in prompt tokens.
inline constexpr std::array<std::int64_t, kContextClasses - 1> kContextBounds{1024, 8192, 24576};
// Bound of a factor: acceptance at longer context rarely rises much above the Short class's.
inline constexpr double kMaxContextFactor = 1.5;
// Cards a factor must be measured on before it is used (one card cannot tell the model from it).
inline constexpr std::size_t kMinContextCards = 2;

[[nodiscard]] ContextClass class_of(std::int64_t prompt_tokens) noexcept;
[[nodiscard]] const char* class_name(ContextClass c) noexcept; // short, mid, long, huge

struct ContextFactors {
    bool present = false; // the registry carried a "context_classes" object
    std::array<std::int64_t, kContextClasses - 1> bounds = kContextBounds;
    std::string engine;                   // stamp: the engine version measured
    std::vector<std::string> measured_on; // stamp: GPU classes measured (lower case)
    std::string source;                   // stamp: free text
    // Per drafter kind, one factor per class (nullopt: not measured).
    std::map<std::string, std::array<std::optional<double>, kContextClasses>, std::less<>> factors;
};

// Parses a "context_classes" object. Throws std::invalid_argument naming what is wrong: bounds not
// three strictly increasing positive integers, a factor array not of length 4, a factor outside
// (0, 1.5] or a Short factor other than 1, a stamp or factors not an object, a missing stamp
// engine, a measured_on name not a GPU class key (ga102, gb203:84sm; read in lower case).
[[nodiscard]] ContextFactors parse_context_factors(const nlohmann::json& object);

struct ClassFactor {
    std::optional<double> value;
    std::string reason; // why it is unknown; empty when known
};

// The factor of `kind` (the drafter's kind, DrafterDecl::kind: dflash, dflash2, mtp, ...; never
// a speculation Mode, which gemma4_assistant and nemotron_h_mtp share with mtp) in class `c`, or
// unknown with a labelled reason: no context_classes, bounds other than kContextBounds, a stamp of
// another engine version than `engine`, fewer than kMinContextCards distinct cards, no data for the
// kind or the class.
[[nodiscard]] ClassFactor factor(const ContextFactors& factors, std::string_view kind,
                                 ContextClass c, std::string_view engine = kSpecEngineVersion);

} // namespace ninferno::spec
