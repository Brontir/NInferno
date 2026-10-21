#pragma once

// Context classes of speculation acceptance: how a drafter's acceptance falls as the prompt grows.
// A request is classed by its prompt length at admission (the bounds are lab constants):
//
//   Short  < 1024 tokens      Mid  < 8192      Long  < 24576      Huge  the rest
//
// Acceptance is a property of the model pair and of the bytes it computes, not of a kernel, so a
// factor is keyed by what acceptance depends on: the target and drafter artifacts, the whole byte
// class (B1: activation precision, summation order, KV codec, decode/verify parity), the sampling,
// the draft depth K, the prompt set (corpus hash) and, under fast activations, the lane count
// (at small T the bytes depend on the batch). The registry's top-level "context_classes":
//
//   {"bounds": [1024, 8192, 24576], "tolerance": 0.05,
//    "entries": [{"model": registry model id, "target_artifact": id,
//                 "drafter": kind, "drafter_artifact": id | "component:<name>" (of the target),
//                 "byte_class": {"activations": "exact" | "fast-a8" | "fast-a4", "kv": ...,
//                                "summation": ..., "parity": ...},
//                 "sampling": "t0" | ..., "depth": K, "lanes": C, "corpus": "sha256:<hex>" | label,
//                 "schedule": the request schedule every card ran, "source": free text,
//                 "cards": [{"gpu_class": "ga102", "route": optional, "factors": [1, f_mid,
//                            f_long, f_huge or null]}]}]}
//
// A factor is the class's acceptance (accepted drafts per round) over the Short class's, so Short
// is 1. It is known only when at least two cards that differ in generation (architecture) or in
// route measured it and agree within the tolerance ((max - min) / mean); the estimate is their
// mean. Otherwise factor() says unknown and why. Unknown is never 1.0.

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
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
// How far the cards of an entry may disagree ((max - min) / mean) for their mean to stand; the
// registry records the value it was checked with ("tolerance").
inline constexpr double kContextTolerance = 0.05;

[[nodiscard]] ContextClass class_of(std::int64_t prompt_tokens) noexcept;
[[nodiscard]] const char* class_name(ContextClass c) noexcept; // short, mid, long, huge

// What an acceptance factor is measured for, and what a lookup asks for.
struct AcceptanceKey {
    std::string model; // registry model id
    std::string target_artifact;
    std::string drafter;          // drafter kind (DrafterDecl::kind)
    std::string drafter_artifact; // its artifact id, or "component:<name>" of the target
    nlohmann::json byte_class = nlohmann::json::object(); // compared whole
    std::string sampling;
    std::int32_t depth = 0;
    std::int32_t lanes = 0; // compared only under fast activations
    std::string corpus;
};

struct CardFactors {
    std::string gpu_class; // lower case: "ga102", "gb203:84sm"
    std::string arch;      // its architecture ("sm86"), the generation
    std::string route;     // optional
    std::array<std::optional<double>, kContextClasses> factors;
};

struct ContextEntry {
    AcceptanceKey key;
    std::string schedule, source;
    std::vector<CardFactors> cards;
};

struct ContextFactors {
    bool present = false; // the registry carried a "context_classes" object
    std::array<std::int64_t, kContextClasses - 1> bounds = kContextBounds;
    double tolerance                                     = kContextTolerance;
    std::vector<ContextEntry> entries;
};

// Parses a "context_classes" object. Throws std::invalid_argument naming what is wrong: bounds not
// three strictly increasing positive integers, a tolerance outside (0, 0.5], an entry missing a
// key field or with a byte class that is not an object of string activations, kv, summation and
// parity, a card whose gpu_class is not a known chip or architecture (ga102, gb203:84sm, sm86)
// or repeats another card's class and route, factors not 4 values in (0, 1.5] or null with a
// Short factor of 1, two entries of the same key.
[[nodiscard]] ContextFactors parse_context_factors(const nlohmann::json& object);

struct ClassFactor {
    std::optional<double> value;
    std::string reason; // why it is unknown; empty when known
};

// The factor for `key` in class `c`, or unknown with a labelled reason: no context_classes, other
// class bounds, no entry for the model, a key field that differs (named), no data for the class,
// fewer than two cards of different generations or routes, cards that disagree beyond the
// tolerance.
[[nodiscard]] ClassFactor factor(const ContextFactors& factors, const AcceptanceKey& key,
                                 ContextClass c);

} // namespace ninferno::spec
