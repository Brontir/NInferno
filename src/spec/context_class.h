#pragma once

// Context classes of speculation acceptance: how a drafter's acceptance falls as the prompt grows.
// A request is classed by its prompt length at admission (the bounds are lab constants):
//
//   Short  < 1024 tokens      Mid  < 8192      Long  < 24576      Huge  the rest
//
// Acceptance is a property of the model pair and of the bytes it computes, not of a kernel, so a
// factor is keyed by what acceptance depends on: the target and drafter by their source identity
// (artifact/source_identity.h: what they were converted from and how, never the artifact id, which
// is drawn at random per conversion), the whole byte class (B1: activation precision, summation
// order, KV codec, decode/verify parity), the sampling, the draft depth K, the prompt set (corpus
// hash) and, under fast activations, the lane count (at small T the bytes depend on the batch).
// The registry's top-level "context_classes":
//
//   {"bounds": [1024, 8192, 24576] (must be kContextBounds),
//    "entries": [{"model": registry model id,
//                 "target": source identity (64 hex) | {"source_identity": 64 hex | null,
//                           "proven": {the parts of the identity record known, as data},
//                           "missing": [the parts not known]} | a placeholder,
//                 "drafter": kind,
//                 "drafter_source": a separate drafter's source identity | "component:<name>"
//                                   (a component of the target: the target's identity and the
//                                   name),
//                 "byte_class": {"activations": "exact" | "fast-a8" | "fast-a4", "kv": ...,
//                                "summation": ..., "parity": ...},
//                 "sampling": "t0" | ..., "depth": K, "lanes": C, "corpus": "sha256:<64 hex>" |
//                 label, "schedule": the request schedule every card ran, "tolerance": 0.05,
//                 "source": free text,
//                 "cards": [{"gpu_class": "ga102", "route": optional route family or name,
//                            "schedule": the entry's, "factors": [1, f_mid, f_long,
//                            f_huge or null]}]}]}
//
// A factor is the class's acceptance (accepted drafts per round) over the Short class's, so Short
// is 1. It is known only when at least two cards that differ in generation (architecture) or both
// name routes of different families (route_family) measured it, each within the tolerance of their
// mean
// (|v - mean| / mean); the estimate is that mean. Routes count by family (the text before the
// first '.', route_family). A key holding a placeholder ("auto", an id "placeholder:...", in any
// case, a target that is no source identity, a target object with a part "missing", a
// drafter_source that is neither a source identity nor "component:<name>", or a corpus that is no
// "sha256:" and 64 lower-case hex digits) is never known. Otherwise factor() says unknown and why.
// Unknown is never 1.0.
//
// No decision reads factor() yet: this is the data and the API (idea 32, step 2); the rule that
// picks a depth per class comes next.

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
// How far each card of an entry may lie from the cards' mean (|v - mean| / mean) for the mean to
// stand. Every entry records it ("tolerance") and must record this value.
inline constexpr double kContextTolerance = 0.05;

[[nodiscard]] ContextClass class_of(std::int64_t prompt_tokens) noexcept;
[[nodiscard]] const char* class_name(ContextClass c) noexcept; // short, mid, long, huge

// What an acceptance factor is measured for, and what a lookup asks for.
struct AcceptanceKey {
    std::string model;          // registry model id
    std::string target;         // its source identity (artifact::source_identity)
    std::string drafter;        // drafter kind (DrafterDecl::kind)
    std::string drafter_source; // a separate drafter's source identity, or "component:<name>"
    nlohmann::json byte_class = nlohmann::json::object(); // compared whole
    std::string sampling;
    std::int32_t depth = 0;
    std::int32_t lanes = 0; // compared only under fast activations
    std::string corpus;
};

struct CardFactors {
    std::string gpu_class; // lower case: "ga102", "gb203:84sm"
    std::string arch;      // its architecture ("sm86"), the generation
    std::string route;     // optional, lower case: a route family or name ("ggml_mmq", "rmma")
    std::array<std::optional<double>, kContextClasses> factors;
};

struct ContextEntry {
    AcceptanceKey key;
    // A target given as an object: its proven parts and the names of the missing ones (the key's
    // target is then "missing:<names>", never known).
    nlohmann::json target_parts = nlohmann::json::object();
    std::vector<std::string> target_missing;
    std::string schedule, source;
    double tolerance = kContextTolerance;
    std::vector<CardFactors> cards;
};

struct ContextFactors {
    bool present = false; // the registry carried a "context_classes" object
    std::array<std::int64_t, kContextClasses - 1> bounds = kContextBounds;
    std::vector<ContextEntry> entries;
};

// Whether a route name belongs to a known GEMM route family (ops/generic/generic_gemm.h route_name:
// scalar, rmma, w4a4, tiled, w4a8, w4a8ws, staged, nvfp4t, r8, ggml_mmq, ggml_vec, i8), as a family
// alone or a full name ("rmma.m1.n8.w4x1"). Lower case only.
[[nodiscard]] bool known_route(std::string_view route);
// The route family a route name belongs to, as route_name assigns them: the text before the first
// '.', with "w4a8ws" (W4A8 with producer warps) folded into "w4a8". Two cards on one architecture
// differ in route only when their families differ.
[[nodiscard]] std::string_view route_family(std::string_view route);

// Parses a "context_classes" object. Throws std::invalid_argument naming what is wrong: bounds
// other than kContextBounds, an entry missing a key field, with a tolerance other than
// kContextTolerance or a byte class that is not an object of string activations, kv, summation and
// parity, a card whose gpu_class is not a known chip or architecture (ga102, gb203:84sm, sm86) or
// names more SMs than its chip has, whose route is unknown, whose schedule is not the entry's, or
// that repeats another card's class and route, factors not 4 values in (0, 1.5] or null with a
// Short factor of 1, two entries of the same key.
[[nodiscard]] ContextFactors parse_context_factors(const nlohmann::json& object);

struct ClassFactor {
    std::optional<double> value;
    std::string reason; // why it is unknown; empty when known
};

// The factor for `key` in class `c`, or unknown with a labelled reason: no context_classes, no
// entry for the model, a key field that differs ("<field> differs"), a placeholder key
// ("placeholder key: <field>=<value>"), no data for the class, fewer than two cards of different
// generations or routes, a card beyond the tolerance of the mean. No caller yet.
[[nodiscard]] ClassFactor factor(const ContextFactors& factors, const AcceptanceKey& key,
                                 ContextClass c);

} // namespace ninferno::spec
