#pragma once

// The generic path's route selection as an ordered list of named authorities (#614, step A1).
//
// An authority is a function of the Ask (the profile, the width and the round's lanes, resolved
// once at entry), the Problem and the Weight. It answers with a route and the evidence the route
// rests on, or abstains. select_route, auto_route and heuristic_route (generic_gemm.h) walk the
// three lists below in the order of the old cascades, and the first answer wins: the routes are
// the ones those cascades gave (ninferno_selector_golden_test holds them byte for byte); what is
// new is that every answer names its authority and its evidence. The precedence itself (one rule
// over the kinds of evidence) is a later step of #614.
//
// Evidence kinds. Pin: a maintainer's choice (a route pin, a route probe). Measured: timed on this
// board (a route table measured here, the dense tuner's entry of this class and build, the R8
// decode table of this board). Ranker: the cost model where it prices the call, else a rule (a
// derivation from the card's facts, or a literal of one or more measured cards). Carried: a
// measurement of a board within kCarryDistance (core/measured_boards.h), every quantity compared.
// Transferred: a measurement of a farther board, carried by the authority's own rule. Borrowed: an
// exact-key entry taken from a farther table because the nearest lacked it (route_table.cpp
// fill_from).

#include "ops/generic/generic_gemm.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ninferno::ops::detail::generic {

// The route profile of a call (ninferno/ops/prefill_routes.h): Speed (decode, verify and the
// speed bands by T), Prefill (one route per problem from 17 tokens, whatever the width) and
// Invariant (the register MMA's decode split at every width).
enum class Profile : std::uint8_t { Speed, Prefill, Invariant };

[[nodiscard]] constexpr const char* profile_name(Profile p) noexcept {
    switch (p) {
    case Profile::Speed:
        return "speed";
    case Profile::Prefill:
        return "prefill";
    case Profile::Invariant:
        return "invariant";
    }
    return "speed";
}

// What is asked of the authorities, resolved once per call.
struct Ask {
    Profile profile;
    std::int32_t tokens;
    int lanes; // the round's sequences (ops::round_lanes()): the MMQ verify rule reads it

    // The prefill band: the prefill profile's calls of 17 tokens or more, which take one route per
    // problem; below it the profile's calls route as the speed profile's.
    [[nodiscard]] bool prefill_band() const noexcept {
        return profile == Profile::Prefill && tokens >= 17;
    }

    [[nodiscard]] bool prefill() const noexcept { return profile == Profile::Prefill; }
};

// The Ask of a call entering select_route: Invariant while the invariant profile is active, else
// Prefill or Speed by `prefill`, the round's lanes as they stand.
[[nodiscard]] Ask resolve_ask(std::int32_t tokens, bool prefill);
// The Ask of a call below the profiles (auto_route, heuristic_route: the pins' A16 answer, the
// tuner's baseline): Prefill or Speed by `prefill` whatever profile is active.
[[nodiscard]] Ask ask_below_profiles(std::int32_t tokens, bool prefill);

struct Evidence {
    enum class Kind : std::uint8_t { Pin, Measured, Ranker, Carried, Transferred, Borrowed };
    enum class Ranker : std::uint8_t { None, Cost, Rule };
    Kind kind     = Kind::Ranker;
    Ranker ranker = Ranker::Rule; // Kind::Ranker: which ranker; None otherwise
    std::string_view source;      // the authority's name (an authority list's entry)
    // The board or table class the answer was measured on, for a Carried, Transferred or Borrowed
    // answer (a route table's class, an R8 decode table's board id, a gate's donor board); empty
    // where the answer is this board's own, a pin's or a ranker's.
    std::string donor;
    double distance = 0; // board_distance to the donor (0: this board)
};

[[nodiscard]] constexpr const char* evidence_kind_name(Evidence::Kind k) noexcept {
    switch (k) {
    case Evidence::Kind::Pin:
        return "pin";
    case Evidence::Kind::Measured:
        return "measured";
    case Evidence::Kind::Ranker:
        return "ranker";
    case Evidence::Kind::Carried:
        return "carried";
    case Evidence::Kind::Transferred:
        return "transferred";
    case Evidence::Kind::Borrowed:
        return "borrowed";
    }
    return "ranker";
}

[[nodiscard]] constexpr const char* evidence_ranker_name(Evidence::Ranker r) noexcept {
    switch (r) {
    case Evidence::Ranker::None:
        return "none";
    case Evidence::Ranker::Cost:
        return "cost";
    case Evidence::Ranker::Rule:
        return "rule";
    }
    return "none";
}

struct RouteAnswer {
    Route route;
    Evidence evidence;
};

// An authority: the first of a list to answer decides the call. nullopt: it abstains (its scope
// does not hold the call, or nothing it names is admitted).
using Authority = std::optional<RouteAnswer> (*)(const Ask&, const Problem&, const Weight&);

struct NamedAuthority {
    std::string_view name; // Evidence::source of its answers
    Authority answer;
};

// The three lists, in the order they are asked.
// select: the invariant profile, a route probe, Volta's tiled route, the maintainer's route pins,
// Volta's HMMA verify route; then auto.
[[nodiscard]] std::span<const NamedAuthority> select_authorities() noexcept;
// auto: the NVFP4 half-L2 rule, the route table (outside the int8 calls), the dense tuner's entry
// (outside the wide int8 calls); then the rules, whose answer the dense tuner's prefill band may
// replace (dense_tune_prefill_band: the one authority that refines an answer rather than precedes
// it, since the band is keyed by the rules' route).
[[nodiscard]] std::span<const NamedAuthority> auto_authorities() noexcept;
// rules: W4A4, the vector kernel, the cost model, R8 (its decode table, else its rules), the MMQ
// prefill gate, I8, W4A8, the MMQ verify rule, the tiled route, the staged MMA, the register MMA,
// its narrow fallbacks, the staged fallback and the scalar route, which always answers.
[[nodiscard]] std::span<const NamedAuthority> rule_authorities() noexcept;

// select_route, auto_route and heuristic_route with the evidence of the answer.
[[nodiscard]] RouteAnswer select_answer(const Problem& problem, const Weight& weight,
                                        std::int32_t tokens, bool prefill);
[[nodiscard]] RouteAnswer auto_answer(const Ask& ask, const Problem& problem, const Weight& weight);
[[nodiscard]] RouteAnswer heuristic_answer(const Ask& ask, const Problem& problem,
                                           const Weight& weight);

} // namespace ninferno::ops::detail::generic
