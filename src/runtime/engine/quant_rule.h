#pragma once

// The quant rule (G3): the largest quantization of a model that fits a card at an asked context x
// streams, from each quantization's plan without the weights (plan_dry_run.h). One rule for
// ninferno-serve's fetch without --quant, ninferno-serve --plan --no-load, and ninferno-ui's
// /api/plan, /api/quants and catalog.
//
// What "fits" means (the owner's flow: the user asks for a context, we show the largest quant that
// fits it):
//   - each stream holds the target context of its own: an asked --max-context, or under an
//     automatic context min(the model's own, kPlanUsefulContext); only when no quantization at or
//     above kPlanQualityFloorBits reaches it does Auto step the target down (16K, then 8K own)
//     before any shared pool;
//   - a shared pool (fewer whole contexts than streams) comes next, and is said in plain words;
//   - below kPlanQualityFloorBits only when nothing at or above the floor fits at all;
//   - last, a quantization that fits only by giving up the class reserve kept for the desktop
//     (a survival rung: the server starts it with a warning).
// Declared constants, no measurement.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::runtime {

struct DryRunPlan;

// Auto's target: each stream gets at least this much context of its own (or the model's own when
// shorter), and the shortest step Auto goes down to before a shared pool.
inline constexpr std::uint32_t kPlanUsefulContext  = 32768;
inline constexpr std::uint32_t kPlanSteppedContext = 16384;
inline constexpr std::uint32_t kPlanFloorContext   = 8192;
// The automatic pick goes below this many bits per weight only when nothing at or above it fits;
// below kPlanLowQualityBits the pick is marked low quality.
inline constexpr double kPlanQualityFloorBits = 3.5;
inline constexpr double kPlanLowQualityBits   = 3.0;

struct QuantCandidate {
    std::string label;             // as --quant takes it
    std::uint64_t bytes    = 0;    // its weights
    int label_bits         = 0;    // from the label (Q4_K_M: 4), 0 when it says none
    double bits_per_weight = 0;    // bytes * 8 / params (0: params unknown)
    bool supported         = true; // the engine serves it here
};

// The bits the rule ranks by: bits per weight, else the label's.
[[nodiscard]] double quant_rank_bits(const QuantCandidate& q) noexcept;

enum class QuantFitState : std::uint8_t {
    Own,         // every stream holds the target context of its own
    Shared,      // only as one pool the streams share (a non-survival rung)
    Desktop,     // only on a survival rung: the reserve kept for the desktop given up
    No,          // nothing fits
    Unknown,     // not planned: the shape could not be read
    Unsupported, // the engine does not serve it
};

[[nodiscard]] std::string_view quant_fit_state_name(QuantFitState state) noexcept;

struct QuantFit {
    QuantFitState state       = QuantFitState::No;
    std::uint32_t context     = 0; // the served plan's context at the ask (0: none)
    std::uint32_t own_context = 0; // the longest context each stream holds on its own (0: none)
};

// What the rule is asked: the context (automatic or pinned), the streams, the model's own context.
struct QuantAsk {
    bool automatic            = true;
    std::uint32_t context     = 0; // the pinned context (automatic: 0)
    std::uint32_t concurrency = 1;
    std::uint32_t native      = 0;
};

// The own-context targets in order: pinned {X}; automatic {min(native, 32K), min(native, 16K),
// min(native, 8K)} without repeats.
[[nodiscard]] std::vector<std::uint32_t> quant_own_targets(const QuantAsk& ask);

// A plan's fit state at the ask, and the own context given (the caller probes it: see
// plan_quants.cpp). `own_context` >= the lowest target makes it Own.
[[nodiscard]] QuantFit quant_fit_of(const DryRunPlan& plan, std::uint32_t own_context,
                                    const QuantAsk& ask);

enum class QuantPickTier : std::uint8_t {
    Own,              // at or above the floor bits, each stream its own target context
    OwnStepped,       // automatic only: the same at a shorter target (16K, then 8K own)
    Shared,           // at or above the floor, one pool the streams share
    OwnBelowFloor,    // below the floor bits, own
    SharedBelowFloor, // below the floor bits, shared
    Desktop,          // only by giving up the reserve kept for the desktop
};

[[nodiscard]] std::string_view quant_pick_tier_name(QuantPickTier tier) noexcept;

struct QuantPick {
    std::size_t index            = 0;
    QuantPickTier tier           = QuantPickTier::Own;
    std::uint32_t target_context = 0; // the own context per stream the tier reached (0: shared)
    bool below_floor             = false;
    bool low_quality             = false;
};

// The rule's pick (nullopt: nothing fits). Tiers in order: Own at the first target, OwnStepped at
// the later ones (automatic), Shared, OwnBelowFloor at the targets, SharedBelowFloor, Desktop.
// Within a tier the order of quant_ranks_above. `fits[i]` is candidate i's.
[[nodiscard]] std::optional<QuantPick>
largest_fitting_quant(const std::vector<QuantCandidate>& candidates,
                      const std::vector<QuantFit>& fits, const QuantAsk& ask);

// The rule's order within a tier, and the quant table's: more bits per weight, then more bytes,
// then the label. A strict weak order. (No quality tie-break: no published KLD is wired in.)
[[nodiscard]] bool quant_ranks_above(const QuantCandidate& a, const QuantCandidate& b) noexcept;

} // namespace ninferno::runtime
