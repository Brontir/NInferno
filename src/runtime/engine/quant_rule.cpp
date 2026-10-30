#include "runtime/engine/quant_rule.h"

#include "runtime/engine/plan_dry_run.h"

#include <algorithm>
#include <stdexcept>

namespace ninferno::runtime {

double quant_rank_bits(const QuantCandidate& q) noexcept {
    return q.bits_per_weight > 0 ? q.bits_per_weight : static_cast<double>(q.label_bits);
}

std::string_view quant_fit_state_name(QuantFitState state) noexcept {
    switch (state) {
    case QuantFitState::Own:
        return "own";
    case QuantFitState::Shared:
        return "shared";
    case QuantFitState::Desktop:
        return "desktop";
    case QuantFitState::No:
        return "no";
    case QuantFitState::Unknown:
        return "unknown";
    case QuantFitState::Unsupported:
        return "unsupported";
    }
    return "no";
}

std::string_view quant_pick_tier_name(QuantPickTier tier) noexcept {
    switch (tier) {
    case QuantPickTier::Own:
        return "own";
    case QuantPickTier::OwnStepped:
        return "own_stepped";
    case QuantPickTier::Shared:
        return "shared";
    case QuantPickTier::OwnBelowFloor:
        return "own_below_floor";
    case QuantPickTier::SharedBelowFloor:
        return "shared_below_floor";
    case QuantPickTier::Desktop:
        return "desktop";
    }
    return "own";
}

std::vector<std::uint32_t> quant_own_targets(const QuantAsk& ask) {
    std::vector<std::uint32_t> out;
    if (!ask.automatic) {
        out.push_back(ask.context);
        return out;
    }
    const std::uint32_t native = ask.native == 0 ? kPlanUsefulContext : ask.native;
    for (const std::uint32_t step : {kPlanUsefulContext, kPlanSteppedContext, kPlanFloorContext}) {
        const std::uint32_t target = std::min(native, step);
        if (out.empty() || target < out.back()) { out.push_back(target); }
    }
    return out;
}

QuantFit quant_fit_of(const DryRunPlan& plan, std::uint32_t own_context, const QuantAsk& ask) {
    QuantFit out;
    if (!plan.planned) {
        out.state = QuantFitState::Unknown;
        return out;
    }
    const DryRunRung* chosen                 = plan.chosen();
    out.context                              = chosen != nullptr ? chosen->rung.max_context : 0;
    out.own_context                          = own_context;
    const std::vector<std::uint32_t> targets = quant_own_targets(ask);
    if (own_context != 0 && own_context >= targets.back()) {
        out.state = QuantFitState::Own;
    } else if (chosen != nullptr && !chosen->rung.survival) {
        // One stream holds its context on its own whatever its length.
        out.state = plan.private_contexts() ? QuantFitState::Own : QuantFitState::Shared;
    } else if (chosen != nullptr) {
        out.state = QuantFitState::Desktop;
    } else {
        out.state = QuantFitState::No;
    }
    return out;
}

bool below_quality_floor(const QuantCandidate& q) noexcept {
    const double bits = quant_rank_bits(q);
    return bits > 0 && bits < kPlanQualityFloorBits;
}

bool quant_ranks_above(const QuantCandidate& a, const QuantCandidate& b) noexcept {
    const double x = quant_rank_bits(a), y = quant_rank_bits(b);
    if (x != y) { return x > y; }
    if (a.bytes != b.bytes) { return a.bytes > b.bytes; }
    return a.label < b.label;
}

std::optional<QuantPick> largest_fitting_quant(const std::vector<QuantCandidate>& candidates,
                                               const std::vector<QuantFit>& fits,
                                               const QuantAsk& ask) {
    if (fits.size() != candidates.size()) {
        throw std::invalid_argument("largest_fitting_quant: a fit for every candidate");
    }
    const auto above_floor = [&](std::size_t i) { return !below_quality_floor(candidates[i]); };
    const auto best_of = [&](auto&& admits) -> std::optional<std::size_t> {
        std::optional<std::size_t> best;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (!candidates[i].supported || !admits(i)) { continue; }
            if (!best || quant_ranks_above(candidates[i], candidates[*best])) { best = i; }
        }
        return best;
    };
    const auto runs = [&](std::size_t i) {
        return fits[i].state == QuantFitState::Own || fits[i].state == QuantFitState::Shared;
    };
    const auto pick = [&](std::size_t i, QuantPickTier tier, std::uint32_t target) {
        const double bits = quant_rank_bits(candidates[i]);
        return QuantPick{.index          = i,
                         .tier           = tier,
                         .target_context = target,
                         .below_floor    = bits != 0 && bits < kPlanQualityFloorBits,
                         .low_quality    = bits != 0 && bits < kPlanLowQualityBits};
    };
    const std::vector<std::uint32_t> targets = quant_own_targets(ask);
    for (const bool floor : {true, false}) {
        for (std::size_t t = 0; t < targets.size(); ++t) {
            const std::uint32_t target = targets[t];
            if (const auto best = best_of([&](std::size_t i) {
                    return above_floor(i) == floor && fits[i].state == QuantFitState::Own &&
                           fits[i].own_context >= target;
                })) {
                const QuantPickTier tier = !floor   ? QuantPickTier::OwnBelowFloor
                                           : t == 0 ? QuantPickTier::Own
                                                    : QuantPickTier::OwnStepped;
                return pick(*best, tier, target);
            }
        }
        if (const auto best =
                best_of([&](std::size_t i) { return above_floor(i) == floor && runs(i); })) {
            // A plan whose streams each hold a context of their own, shorter than the targets.
            if (fits[*best].state == QuantFitState::Own) {
                return pick(*best, floor ? QuantPickTier::OwnStepped : QuantPickTier::OwnBelowFloor,
                            fits[*best].own_context != 0 ? fits[*best].own_context
                                                         : fits[*best].context);
            }
            return pick(*best, floor ? QuantPickTier::Shared : QuantPickTier::SharedBelowFloor, 0);
        }
    }
    if (const auto best =
            best_of([&](std::size_t i) { return fits[i].state == QuantFitState::Desktop; })) {
        return pick(*best, QuantPickTier::Desktop, 0);
    }
    return std::nullopt;
}

} // namespace ninferno::runtime
