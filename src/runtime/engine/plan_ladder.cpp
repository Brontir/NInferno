#include "runtime/engine/plan_ladder.h"

#include "core/paged_kv_cache.h"
#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninferno::runtime {

static_assert(kPlanContextFloor == kAutomaticContextFloor,
              "the ladder's context steps end at the automatic context's floor");
static_assert(kPlanSurvivalContext == kAutomaticContextMinimum,
              "the survival steps end at the automatic context's minimum");
static_assert(kPlanContextFloor % kAutomaticContextGrain == 0 &&
                  kPlanSurvivalContext % kAutomaticContextGrain == 0,
              "the ladder's context bounds are whole pages");

namespace {

// 2^(-r/16) in Q32, r = 0..15, rounded down (exact decimal expansion).
constexpr std::array<std::uint64_t, 16> kSixteenthRoots{
    4294967296ULL, 4112874773ULL, 3938502375ULL, 3771522796ULL, 3611622602ULL, 3458501653ULL,
    3311872529ULL, 3171459999ULL, 3037000499ULL, 2908241642ULL, 2784941737ULL, 2666869344ULL,
    2553802833ULL, 2445529971ULL, 2341847523ULL, 2242560871ULL,
};

std::uint32_t whole_pages(std::uint32_t tokens) {
    return tokens / kAutomaticContextGrain * kAutomaticContextGrain;
}

} // namespace

std::string_view plan_step_name(PlanStep step) noexcept {
    switch (step) {
    case PlanStep::Top:
        return "top";
    case PlanStep::ExtraStates:
        return "extra_states";
    case PlanStep::Pool:
        return "pool";
    case PlanStep::States:
        return "states";
    case PlanStep::Context:
        return "context";
    case PlanStep::Chunk:
        return "chunk";
    case PlanStep::Survival:
        return "survival";
    case PlanStep::LastResort:
        return "last_resort";
    }
    return "unknown";
}

std::string_view plan_reserve_name(PlanReserve reserve) noexcept {
    switch (reserve) {
    case PlanReserve::Kept:
        return "kept";
    case PlanReserve::Soft:
        return "soft";
    case PlanReserve::WarmUp:
        return "warm_up";
    }
    return "unknown";
}

std::uint32_t plan_context_pages(std::uint32_t max_context) {
    return paged_kv_capacity_pages(max_context);
}

std::uint32_t plan_scaled(std::uint32_t x, std::uint32_t step) noexcept {
    const std::uint32_t octaves = step / 16U;
    if (octaves >= 32U) { return 0; }
    const std::uint64_t base = static_cast<std::uint64_t>(x >> octaves);
    return static_cast<std::uint32_t>((base * kSixteenthRoots[step % 16U]) >> 32U);
}

std::vector<PlanRung> plan_ladder(const PlanLadderInputs& in) {
    if (in.context_ceiling == 0) { throw std::invalid_argument("the plan ladder needs a context"); }
    if (in.concurrency == 0) { throw std::invalid_argument("the plan ladder needs a concurrency"); }
    if (in.prefill_chunk == 0) {
        throw std::invalid_argument("the plan ladder needs a prefill chunk");
    }
    if (in.kv_tokens_pinned && *in.kv_tokens_pinned == 0) {
        throw std::invalid_argument("an explicit KV capacity must be positive");
    }
    if (!in.states_pinned && in.states_minimum > in.states_default) {
        throw std::invalid_argument("the fewest cached states exceed the default");
    }
    const std::uint32_t lanes = in.concurrency;
    // An explicit KV capacity bounds one request: the context is pinned to it (as before).
    const bool context_pinned   = in.context_pinned || in.kv_tokens_pinned.has_value();
    const std::uint32_t ceiling = in.context_ceiling;
    const std::uint32_t floor   = std::min(ceiling, kPlanContextFloor);

    const auto minimum_pages = [&](std::uint32_t context) {
        return std::max(plan_context_pages(context), lanes);
    };
    const auto maximum_pages = [&](std::uint32_t context) {
        const std::uint64_t pages = static_cast<std::uint64_t>(plan_context_pages(context)) * lanes;
        if (pages > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("the plan ladder's pool exceeds uint32 pages");
        }
        return static_cast<std::uint32_t>(pages);
    };
    const std::optional<std::uint32_t> pinned_pages =
        in.kv_tokens_pinned ? std::optional<std::uint32_t>(plan_context_pages(*in.kv_tokens_pinned))
                            : std::nullopt;
    const auto pool_at = [&](std::uint32_t context, bool whole) {
        if (pinned_pages) { return *pinned_pages; }
        return whole ? maximum_pages(context) : minimum_pages(context);
    };

    std::vector<PlanRung> ladder;
    const auto push = [&](PlanRung rung) {
        if (!ladder.empty()) {
            const PlanRung& last = ladder.back();
            // The Program prefills min(chunk, context) tokens per call: a chunk at or above the
            // context changes nothing.
            const bool same_chunk = std::min(last.prefill_chunk, last.max_context) ==
                                    std::min(rung.prefill_chunk, rung.max_context);
            if (last.max_context == rung.max_context && last.kv_pages == rung.kv_pages &&
                last.cached_states == rung.cached_states && same_chunk &&
                last.survival == rung.survival) {
                return;
            }
        }
        ladder.push_back(rung);
    };

    const std::uint32_t states_top = in.states_pinned.value_or(in.states_default);
    const std::uint32_t states_low = in.states_pinned.value_or(in.states_minimum);
    // A lane's cached state (one per lane, or the default when it is smaller).
    const std::uint32_t states_per_lane = std::max(states_low, std::min(states_top, lanes));
    // The floor quality: the poorest plan admitted with the class reserve kept. A chunk of
    // kPlanFloorQualityChunk (the configured one when narrower or pinned) and, without a Host
    // tier, a cached state per lane (none with one: the Host tier keeps the StateImages).
    const std::uint32_t states_floor = states_low == 0 ? 0U : states_per_lane;
    const std::uint32_t chunk_floor =
        in.chunk_pinned ? in.prefill_chunk : std::min(in.prefill_chunk, kPlanFloorQualityChunk);
    // The chunks the Program takes, widest first: the configured one, then halved down to the
    // narrowest (only the configured one when it is pinned).
    std::vector<std::uint32_t> chunks{in.prefill_chunk};
    if (!in.chunk_pinned) {
        for (std::uint32_t chunk = in.prefill_chunk / 2; chunk >= kNarrowestAutomaticPrefillChunk;
             chunk /= 2) {
            if (chunk % kPrefillChunkQuantum == 0) { chunks.push_back(chunk); }
        }
    }

    // Every step lowers one axis and keeps the others, so the content (context, KV tokens,
    // cached states, chunk) never grows down the ladder, and more free memory never serves less.
    PlanRung rung{.max_context   = ceiling,
                  .kv_pages      = pool_at(ceiling, true),
                  .cached_states = states_top,
                  .prefill_chunk = in.prefill_chunk,
                  .step          = PlanStep::Top};
    push(rung);
    // The second cached state of each lane (an agent's turn closure beside its endpoint) gives way
    // before any KV, one at a time down to one per lane.
    while (rung.cached_states > states_per_lane) {
        --rung.cached_states;
        rung.step = PlanStep::ExtraStates;
        push(rung);
    }
    // The pool: from C whole contexts down by g to one.
    if (!pinned_pages) {
        const std::uint32_t most  = maximum_pages(ceiling);
        const std::uint32_t least = minimum_pages(ceiling);
        for (std::uint32_t step = 1; rung.kv_pages > least; ++step) {
            rung.kv_pages = std::max(least, plan_scaled(most, step));
            rung.step     = PlanStep::Pool;
            push(rung);
        }
    }
    // The cached states down to the floor quality's: with a Host tier to none, before the
    // context (the Host tier keeps the StateImages, so a Device state is worth less than a step
    // of context); without one a lane's state stays (master fitted the context with one per lane).
    const auto step_states = [&](std::uint32_t down_to, bool survival) {
        while (rung.cached_states > down_to) {
            --rung.cached_states;
            rung.step     = PlanStep::States;
            rung.survival = survival;
            push(rung);
        }
    };
    step_states(states_floor, false);
    // The context, by g down to the floor (the floor itself the last). The grid is N's: the
    // context of step s is N * g^s in whole pages.
    const auto grid = [&](std::uint32_t step) { return whole_pages(plan_scaled(ceiling, step)); };
    std::uint32_t context_step = 1;
    if (!context_pinned) {
        for (; grid(context_step) > floor; ++context_step) {
            rung.max_context = grid(context_step);
            rung.kv_pages    = pool_at(rung.max_context, false);
            rung.step        = PlanStep::Context;
            push(rung);
        }
        if (rung.max_context > floor) {
            rung.max_context = floor;
            rung.kv_pages    = pool_at(floor, false);
            rung.step        = PlanStep::Context;
            push(rung);
        }
    }
    // A configured chunk wider than the floor quality's halves down to it, at the floor.
    for (const std::uint32_t chunk : chunks) {
        if (chunk >= rung.prefill_chunk || chunk < chunk_floor) { continue; }
        rung.prefill_chunk = chunk;
        rung.step          = PlanStep::Chunk;
        push(rung);
    }
    // The survival range, below the floor quality and admitted without the class reserve (the
    // selection admits a survival rung Soft; it is pruned on its own, prune_plan_ladder). With a
    // class reserve it starts at the floor quality itself (the same content for less memory);
    // nothing in it is richer.
    rung.survival = true;
    if (in.class_reserve) {
        rung.step = PlanStep::Survival;
        push(rung);
    }
    // The chunk halves, then the remaining cached states go, then (an automatic context) the
    // context steps below the floor at the narrowest chunk and the fewest states.
    for (const std::uint32_t chunk : chunks) {
        if (chunk >= rung.prefill_chunk) { continue; }
        rung.prefill_chunk = chunk;
        rung.step          = PlanStep::Chunk;
        push(rung);
    }
    step_states(states_low, true);
    if (context_pinned) { return ladder; }
    const std::uint32_t shortest = std::min(ceiling, kPlanSurvivalContext);
    for (; grid(context_step) >= rung.max_context; ++context_step) {}
    for (; rung.max_context > shortest; ++context_step) {
        const std::uint32_t next = std::max(grid(context_step), shortest);
        rung.max_context         = next;
        rung.kv_pages            = pool_at(next, false);
        rung.step                = PlanStep::Context;
        push(rung);
    }
    if (in.last_resort_context != 0 && in.last_resort_context < rung.max_context) {
        rung.max_context = in.last_resort_context;
        rung.kv_pages    = pool_at(rung.max_context, false);
        rung.step        = PlanStep::LastResort;
        push(rung);
    }
    return ladder;
}

std::vector<PlanRung>
prune_plan_ladder(const std::vector<PlanRung>& candidates,
                  const std::function<std::optional<std::uint64_t>(std::size_t)>& bytes) {
    std::vector<PlanRung> out;
    std::optional<std::uint64_t> cheapest;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const std::optional<std::uint64_t> b = bytes(index);
        if (index == 0) {
            // The top rung stays whatever it costs: the selection reports it.
            out.push_back(candidates[index]);
            cheapest = b;
            continue;
        }
        if (!b) { continue; }
        // The survival range is pruned on its own: its rungs need less beside their bytes (no
        // class reserve), so its first is kept whatever it costs against the rungs above.
        const bool first_survival = candidates[index].survival && !out.back().survival;
        if (cheapest && !first_survival && *b >= *cheapest) { continue; }
        out.push_back(candidates[index]);
        cheapest = b;
    }
    return out;
}

std::uint64_t plan_class_reserve(std::uint64_t total_bytes, bool paravirtual, bool drives_display,
                                 bool integrated) noexcept {
    constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;
    const std::uint64_t vram_class =
        total_bytes / kPlanVramClassGrainBytes * kPlanVramClassGrainBytes;
    std::uint64_t reserve = 0;
    if (paravirtual) {
        reserve =
            drives_display ? std::min(vram_class / 8, 4 * kGiB) : std::min(vram_class / 32, kGiB);
    }
    if (integrated) { reserve = std::max(reserve, std::max(vram_class / 8, 4 * kGiB)); }
    return reserve;
}

std::uint64_t plan_needed_bytes(const PlanRungCost& cost, const PlanMargins& margins,
                                PlanReserve reserve) noexcept {
    switch (reserve) {
    case PlanReserve::Kept:
        return cost.bytes + cost.pin_bytes + margins.allowance +
               std::max(margins.base, margins.reserve);
    case PlanReserve::Soft:
        return cost.bytes + cost.pin_bytes + margins.allowance + margins.base;
    case PlanReserve::WarmUp:
        return cost.held_bytes + cost.pin_bytes + margins.base;
    }
    return std::numeric_limits<std::uint64_t>::max();
}

namespace {

// What a rung needs from the free memory at the selection: its bytes, the allowance, and the
// class reserve above the floor (a survival rung gives the reserve up for its context).
std::uint64_t rung_need(const PlanRung& rung, const PlanRungCost& cost,
                        const PlanMargins& margins) {
    return plan_needed_bytes(cost, margins, rung.survival ? PlanReserve::Soft : PlanReserve::Kept);
}

PlanReserve rung_reserve(const PlanRung& rung, const PlanMargins& margins) {
    return rung.survival && margins.reserve > margins.base ? PlanReserve::Soft : PlanReserve::Kept;
}

} // namespace

std::optional<PlanSelection> select_plan_rung(const std::vector<PlanRung>& ladder,
                                              const PlanRungCostFn& cost, std::uint64_t free_bytes,
                                              const PlanMargins& margins, std::size_t from) {
    if (!cost) { throw std::invalid_argument("the plan ladder's selection needs a cost"); }
    // One pass with one need per rung: the first rung that fits. More free memory never picks a
    // lower rung, whatever the needs are.
    for (std::size_t index = from; index < ladder.size(); ++index) {
        const std::optional<PlanRungCost> rung_cost = cost(index);
        if (rung_cost && rung_need(ladder[index], *rung_cost, margins) <= free_bytes) {
            return PlanSelection{.rung = index, .reserve = rung_reserve(ladder[index], margins)};
        }
    }
    // Nothing fits with the allowance: the last rung on the warm-up's measurement (an asked plan
    // the graph bound alone would refuse; the warm-up keeps it only if it then fits).
    if (margins.allowance != 0 && from < ladder.size()) {
        const std::size_t last = ladder.size() - 1U;
        if (const std::optional<PlanRungCost> rung_cost = cost(last);
            rung_cost &&
            plan_needed_bytes(*rung_cost, margins, PlanReserve::WarmUp) <= free_bytes) {
            return PlanSelection{.rung = last, .reserve = PlanReserve::WarmUp};
        }
    }
    return std::nullopt;
}

std::optional<PlanSelection> verify_plan_rung(const std::vector<PlanRung>& ladder,
                                              const PlanRungCostFn& cost, std::uint64_t free_bytes,
                                              const PlanMargins& margins, PlanSelection chosen) {
    if (chosen.rung >= ladder.size()) { throw std::out_of_range("plan rung out of the ladder"); }
    if (!cost) { throw std::invalid_argument("the plan ladder's check needs a cost"); }
    // A WarmUp rung keeps its tier: the held bytes beside the base margin (it was admitted
    // without the allowance and the class reserve; adding them back would refuse it before any
    // graph is measured). It is the last rung: it stays or the plan is refused.
    if (chosen.reserve == PlanReserve::WarmUp) {
        const std::optional<PlanRungCost> rung_cost = cost(chosen.rung);
        if (rung_cost &&
            plan_needed_bytes(*rung_cost, margins, PlanReserve::WarmUp) <= free_bytes) {
            return chosen;
        }
        return std::nullopt;
    }
    // What the allowance was for is measured now.
    PlanMargins measured = margins;
    measured.allowance   = 0;
    if (const std::optional<PlanSelection> kept =
            select_plan_rung(ladder, cost, free_bytes, measured, chosen.rung)) {
        return kept;
    }
    // Nothing from the chosen rung down fits on the margins: the last rung on its held bytes
    // (WarmUp), as the selection would have taken it with this free memory. Without it a rung
    // admitted Soft could be refused where a little less free memory was served WarmUp. Only a
    // plan with a warm-up allowance has a WarmUp tier, as in select_plan_rung.
    if (margins.allowance == 0) { return std::nullopt; }
    const std::size_t last = ladder.size() - 1U;
    if (const std::optional<PlanRungCost> rung_cost = cost(last);
        rung_cost && plan_needed_bytes(*rung_cost, margins, PlanReserve::WarmUp) <= free_bytes) {
        return PlanSelection{.rung = last, .reserve = PlanReserve::WarmUp};
    }
    return std::nullopt;
}

std::optional<PlanSelection> plan_warm_up_check(const std::vector<PlanRung>& ladder,
                                                const PlanRungCostFn& cost, std::uint64_t free_now,
                                                std::uint64_t taken, const PlanMargins& margins,
                                                PlanSelection chosen) {
    const std::uint64_t measured = free_now > taken ? free_now - taken : 0;
    return verify_plan_rung(ladder, cost, measured, margins, chosen);
}

std::string describe_plan_rung(const PlanRung& rung, std::uint32_t concurrency) {
    const auto grouped = [](std::uint64_t value) {
        std::string digits = std::to_string(value);
        for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) {
            digits.insert(static_cast<std::size_t>(at), ",");
        }
        return digits;
    };
    const std::uint64_t kv_tokens =
        static_cast<std::uint64_t>(rung.kv_pages) * kAutomaticContextGrain;
    const std::uint64_t context_pages = plan_context_pages(rung.max_context);
    std::string pool;
    if (context_pages != 0 && rung.kv_pages % context_pages == 0) {
        const std::uint64_t contexts = rung.kv_pages / context_pages;
        pool = std::to_string(contexts) + (contexts == 1 ? " context" : " contexts");
    } else {
        // Tenths of a context, rounded down.
        const std::uint64_t tenths =
            rung.kv_pages * 10ULL / std::max<std::uint64_t>(1, context_pages);
        pool = std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + " contexts";
    }
    return grouped(rung.max_context) + " tokens x " + pool + " (" + grouped(kv_tokens) +
           " KV tokens for " + std::to_string(concurrency) +
           (concurrency == 1 ? " lane" : " lanes") + "), " + std::to_string(rung.cached_states) +
           " cached states, chunk " + std::to_string(rung.prefill_chunk);
}

} // namespace ninferno::runtime
