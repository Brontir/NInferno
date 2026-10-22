// The plan ladder (src/runtime/engine/plan_ladder.h): the rungs are a pure function of the
// options and the model's own context, their bytes never grow down the ladder, the free memory
// picks the first rung that fits at the declared margins, an explicit option pins its axis,
// survival rungs exist below the floor, and the warm-up's re-measurement only moves the plan
// down. A second part replays the user scenarios of the plan-levels review (the arithmetic of
// docs/maintainer/deterministic-decisions.md, Plan levels): the ladder against a continuous fit of
// the same byte model, and against master's continuous fit. CPU.

#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/plan_ladder.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace rt = ninferno::runtime;
using rt::PlanLadderInputs;
using rt::PlanReserve;
using rt::PlanRung;
using rt::PlanRungCost;
using rt::PlanSelection;
using rt::PlanStep;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// A byte model of a hybrid decoder's plan: KV pages, StateImages (the active lanes' and the
// cached ones), a workspace proportional to the prefill chunk, and a CUDA graph allowance per lane
// (held: the real graphs once a probe built them).
struct ByteModel {
    std::uint64_t page_bytes      = 2 * kMiB;
    std::uint64_t state_bytes     = 75 * kMiB;
    std::uint64_t workspace_bytes = 512 * kMiB; // at chunk 1024
    std::uint64_t workspace_lane  = 0;          // more per lane
    std::uint64_t graph_lane      = 12 * kMiB;  // the planner's allowance per lane
    std::uint64_t graph_real_lane = 6 * kMiB;   // what a probe measured
    std::uint32_t concurrency     = 1;

    [[nodiscard]] std::uint64_t fixed(std::uint32_t states, std::uint32_t chunk) const {
        return workspace_bytes * chunk / 1024 + workspace_lane * concurrency +
               (concurrency + states) * state_bytes;
    }

    [[nodiscard]] PlanRungCost cost(const PlanRung& rung, bool held = false) const {
        const std::uint64_t base = fixed(rung.cached_states, rung.prefill_chunk) +
                                   std::uint64_t{rung.kv_pages} * page_bytes;
        const std::uint64_t graphs = (held ? graph_real_lane : graph_lane) * concurrency;
        return {.bytes      = base + graphs,
                .held_bytes = base + graph_real_lane * concurrency,
                .pin_bytes  = 0};
    }

    [[nodiscard]] rt::PlanRungCostFn cost_fn(const std::vector<PlanRung>& ladder,
                                             bool held = false) const {
        return [this, &ladder, held](std::size_t index) -> std::optional<PlanRungCost> {
            return cost(ladder.at(index), held);
        };
    }
};

PlanLadderInputs automatic(std::uint32_t ceiling, std::uint32_t concurrency) {
    return {.context_ceiling  = ceiling,
            .context_pinned   = false,
            .concurrency      = concurrency,
            .kv_tokens_pinned = std::nullopt,
            .states_pinned    = std::nullopt,
            .states_default   = rt::automatic_cached_device_states(concurrency),
            .states_minimum   = 1,
            .prefill_chunk    = 1024};
}

// Whether rung `b` serves no more than rung `a` on any axis (context, KV tokens, cached states,
// chunk).
bool content_at_most(const PlanRung& b, const PlanRung& a) {
    return b.max_context <= a.max_context && b.kv_pages <= a.kv_pages &&
           b.cached_states <= a.cached_states && b.prefill_chunk <= a.prefill_chunk;
}

int scaled_grid() {
    int failures = 0;
    failures += check(rt::plan_scaled(1U << 20, 0) == 1U << 20, "g^0 is one");
    failures += check(rt::plan_scaled(1U << 20, 16) == 1U << 19, "g^16 halves");
    failures += check(rt::plan_scaled(1U << 20, 8) == 741455, "g^8 is 2^-1/2, rounded down");
    failures += check(rt::plan_scaled(262144, 1) == 251029, "g^1 of 262,144");
    failures += check(rt::plan_scaled(7, 600) == 0, "a step past every octave");
    // Every step keeps at least 95.6 % of the one above it (the grid's worst loss).
    for (std::uint32_t step = 0; step < 64; ++step) {
        const std::uint64_t a = rt::plan_scaled(1U << 30, step);
        const std::uint64_t b = rt::plan_scaled(1U << 30, step + 1);
        failures += check(b * 1000 >= a * 956 && b < a, "a grid step of " + std::to_string(step));
    }
    return failures;
}

int ladder_shape() {
    int failures = 0;
    // Qwen3.8-27B at C=4: 262,144 tokens, two states per lane.
    const std::vector<PlanRung> ladder = rt::plan_ladder(automatic(262144, 4));
    failures +=
        check(ladder == rt::plan_ladder(automatic(262144, 4)), "the ladder is deterministic");
    const PlanRung& top = ladder.front();
    failures +=
        check(top.max_context == 262144 && top.kv_pages == 4 * 4096 && top.cached_states == 8 &&
                  top.prefill_chunk == 1024 && top.step == PlanStep::Top,
              "the top rung is every lane's own context with two states per lane");
    // The axes step in order (extra states, pool, context, never back), then the survival range;
    // every step lowers one axis and keeps the others, so the content never grows down.
    const auto order = [](PlanStep step) { return static_cast<int>(step); };
    for (std::size_t i = 1; i < ladder.size(); ++i) {
        const PlanRung& a = ladder[i - 1];
        const PlanRung& b = ladder[i];
        if (!b.survival) {
            failures += check(!a.survival && order(b.step) >= order(a.step) &&
                                  b.step != PlanStep::States && b.prefill_chunk == 1024,
                              "the axes step in order at " + std::to_string(i));
        }
        failures += check(content_at_most(b, a), "the content grows at " + std::to_string(i));
        failures +=
            check(!(a == b) && !(a.max_context == b.max_context && a.kv_pages == b.kv_pages &&
                                 a.cached_states == b.cached_states &&
                                 a.prefill_chunk == b.prefill_chunk && a.survival == b.survival),
                  "equal rungs are listed once");
        if (b.step == PlanStep::Pool) {
            failures +=
                check(b.kv_pages < a.kv_pages && b.kv_pages * 1000ULL >= a.kv_pages * 956ULL,
                      "a pool step keeps 95.6 % at " + std::to_string(i));
        }
        if (b.step == PlanStep::Context && a.step == PlanStep::Context && !b.survival) {
            failures += check(b.max_context * 1000ULL >= a.max_context * 950ULL,
                              "a context step keeps 95 % at " + std::to_string(i));
        }
        if (b.step == PlanStep::Context || b.survival) {
            failures += check(b.kv_pages == std::max(rt::plan_context_pages(b.max_context), 4U),
                              "below the top context the pool is one context");
        }
    }
    const auto find = [&](PlanStep step) {
        return std::find_if(ladder.begin(), ladder.end(),
                            [&](const PlanRung& r) { return r.step == step; });
    };
    failures += check(find(PlanStep::ExtraStates)->kv_pages == 4 * 4096 &&
                          find(PlanStep::Pool)->cached_states == 4,
                      "the second state of each lane gives way before the pool");
    // Without a Host tier a state per lane stays down to the floor quality (master fitted the
    // context with one per lane too): the reserve-kept rungs end at 8192 tokens, chunk 1024 and
    // 4 states.
    const auto first_survival =
        std::find_if(ladder.begin(), ladder.end(), [](const PlanRung& r) { return r.survival; });
    const PlanRung& floor_quality = *std::prev(first_survival);
    failures += check(floor_quality.max_context == 8192 && floor_quality.prefill_chunk == 1024 &&
                          floor_quality.cached_states == 4 && floor_quality.kv_pages == 128,
                      "the reserve-kept rungs end at the floor quality");
    // Below it, without a class reserve: the chunk halves at the floor, then the states go, then
    // the context; each survival rung keeps the axis it stepped.
    std::vector<std::pair<PlanStep, std::uint32_t>> survival;
    for (auto r = first_survival; r != ladder.end() && r->max_context == 8192; ++r) {
        survival.emplace_back(r->step,
                              r->step == PlanStep::Chunk ? r->prefill_chunk : r->cached_states);
    }
    failures += check(survival ==
                          std::vector<std::pair<PlanStep, std::uint32_t>>{
                              {PlanStep::Chunk, 512},
                              {PlanStep::Chunk, 256},
                              {PlanStep::Chunk, 128},
                              {PlanStep::States, 3},
                              {PlanStep::States, 2},
                              {PlanStep::States, 1},
                          },
                      "below the floor quality: the chunk, then the states, at the floor");
    // With a class reserve (WDDM, a GPU sharing the host's memory) the survival range starts at
    // the floor quality itself, the same content without the reserve; nothing in it is richer.
    PlanLadderInputs reserved        = automatic(262144, 4);
    reserved.class_reserve           = true;
    const std::vector<PlanRung> wddm = rt::plan_ladder(reserved);
    const auto soft_floor =
        std::find_if(wddm.begin(), wddm.end(), [](const PlanRung& r) { return r.survival; });
    failures += check(soft_floor->step == PlanStep::Survival &&
                          soft_floor->max_context == floor_quality.max_context &&
                          soft_floor->kv_pages == floor_quality.kv_pages &&
                          soft_floor->cached_states == floor_quality.cached_states &&
                          soft_floor->prefill_chunk == floor_quality.prefill_chunk &&
                          wddm.size() == ladder.size() + 1,
                      "with a class reserve the survival range starts at the floor quality");
    failures += check(ladder.back().max_context == 1024 && ladder.back().survival &&
                          ladder.back().prefill_chunk == 128 && ladder.back().cached_states == 1,
                      "with speculation the survival steps end at 1024 tokens");
    for (const PlanRung& r : ladder) {
        failures += check(r.survival ? r.max_context <= 8192 : r.max_context >= 8192,
                          "survival means at or below the floor");
        failures += check(r.survival || r.step != PlanStep::States,
                          "without a Host tier the states step only in the survival range");
        failures += check(r.max_context % 64 == 0, "every context is whole pages");
    }
    // Without speculation one last rung holds one prompt token and one decode round.
    PlanLadderInputs plain                = automatic(262144, 4);
    plain.last_resort_context             = 64;
    const std::vector<PlanRung> with_last = rt::plan_ladder(plain);
    failures += check(with_last.size() == ladder.size() + 1 &&
                          with_last.back().step == PlanStep::LastResort &&
                          with_last.back().max_context == 64,
                      "the last-resort rung below the survival steps");
    // A Host tier lets the states go down to none, before the context: the floor quality has no
    // cached Device state.
    PlanLadderInputs host                   = automatic(262144, 1);
    host.states_minimum                     = 0;
    const std::vector<PlanRung> host_ladder = rt::plan_ladder(host);
    const auto host_states =
        std::find_if(host_ladder.begin(), host_ladder.end(),
                     [](const PlanRung& r) { return r.step == PlanStep::States; });
    failures +=
        check(host_ladder[1].step == PlanStep::ExtraStates && host_ladder[1].cached_states == 1 &&
                  host_states != host_ladder.end() && !host_states->survival &&
                  host_states->max_context == 262144 && host_states->cached_states == 0 &&
                  std::all_of(
                      host_ladder.begin(), host_ladder.end(),
                      [](const PlanRung& r) { return r.step != PlanStep::States || !r.survival; }),
              "with a Host tier the states step to none before the context");
    // A pinned context: no context steps; the survival range halves the chunk, then the states.
    PlanLadderInputs pinned                   = automatic(32768, 4);
    pinned.context_pinned                     = true;
    const std::vector<PlanRung> pinned_ladder = rt::plan_ladder(pinned);
    bool pinned_ok                            = true;
    for (std::size_t i = 0; i < pinned_ladder.size(); ++i) {
        const PlanRung& r = pinned_ladder[i];
        pinned_ok         = pinned_ok && r.max_context == 32768 &&
                    (r.survival || (r.prefill_chunk == 1024 && r.cached_states >= 4)) &&
                    (i == 0 || content_at_most(r, pinned_ladder[i - 1]));
    }
    failures +=
        check(pinned_ok && pinned_ladder.back().prefill_chunk == 128 &&
                  pinned_ladder.back().cached_states == 1 && pinned_ladder.back().kv_pages == 512,
              "a pinned context keeps the floor quality with the reserve, then survives");
    pinned.class_reserve                    = true;
    const std::vector<PlanRung> pinned_wddm = rt::plan_ladder(pinned);
    failures += check(pinned_wddm.size() == pinned_ladder.size() + 1,
                      "with a class reserve a pinned context's survival range starts at the "
                      "floor quality, and no richer copy follows");
    // A model shorter than the floor: no context steps above it, then survives down to 1024.
    const std::vector<PlanRung> short_model = rt::plan_ladder(automatic(4096, 1));
    failures += check(std::none_of(short_model.begin(), short_model.end(),
                                   [](const PlanRung& r) {
                                       return r.step == PlanStep::Context && !r.survival;
                                   }) &&
                          short_model.back().max_context == 1024,
                      "a model below the floor steps its chunk, then survives down to 1024");
    return failures;
}

// The candidates pruned by a byte model.
std::vector<PlanRung> pruned(const std::vector<PlanRung>& candidates, const ByteModel& model) {
    return rt::prune_plan_ladder(candidates, [&](std::size_t i) -> std::optional<std::uint64_t> {
        return model.cost(candidates[i]).bytes;
    });
}

int bytes_never_grow() {
    int failures = 0;
    for (const std::uint32_t lanes : {1U, 2U, 4U, 8U}) {
        ByteModel model;
        model.concurrency                      = lanes;
        PlanLadderInputs in                    = automatic(262144, lanes);
        in.last_resort_context                 = 64;
        in.class_reserve                       = true;
        const std::vector<PlanRung> candidates = rt::plan_ladder(in);
        const std::vector<PlanRung> ladder     = pruned(candidates, model);
        failures +=
            check(ladder.front() == candidates.front() && ladder.back() == candidates.back() &&
                      ladder.size() <= candidates.size(),
                  "pruning keeps the top and the last rung");
        for (std::size_t i = 1; i < ladder.size(); ++i) {
            // The survival range (without the class reserve) is pruned on its own.
            const bool soft_floor     = ladder[i].survival && !ladder[i - 1].survival;
            const std::uint64_t below = model.cost(ladder[i]).bytes;
            const std::uint64_t above = model.cost(ladder[i - 1]).bytes;
            failures += check(soft_floor || below < above, "bytes do not fall at rung " +
                                                               std::to_string(i) +
                                                               " of C=" + std::to_string(lanes));
            failures += check(content_at_most(ladder[i], ladder[i - 1]),
                              "the content grows at rung " + std::to_string(i));
        }
        const auto soft_floor = std::find_if(ladder.begin(), ladder.end(),
                                             [](const PlanRung& r) { return r.survival; });
        failures += check(soft_floor != ladder.end() && soft_floor->max_context == 8192 &&
                              soft_floor->prefill_chunk == 1024,
                          "pruning keeps the floor admitted without the class reserve");
        // No narrower chunk holds a longer context than a rung above it.
        failures += check(std::none_of(ladder.begin(), ladder.end(),
                                       [](const PlanRung& r) {
                                           return r.prefill_chunk < 1024 && r.max_context > 8192;
                                       }),
                          "a narrower chunk kept a longer context");
    }
    // States that cost nothing (no recurrent state) are no rungs.
    ByteModel stateless;
    stateless.state_bytes              = 0;
    const std::vector<PlanRung> ladder = pruned(rt::plan_ladder(automatic(131072, 1)), stateless);
    failures +=
        check(std::none_of(ladder.begin(), ladder.end(),
                           [](const PlanRung& r) {
                               return r.step == PlanStep::States || r.step == PlanStep::ExtraStates;
                           }),
              "free cached states are no rungs");
    return failures;
}

int selection() {
    int failures = 0;
    ByteModel model;
    const std::vector<PlanRung> ladder = pruned(rt::plan_ladder(automatic(262144, 1)), model);
    const rt::PlanMargins margins{.base = 32 * kMiB, .reserve = 0, .allowance = 64 * kMiB};
    const auto cost = model.cost_fn(ladder);
    // At exactly a rung's need it is picked; a byte less, the next one.
    for (const std::size_t k : {std::size_t{0}, std::size_t{5}, std::size_t{20}}) {
        const std::uint64_t need = rt::plan_needed_bytes(*cost(k), margins, PlanReserve::Kept);
        const auto at            = rt::select_plan_rung(ladder, cost, need, margins);
        const auto less          = rt::select_plan_rung(ladder, cost, need - 1, margins);
        failures += check(at && at->rung == k && at->reserve == PlanReserve::Kept,
                          "the rung at its exact need " + std::to_string(k));
        failures += check(less && less->rung > k, "a byte short picks a lower rung");
        failures += check(at == rt::select_plan_rung(ladder, cost, need, margins),
                          "the selection is deterministic");
    }
    // Monotone in the free memory: more never picks a lower rung.
    std::size_t previous = ladder.size();
    for (std::uint64_t free = 256 * kMiB; free <= 20 * kGiB; free += 37 * kMiB) {
        const auto pick = rt::select_plan_rung(ladder, cost, free, margins);
        if (!pick) { continue; }
        failures += check(pick->rung <= previous, "more free memory picked a lower rung");
        previous = pick->rung;
    }
    // Nothing fits: nullopt, never a guess.
    failures += check(!rt::select_plan_rung(ladder, cost, 100 * kMiB, margins),
                      "a device without room picks nothing");
    // The class reserve (WDDM with a display): kept down to the floor, given up by the survival
    // rungs below it (Soft); when nothing fits with the allowance, the last rung on the warm-up's
    // measurement (WarmUp).
    const rt::PlanMargins shared{.base = 32 * kMiB, .reserve = 2 * kGiB, .allowance = 64 * kMiB};
    PlanLadderInputs reserved_in     = automatic(262144, 1);
    reserved_in.class_reserve        = true;
    const std::vector<PlanRung> wddm = pruned(rt::plan_ladder(reserved_in), model);
    const auto wcost                 = model.cost_fn(wddm);
    const std::size_t floor_rung     = static_cast<std::size_t>(
        std::find_if(wddm.begin(), wddm.end(), [](const PlanRung& r) { return r.survival; }) -
        wddm.begin() - 1);
    const std::uint64_t floor_need =
        rt::plan_needed_bytes(*wcost(floor_rung), shared, PlanReserve::Kept);
    const auto kept = rt::select_plan_rung(wddm, wcost, floor_need, shared);
    failures += check(kept && kept->rung == floor_rung && kept->reserve == PlanReserve::Kept,
                      "the last rung above the floor keeps the reserve");
    const auto soft = rt::select_plan_rung(wddm, wcost, floor_need - 1, shared);
    failures += check(soft && soft->reserve == PlanReserve::Soft && soft->rung == floor_rung + 1 &&
                          wddm[soft->rung].survival && wddm[soft->rung].max_context == 8192 &&
                          wddm[soft->rung].cached_states == wddm[floor_rung].cached_states,
                      "without room for the reserve at the floor, the floor itself without it");
    failures += check(soft && wddm[soft->rung].prefill_chunk == 1024,
                      "the floor without the reserve at the widest chunk first");
    // A byte short of that: the states at the floor, then the contexts below it.
    const std::uint64_t soft_need =
        rt::plan_needed_bytes(*wcost(floor_rung + 1), shared, PlanReserve::Soft);
    const auto below = rt::select_plan_rung(wddm, wcost, soft_need - 1, shared);
    failures += check(below && below->rung > floor_rung + 1 && wddm[below->rung].survival,
                      "below the soft floor the survival steps go on");
    const std::uint64_t last_soft =
        rt::plan_needed_bytes(*wcost(wddm.size() - 1), shared, PlanReserve::Soft);
    const auto warm = rt::select_plan_rung(wddm, wcost, last_soft - 1, shared);
    failures += check(warm && warm->reserve == PlanReserve::WarmUp && warm->rung == wddm.size() - 1,
                      "below the allowance the warm-up decides on the last rung");
    // Monotone across the reserve's edge and the warm-up fallback as well.
    previous = wddm.size();
    for (std::uint64_t free = 200 * kMiB; free <= 8 * kGiB; free += 3 * kMiB) {
        const auto pick = rt::select_plan_rung(wddm, wcost, free, shared);
        if (!pick) { continue; }
        failures += check(pick->rung <= previous, "more free memory picked a lower rung (WDDM)");
        previous = pick->rung;
    }
    failures +=
        check(rt::plan_class_reserve(32 * kGiB, true, true, false) == 4 * kGiB &&
                  rt::plan_class_reserve(16 * kGiB, true, true, false) == 2 * kGiB &&
                  rt::plan_class_reserve(16 * kGiB, true, false, false) == kGiB / 2 &&
                  rt::plan_class_reserve(32 * kGiB, false, true, false) == 0 &&
                  rt::plan_class_reserve(128 * kGiB, false, false, true) == 16 * kGiB &&
                  rt::plan_class_reserve(16 * kGiB + 100 * kMiB, true, true, false) == 2 * kGiB,
              "the class reserve: WDDM an 8th/32nd, unified an 8th, by the 256 MiB class");
    return failures;
}

int warm_up_only_moves_down() {
    int failures = 0;
    ByteModel model;
    const std::vector<PlanRung> ladder = pruned(rt::plan_ladder(automatic(262144, 1)), model);
    const rt::PlanMargins margins{.base = 32 * kMiB, .reserve = 0, .allowance = 64 * kMiB};
    const std::uint64_t free = 5 * kGiB;
    const auto chosen        = rt::select_plan_rung(ladder, model.cost_fn(ladder), free, margins);
    failures += check(chosen.has_value(), "a rung fits 5 GiB");
    // The warm-up took less than its allowance: the rung stays, though a higher one now fits.
    const auto kept = rt::verify_plan_rung(ladder, model.cost_fn(ladder, true), free - 30 * kMiB,
                                           margins, *chosen);
    failures += check(kept && kept->rung == chosen->rung, "a rung that still fits stays");
    const auto roomy =
        rt::verify_plan_rung(ladder, model.cost_fn(ladder, true), 40 * kGiB, margins, *chosen);
    failures += check(roomy && roomy->rung == chosen->rung,
                      "re-measuring more free memory never moves the plan up");
    // It took 300 MiB more: the first lower rung that fits.
    const auto moved = rt::verify_plan_rung(ladder, model.cost_fn(ladder, true),
                                            free - 300 * kMiB - 64 * kMiB, margins, *chosen);
    failures += check(moved && moved->rung > chosen->rung, "a rung that no longer fits moves down");
    return failures;
}

// The Engine's sequence (model_instance.cpp): select on the free memory after the weights, hold
// the graphs from the start on a WarmUp admission (plan_graphs_held), then the warm-up loop's
// first check (plan_warm_up_check) after the prefill-width probe took its 6-40 MiB, and again
// once a decode-graph probe fell short (the shortfall counted as taken).
int select_then_check_pass_zero() {
    int failures = 0;
    // A cost of the Engine's two planners: with the graph allowance until held, held after.
    const auto pass_cost = [](const ByteModel& model, const std::vector<PlanRung>& ladder,
                              PlanSelection chosen, bool probed) {
        return model.cost_fn(ladder, rt::plan_graphs_held(chosen, probed));
    };
    // An asked plan on a GPU Windows shares with a display (Gemma 4 31B IQ3_XXS + MTP at
    // --max-context 4096 --max-concurrency 4 on an RTX 5080, 16 GB: a 2 GiB reserve). Nothing
    // fits, with the reserve or without it (the survival rungs) on the planner's graph bound; the
    // last rung fits on the held graphs: WarmUp, served as master served it.
    {
        ByteModel model;
        model.concurrency      = 4;
        model.state_bytes      = 40 * kMiB;
        model.workspace_bytes  = 600 * kMiB;
        model.graph_lane       = 48 * kMiB; // the planner's bound, about twice the measured
        model.graph_real_lane  = 20 * kMiB;
        PlanLadderInputs asked = automatic(4096, 4);
        asked.context_pinned   = true;
        asked.class_reserve                = true;
        const std::vector<PlanRung> ladder = pruned(rt::plan_ladder(asked), model);
        const rt::PlanMargins wddm{
            .base = 32 * kMiB, .reserve = 2 * kGiB, .allowance = rt::kPlanWarmUpAllowanceBytes};
        const std::size_t last = ladder.size() - 1;
        const std::uint64_t held_need =
            rt::plan_needed_bytes(*model.cost_fn(ladder)(last), wddm, PlanReserve::WarmUp);
        const std::uint64_t free = held_need + 30 * kMiB; // 0.1 GiB short of the graph bound
        const auto selected      = rt::select_plan_rung(ladder, model.cost_fn(ladder), free, wddm);
        failures +=
            check(selected && selected->rung == last && selected->reserve == PlanReserve::WarmUp,
                  "the asked plan is admitted on the warm-up's measurement");
        failures += check(selected && rt::plan_graphs_held(*selected, false),
                          "a WarmUp admission holds the graphs from the warm-up's first pass");
        // Pass 0: the prefill-width probe took 20 MiB for good (kernel code, stack).
        const auto pass0 =
            rt::plan_warm_up_check(ladder, pass_cost(model, ladder, *selected, false),
                                   free - 20 * kMiB, 0, wddm, *selected);
        failures += check(pass0 && *pass0 == *selected,
                          "pass 0 keeps the WarmUp rung on its held bytes (no allowance, no "
                          "class reserve added back)");
        // The old check: the graph allowance and the reserve added back refuse it before any
        // graph is measured.
        failures += check(
            !rt::select_plan_rung(
                ladder, model.cost_fn(ladder), free - 20 * kMiB,
                rt::PlanMargins{.base = wddm.base, .reserve = wddm.reserve, .allowance = 0}, last),
            "the Kept tier on pass 0 would refuse the asked plan");
        // More than the slack taken (a probe's shortfall): the last rung is refused.
        failures += check(!rt::plan_warm_up_check(ladder, pass_cost(model, ladder, *selected, true),
                                                  free, 128 * kMiB, wddm, *selected),
                          "a WarmUp rung the probe outgrew is refused");
    }
    // An automatic plan (Linux): pass 0 keeps the rung (the probe took less than the allowance),
    // a probe shortfall moves it down, never up.
    {
        ByteModel model;
        const std::vector<PlanRung> ladder = pruned(rt::plan_ladder(automatic(262144, 1)), model);
        const rt::PlanMargins headless{.base = 32 * kMiB, .reserve = 0, .allowance = 64 * kMiB};
        const std::uint64_t free = 5 * kGiB;
        const auto selected = rt::select_plan_rung(ladder, model.cost_fn(ladder), free, headless);
        failures += check(selected && selected->reserve == PlanReserve::Kept &&
                              !rt::plan_graphs_held(*selected, false),
                          "a Kept admission keeps the graph allowance until a probe");
        const auto pass0 =
            rt::plan_warm_up_check(ladder, pass_cost(model, ladder, *selected, false),
                                   free - 40 * kMiB, 0, headless, *selected);
        failures += check(pass0 && pass0->rung == selected->rung,
                          "pass 0 keeps a rung the prefill probe took 40 MiB of");
        const auto shortfall =
            rt::plan_warm_up_check(ladder, pass_cost(model, ladder, *selected, false),
                                   free - 40 * kMiB, 512 * kMiB, headless, *selected);
        failures += check(shortfall && shortfall->rung > selected->rung,
                          "a probe's shortfall moves the plan down");
        const auto after = rt::plan_warm_up_check(ladder, pass_cost(model, ladder, *selected, true),
                                                  40 * kGiB, 0, headless, *selected);
        failures += check(after && after->rung == selected->rung,
                          "a later pass with more free memory never moves the plan up");
    }
    // A desktop on a small card, automatic context: the floor without the reserve before 7,808.
    {
        ByteModel model;
        PlanLadderInputs in                = automatic(262144, 1);
        in.class_reserve                   = true;
        const std::vector<PlanRung> ladder = pruned(rt::plan_ladder(in), model);
        const rt::PlanMargins wddm{.base = 32 * kMiB, .reserve = kGiB, .allowance = 64 * kMiB};
        const std::size_t floor_soft =
            static_cast<std::size_t>(std::find_if(ladder.begin(), ladder.end(),
                                                  [](const PlanRung& r) { return r.survival; }) -
                                     ladder.begin());
        const std::uint64_t free =
            rt::plan_needed_bytes(*model.cost_fn(ladder)(floor_soft), wddm, PlanReserve::Soft);
        const auto selected = rt::select_plan_rung(ladder, model.cost_fn(ladder), free, wddm);
        failures += check(selected && selected->rung == floor_soft &&
                              selected->reserve == PlanReserve::Soft &&
                              ladder[selected->rung].max_context == 8192,
                          "the floor admitted Soft comes first in the survival range");
        const auto pass0 =
            rt::plan_warm_up_check(ladder, pass_cost(model, ladder, *selected, false),
                                   free - 40 * kMiB, 0, wddm, *selected);
        failures += check(pass0 && *pass0 == *selected, "pass 0 keeps the Soft floor");
    }
    return failures;
}

int pins() {
    int failures = 0;
    // --max-context: every rung keeps it; the pool, the states and the chunk step.
    PlanLadderInputs context          = automatic(32768, 4);
    context.context_pinned            = true;
    const std::vector<PlanRung> asked = rt::plan_ladder(context);
    failures += check(std::all_of(asked.begin(), asked.end(),
                                  [](const PlanRung& r) { return r.max_context == 32768; }) &&
                          asked.front().kv_pages == 4 * 512 && asked.back().kv_pages == 512 &&
                          asked.back().prefill_chunk == 128 && asked.back().cached_states == 1,
                      "--max-context pins the context");
    // --kv-capacity: the pool is pinned and bounds the context.
    PlanLadderInputs pool              = automatic(65536, 2);
    pool.kv_tokens_pinned              = 65536;
    const std::vector<PlanRung> pooled = rt::plan_ladder(pool);
    failures += check(
        std::all_of(pooled.begin(), pooled.end(),
                    [](const PlanRung& r) { return r.kv_pages == 1024 && r.max_context == 65536; }),
        "--kv-capacity pins the pool and the context");
    // --device-state-slots and --prefill-chunk: every rung keeps them.
    PlanLadderInputs states           = automatic(262144, 2);
    states.states_pinned              = 3;
    states.chunk_pinned               = true;
    const std::vector<PlanRung> fixed = rt::plan_ladder(states);
    failures += check(std::all_of(fixed.begin(), fixed.end(),
                                  [](const PlanRung& r) {
                                      return r.cached_states == 3 && r.prefill_chunk == 1024;
                                  }),
                      "--device-state-slots and --prefill-chunk pin their axes");
    // Everything pinned: one rung, which fits or is refused.
    PlanLadderInputs all = automatic(32768, 1);
    all.context_pinned   = true;
    all.kv_tokens_pinned = 32768;
    all.states_pinned    = 1;
    all.chunk_pinned     = true;
    failures += check(rt::plan_ladder(all).size() == 1, "every axis pinned: one rung");
    return failures;
}

// ---- Scenarios -----------------------------------------------------------------------------
//
// The rows of the plan-levels review (card x model x options; GiB, the review's numbers, ±15 %)
// and four measured rows of an RTX 5070 Ti (16 GB, Linux, headless; the 21.10 stand logs: free
// after weights and master's served plan). For each: master's continuous fit (the longest
// 64-token context that fits after the warm-up, the pool every remaining page, the WDDM reserve
// hard for the context above the floor and soft below it, as kv_capacity.cpp did) and the ladder
// (selected before the warm-up with the declared allowance, then checked after it).

struct Model {
    const char* name;
    std::uint64_t page_bytes; // one 64-token page of every full-attention layer's KV
    std::uint64_t state_bytes;
    std::uint64_t workspace_bytes; // at chunk 1024
    std::uint64_t workspace_lane;
    std::uint32_t native;
};

struct Scenario {
    const char* row;
    Model model;
    double free_gib;    // free after the weights (and a separate drafter), before the warm-up
    double lasting_mib; // what the warm-up kept for good (kernel code, stack)
    double reserve_gib; // the class reserve (WDDM), 0 elsewhere
    std::uint32_t lanes;
    std::uint32_t context; // 0: automatic
    std::uint32_t states_minimum;
    bool speculation;
};

struct Fit {
    std::uint32_t context   = 0;
    std::uint64_t kv_tokens = 0;
    std::uint32_t states    = 0;
    std::uint32_t chunk     = 0;
    std::uint32_t rung      = 0;
    std::uint32_t rungs     = 0;
    std::string reserve;
    bool wider_chunk_fits = false; // a Soft floor whose wider chunk would fit without the reserve
};

std::uint64_t pages_of(std::uint32_t context) { return (context + 63ULL) / 64ULL; }

Fit master_fit(const Scenario& s, bool reserve_soft_below_floor = true,
               bool keep_states_to_floor = false) {
    ByteModel m{.page_bytes      = s.model.page_bytes,
                .state_bytes     = s.model.state_bytes,
                .workspace_bytes = s.model.workspace_bytes,
                .workspace_lane  = s.model.workspace_lane,
                .concurrency     = s.lanes};
    const auto free = static_cast<std::uint64_t>((s.free_gib * 1024.0 - s.lasting_mib) * kMiB);
    const std::uint64_t base = 32 * kMiB;
    const std::uint64_t reserve =
        std::max<std::uint64_t>(base, static_cast<std::uint64_t>(s.reserve_gib * kGiB));
    const std::uint64_t graphs = m.graph_real_lane * s.lanes;
    const auto fits = [&](std::uint32_t context, std::uint32_t states, std::uint32_t chunk,
                          std::uint64_t headroom) {
        const std::uint64_t pages = std::max<std::uint64_t>(pages_of(context), s.lanes);
        return m.fixed(states, chunk) + graphs + pages * m.page_bytes + headroom <= free;
    };
    Fit out;
    std::uint32_t chunk  = 1024;
    std::uint32_t states = s.lanes; // the context is fitted with one cached state per lane
    if (s.context != 0) {
        out.context = s.context;
        while (chunk > 128 && !fits(s.context, states, chunk, base)) { chunk /= 2; }
    } else {
        const auto longest = [&](std::uint32_t st, std::uint32_t ch, std::uint64_t headroom) {
            if (fits(s.model.native, st, ch, headroom)) { return s.model.native; }
            std::uint32_t best = 0;
            for (std::uint32_t k = 1024; k < s.model.native; k += 64) {
                if (fits(k, st, ch, headroom)) {
                    best = k;
                } else {
                    break;
                }
            }
            return best;
        };
        out.context = longest(states, chunk, reserve);
        if (out.context < 8192 && keep_states_to_floor) {
            // The ladder's order: narrower chunks with a state per lane while they reach the
            // floor (the reserve kept), the fewest states only below it.
            for (std::uint32_t ch = 512; ch >= 128 && out.context < 8192; ch /= 2) {
                const std::uint32_t k = longest(states, ch, reserve);
                if (k > out.context) {
                    out.context = k;
                    chunk       = ch;
                }
            }
            if (out.context < 8192) {
                out.context = 0;
                chunk       = 1024;
            }
        }
        if (out.context <
            8192) { // below the floor: the reserve soft, fewer states, narrower chunks
            states = s.states_minimum;
            for (std::uint32_t ch = 1024; ch >= 128; ch /= 2) {
                const std::uint32_t k =
                    longest(states, ch, reserve_soft_below_floor ? base : reserve);
                if (k > out.context) {
                    out.context = k;
                    chunk       = ch;
                }
                if (out.context >= 8192) { break; }
            }
        }
    }
    // The pool: every page left after the reserve (best effort), at most every lane's context.
    const std::uint64_t min_pages = std::max<std::uint64_t>(pages_of(out.context), s.lanes);
    const std::uint64_t used      = m.fixed(states, chunk) + graphs + min_pages * m.page_bytes;
    std::uint64_t pages           = min_pages;
    if (free > used + reserve) {
        pages = std::min<std::uint64_t>(s.lanes * pages_of(out.context),
                                        min_pages + (free - used - reserve) / m.page_bytes);
    }
    out.kv_tokens = pages * 64;
    out.states    = states;
    out.chunk     = chunk;
    return out;
}

Fit ladder_fit(const Scenario& s) {
    ByteModel m{.page_bytes      = s.model.page_bytes,
                .state_bytes     = s.model.state_bytes,
                .workspace_bytes = s.model.workspace_bytes,
                .workspace_lane  = s.model.workspace_lane,
                .concurrency     = s.lanes};
    PlanLadderInputs in    = automatic(s.context != 0 ? s.context : s.model.native, s.lanes);
    in.context_pinned      = s.context != 0;
    in.states_minimum      = s.states_minimum;
    in.last_resort_context = s.speculation ? 0 : 64;
    in.class_reserve                   = s.reserve_gib > 0;
    const std::vector<PlanRung> ladder = pruned(rt::plan_ladder(in), m);
    const rt::PlanMargins margins{.base      = 32 * kMiB,
                                  .reserve   = static_cast<std::uint64_t>(s.reserve_gib * kGiB),
                                  .allowance = rt::kPlanWarmUpAllowanceBytes};
    const auto free = static_cast<std::uint64_t>(s.free_gib * kGiB);
    Fit out;
    out.rungs = static_cast<std::uint32_t>(ladder.size());
    auto pick = rt::select_plan_rung(ladder, m.cost_fn(ladder), free, margins);
    if (!pick) { return out; }
    pick = rt::verify_plan_rung(ladder, m.cost_fn(ladder, true),
                                free - static_cast<std::uint64_t>(s.lasting_mib * kMiB), margins,
                                *pick);
    if (!pick) { return out; }
    out.context   = ladder[pick->rung].max_context;
    out.kv_tokens = std::uint64_t{ladder[pick->rung].kv_pages} * 64;
    out.states    = ladder[pick->rung].cached_states;
    out.chunk     = ladder[pick->rung].prefill_chunk;
    out.rung      = static_cast<std::uint32_t>(pick->rung + 1);
    out.reserve   = std::string(rt::plan_reserve_name(pick->reserve));
    if (pick->reserve == PlanReserve::Soft && s.context == 0 && out.context == 8192) {
        PlanRung wider = ladder[pick->rung];
        for (std::uint32_t chunk = 1024; chunk > out.chunk; chunk /= 2) {
            wider.prefill_chunk = chunk;
            out.wider_chunk_fits =
                out.wider_chunk_fits ||
                rt::plan_needed_bytes(m.cost(wider), margins, PlanReserve::Soft) <= free;
        }
    }
    return out;
}

// The continuous fit of the ladder's own byte model and margins (the allowance, the graphs'
// bound, the class reserve kept down to the floor): the best the grid could do.
Fit continuous_fit(const Scenario& s) {
    Scenario with_allowance = s;
    with_allowance.lasting_mib =
        rt::kPlanWarmUpAllowanceBytes / static_cast<double>(kMiB) + (12.0 - 6.0) * s.lanes;
    return master_fit(with_allowance, false, true);
}

// The selection is monotone in the free memory: as it grows, the selected (context, KV tokens,
// cached states, chunk) never decreases on any axis, for an automatic and a pinned context, with a
// class reserve (WDDM, WSL) and without (Linux), with a Host tier and without (round-3 review:
// 27B C4 pinned at 3.65 GiB served chunk 128 with the reserve kept, poorer than 3.55 GiB without
// it). The warm-up's check moves a rung down by what the warm-up kept for good, which differs per
// rung, so its content is checked on the held graphs only; what is checked at every pass is that a
// free size served is never followed by a larger one refused, also when the warm-up kept more
// than its allowance before the graphs were measured (round-4 review: 696-700 MiB selected Soft
// and refused, 690-694 MiB served WarmUp).
int sweep_free_memory(const ByteModel& m, const PlanLadderInputs& in,
                      const rt::PlanMargins& margins, const std::string& what) {
    int failures                       = 0;
    const std::vector<PlanRung> ladder = pruned(rt::plan_ladder(in), m);
    std::optional<PlanRung> selected_before;
    std::optional<PlanRung> checked_before;
    bool served_first = false; // the first check after the prefill probe, 70 MiB kept
    bool served_held  = false; // a later check on the held graphs, 30 MiB kept
    for (std::uint64_t free = 256 * kMiB; free <= 24 * kGiB; free += 2 * kMiB) {
        const std::string at = what + " at " + std::to_string(free / kMiB) + " MiB";
        const auto pick      = rt::select_plan_rung(ladder, m.cost_fn(ladder), free, margins);
        if (!pick) {
            failures += check(!selected_before, at + ": more memory refused");
            continue;
        }
        const PlanRung& rung = ladder[pick->rung];
        if (selected_before && !content_at_most(*selected_before, rung)) {
            failures += check(false, at + ": the selection serves less");
        }
        selected_before = rung;
        // The first check: the probe kept 70 MiB, more than the 64 MiB allowance, before any graph
        // was measured (the graphs held only for a WarmUp admission).
        const auto first =
            rt::plan_warm_up_check(ladder, m.cost_fn(ladder, rt::plan_graphs_held(*pick, false)),
                                   free, 70 * kMiB, margins, *pick);
        if (first) {
            served_first = true;
        } else {
            failures += check(!served_first, at + ": the first check refuses after a smaller free "
                                                  "size was served");
        }
        // A later check on the held graphs (30 MiB kept for good).
        const auto kept =
            rt::plan_warm_up_check(ladder, m.cost_fn(ladder, rt::plan_graphs_held(*pick, true)),
                                   free, 30 * kMiB, margins, *pick);
        if (!kept) {
            failures +=
                check(!served_held, at + ": the check refuses after a smaller free size served");
            continue;
        }
        served_held            = true;
        const PlanRung& served = ladder[kept->rung];
        if (checked_before && !content_at_most(*checked_before, served)) {
            failures += check(false, at + ": the held check serves less");
        }
        checked_before = served;
    }
    return failures;
}

int content_monotone_in_free_memory() {
    const Model q9{"9B", 8ULL * 4 * 528 * 64, 24 * kMiB, 512 * kMiB, 0, 262144};
    const Model q27{"27B", 16ULL * 4 * 528 * 64, 75 * kMiB, 450 * kMiB, 30 * kMiB, 262144};
    const Model q35{"35B-A3B", 10ULL * 2 * 528 * 64, 31 * kMiB, 75 * kMiB, 4 * kMiB, 262144};
    struct Config {
        const Model* model;
        std::uint32_t lanes;
        std::uint32_t context; // 0: automatic
    };

    const std::vector<Config> configs{{&q27, 4, 32768}, {&q27, 4, 0}, {&q27, 1, 0},    {&q9, 1, 0},
                                      {&q9, 8, 0},      {&q35, 4, 0}, {&q9, 1, 131072}};
    int failures = 0;
    for (const Config& c : configs) {
        for (const double reserve_gib : {0.0, 1.0, 2.0}) {
            for (const std::uint32_t states_minimum : {0U, 1U}) {
                const ByteModel m{.page_bytes      = c.model->page_bytes,
                                  .state_bytes     = c.model->state_bytes,
                                  .workspace_bytes = c.model->workspace_bytes,
                                  .workspace_lane  = c.model->workspace_lane,
                                  .concurrency     = c.lanes};
                PlanLadderInputs in =
                    automatic(c.context != 0 ? c.context : c.model->native, c.lanes);
                in.context_pinned = c.context != 0;
                in.states_minimum = states_minimum;
                in.class_reserve  = reserve_gib > 0;
                const rt::PlanMargins margins{.base = 32 * kMiB,
                                              .reserve =
                                                  static_cast<std::uint64_t>(reserve_gib * kGiB),
                                              .allowance = rt::kPlanWarmUpAllowanceBytes};
                failures += sweep_free_memory(
                    m, in, margins,
                    std::string(c.model->name) + " C" + std::to_string(c.lanes) + " ctx " +
                        std::to_string(c.context) + " reserve " + std::to_string(reserve_gib) +
                        " min states " + std::to_string(states_minimum));
            }
        }
    }
    // The round-4 cliff: --max-context 4096 at C4 on a 16 GB WDDM card with a display (a 2 GiB
    // reserve), the planner's graph bound 48 MiB a lane against 20 measured.
    ByteModel cliff;
    cliff.page_bytes        = 16ULL * 4 * 528 * 64;
    cliff.state_bytes       = 40 * kMiB;
    cliff.workspace_bytes   = 600 * kMiB;
    cliff.graph_lane        = 48 * kMiB;
    cliff.graph_real_lane   = 20 * kMiB;
    cliff.concurrency       = 4;
    PlanLadderInputs pinned = automatic(4096, 4);
    pinned.context_pinned   = true;
    pinned.class_reserve    = true;
    failures += sweep_free_memory(cliff, pinned,
                                  rt::PlanMargins{.base      = 32 * kMiB,
                                                  .reserve   = 2 * kGiB,
                                                  .allowance = rt::kPlanWarmUpAllowanceBytes},
                                  "the round-4 cliff (4x4096, WDDM)");
    return failures;
}

int scenarios() {
    // Bytes per 64-token page: full-attention layers x KV heads x bytes per token and head x 64.
    const Model q9{"9B", 8ULL * 4 * 528 * 64, 24 * kMiB, 512 * kMiB, 0, 262144};
    const Model q27{"27B", 16ULL * 4 * 528 * 64, 75 * kMiB, 450 * kMiB, 30 * kMiB, 262144};
    const Model q35{"35B-A3B", 10ULL * 2 * 528 * 64, 31 * kMiB, 75 * kMiB, 4 * kMiB, 262144};
    const Model oss{"gpt-oss-20b", 12ULL * 8 * 64 * 2 * 64, 0, 400 * kMiB, 0, 131072};
    const std::vector<Scenario> rows{
        // (a) nothing specified, C = 1 (free = class - weights - CUDA context - desktop).
        {"a1 8G Linux H 9B Q4", q9, 2.37, 30, 0, 1, 0, 0, true},
        {"a2 8G Linux D 9B Q4", q9, 1.77, 30, 0, 1, 0, 0, true},
        {"a3 8G WSL D 9B Q4", q9, 1.39, 30, 1.0, 1, 0, 0, true},
        {"a4 12G Linux H 9B Q8", q9, 2.80, 30, 0, 1, 0, 0, true},
        {"a5 12G Linux H 27B IQ3", q27, 1.17, 30, 0, 1, 0, 0, true},
        {"a7 16G Linux H 27B IQ3", q27, 5.17, 30, 0, 1, 0, 0, true},
        {"a8 16G Linux D 27B IQ3", q27, 4.57, 30, 0, 1, 0, 0, true},
        {"a9 16G WSL D 27B IQ3", q27, 4.17, 30, 2.0, 1, 0, 0, true},
        {"a10 24G Linux H 27B Q4", q27, 8.27, 30, 0, 1, 0, 0, true},
        {"a11 24G Linux D 27B Q4", q27, 7.60, 30, 0, 1, 0, 0, true},
        {"a12 24G WSL D 27B Q4", q27, 7.10, 30, 3.0, 1, 0, 0, true},
        {"a14 32G WSL D 27B Q4", q27, 14.5, 30, 4.0, 1, 0, 0, true},
        {"a18 16G Linux H oss+DFlash", oss, 2.09, 30, 0, 1, 0, 0, true},
        // (b) --max-context 131072 on small cards.
        {"b1 8G Linux H 9B Q4 131K", q9, 2.36, 30, 0, 1, 131072, 0, true},
        {"b3 16G Linux D 27B IQ3 131K", q27, 4.57, 30, 0, 1, 131072, 0, true},
        // (c) --max-concurrency 8, automatic context.
        {"c1 24G Linux H 9B Q4 C8", q9, 18.4, 76, 0, 8, 0, 0, true},
        {"c2 24G WSL D 9B Q4 C8", q9, 17.2, 76, 3.0, 8, 0, 0, true},
        {"c3 24G Linux H 27B Q4 C8", q27, 8.27, 76, 0, 8, 0, 0, true},
        {"c4 80G A100 27B Q4 C8", q27, 63.0, 76, 0, 8, 0, 0, true},
        {"c5 32G WSL D 9B Q4 C8", q9, 24.3, 76, 4.0, 8, 0, 0, true},
        // (d) --max-context 32768 --max-concurrency 4.
        {"d1 16G Linux H 27B IQ3 4x32K", q27, 5.17, 76, 0, 4, 32768, 0, true},
        {"d2 16G Linux D 27B IQ3 4x32K", q27, 4.57, 76, 0, 4, 32768, 0, true},
        {"d3 16G WSL D 27B IQ3 4x32K", q27, 4.17, 76, 2.0, 4, 32768, 0, true},
        {"d4 32G WSL D 27B NVFP4 8x64K", q27, 13.9, 76, 4.0, 8, 65536, 0, true},
        // The RTX 5070 Ti stand rows (21.10 logs): free after weights as measured.
        {"s1 5070Ti 27B IQ3 C1 (144,448)", q27, 5.21, 30, 0, 1, 0, 0, true},
        {"s2 5070Ti 27B IQ3 C4 (126,720)", q27, 5.17, 76, 0, 4, 0, 0, true},
        {"s3 5070Ti 35B IQ3 C1 (262,144)", q35, 3.15, 20, 0, 1, 0, 0, false},
        {"s4 5070Ti 35B IQ3 C4 (262,144)", q35, 3.15, 26, 0, 4, 0, 0, false},
        // Round-2 review: the reserve given up. --max-context 32768 x 4 on a 16 GB WDDM card
        // with a display (a 2 GiB reserve), with a Host tier (0) and without one (1); a 9B on an
        // 8 GB one (a 1 GiB reserve), automatic context.
        {"r1 16G WSL D 27B 4x32K 2.0 host", q27, 2.0, 76, 2.0, 4, 32768, 0, true},
        {"r2 16G WSL D 27B 4x32K 2.8 host", q27, 2.8, 76, 2.0, 4, 32768, 0, true},
        {"r3 16G WSL D 27B 4x32K 3.6 host", q27, 3.6, 76, 2.0, 4, 32768, 0, true},
        {"r4 16G WSL D 27B 4x32K 2.0", q27, 2.0, 76, 2.0, 4, 32768, 1, true},
        {"r5 16G WSL D 27B 4x32K 2.8", q27, 2.8, 76, 2.0, 4, 32768, 1, true},
        {"r6 16G WSL D 27B 4x32K 3.6", q27, 3.6, 76, 2.0, 4, 32768, 1, true},
        {"r7 8G WSL D 9B 1.0", q9, 1.0, 30, 1.0, 1, 0, 0, true},
        {"r8 8G WSL D 9B 1.15", q9, 1.15, 30, 1.0, 1, 0, 0, true},
        {"r9 8G WSL D 9B 1.3", q9, 1.3, 30, 1.0, 1, 0, 0, true},
        // Round-3 review: the sweep's 27B C4 pinned edge, and a 35B-A3B on a 32 GB card with a
        // display (a 4 GiB reserve).
        {"r10 16G WSL D 27B 4x32K 3.55 host", q27, 3.55, 76, 2.0, 4, 32768, 0, true},
        {"r11 16G WSL D 27B 4x32K 3.65 host", q27, 3.65, 76, 2.0, 4, 32768, 0, true},
        {"a15 32G WSL D 35B-A3B Q4", q35, 9.5, 30, 4.0, 1, 0, 0, false},
        {"c6 32G WSL D 35B-A3B Q4 C4", q35, 9.5, 76, 4.0, 4, 0, 0, false},
    };
    int failures = 0;
    std::printf("%-32s %8s %9s %3s %4s | %8s %9s | %8s %9s %3s %4s %11s | %6s %6s %6s\n",
                "scenario", "master", "KV", "st", "chnk", "contin.", "KV", "ladder", "KV", "st",
                "chnk", "rung", "ctx/m", "KV/m", "ctx/c");
    for (const Scenario& s : rows) {
        const Fit master        = master_fit(s);
        const Fit cont          = continuous_fit(s);
        const Fit ladder        = ladder_fit(s);
        const double ctx_master = master.context ? double(ladder.context) / master.context : 0;
        const double kv_master = master.kv_tokens ? double(ladder.kv_tokens) / master.kv_tokens : 0;
        const double ctx_cont  = cont.context ? double(ladder.context) / cont.context : 0;
        const std::string rung = std::to_string(ladder.rung) + "/" + std::to_string(ladder.rungs) +
                                 (ladder.reserve == "kept" ? "" : " " + ladder.reserve);
        std::printf(
            "%-32s %8u %9llu %3u %4u | %8u %9llu | %8u %9llu %3u %4u %11s | %6.3f %6.3f %6.3f\n",
            s.row, master.context, static_cast<unsigned long long>(master.kv_tokens), master.states,
            master.chunk, cont.context, static_cast<unsigned long long>(cont.kv_tokens),
            ladder.context, static_cast<unsigned long long>(ladder.kv_tokens), ladder.states,
            ladder.chunk, rung.c_str(), ctx_master, kv_master, ctx_cont);
        // Once the reserve is given up, not the bottom: a Soft floor at the widest chunk that
        // fits, and a pinned context at the highest rung that fits without it.
        failures += check(!ladder.wider_chunk_fits,
                          std::string(s.row) + ": a narrower chunk than fits without the reserve");
        // The floor quality without the reserve: chunk 1024, a state per lane without a Host
        // tier (as master), none with one.
        if (s.row[0] == 'r' && s.context != 0 && s.free_gib >= 2.8) {
            failures += check(ladder.reserve == "soft" && ladder.chunk == 1024 &&
                                  ladder.chunk == master.chunk &&
                                  ladder.states == (s.states_minimum == 0 ? 0U : master.states),
                              std::string(s.row) + ": not the floor quality without the reserve");
        }
        if (s.row[0] == 'r' && s.context == 0 && s.free_gib <= 1.15) {
            failures += check(ladder.context == 8192 && ladder.reserve == "soft",
                              std::string(s.row) + ": not the floor without the reserve");
        }
        // Whatever master served, the ladder serves (no new refusal).
        failures += check(master.context == 0 || ladder.context != 0,
                          std::string(s.row) + ": the ladder refuses what master served");
        // The grid's loss against the continuous fit of the same margins: at most one step of
        // the context (4.4 %) or of the pool when the context is the model's own. (The chunk
        // narrows only at the floor, so a continuous fit at a narrower chunk above it is no
        // rung.)
        if (cont.context != 0 && cont.context >= 8192 && ladder.chunk == cont.chunk) {
            failures += check(ladder.context * 1000ULL >= cont.context * 956ULL,
                              std::string(s.row) + ": more than one grid step below the fit");
        }
    }
    return failures;
}

} // namespace

int main() {
    int failures = scaled_grid() + ladder_shape() + bytes_never_grow() + selection() +
                   warm_up_only_moves_down() + select_then_check_pass_zero() + pins() +
                   content_monotone_in_free_memory() + scenarios();
    if (failures == 0) { std::cout << "plan ladder OK\n"; }
    return failures == 0 ? 0 : 1;
}
