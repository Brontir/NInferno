#pragma once

// The plan ladder (docs/maintainer/deterministic-decisions.md, Plan levels): the startup memory
// plan is one rung of a declared ladder. The rungs come from the artifact (the model's own
// context), the options and the device's discrete facts only; the free Device memory, the one
// observation startup makes, picks the first rung that fits and sizes nothing itself. The rung's
// index and its sizes enter the decision fingerprint's plan section.
//
// The ladder, N the context ceiling (the model's own, under an automatic context's cap and an
// explicit --kv-capacity), C the concurrency, g = 2^(-1/16) (every step keeps about 95.7 % of the
// rung above it, less the rounding to whole pages), F = min(N, kPlanContextFloor). The floor
// quality is the poorest plan admitted with the class reserve kept: the context F (or the pinned
// one), a chunk of kPlanFloorQualityChunk (the configured one when narrower or pinned), and a
// cached Device state per lane without a Host tier, none with one. From the top:
//
//   top       N tokens for every request, a pool of C whole contexts, the default cached Device
//             states (two per lane), the configured prefill chunk;
//   extra     the second cached state of each lane gives way, one at a time, down to one per lane
//             (min(C, default));
//   pool      the pool steps down by g from C contexts to one;
//   states    with a Host tier (the fewest states 0: the Host keeps the StateImages) the cached
//             Device states step down to none; without one, one per lane stays;
//   context   the context steps down by g from N to F, the pool one context;
//   chunk     a configured chunk wider than the floor quality's halves down to it, at F;
//   survival  below the floor quality, admitted without the class reserve (Soft): with a class
//             reserve first the floor quality itself (the same content for less memory), then the
//             chunk halves down to 128, the remaining cached states go (down to 1 without a Host
//             tier), and with an automatic context the context steps on by g below F down to
//             min(N, kPlanSurvivalContext), and, without speculation, one last rung at the
//             shortest context that holds one prompt token and one decode round. A survival rung
//             is served with a warning.
//
// Every step lowers one axis and keeps the others, so the content (context, KV tokens, cached
// states, chunk) never grows down the ladder: with the selection's first fit, more free memory
// never selects less (the warm-up may move a rung down by what it kept, and never refuses what
// less free memory served while the probe's lasting graph memory does not grow with the probed
// rung's context; when it does, a refusal band of a few MiB can appear at the very bottom of the
// ladder, a refusal and never an out-of-memory). An explicit option pins its axis: --max-context the context (also an
// explicit --kv-capacity, which bounds one request), --kv-capacity the pool,
// --device-state-slots the states, --prefill-chunk the chunk; the ladder then steps only the
// others (no context steps with the context pinned). Equal consecutive rungs are listed
// once.
//
// Everything here is pure: no CUDA, no clocks, no files. The Engine passes the costs in.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::runtime {

// The shortest context the context steps reach before the survival rungs (the model's own when
// shorter): kAutomaticContextFloor.
inline constexpr std::uint32_t kPlanContextFloor = 8192;
// The shortest survival context the steps reach (master's automatic minimum, 1024 tokens); only
// the last-resort rung of a plan without speculation is shorter.
inline constexpr std::uint32_t kPlanSurvivalContext = 1024;
// The floor quality's prefill chunk: narrower chunks are survival rungs (the class reserve given
// up first). The configured chunk when it is narrower.
inline constexpr std::uint32_t kPlanFloorQualityChunk = 1024;
// What the kernel warm-up takes for good after the selection's measurement besides the planned
// graphs: kernel code loaded at first launch and the per-thread stack (6-40 MiB measured on an RTX
// 5080; master kept 64 MiB for it in its refusal advice). Declared, not measured.
inline constexpr std::uint64_t kPlanWarmUpAllowanceBytes = std::uint64_t{64} << 20;
// The VRAM class: the device's total memory rounded down to this grain, so the same chip with a
// few MiB more reserved by its firmware keeps one class.
inline constexpr std::uint64_t kPlanVramClassGrainBytes = std::uint64_t{256} << 20;

// The axis a rung stepped from the one above it.
enum class PlanStep : std::uint8_t {
    Top,
    ExtraStates,
    Pool,
    States, // with a Host tier after the pool, else (the rest) in the survival range
    Context,
    Chunk,
    Survival,
    LastResort,
};

[[nodiscard]] std::string_view plan_step_name(PlanStep step) noexcept;

struct PlanRung {
    std::uint32_t max_context   = 0;
    std::uint32_t kv_pages      = 0; // Main KV pages of the pool (64 tokens each)
    std::uint32_t cached_states = 0; // cached Device StateImages beside the active ones
    std::uint32_t prefill_chunk = 0;
    PlanStep step               = PlanStep::Top;
    // Below the floor quality (a survival rung keeps the axis it stepped: "survival chunk"),
    // admitted without the class reserve: served with a warning.
    bool survival = false;

    friend bool operator==(const PlanRung&, const PlanRung&) = default;
};

struct PlanLadderInputs {
    std::uint32_t context_ceiling = 0; // N
    bool context_pinned           = false;
    std::uint32_t concurrency     = 1; // C
    std::optional<std::uint32_t> kv_tokens_pinned;
    std::optional<std::uint32_t> states_pinned;
    std::uint32_t states_default = 0; // the automatic count at the top (two per lane)
    std::uint32_t states_minimum = 1; // 0 with a Host tier
    std::uint32_t prefill_chunk  = 1024;
    bool chunk_pinned            = false;
    // A class reserve applies beside the pool (plan_class_reserve above the base margin: WDDM, a
    // GPU sharing the host's memory). The survival range then starts with the floor quality
    // without it; without one that rung would repeat the one above.
    bool class_reserve = false;
    // The last-resort context (0: none, as with speculation): the shortest that holds one prompt
    // token and one decode round, in whole pages.
    std::uint32_t last_resort_context = 0;
};

// The pages one request's context takes, and the pool bounds of C of them.
[[nodiscard]] std::uint32_t plan_context_pages(std::uint32_t max_context);

// x * 2^(-step/16), rounded down, in exact integer arithmetic (the same on every host).
[[nodiscard]] std::uint32_t plan_scaled(std::uint32_t x, std::uint32_t step) noexcept;

// The ladder's candidates, top first. Pure: they depend only on `inputs`.
[[nodiscard]] std::vector<PlanRung> plan_ladder(const PlanLadderInputs& inputs);

// The ladder: the candidates that take fewer bytes than every rung kept above them (a narrower
// chunk's longer context only where it is cheaper than the floor at the wider chunk; an axis step
// that saves nothing, as cached states of a model without recurrent state, is dropped), so the
// bytes fall strictly down the ladder. The survival range is pruned on its own: its rungs need
// less beside their bytes (no class reserve), so its first rung is kept whatever it costs against
// the rungs above, and the bytes fall strictly within it. The top rung is always kept; a candidate
// the family cannot plan (nullopt) is dropped. Pure: `bytes(i)` is candidate i's Program
// reservation (with the graph allowance), from the artifact, the options and the device's facts.
[[nodiscard]] std::vector<PlanRung>
prune_plan_ladder(const std::vector<PlanRung>& candidates,
                  const std::function<std::optional<std::uint64_t>(std::size_t)>& bytes);

// What a rung costs on the device beside the weights. `bytes`: its Program reservation with the
// CUDA graph allowance (KV, cached states, workspace, graphs); `held_bytes`: the same without the
// allowance, once a warm-up probe has built the graphs and the driver holds their memory (only a
// WarmUp admission reads it, which takes the last rung only: the Engine plans it for that rung
// alone and leaves it 0 elsewhere); `pin`: the page tables pinning the rung's Host tier takes
// (2 MiB per pinned GiB).
struct PlanRungCost {
    std::uint64_t bytes      = 0;
    std::uint64_t held_bytes = 0;
    std::uint64_t pin_bytes  = 0;
};

// nullopt: the rung cannot be planned (the family refuses its shape).
using PlanRungCostFn = std::function<std::optional<PlanRungCost>(std::size_t rung)>;

// What the selection keeps beside a rung. `base`: the driver's own allocations (the automatic KV
// headroom, --kv-auto-headroom-mib; 0 with an explicit --kv-capacity, as before); `reserve`: the
// class reserve of a GPU Windows shares (WDDM, WSL 2) or one sharing the host's memory, which
// includes `base`; `allowance`: what the warm-up takes after the measurement (0 without one).
struct PlanMargins {
    std::uint64_t base      = 0;
    std::uint64_t reserve   = 0;
    std::uint64_t allowance = 0;
};

// The class reserve: under WDDM an 8th of the VRAM class, at most 4 GiB, on a card driving the
// display, a 32nd, at most 1 GiB, on a headless one; on a GPU sharing the host's memory an 8th, at
// least 4 GiB; else 0. The driver model, the display and the integrated flag are discrete facts.
[[nodiscard]] std::uint64_t plan_class_reserve(std::uint64_t total_bytes, bool paravirtual,
                                               bool drives_display, bool integrated) noexcept;

// How a rung was admitted:
//   Kept      with every margin: the class reserve and the warm-up allowance;
//   Soft      a survival rung (at or below the floor quality) without the class reserve (a
//             warning when the reserve is not 0);
//   WarmUp    the last rung without the allowance and the graph allowance, when nothing fits with
//             them: the warm-up measures the graphs and keeps the rung only if it then fits (an
//             asked plan master served on the measured graphs).
enum class PlanReserve : std::uint8_t { Kept, Soft, WarmUp };

[[nodiscard]] std::string_view plan_reserve_name(PlanReserve reserve) noexcept;

struct PlanSelection {
    std::size_t rung    = 0;
    PlanReserve reserve = PlanReserve::Kept;

    friend bool operator==(const PlanSelection&, const PlanSelection&) = default;
};

// The bytes a rung needs from the free memory under an admission (Kept: every margin; Soft:
// without the class reserve; WarmUp: the held bytes beside the base margin only).
[[nodiscard]] std::uint64_t plan_needed_bytes(const PlanRungCost& cost, const PlanMargins& margins,
                                              PlanReserve reserve) noexcept;

// The first rung at or below `from` whose need fits `free_bytes`: one pass, each rung needing its
// bytes, the allowance and (outside the survival range) the class reserve, so more free memory
// never picks a lower rung, and as the content never grows down the ladder, never selects less;
// when none does, the last rung if it fits on the warm-up's measurement (WarmUp). nullopt: none
// fits.
[[nodiscard]] std::optional<PlanSelection>
select_plan_rung(const std::vector<PlanRung>& ladder, const PlanRungCostFn& cost,
                 std::uint64_t free_bytes, const PlanMargins& margins, std::size_t from = 0);

// After the warm-up: the first rung from the chosen one down that fits the measured free memory
// (the allowance taken: what it was for is now measured). The chosen rung keeps the tier it was
// admitted at: a WarmUp rung (the last) needs its held bytes beside the base margin, no class
// reserve, and stays or is refused; a Kept or Soft one, and each rung below it, needs what the
// selection asked less the allowance, and when none of them fits, the last rung is taken on its
// held bytes (WarmUp), as the selection would. Re-measuring never moves the plan up.
[[nodiscard]] std::optional<PlanSelection>
verify_plan_rung(const std::vector<PlanRung>& ladder, const PlanRungCostFn& cost,
                 std::uint64_t free_bytes, const PlanMargins& margins, PlanSelection chosen);

// Whether the warm-up's checks plan the rung with the graphs held (no graph allowance): once a
// probe has built the rung's graphs, and from the first pass on for a WarmUp admission, which was
// made on the held bytes (the probe then measures the graphs; master held them before the
// warm-up as well).
[[nodiscard]] constexpr bool plan_graphs_held(PlanSelection chosen, bool probed) noexcept {
    return probed || chosen.reserve == PlanReserve::WarmUp;
}

// One check of the warm-up loop: the free memory now less what the warm-up took beyond its
// allowance (`taken`: the Host tier's pinning, graph memory a WDDM driver gave back, a probe's
// shortfall), verified from `chosen` down. `cost` plans with the graphs held when
// plan_graphs_held(chosen, probed). nullopt: the measurement refuses the plan.
[[nodiscard]] std::optional<PlanSelection>
plan_warm_up_check(const std::vector<PlanRung>& ladder, const PlanRungCostFn& cost,
                   std::uint64_t free_now, std::uint64_t taken, const PlanMargins& margins,
                   PlanSelection chosen);

// "262,144 tokens x 4 contexts, 8 cached states, chunk 1024" for a rung of C lanes.
[[nodiscard]] std::string describe_plan_rung(const PlanRung& rung, std::uint32_t concurrency);

} // namespace ninferno::runtime
