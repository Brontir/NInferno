#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include "core/per_device.h"
#include "core/device_class.h"
#include "core/weight.h"

#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/rmsnorm.h"
#include "ops/common/column_routes.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

// Widest call the fused 27B norm + gating SIMT kernel serves: the decode/verify widths under the
// prefill route profile, 42 columns otherwise (its measured speed frontier).
inline constexpr std::int32_t kFusedSimt27MaxCols      = 16;
inline constexpr std::int32_t kFusedSimt27SpeedMaxCols = 42;

// The invariant route profile never takes the fused form (bf16_gdn_norm_gating_resolve_plan).
std::int32_t fused_simt27_max_cols() noexcept {
    if (invariant_routes_active()) return 0;
    return prefill_routes_active() ? kFusedSimt27MaxCols : kFusedSimt27SpeedMaxCols;
}

using RouteSpec = ColumnRoute<Bf16GdnGatingScheduleId>;

// The split follows T for speed; the MMA kernel sums K as kBf16GdnCanonicalSegments segments in
// order whatever the split, so a token's g and beta do not follow the size of its chunk.
constexpr std::array<RouteSpec, 6> k27Routes{{
    {1, 1, Bf16GdnGatingScheduleId::GemvPairedRows},
    {2, 8, Bf16GdnGatingScheduleId::SmallTSplit10},
    // As token tiles double, halve SplitK: the band edges are token counts, the same on every
    // board (fitted on the RTX 5090, where they hold the grid near 192 CTAs). Splits 8, 4 and 2
    // sum K as the canonical segments, so an edge moves no byte; the launcher enforces the
    // device's residency (cooperative_from).
    {9, 1024, Bf16GdnGatingScheduleId::MmaCooperativeSplit8},
    {1025, 2048, Bf16GdnGatingScheduleId::MmaCooperativeSplit4},
    {2049, 4096, Bf16GdnGatingScheduleId::MmaCooperativeSplit2},
    {4097, kAnyColumns, Bf16GdnGatingScheduleId::MmaUnsplit},
}};

// Split 16 is wider than the canonical segments, so it sums K its own way. Under the prefill
// route profile it is kept to the decode/verify widths (T <= 16) and every prefill-sized call
// takes a canonical route; otherwise it serves through T=127 (k35SpeedRoutes).
constexpr std::array<RouteSpec, 5> k35Routes{{
    // The same token-count edges (fitted on the RTX 5090, where they hold the grid near 256 CTAs);
    // the launcher enforces the device's residency.
    {1, 16, Bf16GdnGatingScheduleId::MmaCooperativeSplit16},
    {17, 1024, Bf16GdnGatingScheduleId::MmaCooperativeSplit8},
    {1025, 2048, Bf16GdnGatingScheduleId::MmaCooperativeSplit4},
    {2049, 4096, Bf16GdnGatingScheduleId::MmaCooperativeSplit2},
    {4097, kAnyColumns, Bf16GdnGatingScheduleId::MmaUnsplit},
}};

constexpr std::array<RouteSpec, 5> k35SpeedRoutes{{
    {1, 127, Bf16GdnGatingScheduleId::MmaCooperativeSplit16},
    {128, 1024, Bf16GdnGatingScheduleId::MmaCooperativeSplit8},
    {1025, 2048, Bf16GdnGatingScheduleId::MmaCooperativeSplit4},
    {2049, 4096, Bf16GdnGatingScheduleId::MmaCooperativeSplit2},
    {4097, kAnyColumns, Bf16GdnGatingScheduleId::MmaUnsplit},
}};

// The invariant route profile (--activations exact): the MMA routes alone at every width, both
// problems. They sum K as the canonical segments whatever the split, where the decode and verify
// forms (the GEMV, split 10, split 16 and the fused SIMT) sum it their own way: a batched verify of
// more than 16 columns took split 8 and the same row alone split 16, so a token's g and beta, and
// with them the answer, followed its batch neighbours (Qwen3.6-35B + DFlash under exact).
constexpr std::array<RouteSpec, 4> kInvariantRoutes{{
    {1, 1024, Bf16GdnGatingScheduleId::MmaCooperativeSplit8},
    {1025, 2048, Bf16GdnGatingScheduleId::MmaCooperativeSplit4},
    {2049, 4096, Bf16GdnGatingScheduleId::MmaCooperativeSplit2},
    {4097, kAnyColumns, Bf16GdnGatingScheduleId::MmaUnsplit},
}};

static_assert(column_routes_closed(k27Routes));
static_assert(column_routes_closed(kInvariantRoutes));
static_assert(column_routes_closed(k35Routes));
static_assert(column_routes_closed(k35SpeedRoutes));

bool is_27(const Bf16GdnGatingProblem& problem) noexcept {
    return problem.heads == 48 && problem.input_rows == 5120;
}

bool is_35(const Bf16GdnGatingProblem& problem) noexcept {
    return problem.heads == 32 && problem.input_rows == 2048;
}

// Without tensor-core MMA (Volta: the build emulates mma.sync with shuffles and FMAs) the 35B
// decode and verify widths take the warp-per-row SIMT kernel, unfused from the norm, as
// ninfer-v100-sm70 does there.
bool simt_35_decode(const Bf16GdnGatingProblem& problem) noexcept {
    static PerDevice<bool> emulated;
    bool emulated_mma = false;
    try {
        emulated_mma = emulated.get([](int) { return !device_class().features.mma_tensor; });
    } catch (const std::exception&) { return false; }
    return emulated_mma && is_35(problem) && problem.cols <= 16 && !invariant_routes_active();
}

bool schedule_uses_mma(Bf16GdnGatingScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return true;
    case Bf16GdnGatingScheduleId::GemvPairedRows:
    case Bf16GdnGatingScheduleId::SmallTSplit10:
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        return false;
    }
    return false;
}

std::int32_t mma_tile_cols(const Bf16GdnGatingProblem& problem) noexcept {
    return is_35(problem) ? 64 : 128;
}

std::int32_t schedule_split_k(Bf16GdnGatingScheduleId schedule) {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        return 10;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        return 32;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
        return 16;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        return 8;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        return 4;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return 2;
    case Bf16GdnGatingScheduleId::GemvPairedRows:
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return 1;
    }
    throw std::logic_error("BF16 GDN gating: unknown schedule");
}

bool candidate_is_legal(Bf16GdnGatingScheduleId schedule,
                        const Bf16GdnGatingProblem& problem) noexcept {
    if (!bf16_gdn_gating_admits(problem)) { return false; }
    if (is_27(problem)) {
        switch (schedule) {
        case Bf16GdnGatingScheduleId::GemvPairedRows:
            return problem.cols == 1;
        case Bf16GdnGatingScheduleId::SmallTSplit10:
            return problem.cols >= 2 && problem.cols <= 8;
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
            return true;
        case Bf16GdnGatingScheduleId::MmaUnsplit:
            return true;
        case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
            return false;
        }
    }

    switch (schedule) {
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        return problem.cols <= 4 * 65'535;
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        return problem.cols <= 8 * 65'535;
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return true;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return true;
    case Bf16GdnGatingScheduleId::GemvPairedRows:
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        return false;
    }
    return false;
}

std::size_t checked_partial_bytes(std::int32_t heads, std::int32_t split_k, std::int32_t cols) {
    const std::size_t logical_rows = static_cast<std::size_t>(2 * heads);
    const std::size_t split        = static_cast<std::size_t>(split_k);
    const std::size_t tokens       = static_cast<std::size_t>(cols);
    if (tokens > std::numeric_limits<std::size_t>::max() / logical_rows ||
        split > std::numeric_limits<std::size_t>::max() / (tokens * logical_rows)) {
        throw std::overflow_error("BF16 GDN gating workspace element count overflows size_t");
    }
    const std::size_t elements = split * tokens * logical_rows;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("BF16 GDN gating workspace byte count overflows size_t");
    }
    return elements * sizeof(float);
}

void execute_resolved(const Bf16GdnGatingPlan& plan, const Bf16GdnGatingProblem& problem,
                      const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                      const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                      Tensor& beta, DeviceExecutionView execution) {
    auto scratch_scope = ws.scope();
    DeviceSpan scratch{};
    if (plan.workspace_bytes != 0) { scratch = ws.alloc_bytes(plan.workspace_bytes); }
    const auto launch_unsplit = [&] {
        if (is_35(problem)) {
            bf16_gdn_gating_proj_35_mma_unsplit_launch(plan.token_variant, x, a_weight, b_weight,
                                                       A_log, dt_bias, g, beta, execution.stream);
        } else {
            bf16_gdn_gating_proj_mma_unsplit_launch(plan.token_variant, x, a_weight, b_weight,
                                                    A_log, dt_bias, g, beta, execution.stream);
        }
    };
    // One split of K across a cooperative grid; false where the device cannot hold one token tile
    // of it (fewer resident CTAs than heads / 16 x split: a small card, 20 to 30 SMs).
    const auto cooperative = [&](int split) {
        const bool p35 = is_35(problem);
        switch (split) {
        case 32:
            return p35 && bf16_gdn_gating_proj_35_mma_split32_launch(
                              plan.token_variant, x, a_weight, b_weight, A_log, dt_bias,
                              scratch.data, g, beta, execution.multiprocessor_count,
                              execution.stream);
        case 16:
            return p35 && bf16_gdn_gating_proj_35_mma_split16_launch(
                              plan.token_variant, x, a_weight, b_weight, A_log, dt_bias,
                              scratch.data, g, beta, execution.multiprocessor_count,
                              execution.stream);
        case 8:
            return p35 ? bf16_gdn_gating_proj_35_mma_split8_launch(
                             plan.token_variant, x, a_weight, b_weight, A_log, dt_bias,
                             scratch.data, g, beta, execution.multiprocessor_count,
                             execution.stream)
                       : bf16_gdn_gating_proj_mma_split8_launch(
                             plan.token_variant, x, a_weight, b_weight, A_log, dt_bias,
                             scratch.data, g, beta, execution.multiprocessor_count,
                             execution.stream);
        case 4:
            return p35 ? bf16_gdn_gating_proj_35_mma_split4_launch(
                             plan.token_variant, x, a_weight, b_weight, A_log, dt_bias,
                             scratch.data, g, beta, execution.multiprocessor_count,
                             execution.stream)
                       : bf16_gdn_gating_proj_mma_split4_launch(
                             plan.token_variant, x, a_weight, b_weight, A_log, dt_bias,
                             scratch.data, g, beta, execution.multiprocessor_count,
                             execution.stream);
        case 2:
            return p35 ? bf16_gdn_gating_proj_35_mma_split2_launch(
                             plan.token_variant, x, a_weight, b_weight, A_log, dt_bias,
                             scratch.data, g, beta, execution.multiprocessor_count,
                             execution.stream)
                       : bf16_gdn_gating_proj_mma_split2_launch(
                             plan.token_variant, x, a_weight, b_weight, A_log, dt_bias,
                             scratch.data, g, beta, execution.multiprocessor_count,
                             execution.stream);
        default:
            return false;
        }
    };
    // A split the device cannot hold steps down to the next narrower one before the unsplit
    // kernel, which runs heads / 16 CTAs for the whole call (two for the 35B form). Splits 8, 4
    // and 2 sum K as the canonical segments, as the unsplit kernel does, so the step changes no
    // byte of an invariant-profile call; split 16 and 32 serve the speed profile only.
    const auto cooperative_from = [&](int split) {
        for (int s = split; s >= 2; s /= 2) {
            if (cooperative(s)) { return; }
        }
        launch_unsplit();
    };

    switch (plan.schedule) {
    case Bf16GdnGatingScheduleId::GemvPairedRows:
        bf16_gdn_gating_proj_gemv_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta,
                                         execution.stream);
        return;
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        bf16_gdn_gating_proj_small_t_split10_launch(x, a_weight, b_weight, A_log, dt_bias,
                                                    scratch.data, scratch.bytes, g, beta,
                                                    execution.stream);
        return;
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        bf16_gdn_gating_proj_35_simt_c4_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta,
                                               execution.stream);
        return;
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        bf16_gdn_gating_proj_35_simt_c8_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta,
                                               execution.stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        cooperative_from(32);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
        cooperative_from(16);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        cooperative_from(8);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        cooperative_from(4);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        cooperative_from(2);
        return;
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        launch_unsplit();
        return;
    }
    throw std::logic_error("BF16 GDN gating: unknown schedule");
}

template <std::size_t N>
std::size_t route_capacity(const std::array<RouteSpec, N>& routes, const Bf16GdnGatingProblem& base,
                           std::int32_t min_cols, std::int32_t max_cols) {
    std::size_t maximum = 0;
    for (const RouteSpec& route : routes) {
        if (route.last < min_cols || route.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.last, max_cols);
        maximum                     = std::max(maximum, bf16_gdn_gating_resolve_candidate(
                                        route.schedule, {base.heads, base.input_rows, endpoint})
                                                            .workspace_bytes);
    }
    return maximum;
}

} // namespace

bool bf16_gdn_gating_admits(const Bf16GdnGatingProblem& problem) noexcept {
    if (problem.cols < 1) { return false; }
    return is_27(problem) || is_35(problem);
}

Bf16GdnGatingPlan bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId schedule,
                                                    const Bf16GdnGatingProblem& problem) {
    if (!candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("BF16 GDN gating: candidate is not legal for exact problem");
    }
    const bool mma                          = schedule_uses_mma(schedule);
    const Bf16GdnGatingTokenVariant variant = !mma ? Bf16GdnGatingTokenVariant::None
                                                   : ((problem.cols % mma_tile_cols(problem)) == 0
                                                          ? Bf16GdnGatingTokenVariant::Full
                                                          : Bf16GdnGatingTokenVariant::Predicated);
    const std::int32_t split_k              = schedule_split_k(schedule);
    // A split MMA route writes one partial per canonical segment, however few CTAs share K.
    const std::int32_t partials =
        mma ? std::max(split_k, static_cast<std::int32_t>(kBf16GdnCanonicalSegments)) : split_k;
    const std::size_t workspace =
        split_k > 1 ? checked_partial_bytes(problem.heads, partials, problem.cols) : 0;
    return {schedule, variant, workspace};
}

Bf16GdnGatingPlan bf16_gdn_gating_resolve_plan(const Bf16GdnGatingProblem& problem) {
    if (!bf16_gdn_gating_admits(problem)) {
        throw std::invalid_argument(
            "BF16 GDN gating: exact problem or column count is not admitted");
    }
    if (invariant_routes_active()) {
        for (const RouteSpec& route : kInvariantRoutes) {
            if (route.contains(problem.cols)) {
                return bf16_gdn_gating_resolve_candidate(route.schedule, problem);
            }
        }
    } else if (is_27(problem)) {
        for (const RouteSpec& route : k27Routes) {
            if (route.contains(problem.cols)) {
                return bf16_gdn_gating_resolve_candidate(route.schedule, problem);
            }
        }
    } else if (simt_35_decode(problem)) {
        return bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId::SimtWarpRowC8, problem);
    } else {
        for (const RouteSpec& route : prefill_routes_active() ? k35Routes : k35SpeedRoutes) {
            if (route.contains(problem.cols)) {
                return bf16_gdn_gating_resolve_candidate(route.schedule, problem);
            }
        }
    }
    throw std::logic_error("BF16 GDN gating: admitted problem has no covering route");
}

std::size_t bf16_gdn_gating_capacity_workspace_bytes(std::int32_t heads, std::int32_t input_rows,
                                                     std::int32_t min_cols, std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("BF16 GDN gating: invalid column interval");
    }
    const Bf16GdnGatingProblem base{heads, input_rows, 1};
    (void)bf16_gdn_gating_resolve_plan({heads, input_rows, min_cols});
    (void)bf16_gdn_gating_resolve_plan({heads, input_rows, max_cols});
    // For the active route profile: a caller that runs this op under both plans for both.
    if (invariant_routes_active()) {
        return route_capacity(kInvariantRoutes, base, min_cols, max_cols);
    }
    if (is_27(base)) { return route_capacity(k27Routes, base, min_cols, max_cols); }
    const auto& routes = prefill_routes_active() ? k35Routes : k35SpeedRoutes;
    // Without tensor-core MMA the widths up to 16 leave the route table for the SIMT kernel.
    if (simt_35_decode({heads, input_rows, min_cols})) {
        std::size_t bytes = 0;
        for (std::int32_t cols = min_cols; cols <= std::min<std::int32_t>(max_cols, 16); ++cols) {
            bytes = std::max(bytes, bf16_gdn_gating_resolve_candidate(
                                        Bf16GdnGatingScheduleId::SimtWarpRowC8,
                                        {heads, input_rows, cols})
                                        .workspace_bytes);
        }
        return max_cols <= 16 ? bytes
                              : std::max(bytes, route_capacity(routes, base, 17, max_cols));
    }
    return route_capacity(routes, base, min_cols, max_cols);
}

Bf16GdnNormGatingPlan bf16_gdn_norm_gating_resolve_plan(const Bf16GdnGatingProblem& problem) {
    Bf16GdnGatingPlan control            = bf16_gdn_gating_resolve_plan(problem);
    Bf16GdnNormGatingScheduleId schedule = Bf16GdnNormGatingScheduleId::Composed;
    std::int32_t norm_splits             = 0;
    // The fused SIMT form sums in another order than the split-8 MMA; under the prefill route
    // profile it is kept to the decode/verify widths so that every prefill-sized call computes a
    // token's g and beta alike.
    if (is_27(problem) && problem.cols <= fused_simt27_max_cols())
        return {Bf16GdnNormGatingScheduleId::FusedSimt27, control, 0};
    if (is_35(problem) && problem.cols <= 16 && !invariant_routes_active() &&
        !simt_35_decode(problem)) {
        control  = bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId::MmaCooperativeSplit32,
                                                     problem);
        schedule = Bf16GdnNormGatingScheduleId::MmaCooperativeSplit32;
        norm_splits = 32;
    }
    const std::size_t norm_partial_bytes =
        static_cast<std::size_t>(norm_splits) * problem.cols * sizeof(float);
    return {schedule, control, control.workspace_bytes + norm_partial_bytes};
}

std::size_t bf16_gdn_norm_gating_capacity_workspace_bytes(std::int32_t heads,
                                                          std::int32_t input_rows,
                                                          std::int32_t min_cols,
                                                          std::int32_t max_cols) {
    std::size_t maximum =
        bf16_gdn_gating_capacity_workspace_bytes(heads, input_rows, min_cols, max_cols);
    if (invariant_routes_active()) return maximum; // composed at every width
    if (heads == 48 && input_rows == 5120) {
        const std::int32_t fused_last = fused_simt27_max_cols();
        if (max_cols <= fused_last) return 0;
        return bf16_gdn_gating_capacity_workspace_bytes(
            heads, input_rows, std::max(min_cols, fused_last + 1), max_cols);
    }
    if (heads == 32 && input_rows == 2048 && min_cols <= 16) {
        const std::int32_t fused_cols = std::min<std::int32_t>(max_cols, 16);
        maximum                       = std::max(
            maximum,
            bf16_gdn_norm_gating_resolve_plan({heads, input_rows, fused_cols}).workspace_bytes);
    }
    return maximum;
}

void bf16_gdn_gating_execute_plan(const Bf16GdnGatingPlan& plan, const Tensor& x,
                                  const Weight& a_weight, const Weight& b_weight,
                                  const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                  Tensor& g, Tensor& beta, DeviceExecutionView execution) {
    const Bf16GdnGatingProblem problem{g.ne[0], x.ne[0], x.ne[1]};
    const Bf16GdnGatingPlan resolved = bf16_gdn_gating_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.token_variant != plan.token_variant ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("BF16 GDN gating: plan does not match the exact problem");
    }
    execute_resolved(plan, problem, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, execution);
}

void bf16_gdn_gating_dispatch(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                              const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                              Tensor& g, Tensor& beta, DeviceExecutionView execution) {
    const Bf16GdnGatingPlan plan = bf16_gdn_gating_resolve_plan({g.ne[0], x.ne[0], x.ne[1]});
    bf16_gdn_gating_execute_plan(plan, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta,
                                 execution);
}

void bf16_gdn_norm_gating_dispatch(const Tensor& x, const Tensor& norm_weight, float eps, Tensor& h,
                                   const Weight& a_weight, const Weight& b_weight,
                                   const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                   Tensor& g, Tensor& beta, DeviceExecutionView execution) {
    const Bf16GdnGatingProblem problem{g.ne[0], x.ne[0], x.ne[1]};
    const Bf16GdnNormGatingPlan plan = bf16_gdn_norm_gating_resolve_plan(problem);
    if (plan.schedule == Bf16GdnNormGatingScheduleId::FusedSimt27) {
        bf16_gdn_norm_gating_proj_27_launch(x, norm_weight, eps, h, a_weight, b_weight, A_log,
                                            dt_bias, g, beta, execution.stream);
        return;
    }
    if (plan.schedule == Bf16GdnNormGatingScheduleId::Composed) {
        rmsnorm(x, norm_weight, eps, true, h, execution.stream);
        execute_resolved(plan.control, problem, h, a_weight, b_weight, A_log, dt_bias, ws, g, beta,
                         execution);
        return;
    }

    auto scratch_scope = ws.scope();
    DeviceSpan scratch{};
    if (plan.workspace_bytes != 0) { scratch = ws.alloc_bytes(plan.workspace_bytes); }
    if (!bf16_gdn_norm_gating_proj_35_mma_split32_launch(
            plan.control.token_variant, x, norm_weight, eps, h, a_weight, b_weight, A_log, dt_bias,
            scratch.data, g, beta, execution.multiprocessor_count, execution.stream)) {
        rmsnorm(x, norm_weight, eps, true, h, execution.stream);
        const Bf16GdnGatingPlan fallback =
            bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId::MmaUnsplit, problem);
        execute_resolved(fallback, problem, h, a_weight, b_weight, A_log, dt_bias, ws, g, beta,
                         execution);
    }
}

} // namespace ninferno::ops::detail
