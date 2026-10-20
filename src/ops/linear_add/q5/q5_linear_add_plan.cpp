#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "core/weight.h"

#include "core/device_class.h"
#include "core/measured_boards.h"
#include "ops/common/column_routes.h"
#include "ops/common/small_t_bands.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "ninferno/ops/prefill_routes.h"

#include <array>
#include <cstdlib>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

struct SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
};

using RouteSpec = ColumnRoute<Q5LinearAddScheduleId>;

constexpr std::array<SupportSpec, 2> kSupports{{
    {5120, 6144, 6144},
    {5120, 17408, 17408},
}};

constexpr std::array<RouteSpec, 14> kK6144Routes{{
    {1, 1, Q5LinearAddScheduleId::Split2ExactResidual},
    {2, 4, Q5LinearAddScheduleId::SlicedR16T8C4W8S2},
    {5, 8, Q5LinearAddScheduleId::SlicedR16T8W4S2},
    {9, 16, Q5LinearAddScheduleId::SlicedR16T16W4S2},
    {17, 24, Q5LinearAddScheduleId::SlicedR16T24W4S2},
    {25, 32, Q5LinearAddScheduleId::SlicedR32T32W4S2},
    {33, 48, Q5LinearAddScheduleId::SlicedR32T24W4S2Pairwise},
    {49, 64, Q5LinearAddScheduleId::SlicedR32T32W4S2},
    {65, 96, Q5LinearAddScheduleId::MmaResidualR32T32K128},
    {97, 128, Q5LinearAddScheduleId::SlicedR32T32W2S2},
    {129, 160, Q5LinearAddScheduleId::SlicedR32T64W2S1},
    {161, 256, Q5LinearAddScheduleId::MmaResidualR32T128},
    {257, 512, Q5LinearAddScheduleId::MmaResidualR64T128},
    {513, kAnyColumns, Q5LinearAddScheduleId::MmaResidualR64T128Tail},
}};

constexpr std::array<RouteSpec, 14> kK17408Routes{{
    {1, 1, Q5LinearAddScheduleId::Split2ExactResidual},
    {2, 4, Q5LinearAddScheduleId::SlicedR16T8C4W8S2},
    {5, 8, Q5LinearAddScheduleId::SlicedR16T8W4S2},
    {9, 16, Q5LinearAddScheduleId::SlicedR16T16W4S2},
    {17, 24, Q5LinearAddScheduleId::SlicedR16T24W4S2},
    {25, 32, Q5LinearAddScheduleId::SlicedR32T32W4S2},
    {33, 48, Q5LinearAddScheduleId::SlicedR32T24W4S2Pairwise},
    {49, 64, Q5LinearAddScheduleId::SlicedR32T32W4S1},
    {65, 96, Q5LinearAddScheduleId::SlicedR32T32W4S1},
    {97, 128, Q5LinearAddScheduleId::SlicedR32T32W2S2},
    {129, 160, Q5LinearAddScheduleId::SlicedR32T64W2S1},
    {161, 256, Q5LinearAddScheduleId::MmaResidualR32T128},
    {257, 512, Q5LinearAddScheduleId::MmaResidualR64T128},
    {513, kAnyColumns, Q5LinearAddScheduleId::MmaResidualR64T128Tail},
}};

static_assert(column_routes_closed(kK6144Routes) && column_routes_closed(kK17408Routes),
              "Q5 LinearAdd routes must be exact, contiguous, and closed");

bool supported_shape(const Q5LinearAddProblem& problem) noexcept {
    for (const SupportSpec& support : kSupports) {
        if (problem.rows == support.rows && problem.k == support.k &&
            problem.padded_k == support.padded_k) {
            return true;
        }
    }
    return false;
}

// Small-T MMA (q5_linear_add_small_t.cu) for verify widths, by chip: variant 1 takes the call, 0
// leaves it to the column routes above. Bands from bench/ops/q5_linear_add_bench (cold L2, graph)
// against the column routes. Off below sm_80, where bf16 MMA is emulated. The prefill
// and invariant profiles keep the column routes.
constexpr SmallTBand kSmallTGa102Bands[]    = {{1, 0}, {32, 1}};
// RTX 4090 (27B down/out): 67.6 against 74.8..90.1 us at T = 2..8 (K = 17408), equal
// at 10..16, behind the sliced-K from 20. RTX 5090: equal to T = 16, 34.8 against
// 26.6 us (K = 6144) and 88 against 61 us (K = 17408) from T = 20: off, and off on chips not
// measured.
// RTX 5080: 71.7 against 88..90 us at T = 5..8 (K = 17408), 106 against 119..127
// at 20..24, equal at 28..32. RTX 4080 SUPER (q2g_benchn2): 91 against 135..139 us at T = 5..8 and
// 139 against 173..178 at 20..24 (K = 17408); behind the row split at T = 2..4.
// A100 (cold, graph): 1.03 against 1.00 at T = 2..4, 0.71..0.94 at 5..24 and
// 1.01..1.04 at 28..32 (K = 6144 and 17408); with the 3090's bands it took the small-T MMA at
// every width and lost 27B MTP3 C4 4 % in the server (65.5 / 67.7 against 70.2 / 68.7
// tok/s without it).
constexpr SmallTBand kSmallTGa100Bands[]    = {{1, 0}, {4, 1}, {24, 0}, {32, 1}};
constexpr SmallTBand kSmallTAd102Bands[]    = {{1, 0}, {16, 1}, {32, 0}};
constexpr SmallTBand kSmallTGb202Bands[]    = {{32, 0}};
constexpr SmallTBand kSmallTGb203Bands[]    = {{1, 0}, {32, 1}};
constexpr SmallTBand kSmallTAd103Bands[]    = {{4, 0}, {32, 1}};
constexpr SmallTBand kSmallTFallbackBands[] = {{32, 0}};
constexpr SmallTChipBands kSmallTMeasured[] = {{measured::kA100Sxm40, kSmallTGa100Bands},
                                               {measured::kRtx3090, kSmallTGa102Bands},
                                               {measured::kRtx4090, kSmallTAd102Bands},
                                               {measured::kRtx5090, kSmallTGb202Bands},
                                               {measured::kRtx5080, kSmallTGb203Bands},
                                               {measured::kRtx4080Super, kSmallTAd103Bands}};

bool small_t_mma_taken(std::int32_t cols) noexcept {
    if (cols < 1 || cols > 32 || prefill_routes_active() || invariant_routes_active()) return false;
    try {
        if (!device_class().features.mma_bf16_native) return false;
    } catch (...) { return false; }
    thread_local SmallTBandCache cache;
    return small_t_band_variant(kSmallTMeasured, kSmallTFallbackBands, cache, cols) > 0;
}

// Tesla V100 at T = 5..8 (the MTP4 verify width): the split2 entry, whose launcher takes the
// warp-per-row row-split kernel there (q5_linear_add_gemm_simt.cu, ninfer-v100-sm70's route),
// instead of the sliced routes' exact-column direct tile.
bool volta_rowsplit_taken(std::int32_t cols) noexcept {
    if (cols < 5 || cols > 8 || prefill_routes_active() || invariant_routes_active()) return false;
    try {
        return !device_class().features.mma_tensor;
    } catch (...) { return false; }
}

// Keep complete 512-token waves on the wide collective MMA route. A short
// remainder uses the measured narrow schedules to avoid another mostly empty wave.
constexpr std::int32_t kWaveCols       = 512;
constexpr std::int32_t kNarrowTailCols = 192;

void launch_wide_with_narrow_tail(const Tensor& x, const Weight& w, Tensor& residual_out,
                                  WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    const std::int32_t wide = (cols / kWaveCols) * kWaveCols;
    const std::int32_t tail = cols - wide;
    if (wide == 0 || tail == 0 || tail > kNarrowTailCols) {
        q5_linear_add_mma_r64_t128_launch(x, w, residual_out, stream);
        return;
    }

    const Tensor x_wide = x.slice(1, 0, wide);
    Tensor out_wide     = residual_out.slice(1, 0, wide);
    q5_linear_add_mma_r64_t128_launch(x_wide, w, out_wide, stream);

    const Tensor x_tail = x.slice(1, wide, tail);
    Tensor out_tail     = residual_out.slice(1, wide, tail);
    q5_linear_add_execute_plan(
        q5_linear_add_resolve_plan({residual_out.ne[0], x.ne[0], w.padded_shape[1], x_tail.ne[1]}),
        x_tail, w, out_tail, ws, stream);
}

} // namespace

static Q5LinearAddPlan column_plan(const Q5LinearAddProblem& problem);

bool q5_linear_add_registered(std::int32_t rows, std::int32_t k) noexcept {
    for (const SupportSpec& support : kSupports) {
        if (rows == support.rows && k == support.k) { return true; }
    }
    return false;
}

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem) {
    if (!q5_linear_add_admits(problem)) {
        throw std::invalid_argument("q5 linear_add: exact problem or column count is not admitted");
    }
    if (small_t_mma_taken(problem.cols)) { return {Q5LinearAddScheduleId::SmallTMmaResidual, 0}; }
    if (volta_rowsplit_taken(problem.cols)) { return {Q5LinearAddScheduleId::Split2ExactResidual, 0}; }
    return column_plan(problem);
}

static Q5LinearAddPlan column_plan(const Q5LinearAddProblem& problem) {
    const auto resolve_from = [&](const auto& routes) -> Q5LinearAddPlan {
        for (const RouteSpec& route : routes) {
            if (route.contains(problem.cols)) { return {route.schedule, 0}; }
        }
        throw std::logic_error("q5 linear_add: admitted problem has no covering route");
    };
    return problem.k == 6144 ? resolve_from(kK6144Routes) : resolve_from(kK17408Routes);
}

std::size_t q5_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q5 linear_add: invalid column interval");
    }
    (void)q5_linear_add_resolve_plan({rows, k, padded_k, min_cols});
    (void)q5_linear_add_resolve_plan({rows, k, padded_k, max_cols});

    return 0;
}

static void launch_schedule(Q5LinearAddScheduleId schedule, const Tensor& x, const Weight& w,
                            Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream) {
    switch (schedule) {
    case Q5LinearAddScheduleId::Split2ExactResidual:
        q5_linear_add_split2_exact_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR16T8W4S2:
        q5_linear_add_sliced_r16_t8_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR16T8C4W8S2:
        q5_linear_add_sliced_r16_t8_c4_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR16T16W4S2:
        q5_linear_add_sliced_r16_t16_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR16T24W4S2:
        q5_linear_add_sliced_r16_t24_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T32W4S2:
        q5_linear_add_sliced_r32_t32_w4_s2_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T24W4S2Pairwise:
        q5_linear_add_sliced_r32_t24_pairwise_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T32W4S1:
        q5_linear_add_sliced_r32_t32_w4_s1_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T32W2S2:
        q5_linear_add_sliced_r32_t32_w2_s2_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::SlicedR32T64W2S1:
        q5_linear_add_sliced_r32_t64_w2_s1_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR32T32K128:
        q5_linear_add_mma_r32_t32_k128_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR32T128:
        q5_linear_add_mma_r32_t128_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64T128:
        q5_linear_add_mma_r64_t128_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64T128Tail:
        launch_wide_with_narrow_tail(x, w, residual_out, ws, stream);
        return;
    case Q5LinearAddScheduleId::SmallTMmaResidual:
        // A strided activation keeps the column route of its width.
        if (q5_linear_add_small_t_mma_operands(x, residual_out)) {
            q5_linear_add_small_t_mma_launch(x, w, residual_out, stream);
        } else {
            launch_schedule(
                column_plan({residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]}).schedule, x,
                w, residual_out, ws, stream);
        }
        return;
    }
    throw std::logic_error("q5 linear_add: unknown schedule");
}

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan resolved = q5_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q5 linear_add: plan does not match the exact problem");
    }
    launch_schedule(plan.schedule, x, w, residual_out, ws, stream);
}

void q5_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan plan = q5_linear_add_resolve_plan(problem);
    q5_linear_add_execute_plan(plan, x, w, residual_out, ws, stream);
}

} // namespace ninferno::ops::detail
