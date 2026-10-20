#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "core/weight.h"

#include "ninferno/ops/linear.h"
#include "ninferno/ops/silu_mul.h"
#include "core/device_class.h"
#include "core/layout.h"
#include "ops/common/column_routes.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

using RouteSpec = ColumnRoute<Q4LinearSwiGluScheduleId>;

constexpr Q4LinearSwiGluProblem kShape{34816, 17408, 5120, 5120, 1};

constexpr std::array<RouteSpec, 10> kRoutes{{
    {1, 1, Q4LinearSwiGluScheduleId::GemvPair},
    {2, 32, Q4LinearSwiGluScheduleId::SmallTTiled},
    {33, 128, Q4LinearSwiGluScheduleId::Materialized},
    {129, 168, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128Tail},
    {169, 224, Q4LinearSwiGluScheduleId::Materialized},
    {225, 256, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {257, 384, Q4LinearSwiGluScheduleId::Materialized},
    {385, 512, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {513, 640, Q4LinearSwiGluScheduleId::Materialized},
    {641, kAnyColumns, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
}};

static_assert(column_routes_closed(kRoutes),
              "Q4 LinearSwiGLU routes must be exact, contiguous, and closed");

bool supported_shape(const Q4LinearSwiGluProblem& problem) noexcept {
    return problem.gate_up_rows == kShape.gate_up_rows &&
           problem.output_rows == kShape.output_rows && problem.k == kShape.k &&
           problem.padded_k == kShape.padded_k;
}

// Without tensor mma.sync (Volta's build: ops/common/mma.cuh computes the fragments on FP32 FMAs)
// the small-T tile's paired rows run on emulated MMA (27B MTP verify, T = 5, V100: 1378 us a
// layer); materialized, the projection's linear takes the SIMT tiles its sliced-K launches
// redirect to there (through 16 tokens: from 17 the projection's linear takes the tiled route,
// whose workspace the plan does not hold).
Q4LinearSwiGluScheduleId device_schedule(Q4LinearSwiGluScheduleId schedule,
                                         std::int32_t cols) noexcept {
    if (schedule == Q4LinearSwiGluScheduleId::SmallTTiled && cols <= 16 &&
        !build_features().mma_tensor) {
        return Q4LinearSwiGluScheduleId::Materialized;
    }
    return schedule;
}

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows, std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}

} // namespace

bool q4_linear_swiglu_registered(std::int32_t gate_up_rows, std::int32_t k) noexcept {
    return gate_up_rows == kShape.gate_up_rows && k == kShape.k;
}

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem) {
    if (!q4_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "q4 linear_swiglu: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.contains(problem.cols)) { continue; }
        Q4LinearSwiGluPlan plan{
            device_schedule(route.schedule, problem.cols),
            0,
        };
        switch (plan.schedule) {
        case Q4LinearSwiGluScheduleId::GemvPair:
        case Q4LinearSwiGluScheduleId::SmallTTiled:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128Tail:
            return plan;
        case Q4LinearSwiGluScheduleId::Materialized:
            plan.workspace_bytes = materialized_workspace_bytes(problem.gate_up_rows, problem.cols);
            return plan;
        }
    }
    throw std::logic_error("q4 linear_swiglu: admitted problem has no covering route");
}

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q4 linear_swiglu: invalid column interval");
    }
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, min_cols});
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols});

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.last < min_cols || route.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.last, max_cols);
        maximum                     = std::max(maximum, q4_linear_swiglu_resolve_plan(
                                        {gate_up_rows, output_rows, k, padded_k, endpoint})
                                                            .workspace_bytes);
    }
    return maximum;
}

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan resolved = q4_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q4 linear_swiglu: plan does not match the exact problem");
    }

    switch (plan.schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        q4_linear_swiglu_gemv_pair_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::SmallTTiled:
        q4_linear_swiglu_small_t_tiled_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::Materialized: {
        auto scratch_scope = ws.scope();
        Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
        linear(x, w, gate_up, stream);
        silu_mul(gate_up.slice(0, 0, problem.output_rows),
                 gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
        return;
    }
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128Tail:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_tail_launch(x, w, out, stream);
        return;
    }
    throw std::logic_error("q4 linear_swiglu: unknown schedule");
}

void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan plan = q4_linear_swiglu_resolve_plan(problem);
    q4_linear_swiglu_execute_plan(plan, x, w, out, ws, stream);
}

} // namespace ninferno::ops::detail
