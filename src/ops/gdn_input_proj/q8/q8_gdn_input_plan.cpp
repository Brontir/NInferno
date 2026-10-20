#include "ops/gdn_input_proj/q8/q8_gdn_input_plan.h"
#include "core/weight.h"

#include "ninferno/ops/prefill_routes.h"
#include "ops/common/column_routes.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"

#include <array>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

using RouteSpec = ColumnRoute<Q8GdnInputScheduleId>;

constexpr std::array<RouteSpec, 3> kRoutes{{
    {1, 1, Q8GdnInputScheduleId::DecodeR8Direct},
    {2, 96, Q8GdnInputScheduleId::SplitKMmaDirect},
    {97, kAnyColumns, Q8GdnInputScheduleId::MmaR64C128},
}};

// Prefill route profile. The split-K route sums a column's K in another order than the MMA
// routes, so it is kept to the decode/verify widths (T <= 16): a prefill chunk of any size must
// give a token the same bytes. The split-K band's prefill-sized calls take the MMA contraction on
// a narrow tile instead.
constexpr std::array<RouteSpec, 4> kPrefillRoutes{{
    {1, 1, Q8GdnInputScheduleId::DecodeR8Direct},
    {2, 16, Q8GdnInputScheduleId::SplitKMmaDirect},
    {17, 96, Q8GdnInputScheduleId::MmaPrefillSmallT},
    {97, kAnyColumns, Q8GdnInputScheduleId::MmaR64C128},
}};

static_assert(column_routes_closed(kRoutes) && column_routes_closed(kPrefillRoutes),
              "Q8 GDN input routes must be exact and closed");

bool supported_shape(const Q8GdnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.qkv_rows == 8192 && problem.z_rows == 4096 &&
           problem.parent_rows == 12288 && problem.padded_k == 2048;
}

} // namespace

bool q8_gdn_input_admits(const Q8GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

Q8GdnInputPlan q8_gdn_input_resolve_plan(const Q8GdnInputProblem& problem) {
    if (!q8_gdn_input_admits(problem)) {
        throw std::invalid_argument("Q8 GDN input: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> Q8GdnInputPlan {
        return {column_route(routes, problem.cols,
                             "Q8 GDN input: admitted problem has no covering route")};
    };
    return prefill_routes_active() ? resolve_from(kPrefillRoutes) : resolve_from(kRoutes);
}

Q8GdnInputConvPlan q8_gdn_input_conv_resolve_plan(const Q8GdnInputProblem& problem,
                                                  std::int32_t batch_size) {
    if (!q8_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q8 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1) { return {Q8GdnInputConvScheduleId::Materialized}; }
    if (problem.cols == 1) { return {Q8GdnInputConvScheduleId::DecodeFused}; }
    if (problem.cols <= 16) { return {Q8GdnInputConvScheduleId::SplitKMmaFused}; }
    return {Q8GdnInputConvScheduleId::Materialized};
}

void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream) {
    const Q8GdnInputProblem problem{x.ne[0], qkv.ne[0], z.ne[0], weight.n, weight.padded_shape[1],
                                    x.ne[1]};
    const Q8GdnInputPlan plan = q8_gdn_input_resolve_plan(problem);
    switch (plan.schedule) {
    case Q8GdnInputScheduleId::DecodeR8Direct:
        q8_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::SplitKMmaDirect:
        q8_gdn_input_splitk_mma_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::MmaR64C128:
        q8_gdn_input_mma_r64_c128_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::MmaPrefillSmallT:
        q8_gdn_input_mma_prefill_small_t_launch(x, weight, qkv, z, stream);
        return;
    }
    throw std::logic_error("Q8 GDN input: unknown schedule");
}

} // namespace ninferno::ops::detail
