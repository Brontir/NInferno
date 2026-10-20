#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"
#include "core/weight.h"

#include "ninferno/ops/prefill_routes.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"
#include "ops/common/column_routes.h"

#include <array>
#include <stdexcept>

namespace ninferno::ops::detail {

namespace {

using RouteSpec = ColumnRoute<Q8AttnInputScheduleId>;

constexpr std::array<RouteSpec, 4> kTargetRoutes{{
    {1, 1, Q8AttnInputScheduleId::DecodeR8Direct},
    {2, 64, Q8AttnInputScheduleId::SplitKMmaDirect},
    {65, 128, Q8AttnInputScheduleId::MmaR32C128},
    {129, kAnyColumns, Q8AttnInputScheduleId::MmaR64C128},
}};

// Prefill route profile. The split-K route sums a column's K in another order than the MMA
// routes, whose results agree byte for byte with each other, so it is kept to the decode/verify
// widths (T <= 16): a prefill chunk of any size must give a token the same bytes. The split-K
// band's prefill-sized calls take the MMA contraction on a narrow tile instead.
constexpr std::array<RouteSpec, 5> kTargetPrefillRoutes{{
    {1, 1, Q8AttnInputScheduleId::DecodeR8Direct},
    {2, 16, Q8AttnInputScheduleId::SplitKMmaDirect},
    {17, 64, Q8AttnInputScheduleId::MmaPrefillSmallT},
    {65, 128, Q8AttnInputScheduleId::MmaR32C128},
    {129, kAnyColumns, Q8AttnInputScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 9> kCompanionRoutes{{
    {1, 1, Q8AttnInputScheduleId::DecodeR8Direct},
    {2, 96, Q8AttnInputScheduleId::SplitKMmaDirect},
    {97, 192, Q8AttnInputScheduleId::MmaR32C64},
    {193, 288, Q8AttnInputScheduleId::MmaR64C96},
    {289, 320, Q8AttnInputScheduleId::MmaR64C64},
    {321, 384, Q8AttnInputScheduleId::MmaR64C128},
    {385, 448, Q8AttnInputScheduleId::MmaR128C64},
    {449, 560, Q8AttnInputScheduleId::MmaR128C80},
    {561, kAnyColumns, Q8AttnInputScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 6> kDFlash2Routes{{
    {1, 48, Q8AttnInputScheduleId::DFlash2SmallT},
    {49, 63, Q8AttnInputScheduleId::DFlash2MmaR16C64K128},
    {64, 96, Q8AttnInputScheduleId::DFlash2MmaR32C32K128},
    {97, 128, Q8AttnInputScheduleId::DFlash2MmaR32C64K128},
    {129, 192, Q8AttnInputScheduleId::DFlash2MmaR32C64},
    {193, kAnyColumns, Q8AttnInputScheduleId::DFlash2MmaR64C128},
}};

static_assert(column_routes_closed(kTargetRoutes) && column_routes_closed(kTargetPrefillRoutes),
              "Q8 target attention input routes must be exact and closed");
static_assert(column_routes_closed(kCompanionRoutes),
              "Q8 companion attention input routes must be exact and closed");
static_assert(column_routes_closed(kDFlash2Routes),
              "Q8 DFlash2 attention input routes must be exact and closed");

bool is_companion_shape(const Q8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 2048;
}

bool is_dflash2_shape(const Q8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 5120;
}

bool supported_shape(const Q8AttnInputProblem& problem) noexcept {
    const bool target_qkgv =
        problem.query_rows == 4096 && problem.kv_rows == 512 && problem.parent_rows == 9216;
    const bool target_shape = problem.input_rows == 2048 && problem.padded_k == 2048 && target_qkgv;
    return target_shape || is_companion_shape(problem) || is_dflash2_shape(problem);
}

} // namespace

bool q8_attn_input_admits(const Q8AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

Q8AttnInputPlan q8_attn_input_resolve_plan(const Q8AttnInputProblem& problem) {
    if (!q8_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q8 attention input: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> Q8AttnInputPlan {
        return {column_route(routes, problem.cols,
                             "Q8 attention input: admitted problem has no covering route")};
    };
    if (is_dflash2_shape(problem)) { return resolve_from(kDFlash2Routes); }
    if (is_companion_shape(problem)) { return resolve_from(kCompanionRoutes); }
    return prefill_routes_active() ? resolve_from(kTargetPrefillRoutes)
                                   : resolve_from(kTargetRoutes);
}

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const Q8AttnInputPlan resolved = q8_attn_input_resolve_plan(problem);
    if (problem.parent_rows != 9216 || problem.kv_rows != 512 ||
        resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "Q8 attention input: plan does not match exact four-output problem");
    }
    switch (plan.schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        q8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        q8_attn_input_splitk_mma_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SimtR8C4:
        q8_attn_input_simt_r8_c4_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C128:
        q8_attn_input_mma_r32_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C128:
        q8_attn_input_mma_r64_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaPrefillSmallT:
        q8_attn_input_mma_prefill_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C64:
    case Q8AttnInputScheduleId::MmaR64C64:
    case Q8AttnInputScheduleId::MmaR64C96:
    case Q8AttnInputScheduleId::MmaR128C64:
    case Q8AttnInputScheduleId::MmaR128C80:
    case Q8AttnInputScheduleId::DFlash2SmallT:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64:
    case Q8AttnInputScheduleId::DFlash2MmaR64C128:
    case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
        throw std::logic_error("Q8 attention input: three-output schedule in four-output plan");
    }
    throw std::logic_error("Q8 attention input: unknown schedule");
}

void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    q8_attn_input_execute_plan(q8_attn_input_resolve_plan(problem), x, weight, q, gate, k, v,
                               stream);
}

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const Q8AttnInputPlan resolved = q8_attn_input_resolve_plan(problem);
    const bool companion           = is_companion_shape(problem);
    const bool dflash2             = is_dflash2_shape(problem);
    if ((!companion && !dflash2) || resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "Q8 attention input: plan does not match exact three-output problem");
    }
    if (dflash2) {
        switch (plan.schedule) {
        case Q8AttnInputScheduleId::DFlash2SmallT:
            q8_dflash2_attn_input_small_t_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
            q8_dflash2_attn_input_mma_r16_c64_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
            q8_dflash2_attn_input_mma_r32_c32_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
            q8_dflash2_attn_input_mma_r32_c64_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C64:
            q8_dflash2_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR64C128:
            q8_dflash2_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
            return;
        default:
            throw std::logic_error("Q8 attention input: non-DFlash2 schedule in DFlash2 plan");
        }
    }
    switch (plan.schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        q8_attn_input_decode_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        q8_attn_input_splitk_mma_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SimtR8C4:
        q8_attn_input_simt_r8_c4_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C128:
        q8_attn_input_mma_r32_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C64:
        q8_companion_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C64:
        q8_companion_attn_input_mma_r64_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C96:
        q8_companion_attn_input_mma_r64_c96_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C64:
        q8_companion_attn_input_mma_r128_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C80:
        q8_companion_attn_input_mma_r128_c80_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C128:
        q8_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::DFlash2SmallT:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64:
    case Q8AttnInputScheduleId::DFlash2MmaR64C128:
    case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
        throw std::logic_error("Q8 attention input: DFlash2 schedule in companion plan");
    case Q8AttnInputScheduleId::MmaPrefillSmallT:
        throw std::logic_error("Q8 attention input: four-output schedule in companion plan");
    }
    throw std::logic_error("Q8 attention input: unknown schedule");
}

void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    q8_attn_input_execute_plan(q8_attn_input_resolve_plan(problem), x, weight, q, k, v, stream);
}

} // namespace ninferno::ops::detail
