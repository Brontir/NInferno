#include "ops/linear_add/q8/q8_linear_add_plan.h"

#include "core/device_class.h"
#include "core/weight.h"

#include "ninferno/ops/prefill_routes.h"
#include "ops/common/column_routes.h"
#include "ops/linear_add/q8/q8_linear_add_kernels.h"
#include "ops/common/token_slices.h"

#include <array>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

using RouteSpec = ColumnRoute<Q8LinearAddScheduleId>;

// The K=4096 decode GEMV's rows a CTA, from the device's SM count rather than its name, so a new
// card takes a reasoned choice: the widest of 16/8/4 rows whose grid still gives every SM about
// one and a half CTAs (a CTA's rows share each activation load; fewer CTAs than that leave SMs
// idle in the tail). Qwen3.6-35B's K=4096 projection has 2048 rows: RTX 5090 (170 SMs) r8 = 256
// CTAs (r8 the fastest of r4/r8/r16); RTX 4090 (128) r8; RTX 3090 (82) r16 = 128 CTAs.
// NINFERNO-SPECIAL(arch:q8add-k4096-rows): the 4090 and 3090 rows were measured on those boards.
int k4096_decode_rows(std::int32_t rows) {
    const std::int64_t sms = device_class().sms;
    for (const int r : {16, 8}) {
        if (2 * (static_cast<std::int64_t>(rows) / r) >= 3 * sms) { return r; }
    }
    return 4;
}

// T <= 4: the fully unrolled GEMV (every phase's codes in flight; a weight load serves every
// token, and a token sums its row in the same order at every T). The SIMT route took 7.9 us at T=1
// (63 % of the bandwidth) and the 16-row split-K MMA at T=4 128 CTAs on 170 SMs (9.1 us;
// Qwen3.6-35B decode). From T=5 the split-K MMA again: the GEMV at a speculative verify
// width (T=8, DFlash depth 7) cost Qwen3.6-35B --spec auto C=1 5 % on an RTX 4090 and 1 % on an
// RTX 5090 (corpus), while T <= 4 keeps plain C=4 decode's gain.
inline constexpr std::int32_t kK4096DecodeMaxT = 4;

constexpr std::array<RouteSpec, 5> kK4096Routes{{
    {1, kK4096DecodeMaxT, Q8LinearAddScheduleId::DecodeK4096},
    {kK4096DecodeMaxT + 1, 48, Q8LinearAddScheduleId::SplitKMmaExactT},
    {49, 128, Q8LinearAddScheduleId::MediumSplitK},
    {129, 640, Q8LinearAddScheduleId::MmaR32C128},
    {641, kAnyColumns, Q8LinearAddScheduleId::MmaR64C128},
}};

// Prefill route profile. The split-K routes sum a column's K in another order than the MMA
// routes, whose results agree byte for byte with each other, so they are kept to the
// decode/verify widths (T <= 16): a prefill chunk of any size must give a token the same bytes.
constexpr std::array<RouteSpec, 5> kK4096PrefillRoutes{{
    {1, 1, Q8LinearAddScheduleId::SimtR8C4},
    {2, 16, Q8LinearAddScheduleId::SplitKMmaExactT},
    {17, 128, Q8LinearAddScheduleId::MmaPrefillSmallT},
    {129, 640, Q8LinearAddScheduleId::MmaR32C128},
    {641, kAnyColumns, Q8LinearAddScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 33> kK6144Routes{{
    {1, 1, Q8LinearAddScheduleId::DecodeR16},
    {2, 48, Q8LinearAddScheduleId::SplitKMmaExactT},
    {49, 128, Q8LinearAddScheduleId::MediumSplitK},
    {129, 191, Q8LinearAddScheduleId::MmaR32C128},
    {192, 192, Q8LinearAddScheduleId::MmaR32C96},
    {193, 256, Q8LinearAddScheduleId::MmaR32C128},
    {257, 384, Q8LinearAddScheduleId::MmaR32C64},
    {385, 399, Q8LinearAddScheduleId::MmaR32C96},
    {400, 400, Q8LinearAddScheduleId::MmaR32C80},
    {401, 447, Q8LinearAddScheduleId::MmaR32C96},
    {448, 448, Q8LinearAddScheduleId::MmaR32C64},
    {449, 480, Q8LinearAddScheduleId::MmaR32C96},
    {481, 640, Q8LinearAddScheduleId::MmaR32C128},
    {641, 672, Q8LinearAddScheduleId::MmaR48C96},
    {673, 704, Q8LinearAddScheduleId::MmaR48C64},
    {705, 784, Q8LinearAddScheduleId::MmaR48C112},
    {785, 896, Q8LinearAddScheduleId::MmaR48C128},
    {897, 960, Q8LinearAddScheduleId::MmaR64C96},
    {961, 1023, Q8LinearAddScheduleId::MmaR64C112},
    {1024, 1024, Q8LinearAddScheduleId::MmaR64C128},
    {1025, 1120, Q8LinearAddScheduleId::MmaR64C112},
    {1121, 1280, Q8LinearAddScheduleId::MmaR64C128},
    {1281, 1344, Q8LinearAddScheduleId::MmaR128C64},
    {1345, 1408, Q8LinearAddScheduleId::MmaR48C128},
    {1409, 1680, Q8LinearAddScheduleId::MmaR128C80},
    {1681, 1791, Q8LinearAddScheduleId::MmaR48C128},
    {1792, 1792, Q8LinearAddScheduleId::MmaR64C128},
    {1793, 1919, Q8LinearAddScheduleId::MmaR48C128},
    {1920, 1920, Q8LinearAddScheduleId::MmaR64C128},
    {1921, 2016, Q8LinearAddScheduleId::MmaR64C96},
    {2017, 2047, Q8LinearAddScheduleId::MmaR64C112},
    {2048, 2048, Q8LinearAddScheduleId::MmaR64C128},
    {2049, kAnyColumns, Q8LinearAddScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 2> kN5120Routes{{
    {1, 64, Q8LinearAddScheduleId::SplitKMmaCapacity},
    {65, kAnyColumns, Q8LinearAddScheduleId::GroupedSplitK},
}};

static_assert(column_routes_closed(kK4096Routes) && column_routes_closed(kK4096PrefillRoutes) &&
                  column_routes_closed(kK6144Routes) && column_routes_closed(kN5120Routes),
              "Q8 LinearAdd routes must be exact, contiguous, and closed");

std::int32_t schedule_rows(Q8LinearAddScheduleId schedule) {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
    case Q8LinearAddScheduleId::DecodeK4096:
    case Q8LinearAddScheduleId::MediumSplitK:
        break;
    case Q8LinearAddScheduleId::SimtR8C4:
        return 8;
    case Q8LinearAddScheduleId::MmaR32C64:
    case Q8LinearAddScheduleId::MmaR32C80:
    case Q8LinearAddScheduleId::MmaR32C96:
    case Q8LinearAddScheduleId::MmaR32C128:
        return 32;
    case Q8LinearAddScheduleId::MmaR48C64:
    case Q8LinearAddScheduleId::MmaR48C96:
    case Q8LinearAddScheduleId::MmaR48C112:
    case Q8LinearAddScheduleId::MmaR48C128:
        return 48;
    case Q8LinearAddScheduleId::MmaR64C96:
    case Q8LinearAddScheduleId::MmaR64C112:
    case Q8LinearAddScheduleId::MmaR64C128:
        return 64;
    case Q8LinearAddScheduleId::MmaR128C64:
    case Q8LinearAddScheduleId::MmaR128C80:
        return 128;
    case Q8LinearAddScheduleId::SplitKMmaExactT:
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
    case Q8LinearAddScheduleId::GroupedSplitK:
    case Q8LinearAddScheduleId::MmaPrefillSmallT:
        break;
    }
    throw std::logic_error("q8 linear_add: exact-T schedule has no row tile");
}

std::int32_t schedule_cols(Q8LinearAddScheduleId schedule);

std::int32_t schedule_cols(Q8LinearAddScheduleId schedule) {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
    case Q8LinearAddScheduleId::DecodeK4096:
    case Q8LinearAddScheduleId::MediumSplitK:
        break;
    case Q8LinearAddScheduleId::SimtR8C4:
        return 4;
    case Q8LinearAddScheduleId::MmaR32C64:
    case Q8LinearAddScheduleId::MmaR48C64:
    case Q8LinearAddScheduleId::MmaR128C64:
        return 64;
    case Q8LinearAddScheduleId::MmaR32C80:
    case Q8LinearAddScheduleId::MmaR128C80:
        return 80;
    case Q8LinearAddScheduleId::MmaR32C96:
    case Q8LinearAddScheduleId::MmaR48C96:
    case Q8LinearAddScheduleId::MmaR64C96:
        return 96;
    case Q8LinearAddScheduleId::MmaR48C112:
    case Q8LinearAddScheduleId::MmaR64C112:
        return 112;
    case Q8LinearAddScheduleId::MmaR32C128:
    case Q8LinearAddScheduleId::MmaR48C128:
    case Q8LinearAddScheduleId::MmaR64C128:
        return 128;
    case Q8LinearAddScheduleId::SplitKMmaExactT:
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
    case Q8LinearAddScheduleId::GroupedSplitK:
    case Q8LinearAddScheduleId::MmaPrefillSmallT:
        break;
    }
    throw std::logic_error("q8 linear_add: exact-T schedule is not token-sliced");
}

} // namespace

const char* q8_linear_add_schedule_name(Q8LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
        return "linear_add.q8.decode.r16.residual";
    case Q8LinearAddScheduleId::DecodeK4096:
        return "linear_add.q8.decode.k4096.residual";
    case Q8LinearAddScheduleId::GroupedSplitK:
        return "linear_add.q8.grouped_splitk.residual";
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
        return "linear_add.q8.splitk.mma.capacity.residual";
    case Q8LinearAddScheduleId::SplitKMmaExactT:
        return "linear_add.q8.splitk8.mma.r16.exact_t.residual";
    case Q8LinearAddScheduleId::MediumSplitK:
        return "linear_add.q8.medium_splitk.residual";
    case Q8LinearAddScheduleId::SimtR8C4:
        return "linear_add.q8.simt.r8.c4.slab1024.s2.code_ca.scale_pair32";
    case Q8LinearAddScheduleId::MmaR32C64:
        return "linear_add.q8.mma.r32.c64.residual";
    case Q8LinearAddScheduleId::MmaR32C80:
        return "linear_add.q8.mma.r32.c80.residual";
    case Q8LinearAddScheduleId::MmaR32C96:
        return "linear_add.q8.mma.r32.c96.residual";
    case Q8LinearAddScheduleId::MmaR32C128:
        return "linear_add.q8.mma.r32.c128.k64.wr32.wc16.s2.scale_cache8.lb2";
    case Q8LinearAddScheduleId::MmaR48C64:
        return "linear_add.q8.mma.r48.c64.residual";
    case Q8LinearAddScheduleId::MmaR48C96:
        return "linear_add.q8.mma.r48.c96.residual";
    case Q8LinearAddScheduleId::MmaR48C112:
        return "linear_add.q8.mma.r48.c112.residual";
    case Q8LinearAddScheduleId::MmaR48C128:
        return "linear_add.q8.mma.r48.c128.residual";
    case Q8LinearAddScheduleId::MmaR64C96:
        return "linear_add.q8.mma.r64.c96.residual";
    case Q8LinearAddScheduleId::MmaR64C112:
        return "linear_add.q8.mma.r64.c112.residual";
    case Q8LinearAddScheduleId::MmaR64C128:
        return "linear_add.q8.mma.r64.c128.k64.wr64.wc16.s2.scale_cache8.lb2";
    case Q8LinearAddScheduleId::MmaR128C64:
        return "linear_add.q8.mma.r128.c64.residual";
    case Q8LinearAddScheduleId::MmaR128C80:
        return "linear_add.q8.mma.r128.c80.residual";
    case Q8LinearAddScheduleId::MmaPrefillSmallT:
        return "linear_add.q8.mma.prefill_small_t.residual";
    }
    return "linear_add.q8.unknown";
}

bool q8_linear_add_registered(std::int32_t rows, std::int32_t k) noexcept {
    return (rows == 2048 && (k == 4096 || k == 6144)) ||
           (rows == 5120 && (k == 6144 || k == 17408));
}

bool q8_linear_add_admits(const Q8LinearAddProblem& problem) noexcept {
    return q8_linear_add_registered(problem.rows, problem.k) && problem.padded_k == problem.k &&
           problem.cols >= 1;
}

Q8LinearAddPlan q8_linear_add_resolve_plan(const Q8LinearAddProblem& problem) {
    if (!q8_linear_add_admits(problem)) {
        throw std::invalid_argument("q8 linear_add: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> Q8LinearAddPlan {
        return {column_route(routes, problem.cols,
                             "q8 linear_add: admitted problem has no covering route")};
    };
    if (problem.rows == 5120) { return resolve_from(kN5120Routes); }
    if (problem.k == 6144) { return resolve_from(kK6144Routes); }
    return prefill_routes_active() ? resolve_from(kK4096PrefillRoutes) : resolve_from(kK4096Routes);
}

bool use_full(Q8LinearAddScheduleId schedule, const Q8LinearAddProblem& problem) {
    return problem.rows % schedule_rows(schedule) == 0 &&
           problem.cols % schedule_cols(schedule) == 0;
}

Q8LinearAddKernelClass q8_linear_add_kernel_class(const Q8LinearAddPlan& plan,
                                                  const Q8LinearAddProblem& problem) {
    switch (plan.schedule) {
    case Q8LinearAddScheduleId::DecodeK4096:
        return {plan.schedule, false, problem.cols}; // one GEMV instance per T
    case Q8LinearAddScheduleId::DecodeR16:
    case Q8LinearAddScheduleId::GroupedSplitK:
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
        return {plan.schedule, false, 0};
    case Q8LinearAddScheduleId::SplitKMmaExactT:
        return {plan.schedule, false, problem.cols};
    case Q8LinearAddScheduleId::MediumSplitK:
        return {plan.schedule, false, q8_linear_add_medium_band(problem.cols)};
    case Q8LinearAddScheduleId::MmaPrefillSmallT:
        // The admitted rows (2048) and K (4096) fill both tiles' rows and K steps, so the
        // launcher's full test comes down to the columns.
        return {plan.schedule, problem.cols % kQ8LinearAddPrefillSmallTileCols == 0,
                problem.cols <= kQ8LinearAddPrefillSmallR16MaxCols ? 0 : 1};
    default:
        return {plan.schedule, use_full(plan.schedule, problem), 0};
    }
}

void q8_linear_add_execute_plan(const Q8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream) {
    const Q8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q8LinearAddPlan resolved = q8_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("q8 linear_add: plan does not match the exact problem");
    }
    if (plan.schedule == Q8LinearAddScheduleId::GroupedSplitK) {
        q8_linear_add_grouped_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::SplitKMmaCapacity) {
        q8_linear_add_splitk_capacity_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::DecodeR16) {
        q8_linear_add_decode_r16_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::DecodeK4096) {
        q8_linear_add_decode_k4096_launch(x, w, residual_out, k4096_decode_rows(w.n), stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::SplitKMmaExactT) {
        q8_linear_add_splitk_mma_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::MediumSplitK) {
        q8_linear_add_medium_splitk_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::MmaPrefillSmallT) {
        q8_linear_add_mma_prefill_small_t_launch(x, w, residual_out, stream);
        return;
    }
    for_each_token_slice(
        x.ne[1], schedule_cols(plan.schedule), [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice  = x.slice(1, offset, count);
            Tensor residual_slice = residual_out.slice(1, offset, count);
            switch (plan.schedule) {
            case Q8LinearAddScheduleId::DecodeR16:
            case Q8LinearAddScheduleId::DecodeK4096:
            case Q8LinearAddScheduleId::MediumSplitK:
                break;
            case Q8LinearAddScheduleId::SimtR8C4:
                q8_linear_add_simt_r8_c4_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C64:
                q8_linear_add_mma_r32_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C80:
                q8_linear_add_mma_r32_c80_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C96:
                q8_linear_add_mma_r32_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C128:
                q8_linear_add_mma_r32_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C64:
                q8_linear_add_mma_r48_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C96:
                q8_linear_add_mma_r48_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C112:
                q8_linear_add_mma_r48_c112_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C128:
                q8_linear_add_mma_r48_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C96:
                q8_linear_add_mma_r64_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C112:
                q8_linear_add_mma_r64_c112_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C128:
                q8_linear_add_mma_r64_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR128C64:
                q8_linear_add_mma_r128_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR128C80:
                q8_linear_add_mma_r128_c80_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::GroupedSplitK:
            case Q8LinearAddScheduleId::SplitKMmaCapacity:
            case Q8LinearAddScheduleId::SplitKMmaExactT:
            case Q8LinearAddScheduleId::MmaPrefillSmallT:
                break;
            }
            throw std::logic_error("q8 linear_add: unknown tiled schedule");
        });
}

void q8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream) {
    const Q8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    q8_linear_add_execute_plan(q8_linear_add_resolve_plan(problem), x, w, residual_out, stream);
}

} // namespace ninferno::ops::detail
