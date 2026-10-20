#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "core/weight.h"

#include "core/device_class.h"
#include "core/measured_boards.h"
#include "ops/common/column_routes.h"
#include "ops/common/small_t_bands.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "ninferno/ops/prefill_routes.h"

#include <array>
#include <cstdlib>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

using RouteSpec = ColumnRoute<Q4Q5GdnInputScheduleId>;

constexpr std::array<RouteSpec, 4> kRoutes{{
    {1, 12, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {13, 32, Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C32S2},
    {33, 64, Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C64S4},
    {65, kAnyColumns, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128S2},
}};

static_assert(column_routes_closed(kRoutes), "GDN input routes must be exact and closed");

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

// Sliced-K small-T tile for T in [5,32]: the independent direct kernel grows linearly in T past 4
// and the grouped R32 C32 tile reads the activation once per 32-row tile, so both lose to one
// activation slab per CTA with K split across warps. Which variant (1: R16 W8, 2: R16 W4 S2,
// 4: R32 W8) and how far differ per chip, measured cold on the 27B shapes against the column
// route (x at T = 8 / 13 / 24 / 32):
//   RTX 5090  R16 W4 S2 from T = 5: x1.39 / 1.68 / 1.30, then R32 W8 (T = 32: x1.16). Swept again
//             after the K permutation (cold, us): R16 W8 at 9..16 (45..51
//             against 51..55), R32 W4 S2 at 24 and 32 (59 / 74 against 67 / 88), R16 W4 S2
//             elsewhere
//   RTX 4090  R16 W8 from T = 9 (T = 8 loses, x0.94): 2.16 / 1.74 / 1.37; after the K permutation
//             R16 W8 from T = 5 (71 against 79..84 us) and R32 W8 at 25..32 (110..113 against
//             115..135) cold, but not in the server, so kept
//   RTX 4070 Ti SUPER (ad103, 66 SMs)  R16 W8 from T = 5 (x1.11 / 1.25 at 5 / 8, 1.63..1.95 at
//             12..16), R32 W8 at 28..32 (x1.47 / 1.51 against 1.38 / 1.33)
//   RTX 3090  R16 W8 from T = 2: after the K permutation it beats the parent split at every
//             width it was swept (cold): 87 / 90 / 94 us at T = 2 / 5 / 8 against
//             104 / 164 / 159; the column route from T = 25 (x1.00 / 0.94)
//   RTX 2080 Ti  R32 W8 for T = 9..16 (x1.69 at 13), R16 W4 S2 to 24 (x1.22), then the route
// A board without bands takes the nearest measured board's (nearest_measured), and a card of
// an unmeasured architecture the bands every measured native-BF16 board gains on: R16 W8 for
// T = 9..24. After the sliced-K K permutation the verify widths 2..4 gain too,
// Qwen3.8-27B MTP3 at 24K decode tok/s: RTX 5090 R16 W4 S2 170.0 -> 178.7 / 178.9,
// RTX 4090 R16 W8 107.6 -> 113.6. The prefill and invariant profiles keep the column routes.

constexpr SmallTBand kGb202Bands[] = {{1, 0}, {8, 2}, {16, 1}, {22, 2}, {24, 3}, {28, 2}, {32, 3}};
constexpr SmallTBand kAd103Bands[]    = {{4, 0}, {24, 1}, {32, 4}};
constexpr SmallTBand kAd102Bands[]    = {{1, 0}, {4, 1}, {8, 0}, {32, 1}};
constexpr SmallTBand kGa102Bands[]    = {{1, 0}, {24, 1}, {32, 0}};
constexpr SmallTBand kTu102Bands[]    = {{8, 0}, {16, 4}, {24, 2}, {32, 0}};
constexpr SmallTBand kFallbackBands[] = {{8, 0}, {24, 1}, {32, 0}};
constexpr SmallTChipBands kMeasured[] = {{measured::kRtx5090, kGb202Bands},
                                         {measured::kRtx4090, kAd102Bands},
                                         {measured::kRtx4080Super, kAd103Bands},
                                         {measured::kRtx3090, kGa102Bands},
                                         {measured::kRtx2080Ti, kTu102Bands}};

int small_t_variant(std::int32_t cols) noexcept {
    if (cols > 32 || prefill_routes_active() || invariant_routes_active()) { return 0; }
    thread_local SmallTBandCache cache;
    return small_t_band_variant(kMeasured, kFallbackBands, cache, cols);
}

bool small_t_taken(std::int32_t cols) noexcept { return small_t_variant(cols) > 0; }

// Verify and draft widths 2..32 materialize the convolution: the projection-epilogue fused route
// ran the projection on SIMT kernels there (27B DFlash2 verify, T = 6, 5090: 56.8 + 22.0 us a
// layer). Prefill and invariant profiles keep their routes, and so does a build without tensor
// mma.sync (Volta: the small-T MMA kernels run on FP32 FMAs, 864 + 217 us a layer at T = 5 on
// the V100, where the sliced-K bands and the direct SIMT route take the call).
bool small_t_mma_taken(std::int32_t cols) noexcept {
    return cols >= 2 && cols <= 32 && build_features().mma_tensor && !prefill_routes_active() &&
           !invariant_routes_active();
}

// Which projection those widths take, per chip (1: the small-T MMA kernels - one m16 row tile per
// CTA, eight warps splitting each K slab; 0: the sliced-K bands, then the grouped routes). From
// ninferno_gdn_input_proj_bench, 27B form, warm graph: on the 5090 the sliced-K bands lead at every
// width (1.04..1.28x); on the 4090 small-T MMA leads at T = 2..8 (twice the grouped route at
// 5..8) and from 16 up, the sliced-K band at 10..12; on the 3090 small-T MMA leads at every
// width (1.14..1.85x).
constexpr SmallTBand kMmaGb202Bands[]    = {{32, 0}};
constexpr SmallTBand kMmaAd102Bands[]    = {{8, 1}, {12, 0}, {32, 1}};
constexpr SmallTBand kMmaGa102Bands[]    = {{32, 1}};
constexpr SmallTBand kMmaFallbackBands[] = {{32, 1}};
constexpr SmallTChipBands kMmaMeasured[] = {
    {measured::kRtx5090, kMmaGb202Bands}, {measured::kRtx4090, kMmaAd102Bands}, {measured::kRtx3090, kMmaGa102Bands}};

// One token takes the small-T MMA projection and a materialized convolution only where that was
// measured to win: the 3090 decode census at 19.5K had the fused-convolution GEMVs
// at 59.1 + 24.6 us a layer against ninfer-all's small-T MMA 55.0 + 17.1 and its 2.6 us
// convolution.
constexpr SmallTBand kT1MmaGa102Bands[]    = {{1, 1}};
constexpr SmallTBand kT1MmaAd103Bands[]    = {{1, 1}}; // 4080 SUPER: 81.1 against 85.0 us
constexpr SmallTBand kT1MmaFallbackBands[] = {{1, 0}};
constexpr SmallTChipBands kT1MmaMeasured[] = {{measured::kRtx3090, kT1MmaGa102Bands},
                                              {measured::kRtx4080Super, kT1MmaAd103Bands}};

bool one_token_mma(std::int32_t cols) noexcept {
    if (cols != 1 || prefill_routes_active() || invariant_routes_active()) { return false; }
    thread_local SmallTBandCache cache;
    return small_t_band_variant(kT1MmaMeasured, kT1MmaFallbackBands, cache, cols) > 0;
}

bool small_t_mma_projection(std::int32_t cols) noexcept {
    if (one_token_mma(cols)) { return true; }
    if (!small_t_mma_taken(cols)) { return false; }
    thread_local SmallTBandCache cache;
    return small_t_band_variant(kMmaMeasured, kMmaFallbackBands, cache, cols) > 0;
}

} // namespace

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    if (small_t_mma_projection(problem.cols)) { return {Q4Q5GdnInputScheduleId::SmallTMma}; }
    if (small_t_taken(problem.cols)) { return {Q4Q5GdnInputScheduleId::SlicedKSmallT}; }
    for (const RouteSpec& route : kRoutes) {
        if (!route.contains(problem.cols)) { continue; }
        return {route.schedule};
    }
    throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
}

Q4Q5GdnInputConvPlan q4_q5_gdn_input_conv_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                                       std::int32_t batch_size) {
    if (!q4_q5_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input conv: exact problem or column count is not admitted");
    }
    // The invariant route profile materializes every width: the fused form sums K its own way.
    if (batch_size > 1 || small_t_mma_taken(problem.cols) || one_token_mma(problem.cols) ||
        invariant_routes_active()) {
        return {Q4Q5GdnInputConvScheduleId::Materialized};
    }
    switch (problem.cols) {
    case 1:
    case 2:
    case 3:
    case 5:
    case 6:
        return {Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused};
    default:
        return {Q4Q5GdnInputConvScheduleId::Materialized};
    }
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::SmallTMma: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_small_t_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::SlicedKSmallT: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_sliced_launch(x, qk_weight, value_z_weight, qk, value, z,
                                      small_t_variant(problem.cols), stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C32S2:
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C64S4:
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128S2:
        q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, plan.schedule,
                                           stream);
        return;
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, stream);
}

} // namespace ninferno::ops::detail
