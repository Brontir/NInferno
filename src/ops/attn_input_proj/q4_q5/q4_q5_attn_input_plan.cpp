#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "core/weight.h"

#include "core/device_class.h"
#include "core/measured_boards.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#include "ops/common/small_t_bands.h"
#include "ninferno/ops/prefill_routes.h"

#include <cstdlib>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

// Sliced-K small-T tile for T in [5,32] (see the GDN input plan): the parent-split kernel grows
// past T = 4 and the mixed R32 C32 tile pays a 32-row activation pass per tile. Measured cold
// against the column route (x at T = 8 / 13 / 24 / 32):
//   RTX 5090  R16 W4 S2 (2) from T = 5: x1.30 / 1.75 / 1.31; the route from T = 25 (x0.98..1.00).
//             Swept again after the K permutation (cold): R16 W4 S2 from T = 2
//             (33 against 41..45 us at T = 2..4) and R32 W4 S2 (3) at 25..32 (64..66
//             against 84..86)
//   RTX 5080  as the 5090 through T = 24, then R16 W4 S2 to 28 (x1.26) and R32 W8 at 32 (x1.32),
//             where the 5090 keeps the route (e66x2)
//   RTX 4090  R32 W8 (4) from T = 9: 1.94 / 1.71 / 1.53 (R16 W8: 1.83 / 1.52 / 1.30); after the
//             K permutation R16 W8 also wins T = 5..8 cold (62 against 65..82 us), but
//             not in the server (27B MTP3 C4 / C8 flat, 24K -1 % per round), so kept
//   RTX 4070 Ti SUPER (ad103)  R16 W8 (1) from T = 5: 1.20 / 1.41 / 2.36 / 1.94 at 5 / 12 / 13 /
//   24,
//             where the 4090's R32 W8 gives 0.90 / 1.14 / 1.89 / 1.73
//   RTX 3090  R16 W8 for T = 2..20: after the K permutation it beats the parent split and
//             R16 W4 S2 at every width swept (cold): 82 / 89 / 89 us at T = 2 / 5 / 8
//             against 91 / 155 / 101; 1.25 / 1.07 at 13 / 20; R32 W8 loses from T = 24
//             (x0.85..0.90)
//   RTX 2080 Ti  R16 W4 S2 for T = 9..24: 1.59 / 1.20
//   RTX 3090  variant 5, the small-T MMA kernels the T = 1 call already takes (ninfer-all's route
//             at every width): a DFlash2 round census at 19.5K put the pair at 34.1 + 26.3 us at
//             T = 6 against 49.3 + 34.4 for R16 W8
//   RTX 4090  variant 5 at T = 2..8 as well: the DFlash2 round census at 19.5K had
//   the
//             parent split's SIMT pair at 41.6 + 30.3 us at T = 6 against ninfer-all's small-T
//             MMA pair (29.0 us for the gate/value half); the bench 53..57 us for
//             variant 5 at T = 2..16 against 58..77 (variant 1) and 67..74 (variant 4), variant 4
//             from 20
// Other chips as in the GDN input plan; the fallback is R16 W8 for T = 9..20. The prefill and
// invariant profiles keep the column routes.

constexpr SmallTBand kGb202Bands[]    = {{1, 0}, {24, 2}, {32, 3}};
// RTX 5080 / 4080 SUPER: variant 4 53 us at T = 2..8 on the 5080 (the column
// route 61..78), variant 5 59..63 at 10..16; the 4080 SUPER's variant 5 69..77 us at T = 2..16
// against 75..82 (variant 1).
constexpr SmallTBand kGb203Bands[]    = {{1, 0}, {8, 4}, {16, 5}, {24, 2}, {32, 4}};
constexpr SmallTBand kAd103Bands[]    = {{1, 0}, {16, 5}, {24, 1}, {32, 4}};
constexpr SmallTBand kAd102Bands[]    = {{1, 0}, {16, 5}, {32, 4}};
constexpr SmallTBand kGa102Bands[]    = {{1, 0}, {32, 5}};
// A100 (cold, graph): variant 5 at T = 2..16 (57..88 us against 76..95 for variant 1)
// and 28..32, variant 1 at 20..24 (114 / 120 against 122 / 122).
constexpr SmallTBand kGa100Bands[]    = {{1, 0}, {16, 5}, {24, 1}, {32, 5}};
constexpr SmallTBand kTu102Bands[]    = {{8, 0}, {24, 2}, {32, 0}};
// A chip of no measured row and no measured sibling: on the native BF16 MMA (sm_80 on) variant 5 at
// T = 2..16 and 28..32 and variant 1 at 20..24, the pattern A100, RTX 3090, 4060 Ti and 4090 share
// (variant 5 1.3..2.5x the column route at T = 2..8 on each); the RTX 5090 alone prefers variant 2 there (2.6 CTAs per SM). Where the MMA is
// emulated (Volta, Turing) the column route and variant 1.
constexpr SmallTBand kFallbackMmaBands[] = {{1, 0}, {16, 5}, {24, 1}, {32, 5}};
constexpr SmallTBand kFallbackBands[]    = {{8, 0}, {20, 1}, {32, 0}};
constexpr SmallTChipBands kMeasured[] = {{measured::kRtx5090, kGb202Bands}, {measured::kRtx5080, kGb203Bands},
                                         {measured::kRtx4090, kAd102Bands}, {measured::kRtx4080Super, kAd103Bands},
                                         {measured::kA100Sxm40, kGa100Bands}, {measured::kRtx3090, kGa102Bands},
                                         {measured::kRtx2080Ti, kTu102Bands}};

int small_t_variant(std::int32_t cols) noexcept {
    if (cols > 32 || prefill_routes_active() || invariant_routes_active()) return 0;
    thread_local SmallTBandCache cache;
    const bool native_mma = [] {
        try {
            return device_class().features.mma_bf16_native;
        } catch (...) { return false; }
    }();
    const std::span<const SmallTBand> fallback =
        native_mma ? std::span<const SmallTBand>(kFallbackMmaBands) : kFallbackBands;
    return small_t_band_variant(kMeasured, fallback, cache, cols);
}

bool small_t_taken(std::int32_t cols) noexcept { return small_t_variant(cols) > 0; }

} // namespace

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

    if (small_t_variant(problem.cols) == 5) return {Q4Q5AttnInputScheduleId::SmallTMma};
    if (small_t_taken(problem.cols)) return {Q4Q5AttnInputScheduleId::SlicedKSmallT};
    if (problem.cols <= 12) return {Q4Q5AttnInputScheduleId::ParentSplitFixed};
    if (problem.cols <= 32) return {Q4Q5AttnInputScheduleId::MixedR32C32S2};
    if (problem.cols <= 64) return {Q4Q5AttnInputScheduleId::MixedR32C64S3};
    if (problem.cols <= 104) return {Q4Q5AttnInputScheduleId::PairR32C64S3};
    if (problem.cols <= 128 || problem.cols >= 193)
        return {Q4Q5AttnInputScheduleId::MixedR64C128S2};
    return {Q4Q5AttnInputScheduleId::PairR32C64S4};
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::SmallTMma:
        q4_q5_attn_input_small_t_mma_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                            stream);
        return;
    case Q4Q5AttnInputScheduleId::SlicedKSmallT:
        q4_q5_attn_input_sliced_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                       small_t_variant(problem.cols), stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C32S2:
        q4_q5_attn_input_mixed_r32_c32_s2_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        q4_q5_attn_input_mixed_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        q4_q5_attn_input_pair_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR64C128S2:
        q4_q5_attn_input_mixed_r64_c128_s2_launch(x, query_key_weight, gate_value_weight, q, gate,
                                                  k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  stream);
}

} // namespace ninferno::ops::detail
