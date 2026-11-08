#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninferno::ops::detail {

enum class Q8PairScheduleId {
    TwoSimtR8C4,
    DualDecodeR4,
    DualDecodeR8,
    DualDecodeR16,
    DualSplitKMmaExactT,
    DualSplitKMediumC48,
    DualSplitKMediumC64,
    DualSplitKMediumC80,
    DualSplitKMediumC88,
    DualSplitKMediumC96,
    DualSplitKMediumC104,
    DualSplitKMediumC112,
    DualSplitKMediumC128,
    DualSplitKMediumC160,
    DualSplitKMediumC192,
    DualSplitKMediumC224,
    DualSplitKMediumC256,
    DualMmaR32C64,
    DualMmaR32C128,
    ConcatMmaR32C64,
    ConcatMmaR32C80,
    ConcatMmaR32C96,
    ConcatMmaR32C112,
    ConcatMmaR32C128,
    ConcatMmaR48C64,
    ConcatMmaR48C96,
    ConcatMmaR48C112,
    ConcatMmaR48C128,
    ConcatMmaR64C64,
    ConcatMmaR64C80,
    ConcatMmaR64C96,
    ConcatMmaR64C128,
    ConcatMmaR96C64,
    ConcatMmaR96C80,
    ConcatMmaR96C96,
    ConcatMmaR96C112,
    ConcatMmaR128C64,
    ConcatMmaR128C80,
    ExactConcatMmaR32C96,
    ExactConcatMmaR32C128,
    ExactConcatMmaR64C96,
    ExactConcatMmaR64C128,
    ExactConcatMmaR96C96,
    ExactConcatMmaR128C64,
    ExactConcatMmaR128C80,
};

struct Q8PairProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q8PairPlan {
    Q8PairScheduleId schedule;
};

const char* q8_pair_schedule_name(Q8PairScheduleId schedule);

Q8PairProblem q8_pair_problem(const Tensor& x, const Weight& first_weight,
                              const Tensor& first_out) noexcept;
// The (rows, k) problems the paired kernel registers; ops/wrapper/linear_pair.cpp asks it.
bool q8_pair_registered(std::int32_t rows, std::int32_t k) noexcept;
// The [1024,2048] geometry's weights: the DFlash drafter's K and V, adjacent row views (rows 4096
// and 5120) of one 6144-row Q8_G32 parent, which the paired kernel reads as one plane.
bool q8_pair_row_views(const Weight& first_weight, const Weight& second_weight) noexcept;
bool q8_pair_admits(const Q8PairProblem& problem) noexcept;

// The geometry of the launches `plan` makes for a problem, for the traffic they request (the same
// law as generic::route_traffic): each column tile of CTAs requests the weights again
// (ceil(cols / tile_cols) passes over both), each row tile the activation (x_row_blocks times per
// launch). nullopt for the schedules whose tile is not one column tile of a row tile (the decode
// and split-K kernels, the exact tails).
struct Q8PairGeometry {
    std::int32_t launches;     // kernel launches of the call
    std::int32_t x_row_blocks; // row tiles that each request the whole activation, per launch
    std::int32_t tile_cols;    // token columns of one CTA
};
[[nodiscard]] std::optional<Q8PairGeometry> q8_pair_geometry(const Q8PairProblem& problem);
Q8PairPlan q8_pair_resolve_plan(const Q8PairProblem& problem);

// Whether a launch of `plan` for `problem` takes the full-tile kernel variant of its tiled
// schedule (every column tile complete); false for a schedule without a tiled launch. With the
// schedule it names the kernel instance the column count decides; the kernel warm-up covers every
// such class (src/runtime/engine/warmup_widths.cpp).
[[nodiscard]] bool q8_pair_uses_full_tile(const Q8PairPlan& plan, const Q8PairProblem& problem);

void q8_pair_execute_plan(Q8PairPlan plan, const Tensor& x, const Weight& first_weight,
                          const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                          cudaStream_t stream);
void q8_pair_dispatch(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                      Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninferno::ops::detail
