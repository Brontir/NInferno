#pragma once

#include "core/weight.h"
#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops::detail {

enum class Q8LinearAddScheduleId {
    DecodeR16,
    // K=4096 at T <= 4: the fully unrolled GEMV at the call's exact T.
    DecodeK4096,
    SplitKMmaExactT,
    SplitKMmaCapacity,
    GroupedSplitK,
    MediumSplitK,
    SimtR8C4,
    MmaR32C64,
    MmaR32C80,
    MmaR32C96,
    MmaR32C128,
    MmaR48C64,
    MmaR48C96,
    MmaR48C112,
    MmaR48C128,
    MmaR64C96,
    MmaR64C112,
    MmaR64C128,
    MmaR128C64,
    MmaR128C80,
    // Prefill profile, 17..128 columns: the MMA contraction on a narrow tile chosen by T. It
    // sums a column's K exactly as the MMA routes above do.
    MmaPrefillSmallT,
};

struct Q8LinearAddPlan {
    Q8LinearAddScheduleId schedule;
};

struct Q8LinearAddProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

const char* q8_linear_add_schedule_name(Q8LinearAddScheduleId schedule) noexcept;
// The (rows, k) problems this plan registers; ops/wrapper/linear_add.cpp asks it.
bool q8_linear_add_registered(std::int32_t rows, std::int32_t k) noexcept;
bool q8_linear_add_admits(const Q8LinearAddProblem& problem) noexcept;
Q8LinearAddPlan q8_linear_add_resolve_plan(const Q8LinearAddProblem& problem);

// The kernel instance a launch of `plan` for `problem` runs, as far as the column count decides it:
// the schedule, whether its tiled launch takes the full-tile variant (every column tile complete),
// and the band of a launcher that picks its tile inside (the medium split-K bands, the prefill
// small-T tile, T itself for the exact-T split-K). Two column counts of one problem shape with
// equal classes launch the same kernels; the kernel warm-up covers every class
// (src/runtime/engine/warmup_widths.cpp).
struct Q8LinearAddKernelClass {
    Q8LinearAddScheduleId schedule;
    bool full;
    std::int32_t band;
};

[[nodiscard]] Q8LinearAddKernelClass q8_linear_add_kernel_class(const Q8LinearAddPlan& plan,
                                                                const Q8LinearAddProblem& problem);

void q8_linear_add_execute_plan(const Q8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream);
void q8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream);

} // namespace ninferno::ops::detail
