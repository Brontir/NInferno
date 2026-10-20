#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail {

enum class Q8GdnInputScheduleId {
    DecodeR8Direct,
    SplitKMmaDirect,
    MmaR64C128,
    // Prefill profile, 17..96 columns: the MMA contraction on a narrow tile.
    MmaPrefillSmallT,
};

enum class Q8GdnInputConvScheduleId {
    DecodeFused,
    SplitKMmaFused,
    Materialized,
};

struct Q8GdnInputProblem {
    std::int32_t input_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q8GdnInputPlan {
    Q8GdnInputScheduleId schedule;
};

struct Q8GdnInputConvPlan {
    Q8GdnInputConvScheduleId schedule;
};

bool q8_gdn_input_admits(const Q8GdnInputProblem& problem) noexcept;
Q8GdnInputPlan q8_gdn_input_resolve_plan(const Q8GdnInputProblem& problem);
Q8GdnInputConvPlan q8_gdn_input_conv_resolve_plan(const Q8GdnInputProblem& problem,
                                                  std::int32_t batch_size);

void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream);

} // namespace ninferno::ops::detail
