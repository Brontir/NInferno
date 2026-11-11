#pragma once

// Whether the planner offers NVFP4 weights on a card (plan_quants.cpp: a listed NVFP4 quantization
// is planned only there): FP4 tensor cores, compute capability 10.0 and up. `compute_capability`
// is major * 10 + minor; 0 (unknown) offers it. One rule for the planner and the generic path's
// coverage lint (tests/ops/linear/test_coverage_report.cpp), which serves NVFP4 cells only there.

namespace ninferno::product {

[[nodiscard]] constexpr bool nvfp4_planned_on(int compute_capability) noexcept {
    return compute_capability == 0 || compute_capability >= 100;
}

} // namespace ninferno::product
