#pragma once
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace ninferno::ops::detail {
template <int RowsPerBranch>
struct Nvfp4SwiGluRows {
    static constexpr bool kPaired = true;

    __device__ __forceinline__ int weight_row(int begin, int row, int rows) const {
        return begin + row % RowsPerBranch + (row >= RowsPerBranch ? rows / 2 : 0);
    }
};
} // namespace ninferno::ops::detail
