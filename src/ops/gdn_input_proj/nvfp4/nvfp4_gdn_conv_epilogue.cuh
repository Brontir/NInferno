#pragma once
#include "ops/gdn_input_proj/gdn_conv_output.cuh"

namespace ninferno::ops::detail {
template <int Tokens>
struct Nvfp4GdnConvEpilogue {
    static constexpr int kRowTokens = Tokens;
    // Read here too: nvcc counts a member only `requires` asks for as unreferenced (#177).
    static_assert(kRowTokens >= 1);

    template <class Output>
    __device__ __forceinline__ void apply_row(Output output, int row, int,
                                              const float (&values)[Tokens], int) const {
        output.store_row(row, values);
    }
};
} // namespace ninferno::ops::detail
