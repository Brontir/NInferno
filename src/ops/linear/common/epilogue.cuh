#pragma once

#include "ops/common/math.cuh"
#include "ops/linear/common/output.cuh"

namespace ninferno::ops::detail {

// Thread-local operations over a fully reduced accumulator. The contraction
// owns predicates and synchronization; epilogues only receive valid coordinates.
struct LinearIdentityEpilogue {
    __device__ __forceinline__ float apply(int, int, float value) const { return value; }
};

struct LinearResidualAddEpilogue {
    LinearBf16InputView residual;

    // An explicit round-to-nearest add, never contracted with the scaling product in front of
    // it: whether nvcc fuses the two into an FMA differs between the full-tile and the predicated
    // instantiations of one GEMM, and a token's result must not depend on which of them its call
    // took.
    __device__ __forceinline__ float apply(int row, int token, float value) const {
        return __fadd_rn(value, residual.load(row, token));
    }
};

// SwiGLU over a gate row and its paired up row: out = silu(gate) * up.
struct SwiGluPairEpilogue {
    template <class Output>
    __device__ __forceinline__ void apply_pair(Output output, int row, int token, float gate,
                                               float up) const {
        output.store(row, token, silu(gate) * up);
    }
};

// Row consumers run in the single thread owning the fully reduced row. They
// may publish fused Op state, but do not introduce warp/CTA synchronization.
template <class Output, class Epilogue, int Tokens>
__device__ __forceinline__ void
linear_finish_row(const Output& output, const Epilogue& epilogue, int row, int token_begin,
                  const float (&values)[Tokens], int active_tokens) {
    if constexpr (requires {
                      epilogue.apply_row(output, row, token_begin, values, active_tokens);
                  }) {
        epilogue.apply_row(output, row, token_begin, values, active_tokens);
    } else {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if (token < active_tokens) {
                const int global_token = token_begin + token;
                output.store(row, global_token, epilogue.apply(row, global_token, values[token]));
            }
        }
    }
}

} // namespace ninferno::ops::detail
