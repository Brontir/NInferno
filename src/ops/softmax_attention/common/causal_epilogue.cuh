#pragma once

#include "ops/softmax_attention/common/causal_tile_io.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/common/math.cuh"

namespace ninferno::ops::detail {

__device__ __forceinline__ void causal_store_partial_pair(float* target, float a, float b) {
    *reinterpret_cast<float2*>(target) = make_float2(a, b);
}

template <class G>
__device__ __forceinline__ void causal_store_output_pair(__nv_bfloat16* out, int head, int d,
                                                         int token, float a, float b) {
    *reinterpret_cast<unsigned*>(out + causal_q_index<G>(head, d, token)) = pack_bf16x2(a, b);
}

__device__ __forceinline__ void causal_store_output(__nv_bfloat16* out, float value) {
    *out = __float2bfloat16(value);
}

// The attention result, times sigmoid(*gate) when gate is set, bit-exact with ops::sigmoid_mul
// applied after the Op: that kernel reads the stored BF16 result, so this rounds first, widens
// back, multiplies in FP32 and rounds to nearest. Keeping the FP32 value would be more accurate
// and would move tokens. A masked column's zero is multiplied as well: 0 * sigmoid(g) is zero
// for every finite g, but the standalone multiply propagates a NaN gate.
__device__ __forceinline__ __nv_bfloat16 causal_gated_output(float value,
                                                             const __nv_bfloat16* gate) {
    const __nv_bfloat16 rounded = __float2bfloat16(value);
    if (gate == nullptr) return rounded;
    return __float2bfloat16_rn(__bfloat162float(rounded) * sigmoid(__bfloat162float(*gate)));
}

// A warp converts one normalized row from the represented rotated V coordinates.
template <class G>
__device__ __forceinline__ void
causal_store_inverse_rotated_row(const float* row, __nv_bfloat16* out, int head, int token) {
    static_assert(G::kHeadDim == kCausalHeadDim);
    const int lane = threadIdx.x & 31;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = row[lane + 32 * r];
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r)
        causal_store_output(out + causal_q_index<G>(head, lane + 32 * r, token), values[r]);
}

} // namespace ninferno::ops::detail
