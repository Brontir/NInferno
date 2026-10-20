#include "ops/linear/q5/q5_instances.cuh"
#include "ops/linear/q4/q4_instances.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/q5/q5_sliced_k_launch.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

constexpr std::int32_t kSplitRow = 6144;
constexpr std::int32_t kHidden   = 5120;

LinearBf16SplitOutput2<kSplitRow> split_output(Tensor& first, Tensor& second) {
    return {{static_cast<__nv_bfloat16*>(first.data),
             static_cast<std::int64_t>(first.nb[1] / sizeof(__nv_bfloat16)), 0},
            {static_cast<__nv_bfloat16*>(second.data),
             static_cast<std::int64_t>(second.nb[1] / sizeof(__nv_bfloat16)), 0}};
}

// Both parents of the projection on the sliced-K MMA kernels, one call per parent. The grouped
// tiles pad every call to 32 columns and re-read the activation per 32-row tile; the sliced-K form
// stages one activation slab per CTA and splits K across warps, so at 13..32 columns the weight
// pass stays close to one read. Same variants as the GDN input; variant v picks rows per CTA,
// K-warps and stages (A/B only).
template <int BlockRows, int KWarps, int Stages, int MinBlocks, int Tokens>
void launch_pair(const Tensor& x, const Weight& qk_weight, const Weight& gv_weight, Tensor& q,
                 Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    using Q4Schedule = Q4A16SlicedKMmaSchedule<BlockRows, Tokens, KWarps, Stages, Cache::cg,
                                               Cache::ca, MinBlocks, kHidden, Tokens>;
    using Q5Schedule = Q5A16SlicedKMmaSchedule<BlockRows, Tokens, KWarps, Stages, Cache::cg,
                                               Cache::ca, MinBlocks, kHidden, Tokens>;
    launch_q5_a16_sliced_k_mma<Q5Schedule>(q5_linear_operands(x, gv_weight), split_output(gate, v),
                                           LinearIdentityEpilogue{}, stream);
    launch_q4_a16_sliced_k_mma<Q4Schedule>(q4_linear_operands(x, qk_weight), split_output(q, k),
                                           LinearIdentityEpilogue{}, stream);
}

template <int BlockRows, int KWarps, int Stages, int MinBlocks>
void launch_tokens(const Tensor& x, const Weight& qk_weight, const Weight& gv_weight, Tensor& q,
                   Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const int columns = x.ne[1];
    if (columns <= 8) {
        launch_pair<BlockRows, KWarps, Stages, MinBlocks, 8>(x, qk_weight, gv_weight, q, gate, k, v,
                                                             stream);
    } else if (columns <= 16) {
        launch_pair<BlockRows, KWarps, Stages, MinBlocks, 16>(x, qk_weight, gv_weight, q, gate, k,
                                                              v, stream);
    } else if (columns <= 24) {
        launch_pair<BlockRows, KWarps, Stages, MinBlocks, 24>(x, qk_weight, gv_weight, q, gate, k,
                                                              v, stream);
    } else {
        launch_pair<BlockRows, KWarps, Stages, MinBlocks, 32>(x, qk_weight, gv_weight, q, gate, k,
                                                              v, stream);
    }
}

} // namespace

void q4_q5_attn_input_sliced_launch(const Tensor& x, const Weight& qk_weight,
                                    const Weight& gv_weight, Tensor& q, Tensor& gate, Tensor& k,
                                    Tensor& v, int variant, cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4/Q5 attention input sliced-K requires T in [1,32]");
    }
    if (x.ne[0] != kHidden || qk_weight.padded_shape[1] != kHidden ||
        gv_weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("Q4/Q5 attention input sliced-K requires padded K == hidden");
    }
    switch (variant) {
    case 1:
        launch_tokens<16, 8, 1, 3>(x, qk_weight, gv_weight, q, gate, k, v, stream);
        break;
    case 2:
        launch_tokens<16, 4, 2, 4>(x, qk_weight, gv_weight, q, gate, k, v, stream);
        break;
    case 3:
        launch_tokens<32, 4, 2, 3>(x, qk_weight, gv_weight, q, gate, k, v, stream);
        break;
    case 4:
        launch_tokens<32, 8, 1, 2>(x, qk_weight, gv_weight, q, gate, k, v, stream);
        break;
    default:
        throw std::invalid_argument("Q4/Q5 attention input sliced-K: unknown variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
