// The epilogue of the ggml matrix kernel's route (Family::GgmlMmq): its FP32 plane [T][plane_rows]
// to the call's BF16 segments, with the residual add, the gated product or relu^2 of the call.
#include "ops/generic/generic_gemm_internal.h"
#include "ops/common/arch_limits.h"

#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninferno::ops::detail::generic {
namespace {

// GELU's tanh form through the logistic, as the generic route computes it (generic_gelu_tanh): the
// sum 0.5 * (1 + tanh(u)) cancels to 0 for a gate below about -5.
__device__ __forceinline__ float gelu_tanh(float g) {
    constexpr float kTwoSqrt2Pi = 1.59576912160573071176F; // 2 * sqrt(2 / pi)
    return g / (1.0F + expf(-kTwoSqrt2Pi * (g + 0.044715F * g * g * g)));
}

__device__ __forceinline__ float epilogue_value(const GemmArgs& a, Epilogue epilogue, float v,
                                                float up) {
    if (epilogue == Epilogue::SwiGlu) {
        return (a.gelu_gate != 0 ? gelu_tanh(v) : v / (1.0F + __expf(-v))) * up;
    }
    if (a.relu2 != 0) {
        v = fmaxf(v, 0.0F);
        return v * v;
    }
    return v;
}

// V consecutive rows per thread (V = 4: one 16-byte plane load and one 8-byte BF16 store per token,
// where every segment's rows, stride and base keep the four in one aligned word).
template <int V>
__global__ void mmq_epilogue_kernel(GemmArgs a, Epilogue epilogue, const float* __restrict__ plane,
                                    int plane_rows) {
    // A programmatic dependent of the tile kernel or its fixup; it releases nothing early (a
    // following call may quantize into the workspace this call's plane lives in).
    pdl::wait_for_dependencies();
    const int r = (blockIdx.x * blockDim.x + threadIdx.x) * V;
    if (r >= a.rows) return;
    int s = 0;
    while (s + 1 < a.segments && r >= a.segment_end[s]) ++s;
    const int local = r - (s > 0 ? a.segment_end[s - 1] : 0);
    // A grid's y extent stops at 65535: a longer prefill chunk strides its tokens.
    for (int t = blockIdx.y; t < a.tokens; t += gridDim.y) {
        const float* column = plane + static_cast<std::size_t>(t) * plane_rows;
        auto* out =
            static_cast<__nv_bfloat16*>(a.out[s]) + static_cast<std::size_t>(t) * a.ld[s] + local;
        if constexpr (V == 1) {
            float v = epilogue_value(a, epilogue, column[r],
                                     epilogue == Epilogue::SwiGlu ? column[a.rows + r] : 0.0F);
            if (epilogue == Epilogue::AddResidual) v += __bfloat162float(*out);
            *out = __float2bfloat16(v);
        } else {
            const float4 g = *reinterpret_cast<const float4*>(column + r);
            float4 u       = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (epilogue == Epilogue::SwiGlu) {
                u = *reinterpret_cast<const float4*>(column + a.rows + r);
            }
            float v[4] = {epilogue_value(a, epilogue, g.x, u.x),
                          epilogue_value(a, epilogue, g.y, u.y),
                          epilogue_value(a, epilogue, g.z, u.z),
                          epilogue_value(a, epilogue, g.w, u.w)};
            if (epilogue == Epilogue::AddResidual) {
                const uint2 old = *reinterpret_cast<const uint2*>(out);
                const float2 lo =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&old.x));
                const float2 hi =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&old.y));
                v[0] += lo.x;
                v[1] += lo.y;
                v[2] += hi.x;
                v[3] += hi.y;
            }
            const __nv_bfloat162 lo = __floats2bfloat162_rn(v[0], v[1]);
            const __nv_bfloat162 hi = __floats2bfloat162_rn(v[2], v[3]);
            uint2 word;
            word.x                           = *reinterpret_cast<const unsigned*>(&lo);
            word.y                           = *reinterpret_cast<const unsigned*>(&hi);
            *reinterpret_cast<uint2*>(out) = word;
        }
    }
}

// Every segment holds whole aligned four-row words: the four-row kernel's condition.
bool four_rows(const GemmArgs& a, int plane_rows) {
    if ((a.rows % 4) != 0 || (plane_rows % 4) != 0) return false;
    for (int s = 0; s < a.segments; ++s) {
        if ((a.segment_end[s] % 4) != 0 || (a.ld[s] % 4) != 0 ||
            (reinterpret_cast<std::uintptr_t>(a.out[s]) % 8) != 0) {
            return false;
        }
    }
    return true;
}

} // namespace

void launch_mmq_epilogue(Epilogue epilogue, const GemmArgs& args, const float* plane,
                         int plane_rows, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const bool four        = four_rows(args, plane_rows) &&
                      (reinterpret_cast<std::uintptr_t>(plane) % 16) == 0;
    const int per_block    = kThreads * (four ? 4 : 1);
    const dim3 grid((args.rows + per_block - 1) / per_block, std::min(args.tokens, kMaxGridYZ));
    if (four) {
        CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream},
                                         mmq_epilogue_kernel<4>, args, epilogue, plane,
                                         plane_rows));
    } else {
        CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream},
                                         mmq_epilogue_kernel<1>, args, epilogue, plane,
                                         plane_rows));
    }
}

} // namespace ninferno::ops::detail::generic
