// The register MMA's group-sums pre-pass (#619): the activation sums a min-term format's register
// MMA reads (GemmArgs::group_sums), summed once per call instead of by an MMA with ones in every
// warp and row tile. The twin of the W4A8 quantizer's sums plane (generic_w4a8.cu).
#include "ops/common/memory.cuh"
#include "ops/generic/generic_gemm_internal.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninferno::ops::detail::generic {
namespace {

// The sum of v[0 .. N) in one fixed order: the two halves' sums added, each half the same way
// (for 32 values: eight sums of four consecutive values ((v0 + v1) + (v2 + v3)), then a balanced
// tree of the eight). Every token, group, tile, width, board and architecture sums in this order;
// the test's CPU reference repeats it bit for bit.
template <int N>
__device__ __forceinline__ float rmma_group_sum(const float* v) {
    if constexpr (N == 1) {
        return v[0];
    } else {
        return rmma_group_sum<N / 2>(v) + rmma_group_sum<N / 2>(v + N / 2);
    }
}

// One thread per (token, group) of the plane's rows (ld groups, the ones past K zero): the G BF16
// values of the group, widened to FP32 and summed (rmma_group_sum). Vec: x is 16-byte aligned and
// its token stride whole 16-byte words, so a group is G / 8 16-byte loads; otherwise 2-byte loads
// of the same values (the same sum). Rows of padded tokens are not written: no kernel reads them.
template <int G, bool Vec>
__global__ __launch_bounds__(256) void rmma_group_sums_kernel(const __nv_bfloat16* __restrict__ x,
                                                              int ldx, int k, int tokens, int ld,
                                                              float* __restrict__ out) {
    static_assert(G % 8 == 0, "a group of whole 16-byte words");
    const std::int64_t task =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (task >= static_cast<std::int64_t>(tokens) * ld) return;
    const int token = static_cast<int>(task / ld);
    const int group = static_cast<int>(task - static_cast<std::int64_t>(token) * ld);
    float sum       = 0.0F;
    if (group * G < k) {
        const __nv_bfloat16* src = x + static_cast<std::int64_t>(token) * ldx + group * G;
        float v[G];
        if constexpr (Vec) {
#pragma unroll
            for (int i = 0; i < G / 8; ++i) {
                const uint4 w       = load_vec<uint4>(src + 8 * i);
                const unsigned u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    v[8 * i + 2 * j]     = __uint_as_float(u[j] << 16);
                    v[8 * i + 2 * j + 1] = __uint_as_float(u[j] & 0xffff0000U);
                }
            }
        } else {
            const auto* bits = reinterpret_cast<const unsigned short*>(src);
#pragma unroll
            for (int i = 0; i < G; ++i)
                v[i] = __uint_as_float(static_cast<unsigned>(bits[i]) << 16);
        }
        sum = rmma_group_sum<G>(v);
    }
    out[task] = sum;
}

template <int G>
void launch_group_sums(const GemmArgs& args, float* out, cudaStream_t stream) {
    const std::int64_t tasks = static_cast<std::int64_t>(args.tokens) * args.group_sums_ld;
    if (tasks <= 0) return;
    const auto blocks = static_cast<unsigned>((tasks + 255) / 256);
    const auto* x     = static_cast<const __nv_bfloat16*>(args.x);
    const bool vec    = (reinterpret_cast<std::uintptr_t>(args.x) % 16) == 0 && (args.ldx % 8) == 0;
    if (vec) {
        rmma_group_sums_kernel<G, true>
            <<<blocks, 256, 0, stream>>>(x, args.ldx, args.k, args.tokens, args.group_sums_ld, out);
    } else {
        rmma_group_sums_kernel<G, false>
            <<<blocks, 256, 0, stream>>>(x, args.ldx, args.k, args.tokens, args.group_sums_ld, out);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_rmma_group_sums(const GemmArgs& args, float* out, cudaStream_t stream) {
    const int g = args.group_sums_values;
    if (g <= 0 || (args.k % g) != 0) {
        throw std::invalid_argument("generic linear: the group-sums plane needs K of whole groups");
    }
    if (args.group_sums_ld * g < args.k) {
        throw std::invalid_argument("generic linear: the group-sums plane's rows are short of K");
    }
    if (kernel_describe_target() != nullptr) return; // described: nothing runs
    switch (g) {
    case 32:
        launch_group_sums<32>(args, out, stream);
        return;
    case 16:
        launch_group_sums<16>(args, out, stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("generic linear: no group-sums pre-pass for groups of " +
                                std::to_string(g) + " values");
}

} // namespace ninferno::ops::detail::generic
