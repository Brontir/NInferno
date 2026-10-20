#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

// Small-T MMA for the Q5 residual projections of the 27B (down 5120x17408, out 5120x6144) at
// verify widths, from ninfer-all's linear_add: a CTA owns one m16 row tile and streams the whole
// K through a staged activation tile, so its cost stays flat across T = 1..8 where the sliced-K
// tiles grow per column. The plan picks it per chip (q5_linear_add_plan.cpp).

namespace ninferno::ops::detail {
namespace {

constexpr int kRows = 5120;

// residual[col, row] += W x in the bf16 residual stream: one fp32 add and one rounding, as the
// sliced-K residual epilogue does.
struct Q5SmallTResidualEpilogue {
    __nv_bfloat16* out;
    std::int64_t stride;
    int columns;

    __device__ __forceinline__ void add(int row, int col, float value) const {
        __nv_bfloat16& slot = out[static_cast<std::int64_t>(col) * stride + row];
        slot                = __float2bfloat16_rn(__bfloat162float(slot) + value);
    }

    __device__ __forceinline__ void store(int row, int col0, float4 v) const {
        if (col0 < columns) {
            add(row, col0, v.x);
            add(row + 8, col0, v.z);
        }
        if (col0 + 1 < columns) {
            add(row, col0 + 1, v.y);
            add(row + 8, col0 + 1, v.w);
        }
    }
};

template <int K, int XCols, int Stages, int KWarps = 8>
void launch(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const Q5SmallTResidualEpilogue epilogue{
        static_cast<__nv_bfloat16*>(residual_out.data),
        static_cast<std::int64_t>(residual_out.nb[1] / sizeof(__nv_bfloat16)), x.ne[1]};
    q5_small_t_mma_launch<kRows, K, XCols, Stages, Q5SmallTResidualEpilogue, KWarps>(
        stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const std::uint8_t*>(w.scales), epilogue, x.ne[1]);
}

// ninfer-all's widths: up to four columns stage a four-column tile with a two-deep cp.async ring,
// 5..8 eight columns with one stage, and from 16 two row tiles share the activation slab (KWarps 4).
template <int K>
void launch_columns(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const int columns = x.ne[1];
    if (columns <= 4) {
        launch<K, 4, 2>(x, w, residual_out, stream);
    } else if (columns <= 8) {
        launch<K, 8, 1>(x, w, residual_out, stream);
    } else if (columns <= 16) {
        launch<K, 16, 2, 4>(x, w, residual_out, stream);
    } else {
        launch<K, 32, 1, 4>(x, w, residual_out, stream);
    }
}

} // namespace

bool q5_linear_add_small_t_mma_operands(const Tensor& x, const Tensor& residual_out) noexcept {
    return x.ne[1] >= 1 && x.ne[1] <= 32 && residual_out.ne[0] == kRows &&
           x.nb[1] == static_cast<std::size_t>(x.ne[0]) * sizeof(__nv_bfloat16) &&
           residual_out.nb[1] % sizeof(__nv_bfloat16) == 0;
}

void q5_linear_add_small_t_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    if (!q5_linear_add_small_t_mma_operands(x, residual_out) || w.n != kRows) {
        throw std::invalid_argument("q5 linear_add small-T MMA: needs T in [1,32], 5120 rows and "
                                    "a contiguous activation");
    }
    if (w.k == 6144 && w.padded_shape[1] == 6144) {
        launch_columns<6144>(x, w, residual_out, stream);
    } else if (w.k == 17408 && w.padded_shape[1] == 17408) {
        launch_columns<17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add small-T MMA: unsupported exact K");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
