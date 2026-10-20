#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "core/device_class.h"
#include "core/measured_boards.h"
#include "ops/common/small_t_bands.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_simt_launch.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

template <int Cols, int Stride>
void launch_split2(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    using Schedule    = Q5A16DirectSimtSchedule<1, Cols, 2, 8, 16, Stride, true>;
    auto* data        = static_cast<__nv_bfloat16*>(residual_out.data);
    const auto stride = std::int64_t(residual_out.nb[1] / sizeof(__nv_bfloat16));
    launch_q5_a16_direct_simt<Schedule>(q5_linear_operands(x, w),
                                        LinearBf16StridedOutput{data, stride, 0},
                                        LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}

// The warp-per-row split2 kernel the unified templates replaced (q5_rowsplit_gemm_simt.cuh, kept by
// ninfer-all): two warps per row stream one 1024-value slab each through a cp.async double
// buffer. On the 3090 it takes the 27B down projection at T = 1 in 70.8 us against the direct SIMT
// tile's 87.9 (out: 30.7 against 34.1). Needs a contiguous activation and an exact residual stride.
template <int Cols, int FullSlabs, int Stride>
void launch_rowsplit_split2(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(residual_out.ne[0]), 1u, 1u);
    q5_rowsplit_gemm_simt_split2_kernel<Cols, FullSlabs, Stride, true><<<grid, 64, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0], w.padded_shape[1]);
}

// Which split2 kernel a chip takes (1: the row-split kernel above, 0: the direct SIMT tile), from
// bench/ops/q5_linear_add_bench at T = 1..4, row split against the direct tile.
// RTX 4090: 25.6 against 26.6 us at T = 1 (K = 6144), 66.6 against 67.6 (K = 17408). RTX 5090:
// 41.0 against 38.9 us at K = 17408, so off there and on chips not measured.
constexpr SmallTBand kRowSplitFallbackBands[] = {{4, 0}};
constexpr SmallTBand kRowSplitGa102Bands[]    = {{4, 1}};
constexpr SmallTBand kRowSplitAd102Bands[]    = {{4, 1}};
constexpr SmallTBand kRowSplitGb203Bands[]    = {{4, 1}}; // 26.6 against 28.7 us at T = 1, K = 6144
// Tesla V100 at the MTP4 verify width (T = 5, ninfer-v100-sm70's route): 192.6 us against the
// exact-column direct tile's ~230 on the 27B down projection (K = 17408) and 68.7 against ~77 on
// out (K = 6144).
// T = 5..8 reach here through q5_linear_add_resolve_plan's Volta branch.
constexpr SmallTBand kRowSplitGv100Bands[]    = {{4, 0}, {8, 1}};
constexpr SmallTChipBands kRowSplitMeasured[] = {{measured::kRtx3090, kRowSplitGa102Bands},
                                                 {measured::kRtx4090, kRowSplitAd102Bands},
                                                 {measured::kRtx5080, kRowSplitGb203Bands},
                                                 {measured::kV100Sxm2, kRowSplitGv100Bands}};

bool rowsplit_taken(const Tensor& x, const Tensor& residual_out) noexcept {
    if (x.nb[1] != static_cast<std::size_t>(x.ne[0]) * sizeof(__nv_bfloat16) ||
        residual_out.nb[1] != static_cast<std::size_t>(residual_out.ne[0]) * sizeof(__nv_bfloat16)) {
        return false;
    }
    thread_local SmallTBandCache cache;
    return small_t_band_variant(kRowSplitMeasured, kRowSplitFallbackBands, cache, x.ne[1]) > 0;
}

template <int Cols>
void dispatch_shape(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const bool rowsplit = rowsplit_taken(x, residual_out);
    if (w.k == 6144) {
        if (rowsplit) {
            launch_rowsplit_split2<Cols, 6, 6144>(x, w, residual_out, stream);
        } else {
            launch_split2<Cols, 6144>(x, w, residual_out, stream);
        }
    } else if (w.k == 17408) {
        if (rowsplit) {
            launch_rowsplit_split2<Cols, 17, 17408>(x, w, residual_out, stream);
        } else {
            launch_split2<Cols, 17408>(x, w, residual_out, stream);
        }
    } else {
        throw std::invalid_argument("q5 linear_add split2: unsupported exact K");
    }
}

template <class Launch>
void dispatch_cols(std::int32_t cols, Launch&& launch) {
    switch (cols) {
#define NINFERNO_Q5_LINEAR_ADD_EXACT(COLS)                                                         \
    case COLS:                                                                                     \
        launch.template operator()<COLS>();                                                        \
        return
        NINFERNO_Q5_LINEAR_ADD_EXACT(1);
        NINFERNO_Q5_LINEAR_ADD_EXACT(2);
        NINFERNO_Q5_LINEAR_ADD_EXACT(3);
        NINFERNO_Q5_LINEAR_ADD_EXACT(4);
        NINFERNO_Q5_LINEAR_ADD_EXACT(5);
        NINFERNO_Q5_LINEAR_ADD_EXACT(6);
        NINFERNO_Q5_LINEAR_ADD_EXACT(7);
        NINFERNO_Q5_LINEAR_ADD_EXACT(8);
#undef NINFERNO_Q5_LINEAR_ADD_EXACT
    default:
        throw std::invalid_argument("q5 linear_add split2: T must be in [1,8]");
    }
}

} // namespace

void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    dispatch_cols(x.ne[1], [&]<int Cols>() { dispatch_shape<Cols>(x, w, residual_out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninferno::ops::detail
