#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "core/weight.h"
#include "ops/linear_add/q8/q8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_simt_launch.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

constexpr int kRowsPerBlock = 8;
constexpr int kStages       = 2;

template <int RowsPerCta, int K = 6144, int T = 1>
void launch_decode(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8A16GemvSchedule<RowsPerCta, 1, 2, K, K / 256, T>;
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(residual_out.data),
                                  residual_out.ne[0]};
    launch_q8_a16_gemv<Schedule>(q8_linear_operands(x, w), output,
                                 LinearResidualAddEpilogue{{output.data, output.rows, 0}}, stream);
}

template <int ColsPerTile>
void launch_variant(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8A16SimtSchedule<kRowsPerBlock, ColsPerTile, 1, 32, kStages, Cache::cg, 1>;
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(residual_out.data),
                                  residual_out.ne[0]};
    launch_q8_a16_simt<Schedule>(q8_linear_operands(x, w), output,
                                 LinearResidualAddEpilogue{{output.data, output.rows, 0}}, stream);
}

} // namespace

namespace {

// The K=4096 GEMV at the call's exact T (1..8): each weight load serves every token, a token's
// sum runs in the same order at every T.
template <int RowsPerCta>
void launch_decode_k4096(const Tensor& x, const Weight& w, Tensor& residual_out,
                         cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        return launch_decode<RowsPerCta, 4096, 1>(x, w, residual_out, stream);
    case 2:
        return launch_decode<RowsPerCta, 4096, 2>(x, w, residual_out, stream);
    case 3:
        return launch_decode<RowsPerCta, 4096, 3>(x, w, residual_out, stream);
    case 4:
        return launch_decode<RowsPerCta, 4096, 4>(x, w, residual_out, stream);
    case 5:
        return launch_decode<RowsPerCta, 4096, 5>(x, w, residual_out, stream);
    case 6:
        return launch_decode<RowsPerCta, 4096, 6>(x, w, residual_out, stream);
    case 7:
        return launch_decode<RowsPerCta, 4096, 7>(x, w, residual_out, stream);
    case 8:
        return launch_decode<RowsPerCta, 4096, 8>(x, w, residual_out, stream);
    default:
        throw std::invalid_argument("q8 linear_add: the K=4096 decode GEMV takes T=1..8");
    }
}

} // namespace

void q8_linear_add_decode_k4096_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       int rows_per_cta, cudaStream_t stream) {
    switch (rows_per_cta) {
    case 4:
        launch_decode_k4096<4>(x, w, residual_out, stream);
        return;
    case 8:
        launch_decode_k4096<8>(x, w, residual_out, stream);
        return;
    case 16:
        launch_decode_k4096<16>(x, w, residual_out, stream);
        return;
    default:
        throw std::invalid_argument("q8 linear_add: K=4096 decode takes 4, 8 or 16 rows a CTA");
    }
}

void q8_linear_add_decode_r4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream) {
    launch_decode<4>(x, w, residual_out, stream);
}

void q8_linear_add_decode_r8_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream) {
    launch_decode<8>(x, w, residual_out, stream);
}

void q8_linear_add_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    launch_decode<16>(x, w, residual_out, stream);
}

void q8_linear_add_simt_r8_c4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    launch_variant<4>(x, w, residual_out, stream);
}

void q8_linear_add_simt_r8_c8_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    launch_variant<8>(x, w, residual_out, stream);
}

} // namespace ninferno::ops::detail
