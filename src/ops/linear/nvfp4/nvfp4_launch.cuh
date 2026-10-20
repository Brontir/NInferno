#pragma once
#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_simt.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_a16_reg_mma.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/common/shape_routes.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace ninferno::ops::detail {
using Nvfp4A4Launch = void (*)(const Weight&, Tensor&, Nvfp4A4Workspace, std::int32_t,
                               cudaStream_t);

// A selected A4 route: the GEMM to run, and the activation-scale layout that GEMM reads. They
// travel together because the quantizer writes the plane before the GEMM runs and the two must
// agree; a shape selects a route rather than selecting the two halves separately.
struct Nvfp4A4Route {
    Nvfp4A4Launch launch;
    Nvfp4ScaleLayout scales;
};

template <class Geometry, class Schedule>
void nvfp4_linear_a16_gemv(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
    launch_nvfp4_a16_gemv<S>(nvfp4_a16_operands(x, w),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                             LinearIdentityEpilogue{}, stream);
}

template <class Geometry, int Capacity, class Schedule, bool Full = false>
void nvfp4_linear_a16_simt(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, Capacity, Full>;
    launch_nvfp4_a16_simt<S>(nvfp4_a16_operands(x, w),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                             LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void nvfp4_linear_a16_mma(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_nvfp4_a16_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void nvfp4_linear_a16_sliced_k(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

// A16 on tensor cores for small T with weights dequantized in registers: the chunk's tokens against
// the whole weight, activations in BF16 throughout, one K order whatever T (nvfp4_a16_reg_mma.cuh).
template <class Geometry, class Schedule>
void launch_nvfp4_a16_reg_mma(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    if (x.ne[1] > Schedule::kTokens) {
        throw std::logic_error("nvfp4 A16 MMA: chunk exceeds the schedule's token capacity");
    }
    nvfp4_a16_reg_mma_kernel<Geometry, Schedule>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), 1.0F / weight.weight_scale_divisor,
            LinearIdentityEpilogue{},
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
            x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <int Chunk, Nvfp4Launch (*Select)(std::int32_t)>
void launch_nvfp4_a16_chunks(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream) {
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += Chunk) {
        const int count = std::min(Chunk, x.ne[1] - offset);
        auto input      = x.slice(1, offset, count);
        auto output     = out.slice(1, offset, count);
        Select(route_tokens({QType::NVFP4, ShapeCompute::A16, weight.n, weight.k}, count))(
            input, weight, output, stream);
    }
}

// launch_nvfp4_a16_chunks for the Gemma 4 prefill. The narrower schedules sum K in another order
// than the full chunk's (another code run width or another K split across warps). Plain chunking
// would run a call's ragged tail on one of them, so a token's bytes would depend on where its call
// ends, and a prompt cut elsewhere would compute its tokens differently. A call of more than
// kVerifyTokens tokens therefore runs every chunk, the tail included, on the full chunk's schedule
// (the kernel masks the missing columns). Calls of up to kVerifyTokens (decode, speculative verify)
// keep the tuned narrow schedules; the profile-default A4 routes take over from 17 tokens anyway.
template <int Chunk, Nvfp4Launch (*Select)(std::int32_t)>
void launch_nvfp4_a16_chunks_prefill(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    constexpr std::int32_t kVerifyTokens = 16;
    if (x.ne[1] <= kVerifyTokens) {
        launch_nvfp4_a16_chunks<Chunk, Select>(x, weight, out, stream);
        return;
    }
    const Nvfp4Launch full =
        Select(route_tokens({QType::NVFP4, ShapeCompute::A16, weight.n, weight.k}, Chunk));
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += Chunk) {
        const int count = std::min(Chunk, x.ne[1] - offset);
        auto input      = x.slice(1, offset, count);
        auto output     = out.slice(1, offset, count);
        full(input, weight, output, stream);
    }
}

template <class Geometry, class Schedule>
void nvfp4_linear_a4_mma(const Weight& w, Tensor& y, Nvfp4A4Workspace workspace, int tokens,
                         cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(w, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n}, LinearIdentityEpilogue{},
        stream);
}

template <Nvfp4GeometryId Geometry, Nvfp4ScaleLayout Layout>
void nvfp4_linear_a4_tma(const Weight& weight, Tensor& out, Nvfp4A4Workspace scratch, int tokens,
                         cudaStream_t stream) {
    launch_nvfp4_a4_tma_linear(Geometry, nvfp4_a4_operands(weight, scratch, tokens, Layout),
                               static_cast<__nv_bfloat16*>(out.data), stream);
}

template <Nvfp4GeometryId Geometry, Nvfp4ScaleLayout Layout = Nvfp4ScaleLayout::Tiled256>
constexpr Nvfp4A4Route nvfp4_a4_tma_route() {
    return {nvfp4_linear_a4_tma<Geometry, Layout>, Layout};
}

template <class Geometry, class Schedule>
constexpr Nvfp4A4Route nvfp4_a4_mma_route() {
    return {nvfp4_linear_a4_mma<Geometry, Schedule>, Nvfp4ScaleLayout::RowMajor};
}

template <Nvfp4A4Route (*Select)(std::int32_t)>
void launch_nvfp4_a4(const Tensor& x, const Weight& weight, Tensor& out, Nvfp4A4Workspace scratch,
                     cudaStream_t stream) {
    const Nvfp4A4Route route =
        Select(route_tokens({QType::NVFP4, ShapeCompute::A4, weight.n, weight.k}, x.ne[1]));
    launch_nvfp4_a4_quantize(x, weight, scratch, route.scales, stream);
    route.launch(weight, out, scratch, x.ne[1], stream);
}
} // namespace ninferno::ops::detail
