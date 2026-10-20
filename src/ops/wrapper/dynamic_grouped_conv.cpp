#include "core/weight.h"
#include "ninferno/ops/dynamic_grouped_conv.h"

#include "ops/common/tensor_checks.h"
#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_plan.h"
#include "ops/dynamic_grouped_conv/generic/generic_dynamic_grouped_conv.h"
#include "ninferno/ops/linear.h"
#include "core/layout.h"
#include "ninferno/ops/rmsnorm.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

constexpr std::int32_t kHidden          = 5120;
constexpr std::int32_t kGroups          = 320;
constexpr std::int32_t kTaps            = 2;
constexpr std::int32_t kSides           = 2;
constexpr std::int32_t kCoefficientRows = kGroups * kTaps * kSides;
constexpr const char* kPrepareOp        = "dynamic grouped conv prepare";
constexpr const char* kAddOp            = "linear dynamic grouped conv add";

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t d0, std::int32_t d1,
                    std::int32_t d2, std::int32_t d3, const char* op, const char* label) {
    if (tensor.dtype != dtype || tensor.ne[0] != d0 || tensor.ne[1] != d1 || tensor.ne[2] != d2 ||
        tensor.ne[3] != d3 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

// The fused kernels' formats; any other projection takes the decomposed route.
bool fused_prepare(QType format) { return format == QType::BF16; }
bool fused_add(QType format) { return format == QType::Q8_G32_FP16; }

void require_projection(const Weight& weight, std::int32_t rows, std::int32_t cols,
                        const char* op) {
    if (weight.ndim != 2 || weight.n != rows || weight.k != cols || weight.payload == nullptr ||
        weight.qtype == QType::FP32 || weight.qtype == QType::INT32) {
        throw std::invalid_argument(std::string(op) + ": invalid projection weight");
    }
}

std::size_t decomposed_capacity(QType format, std::int32_t rows, std::int32_t cols,
                                std::int32_t intermediate_rows, std::int32_t max_width,
                                std::int32_t max_batch_size) {
    WorkspaceLayoutBuilder layout;
    const std::int32_t columns = max_width * max_batch_size;
    (void)layout.alloc(DType::BF16, {intermediate_rows, columns}, 256);
    (void)layout.alloc(DType::BF16, {rows, columns}, 256);
    (void)layout.alloc_bytes(
        linear_workspace_capacity_bytes(format, rows, cols, LinearPolicy::A16Only, 1, columns));
    return layout.peak_bytes();
}

void require_kernel_projection_weight(const Weight& weight) {
    constexpr std::uint64_t kPayloadBytes =
        static_cast<std::uint64_t>(kCoefficientRows) * kHidden * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < kPayloadBytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != kCoefficientRows || weight.k != kHidden ||
        weight.shape[0] != kCoefficientRows || weight.shape[1] != kHidden ||
        weight.padded_shape[0] != kCoefficientRows || weight.padded_shape[1] != kHidden ||
        weight.qhigh != nullptr || weight.scales != nullptr || weight.group_size != 0 ||
        weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(
            "dynamic grouped conv prepare: invalid kernel_projection_weight");
    }
}

std::uint64_t required_q8_payload_bytes(std::int32_t input_rows) {
    const std::uint64_t rows       = kHidden;
    const std::uint64_t columns    = static_cast<std::uint64_t>(input_rows);
    const std::uint64_t code_bytes = rows * columns;
    const std::uint64_t groups     = columns / 32U;
    return code_bytes + rows * groups * sizeof(std::uint16_t);
}

void require_finish_projection_weight(const Weight& weight, std::int32_t input_rows) {
    const std::uint64_t payload_bytes = required_q8_payload_bytes(input_rows);
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != kHidden || weight.k != input_rows ||
        weight.shape[0] != kHidden || weight.shape[1] != input_rows || weight.shape[2] != 1 ||
        weight.shape[3] != 1 || weight.padded_shape[0] != kHidden ||
        weight.padded_shape[1] != input_rows || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1 || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < payload_bytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument("linear dynamic grouped conv add: invalid projection_weight");
    }
}

void require_finish_nonoverlap(const Tensor& x, const Weight& projection_weight,
                               const Tensor& base_kernel, const Tensor& finish_delta,
                               const Tensor& residual, const WorkspaceArena& workspace) {
    const std::size_t code_bytes =
        static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(x.ne[0]);
    const std::size_t scale_bytes = static_cast<std::size_t>(kHidden) *
                                    static_cast<std::size_t>(x.ne[0] / 32) * sizeof(std::uint16_t);
    const std::array<OperandRange, 7> ranges{{
        {x.data, x.bytes(), "x"},
        {projection_weight.qdata, code_bytes, "projection codes"},
        {projection_weight.scales, scale_bytes, "projection scales"},
        {base_kernel.data, base_kernel.bytes(), "base_kernel"},
        {finish_delta.data, finish_delta.bytes(), "finish_delta"},
        {residual.data, residual.bytes(), "residual"},
        {workspace.base(), workspace.capacity(), "workspace"},
    }};
    require_disjoint(ranges, kAddOp);
}

void require_nonoverlap(const Tensor& residual, const Tensor& norm_weight,
                        const Tensor& base_kernel, const Weight& kernel_projection_weight,
                        const Tensor& prepared, const Tensor& finish_delta,
                        const WorkspaceArena& workspace) {
    constexpr std::size_t kWeightBytes =
        static_cast<std::size_t>(kCoefficientRows) * kHidden * sizeof(std::uint16_t);
    const std::array<OperandRange, 7> ranges{{
        {residual.data, residual.bytes(), "residual"},
        {norm_weight.data, norm_weight.bytes(), "norm_weight"},
        {base_kernel.data, base_kernel.bytes(), "base_kernel"},
        {kernel_projection_weight.qdata, kWeightBytes, "kernel_projection_weight"},
        {prepared.data, prepared.bytes(), "prepared"},
        {finish_delta.data, finish_delta.bytes(), "finish_delta"},
        {workspace.base(), workspace.capacity(), "workspace"},
    }};
    require_disjoint(ranges, "dynamic grouped conv prepare");
}

} // namespace

std::size_t rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
    std::int32_t min_width, std::int32_t max_width, std::int32_t min_batch_size,
    std::int32_t max_batch_size) {
    return detail::bf16_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
        min_width, max_width, min_batch_size, max_batch_size);
}

std::size_t rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
    QType kernel_projection, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch_size, std::int32_t max_batch_size) {
    if (fused_prepare(kernel_projection)) {
        return rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
            min_width, max_width, min_batch_size, max_batch_size);
    }
    if (min_width < 2 || max_width > 16 || max_width < min_width || min_batch_size < 1 ||
        max_batch_size > 8 || max_batch_size < min_batch_size) {
        throw std::invalid_argument("dynamic grouped conv prepare workspace: invalid interval");
    }
    // n [5120, W*B] and d [1280, W*B], then Linear's own scratch.
    return decomposed_capacity(kernel_projection, kCoefficientRows, kHidden, kHidden, max_width,
                               max_batch_size);
}

void rmsnorm_dynamic_grouped_conv_prepare(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Tensor& base_kernel,
                                          const Weight& kernel_projection_weight, Tensor& prepared,
                                          Tensor& finish_delta, WorkspaceArena& workspace,
                                          cudaStream_t stream) {
    const std::int32_t batch_size = residual.ne[2];
    const std::int32_t width      = residual.ne[1];
    if (width < 2 || width > 16) {
        throw std::invalid_argument("dynamic grouped conv prepare: W must be in [2,16]");
    }
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument(
            "dynamic grouped conv prepare: eps must be positive and finite");
    }
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("dynamic grouped conv prepare: B must be in [1,8]");
    }
    require_tensor(residual, DType::BF16, kHidden, width, batch_size, 1, kPrepareOp, "residual");
    require_tensor(norm_weight, DType::BF16, kHidden, 1, 1, 1, kPrepareOp, "norm_weight");
    require_tensor(base_kernel, DType::BF16, kHidden, kTaps, kSides, 1, kPrepareOp, "base_kernel");
    require_tensor(prepared, DType::BF16, kHidden, width, batch_size, 1, kPrepareOp, "prepared");
    require_tensor(finish_delta, DType::BF16, kGroups, kTaps, width, batch_size, kPrepareOp,
                   "finish_delta");
    if (!fused_prepare(kernel_projection_weight.qtype)) {
        require_projection(kernel_projection_weight, kCoefficientRows, kHidden, kPrepareOp);
        const std::int32_t columns = width * batch_size;
        auto scope                 = workspace.scope();
        Tensor n = workspace.alloc(DType::BF16, {kHidden, columns}, 256);
        Tensor d = workspace.alloc(DType::BF16, {kCoefficientRows, columns}, 256);
        rmsnorm(residual.view({kHidden, columns}), norm_weight, eps, false, n, stream);
        linear(n, kernel_projection_weight, d, LinearPolicy::A16Only, workspace, stream);
        detail::generic_dynamic_grouped_conv_prepare_launch(n, d, base_kernel, prepared,
                                                            finish_delta, width, columns, stream);
        return;
    }
    require_kernel_projection_weight(kernel_projection_weight);
    require_nonoverlap(residual, norm_weight, base_kernel, kernel_projection_weight, prepared,
                       finish_delta, workspace);

    detail::bf16_dynamic_grouped_conv_prepare_dispatch(residual, norm_weight, eps, base_kernel,
                                                       kernel_projection_weight, prepared,
                                                       finish_delta, workspace, stream);
}

std::size_t linear_dynamic_grouped_conv_add_workspace_capacity_bytes(std::int32_t input_rows,
                                                                     std::int32_t min_width,
                                                                     std::int32_t max_width,
                                                                     std::int32_t min_batch_size,
                                                                     std::int32_t max_batch_size) {
    return detail::q8_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
        input_rows, min_width, max_width, min_batch_size, max_batch_size);
}

std::size_t linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    QType projection, std::int32_t input_rows, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch_size, std::int32_t max_batch_size) {
    if (fused_add(projection)) {
        return linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
            input_rows, min_width, max_width, min_batch_size, max_batch_size);
    }
    if (min_width < 2 || max_width > 16 || max_width < min_width || min_batch_size < 1 ||
        max_batch_size > 8 || max_batch_size < min_batch_size || input_rows <= 0) {
        throw std::invalid_argument("linear dynamic grouped conv add workspace: invalid interval");
    }
    // z [5120, W*B], then Linear's own scratch (one BF16 plane fewer than prepare: none).
    WorkspaceLayoutBuilder layout;
    const std::int32_t columns = max_width * max_batch_size;
    (void)layout.alloc(DType::BF16, {kHidden, columns}, 256);
    (void)layout.alloc_bytes(linear_workspace_capacity_bytes(projection, kHidden, input_rows,
                                                             LinearPolicy::A16Only, 1, columns));
    return layout.peak_bytes();
}

void linear_dynamic_grouped_conv_add(const Tensor& x, const Weight& projection_weight,
                                     const Tensor& base_kernel, const Tensor& finish_delta,
                                     Tensor& residual, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    const std::int32_t input_rows = x.ne[0];
    const std::int32_t batch_size = x.ne[2];
    const std::int32_t width      = x.ne[1];
    if (width < 2 || width > 16)
        throw std::invalid_argument("linear dynamic grouped conv add: W must be in [2,16]");
    if (input_rows != 4096 && input_rows != 17408) {
        throw std::invalid_argument("linear dynamic grouped conv add: C must be 4096 or 17408");
    }
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("linear dynamic grouped conv add: B must be in [1,8]");
    }
    require_tensor(x, DType::BF16, input_rows, width, batch_size, 1, kAddOp, "x");
    require_tensor(base_kernel, DType::BF16, kHidden, kTaps, kSides, 1, kAddOp, "base_kernel");
    require_tensor(finish_delta, DType::BF16, kGroups, kTaps, width, batch_size, kAddOp,
                   "finish_delta");
    require_tensor(residual, DType::BF16, kHidden, width, batch_size, 1, kAddOp, "residual");
    if (!fused_add(projection_weight.qtype)) {
        require_projection(projection_weight, kHidden, input_rows, kAddOp);
        const std::int32_t columns = width * batch_size;
        auto scope                 = workspace.scope();
        Tensor z = workspace.alloc(DType::BF16, {kHidden, columns}, 256);
        linear(x.view({input_rows, columns}), projection_weight, z, LinearPolicy::A16Only,
               workspace, stream);
        detail::generic_linear_dynamic_grouped_conv_add_launch(z, base_kernel, finish_delta,
                                                               residual, width, columns, stream);
        return;
    }
    require_finish_projection_weight(projection_weight, input_rows);
    require_finish_nonoverlap(x, projection_weight, base_kernel, finish_delta, residual, workspace);

    detail::q8_linear_dynamic_grouped_conv_add_dispatch(x, projection_weight, base_kernel,
                                                        finish_delta, residual, workspace, stream);
}

} // namespace ninferno::ops
