#include "ninferno/ops/rope.h"

#include "ops/common/tensor_checks.h"
#include "ops/kernel/rope_limits.h"
#include "ops/launcher/rope.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

constexpr std::int32_t kTextHeadDim = 256;

int position_axes(const Tensor& positions, std::int32_t tokens) {
    if (positions.ne[0] != tokens || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        (positions.ne[1] != 1 && positions.ne[1] != 2 && positions.ne[1] != 3)) {
        throw std::invalid_argument("rope: positions must have shape [T], [T,2], or [T,3]");
    }
    return positions.ne[1];
}

void require_tensor_layout(const Tensor& tensor, const char* label, std::int32_t head_dim,
                           std::int32_t heads, std::int32_t tokens) {
    if (heads <= 0) {
        throw std::invalid_argument(std::string("rope: ") + label + " must have positive heads");
    }
    if (tensor.ne[0] != head_dim || tensor.ne[1] != heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * static_cast<std::int64_t>(head_dim) * heads ||
        (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " strides");
    }
}

void require_common(const Tensor& positions, int rotary_dim, float theta,
                    DeviceExecutionView execution) {
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument("rope: positive multiprocessor count required");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("rope: positions must be I32");
    }
    if (!(theta > 0.0f) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: theta must be positive and finite");
    }
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0) {
        throw std::invalid_argument("rope: rotary_dim must be positive and even");
    }
}

void require_positions_storage(const Tensor& positions) {
    if (!positions.is_contiguous()) {
        throw std::invalid_argument("rope: positions must be contiguous");
    }
    if (positions.data == nullptr) {
        throw std::invalid_argument("rope: positions data must be non-null");
    }
}

void require_model_mode(int axes, int rotary_dim, std::int32_t head_dim) {
    if (axes == 2) {
        // Each axis rotates a quarter of the head's values (Qwen's encoders: heads of 64 or 72).
        if (rotary_dim != head_dim || head_dim % 4 != 0 || head_dim / 2 > kRopeMaxHalf) {
            throw std::invalid_argument(
                "rope: 2-D Vision mode requires head_dim=rotary_dim, a multiple of 4, at most " +
                std::to_string(2 * kRopeMaxHalf));
        }
        return;
    }
    if (axes == 1 && head_dim == 128 && rotary_dim == 128) { return; }
    if (head_dim != kTextHeadDim || rotary_dim > kTextHeadDim) {
        throw std::invalid_argument("rope: Text mode requires D256 or one-dimensional D128/R128");
    }
    if (axes == 3 && rotary_dim != 64) {
        throw std::invalid_argument("rope: 3-D Text MRoPE requires rotary_dim=64");
    }
}

} // namespace

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          DeviceExecutionView execution) {
    require_common(positions, rotary_dim, theta, execution);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    (void)numel_allow_zero(positions, "rope", "positions");
    const std::int64_t q_numel = numel_allow_zero(q, "rope", "q");
    (void)numel_allow_zero(k, "rope", "k");
    const std::int32_t tokens   = q.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t k_heads  = k.ne[1];
    require_model_mode(axes, rotary_dim, head_dim);
    require_tensor_layout(q, "q", head_dim, q_heads, tokens);
    require_tensor_layout(k, "k", head_dim, k_heads, tokens);
    if (q_numel == 0) { return; }
    require_positions_storage(positions);
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope: q/k data must be non-null");
    }
    detail::rope_launch(positions, rotary_dim, theta, q, k, execution);
}

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
          DeviceExecutionView execution) {
    require_common(positions, rotary_dim, theta, execution);
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope: tensor must be BF16"); }
    (void)numel_allow_zero(positions, "rope", "positions");
    const std::int64_t x_numel  = numel_allow_zero(x, "rope", "tensor");
    const std::int32_t tokens   = x.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = x.ne[0];
    const std::int32_t heads    = x.ne[1];
    require_model_mode(axes, rotary_dim, head_dim);
    require_tensor_layout(x, "tensor", head_dim, heads, tokens);
    if (x_numel == 0) { return; }
    require_positions_storage(positions);
    if (x.data == nullptr) { throw std::invalid_argument("rope: tensor data must be non-null"); }
    detail::rope_single_launch(positions, rotary_dim, theta, x, execution);
}

namespace {

void require_scaled(const Tensor& positions, int rotary_dim, const Tensor& inverse_frequencies,
                    float attention_scale, std::int32_t head_dim, std::int32_t tokens) {
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("rope_scaled: positions must be I32");
    }
    const int axes = position_axes(positions, tokens);
    if (axes == 2 || head_dim != kTextHeadDim || rotary_dim <= 0 || (rotary_dim & 1) != 0 ||
        rotary_dim > kTextHeadDim || (axes == 3 && rotary_dim != 64)) {
        throw std::invalid_argument("rope_scaled: Text modes only (D256; MRoPE needs R64)");
    }
    if (inverse_frequencies.dtype != DType::FP32 || inverse_frequencies.ne[0] != rotary_dim / 2 ||
        inverse_frequencies.ne[1] != 1 || inverse_frequencies.ne[2] != 1 ||
        inverse_frequencies.ne[3] != 1 || !inverse_frequencies.is_contiguous() ||
        inverse_frequencies.data == nullptr) {
        throw std::invalid_argument(
            "rope_scaled: inverse_frequencies must be contiguous device FP32 [rotary_dim/2]");
    }
    if (!(attention_scale > 0.0F) || !std::isfinite(attention_scale)) {
        throw std::invalid_argument("rope_scaled: attention_scale must be positive and finite");
    }
}

} // namespace

void rope_scaled(const Tensor& positions, int rotary_dim, const Tensor& inverse_frequencies,
                 float attention_scale, Tensor& q, Tensor& k, DeviceExecutionView execution) {
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope_scaled: q/k must be BF16");
    }
    (void)numel_allow_zero(positions, "rope", "positions");
    const std::int64_t q_numel = numel_allow_zero(q, "rope", "q");
    (void)numel_allow_zero(k, "rope", "k");
    const std::int32_t tokens = q.ne[2];
    require_scaled(positions, rotary_dim, inverse_frequencies, attention_scale, q.ne[0], tokens);
    require_tensor_layout(q, "q", q.ne[0], q.ne[1], tokens);
    require_tensor_layout(k, "k", q.ne[0], k.ne[1], tokens);
    if (q_numel == 0) { return; }
    require_positions_storage(positions);
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope_scaled: q/k data must be non-null");
    }
    detail::rope_scaled_launch(positions, rotary_dim, inverse_frequencies, attention_scale, &q, &k,
                               execution.stream);
}

void rope_scaled(const Tensor& positions, int rotary_dim, const Tensor& inverse_frequencies,
                 float attention_scale, Tensor& x, DeviceExecutionView execution) {
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope_scaled: tensor must be BF16"); }
    (void)numel_allow_zero(positions, "rope", "positions");
    const std::int64_t x_numel = numel_allow_zero(x, "rope", "tensor");
    const std::int32_t tokens  = x.ne[2];
    require_scaled(positions, rotary_dim, inverse_frequencies, attention_scale, x.ne[0], tokens);
    require_tensor_layout(x, "tensor", x.ne[0], x.ne[1], tokens);
    if (x_numel == 0) { return; }
    require_positions_storage(positions);
    if (x.data == nullptr) { throw std::invalid_argument("rope_scaled: tensor data must be non-null"); }
    detail::rope_scaled_launch(positions, rotary_dim, inverse_frequencies, attention_scale, &x,
                               nullptr, execution.stream);
}

} // namespace ninferno::ops
