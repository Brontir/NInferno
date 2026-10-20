#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/common/shape_routes.h"
#include <array>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Bf16Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{14336, 5120, select_bf16_n14336_k5120},
    ShapeEntry{5120, 6144, select_bf16_n5120_k6144},
    ShapeEntry{256, 5120, select_bf16_n256_k5120},
};
} // namespace

bool bf16_linear_registered(std::int32_t n, std::int32_t k) {
    return find_shape(kShapes, n, k) != nullptr;
}

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("bf16 linear: T must be positive");
    if (const auto* entry = find_shape(kShapes, n, k)) {
        return entry->select(route_tokens({QType::BF16, ShapeCompute::A16, n, k}, t));
    }
    throw std::invalid_argument("bf16 linear: unsupported shape");
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("bf16 linear: unsupported policy");
    return select_bf16_a16_launch(n, k, t);
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    select_bf16_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace ninferno::ops::detail
