// ninferno::ops - l2norm wrapper: public api validation and launcher dispatch.
#include "ninferno/ops/l2norm.h"

#include "ops/common/tensor_checks.h"
#include "ops/launcher/l2norm.h" // detail::l2norm_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

void require_same_shape(const Tensor& a, const Tensor& b) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) { throw std::invalid_argument("l2norm: x/out shapes must match"); }
    }
}

} // namespace

void l2norm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("l2norm: x/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("l2norm: eps must be positive and finite");
    }

    const std::int64_t n = numel_allow_zero(x, "l2norm", "x");
    (void)numel_allow_zero(out, "l2norm", "out");
    require_same_shape(x, out);
    if (n == 0) { return; }

    const std::int64_t rows = n / x.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("l2norm: row count exceeds CUDA grid limit");
    }

    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("l2norm: x/out must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("l2norm: x/out data must be non-null");
    }

    detail::l2norm_launch(x, eps, out, stream);
}

} // namespace ninferno::ops
