// ninferno::ops - rmsnorm wrapper: public api validation and launcher dispatch.
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/gated_rmsnorm.h"

#include "ops/common/tensor_checks.h"
#include "ops/launcher/rmsnorm.h" // detail::rmsnorm_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

void require_same_shape(const Tensor& a, const Tensor& b, const char* b_label) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            throw std::invalid_argument(std::string("rmsnorm: x/") + b_label +
                                        " shapes must match");
        }
    }
}

} // namespace

namespace {

void rmsnorm_impl(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                  const Tensor* z, Tensor& out, std::int32_t multiprocessor_count,
                  cudaStream_t stream, std::int32_t groups = 0) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        (z != nullptr && z->dtype != DType::BF16)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm: eps must be positive and finite");
    }

    const std::int64_t n = numel_allow_zero(x, "rmsnorm", "x");
    (void)numel_allow_zero(out, "rmsnorm", "out");
    (void)numel_allow_zero(weight, "rmsnorm", "weight");
    require_same_shape(x, out, "out");
    if (z != nullptr) {
        (void)numel_allow_zero(*z, "rmsnorm", "z");
        require_same_shape(x, *z, "z");
    }
    if (weight.ne[0] != x.ne[0] || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm: weight must be 1-D with ne[0] == x.ne[0]");
    }
    if (n == 0) { return; }
    const std::int64_t rows = n / x.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds CUDA grid limit");
    }

    if (!x.is_contiguous() || !weight.is_contiguous() || !out.is_contiguous() ||
        (z != nullptr && !z->is_contiguous())) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || out.data == nullptr ||
        (z != nullptr && z->data == nullptr)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out data must be non-null");
    }

    if (groups > 0 && (x.ne[1] <= 0 || x.ne[1] % groups != 0)) {
        throw std::invalid_argument("gated_rmsnorm_regrouped: groups must divide the heads");
    }
    detail::rmsnorm_launch(x, weight, eps, unit_offset, z, out, multiprocessor_count, x.ne[1],
                           groups, stream);
}

} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps, unit_offset, nullptr, out, 0, stream);
}

void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   DeviceExecutionView execution) {
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument("gated_rmsnorm: positive multiprocessor count required");
    }
    rmsnorm_impl(x, weight, eps, false, &z, out, execution.multiprocessor_count, execution.stream);
}

bool gated_rmsnorm_regroups(const Tensor& x, const Tensor& weight, const Tensor& z,
                            const Tensor& out) {
    const std::int64_t d = x.ne[0];
    const auto aligned   = [](const Tensor& t) {
        return (reinterpret_cast<std::uintptr_t>(t.data) & 3U) == 0;
    };
    const auto apart = [&](const Tensor& t) {
        const auto* a = static_cast<const char*>(out.data);
        const auto* b = static_cast<const char*>(t.data);
        return a + out.bytes() <= b || b + t.bytes() <= a;
    };
    const auto same_shape = [&](const Tensor& t) {
        for (int i = 0; i < 4; ++i) {
            if (t.ne[i] != x.ne[i]) return false;
        }
        return true;
    };
    return d >= 64 && d <= 256 && d % 64 == 0 && x.dtype == DType::BF16 && z.dtype == DType::BF16 &&
           out.dtype == DType::BF16 && x.is_contiguous() && z.is_contiguous() &&
           out.is_contiguous() && same_shape(z) && same_shape(out) && aligned(x) &&
           aligned(weight) && aligned(z) && aligned(out) && apart(x) && apart(z);
}

void gated_rmsnorm_regrouped(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                             std::int32_t groups, Tensor& out, DeviceExecutionView execution) {
    if (execution.multiprocessor_count <= 0 || groups <= 0 ||
        !gated_rmsnorm_regroups(x, weight, z, out)) {
        throw std::invalid_argument("gated_rmsnorm_regrouped: positive SM count and groups, D "
                                    "64..256 by 64, aligned tensors, out apart from x and z");
    }
    rmsnorm_impl(x, weight, eps, false, &z, out, execution.multiprocessor_count, execution.stream,
                 groups);
}

} // namespace ninferno::ops
