// ninferno::ops - residual_add wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninferno/ops/residual_add.h"

#include "ops/common/tensor_checks.h"
#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninferno::ops {
void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream) {
    if (y.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("residual_add: y/x must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (y.ne[d] != x.ne[d]) {
            throw std::invalid_argument("residual_add: y/x shapes must match");
        }
    }
    if (numel_allow_zero(x, "residual_add", "x") == 0) { return; }
    if (!y.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("residual_add: y/x must be contiguous");
    }
    if (y.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("residual_add: y/x data must be non-null");
    }

    detail::residual_add_launch(y, x, stream); // single variant -> direct dispatch
}

} // namespace ninferno::ops
