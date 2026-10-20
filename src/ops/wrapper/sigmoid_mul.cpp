// ninferno::ops - sigmoid_mul wrapper: implements the public api, validates parameters,
// and dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninferno/ops/sigmoid_mul.h"

#include "ops/common/tensor_checks.h"
#include "ops/launcher/sigmoid_gate_mul.h" // detail::sigmoid_gate_mul_launch

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninferno::ops {
void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream) {
    if (gate.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("sigmoid_mul: gate/x must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != x.ne[d]) {
            throw std::invalid_argument("sigmoid_mul: gate/x shapes must match");
        }
    }
    if (numel_allow_zero(x, "sigmoid_mul", "x") == 0) { return; }
    if (!x.is_contiguous()) { throw std::invalid_argument("sigmoid_mul: x must be contiguous"); }
    if (gate.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("sigmoid_mul: gate/x data must be non-null");
    }
    if (!gate.is_contiguous()) {
        // Rows of a wider plane (a packed [q; k; gate; v] projection): columns apart, rows dense.
        if (gate.nb[0] != 2 || gate.ne[2] != 1 || gate.ne[3] != 1 ||
            gate.nb[1] < std::int64_t(gate.ne[0]) * 2) {
            throw std::invalid_argument("sigmoid_mul: gate must be contiguous or a row slice");
        }
        detail::sigmoid_gate_mul_rows_launch(gate, x, stream);
        return;
    }

    detail::sigmoid_gate_mul_launch(gate, x, stream); // single variant -> direct dispatch
}

} // namespace ninferno::ops
