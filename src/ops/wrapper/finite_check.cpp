// ninferno::ops - finite_check wrapper: public api validation and launcher dispatch.
#include "ninferno/ops/finite_check.h"

#include "ops/launcher/finite_check.h" // detail::finite_check_launch

#include <cstdint>
#include <stdexcept>

namespace ninferno::ops {

void finite_check(const Tensor& values, const std::int32_t* valid_columns, std::uint32_t* status,
                  cudaStream_t stream) {
    if (values.dtype != DType::BF16 && values.dtype != DType::FP16 && values.dtype != DType::FP32) {
        throw std::invalid_argument("finite_check: values must be BF16, FP16 or FP32");
    }
    if (values.ne[3] != 1) { throw std::invalid_argument("finite_check: values must be [H,C,R]"); }
    if (values.ne[0] <= 0 || values.ne[1] <= 0 || values.ne[2] <= 0) { return; }
    if (!values.is_contiguous()) {
        throw std::invalid_argument("finite_check: values must be contiguous");
    }
    if (values.data == nullptr || status == nullptr) {
        throw std::invalid_argument("finite_check: values/status must be non-null");
    }
    detail::finite_check_launch(values, values.ne[0], values.ne[1], values.ne[2], valid_columns,
                                status, stream);
}

} // namespace ninferno::ops
