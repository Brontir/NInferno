#include "core/layout.h"
#include "core/tensor.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail::generic {

bool ggml_mmq_format(QType qtype) {
    Weight probe;
    probe.qtype     = qtype;
    const auto type = mmq_type(probe);
    return type && ggml_mmq::matrix_supported(*type);
}

bool ggml_vec_format(QType qtype) {
    Weight probe;
    probe.qtype     = qtype;
    const auto type = mmq_type(probe);
    return type && ggml_mmq::vec_supported(*type);
}

} // namespace ninferno::ops::detail::generic
