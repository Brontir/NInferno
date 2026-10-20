#include "ninferno/ops/logit_softcap.h"

#include "ops/launcher/logit_softcap.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninferno::ops {

void logit_softcap(Tensor& logits, float cap, cudaStream_t stream, float scale) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("logit_softcap: logits must be BF16");
    }
    if (!(cap > 0.0F) || !std::isfinite(cap)) {
        throw std::invalid_argument("logit_softcap: cap must be positive and finite");
    }
    if (!(scale > 0.0F) || !std::isfinite(scale)) {
        throw std::invalid_argument("logit_softcap: scale must be positive and finite");
    }
    std::int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (logits.ne[i] < 0) {
            throw std::invalid_argument("logit_softcap: negative dimension");
        }
        if (logits.ne[i] == 0) { return; }
        if (n > std::numeric_limits<std::int64_t>::max() / logits.ne[i]) {
            throw std::overflow_error("logit_softcap: tensor size overflows int64");
        }
        n *= logits.ne[i];
    }
    if (!logits.is_contiguous()) {
        throw std::invalid_argument("logit_softcap: logits must be contiguous");
    }
    if (logits.data == nullptr) {
        throw std::invalid_argument("logit_softcap: logits data must be non-null");
    }
    detail::logit_softcap_launch(logits, cap, scale, stream);
}

} // namespace ninferno::ops
