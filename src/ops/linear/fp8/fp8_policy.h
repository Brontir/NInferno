#pragma once

#include "core/device_class.h"
#include "ninferno/ops/linear.h"

namespace ninferno::ops::detail {

// The A8 routes of FP8 weights issue the E4M3 mma.sync, which sm_89 introduced. On a device
// without it (Ampere) FP8 weights take their A16 routes whatever the policy admits.
[[nodiscard]] inline bool fp8_a8_allowed(LinearPolicy policy) {
    return allows_a8(policy) && device_class().features.mma_fp8;
}

} // namespace ninferno::ops::detail
