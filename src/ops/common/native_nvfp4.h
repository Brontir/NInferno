#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::ops::detail {

// Whether this build carries the native NVFP4 kernels: the sm_120a families that issue the
// block-scaled FP4 MMA and TMA (linear/nvfp4, the fused NVFP4 projections, the SparseMoe and
// Gemma4Moe NVFP4 expert banks, the W4A4 and tiled generic routes). Without them NVFP4 weights run
// on the generic path, which decodes E2M1 on every tensor-core GPU: a native Op admits NVFP4
// only where this is true, and every caller falls back as it does for a format the Op lacks.
#ifdef NINFERNO_WITHOUT_NVFP4
inline constexpr bool kNativeNvfp4 = false;
#else
inline constexpr bool kNativeNvfp4 = true;
#endif

// The refusal of a native NVFP4 Op on a build without its kernels. Callers gate on kNativeNvfp4
// first, so reaching it means one did not: `op` names the Op, `operand` the weight if any.
[[noreturn]] inline void no_native_nvfp4(std::string_view op, std::string_view operand = {}) {
    std::string message(op);
    message += ": ";
    if (!operand.empty()) message.append(operand).append(" ");
    message += "NVFP4 reaches this Op only on sm_120a builds; this build runs NVFP4 on the generic "
               "path, and callers check kNativeNvfp4 first";
    throw std::invalid_argument(message);
}

} // namespace ninferno::ops::detail
