#pragma once

// The narrowest activation width a linear may compute with: a capability (the first layer of
// docs/maintainer/deterministic-decisions.md's rules), read by --activations' startup resolution
// (runtime/engine/activation_precision.h). It reads the weight's format, K and N, its policy,
// whether it carries a calibrated activation divisor, and the device's discrete features, all
// passed in, so it runs on a CPU.
//
// It states the generic linear path's admission (W4A4, W4A8, R8 and the ggml int8 kernels, the
// registered FP8 E4M3 shapes): a call of that matrix may take this width and none narrower. A fused
// plan or an expert bank may run a given call wider (the MoE Op has no W4A4; decode keeps NVFP4 at
// BF16 on sm_120a), never narrower.

#include "core/device_class.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"

#include <cstdint>
#include <string_view>

namespace ninferno::ops {

struct ActivationWidth {
    int bits = 16; // 4, 8 or 16
    // Why the width is not the one the policy allows ("" when it is): read by a startup warning.
    std::string_view reason;
};

// The facts of the device the routes read: its discrete features and whether its FP32-accumulate
// HMMA runs at full rate (GenerationRow::fp32acc_full_rate, which decides where int8 prefill pays:
// generic_gemm.h prefill_a8_policy_on).
struct ActivationDevice {
    DeviceFeatures features;
    bool fp32acc_full_rate = false;
};

// The facts of a device of compute capability major.minor running this build (its features in
// common with build_features()); `mma_slow` is the board's (DeviceClass::features), false for a
// board not at hand.
[[nodiscard]] ActivationDevice activation_device(int major, int minor, bool mma_slow = false);

[[nodiscard]] ActivationWidth narrowest_activation_width(QType qtype, LinearPolicy policy,
                                                         std::int64_t n, std::int64_t k,
                                                         bool has_activation_divisor,
                                                         const DeviceFeatures& features);

// The width of a linear's calls on the generic path, prefill included: a Q4/Q5/Q6 G64 weight
// whose policy is A16Only prefills on int8 activations where they pay (prefill_a8_policy_on;
// `invariant`: under --activations 16 they never do). A head or embedding is never raised: the
// prefill computes its logits for the last token only, below the int8 prefill width.
[[nodiscard]] ActivationWidth linear_activation_width(QType qtype, LinearPolicy policy,
                                                      std::int64_t n, std::int64_t k,
                                                      bool has_activation_divisor,
                                                      bool logits_or_embedding, bool invariant,
                                                      const ActivationDevice& device);

} // namespace ninferno::ops
