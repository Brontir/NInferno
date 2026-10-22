#include "ops/linear/activation_width.h"

#include "ops/generic/generic_gemm.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "core/generation.h"

namespace ninferno::ops {

namespace {

bool fp8_row_format(QType qtype) {
    return qtype == QType::FP8_E4M3FN_ROW_BF16 || qtype == QType::FP8_E4M3FN_ROW_F32 ||
           qtype == QType::FP8_E4M3FN_B128_F32;
}

// Whether the generic path has an int8-activation route for the format at this K: W4A8 (prefill)
// or R8 (decode and verify), which also cover every format of ggml's int8 matrix and vector
// kernels.
bool int8_route(QType qtype, std::int64_t k) {
    const auto k32 = static_cast<std::int32_t>(k);
    return (detail::generic::w4a8_supports(qtype) && detail::generic::w4a8_k_admits(qtype, k32)) ||
           (detail::generic::r8_supports(qtype) && k > 0 &&
            k32 % detail::generic::r8_k_values(qtype) == 0);
}

} // namespace

ActivationWidth narrowest_activation_width(QType qtype, LinearPolicy policy, std::int64_t n,
                                           std::int64_t k, bool has_activation_divisor,
                                           const DeviceFeatures& features) {
    if (!allows_a8(policy)) { return {16, ""}; }
    std::string_view not_four;
    if (allows_a4(policy)) {
        if (qtype != QType::NVFP4) {
            not_four = "the format has no four-bit activation route (NVFP4 only)";
        } else if (!features.mma_mxf4nvf4) {
            not_four = "four-bit activations need the NVFP4 MMA of an sm_120a build";
        } else if (!has_activation_divisor) {
            // Where this build has the W4A4 kernels an NVFP4 weight allowed four bits without a
            // divisor is not W4A8's either (generic_gemm.cpp, allows_w4a8): it stays BF16.
            return {16, "no calibrated activation divisor for the four-bit route"};
        } else {
            return {4, ""};
        }
    }
    if (fp8_row_format(qtype)) {
        if (!features.mma_fp8) { return {16, "FP8 activations need the E4M3 MMA (sm_89 on)"}; }
        if (qtype != QType::FP8_E4M3FN_ROW_BF16 ||
            (n > 0 && k > 0 &&
             !detail::fp8_linear_registered(static_cast<std::int32_t>(n),
                                            static_cast<std::int32_t>(k)))) {
            return {16, "no FP8 E4M3 activation kernel for this format and shape"};
        }
        return {8, not_four};
    }
    if (!int8_route(qtype, k)) { return {16, "the format has no eight-bit activation route"}; }
    return {8, not_four};
}

ActivationDevice activation_device(int major, int minor, bool mma_slow) {
    ActivationDevice out;
    out.features          = common_features(device_features_for(major, minor), build_features());
    out.features.mma_slow = mma_slow;
    out.fp32acc_full_rate = generation_row(major, minor).fp32acc_full_rate;
    return out;
}

ActivationWidth linear_activation_width(QType qtype, LinearPolicy policy, std::int64_t n,
                                        std::int64_t k, bool has_activation_divisor,
                                        bool logits_or_embedding, bool invariant,
                                        const ActivationDevice& device) {
    const LinearPolicy prefill =
        logits_or_embedding
            ? policy
            : detail::generic::prefill_a8_policy_on(
                  qtype, policy,
                  detail::generic::qx_prefill_a8_pays_on(device.features, device.fp32acc_full_rate),
                  invariant);
    auto width =
        narrowest_activation_width(qtype, prefill, n, k, has_activation_divisor, device.features);
    if (prefill != policy && width.bits == 8) {
        width.reason = "recorded 16-bit, prefilled on int8 activations where they pay";
    }
    return width;
}

} // namespace ninferno::ops
