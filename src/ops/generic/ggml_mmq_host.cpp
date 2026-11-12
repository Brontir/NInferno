#include "ops/generic/ggml_mmq/ggml_mmq.h"

#include "core/device_class.h"

#include <cstddef>
#include <exception>

namespace ninferno::ops::ggml_mmq {

bool device_supported() {
    // The device class the selection is for: a selector test's substitute (ScopedDeviceClass)
    // decides as that card would, not as the GPU the test happens to run on.
    try {
        // TU116 / TU117 run its int8 mma.sync without tensor cores (mma_slow): the vector kernel
        // takes their verify widths, as on Volta.
        const auto& d = device_class();
        return d.compute_capability() >= 75 && !d.features.mma_slow;
    } catch (const std::exception&) { return false; }
}

bool vec_supported(Type type) {
    switch (type) {
    case Type::IQ4_XS:
    case Type::Q4_K:
    case Type::Q5_K:
    case Type::Q6_K:
    case Type::Q2_K:
    case Type::IQ3_XXS:
    case Type::IQ3_S:
    case Type::Q3_K:
    case Type::IQ2_XXS:
    case Type::IQ2_XS:
    case Type::IQ2_S:
    case Type::IQ1_S:
    case Type::Q4_0:
    case Type::Q4_1:
    case Type::Q5_0:
    case Type::Q5_1:
    case Type::IQ4_NL:
    case Type::IQ1_M:
        return true;
    default:
        return false;
    }
}

std::size_t activation_bytes(int k, int columns) {
    return std::size_t(columns) * (k / kQ8MmqValues) * kQ8MmqBlockBytes + kActivationSlack;
}

std::size_t vec_activation_bytes(int k, int columns) {
    return vec_scales_offset(k, columns) + std::size_t(columns) * (k / 32) * kHalf2Bytes;
}

std::size_t fixup_bound(int columns) {
    // A stream-k launch holds at most one 128-row tile of the call's column tile per SM. The SMs
    // are device_class()'s: a selector test's substitute counts its own card's. The launch
    // (select_launch, ggml_mmq_internal.cuh) counts the running device's SMs (device_facts()); the
    // two are the same device wherever no substitute is set, and a substitute only prices a
    // workspace for a board it never launches on.
    int j = 16;
    while (j < columns && j < 128) j *= 2;
    return std::size_t(device_class().sms) * j * 128 * sizeof(float);
}

} // namespace ninferno::ops::ggml_mmq
