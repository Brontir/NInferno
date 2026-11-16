#pragma once

// What a decision needs of the chip and of this build, declared as data (#615 F5): a requirement is
// a conjunction of discrete facts, resolved from the facts of one board, and a predicate asks
// whether the board meets it instead of reading the device itself. A requirement names only these:
// - mma_tensor, mma_bf16_native, native_fp4: the instruction sets the device and this build share
//   (DeviceFeatures in core/device_class.h; native_fp4 is mma_mxf4nvf4);
// - mma_slow: the board's (mma.sync without tensor cores, TU116 / TU117 and the GTX 16 boards);
// - fp32acc_full_rate: the generation's tensor-core kind (core/generation.h fp32acc_full_rate_of);
//   a generation with no row of its own has no kind and meets neither answer (#300);
// - build_arch_below: the architecture this tree compiles for (NINFERNO_BUILD_ARCH);
// - cc_any_of: the compute capability, exactly.
// No clock, bandwidth, SM count or L2 size: those are continuous facts, which a byte-class decision
// may not read (docs/maintainer/deterministic-decisions.md, B1).
//
// The facts come from either of two sources, which agree for every board of the PCI-ID table
// (tests/test_arch_requires.cpp):
// - a board registry row (core/board_registry.h, from core/data/gpu_facts.json) with a build's
//   features and architecture: arch_facts_of(row, build, build_arch);
// - a DeviceClass, the probed device or a ScopedDeviceClass substitute: arch_facts_of(d,
//   build_arch). Its features are already the device's and the build's in common, its mma_slow the
//   table row's (board_mma_slow).

#include "core/board_registry.h"
#include "core/device_class.h"
#include "core/generation.h"

#include <array>
#include <cstdint>
#include <optional>

namespace ninferno {

// The discrete facts a requirement may read.
struct ArchFacts {
    int major = 0;
    int minor = 0;
    // The instruction sets the device and this build share; mma_slow is the board's.
    DeviceFeatures features;
    // The generation's tensor-core kind (FP32-accumulate HMMA at full rate or not); nullopt for a
    // generation without a row of its own.
    std::optional<bool> fp32acc_full_rate;
    // The architecture this tree compiles for (NINFERNO_BUILD_ARCH: 120 for "120a"); 0 for a tree
    // that names none.
    int build_arch = 0;
};

// One fact's part in a requirement: not asked, must hold, must not hold.
enum class Need : std::uint8_t { Any, Yes, No };

struct ComputeCapability {
    int major = 0;
    int minor = 0;
};

// A conjunction of the facts above; a field left at its default asks nothing.
struct ArchRequires {
    Need mma_tensor      = Need::Any;
    Need mma_bf16_native = Need::Any;
    Need native_fp4      = Need::Any;
    Need mma_slow        = Need::Any;
    // Yes / No: the generation's own row says so; a generation without one meets neither.
    Need fp32acc_full_rate = Need::Any;
    // > 0: the tree names an architecture and it is below this one.
    int build_arch_below = 0;
    // Compute capabilities, exact; the list ends at the first {0, 0}, and an empty list is any.
    std::array<ComputeCapability, 4> cc_any_of{};
};

[[nodiscard]] constexpr bool meets(Need need, bool fact) noexcept {
    return need == Need::Any || (need == Need::Yes) == fact;
}

// Whether a board with `facts` meets requirement `r`.
[[nodiscard]] constexpr bool satisfied(const ArchRequires& r, const ArchFacts& facts) noexcept {
    const DeviceFeatures& f = facts.features;
    if (!meets(r.mma_tensor, f.mma_tensor) || !meets(r.mma_bf16_native, f.mma_bf16_native) ||
        !meets(r.native_fp4, f.mma_mxf4nvf4) || !meets(r.mma_slow, f.mma_slow)) {
        return false;
    }
    if (r.fp32acc_full_rate != Need::Any &&
        (!facts.fp32acc_full_rate || !meets(r.fp32acc_full_rate, *facts.fp32acc_full_rate))) {
        return false;
    }
    if (r.build_arch_below > 0 &&
        !(facts.build_arch > 0 && facts.build_arch < r.build_arch_below)) {
        return false;
    }
    if (r.cc_any_of[0].major != 0) {
        bool listed = false;
        for (const ComputeCapability& cc : r.cc_any_of) {
            if (cc.major == 0) break;
            listed = listed || (cc.major == facts.major && cc.minor == facts.minor);
        }
        if (!listed) return false;
    }
    return true;
}

// The facts of a board registry row under a build with features `build` and architecture
// `build_arch`: the architecture's features (device_features_for) in common with the build's, and
// the row's mma_slow, as the device query sets them.
[[nodiscard]] inline ArchFacts arch_facts_of(const board_registry::BoardRow& row,
                                             const DeviceFeatures& build, int build_arch) noexcept {
    ArchFacts out{.major    = row.major,
                  .minor    = row.minor,
                  .features = common_features(device_features_for(row.major, row.minor), build),
                  .fp32acc_full_rate = fp32acc_full_rate_of(row.major, row.minor),
                  .build_arch        = build_arch};
    out.features.mma_slow = row.mma_slow;
    return out;
}

// The facts of a device class (the probed device or a substitute) in a tree built for
// `build_arch`.
[[nodiscard]] inline ArchFacts arch_facts_of(const DeviceClass& d, int build_arch) noexcept {
    return ArchFacts{.major             = d.major,
                     .minor             = d.minor,
                     .features          = d.features,
                     .fp32acc_full_rate = fp32acc_full_rate_of(d.major, d.minor),
                     .build_arch        = build_arch};
}

} // namespace ninferno
