#pragma once

// The device as the speculation profiles see it: its class chain and memory from
// core/device_class.h. Header-only, so the spec library itself needs no CUDA; the engines and
// spec-tune, which own a device, include it.

#include "core/device_class.h"
#include "spec/profile.h"

#include <exception>
#include <string>

namespace ninferno::spec {

struct GpuIdentity {
    std::string name; // informational only
    std::string id;   // full class id ("gb202:170sm"): the key of a local tune
    std::string chip; // the chip ("gb202"): the class spec-tune publishes for
    GpuKey key;       // class chain and memory: what select_published matches
};

// An unreadable device is "unknown" and matches only "*" profiles.
inline GpuIdentity gpu_identity(int device) {
    try {
        const auto& d = ninferno::device_class(device);
        // Profiles are looked up along the similarity chain: a card nobody measured takes the
        // most similar measured card's choices (DeviceClass::similarity_chain).
        return {d.name, d.id(), d.chip,
                GpuKey{d.similarity_chain(), d.global_bytes, d.features.mma_mxf4nvf4,
                       d.class_chain().size() - 1, d.sms, d.compute_capability(),
                       d.facts.reference_clock_khz, d.dram_gbps()}};
    } catch (const std::exception&) {
        return {"unknown", "unknown", "unknown", GpuKey{{"*"}, 0, false}};
    }
}

} // namespace ninferno::spec
