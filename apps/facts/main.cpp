// ninferno-facts-probe (#612 step 2): what the engine sees of each visible GPU, as JSON, for
// tools/audit/facts_vs_driver.py. Reads the board through the engine's own query (device_class:
// the driver's values as DeviceClass holds them, the PCI-ID table's facts) beside the architecture
// limits synthetic_device_class writes for the same compute capability, and the raw CUDA
// attributes the query does not keep (reserved shared memory, the default shared limit). Compares
// nothing itself.
//
//   ninferno-facts-probe [--device N]
//
// Output: {"devices": [{"index", "queried": {...}, "synthetic": {...}, "raw": {...}}, ...]}.
#include "core/device_class.h"

#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace {

using Json = nlohmann::ordered_json;

int attribute(cudaDeviceAttr what, int device) {
    int value = 0;
    if (cudaDeviceGetAttribute(&value, what, device) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1; // the driver does not answer
    }
    return value;
}

Json features_of(const ninferno::DeviceFeatures& f) {
    return Json{{"mma_bf16", f.mma_bf16},
                {"mma_bf16_native", f.mma_bf16_native},
                {"mma_tensor", f.mma_tensor},
                {"mma_fp8", f.mma_fp8},
                {"mma_f8f6f4", f.mma_f8f6f4},
                {"mma_mxf4nvf4", f.mma_mxf4nvf4},
                {"tma", f.tma},
                {"pdl", f.pdl},
                {"clusters", f.clusters},
                {"mma_slow", f.mma_slow}};
}

Json class_of(const ninferno::DeviceClass& d) {
    return Json{{"name", d.name},
                {"pci_device_id", d.pci_device_id},
                {"chip", d.chip},
                {"major", d.major},
                {"minor", d.minor},
                {"sms", d.sms},
                {"l2_bytes", d.l2_bytes},
                {"global_bytes", d.global_bytes},
                {"smem_per_block_optin", d.smem_per_block_optin},
                {"smem_per_sm", d.smem_per_sm},
                {"max_threads_per_sm", d.max_threads_per_sm},
                {"max_blocks_per_sm", d.max_blocks_per_sm},
                {"registers_per_sm", d.registers_per_sm},
                {"persisting_l2_max_bytes", d.persisting_l2_max_bytes},
                {"memory_bus_bits", d.memory_bus_bits},
                {"memory_clock_khz", d.memory_clock_khz},
                {"sm_clock_khz", d.sm_clock_khz},
                {"features", features_of(d.features)}};
}

} // namespace

// A whole, non-negative ordinal ("2"), or -1 for anything else ("-1", "abc", "1x").
static int device_ordinal(const char* text) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    return end != text && *end == '\0' && value >= 0 && value < 4096 ? static_cast<int>(value) : -1;
}

int main(int argc, char** argv) {
    int only = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc &&
            (only = device_ordinal(argv[i + 1])) >= 0) {
            ++i;
        } else {
            std::fprintf(stderr, "usage: ninferno-facts-probe [--device N]\n");
            return 2;
        }
    }
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            std::fprintf(stderr, "ninferno-facts-probe: no usable CUDA device\n");
            return 3;
        }
        if (only >= count) {
            std::fprintf(stderr, "ninferno-facts-probe: no device %d (%d visible)\n", only, count);
            return 2;
        }
        Json devices = Json::array();
        for (int device = 0; device < count; ++device) {
            if (only >= 0 && device != only) { continue; }
            const ninferno::DeviceClass& q        = ninferno::device_class(device);
            const ninferno::DeviceClass synthetic = ninferno::synthetic_device_class(
                q.pci_device_id, q.major, q.minor, q.sms, q.l2_bytes, q.global_bytes);
            const ninferno::MachineFacts facts = ninferno::machine_facts_of(q);
            Json raw{
                {"reserved_shared_per_block",
                 attribute(cudaDevAttrReservedSharedMemoryPerBlock, device)},
                {"shared_per_block_default", attribute(cudaDevAttrMaxSharedMemoryPerBlock, device)},
                {"registers_per_block", attribute(cudaDevAttrMaxRegistersPerBlock, device)},
                {"max_threads_per_block", attribute(cudaDevAttrMaxThreadsPerBlock, device)},
                {"warp_size", attribute(cudaDevAttrWarpSize, device)}};
            Json table{{"source", static_cast<int>(facts.source)},
                       {"sku", facts.sku},
                       {"reference_clock_khz", facts.reference_clock_khz},
                       {"memory_bus_bits", facts.memory_bus_bits},
                       {"memory_clock_khz", facts.memory_clock_khz},
                       {"l2_mib", facts.l2_mib}};
            devices.push_back(Json{{"index", device},
                                   {"queried", class_of(q)},
                                   {"synthetic", class_of(synthetic)},
                                   {"raw", std::move(raw)},
                                   {"engine_table_facts", std::move(table)}});
        }
        std::puts(Json{{"devices", std::move(devices)}}.dump(1).c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninferno-facts-probe: %s\n", error.what());
        return 1;
    }
}
