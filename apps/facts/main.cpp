// ninferno-facts-probe (#612 step 2): what the engine sees of each visible GPU, as JSON, for
// tools/audit/facts_vs_driver.py. Reads the board through the engine's own query (device_class:
// the driver's values as DeviceClass holds them, the PCI-ID table's facts) beside the architecture
// limits synthetic_device_class writes for the same compute capability, and the raw CUDA
// attributes the query does not keep (reserved shared memory, the default shared limit). Compares
// nothing itself.
//
//   ninferno-facts-probe [--device N]
//
// Output: {"devices": [{"index", "pci_bus_id", "mig_mode", "queried": {...}, "synthetic": {...},
// "raw": {...}, "engine_occupancy_limits": {...}, "engine_table_facts": {...}, "engine_row_facts":
// {...}}, ...]}. A device the probe cannot read is {"index", "error"}; the others are still
// printed.
#include "core/device_class.h"
#include "core/kernel_occupancy.h"
#include "core/machine_facts.h"
#include "core/per_device.h"
#include "core/sys.h"

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

// MachineFacts as the probe prints them: the query's stored facts and the engine's reading of the
// table row alone have the same keys.
Json facts_json(const ninferno::MachineFacts& facts) {
    return Json{{"source", static_cast<int>(facts.source)},
                {"sku", facts.sku},
                {"reference_clock_khz", facts.reference_clock_khz},
                {"memory_bus_bits", facts.memory_bus_bits},
                {"memory_clock_khz", facts.memory_clock_khz},
                {"l2_mib", facts.l2_mib},
                {"mma_slow", facts.mma_slow}};
}

// machine_facts of the board's row alone: the id and compute capability the driver reports, the
// row's own SM count, and no clock or bus (so no value of the driver's is read). An unknown id
// gives the engine's reading of it, which has no SKU.
ninferno::MachineFacts row_alone_facts(const ninferno::DeviceClass& q) {
    const ninferno::GpuFactsRow* row = ninferno::find_gpu_facts(q.pci_device_id);
    return ninferno::machine_facts({.pci_device_id = q.pci_device_id,
                                    .major         = q.major,
                                    .minor         = q.minor,
                                    .sms           = row != nullptr ? row->sms : 0});
}

// "domain:bus:device.function" of a CUDA ordinal, empty when the driver does not say.
std::string pci_bus_id(int device) {
    char text[32] = {};
    if (cudaDeviceGetPCIBusId(text, sizeof(text), device) != cudaSuccess) {
        (void)cudaGetLastError();
        return {};
    }
    return text;
}

// The MIG mode of the board at a PCI bus id, through NVML: 1 enabled (the CUDA device may be one
// instance of it), 0 disabled, -1 unknown (no NVML, a driver without MIG calls, a board without
// MIG, an unreadable id).
int mig_mode(const std::string& bus_id) {
    if (bus_id.empty()) { return -1; }
    void* nvml = ninferno::sys::load_library("libnvidia-ml.so.1", "nvml.dll");
    if (nvml == nullptr) { return -1; }
    using Init      = int (*)();
    using Handle    = int (*)(const char*, void**);
    using Mode      = int (*)(void*, unsigned*, unsigned*);
    using Shutdown  = int (*)();
    const auto init = reinterpret_cast<Init>(ninferno::sys::symbol(nvml, "nvmlInit_v2"));
    const auto by_bus_id =
        reinterpret_cast<Handle>(ninferno::sys::symbol(nvml, "nvmlDeviceGetHandleByPciBusId_v2"));
    const auto mode_of =
        reinterpret_cast<Mode>(ninferno::sys::symbol(nvml, "nvmlDeviceGetMigMode"));
    const auto shutdown = reinterpret_cast<Shutdown>(ninferno::sys::symbol(nvml, "nvmlShutdown"));
    int result          = -1;
    if (init != nullptr && by_bus_id != nullptr && mode_of != nullptr && shutdown != nullptr &&
        init() == 0) {
        void* board      = nullptr;
        unsigned current = 0;
        unsigned pending = 0;
        if (by_bus_id(bus_id.c_str(), &board) == 0 && mode_of(board, &current, &pending) == 0) {
            result = current == 1 ? 1 : 0; // NVML_DEVICE_MIG_ENABLE
        }
        shutdown();
    }
    ninferno::sys::free_library(nvml);
    return result;
}

// One device's entry. The queried class is the engine's own; everything else is read beside it.
Json device_entry(int device) {
    const ninferno::DeviceClass& q        = ninferno::device_class(device);
    const ninferno::DeviceClass synthetic = ninferno::synthetic_device_class(
        q.pci_device_id, q.major, q.minor, q.sms, q.l2_bytes, q.global_bytes);
    Json raw{
        {"reserved_shared_per_block", attribute(cudaDevAttrReservedSharedMemoryPerBlock, device)},
        {"shared_per_block_default", attribute(cudaDevAttrMaxSharedMemoryPerBlock, device)},
        {"registers_per_block", attribute(cudaDevAttrMaxRegistersPerBlock, device)},
        {"max_threads_per_block", attribute(cudaDevAttrMaxThreadsPerBlock, device)},
        {"warp_size", attribute(cudaDevAttrWarpSize, device)}};
    // The values resident_ctas_per_sm assumes, under the raw attributes' own names.
    Json occupancy{{"reserved_shared_per_block",
                    static_cast<int>(ninferno::reserved_shared_bytes_per_cta(q.major))},
                   {"shared_per_block_default", ninferno::kAssumedSharedPerBlock},
                   {"registers_per_block", ninferno::kAssumedRegsPerBlock},
                   {"max_threads_per_block", ninferno::kAssumedMaxThreadsPerBlock},
                   {"warp_size", ninferno::kAssumedWarpSize}};
    // What the query stored: runtime code reads q.facts, so a query that does not wire it shows
    // here instead of being recomputed from the same inputs.
    const std::string bus_id = pci_bus_id(device);
    return Json{{"index", device},
                {"pci_bus_id", bus_id},
                {"mig_mode", mig_mode(bus_id)},
                {"queried", class_of(q)},
                {"synthetic", class_of(synthetic)},
                {"raw", std::move(raw)},
                {"engine_occupancy_limits", std::move(occupancy)},
                {"engine_table_facts", facts_json(q.facts)},
                {"engine_row_facts", facts_json(row_alone_facts(q))}};
}

// A whole, non-negative ordinal ("2"), or -1 for anything else ("-1", "abc", "1x").
int device_ordinal(const char* text) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    return end != text && *end == '\0' && value >= 0 && value < ninferno::kMaxDevices
               ? static_cast<int>(value)
               : -1;
}

} // namespace

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
            // One board that cannot be read must not lose the others' report.
            try {
                devices.push_back(device_entry(device));
            } catch (const std::exception& error) {
                devices.push_back(Json{{"index", device}, {"error", error.what()}});
            }
        }
        std::puts(Json{{"devices", std::move(devices)}}.dump(1).c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninferno-facts-probe: %s\n", error.what());
        return 1;
    }
}
