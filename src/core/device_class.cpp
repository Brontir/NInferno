#include "core/device_class.h"
#include "core/generation.h"

#include <cuda_runtime.h>
#include "core/sys.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ninferno {

// core/device_image_probe.cu: whether the device can load a kernel compiled with this build.
bool probe_device_kernel_image(int device);

namespace {

constexpr int kMaxDevices = 64;


struct Slot {
    std::once_flag once;
    std::unique_ptr<DeviceClass> value;
    std::once_flag image_once;
    bool image = false;
};

std::array<Slot, kMaxDevices>& slots() {
    static std::array<Slot, kMaxDevices> table;
    return table;
}

void check(cudaError_t err, const char* what, int device) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("device class: ") + what + " failed for device " +
                                 std::to_string(device) + ": " + cudaGetErrorName(err));
    }
}

int attribute(cudaDeviceAttr attr, int device) {
    int value = 0;
    check(cudaDeviceGetAttribute(&value, attr, device), "cudaDeviceGetAttribute", device);
    return value;
}

// An attribute the cost model reads but the device need not report (a virtualized or MIG GPU
// answers the clocks and the bus width as unsupported): 0, which the model takes as unknown.
int optional_attribute(cudaDeviceAttr attr, int device) {
    int value = 0;
    if (cudaDeviceGetAttribute(&value, attr, device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return value > 0 ? value : 0;
}

// The PCI product id (the upper half of NVIDIA's device/vendor word): from sysfs on a native Linux
// driver, else from NVML (WSL2 exposes no PCI sysfs for the GPU), loaded at run time so nothing
// links against it. 0 when neither answers; the chip then comes from the SM count.
std::uint32_t pci_product_id(const cudaDeviceProp& prop, int device) {
    const auto sysfs_word = [&](const char* field) {
        char path[128];
        std::snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.0/%s",
                      prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, field);
        unsigned value = 0;
        if (std::FILE* file = std::fopen(path, "r")) {
            if (std::fscanf(file, "%x", &value) != 1) { value = 0; }
            std::fclose(file);
        }
        return value;
    };
    constexpr unsigned kNvidiaVendor = 0x10DE;
    if (sysfs_word("vendor") == kNvidiaVendor) {
        if (const unsigned value = sysfs_word("device"); value != 0) { return value; }
    }
    // nvmlPciInfo_t (nvml.h, unchanged since NVML's v2 PCI info).
    struct PciInfo {
        char bus_id_legacy[16];
        unsigned domain, bus, device, pci_device_id, pci_subsystem_id;
        char bus_id[32];
    };
    using Init     = int (*)();
    using ByBusId  = int (*)(const char*, void**);
    using GetPci   = int (*)(void*, PciInfo*);
    using Shutdown = int (*)();
    void* nvml = sys::load_library("libnvidia-ml.so.1", "nvml.dll");
    if (nvml == nullptr) { return 0; }
    std::uint32_t id    = 0;
    const auto init     = reinterpret_cast<Init>(sys::symbol(nvml, "nvmlInit_v2"));
    const auto by_bus   = reinterpret_cast<ByBusId>(sys::symbol(nvml, "nvmlDeviceGetHandleByPciBusId_v2"));
    const auto get_pci  = reinterpret_cast<GetPci>(sys::symbol(nvml, "nvmlDeviceGetPciInfo_v3"));
    const auto shutdown = reinterpret_cast<Shutdown>(sys::symbol(nvml, "nvmlShutdown"));
    char bus_id[32]     = {};
    if (init && by_bus && get_pci && shutdown && init() == 0) {
        void* handle = nullptr;
        PciInfo info{};
        if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device) == cudaSuccess &&
            by_bus(bus_id, &handle) == 0 && get_pci(handle, &info) == 0 &&
            (info.pci_device_id & 0xFFFFu) == kNvidiaVendor) {
            id = info.pci_device_id >> 16;
        }
        shutdown();
    }
    sys::free_library(nvml);
    return id;
}

std::unique_ptr<DeviceClass> query(int device) {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties", device);
    auto out                  = std::make_unique<DeviceClass>();
    out->device               = device;
    out->major                = prop.major;
    out->minor                = prop.minor;
    out->sms                  = attribute(cudaDevAttrMultiProcessorCount, device);
    out->l2_bytes             = static_cast<std::size_t>(attribute(cudaDevAttrL2CacheSize, device));
    out->smem_per_block_optin = static_cast<std::size_t>(
        attribute(cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    out->smem_per_sm = static_cast<std::size_t>(
        attribute(cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    out->global_bytes            = prop.totalGlobalMem;
    out->name                    = prop.name;
    out->max_threads_per_sm      = prop.maxThreadsPerMultiProcessor;
    out->max_blocks_per_sm       = prop.maxBlocksPerMultiProcessor;
    out->registers_per_sm        = prop.regsPerMultiprocessor;
    out->persisting_l2_max_bytes = static_cast<std::size_t>(prop.persistingL2CacheMaxSize);
    // cudaDeviceProp lost its clock and bus fields in CUDA 13; the attributes remain.
    out->memory_bus_bits  = optional_attribute(cudaDevAttrGlobalMemoryBusWidth, device);
    out->memory_clock_khz = optional_attribute(cudaDevAttrMemoryClockRate, device);
    out->sm_clock_khz     = optional_attribute(cudaDevAttrClockRate, device);
    out->features = common_features(device_features_for(prop.major, prop.minor), build_features());
    out->pci_device_id     = pci_product_id(prop, device);
    out->chip              = gpu_chip(out->pci_device_id, prop.major, prop.minor, out->sms);
    out->chip_from_pci     = gpu_chip_by_pci(out->pci_device_id, prop.major, prop.minor);
    out->facts             = machine_facts_of(*out);
    out->features.mma_slow = board_mma_slow(out->facts, prop.major, prop.minor, out->chip);
    // A driver that cannot say which board (a paravirtual or vGPU one) leaves it unknown, sorted
    // last; the engine still starts.
    try {
        out->identity = query_device_identity(device);
    } catch (const std::runtime_error&) {
        (void)cudaGetLastError();
        out->identity = {.pci_domain = kUnknownPciDomain};
    }
    return out;
}

} // namespace

int host_gpu_count() noexcept {
    static const int count = [] {
        int n = 0;
        if (void* nvml = sys::load_library("libnvidia-ml.so.1", "nvml.dll")) {
            using Init      = int (*)();
            using Count     = int (*)(unsigned*);
            using ByIndex   = int (*)(unsigned, void**);
            using MigMode   = int (*)(void*, unsigned*, unsigned*);
            using MigMax    = int (*)(void*, unsigned*);
            using MigHandle = int (*)(void*, unsigned, void**);
            using Shutdown  = int (*)();
            const auto init = reinterpret_cast<Init>(sys::symbol(nvml, "nvmlInit_v2"));
            const auto get  = reinterpret_cast<Count>(sys::symbol(nvml, "nvmlDeviceGetCount_v2"));
            const auto by_index =
                reinterpret_cast<ByIndex>(sys::symbol(nvml, "nvmlDeviceGetHandleByIndex_v2"));
            const auto mig_mode =
                reinterpret_cast<MigMode>(sys::symbol(nvml, "nvmlDeviceGetMigMode"));
            const auto mig_max =
                reinterpret_cast<MigMax>(sys::symbol(nvml, "nvmlDeviceGetMaxMigDeviceCount"));
            const auto mig_handle = reinterpret_cast<MigHandle>(
                sys::symbol(nvml, "nvmlDeviceGetMigDeviceHandleByIndex"));
            const auto shutdown = reinterpret_cast<Shutdown>(sys::symbol(nvml, "nvmlShutdown"));
            if (init && get && shutdown && init() == 0) {
                unsigned devices = 0;
                if (get(&devices) == 0) {
                    // A board in MIG mode (NVML_DEVICE_MIG_ENABLE, 1) counts its instances: a
                    // server may run on each (CUDA_VISIBLE_DEVICES=MIG-<uuid>). A driver without
                    // the MIG calls, or a board without instances, counts as one GPU.
                    for (unsigned d = 0; d < devices; ++d) {
                        int units      = 1;
                        void* board    = nullptr;
                        unsigned mode  = 0;
                        unsigned spare = 0;
                        unsigned slots = 0;
                        if (by_index && mig_mode && mig_max && mig_handle &&
                            by_index(d, &board) == 0 && mig_mode(board, &mode, &spare) == 0 &&
                            mode == 1 && mig_max(board, &slots) == 0) {
                            int instances = 0;
                            for (unsigned i = 0; i < slots; ++i) {
                                void* instance = nullptr;
                                if (mig_handle(board, i, &instance) == 0) { ++instances; }
                            }
                            units = std::max(1, instances);
                        }
                        n += units;
                    }
                }
                shutdown();
            }
            sys::free_library(nvml);
        }
        if (n <= 0) {
            int cuda = 0;
            if (cudaGetDeviceCount(&cuda) == cudaSuccess) { n = cuda; }
        }
        return std::max(1, n);
    }();
    return count;
}

DeviceFeatures build_features() noexcept {
#ifdef NINFERNO_BUILD_ARCH
    DeviceFeatures f = device_features_for(NINFERNO_BUILD_ARCH / 10, NINFERNO_BUILD_ARCH % 10);
#    ifndef NINFERNO_BUILD_ARCH_SPECIFIC
    f.mma_f8f6f4   = false;
    f.mma_mxf4nvf4 = false;
#    endif
    return f;
#else
    // A translation unit built outside the tree's CMake: no limit of its own.
    return {.mma_bf16        = true,
            .mma_bf16_native = true,
            .mma_tensor      = true,
            .mma_fp8         = true,
            .mma_f8f6f4      = true,
            .mma_mxf4nvf4    = true,
            .tma             = true,
            .pdl             = true,
            .clusters        = true};
#endif
}

const DeviceClass& device_class(int device) {
    if (device < 0 || device >= kMaxDevices) {
        throw std::runtime_error("device class: invalid CUDA device ordinal " +
                                 std::to_string(device));
    }
    auto& slot = slots()[static_cast<std::size_t>(device)];
    std::call_once(slot.once, [&] { slot.value = query(device); });
    return *slot.value;
}

bool device_has_kernel_image(int device) {
    if (device < 0 || device >= kMaxDevices) {
        throw std::runtime_error("device class: invalid CUDA device ordinal " +
                                 std::to_string(device));
    }
    auto& slot = slots()[static_cast<std::size_t>(device)];
    std::call_once(slot.image_once, [&] { slot.image = probe_device_kernel_image(device); });
    return slot.image;
}



bool substituted_foreign_class() noexcept {
    const std::string* substituted_key = substituted_device_class_key();
    if (substituted_key == nullptr) return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return true;
    }
    try {
        return device_class_key(device_class(device)) != *substituted_key;
    } catch (const std::exception&) {
        return true;
    }
}


DeviceClass synthetic_device_class(std::uint32_t pci_device_id, int major, int minor, int sms,
                                   std::size_t l2_bytes, std::size_t global_bytes) {
    DeviceClass d;
    d.major         = major;
    d.minor         = minor;
    d.sms           = sms;
    d.l2_bytes      = l2_bytes;
    d.global_bytes  = global_bytes;
    d.name          = "synthetic " + gpu_arch_id(major, minor);
    d.pci_device_id = pci_device_id;
    d.chip          = gpu_chip(pci_device_id, major, minor, sms);
    d.chip_from_pci = gpu_chip_by_pci(pci_device_id, major, minor);
    d.features      = device_features_for(major, minor);
    // board_mma_slow as the query sets it (the table's row, else the chip's); `facts` itself stays
    // empty until a test asks for it.
    d.features.mma_slow = board_mma_slow(machine_facts_of(d), major, minor, d.chip);
    // The architecture's limits (CUDA programming guide, compute capability tables).
    const int cc           = major * 10 + minor;
    d.smem_per_block_optin = cc == 75                  ? 64 * 1024
                             : cc == 80 || cc == 87      ? 163 * 1024
                             : cc >= 90 && cc < 120      ? 227 * 1024
                                                         : 99 * 1024;
    d.smem_per_sm          = cc == 75                  ? 64 * 1024
                             : cc == 80 || cc == 87      ? 164 * 1024
                             : cc >= 90 && cc < 120      ? 228 * 1024
                                                         : 100 * 1024;
    d.max_threads_per_sm   = cc == 75 ? 1024 : cc == 80 || (cc >= 90 && cc < 120) ? 2048 : 1536;
    d.max_blocks_per_sm    = cc == 75 ? 16 : cc == 86 ? 16 : cc == 89 ? 24 : 32;
    d.registers_per_sm     = 64 * 1024;
    d.persisting_l2_max_bytes = l2_bytes * 3 / 4;
    return d;
}

DeviceIdentity query_device_identity(int device) {
    char bus_id[32] = {};
    std::optional<DeviceIdentity> identity;
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device) == cudaSuccess) {
        identity = parse_pci_bus_id(bus_id);
    } else {
        (void)cudaGetLastError();
    }
    if (!identity) { identity = DeviceIdentity{.pci_domain = kUnknownPciDomain}; }
    // A UUID the driver does not report stays zero: the PCI address still orders the board.
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        for (std::size_t i = 0; i < identity->uuid.size(); ++i) {
            identity->uuid[i] = static_cast<std::uint8_t>(prop.uuid.bytes[i]);
        }
    } else {
        (void)cudaGetLastError();
    }
    return *identity;
}

std::vector<DeviceIdentity> visible_device_identities() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return {};
    }
    std::vector<DeviceIdentity> identities;
    for (int device = 0; device < count; ++device) {
        try {
            identities.push_back(query_device_identity(device));
        } catch (const std::runtime_error&) {
            (void)cudaGetLastError();
            identities.push_back({.pci_domain = kUnknownPciDomain});
        }
    }
    return identities;
}

int resolve_device(int requested) {
    if (requested != kDefaultDevice) { return requested; }
    const std::vector<DeviceIdentity> identities = visible_device_identities();
    const int device                             = resolve_device(requested, identities);
    return device == kDefaultDevice ? 0 : device;
}

int select_device(const DeviceSelector& selector) {
    const std::vector<DeviceIdentity> identities = visible_device_identities();
    if (identities.empty()) { return 0; }
    if (const auto device = select_device(selector, identities)) { return *device; }
    std::string visible;
    const std::vector<int> order = canonical_device_order(identities);
    for (std::size_t i = 0; i < order.size(); ++i) {
        visible += (i == 0 ? "" : ", ") + std::to_string(i) + " = " +
                   identity_text(identities[static_cast<std::size_t>(order[i])]);
    }
    throw std::invalid_argument("--device " + device_selector_text(selector) +
                                " names no visible GPU; visible in PCI bus order: " + visible);
}

std::string visible_devices_note(int device, const DeviceSelector& selector) {
    const std::vector<DeviceIdentity> identities = visible_device_identities();
    if (identities.size() < 2 || device < 0 || device >= static_cast<int>(identities.size())) {
        return "";
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) { (void)cudaGetLastError(); }
    const std::string how = selector.kind == DeviceSelector::Kind::Default
                                ? "first in PCI order; --device to choose"
                                : "--device " + device_selector_text(selector);
    const DeviceIdentity& board = identities[static_cast<std::size_t>(device)];
    return std::to_string(identities.size()) + " GPUs visible; using " +
           (pci_known(board) ? pci_bus_id_text(board) : std::string("an unreported PCI address")) +
           ' ' + prop.name + " (" + how + ")";
}

const DeviceIdentity& device_identity() { return device_class().identity; }


int kernel_resident_ctas_per_sm(const void* kernel, int threads,
                                std::size_t dynamic_shared_bytes) {
    struct Key {
        const void* kernel;
        int threads;
        std::size_t shared;
        int device;
        bool operator<(const Key& o) const {
            return std::tie(kernel, threads, shared, device) <
                   std::tie(o.kernel, o.threads, o.shared, o.device);
        }
    };
    static std::mutex mutex;
    static std::map<Key, int> cache;
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice", -1);
    const Key key{kernel, threads, dynamic_shared_bytes, device};
    std::lock_guard lock(mutex);
    if (const auto it = cache.find(key); it != cache.end()) { return it->second; }
    int ctas = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, kernel, threads,
                                                        dynamic_shared_bytes),
          "cudaOccupancyMaxActiveBlocksPerMultiprocessor", device);
    cache.emplace(key, ctas);
    return ctas;
}

namespace {

std::mutex& probe_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::pair<std::string, const void*>>& registered_probes() {
    static std::vector<std::pair<std::string, const void*>> probes;
    return probes;
}

} // namespace

void register_kernel_image_probe(const char* library, const void* kernel) {
    if (library == nullptr || kernel == nullptr) {
        throw std::invalid_argument("register_kernel_image_probe: library and kernel required");
    }
    std::lock_guard lock(probe_mutex());
    registered_probes().emplace_back(library, kernel);
}

void require_device_support(const DeviceClass& device, const DeviceFeatures& required,
                            const char* consumer) {
    std::string missing;
    const auto need = [&](bool wanted, bool present, const char* feature) {
        if (wanted && !present) { missing += std::string(missing.empty() ? "" : ", ") + feature; }
    };
    need(true, device_has_kernel_image(device.device),
         "a kernel image for this architecture in this build");
    {
        std::lock_guard lock(probe_mutex());
        for (const auto& [library, kernel] : registered_probes()) {
            if (!kernel_image_loadable(kernel, device.device)) {
                missing += std::string(missing.empty() ? "" : ", ") + "a kernel image for this " +
                           "architecture in " + library;
            }
        }
    }
    need(required.mma_bf16, device.features.mma_bf16, "BF16 mma.sync");
    need(required.mma_fp8, device.features.mma_fp8, "E4M3 mma.sync");
    need(required.mma_f8f6f4, device.features.mma_f8f6f4, "mma.sync .kind::f8f6f4");
    need(required.mma_mxf4nvf4, device.features.mma_mxf4nvf4, "mma.sync .kind::mxf4nvf4");
    need(required.tma, device.features.tma, "TMA");
    need(required.pdl, device.features.pdl, "programmatic dependent launch");
    if (!missing.empty()) {
        throw std::runtime_error(std::string(consumer) + " cannot run on " + device.name + " (" +
                                 device.id() + "): missing " + missing);
    }
}

// The device this thread runs on, for device_class() where no class is substituted.
const DeviceClass& current_device_class() {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice", -1);
    return device_class(device);
}

} // namespace ninferno
