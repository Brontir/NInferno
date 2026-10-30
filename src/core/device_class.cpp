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

// Full-chip SM counts per architecture, ascending.
struct FullChip {
    int major, minor, sms;
    const char* chip;
};

constexpr FullChip kFullChips[] = {
    {12, 0, 20, "gb207"},  {12, 0, 36, "gb206"}, {12, 0, 50, "gb205"}, {12, 0, 84, "gb203"},
    {12, 0, 192, "gb202"}, {8, 9, 24, "ad107"},  {8, 9, 36, "ad106"},  {8, 9, 60, "ad104"},
    {8, 9, 80, "ad103"},   {8, 9, 144, "ad102"}, {8, 6, 20, "ga107"},  {8, 6, 30, "ga106"},
    {8, 6, 48, "ga104"},   {8, 6, 60, "ga103"},  {8, 6, 84, "ga102"},  {7, 5, 16, "tu117"},
    {7, 5, 24, "tu116"},   {7, 5, 36, "tu106"},  {7, 5, 48, "tu104"},  {7, 5, 72, "tu102"},
    {8, 0, 128, "ga100"},  {7, 0, 80, "gv100"},
};

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

bool chip_mma_slow(int major, int minor, std::string_view chip) noexcept {
    return major == 7 && minor == 5 && (chip == "tu116" || chip == "tu117");
}

bool board_mma_slow(const MachineFacts& facts, int major, int minor,
                    std::string_view chip) noexcept {
    return !facts.sku.empty() ? facts.mma_slow : chip_mma_slow(major, minor, chip);
}

DeviceFeatures device_features_for(int major, int minor) noexcept {
    switch (major) {
    case 7: // Turing (7.5): the sm_75 build composes the m16n8k16 / m16n8k32 shapes from its f16
            // m16n8k8 and s8 m8n8k16 and rounds BF16 operands to f16. Volta (7.0, 7.2) has
            // neither: its build computes the same fragments in FP32 FMAs and dp4a over warp
            // shuffles (ops/common/mma.cuh), so every MMA path runs, exactly.
        return {.mma_bf16 = true, .mma_tensor = minor >= 5};
    case 8: // Ampere (8.0, 8.6, 8.7) and Ada (8.9): the E4M3 mma.sync arrived with Ada.
        return {
            .mma_bf16 = true, .mma_bf16_native = true, .mma_tensor = true, .mma_fp8 = minor >= 9};
    case 9: // Hopper
        return {.mma_bf16        = true,
                .mma_bf16_native = true,
                .mma_tensor      = true,
                .mma_fp8         = true,
                .tma             = true,
                .pdl             = true,
                .clusters        = true};
    case 10: // datacenter Blackwell: the block-scaled FP8/FP4 kinds go through tcgen05, not
    case 11: // mma.sync; the un-prefixed E4M3 mma.sync is PTX every target from sm_89 carries
        return {.mma_bf16        = true,
                .mma_bf16_native = true,
                .mma_tensor      = true,
                .mma_fp8         = true,
                .tma             = true,
                .pdl             = true,
                .clusters        = true};
    case 12: // consumer Blackwell (sm_120, sm_121)
        return {.mma_bf16        = true,
                .mma_bf16_native = true,
                .mma_tensor      = true,
                .mma_fp8         = true,
                .mma_f8f6f4      = true,
                .mma_mxf4nvf4    = true,
                .tma             = true,
                .pdl             = true,
                .clusters        = true};
    default:
        // A later generation than this table knows (major > 12) keeps every portable family:
        // mma.sync BF16 and E4M3, cp.async.bulk and griddepcontrol are PTX the newer targets
        // carry; the architecture-specific block-scaled kinds are not assumed. Older unknowns
        // have none.
        if (major > 12) {
            return {.mma_bf16        = true,
                    .mma_bf16_native = true,
                    .mma_tensor      = true,
                    .mma_fp8         = true,
                    .tma             = true,
                    .pdl             = true,
                    .clusters        = true};
        }
        return {};
    }
}

bool has_features(const DeviceFeatures& available, const DeviceFeatures& required) noexcept {
    return (!required.mma_bf16 || available.mma_bf16) &&
           (!required.mma_bf16_native || available.mma_bf16_native) &&
           (!required.mma_tensor || available.mma_tensor) &&
           (!required.mma_fp8 || available.mma_fp8) &&
           (!required.mma_f8f6f4 || available.mma_f8f6f4) &&
           (!required.mma_mxf4nvf4 || available.mma_mxf4nvf4) && (!required.tma || available.tma) &&
           (!required.pdl || available.pdl) && (!required.clusters || available.clusters);
}

DeviceFeatures common_features(const DeviceFeatures& a, const DeviceFeatures& b) noexcept {
    return {.mma_bf16        = a.mma_bf16 && b.mma_bf16,
            .mma_bf16_native = a.mma_bf16_native && b.mma_bf16_native,
            .mma_tensor      = a.mma_tensor && b.mma_tensor,
            .mma_fp8         = a.mma_fp8 && b.mma_fp8,
            .mma_f8f6f4      = a.mma_f8f6f4 && b.mma_f8f6f4,
            .mma_mxf4nvf4    = a.mma_mxf4nvf4 && b.mma_mxf4nvf4,
            .tma             = a.tma && b.tma,
            .pdl             = a.pdl && b.pdl,
            .clusters        = a.clusters && b.clusters,
            .mma_slow        = a.mma_slow || b.mma_slow};
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

std::string gpu_arch_id(int major, int minor) {
    return "sm" + std::to_string(major) + std::to_string(minor);
}

namespace {

// The chip whose PCI id block holds pci_device_id, for a GPU of that chip's architecture; null
// for an unknown id.
const char* pci_block_chip(std::uint32_t pci_device_id, int major, int minor) {
    // NVIDIA gives each chip blocks of PCI product ids. Verified on hardware: 0x2B85 (RTX 5090,
    // gb202) and 0x2C02 (RTX 5080, gb203); the other rows follow the published id lists.
    struct IdBlock {
        std::uint32_t first, last;
        const char* chip;
    };

    static constexpr IdBlock kIds[] = {
        {0x2B80, 0x2BFF, "gb202"}, {0x2C00, 0x2C7F, "gb203"}, {0x2F00, 0x2F7F, "gb205"},
        {0x2D00, 0x2D7F, "gb206"}, {0x2D80, 0x2DFF, "gb207"}, {0x2680, 0x26FF, "ad102"},
        {0x2700, 0x277F, "ad103"}, {0x2780, 0x27FF, "ad104"}, {0x2800, 0x287F, "ad106"},
        {0x2880, 0x28FF, "ad107"}, {0x2200, 0x227F, "ga102"}, {0x2400, 0x247F, "ga103"},
        {0x2480, 0x24FF, "ga104"}, {0x2500, 0x257F, "ga106"}, {0x2580, 0x25FF, "ga107"},
        {0x1E00, 0x1E7F, "tu102"}, {0x1E80, 0x1EFF, "tu104"}, {0x1F00, 0x1F7F, "tu106"},
        {0x1F80, 0x1FFF, "tu117"}, {0x2180, 0x21FF, "tu116"}, {0x1D80, 0x1DFF, "gv100"},
        {0x2080, 0x20FF, "ga100"},
    };
    // A block names its chip only for a GPU of that chip's architecture: Jetson Thor (sm_110)
    // reports 0x2B00, inside GB202's block, and must not take an sm_120 chip's data.
    for (const auto& block : kIds) {
        if (pci_device_id < block.first || pci_device_id > block.last) continue;
        for (const auto& c : kFullChips) {
            if (c.chip == std::string_view(block.chip) && c.major == major && c.minor == minor) {
                return block.chip;
            }
        }
    }
    return nullptr;
}

} // namespace

bool gpu_chip_by_pci(std::uint32_t pci_device_id, int major, int minor) {
    return pci_block_chip(pci_device_id, major, minor) != nullptr;
}

std::vector<std::string> device_fact_notes(const DeviceClass& d) {
    std::vector<std::string> notes;
    if (!d.chip_from_pci) {
        char id[16];
        std::snprintf(id, sizeof(id), "0x%04X", static_cast<unsigned>(d.pci_device_id));
        const std::string why = d.pci_device_id == 0 ? std::string("PCI id unreadable")
                                                     : std::string("PCI id ") + id + " in no chip block";
        const std::string arch = gpu_arch_id(d.major, d.minor);
        // The PCI-ID reference table may still know the board (H100: its chip has no block): its
        // facts are the table's, only the per-chip keys are guessed.
        const GpuFactsRow* row = d.facts.sku.empty() ? nullptr : find_gpu_facts(d.pci_device_id);
        const std::string board =
            row != nullptr ? "; the board's clocks and bandwidth are the reference table's " +
                                 row->name + " (" + row->chip + ")"
                           : std::string();
        notes.push_back(d.chip == arch ? "chip unknown (" + why + "; no chip of " + arch +
                                             " in the chip blocks): per-chip data is keyed by the "
                                             "architecture " +
                                             arch + board
                                       : "chip " + d.chip + " from the SM count (" + why +
                                             "): the smallest chip of " + arch + " holding " +
                                             std::to_string(d.sms) + " SMs" + board);
    }
    std::string missing;
    const auto add = [&](bool absent, const char* what) {
        if (absent) missing += (missing.empty() ? "" : ", ") + std::string(what);
    };
    add(d.sm_clock_khz <= 0, "SM clock");
    add(d.memory_clock_khz <= 0, "memory clock");
    add(d.memory_bus_bits <= 0, "memory bus width");
    const MachineFacts& f = d.facts;
    if (!missing.empty() && f.has_roofline()) {
        // A board the PCI-ID table knows: its row stands in, and no decision reads the driver's.
        notes.push_back(missing +
                        " unreported (a virtualized or MIG GPU, for example): the "
                        "PCI-ID reference table's facts stand in (" +
                        describe(f) + ")");
    } else if (!missing.empty()) {
        notes.push_back(missing +
                        " unreported (a virtualized or MIG GPU, for example): the route cost model "
                        "and the speculative load model are off, routes come from the fixed rules, "
                        "and measured tables are matched to this board without its bandwidth");
    }
    // Facts that only approximate this device (MachineFacts::source extrapolated), with the
    // conservative choice taken.
    if (f.source == FactSource::Extrapolated) {
        std::string why;
        if (f.sku.empty()) {
            why = "PCI id not in the reference table: the driver's clocks on the 2^(1/16) grid";
        } else if (f.form == FormFactor::Laptop) {
            why = f.sku + ", a laptop: priced at the bottom of its published boost range";
        } else if (f.share < 1.0) {
            why = f.sku + ", a slice of " + std::to_string(d.sms) +
                  " SMs (MIG): its share of the board's bandwidth";
        } else if (const GpuFactsRow* row = find_gpu_facts(d.pci_device_id);
                   row != nullptr && row->sms == d.sms) {
            why = f.sku + " on another memory than its row's: the driver's bus width and memory "
                          "clock price it";
        } else {
            why = f.sku + " reporting " + std::to_string(d.sms) + " SMs, not the row's";
        }
        notes.push_back("device facts extrapolated (" + why + "): " + describe(f));
    }
    return notes;
}

std::string gpu_chip(std::uint32_t pci_device_id, int major, int minor, int sms) {
    if (const char* chip = pci_block_chip(pci_device_id, major, minor)) return chip;
    // Unknown id (or unreadable): the smallest chip of the architecture whose full SM count holds
    // the board's. Full-chip SM counts per architecture, ascending.
    for (const auto& c : kFullChips) {
        if (c.major == major && c.minor == minor && sms <= c.sms) { return c.chip; }
    }
    return gpu_arch_id(major, minor);
}

DriverReport driver_report_of(const DeviceClass& d) noexcept {
    return {.pci_device_id    = d.pci_device_id,
            .major            = d.major,
            .minor            = d.minor,
            .sms              = d.sms,
            .sm_clock_khz     = d.sm_clock_khz,
            .memory_clock_khz = d.memory_clock_khz,
            .memory_bus_bits  = d.memory_bus_bits};
}

MachineFacts machine_facts_of(const DeviceClass& d) { return machine_facts(driver_report_of(d)); }

Board board_of(const DeviceClass& d) noexcept {
    return {d.major, d.minor, d.sms, static_cast<double>(d.l2_bytes) / (1 << 20), d.dram_gbps(),
            d.features.mma_slow};
}

namespace {

double board_distance_of(const Board& a, const Board& b, bool hmma_rate) noexcept {
    const DeviceFeatures fa = device_features_for(a.major, a.minor);
    const DeviceFeatures fb = device_features_for(b.major, b.minor);
    if (!has_features(fa, fb) || !has_features(fb, fa) || a.mma_slow != b.mma_slow || a.sms <= 0 ||
        b.sms <= 0 ||
        (hmma_rate && generation_row(a.major, a.minor).fp32acc_full_rate !=
                          generation_row(b.major, b.minor).fp32acc_full_rate)) {
        return std::numeric_limits<double>::infinity();
    }
    const auto apart = [](double x, double y) {
        return x > 0 && y > 0 ? std::abs(std::log2(x) - std::log2(y)) : 0.0; // symmetric to the bit
    };
    return apart(a.sms, b.sms) + apart(a.dram_gbps, b.dram_gbps) + 0.5 * apart(a.l2_mib, b.l2_mib);
}

} // namespace

double board_distance(const Board& a, const Board& b) noexcept {
    return board_distance_of(a, b, true);
}

double board_memory_distance(const Board& a, const Board& b) noexcept {
    return board_distance_of(a, b, false);
}

std::string gpu_chip_arch(std::string_view gpu_class) {
    const auto chip = gpu_class.substr(0, gpu_class.find(':'));
    if (chip.size() > 2 && chip.starts_with("sm") &&
        chip.find_first_not_of("0123456789", 2) == std::string_view::npos) {
        if (chip.size() != gpu_class.size()) { return {}; }
        for (const auto& full : kFullChips) {
            if (gpu_arch_id(full.major, full.minor) == chip) { return std::string(chip); }
        }
        return {};
    }
    for (const auto& full : kFullChips) {
        if (chip == full.chip) { return gpu_arch_id(full.major, full.minor); }
    }
    return {};
}

int gpu_chip_full_sms(std::string_view gpu_class) {
    const auto chip = gpu_class.substr(0, gpu_class.find(':'));
    for (const auto& full : kFullChips) {
        if (chip == full.chip) { return full.sms; }
    }
    return 0;
}

std::string gpu_class_id(std::string_view chip, int sms) {
    return std::string(chip) + ":" + std::to_string(sms) + "sm";
}

std::vector<std::string> gpu_similarity_chain(std::string_view chip, int sms, int major, int minor,
                                              std::optional<bool> mma_slow) {
    std::vector<std::string> chain = gpu_class_chain(chip, sms, gpu_arch_id(major, minor));
    chain.pop_back(); // "*" goes after the architecture's own chips
    const auto add = [&](std::string key) {
        if (std::find(chain.begin(), chain.end(), key) == chain.end()) {
            chain.push_back(std::move(key));
        }
    };
    for (auto& similar : gpu_similar_chips(major, minor, sms, chip, mma_slow)) {
        add(std::move(similar));
    }
    add("*");
    // The other architectures, nearest compute capability first.
    std::vector<std::pair<int, int>> archs;
    for (const auto& c : kFullChips) {
        const std::pair<int, int> arch{c.major, c.minor};
        if (arch != std::pair{major, minor} &&
            std::find(archs.begin(), archs.end(), arch) == archs.end()) {
            archs.push_back(arch);
        }
    }
    const int cc = major * 10 + minor;
    std::stable_sort(archs.begin(), archs.end(), [cc](const auto& a, const auto& b) {
        return std::abs(a.first * 10 + a.second - cc) < std::abs(b.first * 10 + b.second - cc);
    });
    for (const auto& [other_major, other_minor] : archs) {
        for (auto& similar : gpu_similar_chips(other_major, other_minor, sms, "")) {
            add(std::move(similar));
        }
    }
    return chain;
}

std::vector<std::string> gpu_similar_chips(int major, int minor, int sms, std::string_view own_chip,
                                           std::optional<bool> mma_slow) {
    // A board without tensor cores (mma_slow) and one with them are other kinds: measured data of
    // one does not carry to the other. The board's own fact, where given: a GTX board on TU106
    // has none, though its chip does.
    const bool slow = mma_slow.value_or(chip_mma_slow(major, minor, own_chip));
    std::vector<const FullChip*> chips;
    for (const auto& c : kFullChips) {
        if (c.major == major && c.minor == minor && own_chip != c.chip &&
            chip_mma_slow(c.major, c.minor, c.chip) == slow) {
            chips.push_back(&c);
        }
    }
    const double board  = std::log2(static_cast<double>(std::max(sms, 1)));
    const auto distance = [board](const FullChip* c) {
        return std::abs(std::log2(static_cast<double>(c->sms)) - board);
    };
    const auto nearest_first = [&](const FullChip* a, const FullChip* b) {
        const double da = distance(a);
        const double db = distance(b);
        return da != db ? da < db : a->sms > b->sms;
    };
    std::stable_sort(chips.begin(), chips.end(), nearest_first);
    // Then the chips of a sibling architecture: the same major with the same feature row (sm_80
    // and sm_86, sm_121 and sm_120), where the measured data of the other minor is the nearest
    // there is; an architecture with no chip of its own (A100, A30, GB10) otherwise borrows none.
    const DeviceFeatures own = device_features_for(major, minor);
    const auto first_sibling = chips.size();
    for (const auto& c : kFullChips) {
        const DeviceFeatures other = device_features_for(c.major, c.minor);
        if (c.major == major && c.minor != minor && own_chip != c.chip &&
            has_features(own, other) && has_features(other, own)) {
            chips.push_back(&c);
        }
    }
    std::stable_sort(chips.begin() + static_cast<std::ptrdiff_t>(first_sibling), chips.end(),
                     nearest_first);
    std::vector<std::string> out;
    out.reserve(chips.size());
    for (const FullChip* c : chips) { out.emplace_back(c->chip); }
    return out;
}

std::vector<std::string> gpu_class_chain(std::string_view chip, int sms, std::string_view arch) {
    std::vector<std::string> chain;
    for (std::string key : {gpu_class_id(chip, sms), std::string(chip), std::string(arch),
                            std::string("*")}) {
        if (!key.empty() && std::find(chain.begin(), chain.end(), key) == chain.end()) {
            chain.push_back(std::move(key));
        }
    }
    return chain;
}

std::string DeviceClass::id() const { return gpu_class_id(chip, sms); }

std::string DeviceClass::arch() const { return gpu_arch_id(major, minor); }

std::vector<std::string> DeviceClass::class_chain() const {
    return gpu_class_chain(chip, sms, arch());
}

// The board's own mma_slow is features.mma_slow alone: the query and synthetic_device_class set
// it from board_mma_slow, its one source.
std::vector<std::string> DeviceClass::similar_chips() const {
    return gpu_similar_chips(major, minor, sms, chip, features.mma_slow);
}

std::vector<std::string> DeviceClass::similarity_chain() const {
    return gpu_similarity_chain(chip, sms, major, minor, features.mma_slow);
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

namespace {
thread_local const DeviceClass* t_substitute       = nullptr;
thread_local const std::string* t_substitute_key   = nullptr;
} // namespace

ScopedDeviceClass::ScopedDeviceClass(DeviceClass substitute)
    : substitute_(std::move(substitute)), key_(device_class_key(substitute_)),
      previous_(t_substitute), previous_key_(t_substitute_key) {
    t_substitute     = &substitute_;
    t_substitute_key = &key_;
}

ScopedDeviceClass::~ScopedDeviceClass() {
    t_substitute     = previous_;
    t_substitute_key = previous_key_;
}

const DeviceClass* substituted_device_class() noexcept { return t_substitute; }

const std::string* substituted_device_class_key() noexcept { return t_substitute_key; }

bool substituted_foreign_class() noexcept {
    if (t_substitute_key == nullptr) return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return true;
    }
    try {
        return device_class_key(device_class(device)) != *t_substitute_key;
    } catch (const std::exception&) {
        return true;
    }
}

std::string device_class_key(const DeviceClass& d) {
    const auto& f = d.features;
    const auto& m = d.facts;
    std::string key;
    const auto add = [&](const auto& v) {
        key += std::to_string(v);
        key += ',';
    };
    key += d.chip + ',' + m.sku + ',';
    for (const auto v : {d.major, d.minor, d.sms, d.max_threads_per_sm, d.max_blocks_per_sm,
                         d.registers_per_sm, d.memory_bus_bits, d.memory_clock_khz, d.sm_clock_khz,
                         m.reference_clock_khz, m.memory_bus_bits, m.memory_clock_khz, m.tdp_w,
                         m.pcie_gen, static_cast<int>(m.source), static_cast<int>(m.segment),
                         static_cast<int>(m.form)}) {
        add(v);
    }
    for (const auto v : {d.l2_bytes, d.smem_per_block_optin, d.smem_per_sm, d.global_bytes,
                         d.persisting_l2_max_bytes}) {
        add(v);
    }
    add(d.pci_device_id);
    add(m.vram_gib);
    add(m.share);
    for (const bool b : {d.chip_from_pci, f.mma_bf16, f.mma_bf16_native, f.mma_tensor, f.mma_fp8,
                         f.mma_f8f6f4, f.mma_mxf4nvf4, f.tma, f.pdl, f.clusters, f.mma_slow,
                         m.mma_slow, m.driver_read}) {
        key += b ? '1' : '0';
    }
    return key;
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
    check(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device), "cudaDeviceGetPCIBusId", device);
    std::optional<DeviceIdentity> identity = parse_pci_bus_id(bus_id);
    if (!identity) {
        throw std::runtime_error(std::string("device class: unreadable PCI bus id \"") + bus_id +
                                 "\" for device " + std::to_string(device));
    }
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

const DeviceClass& device_class() {
    if (t_substitute != nullptr) { return *t_substitute; }
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice", -1);
    return device_class(device);
}

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

} // namespace ninferno
