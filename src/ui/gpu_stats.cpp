#include "ui/gpu_stats.h"

#include "core/sys.h"

#include <mutex>

namespace ninferno::ui {

double GpuStat::bandwidth_gbs() const noexcept {
    if (memory_bus_bits <= 0 || memory_clock_max_mhz <= 0) { return 0.0; }
    return 2.0 * memory_clock_max_mhz * 1e6 * (memory_bus_bits / 8.0) / 1e9;
}

nlohmann::json GpuStat::to_json() const {
    nlohmann::json processes_json = nlohmann::json::array();
    for (const auto& process : processes) {
        processes_json.push_back({{"pid", process.pid}, {"used_bytes", process.used_bytes}});
    }
    return {{"index", index},
            {"name", name},
            {"uuid", uuid},
            {"bus_id", bus_id},
            {"cc", std::to_string(cc_major) + "." + std::to_string(cc_minor)},
            {"memory_total", memory_total},
            {"memory_used", memory_used},
            {"memory_free", memory_free},
            {"util_gpu", util_gpu},
            {"util_memory", util_memory},
            {"temperature_c", temperature_c},
            {"power_w", power_w},
            {"power_limit_w", power_limit_w},
            {"sm_clock_mhz", sm_clock_mhz},
            {"sm_clock_max_mhz", sm_clock_max_mhz},
            {"memory_clock_mhz", memory_clock_mhz},
            {"bandwidth_gbs", bandwidth_gbs()},
            {"sms", sms},
            {"processes", processes_json}};
}

namespace {

// The NVML structures and entry points used here; their layouts are fixed by the versioned names.
struct NvmlMemory {
    unsigned long long total, free, used;
};

struct NvmlUtilization {
    unsigned gpu, memory;
};

struct NvmlProcessInfo { // nvmlProcessInfo_t as the _v3 process queries fill it
    unsigned pid;
    unsigned long long used_gpu_memory;
    unsigned gpu_instance_id, compute_instance_id;
};

struct NvmlPciInfo {
    char bus_id_legacy[16];
    unsigned domain, bus, device, pci_device_id, pci_subsystem_id;
    char bus_id[32];
};

using Device                      = void*;
constexpr int kOk                 = 0;
constexpr int kInsufficientSize   = 7;
constexpr unsigned kClockGraphics = 0, kClockMemory = 2;
constexpr unsigned long long kUnavailableMemory = ~0ULL; // NVML_VALUE_NOT_AVAILABLE

class NvmlSource final : public GpuSource {
public:
    NvmlSource() {
        library_ = sys::load_library("libnvidia-ml.so.1", "nvml.dll");
        if (library_ == nullptr) {
            error_ = "libnvidia-ml.so.1 not found (no NVIDIA driver?)";
            return;
        }
        bind(init_, "nvmlInit_v2");
        bind(count_, "nvmlDeviceGetCount_v2");
        bind(by_index_, "nvmlDeviceGetHandleByIndex_v2");
        bind(name_, "nvmlDeviceGetName");
        bind(uuid_, "nvmlDeviceGetUUID");
        bind(memory_, "nvmlDeviceGetMemoryInfo");
        bind(utilization_, "nvmlDeviceGetUtilizationRates");
        bind(temperature_, "nvmlDeviceGetTemperature");
        bind(power_, "nvmlDeviceGetPowerUsage");
        bind(power_limit_, "nvmlDeviceGetEnforcedPowerLimit");
        bind(clock_, "nvmlDeviceGetClockInfo");
        bind(max_clock_, "nvmlDeviceGetMaxClockInfo");
        bind(boost_clock_, "nvmlDeviceGetMaxCustomerBoostClock");
        bind(bus_width_, "nvmlDeviceGetMemoryBusWidth");
        bind(capability_, "nvmlDeviceGetCudaComputeCapability");
        bind(pci_, "nvmlDeviceGetPciInfo_v3");
        bind(processes_, "nvmlDeviceGetComputeRunningProcesses_v3");
        bind(cores_, "nvmlDeviceGetNumGpuCores");
        if (init_ == nullptr || count_ == nullptr || by_index_ == nullptr) {
            error_ = "libnvidia-ml.so.1 lacks nvmlInit_v2 / nvmlDeviceGetCount_v2";
            return;
        }
        if (const int rc = init_(); rc != kOk) {
            error_ = "nvmlInit_v2 failed (" + std::to_string(rc) + ")";
            return;
        }
        ready_ = true;
    }

    ~NvmlSource() override {
        if (library_ != nullptr) {
            if (ready_) {
                if (auto shutdown = reinterpret_cast<int (*)()>(sys::symbol(library_, "nvmlShutdown"))) {
                    shutdown();
                }
            }
            sys::free_library(library_);
        }
    }

    std::vector<GpuStat> snapshot() override {
        std::vector<GpuStat> out;
        if (!ready_) { return out; }
        std::lock_guard lock(mutex_);
        unsigned count = 0;
        if (count_(&count) != kOk) { return out; }
        for (unsigned i = 0; i < count; ++i) {
            Device device = nullptr;
            if (by_index_(i, &device) != kOk) { continue; }
            GpuStat stat;
            stat.index    = static_cast<int>(i);
            char text[96] = {};
            if (name_ && name_(device, text, sizeof(text)) == kOk) { stat.name = text; }
            if (uuid_ && uuid_(device, text, sizeof(text)) == kOk) { stat.uuid = text; }
            NvmlPciInfo pci{};
            if (pci_ && pci_(device, &pci) == kOk) {
                stat.bus_id        = pci.bus_id;
                stat.pci_device_id = pci.pci_device_id >> 16; // low half: the vendor
            }
            if (capability_) { capability_(device, &stat.cc_major, &stat.cc_minor); }
            NvmlMemory memory{};
            if (memory_ && memory_(device, &memory) == kOk) {
                stat.memory_total = memory.total;
                stat.memory_used  = memory.used;
                stat.memory_free  = memory.free;
            }
            NvmlUtilization utilization{};
            if (utilization_ && utilization_(device, &utilization) == kOk) {
                stat.util_gpu    = static_cast<int>(utilization.gpu);
                stat.util_memory = static_cast<int>(utilization.memory);
            }
            unsigned value = 0;
            if (temperature_ && temperature_(device, 0, &value) == kOk) {
                stat.temperature_c = static_cast<int>(value);
            }
            if (power_ && power_(device, &value) == kOk) { stat.power_w = value / 1000.0; }
            if (power_limit_ && power_limit_(device, &value) == kOk) {
                stat.power_limit_w = value / 1000.0;
            }
            if (clock_ && clock_(device, kClockGraphics, &value) == kOk) {
                stat.sm_clock_mhz = static_cast<int>(value);
            }
            if (clock_ && clock_(device, kClockMemory, &value) == kOk) {
                stat.memory_clock_mhz = static_cast<int>(value);
            }
            if (max_clock_ && max_clock_(device, kClockMemory, &value) == kOk) {
                stat.memory_clock_max_mhz = static_cast<int>(value);
            }
            // The rated boost clock; the maximum one is the overclocking ceiling (3090 MHz on a
            // 5090).
            if (boost_clock_ && boost_clock_(device, kClockGraphics, &value) == kOk && value > 0) {
                stat.sm_clock_max_mhz = static_cast<int>(value);
            } else if (max_clock_ && max_clock_(device, kClockGraphics, &value) == kOk) {
                stat.sm_clock_max_mhz = static_cast<int>(value);
            }
            if (bus_width_ && bus_width_(device, &value) == kOk) {
                stat.memory_bus_bits = static_cast<int>(value);
            }
            // CUDA cores per SM: 64 on Volta, Turing and GA100, 128 on GA10x, Ada and Blackwell.
            if (cores_ && cores_(device, &value) == kOk && stat.cc_major > 0) {
                const bool wide = stat.cc_major >= 9 || (stat.cc_major == 8 && stat.cc_minor > 0);
                stat.sms        = static_cast<int>(value) / (wide ? 128 : 64);
            }
            if (processes_) {
                unsigned n = 0;
                int rc     = processes_(device, &n, nullptr);
                if (rc == kInsufficientSize || (rc == kOk && n > 0)) {
                    std::vector<NvmlProcessInfo> infos(n + 4);
                    n  = static_cast<unsigned>(infos.size());
                    rc = processes_(device, &n, infos.data());
                    if (rc == kOk) {
                        for (unsigned p = 0; p < n; ++p) {
                            const auto used = infos[p].used_gpu_memory;
                            stat.processes.push_back(
                                {infos[p].pid, used == kUnavailableMemory ? 0 : used});
                        }
                    }
                }
            }
            out.push_back(std::move(stat));
        }
        return out;
    }

    std::string error() const override { return error_; }

private:
    template <typename F>
    void bind(F& slot, const char* symbol) {
        slot = reinterpret_cast<F>(sys::symbol(library_, symbol));
    }

    void* library_ = nullptr;
    bool ready_    = false;
    std::string error_;
    std::mutex mutex_;
    int (*init_)()                                         = nullptr;
    int (*count_)(unsigned*)                               = nullptr;
    int (*by_index_)(unsigned, Device*)                    = nullptr;
    int (*name_)(Device, char*, unsigned)                  = nullptr;
    int (*uuid_)(Device, char*, unsigned)                  = nullptr;
    int (*memory_)(Device, NvmlMemory*)                    = nullptr;
    int (*utilization_)(Device, NvmlUtilization*)          = nullptr;
    int (*temperature_)(Device, unsigned, unsigned*)       = nullptr;
    int (*power_)(Device, unsigned*)                       = nullptr;
    int (*power_limit_)(Device, unsigned*)                 = nullptr;
    int (*clock_)(Device, unsigned, unsigned*)             = nullptr;
    int (*max_clock_)(Device, unsigned, unsigned*)         = nullptr;
    int (*boost_clock_)(Device, unsigned, unsigned*)       = nullptr;
    int (*bus_width_)(Device, unsigned*)                   = nullptr;
    int (*capability_)(Device, int*, int*)                 = nullptr;
    int (*pci_)(Device, NvmlPciInfo*)                      = nullptr;
    int (*cores_)(Device, unsigned*)                       = nullptr;
    int (*processes_)(Device, unsigned*, NvmlProcessInfo*) = nullptr;
};

} // namespace

std::unique_ptr<GpuSource> make_nvml_source() { return std::make_unique<NvmlSource>(); }

} // namespace ninferno::ui
