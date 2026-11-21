#pragma once

// The GPUs as ninferno-ui shows them: read through NVML (libnvidia-ml.so.1, opened at run time like
// core/device_class.cpp does), so the UI itself needs no CUDA context and holds no device memory.
// Indices are NVML's, which follow the PCI bus as --device N counts; a process started on GPU i
// gets CUDA_VISIBLE_DEVICES=<its UUID> (or, without one, CUDA_DEVICE_ORDER=PCI_BUS_ID and
// CUDA_VISIBLE_DEVICES=i), so both name the same card.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ninferno::ui {

struct GpuProcess {
    std::uint32_t pid        = 0;
    std::uint64_t used_bytes = 0; // 0 when the driver does not say (WSL)
    // The NVIDIA MPS server (nvidia-cuda-mps-server, by NVML's process name): an exclusive-process
    // card it holds still takes MPS clients, as in the default mode.
    bool mps_server = false;
};

struct GpuStat {
    int index = 0;
    std::string name;
    std::string uuid;
    std::string bus_id;
    int cc_major = 0, cc_minor = 0;
    std::uint64_t memory_total = 0, memory_used = 0, memory_free = 0;
    int util_gpu = -1, util_memory = -1; // percent, -1 when unknown
    int temperature_c = -1;
    double power_w = -1, power_limit_w = -1;
    int sm_clock_mhz = -1, memory_clock_mhz = -1, memory_clock_max_mhz = -1;
    int sm_clock_max_mhz        = -1; // the rated boost clock the tensor throughput estimate uses
    int memory_bus_bits = -1;
    std::uint32_t pci_device_id = 0; // PCI product id (the chip table's key), 0 when unknown
    int sms                     = 0; // from NVML's CUDA core count, 0 when unknown
    std::vector<GpuProcess> processes;
    // NVML's compute mode: 0 default, 1 exclusive thread, 2 prohibited, 3 exclusive process; -1
    // when it does not say.
    int compute_mode = -1;

    // NVML's word that this card takes no new CUDA process now: a prohibited compute mode, or an
    // exclusive one that a running process holds and no MPS server serves (#891). Empty when it
    // takes one or NVML does not say. NVML still reports such a card's free memory, which alone
    // reads as a fit.
    [[nodiscard]] std::string refusal_now() const;

    // Peak memory bandwidth from the bus width and the maximum memory clock (GDDR: two transfers
    // per clock as NVML reports it), 0 when either is unknown.
    [[nodiscard]] double bandwidth_gbs() const noexcept;
    [[nodiscard]] nlohmann::json to_json() const;
};

class GpuSource {
public:
    virtual ~GpuSource() = default;
    // Empty when there is no driver or no GPU; `error` says why.
    [[nodiscard]] virtual std::vector<GpuStat> snapshot() = 0;
    [[nodiscard]] virtual std::string error() const       = 0;
};

// NVML, initialised once and kept for the process's life.
[[nodiscard]] std::unique_ptr<GpuSource> make_nvml_source();

} // namespace ninferno::ui
