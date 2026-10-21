#pragma once

// Machine facts: what a board is, from the PCI-ID reference table (core/data/gpu_facts.json,
// compiled in), never what it should run. The canonical continuous facts a B2 decision may read
// (docs/maintainer/deterministic-decisions.md): the reference clock, the memory bus width and data
// rate, TDP, nominal VRAM, the highest PCIe generation; and the per-board discrete facts the table
// knows better than the compute capability (segment, mma_slow: no tensor cores).
//
// machine_facts() is a pure function of the PCI id and the values the driver reported, so a CPU
// test reproduces any device's facts. A known id takes the table's values whatever the driver said
// (overclocks, a WDDM or MIG device that reports no clocks): source Verified. An unknown id takes
// the driver's clocks rounded to a 2^(1/16) grid: source Extrapolated, and none at all when the
// driver reported none (FactSource::None: no roofline, the cost model stands aside as before).
// This header and its source need no CUDA.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno {

enum class FactSource : std::uint8_t {
    None,         // nothing known: unknown id and no driver clocks
    Verified,     // the reference table's row for this PCI id
    Extrapolated, // driver values on the 2^(1/16) grid (unknown id), or a table row that is only
                  // an approximation of this device (a laptop, whose clocks follow the OEM's power
                  // limit; a MIG slice or a board reporting other SMs than the row)
};

enum class Segment : std::uint8_t { Unknown, GeForce, Pro, Datacenter };

enum class FormFactor : std::uint8_t { Unknown, Desktop, Laptop, Server };

[[nodiscard]] std::string_view to_string(FactSource source) noexcept;
[[nodiscard]] std::string_view to_string(Segment segment) noexcept;
[[nodiscard]] std::string_view to_string(FormFactor form) noexcept;

// One SKU of the reference table.
struct GpuFactsRow {
    std::uint32_t pci_device_id = 0;
    std::string name;
    std::string chip; // as gpu_chip spells it: "gb202"
    int major = 0, minor = 0;
    Segment segment = Segment::Unknown;
    FormFactor form = FormFactor::Unknown;
    int sms         = 0;
    int boost_mhz   = 0; // the reference clock (see MachineFacts::reference_clock_khz)
    int bus_bits    = 0;
    double gbps     = 0; // memory data rate per pin
    int tdp_w       = 0;
    double vram_gib = 0;
    int pcie_gen    = 0;     // highest generation the link supports
    bool mma_slow   = false; // mma.sync without tensor cores (TU116, TU117, GTX 16 boards)
    bool target     = true;  // false: listed for identification only (H100, B200)
    std::vector<std::string> sources;
    std::string note;
};

// The reference table, parsed once from the embedded JSON. Throws std::runtime_error when the
// embedded text is malformed (a build defect: the CPU test parses the same text).
[[nodiscard]] std::span<const GpuFactsRow> gpu_facts_table();

// The row of a PCI product id, null when the table has none.
[[nodiscard]] const GpuFactsRow* find_gpu_facts(std::uint32_t pci_device_id);

// Parses a table text (the embedded one, or a test's). Throws std::runtime_error naming the row.
[[nodiscard]] std::vector<GpuFactsRow> parse_gpu_facts(std::string_view json_text);

// What the driver reported about a device: the inputs of machine_facts(). 0 is "not reported".
struct DriverReport {
    std::uint32_t pci_device_id = 0;
    int major = 0, minor = 0;
    int sms              = 0;
    int sm_clock_khz     = 0; // cudaDevAttrClockRate
    int memory_clock_khz = 0; // cudaDevAttrMemoryClockRate: half the data rate
    int memory_bus_bits  = 0; // cudaDevAttrGlobalMemoryBusWidth
};

struct MachineFacts {
    FactSource source = FactSource::None;
    std::string sku; // the table row's name; empty when the id is unknown
    Segment segment = Segment::Unknown;
    FormFactor form = FormFactor::Unknown;
    // The reference clock: the published boost clock of NVIDIA's reference board (the top of the
    // published range on a laptop), which is what the driver reports as the peak SM clock of a
    // stock board. 0: unknown.
    int reference_clock_khz = 0;
    int memory_bus_bits     = 0;
    // Half the memory data rate, as the driver spells the memory clock (28 Gbps GDDR7: 14 000 000).
    int memory_clock_khz = 0;
    int tdp_w            = 0;
    double vram_gib      = 0;
    int pcie_gen         = 0;
    bool mma_slow        = false;
    // The fraction of the board this device is: its SMs over the row's when fewer (a MIG slice);
    // the bandwidth it sees scales with it. 1 otherwise.
    double share = 1.0;

    // Whether the cost model has a roofline: a clock, a bus width and a memory clock.
    [[nodiscard]] bool has_roofline() const noexcept {
        return reference_clock_khz > 0 && memory_clock_khz > 0 && memory_bus_bits > 0;
    }

    [[nodiscard]] double reference_clock_hz() const noexcept { return reference_clock_khz * 1e3; }

    // DRAM bandwidth in bytes per second: two transfers per memory clock across the bus, times
    // share.
    [[nodiscard]] double dram_bytes_per_s() const noexcept {
        return 2.0 * memory_clock_khz * 1e3 * memory_bus_bits / 8.0 * share;
    }

    [[nodiscard]] double dram_gbps() const noexcept { return dram_bytes_per_s() / 1e9; }
};

// x rounded to the nearest point 2^(k/16) (k an integer); 0 for x <= 0. Ties go up.
[[nodiscard]] double snap_to_fact_grid(double x) noexcept;

// The device's facts from its PCI id and what the driver reported (see the header comment).
// `table` defaults to the embedded table.
[[nodiscard]] MachineFacts machine_facts(const DriverReport& driver);
[[nodiscard]] MachineFacts machine_facts(const DriverReport& driver,
                                         std::span<const GpuFactsRow> table);

// "verified RTX 4090 ref 2520 MHz 1008 GB/s" and the like, for logs and the fingerprint.
[[nodiscard]] std::string describe(const MachineFacts& facts);

} // namespace ninferno
