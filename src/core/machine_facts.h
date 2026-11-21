#pragma once

// Machine facts: what a board is, from the PCI-ID reference table (core/data/gpu_facts.json,
// compiled in), never what it should run. The canonical continuous facts a B2 decision may read
// (docs/maintainer/deterministic-decisions.md): the reference clock, the memory bus width and data
// rate, TDP, nominal VRAM, the highest PCIe generation; and the per-board discrete facts the table
// knows better than the compute capability (segment, mma_slow: no tensor cores).
//
// machine_facts() is a pure function of the PCI id and the values the driver reported, so a CPU
// test reproduces any device's facts. A known id takes the table's values whatever the driver said
// (overclocks, a WDDM or MIG device that reports no clocks): source Verified. A laptop row is
// priced at the bottom of its published boost range (principle 7: the unknown is conservative),
// or, where the table lacks it, at the lower of the row's top and the driver's clock on the grid:
// source Extrapolated. An unknown id takes
// the driver's clocks rounded to a 2^(1/16) grid: source Extrapolated, and none at all when the
// driver reported none (FactSource::None: no roofline, the cost model stands aside as before).
// This header and its source need no CUDA.

#include "core/board_facts.h"

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
// A board sold under a row's PCI product id that its PCI subsystem id names (#903: the TITAN V CEO
// Edition, subsystem 10DE:12EB, under the Quadro GV100's 0x1DBA, same SMs and memory, a lower boost
// clock). Its values are the row's except where gpu_facts.json gives its own.
struct GpuFactsSubsystem {
    std::uint32_t id = 0; // subsystem vendor << 16 | subsystem device: 0x10DE12EB
    std::string name;
    int boost_mhz   = 0;
    int bus_bits    = 0;
    double gbps     = 0;
    int tdp_w       = 0;
    double vram_gib = 0;
    std::vector<std::string> sources;
};

struct GpuFactsRow {
    std::uint32_t pci_device_id = 0;
    std::string name;
    std::string chip; // as gpu_chip spells it: "gb202"
    int major = 0, minor = 0;
    Segment segment = Segment::Unknown;
    FormFactor form = FormFactor::Unknown;
    int sms         = 0;
    int boost_mhz   = 0; // the reference clock (see MachineFacts::reference_clock_khz)
    // A laptop's bottom of the published boost range (0: not given): the clock it is priced at.
    int boost_min_mhz = 0;
    int bus_bits    = 0;
    double gbps     = 0; // memory data rate per pin
    int tdp_w       = 0;
    double vram_gib = 0;
    int pcie_gen    = 0;     // highest generation the link supports
    bool mma_slow   = false; // mma.sync without tensor cores (TU116, TU117, GTX 16 boards)
    double l2_mib   = 0;     // the L2 cache (0: not in the row)
    bool target     = true;  // false: listed for identification only (H100, B200)
    std::vector<std::string> sources;
    std::string note;
    // The other boards of the id that the subsystem id tells apart (machine_facts reads them).
    std::vector<GpuFactsSubsystem> subsystems;
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
    // The PCI subsystem, vendor << 16 | device (0x10DE12EB). 0 where none was asked: a board named
    // by its row (a plan's, a test's) is the row's own board.
    std::uint32_t pci_subsystem_id = 0;
    // The driver was asked for the subsystem and could not say: on an id whose row lists boards by
    // subsystem, which of them this is stays unknown (machine_facts prices the slowest).
    bool subsystem_unread = false;
    // The device's memory (cudaDevAttr / totalGlobalMem; 0: not reported). A MIG GPU instance
    // reports its own memory slices, which carry the memory controllers and so its share of the
    // board's bandwidth (#906); a compute instance reports its GPU instance's.
    std::uint64_t global_bytes = 0;
};

struct MachineFacts {
    FactSource source = FactSource::None;
    std::string sku; // the table row's name; empty when the id is unknown
    Segment segment = Segment::Unknown;
    FormFactor form = FormFactor::Unknown;
    // The reference clock: the published boost clock of NVIDIA's reference board, which is what
    // the driver reports as the peak SM clock of a stock board; on a laptop the bottom of the
    // published range (machine_facts). 0: unknown.
    int reference_clock_khz = 0;
    int memory_bus_bits     = 0;
    // Half the memory data rate, as the driver spells the memory clock (28 Gbps GDDR7: 14 000 000).
    int memory_clock_khz = 0;
    int tdp_w            = 0;
    double vram_gib      = 0;
    int pcie_gen         = 0;
    bool mma_slow        = false;
    // The board's L2 cache, MiB (0: its row does not give it).
    double l2_mib = 0;
    // The fraction of the board this device is: its SMs over the row's when fewer (a MIG slice).
    // 1 otherwise.
    double share = 1.0;
    // The fraction of the board's memory bandwidth it sees: a slice's reported memory over the
    // row's (a MIG GPU instance owns whole memory slices, each with its controllers: the published
    // profiles pair 1/7 of the SMs with 1/8 or 2/8 of the memory, #906), never above 1; where the
    // slice reports no memory, its SM share. 1 for a whole board. The reported memory is a little
    // under the profile's nominal size (the driver's reserve), so the bandwidth errs low.
    double memory_share = 1.0;
    // Whether a driver clock or bus width fed a fact: an unknown id (its clocks on the grid), a
    // laptop row without the bottom of its range (its clock bounded by the driver's), or a whole
    // board on another memory than its row's (its bus width or slower memory clock). A slice's
    // SM count feeds `share`, which the decision fingerprint always hashes, so it does not set
    // this. The fingerprint hashes the driver's values only when this is set.
    bool driver_read = false;
    // The row lists other boards under the id by subsystem and the driver could not say which
    // this is: the facts are the slowest of them (extrapolated, #903).
    bool subsystem_unknown = false;

    // Whether the cost model has a roofline: a clock, a bus width and a memory clock.
    [[nodiscard]] bool has_roofline() const noexcept {
        return reference_clock_khz > 0 && memory_clock_khz > 0 && memory_bus_bits > 0;
    }

    [[nodiscard]] double reference_clock_hz() const noexcept { return reference_clock_khz * 1e3; }

    // DRAM bandwidth in bytes per second: two transfers per memory clock across the bus, times
    // memory_share.
    [[nodiscard]] double dram_bytes_per_s() const noexcept {
        return board_facts::dram_bytes_per_s(memory_clock_khz, memory_bus_bits, memory_share);
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
