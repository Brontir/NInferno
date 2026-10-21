// The PCI-ID reference table and machine_facts() (core/machine_facts.h), no GPU: the embedded
// table is well formed and sourced, agrees with the chip table, and the facts a decision reads are
// a pure function of the id and the driver's values. For the lab cards the facts are compared with
// the driver values the cost model read before (the CUDA query of each stand, as the selector
// golden test records them): a change there moves routes, so the differences are printed and
// bounded.
#include "core/device_class.h"
#include "core/machine_facts.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninferno;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

bool same(const MachineFacts& a, const MachineFacts& b) {
    return a.source == b.source && a.sku == b.sku && a.segment == b.segment && a.form == b.form &&
           a.reference_clock_khz == b.reference_clock_khz &&
           a.memory_bus_bits == b.memory_bus_bits && a.memory_clock_khz == b.memory_clock_khz &&
           a.tdp_w == b.tdp_w && a.vram_gib == b.vram_gib && a.pcie_gen == b.pcie_gen &&
           a.mma_slow == b.mma_slow && a.share == b.share;
}

void table_well_formed() {
    const auto table = gpu_facts_table();
    expect(table.size() >= 40,
           "the table has at least 40 rows (" + std::to_string(table.size()) + ")");
    for (const GpuFactsRow& r : table) {
        char id[16];
        std::snprintf(id, sizeof(id), "0x%04X", r.pci_device_id);
        const std::string where = std::string(id) + " " + r.name;
        bool pci_ids            = false;
        for (const auto& s : r.sources) pci_ids |= s.rfind("https://pci-ids.ucw.cz/", 0) == 0;
        expect(r.sources.size() >= 2 && pci_ids,
               where + ": a spec source and the pci-ids entry of its id");
        expect(find_gpu_facts(r.pci_device_id) == &r, where + ": found by its id");
        // The chip names the same chip gpu_chip finds by the id's block, where it knows the chip.
        const std::string chip = gpu_chip(r.pci_device_id, r.major, r.minor, r.sms);
        if (chip != gpu_arch_id(r.major, r.minor)) {
            expect(chip == r.chip, where + ": chip " + r.chip + " where gpu_chip says " + chip);
        }
        // mma_slow: the chips without tensor cores (TU116, TU117) and the GTX 16 boards of TU106,
        // the same fact chip_mma_slow (DeviceFeatures::mma_slow) keys by chip.
        const bool gtx_on_tu106 = r.chip == "tu106" && r.name.find("GTX") != std::string::npos;
        const bool slow_chip    = r.chip == "tu116" || r.chip == "tu117";
        expect(r.mma_slow == (slow_chip || gtx_on_tu106), where + ": mma_slow");
        // A laptop row carries the bottom of its published boost range (the clock it is priced
        // at); no other row does (the parser refuses one).
        expect((r.form == FormFactor::Laptop) == (r.boost_min_mhz > 0),
               where + ": boost_min_mhz on laptops only, and on every laptop");
        // Bandwidth plausibility: 64..9000 GB/s.
        const double gbs = r.gbps * r.bus_bits / 8.0;
        expect(gbs >= 64 && gbs <= 9000, where + ": bandwidth " + std::to_string(gbs));
    }
}

// The driver values of the lab cards (tests/ops/linear/test_selector_golden.cpp: boost SM clock,
// memory clock = data rate / 2, bus width; the RTX 4070 Ti and 3060, not golden boards, spelled the
// same way from their datasheets) against the table's facts.
struct LabCard {
    const char* name;
    std::uint32_t id;
    int major, minor, sms;
    int sm_clock_mhz, memory_clock_mhz, bus_bits;
};

constexpr LabCard kLab[] = {
    {"rtx5090", 0x2B85, 12, 0, 170, 2407, 14001, 512},
    {"rtx5080", 0x2C02, 12, 0, 84, 2617, 15001, 256},
    {"rtx5070ti", 0x2C05, 12, 0, 70, 2452, 14001, 256},
    {"rtx5060ti", 0x2D04, 12, 0, 36, 2572, 14001, 128},
    {"rtx4090", 0x2684, 8, 9, 128, 2520, 10501, 384},
    {"rtx4070ti", 0x2782, 8, 9, 60, 2610, 10501, 192},
    {"rtx4060ti", 0x2805, 8, 9, 34, 2535, 9001, 128},
    {"l4", 0x27B8, 8, 9, 58, 2040, 6251, 192},
    {"rtx3090", 0x2204, 8, 6, 82, 1695, 9751, 384},
    {"rtx3060", 0x2504, 8, 6, 28, 1777, 7501, 192},
    {"a10", 0x2236, 8, 6, 72, 1695, 6251, 384},
    {"a100", 0x20B0, 8, 0, 108, 1410, 1215, 5120},
    {"rtx2080ti", 0x1E07, 7, 5, 68, 1545, 7000, 352},
    {"v100", 0x1DB5, 7, 0, 80, 1530, 877, 4096},
    {"h100-sxm", 0x2330, 9, 0, 132, 1980, 2619, 5120},
};

void lab_cards_old_vs_new() {
    std::printf("%-10s %-24s %9s %9s %7s %9s %9s %7s\n", "card", "facts", "clk old", "clk new",
                "d clk", "GB/s old", "GB/s new", "d bw");
    for (const LabCard& c : kLab) {
        const DriverReport driver{
            c.id,      c.major, c.minor, c.sms, c.sm_clock_mhz * 1000, c.memory_clock_mhz * 1000,
            c.bus_bits};
        const MachineFacts f = machine_facts(driver);
        const double clk_old = c.sm_clock_mhz * 1e6;
        const double bw_old  = 2.0 * c.memory_clock_mhz * 1e6 * c.bus_bits / 8.0;
        const double d_clk   = f.reference_clock_hz() / clk_old - 1.0;
        const double d_bw    = f.dram_bytes_per_s() / bw_old - 1.0;
        std::printf("%-10s %-24s %9.0f %9.0f %+6.3f%% %9.1f %9.1f %+6.3f%%\n", c.name,
                    std::string(to_string(f.source)).c_str(), clk_old / 1e6,
                    f.reference_clock_hz() / 1e6, 100 * d_clk, bw_old / 1e9, f.dram_gbps(),
                    100 * d_bw);
        expect(f.source == FactSource::Verified, std::string(c.name) + ": verified");
        expect(std::abs(d_clk) < 1e-9,
               std::string(c.name) + ": the reference clock is the driver's");
        expect(std::abs(d_bw) < 2e-4, std::string(c.name) + ": bandwidth within 0.02 %");
    }
}

void deterministic_and_driver_blind() {
    // A known id takes the table's values whatever the driver reports: an overclocked board, a
    // driver that rounds, a MIG or WSL device that reports no clocks.
    const DriverReport stock{0x2684, 8, 9, 128, 2520000, 10501000, 384};
    const MachineFacts a = machine_facts(stock);
    expect(same(a, machine_facts(stock)), "the same report, the same facts");
    for (const DriverReport other :
         {DriverReport{0x2684, 8, 9, 128, 2610000, 10751000, 384},
          DriverReport{0x2684, 8, 9, 128, 0, 0, 0}, DriverReport{0x2684, 0, 0, 0, 0, 0, 0}}) {
        expect(same(a, machine_facts(other)), "the 4090's facts do not follow the driver's clocks");
    }
    expect(a.sku == "GeForce RTX 4090" && a.segment == Segment::GeForce && a.tdp_w == 450 &&
               a.vram_gib == 24 && a.pcie_gen == 4 && !a.mma_slow,
           "the 4090's discrete facts");
    expect(machine_facts(DriverReport{0x2182, 7, 5, 24, 0, 0, 0}).mma_slow, "TU116: mma_slow");
    expect(machine_facts(DriverReport{0x1F82, 7, 5, 14, 0, 0, 0}).mma_slow, "TU117: mma_slow");
    expect(!machine_facts(DriverReport{0x1E07, 7, 5, 68, 0, 0, 0}).mma_slow, "TU102: tensor cores");
    // A laptop's row is an approximation: its clocks follow the OEM's power limit.
    expect(machine_facts(DriverReport{0x2717, 8, 9, 76, 1755000, 10001000, 256}).source ==
               FactSource::Extrapolated,
           "a laptop is extrapolated");
    // An id the table has under another architecture is unknown (Jetson Thor's 0x2B00 sits in
    // GB202's block; here GB202's own id reported as sm_110).
    const MachineFacts reused =
        machine_facts(DriverReport{0x2B85, 11, 0, 20, 1575000, 4266000, 256});
    expect(reused.sku.empty() && reused.source == FactSource::Extrapolated,
           "an id reported by another architecture is not the table's row");
}

// Principle 7: a laptop is priced at the bottom of its published boost range (an RTX 4090 Laptop
// GPU runs 80 to 150 W, 1455 to 2040 MHz), whatever its driver reports; a laptop row without the
// bottom takes the lower of its top and the driver's clock on the grid.
void laptops_conservative() {
    for (const DriverReport report : {DriverReport{0x2717, 8, 9, 76, 2040000, 9001000, 256},
                                      DriverReport{0x2717, 8, 9, 76, 1755000, 9001000, 256},
                                      DriverReport{0x2717, 8, 9, 76, 0, 0, 0}}) {
        const MachineFacts f = machine_facts(report);
        expect(f.source == FactSource::Extrapolated && f.form == FormFactor::Laptop &&
                   f.reference_clock_khz == 1455000 && f.memory_clock_khz == 9000000,
               "RTX 4090 Laptop GPU: the bottom of 1455-2040 MHz, whatever the driver says");
    }
    expect(
        machine_facts(DriverReport{0x2520, 8, 6, 30, 1703000, 7001000, 192}).reference_clock_khz ==
            1283000,
        "RTX 3060 Laptop GPU: 1283 MHz, the bottom of 1283-1703");
    const std::string row =
        R"({"schema":1,"gpus":[{"id":"0x2717","name":"x","chip":"ad103","cc":89,)"
        R"("segment":"geforce","form":"laptop","sms":76,"boost_mhz":2040,"bus_bits":256,)"
        R"("gbps":18.0,"tdp_w":150,"vram_gib":16,"pcie":4,"sources":["https://example.org/"]}]})";
    const std::vector<GpuFactsRow> table = parse_gpu_facts(row);
    const auto clock                     = [&](int driver_khz) {
        return machine_facts(DriverReport{0x2717, 8, 9, 76, driver_khz, 9001000, 256}, table)
            .reference_clock_khz;
    };
    const int low = static_cast<int>(std::llround(snap_to_fact_grid(1500000)));
    expect(clock(1500000) == low && low < 2040000,
           "no bottom given: a lower driver clock, on the grid");
    expect(clock(2400000) == 2040000, "no bottom given: never above the row's top");
    expect(clock(0) == 2040000, "no bottom given, no driver clock: the row's top");
    std::printf(
        "laptop 4090: %s\n",
        describe(machine_facts(DriverReport{0x2717, 8, 9, 76, 2040000, 9001000, 256})).c_str());
}

void grid() {
    expect(snap_to_fact_grid(0) == 0 && snap_to_fact_grid(-5) == 0, "the grid of 0 is 0");
    for (int k = -32; k <= 400; ++k) {
        const double p = std::exp2(k / 16.0);
        expect(std::abs(snap_to_fact_grid(p) / p - 1) < 1e-12, "a grid point is its own snap");
        expect(std::abs(snap_to_fact_grid(p * 1.02) / p - 1) < 1e-12, "+2 % snaps down");
        expect(std::abs(snap_to_fact_grid(p * 0.98) / p - 1) < 1e-12, "-2 % snaps up");
    }
    // The integers either side of each midpoint 2^((k+1/2)/16), up to 2^31, snap to the grid
    // points either side (the margin snap_to_fact_grid's comment states holds on this libm).
    for (int k = 0; k < 496; ++k) {
        const double mid = std::exp2((k + 0.5) / 16.0);
        const double lo = std::floor(mid), hi = std::ceil(mid);
        // Only where the integer stays within the half-step on its side.
        if (lo >= std::exp2((k - 0.5) / 16.0)) {
            expect(snap_to_fact_grid(lo) == std::exp2(k / 16.0), "below a midpoint snaps down");
        }
        if (hi < std::exp2((k + 1.5) / 16.0)) {
            expect(snap_to_fact_grid(hi) == std::exp2((k + 1) / 16.0), "above a midpoint snaps up");
        }
    }
    double last = 0;
    for (double x = 1000; x < 4e6; x *= 1.003) {
        const double s = snap_to_fact_grid(x);
        expect(s >= last, "the grid is monotonic");
        expect(std::abs(std::log2(s / x)) <= 1.0 / 32 + 1e-12, "within half a step");
        last = s;
    }
    // An unknown id: the driver's clocks on the grid, so stock and factory-overclocked boards of
    // one unlisted kind land on the same point; the bus width is kept.
    const MachineFacts u =
        machine_facts(DriverReport{0, 12, 0, 48, 2500000, 14001000, 192}, gpu_facts_table());
    const MachineFacts oc = machine_facts(DriverReport{0, 12, 0, 48, 2530000, 14001000, 192});
    expect(u.source == FactSource::Extrapolated && u.sku.empty(), "unknown: extrapolated");
    expect(u.reference_clock_khz == oc.reference_clock_khz,
           "a 1.2 % overclock snaps to the same clock");
    expect(u.memory_bus_bits == 192, "the bus width is kept");
    expect(u.reference_clock_khz == static_cast<int>(std::llround(std::exp2(
                                        std::floor(16 * std::log2(2500000.0) + 0.5) / 16))),
           "the clock is the grid point");
    std::printf("unknown sm120 48 SM, driver 2500 MHz / 14001 MHz: %s\n", describe(u).c_str());
}

void mig_and_zero_clocks() {
    // A MIG slice of an A100 (14 of 108 SMs) whose driver reports no clocks: the board's facts,
    // labelled extrapolated, with its share of the bandwidth.
    const MachineFacts slice = machine_facts(DriverReport{0x20B0, 8, 0, 14, 0, 0, 0});
    expect(slice.source == FactSource::Extrapolated, "MIG slice: extrapolated");
    expect(slice.has_roofline() && slice.reference_clock_khz == 1410000,
           "MIG slice: the table's clock");
    expect(std::abs(slice.share - 14.0 / 108) < 1e-12, "MIG slice: its share of the bandwidth");
    expect(std::abs(slice.dram_gbps() - 1555.2 * 14 / 108) < 1e-6, "MIG slice: bandwidth");
    // The whole A100 with zero clocks (a WSL driver): verified table facts.
    const MachineFacts whole = machine_facts(DriverReport{0x20B0, 8, 0, 108, 0, 0, 0});
    expect(whole.source == FactSource::Verified && std::abs(whole.dram_gbps() - 1555.2) < 1e-6,
           "zero clocks on a known id: the table");
    // Unknown id, zero clocks: nothing known, no roofline (the cost model stands aside).
    const MachineFacts none = machine_facts(DriverReport{0, 8, 0, 14, 0, 0, 0});
    expect(none.source == FactSource::None && !none.has_roofline() && none.dram_gbps() == 0,
           "unknown id, zero clocks: none");
    // Clock but no memory clock: extrapolated, still no roofline.
    const MachineFacts half = machine_facts(DriverReport{0, 8, 9, 30, 2000000, 0, 128});
    expect(half.source == FactSource::Extrapolated && !half.has_roofline(),
           "half known: no roofline");
}

void device_class_wiring() {
    DeviceClass d =
        synthetic_device_class(0x2B85, 12, 0, 170, std::size_t{96} << 20, std::size_t{32} << 30);
    expect(d.facts.source == FactSource::None && d.dram_gbps() == 0,
           "a synthetic class has no facts until asked");
    d.facts = machine_facts_of(d);
    expect(d.facts.source == FactSource::Verified && std::abs(d.dram_gbps() - 1792) < 1e-9,
           "machine_facts_of: the 5090's row");
    expect(board_of(d).dram_gbps == d.dram_gbps(), "the board distance reads the facts");
}

void malformed_tables_refused() {
    const auto refused = [](std::string_view text, const char* what) {
        try {
            (void)parse_gpu_facts(text);
        } catch (const std::exception&) { return; }
        expect(false, std::string("refused: ") + what);
    };
    const std::string row =
        R"({"id":"0x2684","name":"x","chip":"ad102","cc":89,"segment":"geforce","form":"desktop",)"
        R"("sms":128,"boost_mhz":2520,"bus_bits":384,"gbps":21.0,"tdp_w":450,"vram_gib":24,"pcie":4,)";
    const std::string src = R"("sources":["https://example.org/"]})";
    expect(parse_gpu_facts(R"({"schema":1,"gpus":[)" + row + src + "]}").size() == 1, "one row");
    refused(R"({"schema":1,"gpus":[)" + row + R"("sources":[]}]})", "a row without a source");
    refused(R"({"schema":1,"gpus":[)" + row + R"("sources":["http://x"]}]})",
            "a plain-http source");
    refused(R"({"schema":1,"gpus":[)" + row + src + "," + row + src + "]}", "a repeated id");
    refused(R"({"schema":2,"gpus":[]})", "another schema");
    std::string desktop_min = row;
    desktop_min.replace(desktop_min.find("\"bus_bits\""), 0, "\"boost_min_mhz\":1500,");
    refused(R"({"schema":1,"gpus":[)" + desktop_min + src + "]}", "boost_min_mhz on a desktop");
    std::string laptop_min = desktop_min;
    laptop_min.replace(laptop_min.find("desktop"), 7, "laptop");
    expect(parse_gpu_facts(R"({"schema":1,"gpus":[)" + laptop_min + src + "]}")[0].boost_min_mhz ==
               1500,
           "boost_min_mhz on a laptop");
    laptop_min.replace(laptop_min.find("1500"), 4, "2600");
    refused(R"({"schema":1,"gpus":[)" + laptop_min + src + "]}", "boost_min_mhz above boost_mhz");
    std::string bad = row;
    bad.replace(bad.find("\"pcie\":4"), 8, "\"pcie\":0");
    refused(R"({"schema":1,"gpus":[)" + bad + src + "]}", "PCIe generation 0");
}

} // namespace

int main() {
    try {
        table_well_formed();
        lab_cards_old_vs_new();
        deterministic_and_driver_blind();
        laptops_conservative();
        grid();
        mig_and_zero_clocks();
        device_class_wiring();
        malformed_tables_refused();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: threw " << e.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "PASS" : "FAILED") << " (" << gpu_facts_table().size()
              << " rows)\n";
    return failures == 0 ? 0 : 1;
}
