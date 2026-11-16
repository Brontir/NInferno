// The one class builder for a named board (core board_class, #612 step 10 PR-A), no GPU. Each of
// its three product callers gives, for every board of the registry, the class its own builder
// gave before: runtime::plan_device_class (the plan of a named board), the selector golden boards'
// board_device_class (at stock clocks, scaled clocks and a driver's rounding) and the price tool's
// class. The old builders are kept here verbatim as the reference; every DeviceClass field the
// callers read is compared, the decision content key (device_class_key) among them.
#include "core/device_class.h"
#include "core/machine_facts.h"
#include "core/units.h"
#include "ops/generic/price_tool.h"
#include "ops/generic/selector_facts.h"
#include "runtime/engine/plan_dry_run.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// Every field of the class, doubles exactly (%a): the content key and what it leaves out (the
// name, the ordinal, the facts' L2, VRAM and share).
std::string fields(const DeviceClass& d) {
    const MachineFacts& m = d.facts;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "device %d name '%s' facts: source %d sku '%s' ref %d bus %d mem %d tdp %d "
                  "vram %a pcie %d mma_slow %d l2 %a share %a driver_read %d",
                  d.device, d.name.c_str(), static_cast<int>(m.source), m.sku.c_str(),
                  m.reference_clock_khz, m.memory_bus_bits, m.memory_clock_khz, m.tdp_w, m.vram_gib,
                  m.pcie_gen, m.mma_slow ? 1 : 0, m.l2_mib, m.share, m.driver_read ? 1 : 0);
    return device_class_key(d) + " | " + buf;
}

void expect_same(const DeviceClass& now, const DeviceClass& before, const std::string& what) {
    const std::string a = fields(now);
    const std::string b = fields(before);
    expect(a == b, what + "\n  now:    " + a + "\n  before: " + b);
}

// ---- The three builders as they were before board_class (master 248b9e235).

DeviceClass before_plan_device_class(const runtime::PlanDeviceFacts& device) {
    DeviceClass d =
        synthetic_device_class(device.pci_device_id, device.compute_capability / 10,
                               device.compute_capability % 10, device.sms, 0, device.total_bytes);
    d.facts = machine_facts_of(d);
    return d;
}

DeviceClass before_board_device_class(const generic::FactsBoard& b, double sm_scale,
                                      double memory_scale) {
    DeviceClass d      = synthetic_device_class(b.pci_device_id, b.major, b.minor, b.sms,
                                                static_cast<std::size_t>(b.l2_mib * kMiB),
                                                b.memory_gib * 1024 * kMiB);
    d.sm_clock_khz     = static_cast<int>(b.sm_clock_mhz * 1000 * sm_scale);
    d.memory_clock_khz = static_cast<int>(b.memory_clock_mhz * 1000 * memory_scale);
    d.memory_bus_bits  = b.bus_bits;
    d.facts            = machine_facts_of(d);
    return d;
}

DeviceClass before_price_board_class(const GpuFactsRow& row) {
    DeviceClass d      = synthetic_device_class(row.pci_device_id, row.major, row.minor, row.sms,
                                                static_cast<std::size_t>(row.l2_mib * kMiB),
                                                static_cast<std::size_t>(row.vram_gib * 1024 * kMiB));
    d.sm_clock_khz     = row.boost_mhz * 1000;
    d.memory_clock_khz = static_cast<int>(row.gbps * 1e6 / 2.0);
    d.memory_bus_bits  = row.bus_bits;
    d.facts            = machine_facts_of(d);
    return d;
}

// ---- Inputs.

// The facts plan_device_of_board takes from a row (tests/support/named_board.h spells the same).
runtime::PlanDeviceFacts plan_facts_of(const GpuFactsRow& row) {
    runtime::PlanDeviceFacts out;
    out.name               = row.name;
    out.total_bytes        = runtime::plan_board_total_bytes(row.vram_gib);
    out.compute_capability = row.major * 10 + row.minor;
    out.pci_device_id      = row.pci_device_id;
    out.sms                = row.sms;
    return out;
}

// A row as a golden board spells it (selector_facts.h registry_facts_board, from the parsed row).
generic::FactsBoard facts_board_of(const GpuFactsRow& row) {
    return {row.name.c_str(),
            row.pci_device_id,
            row.major,
            row.minor,
            row.sms,
            row.l2_mib,
            static_cast<std::size_t>(row.vram_gib),
            row.boost_mhz,
            static_cast<int>(row.gbps * 500 + 0.5),
            row.bus_bits};
}

constexpr double kScales[][2] = {{1.0, 1.0}, {0.9, 1.0}, {1.0, 0.75}, {1.1, 1.05}};

void every_row_keeps_its_three_classes() {
    for (const GpuFactsRow& row : gpu_facts_table()) {
        const std::string at = row.name + " [" + std::to_string(row.pci_device_id) + "]";
        const runtime::PlanDeviceFacts plan = plan_facts_of(row);
        expect_same(runtime::plan_device_class(plan), before_plan_device_class(plan),
                    at + ": plan_device_class");
        expect(runtime::plan_device_class(plan).l2_bytes == 0,
               at + ": the plan's class still has no L2 (the row's comes in a later PR)");
        generic::FactsBoard board = facts_board_of(row);
        for (const auto& s : kScales) {
            expect_same(generic::board_device_class(board, s[0], s[1]),
                        before_board_device_class(board, s[0], s[1]),
                        at + ": board_device_class x" + std::to_string(s[0]) + " x" +
                            std::to_string(s[1]));
        }
        // A driver's rounding of the memory clock (as test_decision_properties spells it).
        board.memory_clock_mhz += 1;
        expect_same(generic::board_device_class(board, 1.0, 1.0),
                    before_board_device_class(board, 1.0, 1.0),
                    at + ": board_device_class, memory clock rounded up");
        expect_same(tools::price_board_class(row), before_price_board_class(row),
                    at + ": price_board_class");
    }
}

// The golden boards themselves, the three id-less ones (b300, gb10-spark, an unknown sm_120)
// included: their driver values are the FactsBoard's.
void golden_boards_keep_their_class() {
    for (const generic::FactsBoard& b : generic::kFactsBoards) {
        for (const auto& s : kScales) {
            expect_same(generic::board_device_class(b, s[0], s[1]),
                        before_board_device_class(b, s[0], s[1]),
                        std::string(b.name) + ": board_device_class x" + std::to_string(s[0]) +
                            " x" + std::to_string(s[1]));
        }
    }
}

// A golden board of the registry is its row: board_class with the row's own driver view
// (registry_driver_view, the default) is the golden class, content key and all.
void golden_board_is_its_row() {
    for (const generic::FactsBoard& b : generic::kFactsBoards) {
        if (b.pci_device_id == 0) continue;
        const DeviceClass from_row = board_class({.pci_device_id = b.pci_device_id,
                                                  .major         = b.major,
                                                  .minor         = b.minor,
                                                  .sms           = b.sms,
                                                  .global_bytes  = b.memory_gib * 1024 * kMiB});
        expect_same(from_row, generic::board_device_class(b), std::string(b.name) + ": its row");
        expect(from_row.facts.source == FactSource::Verified,
               std::string(b.name) + ": the row's facts are verified");
    }
}

// The spec's two optional fields: an id the registry lacks has no driver view, and an engine's
// features replace the architecture's row but keep the board's own mma_slow.
void spec_defaults() {
    const GpuFactsRow& row = gpu_facts_table().front();
    const DeviceClass unknown =
        board_class({.major = row.major, .minor = row.minor, .sms = row.sms});
    expect(unknown.l2_bytes == 0 && unknown.sm_clock_khz == 0 && unknown.memory_clock_khz == 0 &&
               unknown.memory_bus_bits == 0 && unknown.facts.source == FactSource::None,
           "an id without a row: no driver view, no facts");
    // The row's id under another architecture is not the row (machine_facts' rule).
    const DeviceClass other_arch = board_class({.pci_device_id = row.pci_device_id,
                                                .major         = row.major,
                                                .minor         = row.minor + 1,
                                                .sms           = row.sms});
    expect(other_arch.l2_bytes == 0 && other_arch.facts.sku.empty(),
           "the row's id under another architecture has no driver view");
    const DeviceClass with_row = board_class({.pci_device_id = row.pci_device_id,
                                              .major         = row.major,
                                              .minor         = row.minor,
                                              .sms           = row.sms});
    const DeviceClass engine =
        board_class({.pci_device_id = row.pci_device_id,
                     .major         = row.major,
                     .minor         = row.minor,
                     .sms           = row.sms,
                     .engine        = DeviceFeatures{.mma_bf16 = true, .mma_slow = !row.mma_slow}});
    expect(engine.features.mma_bf16 && !engine.features.mma_tensor,
           "the engine's features replace the architecture's row");
    expect(engine.features.mma_slow == with_row.features.mma_slow,
           "the board's own mma_slow is kept under an engine's features");
    expect(engine.facts.sku == with_row.facts.sku, "the engine does not move the facts");
}

} // namespace

int main() {
    try {
        every_row_keeps_its_three_classes();
        golden_boards_keep_their_class();
        golden_board_is_its_row();
        spec_defaults();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "board_class: every registry board keeps its plan, selector and price class\n";
    return 0;
}
