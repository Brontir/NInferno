// The architecture table (core/arch_registry.h, the "archs" array of core/data/gpu_facts.json), no
// GPU: the three places that held per-architecture SM limits before it (the build's limits
// ops/common/arch_limits.h, the MMQ tile widths core/mmq_columns.h, the synthetic device class
// core/device_class.cpp) return the same value as before, for every compute capability they covered
// and for one the table lacks. The legacy namespace is the old code, copied verbatim; the
// exceptions are listed where they are tested.
#include "core/arch_registry.h"
#include "core/device_class.h"
#include "core/machine_facts.h"
#include "core/mmq_columns.h"
#include "ops/common/arch_limits.h"

#include <cstddef>
#include <iostream>
#include <string>

namespace {

using namespace ninferno;

// The rows the old code covered, pinned at compile time.
static_assert(arch_registry::find(12, 0).smem_per_block_optin == 99 * 1024);
static_assert(arch_registry::find(12, 0).smem_per_sm == 100 * 1024);
static_assert(arch_registry::find(12, 0).max_threads_per_sm == 1536);
static_assert(arch_registry::find(12, 0).max_blocks_per_sm == 24);
static_assert(arch_registry::find(12, 1).max_blocks_per_sm == 24);
static_assert(arch_registry::find(8, 7).max_blocks_per_sm == 16);
static_assert(arch_registry::find(7, 0).max_threads_per_sm == 2048);
static_assert(arch_registry::find(7, 0).max_blocks_per_sm == 32);
static_assert(arch_registry::find(8, 9).max_blocks_per_sm == 24);
static_assert(arch_registry::find(8, 6).max_blocks_per_sm == 16);
static_assert(arch_registry::find(8, 0).smem_per_block_optin == 163 * 1024);
static_assert(arch_registry::find(8, 0).smem_per_sm == 164 * 1024);
static_assert(arch_registry::find(8, 0).max_threads_per_sm == 2048);
static_assert(arch_registry::find(7, 5).smem_per_block_optin == 64 * 1024);
static_assert(arch_registry::find(7, 5).max_threads_per_sm == 1024);
static_assert(arch_registry::find(7, 5).max_blocks_per_sm == 16);
static_assert(arch_registry::find(7, 0).smem_per_block_optin == 96 * 1024);
static_assert(arch_registry::find(9, 0).smem_per_block_optin == 227 * 1024);
static_assert(arch_registry::find(9, 0).smem_per_sm == 228 * 1024);
static_assert(arch_registry::find(12, 0).registers_per_sm == 64 * 1024);
// An unlisted capability gets row 0, and row 0 is GA10x/Ada/consumer Blackwell's shared memory.
static_assert(!arch_registry::listed(85) && &arch_registry::find(85) == &arch_registry::kRows[0]);
static_assert(arch_registry::kRows[0].smem_per_block_optin == 99 * 1024);
static_assert(arch_registry::shared_reserved_per_block(61) == 0);
static_assert(arch_registry::shared_reserved_per_block(75) == 0);
static_assert(arch_registry::shared_reserved_per_block(80) == 1024);
static_assert(arch_registry::shared_reserved_per_block(130) == 1024);
// The build's own limits are the table's.
static_assert(ops::kMaxSharedBytesPerBlock ==
              arch_registry::find(NINFERNO_BUILD_ARCH).build_smem_per_block_optin);
// The build keeps the old limits above sm_90, below the device's.
static_assert(arch_registry::find(9, 0).build_smem_per_block_optin == 99 * 1024);
static_assert(arch_registry::find(11, 0).build_smem_per_sm == 100 * 1024);
static_assert(arch_registry::find(9, 5).smem_per_block_optin == 227 * 1024);

// The old code, verbatim.
namespace legacy {

// ops/common/arch_limits.h
constexpr std::size_t arch_block(int arch) {
    if (arch == 75) { return 64 * 1024; }
    if (arch >= 70 && arch < 75) { return 96 * 1024; }
    if (arch == 80 || arch == 87) { return 163 * 1024; }
    return 99 * 1024;
}

constexpr std::size_t arch_sm(int arch) {
    if (arch == 75) { return 64 * 1024; }
    if (arch >= 70 && arch < 75) { return 96 * 1024; }
    if (arch == 80 || arch == 87) { return 164 * 1024; }
    return 100 * 1024;
}

constexpr int arch_reserved(int arch) { return arch >= 80 ? 1024 : 0; }

// core/mmq_columns.h
constexpr std::size_t mmq_block(int major, int minor) {
    const int cc = major * 10 + minor;
    if (cc == 70 || cc == 72) { return 96 * 1024; }
    if (cc == 75) { return 64 * 1024; }
    if (cc == 80 || cc == 87) { return 163 * 1024; }
    if (cc >= 90 && cc < 120) { return 227 * 1024; }
    return 99 * 1024;
}

// core/device_class.cpp, synthetic_device_class
struct Synthetic {
    std::size_t block, sm;
    int threads, blocks, registers;
};

constexpr Synthetic synthetic(int major, int minor) {
    const int cc = major * 10 + minor;
    return {cc == 75               ? std::size_t{64 * 1024}
            : cc == 80 || cc == 87 ? std::size_t{163 * 1024}
            : cc >= 90 && cc < 120 ? std::size_t{227 * 1024}
                                   : std::size_t{99 * 1024},
            cc == 75               ? std::size_t{64 * 1024}
            : cc == 80 || cc == 87 ? std::size_t{164 * 1024}
            : cc >= 90 && cc < 120 ? std::size_t{228 * 1024}
                                   : std::size_t{100 * 1024},
            cc == 75                             ? 1024
            : cc == 80 || (cc >= 90 && cc < 120) ? 2048
                                                 : 1536,
            cc == 75   ? 16
            : cc == 86 ? 16
            : cc == 89 ? 24
                       : 32,
            64 * 1024};
}

} // namespace legacy

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// The listed capabilities, and what the old code did for each in the three places.
void every_covered_capability_keeps_its_value() {
    // cc up to 129, every minor digit: the capabilities that exist and the ones that do not.
    for (int cc = 0; cc < 130; ++cc) {
        const int major = cc / 10, minor = cc % 10;
        const std::string at = "cc " + std::to_string(cc) + ": ";
        const auto& row      = arch_registry::find(cc);

        // arch_limits.h: every capability, sm_71/73/74 and sm_90..119 included (the build keeps the
        // 99/100 KiB it had for the latter, below the device's 227/228 KiB, until a measured
        // change).
        expect(ops::shared_bytes_per_block_for_arch(cc) == legacy::arch_block(cc),
               at + "arch_limits per block");
        expect(ops::shared_bytes_per_sm_for_arch(cc) == legacy::arch_sm(cc),
               at + "arch_limits per SM");
        expect(ops::shared_bytes_reserved_per_block_for_arch(cc) == legacy::arch_reserved(cc),
               at + "arch_limits reserved per block");

        expect(mmq_columns::shared_per_block(major, minor) == legacy::mmq_block(major, minor),
               at + "mmq_columns shared per block");
        expect(arch_registry::listed(cc) ? (row.cc <= cc && cc <= row.cc_to) : row.cc == 0,
               at + "row or the unlisted row");
    }
}

// The CUDA programming guide's limits per compute capability (Compute Capabilities: Table 33,
// resident threads and blocks per SM; Table 34, shared memory per SM and per block, opt-in). Where
// the corrections of #662 differ from what the engine carried before: 7.0 and 7.2 (2048 threads,
// 96 KiB; the current guide no longer lists 7.0, earlier editions give these), 8.7 (16 blocks),
// 12.x (24 blocks). 7.2 and 12.1 are the same rows as 7.0 and 12.0 in the guide's columns.
struct GuideRow {
    int cc, threads, blocks, sm_kib, block_kib;
};
constexpr GuideRow kGuide[] = {
    {70, 2048, 32, 96, 96},  {72, 2048, 32, 96, 96},  {75, 1024, 16, 64, 64},
    {80, 2048, 32, 164, 163}, {86, 1536, 16, 100, 99}, {87, 1536, 16, 164, 163},
    {89, 1536, 24, 100, 99},  {90, 2048, 32, 228, 227}, {120, 1536, 24, 100, 99},
    {121, 1536, 24, 100, 99},
};

void the_synthetic_class_matches_the_guide() {
    // Every cc 0..129, listed or not.
    for (int cc = 0; cc < 130; ++cc) {
        const int major = cc / 10, minor = cc % 10;
        const std::string at =
            "synthetic " + std::to_string(major) + "." + std::to_string(minor) + ": ";
        const DeviceClass d =
            synthetic_device_class(0, major, minor, 40, std::size_t{8} << 20, std::size_t{8} << 30);
        legacy::Synthetic old = legacy::synthetic(major, minor);
        // The capabilities of the CUDA programming guide's table (kGuide) must equal it; the
        // others (unlisted ones, sm_90..119 beyond 9.0) keep what the legacy class (verbatim above)
        // gave.
        const GuideRow* g = nullptr;
        for (const GuideRow& row : kGuide) {
            if (row.cc == cc) g = &row;
        }
        const std::size_t block = g ? std::size_t{g->block_kib} * 1024 : old.block;
        const std::size_t sm    = g ? std::size_t{g->sm_kib} * 1024 : old.sm;
        const int threads       = g ? g->threads : old.threads;
        const int blocks        = g ? g->blocks : old.blocks;
        expect(d.smem_per_block_optin == block, at + "shared per block");
        expect(d.smem_per_sm == sm, at + "shared per SM");
        expect(d.max_threads_per_sm == threads, at + "threads per SM");
        expect(d.max_blocks_per_sm == blocks, at + "blocks per SM");
        expect(d.registers_per_sm == old.registers, at + "registers per SM");
    }
}

// The table: row 0 first, capabilities once, shared memory of an SM holds a block's.
void the_table_is_well_formed() {
    expect(arch_registry::kRows[0].cc == 0, "row 0 is the unlisted capability");
    for (std::size_t i = 0; i < std::size(arch_registry::kRows); ++i) {
        const auto& a = arch_registry::kRows[i];
        expect(a.smem_per_sm >= a.smem_per_block_optin,
               "row " + std::to_string(a.cc) + ": SM holds a block");
        expect(a.max_threads_per_sm % 32 == 0 && a.max_threads_per_sm > 0,
               "row " + std::to_string(a.cc) + ": threads");
        for (std::size_t j = i + 1; j < std::size(arch_registry::kRows); ++j) {
            expect(arch_registry::kRows[j].cc != a.cc,
                   "capability " + std::to_string(a.cc) + " once");
        }
    }
    // Every board of the PCI table has a listed capability.
    for (const GpuFactsRow& g : gpu_facts_table()) {
        expect(arch_registry::listed(g.major * 10 + g.minor),
               "board " + g.name + ": capability listed");
    }
}

} // namespace

int main() {
    every_covered_capability_keeps_its_value();
    the_synthetic_class_matches_the_guide();
    the_table_is_well_formed();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "arch registry: ok\n";
    return 0;
}
