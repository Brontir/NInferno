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

void the_synthetic_class_keeps_its_value() {
    // Every cc 0..129, listed or not.
    for (int cc = 0; cc < 130; ++cc) {
        const int major = cc / 10, minor = cc % 10;
        const std::string at =
            "synthetic " + std::to_string(major) + "." + std::to_string(minor) + ": ";
        const DeviceClass d =
            synthetic_device_class(0, major, minor, 40, std::size_t{8} << 20, std::size_t{8} << 30);
        legacy::Synthetic old = legacy::synthetic(major, minor);
        // Volta's synthetic class has always read the unlisted row's shared memory and threads
        // (99 and 100 KiB, 1536) for the capabilities of major 7 below 7.5; they stay so (see
        // synthetic_device_class); real Volta is 96 KiB, 96 KiB, 2048.
        expect(d.smem_per_block_optin == old.block, at + "shared per block");
        expect(d.smem_per_sm == old.sm, at + "shared per SM");
        expect(d.max_threads_per_sm == old.threads, at + "threads per SM");
        expect(d.max_blocks_per_sm == old.blocks, at + "blocks per SM");
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
    the_synthetic_class_keeps_its_value();
    the_table_is_well_formed();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "arch registry: ok\n";
    return 0;
}
