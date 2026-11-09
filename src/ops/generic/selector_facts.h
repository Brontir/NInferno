#pragma once

// The compiled-kernel facts the selector's register-MMA price reads, as a digest (#556), and the
// boards the selector goldens are kept for. One implementation for the selector golden test and
// for the release (`ninferno-serve selector-facts`, tools/release/package.sh), so the digest the
// goldens are keyed by and the digest a release build computes of its own engine cannot drift.

#include "core/device_class.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace ninferno::ops::detail::generic {

struct RmmaFactRead; // generic_gemm.h: one read the price made (ScopedRmmaFactsSink)

// Datasheet numbers as the CUDA device query reports them (boost SM clock, memory clock = data
// rate / 2, bus width).
struct FactsBoard {
    const char* name;
    std::uint32_t pci_device_id;
    int major, minor, sms;
    std::size_t l2_mib, memory_gib;
    int sm_clock_mhz, memory_clock_mhz, bus_bits;
};

inline constexpr FactsBoard kFactsBoards[] = {
    {"rtx5090", 0x2B85, 12, 0, 170, 96, 32, 2407, 14001, 512},
    {"rtx5080", 0x2C02, 12, 0, 84, 64, 16, 2617, 15001, 256},
    {"rtx5070ti", 0x2C05, 12, 0, 70, 48, 16, 2452, 14001, 256},
    {"rtx5060ti", 0x2D04, 12, 0, 36, 32, 16, 2572, 14001, 128},
    {"rtx4090", 0x2684, 8, 9, 128, 72, 24, 2520, 10501, 384},
    {"rtx4060ti", 0x2805, 8, 9, 34, 32, 16, 2535, 9001, 128},
    {"l4", 0x27B8, 8, 9, 58, 48, 24, 2040, 6251, 192},
    {"rtx3090", 0x2204, 8, 6, 82, 6, 24, 1695, 9751, 384},
    {"a10", 0x2236, 8, 6, 72, 6, 24, 1695, 6251, 384},
    {"a100", 0x20B0, 8, 0, 108, 40, 40, 1410, 1215, 5120},
    {"rtx2080ti", 0x1E07, 7, 5, 68, 5, 11, 1545, 7000, 352},
    {"v100", 0x1DB5, 7, 0, 80, 6, 32, 1530, 877, 4096},
    // Not targets, but the rows where spellings of one generation fact used to disagree.
    {"h100-sxm", 0x2330, 9, 0, 132, 50, 80, 1980, 2619, 5120},
    {"b300", 0, 10, 3, 148, 126, 268, 2032, 3996, 8192},
    {"gb10-spark", 0, 12, 1, 48, 24, 128, 2418, 4266, 256},
    {"unknown-sm120-48sm", 0, 12, 0, 48, 32, 12, 2500, 14001, 192},
};

// The board as a substituted DeviceClass; the scales multiply the clocks the driver reports.
[[nodiscard]] DeviceClass board_device_class(const FactsBoard& board, double sm_scale = 1.0,
                                             double memory_scale = 1.0);

// Whether the board's register-MMA price reads compiled kernel facts (rmma_terms_measured).
[[nodiscard]] bool board_prices_from_facts(const FactsBoard& board);

// One fact the price read: (format, schedule, epilogue, registers, spill bytes, threads, latency
// form).
using RmmaFactKey = std::tuple<int, int, int, int, std::size_t, int, bool>;

// The distinct facts of a ScopedRmmaFactsSink's record.
[[nodiscard]] std::set<RmmaFactKey> rmma_fact_keys(const std::vector<RmmaFactRead>& reads);

// The facts the register-MMA price of `board` reads over every format with sub-block scales
// (has_subblock_scales), a fixed set of shapes, epilogues and widths, on this build's kernels.
// The selector golden test checks that its dumps read no fact outside this set. Throws where the
// price reads none.
[[nodiscard]] std::set<RmmaFactKey> selector_facts_probe(const FactsBoard& board);

// The digest (12 hex digits) of a set of facts, and of the probe's.
[[nodiscard]] std::string facts_digest(const std::set<RmmaFactKey>& keys);
[[nodiscard]] std::string selector_facts_digest(const FactsBoard& board);

} // namespace ninferno::ops::detail::generic

namespace ninferno::tools {

// `ninferno-serve selector-facts`: one line "<board> <digest>" for each facts-priced golden board
// of this build's compute capability (NINFERNO_BUILD_ARCH), from its own kernels, read on the
// first GPU that runs them (the same major and an equal or higher minor; the same minor for an
// architecture-specific build). Exit 2 where this build prices no board from facts or no GPU runs
// its kernels (a release then reports the check as not run); 1 on any other error.
int selector_facts_main(int argc, char** argv);

} // namespace ninferno::tools
