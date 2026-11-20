#include "ops/generic/mmq_unit.h"

#include "core/launch_facts.h"
#include "core/mmq_columns.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/route_cost_constants.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <mutex>
#include <utility>

namespace ninferno::mmq_unit {
namespace {

namespace kernel_facts = ops::detail::generic::kernel_facts;

constexpr double kTileRows           = 128;
constexpr double kActivationBytes    = 2 * 144; // two block_q8_1_mmq a column per 256 values
constexpr double kValuesPerIteration = 256;

// -1 no override, 0 off, 1 on.
std::atomic<int> g_counted_override{-1};

} // namespace

bool counted() noexcept {
    const int o = g_counted_override.load(std::memory_order_relaxed);
    return o < 0 ? ops::detail::generic::route_cost::kMmqCountedUnit : o == 1;
}

void counted_for_test(std::optional<bool> on) noexcept {
    g_counted_override.store(on ? (*on ? 1 : 0) : -1, std::memory_order_relaxed);
}

bool scored(QType q) noexcept {
    return q == QType::GGML_Q4_K || q == QType::GGML_Q5_K || q == QType::GGML_Q6_K;
}

double bytes_per_256(QType q) noexcept {
    return ggml_bits_per_value(q) * kValuesPerIteration / 8.0;
}

bool scored_arch(std::string_view arch) noexcept {
    return arch == "86" || arch == "89" || arch == "120a";
}

std::optional<double> unit_seconds(std::string_view arch, QType q, int j, const IsaRow& isa,
                                   const UnitFacts& f) {
    const int type = format_facts(q).ggml_type;
    const double b = bytes_per_256(q);
    if (!scored(q) || type < 0 || !(b > 0) || f.weight_tiles <= 0 || f.ctas <= 0 ||
        !(f.clock_hz > 0) || !(f.dram_bytes_per_s > 0) || !isa.dram_latency.w1 ||
        !isa.shared_latency || !isa.l2_bytes_per_sm_clock) {
        return std::nullopt;
    }
    std::optional<kernel_facts::KernelFacts> facts;
    try {
        facts = kernel_facts::mmq_facts(arch, {type, j, false, 30});
    } catch (const std::exception& e) {
        // A snapshot that does not read: the fitted price, but never silently (#973: a broken
        // merge of a snapshot passed unnoticed outside test_kernel_facts). Said once a process.
        static std::once_flag said;
        std::call_once(said, [&] {
            std::fprintf(stderr,
                         "ninferno: the kernel facts of sm_%.*s do not read (%s); the MMQ unit "
                         "falls back to the fitted price\n",
                         static_cast<int>(arch.size()), arch.data(), e.what());
        });
        return std::nullopt;
    }
    if (!facts || !facts->loop || !facts->loop->schedule || !facts->loop->pipes)
        return std::nullopt;
    const double b_dram = f.weight_tiles * kTileRows * b;
    const double b_l2   = f.ctas * (kTileRows * b + j * kActivationBytes);
    const double load   = std::max(b_dram / f.dram_bytes_per_s,
                                   b_l2 / (*isa.l2_bytes_per_sm_clock * f.ctas * f.clock_hz)) +
                        *isa.dram_latency.w1 / f.clock_hz;
    const double chain = (facts->loop->schedule->stall_sum +
                          facts->loop->pipes->sb_waits_shared * *isa.shared_latency) /
                         f.clock_hz;
    return load + chain;
}

std::optional<CountedBoard> counted_board(int major, int minor) {
    if (!counted()) return std::nullopt;
    std::string arch = kernel_facts::release_arch(major, minor);
    if (!scored_arch(arch)) return std::nullopt;
    std::optional<IsaRow> isa = isa_row(major, minor);
    if (!isa) return std::nullopt;
    return CountedBoard{
        std::move(arch), *isa,
        ninferno::launch_seconds(major, minor, LaunchKind::GraphNode).value_or(0.0)};
}

UnitFacts unit_facts_of(int rows, int k, int columns, int j, int sms, double clock_hz,
                        double dram_bytes_per_s) noexcept {
    const mmq_columns::ColumnLaunch l = mmq_columns::launch_of(rows, k, columns, j, sms);
    const int row_tiles               = mmq_columns::tile_count(rows, mmq_columns::kTile);
    const int steps = std::max(1, (k + mmq_columns::kUnitK - 1) / mmq_columns::kUnitK);
    // The resident CTAs: a tile each up to the SMs, or under stream-k the launch's CTAs.
    const int ctas = std::max(1, l.split ? l.blocks : std::min(l.tiles, sms));
    // The weights come from DRAM once over the launch, whichever CTAs share a tile in the same
    // iteration or the next (an L2 hit): the row tiles' K steps spread over the launch's units,
    // at most one weight tile a CTA an iteration.
    const long long reads = static_cast<long long>(std::max(1, row_tiles)) * steps;
    const long long units = std::max(1, l.units);
    const int weight_tiles =
        static_cast<int>(std::min<long long>(ctas, (reads + units - 1) / units));
    return UnitFacts{std::max(1, weight_tiles), ctas, clock_hz, dram_bytes_per_s};
}

std::optional<double> launch_seconds(std::string_view arch, QType q, const IsaRow& isa, int rows,
                                     int k, int columns, int j, int sms, double clock_hz,
                                     double dram_bytes_per_s, double fixup_launch_seconds) {
    const auto u = unit_seconds(
        arch, q, j, isa, unit_facts_of(rows, k, columns, j, sms, clock_hz, dram_bytes_per_s));
    if (!u) return std::nullopt;
    const mmq_columns::ColumnLaunch l = mmq_columns::launch_of(rows, k, columns, j, sms);
    if (!l.fixup) return l.units * *u;
    // unit_seconds read the L2 rate, so it is there. The fixup's working CTAs move the bytes, not
    // the stream-k grid (mmq_columns::fixup_ctas).
    const double rate  = *isa.l2_bytes_per_sm_clock * mmq_columns::fixup_ctas(l, sms) * clock_hz;
    const double bytes = mmq_columns::fixup_bytes(l) / rate;
    // ...but no faster than its walk: the CTA of a split tile reads the preceding CTAs' partials
    // one after another (mmq_columns::ColumnLaunch::rounds), each an L2 round trip of its four
    // warps (the fixup CTA's 128 threads; isa_facts l2_latency), then adds the sum into the
    // output, one trip more (#1095). Not counted yet: each round's 64-bit divide of the block
    // index ahead of its load (no kernel-facts row of the fixup kernel). Where the row has no
    // latency the bytes stand.
    const std::optional<double> trip = isa.l2_latency.w4 ? isa.l2_latency.w4 : isa.l2_latency.w1;
    const double walk = trip && l.rounds > 0 ? (l.rounds + 1) * *trip / clock_hz : 0.0;
    return l.units * *u + std::max(bytes, walk) + fixup_launch_seconds;
}

} // namespace ninferno::mmq_unit
