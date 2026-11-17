#include "ops/generic/mmq_unit.h"

#include "core/mmq_columns.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/route_cost_constants.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <vector>

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
    } catch (...) {
        return std::nullopt; // a snapshot that does not read: the fitted price
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

std::optional<MmqLaunchLaw> law(std::string_view arch, QType q, const IsaRow& isa,
                                const UnitFacts& f) {
    MmqLaunchLaw out;
    for (std::size_t i = 0; i < std::size(mmq_columns::kTilesWork); ++i) {
        const auto u = unit_seconds(arch, q, mmq_columns::kTilesWork[i], isa, f);
        if (!u) return std::nullopt;
        out.unit_clocks[i] = *u * f.clock_hz;
    }
    return out;
}

UnitFacts unit_facts_of(int rows, int k, int columns, int j, int sms, double clock_hz,
                        double dram_bytes_per_s) noexcept {
    const mmq_columns::ColumnLaunch l = mmq_columns::launch_of(rows, k, columns, j, sms);
    const int column_tiles            = std::max(1, mmq_columns::tile_count(columns, j));
    const int steps = std::max(1, (k + mmq_columns::kUnitK - 1) / mmq_columns::kUnitK);
    // The resident CTAs: a tile each up to the SMs, or under stream-k the launch's CTAs, each from
    // its own first unit (ggml: CTA b from unit b W / B).
    const int ctas       = std::max(1, l.split ? l.blocks : std::min(l.tiles, sms));
    const long long work = static_cast<long long>(std::max(1, l.tiles)) * steps;
    // The weight tile a CTA reads in its first iteration: (row tile, K step) of its first unit.
    std::vector<long long> reads;
    reads.reserve(static_cast<std::size_t>(ctas));
    for (long long b = 0; b < ctas; ++b) {
        const long long unit = l.split ? b * work / ctas : b * steps;
        reads.push_back((unit / steps / column_tiles) * steps + unit % steps);
    }
    std::sort(reads.begin(), reads.end());
    const int weight_tiles =
        static_cast<int>(std::unique(reads.begin(), reads.end()) - reads.begin());
    return UnitFacts{std::max(1, weight_tiles), ctas, clock_hz, dram_bytes_per_s};
}

std::optional<double> launch_seconds(std::string_view arch, QType q, const IsaRow& isa, int rows,
                                     int k, int columns, int j, int sms, double clock_hz,
                                     double dram_bytes_per_s, double fixup_launch_seconds) {
    const auto u = unit_seconds(
        arch, q, j, isa, unit_facts_of(rows, k, columns, j, sms, clock_hz, dram_bytes_per_s));
    if (!u) return std::nullopt;
    const mmq_columns::ColumnLaunch l = mmq_columns::launch_of(rows, k, columns, j, sms);
    return l.units * *u +
           (l.fixup ? mmq_columns::fixup_bytes(l) / dram_bytes_per_s + fixup_launch_seconds : 0.0);
}

} // namespace ninferno::mmq_unit
