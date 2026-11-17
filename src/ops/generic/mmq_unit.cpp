#include "ops/generic/mmq_unit.h"

#include "core/mmq_columns.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/route_cost_constants.h"

#include <algorithm>
#include <cstddef>

namespace ninferno::mmq_unit {
namespace {

namespace kernel_facts = ops::detail::generic::kernel_facts;

constexpr double kTileRows           = 128;
constexpr double kActivationBytes    = 2 * 144; // two block_q8_1_mmq a column per 256 values
constexpr double kValuesPerIteration = 256;

std::optional<bool> g_counted_override;

} // namespace

bool counted() noexcept {
    return g_counted_override.value_or(ops::detail::generic::route_cost::kMmqCountedUnit);
}

void counted_for_test(std::optional<bool> on) noexcept { g_counted_override = on; }

double bytes_per_256(QType q) noexcept {
    return ggml_bits_per_value(q) * kValuesPerIteration / 8.0;
}

std::optional<double> unit_seconds(std::string_view arch, QType q, int j, const IsaRow& isa,
                                   const UnitFacts& f) {
    const int type = format_facts(q).ggml_type;
    const double b = bytes_per_256(q);
    if (type < 0 || !(b > 0) || f.weight_tiles <= 0 || f.ctas <= 0 || !(f.clock_hz > 0) ||
        !(f.dram_bytes_per_s > 0) || !isa.dram_latency.w1 || !isa.shared_latency ||
        !isa.l2_bytes_per_sm_clock) {
        return std::nullopt;
    }
    const auto facts = kernel_facts::mmq_facts(arch, {type, j, false, 30});
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

UnitFacts unit_facts_of(int rows, int columns, int j, int sms, double clock_hz,
                        double dram_bytes_per_s) noexcept {
    const int row_tiles    = mmq_columns::tile_count(rows, mmq_columns::kTile);
    const int column_tiles = std::max(1, mmq_columns::tile_count(columns, j));
    const int ctas         = std::max(1, std::min(row_tiles * column_tiles, sms));
    const int shared_by    = (ctas + column_tiles - 1) / column_tiles;
    return UnitFacts{std::max(1, std::min(row_tiles, shared_by)), ctas, clock_hz, dram_bytes_per_s};
}

std::optional<double> launch_seconds(std::string_view arch, QType q, const IsaRow& isa, int rows,
                                     int k, int columns, int j, int sms, double clock_hz,
                                     double dram_bytes_per_s, double fixup_launch_seconds) {
    const auto u = unit_seconds(arch, q, j, isa,
                                unit_facts_of(rows, columns, j, sms, clock_hz, dram_bytes_per_s));
    if (!u) return std::nullopt;
    const mmq_columns::ColumnLaunch l = mmq_columns::launch_of(rows, k, columns, j, sms);
    return l.units * *u +
           (l.fixup ? mmq_columns::fixup_bytes(l) / dram_bytes_per_s + fixup_launch_seconds : 0.0);
}

} // namespace ninferno::mmq_unit
