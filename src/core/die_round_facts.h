#pragma once

// Probe facts a die for the decode price (ops/generic/route_analyzer.cpp, its lab copy
// tools/route_pick/analyzer.py, analyzer rounds 50-105): what one card of a die measured, read by
// every board of that die. core/data/die_round_facts.json, compiled in; the Python reads the same
// file, so the two prices cannot disagree on a fact.
//
// Per die (keyed as gpu_facts spells the chip, "ga102"):
// - stream_share: pure weight streaming over the peak bus (the burst_wait probe's E0 rows);
// - head_t1_ns: the once-a-launch head of the register-MMA latency instance (the rounds probe's
//   intercept less the stream launch);
// - in_phase_waves / in_phase_share: the waves a lone CTA an SM stays in phase (at most this many
//   counted drains a launch), or a share of a drain every round (a die that never leaves phase);
// - row_flat: the rows an SM keeps open at the full DRAM rate (the stride4k probe's last flat
//   point, in its 2720-B rows);
// - rows_open: the rows-open probe curves, by warps_k and row sets, as [rows open an SM, share of
//   the die's flat rate]: at the probe's 2720-B rows (`curve`, the request width it loads with)
//   and at the exam row pitches (`pitch`, the stride4z / stride4q8 sweeps, keyed by the pitch in
//   bytes).
// A die without a field takes its architecture's (`arch`: stream_share by build arch, row_flat by
// arch with a default); a die the file does not name has none of the die terms.

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno {

// [rows open an SM, share of the die's flat rate], rows ascending.
using RowsOpenCurve = std::vector<std::pair<double, double>>;

struct DieRoundFacts {
    std::string chip;
    std::optional<double> stream_share;
    std::optional<double> head_t1_ns;
    std::optional<double> in_phase_waves;
    std::optional<double> in_phase_share;
    std::optional<double> row_flat;
    std::optional<std::string> request; // the stride4k probe's load width ("64 B")
    // "wk_sets" -> curve; pitch bytes -> ("wk_sets" -> curve).
    std::map<std::string, RowsOpenCurve> curve;
    std::map<double, std::map<std::string, RowsOpenCurve>> pitch;
};

struct DieRoundTable {
    double probe_row_bytes = 0.0;
    std::map<std::string, double> arch_stream_share; // build arch ("86", "120a")
    std::map<std::string, double> arch_row_flat;     // build arch, "" the default
    std::map<std::string, DieRoundFacts> dies;
};

[[nodiscard]] DieRoundTable parse_die_round_facts(std::string_view json_text);
[[nodiscard]] const DieRoundTable& die_round_facts();
[[nodiscard]] const DieRoundFacts* die_round_row(std::string_view chip);

} // namespace ninferno
