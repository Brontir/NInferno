#pragma once

// SM DRAM rate facts: what one SM of an architecture moves from DRAM, measured by the
// bytes-in-flight probe on one card of it (tools/bench/warp_inflight/, its rows in
// core/data/sm_rate_facts.json, compiled in). Per architecture: P, the most bytes per SM clock one
// SM reaches with the L1-largest carve-out (`free`) and with the shared-largest one (`placed`);
// L0, the fetch time of one warp alone with one 16-byte streaming load a lane in flight; and the
// probe's two-slot ring rows (2048 B a warp in flight by count) that verify the decode price's
// per-SM bound min(P, W x b_w / L0) (ops/generic/kernel_roofline.h dram_time, #690).
//
// The table is keyed like the ISA facts table (the build's arch, "120a"; the row's "cc" field). An
// architecture without a row is unknown (sm_rate_row() gives nullopt) and the price keeps its
// legacy DRAM law there: a row speaks for its own architecture only. Units are the table's and are
// checked on parse (B/clk, GB/s, MHz, ns, B, warps). This header and its source need no CUDA.

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno {

// One architecture's row of the SM rate facts table. Every fact is required: a row is one run of
// one probe version, which prints them all.
struct SmRateRow {
    // Where the row comes from.
    std::string arch; // the table's key: the build's arch ("86", "120a")
    int major = 0, minor = 0;
    std::string card;       // the card the probe ran on
    std::string probe;      // the probe version ("warp_inflight 3")
    std::string provenance; // "Measured"
    int sms = 0;            // the probe card's SMs

    // P: the largest DRAM rate one SM reached (k = 1), per SM clock of the row it was read at.
    double bytes_per_clock_free   = 0;         // B/clk, L1-largest carve-out
    double bytes_per_clock_placed = 0;         // B/clk, shared-largest carve-out
    double gbps_free = 0, gbps_placed = 0;     // GB/s an SM, the rates P was read from
    double sm_mhz_free = 0, sm_mhz_placed = 0; // MHz, the SM clock of those rows
    // L0: one warp alone on one SM, one 16-byte load a lane in flight, its fetch time (ns).
    double dram_stream_latency_ns = 0;
    // The two-slot ring (2048 B a warp by count): b_w at W = 1 (B) and the largest W up to which
    // every row keeps 0.9 of it; 4 loads a lane in one chain, and the 4-load ring, at W = 1.
    double ring2_b_w_w1_free = 0, ring2_b_w_w1_placed = 0;
    double ring2_linear_to_w_free = 0, ring2_linear_to_w_placed = 0;
    double chain4_b_w_w1_free = 0, ring4_b_w_w1_free = 0;
    // The ring2 rows at k = 1: (W, GB/s an SM), in W order.
    std::vector<std::pair<double, double>> ring2_free, ring2_placed;
};

// The table, parsed once from the embedded JSON. Throws std::runtime_error when the embedded text
// is malformed (a build defect: the host test parses the same text).
[[nodiscard]] std::span<const SmRateRow> sm_rate_facts_table();

// The row of compute capability major.minor, nullopt when the table has none.
[[nodiscard]] std::optional<SmRateRow> sm_rate_row(int major, int minor);

// Parses a table text (the embedded one, or a test's). Throws std::runtime_error naming the row
// and the fact: a schema other than 1, a key that is not the row's cc, a cc twice, a fact missing,
// in another unit than the reader's or not a positive finite number, a ring row that is not a
// positive (W, GB/s) pair in rising W.
[[nodiscard]] std::vector<SmRateRow> parse_sm_rate_facts(std::string_view json_text);

// The names of the facts the reader takes from a row, in the table's spelling.
[[nodiscard]] std::span<const std::string> sm_rate_fact_names();

} // namespace ninferno
