#pragma once

// The unit of ggml's matrix kernel from counts and chip facts (#613 PR-5): one 256-K iteration of
// a 128-row x J-column tile, the unit core/mmq_columns.h's launch law prices a launch in
// (core/generation.h MmqLaunchLaw), here counted where the generation row measured it.
//
// The kernel loads an iteration's tiles, synchronises, multiplies, synchronises: the load phase
// and the warp's chain do not overlap inside a CTA. So a unit takes
//
//   load  = max(B_dram / BW_dram, B_l2 / BW_l2) + L_dram / f
//   chain = (stall_sum + sb_waits_shared x L_shared) / f
//
//   B_dram  the distinct weight tiles the launch's CTAs read in one iteration, 128 rows x the
//           format's bytes per 256 values each, from DRAM once;
//   B_l2    what every resident CTA reads through the L2: its weight tile and its activation
//           tile (two block_q8_1_mmq of 144 bytes per column and 256 values);
//   BW_dram the board's DRAM rate; BW_l2 the L2's bytes per SM clock per SM (isa_probe 18) x the
//           CTAs' SMs x f; L_dram, L_shared the arch's isa_probe latencies (dram_latency_w1,
//           shared_latency); stall_sum, sb_waits_shared the counted iteration of the kernel's
//           SASS (the kernel-facts snapshot's mmq section); f the clock the units are stated at.
//
// Scored before it was written here, frozen first (tests/fixtures/mmq_law/README.md): mean |err|
// 4.4 % (RTX 3090) to 9.5 % (RTX 4060 Ti) over q4_K, q5_K, q6_K and J 16..128 on the launch
// law's grid; J 16-32 over by up to 28 %, J >= 80 under by up to 12 %.
//
// Host code over embedded facts only: the selector (ninferno_ops_host) and the choice call it,
// no launch code.

#include "core/format_facts.h"
#include "core/generation.h"
#include "core/isa_facts.h"

#include <optional>
#include <string_view>

namespace ninferno::mmq_unit {

// Whether the selector's A8 candidate and the choice price ggml MMQ by the counted unit:
// route_cost::kMmqCountedUnit (ops/generic/route_cost_constants.h) unless a test or the price tool
// set an override for its process (nullopt clears it). Off, every price is the one before.
[[nodiscard]] bool counted() noexcept;
void counted_for_test(std::optional<bool> on) noexcept;

// The launch a unit is priced in, and the board it runs on.
struct UnitFacts {
    int weight_tiles        = 0; // distinct weight tiles the resident CTAs read an iteration
    int ctas                = 0; // resident CTAs, one an SM
    double clock_hz         = 0; // the clock the unit is stated at (the board's reference clock)
    double dram_bytes_per_s = 0;
};

// The format's bytes per 256 values (Q4_K 144, Q5_K 176, Q6_K 210); 0 for a non-ggml format.
[[nodiscard]] double bytes_per_256(QType q) noexcept;

// One unit's seconds at the column tile `j`, from the counts of `arch`'s snapshot (the kernel's
// stall sum and shared waits) and `isa`'s facts; nullopt where a count or a fact is missing.
[[nodiscard]] std::optional<double> unit_seconds(std::string_view arch, QType q, int j,
                                                 const IsaRow& isa, const UnitFacts& f);

// The launch law with the counted units at the six work widths (unit_clocks at f.clock_hz);
// stream_k_unit 1 and launch_seconds 0, not yet counted. nullopt where any width is not priced.
[[nodiscard]] std::optional<MmqLaunchLaw> law(std::string_view arch, QType q, const IsaRow& isa,
                                              const UnitFacts& f);

// The resident CTAs and the distinct weight tiles they read in one launch of `rows` x `columns`
// at the tile `j` on `sms` SMs: one CTA an SM (the launch law counts one). ggml's stream-k index
// takes the column tile fastest (mmq.cuh: jt = tile % ntx, it = tile / ntx), so the resident CTAs
// share each weight tile among the column tiles: ceil(CTAs / column tiles) of them, up to the row
// tiles.
[[nodiscard]] UnitFacts unit_facts_of(int rows, int columns, int j, int sms, double clock_hz,
                                      double dram_bytes_per_s) noexcept;

// One mul_mat_q launch: the longest CTA's units (core/mmq_columns.h launch_of) at the unit its own
// resident CTAs give, and where the CTAs do not divide the tiles, the fixup's bytes at the DRAM
// rate and `fixup_launch_seconds` (its graph node). A stream-k CTA's unit is the tile CTA's (not
// yet counted apart). nullopt where the unit is not priced.
[[nodiscard]] std::optional<double> launch_seconds(std::string_view arch, QType q,
                                                   const IsaRow& isa, int rows, int k, int columns,
                                                   int j, int sms, double clock_hz,
                                                   double dram_bytes_per_s,
                                                   double fixup_launch_seconds);

} // namespace ninferno::mmq_unit
