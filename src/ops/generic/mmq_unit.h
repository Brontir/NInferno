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
//   B_dram  the weight tiles the launch reads from DRAM in one iteration (unit_facts_of), 128
//           rows x the format's bytes per 256 values each;
//   B_l2    what every resident CTA reads through the L2: its weight tile and its activation
//           tile (two block_q8_1_mmq of 144 bytes per column and 256 values);
//   BW_dram the board's DRAM rate; BW_l2 the L2's bytes per SM clock per SM (isa_probe 18) x the
//           CTAs' SMs x f; L_dram, L_shared the arch's isa_probe latencies (dram_latency_w1,
//           shared_latency); stall_sum, sb_waits_shared the counted iteration of the kernel's
//           SASS (the kernel-facts snapshot's mmq section); f the clock the units are stated at.
//
// Scored before it was written here, frozen first (tests/fixtures/mmq_law/README.md): mean |err|
// 4.4 % (RTX 3090) to 9.5 % (RTX 4060 Ti) over q4_K, q5_K, q6_K and J 16..128 on the launch
// law's grid; J 16-32 over by up to 28 %, J >= 80 under by up to 12 %. That score took each card's
// own L2 rate (a 32-warp probe); here the rate is the arch's isa_probe 18 row, measured on one
// card of the arch (the L2 rate is a card's, not an arch's: RTX 4060 Ti 17.2 against the RTX
// 4090's 15.1), so a card other than the row's carries that difference. Only the scored formats
// are priced (scored()); every other format keeps the fitted price.
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

// The formats the law was scored on: Q4_K, Q5_K, Q6_K.
[[nodiscard]] bool scored(QType q) noexcept;

// The format's bytes per 256 values (Q4_K 144, Q5_K 176, Q6_K 210); 0 for a non-ggml format.
[[nodiscard]] double bytes_per_256(QType q) noexcept;

// One unit's seconds at the column tile `j`, from the counts of `arch`'s snapshot (the kernel's
// stall sum and shared waits) and `isa`'s facts; nullopt for a format not scored(), and where a
// count or a fact is missing or the snapshot does not read.
[[nodiscard]] std::optional<double> unit_seconds(std::string_view arch, QType q, int j,
                                                 const IsaRow& isa, const UnitFacts& f);

// The launch law with the counted units at the six work widths (unit_clocks at f.clock_hz);
// stream_k_unit 1 and launch_seconds 0, not yet counted. nullopt where any width is not priced.
[[nodiscard]] std::optional<MmqLaunchLaw> law(std::string_view arch, QType q, const IsaRow& isa,
                                              const UnitFacts& f);

// The resident CTAs and the weight tiles they read from DRAM in one iteration of a launch of
// `rows` x `k` x `columns` at the tile `j` on `sms` SMs (core/mmq_columns.h launch_of): a tile
// launch runs a tile a CTA up to the SMs, a stream-k launch launch_of's CTAs. The weights come
// from DRAM once over the launch (CTAs on one row tile in the same or the next iteration share it
// through the L2: ggml takes the column tile fastest, mmq.cuh jt = tile % ntx), so an iteration
// reads the row tiles' K steps over the launch's units, at most one weight tile a CTA.
[[nodiscard]] UnitFacts unit_facts_of(int rows, int k, int columns, int j, int sms, double clock_hz,
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
