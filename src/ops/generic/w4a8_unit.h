#pragma once

// The W4A8 tiles' K step from counts and chip facts (#613 PR-6, #978): one 128-value K tile of
// generic_w4a8.cuh's w4a8_kernel (every warp loads, decodes and multiplies) and w4a8_ws_kernel
// (producer warps decode into a ring, MMA warps multiply), in SM clocks, for the critical MMA warp:
//
//   cycles = S + (W - W_glb) x L_sh + I_other + T_out
//
//   S       the MMA warp's K-loop stall sum (the static schedule of the kernel-facts snapshot's
//           w4a8 row: the clocks one warp alone takes to issue one iteration);
//   W       its scoreboard waits; W_glb those on a global load. The global load (the next K tile's
//           weight words, or the producers' copy) is issued an iteration ahead and an iteration
//           is longer than the DRAM latency, so it is not exposed; the other waits (shared loads,
//           the MMA results, the ring) are priced at the shared latency L_sh (isa_probe 18);
//   I_other the instructions the other warps on the MMA warp's scheduler issue in that iteration:
//           the ring's bar.sync releases every warp of the CTA together, so they take the issue
//           slots the critical warp waits for. Warps a scheduler = the CTA's warps x the CTAs an
//           SM holds / 4: the other MMA warps' K loop and, on the producer-warp tile, the
//           producers' loop (the snapshot's prod_ columns);
//   T_out   the code outside the loop (prologue, epilogue: outside_stall + outside_waits x L_sh)
//           shared over the CTA's K tiles, as the measured time per K tile carries it.
//
// The tensor pipe is not a term: isa_probe 19 measures IMMA.16832 at 16 clk a partition and
// IMMA.16816 at 8 (RTX 3090), so a Q4_K and a Q6_K K tile both hold 1024 clk of it a scheduler,
// under the cycles above (#978).
//
// Frozen and scored before it was written here (#978): the producer-warp tile 128x64, Q4_K and
// Q6_K, 23 of 24 cells within 5 % on the RTX 4060 Ti (sm_89) and RTX 3060 (sm_86) at T 64..1024;
// refuted on the RTX 4090 at T >= 256 (+7 .. +22 %, an L2 term of the 128-SM board, not counted
// yet). The classic tile's frozen table was refuted on the RTX 3060 (17 of 60 cells within 5 %, a
// per-tile error of -18 .. +6 %: its MMA warps wait on their own global loads), so only the
// producer-warp tile has a counted price; the classic tile keeps the fitted step (#978).
// Host code over embedded facts only (ninferno_ops_host), no launch code.

#include "core/format_facts.h"
#include "core/isa_facts.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/kernel_facts.h"

#include <optional>
#include <string_view>

namespace ninferno::w4a8_unit {

// Whether the selector prices the W4A8 tiles by the counted K step: route_cost::kW4a8CountedStep
// unless a test or the price tool set an override for its process (nullopt clears it). Off,
// every price is the one before (the fitted kW4a8Efficiency and step cycles).
[[nodiscard]] bool counted() noexcept;
void counted_for_test(std::optional<bool> on) noexcept;

// The architectures the law was scored on (sm_86, sm_89), and the formats (Q4_K, Q6_K).
[[nodiscard]] bool scored_arch(std::string_view arch) noexcept;
[[nodiscard]] bool scored(QType q) noexcept;

// The snapshot key of schedule `schedule` (generic_schedules.h NINFERNO_GENERIC_W4A8_SCHEDULES) for
// a scored format, the epilogue and the release arch (the classic tile's Flavor is the build's,
// generic_w4a8.cuh w4a8_classic_flavor); nullopt for a format that is not scored.
[[nodiscard]] std::optional<ops::detail::generic::kernel_facts::W4a8KernelKey>
kernel_key(int schedule, QType q, ops::detail::generic::Epilogue epilogue, std::string_view arch);

// The clocks one K tile of the kernel `key` takes on `arch`, with `ctas_per_sm` CTAs resident and
// `k_tiles` K tiles a CTA; nullopt for a classic tile (no producer warps: the law does not hold
// there), where the snapshot has no such kernel or no counted loop, the producer-warp tile has no
// counted producer loop, or the arch has no shared latency.
[[nodiscard]] std::optional<double>
tile_cycles(std::string_view arch, const ops::detail::generic::kernel_facts::W4a8KernelKey& key,
            const IsaRow& isa, int ctas_per_sm, int k_tiles);

// The same from a row's facts (the test's entry point).
[[nodiscard]] std::optional<double>
tile_cycles_of(const ops::detail::generic::kernel_facts::KernelFacts& facts,
               const ops::detail::generic::kernel_facts::W4a8KernelKey& key, double shared_latency,
               int ctas_per_sm, int k_tiles);

} // namespace ninferno::w4a8_unit
