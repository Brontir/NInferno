#pragma once

// The register-MMA variant rules on their inputs (generic_routes.cpp), declared apart from
// generic_gemm_internal.h so the host part of the selector needs no CUDA runtime (#620).

#include "ops/generic/route_cost_constants.h"

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail::generic {

// The DRAM bytes per SM cycle below which the wide-load Q6_K (GgmlQ6KWideDecoder) of one token
// tile pays: there the plain kernel's twice-fetched sectors fill the load queue (ncu, RTX 4090,
// Gemma 4 12B ffn_down 3840x15360 at T = 6: lg_throttle a third of the stalls, 54 % of DRAM).
// A/B, T = 1..8, bytes per SM cycle: RTX 4090 3.1 (that add 71 -> 59 us), 5070 Ti 5.2 (every
// problem 1-8 % faster); on the 3060 (7.2), whose plain kernel streams at 84 % of DRAM, the
// extra shuffles cost 2-10 %. The boundary of rmma_latency_pays' one row set, from the other side.
inline constexpr double kRmmaWideBytes = 5.5;

// Whether a variant kernel of a register-MMA call keeps the plain kernel's occupancy where it
// counts: its grid of `grid_ctas` CTAs takes no more waves on `sms` SMs at the variant's resident
// CTAs per SM (registers, shared memory and threads: cudaOccupancy*) than at the plain kernel's.
// In flight a SM holds tens of KB of weight loads at either count (rmma_kernel's ring: 37-74 KB
// against the 2-4 KB that Little's law asks at 230-310 ns of DRAM latency, generic_gemm.cuh), so
// a CTA fewer per SM costs nothing while the grid still fits in as many waves; a grid it pushes
// into another, part-empty wave runs that wave on SMs the rest left idle. Q4_K add K = 4096 w1x4
// at T = 8, held to 4, 3 and 2 CTAs per SM by a pad of shared memory (time against 4 CTAs):
// RTX 5070 Ti (70 SMs), grids of 192..768 CTAs: as many waves 0.97 and 0.97, a wave more
// 1.16-1.61 (8 of 8); RTX 4060 Ti (34 SMs), 96..384 CTAs: as many waves 0.98 and 0.99, a wave
// more 1.02-1.28 (8 of 8). The loss of a wave more lies between the bytes (1.0: the board
// streams at its bandwidth whatever the residency) and the waves' ratio (a latency-bound CTA
// takes as long alone), nearer the latter on the latency-bound 5070 Ti. The CTA count alone
// decided before: Q5_K's w1x4 latency kernel at 129 registers against 127 (sm_120, 4 -> 3 CTAs
// per SM) lost 12-30 % on Qwen3.5-9B calls, the 256-CTA ones a wave more at 3. A variant that
// holds no CTA, or unknown counts (0), never pays.
[[nodiscard]] bool rmma_variant_occupancy_rule(std::int64_t grid_ctas, int sms, int variant_ctas,
                                               int plain_ctas);

// rmma_wide_pays' rule on its inputs: below kRmmaWideBytes (0: unknown, no), not before sm_75
// (Volta's register MMA is FMA-emulated, compute-bound), and only where the wide kernel keeps the
// plain one's occupancy (rmma_variant_occupancy_rule: `occupancy`) and spills at most
// kRmmaWideSpillSlack bytes a thread more
// (sm_89's one-row-set tiles, held to 128 registers, keep 8 bytes of loop-invariant state in
// local memory; SwiGLU's 128-register tiles spill 160 bytes in the loop, RTX 4090 w1x16 at
// T = 6 100.3 -> 103.9 us).
inline constexpr std::size_t kRmmaWideSpillSlack = 16;
[[nodiscard]] bool rmma_wide_rule(double bytes_per_sm_cycle, int compute_capability, bool occupancy,
                                  std::size_t wide_local_bytes, std::size_t plain_local_bytes);

} // namespace ninferno::ops::detail::generic
