#pragma once

// The kernel roofline (#613 PR-4): one pure price of a tile on a chip, every term named, so that a
// caller prints the breakdown and a CPU replica prices any board without a GPU. Its inputs are
// plain values: what the chip is (the board registry's facts, the generation row's rate, the
// architecture's residency limits), what the kernel instance is (its tile, threads, and the
// registers and spill the kernel-facts snapshot or the describe report), and the call. The
// constants it does not derive yet are gathered in Calibration, each still the one tagged value of
// route_cost_constants.h, so a later step replaces a fitted one by a counted term without touching
// the clients. Today's clients: price_routes' register-MMA and pipelined tiles (generic_cost.cpp),
// byte-equal to the formula they replaced (tests/ops/linear/test_kernel_roofline.cpp). No CUDA.
//
// The model, in words: a tile's time on the busiest SM is its bytes over that SM's share of the
// memory system plus its MMA work over the SM's tensor rate (the register MMA runs them in series;
// the pipelined tiles take the longer of the two at a fraction of the rate), slowed in proportion
// while the SM holds fewer warps than saturate it, plus the exposed dependent chain of its K steps
// per residency round, plus the reduction of a split K.

#include "core/device_class.h"
#include "core/generation.h"
#include "ops/generic/route_cost_constants.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace ninferno::ops::detail::generic::kernel_roofline {

// What the price reads of the chip. Facts of the board (core/machine_facts.h), the generation
// (core/generation.h) and the architecture (the CUDA residency limits); chip_facts_of fills it
// from a DeviceClass.
struct ChipFacts {
    double sms                = 0;
    double clock_hz           = 0; // the reference SM clock
    double dram_bytes_per_s   = 0; // bus width x memory data rate
    double tensor_rate        = 0; // the BF16 MMA rate per SM and clock (GenerationRow::bf16)
    double max_threads_per_sm = 0;
    double max_ctas_per_sm    = 0;
    double registers_per_sm   = 0;
    double smem_per_sm        = 0;
};

// The constants the price does not derive yet, with their tags in route_cost_constants.h. The
// defaults are today's values, so a client that passes calibration_of(major, minor) prices as it
// did; a replica may move one and read the terms it moves.
struct Calibration {
    // Register MMA.
    double rmma_step_cycles            = route_cost::kRmmaStepCycles.value();
    double rmma_step_k                 = route_cost::kRmmaStepK.value();
    double rmma_saturating_warps       = route_cost::kRmmaSaturatingWarps.value();
    double rmma_heavy_saturating_warps = route_cost::kRmmaHeavySaturatingWarps.value();
    double l2_bytes_per_sm_clock       = route_cost::kL2BytesPerSmClockGa10x.value();
    double rmma_heavy_step_k           = route_cost::kRmmaHeavyStepK.value();
    double spill_accesses_per_step     = route_cost::kSpillAccessesPerStep.value();
    double l1_bytes_per_sm_clock       = route_cost::kL1BytesPerSmClock.value();
    double reg_alloc_per_warp          = route_cost::kRegAllocPerWarp.value();
    double cluster_reduce_cycles       = route_cost::kClusterReduceCycles.value();
    double warp_reduce_cycles          = route_cost::kWarpReduceCycles.value();
    // Pipelined tiles (tiled, staged).
    double pipelined_stage_cycles        = route_cost::kPipelinedStageCycles.value();
    double pipelined_stage_k             = route_cost::kPipelinedStageK.value();
    double pipelined_saturating_warps    = route_cost::kPipelinedSaturatingWarps.value();
    double pipelined_stage_bytes_per_row = route_cost::kPipelinedStageBytesPerRow.value();
    double max_tile_shared_bytes         = route_cost::kMaxTileSharedBytes.value();
    // Every tile.
    double min_warp_fill = route_cost::kMinWarpFill.value();
};

// The call: an n x k projection of `tokens` tokens; `sets` 2 under SwiGLU (the weight has 2 n
// rows, every output row multiplies both); `bpw` the weight's payload bytes over n x k (under
// SwiGLU the gate and up bytes of an output row together).
struct Call {
    double n      = 0;
    double k      = 0;
    double tokens = 0;
    double sets   = 1;
    double bpw    = 0;
};

// How a grid of CTAs sits on the chip: what every form of the price computes first.
struct Grid {
    double ctas      = 0; // of the whole call
    double waves     = 0; // CTAs on the busiest SM: ceil(ctas / SMs)
    double occupancy = 0; // CTAs one SM holds at once
    double rounds    = 0; // residency rounds on the busiest SM: ceil(waves / occupancy)
    double warps     = 0; // warps resident on a busy SM
    double fill      = 0; // the slowdown factor of too few warps, in (min_warp_fill, 1]
};

// ---- Register MMA ----

// One register-MMA kernel instance for one call: its tile (route_tile), its CTA, and the
// compiled facts the price reads where it keeps the heavy-decoder terms (`heavy`:
// has_subblock_scales on a generation whose cards confirm them, rmma_terms_measured). `registers`
// absent: priced by threads alone, as a process without the kernel's facts is.
struct RegisterMmaKernel {
    double rows    = 0;           // output rows of the tile
    double cols    = 0;           // token columns of the tile
    double cluster = 1;           // K over a thread-block cluster of that many CTAs
    double threads = 0;           // per CTA
    double warps_k = 1;           // K over that many warps of the CTA
    std::optional<int> registers; // per thread
    double local_bytes = 0;       // the spill per thread
    bool heavy         = false;
};

struct RegisterMmaTerms {
    Grid grid;
    // Bytes, in seconds. Heavy form: the matrix once over the DRAM bus, every CTA's weights and
    // token tile through its SM's L2 port, the spill through the L1; these add (a max of the DRAM
    // and the L2 time did worse on the 5090 and 4090). Plain form: the busiest SM's DRAM share
    // of every CTA's bytes, or the whole matrix over the whole bus, whichever is longer.
    double dram          = 0;
    double l2_weight     = 0;
    double l2_activation = 0;
    double spill         = 0;
    double mem           = 0;
    double bytes         = 0; // the form's sum
    // The MMA work over the tensor rate.
    double mma = 0;
    // K split over a cluster and over the CTA's K warps.
    double reduce_cluster = 0;
    double reduce_warps   = 0;
    double reduce         = 0;
    // The warps' dependent chain of K steps per residency round (not slowed by the fill).
    double chain = 0;
    // (bytes + mma + reduce) / fill + chain.
    double total = 0;
};

// The CTAs of `threads` threads one SM holds: its thread and CTA limits and, where the kernel's
// registers are known, the register file (allocated per warp in units of reg_alloc_per_warp). A
// 16-row, 32-token tile of a K-quant keeps 229 registers: two 128-thread CTAs, not the twelve the
// thread limit allows.
[[nodiscard]] inline double register_mma_occupancy(const ChipFacts& chip, const Calibration& cal,
                                                   double threads, std::optional<int> registers) {
    const double limit =
        std::min(std::floor(chip.max_threads_per_sm / threads), chip.max_ctas_per_sm);
    if (!registers) return limit;
    const double unit    = cal.reg_alloc_per_warp;
    const double per_cta = std::ceil(*registers * 32.0 / unit) * unit * (threads / 32.0);
    return std::max(1.0, std::min(limit, std::floor(chip.registers_per_sm / per_cta)));
}

[[nodiscard]] inline RegisterMmaTerms price_register_mma(const ChipFacts& chip,
                                                         const Calibration& cal, const Call& call,
                                                         const RegisterMmaKernel& k) {
    const auto cdiv = [](double a, double b) { return std::ceil(a / b); };
    RegisterMmaTerms t;
    Grid& g            = t.grid;
    const double tiles = cdiv(call.tokens, k.cols);
    const double gx    = cdiv(call.n, k.rows);
    g.ctas             = gx * tiles * k.cluster;
    const double kc    = call.k / k.cluster;
    const double q     = std::ceil(g.ctas / chip.sms);
    g.waves            = q;
    g.occupancy        = register_mma_occupancy(chip, cal, k.threads, k.registers);
    const double occ   = g.occupancy;
    // The warps resident on a busy SM: SMs left idle by a small grid do not slow the busiest one,
    // whose bytes are already in `mem`.
    g.warps              = std::min(q, std::max(occ, 1.0)) * k.threads / 32;
    const double bw      = chip.dram_bytes_per_s;
    const double bw_sm   = bw / chip.sms;
    const double peak_sm = chip.tensor_rate * chip.clock_hz;
    const double clk     = chip.clock_hz;
    t.mem = std::max(q * k.rows * kc * call.bpw / bw_sm, call.n * call.k * call.bpw / bw);
    t.mma = q * 2.0 * k.rows * call.sets * k.cols * kc / peak_sm;
    // The CTAs the busiest SM runs one residency after another each pay their warps' dependent
    // chain: the exposed load latency of a K step (16 K per warp, 150 cycles).
    g.rounds            = std::ceil(q / std::max(occ, 1.0));
    const double rounds = g.rounds;
    // The heavy decoders' bytes. The matrix crosses the DRAM bus once whatever its token tiles
    // (the CTAs of a weight row block after the first read it from the L2); every CTA pulls its
    // weights and its token tile of x through its SM's L2 port. The grid runs the row blocks
    // first (blockIdx.x) and the token tiles after (blockIdx.y), so the `occ` x `sms` CTAs
    // resident together span ceil(occ x sms / gx) token tiles (at most min(tiles, occ) on one
    // SM); the CTAs of one tile on an SM read it at the same K, so a residency round fetches each
    // of its tiles once for all of them. A spilled register goes through the L1 twice a K step.
    const double l2_sm = cal.l2_bytes_per_sm_clock * clk;
    t.dram             = call.n * call.k * call.bpw / bw;
    t.l2_weight        = q * kc * k.rows * call.bpw / l2_sm;
    const double per_round =
        std::min({tiles, std::max(occ, 1.0), q, std::max(1.0, std::ceil(occ * chip.sms / gx))});
    t.l2_activation = rounds * per_round * kc * (call.tokens / tiles) * 2.0 / l2_sm;
    t.spill         = q * k.threads * k.local_bytes * cal.spill_accesses_per_step *
              std::ceil(kc / (cal.rmma_heavy_step_k * k.warps_k)) /
              (cal.l1_bytes_per_sm_clock * clk);
    t.bytes                 = k.heavy ? t.dram + t.l2_weight + t.l2_activation + t.spill : t.mem;
    const double fill_warps = k.heavy ? cal.rmma_heavy_saturating_warps : cal.rmma_saturating_warps;
    t.reduce_cluster =
        q * (k.cluster - 1) * k.rows * call.sets * k.cols * cal.cluster_reduce_cycles / clk;
    t.reduce_warps =
        rounds * (k.warps_k - 1.0) * k.rows * call.sets * k.cols * cal.warp_reduce_cycles / clk;
    t.reduce = t.reduce_cluster + t.reduce_warps;
    t.chain  = rounds * std::ceil(kc / (cal.rmma_step_k * k.warps_k)) * cal.rmma_step_cycles / clk;
    g.fill   = std::clamp(g.warps / fill_warps, cal.min_warp_fill, 1.0);
    t.total  = (t.bytes + t.mma + t.reduce) / g.fill + t.chain;
    return t;
}

// ---- Pipelined tiles (tiled, staged) ----

// One shared-memory tile for one call: its output tile (route_tile: `rows` halved under SwiGLU)
// and the stage it buffers (`stage_rows` weight rows and `stage_cols` tokens, as the schedule
// declares them), at `tensor_fraction` of the tensor rate.
struct PipelinedKernel {
    double rows            = 0;
    double cols            = 0;
    double threads         = 0;
    double stages          = 0;
    double stage_rows      = 0;
    double stage_cols      = 0;
    double tensor_fraction = 1;
};

struct PipelinedTerms {
    Grid grid;
    double mem   = 0; // the busiest SM's DRAM share, or the whole matrix over the bus
    double mma   = 0; // the MMA work over the fraction of the tensor rate
    double chain = 0; // one stage's barrier and cp.async wait per stage and residency round
    double total = 0; // max(mem, mma) / fill + chain
};

// The CTAs one SM holds of a pipelined tile: its thread and CTA limits and the shared memory of
// its stages (capped at what one CTA may take).
[[nodiscard]] inline double pipelined_occupancy(const ChipFacts& chip, const Calibration& cal,
                                                const PipelinedKernel& k) {
    const double tile =
        std::min(k.stages * (k.stage_rows + k.stage_cols) * cal.pipelined_stage_bytes_per_row,
                 cal.max_tile_shared_bytes);
    return std::min({std::floor(chip.max_threads_per_sm / k.threads), chip.max_ctas_per_sm,
                     std::floor(chip.smem_per_sm / tile)});
}

[[nodiscard]] inline PipelinedTerms price_pipelined(const ChipFacts& chip, const Calibration& cal,
                                                    const Call& call, const PipelinedKernel& k) {
    const auto cdiv = [](double a, double b) { return std::ceil(a / b); };
    PipelinedTerms t;
    Grid& g              = t.grid;
    const double tiles   = cdiv(call.tokens, k.cols);
    g.ctas               = cdiv(call.n, k.rows) * tiles;
    const double q       = std::ceil(g.ctas / chip.sms);
    g.waves              = q;
    g.occupancy          = pipelined_occupancy(chip, cal, k);
    const double occ     = g.occupancy;
    g.warps              = std::min(q, std::max(occ, 1.0)) * k.threads / 32;
    const double bw      = chip.dram_bytes_per_s;
    const double bw_sm   = bw / chip.sms;
    const double peak_sm = chip.tensor_rate * chip.clock_hz;
    t.mem    = std::max(q * k.rows * call.k * call.bpw / bw_sm, call.n * call.k * call.bpw / bw);
    t.mma    = q * 2.0 * k.rows * call.sets * k.cols * call.k / peak_sm / k.tensor_fraction;
    g.rounds = std::ceil(q / std::max(occ, 1.0));
    t.chain  = g.rounds * std::ceil(call.k / cal.pipelined_stage_k) * cal.pipelined_stage_cycles /
              chip.clock_hz;
    g.fill  = std::clamp(g.warps / cal.pipelined_saturating_warps, cal.min_warp_fill, 1.0);
    t.total = std::max(t.mem, t.mma) / g.fill + t.chain;
    return t;
}

// ---- From a device class ----

// The chip facts of a class: its facts' reference clock and bandwidth (never the driver's
// clocks), its generation row's BF16 rate, its residency limits (a limit the query left at 0
// reads as 1, as the price always did).
[[nodiscard]] inline ChipFacts chip_facts_of(const DeviceClass& d) {
    ChipFacts c;
    c.sms                = d.sms;
    c.clock_hz           = d.facts.reference_clock_hz();
    c.dram_bytes_per_s   = d.facts.dram_bytes_per_s();
    c.tensor_rate        = generation_row(d.major, d.minor).bf16;
    c.max_threads_per_sm = std::max(d.max_threads_per_sm, 1);
    c.max_ctas_per_sm    = std::max(d.max_blocks_per_sm, 1);
    c.registers_per_sm   = d.registers_per_sm;
    c.smem_per_sm        = static_cast<double>(d.smem_per_sm);
    return c;
}

// Today's calibration of compute capability major.minor: the constants above, the L2 rate the
// generation was fitted with (gb20x its own, every other part GA10x's).
[[nodiscard]] inline Calibration calibration_of(int major, int minor) {
    Calibration cal;
    cal.l2_bytes_per_sm_clock = (gb20x_cc(major, minor) ? route_cost::kL2BytesPerSmClockGb20x
                                                        : route_cost::kL2BytesPerSmClockGa10x)
                                    .value();
    return cal;
}

} // namespace ninferno::ops::detail::generic::kernel_roofline
