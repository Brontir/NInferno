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
// The model, in words: a tile's time on the busiest SM is its DRAM time plus its MMA work over the
// SM's tensor rate (the register MMA runs them in series; the pipelined tiles take the longer of
// the two at a fraction of the rate), the MMA and the L2 traffic slowed in proportion while the SM
// holds fewer warps than saturate it, plus the exposed dependent chain of its K steps and the
// reduction of a split K per residency round.
//
// The DRAM time (issue #690, dram_time below): every CTA's bytes over the chip's shared bus, or the
// busiest SM's residency rounds at the rate its resident warps reach, whichever is longer. The rate
// one SM reaches with W resident warps is min(P, W x b_w / L0) (Little's law), from the facts the
// bytes-in-flight probe measured once per architecture (core/sm_rate_facts.h): L0 the fetch time of
// one warp's 16-byte streaming loads alone on an SM, P the most bytes per SM clock one SM reaches
// (with the L1-largest carve-out for a kernel without shared memory, the shared-largest one for a
// kernel with some). b_w, the bytes a warp keeps in flight, is counted per kernel from its loop
// (the register MMA's two ring slots of a step's weight runs, the pipelined and A8 tiles' cp.async
// stages ahead), and held to b_w_max, the most one warp kept in flight in the probe, on the
// architectures where the probe saw one warp saturate (sm_70; elsewhere no cap). The rate
// depends on no other SM, so a board with more SMs, all else equal, is never priced slower for
// it. A chip whose architecture has no SM rate row (sm_80, sm_90, sm_100, sm_121 today), or whose
// row another card of it refutes (sm_120a: the RTX 5070 Ti), keeps the law #643 extracted
// (`legacy_dram`): the busiest SM's CTAs at bw / SMs, under the warp fill. That law bills idle SMs
// (#690) and stays until a probe row that holds exists for the architecture.

#include "core/device_class.h"
#include "core/generation.h"
#include "core/isa_facts.h"
#include "core/launch_facts.h"
#include "core/sm_rate_facts.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/rmma_counted_terms.h"
#include "ops/generic/rmma_geometry.h"
#include "ops/generic/route_cost_constants.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace ninferno::ops::detail::generic::kernel_roofline {

// What the price reads of the chip. Facts of the board (core/machine_facts.h), the generation
// (core/generation.h) and the architecture (the CUDA residency limits); chip_facts_of fills it
// from a DeviceClass.
struct ChipFacts {
    double sms                = 0;
    double clock_hz           = 0; // the reference SM clock
    double dram_bytes_per_s   = 0; // bus width x memory data rate
    // The architecture's SM DRAM rate facts (core/sm_rate_facts.h), absent where the table has
    // no row for it: the price then keeps the legacy DRAM law (see the top). L0: one warp's
    // 16-byte streaming loads alone on an SM, their fetch time in seconds; P: the most bytes per
    // SM clock one SM moves from DRAM, with the L1-largest carve-out (`free`) and with the
    // shared-largest one (`placed`). Absent too where another card of the architecture refutes
    // the row (its transfer check). sm_max_warp_bytes: b_w_max, the most bytes one warp keeps in
    // flight, only where the probe saw a warp saturate (warp_bytes_saturated); absent, no cap.
    std::optional<double> dram_latency_s;
    std::optional<double> sm_dram_bytes_per_clock_free;
    std::optional<double> sm_dram_bytes_per_clock_placed;
    std::optional<double> sm_max_warp_bytes;
    double tensor_rate        = 0; // the BF16 MMA rate per SM and clock (GenerationRow::bf16)
    double max_threads_per_sm = 0;
    double max_ctas_per_sm    = 0;
    double registers_per_sm   = 0;
    double smem_per_sm        = 0;
    int major                 = 0; // the architecture: its pipes' ISA facts (rmma_counted_terms.h)
    // The architecture's launch facts (core/launch_facts.h, #613 PR-8b), absent where its ISA row
    // has no launch entry or the entry lacks the fact: the graph node (one launch as a node of a
    // captured graph) and the CTA dispatch ramp's four measured points (ramp_points; ramp_from
    // reads them piecewise linear).
    std::optional<double> launch_node_s;
    std::optional<RampPoints> launch_ramp;
    // The architecture's dependent L2 load latency in SM clocks alone and under 2, 4 and 8 chasing
    // warps an SM (core/isa_facts.h l2_latency), absent without an ISA row: what Little's law reads
    // for the L2 rate a warp count reaches (l2_little_bytes_per_clock). Printed, not priced.
    IsaByWarps l2_latency_clk;
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
    // The register-MMA law (#613 PR-7a): false, the fitted spill (spill_accesses_per_step and
    // l1_bytes_per_sm_clock above) and price_register_mma's weight-only tensor term
    // (2 x rows x sets x cols x kc over the tensor rate); true, the terms counted from the
    // kernel's K loop (RegisterMmaKernel::loop) where it carries a usable one, its spill waits at
    // l1_hit_latency_cycles. The binary's law is route_cost::kRmmaCountedLaw; price_routes takes
    // rmma_counted_law() (a test's override).
    bool rmma_counted_law        = route_cost::kRmmaCountedLaw;
    double l1_hit_latency_cycles = route_cost::kL1HitLatencyCycles.value();
    // Pipelined tiles (tiled, staged).
    double pipelined_stage_cycles        = route_cost::kPipelinedStageCycles.value();
    double pipelined_stage_k             = route_cost::kPipelinedStageK.value();
    double pipelined_saturating_warps    = route_cost::kPipelinedSaturatingWarps.value();
    double pipelined_stage_bytes_per_row = route_cost::kPipelinedStageBytesPerRow.value();
    double max_tile_shared_bytes         = route_cost::kMaxTileSharedBytes.value();
    // Every tile.
    double min_warp_fill = route_cost::kMinWarpFill.value();
    // The completeness terms (#613 PR-8a, Completeness below): false, the price offers `total`
    // (what every golden holds) and the terms are only computed (`complete`, price_tool); true, it
    // offers total x the A16 scale + the terms. The binary's switch is
    // route_cost::kCompletenessTerms; price_routes takes completeness_terms() (a test's override).
    bool completeness_terms = route_cost::kCompletenessTerms;
};

// The call: an n x k projection of `tokens` tokens; `sets` 2 under SwiGLU (the weight has 2 n
// rows, every output row multiplies both); `bpw` the weight's payload bytes over n x k (under
// SwiGLU the gate and up bytes of an output row together); `residual` under AddResidual (the
// epilogue reads the output's old value before it writes the sum).
struct Call {
    double n      = 0;
    double k      = 0;
    double tokens = 0;
    double sets   = 1;
    double bpw    = 0;
    bool residual = false;
};

// ---- Completeness terms (#613 PR-8a) ----

// What every family's kernel costs beyond its tiles' bytes and MMA work, each counted or read from
// a fact, never fitted:
//  - output: the BF16 output written once, n x T x 2 bytes (route_traffic's output_bytes; SwiGLU
//    writes one row a gate and up pair, a cluster's K split reduces in distributed shared memory)
//    over the chip's DRAM bus, as projection_chain's projection_step_seconds prices it (#603);
//  - residual: AddResidual reads the old output first, the same bytes again;
//  - quant: the A8 activation quantizer (a8_candidates: its counted bytes over the bus and its own
//    launch); 0 for an A16 tile, which reads BF16 as it is;
//  - launch: the kernel's launch as a graph node plus the dispatch ramp of its grid, read piecewise
//    linear between the probe's four measured grids (core/launch_facts.h launch_seconds GraphNode
//    + ramp_seconds; past 1024 CTAs the last segment's slope, `launch_extrapolated`), where the
//    architecture's launch facts are known; unknown (`launch_known` false), it is 0 and says so: an
//    unknown fact is not given a number.
// The output and residual are additive, not under the warp fill: the epilogue is the CTA's tail,
// after its last MMA, and its bytes add to the bus. A term common to every route of a call moves no
// argmin among them; the quantizer and the second launch separate A8 from A16, and the ramp the
// grids. Computed always; offered only under Calibration::completeness_terms.
struct Completeness {
    double output_bytes = 0; // n x T x 2
    double output       = 0;
    double residual     = 0;
    double quant        = 0;
    double launch       = 0;
    bool launch_known        = false;
    bool launch_extrapolated = false; // the grid is past the probe's largest (1024 CTAs)

    [[nodiscard]] double sum() const { return output + residual + quant + launch; }
};

// One launch of a grid of `ctas` CTAs as a graph node: node + the dispatch ramp of the grid
// (ramp_from: piecewise linear between the measured points, extrapolated past the last and saying
// so); nullopt where either fact is unknown.
[[nodiscard]] inline std::optional<Ramp> launch_seconds_of(const ChipFacts& chip, double ctas) {
    if (!chip.launch_node_s || !chip.launch_ramp) return std::nullopt;
    const std::optional<Ramp> ramp = ramp_from(*chip.launch_ramp, ctas);
    if (!ramp) return std::nullopt;
    return Ramp{*chip.launch_node_s + ramp->seconds, ramp->extrapolated};
}

// The completeness terms of a call whose kernel launches `ctas` CTAs, with `quant` the quantizer's
// seconds (0 for an A16 tile).
[[nodiscard]] inline Completeness completeness(const ChipFacts& chip, const Call& call, double ctas,
                                               double quant = 0) {
    Completeness c;
    c.output_bytes                     = call.n * call.tokens * 2.0;
    c.output                           = c.output_bytes / chip.dram_bytes_per_s;
    c.residual                         = call.residual ? c.output : 0.0;
    c.quant                            = quant;
    const std::optional<Ramp> launch   = launch_seconds_of(chip, ctas);
    c.launch_known                     = launch.has_value();
    c.launch                           = launch ? launch->seconds : 0.0;
    c.launch_extrapolated              = launch && launch->extrapolated;
    return c;
}

// The L2 rate W resident warps reach by Little's law, W x b_w / l2_latency(W) bytes an SM clock,
// b_w the bytes a warp keeps in flight (the count the DRAM law reads), l2_latency(W) the ISA row's
// dependent L2 latency under the largest measured chase count at most W (1, 2, 4, 8 warps; past 8
// the 8-warp figure). Uncapped: no per-SM L2 port rate is a cited fact in this tree. nullopt
// without the latency or a warp. Printed beside the fitted kL2BytesPerSmClock (price_tool), not
// priced: the counted L2 form is PR-7b's, one law for the chain and the L2.
[[nodiscard]] inline std::optional<double>
l2_little_bytes_per_clock(const ChipFacts& chip, double warps, double warp_bytes) {
    if (!(warps >= 1.0) || !(warp_bytes > 0)) return std::nullopt;
    const IsaByWarps& l             = chip.l2_latency_clk;
    const std::optional<double> lat = warps >= 8 && l.w8   ? l.w8
                                      : warps >= 4 && l.w4 ? l.w4
                                      : warps >= 2 && l.w2 ? l.w2
                                                           : l.w1;
    if (!lat || !(*lat > 0)) return std::nullopt;
    return warps * warp_bytes / *lat;
}

// How a grid of CTAs sits on the chip: what every form of the price computes first.
struct Grid {
    double ctas      = 0; // of the whole call
    double waves     = 0; // CTAs on the busiest SM: ceil(ctas / SMs)
    double occupancy = 0; // CTAs one SM holds at once
    double rounds    = 0; // residency rounds on the busiest SM: ceil(waves / occupancy)
    double warps     = 0; // warps resident on a busy SM
    double fill      = 0; // the slowdown factor of too few warps, in (min_warp_fill, 1]
};

// Whether the chip is priced by the DRAM law of #690 (dram_time) or keeps the legacy one: the
// architecture has an SM rate row that no other card of it refutes (its L0 and both P).
[[nodiscard]] inline bool legacy_dram(const ChipFacts& chip) {
    return !chip.dram_latency_s || !chip.sm_dram_bytes_per_clock_free ||
           !chip.sm_dram_bytes_per_clock_placed;
}

// dram_time with the terms it took (price_tool prints them).
struct DramTerms {
    double bus         = 0; // every CTA's bytes over the chip's bandwidth
    double sm          = 0; // the busiest SM's residency rounds at its own rate
    double peak        = 0; // P of the carve-out at the reference clock, bytes per second
    double counted     = 0; // the bytes the kernel's loop keeps in flight a warp, by count
    double warp_bytes  = 0; // b_w: that, at most b_w_max where a warp saturates
    double round_rate  = 0; // R_sm of the busiest SM's first round: min(P, W x b_w / L0)
    double total       = 0; // max(bus, sm)
    bool bus_bound     = false;
    bool shared_carved = false; // P of the shared-largest carve-out
};

// The DRAM time of a grid of `ctas` CTAs, each moving `bytes_per_cta` over the DRAM bus, an SM
// holding `occupancy` of them at once, each of `warps_per_cta` warps keeping `counted_bytes` in
// flight by the count of its loop, with the shared-largest carve-out (`shared_carveout`: a kernel
// with shared memory) or the L1-largest one (a kernel with none) (issue #690). Two bounds, the
// longer one is the time:
//  - the bus: every CTA's bytes over the chip's bandwidth;
//  - the busiest SM: its q = ceil(ctas / SMs) CTAs run in residency rounds of `occupancy` (the last
//    one holds the rest), and a round of c CTAs moves c x bytes_per_cta at the rate its c x warps
//    reach, min(P, W x b_w / L0), P of the carve-out at the chip's reference clock, b_w the
//    counted bytes, at most b_w_max where the probe saw one warp saturate (sm_70, and sm_120a
//    whose row is refuted: the 4-load ring keeps x1.20 and x1.00 of the 2-load ring's b_w; the
//    others' x1.80-1.81 show no limit, and their counted bytes stand, b_w_max a lower bound of
//    the limit). The probe measured b_w_max with 16-byte ld.global loads; the cp.async of the
//    shared-memory tiles is taken to have the same per-warp limit (not measured).
// The bus bound reads no SM count; the busiest SM's rounds read it only through q, which does not
// grow with more SMs. A round's rate is its own SM's, never a share of the bus: SMs left idle by
// a small grid cost nothing, and the bus bound holds the grid that fills every SM. Requires the
// chip's SM rate facts (legacy_dram false).
[[nodiscard]] inline DramTerms dram_terms(const ChipFacts& chip, double ctas, double occupancy,
                                          double warps_per_cta, double bytes_per_cta,
                                          double counted_bytes, bool shared_carveout) {
    DramTerms t;
    t.shared_carved   = shared_carveout;
    t.counted         = counted_bytes;
    const double occ  = std::max(occupancy, 1.0);
    const double q    = std::ceil(ctas / chip.sms);
    const double full = std::floor(q / occ);
    const double last = q - full * occ;
    t.peak            = (shared_carveout ? chip.sm_dram_bytes_per_clock_placed.value()
                                         : chip.sm_dram_bytes_per_clock_free.value()) *
             chip.clock_hz;
    t.warp_bytes =
        chip.sm_max_warp_bytes ? std::min(counted_bytes, *chip.sm_max_warp_bytes) : counted_bytes;
    const double per_w = t.warp_bytes / chip.dram_latency_s.value();
    const auto rate    = [&](double c) { return std::min(t.peak, c * warps_per_cta * per_w); };
    const auto round   = [&](double c) { return c * bytes_per_cta / rate(c); };
    t.round_rate       = rate(std::min(q, occ));
    t.sm               = full * round(occ) + (last > 0 ? round(last) : 0.0);
    t.bus              = ctas * bytes_per_cta / chip.dram_bytes_per_s;
    t.bus_bound        = t.bus >= t.sm;
    t.total            = std::max(t.bus, t.sm);
    return t;
}

[[nodiscard]] inline double dram_time(const ChipFacts& chip, double ctas, double occupancy,
                                      double warps_per_cta, double bytes_per_cta,
                                      double counted_bytes, bool shared_carveout) {
    return dram_terms(chip, ctas, occupancy, warps_per_cta, bytes_per_cta, counted_bytes,
                      shared_carveout)
        .total;
}

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
    double m_tiles = 1;           // 16-row MMA tiles of a warp (RmmaScheduleInfo::m_tiles)
    // K values of one step of the decoder (rmma_step_values: 4 x kRun); no default, the DRAM law
    // of #690 refuses a kernel without one.
    double k_step = 0;
    // The CTA's static shared memory (the K warps' and the cluster's partials, the staged
    // activations, a codebook): any picks the shared-largest carve-out's P (dram_time).
    double shared_bytes = 0;
    std::optional<int> registers; // per thread
    double local_bytes = 0;       // the spill per thread
    bool heavy         = false;
    // The kernel's K loop counted off its SASS (kernel_facts::LoopCounts, #650), where the price
    // reads its compiled facts (the heavy form): under Calibration::rmma_counted_law its MMAs and
    // LDL / STL give the tensor and spill terms (rmma_counted_terms.h). Absent, or the law off:
    // the fitted terms.
    std::optional<kernel_facts::LoopCounts> loop;
};

// The static shared memory of a register-MMA instance of `decoder` (its kDecoderGeometry
// spelling, rmma_geometry_decoder) on a chip of `compute_capability`, per CTA: the K warps'
// partials, a cluster's (the cluster schedules launch only in the build with the NVFP4 kernels),
// the decoder's codebook and its staged SwiGLU activations (rmma_geometry.h
// rmma_decoder_static_shared_bytes, the rule the kernel is static_asserted against). Whether the
// launch takes the latency variant, the one that stages, is decided from the compiled kernels'
// facts and occupancy (rmma_latency_pays); the price takes the staging form where it may stage,
// a bound: its placed P is the lower one. The dynamic shared memory of the spread decoders
// (GemmArgs::spread_smem, sm_120-class builds, legacy DRAM law today) is not counted. Where the
// price reads the compiled facts (the heavy form) a caller passes their static_smem instead.
[[nodiscard]] inline double register_mma_static_shared_bytes(std::string_view decoder, int warps_m,
                                                             int warps_k, int m_tiles, int n_tiles,
                                                             int cluster, bool swiglu,
                                                             int compute_capability) {
    return static_cast<double>(geometry::rmma_decoder_static_shared_bytes(
        decoder, warps_m, warps_k, m_tiles, n_tiles, cluster, swiglu, true, compute_capability,
        true));
}

struct RegisterMmaTerms {
    Grid grid;
    // The DRAM law the price took: false the one of #690 (dram_time), true the legacy one (an
    // architecture without an SM rate row, legacy_dram).
    bool legacy = true;
    // The DRAM law's terms of `mem` (plain form) or `dram` (heavy form); empty under the legacy
    // law.
    DramTerms dram_law;
    // Bytes, in seconds. Heavy form: the matrix once over the DRAM bus, every CTA's weights and
    // token tile through its SM's L2 port, the spill through the L1; these add (a max of the DRAM
    // and the L2 time did worse on the 5090 and 4090). Plain form: every CTA's bytes over DRAM.
    // The DRAM time is dram_time's (in the heavy form a CTA reads its weights from DRAM for one of
    // the token tiles and from the L2 for the others: bytes_per_cta / tiles). Legacy: the matrix
    // once over the whole bus (heavy); the busiest SM's share of the bus for its CTAs' bytes, or
    // the whole matrix over the whole bus, whichever is longer (plain).
    double dram          = 0;
    double l2_weight     = 0;
    double l2_activation = 0;
    double spill         = 0;
    double mem           = 0;
    double bytes         = 0; // the form's sum
    // The MMA work over the tensor rate.
    double mma = 0;
    // Whether `mma` and `spill` are the counted law's (rmma_counted_terms.h: every MMA of the K
    // loop, the min-term ones included; its LDL / STL at the L1 latency over the warps a scheduler
    // holds) rather than the fitted ones.
    bool counted = false;
    // K split over a cluster and over the CTA's K warps.
    double reduce_cluster = 0;
    double reduce_warps   = 0;
    double reduce         = 0;
    // The warps' dependent chain of K steps per residency round (not slowed by the fill).
    double chain = 0;
    // The DRAM time + (the L2 and L1 bytes + mma) / fill + reduce + chain: the reduce is a
    // per-round latency, like the chain, and the DRAM rate already reads the resident warps
    // (dram_time). Legacy: (bytes + mma + reduce) / fill + chain.
    double total = 0;
    // The completeness terms of the tile's grid (Completeness; quant 0: an A16 tile), the launch
    // unknown and 0 where the architecture has no launch facts, and `complete` = total + output +
    // residual + quant + launch. Computed whatever Calibration::completeness_terms says.
    double output = 0, residual = 0, quant = 0, launch = 0;
    bool launch_known        = false;
    bool launch_extrapolated = false; // the grid's ramp past the probe's 1024 CTAs
    double complete          = 0;
    // Little's-law L2 rate of the busiest SM's resident warps (l2_little_bytes_per_clock) at the
    // counted bytes a warp keeps in flight, beside cal.l2_bytes_per_sm_clock: printed, not priced.
    std::optional<double> l2_little_bytes_per_clock;
};

// Writes the completeness terms `c` into a family's terms `t` and their sum with t.total.
template <class Terms>
void set_completeness(Terms& t, const Completeness& c) {
    t.output       = c.output;
    t.residual     = c.residual;
    t.quant        = c.quant;
    t.launch              = c.launch;
    t.launch_extrapolated = c.launch_extrapolated;
    t.launch_known = c.launch_known;
    t.complete     = t.total + c.sum();
}

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
    t.legacy             = legacy_dram(chip);
    const double bw      = chip.dram_bytes_per_s;
    const double bw_sm   = bw / chip.sms;
    const double peak_sm = chip.tensor_rate * chip.clock_hz;
    const double clk     = chip.clock_hz;
    const double cta_b   = k.rows * kc * call.bpw; // one CTA's weight bytes
    // b_w by count: the K loop fetches the next step's weight runs into the ring's second slot
    // before it computes this step's (generic_gemm.cuh), a step of a warp being its m_tiles x 16
    // rows (of each set) over k_step K values: a lane holds two runs of each set and m tile (its
    // row in either 8-row half, one quarter of the step's K each), so the warp's 32 lanes hold
    // 2 slots x m_tiles x 16 x k_step x bpw bytes (bpw holds both sets' bytes under SwiGLU; the
    // runs' scales and heads with the codes, so BF16 m1 Store is 2048 B, Q4_K 2304, Q6_K 3360).
    // The P of the carve-out the kernel's shared memory takes: the probe measured the two ends
    // only (none, the shared-largest), so any shared memory takes the lower, placed P, a bound.
    // In the heavy form a CTA reads its weights from DRAM for one of the token tiles and from the
    // L2 for the others.
    if (!t.legacy) {
        if (!(k.k_step > 0)) {
            throw std::invalid_argument(
                "kernel_roofline: a register-MMA kernel without its K step");
        }
        const double counted = 2.0 * k.m_tiles * 16.0 * k.k_step * call.bpw;
        t.dram_law = dram_terms(chip, g.ctas, occ, k.threads / 32, k.heavy ? cta_b / tiles : cta_b,
                                counted, k.shared_bytes > 0);
    }
    // (The legacy terms keep their operation order: the kernel-roofline golden holds them to the
    // bit.)
    t.mem = t.legacy ? std::max(q * k.rows * kc * call.bpw / bw_sm, call.n * call.k * call.bpw / bw)
                     : t.dram_law.total;
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
    // of its tiles once for all of them, and never more tiles than its q CTAs (the legacy law does
    // not cap that). With the counted law off, a spilled register goes through the L1 twice a K
    // step; under it, the loop's LDL / STL wait the L1 latency (below).
    const double l2_sm = cal.l2_bytes_per_sm_clock * clk;
    t.dram             = t.legacy ? call.n * call.k * call.bpw / bw : t.dram_law.total;
    t.l2_weight        = q * kc * k.rows * call.bpw / l2_sm;
    const double per_round =
        std::min({tiles, std::max(occ, 1.0), q, std::max(1.0, std::ceil(occ * chip.sms / gx))});
    const double reads = t.legacy ? rounds * per_round : std::min(rounds * per_round, q);
    t.l2_activation    = reads * kc * (call.tokens / tiles) * 2.0 / l2_sm;
    t.spill         = q * k.threads * k.local_bytes * cal.spill_accesses_per_step *
              std::ceil(kc / (cal.rmma_heavy_step_k * k.warps_k)) /
              (cal.l1_bytes_per_sm_clock * clk);
    // Under the counted law, the loop's own counts: every MMA of an iteration (the min-term ones
    // included) over the tensor rate, its LDL / STL waits at the L1 latency shared by the warps a
    // scheduler holds.
    if (cal.rmma_counted_law && k.loop && rmma_counted::usable(*k.loop)) {
        const auto c = rmma_counted::terms(*k.loop, q, k.threads / 32.0, kc, k.warps_k, peak_sm,
                                           g.warps, cal.l1_hit_latency_cycles, clk, chip.major);
        t.mma        = c.mma;
        t.spill      = c.spill;
        t.counted    = true;
    }
    t.bytes                 = k.heavy ? t.dram + t.l2_weight + t.l2_activation + t.spill : t.mem;
    const double fill_warps = k.heavy ? cal.rmma_heavy_saturating_warps : cal.rmma_saturating_warps;
    t.reduce_cluster =
        q * (k.cluster - 1) * k.rows * call.sets * k.cols * cal.cluster_reduce_cycles / clk;
    t.reduce_warps =
        rounds * (k.warps_k - 1.0) * k.rows * call.sets * k.cols * cal.warp_reduce_cycles / clk;
    t.reduce = t.reduce_cluster + t.reduce_warps;
    t.chain  = rounds * std::ceil(kc / (cal.rmma_step_k * k.warps_k)) * cal.rmma_step_cycles / clk;
    g.fill   = std::clamp(g.warps / fill_warps, cal.min_warp_fill, 1.0);
    if (t.legacy) {
        t.total = (t.bytes + t.mma + t.reduce) / g.fill + t.chain;
    } else if (k.heavy) {
        t.total = t.dram + (t.l2_weight + t.l2_activation + t.spill + t.mma) / g.fill + t.reduce +
                  t.chain;
    } else {
        t.total = t.mem + t.mma / g.fill + t.reduce + t.chain;
    }
    set_completeness(t, completeness(chip, call, g.ctas));
    if (k.k_step > 0) {
        t.l2_little_bytes_per_clock =
            l2_little_bytes_per_clock(chip, g.warps, 2.0 * k.m_tiles * 16.0 * k.k_step * call.bpw);
    }
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
    bool legacy = true; // the DRAM law the price took (RegisterMmaTerms::legacy)
    DramTerms dram_law; // the DRAM law's terms of `mem`; empty under the legacy law
    double mem   = 0;   // dram_time; legacy: the busiest SM's DRAM share, or the whole matrix
    double mma   = 0;   // the MMA work over the fraction of the tensor rate
    double chain = 0;   // one stage's barrier and cp.async wait per stage and residency round
    double total = 0;   // max(mem, mma / fill) + chain; legacy: max(mem, mma) / fill + chain
    // The completeness terms (RegisterMmaTerms' fields of the same names).
    double output = 0, residual = 0, quant = 0, launch = 0;
    bool launch_known        = false;
    bool launch_extrapolated = false;
    double complete   = 0;
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
    t.legacy             = legacy_dram(chip);
    const double bw      = chip.dram_bytes_per_s;
    const double bw_sm   = bw / chip.sms;
    const double peak_sm = chip.tensor_rate * chip.clock_hz;
    // b_w: the cp.async ring keeps stages - 1 K stages of the CTA's weight rows in flight while it
    // multiplies one, over the CTA's warps.
    if (!t.legacy) {
        const double in_flight =
            (k.stages - 1.0) * k.rows * cal.pipelined_stage_k * call.bpw / (k.threads / 32);
        t.dram_law = dram_terms(chip, g.ctas, occ, k.threads / 32, k.rows * call.k * call.bpw,
                                in_flight, true);
    }
    t.mem    = t.legacy
                   ? std::max(q * k.rows * call.k * call.bpw / bw_sm, call.n * call.k * call.bpw / bw)
                   : t.dram_law.total;
    t.mma    = q * 2.0 * k.rows * call.sets * k.cols * call.k / peak_sm / k.tensor_fraction;
    g.rounds = std::ceil(q / std::max(occ, 1.0));
    t.chain  = g.rounds * std::ceil(call.k / cal.pipelined_stage_k) * cal.pipelined_stage_cycles /
              chip.clock_hz;
    g.fill  = std::clamp(g.warps / cal.pipelined_saturating_warps, cal.min_warp_fill, 1.0);
    t.total = t.legacy ? std::max(t.mem, t.mma) / g.fill + t.chain
                       : std::max(t.mem, t.mma / g.fill) + t.chain;
    set_completeness(t, completeness(chip, call, g.ctas));
    return t;
}

// ---- From a device class ----

// The SM DRAM rate facts of a chip from its architecture's row of core/sm_rate_facts.h (L0 in
// seconds, P per SM clock in both carve-outs, b_w_max in bytes where a warp saturates, else no
// cap); left absent without a row, and
// where another card of the architecture refutes the row (the law of the row falls outside the
// check's bounds over that card's run: sm_120a, whose RTX 5070 Ti fetches 12 % faster a warp
// and moves 9 % more a clock in the placed carve-out than the RTX 5090 row says). A board of
// the architecture other than the row's card is priced by the row: a transfer, checked against
// every other card's run that is committed.
inline void fill_sm_rate_facts(ChipFacts& c, int major, int minor) {
    for (const SmRateRow& row : sm_rate_facts_table()) {
        if (row.major != major || row.minor != minor) continue;
        if (row.transfer_refuted) return;
        c.dram_latency_s                 = row.dram_stream_latency_ns * 1e-9;
        c.sm_dram_bytes_per_clock_free   = row.bytes_per_clock_free;
        c.sm_dram_bytes_per_clock_placed = row.bytes_per_clock_placed;
        if (row.warp_bytes_saturated) c.sm_max_warp_bytes = row.max_warp_bytes_in_flight;
        return;
    }
}

// The launch facts and the L2 latencies of compute capability major.minor from its ISA row (its
// launch entry: the graph node and the dispatch ramp's measured points, core/launch_facts.h), each
// left absent where the row or the fact is absent (the same figures as launch_seconds(GraphNode)
// and ramp_seconds read; the table is walked in place, since the price runs at every launch the
// selector prices).
inline void fill_isa_facts(ChipFacts& c, int major, int minor) {
    for (const IsaRow& row : isa_facts_table()) {
        if (row.major != major || row.minor != minor) continue;
        c.l2_latency_clk = row.l2_latency;
        if (row.launch) {
            if (row.launch->node_ns) c.launch_node_s = *row.launch->node_ns * 1e-9;
            c.launch_ramp = ramp_points(*row.launch);
        }
        return;
    }
}

// The chip facts of a class: its facts' reference clock and bandwidth (never the driver's
// clocks), its generation row's BF16 rate, its residency limits (a limit the query left at 0
// reads as 1, as the price always did), its architecture's SM DRAM rate facts, launch facts and
// L2 latencies.
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
    c.major              = d.major;
    fill_sm_rate_facts(c, d.major, d.minor);
    fill_isa_facts(c, d.major, d.minor);
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
