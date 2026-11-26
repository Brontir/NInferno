#pragma once

// The route analyzer of decode and verify widths (T 1-8): it prices every route of one call from
// counted chip and kernel facts, and the cheapest wins (analyzer rounds 15-20, owner 25.11: the
// one decision source at these widths, part of the engine, no Python on the decision path). It
// prices any shape the build's kernels take, so a model nobody measured is routed by the same law
// as a release model.
//
// Inputs, all compiled-in facts: the board's SMs, reference clock and DRAM bus (machine facts);
// its architecture's register, thread, block and shared-memory limits (arch_registry); the DRAM
// latency L0 and a warp's fill rate (sm_rate_facts); the launch facts; the MMA latency and cycles,
// ALU latencies and the L1 / L2 / shared latencies (isa_facts); the narrow-load facts (below);
// each kernel's registers, K step, MMAs, loads and their bytes, instruction counts and static
// stall clocks (kernel_facts, the build architecture's snapshot).
//
// The price (ns):
//   rmma      launch + rounds of the CTAs the SMs hold; a round = hypot(SM side, DRAM side).
//             SM side = K trips x tau + the code outside the loop + the wave term (each warp's K
//             step waits one K step's bytes of every row of the round at the bus rate, / warps_k);
//             tau = the warp's chain (its DRAM round trips, each a window of the bytes it keeps in
//             flight, + the loop's static stall clocks + an emulated MMA's latency on sm_70) or an
//             issue bound. DRAM side = the round's bytes / bus x (rows open on the busiest SM /
//             L1) ^ G.
//   r8        launch + pre-pass + rounds; a round = hypot(the warp's chain: dp4a loop stall clocks
//             + a DRAM window per global-wait generation, the quantizer's loop at T 1, the code
//             outside; the round's bytes / bus).
//   ggml_mmq  launch + pre-pass + waves of 128-row tiles; a unit = max(its MMA issue, its loop's
//             stall clocks) + bytes / bus; the stream-k fixup where it runs; the epilogue kernel.
//   ggml_vec  two launches (quantizer, product) + L0 + rounds of the persistent grid; a round =
//             hypot(SM side: the loop's chain, the issue bound and the load unit by the
//             narrow-load rule; the round's bytes / bus).
//   sm_70     every rmma.* route is the Volta substitute the launch takes (HMMA before GEMV),
//             priced from its own counted kernel (sections vhmma / vgemv / vgemvr).
// FITTED: Settings::g alone. Not priced: the W4A8, tiled and staged tiles.
//
// tools/route_pick/analyzer.py is the same law in Python, kept while
// ninferno_route_analyzer_parity_test proves the two equal on every cell; this file is the
// product's.

#include "core/format_facts.h"
#include "ops/generic/generic_gemm.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno {
struct DeviceClass;
struct GpuFactsRow;
} // namespace ninferno

namespace ninferno::ops::route_analyzer {

// How a launch is priced: a graph node whose floor grows with its CTAs (the product's decode
// graph), or the same under a stream launch (route_tune times stream launches: the exam).
enum class LaunchMode : std::uint8_t { Graph, Stream };

// ggml_vec's round: hypot(SM side, bytes / bus), their sum or their max.
enum class RoundForm : std::uint8_t { Hypot, Sum, Max };

// The analyzer's settings. Every field but `g` chooses a form, no number; the defaults are the
// shipped forms (analyzer.py P).
struct Settings {
    double g = 0.1240; // FITTED (round 9): rmma's DRAM slowdown exponent, rows open on an SM
    LaunchMode launch  = LaunchMode::Stream;
    bool narrow_loads  = true; // ggml_vec's loads by the narrow-load rule
    RoundForm vec_form = RoundForm::Hypot;
    // ggml_vec's SM side: the code outside the K loop, the issue bound, the load unit.
    bool vec_outside = true, vec_issue = true, vec_load = true;
    // ggml_vec.mma's SM side.
    bool mma_outside = true, mma_issue = true;
    bool second_kernel     = true; // a route's second kernel costs one more stream launch
    bool volta_substitutes = true; // sm_70: every rmma route is the Volta substitute it launches
};

// The board numbers and architecture facts a price reads. Built by card_of; nullopt where the
// architecture lacks a fact the price needs (no price is then better than a guess).
struct Card {
    int cc = 0;
    std::string arch; // the kernel-facts snapshot ("86", "120a")
    double sms = 0, clk = 0, bus = 0,
           bpc  = 0; // SMs, SM clocks a ns, DRAM bytes a ns, bytes an SM clock
    int regs_sm = 0, thr_sm = 0, blk_sm = 0;
    double l1 = 0, smem_sm = 0, smem_res = 0; // bytes
    // isa / rate / launch facts
    double l0 = 0, node = 0, mma_lat = 0, hmma = 0, lat = 0, fill = 0, l1lat = 0, l2lat_ns = 0;
    double int_cyc = 0, fp_cyc = 0, shfl = 0, smem_lat = 0, l1cyc = 0, l2bpc = 0, dramlat = 0;
    double l2trip = 0;
    std::optional<double> stream, imma_n, imma_w;
    bool pdl = false;
    std::vector<std::pair<double, double>> floor; // (CTAs, ns), sorted
    // the narrow-load facts: a warp load's issue cap (clk), a wavefront's clocks, the bank width
    // (B)
    double nr_cap = 0, nr_wf = 0;
    int nr_bank = 0;

    // The cluster facts by cluster size (sm_90+, isa_facts' cluster entry, #1228): one
    // cluster.sync() and rank 0's read of one other rank's partial of 1 / 2 weight sets (ns),
    // and the CTAs a launch of C-CTA clusters holds an SM (the probe card's slots over its SMs).
    // A size the entry lacks a fact of is not here: its cN schedules are not priced.
    struct ClusterSize {
        double sync = 0, rank1 = 0, rank2 = 0, slots_sm = 0;
    };

    std::optional<ClusterSize> cluster2, cluster8;

    [[nodiscard]] const ClusterSize* cluster(int C) const {
        const auto& o = C == 2 ? cluster2 : cluster8;
        return (C == 2 || C == 8) && o ? &*o : nullptr;
    }
};

[[nodiscard]] std::optional<Card> card_of(const GpuFactsRow& row);
// The device's board: its machine facts' row where the PCI id is known, else the driver values on
// the facts grid (machine_facts).
[[nodiscard]] std::optional<Card> card_of(const DeviceClass& device);

// One call: the weight's format, the epilogue, the output rows (SwiGLU: the gate rows), K and
// the tokens.
struct Call {
    QType qtype                        = QType::BF16;
    detail::generic::Epilogue epilogue = detail::generic::Epilogue::Store;
    std::int32_t n = 0, k = 0, tokens = 1;
};

// A route as the analyzer names it: "rmma.m1.n2.w1x4", "rmma.m1.n1.w1x8.c8", "r8.w4r1",
// "ggml_mmq", "ggml_vec" (and the lab's instances "ggml_vec.ks", "ggml_vec.loop", "ggml_vec.mma").
// ns of one call, or nullopt: no facts for the card, the format or the route's kernel, or a
// route the launch would not take as named.
[[nodiscard]] std::optional<double> price_ns(const Card& card, const Call& call,
                                             std::string_view route, const Settings& s = {});

// The routes the product can launch for the call that this law prices, sorted by name: ggml_vec
// (T <= 8), ggml_mmq (T >= 5, whole tiles), every rmma and r8 (T <= 8) schedule with a kernel
// facts row.
[[nodiscard]] std::vector<std::string> candidates(const Card& card, const Call& call,
                                                  const Settings& s = {});

struct Priced {
    std::string route;
    double ns = 0;
};

// Every priced candidate, cheapest first (ties: the earlier name): the list the engine walks,
// taking the first route its policy admits.
[[nodiscard]] std::vector<Priced> rank(const Card& card, const Call& call, const Settings& s = {});

// The chips whose decode calls the analyzer does not answer yet (the switch's gate: no card below
// -1 % end to end against the rules it replaces). AD102: rmma priced 12-16 % below its time at
// T 2-4 (#1211). AD106 / GA106: w2x2 priced 8-19 % above its time, w1x8 1-18 % below (the wave
// term, #1224). TU102 / TU104 / TU106: the activation-ring latency kernel priced 1.2-1.3x (#1183).
inline constexpr std::string_view kHeldChips[] = {"ad102", "ad106", "ga106",
                                                  "tu102", "tu104", "tu106"};

// The widths the analyzer answers: decode and verify calls.
inline constexpr std::int32_t kMaxTokens = 8;

// The engine's question (route_authorities.cpp analyzer_entry): the analyzer's routes of a decode
// or verify call on this thread's device (device_class()), cheapest first, cut after the first
// register-MMA schedule (an A16 route every policy admits, so no route past it is ever taken), as
// engine routes (a name this build does not compile is left out). Memoized per board, kernel
// snapshot and call: the facts are compiled in, so a build prices a call once. Empty for T outside
// 1-8, a held chip, the legacy route policy, a board or format without facts, or a device whose
// kernels are not the snapshot the price reads.
[[nodiscard]] std::vector<detail::generic::Route>
device_routes(const detail::generic::Problem& problem, std::int32_t tokens);

// The format's route-table name ("q4k"), as the analyzer's lists and the exam grids spell it.
[[nodiscard]] std::string_view format_name(QType qtype);
// The format of that name; nullopt for a name the law does not price.
[[nodiscard]] std::optional<QType> format_of_name(std::string_view name);

} // namespace ninferno::ops::route_analyzer
