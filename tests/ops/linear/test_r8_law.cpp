// The R8 decode schedule law (ops/generic/r8_law.h, #613 PR-10): each term against the count it
// is made of, on constructed facts and a facts board (the RTX 4090's). No GPU, no snapshot read:
// the law's own arithmetic, so a later change of a term is a deliberate edit here.

#include "core/device_class.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/r8_law.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>

namespace {

namespace g  = ninferno::ops::detail::generic;
namespace kf = g::kernel_facts;
namespace kr = g::kernel_roofline;
namespace rl = g::r8_law;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (ok) return;
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

bool near(double a, double b) { return std::fabs(a - b) <= 1e-9 * std::max(std::fabs(b), 1.0); }

// A dp4a loop of `k_iter` K values a row-token product, with the quantizer loop beside it: counts
// of the size of the sm_89 Q4_K w4r1 row (258 loop instructions, a 549-clock loop schedule), not
// its exact values; the outside counts hold the quantizer loop once, as the snapshot's do.
kf::KernelFacts facts(int registers, int k_iter, int total, int stall, bool prologue = true) {
    kf::KernelFacts f;
    f.registers = registers;
    kf::LoopCounts l;
    l.loops     = 1;
    l.k_iter    = k_iter;
    l.total     = total;
    l.int_alu   = total / 2;
    l.fp_alu    = total / 8;
    l.ldg_bytes = 1024;
    kf::StaticSchedule s;
    s.stall_sum     = stall;
    s.outside_total = 480;
    s.outside_stall = 1500;
    l.schedule      = s;
    l.pipes         = kf::PipeCounts{};
    f.loop          = l;
    if (prologue) {
        kf::PrologueCounts p;
        p.loops     = 1;
        p.total     = 120;
        p.int_alu   = 40;
        p.fp_alu    = 60;
        p.stall_sum = 400;
        f.prologue  = p;
    }
    return f;
}

std::optional<ninferno::DeviceClass> board(const char* name) {
    const auto it =
        std::find_if(std::begin(g::kFactsBoards), std::end(g::kFactsBoards),
                     [&](const g::FactsBoard& b) { return std::strcmp(b.name, name) == 0; });
    if (it == std::end(g::kFactsBoards)) return std::nullopt;
    return g::board_device_class(*it);
}

void check_terms() {
    const auto device = board("rtx4090");
    expect(device.has_value(), "the rtx4090 facts board");
    if (!device) return;
    const kr::ChipFacts chip = kr::chip_facts_of(*device);
    expect(!kr::legacy_dram(chip), "the rtx4090 has SM rate facts");
    if (kr::legacy_dram(chip)) return;

    rl::Call call;
    call.n                      = 4096;
    call.k                      = 4096;
    call.nt                     = 1;
    call.sums                   = true;
    call.weight_bytes_per_value = 144.0 / 256.0; // Q4_K
    const kf::KernelFacts f     = facts(77, 4096, 259, 624);

    // A missing fact gives no answer: the rules decide.
    expect(!rl::terms(chip, *device, facts(77, 4096, 259, 624, false), call, {4, 1}),
           "no answer without the quantizer's counts");
    expect(!rl::terms(chip, *device, facts(77, 0, 259, 624), call, {4, 1}),
           "no answer without k_iter");
    kr::ChipFacts legacy  = chip;
    legacy.dram_latency_s = std::nullopt;
    expect(!rl::terms(legacy, *device, f, call, {4, 1}), "no answer on a legacy-DRAM chip");
    kf::KernelFacts spills = f;
    spills.stack_bytes     = 8;
    expect(!rl::terms(chip, *device, spills, call, {4, 1}), "no answer for a kernel that spills");

    const auto t = rl::terms(chip, *device, f, call, {4, 1});
    expect(t.has_value(), "an answer with every fact");
    if (!t) return;
    // Grid and rounds: 1024 CTAs on 128 SMs, 8 a busy SM.
    expect(t->ctas == 1024 && t->waves == 8, "4096 rows in CTAs of 4: 1024, 8 a busy SM");
    expect(t->rounds == std::ceil(t->waves / t->occupancy), "rounds = ceil(q / c)");
    // The quantizer: 128 blocks over 128 threads, one pass; K 5120 needs a second, eight warps
    // take it in one again (the K > 1024 W step the plan names).
    expect(t->passes == 1, "K 4096 on four warps: one quantizer pass");
    rl::Call wide = call;
    wide.k        = 5120;
    const auto t5 = rl::terms(chip, *device, f, wide, {4, 1});
    const auto t8 = rl::terms(chip, *device, f, wide, {8, 1});
    expect(t5 && t5->passes == 2, "K 5120 on four warps: two passes");
    expect(t8 && t8->passes == 1, "K 5120 on eight warps: one pass");
    // Warp passes a CTA: 128 tasks (160 at K 5120) over 32 lanes, whatever the warps.
    expect(t->warp_passes == 4, "K 4096: four warp passes a CTA");
    expect(t5 && t8 && t5->warp_passes == 5 && t8->warp_passes == 5,
           "K 5120: five warp passes a CTA on four or eight warps");
    // The dp4a loop: two slots an iteration, the last one may stop after its first slot.
    expect(t->iterations == 1, "K 4096 at 4096 an iteration: one");
    expect(t5 && t5->iterations == 1.5, "K 5120 at 4096 an iteration: one and a half");
    rl::Call volta = wide;
    volta.slots    = 1;
    const auto t1  = rl::terms(chip, *device, f, volta, {4, 1});
    expect(t1 && t1->iterations == 2, "one slot: K 5120 at 4096 an iteration takes two");
    expect(rl::slots_of(*device) == 2, "two slots on Ada");
    // The DRAM term is the DRAM law's, on the CTA's bytes.
    const double cta_bytes = 4 * 4096 * 144.0 / 256.0;
    expect(near(t->cta_bytes, cta_bytes), "a CTA's bytes: rows x K x bytes per value");
    // A warp's bytes in flight: the loop's 1024 load bytes, under one iteration's 2304 weight
    // bytes.
    const auto d = kr::dram_terms(chip, 1024, t->occupancy, 4, cta_bytes, 1024, true);
    expect(near(t->dram, d.sm) && near(t->bus, d.bus), "dram and bus are dram_terms of the call");
    kf::KernelFacts lookups = f;
    lookups.loop->ldg_bytes = 4096;
    const auto tl           = rl::terms(chip, *device, lookups, call, {4, 1});
    expect(
        tl && near(tl->dram, kr::dram_terms(chip, 1024, t->occupancy, 4, cta_bytes, 2304, true).sm),
        "loads past one iteration's weight bytes hit the cache");
    // Issue: the busiest SM's warp instructions over four schedulers, the INT32 pipe's two clocks
    // where they take longer.
    // A CTA: four warp passes of the quantizer, four warps of one iteration and the rest outside
    // (480 less the quantizer's 120).
    const double per_cta = std::max(
        {4.0 * 120 + 4 * (259.0 + 360), 2.0 * (4 * 40 + 4 * 129), 1.0 * (4 * 60 + 4 * 32)});
    expect(near(t->issue, 8 * per_cta / (4 * chip.clock_hz)), "issue from the counts");
    // Chain: the static schedule of one warp, every round (outside 1500 less the quantizer's 400).
    expect(near(t->chain, t->rounds * (400.0 + 624.0 + 1100.0) / chip.clock_hz),
           "chain from the stall sums");
    expect(t->rank == std::max({t->dram, t->issue, t->chain}), "the rank is the longest SM term");
    expect(t->total == std::max(t->bus, t->rank), "the time is the rank or the bus");
    // SwiGLU doubles a CTA's bytes; a quantized activation (no quantizer loop) has no passes.
    rl::Call gate = call;
    gate.swiglu   = true;
    const auto ts = rl::terms(chip, *device, f, gate, {4, 1});
    expect(ts && near(ts->cta_bytes, 2 * cta_bytes), "SwiGLU: two row sets of bytes");
    kf::KernelFacts pre = f;
    pre.prologue->loops = 0;
    const auto tp       = rl::terms(chip, *device, pre, call, {4, 1});
    expect(tp && tp->passes == 0, "a quantized activation: no quantizer pass");
    // More registers, fewer resident CTAs, more rounds for the same grid.
    const auto heavy = rl::terms(chip, *device, facts(209, 4096, 259, 624), call, {4, 1});
    expect(heavy && heavy->occupancy < t->occupancy && heavy->rounds >= t->rounds,
           "209 registers hold fewer CTAs than 77");
}

} // namespace

int main() {
    check_terms();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "r8 law: OK\n";
    return 0;
}
