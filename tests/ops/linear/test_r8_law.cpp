// The R8 decode schedule law (ops/generic/r8_law.h, #613 PR-10): each term against the count it
// is made of, on constructed facts and a facts board (the RTX 4090's). No GPU, no snapshot read:
// the law's own arithmetic, so a later change of a term is a deliberate edit here.

#include "core/device_class.h"
#include "core/isa_facts.h"
#include "core/format_facts.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/kernel_roofline.h"
#include "ops/generic/r8_decode_table.h"
#include "ops/generic/r8_law.h"
#include "ops/generic/route_authorities.h"
#include "ops/generic/route_query.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
    // Waits on global loads expose their latency: the loop's at DRAM's L0, the quantizer's and
    // the rest's at the ISA table's L2 hit (one warp alone); the outside count holds the quantizer
    // loop's once, which the law takes out.
    kf::KernelFacts waits                = f;
    waits.prologue->loop_global_waits    = 2;
    waits.prologue->global_waits         = 1;
    waits.prologue->outside_global_waits = 3;
    const auto tw                        = rl::terms(chip, *device, waits, call, {4, 1});
    const auto isa                       = ninferno::isa_row(device->major, device->minor);
    expect(isa && isa->l2_latency.w1.has_value(), "the rtx4090's ISA row has an L2 latency");
    if (tw && isa && isa->l2_latency.w1) {
        const double exposed =
            1 * 2 * *chip.dram_latency_s * chip.clock_hz + (1 * 1 + (3 - 1)) * *isa->l2_latency.w1;
        expect(near(tw->exposed_clocks, exposed), "exposed: waits x their latency");
        expect(near(tw->chain, t->rounds * (400.0 + 624.0 + 1100.0 + exposed) / chip.clock_hz),
               "chain adds the exposed latency, round after round");
    }
    expect(t->exposed_clocks == 0, "no waits on global loads, no exposed latency");
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
    // Eight tokens at K 12288 stage 8 x 384 x 40 = 122880 activation bytes: more shared memory
    // than an Ada SM has, so no CTA is resident. The pre-quantized form stages none.
    rl::Call verify = call;
    verify.nt       = 8;
    verify.k        = 12288;
    expect(!rl::terms(chip, *device, pre, verify, {4, 1}),
           "8 tokens at K 12288 staged: no CTA fits");
    verify.pre     = true;
    const auto tw8 = rl::terms(chip, *device, pre, verify, {4, 1});
    expect(tw8 && tw8->occupancy > 0, "the pre-quantized form stages no activation: CTAs fit");
    // More registers, fewer resident CTAs, more rounds for the same grid.
    const auto heavy = rl::terms(chip, *device, facts(209, 4096, 259, 624), call, {4, 1});
    expect(heavy && heavy->occupancy < t->occupancy && heavy->rounds >= t->rounds,
           "209 registers hold fewer CTAs than 77");
}

#ifndef NINFERNO_SOURCE_DIR
#    error "NINFERNO_SOURCE_DIR is defined by the test's CMake target"
#endif

// The formats of the frozen prediction files (r8_law_predict.cpp kFormats): their name there, the
// codec name their snapshot rows carry, the QType.
struct LawFormat {
    const char* name;
    const char* codec;
    ninferno::QType q;
};

constexpr LawFormat kLawFormats[] = {
    {"q4k", "W4a8Q45K<4>", ninferno::QType::GGML_Q4_K},
    {"q6k", "W4a8Q6K", ninferno::QType::GGML_Q6_K},
    {"q3k", "W4a8Q3K", ninferno::QType::GGML_Q3_K},
    {"iq2xxs", "W4a8Iq2Xxs", ninferno::QType::GGML_IQ2_XXS},
    {"iq2xs", "W4a8Iq2Xs", ninferno::QType::GGML_IQ2_XS},
    {"iq2s", "W4a8Iq2S", ninferno::QType::GGML_IQ2_S},
    {"iq3xxs", "W4a8Iq3Xxs", ninferno::QType::GGML_IQ3_XXS},
    {"iq3s", "W4a8Iq3S", ninferno::QType::GGML_IQ3_S},
};

std::string schedule_name(int i) {
    const g::R8ScheduleInfo s = g::r8_schedule_info(i);
    return "w" + std::to_string(s.warps) + "r" + std::to_string(s.rows);
}

// The launch names the kernel instance whose snapshot row the law reads (r8_kernel_key): the
// predictor's codec spelling, every schedule's warps and rows, one token slot, the epilogue, the
// in-kernel quantizer; and the pre-quantized verify form where asked.
void check_kernel_keys() {
    for (const LawFormat& f : kLawFormats) {
        for (int i = 0; i < g::r8_schedule_count(); ++i) {
            const g::R8ScheduleInfo s = g::r8_schedule_info(i);
            const auto key            = g::r8_kernel_key(i, f.q, g::Epilogue::SwiGlu, 1, false);
            expect(key.has_value(), std::string("no R8 kernel key: ") + f.name);
            if (!key) continue;
            expect(key->codec == f.codec && key->warps == s.warps && key->rows == s.rows &&
                       key->nt == 1 && key->epilogue == 2 && !key->pre,
                   std::string("R8 kernel key of ") + f.name + " " + schedule_name(i) + ": " +
                       key->codec + " " + std::to_string(key->warps) + " " +
                       std::to_string(key->rows) + " " + std::to_string(key->nt) + " " +
                       std::to_string(key->epilogue));
            const auto verify = g::r8_kernel_key(i, f.q, g::Epilogue::Store, 3, true);
            // Q3_K has no pre-quantized form (compiled_codecs.h PREQUANT false): no key.
            expect(g::r8_verify_compiled(f.q)
                       ? verify && verify->nt == 4 && verify->pre && verify->epilogue == 0
                       : !verify,
                   std::string("R8 verify kernel key of ") + f.name);
        }
    }
    // The host's key (r8_decoder_name's table, the launch's widths) is the launch's own, for every
    // format, schedule, epilogue, token count and form.
    int compared = 0;
    for (const ninferno::FormatFacts& f : ninferno::kFormatFacts) {
        for (int i = 0; i < g::r8_schedule_count(); ++i) {
            for (const g::Epilogue e :
                 {g::Epilogue::Store, g::Epilogue::AddResidual, g::Epilogue::SwiGlu}) {
                for (std::int32_t t = 1; t <= 9; ++t) {
                    for (const bool pre : {false, true}) {
                        const auto host   = g::r8_kernel_key(i, f.q, e, t, pre);
                        const auto launch = g::r8_launch_key(i, f.q, e, t, pre);
                        const bool same =
                            host.has_value() == launch.has_value() &&
                            (!host ||
                             (host->codec == launch->codec && host->warps == launch->warps &&
                              host->rows == launch->rows && host->nt == launch->nt &&
                              host->epilogue == launch->epilogue && host->pre == launch->pre));
                        expect(same,
                               std::string("the host's R8 kernel key is not the launch's: ") +
                                   std::string(f.name) + " " + schedule_name(i) +
                                   " T=" + std::to_string(t) + (pre ? " pre" : "") + ": host " +
                                   (host ? host->codec + " nt " + std::to_string(host->nt)
                                         : std::string("none")) +
                                   ", launch " +
                                   (launch ? launch->codec + " nt " + std::to_string(launch->nt)
                                           : std::string("none")));
                        compared += launch ? 1 : 0;
                    }
                }
            }
        }
    }
    expect(compared > 0, "no R8 launch named a kernel");
    std::cout << "R8 kernel keys: the host's equal the launch's on " << compared << " launches\n";
    expect(!g::r8_kernel_key(0, ninferno::QType::BF16, g::Epilogue::Store, 1, false),
           "a format without an R8 codec has a kernel key");
    expect(!g::r8_kernel_key(g::r8_schedule_count(), ninferno::QType::GGML_Q4_K, g::Epilogue::Store,
                             1, false),
           "an undeclared R8 schedule has a kernel key");
}

// The selector's law (law_r8) is the frozen prediction: on every board the law decides on, every
// call of its unread holdout grid's T = 1 file (tests/fixtures/r8_law, frozen at 9bdd25031) takes
// the schedule the file picks; where the law does not decide (GA10x, Turing, GB20x) it answers
// nothing; and the route's evidence names the law, a cost ranker, where it answered.
void check_selector_is_the_prediction() {
    struct HoldBoard {
        const char* name;
        std::uint32_t pci;
        bool decides;
    };

    constexpr HoldBoard kBoards[] = {
        {"rtx4090", 0x2684, true},   {"rtx4070ti", 0x2782, true}, {"rtx4060ti", 0x2805, true},
        {"gv100", 0x1DBA, true},     {"rtx3090", 0x2204, false},  {"rtx3060", 0x2504, false},
        {"rtx2080ti", 0x1E07, true}, {"rtx5090", 0x2B85, false},
    };
    const auto dir = std::filesystem::path(NINFERNO_SOURCE_DIR) / "tests/fixtures/r8_law";
    for (const HoldBoard& b : kBoards) {
        const ninferno::DeviceClass d =
            g::board_device_class(g::registry_facts_board(b.name, b.pci));
        expect(g::r8_law_generation(d) == b.decides,
               std::string("the law's generations: ") + b.name);
        const ninferno::ScopedDeviceClass scoped(d);
        if (!b.decides) {
            const g::Problem p{ninferno::QType::GGML_Q4_K, ninferno::ops::LinearPolicy::AllowA8,
                               4096, 4096, g::Epilogue::Store};
            expect(!g::law_r8(p), std::string("the law answered off its generations: ") + b.name);
            continue;
        }
        std::ifstream in(dir / ("predicted_v2_hold_" + std::string(b.name) + "_t1.csv"));
        expect(static_cast<bool>(in), std::string("no T=1 prediction file: ") + b.name);
        // (format, epilogue, n, k) -> the schedule the file picks.
        std::map<std::string, std::string> picks;
        for (std::string line; std::getline(in, line);) {
            std::vector<std::string> c;
            std::stringstream fields(line);
            for (std::string v; std::getline(fields, v, ',');) c.push_back(v);
            if (c.size() != 15 || c[14] != "1") continue;
            picks[c[0] + ' ' + c[1] + ' ' + c[2] + ' ' + c[3]] = c[4];
        }
        expect(!picks.empty(), std::string("no picks in the T=1 prediction: ") + b.name);
        std::size_t agree = 0;
        for (const auto& [call, pick] : picks) {
            std::stringstream fields(call);
            std::string format, epilogue;
            std::int32_t n = 0, k = 0;
            fields >> format >> epilogue >> n >> k;
            const auto* f = std::find_if(std::begin(kLawFormats), std::end(kLawFormats),
                                         [&](const LawFormat& x) { return format == x.name; });
            expect(f != std::end(kLawFormats), "an unknown format in a prediction: " + format);
            if (f == std::end(kLawFormats)) continue;
            const g::Epilogue e = epilogue == "swiglu" ? g::Epilogue::SwiGlu
                                  : epilogue == "add"  ? g::Epilogue::AddResidual
                                                       : g::Epilogue::Store;
            const g::Problem p{f->q, ninferno::ops::LinearPolicy::AllowA8, n, k, e};
            const auto law        = g::law_r8(p);
            const std::string got = law ? schedule_name(*law) : "none";
            expect(got == pick, std::string(b.name) + " " + call + ": the selector's law takes " +
                                    got + ", the frozen prediction " + pick);
            agree += got == pick ? 1 : 0;
        }
        std::cout << b.name << ": the law is the frozen T=1 prediction on " << agree << " of "
                  << picks.size() << " calls\n";
        // A call no measured table holds (none has 4608 x 3584) takes the law at one token, and at
        // two the law is not asked (the selector may route it off R8: the vec rule).
        const auto at = [&](std::int32_t t) {
            return g::route_answer_query(d, ninferno::QType::GGML_Q4_K,
                                         ninferno::ops::LinearPolicy::AllowA8, g::Epilogue::Store,
                                         4608, 3584, t);
        };
        const auto one = at(1);
        const auto two = at(2);
        expect(one && one->evidence.source == "r8-law" &&
                   one->evidence.kind == g::Evidence::Kind::Ranker &&
                   one->evidence.ranker == g::Evidence::Ranker::Cost,
               std::string("T=1 evidence is not the law's: ") + b.name + " " +
                   (one ? g::evidence_text(one->evidence) : std::string("none")));
        expect(two && two->evidence.source != "r8-law",
               std::string("T=2 evidence is the law's: ") + b.name + " " +
                   (two ? g::evidence_text(two->evidence) : std::string("none")));
    }
}

// How many entries of each measured decode table (r8_decode_table.h) on a generation the law
// decides on it reproduces (#613 PR-10: an entry the law reproduces can be retired), on the table's
// own board. Printed, not failed: the tables stay first, so a law that differs moves no call.
void count_reproduced_entries() {
    // The tables' boards by PCI id (r8_decode_table.h kR8DecodeTables: board_of_pci).
    constexpr std::pair<std::string_view, std::uint32_t> kTableBoards[] = {
        {"ad102:128sm", 0x2684}, {"ad103:80sm", 0x2702}, {"ad104:60sm", 0x2782},
        {"ad106:34sm", 0x2805},  {"tu102:68sm", 0x1E07}, {"ga102:82sm", 0x2204}};
    for (const g::R8DecodeTable& t : g::kR8DecodeTables) {
        const auto* b = std::find_if(std::begin(kTableBoards), std::end(kTableBoards),
                                     [&](const auto& x) { return x.first == t.gpu; });
        if (b == std::end(kTableBoards)) continue;
        const ninferno::DeviceClass d =
            g::board_device_class(g::registry_facts_board(std::string(t.gpu).c_str(), b->second));
        if (!g::r8_law_generation(d)) continue;
        const ninferno::ScopedDeviceClass scoped(d);
        std::size_t same = 0, none = 0;
        for (const g::R8DecodeEntry& e : t.entries) {
            const g::Problem p{e.qtype, ninferno::ops::LinearPolicy::AllowA8, e.n, e.k, e.epilogue};
            const auto law = g::law_r8(p);
            if (!law) {
                ++none;
                continue;
            }
            const g::R8ScheduleInfo s = g::r8_schedule_info(*law);
            same += s.warps == e.warps && s.rows == e.rows ? 1 : 0;
        }
        std::cout << t.gpu << ": the law reproduces " << same << " of " << t.entries.size()
                  << " decode table entries (" << none << " unanswered)\n";
    }
}

// A board whose class id is a decode table's but whose facts differ a little (a driver's clocks,
// another memory clock) takes its own table as measured here, distance 0, before the law
// (r8_hit_here); the same table's entry on another class is carried and yields to the law.
void check_table_here() {
    ninferno::DeviceClass d  = g::board_device_class(g::registry_facts_board("rtx4090", 0x2684));
    d.memory_clock_khz       = static_cast<int>(d.memory_clock_khz * 1.05);
    d.facts.memory_clock_khz = static_cast<int>(d.facts.memory_clock_khz * 1.05);
    // ad102:128sm's first entry: IQ2_S store 4096 x 2048, w4r1.
    const auto a =
        g::route_answer_query(d, ninferno::QType::GGML_IQ2_S, ninferno::ops::LinearPolicy::AllowA8,
                              g::Epilogue::Store, 4096, 2048, 1);
    expect(a && a->evidence.source == "r8-table" &&
               a->evidence.kind == g::Evidence::Kind::Measured && a->evidence.distance == 0 &&
               a->route.family == g::Family::R8Mma &&
               g::r8_schedule_info(a->route.schedule).warps == 4 &&
               g::r8_schedule_info(a->route.schedule).rows == 1,
           std::string("the RTX 4090 with other memory facts does not take its own table: ") +
               (a ? g::evidence_text(a->evidence) : std::string("none")));
}

} // namespace

int main() {
    check_terms();
    check_kernel_keys();
    check_selector_is_the_prediction();
    count_reproduced_entries();
    check_table_here();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "r8 law: OK\n";
    return 0;
}
