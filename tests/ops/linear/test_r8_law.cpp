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
#include "ops/generic/route_analyzer.h"
#include "ops/generic/route_authorities.h"
#include "ops/generic/route_query.h"
#include "ops/generic/selector_facts.h"
#include "ops/common/arch_limits.h"
#include "ops/generic/generic_i8_schedules.h"
#include "ops/generic/w4a8_unit.h"
#include "ops/route_pins.h"

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
    expect(rl::slots_of("89") == 2, "two slots on Ada");
    expect(rl::label_number("120a") == 120 && rl::label_number("89") == 89 &&
               rl::label_number("") == 0,
           "a SASS label's compute capability");
    expect(rl::slots_of("70") == 1 && rl::slots_of("120a") == 2 && rl::slots_of("75") == 2,
           "one slot below sm_75 only, read from the SASS label");
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

// #938: the A8 tiles' occupancy prices from the snapshot's compiled registers, and a key the
// snapshot lacks falls back to the hand count without a word (generic_cost.cpp `resident`). So a
// key the price builds names its instance exactly: each W4A8 schedule x decoder x epilogue that the
// snapshot compiles under some flavor or ring is found under the price's (a tile not compiled for a
// codec, as the producer-warp tile for most, is not offered); on every snapshot holding I8 kernels,
// each I8 schedule x code width x epilogue at the stages its arch compiles is there. And every row
// of either section is reached by exactly one price key: a key built wrong for a whole family (a
// decoder spelling, a block or epilogue mapping) finds no variant either, which only the counts
// catch.
void check_a8_occupancy_keys() {
    const g::Epilogue epilogues[] = {g::Epilogue::Store, g::Epilogue::AddResidual,
                                     g::Epilogue::SwiGlu};
    for (const std::string_view arch : kf::snapshot_archs()) {
        int have = 0, miss = 0;
        std::string first;
        for (const ninferno::FormatFacts& f : ninferno::kFormatFacts) {
            if (!f.codec) continue;
            const std::string_view dec = g::w4a8_decoder_name(*f.codec);
            if (dec.empty()) continue;
            for (int i = 0; i < g::w4a8_schedule_count(); ++i) {
                for (const g::Epilogue e : epilogues) {
                    const auto key = ninferno::w4a8_unit::kernel_key_of(i, dec, e, arch);
                    if (!key) continue;
                    if (kf::w4a8_facts(arch, *key)) {
                        ++have;
                        continue;
                    }
                    // Not this key: is the instance compiled under another flavor or ring (a key
                    // built wrong, the bug this checks), or not at all (a codec the tile is not
                    // compiled for, which route_admits never offers: skipped)?
                    bool compiled = false;
                    for (int flavor = 0; flavor <= 3 && !compiled; ++flavor) {
                        for (int stages = 0; stages <= 4 && !compiled; ++stages) {
                            kf::W4a8KernelKey other = *key;
                            other.flavor            = key->producers > 0 ? 0 : flavor;
                            other.stages            = key->producers > 0 ? stages : 0;
                            compiled                = kf::w4a8_facts(arch, other).has_value();
                        }
                    }
                    if (compiled && ++miss == 1) {
                        first = key->codec + " schedule " + std::to_string(i) + " epilogue " +
                                std::to_string(static_cast<int>(e));
                    }
                }
            }
        }
        expect(miss == 0,
               "sm_" + std::string(arch) + ": " + std::to_string(miss) +
                   " W4A8 keys the price builds name no compiled instance of their tile (" +
                   std::to_string(have) + " found; first: " + first + ")");
        if (const auto header = kf::section_header(arch, "w4a8")) {
            expect(static_cast<std::size_t>(have) == header->kernels,
                   "sm_" + std::string(arch) + ": the price's W4A8 keys reach " +
                       std::to_string(have) + " of the snapshot's " +
                       std::to_string(header->kernels) + " W4A8 rows");
        }
        // The I8 tiles: the stages the arch's block holds (generic_cost.cpp's key).
        const int arch_number = std::stoi(std::string(arch));
        int i8_have = 0, i8_miss = 0;
        std::string i8_first;
        for (int i = 0; i < g::i8_schedule_count(); ++i) {
            const g::I8ScheduleInfo si = g::i8_schedule_info(i);
            for (const int bits : {4, 5, 6}) {
                const int stages =
                    g::i8_stages(si.block_m, si.block_n, bits,
                                 ninferno::ops::shared_bytes_per_block_for_arch(arch_number));
                const int wn = si.block_n / 64, wt = si.threads / 32 / std::max(wn, 1);
                for (const g::Epilogue e : epilogues) {
                    const kf::I8KernelKey key{wt, wn, stages, bits, static_cast<int>(e)};
                    if (kf::i8_facts(arch, key)) {
                        ++i8_have;
                    } else if (++i8_miss == 1) {
                        i8_first = std::to_string(si.block_m) + "x" + std::to_string(si.block_n) +
                                   " bits " + std::to_string(bits) + " stages " +
                                   std::to_string(stages) + " epilogue " +
                                   std::to_string(static_cast<int>(e));
                    }
                }
            }
        }
        expect(i8_have == 0 || i8_miss == 0,
               "sm_" + std::string(arch) + ": " + std::to_string(i8_miss) +
                   " I8 keys the price builds are not in the snapshot (first: " + i8_first + ")");
        if (const auto header = kf::section_header(arch, "i8")) {
            expect(static_cast<std::size_t>(i8_have) == header->kernels,
                   "sm_" + std::string(arch) + ": the price's I8 keys reach " +
                       std::to_string(i8_have) + " of the snapshot's " +
                       std::to_string(header->kernels) + " I8 rows");
        }
    }
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
    // The host's key (w4a8_decoder_name's table, the launch's widths) is the launch's own, for
    // every format, schedule, epilogue, token count and form.
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
// call of its holdout grid's T = 1 file (tests/fixtures/r8_law, law v3: the DRAM window, #771)
// takes the schedule the file picks; where the law does not decide (GA10x, Turing, GB20x) it
// answers nothing; and the route's evidence names the law, a cost ranker, where it answered.
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
        std::ifstream in(dir / ("predicted_v3_hold_" + std::string(b.name) + "_t1.csv"));
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
            // A format the law is not taken on (r8_law_format: Q4_K) answers nothing.
            const std::string want = g::r8_law_format(f->q) ? pick : "none";
            expect(got == want, std::string(b.name) + " " + call + ": the selector's law takes " +
                                    got + ", the frozen prediction " + pick);
            agree += got == want ? 1 : 0;
        }
        std::cout << b.name << ": the law is the frozen T=1 prediction on " << agree << " of "
                  << picks.size() << " calls\n";
        // A call no measured table holds (none has 4608 x 3584) takes the law at one token, and at
        // two the law is not asked (the selector may route it off R8: the vec rule).
        const auto at = [&](std::int32_t t) {
            return g::route_answer_query(d, ninferno::QType::GGML_Q6_K,
                                         ninferno::ops::LinearPolicy::AllowA8, g::Epilogue::Store,
                                         4608, 3584, t);
        };
        const auto one = at(1);
        const auto two = at(2);
        // Where the engine's analyzer answers the board (route_analyzer.h device_routes: any
        // shape, not only the tables'), its answer stands above the law (route_authorities.cpp,
        // the authority "analyzer"); on a held chip or Volta the law answers.
        const g::Problem off_table{ninferno::QType::GGML_Q6_K, ninferno::ops::LinearPolicy::AllowA8,
                                   4608, 3584, g::Epilogue::Store};
        const bool analyzer = !ninferno::ops::route_analyzer::device_routes(off_table, 1).empty();
        expect(one && (analyzer ? one->evidence.source == "analyzer"
                                : one->evidence.source == "r8-law" &&
                                      one->evidence.kind == g::Evidence::Kind::Ranker &&
                                      one->evidence.ranker == g::Evidence::Ranker::Cost),
               std::string("T=1 evidence is not the ") + (analyzer ? "analyzer's: " : "law's: ") +
                   b.name + " " + (one ? g::evidence_text(one->evidence) : std::string("none")));
        if (analyzer) std::cout << b.name << ": the analyzer answers the off-table T=1 call\n";
        expect(two && two->evidence.source != "r8-law",
               std::string("T=2 evidence is the law's: ") + b.name + " " +
                   (two ? g::evidence_text(two->evidence) : std::string("none")));
    }
}

// No measured decode table (r8_decode_table.h) on a generation the law decides on holds an entry
// the law reproduces on the table's own board (#613 PR-10): such an entry moves no call (the law
// answers before a carried entry, with the same schedule), so it is retired
// (tools/bench/data/r8_law_retired.txt, r8_decode_table.py --retire). A law change that comes to
// reproduce another entry names it here; the entries it answers otherwise are counted.
void count_reproduced_entries() {
    // The tables' boards by PCI id (r8_decode_table.h kR8DecodeTables: board_of_pci).
    constexpr std::pair<std::string_view, std::uint32_t> kTableBoards[] = {
        {"ad102:128sm", 0x2684}, {"ad103:80sm", 0x2702}, {"ad104:60sm", 0x2782},
        {"ad106:34sm", 0x2805},  {"tu102:68sm", 0x1E07}, {"ga102:82sm", 0x2204}};
    for (const g::R8DecodeTable& t : g::kR8DecodeTables) {
        const auto* b = std::find_if(std::begin(kTableBoards), std::end(kTableBoards),
                                     [&](const auto& x) { return x.first == t.gpu; });
        if (b == std::end(kTableBoards)) {
            // Only a table this map names can be held; a table on a generation the law decides
            // on (its own board's, r8_law_generation) must be named.
            ninferno::DeviceClass on;
            on.major = t.board.major;
            on.minor = t.board.minor;
            expect(!g::r8_law_generation(on),
                   std::string(t.gpu) + ": a law generation's table the test does not map");
            continue;
        }
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
            if (s.warps == e.warps && s.rows == e.rows) {
                ++same;
                std::cerr << "FAIL: " << t.gpu << " " << ninferno::format_facts(e.qtype).name
                          << " n " << e.n << " k " << e.k
                          << ": the law reproduces this entry; retire it "
                          << "(tools/bench/data/r8_law_retired.txt)\n";
            }
        }
        expect(same == 0, std::string(t.gpu) + ": decode table entries the law reproduces");
        std::cout << t.gpu << ": the law answers " << t.entries.size() - none - same << " of "
                  << t.entries.size() << " decode table entries otherwise (" << none
                  << " unanswered)\n";
    }
}

// Every retired entry (tools/bench/data/r8_law_retired.txt) is still the law's answer on its
// table's board: the schedule the entry measured. A law change that moves a retired call fails
// here, where the table would have kept it (#613 PR-10, #974 review).
void check_retired_entries() {
    constexpr std::pair<std::string_view, std::uint32_t> kBoards[]    = {{"ad102:128sm", 0x2684},
                                                                         {"ad103:80sm", 0x2702},
                                                                         {"ad104:60sm", 0x2782},
                                                                         {"ad106:34sm", 0x2805},
                                                                         {"tu102:68sm", 0x1E07}};
    constexpr std::pair<std::string_view, ninferno::QType> kFormats[] = {
        {"GGML_IQ2_S", ninferno::QType::GGML_IQ2_S},
        {"GGML_IQ2_XS", ninferno::QType::GGML_IQ2_XS},
        {"GGML_IQ2_XXS", ninferno::QType::GGML_IQ2_XXS},
        {"GGML_IQ3_S", ninferno::QType::GGML_IQ3_S},
        {"GGML_IQ3_XXS", ninferno::QType::GGML_IQ3_XXS},
        {"GGML_Q3_K", ninferno::QType::GGML_Q3_K},
        {"GGML_Q6_K", ninferno::QType::GGML_Q6_K}};
    std::ifstream in(std::filesystem::path(NINFERNO_SOURCE_DIR) /
                     "tools/bench/data/r8_law_retired.txt");
    expect(in.good(), "tools/bench/data/r8_law_retired.txt reads");
    std::string line;
    int checked = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string gpu, qtype, epilogue;
        std::int32_t n = 0, k = 0;
        int warps = 0, rows = 0;
        fields >> gpu >> qtype >> epilogue >> n >> k >> warps >> rows;
        const auto* b = std::find_if(std::begin(kBoards), std::end(kBoards),
                                     [&](const auto& x) { return x.first == gpu; });
        const auto* q = std::find_if(std::begin(kFormats), std::end(kFormats),
                                     [&](const auto& x) { return x.first == qtype; });
        if (fields.fail() || b == std::end(kBoards) || q == std::end(kFormats) ||
            epilogue != "Store") {
            expect(false, "r8_law_retired.txt: a line this test cannot read: " + line);
            continue;
        }
        const ninferno::DeviceClass d =
            g::board_device_class(g::registry_facts_board(gpu.c_str(), b->second));
        const ninferno::ScopedDeviceClass scoped(d);
        const g::Problem p{q->second, ninferno::ops::LinearPolicy::AllowA8, n, k,
                           g::Epilogue::Store};
        const auto law  = g::law_r8(p);
        const bool same = law && g::r8_schedule_info(*law).warps == warps &&
                          g::r8_schedule_info(*law).rows == rows;
        expect(same, "retired " + line + ": the law no longer gives its measured schedule");
        ++checked;
    }
    expect(checked == 29, "29 retired entries checked, not " + std::to_string(checked));
    std::cout << "retired R8 table entries: the law gives all " << checked
              << " their measured schedule\n";
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

// The three T = 1 paths, each with its evidence, and heuristic_r8 (route_probe, the decode pin)
// taking the route's schedule on every one: this board's own table (the RTX 4090, Measured), a
// table carried from a nearby board where the law answers (an RTX 4090 D, ad102:114sm, 0.17 from
// the 4090: the law, a cost ranker, no donor), and a carried table where the law does not decide
// (an RTX 3090 Ti, ga102:84sm, 0.14 from the 3090, sm_86: Carried, the 3090 its donor).
void check_evidence_paths() {
    struct Path {
        const char* name;
        std::uint32_t pci;
        ninferno::QType q;
        std::int32_t n, k;
        g::Evidence::Kind kind;
        const char* source;
    };

    const Path paths[] = {
        {"rtx4090", 0x2684, ninferno::QType::GGML_IQ2_S, 4096, 2048, g::Evidence::Kind::Measured,
         "r8-table"},
        {"rtx4090d", 0x2685, ninferno::QType::GGML_IQ2_S, 4096, 2048, g::Evidence::Kind::Ranker,
         "r8-law"},
        {"rtx3090ti", 0x2203, ninferno::QType::GGML_IQ4_NL, 4096, 5120, g::Evidence::Kind::Carried,
         "r8-table"},
    };
    // Under evidence the analyzer's ranked list (routes/pick: the board's own, lists are never
    // carried) answers T = 1 ahead of the R8 table and law (route_authorities.cpp analyzer_entry);
    // legacy reads no list, so every R8 path is asked there. AD102 is held out of the lists (gen.py
    // HELD_CHIPS, #1211), so its two paths (the 4090's table, the 4090 D's law) are asked under
    // evidence as well. The 3090 Ti's carried path is asked under legacy only: under evidence
    // ga102-84sm.json's own list answers that call (the analyzer, r8.w4r2 first).
    for (const bool evidence : {false, true}) {
        const ninferno::ops::ScopedRoutePolicy policy(evidence ? ninferno::RoutePolicy::Evidence
                                                               : ninferno::RoutePolicy::Legacy);
        for (const Path& path : paths) {
            if (evidence && path.kind == g::Evidence::Kind::Carried) continue;
            const ninferno::DeviceClass d =
                g::board_device_class(g::registry_facts_board(path.name, path.pci));
            const auto a = g::route_answer_query(d, path.q, ninferno::ops::LinearPolicy::AllowA8,
                                                 g::Epilogue::Store, path.n, path.k, 1);
            const std::string at = std::string(path.name) + " " + std::to_string(path.n) + "x" +
                                   std::to_string(path.k) + (evidence ? " (evidence): " : ": ");
            expect(a && a->route.family == g::Family::R8Mma && a->evidence.kind == path.kind &&
                       a->evidence.source == path.source,
                   at + "evidence " + (a ? g::evidence_text(a->evidence) : std::string("none")));
            if (!a) continue;
            const bool carried = path.kind == g::Evidence::Kind::Carried;
            expect(carried == !a->evidence.donor.empty(),
                   at + "a donor exactly on the carried path: " + g::evidence_text(a->evidence));
            const ninferno::ScopedDeviceClass scoped(d);
            const g::Problem p{path.q, ninferno::ops::LinearPolicy::AllowA8, path.n, path.k,
                               g::Epilogue::Store};
            expect(g::heuristic_r8(p, 1) == a->route.schedule,
                   at + "heuristic_r8 takes " + schedule_name(g::heuristic_r8(p, 1)) +
                       ", the route " + schedule_name(a->route.schedule));
            // The carried entry stands against the law only where the law could not price it.
            if (carried) expect(!g::law_r8(p), at + "the law answered off its generations");
        }
    }
}

// law_r8_before: where the law decides (an RTX 4090 D: Ada, no table of its own), an entry carried
// from another board that names a schedule the law could not price (a spilling instance: Q6_K
// SwiGLU's w4r2 and w8r2 on sm_89) stands, and one naming the law's own pick yields to it; without
// a carried entry it is law_r8.
void check_law_before() {
    const ninferno::DeviceClass d =
        g::board_device_class(g::registry_facts_board("rtx4090d", 0x2685));
    const ninferno::ScopedDeviceClass scoped(d);
    const g::Problem p{ninferno::QType::GGML_Q6_K, ninferno::ops::LinearPolicy::AllowA8, 12288,
                       4096, g::Epilogue::SwiGlu};
    const auto law = g::law_r8(p);
    expect(law.has_value(), "rtx4090d Q6_K SwiGLU 12288x4096: the law does not answer");
    if (!law) return;
    expect(g::law_r8_before(p, std::nullopt) == law, "law_r8_before without a carried entry");
    expect(g::law_r8_before(p, *law) == law,
           "a carried entry naming the law's pick does not yield");
    int spilling = 0;
    for (int i = 0; i < g::r8_schedule_count(); ++i) {
        const auto key   = g::r8_kernel_key(i, p.qtype, p.epilogue, 1, false);
        const auto facts = key ? g::kernel_facts::r8_facts("89", *key) : std::nullopt;
        if (!facts || facts->spill_bytes() == 0) continue;
        ++spilling;
        expect(!g::law_r8_before(p, i),
               "a carried " + schedule_name(i) +
                   " the law could not price (it spills) yields to the law");
    }
    expect(spilling > 0,
           "no spilling Q6_K SwiGLU R8 instance on sm_89: the case this checks is gone");
}

} // namespace

int main() {
    check_terms();
    check_kernel_keys();
    check_a8_occupancy_keys();
    check_selector_is_the_prediction();
    count_reproduced_entries();
    check_retired_entries();
    check_table_here();
    check_evidence_paths();
    check_law_before();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "r8 law: OK\n";
    return 0;
}
