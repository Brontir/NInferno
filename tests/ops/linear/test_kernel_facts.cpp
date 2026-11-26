// The kernel-facts snapshots the register-MMA price reads (ops/generic/kernel_facts.h, #612), on
// any host, a CPU included: every committed snapshot parses; the release engine a board's
// capability runs is found by the dispatcher's rule; a type name is spelled as the tool spells
// it; and the price of every facts-priced golden board reads facts (the snapshot of its release
// engine), with the registers and threads a kernel has. No GPU: the whole point is that the
// decisions need none.
#include "core/device_class.h"
#include "core/kernel_occupancy.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::ops::detail::generic {
// A type in the generic namespace, as the decoders are: its canonical name drops the namespace.
template <int N>
struct KernelFactsTestDecoder {};
// An enumerator argument, as Ggml32Decoder has: the key spells its value, as cu++filt does.
enum class KernelFactsTestFormat { A, B, C };

template <KernelFactsTestFormat F>
struct KernelFactsEnumDecoder {
    static constexpr int kKeyArgument = static_cast<int>(F);
};
} // namespace ninferno::ops::detail::generic

namespace {

namespace generic = ninferno::ops::detail::generic;
namespace kf      = generic::kernel_facts;

int failures = 0;

void fail(const std::string& what) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

void expect(bool ok, const std::string& what) {
    if (!ok) fail(what);
}

void snapshots_parse() {
    const auto archs = kf::snapshot_archs();
    expect(!archs.empty(), "no kernel-facts snapshot is committed");
    for (const auto arch : archs) {
        const auto header = kf::snapshot_header(arch);
        expect(header.has_value(), "no header for sm_" + std::string(arch));
        if (!header) continue;
        std::cout << "sm_" << arch << ": " << header->kernels << " kernels, nvcc " << header->nvcc
                  << ", source " << header->source.substr(0, 12) << '\n';
        expect(header->kernels > 0, "sm_" + std::string(arch) + " holds no kernel");
        // The fingerprint's digest of the snapshot (#612 step 10): 64 hex digits, one per label.
        const std::string digest = kf::snapshot_sha256(arch);
        expect(digest.size() == 64 &&
                   digest.find_first_not_of("0123456789abcdef") == std::string::npos,
               "sm_" + std::string(arch) + ": the snapshot's sha256");
        for (const auto other : archs) {
            expect(other == arch || kf::snapshot_sha256(other) != digest,
                   "sm_" + std::string(arch) + " and sm_" + std::string(other) + ": one digest");
        }
        expect(!header->nvcc.empty() && header->nvcc != "unknown",
               "sm_" + std::string(arch) + " records no nvcc");
    }
    // A known kernel of every snapshot: the BF16 decoder on the four-warp K-split schedule, plain
    // store epilogue, compiled for every format and architecture.
    for (const auto arch : archs) {
        kf::RmmaKernelKey key;
        key.decoder = "Bf16Decoder";
        key.warps_m = 1;
        key.warps_k = 4;
        key.m_tiles = key.n_tiles = key.cluster = 1;
        const auto facts                        = kf::rmma_facts(arch, key);
        expect(facts.has_value(), "sm_" + std::string(arch) + ": no Bf16Decoder 1 4 1 1 1 row");
        if (facts) expect(facts->registers > 0, "sm_" + std::string(arch) + ": 0 registers");
    }
    expect(!kf::rmma_facts("no-such-arch", kf::RmmaKernelKey{}).has_value(),
           "an unknown arch has facts");
}

void release_engines() {
    // The lab boards' release engines (the dispatcher's rule over the committed snapshots): a
    // board runs the architecture-specific engine of its exact capability, else the newest plain
    // one of its major not above it. The expectations hold once the snapshot is committed; a
    // missing snapshot is named, not asserted (the test of a tree that lacks one still runs).
    struct Case {
        int major, minor;
        const char* engine;
    };

    const Case cases[] = {{7, 0, "70"}, {7, 5, "75"}, {8, 0, "80"},   {8, 6, "86"},
                          {8, 9, "89"}, {9, 0, "90"}, {10, 3, "100"}, {12, 0, "120a"}};
    for (const auto& c : cases) {
        const std::string got = kf::release_arch(c.major, c.minor);
        if (!kf::snapshot_header(c.engine)) {
            std::cout << "sm_" << c.engine << ": no snapshot committed (cc " << c.major << "."
                      << c.minor << " answers '" << got << "')\n";
            continue;
        }
        expect(got == c.engine, "cc " + std::to_string(c.major) + "." + std::to_string(c.minor) +
                                    " runs sm_" + got + ", expected sm_" + c.engine);
    }
    // A capability no engine's SASS runs (the driver compiles PTX): no facts.
    expect(kf::release_arch(6, 1).empty(), "cc 6.1 names an engine");
    // An sm_86 board never runs the sm_89 engine (a cubin runs on its major from its minor up).
    expect(kf::release_arch(8, 6) != "89", "cc 8.6 runs sm_89");
}

void canonical_names() {
    // cu++filt's spelling and the compiler's, to one form.
    expect(kf::canonical_type_name("ninferno::ops::detail::generic::GgmlQ45KQuadDecoder<(int)4>") ==
               "GgmlQ45KQuadDecoder<4>",
           "cu++filt spelling");
    expect(
        kf::canonical_type_name("struct ninferno::ops::detail::generic::GroupwiseG64Decoder<4>") ==
            "GroupwiseG64Decoder<4>",
        "MSVC spelling");
    expect(kf::canonical_type_name("ninferno::ops::detail::generic::Bf16Decoder") == "Bf16Decoder",
           "plain spelling");
    const std::string& name = kf::decoder_name<generic::KernelFactsTestDecoder<7>>();
    expect(name == "KernelFactsTestDecoder<7>", "decoder_name gave '" + name + "'");
    const std::string& by_value =
        kf::decoder_name<generic::KernelFactsEnumDecoder<generic::KernelFactsTestFormat::C>>();
    expect(by_value == "KernelFactsEnumDecoder<2>", "enum decoder_name gave '" + by_value + "'");
    std::cout << "decoder_name: " << name << " (raw '"
              << std::string(kf::raw_type_name<generic::KernelFactsTestDecoder<7>>()) << "')\n";
}

void boards_read_facts() {
    // Every facts-priced golden board reads facts from the snapshot of its release engine, on
    // this host (no GPU is needed; a GPU present is not asked), and what it reads is a kernel's:
    // registers and threads, the latency form only where the schedule has one.
    for (const auto& board : generic::kFactsBoards) {
        if (!generic::board_prices_from_facts(board)) continue;
        const std::string engine = kf::release_arch(board.major, board.minor);
        if (engine.empty() || !kf::snapshot_header(engine)) {
            fail(std::string(board.name) + ": no snapshot for its release engine sm_" + engine);
            continue;
        }
        std::vector<generic::RmmaFactRead> reads;
        std::size_t count = 0;
        try {
            const generic::ScopedRmmaFactsSink sink(reads);
            count = generic::selector_facts_probe(board);
        } catch (const std::exception& error) {
            fail(std::string(board.name) + ": " + error.what());
            continue;
        }
        int bad = 0;
        for (const auto& r : reads) {
            const auto s = generic::rmma_schedule_info(r.schedule);
            if (r.facts.registers <= 0 || r.facts.registers > 255 ||
                r.facts.threads != 32 * s.warps_m * s.warps_k || r.facts.binary_version == 0) {
                ++bad;
            }
        }
        expect(bad == 0, std::string(board.name) + ": " + std::to_string(bad) +
                             " reads with no kernel's registers or threads");
        std::cout << board.name << ": " << count << " facts read from sm_" << engine << '\n';
    }
}

void build_kernels_find_facts() {
    // Every register-MMA kernel this build launches finds its row in the build's own snapshot,
    // as the engine spells the key (decoder_name, rmma_kernel_key): a schedule the weight holds is
    // described, one it refuses is not, on a board of the build's architecture, no GPU asked. The
    // binary's kernel set is kernel_facts.py check's; this is the other side of the key.
    const std::string own(kf::build_arch());
    if (!kf::snapshot_header(own)) {
        std::cout << "build kernels: no snapshot for sm_" << own << "; skipped\n";
        return;
    }
    namespace qw        = ninferno::test::quantized_weight;
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    int boards = 0, described = 0;
    for (const auto& board : generic::kFactsBoards) {
        if (kf::release_arch(board.major, board.minor) != own) continue;
        ++boards;
        const ninferno::DeviceClass device = generic::board_device_class(board);
        const ninferno::ScopedDeviceClass scoped(device);
        for (int v = 0; v < 256; ++v) {
            const auto qtype = static_cast<ninferno::QType>(v);
            for (const auto [n, k] : {std::pair{4096, 4096}, std::pair{1024, 12288}}) {
                std::optional<qw::PackedWeight> host;
                try {
                    qw::PatternedWeightOptions options;
                    if (qtype == ninferno::QType::NVFP4)
                        options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
                    host = qw::make_patterned_weight(qtype, n, k, 7U, options);
                } catch (const std::exception&) { continue; } // no fixture for the value
                const ninferno::Weight w = host->device_weight(payload);
                if (!generic::supports(w)) continue;
                for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
                    const auto why  = generic::rmma_refusal(i, w);
                    const bool held = why != generic::ScheduleRefusal::NotCompiled &&
                                      why != generic::ScheduleRefusal::CodecTile &&
                                      why != generic::ScheduleRefusal::Format &&
                                      why != generic::ScheduleRefusal::Extent;
                    for (const std::int32_t t : {1, 4, 8}) {
                        const auto r = generic::rmma_resources(i, w, generic::Epilogue::Store, t);
                        // Without tensor mma.sync Volta's HMMA and GEMV kernels take held calls;
                        // on Volta only a refused call described is checked (a held call with no
                        // row is not told apart from one those kernels take).
                        if (r.has_value() != held && (r || device.features.mma_tensor)) {
                            fail(std::string(board.name) + " " + generic::qtype_name(qtype) +
                                 " schedule " + std::to_string(i) + " T=" + std::to_string(t) +
                                 ": described " + (r ? "yes" : "no") + ", refusal " +
                                 std::string(generic::schedule_refusal_name(why)));
                        }
                        described += r.has_value() ? 1 : 0;
                    }
                }
            }
        }
    }
    // A snapshot with no board, or boards whose every fixture failed, would check nothing.
    expect(boards > 0 && described > 0, "build kernels: nothing checked for sm_" + own);
    std::cout << "build kernels: " << described << " calls described on " << boards
              << " boards of sm_" << own << '\n';
}

void occupancy_from_facts() {
    // The pure occupancy on the synthetic boards: the documented cases of the register-MMA price
    // (a 16-row, 32-token K-quant tile at 229 registers keeps two 128-thread CTAs per SM on
    // sm_120, not the twelve the thread limit allows; generic_cost.cpp), and a kernel that does
    // not fit.
    const ninferno::DeviceClass sm120 =
        generic::board_device_class(generic::kFactsBoards[0]); // rtx5090
    expect(ninferno::resident_ctas_per_sm(sm120, {229, 0, 128, 0}) == 2, "229 registers x 128");
    expect(ninferno::resident_ctas_per_sm(sm120, {32, 0, 128, 0}) == 12, "32 registers x 128");
    expect(ninferno::resident_ctas_per_sm(sm120, {128, 0, 128, 64 * 1024}) == 0,
           "64 KiB of dynamic shared memory without the opt-in fits");
    expect(ninferno::resident_ctas_per_sm(sm120, {0, 0, 128, 0}) == 0, "unknown registers fit");
}

void loop_counts_read() {
    // The K loop's counts (#613 PR-3) of the BF16 decoder on the four-warp K-split
    // schedule, in every snapshot that carries them: one loop, two K steps of 4 * kRun = 32 values
    // (k_iter 64), one m16n8k16 MMA per 16 K per tile (4 per iteration) on a tensor-core build,
    // the weights read by global loads, and the total above the parts it is made of. A snapshot
    // without the columns (schema 1, `kernel_facts.py extract --no-sass`) fails: the tool writes
    // schema 4, so a schema 1 file is one nobody regenerated since the loop columns came (#758:
    // sm_80, sm_90 and sm_100 stayed so, because no stand builds those architectures and
    // ninferno_kernel_facts_test checks only a build's own; tools/dev/kernel_facts_archs.sh checks
    // them now, #762). kResourcesOnly names the snapshots whose loop counts cannot be written yet,
    // each with its issue.
    // None today: sm_70's emulated BF16 / F16 MMA calls count as their HMMA (#762).
    constexpr std::array<std::string_view, 0> kResourcesOnly{};
    kf::RmmaKernelKey key;
    key.decoder = "Bf16Decoder";
    key.warps_m = 1;
    key.warps_k = 4;
    key.m_tiles = key.n_tiles = key.cluster = 1;
    for (const auto arch : kf::snapshot_archs()) {
        const std::string name = "sm_" + std::string(arch);
        const auto facts       = kf::rmma_facts(arch, key);
        if (!facts) continue; // snapshots_parse reports it
        const bool waived = std::find(std::begin(kResourcesOnly), std::end(kResourcesOnly), arch) !=
                            std::end(kResourcesOnly);
        if (!facts->loop) {
            std::cout << name << ": no loop counts (schema 1)\n";
            expect(waived, name + ": a schema 1 snapshot (no loop counts): regenerate it from a "
                                  "build of the architecture (ninja ninferno_kernel_facts_update)");
            continue;
        }
        expect(!waived, name + ": carries loop counts, take it off kResourcesOnly");
        const kf::LoopCounts& l = *facts->loop;
        std::cout << name << ": Bf16Decoder 1 4 1 1 1 loop: " << l.mma << ' ' << l.mma_shape
                  << ", k_iter " << l.k_iter << ", " << l.ldg << " LDG (" << l.ldg_bytes << " B), "
                  << l.total << " instructions\n";
        expect(l.loops == 1, name + ": " + std::to_string(l.loops) + " candidate loops");
        expect(l.total > 0 && l.total >= l.mma + l.ldg + l.ld + l.int_alu + l.fp_alu,
               name + ": the total is below its parts");
        expect(l.ldg > 0 && l.ldg_bytes == 32 * 16 * static_cast<std::size_t>(l.ldg),
               name + ": the BF16 weights are 16-byte global loads");
        if (arch == "70") {
            // Volta's BF16 product is a call to the emulated m16n8k16 (mma_bf16_fma_call, FP32
            // FMAs inside the callee), counted as that HMMA (sass_counts.py EMULATED_MMA, #762).
            expect(l.mma_shape == "HMMA.16816.F32.BF16.EMU",
                   name + ": the emulated BF16 MMA, " + l.mma_shape);
        }
        expect(l.k_iter == 64, name + ": k_iter " + std::to_string(l.k_iter));
        // 64 K per iteration in m16n8k16 MMAs, or in Turing's m16n8k8 ones (HMMA.1688, twice as
        // many).
        const int per_k16 = l.mma_shape.find(".1688.") != std::string::npos ? 2 : 1;
        expect(l.mma == 4 * per_k16 && l.mma_ones == 0,
               name + ": " + std::to_string(l.mma) + " MMAs of " + l.mma_shape);
        expect(l.mma_shape.rfind("HMMA.", 0) == 0, name + ": shape " + l.mma_shape);
    }
    // A min-term decoder's loop counts its ones-MMAs apart, and they never make K.
    key.decoder = "GgmlQ45KQuadDecoder<4>";
    for (const auto arch : kf::snapshot_archs()) {
        const auto facts = kf::rmma_facts(arch, key);
        if (!facts || !facts->loop || arch == "70") continue;
        const kf::LoopCounts& l = *facts->loop;
        expect(l.mma_ones > 0 && l.mma_ones < l.mma && l.k_iter == 256,
               "sm_" + std::string(arch) + ": Q4_K ones-MMAs " + std::to_string(l.mma_ones) +
                   " of " + std::to_string(l.mma) + ", k_iter " + std::to_string(l.k_iter));
    }
}

void schedules_read() {
    // The static schedule (schema 4, #613 PR-C) of every row of every snapshot that carries it,
    // within the bounds a kernel's control codes allow: every issued instruction stalls at least
    // one clock, so a stall sum is at least its instruction count; the waits, the local waits and
    // the barriers are instructions of their part; a found loop has instructions outside it (the
    // prologue and the store at least). Every row of a snapshot with a loop has a schedule (the
    // reader refuses a schema 2 one, see old_columns_refused). No value is pinned: a recompile
    // moves them.
    for (const auto arch : kf::snapshot_archs()) {
        const std::string name = "sm_" + std::string(arch);
        const auto rows        = kf::rmma_rows(arch);
        expect(rows.size() == kf::snapshot_header(arch)->kernels,
               name + ": " + std::to_string(rows.size()) + " rows read");
        int with = 0, without = 0, bad = 0;
        std::string first;
        for (const auto& f : rows) {
            if (!f.loop) continue;
            if (!f.loop->schedule) {
                ++without;
                continue;
            }
            ++with;
            const kf::LoopCounts& l     = *f.loop;
            const kf::StaticSchedule& d = *l.schedule;
            const bool ok =
                d.stall_sum >= l.total && 0 <= d.sb_waits_local && d.sb_waits_local <= d.sb_waits &&
                d.sb_waits <= l.total && d.outside_stall >= d.outside_total &&
                0 <= d.outside_waits && d.outside_waits <= d.outside_total && 0 <= d.outside_bar &&
                d.outside_bar <= d.outside_total && (l.loops == 0 || d.outside_total > 0);
            if (!ok && bad++ == 0) {
                first = "stall_sum " + std::to_string(d.stall_sum) + " total " +
                        std::to_string(l.total) + " outside " + std::to_string(d.outside_total);
            }
        }
        expect(without == 0,
               name + ": " + std::to_string(without) + " loop rows without a schedule");
        expect(bad == 0,
               name + ": " + std::to_string(bad) + " schedules out of bounds, first " + first);
        std::cout << name << ": " << rows.size() << " rows, " << with << " with a schedule\n";
    }
}

void sections_read() {
    // The section families (#785): ggml's matrix kernel and the W4A8 tiles, in every snapshot
    // that has their sections (no price reads them yet). Every row parses with its schedule and
    // pipe counts within a kernel's bounds; and the rows the launch laws of #785 read are there,
    // counted as the unit they price: Q4_K, Q5_K and Q6_K at J 32 and 128, one or two candidate
    // loops (the stream-k body holds the unit's loop twice), four CTA barriers per iteration and
    // k_iter 32 J (a 128-row x J-column tile over 256 K on 8 warps, per m16n8 tile); the int8 MMA
    // the generation has (Turing's m8n8k16, m16n8k32 from sm_80 on); and the W4A8 32 x 128 tile
    // (Q4_K, Q6_K) with one loop, two barriers per 128-value K tile and k_iter 16 x 32.
    // sm_70's snapshot (CUDA 12.8, regenerated in full from a GV100 build, #762) has every
    // section: its mmq rows with their resources alone (ggml's matrix kernel runs from sm_75 on,
    // route_requires, so those instances are never launched and RESOURCES_ONLY writes no loop),
    // its i8 rows from an empty build (as sm_75's: generic_i8.cu compiles the tiles empty below
    // sm_80), and the routed-expert families. sm_100 has no moer8 / moegemm section yet (six of
    // its W4a8Nvfp4 GateUp rows read as unknown flow; no lab board is an sm_100 part).
    constexpr std::array<std::string_view, 0> kWithoutSections{};
    constexpr std::string_view kWithoutMoe[]      = {"100"};
    int with_sections                             = 0;
    for (const auto arch : kf::snapshot_archs()) {
        const std::string name = "sm_" + std::string(arch);
        for (const auto family : kf::kSectionFamilies) {
            const bool moe   = family == "moer8" || family == "moegemm";
            const bool volta = family == "vhmma" || family == "vgemv" || family == "vgemvr";
            const bool waived =
                volta ? arch != "70"
                : moe
                    ? std::find(std::begin(kWithoutMoe), std::end(kWithoutMoe), arch) !=
                          std::end(kWithoutMoe)
                    : family != "gemma4" && family != "r8" && family != "w4a8" && family != "vec" &&
                          std::find(std::begin(kWithoutSections), std::end(kWithoutSections),
                                    arch) != std::end(kWithoutSections);
            const auto header = kf::section_header(arch, family);
            expect(header.has_value() != waived,
                   name + (waived ? ": has a " : ": no ") + std::string(family) + " section");
            if (!header) continue;
            const auto rows = kf::section_rows(arch, family);
            expect(rows.size() == header->kernels && !rows.empty(),
                   name + " " + std::string(family) + ": " + std::to_string(rows.size()) +
                       " rows read, the section says " + std::to_string(header->kernels));
            expect(!header->nvcc.empty() && header->nvcc != "unknown",
                   name + " " + std::string(family) + " records no nvcc");
            int bad = 0;
            std::string first;
            for (const auto& [key, f] : rows) {
                if (!f.loop) continue;
                const kf::LoopCounts& l = *f.loop;
                const bool ok =
                    l.schedule && l.pipes && l.schedule->stall_sum >= l.total &&
                    l.schedule->sb_waits <= l.total && l.pipes->bar <= l.total &&
                    l.pipes->i2f + l.pipes->i2fp + l.pipes->ffma + l.pipes->fmul + l.pipes->fadd <=
                        l.fp_alu &&
                    l.pipes->imad <= l.int_alu &&
                    l.pipes->sb_waits_global + l.pipes->sb_waits_shared <= l.schedule->sb_waits;
                if (!ok && bad++ == 0) first = key;
            }
            expect(bad == 0, name + " " + std::string(family) + ": " + std::to_string(bad) +
                                 " rows out of bounds, first " + first);
            std::cout << name << ' ' << family << ": " << rows.size() << " rows, nvcc "
                      << header->nvcc << '\n';
        }
        // The mmq rows where the arch has them; the w4a8 and r8 checks below run either way.
        const bool mmq_section = kf::section_header(arch, "mmq").has_value();
        if (mmq_section) ++with_sections;
        for (const int type : {12, 13, 14}) {
            if (!mmq_section || arch == "70") break; // sm_70's rows: resources alone (above)
            for (const int J : {32, 128}) {
                const auto f = kf::mmq_facts(arch, {type, J, false, 30});
                const std::string what =
                    name + " mmq type " + std::to_string(type) + " J " + std::to_string(J);
                expect(f && f->loop, what + ": no row with a loop");
                if (!f || !f->loop) continue;
                const kf::LoopCounts& l = *f->loop;
                expect(l.loops == 1 || l.loops == 2,
                       what + ": " + std::to_string(l.loops) + " loops");
                expect(l.pipes && l.pipes->bar == 4, what + ": not four barriers");
                expect(l.k_iter == 32 * J, what + ": k_iter " + std::to_string(l.k_iter));
                // Q6_K's 16-value scales take the k16 form from sm_80 on.
                const bool shape = arch == "75" ? l.mma_shape.rfind("IMMA.8816.", 0) == 0
                                                : l.mma_shape.rfind("IMMA.16832.", 0) == 0 ||
                                                      l.mma_shape.rfind("IMMA.16816.", 0) == 0;
                expect(l.mma > 0 && shape,
                       what + ": " + std::to_string(l.mma) + " MMAs of " + l.mma_shape);
            }
        }
        if (!kf::section_header(arch, "w4a8")) continue;
        for (const char* codec : {"W4a8Q45K<4>", "W4a8Q6K"}) {
            bool found = false;
            for (const auto& [key, f] : kf::section_rows(arch, "w4a8")) {
                if (key.rfind(std::string(codec) + " 32 128 2 4 0 ", 0) != 0 || !f.loop) continue;
                found                   = true;
                const kf::LoopCounts& l = *f.loop;
                expect(l.loops == 1 && l.k_iter == 16 * 32 && l.pipes && l.pipes->bar == 2,
                       name + " w4a8 " + key + ": loops " + std::to_string(l.loops) + ", k_iter " +
                           std::to_string(l.k_iter));
            }
            expect(found, name + " w4a8: no " + codec + " 32 x 128 store row with a loop");
        }
        // The R8 decode kernels (#613 PR-10): one loop marked by the dp4a, no MMA and no CTA
        // barrier in it, and one quantizer loop where the kernel quantizes (pre 0), none where it
        // reads a quantized activation; the Q4_K four-warp single-row decode kernel is among them.
        // The routed-expert R8 kernels (moer8, #619 step 11) are counted as the r8 rows, but for
        // what the compiler splits: their Down stage (key "... 2 0") quantizes in two loops (four
        // or five on sm_120a, nvcc 13), and sm_120a peels the dp4a loop of 72 of its pre-quantized
        // Up rows into two.
        if (!kf::section_header(arch, "r8")) continue;
        for (const std::string_view family : {"r8", "moer8"}) {
            if (!kf::section_header(arch, family)) continue;
            int r8_bad = 0;
            std::string r8_first;
            for (const auto& [key, f] : kf::section_rows(arch, family)) {
                const bool pre  = key.size() >= 2 && key.compare(key.size() - 2, 2, " 1") == 0;
                const bool down = family == "moer8" && key.size() >= 4 &&
                                  key.compare(key.size() - 4, 4, " 2 0") == 0;
                const int loops  = f.loop ? f.loop->loops : 0;
                const int quant  = f.prologue ? f.prologue->loops : -1;
                const bool split = family == "moer8" && arch == "120a";
                const bool up    = key.size() >= 4 && key.compare(key.size() - 4, 4, " 3 1") == 0;
                const bool ok    = f.loop && (loops == 1 || (split && up && loops == 2)) &&
                                f.loop->mma == 0 && f.loop->pipes && f.loop->pipes->bar == 0 &&
                                f.loop->k_iter > 0 && f.prologue &&
                                (pre    ? quant == 0
                                 : down ? (quant == 2 || (split && (quant == 4 || quant == 5)))
                                        : quant == 1);
                if (!ok && r8_bad++ == 0) r8_first = key;
            }
            expect(r8_bad == 0,
                   name + " " + std::string(family) + ": " + std::to_string(r8_bad) +
                       " rows without one barrier-free dp4a loop and its quantizer loop, first " +
                       r8_first);
        }
        // The routed-expert register-MMA kernels (moegemm): one K loop of MMAs, no CTA barrier.
        int gemm_bad = 0;
        std::string gemm_first;
        for (const auto& [key, f] : kf::section_rows(arch, "moegemm")) {
            const bool ok = f.loop && f.loop->loops == 1 && f.loop->mma > 0 && f.loop->pipes &&
                            f.loop->pipes->bar == 0 && f.loop->k_iter > 0;
            if (!ok && gemm_bad++ == 0) gemm_first = key;
        }
        expect(gemm_bad == 0, name + " moegemm: " + std::to_string(gemm_bad) +
                                  " rows without one barrier-free MMA loop, first " + gemm_first);
        expect(kf::r8_facts(arch, {"W4a8Q45K<4>", 4, 1, 1, 0, false}).has_value(),
               name + " r8: no W4a8Q45K<4> 4 1 1 0 0 row");
        // The I8 tiles (#613 PR-6): from sm_80 on every row has one K loop of int8 MMAs (one
        // cp.async stage an iteration); below, generic_i8.cu compiles them empty.
        if (!kf::section_header(arch, "i8")) continue;
        const bool i8_built = arch != "75" && arch != "70";
        int i8_bad          = 0;
        std::string i8_first;
        for (const auto& [key, f] : kf::section_rows(arch, "i8")) {
            const bool looped = f.loop && f.loop->loops == 1 && f.loop->mma > 0 &&
                                f.loop->mma_shape.rfind("IMMA.", 0) == 0;
            const bool ok = i8_built ? looped : !f.loop || f.loop->loops == 0;
            if (!ok && i8_bad++ == 0) i8_first = key;
        }
        // The reader's key: the 4x2-warp, 4-bit Store tile (three stages on sm_75's empty build,
        // four on sm_70's).
        const int i8_stages = arch == "75" ? 3 : 4;
        expect(kf::i8_facts(arch, {4, 2, i8_stages, 4, 0}).has_value(),
               name + " i8: no 4 2 " + std::to_string(i8_stages) + " 4 0 row");
        expect(i8_bad == 0, name + " i8: " + std::to_string(i8_bad) +
                                (i8_built ? " rows without one int8-MMA loop, first "
                                          : " rows with a loop in an empty build, first ") +
                                i8_first);
    }
    std::cout << "sections: " << with_sections << " snapshots with the mmq section\n";
    // The reader takes a section after the rows and refuses one that breaks its form.
    const std::string base = "arch 89\nnvcc 13.4.92\nkernels 0\ncolumns family decoder warps_m "
                             "warps_k m_tiles n_tiles cluster epilogue latency sums registers "
                             "stack local shared\n";
    const std::string head = "family mmq\nfamily_nvcc 13.4.92\nfamily_kernels 1\n"
                             "family_columns family type J fallback prec registers stack local "
                             "shared\n";
    const std::string row  = "mmq 12 32 0 30 128 0 0 0\n";
    expect(kf::snapshot_text_accepted(base + head + row), "a resources-only mmq section is read");
    expect(!kf::snapshot_text_accepted(base + head + "mmq 12 32 0 30 128 0 0\n"),
           "a short mmq row is refused");
    expect(!kf::snapshot_text_accepted(base + head + row +
                                       "rmma Bf16Decoder 1 4 1 1 1 0 0 0 48 0 0 1536\n"),
           "an rmma row inside a section is refused");
    expect(!kf::snapshot_text_accepted(base + head + row + head + row),
           "a repeated section is refused");
    expect(!kf::snapshot_text_accepted(base + head + "mmq q4 32 0 30 128 0 0 0\n"),
           "a text key in the mmq section is refused");
    expect(!kf::snapshot_text_accepted(base + head), "a section without its rows is refused");
}

void section_wait_columns() {
    // The wait classes (sb_waits_global, sb_waits_shared) follow the pipe counts in the mmq and
    // w4a8 sections only; the r8 section has its quantizer loop there instead (#613 PR-5, PR-10).
    // A header with the other family's columns is refused.
    const std::string base = "arch 89\nkernels 0\ncolumns family decoder warps_m warps_k m_tiles "
                             "n_tiles cluster epilogue latency sums registers stack local shared\n";
    const std::string loop =
        " loops k_iter mma mma_ones mma_shape ldg ldg_bytes ld ld_bytes ldgsts ldgsts_bytes lds "
        "lds_bytes sts sts_bytes local local_bytes shfl int fp uniform other total stall_sum "
        "sb_waits sb_waits_local outside_total outside_stall outside_waits outside_bar";
    const std::string pipes    = " i2f i2fp ffma fmul fadd imad bar";
    const std::string waits    = " sb_waits_global sb_waits_shared";
    const std::string prologue = " pro_loops pro_total pro_int pro_fp pro_ldg_bytes pro_sts_bytes "
                                 "pro_stall glb_waits outside_glb_waits pro_glb_waits";
    // A row's values past the resources: the loop (mma_shape is text), the schedule, the pipes.
    const auto values = [](int n) {
        std::string s;
        for (int i = 0; i < n; ++i) s += i == 4 ? " -" : " 1";
        return s;
    };
    const auto mmq = [&](const std::string& extra, int fields) {
        return base +
               "family mmq\nfamily_nvcc 13.0\nfamily_kernels 1\nfamily_columns family type "
               "J fallback prec registers stack local shared" +
               loop + pipes + extra + "\nmmq 12 32 0 30 128 0 0 0" + values(fields) + "\n";
    };
    const auto r8 = [&](const std::string& extra, int fields) {
        return base +
               "family r8\nfamily_nvcc 13.0\nfamily_kernels 1\nfamily_columns family codec "
               "warps rows nt epilogue pre registers stack local shared" +
               loop + pipes + extra + "\nr8 W4a8Q8 4 1 1 0 0 64 0 0 0" + values(fields) + "\n";
    };
    expect(kf::snapshot_text_accepted(mmq(waits, 30 + 7 + 2)), "an mmq section with its waits");
    expect(!kf::snapshot_text_accepted(mmq("", 30 + 7)), "an mmq section without the waits");
    expect(kf::snapshot_text_accepted(r8(prologue, 30 + 7 + 10)), "an r8 section, its prologue");
    expect(!kf::snapshot_text_accepted(r8(waits + prologue, 30 + 7 + 2 + 10)),
           "an r8 section with the mmq waits");
    // The w4a8 section carries the producer warps' loop after its waits (#613 PR-6); an mmq
    // section with it, and a w4a8 section without it, are refused.
    const std::string producer =
        " prod_loops prod_total prod_stall prod_sb_waits prod_glb_waits prod_shd_waits prod_int";
    const auto w4a8 = [&](const std::string& extra, int fields) {
        return base +
               "family w4a8\nfamily_nvcc 13.0\nfamily_kernels 1\nfamily_columns family codec "
               "block_m block_n warps_m warps_n epilogue flavor stages producers registers stack "
               "local shared" +
               loop + pipes + extra + "\nw4a8 W4a8Q6K 128 64 4 2 0 0 3 4 168 0 0 0" +
               values(fields) + "\n";
    };
    expect(kf::snapshot_text_accepted(w4a8(waits + producer, 30 + 7 + 2 + 7)),
           "a w4a8 section with its waits and producer loop");
    expect(!kf::snapshot_text_accepted(w4a8(waits, 30 + 7 + 2)),
           "a w4a8 section without the producer loop");
    expect(!kf::snapshot_text_accepted(mmq(waits + producer, 30 + 7 + 2 + 7)),
           "an mmq section with the producer loop");
    // The vec section carries the K loop's loads in flight after its waits (#1185); one without
    // them, with the waits alone or with them before the waits, is refused, as is an mmq section
    // with them.
    const std::string inflight = " ldg_peak ldg_peak_bytes";
    const auto vec             = [&](const std::string& extra, int fields) {
        return base +
               "family vec\nfamily_nvcc 13.0\nfamily_kernels 1\nfamily_columns family type T KS "
                           "gated bounded registers stack local shared" +
               loop + pipes + extra + "\nvec 12 5 0 0 0 40 0 0 0" + values(fields) + "\n";
    };
    expect(kf::snapshot_text_accepted(vec(waits + inflight, 30 + 7 + 2 + 2)),
           "a vec section with its waits and loads in flight");
    expect(!kf::snapshot_text_accepted(vec(waits, 30 + 7 + 2)),
           "a vec section without the loads in flight");
    expect(!kf::snapshot_text_accepted(vec(inflight + waits, 30 + 7 + 2 + 2)),
           "a vec section with the loads in flight before the waits");
    expect(!kf::snapshot_text_accepted(vec(waits + inflight, 30 + 7 + 2)),
           "a vec row short of its loads in flight");
    expect(!kf::snapshot_text_accepted(mmq(waits + inflight, 30 + 7 + 2 + 2)),
           "an mmq section with the loads in flight");
}

void old_columns_refused() {
    // The reader takes the full (schema 4) columns line and the resources-only one, and refuses
    // the schema 2 line (the loop's counts without the schedule): its rows would carry no
    // schedule, and the Python tool refuses it as well.
    const std::string resources = "family decoder warps_m warps_k m_tiles n_tiles cluster epilogue "
                                  "latency sums registers "
                                  "stack local shared";
    const std::string loop =
        " loops k_iter mma mma_ones mma_shape ldg ldg_bytes ld ld_bytes ldgsts "
        "ldgsts_bytes lds lds_bytes sts sts_bytes local local_bytes shfl int fp "
        "uniform other total";
    const std::string schedule = " stall_sum sb_waits sb_waits_local outside_total outside_stall "
                                 "outside_waits outside_bar";
    const auto text            = [](const std::string& columns) {
        return "arch 89\nkernels 0\ncolumns " + columns + "\n";
    };
    expect(kf::snapshot_text_accepted(text(resources + loop + schedule)),
           "the schema 4 columns line is read");
    expect(kf::snapshot_text_accepted(text(resources)), "the resources-only columns line is read");
    expect(!kf::snapshot_text_accepted(text(resources + loop)),
           "the schema 2 columns line (no schedule) is refused");
}

} // namespace

// The vector kernels' rows through vec_facts (#1033): every arch has one row per launchable
// instance (18 formats x 8 token counts x 6 forms, 4 ungated K and 2 gated, plus the 18 bounded
// twins at T 4, the runtime K, ungated: 882), and each row of the section, looked up by its key's
// fields through VecKernelKey, gives that row's resources. A swapped field in vec_facts' key, or a
// form the snapshot lacks, fails here and not at describe time.
void vec_rows_look_up() {
    constexpr std::size_t kVecInstances = 18 * 8 * 6 + 18;
    for (const auto arch : kf::snapshot_archs()) {
        const std::string name = "sm_" + std::string(arch);
        const auto rows        = kf::section_rows(arch, "vec");
        expect(rows.size() == kVecInstances, name + " vec: " + std::to_string(rows.size()) +
                                                 " rows, " + std::to_string(kVecInstances) +
                                                 " launchable instances");
        int looked = 0, twins = 0;
        for (const auto& [key, f] : rows) {
            std::istringstream in(key);
            int type = -1, t = -1, ks = -1, gated = -1, bounded = -1;
            in >> type >> t >> ks >> gated >> bounded;
            if (!in || (gated != 0 && gated != 1) || (bounded != 0 && bounded != 1)) {
                fail(name + " vec: the key \"" + key + "\" is not type T KS gated bounded");
                continue;
            }
            expect(ks == 0 || ks == 5120 || ks == 6144 || ks == 17408,
                   name + " vec " + key + ": a static K no instance has");
            expect(bounded == 0 || (t == 4 && ks == 0 && gated == 0),
                   name + " vec " + key + ": a bounded twin no launch takes");
            twins += bounded;
            const auto found = kf::vec_facts(arch, {type, t, ks, gated == 1, bounded == 1});
            expect(found && found->registers == f.registers &&
                       found->static_shared_bytes == f.static_shared_bytes &&
                       found->spill_bytes() == f.spill_bytes(),
                   name + " vec " + key + ": vec_facts gives another row");
            ++looked;
        }
        expect(twins == 18, name + " vec: " + std::to_string(twins) + " bounded twins, not 18");
        std::cout << name << " vec: " << looked << " rows looked up by key\n";
    }
}

int main() {
    try {
        snapshots_parse();
        release_engines();
        canonical_names();
        loop_counts_read();
        schedules_read();
        sections_read();
        vec_rows_look_up();
        section_wait_columns();
        old_columns_refused();
        occupancy_from_facts();
        boards_read_facts();
        build_kernels_find_facts();
    } catch (const std::exception& error) { fail(error.what()); }
    if (failures != 0) {
        std::cerr << "kernel facts: " << failures << " failures\n";
        return 1;
    }
    std::cout << "kernel facts: PASS\n";
    return 0;
}
