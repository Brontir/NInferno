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
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
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

} // namespace

int main() {
    try {
        snapshots_parse();
        release_engines();
        canonical_names();
        occupancy_from_facts();
        boards_read_facts();
    } catch (const std::exception& error) { fail(error.what()); }
    if (failures != 0) {
        std::cerr << "kernel facts: " << failures << " failures\n";
        return 1;
    }
    std::cout << "kernel facts: PASS\n";
    return 0;
}
