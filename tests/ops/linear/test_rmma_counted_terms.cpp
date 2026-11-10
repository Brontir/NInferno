// The register MMA's counted terms (ops/generic/rmma_counted_terms.h, #613 PR-7a) against the
// committed kernel-facts snapshots: the counted min-term share equals the one #598 reasoned from
// mma_step, (sets x m_tiles + 1) / (sets x m_tiles), on every min-term kernel that takes the MMA
// with ones, and the ones count is zero on every kernel that adds the sums or has no min term;
// the loop's iterations cover a CTA's K range; the terms of a constructed loop are the arithmetic
// written down. No GPU.
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/rmma_counted_terms.h"
#include "ops/generic/route_cost_constants.h"

#include <cmath>
#include <iostream>
#include <optional>
#include <string>

namespace {

namespace kf = ninferno::ops::detail::generic::kernel_facts;
namespace rc = ninferno::ops::detail::generic::rmma_counted;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

struct Schedule {
    int warps_m, warps_k, m_tiles, n_tiles, cluster;
};

constexpr Schedule kSchedules[] = {
#define NINFERNO_SCHEDULE(i, wm, wk, mt, nt, c) {wm, wk, mt, nt, c},
    NINFERNO_GENERIC_RMMA_SCHEDULES(NINFERNO_SCHEDULE)
#undef NINFERNO_SCHEDULE
};

// The decoders with a min term (MinTermOf in generic_gemm.cuh: rmma_has_min_term) and some
// without, as the snapshots spell them.
struct Decoder {
    const char* name;
    bool min_term;
};

constexpr Decoder kDecoders[] = {{"GgmlQ45KQuadDecoder<4>", true},
                                 {"GgmlQ45KQuadDecoder<5>", true},
                                 {"GgmlQ2KDecoder", true},
                                 {"Ggml32Decoder<1>", true},
                                 {"Ggml32Decoder<3>", true},
                                 {"Bf16Decoder", false},
                                 {"GgmlQ6KDecoder", false},
                                 {"GgmlQ3KDecoder", false},
                                 {"Ggml32Decoder<0>", false},
                                 {"GgmlIq4XsDecoder", false},
                                 {"Q8Decoder", false}};

void ones_share_on_the_snapshots() {
    int checked = 0, with_ones = 0;
    for (const auto arch : kf::snapshot_archs()) {
        const std::string name = "sm_" + std::string(arch);
        for (const Decoder& d : kDecoders) {
            const char* decoder = d.name;
            const bool min_term = d.min_term;
            {
                for (const Schedule& s : kSchedules) {
                    for (int epilogue = 0; epilogue < 3; ++epilogue) {
                        for (const bool latency : {false, true}) {
                            for (const bool add_sums : {false, true}) {
                                kf::RmmaKernelKey key;
                                key.decoder      = decoder;
                                key.warps_m      = s.warps_m;
                                key.warps_k      = s.warps_k;
                                key.m_tiles      = s.m_tiles;
                                key.n_tiles      = s.n_tiles;
                                key.cluster      = s.cluster;
                                key.epilogue     = epilogue;
                                key.latency      = latency;
                                key.add_sums     = add_sums;
                                const auto facts = kf::rmma_facts(arch, key);
                                if (!facts || !facts->loop) continue;
                                const kf::LoopCounts& l = *facts->loop;
                                if (arch == "70") continue; // FFMA loop: no tensor MMA
                                const std::string what =
                                    name + " " + decoder + " " + std::to_string(s.warps_m) + "x" +
                                    std::to_string(s.warps_k) + " m" + std::to_string(s.m_tiles) +
                                    " n" + std::to_string(s.n_tiles) + " e" +
                                    std::to_string(epilogue) + (latency ? " latency" : "") +
                                    (add_sums ? " add_sums" : "");
                                ++checked;
                                expect(rc::usable(l), what + ": counts not usable");
                                const int sets = epilogue == 2 ? 2 : 1;
                                // The FP32 adds replace the MMA with ones where the call asks
                                // (add_sums) on BF16 operands; Turing's f16 operands never add
                                // (mma_step: F16Of), so its add-sums instances keep the MMA. The
                                // counts say so; #598's rule did not.
                                const bool adds = add_sums && arch != "75";
                                const double expect_share =
                                    min_term && !adds
                                        ? (sets * s.m_tiles + 1.0) / (sets * s.m_tiles)
                                        : 1.0;
                                const double share = rc::ones_factor(l);
                                expect(std::fabs(share - expect_share) < 1e-9,
                                       what + ": ones share " + std::to_string(share) +
                                           ", expected " + std::to_string(expect_share));
                                if (l.mma_ones > 0) ++with_ones;
                                // Two K steps of 4 x kRun per iteration: the K range of a 4096
                                // call splits into whole iterations of the warps.
                                const double it = rc::iterations(l, 4096.0 / s.cluster, s.warps_k);
                                expect(it > 0 && std::fabs(it * l.k_iter * s.warps_k * s.cluster -
                                                           4096.0) < 1e-9,
                                       what + ": iterations " + std::to_string(it));
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << checked << " kernels checked, " << with_ones << " with min-term MMAs\n";
    expect(checked > 0, "no snapshot carries loop counts");
    expect(with_ones > 0, "no kernel takes the MMA with ones");
}

void terms_arithmetic() {
    // A constructed loop: 128 MMAs of m16n8k16 (64 of them ones), 256 K per iteration, 4 LDL / STL
    // of 2 KB, 1000 instructions of which 400 INT and 390 FP; a CTA of 4 warps on a 2 GHz chip at
    // 512 flop per clock, 8 warps resident (2 a scheduler), 32 clocks of L1 latency; 2 waves, kc
    // 4096 unsplit.
    kf::LoopCounts l;
    l.loops          = 1;
    l.k_iter         = 256;
    l.mma            = 128;
    l.mma_ones       = 64;
    l.mma_shape      = "HMMA.16816.F32.BF16";
    l.local          = 4;
    l.local_bytes    = 2048;
    l.int_alu        = 400;
    l.fp_alu         = 390;
    l.total          = 1000;
    const double clk = 2e9;
    const auto t     = rc::terms(l, 2.0, 4.0, 4096.0, 1.0, 512.0 * clk, 8.0, 32.0, clk, 8);
    expect(std::fabs(t.warp_iterations - 2.0 * 4.0 * 16.0) < 1e-9, "warp iterations");
    expect(std::fabs(t.mma - 128.0 * 128 * 4096.0 / (512.0 * clk)) < 1e-15, "mma seconds");
    expect(std::fabs(t.spill - 128.0 * 4.0 * 32.0 / 2.0 / clk) < 1e-15, "spill seconds");
    // One warp a scheduler hides nothing; a quarter of a warp is still one.
    const auto lone = rc::terms(l, 2.0, 4.0, 4096.0, 1.0, 512.0 * clk, 1.0, 32.0, clk, 8);
    expect(std::fabs(lone.spill - 128.0 * 4.0 * 32.0 / clk) < 1e-15, "spill at one warp");
    expect(std::fabs(t.issue_slots - 128.0 * 1000.0 / (4.0 * clk)) < 1e-15, "issue slots");
    expect(std::fabs(t.int_pipe - 128.0 * 400.0 * 2.0 / (4.0 * clk)) < 1e-15, "int pipe");
    expect(std::fabs(t.fp_pipe - 128.0 * 390.0 / (4.0 * clk)) < 1e-15, "fp pipe");
    expect(t.issue == t.issue_slots, "the issue slots bind here");
    expect(std::fabs(rc::ones_factor(l) - 2.0) < 1e-12, "ones factor 2 at one tile");
    // Turing's shapes and pipes.
    l.mma_shape = "HMMA.1688.F32";
    expect(rc::mma_flops(l) == 2048.0, "m16n8k8 flops");
    const auto t75 = rc::terms(l, 2.0, 4.0, 4096.0, 1.0, 1536.0 * clk, 8.0, 32.0, clk, 7);
    expect(std::fabs(t75.fp_pipe - 128.0 * 390.0 * 2.0 / (4.0 * clk)) < 1e-15, "Turing fp pipe");
    // Not usable: no loop, no K, no MMA.
    kf::LoopCounts none;
    expect(!rc::usable(none) && rc::terms(none, 1, 1, 1, 1, 1, 1, 1, 1, 8).mma == 0, "unusable");
}

void law_gate() {
    // The binary prices one law (route_cost::kRmmaCountedLaw, false until the A/B); a test flips
    // it for its process and clears the override.
    namespace g = ninferno::ops::detail::generic;
    expect(g::rmma_counted_law() == g::route_cost::kRmmaCountedLaw, "the registry's law");
    g::rmma_counted_law_for_test(!g::route_cost::kRmmaCountedLaw);
    expect(g::rmma_counted_law() != g::route_cost::kRmmaCountedLaw, "the override");
    g::rmma_counted_law_for_test(std::nullopt);
    expect(g::rmma_counted_law() == g::route_cost::kRmmaCountedLaw, "cleared");
}

} // namespace

int main() {
    try {
        ones_share_on_the_snapshots();
        terms_arithmetic();
        law_gate();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0) {
        std::cerr << "rmma counted terms: " << failures << " failures\n";
        return 1;
    }
    std::cout << "rmma counted terms: PASS\n";
    return 0;
}
