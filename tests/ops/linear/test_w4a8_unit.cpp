// The counted W4A8 K tile (ops/generic/w4a8_unit.h, #613 PR-6, #978):
// - the snapshot key of every schedule for Q4_K and Q6_K is a row of the committed sm_75, sm_86,
//   sm_89 and sm_120a snapshots (the classic tile's Flavor follows the build, as
//   generic_w4a8.cuh w4a8_classic_flavor compiles it), for every epilogue;
// - the law's arithmetic on one row: stall sum + non-global waits x L_sh + the other warps'
//   issue a scheduler + the code outside the loop over the K tiles, which the frozen prediction
//   on #978 printed for the producer-warp tile 128x64 Q4_K on sm_89 (3082 clk);
// - no price without the counted producer loop, none for a classic tile (refuted on the RTX 3060,
//   #978), and none for a format the law was not scored on.
// No GPU.

#include "core/format_facts.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/w4a8_unit.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {

using namespace ninferno;
namespace kf = ops::detail::generic::kernel_facts;
using ops::detail::generic::Epilogue;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void keys_are_snapshot_rows() {
    for (const char* arch : {"75", "80", "86", "89", "90", "100", "120a"}) {
        for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q6_K}) {
            for (const Epilogue e : {Epilogue::Store, Epilogue::AddResidual, Epilogue::SwiGlu}) {
                for (int s = 0; s < 6; ++s) {
                    const auto key = w4a8_unit::kernel_key(s, q, e, arch);
                    expect(key.has_value(), "a key for schedule " + std::to_string(s));
                    if (!key) continue;
                    // sm_75 has no producer-warp tile (no snapshot row for any codec).
                    const bool absent_ok = key->producers > 0 && std::string(arch) == "75";
                    const auto facts     = kf::w4a8_facts(arch, *key);
                    expect(facts.has_value() || absent_ok,
                           std::string("sm_") + arch + " has the row " + key->codec + " " +
                               std::to_string(key->block_m) + "x" + std::to_string(key->block_n) +
                               " flavor " + std::to_string(key->flavor) + " epilogue " +
                               std::to_string(static_cast<int>(e)));
                    // Every producer-warp row of a scored arch prices: no silent fitted fallback.
                    if (key->producers > 0 && w4a8_unit::scored_arch(arch)) {
                        const auto isa = isa_row(8, std::string(arch) == "86" ? 6 : 9);
                        const auto c =
                            isa ? w4a8_unit::tile_cycles(arch, *key, *isa, 1, 32) : std::nullopt;
                        expect(c && *c > 0, std::string("sm_") + arch + " prices " + key->codec +
                                                " " + std::to_string(key->block_m) + "x" +
                                                std::to_string(key->block_n) + " epilogue " +
                                                std::to_string(static_cast<int>(e)));
                    }
                }
            }
        }
    }
    expect(!w4a8_unit::kernel_key(0, QType::GGML_Q5_K, Epilogue::Store, "89"),
           "Q5_K is not scored");
    expect(!w4a8_unit::kernel_key(6, QType::GGML_Q4_K, Epilogue::Store, "89"),
           "schedule 6 does not exist");
}

void law_arithmetic() {
    // sm_89's w4a8_ws_kernel<Q4_K, 128x64 (4x2 warps), Store, 3 stages, 4 producers> row and
    // its producer loop (#978): stall 853, waits 37 (0 global), 662 instructions; outside 5845
    // stall and 149 waits; producers 379 instructions. L_sh 24.127, 32 K tiles.
    kf::KernelFacts f;
    kf::LoopCounts loop;
    loop.loops = 1;
    loop.total = 662;
    kf::StaticSchedule sched;
    sched.stall_sum     = 853;
    sched.sb_waits      = 37;
    sched.outside_stall = 5845;
    sched.outside_waits = 149;
    loop.schedule       = sched;
    kf::PipeCounts pipes;
    pipes.sb_waits_global = 0;
    loop.pipes            = pipes;
    f.loop                = loop;
    kf::ProducerCounts prod;
    prod.loops = 1;
    prod.total = 379;
    f.producer = prod;
    kf::W4a8KernelKey key;
    key.codec         = "W4a8Q45K<4>";
    key.block_m       = 128;
    key.block_n       = 64;
    key.warps_m       = 4;
    key.warps_n       = 2;
    key.stages        = 3;
    key.producers     = 4;
    const double l_sh = 24.127;
    const double expect_cycles =
        853 + 37 * l_sh + (8.0 / 4 - 1) * 662 + 4.0 / 4 * 379 + (5845 + 149 * l_sh) / 32;
    const auto got = w4a8_unit::tile_cycles_of(f, key, l_sh, 1, 32);
    expect(got && std::abs(*got - expect_cycles) < 1e-9, "the law's sum");
    expect(got && std::abs(*got - 3082) < 1, "3082 clk, the frozen sm_89 figure");
    // A global wait is hidden: one more wait on a global load adds nothing.
    f.loop->schedule->sb_waits     = 38;
    f.loop->pipes->sb_waits_global = 1;
    const auto hidden              = w4a8_unit::tile_cycles_of(f, key, l_sh, 1, 32);
    expect(hidden && std::abs(*hidden - expect_cycles) < 1e-9, "a global wait is not added");
    // Two CTAs an SM: the second CTA's warps share the schedulers.
    const auto two = w4a8_unit::tile_cycles_of(f, key, l_sh, 2, 32);
    expect(two && std::abs(*two - (expect_cycles + 8.0 / 4 * 662 + 4.0 / 4 * 379)) < 1e-9,
           "a second CTA's warps add their issue");
    // A classic tile (no producer warps) has no counted price: the law was refuted there.
    kf::W4a8KernelKey classic = key;
    classic.producers         = 0;
    classic.stages            = 0;
    classic.flavor            = 3;
    expect(!w4a8_unit::tile_cycles_of(f, classic, l_sh, 1, 32), "no price for a classic tile");
    // The producer-warp tile without its counted producer loop has no counted price.
    f.producer.reset();
    expect(!w4a8_unit::tile_cycles_of(f, key, l_sh, 1, 32), "no producer loop, no price");
}

// The law from the committed snapshot rows and isa_probe rows, end to end: the frozen #978
// figures of the producer-warp tile 128x64, Store, one CTA an SM, 32 K tiles (K 4096).
void frozen_figures_from_the_snapshots() {
    struct Case {
        const char* arch;
        int major, minor;
        QType q;
        double clk;
    };

    const Case cases[] = {{"89", 8, 9, QType::GGML_Q4_K, 3082},
                          {"89", 8, 9, QType::GGML_Q6_K, 4126},
                          {"86", 8, 6, QType::GGML_Q4_K, 3039},
                          {"86", 8, 6, QType::GGML_Q6_K, 4074}};
    for (const Case& c : cases) {
        int schedule = -1;
        for (int s = 0; s < 6; ++s) {
            const auto k = w4a8_unit::kernel_key(s, c.q, Epilogue::Store, c.arch);
            if (k && k->producers > 0 && k->block_m == 128 && k->block_n == 64) schedule = s;
        }
        expect(schedule >= 0, "a producer-warp 128x64 schedule");
        const auto key = w4a8_unit::kernel_key(schedule, c.q, Epilogue::Store, c.arch);
        const auto isa = isa_row(c.major, c.minor);
        expect(key && isa, std::string("sm_") + c.arch + " key and ISA row");
        if (!key || !isa) continue;
        const auto got = w4a8_unit::tile_cycles(c.arch, *key, *isa, 1, 32);
        expect(got && std::abs(*got - c.clk) < 1,
               std::string("sm_") + c.arch + " " + key->codec + ": " +
                   (got ? std::to_string(*got) : std::string("none")) + " clk, frozen " +
                   std::to_string(c.clk));
    }
}

} // namespace

int main() {
    try {
        keys_are_snapshot_rows();
        law_arithmetic();
        frozen_figures_from_the_snapshots();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        ++failures;
    }
    if (failures != 0) {
        std::cerr << "w4a8_unit: " << failures << " failures\n";
        return 1;
    }
    std::cout << "w4a8_unit: ok\n";
    return 0;
}
