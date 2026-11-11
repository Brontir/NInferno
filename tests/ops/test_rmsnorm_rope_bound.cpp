// The fused text qk-norm + RoPE's width bound as the dispatch reads it
// (rmsnorm_rope_text_fused_max_tokens, #617 G), on substituted boards: the production composition,
// not a copy of its arithmetic. The Fitted chain excess is pending (route_cost::kPendingFits), so
// every board and geometry answers 256 (the RTX 5090's measured width). The law's own composition
// (launch node of the generation, SMs, resident warps from the kernel's facts, clock, heads of the
// geometry) is checked with an explicit excess and explicit kernel facts: the snapshots carry no
// facts of the text kernel yet, so the law reading them refuses. No GPU.
#include "core/device_class.h"
#include "core/kernel_occupancy.h"
#include "core/units.h"
#include "ninferno/ops/rmsnorm_rope.h"
#include "ops/rmsnorm_rope/launch.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using namespace ninferno;
using ninferno::ops::detail::generic::kernel_facts::KernelFacts;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "rmsnorm_rope bound: %s\n", what.c_str());
        ++failures;
    }
}

constexpr int kGeometries[][2] = {{8, 2}, {16, 2}, {16, 4}, {24, 4}, {32, 2}};

KernelFacts facts(int registers, std::size_t static_shared_bytes) {
    KernelFacts f;
    f.registers           = registers;
    f.static_shared_bytes = static_shared_bytes;
    return f;
}

} // namespace

int main() {
    const DeviceClass rtx5090 =
        synthetic_device_class(0x2B85, 12, 0, 170, 96 * kMiB, 32ULL * 1024 * kMiB);
    const DeviceClass rtx4090 =
        synthetic_device_class(0x2684, 8, 9, 128, 72 * kMiB, 24ULL * 1024 * kMiB);
    const DeviceClass rtx4060ti =
        synthetic_device_class(0x2803, 8, 9, 34, 32 * kMiB, 16ULL * 1024 * kMiB);
    const DeviceClass rtx2080ti =
        synthetic_device_class(0x1E04, 7, 5, 68, 6 * kMiB, 11ULL * 1024 * kMiB);

    // Uncalibrated: 256 everywhere, at every geometry; an uninstantiated geometry too.
    for (const DeviceClass* d : {&rtx5090, &rtx4090, &rtx4060ti, &rtx2080ti}) {
        for (const auto& g : kGeometries) {
            expect(ops::rmsnorm_rope_text_fused_max_tokens(*d, g[0], g[1]) == 256,
                   d->name + ": 256 while the excess is uncalibrated");
        }
        expect(ops::rmsnorm_rope_text_fused_max_tokens(*d, 12, 3) == 256, "default off-table");
    }

    namespace d         = ninferno::ops::detail;
    const double excess = 2000.0;

    // No snapshot carries the text kernel: the law that reads the committed facts refuses (0) on
    // every board, even with an excess, instead of falling back to the device's limits alone.
    for (const DeviceClass* b : {&rtx5090, &rtx4090, &rtx4060ti, &rtx2080ti}) {
        for (const auto& g : kGeometries) {
            expect(!d::text_kernel_facts(*b, g[0], g[1]).has_value(),
                   b->name + ": no committed facts of the text kernel");
            expect(d::text_fused_max_tokens_with_excess(*b, g[0], g[1], excess) == 0,
                   b->name + ": no facts, no bound");
        }
    }

    // The occupancy of the facts: the CUDA occupancy calculator over the device's limits. At 32
    // registers and no shared memory the thread limit binds on Ada (1536 / 96 = 16 of 24 slots);
    // at 128 registers a warp holds 4096 of a partition's 16384 registers, so 16 warps, 5 CTAs; at
    // 20 KiB of static shared memory (plus the 1 KiB reserved a CTA) 100 KiB holds 4.
    const KernelFacts light     = facts(32, 0);
    const KernelFacts registers = facts(128, 0);
    const KernelFacts shared    = facts(32, 20 * 1024);
    expect(d::text_kernel_facts_resident_ctas(rtx4090, light) == 16,
           "occupancy: the thread limit binds a light kernel on Ada");
    expect(d::text_kernel_facts_resident_ctas(rtx4090, registers) == 5,
           "occupancy: 128 registers leave 5 CTAs on Ada");
    expect(d::text_kernel_facts_resident_ctas(rtx4090, shared) == 4,
           "occupancy: 20 KiB of shared memory leaves 4 CTAs on Ada");
    expect(d::text_kernel_facts_resident_ctas(rtx4090, registers) ==
               resident_ctas_per_sm(rtx4090, KernelShape{128, 0, 96, 0}),
           "occupancy: the core calculator at three warps, no dynamic shared memory");

    // The law with an explicit chain excess (cycles) and facts: a composition check of the
    // dispatch's inputs (launch node of the generation, SMs, resident warps, clock, heads of the
    // geometry), not a prediction of any board. Ada has a launch node; gb20x and Turing have none
    // yet and answer 0.
    const auto w4090 = d::text_fused_max_tokens_with_facts(rtx4090, 32, 2, light, excess);
    const auto w4060 = d::text_fused_max_tokens_with_facts(rtx4060ti, 32, 2, light, excess);
    expect(w4090 > 0 && w4060 > 0, "Ada: a launch node, so a bound");
    const double ratio = static_cast<double>(w4090) / static_cast<double>(w4060);
    const double want =
        (128.0 / 34.0) * (machine_facts_of(rtx4090).reference_clock_khz /
                          static_cast<double>(machine_facts_of(rtx4060ti).reference_clock_khz));
    expect(ratio > 0.97 * want && ratio < 1.03 * want, "the width composes SMs and clock");
    // The clock is the canonical one the device class carries (device.facts), as in the other cost
    // models and the decision fingerprint, not one rebuilt from the raw driver fields.
    DeviceClass canonical = rtx4090;
    canonical.facts.reference_clock_khz *= 2;
    const auto w_canonical = d::text_fused_max_tokens_with_facts(canonical, 32, 2, light, excess);
    expect(std::abs(static_cast<double>(w_canonical) / w4090 - 2.0) < 0.03,
           "the law reads device.facts' reference clock (twice the clock, twice the width)");
    // The facts' occupancy is the law's: a register-limited kernel (5 CTAs against 16) and a
    // shared-memory-limited one (4) fuse proportionally narrower.
    const auto w_registers = d::text_fused_max_tokens_with_facts(rtx4090, 32, 2, registers, excess);
    const auto w_shared    = d::text_fused_max_tokens_with_facts(rtx4090, 32, 2, shared, excess);
    expect(w_registers > 0 && w_registers < w4090 &&
               std::abs(static_cast<double>(w4090) / w_registers - 16.0 / 5.0) < 0.03,
           "register-limited: the width scales with the facts' resident CTAs (16 over 5)");
    expect(w_shared > 0 && w_shared < w_registers &&
               std::abs(static_cast<double>(w4090) / w_shared - 16.0 / 4.0) < 0.03,
           "shared-memory-limited: the width scales with the facts' resident CTAs (16 over 4)");
    expect(d::text_fused_max_tokens_with_facts(rtx4090, 32, 2, facts(0, 0), excess) == 0,
           "unknown registers: no occupancy, no bound");
    const auto narrow = d::text_fused_max_tokens_with_facts(rtx4090, 16, 4, light, excess);
    const auto wide   = d::text_fused_max_tokens_with_facts(rtx4090, 32, 2, light, excess);
    // A token's launched warp slots: 3 x ceil((Q + K) / 3), 36 for 32+2 and 21 for 16+4.
    expect(narrow > wide && std::abs(static_cast<double>(narrow) / wide - 36.0 / 21.0) < 0.03,
           "per geometry: the width composes the launched heads a token (16+4 over 32+2 is 36/21)");
    // 8+2 launches four CTAs of three warps a token: 12 slots, not 10 (the last two warps idle).
    const auto ten    = d::text_fused_max_tokens_with_facts(rtx4090, 8, 2, light, excess);
    const auto twelve = d::text_fused_max_tokens_with_facts(rtx4090, 16, 2, light, excess);
    expect(ten > 0 && std::abs(static_cast<double>(ten) / twelve - 18.0 / 12.0) < 0.03,
           "a partial head group's idle warps are priced (8+2 takes 12 slots, 16+2 takes 18)");
    expect(d::text_fused_max_tokens_with_facts(rtx5090, 32, 2, light, excess) == 0 &&
               d::text_fused_max_tokens_with_facts(rtx2080ti, 32, 2, light, excess) == 0,
           "a generation without a launch node answers 0");
    expect(d::text_fused_max_tokens_with_facts(rtx4090, 12, 3, light, excess) == 0,
           "an uninstantiated geometry answers 0");
    expect(d::text_fused_max_tokens_with_facts(rtx4090, 32, 2, light, 0.0) == 0,
           "no excess calibrated: 0");

    if (failures == 0) std::printf("rmsnorm_rope bound: ok\n");
    return failures == 0 ? 0 : 1;
}
