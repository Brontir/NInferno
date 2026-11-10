// The fused text qk-norm + RoPE's width bound as the dispatch reads it
// (rmsnorm_rope_text_fused_max_tokens, #617 G), on substituted boards: the production composition,
// not a copy of its arithmetic. The Fitted chain excess is pending (route_cost::kPendingFits), so
// every board and geometry answers 256 (the RTX 5090's measured width). The law's own composition
// (launch node of the generation, DRAM rate of the board, bytes a geometry moves) is checked with
// an explicit excess. No GPU.
#include "core/device_class.h"
#include "core/units.h"
#include "ninferno/ops/rmsnorm_rope.h"
#include "ops/rmsnorm_rope/launch.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using namespace ninferno;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "rmsnorm_rope bound: %s\n", what.c_str());
        ++failures;
    }
}

constexpr int kGeometries[][2] = {{8, 2}, {16, 2}, {16, 4}, {24, 4}, {32, 2}};

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

    // The law with an explicit chain excess (cycles): a composition check of the dispatch's inputs
    // (launch node of the generation, SMs, resident warps, clock, heads of the geometry), not a
    // prediction of any board. Ada has a launch node; gb20x and Turing have none yet and answer 0.
    namespace d         = ninferno::ops::detail;
    const double excess = 2000.0;
    const auto w4090    = d::text_fused_max_tokens_with_excess(rtx4090, 32, 2, excess);
    const auto w4060    = d::text_fused_max_tokens_with_excess(rtx4060ti, 32, 2, excess);
    expect(w4090 > 0 && w4060 > 0, "Ada: a launch node, so a bound");
    const double ratio = static_cast<double>(w4090) / static_cast<double>(w4060);
    const double want =
        (128.0 / 34.0) * (machine_facts_of(rtx4090).reference_clock_khz /
                          static_cast<double>(machine_facts_of(rtx4060ti).reference_clock_khz));
    expect(ratio > 0.97 * want && ratio < 1.03 * want, "the width composes SMs and clock");
    const auto narrow = d::text_fused_max_tokens_with_excess(rtx4090, 16, 4, excess);
    const auto wide   = d::text_fused_max_tokens_with_excess(rtx4090, 32, 2, excess);
    expect(narrow > wide && std::abs(static_cast<double>(narrow) / wide - 34.0 / 20.0) < 0.03,
           "per geometry: the width composes the heads a token (16+4 over 32+2 is 34/20)");
    expect(d::text_fused_max_tokens_with_excess(rtx5090, 32, 2, excess) == 0 &&
               d::text_fused_max_tokens_with_excess(rtx2080ti, 32, 2, excess) == 0,
           "a generation without a launch node answers 0");
    expect(d::text_fused_max_tokens_with_excess(rtx4090, 12, 3, excess) == 0,
           "an uninstantiated geometry answers 0");
    expect(d::text_fused_max_tokens_with_excess(rtx4090, 32, 2, 0.0) == 0,
           "no excess calibrated: 0");

    if (failures == 0) std::printf("rmsnorm_rope bound: ok\n");
    return failures == 0 ? 0 : 1;
}
