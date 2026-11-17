#include "ops/generic/w4a8_unit.h"

#include "ops/generic/family_facts.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/route_cost_constants.h"

#include <array>
#include <atomic>

namespace ninferno::w4a8_unit {
namespace {

namespace kernel_facts = ops::detail::generic::kernel_facts;

// -1 no override, 0 off, 1 on.
std::atomic<int> g_counted_override{-1};

constexpr int kSchedulers = 4; // warp schedulers an SM, every architecture the tiles build for

struct Schedule {
    int block_m, block_n, warps_m, warps_n, producers;
};

#define NINFERNO_W4A8_UNIT_INFO(i, bm, bn, wm, wn, p) Schedule{bm, bn, wm, wn, p},
constexpr std::array kSchedules{NINFERNO_GENERIC_W4A8_SCHEDULES(NINFERNO_W4A8_UNIT_INFO)};
#undef NINFERNO_W4A8_UNIT_INFO

// generic_w4a8.cuh w4a8_classic_flavor: float4 scales (1), mid-tile decode (2); a build without
// NVFP4 (NINFERNO_WITHOUT_NVFP4: every release arch before sm_100, whose labels do not start with
// "1") takes its own on 64x64 and 128x128.
int classic_flavor(int bm, int bn, bool without_nvfp4) {
    if (bm == 32) return 1;
    if (bm == 64 && bn == 64) return without_nvfp4 ? 1 : 3;
    if (bm == 128 && bn == 128) return without_nvfp4 ? 2 : 3;
    return 3;
}

} // namespace

bool counted() noexcept {
    const int o = g_counted_override.load(std::memory_order_relaxed);
    return o < 0 ? ops::detail::generic::route_cost::kW4a8CountedStep : o == 1;
}

void counted_for_test(std::optional<bool> on) noexcept {
    g_counted_override.store(on ? (*on ? 1 : 0) : -1, std::memory_order_relaxed);
}

bool scored_arch(std::string_view arch) noexcept { return arch == "86" || arch == "89"; }

bool scored(QType q) noexcept { return q == QType::GGML_Q4_K || q == QType::GGML_Q6_K; }

std::optional<kernel_facts::W4a8KernelKey>
kernel_key(int schedule, QType q, ops::detail::generic::Epilogue epilogue, std::string_view arch) {
    if (!scored(q) || schedule < 0 || schedule >= static_cast<int>(kSchedules.size()))
        return std::nullopt;
    const Schedule& s = kSchedules[static_cast<std::size_t>(schedule)];
    kernel_facts::W4a8KernelKey k;
    k.codec    = q == QType::GGML_Q4_K ? "W4a8Q45K<4>" : "W4a8Q6K";
    k.block_m  = s.block_m;
    k.block_n  = s.block_n;
    k.warps_m  = s.warps_m;
    k.warps_n  = s.warps_n;
    k.epilogue = static_cast<int>(epilogue);
    if (s.producers > 0) {
        k.stages    = ops::detail::generic::kW4a8MmaFacts.producer_stages;
        k.producers = s.producers;
    } else {
        k.flavor = classic_flavor(s.block_m, s.block_n, arch.empty() || arch.front() != '1');
    }
    return k;
}

std::optional<double> tile_cycles_of(const kernel_facts::KernelFacts& facts,
                                     const kernel_facts::W4a8KernelKey& key, double shared_latency,
                                     int ctas_per_sm, int k_tiles) {
    // The classic tile: refuted (#978, RTX 3060), the fitted step prices it.
    if (key.producers <= 0) return std::nullopt;
    if (!facts.loop || !facts.loop->schedule || !facts.loop->pipes || facts.loop->loops != 1 ||
        ctas_per_sm < 1 || k_tiles < 1 || !(shared_latency > 0)) {
        return std::nullopt;
    }
    const auto& loop    = *facts.loop;
    const auto& sched   = *loop.schedule;
    const int mma_warps = key.warps_m * key.warps_n;
    // The other warps on the critical warp's scheduler, from the CTAs resident on the SM.
    const double mma_per_scheduler = static_cast<double>(mma_warps) * ctas_per_sm / kSchedulers;
    double other                   = (mma_per_scheduler - 1.0) * loop.total;
    if (!facts.producer || facts.producer->loops != 1) return std::nullopt;
    other += static_cast<double>(key.producers) * ctas_per_sm / kSchedulers * facts.producer->total;
    const double waits =
        static_cast<double>(sched.sb_waits - loop.pipes->sb_waits_global) * shared_latency;
    const double outside =
        (sched.outside_stall + static_cast<double>(sched.outside_waits) * shared_latency) / k_tiles;
    return sched.stall_sum + waits + other + outside;
}

std::optional<double> tile_cycles(std::string_view arch, const kernel_facts::W4a8KernelKey& key,
                                  const IsaRow& isa, int ctas_per_sm, int k_tiles) {
    if (!isa.shared_latency) return std::nullopt;
    std::optional<kernel_facts::KernelFacts> facts;
    try {
        facts = kernel_facts::w4a8_facts(arch, key);
    } catch (...) {
        return std::nullopt; // a snapshot that does not read: the fitted price
    }
    if (!facts) return std::nullopt;
    return tile_cycles_of(*facts, key, *isa.shared_latency, ctas_per_sm, k_tiles);
}

} // namespace ninferno::w4a8_unit
