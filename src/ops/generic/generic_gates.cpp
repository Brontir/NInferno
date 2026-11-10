// The generic path's per-board launch gates that the selector reads (#620): the spread of a thin
// NVFP4 / FP8 call over SMs (spread_smem), PDL, and the add-sums form of the register MMA. Host
// code, no CUDA: the device is device_class()'s and the ordinal comes through core/per_device.h.
#include "core/device_class.h"
#include "core/measured_boards.h"
#include "core/per_device.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/arch_limits.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_select_detail.h"
#include "ops/generic/generic_select_internal.h"

#include <cstdint>
#include <exception>
#include <span>

namespace ninferno::ops::detail::generic {
namespace {

// GemmArgs::spread_smem. The eight-warp decode CTA (w1x8, one token tile, 72 registers) fits
// three to an SM, and the block scheduler fills SMs that deep before it uses the rest: a call of
// at most two CTAs per SM then runs on as few as a third of the SMs' memory pipes. Reserving a
// third of an SM's shared memory per CTA holds it to two, spread over every SM. Same kernel, same
// bytes. RTX 5090 at T = 1 under PDL (a kernel micro-bench, L2-cold weights): NVFP4 q/k/v
// 4608x6656 11.8 -> 11.25 us, gate 4096x6656 11.1 -> 10.5 us; FP8 row-f32 20.2 -> 19.5 and 18.5
// -> 17.8 us. At 416 CTAs and more it loses (6656x4096 NVFP4 9.4 -> 12.6 us) and so does SwiGLU
// (two CTAs per SM already: 116 registers), which is why it is limited to thin Store calls. A
// joined call (the attention gate beside q/k/v) takes it too: Muse Glimmer 30B NVFP4 decode 79.07
// -> 79.97 t/s, and 78.93 with the chain head alone spread. Not Q8_G32: its six Muse layers side by
// side lose what they gained with it (79.99 -> 79.63 t/s). The boards it was measured on, and any
// other board by the nearest of them within measured::kCarryDistance (else off), never across
// instruction sets: what decides is how many SMs a thin call spreads over, which the boards' SM
// count, bandwidth and L2 carry. The RTX 5080 (84 SMs, same micro-bench): NVFP4 2048x4096 6.66
// -> 6.10 us, 2560x2560 4.93 -> 4.68 us, 2688x4096 7.43 -> 7.06 us; FP8 2048x4096 11.04 -> 10.20
// us, 2688x4096 13.21 -> 12.48 us; 64 CTAs (1024 rows) even. The RTX 5060 Ti (36 SMs; same
// micro-bench, NVFP4/FP8/FP8 row-f32 at 1024..6656 rows, spread against packed, three rounds): even
// on every problem, since only 1024 rows stay within two CTAs per SM there; the RTX 4080 SUPER:
// even.

struct SpreadMeasured {
    Board board;
    bool faster;
};

constexpr SpreadMeasured kSpreadMeasured[] = {{measured::kRtx5090, true},
                                              {measured::kRtx5080, true},
                                              {measured::kRtx5060Ti, false},
                                              {measured::kRtx4080Super, false}};

static_assert(measured_boards_known(std::span(kSpreadMeasured)));

bool spread_by_board(const DeviceClass& d) {
    const SpreadMeasured* m =
        nearest_known_measured(std::span(kSpreadMeasured), board_of(d), measured::kCarryDistance);
    return m != nullptr && m->faster;
}

// The current device's gates, once per ordinal: a process may drive boards of different classes
// (an RTX 5090 beside a 4090), and each launch takes its own device's. Through PerDevice, so a
// substituted DeviceClass (ScopedDeviceClass: the selector goldens, route_query) reads its own
// board's gates, not the ones the host's GPU cached first. A device whose class cannot be read
// throws and is asked again next time (caching "none" would turn PDL and the spread off for the
// process on one transient failure).
struct DeviceGates {
    bool spread = false; // spread_smem measured faster on the nearest measured board
    bool pdl    = false; // programmatic dependent launch (pdl_prefetch)
};

const DeviceGates& device_gates() {
    static PerDevice<DeviceGates> gates;
    static const DeviceGates none{};
    // No current device (a host-only process without a substitute): no gates, as before.
    if (substituted_device_class_key() == nullptr && current_device_ordinal_or_none() < 0) {
        return none;
    }
    return gates.get([](int) {
        const DeviceClass& d = device_class();
        return DeviceGates{spread_by_board(d), d.features.pdl};
    });
}

bool spread_measured() { return device_gates().spread; }

} // namespace

bool pdl_device() { return device_gates().pdl; }

int spread_smem(Route route, WeightCodec codec, Epilogue epilogue, const GemmArgs& args) {
    if (route.family != Family::RegisterMma || route.schedule != rmma_index(1, 8, 1, 1, 1) ||
        epilogue != Epilogue::Store || args.tokens > 8 ||
        !(codec == WeightCodec::Nvfp4 || codec == WeightCodec::Fp8 ||
          codec == WeightCodec::Fp8RowF32) ||
        !spread_measured()) {
        return 0;
    }
    try {
        const auto& device      = device_class();
        const std::int64_t ctas = (static_cast<std::int64_t>(args.rows) + 15) / 16;
        const std::size_t pad   = (device.smem_per_sm / 3 + 1024) / 1024 * 1024;
        if (ctas > 2 * static_cast<std::int64_t>(device.sms) || pad > kStaticSharedBytesPerBlock)
            return 0;
        return static_cast<int>(pad);
    } catch (const std::exception&) { return 0; }
}

bool rmma_add_sums() {
    return current_features().mma_tensor && bf16_mma_native() && !invariant_routes_active() &&
           !prefill_routes_active();
}

} // namespace ninferno::ops::detail::generic
