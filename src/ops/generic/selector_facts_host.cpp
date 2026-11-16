#include "ops/generic/selector_facts.h"
#include "core/generation.h"
#include "core/units.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_query.h"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::ops::detail::generic {

// core's board_class with the board's own driver values (a FactsBoard is the CUDA query's view, and
// a test may spell a driver's rounding in it), the clocks scaled; `facts` as the query derives
// them from these values.
DeviceClass board_device_class(const FactsBoard& b, double sm_scale, double memory_scale) {
    return board_class(
        {.pci_device_id = b.pci_device_id,
         .major         = b.major,
         .minor         = b.minor,
         .sms           = b.sms,
         .global_bytes  = b.memory_gib * 1024 * kMiB,
         .driver        = BoardDriverView{
                    .l2_bytes         = static_cast<std::size_t>(b.l2_mib * kMiB),
                    .sm_clock_khz     = static_cast<int>(b.sm_clock_mhz * 1000 * sm_scale),
                    .memory_clock_khz = static_cast<int>(b.memory_clock_mhz * 1000 * memory_scale),
                    .memory_bus_bits  = b.bus_bits}});
}

bool board_prices_from_facts(const FactsBoard& b) { return rmma_terms_measured(b.major, b.minor); }

} // namespace ninferno::ops::detail::generic
