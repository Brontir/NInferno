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

DeviceClass board_device_class(const FactsBoard& b, double sm_scale, double memory_scale) {
    DeviceClass d      = synthetic_device_class(b.pci_device_id, b.major, b.minor, b.sms,
                                                b.l2_mib * kMiB, b.memory_gib * 1024 * kMiB);
    d.sm_clock_khz     = static_cast<int>(b.sm_clock_mhz * 1000 * sm_scale);
    d.memory_clock_khz = static_cast<int>(b.memory_clock_mhz * 1000 * memory_scale);
    d.memory_bus_bits  = b.bus_bits;
    d.facts            = machine_facts_of(d); // as the query derives them from these values
    return d;
}

bool board_prices_from_facts(const FactsBoard& b) { return rmma_terms_measured(b.major, b.minor); }

} // namespace ninferno::ops::detail::generic
