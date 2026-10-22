#pragma once

// A GPU as the plan without the weights sees it, from the same source a start reads (the CUDA
// device's totalGlobalMem, the driver model, the kernel-run watchdog as the display flag:
// plan_device_of_cuda), for a process that knows the GPU by its PCI bus id (ninferno-ui's NVML
// list). No CUDA context is created and no memory is measured: an idle card.

#include "runtime/engine/plan_dry_run.h"

#include <optional>
#include <string>

namespace ninferno::product {

// nullopt when the CUDA runtime does not know the bus id (no driver, a GPU hidden from CUDA).
[[nodiscard]] std::optional<runtime::PlanDeviceFacts> plan_device_of_bus(const std::string& bus_id);

} // namespace ninferno::product
