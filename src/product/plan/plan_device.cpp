#include "product/plan/plan_device.h"

#include "product/plan/plan_facts.h"

#include <cuda_runtime.h>

#include <exception>

namespace ninferno::product {

std::optional<runtime::PlanDeviceFacts> plan_device_of_bus(const std::string& bus_id) {
    if (bus_id.empty()) { return std::nullopt; }
    int device = -1;
    if (cudaDeviceGetByPCIBusId(&device, bus_id.c_str()) != cudaSuccess || device < 0) {
        (void)cudaGetLastError();
        return std::nullopt;
    }
    try {
        return plan_device_of_cuda(device, false);
    } catch (const std::exception&) { return std::nullopt; }
}

} // namespace ninferno::product
