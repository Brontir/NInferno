#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"

#include <cstddef>
#include <optional>

// The answers of the generic path that belong to a device or to the CUDA build, in a build with no
// CUDA runtime (#620 step 11): nothing is ever timed, so no tuned route exists (generic_tune.cpp
// holds the real ones: the tuner's tables, filled at the first eager call on a device), and the
// kernel an executed route would launch is not described (route_resources,
// rmma_resources_by_launch: they read the device's binary; the last two stubs are here for the
// link only, nothing a host-only test calls reaches them). The Volta CUTLASS route's sizes are no
// seam: they are host code (generic_volta_cutlass_sizes.cpp), compiled with the route where the
// tree has it. A host-only test adds this file to its own sources; nothing beside ninferno_ops
// links it, which defines the real ones.
namespace ninferno::ops::detail::generic {

std::optional<Route> tuned_dense_route(const Problem&, std::int32_t) { return std::nullopt; }

std::optional<Route> tuned_prefill_route(const Problem&, std::int32_t, Route) {
    return std::nullopt;
}

// The kernel a route would launch is read from the device's binary (generic_describe.cpp): a host
// with no device describes none.
std::optional<KernelResources> route_resources(Route, const Weight&, Epilogue, std::int32_t,
                                               std::int32_t, std::optional<bool>) {
    return std::nullopt;
}

// The register-MMA launch run in describe mode (generic_describe.cpp): rmma_resources reaches it
// only under the maintainer's live check, which needs a GPU.
std::optional<KernelResources> rmma_resources_by_launch(int, const Weight&, Epilogue, std::int32_t,
                                                        std::int32_t) {
    return std::nullopt;
}

} // namespace ninferno::ops::detail::generic
