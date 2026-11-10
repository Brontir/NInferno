#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"

#include <cstddef>
#include <optional>

// The answers of the generic path that belong to a device or to the CUDA build, in a build with no
// CUDA runtime (#620 step 11): nothing is ever timed, so no tuned route exists (generic_tune.cpp
// holds the real ones: the tuner's tables, filled at the first eager call on a device), and the
// Volta CUTLASS route is compiled out (generic_volta_cutlass.cu's answer of a build without it).
// A host-only test adds this file to its own sources; nothing beside ninferno_ops links it, which
// defines the real ones.
namespace ninferno::ops::detail::generic {

std::optional<Route> tuned_dense_route(const Problem&, std::int32_t) { return std::nullopt; }

std::optional<Route> tuned_prefill_route(const Problem&, std::int32_t, Route) {
    return std::nullopt;
}

bool volta_cutlass_compiled() noexcept { return false; }

std::size_t volta_cutlass_workspace_bytes(std::int32_t, std::int32_t, std::int32_t, bool) {
    return 0;
}

std::size_t volta_cutlass_chunk_bytes(std::int32_t, std::int32_t, std::int32_t, bool, std::size_t) {
    return 0;
}

// The kernel a route would launch is read from the device's binary (generic_describe.cpp): a host
// with no device describes none.
std::optional<KernelResources> route_resources(Route, const Weight&, Epilogue, std::int32_t,
                                               std::int32_t, std::optional<bool>) {
    return std::nullopt;
}

} // namespace ninferno::ops::detail::generic
