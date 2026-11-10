#include "ops/generic/generic_gemm.h"

#include <optional>

// The dense tuner's answers in a build with no CUDA runtime (#620 step 11): nothing is ever timed,
// so no tuned route exists (generic_tune.cpp holds the real ones: the tuner's tables, filled at the
// first eager call on a device). A host-only test adds this file to its own sources; nothing beside
// ninferno_ops links it.
namespace ninferno::ops::detail::generic {

std::optional<Route> tuned_dense_route(const Problem&, std::int32_t) { return std::nullopt; }

std::optional<Route> tuned_prefill_route(const Problem&, std::int32_t, Route) {
    return std::nullopt;
}

} // namespace ninferno::ops::detail::generic
