#pragma once

#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"

#include <cstddef>
#include <cstdint>

namespace ninferno::test {

// Workspace capacity queries cover both route profiles (ninferno/ops/prefill_routes.h). On the
// cards where int8 prefill activations pay (prefill_a8_format), a Q4/Q5/Q6 G64 call of 17 columns
// or more takes the int8 tiers only inside a prefill: outside one it runs A16 within a query sized
// for both, so its high water may stay below the query. Everywhere else the two must match.
inline bool qx_query_covers_prefill_a8(QType qtype, std::int32_t columns) {
    if (ops::prefill_routes_active()) return false;
    // ninfer-all's dedicated int8 kernels (q4a8) run inside a prefill only, from 128 columns.
    if (columns >= 128 && ((qtype == QType::Q4_G64_FP16 && ops::detail::q4a8_swiglu_pays()) ||
                           ((qtype == QType::Q4_G64_FP16 || qtype == QType::Q5_G64_FP16) &&
                            ops::detail::a8_add_pays()))) {
        return true;
    }
    return (qtype == QType::Q4_G64_FP16 || qtype == QType::Q5_G64_FP16 ||
            qtype == QType::Q6_G64_FP16) &&
           ops::detail::generic::prefill_a8_format(qtype) &&
           columns >= ops::detail::generic::kQ8PrefillA8Tokens;
}

inline bool workspace_query_holds(std::size_t peak, std::size_t query, bool covers_prefill_a8) {
    return covers_prefill_a8 ? peak <= query : peak == query;
}

} // namespace ninferno::test
