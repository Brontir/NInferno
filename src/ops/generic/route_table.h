#pragma once

// The generic projection path's view of the per-GPU-class route table
// (include/ninferno/ops/route_table.h; loaded by src/ops/route_table.cpp): "bands" and "prefill".

#include "ninferno/ops/route_table.h"
#include "ops/common/route_compute.h"
#include "ops/generic/generic_gemm.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::ops::detail::generic {

// The generic path computes in A16 or A4; A8 is a shape-owned FP8 table's class only.
using ComputeClass = RouteCompute;

struct RouteBand {
    std::int32_t t_min = 1;
    std::int32_t t_max = 1;
    Route route;
};

struct RouteEntry {
    QType qtype;
    ComputeClass compute;
    Epilogue epilogue;
    std::int32_t n;
    std::int32_t k;
    std::vector<RouteBand> bands;
    std::optional<Route> prefill;
    // Under the prefill profile (T >= 17) a band here takes the call instead of `prefill`; every
    // route in it sums as `prefill` does (same_summation), so bytes stay independent of the width.
    std::vector<RouteBand> prefill_bands;
    // "claim": the problem's shape also has a shape-owned Linear table (ops/linear/<format>/shapes),
    // which the dispatcher prefers; a claiming entry's routes replace that table's own choice at the
    // widths the entry covers (store epilogue only; measured against the table's kernels).
    bool claim = false;
    // The class whose table lent the entry when the nearest board's table lacked it
    // (route_table.cpp fill_from); empty: the table's own.
    std::string lender;
    // The lending table's board distance to this device (table_distance); 0 for the table's own.
    double lender_distance = 0;
};

[[nodiscard]] ComputeClass compute_class(QType qtype, LinearPolicy policy);
[[nodiscard]] const char* compute_class_name(ComputeClass c);
[[nodiscard]] const char* epilogue_name(Epilogue e);
[[nodiscard]] const char* qtype_name(QType q);
// "mxfp4 (format 9)": the name and the number, for errors about a format this build cannot use (an
// artifact from another converter or a newer one can store one).
[[nodiscard]] std::string format_label(QType q);

// The tuned route for one call, if a loaded table holds the exact problem (loads the default
// table on first use).
[[nodiscard]] std::optional<Route> table_route(const Problem& problem, std::int32_t tokens,
                                               bool prefill);

// table_route's answer with what it rests on: the board distance of the table to this device
// (0: measured here, which an unboarded table keyed by one of the device's own classes counts as;
// a similar chip's unboarded table 1e6 and up), the table's class, and the class whose table lent
// the entry (RouteEntry::lender; empty: the table's own) with that lender's distance. The strings
// are copies: a table published meanwhile does not leave them dangling.
struct TableAnswer {
    Route route;
    double distance = 0;
    std::string gpu_class;
    std::string lender;
    double lender_distance = 0;
};

[[nodiscard]] std::optional<TableAnswer> table_route_answer(const Problem& problem,
                                                            std::int32_t tokens, bool prefill);

// Whether a claiming entry (RouteEntry::claim) holds the exact problem and a route for this call
// under an A16Only policy: then the generic path computes it even though its shape has a
// shape-owned Linear table.
[[nodiscard]] bool table_claims(const Problem& problem, std::int32_t tokens, bool prefill);

// Serializes entries as a table file for `gpu_class` (the tuner's writer).
[[nodiscard]] std::string write_route_table(const std::string& gpu_class,
                                            const std::string& gpu_name,
                                            const std::vector<RouteEntry>& entries);

} // namespace ninferno::ops::detail::generic
