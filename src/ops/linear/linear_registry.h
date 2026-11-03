#pragma once

#include "core/weight.h"
#include "ninferno/ops/linear.h" // LinearPolicy

#include <cstdint>
#include <string>
#include <vector>

namespace ninferno::ops::detail {

// Whether the shape-owned linear table of this format registers the exact [N,K]. A shape that is
// not registered takes the generic path (ops/generic/generic_gemm.h).
[[nodiscard]] bool linear_shape_registered(QType qtype, std::int32_t n, std::int32_t k);

// Whether ops::linear(x, weight, out, policy) of `tokens` columns may instead run joined to the
// call before it in one segmented projection (generic Call::join_previous), as the dense pass's
// side-by-side projections do: the generic path takes it (no shape-owned table) with the register
// MMA, in a format the joined launch was measured on (NVFP4, FP8, Q8_G32), or with the R8 decode
// kernel in any format it reads (the GGUF K-quants, i-quants and legacy blocks): the route the
// launch runs (generic::resolve_route) and joins (generic::launch_overlap). The bytes must
// equal ops::linear's (tests/ops/linear/test_side_by_side_projection.cpp checks every format
// against this answer, so a format added here is checked there too).
[[nodiscard]] bool side_by_side_admits(const Weight& weight, LinearPolicy policy,
                                       std::int32_t tokens);

// Why ops::linear runs a call on the generic path or on the shape's own kernels (#306): the one
// decision dispatch_linear takes. Owned is the shape-owned table's; every other names the first
// rule that hands the call to the generic path. `policy` is the policy the call runs under (an
// int8 prefill promotes it, generic::prefill_a8_policy).
enum class LinearClaim {
    Owned,            // a registered shape the generic path does not claim
    Invariant,        // the invariant route profile (--activations exact)
    Unregistered,     // no shape-owned kernel for this [N,K]
    OwnedDecodeR8,    // the R8 decode kernel takes the owned shape (generic::owned_decode_r8)
    PrefillA8,        // an int8 prefill (generic::q8_prefill_a8)
    OwnedPrefillW4a8, // an NVFP4 prefill under A8 but not A4 (generic::owned_prefill_w4a8)
    VoltaTiled,       // the tiled tensor-core route without mma.sync (generic::volta_tiled_route)
    RouteTable,       // a route-table entry claims these widths (generic::table_claims)
};
struct LinearDecision {
    LinearClaim claim   = LinearClaim::Owned;
    LinearPolicy policy = LinearPolicy::A16Only;
};
[[nodiscard]] LinearDecision linear_decision(const Weight& weight, LinearPolicy policy,
                                             std::int32_t tokens);
[[nodiscard]] const char* linear_claim_name(LinearClaim claim);

// Every implementation ops::linear has for one call, and which one it runs (#306): the shape-owned
// kernel ("owned:q8.n<N>_k<K>.launch<i>" for Q8_G32: the launch its table selects at the width,
// q8_launch_id; another format "owned:<format>") and the generic route ("generic:<route>", the
// route select_route asks for; the call's layout and workspace may still resolve it lower,
// generic::resolve_route).
// Selected: the one dispatch_linear runs (none where it refuses the call; `why` of the generic
// route: the claim that handed it the call). Inadmissible: available, not taken (`why`: the claim
// that took the call from the shape's kernels, or "owned" for a route the shape's kernels win
// over). Unavailable: no such implementation for the weight (`why`: what lacks; a ggml_frag_v1
// weight, which ops::linear refuses, has neither). T must be positive.
struct LinearCandidate {
    enum class Status { Selected, Inadmissible, Unavailable };
    std::string id;
    Status status = Status::Unavailable;
    std::string why;
};
[[nodiscard]] std::vector<LinearCandidate> linear_candidates(const Weight& weight,
                                                             LinearPolicy policy,
                                                             std::int32_t tokens);

// ops::linear_pair's implementations for one call (#310): the fused Q8 pair kernel ("fused:q8_pair",
// its two registered geometries in the converter's RowSplit Q8_G32 layout) and the two separate
// linear() calls ("separate", A16 as linear_pair's contract has it), each with the candidates of
// its two calls (`parts`: linear_candidates of each weight, the first then the second). Selected:
// the one linear_pair runs; the fused kernel is Unavailable with what lacks where it does not
// apply; the separate calls are Inadmissible ("fused") where the fused kernel takes the pair,
// Unavailable where the shapes differ or linear() runs nothing for a weight.
struct LinearPairCandidate {
    std::string id;
    LinearCandidate::Status status = LinearCandidate::Status::Unavailable;
    std::string why;
    std::vector<LinearCandidate> parts;
};
[[nodiscard]] std::vector<LinearPairCandidate> linear_pair_candidates(const Weight& first,
                                                                      const Weight& second,
                                                                      std::int32_t tokens);

} // namespace ninferno::ops::detail
