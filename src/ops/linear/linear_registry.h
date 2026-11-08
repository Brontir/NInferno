#pragma once

#include "core/weight.h"
#include "ninferno/ops/linear.h" // LinearPolicy

#include <cstdint>
#include <optional>
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
// kernel ("owned:q8.n<N>_k<K>.launch<i>" for Q8_G32, "owned:<format>.n<N>_k<K>.launch<i>" for
// Q4/Q5/Q6_G64 and BF16: the launch its table selects at the width, owned_launch_id; NVFP4 and
// FP8 "owned:<format>") and the generic route ("generic:<route>", the
// route select_route asks for; the call's layout and workspace may still resolve it lower,
// generic::resolve_route).
// Selected: the one dispatch_linear runs (none where it refuses the call; `why` of the generic
// route: the claim that handed it the call). Inadmissible: available, not taken (`why`: the claim
// that took the call from the shape's kernels, or "owned" for a route the shape's kernels win
// over). Unavailable: no such implementation for the weight (`why`: what lacks; a ggml_frag_v1
// weight, which ops::linear refuses, has neither). T must be positive.
// After these two, each other generic family ("generic.<family>:<route>", generic::
// family_alternatives) as an alternative to the selected route, where the generic path reads the
// weight:
// - Unavailable: no schedule compiled for the format.
// - Inadmissible: no schedule takes the call (`why`: the refusal).
// - Dominated: priced, and no cheaper than the selected route by the same model.
// - PricedBelow: priced cheaper than the selected route, so the selection is not the model's
//   (a route table, the tuned prefill route or a pin chose it).
// - Unpriced: takes the call, but there is no comparable cost (a call the model does not price,
//   a selected route it does not price, or the shape's own kernels selected).
// `seconds`: the model's time where it prices the candidate (the selected generic route too: the
// route select_route asks for; the call's layout and workspace may still resolve it lower).
struct LinearCandidate {
    enum class Status { Selected, Inadmissible, Unavailable, Dominated, PricedBelow, Unpriced };
    std::string id;
    Status status = Status::Unavailable;
    std::string why;
    std::optional<double> seconds;
};
// The launch a shape-owned table selects for [n,k] at width t (#306): the shape and the launch's
// number in the order the widths 1..t first select its launches ("n1024_k2048.launch2"). The
// Q8_G32, Q4/Q5/Q6_G64 and BF16 tables; nullopt for a format whose dispatch chooses inside itself
// (NVFP4, FP8). The shape must be registered.
[[nodiscard]] std::optional<std::string> owned_launch_id(QType qtype, std::int32_t n,
                                                         std::int32_t k, std::int32_t t);
[[nodiscard]] std::vector<LinearCandidate> linear_candidates(const Weight& weight,
                                                             LinearPolicy policy,
                                                             std::int32_t tokens);

// The bytes the CTAs of ops::linear's call request for one Q8_G32 weight at `tokens` (A16), before
// any cache serves them (generic::route_traffic's law): the weights once for every column tile of
// CTAs, the activation once for every row tile, the output once. Priced where the shape's own table
// runs the call and gives its launch's tile (q8_launch_tile); nullopt where another implementation
// runs it or the tile is not known.
[[nodiscard]] std::optional<double> linear_q8_requested_bytes(const Weight& weight,
                                                              std::int32_t tokens);

// ops::linear_pair's implementations for one call (#310): the fused Q8 pair kernel
// ("fused:q8_pair", its two registered geometries in the converter's RowSplit Q8_G32 layout) and
// the two separate linear() calls ("separate", A16 as linear_pair's contract has it), each with the
// candidates of its two calls (`parts`: linear_candidates of each weight, the first then the
// second). Selected: the one linear_pair runs; the fused kernel is Unavailable with what lacks
// where it does not apply (Dominated where it applies but its chain requests more bytes than the
// two calls' and fuses nothing); the separate calls are Inadmissible ("fused") where the fused
// kernel takes the pair, Unavailable where the shapes differ or linear() runs nothing for a weight.
struct LinearPairCandidate {
    std::string id;
    LinearCandidate::Status status = LinearCandidate::Status::Unavailable;
    std::string why;
    std::vector<LinearCandidate> parts;
};

// The bytes the CTAs of each alternative request for a call of `tokens` columns (its chain price,
// ops/wrapper/linear_pair.cpp): the fused kernel's and the two linear() calls', nullopt where the
// tiling of that alternative is not known (then the fused kernel keeps the call).
struct LinearPairRequested {
    std::optional<double> fused;
    std::optional<double> separate;
};

[[nodiscard]] LinearPairRequested
linear_pair_requested_bytes(const Weight& first, const Weight& second, std::int32_t tokens);
[[nodiscard]] std::vector<LinearPairCandidate> linear_pair_candidates(const Weight& first,
                                                                      const Weight& second,
                                                                      std::int32_t tokens);

} // namespace ninferno::ops::detail
