// ops::linear's candidates for one call (#306, ops/linear/linear_registry.h linear_candidates): the
// shape-owned Q8_G32 kernel with the launch its table selects at each width, and the generic route;
// the one dispatch_linear runs is Selected, and it is the one linear_decision names. The Q4/Q5/Q6_G64
// and BF16 tables' launches are named the same way (owned_launch_id).
#include "core/weight_view.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/linear/linear_registry.h"
#include "ops/generic/route_table.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q8/q8_dispatch.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace ops = ninferno::ops;
using ninferno::QType;
using ninferno::Weight;
using Candidate = ops::detail::LinearCandidate;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// A Q8_G32 weight as the converter lays it out, its bytes never read (an aligned address stands
// in), as ops/generic/route_query.cpp describes one.
struct Described {
    ninferno::WeightParent parent;
    ninferno::WeightView view;
    Weight weight;
    Described(std::int32_t n, std::int32_t k, QType qtype = QType::Q8_G32_FP16) {
        parent.data = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
        const std::uint64_t shape[2] = {static_cast<std::uint64_t>(n),
                                        static_cast<std::uint64_t>(k)};
        parent.geometry = ninferno::weight_geometry(
            qtype, ninferno::default_weight_layout(qtype), shape);
        view.shape  = {shape[0], shape[1]};
        view.parts  = {ninferno::WeightRegion{&parent, 0, shape[0] * shape[1]}};
        weight      = ninferno::native_weight(view, 0.0F);
    }
    Described(const Described&)            = delete;
    Described& operator=(const Described&) = delete;
};

const char* status_name(Candidate::Status s) {
    return s == Candidate::Status::Selected       ? "selected"
           : s == Candidate::Status::Inadmissible ? "inadmissible"
                                                  : "unavailable";
}

std::string describe(const std::vector<Candidate>& c) {
    std::string out;
    for (const Candidate& x : c) {
        out += " [" + x.id + " " + status_name(x.status) + (x.why.empty() ? "" : " " + x.why) + "]";
    }
    return out;
}

// The candidates of one call held to the decision: the owned kernel is Selected exactly where the
// decision keeps the call on it, the generic route exactly where a rule claims it; its launch is
// the table's own at this width.
void call(std::int32_t n, std::int32_t k, std::int32_t t, ops::LinearPolicy policy,
          const std::string& name) {
    const Described d(n, k);
    const auto decision = ops::detail::linear_decision(d.weight, policy, t);
    const auto c        = ops::detail::linear_candidates(d.weight, policy, t);
    const std::string at = name + " T=" + std::to_string(t) + ":" + describe(c);
    check(c.size() == 2, at + ": two candidates");
    if (c.size() != 2) return;
    const bool owned_claims = decision.claim == ops::detail::LinearClaim::Owned;
    const bool registered   = ops::detail::q8_linear_registered(n, k);
    check(c[0].id.rfind("owned:q8", 0) == 0, at + ": the owned candidate first");
    check(c[1].id.rfind("generic:", 0) == 0, at + ": the generic route second");
    check((c[0].status == Candidate::Status::Selected) == owned_claims,
          at + ": the owned kernel selected exactly where the decision keeps it");
    check((c[1].status == Candidate::Status::Selected) == !owned_claims,
          at + ": the generic route selected exactly where a rule claims the call");
    if (registered) {
        check(c[0].id == "owned:q8." + ops::detail::q8_launch_id(n, k, t),
              at + ": the owned candidate names its launch");
        // The identity changes from the width before exactly where the table's launch does.
        if (t > 1) {
            const bool same_launch = ops::detail::select_q8_launch(n, k, t, decision.policy) ==
                                     ops::detail::select_q8_launch(n, k, t - 1, decision.policy);
            check(same_launch == (ops::detail::q8_launch_id(n, k, t) ==
                                  ops::detail::q8_launch_id(n, k, t - 1)),
                  at + ": the launch identity follows the table's launch");
        }
        check(c[0].status != Candidate::Status::Unavailable, at + ": a registered shape's kernel");
    } else {
        check(c[0].status == Candidate::Status::Unavailable &&
                  decision.claim == ops::detail::LinearClaim::Unregistered,
              at + ": an unregistered shape has no owned kernel and goes generic for it");
        check(c[1].why == "unregistered", at + ": the generic route says why it runs");
    }
}

// A shape-owned table's launches at every width 1..max_t: the identity changes from the width
// before exactly where the table's selection does, starts at launch0, and the owned candidate of a
// weight of that format and shape carries it.
template <class Select>
void table_ids(QType qtype, Select select, std::int32_t n, std::int32_t k, const std::string& name) {
    const std::string shape = "n" + std::to_string(n) + "_k" + std::to_string(k) + ".launch";
    std::string before;
    for (std::int32_t t = 1; t <= 512; ++t) {
        const auto id = ops::detail::owned_launch_id(qtype, n, k, t);
        const std::string at = name + " T=" + std::to_string(t) + ": " + id.value_or("none");
        check(id && id->rfind(shape, 0) == 0, at + ": the shape and a launch number");
        if (!id) return;
        if (t == 1) {
            check(*id == shape + "0", at + ": the first width's launch is launch0");
        } else {
            check((select(n, k, t) == select(n, k, t - 1)) == (*id == before),
                  at + ": the launch identity follows the table's launch");
        }
        before = *id;
    }
    const Described d(n, k, qtype);
    for (const std::int32_t t : {1, 17, 512}) {
        const auto c = ops::detail::linear_candidates(d.weight, ops::LinearPolicy::A16Only, t);
        const std::string want = "owned:" + ops::detail::generic::format_label(qtype) + "." +
                                 ops::detail::owned_launch_id(qtype, n, k, t).value_or("");
        check(!c.empty() && c[0].id == want,
              name + " T=" + std::to_string(t) + ": the owned candidate names its launch," +
                  describe(c));
    }
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "CUDA unavailable; skipping linear candidates test\n";
        return 77;
    }
    // Registered shapes (q8_dispatch.cpp kShapes), their neighbours 128 values of K or 64 rows off,
    // at the decode, verify and prefill widths on both policies.
    const struct {
        std::int32_t n, k;
    } shapes[] = {{1024, 2048}, {5120, 25600}, {248320, 5120}, {2816, 16896}, {4096, 1024},
                  {1024, 2176}, {5184, 25600}, {2816, 17024}, {4160, 1024}};
    const std::int32_t widths[] = {1, 2, 4, 8, 9, 16, 17, 32, 64, 65, 128, 256, 512, 2048};
    for (const auto& s : shapes) {
        for (const auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
            for (const std::int32_t t : widths) {
                call(s.n, s.k, t, policy,
                     "q8 " + std::to_string(s.n) + "x" + std::to_string(s.k) +
                         (policy == ops::LinearPolicy::A16Only ? " a16" : " a8"));
            }
        }
    }
    // The invariant profile: the generic path takes every call; the owned kernel stays available
    // and says why it is not taken.
    {
        const ops::ScopedInvariantRoutes invariant(true);
        const Described d(1024, 2048);
        const auto c = ops::detail::linear_candidates(d.weight, ops::LinearPolicy::A16Only, 1);
        check(c.size() == 2 && c[0].status == Candidate::Status::Inadmissible &&
                  c[0].why == "invariant" && c[1].status == Candidate::Status::Selected,
              "invariant profile:" + describe(c));
    }
    // Every width 1..512 of a shape whose table changes launch often: the identity follows it.
    for (std::int32_t t = 1; t <= 512; ++t) {
        call(5120, 25600, t, ops::LinearPolicy::A16Only, "q8 5120x25600 sweep");
    }
    // The other tables that name their launches: a registered shape of each (q4/q5/q6/bf16
    // _dispatch.cpp kShapes).
    table_ids(QType::Q4_G64_FP16, ops::detail::select_q4_a16_launch, 1024, 5120, "q4 1024x5120");
    table_ids(QType::Q5_G64_FP16, ops::detail::select_q5_a16_launch, 5120, 17408, "q5 5120x17408");
    table_ids(QType::Q6_G64_FP16, ops::detail::select_q6_a16_launch, 248320, 5120, "q6 248320x5120");
    table_ids(QType::BF16, ops::detail::select_bf16_a16_launch, 5120, 6144, "bf16 5120x6144");
    check(!ops::detail::owned_launch_id(QType::NVFP4, 5120, 5120, 1),
          "a table that chooses inside its dispatch names no launch");
    // A fragment-order weight: dispatch_linear refuses it, so nothing is selected.
    {
        Described d(1024, 2048);
        d.weight.layout = ninferno::QuantLayout::GgmlFrag;
        const auto c = ops::detail::linear_candidates(d.weight, ops::LinearPolicy::A16Only, 1);
        check(c.size() == 2 && c[0].status == Candidate::Status::Unavailable &&
                  c[1].status == Candidate::Status::Unavailable,
              "ggml_frag_v1:" + describe(c));
    }
    if (failures != 0) {
        std::cerr << "linear candidates: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "linear candidates: the owned kernels (Q8, Q4/Q5/Q6, BF16 named by launch) and the "
                 "generic route, one selected as dispatch_linear runs\n";
    return 0;
}
