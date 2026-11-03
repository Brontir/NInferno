#include "core/device_class.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/prefill_routes.h"

#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/route_pins.h"
#include "ops/generic/route_table.h"
#include "ops/linear/linear_registry.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear_add/q8/q8_linear_add_plan.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::ops {
namespace detail {

bool linear_shape_registered(QType qtype, std::int32_t n, std::int32_t k) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return q4_linear_registered(n, k);
    case QType::Q5_G64_FP16:
        return q5_linear_registered(n, k);
    case QType::Q6_G64_FP16:
        return q6_linear_registered(n, k);
    case QType::Q8_G32_FP16:
        return q8_linear_registered(n, k);
    case QType::BF16:
        return bf16_linear_registered(n, k);
    case QType::NVFP4:
#ifdef NINFERNO_WITHOUT_NVFP4
        return false;
#else
        return nvfp4_linear_registered(n, k);
#endif
    case QType::FP8_E4M3FN_ROW_BF16:
        return fp8_linear_registered(n, k);
    case QType::MXFP4:
    case QType::FP8_E4M3FN_ROW_F32:
    case QType::FP8_E4M3FN_B128_F32:
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    case QType::GGML_TQ1_0:
    case QType::GGML_TQ2_0:
    case QType::GGML_Q1_0:
    case QType::GGML_Q2_0:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    return false;
}

bool side_by_side_admits(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    namespace generic = detail::generic;
    // The gain is the joined launch (generic Call::join_previous, a programmatic dependent
    // launch): without PDL the slices run one after another and the segmented call only adds its
    // own overhead (RTX 4090 and 3090: Muse Glimmer 30B -0.8 % side by side).
    if (!device_class().features.pdl) { return false; }
    if (linear_shape_registered(weight.qtype, weight.n, weight.k) || !generic::supports(weight)) {
        return false;
    }
    const generic::Problem problem{weight.qtype, policy, weight.n, weight.k,
                                   generic::Epilogue::Store};
    const generic::Family family =
        generic::select_route(problem, weight, tokens, prefill_routes_active()).family;
    // The R8 decode kernel joins in every format it reads (a GGUF layer's q/k/v and gate).
    if (family == generic::Family::R8Mma) { return true; }
    const generic::WeightCodec codec = generic::codec_of(weight);
    if (!(codec == generic::WeightCodec::Nvfp4 || codec == generic::WeightCodec::Fp8 ||
          codec == generic::WeightCodec::Fp8RowF32 || codec == generic::WeightCodec::Q8)) {
        return false;
    }
    return family == generic::Family::RegisterMma;
}

LinearDecision linear_decision(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    LinearDecision d;
    const LinearPolicy a8_policy = generic::prefill_a8_policy(w.qtype, policy);
    const bool prefill_a8        = generic::q8_prefill_a8(w, a8_policy, tokens);
    d.policy                     = prefill_a8 ? a8_policy : policy;
    // A shape the tuned tables do not own takes the generic path (the tables win when they do);
    // under the invariant route profile every shape the generic path reads does (prefill_routes.h),
    // and so does an owned shape whose route-table entry claims this call (RouteEntry::claim: the
    // generic route measured faster than the shape's own kernels at these widths), and, on a GPU
    // without tensor mma.sync, a call the tiled tensor-core route takes (volta_tiled_route). An
    // NVFP4 prefill call (T >= 17) whose policy allows A8 but not A4 takes the generic W4A8
    // (owned_prefill_w4a8): the native NVFP4 shapes compute A16 and A4 only.
    if (invariant_routes_active()) {
        d.claim = LinearClaim::Invariant;
    } else if (!linear_shape_registered(w.qtype, w.n, w.k)) {
        d.claim = LinearClaim::Unregistered;
    } else if (generic::owned_decode_r8(w, d.policy, tokens)) {
        d.claim = LinearClaim::OwnedDecodeR8;
    } else if (prefill_a8) {
        d.claim = LinearClaim::PrefillA8;
    } else if (generic::owned_prefill_w4a8(w, d.policy, tokens)) {
        d.claim = LinearClaim::OwnedPrefillW4a8;
    } else if (generic::volta_tiled_route(w, d.policy, tokens)) {
        d.claim = LinearClaim::VoltaTiled;
    } else if (generic::table_claims({w.qtype, d.policy, w.n, w.k, generic::Epilogue::Store},
                                     tokens, prefill_routes_active())) {
        d.claim = LinearClaim::RouteTable;
    }
    return d;
}

const char* linear_claim_name(LinearClaim claim) {
    switch (claim) {
    case LinearClaim::Owned: return "owned";
    case LinearClaim::Invariant: return "invariant";
    case LinearClaim::Unregistered: return "unregistered";
    case LinearClaim::OwnedDecodeR8: return "owned_decode_r8";
    case LinearClaim::PrefillA8: return "prefill_a8";
    case LinearClaim::OwnedPrefillW4a8: return "owned_prefill_w4a8";
    case LinearClaim::VoltaTiled: return "volta_tiled";
    case LinearClaim::RouteTable: return "route_table";
    }
    return "unknown";
}

namespace {
// The launch a shape-owned table selects at width t, named as q8_launch_id names Q8's: the shape,
// and the launch's number in the order the widths 1..t first select its launches (they are
// template instances in each shape's translation unit, with no name here).
template <class Select>
std::string table_launch_id(Select select, std::int32_t n, std::int32_t k, std::int32_t t) {
    const auto at = select(n, k, t);
    std::vector<decltype(at)> seen;
    for (std::int32_t width = 1; width <= t; ++width) {
        const auto launch = select(n, k, width);
        if (std::find(seen.begin(), seen.end(), launch) == seen.end()) { seen.push_back(launch); }
        if (launch == at) { break; }
    }
    const auto index = std::find(seen.begin(), seen.end(), at) - seen.begin();
    return "n" + std::to_string(n) + "_k" + std::to_string(k) + ".launch" + std::to_string(index);
}
} // namespace

std::optional<std::string> owned_launch_id(QType qtype, std::int32_t n, std::int32_t k,
                                           std::int32_t t) {
    switch (qtype) {
    case QType::Q8_G32_FP16: return q8_launch_id(n, k, t);
    case QType::Q4_G64_FP16: return table_launch_id(select_q4_a16_launch, n, k, t);
    case QType::Q5_G64_FP16: return table_launch_id(select_q5_a16_launch, n, k, t);
    case QType::Q6_G64_FP16: return table_launch_id(select_q6_a16_launch, n, k, t);
    case QType::BF16: return table_launch_id(select_bf16_a16_launch, n, k, t);
    default: return std::nullopt; // NVFP4 and FP8 choose inside their dispatch
    }
}

std::vector<LinearCandidate> linear_candidates(const Weight& w, LinearPolicy policy,
                                               std::int32_t tokens) {
    using Status = LinearCandidate::Status;
    if (tokens <= 0) { throw std::invalid_argument("linear candidates: T must be positive"); }
    // dispatch_linear refuses a fragment-order weight before it decides (generic::refuse_frag):
    // nothing runs it.
    if (w.layout == QuantLayout::GgmlFrag) {
        const std::string why = "ggml_frag_v1 is the grouped MoE kernel's layout only";
        return {{"owned:" + generic::format_label(w.qtype), Status::Unavailable, why},
                {"generic", Status::Unavailable, why}};
    }
    const LinearDecision decision = linear_decision(w, policy, tokens);
    const bool generic_reads      = generic::supports(w);
    const bool registered         = linear_shape_registered(w.qtype, w.n, w.k);
    // As dispatch_linear runs the call: the generic path where the decision hands it the call and
    // it reads the weight, else the shape's own kernels where the table registers the shape (the
    // invariant profile refuses instead).
    const bool runs_generic = decision.claim != LinearClaim::Owned && generic_reads;
    const bool runs_owned =
        !runs_generic && registered && decision.claim != LinearClaim::Invariant;
    std::vector<LinearCandidate> out;
    LinearCandidate owned;
    owned.id = w.qtype == QType::Q8_G32_FP16 ? "owned:q8"
                                             : "owned:" + generic::format_label(w.qtype);
    if (!registered) {
        owned.why = "no shape-owned kernel for this [N,K]";
    } else {
        if (const auto launch = owned_launch_id(w.qtype, w.n, w.k, tokens)) {
            owned.id += "." + *launch;
        }
        owned.status = runs_owned ? Status::Selected : Status::Inadmissible;
        if (!runs_owned) { owned.why = linear_claim_name(decision.claim); }
    }
    out.push_back(std::move(owned));
    LinearCandidate route;
    route.id = "generic";
    if (!generic_reads) {
        route.why = "the generic path does not read this format or layout";
    } else {
        const generic::Problem problem{w.qtype, decision.policy, w.n, w.k,
                                       generic::Epilogue::Store};
        route.id += ":" + generic::route_name(
                              generic::select_route(problem, w, tokens, prefill_routes_active()));
        route.status = runs_generic ? Status::Selected : Status::Inadmissible;
        route.why    = runs_generic ? linear_claim_name(decision.claim) : "owned";
    }
    out.push_back(std::move(route));
    return out;
}

} // namespace detail

namespace {

// Names the format: "unsupported" alone does not say which.
std::string unsupported_weight_format(const char* op, QType qtype) {
    return std::string(op) + ": this build has no kernels for weights stored as " +
           detail::generic::format_label(qtype) +
           "; reconvert the model with this build's ninferno-convert";
}

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

bool dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream, bool release_next = false) {
    detail::generic::refuse_frag(w, "linear");
    // The decision (linear_decision, #306): the generic path where a rule claims the call, the
    // shape's own kernels otherwise.
    const detail::LinearDecision decision = detail::linear_decision(w, policy, x.ne[1]);
    policy                                = decision.policy;
    if (decision.claim != detail::LinearClaim::Owned && detail::generic::supports(w)) {
        if (release_next) {
            return detail::generic::linear_release_next(x, w, out, policy, workspace, stream);
        }
        detail::generic::linear(x, w, out, policy, workspace, stream);
        return false;
    }
    if (invariant_routes_active()) {
        throw std::invalid_argument(
            "linear: the invariant route profile (--activations exact) runs on the generic path, "
            "which cannot read this weight's format or layout");
    }
    switch (w.qtype) {
    case QType::Q4_G64_FP16:
        detail::q4_dispatch(x, w, out, policy, stream);
        return false;
    case QType::Q5_G64_FP16:
        detail::q5_dispatch(x, w, out, policy, stream);
        return false;
    case QType::Q6_G64_FP16:
        detail::q6_dispatch(x, w, out, policy, stream);
        return false;
    case QType::Q8_G32_FP16:
        detail::q8_dispatch(x, w, out, policy, stream);
        return false;
    case QType::BF16:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return false;
    case QType::NVFP4:
#ifdef NINFERNO_WITHOUT_NVFP4
        throw std::invalid_argument("NVFP4 linear is a sm_120a path; this build has no kernels");
#else
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
#endif
        return false;
    case QType::FP8_E4M3FN_ROW_BF16:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return false;
    case QType::MXFP4:
    case QType::FP8_E4M3FN_ROW_F32:
    case QType::FP8_E4M3FN_B128_F32:
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    case QType::GGML_TQ1_0:
    case QType::GGML_TQ2_0:
    case QType::GGML_Q1_0:
    case QType::GGML_Q2_0:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument(unsupported_weight_format("linear", w.qtype));
}

} // namespace

static std::size_t own_linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                       std::int32_t input_rows, LinearPolicy policy,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }
    const auto generic_bytes = [&] {
        return detail::generic::workspace_capacity_bytes(
            {qtype, policy, output_rows, input_rows, detail::generic::Epilogue::Store}, nullptr,
            min_tokens, max_tokens);
    };
    // As dispatch_linear routes: a shape the tables do not own, and every shape under the
    // invariant route profile, runs on the generic path.
    if (invariant_routes_active() ||
        !detail::linear_shape_registered(qtype, output_rows, input_rows)) {
        // --prefill-route w4a8 / i8 (ops/route_pins.h) promote a Q4/Q5/Q6 G64 weight recorded
        // A16Only to int8 at prefill on every card (prefill_a8_policy), which dispatch_linear
        // applies to a shape the tables do not own too: its planes, reserved under that policy.
        if (const PrefillRoute pin = ops::route_pins().prefill;
            !invariant_routes_active() && (pin == PrefillRoute::W4a8 || pin == PrefillRoute::I8)) {
            return std::max(generic_bytes(), detail::generic::prefill_a8_workspace_bytes(
                                                 {qtype, policy, output_rows, input_rows,
                                                  detail::generic::Epilogue::Store},
                                                 min_tokens, max_tokens));
        }
        return generic_bytes();
    }
    if (const std::size_t a8 = detail::generic::prefill_a8_workspace_bytes(
            {qtype, policy, output_rows, input_rows, detail::generic::Epilogue::Store}, min_tokens,
            max_tokens);
        a8 != 0) {
        return std::max(generic_bytes(), a8);
    }

    switch (qtype) {
    case QType::Q4_G64_FP16:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5_G64_FP16:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6_G64_FP16:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q8_G32_FP16:
        (void)detail::select_q8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::BF16:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
#ifdef NINFERNO_WITHOUT_NVFP4
        // Without the NVFP4 kernels (sm_89, sm_86 builds) every NVFP4 call runs on the generic
        // path, whose W4A8 route quantizes the activation into the workspace.
        return generic_bytes();
#else
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
#endif
    case QType::FP8_E4M3FN_ROW_BF16:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::MXFP4:
    case QType::FP8_E4M3FN_ROW_F32:
    case QType::FP8_E4M3FN_B128_F32:
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    case QType::GGML_TQ1_0:
    case QType::GGML_TQ2_0:
    case QType::GGML_Q1_0:
    case QType::GGML_Q2_0:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument(unsupported_weight_format("linear workspace", qtype));
}

LinearPolicy verify_linear_policy(LinearPolicy recorded) noexcept {
    return recorded == LinearPolicy::AllowA4 ? LinearPolicy::AllowA8 : recorded;
}

LinearPolicy drafter_linear_policy(const Weight& weight, LinearPolicy recorded, DrafterOp op) {
    // R8 takes an eight-token block of Q8 only (r8_verify_widest). A Q8 shape the table of the op
    // that runs it owns keeps that table's A16 kernels at 2..8 tokens (the dispatcher sends only a
    // one-token call of it to R8), so lifting its policy would change nothing; the segmented
    // projection reads no table.
    const bool owned = [&] {
        switch (op) {
        case DrafterOp::Linear:
            return detail::linear_shape_registered(weight.qtype, weight.n, weight.k);
        case DrafterOp::LinearAdd:
            return detail::q8_linear_add_registered(weight.n, weight.k);
        case DrafterOp::LinearSwiGlu:
            return detail::q8_linear_swiglu_registered(weight.n, weight.k);
        case DrafterOp::Generic:
            return false;
        }
        return true;
    }();
    if (recorded != LinearPolicy::A16Only || weight.qtype != QType::Q8_G32_FP16 || owned ||
        !detail::generic::r8_takes_block(weight)) {
        return recorded;
    }
    return LinearPolicy::AllowA8;
}

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    // A Volta prefill call decodes its weight for CUTLASS in this workspace, as linear_add's and
    // linear_swiglu's queries count; without it an owned shape fell back to the tiled kernel.
    const std::size_t own =
        std::max(own_linear_workspace_capacity_bytes(qtype, output_rows, input_rows, policy,
                                                     min_tokens, max_tokens),
                 detail::generic::volta_prefill_workspace_bytes(qtype, output_rows, input_rows,
                                                                max_tokens, false));
    // A decode call of an owned shape may take the generic R8 (owned_decode_r8).
    if (!allows_a8(policy) ||
        min_tokens > detail::generic::kR8DecodeTokens) {
        return own;
    }
    return std::max(own,
                    detail::generic::workspace_capacity_bytes(
                        {qtype, policy, output_rows, input_rows, detail::generic::Epilogue::Store},
                        nullptr, min_tokens, detail::generic::kR8DecodeTokens));
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

bool linear_release_next(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    return dispatch_linear(x, w, out, policy, &workspace, stream, true);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

} // namespace ninferno::ops
