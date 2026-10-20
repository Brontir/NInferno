#include "core/device_class.h"
#include "core/generation.h"
#include "core/weight.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/prefill_routes.h"

#include "ops/common/native_nvfp4.h"
#include "ops/generic/generic_gemm.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops {
namespace {

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

// Whether a shape-owned linear_swiglu plan registers the problem. Anything else takes the generic
// path, which applies SiLU to the FP32 gate accumulator and multiplies by the up accumulator
// before its one rounding.
bool registered_shape(QType qtype, std::int32_t gate_up_rows, std::int32_t k) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return detail::q4_linear_swiglu_registered(gate_up_rows, k);
    case QType::Q8_G32_FP16:
        return detail::q8_linear_swiglu_registered(gate_up_rows, k);
    case QType::NVFP4:
        return detail::kNativeNvfp4 && detail::nvfp4_linear_swiglu_registered(gate_up_rows, k);
    case QType::FP8_E4M3FN_ROW_BF16:
        return detail::fp8_linear_swiglu_registered(gate_up_rows, k);
    default:
        return false;
    }
}

// Whether the shape's own plan runs the call. Under the invariant route profile the generic path
// owns every shape it reads, except the NVFP4 plan at A16Only: its ExactA16 route gives a column
// the same bytes at every width (the exact contract it was built for).
bool shape_plan_runs(QType qtype, std::int32_t gate_up_rows, std::int32_t k, LinearPolicy policy) {
    return registered_shape(qtype, gate_up_rows, k) &&
           (!invariant_routes_active() ||
            (qtype == QType::NVFP4 && policy == LinearPolicy::A16Only));
}

} // namespace

static std::size_t
own_linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                           std::int32_t input_rows, LinearPolicy policy,
                                           std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
    if (!shape_plan_runs(qtype, gate_up_rows, input_rows, policy)) {
        return detail::generic::workspace_capacity_bytes(
            {qtype, policy, gate_up_rows / 2, input_rows, detail::generic::Epilogue::SwiGlu},
            nullptr, min_tokens, max_tokens);
    }
    if (qtype == QType::Q8_G32_FP16) {
        (void)detail::q8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens});
        (void)detail::q8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, max_tokens});
        return 0;
    }
    if (qtype == QType::Q4_G64_FP16) {
        return detail::q4_linear_swiglu_capacity_workspace_bytes(
            gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4 && gate_up_rows == 34816 && input_rows == 5120) {
#ifdef NINFERNO_WITHOUT_NVFP4
        detail::no_native_nvfp4("linear_swiglu");
#else
        return detail::nvfp4_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
#endif
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16 && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::fp8_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    throw std::invalid_argument("linear_swiglu workspace: unsupported weight format");
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    // Volta: from 17 tokens a tiled format runs on the generic tiled route (linear_swiglu's
    // volta_tiled_route), not the shape's plan, so the plan holds only the narrower calls and the
    // generic query the rest.
    if (detail::generic::volta_tiled_format(qtype) && max_tokens >= 17) {
        const std::size_t own = min_tokens <= 16
                                    ? own_linear_swiglu_workspace_capacity_bytes(
                                          qtype, gate_up_rows, input_rows, policy, min_tokens, 16)
                                    : 0;
        return std::max(own, detail::generic::workspace_capacity_bytes(
                                 {qtype, policy, gate_up_rows / 2, input_rows,
                                  detail::generic::Epilogue::SwiGlu},
                                 nullptr, std::max(min_tokens, 17), max_tokens));
    }
    // A call of 17 tokens or more takes int8 activations inside a prefill (prefill_a8_policy) and
    // the shape's plan outside one (a DFlash verify of 2 x 16 columns), so the query covers both: it
    // returned the int8 tier's alone (Q4 SwiGLU T = 33: 211 KB against 2.3 MB the plan ran).
    const std::size_t a8 = detail::generic::prefill_a8_workspace_bytes(
        {qtype, policy, gate_up_rows / 2, input_rows, detail::generic::Epilogue::SwiGlu}, min_tokens,
        max_tokens);
    // A Volta prefill call decodes its gate and up rows for CUTLASS in this workspace.
    const std::size_t own = std::max(
        own_linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows, policy,
                                                   min_tokens, max_tokens),
        detail::generic::volta_prefill_workspace_bytes(qtype, gate_up_rows / 2, input_rows,
                                                       max_tokens, true));
    const std::size_t fork_a8 =
        qtype == QType::Q4_G64_FP16 && gate_up_rows == 34816 && input_rows == 5120 &&
                max_tokens >= 128 && detail::q4a8_swiglu_pays()
            ? detail::q4a8_swiglu_workspace_capacity_bytes(std::max(min_tokens, 128), max_tokens)
            : 0;
    return std::max({a8, own, fork_a8});
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                  LinearPolicy::A16Only, min_tokens, max_tokens);
}

namespace {
// ninfer-all's dedicated int8 gate_up kernel (q4a8_linear_swiglu) for the 27B Q4 G64 34816x5120,
// on the cards where it beats the generic route (q4a8_swiglu_pays). It takes the place of the
// prefill int8 tier, so it runs where that tier runs; where FP32-accumulate HMMA runs at full
// rate (the A100) the tier stays off only because its generic kernel loses there
// (prefill_a8_policy), and this one runs in its place inside a prefill.
// --activations exact keeps A16 everywhere.
bool dedicated_q4a8_swiglu(bool prefill_a8) {
    if (invariant_routes_active() || !detail::q4a8_swiglu_pays()) return false;
    const bool full_rate = [] {
        try {
            const auto& d = device_class();
            return generation_row(d.major, d.minor).fp32acc_full_rate;
        } catch (const std::exception&) { return false; }
    }();
    return prefill_a8 || (full_rate && prefill_routes_active());
}
} // namespace

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    detail::generic::refuse_frag(gate_up_weight, "linear_swiglu");
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t         = x.ne[1];
    const LinearPolicy a8_policy = detail::generic::prefill_a8_policy(gate_up_weight.qtype, policy);
    const bool prefill_a8 = detail::generic::q8_prefill_a8_fused(gate_up_weight, a8_policy, t);
    if (dedicated_q4a8_swiglu(prefill_a8) && detail::q4a8_swiglu_supported(gate_up_weight, t) &&
        x.dtype == DType::BF16 && x.is_contiguous() && aligned_to(x.data, 16) &&
        out.is_contiguous()) {
        detail::q4a8_swiglu_launch(x, gate_up_weight, out, ws, stream);
        return;
    }
    // An NVFP4 prefill call the generic W4A8 takes (owned_prefill_w4a8) leaves the plan, which has
    // no 8-bit form.
    const bool registered =
        shape_plan_runs(gate_up_weight.qtype, gate_up_weight.n, gate_up_weight.k, policy) &&
        (gate_up_weight.layout != QuantLayout::RowSplit ||
         gate_up_weight.padded_shape[1] == gate_up_weight.k) &&
        !prefill_a8 && !detail::generic::owned_prefill_w4a8(gate_up_weight, policy, t) &&
        !detail::generic::volta_tiled_route(gate_up_weight, policy, t);
    if (!registered && detail::generic::supports(gate_up_weight)) {
        if (t <= 0 || x.ne[0] != gate_up_weight.k || out.ne[0] * 2 != gate_up_weight.n ||
            out.ne[1] != t) {
            throw std::invalid_argument("linear_swiglu: invalid tensor shape");
        }
        detail::generic::linear_swiglu(x, gate_up_weight, out, prefill_a8 ? a8_policy : policy, &ws,
                                       stream);
        return;
    }
    if (!registered && invariant_routes_active()) {
        throw std::invalid_argument(
            "linear_swiglu: the invariant route profile (--activations exact) runs on the generic "
            "path, which cannot read this weight's format or layout");
    }
    const bool large_shape = x.ne[0] == 5120 && out.ne[0] == 17408 && gate_up_weight.n == 34816 &&
                             gate_up_weight.k == 5120 && gate_up_weight.padded_shape[0] == 34816 &&
                             gate_up_weight.padded_shape[1] == 5120;
    const bool q8_shape = x.ne[0] == 2048 && out.ne[0] == 6144 && gate_up_weight.n == 12288 &&
                          gate_up_weight.k == 2048 && gate_up_weight.padded_shape[0] == 12288 &&
                          gate_up_weight.padded_shape[1] == 2048;
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || (!large_shape && !q8_shape)) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_swiglu: x/out must be non-null and 16-byte aligned");
    }

    const bool common_row_split =
        gate_up_weight.layout == QuantLayout::RowSplit &&
        gate_up_weight.scale_dtype == DType::FP16 && gate_up_weight.ndim == 2 &&
        gate_up_weight.shape[0] == gate_up_weight.n &&
        gate_up_weight.shape[1] == gate_up_weight.k && gate_up_weight.qdata != nullptr &&
        gate_up_weight.scales != nullptr;
    const bool q4_weight = large_shape && gate_up_weight.qtype == QType::Q4_G64_FP16 &&
                           gate_up_weight.group_size == 64 && gate_up_weight.group == 64 &&
                           common_row_split;
    const bool q8_weight =
        (q8_shape || large_shape) && gate_up_weight.qtype == QType::Q8_G32_FP16 &&
        gate_up_weight.group_size == 32 && gate_up_weight.group == 32 &&
        gate_up_weight.qhigh == nullptr && gate_up_weight.high_plane_bytes == 0 && common_row_split;
    const bool nvfp4_weight = large_shape && gate_up_weight.qtype == QType::NVFP4;
    const bool fp8_weight   = large_shape && gate_up_weight.qtype == QType::FP8_E4M3FN_ROW_BF16;
    if (!q4_weight && !q8_weight && !nvfp4_weight && !fp8_weight) {
        throw std::invalid_argument("linear_swiglu: unsupported weight");
    }

    if (fp8_weight) {
        (void)detail::validate_fp8_weight(gate_up_weight, "fp8 linear_swiglu");
        detail::fp8_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (nvfp4_weight) {
#ifdef NINFERNO_WITHOUT_NVFP4
        detail::no_native_nvfp4("linear_swiglu");
#else
        (void)detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 linear_swiglu");
        detail::nvfp4_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
#endif
    }

    if (!aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, q8_weight ? 16 : 4)) {
        throw std::invalid_argument("linear_swiglu: required code/scale alignment is missing");
    }

    if (q8_weight) {
        detail::q8_linear_swiglu_dispatch(x, gate_up_weight, out, stream);
    } else {
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, ws, stream);
    }
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, stream);
}

bool linear_geglu_supports(const Weight& gate_up_weight) {
    return gate_up_weight.layout != QuantLayout::GgmlFrag && (gate_up_weight.n % 2) == 0 &&
           detail::generic::supports(gate_up_weight);
}

std::size_t linear_geglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                  std::int32_t input_rows, LinearPolicy policy,
                                                  std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_geglu workspace: invalid profile or token interval");
    }
    return detail::generic::workspace_capacity_bytes(
        {qtype, policy, gate_up_rows / 2, input_rows, detail::generic::Epilogue::SwiGlu}, nullptr,
        min_tokens, max_tokens);
}

void linear_geglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    detail::generic::refuse_frag(gate_up_weight, "linear_geglu");
    if (!linear_geglu_supports(gate_up_weight)) {
        throw std::invalid_argument("linear_geglu: the generic path cannot read this weight");
    }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_geglu: x/out must be BF16");
    }
    const std::int32_t t = x.ne[1];
    if (t <= 0 || x.ne[0] != gate_up_weight.k || out.ne[0] * 2 != gate_up_weight.n ||
        out.ne[1] != t) {
        throw std::invalid_argument("linear_geglu: invalid tensor shape");
    }
    detail::generic::linear_geglu(x, gate_up_weight, out, policy, &ws, stream);
}

} // namespace ninferno::ops
