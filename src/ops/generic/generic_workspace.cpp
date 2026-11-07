// The generic path's workspace (#refactor B1): the planes a quantized activation takes from the
// caller's arena (activation_planes, the one description run_route allocates by), a route's bytes
// (route_workspace_bytes), and the capacity a caller reserves for a token interval
// (workspace_capacity_bytes, volta_prefill_workspace_bytes).
#include "ops/generic/generic_gemm.h"
#include "core/per_device.h"
#include "ops/common/arch_limits.h"
#include "ops/common/native_nvfp4.h"

#include "core/device_class.h"
#include "core/generation.h"
#include "core/measured_boards.h"
#include "core/tensor.h"
#include "core/layout.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/route_probe.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/route_cost_constants.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/r8_decode_table.h"
#include "ops/generic/route_table.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/route_pins.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::ops::detail::generic {

// The quantized activation's planes a route reads from the caller's workspace, in bytes and in
// the order run_route allocates them (each 256-aligned): the codes, the scales and the sums (W4A4
// has no sums plane: 0). One description for both, so the workspace a query sizes is the one the
// launch takes. I8 pads tokens (i8_padded_tokens) and groups K by 64; W4A8 reads rows of whole
// tiles (w4a8_ld), R8's verify rows of K, both in blocks of 32; W4A4 two codes a byte and one
// scale per 16 values.
ActivationPlanes activation_planes(Route route, std::int32_t k, std::int32_t tokens) {
    switch (route.family) {
    case Family::I8Mma: {
        const auto t      = static_cast<std::size_t>(i8_padded_tokens(tokens));
        const auto groups = static_cast<std::size_t>(k / 64);
        return {t * static_cast<std::size_t>(k), t * groups * sizeof(float),
                t * groups * sizeof(int)};
    }
    case Family::W4a8Mma:
    case Family::R8Mma: {
        const std::int32_t ld = route.family == Family::W4a8Mma ? w4a8_ld(k) : k;
        const auto t          = static_cast<std::size_t>(tokens);
        const auto blocks     = static_cast<std::size_t>(ld / 32);
        return {t * static_cast<std::size_t>(ld), t * blocks * sizeof(float),
                t * blocks * sizeof(float)};
    }
    case Family::W4a4Mma: {
        const auto t = static_cast<std::size_t>(tokens);
        return {t * static_cast<std::size_t>(k / 2), t * static_cast<std::size_t>(k / 16), 0};
    }
    default:
        return {};
    }
}
} // namespace

std::size_t route_workspace_bytes(Route route, std::int32_t k, std::int32_t tokens) {
    if (route.family == Family::GgmlVec) return vec_workspace_bytes(k, tokens);
    // R8 at verify widths reads the W4A8 planes (run_route's r8_pre: every r8_verify_compiled
    // format, whichever device routes to it).
    const bool r8_verify =
        route.family == Family::R8Mma && tokens >= 2 && tokens <= kR8VerifyTokens;
    if (route.family != Family::I8Mma && route.family != Family::W4a8Mma &&
        route.family != Family::W4a4Mma && !r8_verify) {
        return 0;
    }
    const ActivationPlanes planes = activation_planes(route, k, tokens);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(planes.codes, 256);
    (void)layout.alloc_bytes(planes.scales, 256);
    if (planes.sums > 0) (void)layout.alloc_bytes(planes.sums, 256);
    return layout.peak_bytes(1);
}

std::size_t volta_prefill_workspace_bytes(QType qtype, std::int32_t rows, std::int32_t k,
                                          std::int32_t max_tokens, bool swiglu) {
    if (max_tokens < 17 || !volta_cutlass_runs(volta_cutlass_only_qtype(qtype), max_tokens) ||
        !volta_cutlass_qtype(qtype))
        return 0;
    return volta_cutlass_workspace_bytes(rows, k, max_tokens, swiglu);
}

std::size_t workspace_capacity_bytes(const Problem& problem, const Weight* weight,
                                     std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("generic linear workspace: invalid token interval");
    }
    // A u4 weight stored in another column order gathers its input into [K, T] first, ahead of
    // any int8 planes (run allocates both). The callers that size a workspace by format alone
    // reserve it for every u4 weight. Before, the gather was all a u4 weight reserved, short of
    // the W4A8 / R8 planes its routes allocate after it.
    std::size_t gather = 0;
    if (u4_zp_group(problem.qtype) && (weight == nullptr || weight->k_order != nullptr)) {
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc_bytes(static_cast<std::size_t>(max_tokens) *
                                     static_cast<std::size_t>(problem.k) * 2,
                                 256);
        gather = layout.peak_bytes(1);
    }
    const auto with_gather = [gather](std::size_t planes) {
        if (gather == 0) return planes;
        if (planes == 0) return gather;
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc_bytes(gather, 256);
        (void)layout.alloc_bytes(planes, 256);
        return layout.peak_bytes(1);
    };
    // The W4A4 plane grows with T; any W4A4 route in the interval is bounded by the largest T.
    // A build without the W4A4 kernels (sm_89, sm_86) runs those calls on W4A8 (below).
    const std::int8_t forced = route_probe_state.quantized_act;
    const bool a4            = problem.qtype == QType::NVFP4 && w4a4_schedule_count() > 0 &&
                    allows_a4(problem.policy) &&
                    (weight == nullptr || allows_w4a4(*weight, problem.policy)) && forced != 0 &&
                    (max_tokens >= 17 || forced == 1);
    if (a4) {
        return with_gather(route_workspace_bytes(Route{Family::W4a4Mma, 0}, problem.k, max_tokens));
    }
    // The W4A8 planes likewise (int8 codes, FP32 scales and sums). Without tensor mma.sync a call
    // of 17 tokens or more whose format has a tiled codec takes the tiled route whatever the policy
    // allows (volta_tiled_route) and reads no planes: an exact query of such a width reserved them.
    const bool volta_tiled = !current_features().mma_tensor && volta_tiled_qtype(problem.qtype);
    const bool a8          = w4a8_supports(problem.qtype) && allows_a8(problem.policy) &&
                    w4a8_k_admits(problem.qtype, problem.k) &&
                    (weight == nullptr || allows_w4a8(*weight, problem.policy)) && forced != 0 &&
                    ((max_tokens >= 17 && !volta_tiled) || forced == 1);
    // R8 at verify widths reads the same planes (the activation quantized once per call). Every
    // format with the pre-quantized form: the dense tuner may give it R8 at verify widths.
    const bool r8_verify = r8_verify_compiled(problem.qtype) && allows_a8(problem.policy) &&
                           (problem.k % r8_k_values(problem.qtype)) == 0 &&
                           (weight == nullptr || allows_r8(*weight, problem.policy)) &&
                           forced != 0 && max_tokens >= 2 && min_tokens <= kR8VerifyTokens;
    // ggml's matrix kernel: its activation, the FP32 plane (SwiGlu: gate and up rows) and the
    // stream-k fixup, at the widest call of its band in the interval.
    std::size_t mmq = 0;
    if (weight == nullptr || mmq_admits(*weight, problem.policy)) {
        // A prefill runs any width of 17 tokens or more in slices of at most kMmqPrefillSlice: an
        // interval wholly above the slice (513..1024) still needs one slice's planes.
        const bool prefill    = ggml_mmq_format(problem.qtype);
        const std::int32_t hi = prefill ? max_tokens : std::min(max_tokens, kMmqMaxTokens);
        const std::int32_t t  = std::min(max_tokens, prefill ? kMmqPrefillSlice : kMmqMaxTokens);
        if (ggml_mmq_format(problem.qtype) && allows_a8(problem.policy) && forced != 0 &&
            hi >= std::max(min_tokens, kMmqMinTokens) && (problem.n % ggml_mmq::kRowMultiple) == 0 &&
            (problem.k % ggml_mmq::kKMultiple) == 0 && ggml_mmq::device_supported()) {
            const std::int32_t plane_rows =
                problem.epilogue == Epilogue::SwiGlu ? 2 * problem.n : problem.n;
            mmq = mmq_workspace_bytes(problem.k, plane_rows, t) + kGgmlArenaSlack;
        }
    }
    // ggml's vector kernel: its activation at the widest call of 1..8 tokens in the interval (a
    // part of eight of a wider call where it takes one, vec_max_tokens).
    if (ggml_vec_format(problem.qtype) && allows_a8(problem.policy) && forced != 0 &&
        (problem.k % 32) == 0 && min_tokens <= vec_max_tokens() &&
        (weight == nullptr || vec_admits(*weight, problem.policy))) {
        mmq = std::max(
            mmq, vec_workspace_bytes(problem.k,
                                     std::min<std::int32_t>(max_tokens, ggml_mmq::kVecMaxColumns)) +
                     kGgmlArenaSlack);
    }
    // Volta's tiled route decodes into FP16 rows for CUTLASS (volta_cutlass_runs).
    const bool only = weight != nullptr ? volta_cutlass_only(codec_of(*weight))
                                        : volta_cutlass_only_qtype(problem.qtype);
    const std::size_t volta =
        volta_cutlass_runs(only, max_tokens) &&
                (weight == nullptr || tiled_takes(codec_of(*weight)) || only)
            ? volta_cutlass_workspace_bytes(problem.n, problem.k, max_tokens,
                                            problem.epilogue == Epilogue::SwiGlu)
            : 0;
    mmq = std::max(mmq, volta);
    if (!a8 && !r8_verify) return with_gather(mmq);
    const std::size_t i8 = (problem.qtype == QType::Q4_G64_FP16 ||
                            problem.qtype == QType::Q5_G64_FP16 ||
                            problem.qtype == QType::Q6_G64_FP16)
                               ? route_workspace_bytes(Route{Family::I8Mma, 0}, problem.k, max_tokens)
                               : 0;
    return with_gather(std::max(
        {mmq, i8, route_workspace_bytes(Route{Family::W4a8Mma, 0}, problem.k, max_tokens)}));
}

} // namespace ninferno::ops::detail::generic
