#include "ops/generic/generic_gemm.h"
#include "ops/generic/format_traits.h"
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
#include "ops/generic/generic_i8_schedules.h"
#include "ops/generic/generic_select_detail.h"
#include "ops/generic/generic_select_detail.h"
#include "ops/generic/generic_schedule_tables.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/r8_decode_table.h"
#include "ops/generic/route_table.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/route_pins.h"


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

GemmArgs weight_args(const Weight& w) {
    GemmArgs a;
    a.k      = w.k;
    a.codes  = static_cast<const std::uint8_t*>(w.qdata);
    a.high   = static_cast<const std::uint8_t*>(w.qhigh);
    a.scales = static_cast<const std::uint8_t*>(w.scales);
    switch (w.qtype) {
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
        a.groups_per_row = w.padded_shape[1] / 64;
        break;
    case QType::Q8_G32_FP16:
        a.groups_per_row = w.padded_shape[1] / 32;
        break;
    case QType::FP8_E4M3FN_B128_F32:
        a.scale_tiles = (w.k + 127) / 128;
        break;
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
        a.groups_per_row = w.k / static_cast<int>(u4_zp_group(w.qtype));
        break;
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
        a.groups_per_row = w.padded_shape[1] / static_cast<int>(q3_group(w.qtype));
        break;
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
        a.groups_per_row = w.k / static_cast<int>(ggml_block_values(w.qtype));
        break;
    case QType::NVFP4:
        a.scale_tiles     = w.k / 64;
        a.inverse_divisor = 1.0F / w.weight_scale_divisor;
        if (w.weight_divisors != nullptr && w.weight_divisor_rows > 0 &&
            w.weight_divisor_rows < w.n) {
            a.divisors     = static_cast<const float*>(w.weight_divisors);
            a.divisor_rows = w.weight_divisor_rows;
        }
        break;
    default:
        break;
    }
    return a;
}

bool allows_i8(const Weight& w, LinearPolicy policy) {
    return allows_a8(policy) && i8_format(w.qtype) && i8_extent(w) && i8_planes_aligned(w);
}

namespace {

Route w4a8_classic_twin(Route route) {
    const auto s = w4a8_schedule_info(route.schedule);
    for (int i = 0; i < w4a8_schedule_count(); ++i) {
        const auto c = w4a8_schedule_info(i);
        if (c.producers == 0 && c.block_m == s.block_m && c.block_n == s.block_n) {
            return Route{Family::W4a8Mma, static_cast<std::uint8_t>(i)};
        }
    }
    throw std::logic_error("generic W4A8: a producer-warp tile without a classic twin");
}

Route w4a8_ws_twin(Route route) {
    const auto s = w4a8_schedule_info(route.schedule);
    for (int i = 0; i < w4a8_schedule_count(); ++i) {
        const auto c = w4a8_schedule_info(i);
        if (c.producers > 0 && c.block_m == s.block_m && c.block_n == s.block_n) {
            return Route{Family::W4a8Mma, static_cast<std::uint8_t>(i)};
        }
    }
    return route;
}


} // namespace

std::size_t vec_workspace_bytes(std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(ggml_mmq::vec_activation_bytes(k, tokens), 256);
    return layout.peak_bytes(1);
}

std::size_t mmq_workspace_bytes(std::int32_t k, std::int32_t plane_rows, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(ggml_mmq::activation_bytes(k, tokens), 256);
    (void)layout.alloc_bytes(static_cast<std::size_t>(plane_rows) * tokens * sizeof(float), 256);
    (void)layout.alloc_bytes(ggml_mmq::fixup_bound(tokens), 256);
    return layout.peak_bytes(1);
}

Route w4a8_executed_route(Route route, const Weight& w) {
    if (route.family != Family::W4a8Mma) return route;
    const bool producers = w4a8_schedule_info(route.schedule).producers > 0;
    if (producers && !w4a8_ws_admits(w)) return w4a8_classic_twin(route);
    if (!producers && w4a8_ws_extra(w.qtype) && w4a8_ws_admits(w)) return w4a8_ws_twin(route);
    return route;
}

bool volta_tiled_qtype(QType qtype) {
    Weight w{};
    w.qtype = qtype;
    try {
        return tiled_takes(codec_of(w));
    } catch (const std::exception&) { return false; }
}

bool volta_cutlass_qtype(QType qtype) {
    if (volta_tiled_qtype(qtype)) return true;
    Weight w{};
    w.qtype = qtype;
    try {
        return volta_cutlass_only(codec_of(w));
    } catch (const std::exception&) { return false; }
}

bool volta_cutlass_only_qtype(QType qtype) {
    Weight w{};
    w.qtype = qtype;
    try {
        return volta_cutlass_only(codec_of(w));
    } catch (const std::exception&) { return false; }
}

std::optional<RouteTile> route_tile(Route route, Epilogue epilogue, std::int32_t tokens) {
    const int sets = epilogue == Epilogue::SwiGlu ? 2 : 1;
    switch (route.family) {
    case Family::RegisterMma: {
        const auto s = rmma_schedule_info(route.schedule);
        return RouteTile{16 * s.m_tiles * s.warps_m, 8 * s.n_tiles, s.cluster};
    }
    case Family::TiledMma: {
        const auto s = tiled_schedule_info(route.schedule);
        return RouteTile{s.block_n / sets, s.block_m, 1};
    }
    case Family::StagedMma: {
        const auto s = staged_schedule_info(route.schedule);
        return RouteTile{s.rows / sets, s.tokens, 1};
    }
    case Family::W4a8Mma: {
        const auto s = w4a8_schedule_info(route.schedule);
        return RouteTile{s.block_n / sets, s.block_m, 1};
    }
    case Family::I8Mma: {
        const auto s = i8_schedule_info(route.schedule);
        return RouteTile{s.block_n / sets, s.block_m, 1};
    }
    default: return std::nullopt;
    }
}

std::string_view demotion_name(Demotion demotion) noexcept {
    switch (demotion) {
    case Demotion::None: return "none";
    case Demotion::Alignment: return "alignment";
    case Demotion::W4a8Admission: return "w4a8_admission";
    case Demotion::R8Fit: return "r8_fit";
    case Demotion::GgmlLayout: return "ggml_layout";
    case Demotion::GgmlWorkspace: return "ggml_workspace";
    case Demotion::Workspace: return "workspace";
    }
    return "none";
}

Route workspace_free_route(Route selected, const Problem& problem, const Weight& weight,
                           std::int32_t tokens, bool prefill, bool workspace) {
    if ((selected.family == Family::W4a4Mma || selected.family == Family::W4a8Mma ||
         selected.family == Family::I8Mma) &&
        !workspace) {
        return heuristic_route(
            Problem{problem.qtype, LinearPolicy::A16Only, problem.n, problem.k, problem.epilogue},
            weight, tokens, prefill);
    }
    return selected;
}

ResolvedRoute resolve_route(Route asked, const Weight& w, const CallLayout& call) {
    ResolvedRoute out{.asked = asked, .executed = asked};
    Route& route      = out.executed;
    const bool x16    = call.x_aligned16;
    const bool staged = x16 && (call.ldx % 8) == 0; // each token in 16-byte copies
    const auto demote = [&](Route to, Demotion why) {
        if (to.family != route.family || to.schedule != route.schedule) out.reason = why;
        route = to;
    };
    if (!x16 && route.family == Family::RegisterMma) demote(Route{Family::Scalar, 0}, Demotion::Alignment);
    if (route.family == Family::TiledMma && !staged) demote(Route{Family::Scalar, 0}, Demotion::Alignment);
    // The NVFP4 tiled route stages activations in 16-byte copies; otherwise the register MMA of
    // the same order takes the call (or the scalar route where that cannot read x either).
    if (route.family == Family::Nvfp4TiledMma && !staged) {
        const int classes = nvfp4_tiled_schedule_info(route.schedule).classes;
        demote(x16 ? Route{Family::RegisterMma,
                           static_cast<std::uint8_t>(classes == 1 ? rmma_index(4, 1, 1, 8)
                                                                  : rmma_index(1, classes, 1, 4))}
                   : Route{Family::Scalar, 0},
               Demotion::Alignment);
    }
    // The W4A8 quantizer reads 32 BF16 values of a token as four 16-byte words, the I8 one 64.
    if ((route.family == Family::W4a8Mma || route.family == Family::I8Mma) && !staged) {
        demote(Route{Family::Scalar, 0}, Demotion::Alignment);
    }
    // A producer-warp W4A8 tile the weight is not admitted to (another format, an IQ3_XXS sign
    // plane off 8 bytes) runs the classic tile of its shape: the same bytes.
    demote(w4a8_executed_route(route, w), Demotion::W4a8Admission);
    if (route.family == Family::StagedMma && !staged) {
        // The register MMA with WarpsK == 1 gives the same bytes; the scalar kernel where even it
        // cannot read x, as for the register MMA.
        demote(x16 ? Route{Family::RegisterMma, static_cast<std::uint8_t>(rmma_index(4, 1, 1, 8))}
                   : Route{Family::Scalar, 0},
               Demotion::Alignment);
    }
    // The A16 route a call the quantized families cannot take runs.
    const auto a16 = [&] {
        const Problem problem{w.qtype, LinearPolicy::A16Only, call.rows, w.k, call.epilogue};
        Route r{Family::RegisterMma,
                static_cast<std::uint8_t>(heuristic_rmma(problem, call.tokens, false))};
        if (!x16 || (w.k % k_multiple(codec_of(w))) != 0) r = Route{Family::Scalar, 0};
        return r;
    };
    // R8 quantizes 32 BF16 values of a token as four 16-byte words, and holds the call's tokens in
    // shared memory; a call it cannot take (a route table or a probe naming it) runs A16.
    // At verify widths a format with the pre-quantized form needs the workspace instead of the fit.
    // Its planes in the workspace free to the call: a workspace short of them takes the CTA's own
    // quantization where the tokens fit it, else A16 (said as a demotion), not an allocation
    // failure at the launch.
    out.r8_prequantized = route.family == Family::R8Mma && call.tokens >= 2 &&
                          call.tokens <= kR8VerifyTokens && call.workspace_free.has_value() &&
                          r8_verify_compiled(w.qtype) &&
                          *call.workspace_free >= route_workspace_bytes(route, w.k, call.tokens);
    if (route.family == Family::R8Mma &&
        (!staged || !(out.r8_prequantized || r8_fits(w.qtype, w.k, call.tokens)))) {
        demote(a16(), staged ? Demotion::R8Fit : Demotion::Alignment);
    }
    if (route.family != Family::R8Mma) out.r8_prequantized = false;
    // The quantized-activation families take their planes from the workspace: given less than
    // they need, the call runs A16 (said as a demotion) instead of failing to allocate at the
    // launch. Without any workspace run() already took A16 (workspace_free_route).
    if ((route.family == Family::W4a4Mma || route.family == Family::W4a8Mma ||
         route.family == Family::I8Mma) &&
        call.workspace_free && *call.workspace_free < route_workspace_bytes(route, w.k, call.tokens)) {
        demote(a16(), Demotion::Workspace);
    }
    // ggml's matrix kernel needs its planes in the workspace and x in 8-byte words; a call it
    // cannot take runs A16 (the same route the heuristic gives without the band).
    const int plane_rows = call.epilogue == Epilogue::SwiGlu ? 2 * call.rows : call.rows;
    const auto free_for  = [&](std::size_t bytes) {
        return call.workspace_free.has_value() && bytes <= *call.workspace_free;
    };
    if (route.family == Family::GgmlMmq) {
        if (!call.workspace_free) {
            demote(a16(), Demotion::GgmlWorkspace);
        } else if (!x16 || (call.ldx % 4) != 0) {
            demote(a16(), Demotion::Alignment);
        } else if ((call.rows % ggml_mmq::kRowMultiple) != 0) {
            demote(a16(), Demotion::GgmlLayout);
        } else if (!free_for(mmq_workspace_bytes(w.k, plane_rows,
                                                 std::min(call.tokens, kMmqPrefillSlice)))) {
            demote(a16(), Demotion::GgmlWorkspace);
        }
        out.ggml_fallback = route.family != Family::GgmlMmq;
    } else if (route.family == Family::GgmlVec) {
        if (call.segments > ggml_mmq::kVecMaxSegments) {
            demote(a16(), Demotion::GgmlLayout);
        } else if (!free_for(vec_workspace_bytes(w.k, std::min(call.tokens, ggml_mmq::kVecMaxColumns)))) {
            demote(a16(), Demotion::GgmlWorkspace);
        }
        out.ggml_fallback = route.family != Family::GgmlVec;
    }
    return out;
}

} // namespace ninferno::ops::detail::generic
