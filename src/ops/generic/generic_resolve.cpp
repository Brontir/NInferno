#include "ops/generic/generic_gemm.h"
#include "ops/generic/format_traits.h"

#include "core/layout.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_select_detail.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"

#include <algorithm>
#include <exception>
#include <optional>
#include <cstdint>
#include <stdexcept>
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

// The classic W4A8 tile of a producer-warp tile's shape (the same bytes).
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

// The producer-warp tile of a classic W4A8 tile's shape, or the route itself.
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

// The W4A8 schedule run_route executes for `route`: a producer-warp tile the weight is not admitted
// to runs the classic tile of its shape, and a w4a8_ws_extra format's classic tile its
// producer-warp twin (the same bytes either way).
Route w4a8_executed_route(Route route, const Weight& w) {
    if (route.family != Family::W4a8Mma) return route;
    const bool producers = w4a8_schedule_info(route.schedule).producers > 0;
    if (producers && !w4a8_ws_admits(w)) return w4a8_classic_twin(route);
    if (!producers && w4a8_ws_extra(w.qtype) && w4a8_ws_admits(w)) return w4a8_ws_twin(route);
    return route;
}

// tiled_takes for a format named by its QType alone (workspace queries without a weight).
bool volta_tiled_qtype(QType qtype) {
    Weight w{};
    w.qtype = qtype;
    try {
        return tiled_takes(codec_of(w));
    } catch (const std::exception&) { return false; }
}

// volta_tiled_qtype, or a CUTLASS-only format (volta_cutlass_only): whose wide calls take the
// decoded copy.
bool volta_cutlass_qtype(QType qtype) {
    if (volta_tiled_qtype(qtype)) return true;
    Weight w{};
    w.qtype = qtype;
    try {
        return volta_cutlass_only(codec_of(w));
    } catch (const std::exception&) { return false; }
}

// volta_cutlass_only for a format named by its QType alone.
bool volta_cutlass_only_qtype(QType qtype) {
    Weight w{};
    w.qtype = qtype;
    try {
        return volta_cutlass_only(codec_of(w));
    } catch (const std::exception&) { return false; }
}

// The int8-activation tiles as candidates of cost_route, where the policy admits them: the W4A8
// tiles (pipelined, at the generation's int8 tensor rate, GenerationRow::s8: 2048 per SM and
// clock, 4096 on the full-rate datacenter rows ga100, hopper and gb100) and ggml's matrix kernel
// (128-row tiles over the SMs, 512 tokens a slice), each with its activation quantizer (1.7 us
// and 4.3 bytes of traffic per value) and a K chain per CTA (W4A8: 1390 cycles a 128-K step, the
// producer-warp tile 1690; mmq: 4075 cycles per 862 K shared by the SMs). The int8 tile pays where
// its grid fills the SMs and its rate outruns the BF16 MMA; a narrow weight (n <= 1024 rows below
// 512 tokens) keeps the A16 split-K stream. A tile reaches a fraction of the int8 peak that depends
// on its shape (its warp tile and what it keeps in registers): 0.31 / 0.28 / 0.24 for 32x128 /
// 64x128 / 128x128, 0.47 for 128x64, 0.37 for 64x64, and 0.98 for the producer-warp 128x64, whose
// producers hide the decode behind the multiplies; the minor-0 parts (A100) reach 0.46 of it.
// SwiGLU: the weight has 2 n rows, a tile of block_n rows makes block_n / 2 outputs (gate and up
// rows interleave), so a call takes twice the tiles and each multiplies block_n rows; `bpw` is the
// payload over the n output rows, i.e. twice the bytes per weight. ggml's kernel takes its calls of
// 17 tokens or more in 512-token slices on every device (ggml_mmq_slice_tokens, whose workspace the
// planner reserves), so it is a candidate wherever the policy admits it; not yet on the minor-0
// datacenter row, where its microbenchmark win (1.26..1.8x the W4A8 tile) and the 24K prefill
// A/B (-15 %) disagree. ninferno_route_tune --a8 and --a8 --epilogue timings of the RTX 4090, 5090,
// 4060 Ti, L4, A10 and A100 (six formats, n 48..28672, T 17..2048, quantizer in the sample; SwiGLU
// and AddResidual on the A100), constants refit with the board held out: 1.06..1.09 of the best
// route of either class on the consumer boards (the minor-0 row has one board: 1.13 in sample).
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
    default:
        return std::nullopt;
    }
}

std::string_view demotion_name(Demotion demotion) noexcept {
    switch (demotion) {
    case Demotion::None:
        return "none";
    case Demotion::Alignment:
        return "alignment";
    case Demotion::W4a8Admission:
        return "w4a8_admission";
    case Demotion::R8Fit:
        return "r8_fit";
    case Demotion::GgmlLayout:
        return "ggml_layout";
    case Demotion::GgmlWorkspace:
        return "ggml_workspace";
    case Demotion::Workspace:
        return "workspace";
    case Demotion::VoltaCopy:
        return "volta_copy";
    case Demotion::KStep:
        return "k_step";
    case Demotion::GroupSums:
        return "group_sums";
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

bool r8_prequantized_takes(Route route, QType qtype, std::int32_t k, std::int32_t tokens,
                           std::optional<std::size_t> workspace_free) {
    return route.family == Family::R8Mma && tokens >= 2 && tokens <= kR8VerifyTokens &&
           workspace_free.has_value() && r8_verify_compiled(qtype) &&
           *workspace_free >= route_workspace_bytes(route, k, tokens);
}

// FUSION-SITE: r8_prequantized
ResolvedRoute resolve_route(Route asked, const Weight& w, const CallLayout& call) {
    ResolvedRoute out{.asked = asked, .executed = asked};
    Route& route      = out.executed;
    const bool x16    = call.x_aligned16;
    const bool staged = x16 && (call.ldx % 8) == 0; // each token in 16-byte copies
    const auto demote = [&](Route to, Demotion why) {
        if (to.family != route.family || to.schedule != route.schedule) out.reason = why;
        route = to;
    };
    if (!x16 && route.family == Family::RegisterMma)
        demote(Route{Family::Scalar, 0}, Demotion::Alignment);
    if (route.family == Family::TiledMma && !staged)
        demote(Route{Family::Scalar, 0}, Demotion::Alignment);
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
    // K off the register kernel's step runs the scalar kernel (other bytes and speed). The
    // selector, the tuner and route-tune never ask it (rmma_refusal refuses the K, #693); a route
    // asked past route_admits is resolved here, so the run and its description read one rule.
    if (route.family == Family::RegisterMma && (w.k % k_multiple(codec_of(w))) != 0) {
        demote(Route{Family::Scalar, 0}, Demotion::KStep);
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
    out.r8_prequantized =
        r8_prequantized_takes(route, w.qtype, w.k, call.tokens, call.workspace_free);
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
        call.workspace_free &&
        *call.workspace_free <
            executed_workspace_bytes(route, false, w.k, call.rows, call.epilogue, call.tokens)) {
        demote(a16(), Demotion::Workspace);
    }
    // ggml's matrix kernel needs its planes in the workspace and x in 8-byte words; a call it
    // cannot take runs A16 (the same route the heuristic gives without the band).
    const auto free_for = [&](std::size_t bytes) {
        return call.workspace_free.has_value() && bytes <= *call.workspace_free;
    };
    if (route.family == Family::GgmlMmq) {
        if (!call.workspace_free) {
            demote(a16(), Demotion::GgmlWorkspace);
        } else if (!x16 || (call.ldx % 4) != 0) {
            demote(a16(), Demotion::Alignment);
        } else if ((call.rows % ggml_mmq::kRowMultiple) != 0) {
            demote(a16(), Demotion::GgmlLayout);
        } else if (!free_for(executed_workspace_bytes(route, false, w.k, call.rows, call.epilogue,
                                                      call.tokens))) {
            demote(a16(), Demotion::GgmlWorkspace);
        }
        out.ggml_fallback = route.family != Family::GgmlMmq;
    } else if (route.family == Family::GgmlVec) {
        if (call.segments > ggml_mmq::kVecMaxSegments) {
            demote(a16(), Demotion::GgmlLayout);
        } else if (!free_for(executed_workspace_bytes(route, false, w.k, call.rows, call.epilogue,
                                                      call.tokens))) {
            demote(a16(), Demotion::GgmlWorkspace);
        }
        out.ggml_fallback = route.family != Family::GgmlVec;
    }
    // A min-term register-MMA call reads its group sums from a plane in its workspace where it
    // takes one (rmma_group_sums, and the instance the launch takes: rmma_plane_taken); without
    // room for it the same route runs the MMA with ones (other bytes: a demotion, counted and
    // said by run_route, unless the call was demoted before: the first reason stands). The K the
    // register kernel cannot step runs scalar at the launch, and so does Volta: neither reads a
    // plane. The plane is off by default (rmma_group_sums), which answers before the launch's
    // description is asked.
    if (route.family == Family::RegisterMma && (w.k % k_multiple(codec_of(w))) == 0 &&
        rmma_group_sums(codec_of(w)) &&
        rmma_plane_taken(route.schedule, w, call.epilogue, call.tokens, call.rows)) {
        if (free_for(rmma_group_sums_bytes(codec_of(w), w.k, call.tokens))) {
            out.group_sums = true;
        } else if (out.reason == Demotion::None) {
            out.reason = Demotion::GroupSums;
        }
    }
    return out;
}

} // namespace ninferno::ops::detail::generic
