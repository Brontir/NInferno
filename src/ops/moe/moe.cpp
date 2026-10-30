// The generic MoE Op: validation, route choice, workspace layout and the launch sequence
// (include/ninferno/ops/moe.h, src/ops/moe/moe_internal.h).
#include "ninferno/ops/moe.h"

#include "core/device.h" // CUDA_CHECK
#include "core/device_class.h"
#include "core/measured_boards.h"
#include "core/ggml_frag.h"
#include "core/layout.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/math.h"
#include "ops/common/native_nvfp4.h"
#include "ops/generic/generic_gemm.h"
#include "ops/moe/moe_internal.h"
#include "ops/common/arch_limits.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::ops {
namespace detail::moe {

namespace {

constexpr std::array<ScheduleInfo, 9> kSchedules{{
#define NINFERNO_MOE_INFO(I, WM, WK, MT, NT) {WM, WK, MT, NT},
    NINFERNO_MOE_SCHEDULES(NINFERNO_MOE_INFO)
#undef NINFERNO_MOE_INFO
}};

constexpr std::array<R8ScheduleInfo, 4> kR8Schedules{{
#define NINFERNO_MOE_R8_INFO(I, W, R) {W, R},
    NINFERNO_MOE_R8_SCHEDULES(NINFERNO_MOE_R8_INFO)
#undef NINFERNO_MOE_R8_INFO
}};

} // namespace

int schedule_count() { return static_cast<int>(kSchedules.size()); }

ScheduleInfo schedule_info(int index) { return kSchedules.at(static_cast<std::size_t>(index)); }

int r8_schedule_count() { return static_cast<int>(kR8Schedules.size()); }

R8ScheduleInfo r8_schedule_info(int index) {
    return kR8Schedules.at(static_cast<std::size_t>(index));
}

bool r8_expert_format(QType format) {
    switch (format) {
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_Q3_K:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ4_XS:
    case QType::MXFP4:
    // The native formats on the codecs generic R8 reads them with (moe_r8_native.cu).
    case QType::Q8_G32_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
        return true;
    // NVFP4 experts decode on R8 only without the native NVFP4 kernels (moe_r8_ggml4.cu).
    case QType::NVFP4:
        return !kNativeNvfp4;
    default: return false;
    }
}

bool r8_expert_supports(const Weight& w) {
    if (!r8_expert_format(w.qtype)) { return false; }
    // The codecs load vectors of up to 16 bytes, aligned within a bank that starts on a 16-byte
    // boundary. A ggml_frag_v1 bank has fragment-order codecs for IQ2_S, IQ3_XXS, Q4_K and Q4_0
    // (moe_r8.cuh) and stays on A16 otherwise.
    if (w.layout == QuantLayout::GgmlFrag && w.qtype != QType::GGML_IQ2_S &&
        w.qtype != QType::GGML_IQ3_XXS && w.qtype != QType::GGML_Q4_K &&
        w.qtype != QType::GGML_Q4_0) {
        return false;
    }
    // K in whole load groups, as generic R8 (Gemma 4 26B's Q4_0 down K 704, NemotronH's 1856; a
    // ggml_frag_v1 Q4_0 row's last K/32 % 4 blocks stay natural).
    return w.k_order == nullptr && generic::r8_supports(w.qtype) &&
           w.k % generic::r8_k_values(w.qtype) == 0 &&
           reinterpret_cast<std::uintptr_t>(w.qdata) % 16 == 0;
}

// Lanes per row of an R8 warp on this bank (GemmArgs::r8_lanes): the fewest of 8, 16, 32 that
// cover the row's loads at three a lane (K / 32 blocks over the blocks one codec load holds: two
// for Q4_K and Q5_K, four for Q6_K and Q3_K, one otherwise). Only the speed depends on it
// (moe_r8.cuh).
int r8_lanes(const Weight& w) {
    int per_load = 1;
    switch (w.qtype) {
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
        per_load = 2;
        break;
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
        per_load = 4;
        break;
    default:
        break;
    }
    // Up to three loads a lane: fewer lanes per row keep more rows in flight per warp, and a
    // short row spread one load a lane over 32 lanes left most of the warp's loads empty (RTX
    // 5090, T = 1: Gemma 4 26B Q4_0 down K 704 (22 loads) 14.0 us on 32 lanes, 11.5 on 8;
    // Qwen3.6-35B Q4_K gate/up K 2048 (32 loads) 13.5 on 32, 11.5 on 16; rows of 58..90 loads
    // fastest on 32).
    const auto loads = static_cast<std::int64_t>(w.k) / 32 / per_load;
    return loads <= 24 ? 8 : loads <= 48 ? 16 : 32;
}

void launch_r8_gemm(generic::WeightCodec codec, bool frag, Stage stage, int schedule, int tokens,
                    const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    if (launch_r8_gemm_iq2(codec, frag, stage, schedule, tokens, args, grid, stream) ||
        launch_r8_gemm_iq3(codec, frag, stage, schedule, tokens, args, grid, stream) ||
        launch_r8_gemm_kquant(codec, frag, stage, schedule, tokens, args, grid, stream) ||
        launch_r8_gemm_ggml4(codec, frag, stage, schedule, tokens, args, grid, stream) ||
        launch_r8_gemm_native(codec, frag, stage, schedule, tokens, args, grid, stream)) {
        return;
    }
    throw std::invalid_argument("moe: no grouped R8 kernel for this expert format");
}

namespace {

// The two route choices tests compare against their alternatives (set_mx8, set_fused_combine).
std::atomic<bool> g_mx8{true};
std::atomic<bool> g_fused_combine{true};

} // namespace

// Programmatic dependent launches through a decode layer's MoE chain, wherever the pass allows
// them (MoeConfig::pdl). A layer with a shared expert gained first (RTX 5090, C=1: Nemotron
// 3.5 +1.7 %); one without lost before the one-launch Gemma 4 attention (gpt-oss-20b -0.4 %,
// Gemma 4 26B -1.0 %) and gains since (A B B A: Gemma 4 26B C=1 +0.9 %, C=4 +1.2 %;
// gpt-oss-20b +0.9 %, +0.6 %). (Nemotron's
// warm-up failure with it was the decode graph's egress copy queued ahead of the token mask
// release on the copy engine, which the egress copy by kernel removed: ops::copy_to_pinned_host.)

void launch_gemm(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                 const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    if (frag) {
        switch (codec) {
        case generic::WeightCodec::GgmlQ40:
            return launch_gemm_ggml_q40_frag(stage, schedule, args, grid, stream);
        case generic::WeightCodec::GgmlQ4K:
            return launch_gemm_ggml_q4k_frag(stage, schedule, args, grid, stream);
        case generic::WeightCodec::GgmlIq3Xxs:
            return launch_gemm_ggml_iq3xxs_frag(stage, schedule, args, grid, stream);
        case generic::WeightCodec::GgmlIq2S:
            return launch_gemm_ggml_iq2s_frag(stage, schedule, args, grid, stream);
        default:
            throw std::invalid_argument("moe: no ggml_frag_v1 grouped kernel for this format");
        }
    }
    switch (codec) {
    case generic::WeightCodec::Bf16:
        return launch_gemm_bf16(stage, schedule, args, grid, stream);
    case generic::WeightCodec::Fp8:
        return launch_gemm_fp8(stage, schedule, args, grid, stream);
    case generic::WeightCodec::Nvfp4:
        return launch_gemm_nvfp4(stage, schedule, args, grid, stream);
    case generic::WeightCodec::Mxfp4:
        return launch_gemm_mxfp4(stage, schedule, args, grid, stream);
    case generic::WeightCodec::Q4:
        return launch_gemm_q4(stage, schedule, args, grid, stream);
    case generic::WeightCodec::Q5:
        return launch_gemm_q5(stage, schedule, args, grid, stream);
    case generic::WeightCodec::Q6:
        return launch_gemm_q6(stage, schedule, args, grid, stream);
    case generic::WeightCodec::Q8:
        return launch_gemm_q8(stage, schedule, args, grid, stream);
    case generic::WeightCodec::Fp8RowF32:
        return launch_gemm_fp8_row_f32(stage, schedule, args, grid, stream);
    case generic::WeightCodec::Fp8B128:
        return launch_gemm_fp8_b128(stage, schedule, args, grid, stream);
    case generic::WeightCodec::U4Zp:
        return launch_gemm_u4zp(stage, schedule, args, grid, stream);
    case generic::WeightCodec::U4ZpG32:
        return launch_gemm_u4zp_g32(stage, schedule, args, grid, stream);
    case generic::WeightCodec::U4ZpG64:
        return launch_gemm_u4zp_g64(stage, schedule, args, grid, stream);
    case generic::WeightCodec::Q3G128:
    case generic::WeightCodec::Q3G64:
        throw std::invalid_argument("moe: no grouped kernel for 3-bit groupwise experts");
    case generic::WeightCodec::GgmlQ4K:
        return launch_gemm_ggml_q4k(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlQ5K:
        return launch_gemm_ggml_q5k(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlQ6K:
        return launch_gemm_ggml_q6k(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlQ3K:
        return launch_gemm_ggml_q3k(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlQ2K:
        return launch_gemm_ggml_q2k(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlIq4Xs:
        return launch_gemm_ggml_iq4xs(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlIq3Xxs:
        return launch_gemm_ggml_iq3xxs(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlIq3S:
        return launch_gemm_ggml_iq3s(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlQ40:
        return launch_gemm_ggml_q40(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlQ41:
        return launch_gemm_ggml_q41(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlQ50:
        return launch_gemm_ggml_q50(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlQ51:
        return launch_gemm_ggml_q51(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlIq4Nl:
        return launch_gemm_ggml_iq4nl(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlIq2Xxs:
        return launch_gemm_ggml_iq2xxs(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlIq2Xs:
        return launch_gemm_ggml_iq2xs(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlIq2S:
        return launch_gemm_ggml_iq2s(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlIq1S:
        return launch_gemm_ggml_iq1s(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlIq1M:
        return launch_gemm_ggml_iq1m(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlTq10:
        return launch_gemm_ggml_tq10(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlTq20:
        return launch_gemm_ggml_tq20(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlQ10:
        return launch_gemm_ggml_q10(stage, schedule, args, grid, stream);
    case generic::WeightCodec::GgmlQ20:
        return launch_gemm_ggml_q20(stage, schedule, args, grid, stream);
    }
    throw std::invalid_argument("moe: unknown weight codec");
}

namespace {

[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument("moe: " + what); }

bool present(const Weight& w) { return w.qdata != nullptr; }

// The layer's geometry, read off (and checked against) its weights.
struct Geometry {
    // input: the gate/up bank's K, hidden or a zero-padded width above it (moe.h).
    std::int32_t experts = 0, hidden = 0, input = 0, inter = 0, shared_inter = 0;
    bool shared = false, shared_gate = false;
};

bool is_frag(const Weight& w) { return w.layout == QuantLayout::GgmlFrag; }

// A ggml_frag_v1 bank as the generic path's checks see it: its published layout (the planes,
// K and alignment are the same; only the order of each row's code bytes differs).
Weight published_view(const Weight& w) {
    Weight out = w;
    if (is_frag(w)) { out.layout = ggml_published_layout(w.qtype); }
    return out;
}

void require_weight(const Weight& w, const char* name, bool expert_bank) {
    // The grouped kernels read the input in its natural column order.
    if (w.k_order != nullptr) {
        fail(std::string(name) + ": a weight stored in another column order (act-order) has no "
                                 "grouped MoE route");
    }
    if (is_frag(w) && (!expert_bank || !ggml_frag_format(w.qtype))) {
        fail(std::string(name) + ": ggml_frag_v1 is read only for the routed experts' gate_up and "
                                 "down banks of ggml_q4_0, ggml_q4_k, ggml_iq3_xxs, ggml_iq2_s");
    }
    const Weight published = published_view(w);
    if (!generic::supports(published) ||
        !generic::route_admits(generic::Route{generic::Family::RegisterMma, 0}, published,
                               LinearPolicy::A16Only)) {
        fail(std::string(name) + ": format, layout, K or alignment the grouped kernel cannot read");
    }
}

void require_vector(const Tensor& t, std::int64_t n, const char* name, bool fp32_only) {
    if (t.data == nullptr) { return; }
    const bool type_ok = t.dtype == DType::FP32 || (!fp32_only && t.dtype == DType::BF16);
    if (!type_ok || !t.is_contiguous() || t.numel() != n) {
        fail(std::string(name) + " must be a contiguous " + (fp32_only ? "FP32" : "FP32/BF16") +
             " [" + std::to_string(n) + "]");
    }
}

Geometry validate(const MoeWeights& w, const MoeConfig& c) {
    Geometry g;
    if (!present(w.router) || !present(w.gate_up) || !present(w.down)) {
        fail("router, gate_up and down are required");
    }
    g.experts = w.router.n;
    g.hidden  = w.router.k;
    if (g.experts < 1 || g.experts > kMaxExperts) {
        fail("expert count " + std::to_string(g.experts) + " outside [1, 1024]");
    }
    if (c.top_k < 1 || c.top_k > std::min(g.experts, kMaxTopK)) {
        fail("top_k " + std::to_string(c.top_k) + " outside [1, min(E, 32)]");
    }
    const bool gated = c.activation != MoeActivation::Relu2;
    const int sets   = gated ? 2 : 1;
    if (w.gate_up.n % (sets * g.experts) != 0 || w.gate_up.k < g.hidden) {
        fail(gated ? "gate_up must be [E * 2F, K >= H]"
                   : "a non-gated gate_up must be [E * F, K >= H]");
    }
    g.input = w.gate_up.k;
    g.inter = w.gate_up.n / (sets * g.experts);
    if (w.down.n != g.experts * g.hidden || w.down.k != g.inter) { fail("down must be [E * H, F]"); }
    if (g.hidden % 8 != 0 || g.input % 8 != 0 || g.inter % 8 != 0) {
        fail("H, the gate/up K and F must be multiples of 8");
    }
    require_weight(w.router, "router", false);
    require_weight(w.gate_up, "gate_up", true);
    require_weight(w.down, "down", true);
    require_vector(w.router_bias, g.experts, "router_bias", false);
    require_vector(w.selection_bias, g.experts, "selection_bias", true);
    require_vector(w.expert_scale, g.experts, "expert_scale", true);
    require_vector(w.gate_up_bias, static_cast<std::int64_t>(w.gate_up.n), "gate_up_bias", false);
    require_vector(w.down_bias, static_cast<std::int64_t>(w.down.n), "down_bias", false);
    g.shared = present(w.shared_gate_up);
    if (g.shared != present(w.shared_down)) { fail("shared_gate_up and shared_down go together"); }
    if (g.shared) {
        if (w.shared_gate_up.k != g.hidden || w.shared_gate_up.n % sets != 0 ||
            w.shared_down.n != g.hidden || w.shared_down.k != w.shared_gate_up.n / sets) {
            fail(gated ? "shared expert must be gate_up [2Fs, H] and down [H, Fs]"
                       : "a non-gated shared expert must be up [Fs, H] and down [H, Fs]");
        }
        if (c.activation != MoeActivation::SwiGlu && c.activation != MoeActivation::Relu2) {
            fail("a shared expert needs SwiGlu or Relu2");
        }
        if (!generic::supports(w.shared_gate_up) || !generic::supports(w.shared_down) ||
            w.shared_gate_up.k_order != nullptr || w.shared_down.k_order != nullptr) {
            fail("shared expert format the generic path cannot read");
        }
        g.shared_inter = w.shared_gate_up.n / sets;
        g.shared_gate  = present(w.shared_gate);
        if (g.shared_gate &&
            (w.shared_gate.n != 1 || w.shared_gate.k != g.hidden || !generic::supports(w.shared_gate) ||
             w.shared_gate.k_order != nullptr)) {
            fail("shared_gate must be [1, H]");
        }
    } else if (present(w.shared_gate)) {
        fail("shared_gate without a shared expert");
    }
    if (c.activation == MoeActivation::ClampedSwiGlu && !(c.limit > 0.0F)) {
        fail("ClampedSwiGlu needs a positive limit");
    }
    return g;
}

// Route choice. Decode widths (T <= 16, or any T outside the prefill profile below 64) stream
// weights with K split across warps; the prefill profile's calls of 17 tokens or more take one
// WarpsK == 1 schedule, so a token's bytes do not depend on the call.

int streaming_schedule(std::int32_t k) {
    if (k >= 1024) { return 0; } // 1 x 8 warps, 16 rows
    if (k >= 512) { return 1; }  // 1 x 4
    return 3;                    // 4 x 1
}

constexpr int kPrefillSchedule = 5; // 4 x 1 warps, 64 rows x 64 columns
constexpr int kWideSplit8      = 7; // 1 x 8 warps, 16 rows x 32 columns
constexpr int kWideSplit4      = 8; // 1 x 4 warps, 16 rows x 32 columns

// The invariant route profile (prefill_routes.h): the streaming K split of every stage at every
// T, eight columns per CTA through T = 8 and 32 above (the order depends on WarpsK alone).
MoePlan invariant_plan(const Geometry& g, std::int32_t tokens) {
    const auto wide = [&](int schedule) {
        if (tokens <= 8) { return schedule; }
        return schedule == 0 ? kWideSplit8 : schedule == 1 ? kWideSplit4 : kPrefillSchedule;
    };
    MoePlan p;
    p.router_schedule  = wide(streaming_schedule(g.hidden));
    p.gate_up_schedule = wide(streaming_schedule(g.input));
    p.down_schedule    = wide(streaming_schedule(g.inter));
    p.job_columns      = 8;
    for (const int s : {p.gate_up_schedule, p.down_schedule}) {
        p.job_columns = std::max(p.job_columns, schedule_info(s).tokens_per_cta());
    }
    return p;
}

MoePlan plan_for(const Geometry& g, std::int32_t tokens, bool prefill) {
    MoePlan p;
    const bool wide = (prefill && tokens >= 17) || tokens >= 64;
    if (wide) {
        p.router_schedule = p.gate_up_schedule = p.down_schedule = kPrefillSchedule;
        p.job_columns = kJobColumns;
    } else {
        p.router_schedule  = streaming_schedule(g.hidden);
        p.gate_up_schedule = streaming_schedule(g.input);
        p.down_schedule    = streaming_schedule(g.inter);
        p.job_columns      = 8; // every streaming schedule covers eight columns
    }
    return p;
}

struct Workspace {
    Tensor logits, ids, weights, packed_token, slot_of, job_expert, job_begin, job_count, job_total,
        mid, y, shared_mid, shared_out, shared_gate, padded_x;
    // MoePlan::w4a8: the quantized x ([T] rows) and intermediate ([T * top_k] slot rows).
    Tensor x_codes, x_scales, x_sums, mid_codes, mid_scales, mid_sums;
};

// The router's CTAs quantize x for an R8 gate/up (GemmArgs::xq) where x is the bank's input as
// it is (no zero-padded copy).
bool router_quantizes_x(const Geometry& g, const MoePlan& p, std::int32_t tokens) {
    return p.gate_up_r8 >= 0 && !p.w4a8 && g.input == g.hidden && tokens <= kR8Tokens;
}

template <class Arena>
Workspace allocate(Arena& arena, const Geometry& g, const MoeConfig& c, std::int32_t tokens,
                   const MoePlan& p) {
    Workspace w;
    const std::int32_t a    = tokens * c.top_k;
    const std::int32_t jobs = max_jobs(a, g.experts, p.job_columns);
    w.logits       = arena.alloc(DType::FP32, {g.experts, tokens});
    w.ids          = arena.alloc(DType::I32, {a});
    w.weights      = arena.alloc(DType::FP32, {a});
    w.packed_token = arena.alloc(DType::I32, {a});
    w.slot_of      = arena.alloc(DType::I32, {a});
    w.job_expert   = arena.alloc(DType::I32, {jobs});
    w.job_begin    = arena.alloc(DType::I32, {jobs});
    w.job_count    = arena.alloc(DType::I32, {jobs});
    w.job_total    = arena.alloc(DType::I32, {1});
    w.mid          = arena.alloc(DType::BF16, {g.inter, a});
    w.y            = arena.alloc(DType::BF16, {g.hidden, a});
    if (g.input != g.hidden) { w.padded_x = arena.alloc(DType::BF16, {g.input, tokens}); }
    if (p.w4a8 || router_quantizes_x(g, p, tokens)) {
        // x at the zero-padded bank's K (g.input): the quantizer writes the tail as zeros (an R8
        // gate/up's x, quantized by the router, takes the same rows).
        const std::int32_t lh = moe_w4a8_ld(g.input);
        w.x_codes    = arena.alloc(DType::I8, {lh, tokens});
        w.x_scales   = arena.alloc(DType::FP32, {lh / 32, tokens});
        w.x_sums     = arena.alloc(DType::FP32, {lh / 32, tokens});
    }
    if (p.w4a8) {
        const std::int32_t li = moe_w4a8_ld(g.inter);
        w.mid_codes  = arena.alloc(DType::I8, {li, a});
        w.mid_scales = arena.alloc(DType::FP32, {li / 32, a});
        w.mid_sums   = arena.alloc(DType::FP32, {li / 32, a});
    }
    if (g.shared) {
        w.shared_mid = arena.alloc(DType::BF16, {g.shared_inter, tokens});
        w.shared_out = arena.alloc(DType::BF16, {g.hidden, tokens});
        if (g.shared_gate) { w.shared_gate = arena.alloc(DType::BF16, {1, tokens}); }
    }
    return w;
}

generic::GemmArgs weight_args(const Weight& w) {
    // The generic path's own argument builder, over a one-token dummy call: only the weight
    // planes and K are read by the grouped kernel.
    generic::GemmArgs a;
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

unsigned blocks(std::int64_t n, std::int64_t per) { return static_cast<unsigned>(div_up(n, per)); }

// Whether the layer's expert banks may take the grouped W4A8 (MoePlan::w4a8), whatever T: an A8
// expert policy and both banks in a format the int8 tile decodes (a ggml_frag_v1 bank: each format
// with a map, moe_w4a8_frag.cu).
// A fragment-ordered bank takes the grouped W4A8 only for the formats ggml_frag_w4a8 names
// (core/ggml_frag.h: Q4_0, Q4_K).
bool w4a8_frag_admits(QType qtype) { return ggml_frag_w4a8(qtype); }

// NVFP4 banks take it under a policy that allows A8 and, where the build has the W4A4 kernels
// (sm_120a), not A4 (generic::nvfp4_prefill_a8_policy): an A4 expert policy (either bank recorded
// for A4, moe_policy) keeps both banks on the A16 register MMA there as before, and a W4A16
// checkpoint under --activations int8 prefills its experts on int8 activations as on the builds
// without those kernels.
bool w4a8_banks_admit(const MoeWeights& w) {
    const auto bank = [&w](const Weight& b) {
        return generic::w4a8_supports(b.qtype) && b.k % 32 == 0 &&
               (b.qtype != QType::NVFP4 ||
                generic::nvfp4_prefill_a8_policy(b.qtype, w.expert_policy)) &&
               (!is_frag(b) || (ggml_frag_format(b.qtype) && w4a8_frag_admits(b.qtype)));
    };
    return allows_a8(w.expert_policy) && bank(w.gate_up) && bank(w.down);
}

// Whether the heuristic prefill plan takes the grouped W4A8 where the banks admit it: on by
// default (RTX 5090, one MoE layer at T = 64..8192 against the register MMA:
// Qwen3.6-35B-A3B Q4_K/Q6_K 0.64-0.69, IQ2_S/IQ3_XXS 0.89-0.91, Gemma 4 26B Q4_0 0.80-0.86,
// Nemotron 3.5 Q4_0 relu2 0.70-0.83).

// MXFP4 expert banks (gate/up and down) take the block-scaled MMA with MXFP8 activations instead of
// the int8 W4A8 on sm_120a builds: gpt-oss prefill experts 1.25-1.32x at T = 1K-8K on the RTX 5090,
// serve prefill +16 %, the same perplexity on llama.cpp's tokens.
bool mx8_enabled() {
#ifdef NINFERNO_WITHOUT_NVFP4
    return false;
#else
    return g_mx8.load(std::memory_order_relaxed);
#endif
}

bool mx8_banks(const MoeWeights& w) {
    return mx8_enabled() && !is_frag(w.gate_up) && !is_frag(w.down) &&
           generic::codec_of(w.gate_up) == generic::WeightCodec::Mxfp4 &&
           generic::codec_of(w.down) == generic::WeightCodec::Mxfp4;
}

MoeW4a8Buffers w4a8_buffers(const Tensor& codes, const Tensor& scales, const Tensor& sums) {
    return {static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data),
            static_cast<float*>(sums.data), static_cast<int>(codes.ne[0])};
}

void launch_w4a8(generic::WeightCodec codec, bool frag, Stage stage, const GemmArgs& args,
                 const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream) {
    if (frag) {
        if (launch_moe_w4a8_frag(codec, stage, args, act, grid, stream)) { return; }
        fail("no grouped W4A8 tile for this ggml_frag_v1 format");
    }
    if (launch_moe_w4a8_ggml32(codec, stage, args, act, grid, stream) ||
        launch_moe_w4a8_superblock(codec, stage, args, act, grid, stream) ||
        launch_moe_w4a8_codebook(codec, stage, args, act, grid, stream) ||
        launch_moe_w4a8_native(codec, stage, args, act, grid, stream)) {
        return;
    }
    fail("no grouped W4A8 tile for this expert format");
}

// The policy one shared-expert projection runs under. AllowA4 (a projection of the pair recorded
// for A4, moe_policy) runs as the AllowA8 it collapsed to before: on the generic path no W4A4, and
// where the build has the W4A4 kernels no W4A8 for NVFP4 either, so an NVFP4 projection runs A16
// there (the routes AllowA8 gave it; the dense tuner keys it a16) and every other one AllowA8.
LinearPolicy shared_projection_policy(const MoeWeights& w, const Weight& m) {
    if (!allows_a4(w.shared_policy)) { return w.shared_policy; }
    return m.qtype == QType::NVFP4 && generic::w4a4_schedule_count() > 0 ? LinearPolicy::A16Only
                                                                         : LinearPolicy::AllowA8;
}

// The activation planes an A8 shared expert's projections take on the generic path (where they
// run, whatever the Linear tables own: a Q8 shape with its own kernels has no A8 there).
std::size_t shared_planes_bytes(const MoeWeights& w, std::int32_t min_tokens,
                                std::int32_t max_tokens) {
    const auto bytes = [&](const Weight& m) {
        return generic::workspace_capacity_bytes(
            {m.qtype, shared_projection_policy(w, m), m.n, m.k, generic::Epilogue::Store}, &m,
            min_tokens, max_tokens);
    };
    return std::max(bytes(w.shared_gate_up), bytes(w.shared_down));
}

} // namespace
} // namespace detail::moe

std::size_t moe_workspace_bytes(const MoeWeights& weights, const MoeConfig& config,
                                std::int32_t min_tokens, std::int32_t max_tokens) {
    using namespace detail::moe;
    const Geometry g = validate(weights, config);
    if (min_tokens < 1 || max_tokens < min_tokens) {
        throw std::invalid_argument("moe workspace: invalid token interval");
    }
    // Every buffer grows with T; the job list also grows as the job width narrows, and the route
    // table may give any band or the prefill plan any width. The largest T at the narrowest width
    // any plan can have (one CTA's columns) bounds every plan the Op may select.
    MoePlan narrowest;
    narrowest.job_columns = kJobColumns;
    for (int s = 0; s < schedule_count(); ++s) {
        narrowest.job_columns = std::min(narrowest.job_columns, schedule_info(s).tokens_per_cta());
    }
    narrowest.w4a8 = w4a8_banks_admit(weights);
    const auto bound = [&](std::int32_t tokens, const MoePlan& plan) {
        WorkspaceLayoutBuilder layout;
        (void)allocate(layout, g, config, tokens, plan);
        // An A8 shared expert's projections take their activation planes after the Op's buffers.
        if (g.shared && weights.shared_policy != LinearPolicy::A16Only) {
            (void)layout.alloc_bytes(shared_planes_bytes(weights, min_tokens, max_tokens), 256);
        }
        return layout.peak_bytes(256);
    };
    std::size_t bytes = bound(max_tokens, narrowest);
    // An R8 gate/up of at most kR8Tokens tokens reads x as the router's CTAs quantized it
    // (router_quantizes_x): planes the bound above holds only when its banks take W4A8. The
    // widest T the bank admits R8 at (a wide K admits it at T = 1 only) sizes them.
    if (g.input == g.hidden) {
        for (std::int32_t t = std::min(max_tokens, kR8Tokens); t >= min_tokens; --t) {
            if (!r8_admits(weights, weights.gate_up, t)) continue;
            MoePlan xq    = narrowest;
            xq.gate_up_r8 = 0;
            bytes         = std::max(bytes, bound(t, xq));
            break;
        }
    }
    return bytes;
}

namespace detail::moe {

bool plan_compiled(const MoeWeights& w, const MoePlan& plan) {
    const auto compiled = [](const Weight& weight, int schedule) {
        const auto s = schedule_info(schedule);
        return generic::rmma_tile_compiled(generic::codec_of(weight), s.m_tiles, s.n_tiles,
                                           weight.k);
    };
    const auto stage = [&](const Weight& weight, int schedule, int r8) {
        if (r8 >= 0) { return r8 < r8_schedule_count() && r8_expert_supports(weight); }
        return compiled(weight, schedule);
    };
    return compiled(w.router, plan.router_schedule) &&
           stage(w.gate_up, plan.gate_up_schedule, plan.gate_up_r8) &&
           stage(w.down, plan.down_schedule, plan.down_r8);
}

// An R8 stage sums a slot's dot products in one order whatever T is, but not in the register
// MMA's: a plan with one is not the prefill profile's.
bool plan_is_token_invariant(const MoePlan& plan) {
    return plan.gate_up_r8 < 0 && plan.down_r8 < 0 &&
           schedule_info(plan.router_schedule).warps_k == 1 &&
           schedule_info(plan.gate_up_schedule).warps_k == 1 &&
           schedule_info(plan.down_schedule).warps_k == 1;
}

bool r8_admits(const MoeWeights& w, const Weight& bank, std::int32_t tokens) {
    if (!allows_a8(w.expert_policy) || tokens < 1 || tokens > kR8Tokens ||
        !r8_expert_supports(bank)) {
        return false;
    }
    // The kernel's quantized activation for its widest job (1, 2 or 4 slots: a job has at most T;
    // 36 bytes per 32 values and slot, 40 for a codec with a min, which keeps each block's sum)
    // within the default 48 KiB: K <= 43690 at T = 1, 21845 at T = 2, 10922 at T = 3 or 4 (with a
    // min 39321, 19660, 9830).
    std::int64_t slots = 1;
    while (slots < tokens) { slots *= 2; }
    const auto q    = bank.qtype;
    const bool sums = q == QType::GGML_Q4_1 || q == QType::GGML_Q5_1 || q == QType::GGML_Q4_K ||
                      q == QType::GGML_Q5_K;
    return slots * (static_cast<std::int64_t>(bank.k) / 32) * (sums ? 40 : 36) <=
           kStaticSharedBytesPerBlock;
}

namespace {

// The plan with R8 schedules gate_up_r8 / down_r8 (-1: the stage's register MMA). A call of at most
// kR8Tokens tokens has no job of more slots whatever the job width (a token selects an expert
// once), so an R8 stage adds no job-width constraint.
MoePlan with_r8(MoePlan p, int gate_up_r8, int down_r8) {
    p.gate_up_r8 = gate_up_r8;
    p.down_r8    = down_r8;
    return p;
}

// Whether the plan's R8 stages may run at T on this layer (moe_with_plan's check); x is the Op's
// activation (the gate/up stage reads it in 16-byte words; the down stage reads the workspace).
bool r8_stages_admitted(const MoeWeights& w, const MoePlan& p, std::int32_t tokens) {
    return (p.gate_up_r8 < 0 || r8_admits(w, w.gate_up, tokens)) &&
           (p.down_r8 < 0 || r8_admits(w, w.down, tokens));
}

std::string schedule_name(int index) {
    const auto s = schedule_info(index);
    return "rmma.m" + std::to_string(s.m_tiles) + ".n" + std::to_string(s.n_tiles) + ".w" +
           std::to_string(s.warps_m) + "x" + std::to_string(s.warps_k);
}

std::string r8_schedule_name(int index) {
    const auto s = r8_schedule_info(index);
    return "r8.w" + std::to_string(s.warps) + "r" + std::to_string(s.rows);
}

// A grouped stage's route: a register-MMA schedule, or (r8) an R8 schedule.
struct StageRoute {
    int schedule = 0;
    bool r8      = false;
};

std::optional<StageRoute> parse_stage(std::string_view name) {
    for (int i = 0; i < schedule_count(); ++i) {
        if (schedule_name(i) == name) { return StageRoute{i, false}; }
    }
    for (int i = 0; i < r8_schedule_count(); ++i) {
        if (r8_schedule_name(i) == name) { return StageRoute{i, true}; }
    }
    return std::nullopt;
}

// The default A8 route per GPU class, measured where it is listed (bench/ops/moe_bench.cu --a8 1
// --plans sweep [--frag 1] on the Qwen3.6-35B-A3B UD-IQ3_XXS experts: IQ2_S gate/up, IQ3_XXS
// down, H 2048, F 512): the R8 schedule of a codebook gate/up bank at each T = 1..4, published and
// in ggml_frag_v1 (whose 2-byte gathers favour one row per warp), and the schedule an IQ3_XXS
// down of width 512..1023 then takes instead of the streaming 1 x 4 warps, the same at every T of
// an A8 call so that a token's bytes do not depend on T (the A16 route keeps the streaming order).
// Down stays on the register MMA everywhere: its K (the expert width) is short, so an R8 warp's
// lanes share a few 32-value blocks. A board of an instruction set without an entry takes w4r1 on
// gate/up at every T
// and keeps the down (kFallbackA8Route). Gate/up us, R8
// against the register MMA:
//   ad102 published, T = 1 / 2 / 4: w4r1 17.4 / 24.8 / 44.0 against 19.5 / 33.8 / 54.3; down
//          2 x 2 12.3 / 20.5 / 31.7 against 1 x 4 13.3 / 19.5 / 30.8. Frag: the published route.
//   gb202 published, T = 1 / 2 / 4: w4r1 14.3, w4r2 22.5 / 34.8, against 20.4 / 30.7 / 41.0;
//          down, T = 1..4: 2 x 2 no faster (12.3 / 16.4 / 20.5 / 24.6 both; frag 2 us ahead
//          at T = 3 and 4 only, the call no faster), 4 x 1 2 us behind at T = 1; it keeps the
//          streaming 1 x 4. Frag, T = 1..4: w4r1 14.3, w4r1 22.5, w8r1 30.7 / 36.9 against
//          18.4 / 28.7 / 36.9 / 41.0 (w4r2 16.4 / 26.6 / 36.9 / 43.0).
//   gb203 published, T = 1 / 2 / 4: w8r1 18.1, w4r2 30.4 / 53.2, against 30.4 / 44.7 / 67.6;
//          down, T = 1..4: 4 x 1 16.4 / 22.5 / 26.6 / 34.8 against 1 x 4 16.4 / 22.5 / 32.5 /
//          38.9 (2 x 2 16.4 / 24.6 / 32.8 / 40.6); Qwen3.6 decode x1.004 at C = 1, x1.015 at
//          C = 4. Frag, T = 1..4: w4r1 20.2, w4r1 30.4, w8r1 46.8 / 57.3 against 28.4 / 43.0
//          / 55.3 / 65.5 (w4r2 38.6 / 59.1 / 71.3 from T = 2).
// The lists name the boards (ad102: the RTX 4090, gb202: the RTX 5090, gb203: the RTX 5080); a
// device takes the nearest measured board's route within measured::kCarryDistance, else the
// fallback below (nearest_measured).
struct ClassA8Route {
    Board board;
    A8Route route;
};
// R8 schedules (NINFERNO_MOE_R8_SCHEDULES): 0 w4r1, 1 w4r2, 2 w8r1, 3 w2r1.
constexpr ClassA8Route kA8Routes[] = {
    {measured::kRtx4090, {{0, 0, 0, 0}, {0, 0, 0, 0}, 2}},
    {measured::kRtx5090, {{0, 1, 1, 1}, {0, 0, 2, 2}, -1}},
    {measured::kRtx5080, {{2, 1, 1, 1}, {0, 0, 2, 2}, 3}},
};
// Every other class. The MoE tuner's choices of eight cards (A100, RTX 2080 Ti, 3090, 4060 Ti,
// 5060 Ti, V100, and the 4090 and 5090 above) put an R8 schedule on gate/up at T = 1..4 for
// nearly every A8 layer, mostly w4r1, where the register MMA lost by x1.15..1.36 per card (median
// of the layers; V100 x3.5..4); ad102's route, w4r1 at every T and the streaming down.
constexpr A8Route kFallbackA8Route{{0, 0, 0, 0}, {0, 0, 0, 0}, -1};

// The class's default A8 route on top of plan p: R8 on gate/up where the layer admits it and, on a
// heuristic plan (not a route-table band's, whose down schedule was measured), the class's down.
MoePlan heuristic_r8(const MoeWeights& w, const Geometry& g, MoePlan p, std::int32_t tokens,
                     bool tuned = false) {
    const auto route = a8_default_route();
    if (!route || !r8_admits(w, w.gate_up, tokens)) { return p; }
    const int gate_up = route->gate_up_at(tokens, is_frag(w.gate_up));
    if (gate_up < 0) { return p; }
    if (!tuned && route->down >= 0 && w.down.qtype == QType::GGML_IQ3_XXS &&
        streaming_schedule(g.inter) == 1 && p.down_schedule == 1) {
        p.down_schedule = route->down;
    }
    return with_r8(p, gate_up, -1);
}

} // namespace

std::optional<A8Route> a8_default_route() {
    const ClassA8Route* entry = nearest_measured(std::span(kA8Routes), board_of(device_class()),
                                                 measured::kCarryDistance);
    return entry != nullptr ? entry->route : kFallbackA8Route;
}

namespace {

// The plan without its R8 stages: each on the streaming register MMA.
MoePlan without_r8(MoePlan p, const Geometry& g) {
    if (p.gate_up_r8 >= 0) { p.gate_up_schedule = streaming_schedule(g.input); }
    if (p.down_r8 >= 0) { p.down_schedule = streaming_schedule(g.inter); }
    p.gate_up_r8 = p.down_r8 = -1;
    p.job_columns = std::max({8, schedule_info(p.gate_up_schedule).tokens_per_cta(),
                              schedule_info(p.down_schedule).tokens_per_cta()});
    return p;
}

} // namespace

std::string plan_name(const MoePlan& plan) {
    const std::string down =
        plan.down_r8 >= 0 ? r8_schedule_name(plan.down_r8) : schedule_name(plan.down_schedule);
    return (plan.gate_up_r8 >= 0 ? r8_schedule_name(plan.gate_up_r8)
                                 : schedule_name(plan.gate_up_schedule)) +
           "|" + down + (plan.a16 ? "|a16" : "") + (plan.w4a8 ? "|w4a8" : "");
}

std::optional<MoePlan> parse_plan(std::string_view name, std::int32_t hidden) {
    // "|a16": a band measured on A16 (register-MMA stages only).
    constexpr std::string_view kA16 = "|a16";
    const bool a16 = name.size() > kA16.size() && name.ends_with(kA16);
    if (a16) { name.remove_suffix(kA16.size()); }
    const auto bar = name.find('|');
    if (bar == std::string_view::npos) { return std::nullopt; }
    const auto gu = parse_stage(name.substr(0, bar));
    const auto dn = parse_stage(name.substr(bar + 1));
    if (!gu || !dn || (a16 && (gu->r8 || dn->r8))) { return std::nullopt; }
    MoePlan p;
    // An R8 stage keeps the streaming schedule's slot (never launched) and adds no job width.
    p.gate_up_schedule = gu->r8 ? 0 : gu->schedule;
    p.down_schedule    = dn->r8 ? 0 : dn->schedule;
    p.job_columns      = 8;
    for (const auto& s : {*gu, *dn}) {
        if (!s.r8) {
            p.job_columns = std::max(p.job_columns, schedule_info(s.schedule).tokens_per_cta());
        }
    }
    p = with_r8(p, gu->r8 ? gu->schedule : -1, dn->r8 ? dn->schedule : -1);
    // The router follows: token-invariant with a token-invariant plan, streaming otherwise.
    const bool invariant = !gu->r8 && !dn->r8 && schedule_info(gu->schedule).warps_k == 1 &&
                           schedule_info(dn->schedule).warps_k == 1;
    p.router_schedule = invariant ? kPrefillSchedule : streaming_schedule(hidden);
    p.a16             = a16;
    return p;
}

MoeRouteKey route_key(const MoeWeights& weights, const MoeConfig& config) {
    const Geometry g = validate(weights, config);
    return {weights.gate_up.qtype, weights.down.qtype, g.experts, config.top_k, g.hidden, g.inter};
}

bool table_covers(const MoeWeights& weights, const MoeConfig& config, std::int32_t tokens) {
    return table_plan(route_key(weights, config), tokens, false).has_value();
}

MoePlan select_plan(const MoeWeights& weights, const MoeConfig& config, std::int32_t tokens,
                    bool prefill) {
    // A route-table band first (measured and published), then the load-time tuner's plan, then
    // the heuristics. The tuner covers decode widths only, under the default route profile.
    if (!invariant_routes_active() && tokens <= kTuneMaxTokens) {
        if (auto tuned = tuned_plan(weights, config, tokens);
            tuned && !table_covers(weights, config, tokens) &&
            r8_stages_admitted(weights, *tuned, tokens)) {
            return *tuned;
        }
    }
    return untuned_plan(weights, config, tokens, prefill);
}

MoePlan untuned_plan(const MoeWeights& weights, const MoeConfig& config, std::int32_t tokens,
                     bool prefill) {
    const Geometry g = validate(weights, config);
    if (invariant_routes_active()) { return invariant_plan(g, tokens); }
    // A wide call outside the prefill profile takes the grouped W4A8 on the plan's own register-MMA
    // schedules where the banks admit it, as the prefill profile does from T=17: the int8 tier is
    // 2-4x BF16 on every card (ad102 op bench, Gemma 4 26B Q4_0 experts: T=64 947 -> 771 us,
    // T=8192 7776 -> 6054 us), and Gemma 4 prefills in the speed profile. Not NVFP4 banks where the
    // build has the W4A4 kernels: there the grouped W4A8 is the int8 prefill of a W4A16 checkpoint
    // (prefill profile only, as owned_prefill_w4a8 for the dense projections), and a wide verify
    // keeps its A16 plan.
    const bool nvfp4_native =
        generic::w4a4_schedule_count() > 0 &&
        (weights.gate_up.qtype == QType::NVFP4 || weights.down.qtype == QType::NVFP4);
    const auto speed_w4a8 = [&](MoePlan p) {
        p.w4a8 = !prefill && !nvfp4_native && tokens >= kMoeW4a8SpeedMinTokens && !p.a16 &&
                 p.gate_up_r8 < 0 && p.down_r8 < 0 && p.job_columns % kMoeW4a8BlockM == 0 &&
                 w4a8_banks_admit(weights);
        return p;
    };
    // A prefill call of 17 tokens or more that admits the grouped W4A8 takes it, as a dense W4A8
    // call does (generic_gemm.cpp select_route): the tables' prefill plans are A16 schedules,
    // measured before the grouped W4A8 was the default, and one would hide it (RTX 5080, gpt-oss
    // MXFP4 experts: the gb203 entry 2950 us at T=1024, the W4A8 2110 us).
    MoePlan p               = plan_for(g, tokens, prefill);
    const bool w4a8_prefill = prefill && tokens >= 17 &&
                              w4a8_banks_admit(weights) && p.job_columns % kMoeW4a8BlockM == 0;
    if (auto tuned =
            w4a8_prefill ? std::nullopt : table_plan(route_key(weights, config), tokens, prefill)) {
        if (!(prefill && tokens >= 17) || plan_is_token_invariant(*tuned)) {
            if (const MoePlan wide = speed_w4a8(*tuned); wide.w4a8) { return wide; }
            // A table's R8 stages run only where this layer admits them (the streaming register
            // MMA otherwise). A band of register-MMA stages takes the class's A8 gate/up on top
            // unless it is marked A16; its schedules stay as measured.
            if (!r8_stages_admitted(weights, *tuned, tokens)) { return without_r8(*tuned, g); }
            if (tuned->gate_up_r8 < 0 && tuned->down_r8 < 0 && !tuned->a16) {
                return heuristic_r8(weights, g, *tuned, tokens, true);
            }
            return *tuned;
        }
    }
    if (prefill && tokens >= 17) {
        p.w4a8 = w4a8_prefill;
        return p;
    }
    if (const MoePlan wide = speed_w4a8(p); wide.w4a8) { return wide; }
    return heuristic_r8(weights, g, p, tokens);
}

std::size_t plan_workspace_bytes(const MoeWeights& weights, const MoeConfig& config,
                                 std::int32_t tokens, const MoePlan& plan) {
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, validate(weights, config), config, tokens, plan);
    return layout.peak_bytes(256);
}

namespace {

// The shared expert's branch: this thread's second stream on the current device, which a call
// forks to after its workspace is laid out and joins back before the combine (a capture follows
// the fork into the graph as two parallel chains), and the fork's two events. The shared expert
// is a separate small GEMV pair at decode (Qwen3.6-35B-A3B GGUF: Q8 [1024,2048] + [2048,512],
// two to four CTAs of rows per SM at most), so beside the router, plan and grouped experts it
// fills SMs those leave idle instead of following them.
struct SharedBranch {
    cudaStream_t stream = nullptr;
    cudaEvent_t fork    = nullptr;
    cudaEvent_t join    = nullptr;
};

SharedBranch& shared_branch() {
    thread_local std::array<SharedBranch, 16> branches{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= static_cast<int>(branches.size())) {
        fail("the shared-expert branch covers 16 devices");
    }
    SharedBranch& b = branches[static_cast<std::size_t>(device)];
    if (b.stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&b.fork, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&b.join, cudaEventDisableTiming));
    }
    return b;
}

// The R8 down stage combines in its epilogue; set_fused_combine(false) keeps the separate combine
// kernel, which the test compares it with in one process (read at every call).
bool fused_combine_enabled() { return g_fused_combine.load(std::memory_order_relaxed); }

CombineArgs combine_args(const Workspace& ws, const Geometry& g, const MoeConfig& c,
                         MoeEpilogue epilogue, Tensor& out, std::int32_t t, bool pdl) {
    CombineArgs args;
    args.y            = ws.y.data;
    args.slot_of      = static_cast<const int*>(ws.slot_of.data);
    args.weights      = static_cast<const float*>(ws.weights.data);
    args.shared       = g.shared ? ws.shared_out.data : nullptr;
    args.shared_gate  = g.shared_gate ? ws.shared_gate.data : nullptr;
    args.out          = out.data;
    args.hidden       = g.hidden;
    args.top_k        = c.top_k;
    args.tokens       = t;
    args.add_residual = epilogue == MoeEpilogue::AddResidual;
    args.pdl          = pdl;
    return args;
}

// The launch sequence.
void run(const Tensor& x, const MoeWeights& w, const MoeConfig& c, MoeEpilogue epilogue,
         Tensor& out, WorkspaceArena& workspace, cudaStream_t stream, const MoePlan& p,
         const Tensor* router_input = nullptr) {
    namespace generic = detail::generic;
    const Geometry g = validate(w, c);
    for (const int s : {p.router_schedule, p.gate_up_schedule, p.down_schedule}) {
        if (s < 0 || s >= schedule_count()) { fail("unknown schedule in the plan"); }
    }
    if (!plan_compiled(w, p)) { fail("a schedule of the plan is not compiled for its format"); }
    if (p.job_columns < 1 || p.job_columns > kJobColumns ||
        (p.gate_up_r8 < 0 &&
         p.job_columns % schedule_info(p.gate_up_schedule).tokens_per_cta() != 0) ||
        (p.down_r8 < 0 && p.job_columns % schedule_info(p.down_schedule).tokens_per_cta() != 0)) {
        fail("the job width must be a multiple of both grouped schedules' columns, <= 64");
    }
    if (p.w4a8 && (p.job_columns % kMoeW4a8BlockM != 0 || !w4a8_banks_admit(w) ||
                   p.gate_up_r8 >= 0 || p.down_r8 >= 0)) {
        fail("a W4A8 plan needs admitted banks, no R8 stage and a job width of whole W4A8 tiles");
    }
    const std::int32_t t = x.ne[1];
    // Every plan reads the activation in 16-byte words: the router's BF16 register MMA (and the
    // gate/up stage's, R8 or register MMA, for most formats). An A16 fallback would not help an
    // activation off that alignment, so every plan refuses it before a launch.
    const auto aligned16 = [](const void* ptr) {
        return reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0;
    };
    if (!aligned16(x.data) || (router_input != nullptr && !aligned16(router_input->data))) {
        fail("x (and router_input) must be 16-byte aligned: the grouped kernels read the "
             "activation in 16-byte words");
    }
    if (!r8_stages_admitted(w, p, t)) {
        fail("an R8 stage needs an A8 expert policy, a codebook format, at most 4 tokens and an "
             "activation that fits a CTA");
    }
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != g.hidden || t < 1 ||
        x.ne[2] != 1 || x.ne[3] != 1) {
        fail("x must be contiguous BF16 [H, T]");
    }
    if (out.dtype != DType::BF16 || !out.is_contiguous() || out.ne[0] != g.hidden ||
        out.ne[1] != t || out.ne[2] != 1 || out.ne[3] != 1) {
        fail("out must be contiguous BF16 [H, T]");
    }
    if (router_input != nullptr &&
        (router_input->dtype != DType::BF16 || !router_input->is_contiguous() ||
         router_input->ne[0] != g.hidden || router_input->ne[1] != t || router_input->ne[2] != 1 ||
         router_input->ne[3] != 1)) {
        fail("router_input must be contiguous BF16 [H, T] like x");
    }
    auto scope       = workspace.scope();
    const Workspace ws = allocate(workspace, g, c, t, p);
    const std::int32_t a = t * c.top_k;
    const std::int32_t jobs = max_jobs(a, g.experts, p.job_columns);
    bool fused_combine      = false; // the R8 down stage combines (GemmArgs::combine_tickets)

    // The shared expert, through the generic projection path: x and its own buffers only, and
    // (A8) its activation planes in `arena`. On the branch it runs beside steps 1-5, which take
    // nothing from the workspace past ws, so its planes get a region of their own right after ws
    // (moe_workspace_bytes reserves it there).
    const auto run_shared = [&](cudaStream_t s, WorkspaceArena* arena) {
        if (!g.shared) return;
        Tensor mid  = ws.shared_mid;
        Tensor sout = ws.shared_out;
        const LinearPolicy gate_up_policy = shared_projection_policy(w, w.shared_gate_up);
        if (c.activation == MoeActivation::Relu2) {
            generic::linear_relu2(x, w.shared_gate_up, mid, gate_up_policy, arena, s);
        } else {
            generic::linear_swiglu(x, w.shared_gate_up, mid, gate_up_policy, arena, s);
        }
        generic::linear(mid, w.shared_down, sout, shared_projection_policy(w, w.shared_down), arena,
                        s);
        if (g.shared_gate) {
            Tensor gate = ws.shared_gate;
            generic::linear(x, w.shared_gate, gate, LinearPolicy::A16Only, arena, s);
        }
    };
    // The shared expert runs on the branch (on the stream: RTX 4090, Qwen3.6-35B-A3B -6..-10 %,
    // Nemotron -3 %), A8 with its activation planes in a region of its own. One of a weight stored
    // in another column order stays on the stream: its gathered activation takes workspace no
    // capacity bound covers here.
    SharedBranch* branch = nullptr;
    std::optional<WorkspaceArena> branch_arena;
    if (g.shared && w.shared_gate_up.k_order == nullptr && w.shared_down.k_order == nullptr &&
        (!g.shared_gate || w.shared_gate.k_order == nullptr)) {
        if (w.shared_policy != LinearPolicy::A16Only) {
            const std::size_t bytes = shared_planes_bytes(w, t, t);
            // A route that takes no planes at this T reads no workspace.
            if (bytes > 0) { branch_arena.emplace(workspace.alloc_bytes(bytes, 256)); }
        }
        branch = &shared_branch();
        CUDA_CHECK(cudaEventRecord(branch->fork, stream));
        CUDA_CHECK(cudaStreamWaitEvent(branch->stream, branch->fork, 0));
    }
    // The shared expert's launches go to the branch after the router's and the plan's. The fork
    // is recorded before the router, so the branch depends on nothing the router writes; only the
    // order the two chains reach the GPU is late. Enqueued first, the shared expert's grids
    // (Nemotron: 1000+ CTAs of R8) fill the SMs and the router's 32 CTAs, on the path that sets
    // the layer's length, wait behind them (decode census: 2.5 us alone, 7.7-10 us beside them).
    const auto run_shared_late = [&] {
        if (branch != nullptr) {
            run_shared(branch->stream, branch_arena ? &*branch_arena : &workspace);
        }
    };

    // 2. Selection and 3. plan: one launch for a call of at most kPlanWarps tokens.
    SelectArgs select;
    {
        SelectArgs& args    = select;
        args.logits         = static_cast<const float*>(ws.logits.data);
        args.selection_bias = static_cast<const float*>(w.selection_bias.data);
        args.expert_scale   = static_cast<const float*>(w.expert_scale.data);
        args.experts        = g.experts;
        args.top_k          = c.top_k;
        args.tokens         = t;
        args.sigmoid        = c.scoring == MoeScoring::Sigmoid;
        args.renormalize    = c.renormalize;
        args.routed_scale   = c.routed_scale;
        args.ids            = static_cast<int*>(ws.ids.data);
        args.weights        = static_cast<float*>(ws.weights.data);
    }
    PlanArgs plan;
    {
        plan.ids          = static_cast<const int*>(ws.ids.data);
        plan.assignments  = a;
        plan.top_k        = c.top_k;
        plan.experts      = g.experts;
        plan.job_columns  = p.job_columns;
        plan.packed_token = static_cast<int*>(ws.packed_token.data);
        plan.slot_of      = static_cast<int*>(ws.slot_of.data);
        plan.job_expert   = static_cast<int*>(ws.job_expert.data);
        plan.job_begin    = static_cast<int*>(ws.job_begin.data);
        plan.job_count    = static_cast<int*>(ws.job_count.data);
        plan.job_total    = static_cast<int*>(ws.job_total.data);
    }
    // The SIMT router: a BF16 router at decode widths outside the invariant profile (whose router
    // streams with the K split of every width) and outside a token-invariant plan (the prefill
    // schedule), x within the default shared memory.
    const bool router_simt = t <= kRouterSimtMaxTokens &&
                             generic::codec_of(w.router) == generic::WeightCodec::Bf16 &&
                             !invariant_routes_active() && p.router_schedule != kPrefillSchedule &&
                             router_simt_smem_bytes(t, g.hidden) <= kStaticSharedBytesPerBlock;
    // Programmatic dependent launches (decode chain): each kernel below waits for the previous one
    // before its first load; a copy or a quantizer between two stages keeps the plain launch.
    const bool pdl = device_class().features.pdl && c.pdl;
    // The SIMT router's last CTA selects and plans (one launch fewer) when the warp plan covers
    // the call (at most 32 assignments, on the device) and the chain has no programmatic
    // dependent launches: without PDL each launch is a gap (RTX 3090: Gemma 4 26B C=1
    // +1.3 %, C=4 +2.1 %, Qwen3.6-35B and Nemotron C=1 +1.2 / +1.4 %), with it the separate
    // launch already overlaps and the last CTA's serial work costs more (RTX 5090:
    // Qwen3.6-35B C=4 -3.3 %, Nemotron -2.3 %).
    const bool route_in_router = router_simt && t <= kRouterSimtWarps && a <= 32 && !pdl;
    // 1. Router logits.
    {
        GemmArgs args;
        args.pdl         = pdl;
        args.w           = weight_args(w.router);
        args.x           = router_input != nullptr ? router_input->data : x.data;
        args.ldx         = g.hidden;
        args.tokens      = t;
        args.rows        = g.experts;
        args.expert_rows = 0;
        args.bias        = w.router_bias.data;
        args.bias_bf16   = w.router_bias.dtype == DType::BF16;
        args.out_f32     = static_cast<float*>(ws.logits.data);
        if (router_quantizes_x(g, p, t)) {
            args.xq        = x.data;
            args.xq_ldx    = g.hidden;
            args.xq_k      = g.hidden;
            args.xq_ld     = static_cast<int>(ws.x_codes.ne[0]);
            args.xq_codes  = static_cast<std::int8_t*>(ws.x_codes.data);
            args.xq_scales = static_cast<float*>(ws.x_scales.data);
            args.xq_sums   = static_cast<float*>(ws.x_sums.data);
        }
        if (router_simt) {
            launch_router_simt(args, stream, route_in_router ? &select : nullptr, &plan);
        } else {
            const auto s = schedule_info(p.router_schedule);
            launch_gemm(generic::codec_of(w.router), false, Stage::Router, p.router_schedule, args,
                        dim3(blocks(g.experts, s.rows_per_cta()), blocks(t, s.tokens_per_cta())),
                        stream);
        }
    }
    // 3. Plan (with 2. selection in the same launch when it is short enough).
    if (route_in_router) {
        // Done by the router's last CTA.
    } else if (t <= kPlanWarps) {
        launch_select_plan(select, plan, stream);
    } else {
        launch_select(select, stream);
        launch_plan(plan, stream);
    }
    run_shared_late();
    GemmArgs grouped;
    grouped.job_expert   = static_cast<const int*>(ws.job_expert.data);
    grouped.job_begin    = static_cast<const int*>(ws.job_begin.data);
    grouped.job_count    = static_cast<const int*>(ws.job_count.data);
    grouped.job_total    = static_cast<const int*>(ws.job_total.data);
    // 4. Gate/up and the activation, per slot. A zero-padded bank reads x widened with zeros.
    const void* gate_up_x = x.data;
    if (g.input != g.hidden && !p.w4a8) {
        const std::size_t row  = sizeof(std::uint16_t) * static_cast<std::size_t>(g.input);
        const std::size_t live = sizeof(std::uint16_t) * static_cast<std::size_t>(g.hidden);
        auto* padded           = static_cast<char*>(ws.padded_x.data);
        CUDA_CHECK(cudaMemcpy2DAsync(padded, row, x.data, live, live, static_cast<std::size_t>(t),
                                     cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemset2DAsync(padded + live, row, 0, row - live, static_cast<std::size_t>(t),
                                     stream));
        gate_up_x = padded;
    }
    {
        GemmArgs args     = grouped;
        args.w            = weight_args(w.gate_up);
        args.x                 = gate_up_x;
        args.ldx               = g.input;
        args.packed_token = static_cast<const int*>(ws.packed_token.data);
        args.rows         = g.inter;
        const bool gated  = c.activation != MoeActivation::Relu2;
        args.expert_rows  = gated ? 2 * g.inter : g.inter;
        const bool interleaved = gated && c.gate_up_layout == MoeGateUpLayout::Interleaved;
        args.row_step     = interleaved ? 2 : 1;
        args.up_offset    = interleaved ? 1 : g.inter; // GateUp's second set (Up has one)
        args.bias         = w.gate_up_bias.data;
        args.bias_bf16    = w.gate_up_bias.dtype == DType::BF16;
        args.activation   = static_cast<int>(c.activation);
        args.alpha        = c.alpha;
        args.limit        = c.limit;
        args.out_bf16     = ws.mid.data;
        args.pdl               = pdl && gate_up_x == x.data && !p.w4a8;
        // A non-gated expert reads its one up row once, not as both sets of a gated one.
        const Stage first = gated ? Stage::GateUp : Stage::Up;
        args.ldo          = g.inter;
        if (p.w4a8) {
            const auto xq  = w4a8_buffers(ws.x_codes, ws.x_scales, ws.x_sums);
            const bool mx  = mx8_banks(w);
            args.sub_tiles = p.job_columns / kMoeW4a8BlockM;
            const dim3 grid(blocks(g.inter, gated ? kMoeW4a8BlockN / 2 : kMoeW4a8BlockN),
                            static_cast<unsigned>(jobs * args.sub_tiles));
            if (mx) {
                launch_moe_mx8_quantize(x.data, g.hidden, g.hidden, t, xq, stream);
                if (!launch_moe_w4a8_mx(generic::codec_of(w.gate_up), first, args, xq.view(), grid,
                                        stream)) {
                    fail("no block-scaled MoE tile for this expert format");
                }
            } else {
                launch_moe_w4a8_quantize(x.data, g.hidden, g.hidden, t, xq, stream);
                launch_w4a8(generic::codec_of(w.gate_up), is_frag(w.gate_up), first, args,
                            xq.view(), grid, stream);
            }
        } else if (p.gate_up_r8 >= 0) {
            if (router_quantizes_x(g, p, t)) {
                args.x_codes  = static_cast<const std::int8_t*>(ws.x_codes.data);
                args.x_scales = static_cast<const float*>(ws.x_scales.data);
                args.x_sums   = static_cast<const float*>(ws.x_sums.data);
                args.x_ld     = static_cast<int>(ws.x_codes.ne[0]);
            }
            args.r8_lanes  = r8_lanes(w.gate_up);
            const int rows = r8_schedule_info(p.gate_up_r8).rows_per_cta() * (32 / args.r8_lanes);
            launch_r8_gemm(generic::codec_of(w.gate_up), is_frag(w.gate_up), first, p.gate_up_r8, t,
                           args, dim3(blocks(g.inter, rows), static_cast<unsigned>(jobs)), stream);
        } else {
            const auto s   = schedule_info(p.gate_up_schedule);
            args.sub_tiles = p.job_columns / s.tokens_per_cta();
            launch_gemm(generic::codec_of(w.gate_up), is_frag(w.gate_up), first,
                        p.gate_up_schedule, args,
                        dim3(blocks(g.inter, s.rows_per_cta()),
                             static_cast<unsigned>(jobs * args.sub_tiles)),
                        stream);
        }
    }
    // 5. Down, per slot.
    {
        GemmArgs args    = grouped;
        args.w           = weight_args(w.down);
        args.x           = ws.mid.data;
        args.ldx         = g.inter;
        args.rows        = g.hidden;
        args.expert_rows = g.hidden;
        args.bias        = w.down_bias.data;
        args.bias_bf16   = w.down_bias.dtype == DType::BF16;
        args.out_bf16    = ws.y.data;
        args.ldo         = g.hidden;
        args.pdl         = pdl && !p.w4a8;
        if (p.w4a8) {
            const auto mq  = w4a8_buffers(ws.mid_codes, ws.mid_scales, ws.mid_sums);
            args.sub_tiles = p.job_columns / kMoeW4a8BlockM;
            const dim3 grid(blocks(g.hidden, kMoeW4a8BlockN),
                            static_cast<unsigned>(jobs * args.sub_tiles));
            if (mx8_banks(w)) {
                launch_moe_mx8_quantize(ws.mid.data, g.inter, g.inter, a, mq, stream);
                if (!launch_moe_w4a8_mx(generic::codec_of(w.down), Stage::Down, args, mq.view(),
                                        grid, stream)) {
                    fail("no block-scaled MoE tile for this expert format");
                }
            } else {
                launch_moe_w4a8_quantize(ws.mid.data, g.inter, g.inter, a, mq, stream);
                launch_w4a8(generic::codec_of(w.down), is_frag(w.down), Stage::Down, args,
                            mq.view(), grid, stream);
            }
        } else if (p.down_r8 >= 0) {
            args.r8_lanes  = r8_lanes(w.down);
            const int rows = r8_schedule_info(p.down_r8).rows_per_cta() * (32 / args.r8_lanes);
            if (!g.shared && branch == nullptr && fused_combine_enabled() &&
                blocks(g.hidden, rows) <= kCombineTickets) {
                args.combine_tickets = combine_tickets();
                args.combine         = combine_args(ws, g, c, epilogue, out, t, false);
                fused_combine        = true;
            }
            launch_r8_gemm(generic::codec_of(w.down), is_frag(w.down), Stage::Down, p.down_r8, t,
                           args, dim3(blocks(g.hidden, rows), static_cast<unsigned>(jobs)), stream);
        } else {
            const auto s   = schedule_info(p.down_schedule);
            args.sub_tiles = p.job_columns / s.tokens_per_cta();
            launch_gemm(generic::codec_of(w.down), is_frag(w.down), Stage::Down,
                        p.down_schedule, args,
                        dim3(blocks(g.hidden, s.rows_per_cta()),
                             static_cast<unsigned>(jobs * args.sub_tiles)),
                        stream);
        }
    }
    // The shared expert (forked at the top of the call, or here on the stream).
    if (branch != nullptr) {
        CUDA_CHECK(cudaEventRecord(branch->join, branch->stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, branch->join, 0));
    } else {
        run_shared(stream, &workspace);
    }
    // 6. Combine (unless the R8 down stage's last CTAs did it).
    if (!fused_combine) {
        // After the join the stream's previous work is the shared branch's too: a plain launch.
        launch_combine(combine_args(ws, g, c, epilogue, out, t, pdl && branch == nullptr), stream);
    }
}

} // namespace

void set_mx8(bool enabled) { g_mx8.store(enabled, std::memory_order_relaxed); }

void set_fused_combine(bool enabled) { g_fused_combine.store(enabled, std::memory_order_relaxed); }

void moe_with_plan(const Tensor& x, const MoeWeights& w, const MoeConfig& c,
                   MoeEpilogue epilogue, Tensor& out, WorkspaceArena& workspace,
                   cudaStream_t stream, const MoePlan& p, const Tensor* router_input) {
    run(x, w, c, epilogue, out, workspace, stream, p, router_input);
}

} // namespace detail::moe

void moe(const Tensor& x, const MoeWeights& w, const MoeConfig& c, MoeEpilogue epilogue,
         Tensor& out, WorkspaceArena& workspace, cudaStream_t stream,
         const Tensor* router_input) {
    const std::int32_t t = x.ne[1];
    // Without tensor mma.sync (Volta's build) the register-MMA stages run on FP32 FMAs, and R8
    // stops at kR8Tokens: a verify step of 5..16 tokens runs as calls of at most four, each on its
    // own plan, where four tokens take R8 (V100, Gemma 4 26B experts, tuned: R8 at T = 4 214 us
    // against the register MMA's 1535 at T = 5). R8 sums a token the same way at every T, so a
    // column's bytes are those of plain decode.
    if (!build_features().mma_tensor && t > detail::moe::kR8Tokens && t <= 16 &&
        !prefill_routes_active() && !invariant_routes_active() &&
        detail::moe::select_plan(w, c, detail::moe::kR8Tokens, false).gate_up_r8 >= 0) {
        for (std::int32_t begin = 0; begin < t; begin += detail::moe::kR8Tokens) {
            const std::int32_t n = std::min(detail::moe::kR8Tokens, t - begin);
            const Tensor xs      = x.slice(1, begin, n);
            Tensor outs          = out.slice(1, begin, n);
            const Tensor routes  = router_input != nullptr ? router_input->slice(1, begin, n) : xs;
            const auto plan      = detail::moe::select_plan(w, c, n, false);
            detail::moe::moe_with_plan(xs, w, c, epilogue, outs, workspace, stream, plan,
                                       router_input != nullptr ? &routes : nullptr);
        }
        return;
    }
    const auto plan = detail::moe::select_plan(w, c, t, prefill_routes_active());
    detail::moe::moe_with_plan(x, w, c, epilogue, out, workspace, stream, plan, router_input);
}

LinearPolicy moe_policy(LinearPolicy gate_up, LinearPolicy down) {
    if (!allows_a8(gate_up) || !allows_a8(down)) { return LinearPolicy::A16Only; }
    return allows_a4(gate_up) || allows_a4(down) ? LinearPolicy::AllowA4 : LinearPolicy::AllowA8;
}

LinearPolicy shared_moe_policy(LinearPolicy gate_up, LinearPolicy down) {
    return moe_policy(gate_up, down);
}

} // namespace ninferno::ops
