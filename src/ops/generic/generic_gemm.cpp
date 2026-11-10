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

namespace {

// The host tables, from the lists the CUDA files instantiate (generic_schedules.h): one source,
// so a schedule cannot be listed here and missing there, or listed in another order (#307).
template <std::size_t N>
constexpr bool indexed_in_order(const int (&index)[N]) {
    for (std::size_t i = 0; i < N; ++i) {
        if (index[i] != static_cast<int>(i)) { return false; }
    }
    return true;
}
#define NINFERNO_SCHEDULE_INDEX(i, ...) i,
static_assert(indexed_in_order({NINFERNO_GENERIC_RMMA_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
static_assert(indexed_in_order({NINFERNO_GENERIC_W4A4_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
static_assert(indexed_in_order({NINFERNO_GENERIC_W4A8_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
static_assert(indexed_in_order({NINFERNO_GENERIC_R8_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
static_assert(indexed_in_order({NINFERNO_GENERIC_TILED_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
static_assert(indexed_in_order({NINFERNO_GENERIC_STAGED_SCHEDULES(NINFERNO_SCHEDULE_INDEX)}));
#undef NINFERNO_SCHEDULE_INDEX

// warps_m, warps_k, m_tiles, n_tiles, cluster.
#define NINFERNO_RMMA_INFO(i, wm, wk, mt, nt, cl) RmmaScheduleInfo{wm, wk, mt, nt, cl},
constexpr std::array kRmmaSchedules{NINFERNO_GENERIC_RMMA_SCHEDULES(NINFERNO_RMMA_INFO)};
#undef NINFERNO_RMMA_INFO

// block_m, block_n.
#define NINFERNO_W4A4_INFO(i, bm, bn, wm, wn, st) W4a4ScheduleInfo{bm, bn},
constexpr std::array kW4a4Schedules{NINFERNO_GENERIC_W4A4_SCHEDULES(NINFERNO_W4A4_INFO)};
#undef NINFERNO_W4A4_INFO

// block_m (tokens), block_n (rows), producer warps (0: every warp loads and multiplies).
#define NINFERNO_W4A8_INFO(i, bm, bn, wm, wn, pr) W4a8ScheduleInfo{bm, bn, pr},
constexpr std::array kW4a8Schedules{NINFERNO_GENERIC_W4A8_SCHEDULES(NINFERNO_W4A8_INFO)};
#undef NINFERNO_W4A8_INFO

// Warps per CTA, rows per warp.
#define NINFERNO_R8_INFO(i, w, r) R8ScheduleInfo{w, r},
constexpr std::array kR8Schedules{NINFERNO_GENERIC_R8_SCHEDULES(NINFERNO_R8_INFO)};
#undef NINFERNO_R8_INFO

// block_m (tokens), block_n (rows).
#define NINFERNO_TILED_INFO(i, bm, bn, wm, wn, st) TiledScheduleInfo{bm, bn},
constexpr std::array kTiledSchedules{NINFERNO_GENERIC_TILED_SCHEDULES(NINFERNO_TILED_INFO)};
#undef NINFERNO_TILED_INFO

// Rows, tokens, warps_m, warps_n, stages. 6 (48 KiB) runs two CTAs per SM, for grids of fewer
// tiles than twice the SMs; 9 (64 KiB) two where an SM holds 128 KiB; 10-12 (144-160 KiB) only
// where one block holds them (GA100, Orin).
#define NINFERNO_STAGED_INFO(i, r, t, wm, wn, st) StagedScheduleInfo{r, t, wm, wn, st},
constexpr std::array kStagedSchedules{NINFERNO_GENERIC_STAGED_SCHEDULES(NINFERNO_STAGED_INFO)};
#undef NINFERNO_STAGED_INFO
static_assert(kRmmaSchedules.size() == 20 && kW4a4Schedules.size() == 4 && kW4a8Schedules.size() == 6 &&
              kR8Schedules.size() == 4 && kTiledSchedules.size() == 3 && kStagedSchedules.size() == 13);

// The NVFP4 tiled schedules (generic_nvfp4_tiled.cu dispatches by classes, no list of its own):
// classes (the register MMA's WarpsK whose order it reproduces), block_m (tokens), block_n (rows).
constexpr std::array<Nvfp4TiledScheduleInfo, 3> kNvfp4TiledSchedules{{
    {1, 64, 128},
    {4, 64, 128},
    {8, 64, 128},
}};

int run_values(WeightCodec codec) {
    switch (codec) {
    case WeightCodec::Bf16:
    case WeightCodec::Q8:
    case WeightCodec::U4ZpG32:
        return 8;
    case WeightCodec::Fp8:
    case WeightCodec::Fp8RowF32:
    case WeightCodec::Fp8B128:
    case WeightCodec::Nvfp4:
    case WeightCodec::Q4:
    case WeightCodec::Q5:
    case WeightCodec::Q6:
    case WeightCodec::U4ZpG64:
    case WeightCodec::Q3G128:
    case WeightCodec::Q3G64:
        return 16;
    case WeightCodec::Mxfp4:
    case WeightCodec::U4Zp:
    case WeightCodec::GgmlQ4K:
    case WeightCodec::GgmlQ5K:
    case WeightCodec::GgmlQ6K:
    case WeightCodec::GgmlQ3K:
    case WeightCodec::GgmlQ2K:
    case WeightCodec::GgmlIq4Xs:
    case WeightCodec::GgmlIq3Xxs:
    case WeightCodec::GgmlIq3S:
    case WeightCodec::GgmlIq2Xxs:
    case WeightCodec::GgmlIq2Xs:
    case WeightCodec::GgmlIq2S:
    case WeightCodec::GgmlIq1S:
    case WeightCodec::GgmlIq1M:
    case WeightCodec::GgmlTq10:
    case WeightCodec::GgmlTq20:
        return 32;
    // A K step is four 32-value blocks, each lane holding a slice of all four: K must be whole
    // steps for the register MMA (the scalar path takes any K of whole blocks).
    case WeightCodec::GgmlQ40:
    case WeightCodec::GgmlQ41:
    case WeightCodec::GgmlQ50:
    case WeightCodec::GgmlQ51:
    case WeightCodec::GgmlIq4Nl:
    case WeightCodec::GgmlQ10:
    case WeightCodec::GgmlQ20:
        return 128;
    }
    return 16;
}

} // namespace

// W4A4: NVFP4 with A4 allowed and a finite, positive input-scale divisor.
bool allows_w4a4(const Weight& w, LinearPolicy policy) {
    return w.qtype == QType::NVFP4 && allows_a4(policy) && std::isfinite(w.input_scale_divisor) &&
           w.input_scale_divisor > 0.0F;
}

// W4A8: a format with a W4A8 codec, a K of whole tiles (w4a8_k_admits), 16-byte code rows, A8
// allowed. Where this build has the W4A4 kernels an NVFP4 weight under a policy that allows A4 is
// not W4A8's, whatever its divisor: four-bit activations stay the route of the weights recorded
// for them (A16 where no divisor lets W4A4 run, as before), W4A8 the one of those whose policy
// allows eight bits only (a W4A16 checkpoint under --activations int8). The weight-less
// nvfp4_prefill_a8_policy and the workspace query decide the same.
// The policy part, which w4a8_refusal names Policy.
namespace {
bool w4a8_policy_allows(const Weight& w, LinearPolicy policy) {
    return allows_a8(policy) &&
           (w.qtype != QType::NVFP4 || nvfp4_prefill_a8_policy(w.qtype, policy));
}
} // namespace

bool allows_w4a8(const Weight& w, LinearPolicy policy) {
    return w4a8_policy_allows(w, policy) && w4a8_supports(w.qtype) && w4a8_k_admits(w.qtype, w.k) &&
           aligned_to(w.qdata, 16);
}

// I8: Q4 G64 with A8 allowed, K in whole 512-value runs (the weight scales of eight groups are
// one 16-byte copy per row), 16-byte code and scale planes.
// Its parts, which i8_refusal names Format, Alignment and Extent.
namespace {
bool i8_format(QType qtype) { return format_traits(qtype).qx_g64; }

bool i8_planes_aligned(const Weight& w) {
    const bool q4 = w.qtype == QType::Q4_G64_FP16, q5 = w.qtype == QType::Q5_G64_FP16;
    return aligned_to(w.qdata, 16) && aligned_to(w.scales, 16) &&
           (q4 || aligned_to(w.qhigh, q5 ? 8 : 16));
}

bool i8_extent(const Weight& w) { return (w.k % 512) == 0 && (w.padded_shape[1] % 512) == 0; }
} // namespace

bool allows_i8(const Weight& w, LinearPolicy policy) {
    return allows_a8(policy) && i8_format(w.qtype) && i8_extent(w) && i8_planes_aligned(w);
}

namespace {

// Whether the producer-warp tile can read the weight: its format, and IQ3_XXS's sign plane (read
// 8 bytes at a time) on an 8-byte boundary.
bool w4a8_ws_admits(const Weight& w) {
    return w4a8_ws_supports(w.qtype) && (w.qtype != QType::GGML_IQ3_XXS || aligned_to(w.qhigh, 8));
}
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

GemmArgs make_args(const Call& call) {
    GemmArgs a = weight_args(*call.weight);
    a.x         = call.x->data;
    a.ldx       = call.x->ne[0];
    a.tokens    = call.x->ne[1];
    a.row_begin = call.row_begin;
    a.rows      = call.rows;
    a.up_offset = call.up_offset;
    a.gelu_gate = call.gelu_gate ? 1 : 0;
    a.relu2     = call.relu2 ? 1 : 0;
    a.segments  = call.segments;
    int end     = 0;
    for (int s = 0; s < call.segments; ++s) {
        end += call.segment_rows[s];
        a.out[s]         = static_cast<std::uint16_t*>(call.out[s]->data) + call.out_row_offset[s];
        a.ld[s]          = call.out[s]->ne[0];
        a.segment_end[s] = end;
    }
    for (int s = call.segments; s < kMaxSegments; ++s) a.segment_end[s] = end;
    return a;
}

void validate_call(const Call& call) {
    if (call.x == nullptr || call.weight == nullptr) {
        throw std::invalid_argument("generic linear: missing input or weight");
    }
    const Weight& w = *call.weight;
    refuse_frag(w, "generic linear");
    const Tensor& x = *call.x;
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != w.k || x.ne[1] <= 0 ||
        x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("generic linear: x must be contiguous BF16 [K,T]");
    }
    if (call.segments < 1 || call.segments > kMaxSegments) {
        throw std::invalid_argument("generic linear: invalid output segment count");
    }
    int total = 0;
    for (int s = 0; s < call.segments; ++s) {
        const Tensor* out = call.out[s];
        if (out == nullptr || out->dtype != DType::BF16 || !out->is_contiguous() ||
            call.out_row_offset[s] < 0 ||
            out->ne[0] < call.out_row_offset[s] + call.segment_rows[s] ||
            static_cast<std::int64_t>(out->ne[1]) * out->ne[2] * out->ne[3] != x.ne[1]) {
            throw std::invalid_argument("generic linear: output segment must be BF16 [rows,T]");
        }
        total += call.segment_rows[s];
    }
    if (total != call.rows || call.rows <= 0 || call.row_begin < 0) {
        throw std::invalid_argument("generic linear: output segments do not cover the rows");
    }
    const int sets = call.epilogue == Epilogue::SwiGlu ? 2 : 1;
    const std::int64_t last =
        static_cast<std::int64_t>(call.row_begin) + call.rows + (sets == 2 ? call.up_offset : 0);
    if (last > w.n || (sets == 2 && call.up_offset < call.rows)) {
        throw std::invalid_argument("generic linear: rows exceed the weight");
    }
}

} // namespace

// R8: a format with an R8 codec, whole load groups of K, 16-byte code rows, A8 allowed. A lane
// strides K by groups, so K need not fill 128-value tiles: Gemma 4 26B's dense down projection
// (K = 2112) and gpt-oss's hidden (2880) take R8 at decode instead of the A16 register MMA.
bool allows_r8(const Weight& w, LinearPolicy policy) {
    return allows_a8(policy) && r8_supports(w.qtype) && (w.k % r8_k_values(w.qtype)) == 0 &&
           aligned_to(w.qdata, 16);
}

namespace {

// Whether a call of T tokens takes R8 (decode, T <= kR8DecodeTokens, and the activation fits):
// r8_call_refusal names why not.
bool r8_call(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return r8_call_refusal(w, policy, tokens) == R8CallRefusal::None;
}

// The bytes per block of each plane: the code plane (the whole block where blocks are unchanged),
// the high and the scale plane (core/weight.h ggml_plane_bytes).
struct MmqPlaneBytes {
    int codes, high, scales;
};

MmqPlaneBytes mmq_plane_bytes(QType q) {
    if (q == QType::GGML_Q4_K) return {144, 0, 0};
    if (q == QType::GGML_Q5_K) return {176, 0, 0};
    const auto p = ggml_plane_bytes(q);
    return {static_cast<int>(p.code), static_cast<int>(p.high), static_cast<int>(p.scale)};
}

// The weight in the layout ggml's kernels read: whole blocks for Q4_K / Q5_K, the planes otherwise.
bool ggml_layout_admits(const Weight& w) {
    const bool planes = w.qtype != QType::GGML_Q4_K && w.qtype != QType::GGML_Q5_K;
    if (planes != (w.layout == QuantLayout::GgmlPlanes)) return false;
    const auto bytes = mmq_plane_bytes(w.qtype);
    if ((bytes.high > 0 && w.qhigh == nullptr) || (bytes.scales > 0 && w.scales == nullptr)) {
        return false;
    }
    // ggml's loaders read every plane in vector words: a sign or scale plane off a 16-byte
    // boundary (a view into a packed buffer) takes another route.
    return aligned_to(w.qdata, 16) && (bytes.high == 0 || aligned_to(w.qhigh, 16)) &&
           (bytes.scales == 0 || aligned_to(w.scales, 16));
}

// Prefill (17 tokens and up) of the ggml block formats on ggml's matrix kernel instead of the W4A8
// tiles, kMmqPrefillSlice tokens at a time. ninfer-all's 24K prefill of ISTA GSQ IQ3_S led ours on
// the 16 GB cards, and its nsys is 75 % mul_mat_q (J 16..128, stream-k); ours ran the same kernel
// at verify widths only. 24K prompt, two passes: RTX 4060 Ti 931 -> 971 tok/s (ninfer-all 1004),
// the depth decode after it 47.7 -> 52.3; L4 763 -> 821; A10 (sm_86) 819 -> 897; RTX 2080 Ti
// (Qwen3.8 27B IQ2_XXS) 611 -> 671, where the W4A8 tiles' BF16 path is emulated; RTX 4090 with
// Unsloth's UD-Q4_K_XL 2977 -> 3330 (C1 and C8 even); RTX 5090 3930 -> 4306, UD-Q4_K_XL
// 3771 -> 4152. Every format the kernel reads: only the classic-tile ones (IQ4_XS / IQ3_S /
// IQ2_XS / IQ2_XXS / Q2_K) gave 949 on the 4060 Ti. The small sm_120 keeps the tiles (RTX 5060 Ti,
// 36 SMs: 1047 -> 1015), and so does sm_80: A100 40 GB 1340 -> 1142. Every other board takes the
// nearest measured board's answer of its instruction set at any distance, as the "64 SMs" line
// between the 5060 Ti and the 5090 did (an RTX 5080 the 5090's, an RTX 5070 Ti, 5070 or 5060 the
// 5060 Ti's: the 5070 Ti's route profile prefills on the tiles, DECISIONS 62); Volta and the
// architectures nobody measured keep the tiles. The gate's table: measured::kMmqPrefillPays
// (core/measured_boards.h), which the product's estimate reads too.
bool mmq_prefill_device() { return measured::mmq_prefill_device_pays(); }

} // namespace

// Without tensor mma.sync (Volta) ggml's matrix kernel is out (device_supported: sm_75 on) and the
// register-MMA tiles run on FP32 FMAs, so a call of 9..16 tokens (a deep speculative verify) runs
// the vector kernel eight columns at a time (V100, Gemma 4 31B Q4_K_XL MTP, draft window 10: the
// emulated tiles 1.0..2.6 ms a call, 67 % of the run); Q4_K / Q5_K / Q6_K / Q8 take Volta's HMMA
// route from five tokens instead (volta_hmma_route).
std::int32_t vec_max_tokens() {
    try {
        const auto& f = device_class().features;
        return f.mma_tensor && !f.mma_slow ? ggml_mmq::kVecMaxColumns
                                           : 2 * ggml_mmq::kVecMaxColumns;
    } catch (const std::exception&) { return ggml_mmq::kVecMaxColumns; }
}

namespace {

// The verify widths the vector kernel takes by rule. Decode (T = 1) keeps R8 and the dense tuner:
// on the RTX 2080 Ti the vector kernel lost decode by 10 % (9B Q4_K_M C1 88 -> 79 tok/s) and won
// C4 by 10 %. Its integer dot products grow with the width on the CUDA cores while the register
// MMA's do not, so with tensor mma.sync the register MMA takes the call from five tokens: in the
// dense tuner's timings
// of eight cards the vector kernel was within 4 % of the best route at T = 2..4 and behind it at
// T = 8 on all of them, the most where a card has the least compute per byte of bandwidth
// (A100: x1.5..1.6, RTX 3090 x1.4..1.5, RTX 5090 x1.1, RTX 4090 x1.03).
// Without tensor mma.sync (Volta) the register MMA runs on FP32 FMAs, so the vector kernel takes
// every width it reads (vec_max_tokens) but where Volta's HMMA route takes the call (Q4_K / Q5_K /
// Q6_K / Q8 from five tokens, volta_hmma_route): V100, Qwen3.8 27B UD-Q4_K_XL, the register MMA at
// T = 5..16 about 0.5 ms a call against 0.08 for the vector kernel at T = 4; C8 aggregate 37.3
// tok/s against C4 86.3, DFlash2 (verify 6) C1 15.7 against plain 30.1.
constexpr std::int32_t kVecRuleMaxTokens = 4;

std::int32_t vec_rule_max_tokens() {
    // vec_max_tokens() reaches past eight columns only without tensor mma.sync.
    const std::int32_t widest = vec_max_tokens();
    return widest > ggml_mmq::kVecMaxColumns ? widest : kVecRuleMaxTokens;
}

bool vec_rule(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= 2 && tokens <= vec_rule_max_tokens() && vec_call(w, policy, tokens);
}

// Speculative verify (T 9..16) of the codebook formats takes ggml's matrix kernel on
// every device, ahead of the route tables (which timed A16 schedules only) and the A16 rules: its
// int8 tile decodes a weight once per call where the register MMA's dequant runs per token tile.
// End to end, ISTA IQ3_S 27B C4 MTP3 (16-token verify), every iq / K-quant band 9..16 on mmq:
// L4 79.4 -> 89.2 tok/s (+12.3 %), RTX 5070 Ti 242.7 -> 270.9 (+11.6 %); ninfer-all runs
// mul_mat_q<16> there (92.9 / 280.1). The route_tune grid, one call with L2 flushed, does not show
// it (L4 rule 1.06 of its best). Where the BF16 MMA is emulated (Turing) every mmq format takes
// it from 9 tokens, and the codebook formats below that too (mmq_verify_rule).
bool bf16_mma_native();
// The codebook formats only: their register-MMA dequant looks every value up in a grid table per
// token tile, which the matrix kernel's one decode a call saves. The K-quants' affine dequant is
// cheap, and there the register MMA keeps the band (RTX 4090, cellgt ABBA: UD-Q4_K_XL 27B DFlash2
// C1 115.7 -> 111.7 and C4 271.5 -> 263.1 with them in the rule, Muse 30B Q4_K_M C4 282.7 -> 267.6).
bool mmq_verify_format(QType q) { return format_traits(q).mmq_verify; }
constexpr std::int32_t kMmqVerifyMinTokens = 9;
bool mmq_verify_rule(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens) {
    // The measured band is 9..16 tokens of several sequences (a C4 verify). One sequence keeps its
    // register MMA at every width (ops::round_lanes): a C1 MTP round of 5..8 tokens and a C1 copy
    // round of 16 lost on mmq (RTX 5070 Ti C1 117.0 -> 115.2 tok/s, all of it in the copy rounds:
    // with them off, 116.5 both ways), while the C4 verify gains (233.5 -> 249.3).
    // Where the BF16 MMA is emulated (Turing) a verify of 5..8 tokens keeps mmq for the codebook
    // formats only: the affine ones (the K-quants, Q4_0) dequant cheaply on the register MMA (RTX
    // 2080 Ti, one call each, T 5..8: Q3_K-Q6_K 10-58 % faster off mmq; Gemma 4 12B Q4_K_M MTP5
    // 116.7 -> 136.1 tok/s), while the codebook IQ4_XS lost 23-38 % there.
    if (!bf16_mma_native())
        return mmq_call(w, policy, rows, tokens) &&
               (tokens >= kMmqVerifyMinTokens || mmq_verify_format(w.qtype));
    return mmq_call(w, policy, rows, tokens) && tokens >= kMmqVerifyMinTokens &&
           ops::round_lanes() != 1 && mmq_verify_format(w.qtype);
}

// The bytes free in the rest of the arena from its next 256-byte address (nullopt: no arena; 0
// where that address is already past its end: no route's planes are empty), against
// which mmq_workspace_bytes' layout is held (resolve_route). Its blocks are 256-aligned, so from
// there they land exactly as the layout counted them: an arena the planner sized to the route
// holds it with nothing to spare.
std::optional<std::size_t> workspace_free(const DeviceArena* workspace) {
    if (workspace == nullptr) return std::nullopt;
    const auto base = reinterpret_cast<std::uintptr_t>(workspace->base());
    const auto start =
        ((base + workspace->used() + 255U) & ~static_cast<std::uintptr_t>(255U)) - base;
    return start <= workspace->capacity() ? workspace->capacity() - start : 0;
}

std::atomic<std::uint64_t> mmq_fallbacks{0};
std::atomic<std::uint64_t> route_demotions{0};
std::atomic<std::uint64_t> joined_launches{0};

// A demotion that keeps the bytes: the staged MMA and the NVFP4 tiled route hand a call they
// cannot read to the register MMA of the same summation order.
bool same_bytes(Route asked, Route ran) {
    return ran.family == Family::RegisterMma &&
           (asked.family == Family::StagedMma || asked.family == Family::Nvfp4TiledMma);
}

// A call the asked route could not take ran another route, with other bytes and speed than a
// route table or the heuristic chose: said once per pair of routes, with the call's facts, and
// counted (route_demotion_count). No product run demoted in the lab (perplexity, generation and
// serving at 1 and 4 streams on Ada, Blackwell and Turing); one names a caller that sized or
// aligned its call short of its route.
void note_demotion(Route asked, Route ran, int rows, int tokens, int k, bool x16, std::int64_t ldx,
                   std::size_t workspace) {
    route_demotions.fetch_add(1, std::memory_order_relaxed);
    static std::mutex mutex;
    static std::set<std::pair<std::string, std::string>> said;
    const std::lock_guard lock(mutex);
    if (!said.emplace(route_name(asked), route_name(ran)).second) { return; }
    std::fprintf(stderr,
                 "ninferno: warning: route %s cannot take a call of %d rows x %d tokens (k %d, x "
                 "%s16-byte aligned, ldx %lld, workspace %zu bytes): it ran %s, with other bytes "
                 "and speed\n",
                 route_name(asked).c_str(), rows, tokens, k, x16 ? "" : "not ",
                 static_cast<long long>(ldx), workspace, route_name(ran).c_str());
}

int sm_count() {
    try {
        return device_class().sms;
    } catch (const std::exception&) {
        return 1;
    }
}

// Index of the register-MMA schedule with this shape; the list is generic_gemm_launch.cuh's.
int rmma_index(int warps_m, int warps_k, int m_tiles, int n_tiles, int cluster = 1) {
#ifdef NINFERNO_WITHOUT_NVFP4
    if (cluster > 1) {
        // Thread-block clusters are sm_90+; the Ada build compiles C == 1 schedules only and
        // the selector must never offer a cluster split there.
        cluster = 1;
    }
#endif
    for (std::size_t i = 0; i < kRmmaSchedules.size(); ++i) {
        const auto& s = kRmmaSchedules[i];
        if (s.warps_m == warps_m && s.warps_k == warps_k && s.m_tiles == m_tiles &&
            s.n_tiles == n_tiles && s.cluster == cluster) {
            return static_cast<int>(i);
        }
    }
    return 0;
}

// ggml formats that the decode heuristic below measured faster on the four-warp CTA once the
// eight-warp grid needs a second wave (RTX 5080, T = 1): Q4_K, Q5_K, IQ4_XS, IQ3_XXS and the Q4_0
// decoder family at n = 3840..8192, and the codebook formats IQ2_S and IQ3_S, whose
// decode rather than the weight stream bounds them, by 11-16% at n = 4096..12288 (IQ3_S
// 4096x2560 11.4 -> 9.7 us, IQ2_S 9216x2560 21.0 -> 18.6 us). Q8 (Q8_0) and Q6_K at K >= 12288
// were faster on eight warps at every size.
bool light_ggml_step(QType q) { return format_traits(q).light_step; }

// Shape-class heuristic for the register MMA. It follows what ninferno_route_tune measured across
// the recorded problems (Qwen3.5 0.8B in four formats on an RTX 5090): at T <= 16 one 16-row M tile
// per CTA with K split across as many warps as keep >= 128 K values each (weight streaming wants
// many CTAs each reading short contiguous runs); from 17 tokens outside the prefill profile by T
// and the waves its tiles fill (below, measured on seven boards); under the prefill profile
// WarpsK == 1 (so every call of 17 tokens or more sums K in one order) with eight N tiles, four
// for a band that eight would pad (below).
//
// Decode (T <= 8, one N tile) with K split eight ways also looks at how many CTAs the call makes
// (one per 16 rows) against the SMs (RTX 5080, Qwen3.5-4B/9B and Gemma 4 12B shapes):
// - a call of at most a quarter as many CTAs as SMs (the 32-row GDN a/b projections: 2 CTAs, 12.6
//   us for 139 KB) splits K over a cluster of eight CTAs as well: 7.5 -> 3.1 us;
// - a call of at most one CTA per SM (an output projection of 2048..2688 rows: 128..168 CTAs on
//   170 SMs) splits K over a cluster of two: half of every CTA's K runs on an SM the call left
//   idle (RTX 5090, T = 1..4, L2 flushed: Q8 2048x4096 23.7 -> 17.6 us, 2688x4096 25.5 -> 17.6
//   us, 2048x8192 39.8 -> 30.1 us, Q4_K 2048x4096 13.5 -> 9.4 us; a cluster of four was no
//   faster, and from 176 CTAs (2816 rows) the pair only ties); without clusters (the Ada build)
//   sixteen K-split warps in one CTA instead (RTX 4090, 128 SMs, T = 1..4: Q8 1280x4096 26.3 ->
//   16.2 us, 2048x4096 26.6 -> 20.5 us, Q4_K 1536x4096 16.4 -> 12.3 us, BF16 2048x6144 46.1 ->
//   45.1 us; from 160 CTAs it loses: Q4_K 2816x4096 17.4 -> 21.5 us);
// - past two CTAs per SM (what the eight-warp CTA holds) the eight-warp grid runs a second, part
//   empty wave; for the light ggml formats four warps per CTA (four CTAs per SM, one wave) is
//   2-10% faster at n = 3840..8192 (Q4_K 4096x4096 16.0 -> 14.7 us, Q5_K 18.1 -> 16.6 us, Q4_0
//   13.3 -> 12.8 us, IQ4_XS 15.1 -> 14.2 us); at n = 2560 (160 CTAs) eight warps stay faster.
//   SwiGLU (gate and up rows: twice the loads and accumulators per warp) is not: Qwen3.5-9B's
//   Q4_K gate/up at 24576x4096 took 76.0 us on eight warps and 93.9 us on four.
// K over the SMs a call of at most one CTA per SM leaves idle: sixteen warps of one CTA where
// clusters do not exist, else a cluster of eight CTAs for a call of at most a quarter as many CTAs
// as SMs and a CTA pair for the rest (heuristic_rmma).
int one_wave_split([[maybe_unused]] std::int64_t ctas, [[maybe_unused]] std::int64_t sms) {
#ifdef NINFERNO_WITHOUT_NVFP4
    return rmma_index(1, 16, 1, 1);
#else
    return 4 * ctas <= sms ? rmma_index(1, 8, 1, 1, 8) : rmma_index(1, 8, 1, 1, 2);
#endif
}

// Waves of the SMs a grid of `rows` x `cols` tiles fills on the call (n rows, `tokens` tokens).
double speed_waves(std::int64_t n, std::int32_t tokens, int rows, int cols) {
    const std::int64_t tiles = ((n + rows - 1) / rows) * ((tokens + cols - 1) / cols);
    return static_cast<double>(tiles) / static_cast<double>(sm_count());
}

// Whether a format's register-MMA decode is heavy: more than 128 registers on the m1.n4 tile of
// four K warps (cuobjdump of every compiled format, sm_75..sm_120), every format but BF16 on a
// native BF16 MMA. A heavy decode wants fewer K warps per token tile and longer K steps.
bool heavy_decode(QType q) { return q != QType::BF16; }

// The format whose 64x64 register-MMA tile spills where its 128x32 tile spills less (ptxas,
// sm_86..sm_120): Q4_K, whose 128x32 tile was 10..20 % faster on all seven boards from 97 tokens.
bool tall_tile_format(QType q) { return format_traits(q).tall_tile; }

bool bf16_mma_native();

// Q6_K's one-tile call on a grid of one to two eight-warp CTAs per SM (sms < ctas <= 2 sms),
// where the board's DRAM bytes per SM cycle are below this: the four-warp CTA, half the warps on
// the SM issuing the plain decoder's narrow loads into a load queue they fill (rmma_wide_rule).
// Store / add, T = 5 and 8, K 4096 and 15360, n at 1.25..2.0 CTAs per SM (w1x4 against w1x8):
// RTX 4090 (3.1) 0.82-0.98, 4070 Ti (3.2) 0.80-0.99, 4060 Ti (3.4) 0.87-1.00; on the 5070 Ti
// (5.2) 0.98-1.12. The boundary sits below the RTX 5080 and 5090 (4.4), not measured, which keep
// eight warps. Q4_K lost on w1x4 on all four boards (1.04-1.53), and a grid of more than two CTAs
// per SM keeps eight warps (light_ggml_step).
inline constexpr double kRmmaQ6kFourWarpBytes = 4.0;

// K-quant SwiGLU at T <= 8 on a board without cp.async (sm_75) whose bytes per SM cycle take the
// latency variant: four K warps, the CTA whose register cap holds the activation ring of both row
// sets (kRmmaAheadForm, generic_gemm.cuh; 214-217 registers on sm_75: the eight-warp CTA, whose
// two CTAs a SM cap it at 128, has no ring form, and its latency kernel without one lost 7-8 %).
// RTX 2080 Ti (5.9), Qwen3.5-9B gate/up 12288x4096 Q4_K: w1x8 142.4 us (plain, the old pick)
// against 122.3 us w1x4 with the ring at T = 8, 141.2 -> 118.7 at T = 5, 136.5 -> 111.8 at T = 1.
bool ring_swiglu_four_warps(QType q) {
    // Q4_K and Q5_K: the formats measured (Q3_K and Q6_K SwiGLU keep eight warps until they are).
    if (q != QType::GGML_Q4_K && q != QType::GGML_Q5_K) return false;
    try {
        const auto& d = device_class();
        return turing_cc(d.major, d.minor) && dram_bytes_per_sm_cycle() >= kRmmaLatencySwiGluBytes;
    } catch (const std::exception&) {
        return false;
    }
}

// The K values one register-MMA step of the format covers: four runs of its decoder (rmma_kernel's
// KStep = 4 x Decoder::kRun): 32 for BF16, Q8 and U4 G32 (eight-value runs), 64 for FP8, NVFP4
// and the 16-value groupwise runs, 128 for the 32-value decoders (K-quants, IQ, MXFP4, the
// quad-run U4 G32 / G64 at K % 128 == 0) and for the ggml 32-blocks stepped four at a time
// (run_values 128).
std::int64_t rmma_k_step(QType qtype, std::int32_t k) {
    Weight weight;
    weight.qtype = qtype;
    WeightCodec codec{};
    try {
        codec = codec_of(weight);
    } catch (const std::invalid_argument&) {
        return 128;
    }
    const bool quad_u4 = (codec == WeightCodec::U4ZpG32 || codec == WeightCodec::U4ZpG64) &&
                         k > 0 && (k % 128) == 0;
    const int run = quad_u4 ? 32 : run_values(codec);
    return run >= 128 ? run : 4 * run;
}

// The K split of a one-tile store / add call of more CTAs than SMs (T <= 8), held to one fact of
// the addressing, never a timing: the CTA's longest warp walks more K steps of its rows than its
// load ring holds (two steps, rmma_kernel's ring; a step is rmma_k_step values): steps / warps
// rounded up, at least three. The CTA ends with that warp; with two steps or fewer its ring never
// refills, every step's load latency and the cross-warp reduction exposed (Q4_K K = 4096 on
// sixteen warps, two steps each, streamed DRAM at 57..70 % of peak on all five lab boards against
// 71..74 % for the best split). K = 2560 on eight warps (three steps for half the warps) keeps up:
// Q6_K 4096x2560 at T = 5 / 8 on eight warps against four, RTX 4070 Ti 19.6 against 21.1 us, RTX
// 3060 32.4 against 32.9 (the RTX 5070 Ti and 2080 Ti ran four warps faster there: DECISIONS 61).
// Its callers start at eight or four warps; eight that break it take four where four keep it and
// the grid fills their CTA slots, else the pick stands. Sixteen warps hold one CTA a SM
// (kRmmaMinBlocks caps the split's one-tile store / add kernels at 128 registers a thread, so `w` K
// warps hold 16 / w CTAs a SM): fewer warps a CTA keep the warps a SM holds only on a grid of at
// least that many CTAs a SM. Below it the split trades resident warps (the loads in flight) for
// ring refills: K = 1024 (2048 rows, 128 CTAs) on four warps against eight, RTX 5070 Ti (1.8 CTAs
// a SM) 20-37 % slower, RTX 5090 (0.75) 18-38 %, RTX 4070 Ti (2.1) within 3 %; the RTX 4060 Ti
// (3.8, just under its four slots) ran four warps 2-16 % faster, which this leaves on eight.
// Open (DECISIONS): how much a grid's last, partly filled round of resident CTAs costs. Pricing
// it by its empty slots was wrong on two of four boards (RTX 4090 Q6_K 3840x15360 add: four warps
// in one half-filled round 65 us, eight warps in one full round 79; RTX 3060 Q4_K 4096x2560:
// eight warps 8-10 % slower although they fill more) and right on two (RTX 4070 Ti / 5070 Ti
// light K-quants 6-12 % faster on eight): a round under the warps that saturate DRAM is what
// costs, and that count is not a fact we hold yet.
int rmma_k_split_fit(int index, const Problem& problem, std::int64_t ctas, std::int64_t sms) {
    if (problem.epilogue == Epilogue::SwiGlu || sms <= 0 || ctas <= sms) return index;
    const RmmaScheduleInfo info = rmma_schedule_info(index);
    if (info.warps_m != 1 || info.m_tiles != 1 || info.n_tiles != 1 || info.cluster != 1) {
        return index;
    }
    const std::int64_t step  = rmma_k_step(problem.qtype, problem.k);
    const std::int64_t steps = (static_cast<std::int64_t>(problem.k) + step - 1) / step;
    const auto refills = [&](int warps_k) { return (steps + warps_k - 1) / warps_k > 2; };
    // The warps a SM holds at 128 registers a thread (kRmmaMinBlocks).
    constexpr std::int64_t kSlotWarps = 65536 / (32 * 128);
    if (info.warps_k <= 4 || refills(info.warps_k) || !refills(4) || ctas * 4 < kSlotWarps * sms) {
        return index;
    }
    return rmma_index(1, 4, 1, 1);
}

// Outside the prefill profile a tiled codec's 128x128 kernel pays once its tiles fill kSpeedFill
// waves of the SMs (from 33 tokens, on a native BF16 MMA): below that the register MMA's
// narrower tiles fill them (Q4_K n = 1024 at T 64..128: the tiled kernel 5..7x on every board).
bool tiled_speed_pays(std::int32_t n, std::int32_t tokens) {
    return tokens >= kTiledSpeedMinTokens && bf16_mma_native() &&
           speed_waves(n, tokens, 128, 128) >= kSpeedFill;
}

// The invariant profile's register MMA (prefill_routes.h): the decode split of K for every T, so
// a token's bytes are the same at every width (the order depends on WarpsK alone,
// generic_gemm.cuh). Eight tokens per CTA through T = 8, where decode batches live; 32 above, so
// a prefill chunk reads each weight a quarter as often.
int invariant_rmma(std::int32_t k, std::int32_t tokens) {
    const int n_tiles = tokens <= 8 ? 1 : 4;
    if (k >= 1024) return rmma_index(1, 8, 1, n_tiles);
    if (k >= 512) return rmma_index(1, 4, 1, n_tiles);
    return tokens <= 8 ? rmma_index(4, 1, 1, 1) : rmma_index(4, 1, 1, 8);
}

// W4A4 tiles. Under the prefill profile the widest tile (the tuner's prefill winner on every
// recorded NVFP4 problem but one, where it was within noise); otherwise the widest that still
// gives every SM a CTA, else the narrow ones.
int heuristic_w4a4(std::int32_t rows, std::int32_t tokens, bool prefill_band) {
    if (prefill_band) return w4a4_schedule_count() - 1;
    const int sms   = sm_count();
    const auto grid = [&](int block_m, int block_n) {
        return ((rows + block_n - 1) / block_n) * ((tokens + block_m - 1) / block_m);
    };
    if (grid(128, 128) >= sms) return 3;
    if (grid(64, 128) >= sms) return 2;
    if (grid(32, 128) >= sms) return 1;
    return 0;
}

// W4A8 tiles (indices into kW4a8Schedules), chosen by format and T in the product (RTX 5080,
// Qwen3.8-27B UD-IQ4_XS p512 under fast). 64 tokens by 64 rows by default (the fastest of ten
// tile shapes on Gemma 4 12B Q4_0, SmolLM3 3B Q4_0 and Q4_K_M and TinyLlama Q5_0). From 128 tokens,
// 128x64 for the GGUF lookup and K-quant formats of that model: IQ4_XS (0.82-0.97 per call
// against 64x64, p512 0.93x -> 0.98x llama.cpp), IQ2_S, IQ2_XS and Q6_K (0.53-0.92), then
// IQ3_S, IQ3_XXS, Q4_K and Q5_K (p512 2155-2162 -> 2195-2203 = 0.98x -> 1.00x llama.cpp,
// A B B A twice). 128x128 was slower in the product for IQ4_XS (1.06-1.29x per call, p512
// 1890) though faster in an isolated benchmark: the product decides. Every W4A8 tile sums K
// in one order, so the choice never changes a bit of the output (all tiles byte-identical).
//
// From 128 tokens the 128x64 tile with four producer warps (w4a8_ws_kernel) replaces the classic
// 128x64 tile of the formats measured in the product, on the GPUs measured: the same grid, a
// faster CTA, the same bytes. RTX 5080 (sm_120), Muse 30B IQ3_XXS (IQ3_XXS, Q6_K and Q4_K layers)
// at a 29.7k-token prompt, chunk 1024: prefill 2103 -> 2431 t/s, W4A8 kernel time 12.37 -> 10.40 s;
// Gemma 4 31B UD-IQ3_XXS (IQ3_XXS, IQ2_S) at 7.9k tokens: 1696 -> 1912 t/s; greedy outputs
// byte-identical. Per call (W4A8 micro-bench, T = 200 and 1024,
// time against the best classic tile), RTX 5080 and RTX 5090: IQ3_XXS
// 0.79-0.93, IQ2_S 0.86-0.95, Q4_K 0.88-0.95, Q6_K 0.90-0.98; RTX 4090 (sm_89): IQ3_XXS 0.78-0.94,
// IQ2_S 0.89-0.93, Q6_K Store 0.95-0.96 (at down-projection and attention shapes: Store
// geomean 0.975, AddResidual 0.985, 0.92-1.06), while Q4_K (1.06-1.13) and Q6_K SwiGLU (1.10) lose
// there and keep the classic tile. IQ3_S on the global grid lost (1.01-1.08: its decode did not
// keep four producer warps ahead of the MMA warps); on its shared-table codec (W4a8Iq3SPaired) it
// is taken on sm_89 / sm_120 (w4a8_ws_extra). The
// micro-bench also has the producer-warp tile ahead for IQ4_XS, IQ2_XS, Q5_K and, against 64x64,
// for IQ2_XXS, Q8, Q4_0/Q4_1/Q5_0/Q5_1 and IQ4_NL on sm_120: not taken until measured in the
// product (a 64x64 format moving to 128 tokens per CTA halves its grid); of them only Q4_0 is
// compiled for it (w4a8_ws_supports) and taken, on sm_89 (below). Only the measured
// architectures take it: sm_120 (RTX 5080, 5090) and sm_89 (RTX 4090) for the formats below,
// Ampere (sm_80/86) for every format it reads (w4a8_ws_wins); sm_121 and the rest keep the
// classic tiles. IQ3_XXS's producer decode reads the sign plane 8 bytes at a time: a plane not
// 8-byte aligned keeps the classic tile.
//
// Q4_0 on the RTX 4090, from 64x64, for matrices of at least 5376 x 5376: Gemma 4 31B QAT (all
// Q4_0, every problem 5376 or more each way) prefill 1914 -> 2149 t/s at 22.6-31.4k tokens
// (A B B A, greedy byte-identical); per call (W4A8 micro-bench, T 1024 / 512) gate+up 43008x5376
// 0.51 / 0.53, q/k/v 0.84-0.90, o and down 0.81-0.90. Gemma 4 26B QAT's dense Q4_0 (K 2816, or 2816
// rows: 13-15 MB, read cold in the product but from L2 in the micro-bench) ran 0.357 -> 0.383 s per
// prompt there (nsys) and keeps 64x64. The bound is the smallest measured win: nothing
// between 2816 and 5376 was measured (a 4096-wide 8B-class model, 9.4 MB matrices), nor any sm_89
// card with less L2 than the RTX 4090's 72 MB (RTX 4060 Ti, 4070 Ti Super, 4080).
// IQ4_XS, IQ2_XS, IQ2_XXS and IQ3_S on sm_89 / sm_120 (see the note below): the producer-warp
// tile also replaces their classic 128x64 tile where a measured choice (the shipped dense tuner
// table) names it, since those measurements had no producer-warp tile of these formats to time.
bool w4a8_ws_extra(QType q) {
    if (!format_traits(q).w4a8_ws_tuned) return false;
    static PerDevice<bool> on;
    return on.get([](int) {
        try {
            const auto& d = device_class();
            return gb20x_cc(d.major, d.minor) || ada_cc(d.major, d.minor);
        } catch (const std::exception&) {
            return false;
        }
    });
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

bool w4a8_ws_wins(const Problem& p, const Weight& w) {
    if (!w4a8_ws_admits(w)) return false;
    bool blackwell = false, ada = false, ampere = false;
    try {
        const auto& d = device_class();
        blackwell     = gb20x_cc(d.major, d.minor);
        ada           = ada_cc(d.major, d.minor);
        ampere        = ampere_cc(d.major, d.minor);
    } catch (const std::exception&) {
        return false;
    }
    // Ampere, sm_80 and sm_86 alike (measured on an RTX 3090, ga102: the tile forced on every
    // format it reads, one 16K prompt, D WS
    // WS D): Muse 30B Q4_K_M prefill 1278 / 1273 -> 1398 / 1399 t/s, Qwen3.8 27B UD-Q4_K_XL 1290 /
    // 1289 -> 1314 / 1314, Gemma 4 31B 1071 -> 1081 / 1081; no format lost.
    if (ampere) return true;
    switch (p.qtype) {
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ2_S:
        return blackwell || ada;
    // Q6_K on sm_89 lost once the 128-token tiles run token-major there (kW4a8TokenMajor): RTX
    // 4090, Muse 30B Q4_K_M (Q6_K Store/AddResidual is its only producer-warp format on sm_89),
    // 16K prompt: 2935 / 2931 t/s with the tile against 2997 / 3001 without (+2.2 %).
    case QType::GGML_Q6_K:
        return blackwell;
    case QType::GGML_Q4_K:
        return blackwell;
    case QType::GGML_Q4_0:
        return ada && p.n >= 5376 && p.k >= 5376;
    // ISTA GSQ IQ3_S mix, 24K prefill against ninfer-all's MMQ (RTX 5060 Ti nsys, same
    // tokens): the classic-tile formats carried the whole gap - IQ4_XS 16.1 s against 10.99,
    // Q2_K 3.10 / 2.51, IQ2_XS 2.81 / 2.32, IQ2_XXS 1.00 / 0.79 - while the producer-warp formats
    // were even (IQ3_XXS 12.0 / 11.8, IQ2_S 3.96 / 3.67). In the product (24K prompt, A B B A,
    // greedy byte-identical): RTX 5060 Ti 981 -> 1011 t/s with IQ4_XS / IQ2_XS / IQ2_XXS;
    // RTX 4060 Ti 900 -> 901 with those, 926 with IQ3_S on its shared-table codec too.
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ3_S:
        return w4a8_ws_extra(p.qtype);
    default:
        return false;
    }
}

// Q4_K from 128 tokens on the RTX 4090 (sm_89, token-major grid, the 128x128 tile's Ada flavor):
// 128 tokens by 128 rows wherever that grid still gives three quarters of the SMs a CTA. Muse 30B
// Q4_K_M shapes (W4A8 micro-bench, time against the classic 128x64 at T = 1024 / 512 / 200): SwiGLU
// 19968x6656 0.86 / 0.86 / 0.94, Store 6656x19968 0.98 / 0.84 / 0.87, 4096x6656 0.89 / 0.90 and at
// T = 200 (64 CTAs) 1.65, 256x6656 1.71: those keep 128x64.
bool w4a8_wide_q4k(const Problem& p, std::int32_t tokens) {
    if (p.qtype != QType::GGML_Q4_K) return false;
    try {
        const auto& d = device_class();
        if (!ada_cc(d.major, d.minor)) return false;
        const std::int64_t rows_per_block = p.epilogue == Epilogue::SwiGlu ? 64 : 128;
        const std::int64_t ctas =
            ((p.n + rows_per_block - 1) / rows_per_block) * ((tokens + 127) / 128);
        return 4 * ctas >= 3 * static_cast<std::int64_t>(d.sms);
    } catch (const std::exception&) {
        return false;
    }
}

// R8 decode (T = 1) on the RTX 5080 at Qwen3.5-4B/9B shapes: Q8 is fastest on
// four warps of one row each everywhere but 2560x9216 (within 2%); the codebook formats move to
// eight warps of two rows from K >= 8192 (the activation each CTA quantizes grows with K: IQ3_XXS
// 4096x12288 32.1 -> 27.9 us), and IQ2_XXS/XS/S also for N >= 12288 (IQ2_S 12288x4096 28.3 -> 24.9
// us) and to two rows per warp for N >= 4096 (IQ2_XS 9216x2560 14.1 -> 12.8 us). The ggml 4/5-bit
// blocks and K-quants (27B/31B/35B shapes) follow the same K rule within 2 % of
// their best schedule. Every R8 schedule sums a row in the same order (lane l takes block groups l,
// l + 32, ...), so the choice changes no bit.
int r8_index(int warps, int rows) {
    for (std::size_t i = 0; i < kR8Schedules.size(); ++i) {
        if (kR8Schedules[i].warps == warps && kR8Schedules[i].rows == rows) {
            return static_cast<int>(i);
        }
    }
    return 0;
}

// Verify widths (the pre-quantized form, 27B/31B/35B shapes): four warps of one
// row, IQ2_* two rows at T = 4 and from N = 34816 (within 3 % of the best schedule elsewhere).
//
// Decode at K = 5120..6656 on the RTX 5080 (gb203:84sm; T = 1, store epilogue,
// Gemma 4 31B / Qwen3.8 shapes): each CTA quantizes the whole activation before its first block,
// and at this K a CTA of four single-row warps spends a visible share on that prologue. IQ3_XXS
// moves to two rows per warp from N = 8192 (43008x5376 112.9 -> 106.6 us, 8192x5376 26.4 -> 24.0;
// 5120x6144 gains 1 %, and at K <= 4096, w4r1 is ahead at every N), IQ3_S and IQ2_S to
// eight warps of two rows from N = 5120 (IQ2_S 5120x6144 20.5 -> 15.9 us, IQ3_S 8192x5376 28.5 ->
// 27.5; never slower on the larger shapes). A SwiGLU warp already reads two rows (gate and up),
// which is where these rules land a store call, so SwiGLU calls keep the schedules above (5080,
// repack_bench: Gemma 31B's 21504x5376 w4r1 104.1, w4r2 105.2 us); so do IQ2_XXS and IQ2_XS, not
// measured at this K. Other GPUs, 5080-class boards with another SM count included, run these
// shapes differently (the RTX 4090: w4r1 fastest on IQ3_XXS 43008x5376, 96.7 against w4r2 99.8
// us): they keep the shape-class rules.
//
// The K >= 5120 rule holds on the sm_12x family, not on one board: the RTX 5060 Ti (gb206:36sm,
// u3r, ninferno_r8_schedule_bench on the IQ3 16 GB artifacts' 63 problems) is fastest with eight
// warps of two rows on every IQ2_*/IQ3_S store and add from N = 5120 (IQ2_XXS 10240x5120 w4r2
// 43.5 -> w8r2 40.0 us, IQ2_XS 17408x5120 83.6 -> 79.4, IQ3_S 16384x5120 w4r1 100.6 -> 94.5) and
// IQ3_XXS two rows per warp from N = 5120 (16384x5120 92.8 -> 82.5, 6144x5120 36.0 -> 34.0), so
// the rule now keys on the architecture, with IQ2_XXS/XS joining IQ2_S. Q3_K at K >= 8192 stays at
// four single-row warps: Qwen3.8's 5120x17408 add is 7-10 % faster so on the 5060 Ti, the
// RTX 4080 SUPER and the RTX 4070 Ti SUPER (113.2 -> 103.0, 64.8 -> 60.3, 73.0 -> 66.3 us).
// Whether this tree's R8 kernel is the one-slot form (generic_r8.cuh NINFERNO_R8_ONE_SLOT: an
// sm_70 build; a tree compiles one architecture).
constexpr bool r8_one_slot_build() {
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
    return true;
#else
    return false;
#endif
}

int rules_r8(const Problem& p, std::int32_t tokens) {
    if (p.qtype == QType::Q8_G32_FP16) return r8_index(4, 1);
    // IQ1_S / IQ1_M: grid-lookup decoders of IQ2's weight, scheduled with them (sched_iq2).
    const bool iq2 = format_traits(p.qtype).sched_iq2;
    if (tokens >= 2) return iq2 && (tokens >= 4 || p.n >= 34816) ? r8_index(4, 2) : r8_index(4, 1);
    // Turing (the RTX 2080 Ti, tu102:68sm, ninferno_r8_schedule_bench on the IQ3 16 GB
    // artifacts' 63 problems: picks +6.25 % above the best schedules in sum) decodes the codebook
    // formats and Q3_K slower per word, and from K = 5120 eight warps of two rows win at every N
    // and epilogue (IQ3_XXS 17408x5120 w4r1 110.2 -> w8r2 84.5 us, IQ3_S 16384x5120 131.3 -> 114.2,
    // Q3_K 5120x17408 add 112.9 -> 89.0); Q6_K gains two rows per warp from N = 8192 or K = 4096
    // (the 248320x2048 lm_head 807 -> 723 us, 2816x8192 w8r2 42.3 -> w4r2 40.3).
    if (!device_class().features.mma_bf16_native) {
        const bool decode_heavy = format_traits(p.qtype).decode_heavy;
        if (decode_heavy && p.k >= 5120) return r8_index(8, 2);
        // Volta (an sm_70 build, whose R8 kernel is the one-slot form: generic_r8.cuh
        // NINFERNO_R8_ONE_SLOT; a GPU without mma.sync) keeps one row per warp for Q6_K at every N
        // and K, where the two-slot form took two. With one slot a second row per warp raises the
        // registers from 137 to 213 (3 CTAs per SM to 2) and the code a cold SM fetches first from
        // 960 to 1480 instructions. GV100 with ten other kernels between launches, as in a model
        // step: 4096x12288 add w4r1 88.8 against w4r2 94.7 us, 8192x4096 store 57.9 against 63.3.
        if (p.qtype == QType::GGML_Q6_K && p.epilogue != Epilogue::SwiGlu && r8_one_slot_build() &&
            !device_class().features.mma_tensor)
            return r8_index(4, 1);
        if (p.qtype == QType::GGML_Q6_K && p.epilogue != Epilogue::SwiGlu &&
            (p.n >= 8192 || p.k >= 4096))
            return r8_index(4, 2);
    }
    if (p.qtype == QType::GGML_Q3_K && p.k >= 8192) return r8_index(4, 1);
    if (p.k >= 8192 || (iq2 && p.n >= 12288)) return r8_index(8, 2);
    if (p.epilogue != Epilogue::SwiGlu && p.k >= 5120 &&
        consumer_blackwell_major(device_class().major)) {
        if (iq2 && p.n >= 5120) return r8_index(8, 2);
        // An SM-count rule, its edge between the measured boards: at 96 SMs or fewer (the
        // 5080's 84) IQ3 takes the wider warps; at 170 (the 5090) it keeps four single-row
        // warps: 10240x5120 IQ3_S w4r1 18.4 us against w8r2 20.3, IQ3_XXS 16.0 against w4r2
        // 18.0; IQ2 still gains (IQ2_S 10240x5120 w4r2 29.9 -> w8r2 20.1 us).
        if (device_class().sms <= 96) {
            if (p.qtype == QType::GGML_IQ3_S && p.n >= 5120) return r8_index(8, 2);
            if (p.qtype == QType::GGML_IQ3_XXS && p.n >= 5120) return r8_index(4, 2);
        }
    }
    if (iq2 && p.n >= 4096) return r8_index(4, 2);
    return r8_index(4, 1);
}

// Before the rules, a decode call takes the entry for the exact problem in the table of the
// nearest measured board (r8_decode_table.h), when that board is a variant of this one: the
// codebook problems of Gemma 4 31B, Muse Glimmer 30B, Qwen3.6-35B-A3B and Qwen3.8-27B where another
// schedule measured faster on that board. A table carries to boards within kR8CarryDistance of it
// (board_distance: a 4090 D or a 3090 Ti takes the 4090's or the 3090's, at 0.17 and 0.14) and no
// further: the 4090's 80 entries on the 4080 SUPER (distance 1.2; 3 sweeps) won 41, lost 30 (to
// -22 %: IQ2_XS 4096x6656 w8r2), +0.1 % in the mean, where its own sweep keeps 52 that win.
constexpr double kR8CarryDistance = 0.5;

// The current device's R8 decode table, once per ordinal (a process may drive boards of
// different classes); null where no measured board is near enough.
// A device query that fails is not cached (PerDevice retries a make that throws): the next call
// asks again rather than leaving the device on the heuristics for the process's life.
const R8DecodeTable* r8_decode_table() {
    static PerDevice<const R8DecodeTable*> table;
    try {
        return table.get([](int) {
            return nearest_known_measured(std::span(kR8DecodeTables), board_of(device_class()),
                                          kR8CarryDistance);
        });
    } catch (const std::exception&) {
        return nullptr;
    }
}

std::optional<int> tuned_r8(const Problem& p) {
    const R8DecodeTable* table = r8_decode_table();
    if (table == nullptr) return std::nullopt;
    const Epilogue epilogue = p.epilogue == Epilogue::AddResidual ? Epilogue::Store : p.epilogue;
    for (const R8DecodeEntry& e : table->entries) {
        if (e.qtype == p.qtype && e.epilogue == epilogue && e.n == p.n && e.k == p.k) {
            return r8_index(e.warps, e.rows);
        }
    }
    return std::nullopt;
}

// Every distinct problem that reached the generic path on a thread collecting them (seen_problems:
// an Engine's load, on its own thread, so another live Engine's calls stay out), and under
// NINFERNO_ROUTE_RECORD=<file> each distinct (format, compute, epilogue, N, K) appended to the file
// in ninferno_route_tune's --shapes format, so a model's shapes come from running it.
using SeenSet = std::set<std::tuple<int, int, int, std::int32_t, std::int32_t, std::int32_t, int>>;
struct ThreadSeen {
    bool collecting = false;
    SeenSet seen;
};
ThreadSeen& thread_seen() {
    thread_local ThreadSeen state;
    return state;
}
struct RecordedProblems {
    std::mutex mutex;
    std::set<std::tuple<int, int, int, int, int>> recorded;
};
RecordedProblems& recorded_state() {
    static RecordedProblems state;
    return state;
}

void record_problem(const Problem& p, const Weight& w) {
    static const char* path = std::getenv("NINFERNO_ROUTE_RECORD");
    if (ThreadSeen& t = thread_seen(); t.collecting) {
        t.seen.insert({static_cast<int>(p.qtype), static_cast<int>(p.policy),
                       static_cast<int>(p.epilogue), p.n, p.k, w.n, static_cast<int>(w.layout)});
    }
    if (path == nullptr || *path == 0) return;
    RecordedProblems& s = recorded_state();
    std::lock_guard lock(s.mutex);
    const auto compute = compute_class(p.qtype, p.policy);
    if (!s.recorded.insert({static_cast<int>(p.qtype), static_cast<int>(compute),
                            static_cast<int>(p.epilogue), p.n, p.k})
             .second) {
        return;
    }
    std::ofstream(path, std::ios::app) << qtype_name(p.qtype) << ' ' << compute_class_name(compute)
                                       << ' ' << epilogue_name(p.epilogue) << ' ' << p.n << ' '
                                       << p.k << '\n';
}

struct RecordedProjections {
    std::mutex mutex;
    std::set<std::tuple<std::string, std::string, std::int32_t, std::int32_t, std::int32_t>> seen;
};

RecordedProjections& recorded_projections() {
    static RecordedProjections state;
    return state;
}

} // namespace

void record_projection_route(const char* op, const char* route, std::int32_t tokens, std::int32_t n,
                             std::int32_t k) {
    static const char* path = std::getenv("NINFERNO_PROJECTION_RECORD");
    if (path == nullptr || *path == 0) return;
    RecordedProjections& s = recorded_projections();
    std::lock_guard lock(s.mutex);
    if (!s.seen.insert({op, route, tokens, n, k}).second) return;
    std::ofstream(path, std::ios::app)
        << op << ' ' << route << ' ' << tokens << ' ' << n << ' ' << k << '\n';
}

bool w4a8_ws_supports(QType q) { return format_traits(q).w4a8_ws; }

int heuristic_rmma(const Problem& problem, std::int32_t tokens, bool prefill_band) {
    const std::int32_t k = problem.k;
    const int split      = k >= 1024 ? 8 : k >= 512 ? 4 : 1;
    const auto narrow    = [&] {
        if (split == 8) {
            if (tokens <= 8) {
                const std::int64_t ctas = (static_cast<std::int64_t>(problem.n) + 15) / 16;
                const std::int64_t sms  = sm_count();
                if (ctas <= sms) return one_wave_split(ctas, sms);
                if (ctas > 2 * sms && light_ggml_step(problem.qtype) &&
                    problem.epilogue != Epilogue::SwiGlu) {
                    return rmma_k_split_fit(rmma_index(1, 4, 1, 1), problem, ctas, sms);
                }
                if (ctas > 2 * sms && problem.epilogue == Epilogue::SwiGlu &&
                    ring_swiglu_four_warps(problem.qtype)) {
                    return rmma_index(1, 4, 1, 1);
                }
                if (ctas <= 2 * sms && problem.qtype == QType::GGML_Q6_K &&
                    problem.epilogue != Epilogue::SwiGlu) {
                    double bytes = 0.0; // a device whose class cannot be read: eight warps
                    try {
                        bytes = dram_bytes_per_sm_cycle();
                    } catch (const std::exception&) {
                    }
                    if (bytes > 0.0 && bytes < kRmmaQ6kFourWarpBytes) {
                        return rmma_k_split_fit(rmma_index(1, 4, 1, 1), problem, ctas, sms);
                    }
                }
                return rmma_k_split_fit(rmma_index(1, 8, 1, 1), problem, ctas, sms);
            } else if (tokens <= 16) {
                // 9..16 tokens (verify of an 8- or 16-token draft): one 8-token tile per CTA would
                // put two or three CTA columns on the token axis, each streaming and decoding the
                // whole weight. Four K-split warps with two tiles cover 16 tokens per column: 1.00-1.12
                // of the best of seven schedules against 1.24-1.58 for w1x8 n1 (geomean per format,
                // RTX 5080, Q8/Q4_0/IQ4_XS/Q4_K/Q5_K/Q6_K/IQ3_XXS at 27B/31B/35B
                // shapes; at T = 24 Q8 1.35 against 1.68).
                // A call of at most one CTA per SM through T = 16 takes the decode split of K
                // above (two CTA columns): the four-warp tiles leave SMs idle on that grid. The
                // dense tuner's timings of nine cards (sm_70 to sm_120) had the rule above lose
                // such calls (n = 48..2048 rows: K/V, gates, small outputs) by x1.4..2.1 (median
                // per card) to the sixteen-warp split; on the RTX 5090 the CTA pair or cluster of
                // eight was 0..17 % faster still (n = 48 and 1024, T = 12 and 16: 8..20 us against
                // 16..41 for the rule).
                const std::int64_t ctas = (static_cast<std::int64_t>(problem.n) + 15) / 16;
                const std::int64_t sms  = sm_count();
                if (ctas <= sms) return one_wave_split(ctas, sms);
                return rmma_index(1, 4, 1, 2);
            }
            return rmma_index(1, 8, 1, 1);
        }
        if (split == 4) return rmma_index(1, 4, 1, 1);
        return rmma_index(4, 1, 1, 1);
    };
    if (prefill_band) {
        // 64 tokens per CTA column. A narrower band (a chunk's tail) pads that column: through 32
        // tokens 32 per column cover it with the same grid and half the MMA work, and through 64
        // two columns of 32 while one column leaves SMs idle (fewer than two CTAs per SM). RTX 4080
        // Super (ad103:80sm, the dense tuner's prefill bands, A16 IQ1_S, IQ1_M and Q2_K at 5120 to
        // 17408 rows, time of m1.n8 over the best): T = 17 and 32 1.48-3.21 on every problem; T = 64
        // 1.20-1.58 below 160 CTAs, 1.00 at 272. RTX 5090 (gb202:170sm), Q6_K A16 output head
        // (248320 rows): m1.n4 the best at T = 17 (1.22) and 32 (1.54), m1.n8 at 64. RTX 5060 Ti
        // (gb206:36sm, the same artifacts as the 4080 SUPER): T = 17 and 32 1.60-2.14 on every
        // problem. In the product (ninferno-perplexity at context 32, tuner off on both arms): 4080
        // SUPER Qwen3.8-27B IQ3 246.6 -> 252.9 tok/s, the 5090's Q4_K_XL even.
        const std::int64_t ctas = (static_cast<std::int64_t>(problem.n) + 63) / 64;
        const bool narrow_band = tokens <= 32 || (tokens <= 64 && ctas < 2 * sm_count());
        return rmma_index(4, 1, 1, narrow_band ? 4 : 8);
    }
    if (tokens <= 16 || (split != 8 && tokens < kRmmaMidMinTokens)) return narrow();
    // From 17 tokens outside the prefill profile (a batched speculative verify: C4 x MTP3, C8 x
    // DFlash; a DFlash drafter's context) one rule for every board, from the call's shape, the
    // SMs and whether the BF16 MMA is native: the largest tile whose grid still fills
    // kSpeedFill waves of the SMs (speed_waves), else a narrower token tile, else K split over
    // warps. ninferno_route_tune's timings of seven boards (RTX 5090, 4090, 3090, 2080 Ti, A10,
    // L4, A100: Qwen3.5/3.8 and ISTA shapes in seven formats, n 48..248320, T 17..2048), the rule
    // fitted on six and scored on the seventh: 1.116 of the best (geomean) against 1.219 for the
    // bands by T and CTAs per SM it replaced; a separate rule per board gained 1.1 % at most, so
    // a board nobody measured takes the same tiles. What moved the best tile besides the waves
    // was the format's kernel, not the board (heavy_decode, tall_tile_format).
    const bool emulated  = !bf16_mma_native();
    const bool heavy     = heavy_decode(problem.qtype) || emulated;
    const std::int64_t n = problem.n;
    const auto waves     = [&](int rows, int cols) { return speed_waves(n, tokens, rows, cols); };
    // Q4_K: 128 rows by 32 tokens, half the activation loads per MMA of the 64x64 tile, 10..20 %
    // faster on every board where its tiles fill the SMs.
    if (!emulated && tokens >= kRmmaMidMinTokens && tall_tile_format(problem.qtype) &&
        waves(128, 32) >= kSpeedFill) {
        return rmma_index(4, 1, 2, 4);
    }
    // Four row warps of 64 rows and the widest token tile of under twice the call's tokens that
    // fills the SMs. Where the BF16 MMA is emulated the token tiles stop at 32, and that path
    // takes this regime only past the stream band.
    const int nt_cap = emulated ? 4 : 8;
    if (tokens > kStreamMaxTokens ||
        (!emulated && tokens >= kRmmaMidMinTokens && waves(64, 8 * nt_cap) >= kSpeedFill)) {
        for (int nt = nt_cap; nt > 1; nt /= 2) {
            if (8 * nt < 2 * std::max(8, static_cast<int>(tokens)) &&
                waves(64, 8 * nt) >= kSpeedFill) {
                return rmma_index(4, 1, 1, nt);
            }
        }
        return rmma_index(4, 1, 1, 1);
    }
    // The stream regime: one 16-row M tile per CTA with K over the warps, as in decode, and as
    // many 8-token tiles (4, 2, 1) as the call's tokens use while the grid fills the SMs. With one
    // token tile sixteen K warps where the call has at most one CTA per SM or a long K (from 72
    // K steps), else eight.
    const int pad = static_cast<int>(std::bit_ceil(static_cast<unsigned>((tokens + 7) / 8)));
    if (pad >= 4 && waves(16, 32) >= kSpeedFill) {
        return heavy ? rmma_index(1, 4, 1, 4) : rmma_index(1, 8, 1, 4);
    }
    if (pad >= 2 && waves(16, 16) >= kSpeedFill) return rmma_index(1, 4, 1, 2);
    const std::int64_t step  = heavy ? 128 : 32;
    const std::int64_t steps = (k + step - 1) / step;
    const std::int64_t ctas  = (n + 15) / 16;
    return ctas <= sm_count() || 2 * steps >= 9 * 16 ? rmma_index(1, 16, 1, 1)
                                                     : rmma_index(1, 8, 1, 1);
}

int heuristic_w4a8(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    constexpr int k32x128 = 0, k128x128 = 2, k128x64 = 3, k64x64 = 4, k128x64ws = 5;
    // 17..32 tokens (verify of a 16-token draft, short prompts): 32 tokens by 128 rows wastes less of
    // the tile than 64x64 and reads each weight tile once per 128 rows. (RTX 5080,
    // 7 GGUF formats at 27B/31B/35B shapes, time over the best of seven routes, geomean per format)
    // at T = 17: 32x128 1.05-1.45 against 64x64 1.14-1.57, better for every format at 17, 20, 24, 32.
    if (tokens <= 32) return k32x128;
    if (tokens <= 64) return k64x64;
    // From 65 tokens a 128-token tile reads each weight tile once where 64x64 reads it twice, but
    // it halves the grid: it pays while that grid still gives half the SMs a CTA (rows split as
    // the kernel does, SwiGLU's gate and up pairs halving a tile's output rows). The dense tuner's
    // prefill bands (perplexity at context 2048 on Qwen3.8-27B shapes, time of the old choice
    // over the best), 64x64 against 128x64 below that grid: RTX 4080 SUPER (ad103:80sm) at 1024
    // rows (16 CTAs at T = 128, 32 at 256) 1.33-1.37 for Q4_K, Q5_K and IQ4_XS; RTX 5090
    // (gb202:170sm) at 5120 rows and T = 128 (80 CTAs) 1.15-1.36 for Q4_K, Q5_K and IQ4_XS, at 1024
    // rows and T = 128..512 (16..64 CTAs) 1.14-1.35 for Q4_K and Q5_K, 1.07 for Q6_K; RTX 5060 Ti
    // (gb206:36sm) at 1024 rows and T = 128 1.19-1.50 for Q4_K, Q5_K and IQ4_XS.
    const std::int64_t rows_per_cta = problem.epilogue == Epilogue::SwiGlu ? 32 : 64;
    const std::int64_t wide_ctas =
        ((problem.n + rows_per_cta - 1) / rows_per_cta) * ((tokens + 127) / 128);
    if (2 * wide_ctas < sm_count()) return k64x64;
    bool wide_format = false;
    switch (problem.qtype) {
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ2_XS:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
        wide_format = true;
        break;
    default:
        break;
    }
    // 65..127 tokens: the formats that take a 128-token tile from 128 tokens take 128x64 already,
    // and so do IQ2_XXS and Q3_K, whose wider tiles were not measured in the product. The same
    // 4080 SUPER: 128x64 against 64x64 1.10-1.56 on IQ2_*, IQ3_*, IQ4_XS from 5120 rows (80+ CTAs),
    // Q3_K 1.09, IQ2_XXS 1.10-1.14, Q4_K 1.09 at 17408 rows (at 5120 rows 64x64 was as fast). The
    // 5090: Q4_K 1.05-1.09, Q5_K 1.07-1.17, Q6_K 1.17-1.20 from 6144 rows (96+ CTAs; 5120 rows
    // stay under its grid bound and on 64x64, the tuner's choice there too). IQ4_XS takes 128x64
    // too although the 5090's tuner kept 64x64 for it (the 4080 SUPER's gain is the measured one;
    // the tuner corrects the other). The 5060 Ti: IQ2_S, IQ2_XS, IQ3_S, IQ3_XXS 1.08-1.29 at every
    // row count. Q4_0 and the rest (64x64 from 128 tokens too) keep 64x64. In the product
    // (ninferno-perplexity at context 100, tuner off on both arms, A B A B A B, perplexity
    // bit-identical): 4080 SUPER Qwen3.8-27B IQ3 412.9 -> 472.4 tok/s, 5090 Q4_K_XL 738 -> 757.
    if (tokens < kW4a8WideMinTokens) {
        return wide_format || problem.qtype == QType::GGML_IQ2_XXS ||
                       problem.qtype == QType::GGML_Q3_K
                   ? k128x64
                   : k64x64;
    }
    if (w4a8_ws_wins(problem, weight)) return k128x64ws;
    if (w4a8_wide_q4k(problem, tokens)) return k128x128;
    return wide_format ? k128x64 : k64x64;
}

int heuristic_r8(const Problem& p, std::int32_t tokens) {
    if (tokens == 1) {
        if (const auto tuned = tuned_r8(p)) return *tuned;
    }
    return rules_r8(p, tokens);
}

int k_multiple(WeightCodec codec) {
    switch (codec) {
    case WeightCodec::GgmlQ40:
    case WeightCodec::GgmlQ41:
    case WeightCodec::GgmlQ50:
    case WeightCodec::GgmlQ51:
    case WeightCodec::GgmlIq4Nl:
        return 32;
    default:
        return run_values(codec);
    }
}

int i8_bits(QType q) { return q == QType::Q5_G64_FP16 ? 5 : q == QType::Q6_G64_FP16 ? 6 : 4; }

std::optional<ggml_mmq::Type> mmq_type(const Weight& w) {
    switch (w.qtype) {
    case QType::GGML_Q4_K:
        return ggml_mmq::Type::Q4_K;
    case QType::GGML_Q5_K:
        return ggml_mmq::Type::Q5_K;
    case QType::GGML_Q6_K:
        return ggml_mmq::Type::Q6_K;
    case QType::GGML_IQ4_XS:
        return ggml_mmq::Type::IQ4_XS;
    case QType::GGML_IQ3_XXS:
        return ggml_mmq::Type::IQ3_XXS;
    case QType::GGML_IQ3_S:
        return ggml_mmq::Type::IQ3_S;
    case QType::GGML_Q4_0:
        return ggml_mmq::Type::Q4_0;
    case QType::GGML_Q4_1:
        return ggml_mmq::Type::Q4_1;
    case QType::GGML_Q5_0:
        return ggml_mmq::Type::Q5_0;
    case QType::GGML_Q5_1:
        return ggml_mmq::Type::Q5_1;
    case QType::GGML_IQ4_NL:
        return ggml_mmq::Type::IQ4_NL;
    case QType::GGML_Q2_K:
        return ggml_mmq::Type::Q2_K;
    case QType::GGML_Q3_K:
        return ggml_mmq::Type::Q3_K;
    case QType::GGML_IQ2_XXS:
        return ggml_mmq::Type::IQ2_XXS;
    case QType::GGML_IQ2_XS:
        return ggml_mmq::Type::IQ2_XS;
    case QType::GGML_IQ2_S:
        return ggml_mmq::Type::IQ2_S;
    case QType::GGML_IQ1_S:
        return ggml_mmq::Type::IQ1_S;
    case QType::GGML_IQ1_M:
        return ggml_mmq::Type::IQ1_M;
    default:
        return std::nullopt;
    }
}

bool mmq_admits(const Weight& w, LinearPolicy policy) {
    const auto type = mmq_type(w);
    if (!type || !ggml_mmq::matrix_supported(*type) || !allows_a8(policy) ||
        (w.k % ggml_mmq::kKMultiple) != 0) {
        return false;
    }
    return ggml_layout_admits(w) && ggml_mmq::device_supported();
}

bool mmq_prefill_wins(const Problem& p, const Weight& w) {
    return mmq_prefill_device() && ggml_mmq_format(p.qtype) &&
           (p.n % ggml_mmq::kRowMultiple) == 0 && mmq_admits(w, p.policy);
}

bool vec_admits(const Weight& w, LinearPolicy policy) {
    const auto type = mmq_type(w);
    if (!type || !ggml_mmq::vec_supported(*type) || !allows_a8(policy) ||
        (w.k % static_cast<std::int32_t>(ggml_block_values(w.qtype))) != 0) {
        return false;
    }
    return ggml_layout_admits(w);
}

bool vec_call(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= 1 && tokens <= vec_max_tokens() && vec_admits(w, policy);
}

std::size_t vec_workspace_bytes(std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(ggml_mmq::vec_activation_bytes(k, tokens), 256);
    return layout.peak_bytes(1);
}

ggml_mmq::Planes ggml_rows_at(const GemmArgs& args, QType qtype, int bpr, int row) {
    const auto bytes  = mmq_plane_bytes(qtype);
    const auto blocks = static_cast<std::int64_t>(row) * bpr;
    ggml_mmq::Planes p;
    p.codes = args.codes + blocks * bytes.codes;
    if (bytes.high > 0) p.high = args.high + blocks * bytes.high;
    if (bytes.scales > 0) p.scales = args.scales + blocks * bytes.scales;
    return p;
}

bool mmq_call(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens) {
    return tokens >= kMmqMinTokens && tokens <= kMmqMaxTokens &&
           (rows % ggml_mmq::kRowMultiple) == 0 && mmq_admits(w, policy);
}

std::size_t mmq_workspace_bytes(std::int32_t k, std::int32_t plane_rows, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(ggml_mmq::activation_bytes(k, tokens), 256);
    (void)layout.alloc_bytes(static_cast<std::size_t>(plane_rows) * tokens * sizeof(float), 256);
    (void)layout.alloc_bytes(ggml_mmq::fixup_bound(tokens), 256);
    return layout.peak_bytes(1);
}

void clear_seen_problems() {
    ThreadSeen& t = thread_seen();
    t.collecting  = true;
    t.seen.clear();
}

std::vector<SeenProblem> seen_problems() {
    const ThreadSeen& t = thread_seen();
    std::vector<SeenProblem> out;
    out.reserve(t.seen.size());
    for (const auto& [qtype, policy, epilogue, n, k, parent_rows, layout] : t.seen) {
        out.push_back({Problem{static_cast<QType>(qtype), static_cast<LinearPolicy>(policy), n, k,
                               static_cast<Epilogue>(epilogue)},
                       parent_rows, static_cast<QuantLayout>(layout)});
    }
    return out;
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

// The DRAM bytes the board's bandwidth asks of each SM per cycle: the facts' bandwidth over the
// SMs times the reference clock (rmma_latency_pays reads it); 0 when a fact is unknown.
double dram_bytes_per_sm_cycle() {
    static PerDevice<double> bytes;
    // A device whose class cannot be read throws, and PerDevice retries on the next call (the
    // launch takes the plain kernel meanwhile): caching 0 would turn the variant off for good.
    return bytes.get([](int) {
        const auto& d = device_class();
        const auto& f = d.facts;
        if (!f.has_roofline() || d.sms <= 0) return 0.0;
        return f.dram_bytes_per_s() / (d.sms * f.reference_clock_hz());
    });
}

bool rmma_variant_occupancy_rule(std::int64_t grid_ctas, int sms, int variant_ctas,
                                 int plain_ctas) {
    if (variant_ctas <= 0 || plain_ctas <= 0 || sms <= 0) return false;
    if (variant_ctas >= plain_ctas) return true;
    const auto waves = [&](int resident) {
        const std::int64_t wave = static_cast<std::int64_t>(resident) * sms;
        return (grid_ctas + wave - 1) / wave;
    };
    return waves(variant_ctas) <= waves(plain_ctas);
}

bool rmma_wide_rule(double bytes_per_sm_cycle, int compute_capability, bool occupancy,
                    std::size_t wide_local_bytes, std::size_t plain_local_bytes) {
    return bytes_per_sm_cycle > 0.0 && bytes_per_sm_cycle < kRmmaWideBytes &&
           compute_capability >= 75 && occupancy &&
           wide_local_bytes <= plain_local_bytes + kRmmaWideSpillSlack;
}

bool operator==(Route a, Route b) { return a.family == b.family && a.schedule == b.schedule; }

int rmma_schedule_count() { return static_cast<int>(kRmmaSchedules.size()); }
RmmaScheduleInfo rmma_schedule_info(int index) {
    return kRmmaSchedules.at(static_cast<std::size_t>(index));
}
bool rmma_schedule_launchable(int index) {
#ifdef NINFERNO_WITHOUT_NVFP4
    return rmma_schedule_info(index).cluster == 1;
#else
    return index >= 0 && index < rmma_schedule_count();
#endif
}
int w4a4_schedule_count() {
#ifdef NINFERNO_WITHOUT_NVFP4
    // Below sm_120a the W4A4 kernel TU is not compiled (kind::mxf4nvf4 is Blackwell-only), so no
    // W4A4 route is offered, admitted or run; NVFP4 weights take the register-MMA A16 path. The
    // names still parse (parse_route), so one route table serves every build.
    return 0;
#else
    return static_cast<int>(kW4a4Schedules.size());
#endif
}

int w4a4_schedule_names() { return static_cast<int>(kW4a4Schedules.size()); }
W4a4ScheduleInfo w4a4_schedule_info(int index) {
    return kW4a4Schedules.at(static_cast<std::size_t>(index));
}
int w4a8_schedule_count() { return static_cast<int>(kW4a8Schedules.size()); }
W4a8ScheduleInfo w4a8_schedule_info(int index) {
    return kW4a8Schedules.at(static_cast<std::size_t>(index));
}

bool w4a8_fits(int schedule, QType qtype) {
    const auto s = w4a8_schedule_info(schedule);
    // Two buffers of int8 codes and two floats per 32-value block for block_m + block_n rows, and
    // Q2_K's two mins per block and weight row (W4a8BufferOf::kSmem); the producer ring is sized
    // for 99 KiB.
    if (s.producers > 0) return kMaxSharedBytesPerBlock >= 99 * 1024;
    const int half_min = qtype == QType::GGML_Q2_K ? s.block_n * 32 : 0;
    return 2 * ((s.block_m + s.block_n) * (128 + 32) + half_min) <=
           static_cast<int>(kMaxSharedBytesPerBlock);
}
int r8_schedule_count() { return static_cast<int>(kR8Schedules.size()); }
R8ScheduleInfo r8_schedule_info(int index) {
    return kR8Schedules.at(static_cast<std::size_t>(index));
}

bool r8_supports(QType qtype) { return admits(format_traits(qtype).r8, kNativeNvfp4); }

std::int32_t r8_k_values(QType qtype) { return format_traits(qtype).r8_k_values; }

bool r8_stage_grid() {
    try {
        return !device_class().features.mma_bf16_native;
    } catch (const std::exception&) { return false; }
}

bool r8_fits(QType qtype, std::int32_t k, std::int32_t tokens) {
    if (tokens <= 0 || tokens > kR8MaxTokens || k <= 0) return false;
    std::int64_t slots = 1;
    while (slots < tokens) slots *= 2;
    // A codec with a min also holds each block's sum (r8_smem_bytes).
    const bool sums = qtype == QType::GGML_Q4_1 || qtype == QType::GGML_Q5_1 ||
                      qtype == QType::GGML_Q4_K || qtype == QType::GGML_Q5_K;
    // A codebook format's grid follows it where r8_kernel stages it (r8_stage_grid).
    std::int64_t grid = 0;
    if (r8_stage_grid()) switch (qtype) {
        case QType::GGML_IQ3_XXS:
            grid = 1024;
            break;
        case QType::GGML_IQ3_S:
        case QType::GGML_IQ2_XXS:
            grid = 2048;
            break;
        case QType::GGML_IQ2_XS:
            grid = 4096;
            break;
        case QType::GGML_IQ2_S:
        case QType::GGML_IQ1_S:
        case QType::GGML_IQ1_M:
            grid = 8192;
            break;
        default:
            break;
        }
    if (grid > 0) grid += 4; // r8_grid_pad (generic_r8.cuh): the grid starts 8-byte aligned
    return slots * (static_cast<std::int64_t>(k) / 32) * (sums ? 40 : 36) + grid <=
           kStaticSharedBytesPerBlock;
}

bool r8_verify_supports(QType qtype) { return format_traits(qtype).r8_verify; }

std::int32_t r8_verify_widest(QType qtype) {
    return qtype == QType::Q8_G32_FP16 ? kR8VerifyTokens : 4;
}

bool r8_verify_compiled(QType qtype) {
    return r8_supports(qtype) && format_traits(qtype).r8_prequant;
}

// Q4_0 at the verify widths (T = 2..4): R8 pays on a large matrix and loses on a small one
// (A B B A, C = 4: Gemma 4 31B QAT, every dense matrix >= 44M values, RTX 4090 +15.7 %,
// RTX 3090 +26 %; Gemma 4 26B QAT, at most 12M, -0.7 % and -2.7 %, its MTP verify -1.1 %).
inline constexpr std::int64_t kR8VerifyQ40MinValues = 24ll << 20;

bool r8_verify_takes_shape(QType qtype, std::int64_t n, std::int64_t k) {
    if (qtype != QType::GGML_Q4_0 || n * k >= kR8VerifyQ40MinValues) {
        return r8_verify_takes(qtype);
    }
    // A small Q4_0 matrix keeps the A16 register MMA where it is native (Turing: R8, as for
    // every compiled format).
    try {
        return !device_class().features.mma_bf16_native && r8_verify_compiled(qtype);
    } catch (const std::exception&) { return false; }
}

namespace {
// Whether the device's BF16 MMA is native (sm_80 and up); a device that cannot be queried
// counts as native, the routes that assume nothing about it.
bool bf16_mma_native() {
    try {
        return device_class().features.mma_bf16_native;
    } catch (const std::exception&) { return true; }
}
} // namespace

bool r8_verify_takes(QType qtype) {
    if (r8_verify_supports(qtype)) return true;
    try {
        return !device_class().features.mma_bf16_native && r8_verify_compiled(qtype);
    } catch (const std::exception&) { return false; }
}

// K for the W4A8 tiles: whole 128-value tiles, or for a format of 32-value blocks any whole number
// of blocks, the last tile's blocks past K zero (weights not loaded, activation quantized as
// zeros). Gemma 4 26B's shared MLP down (K 2112) ran the BF16 register MMA before: RTX 5090 nsys,
// chunk 1024, 114 us a call against 102 us for the twice-as-large gate/up on W4A8.
bool w4a8_k_admits(QType qtype, std::int32_t k) {
    if (k <= 0) return false;
    if (k % 128 == 0) return true;
    const std::int32_t tail = format_traits(qtype).w4a8_tail_k;
    return tail != 0 && k % tail == 0;
}

bool w4a8_supports(QType qtype) { return admits(format_traits(qtype).w4a8, kNativeNvfp4); }
int nvfp4_tiled_schedule_count() { return static_cast<int>(kNvfp4TiledSchedules.size()); }
Nvfp4TiledScheduleInfo nvfp4_tiled_schedule_info(int index) {
    return kNvfp4TiledSchedules.at(static_cast<std::size_t>(index));
}
int tiled_schedule_count() { return static_cast<int>(kTiledSchedules.size()); }
TiledScheduleInfo tiled_schedule_info(int index) {
    return kTiledSchedules.at(static_cast<std::size_t>(index));
}
int staged_schedule_count() { return static_cast<int>(kStagedSchedules.size()); }
// The build admits a tile its architecture holds; an sm_80 build run on a later 99 KiB part
// (forward-compatible SASS) must still turn the 144-160 KiB tiles away, so the device decides too.
bool staged_device_fits(int schedule) {
    const auto s     = kStagedSchedules.at(static_cast<std::size_t>(schedule));
    const auto bytes = static_cast<std::size_t>(s.rows + s.tokens) * 64 * 2 * s.stages;
    std::size_t limit = 0;
    try {
        limit = device_class().smem_per_block_optin;
    } catch (const std::exception&) {
        return true; // no device (a CPU-only tool): staged_fits holds the build ceiling
    }
    return limit == 0 || bytes <= limit;
}
StagedScheduleInfo staged_schedule_info(int index) {
    return kStagedSchedules.at(static_cast<std::size_t>(index));
}

std::string route_name(Route route) {
    switch (route.family) {
    case Family::Scalar:
        return "scalar";
    case Family::RegisterMma: {
        const auto s = rmma_schedule_info(route.schedule);
        return "rmma.m" + std::to_string(s.m_tiles) + ".n" + std::to_string(s.n_tiles) + ".w" +
               std::to_string(s.warps_m) + "x" + std::to_string(s.warps_k) +
               (s.cluster > 1 ? ".c" + std::to_string(s.cluster) : std::string());
    }
    case Family::W4a4Mma: {
        const auto s = w4a4_schedule_info(route.schedule);
        return "w4a4." + std::to_string(s.block_m) + "x" + std::to_string(s.block_n);
    }
    case Family::TiledMma: {
        const auto s = tiled_schedule_info(route.schedule);
        return "tiled." + std::to_string(s.block_m) + "x" + std::to_string(s.block_n);
    }
    case Family::W4a8Mma: {
        const auto s = w4a8_schedule_info(route.schedule);
        return (s.producers > 0 ? "w4a8ws." : "w4a8.") + std::to_string(s.block_m) + "x" +
               std::to_string(s.block_n) +
               (s.producers > 0 ? ".p" + std::to_string(s.producers) : std::string());
    }
    case Family::StagedMma: {
        const auto s = staged_schedule_info(route.schedule);
        return "staged." + std::to_string(s.rows) + "x" + std::to_string(s.tokens) + ".w" +
               std::to_string(s.warps_m) + "x" + std::to_string(s.warps_n) + ".s" +
               std::to_string(s.stages);
    }
    case Family::Nvfp4TiledMma: {
        const auto s = nvfp4_tiled_schedule_info(route.schedule);
        return "nvfp4t.c" + std::to_string(s.classes) + "." + std::to_string(s.block_m) + "x" +
               std::to_string(s.block_n);
    }
    case Family::R8Mma: {
        const auto s = r8_schedule_info(route.schedule);
        return "r8.w" + std::to_string(s.warps) + "r" + std::to_string(s.rows);
    }
    case Family::GgmlMmq:
        return "ggml_mmq";
    case Family::GgmlVec:
        return "ggml_vec";
    case Family::I8Mma: {
        const auto s = i8_schedule_info(route.schedule);
        return "i8." + std::to_string(s.block_m) + "x" + std::to_string(s.block_n);
    }
    }
    return "unknown";
}

std::string_view family_name(Family family) noexcept {
    switch (family) {
    case Family::RegisterMma: return "rmma";
    case Family::Scalar: return "scalar";
    case Family::W4a4Mma: return "w4a4";
    case Family::TiledMma: return "tiled";
    case Family::W4a8Mma: return "w4a8";
    case Family::StagedMma: return "staged";
    case Family::Nvfp4TiledMma: return "nvfp4t";
    case Family::R8Mma: return "r8";
    case Family::GgmlMmq: return "ggml_mmq";
    case Family::GgmlVec: return "ggml_vec";
    case Family::I8Mma: return "i8";
    }
    return "unknown";
}

std::optional<Route> parse_route(std::string_view name) {
    if (name == "scalar") return Route{Family::Scalar, 0};
    if (name == "ggml_mmq") return Route{Family::GgmlMmq, 0};
    if (name == "ggml_vec") return Route{Family::GgmlVec, 0};
    for (int i = 0; i < rmma_schedule_count(); ++i) {
        const Route r{Family::RegisterMma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    // Every W4A4 name, compiled or not: a shipped Blackwell table names them, and route_admits
    // refuses them on a build without the kernels.
    for (int i = 0; i < w4a4_schedule_names(); ++i) {
        const Route r{Family::W4a4Mma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < tiled_schedule_count(); ++i) {
        const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < w4a8_schedule_count(); ++i) {
        const Route r{Family::W4a8Mma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < staged_schedule_count(); ++i) {
        const Route r{Family::StagedMma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < nvfp4_tiled_schedule_count(); ++i) {
        const Route r{Family::Nvfp4TiledMma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < r8_schedule_count(); ++i) {
        const Route r{Family::R8Mma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    for (int i = 0; i < i8_schedule_count(); ++i) {
        const Route r{Family::I8Mma, static_cast<std::uint8_t>(i)};
        if (route_name(r) == name) return r;
    }
    return std::nullopt;
}

WeightCodec codec_of(const Weight& weight) {
    switch (weight.qtype) {
    case QType::BF16:
        return WeightCodec::Bf16;
    case QType::FP8_E4M3FN_ROW_BF16:
        return WeightCodec::Fp8;
    case QType::NVFP4:
        return WeightCodec::Nvfp4;
    case QType::Q4_G64_FP16:
        return WeightCodec::Q4;
    case QType::Q5_G64_FP16:
        return WeightCodec::Q5;
    case QType::Q6_G64_FP16:
        return WeightCodec::Q6;
    case QType::Q8_G32_FP16:
        return WeightCodec::Q8;
    case QType::MXFP4:
        return WeightCodec::Mxfp4;
    case QType::FP8_E4M3FN_ROW_F32:
        return WeightCodec::Fp8RowF32;
    case QType::FP8_E4M3FN_B128_F32:
        return WeightCodec::Fp8B128;
    case QType::U4_G128_ZP_FP16:
        return WeightCodec::U4Zp;
    case QType::U4_G32_ZP_FP16:
        return WeightCodec::U4ZpG32;
    case QType::U4_G64_ZP_FP16:
        return WeightCodec::U4ZpG64;
    case QType::Q3_G128_FP16:
        return WeightCodec::Q3G128;
    case QType::Q3_G64_FP16:
        return WeightCodec::Q3G64;
    case QType::GGML_Q4_K:
        return WeightCodec::GgmlQ4K;
    case QType::GGML_Q5_K:
        return WeightCodec::GgmlQ5K;
    case QType::GGML_Q6_K:
        return WeightCodec::GgmlQ6K;
    case QType::GGML_Q3_K:
        return WeightCodec::GgmlQ3K;
    case QType::GGML_Q2_K:
        return WeightCodec::GgmlQ2K;
    case QType::GGML_IQ4_XS:
        return WeightCodec::GgmlIq4Xs;
    case QType::GGML_IQ3_XXS:
        return WeightCodec::GgmlIq3Xxs;
    case QType::GGML_IQ3_S:
        return WeightCodec::GgmlIq3S;
    case QType::GGML_Q4_0:
        return WeightCodec::GgmlQ40;
    case QType::GGML_Q4_1:
        return WeightCodec::GgmlQ41;
    case QType::GGML_Q5_0:
        return WeightCodec::GgmlQ50;
    case QType::GGML_Q5_1:
        return WeightCodec::GgmlQ51;
    case QType::GGML_IQ4_NL:
        return WeightCodec::GgmlIq4Nl;
    case QType::GGML_IQ2_XXS:
        return WeightCodec::GgmlIq2Xxs;
    case QType::GGML_IQ2_XS:
        return WeightCodec::GgmlIq2Xs;
    case QType::GGML_IQ2_S:
        return WeightCodec::GgmlIq2S;
    case QType::GGML_IQ1_S:
        return WeightCodec::GgmlIq1S;
    case QType::GGML_IQ1_M:
        return WeightCodec::GgmlIq1M;
    case QType::GGML_TQ1_0:
        return WeightCodec::GgmlTq10;
    case QType::GGML_TQ2_0:
        return WeightCodec::GgmlTq20;
    case QType::GGML_Q1_0:
        return WeightCodec::GgmlQ10;
    case QType::GGML_Q2_0:
        return WeightCodec::GgmlQ20;
    default:
        throw std::invalid_argument("generic linear: unsupported weight format");
    }
}

void refuse_frag(const Weight& w, const char* op) {
    if (w.layout != QuantLayout::GgmlFrag) return;
    throw std::invalid_argument(
        std::string(op) +
        ": the weight is a ggml_frag_v1 routed-expert bank, which only the grouped MoE kernel "
        "reads; a projection needs its published layout (ninferno-convert --relayout published)");
}

namespace {
// A registered shape's own kernels of these formats compute BF16 activations only (the int8
// tiers and R8 are the generic path's): a call such a gate keeps on its table is A16. The NVFP4
// and FP8 shapes choose their quantized kernels themselves (their dispatch notes them).
bool owned_tables_a16(QType qtype) {
    return qtype != QType::NVFP4 && qtype != QType::FP8_E4M3FN_ROW_BF16;
}
} // namespace

bool r8_takes_block(const Weight& weight) {
    return supports(weight) && r8_call(weight, LinearPolicy::AllowA8, kR8VerifyTokens);
}

bool owned_decode_r8(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    const bool r8 =
        tokens <= kR8DecodeTokens && supports(weight) && r8_call(weight, policy, tokens);
    // --decode-route: R8 keeps the rule (a registered shape hands R8 only the decode calls its
    // workspace query covers; the generic path notes the call); any other pin keeps the call on
    // the shape's own kernel, which reads BF16 activations.
    const DecodeRoute pin = route_pins().decode;
    if (pin == DecodeRoute::Auto || pin == DecodeRoute::R8 || tokens < 1 ||
        tokens > kDecodePinMaxTokens) {
        return r8;
    }
    if (owned_tables_a16(weight.qtype)) {
        note_decode_pin(pin == DecodeRoute::A16,
                        family_bit(DecodeRoute::A16) | (r8 ? family_bit(DecodeRoute::R8) : 0U));
    }
    return false;
}

bool owned_prefill_w4a8(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    // Without tensor mma.sync the tiled route owns such a call, as in select_route.
    const bool w4a8 = tokens >= kQ8PrefillA8Tokens && prefill_routes_active() &&
                      weight.qtype == QType::NVFP4 && supports(weight) &&
                      allows_w4a8(weight, policy) && !volta_tiled_route(weight, policy, tokens);
    // --prefill-route a16: the shape's own plan keeps the call on its A16 route.
    if (w4a8 && route_pins().prefill == PrefillRoute::A16) {
        note_prefill_pin(true, family_bit(PrefillRoute::A16) | family_bit(PrefillRoute::W4a8));
        return false;
    }
    return w4a8;
}

bool nvfp4_prefill_a8_policy(QType qtype, LinearPolicy policy) {
    return qtype == QType::NVFP4 && allows_a8(policy) &&
           !(w4a4_schedule_count() > 0 && allows_a4(policy));
}

namespace {
// Q4/Q5/Q6 G64 weights take int8 activations at prefill wherever the route admits W4A8 (the
// artifact's A16Only uses included) on the cards where they pay (qx_prefill_a8_pays);
// --activations exact keeps A16 (invariant routes). Qwen3.8-27B book 8K KLD 0.0047, top-1 97.1 %.
// Int8 activations pay where the card's int8 mma runs well ahead of its bf16 mma with fp32
// accumulation: the GeForce and Ada/Ampere inference parts (half-rate fp32-accumulate HMMA, a
// quarter of int8). Qwen3.8-27B Q4/Q5 G64 24K prefill, MTP 3: RTX 4090
// +18.7 %, L4 +14 %, A10 +7.8 %. The HBM training parts run fp32-accumulate HMMA at full rate,
// and the activation quantization pass then costs more than the faster mma wins: A100 -3.5 %.
bool qx_prefill_a8_pays() {
    try {
        const auto& d = device_class();
        // Not on a chip whose mma.sync is emulated (mma_slow: TU116 / TU117), as for ggml's mmq.
        // A generation without a row of its own has no measured kind: A16 (#300).
        const auto full_rate = fp32acc_full_rate_of(d.major, d.minor);
        return d.features.mma_tensor && !d.features.mma_slow && full_rate && !*full_rate;
    } catch (const std::exception&) {
        return false;
    }
}

bool qx_format(QType qtype) { return format_traits(qtype).qx_g64; }

// Whether the Q4/Q5/Q6 G64 weights take the int8 tiers at prefill: where int8 activations pay
// (qx_prefill_a8_pays, `pays`), or as a route pin asks (--prefill-route w4a8 or i8: on every card;
// a16: on none).
bool qx_prefill_a8_takes(bool pays) {
    switch (route_pins().prefill) {
    case PrefillRoute::W4a8:
    case PrefillRoute::I8:
        return true;
    case PrefillRoute::A16:
        return false;
    default:
        return pays;
    }
}
} // namespace

LinearPolicy prefill_a8_policy(QType qtype, LinearPolicy policy) {
    if (policy == LinearPolicy::A16Only && qx_format(qtype) &&
        qx_prefill_a8_takes(qx_prefill_a8_pays()) && !invariant_routes_active()) {
        return LinearPolicy::AllowA8;
    }
    return policy;
}

std::size_t prefill_a8_workspace_bytes(const Problem& problem, std::int32_t min_tokens,
                                      std::int32_t max_tokens) {
    // NVFP4 under a policy with A8 and without W4A4 (owned_prefill_w4a8): the W4A8 planes of its
    // calls of 17 tokens or more, which the shape's own plan does not run.
    if (nvfp4_prefill_a8_policy(problem.qtype, problem.policy)) {
        if (max_tokens < 17) return 0;
        return workspace_capacity_bytes(problem, nullptr, std::max(min_tokens, 17), max_tokens);
    }
    const LinearPolicy a8_policy = prefill_a8_policy(problem.qtype, problem.policy);
    if (!prefill_a8_format(problem.qtype) || !allows_a8(a8_policy) ||
        max_tokens < kQ8PrefillA8Tokens) {
        return 0;
    }
    Problem a8 = problem;
    a8.policy  = a8_policy;
    return workspace_capacity_bytes(a8, nullptr, min_tokens, max_tokens);
}

bool prefill_a8_format(QType qtype) {
    static PerDevice<bool> pays;
    const bool qx = qx_prefill_a8_takes(pays.get([](int) { return qx_prefill_a8_pays(); }));
    switch (qtype) {
    case QType::Q8_G32_FP16:
        // --prefill-route a16 keeps Q8 on its shape tables' A16 kernels too.
        return route_pins().prefill != PrefillRoute::A16;
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
        return qx;
    default:
        return false;
    }
}

bool q8_prefill_a8(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    const bool band = tokens >= kQ8PrefillA8Tokens && prefill_routes_active();
    bool tier = band && prefill_a8_format(weight.qtype) && allows_a8(policy) && supports(weight);
    // --prefill-route w4a8 / i8: the int8 tier of that family only, where it reads the weight;
    // a16 has no tier (prefill_a8_format). A call kept on its shape table runs A16 there.
    const PrefillRoute pin = route_pins().prefill;
    if (pin == PrefillRoute::W4a8) tier = tier && allows_w4a8(weight, policy);
    if (pin == PrefillRoute::I8) tier = tier && allows_i8(weight, policy);
    if (band && !tier && pin != PrefillRoute::Auto && owned_tables_a16(weight.qtype)) {
        const bool a8 = allows_a8(policy) && supports(weight);
        note_prefill_pin(
            pin == PrefillRoute::A16,
            family_bit(PrefillRoute::A16) |
                (a8 && allows_w4a8(weight, policy) ? family_bit(PrefillRoute::W4a8) : 0U) |
                (a8 && allows_i8(weight, policy) ? family_bit(PrefillRoute::I8) : 0U));
    }
    return tier;
}

bool q8_prefill_a8_fused(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    return q8_prefill_a8(weight, policy, tokens);
}

LinearPolicy fixed_parent_prefill_a8_policy(QType qtype) {
    return prefill_a8_policy(qtype, LinearPolicy::A16Only);
}

bool fixed_pair_prefill_a8(const Weight& first, const Weight& second, std::int32_t tokens) {
    return q8_prefill_a8(first, fixed_parent_prefill_a8_policy(first.qtype), tokens) &&
           q8_prefill_a8(second, fixed_parent_prefill_a8_policy(second.qtype), tokens);
}

std::size_t fixed_parents_prefill_a8_workspace_bytes(std::initializer_list<FixedParent> parents,
                                                     std::int32_t k, std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    std::size_t bytes = 0;
    for (const FixedParent& parent : parents) {
        const LinearPolicy policy = fixed_parent_prefill_a8_policy(parent.qtype);
        if (!prefill_a8_format(parent.qtype) || !allows_a8(policy) ||
            max_tokens < kQ8PrefillA8Tokens) {
            return 0;
        }
        bytes = std::max(
            bytes, workspace_capacity_bytes({parent.qtype, policy, parent.rows, k, Epilogue::Store},
                                            nullptr, std::max(min_tokens, kQ8PrefillA8Tokens), max_tokens));
    }
    return bytes;
}

bool supports(const Weight& w) {
    if (w.n <= 0 || w.k <= 0 || w.qdata == nullptr) return false;
    switch (w.qtype) {
    case QType::BF16:
        return w.layout == QuantLayout::Contiguous;
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::FP8_E4M3FN_ROW_F32:
        return w.layout == QuantLayout::RowScale && w.scales != nullptr;
    case QType::FP8_E4M3FN_B128_F32:
        return w.layout == QuantLayout::BlockScale128F32 && w.scales != nullptr;
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
        return (w.layout == QuantLayout::RowGroupZp ||
                (w.layout == QuantLayout::RowGroupZpOrdered && w.k_order != nullptr)) &&
               w.scales != nullptr && w.qhigh != nullptr &&
               (w.k % static_cast<int>(u4_zp_group(w.qtype))) == 0;
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
        return w.layout == QuantLayout::GgmlBlocks && (w.k % 256) == 0;
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
    {
        const auto plane = ggml_plane_bytes(w.qtype);
        return w.layout == QuantLayout::GgmlPlanes && w.scales != nullptr &&
               (w.qhigh != nullptr || plane.high == 0) &&
               (w.k % static_cast<std::int32_t>(plane.values)) == 0;
    }
    case QType::NVFP4:
        // Any N: every generic route guards its rows, and the scale plane holds whole 128-row
        // tiles (a last tile N does not fill included).
        return w.layout == QuantLayout::BlockScaleK16M128x4 && w.scales != nullptr &&
               (w.k % 64) == 0 && std::isfinite(w.weight_scale_divisor) &&
               w.weight_scale_divisor > 0.0F;
    case QType::MXFP4:
        return w.layout == QuantLayout::BlockScaleK32E8M0 && w.scales != nullptr &&
               (w.k % 32) == 0;
    case QType::Q4_G64_FP16:
    case QType::Q8_G32_FP16:
        return w.layout == QuantLayout::RowSplit && w.scales != nullptr &&
               w.padded_shape[1] >= w.k && (w.padded_shape[1] % 128) == 0;
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
        return w.layout == QuantLayout::RowSplit && w.scales != nullptr && w.qhigh != nullptr &&
               w.padded_shape[1] >= w.k && (w.padded_shape[1] % 128) == 0;
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
        return w.layout == QuantLayout::RowSplit && w.scales != nullptr &&
               w.padded_shape[1] >= w.k && (w.padded_shape[1] % 128) == 0;
    default:
        return false;
    }
}

bool rmma_tile_compiled(WeightCodec codec, int m_tiles, int n_tiles, std::int32_t k) {
    // run_values: 32 for a decoder of 32-value blocks, 128 for the ggml 32-blocks that step four
    // blocks at a time; both read 32 values per decode (kRun == 32). So does the u4 G32/G64
    // quad-run decoder that launch_rmma_u4zp_g32/_g64 (and the MoE launches) take at K % 128 == 0.
    const bool quad_u4 = (codec == WeightCodec::U4ZpG32 || codec == WeightCodec::U4ZpG64) &&
                         k > 0 && (k % 128) == 0;
    return !((run_values(codec) >= 32 || quad_u4) && m_tiles * n_tiles >= 16);
}

DeviceFeatures route_requires(Route route) {
    DeviceFeatures f;
    switch (route.family) {
    case Family::Scalar:
        return f;
    case Family::RegisterMma:
        f.mma_bf16 = true;
        f.clusters = route.schedule < rmma_schedule_count() &&
                     rmma_schedule_info(route.schedule).cluster > 1;
        return f;
    case Family::W4a4Mma:
        f.mma_bf16     = true;
        f.mma_mxf4nvf4 = true;
        return f;
    case Family::Nvfp4TiledMma:
        // The E2M1 converter of the sm_120 family decodes its tiles.
        f.mma_bf16     = true;
        f.mma_mxf4nvf4 = true;
        return f;
    case Family::TiledMma:
    case Family::W4a8Mma: // int8 m16n8k32: part of mma_bf16 (core/device_class.h)
    case Family::StagedMma:
    case Family::R8Mma:
        f.mma_bf16 = true;
        return f;
    case Family::GgmlMmq: // sm_75 and up, held by mmq_admits
    case Family::GgmlVec: // dp4a
        return f;
    case Family::I8Mma: // m16n8k32 s8 x u8, ldmatrix, cp.async: sm_80 on
        f.mma_bf16        = true;
        f.mma_bf16_native = true;
        return f;
    }
    return f;
}

DeviceFeatures current_features() {
    try {
        return device_class().features;
    } catch (const std::exception&) {
        return {.mma_bf16        = true,
                .mma_bf16_native = true,
                .mma_tensor      = true,
                .mma_fp8         = true,
                .mma_f8f6f4      = true,
                .mma_mxf4nvf4    = true,
                .tma             = true,
                .pdl             = true,
                .clusters        = true};
    }
}

bool volta_cutlass_takes(std::int32_t tokens);
bool volta_cutlass_runs(bool only, std::int32_t tokens);

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

bool tiled_takes(WeightCodec codec) {
    return tiled_supports(codec) || (codec == WeightCodec::Q8 && !current_features().mma_tensor);
}

bool volta_cutlass_only(WeightCodec codec) {
    return !tiled_takes(codec) && !current_features().mma_tensor;
}

bool volta_tiled_format(QType qtype) {
    return !current_features().mma_tensor && volta_tiled_qtype(qtype);
}

std::string_view schedule_refusal_name(ScheduleRefusal why) noexcept {
    switch (why) {
    case ScheduleRefusal::None: return "none";
    case ScheduleRefusal::NotDeclared: return "not_declared";
    case ScheduleRefusal::NotCompiled: return "not_compiled";
    case ScheduleRefusal::Format: return "format";
    case ScheduleRefusal::Instruction: return "instruction";
    case ScheduleRefusal::Alignment: return "alignment";
    case ScheduleRefusal::Extent: return "extent";
    case ScheduleRefusal::CodecTile: return "codec_tile";
    case ScheduleRefusal::Policy: return "policy";
    case ScheduleRefusal::Resource: return "resource";
    }
    return "none";
}

std::string_view r8_call_refusal_name(R8CallRefusal why) noexcept {
    switch (why) {
    case R8CallRefusal::None: return "none";
    case R8CallRefusal::NotAdmitted: return "not_admitted";
    case R8CallRefusal::DecodeFit: return "decode_fit";
    case R8CallRefusal::VerifyWidth: return "verify_width";
    case R8CallRefusal::VerifyShape: return "verify_shape";
    }
    return "none";
}

R8CallRefusal r8_call_refusal(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    if (!allows_r8(w, policy)) return R8CallRefusal::NotAdmitted;
    if (tokens <= kR8DecodeTokens) {
        return r8_fits(w.qtype, w.k, tokens) ? R8CallRefusal::None : R8CallRefusal::DecodeFit;
    }
    if (tokens > r8_verify_widest(w.qtype)) return R8CallRefusal::VerifyWidth;
    return r8_verify_takes_shape(w.qtype, w.n, w.k) ? R8CallRefusal::None
                                                     : R8CallRefusal::VerifyShape;
}

ScheduleRefusal r8_refusal(int index, const Weight& w, LinearPolicy policy) {
    if (index < 0 || index >= r8_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!supports(w) || !r8_supports(w.qtype)) return ScheduleRefusal::Format;
    if (!allows_a8(policy)) return ScheduleRefusal::Policy;
    const Route route{Family::R8Mma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // A ggml block format's high plane off a 16-byte boundary (generic_refusal), or the codes.
    if ((w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) ||
        !aligned_to(w.qdata, 16)) {
        return ScheduleRefusal::Alignment;
    }
    if ((w.k % r8_k_values(w.qtype)) != 0) return ScheduleRefusal::Extent;
    return ScheduleRefusal::None;
}

ScheduleRefusal w4a8_refusal(int index, const Weight& w, LinearPolicy policy) {
    if (index < 0 || index >= w4a8_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!supports(w) || !w4a8_supports(w.qtype)) return ScheduleRefusal::Format;
    const bool producer = w4a8_schedule_info(index).producers > 0;
    if (producer && !w4a8_ws_supports(w.qtype)) return ScheduleRefusal::NotCompiled;
    if (!w4a8_fits(index, w.qtype)) return ScheduleRefusal::Resource;
    if (!w4a8_policy_allows(w, policy)) return ScheduleRefusal::Policy;
    const Route route{Family::W4a8Mma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // The classic tiles read a ggml high plane a byte at a time; the producer tiles in vector
    // words (generic_refusal), and IQ3_XXS's sign plane 8 bytes at a time (w4a8_ws_admits).
    const bool high_off = w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr &&
                          !aligned_to(w.qhigh, 16);
    if (!aligned_to(w.qdata, 16) || (producer && high_off) ||
        (producer && w.qtype == QType::GGML_IQ3_XXS && !aligned_to(w.qhigh, 8))) {
        return ScheduleRefusal::Alignment;
    }
    if (!w4a8_k_admits(w.qtype, w.k)) return ScheduleRefusal::Extent;
    return ScheduleRefusal::None;
}

ScheduleRefusal rmma_refusal(int index, const Weight& w) {
    if (index < 0 || index >= rmma_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!rmma_schedule_launchable(index)) return ScheduleRefusal::NotCompiled;
    if (!supports(w)) return ScheduleRefusal::Format;
    // Whether the build holds the tile for the codec, ahead of the weight's own checks, so an
    // alignment or extent refusal does not hide it (rmma_coverage counts compiled tiles by it).
    const WeightCodec codec = codec_of(w);
    const auto s            = rmma_schedule_info(index);
    if (!rmma_tile_compiled(codec, s.m_tiles, s.n_tiles, w.k)) return ScheduleRefusal::CodecTile;
    const Route route{Family::RegisterMma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // A ggml block format's high plane off a 16-byte boundary (generic_refusal).
    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) {
        return ScheduleRefusal::Alignment;
    }
    if ((w.k % k_multiple(codec)) != 0) return ScheduleRefusal::Extent;
    if (codec == WeightCodec::Fp8 || codec == WeightCodec::Fp8RowF32 ||
        codec == WeightCodec::Fp8B128 || codec == WeightCodec::Bf16 ||
        codec == WeightCodec::Mxfp4 || codec == WeightCodec::U4Zp ||
        codec == WeightCodec::U4ZpG64 ||
        codec == WeightCodec::GgmlQ4K || codec == WeightCodec::GgmlQ5K ||
        codec == WeightCodec::GgmlQ6K || codec == WeightCodec::GgmlQ3K ||
        codec == WeightCodec::GgmlQ2K || codec == WeightCodec::GgmlIq4Xs ||
        codec == WeightCodec::GgmlIq3Xxs || codec == WeightCodec::GgmlIq3S ||
        codec == WeightCodec::GgmlQ40 || codec == WeightCodec::GgmlQ41 ||
        codec == WeightCodec::GgmlQ50 || codec == WeightCodec::GgmlQ51 ||
        codec == WeightCodec::GgmlIq4Nl || is_ggml_block(w.qtype)) {
        return aligned_to(w.qdata, 16) ? ScheduleRefusal::None : ScheduleRefusal::Alignment;
    }
    return aligned_to(w.qdata, 8) ? ScheduleRefusal::None : ScheduleRefusal::Alignment;
}

ScheduleRefusal tiled_refusal(int index, const Weight& w) {
    if (index < 0 || index >= tiled_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!supports(w)) return ScheduleRefusal::Format;
    const WeightCodec codec = codec_of(w);
    const bool cutlass_only = volta_cutlass_only(codec);
    if (!cutlass_only && !tiled_takes(codec)) return ScheduleRefusal::Format;
    // A CUTLASS-only format (volta_cutlass_only) runs no tile of its own: schedule 0 names it.
    if (cutlass_only && index != 0) return ScheduleRefusal::CodecTile;
    if (!cutlass_only && !tiled_fits(index, codec)) return ScheduleRefusal::Resource;
    const Route route{Family::TiledMma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // A ggml block format's high plane off a 16-byte boundary (generic_refusal), or the codes where
    // the tiles read them.
    if ((w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) ||
        (!cutlass_only && !aligned_to(w.qdata, 16))) {
        return ScheduleRefusal::Alignment;
    }
    if ((w.k % 64) != 0) return ScheduleRefusal::Extent;
    return ScheduleRefusal::None;
}

ScheduleRefusal staged_refusal(int index, const Weight& w) {
    if (index < 0 || index >= staged_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!supports(w) || codec_of(w) != WeightCodec::Bf16) return ScheduleRefusal::Format;
    if (!staged_fits(index) || !staged_device_fits(index)) return ScheduleRefusal::Resource;
    const Route route{Family::StagedMma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // A ggml block format's high plane off 16 bytes (generic_refusal; a BF16 weight has none), or
    // the codes.
    if ((w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) ||
        !aligned_to(w.qdata, 16)) {
        return ScheduleRefusal::Alignment;
    }
    if ((w.k % 64) != 0) return ScheduleRefusal::Extent;
    return ScheduleRefusal::None;
}

ScheduleRefusal i8_refusal(int index, const Weight& w, LinearPolicy policy) {
    if (index < 0 || index >= i8_schedule_count()) return ScheduleRefusal::NotDeclared;
    if (!supports(w) || !i8_format(w.qtype)) return ScheduleRefusal::Format;
    if (!i8_fits(index, w.qtype)) return ScheduleRefusal::Resource;
    if (!allows_a8(policy)) return ScheduleRefusal::Policy;
    const Route route{Family::I8Mma, static_cast<std::uint8_t>(index)};
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    // A ggml block format's high plane off 16 bytes (generic_refusal; a G64 weight has none), the
    // codes or scales off 16 (one 16-byte copy per row), a Q5 / Q6 high plane off 8 / 16.
    if ((w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) ||
        !i8_planes_aligned(w)) {
        return ScheduleRefusal::Alignment;
    }
    if (!i8_extent(w)) return ScheduleRefusal::Extent;
    return ScheduleRefusal::None;
}

namespace {
// The checks every family's refusal shares (all but the scalar path's, which route_admits keeps):
// the generic path reads the weight (Format), the device has the route's instructions, and a ggml
// block format's high (sign / bit) plane lies on a 16-byte boundary. A view into a packed buffer
// can leave it off one (a row view stays aligned: 32 bytes a row and 256 values), and the kernels
// read it in vector words, all but the classic W4A8 tiles (one byte at a time; w4a8_refusal). The
// 2080 Ti and A100 gates faulted (misaligned address) on an IQ3_XXS plane 4 bytes off once the cost
// model's pick left the classic tile. NotDeclared stands for none.
ScheduleRefusal generic_refusal(Route route, const Weight& w) {
    if (!supports(w)) return ScheduleRefusal::Format;
    if (!has_features(current_features(), route_requires(route))) {
        return ScheduleRefusal::Instruction;
    }
    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16)) {
        return ScheduleRefusal::Alignment;
    }
    return ScheduleRefusal::NotDeclared;
}
} // namespace

ScheduleRefusal w4a4_refusal(int index, const Weight& w, LinearPolicy policy) {
    if (index < 0 || index >= w4a4_schedule_names()) return ScheduleRefusal::NotDeclared;
    if (index >= w4a4_schedule_count()) return ScheduleRefusal::NotCompiled;
    if (!supports(w) || w.qtype != QType::NVFP4 || !std::isfinite(w.input_scale_divisor) ||
        w.input_scale_divisor <= 0.0F) {
        return ScheduleRefusal::Format;
    }
    if (!allows_a4(policy)) return ScheduleRefusal::Policy;
    const auto common =
        generic_refusal(Route{Family::W4a4Mma, static_cast<std::uint8_t>(index)}, w);
    return common == ScheduleRefusal::NotDeclared ? ScheduleRefusal::None : common;
}

ScheduleRefusal nvfp4_tiled_refusal(int index, const Weight& w) {
    if (index < 0 || index >= nvfp4_tiled_schedule_count()) return ScheduleRefusal::NotDeclared;
    // The schedules parse on every build, but their kernels are 120a-only.
    if (!kNativeNvfp4) return ScheduleRefusal::NotCompiled;
    if (!supports(w) || w.qtype != QType::NVFP4) return ScheduleRefusal::Format;
    const auto common =
        generic_refusal(Route{Family::Nvfp4TiledMma, static_cast<std::uint8_t>(index)}, w);
    if (common != ScheduleRefusal::NotDeclared) return common;
    // supports() holds K to whole 64-value steps; a lane reads eight code bytes at once.
    return aligned_to(w.qdata, 8) ? ScheduleRefusal::None : ScheduleRefusal::Alignment;
}

ScheduleRefusal ggml_refusal(Family family, int index, const Weight& w, LinearPolicy policy) {
    const bool matrix = family == Family::GgmlMmq;
    if (index != 0 || (!matrix && family != Family::GgmlVec)) return ScheduleRefusal::NotDeclared;
    if (!supports(w)) return ScheduleRefusal::Format;
    const auto type = mmq_type(w);
    if (!type || !(matrix ? ggml_mmq::matrix_supported(*type) : ggml_mmq::vec_supported(*type))) {
        return ScheduleRefusal::Format;
    }
    // ggml_layout_admits: Q4_K / Q5_K as blocks, the others as planes, each plane it reads there.
    const bool planes = w.qtype != QType::GGML_Q4_K && w.qtype != QType::GGML_Q5_K;
    const auto bytes  = mmq_plane_bytes(w.qtype);
    if (planes != (w.layout == QuantLayout::GgmlPlanes) || (bytes.high > 0 && w.qhigh == nullptr) ||
        (bytes.scales > 0 && w.scales == nullptr)) {
        return ScheduleRefusal::Format;
    }
    if (!allows_a8(policy)) return ScheduleRefusal::Policy;
    const auto common = generic_refusal(Route{family, 0}, w);
    if (common != ScheduleRefusal::NotDeclared) return common;
    if (matrix && !ggml_mmq::device_supported()) return ScheduleRefusal::Instruction;
    // ggml's loaders read every plane in vector words.
    if (!aligned_to(w.qdata, 16) || (bytes.high > 0 && !aligned_to(w.qhigh, 16)) ||
        (bytes.scales > 0 && !aligned_to(w.scales, 16))) {
        return ScheduleRefusal::Alignment;
    }
    const auto multiple = matrix ? static_cast<std::int32_t>(ggml_mmq::kKMultiple)
                                 : static_cast<std::int32_t>(ggml_block_values(w.qtype));
    return (w.k % multiple) == 0 ? ScheduleRefusal::None : ScheduleRefusal::Extent;
}

bool ggml_mmq_admits(const Weight& w, LinearPolicy policy) { return mmq_admits(w, policy); }
bool ggml_vec_admits(const Weight& w, LinearPolicy policy) { return vec_admits(w, policy); }

bool route_admits(Route route, const Weight& w, LinearPolicy policy) {
    if (route.family == Family::RegisterMma) {
        return rmma_refusal(route.schedule, w) == ScheduleRefusal::None;
    }
    if (route.family == Family::R8Mma) {
        return r8_refusal(route.schedule, w, policy) == ScheduleRefusal::None;
    }
    if (route.family == Family::W4a8Mma) {
        return w4a8_refusal(route.schedule, w, policy) == ScheduleRefusal::None;
    }
    if (route.family == Family::TiledMma) {
        return tiled_refusal(route.schedule, w) == ScheduleRefusal::None;
    }
    if (route.family == Family::StagedMma) {
        return staged_refusal(route.schedule, w) == ScheduleRefusal::None;
    }
    if (route.family == Family::I8Mma) {
        return i8_refusal(route.schedule, w, policy) == ScheduleRefusal::None;
    }
    if (route.family == Family::W4a4Mma) {
        return w4a4_refusal(route.schedule, w, policy) == ScheduleRefusal::None;
    }
    if (route.family == Family::Nvfp4TiledMma) {
        return nvfp4_tiled_refusal(route.schedule, w) == ScheduleRefusal::None;
    }
    if (route.family == Family::GgmlMmq || route.family == Family::GgmlVec) {
        return ggml_refusal(route.family, route.schedule, w, policy) == ScheduleRefusal::None;
    }
    // The scalar path: every other family returned its refusal above (generic_refusal: a ggml high
    // plane off 16 bytes among them). The scalar path reads that plane byte by byte. A family with
    // no refusal of its own is refused here.
    return route.family == Family::Scalar && supports(w) &&
           has_features(current_features(), route_requires(route));
}

// Without tensor mma.sync a one-tile register-MMA call of at most four tokens sums lane by lane
// (GemmArgs::lane_local) outside the invariant profile: a token's sums then depend on its tile.
namespace {
bool rmma_sums_lane_local(int schedule) {
    return !current_features().mma_tensor && !invariant_routes_active() &&
           rmma_schedule_info(schedule).n_tiles == 1;
}
} // namespace

bool rmma_add_sums() {
    return current_features().mma_tensor && bf16_mma_native() && !invariant_routes_active() &&
           !prefill_routes_active();
}

bool route_is_token_invariant(Route route) {
    switch (route.family) {
    case Family::RegisterMma:
        return rmma_schedule_info(route.schedule).warps_k == 1 &&
               !rmma_sums_lane_local(route.schedule);
    case Family::W4a4Mma:
    case Family::TiledMma:
    case Family::W4a8Mma:
    case Family::StagedMma:
    case Family::R8Mma: // a token's sums depend on its own activation only
        return true;
    case Family::Nvfp4TiledMma:
        return nvfp4_tiled_schedule_info(route.schedule).classes == 1; // rmma WarpsK == 1
    case Family::Scalar:
        return true; // one lane order per element, independent of T
    case Family::GgmlMmq: // stream-k splits K by the call's tile count
        return false;
    case Family::GgmlVec: // a lane's slices and one butterfly per token, whatever T
        return true;
    case Family::I8Mma: // K groups ascending, never split: a token's sums are its own
        return true;
    }
    return false;
}

SummationOrder summation_order(Route route) {
    // The orders of the prefill and invariant profiles, where its users act (the prefill route's
    // bands, in the route tables, route-tune and auto_route): there no register-MMA tile adds its
    // min-term sums in FP32 (rmma_add_sums), so the WarpsK == 1 tiles of every width and the staged
    // MMA sum alike. Outside them the tiles of four or more token tiles do not.
    if ((route.family == Family::RegisterMma && route.schedule < rmma_schedule_count() &&
         rmma_schedule_info(route.schedule).warps_k == 1 &&
         !rmma_sums_lane_local(route.schedule)) ||
        (route.family == Family::StagedMma && route.schedule < staged_schedule_count())) {
        return {SummationOrder::Kind::Register, Route{}};
    }
    if (route.family == Family::I8Mma) return {SummationOrder::Kind::I8, Route{}};
    return {SummationOrder::Kind::Own, route};
}

bool same_summation(Route a, Route b) { return summation_order(a) == summation_order(b); }

namespace {
// The NVFP4 tiled route in the order of the register MMA with WarpsK = `classes`, for calls of two
// token tiles or more whose tiles give every SM at least one CTA; nullopt otherwise. Against the
// invariant register MMA (WarpsK 8) on an RTX 5090: -8..-40% from there on the
// Muse Glimmer 30B MLP and a synthetic [4096, 4096]; below one wave, or at one token tile, it
// loses (up to 5x at T = 32).
std::optional<Route> nvfp4_tiled_route(const Problem& problem, const Weight& weight,
                                       [[maybe_unused]] int classes,
                                       [[maybe_unused]] std::int32_t tokens) {
    if (problem.qtype != QType::NVFP4) return std::nullopt;
#ifdef NINFERNO_WITHOUT_NVFP4
    return std::nullopt;
#endif
    for (int i = 0; i < nvfp4_tiled_schedule_count(); ++i) {
        const auto s = nvfp4_tiled_schedule_info(i);
        if (s.classes != classes) continue;
        const Route r{Family::Nvfp4TiledMma, static_cast<std::uint8_t>(i)};
        if (!route_admits(r, weight, problem.policy)) return std::nullopt;
        const std::int64_t rows_per_cta =
            problem.epilogue == Epilogue::SwiGlu ? s.block_n / 2 : s.block_n;
        const std::int64_t ctas = ((problem.n + rows_per_cta - 1) / rows_per_cta) *
                                  ((tokens + s.block_m - 1) / s.block_m);
        if (tokens < 2 * s.block_m || ctas < sm_count()) return std::nullopt;
        return r;
    }
    return std::nullopt;
}
} // namespace

Route r8_rules_route(const Problem& problem, std::int32_t tokens) {
    return Route{Family::R8Mma, static_cast<std::uint8_t>(rules_r8(problem, tokens))};
}

// The I8 tiles (generic_i8.cu): 256x128 (8 warps), 128x128 and 128x64 (4 and 2 warps), four
// stages each. All give the same bytes.
constexpr I8ScheduleInfo kI8Schedules[] = {
    {256, 128, 256, 4, 4 * (256 * 64 + 128 * 32 + 256 * 8) + 2 * 128 * 16},
    {128, 128, 128, 4, 4 * (128 * 64 + 128 * 32 + 128 * 8) + 2 * 128 * 16},
    {128, 64, 64, 4, 4 * (128 * 64 + 64 * 32 + 128 * 8) + 2 * 64 * 16},
};
int i8_schedule_count() { return static_cast<int>(std::size(kI8Schedules)); }

// Each I8 tile's fraction of the int8 peak at a full grid (stage-1 probe, RTX 3090, 4090 and
// 5070 Ti, 27B shapes): 256x128 0.53, 128x128 0.46, 128x64 0.42.
constexpr auto kI8Eff = route_cost::kI8Efficiency;
static_assert(kI8Eff.values.size() == std::size(kI8Schedules));
bool i8_fits(int schedule, QType qtype) {
    // The high plane takes 0 / 8 / 16 bytes a group and weight row in every stage; a tile runs
    // three stages where four do not fit (generic_i8.cu kI8Stages).
    const auto s       = i8_schedule_info(schedule);
    const int high     = i8_bits(qtype) == 4 ? 0 : i8_bits(qtype) == 5 ? 8 : 16;
    const auto ring    = static_cast<std::size_t>(2 * s.block_n * 16);
    const auto stage   = (s.smem_bytes - ring) / static_cast<std::size_t>(s.stages) +
                       static_cast<std::size_t>(s.block_n) * high;
    return 3 * stage + ring <= kMaxSharedBytesPerBlock;
}
I8ScheduleInfo i8_schedule_info(int index) {
    if (index < 0 || index >= i8_schedule_count()) {
        throw std::out_of_range("generic I8: schedule index");
    }
    return kI8Schedules[index];
}

int i8_pick(const Problem& problem, QType qtype, std::int32_t tokens) {
    const double sets = problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0;
    const double sms  = std::max(sm_count(), 1);
    int best          = -1;
    double best_cost  = 0;
    for (int i = 0; i < i8_schedule_count(); ++i) {
        if (!i8_fits(i, qtype)) continue;
        const auto s       = kI8Schedules[i];
        const double ctas  = std::ceil(problem.n * sets / s.block_n) * std::ceil(tokens / double(s.block_m));
        const double waves = std::ceil(ctas / sms);
        const double cost  = waves * s.block_m * s.block_n / kI8Eff[i];
        if (best < 0 || cost < best_cost) {
            best      = i;
            best_cost = cost;
        }
    }
    return best;
}

// i8_pick's tile where the weight is admitted to it. The schedules differ in what they admit only
// by i8_fits (i8_refusal), so this is the best tile of those the weight is admitted to.
std::optional<Route> i8_route(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    const int pick = i8_pick(problem, weight.qtype, tokens);
    if (pick < 0) return std::nullopt;
    const Route r{Family::I8Mma, static_cast<std::uint8_t>(pick)};
    if (!route_admits(r, weight, problem.policy)) return std::nullopt;
    return r;
}

// The heuristic's W4A8 tile, else the first schedule that fits the build (Turing: the 128x128
// tile needs 80 KiB).
std::optional<Route> w4a8_route(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    const Route tuned{Family::W4a8Mma,
                      static_cast<std::uint8_t>(heuristic_w4a8(problem, weight, tokens))};
    if (route_admits(tuned, weight, problem.policy)) return tuned;
    for (int i = 0; i < w4a8_schedule_count(); ++i) {
        const Route r{Family::W4a8Mma, static_cast<std::uint8_t>(i)};
        if (route_admits(r, weight, problem.policy)) return r;
    }
    return std::nullopt;
}

// Outside the prefill profile BF16 weights take the staged MMA from 33 tokens where one of its
// tiles fills the SMs: 256x128 (two stages) from two waves, 128x64 (two stages, two CTAs per SM)
// from one, 64x64 (three stages) from half a wave; else the register MMA. ninferno_route_tune's
// BF16 timings on seven boards, with the register-MMA rule above: 1.116 of the best (geomean)
// against 1.15..1.36 for the largest tile that filled the SMs and 1.33..1.64 for the old gate
// (from 256 tokens, the first tile); a wider tile than the call's grid leaves SMs idle (n = 1024
// at T = 256: the 128x128 tile 2.5x the register MMA, n = 48: 6x).
std::optional<Route> staged_speed_route(const Problem& problem, const Weight& weight,
                                        std::int32_t tokens) {
    if (tokens < kRmmaMidMinTokens) return std::nullopt;
    struct Pick {
        int rows, cols, stages;
        double waves;
    };
    for (const Pick p : {Pick{256, 128, 2, 2.0}, Pick{128, 64, 2, 1.0}, Pick{64, 64, 3, 0.5}}) {
        if (speed_waves(problem.n, tokens, p.rows, p.cols) < p.waves) continue;
        for (int i = 0; i < staged_schedule_count(); ++i) {
            const auto st = staged_schedule_info(i);
            if (st.rows != p.rows || st.tokens != p.cols || st.stages != p.stages) continue;
            const Route r{Family::StagedMma, static_cast<std::uint8_t>(i)};
            if (route_admits(r, weight, problem.policy)) return r;
        }
    }
    return std::nullopt;
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
    default: return std::nullopt;
    }
}

std::optional<RouteTraffic> route_traffic(Route route, const Weight& weight,
                                          const Problem& problem, std::int32_t tokens) {
    const auto tile = route_tile(route, problem.epilogue, tokens);
    if (!tile || problem.n <= 0 || problem.k <= 0 || tokens <= 0) return std::nullopt;
    const double n = problem.n, k = problem.k, t = tokens;
    const auto cdiv = [](double a, double b) { return std::ceil(a / b); };
    // The activation in the route's representation: bytes a value, the K and the tokens it holds.
    // BF16 for the A16 tiles; W4A8 one byte a value and an FP16 scale a 32 (and the block sum,
    // 40 bytes a 32, for a format with a min), K in whole 128-value tiles; I8 an FP32 scale and an int32 sum a 64-value group, tokens padded to whole 256-token
    // tiles.
    double x_bytes = 2.0, x_k = k, x_t = t;
    const QType q  = problem.qtype;
    const bool has_min = q == QType::GGML_Q4_1 || q == QType::GGML_Q5_1 || q == QType::GGML_Q4_K ||
                         q == QType::GGML_Q5_K;
    if (route.family == Family::W4a8Mma) {
        x_bytes = (has_min ? 40.0 : 36.0) / 32.0;
        x_k     = cdiv(k, 128.0) * 128.0;
    } else if (route.family == Family::I8Mma) {
        x_bytes = 1.0 + 8.0 / 64.0;
        x_t     = i8_padded_tokens(tokens);
    }
    const double weight_reads = cdiv(t, tile->tokens);
    // The register MMA holds its operands in registers: each of a CTA's row warps loads the
    // activation slice itself (the shared-memory tiles stage it once for the CTA).
    const double x_readers =
        route.family == Family::RegisterMma ? rmma_schedule_info(route.schedule).warps_m : 1.0;
    const double bpw = static_cast<double>(weight.payload_bytes) / (n * k);
    return RouteTraffic{n * k * bpw * weight_reads,
                        x_t * x_k * x_bytes * cdiv(n, tile->out_rows) * x_readers, n * t * 2.0};
}

Route heuristic_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                      bool prefill) {
    // Under the prefill profile every call of 17 tokens or more takes one route; outside it, routes
    // by T band. A4 only where the policy allows it, and never at T <= 16, which is where decode
    // and speculative verify run.
    const bool prefill_band = prefill && tokens >= 17;
    if (tokens >= 17 && allows_w4a4(weight, problem.policy)) {
        const Route r{Family::W4a4Mma,
                      static_cast<std::uint8_t>(heuristic_w4a4(problem.n, tokens, prefill_band))};
        if (route_admits(r, weight, problem.policy)) return r;
    }
    // A8 where the policy allows it: ggml's vector kernel for speculative verify (2..8 tokens) of
    // the ggml block formats it decodes (vec_rule), R8 for decode and for the verify of the other
    // formats with an R8 codec, W4A8 from 17 tokens; the rest stays A16. The vector kernel streams
    // each weight row once for every token of the call and runs the epilogue in place: on the RTX
    // 5090 ISTA IQ3_S MTP3 it beat the R8 and A16 schedules the dense tuner picked there (C1 166 ->
    // 174 tok/s), whose L2-cold timings of one call at a time miss the stream of a verify step. The
    // tuner still replaces it where a route beats it by its margin (Muse Glimmer 30B Q4_K_M: the
    // rule alone lost C1 by 19 %).
    if (vec_rule(weight, problem.policy, tokens)) return Route{Family::GgmlVec, 0};
    // From 17 tokens outside the prefill profile the cost model's route of either activation class
    // (cost_route, a8_candidates); the rules below where the device reports no clocks.
    if (!prefill_band && tokens >= 17 && problem.qtype != QType::NVFP4) {
        if (const auto r = cost_route(problem, weight, tokens)) return *r;
    }
    if (r8_call(weight, problem.policy, tokens)) {
        const Route r{Family::R8Mma, static_cast<std::uint8_t>(heuristic_r8(problem, tokens))};
        if (route_admits(r, weight, problem.policy)) return r;
    }
    if (tokens >= 17 && mmq_prefill_wins(problem, weight)) return Route{Family::GgmlMmq, 0};
    if (tokens >= 17) {
        if (const auto r = i8_route(problem, weight, tokens)) return *r;
    }
    if (tokens >= 17 && allows_w4a8(weight, problem.policy)) {
        if (const auto r = w4a8_route(problem, weight, tokens)) return *r;
    }
    // Where the BF16 MMA is emulated (Turing), the band between R8 / the vector kernel and W4A8
    // takes ggml's int8 matrix kernel for the formats it reads: the register MMA there runs on
    // the emulation, about three times its native cost.
    if (mmq_verify_rule(weight, problem.policy, problem.n, tokens)) return Route{Family::GgmlMmq, 0};
    // NVFP4 A16 outside the invariant profile stays on the register MMA: against its prefill
    // schedule the tiled route loses on the Muse gate/up and synthetic shapes (+11..+20%)
    // and wins on the down projection only from T = 2048. A route table may still name it.
    // A format with a tiled codec takes it for every prefill-profile call of 17 tokens or more (one
    // summation order for all of them) and, outside the profile, where its tiles fill the SMs
    // (tiled_speed_pays).
    // The first schedule whose tiles fit the build: the tuned 128x128 everywhere but Turing.
    if (prefill_band || tiled_speed_pays(problem.n, tokens)) {
        for (int i = 0; i < tiled_schedule_count(); ++i) {
            const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
            if (route_admits(r, weight, problem.policy)) return r;
        }
    }
    // BF16 weights take the staged register MMA (same bytes, operands through shared memory):
    // under the prefill profile the first schedule that fits the build, and outside it once its
    // 64x64 tiles cover half the SMs, with the largest tile whose grid still fills them
    // (staged_speed_route). Only on the native BF16 MMA: where it is emulated its tiles lost to
    // the register MMA, x2.5..4 on Turing (RTX 2080 Ti, BF16 N 1536 K 2048, T 17..1024: 142..1005
    // us against 35..402) and x6..8 on Volta's FP32 FMAs (V100, T 17..128: 942 us against 110).
    if (bf16_mma_native()) {
        if (prefill_band) {
            for (int i = 0; i < staged_schedule_count(); ++i) {
                const Route r{Family::StagedMma, static_cast<std::uint8_t>(i)};
                if (route_admits(r, weight, problem.policy)) return r;
            }
        } else if (const auto r = staged_speed_route(problem, weight, tokens)) {
            return *r;
        }
    }
    const Route r{Family::RegisterMma,
                  static_cast<std::uint8_t>(heuristic_rmma(problem, tokens, prefill_band))};
    if (route_admits(r, weight, problem.policy)) return r;
    // A register-MMA tile the build does not hold for the call: the narrower tiles the dense
    // tuner picks where the BF16 MMA is emulated (m1.n2 and m1.n1 of four row warps: V100 and RTX
    // 2080 Ti, BF16 N 1536 K 2048, T 17..2048), then the staged MMA, before the scalar route (V100,
    // gpt-oss BF16 attention at T = 79: the scalar route took 7.3 ms a call).
    for (const int n_tiles : {2, 1}) {
        const int index = rmma_index(4, 1, 1, n_tiles);
        const Route narrow{Family::RegisterMma, static_cast<std::uint8_t>(index)};
        if (route_admits(narrow, weight, problem.policy)) return narrow;
    }
    if (prefill_band || tokens >= kStagedSpeedMinTokens) {
        for (int i = 0; i < staged_schedule_count(); ++i) {
            const Route staged{Family::StagedMma, static_cast<std::uint8_t>(i)};
            if (route_admits(staged, weight, problem.policy)) return staged;
        }
    }
    return Route{Family::Scalar, 0};
}

std::optional<Route> volta_tiled_route(const Weight& weight, LinearPolicy policy,
                                       std::int32_t tokens) {
    // Without tensor mma.sync (Volta) every register-MMA, W4A8 and W4A4 tile runs on FP32 FMAs or
    // dp4a over warp shuffles (ops/common/mma.cuh), while the tiled kernel issues the tensor cores'
    // own mma.m8n8k4 (generic_tiled.cuh): a call of 17 tokens or more whose format has a tiled
    // codec takes it whatever the policy allows (f16 activations, not int8 ones). Decode and verify
    // (T <= 16) keep their routes, so the switch is at T = 17.
    if (tokens < 17 || current_features().mma_tensor || !supports(weight)) return std::nullopt;
    if (volta_cutlass_only(codec_of(weight))) {
        // From 17 tokens, as the tiled formats: a prefill token's bytes do not depend on its call's
        // width (a narrower call on another route summed in another order).
        const Route r{Family::TiledMma, 0};
        if (volta_cutlass_runs(true, tokens) && route_admits(r, weight, policy)) return r;
        return std::nullopt;
    }
    if (!tiled_takes(codec_of(weight))) return std::nullopt;
    for (int i = 0; i < tiled_schedule_count(); ++i) {
        const Route r{Family::TiledMma, static_cast<std::uint8_t>(i)};
        if (route_admits(r, weight, policy)) return r;
    }
    return std::nullopt;
}

namespace {
Route auto_route(const Problem& problem, const Weight& weight, std::int32_t tokens, bool prefill);
std::optional<Route> pinned_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                                  bool prefill);

// Without tensor mma.sync (Volta) a call of 1..16 tokens of Q4_K / Q5_K / Q6_K or Q8 G32 takes
// the HMMA kernel (generic_volta_hmma.cu): it runs as a register-MMA route (run_route launches it
// in place of the emulated tiles, volta_hmma_takes), f16 activations as the tiled route's from 17
// tokens. Its alternatives grow with T on Volta's CUDA cores: the register MMA's emulated
// fragments (FP32 FMAs and shuffles), ggml's vector kernel (dp4a, the activation re-read for every
// column) and R8 (dp4a); the tensor cores multiply eight or sixteen columns for the price of one,
// so the call costs reading its weight once, whatever its width up to eight (sixteen: two column
// groups). An A16 call takes it from one token. A call with an int8 route keeps that route
// through the widths it holds where tensor mma.sync exists (kVecRuleMaxTokens): there an HMMA
// leaves at least half of its eight columns empty while the int8 kernels' per-column work is still
// small, and from five tokens the HMMA kernel takes over, where those cards move to their MMA.
// One card (GV100) checked the band, Qwen3.5-9B shapes: at T = 5..8 the HMMA kernel took 0.52..0.91
// of the vector kernel's time, except Q4_K gate/up at T = 5 (140 against 128 us); at T = 2..4 the
// vector kernel was up to 1.4x faster.
std::optional<Route> volta_hmma_route(const Problem& problem, const Weight& weight,
                                      std::int32_t tokens) {
    if (tokens < kVoltaHmmaMinTokens || tokens > kVoltaHmmaMaxTokens ||
        current_features().mma_tensor || !supports(weight) ||
        !volta_hmma_codec(codec_of(weight)) ||
        (weight.k % volta_hmma_k_multiple(codec_of(weight))) != 0) {
        return std::nullopt;
    }
    if (tokens <= kVecRuleMaxTokens &&
        (vec_call(weight, problem.policy, tokens) || r8_call(weight, problem.policy, tokens))) {
        return std::nullopt;
    }
    const Route r{Family::RegisterMma,
                  static_cast<std::uint8_t>(heuristic_rmma(problem, tokens, false))};
    if (route_admits(r, weight, problem.policy)) return r;
    return std::nullopt;
}
} // namespace

std::optional<Route> volta_fixed_route(const Problem& problem, const Weight& weight,
                                       std::int32_t tokens) {
    if (const auto r = volta_tiled_route(weight, problem.policy, tokens)) return r;
    return volta_hmma_route(problem, weight, tokens);
}

Route select_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                   bool prefill) {
    // The invariant profile: one summation order per problem at every width, BF16 activations
    // whatever the policy allows. The scalar route (one lane order per element) takes what the
    // register MMA cannot read.
    if (invariant_routes_active()) {
        const Route r{Family::RegisterMma,
                      static_cast<std::uint8_t>(invariant_rmma(problem.k, tokens))};
        if (!route_admits(r, weight, problem.policy)) return Route{Family::Scalar, 0};
        // NVFP4 wide calls: the tiled kernel sums in the register MMA's order (the same bytes).
        if (const auto t =
                nvfp4_tiled_route(problem, weight, rmma_schedule_info(r.schedule).warps_k, tokens)) {
            return *t;
        }
        return r;
    }
    // A route probe forcing one activation route for every width (route_probe.h).
    if (const std::int8_t forced = route_probe_state.quantized_act; forced >= 0) {
        const bool prefill_band = prefill && tokens >= 17;
        if (forced == 1 && allows_w4a4(weight, problem.policy)) {
            const Route r{Family::W4a4Mma,
                          static_cast<std::uint8_t>(heuristic_w4a4(problem.n, tokens, prefill_band))};
            if (route_admits(r, weight, problem.policy)) return r;
        }
        if (forced == 1 && r8_call(weight, problem.policy, tokens)) {
            const Route r{Family::R8Mma, static_cast<std::uint8_t>(heuristic_r8(problem, tokens))};
            if (route_admits(r, weight, problem.policy)) return r;
        }
        if (forced == 1 && allows_w4a8(weight, problem.policy)) {
            if (const auto r = w4a8_route(problem, weight, tokens)) return *r;
        }
        const Route r{Family::RegisterMma,
                      static_cast<std::uint8_t>(heuristic_rmma(problem, tokens, prefill_band))};
        if (route_admits(r, weight, problem.policy)) return r;
        return Route{Family::Scalar, 0};
    }
    if (const auto r = volta_tiled_route(weight, problem.policy, tokens)) {
        // Volta keeps its tiled route under a prefill pin (f16 activations: the a16 family's
        // there), and the ledger notes it, so a pin no prefill call can take is refused.
        if (tokens >= kPrefillPinMinTokens && route_pins().prefill != PrefillRoute::Auto) {
            note_prefill_pin(route_pins().prefill == PrefillRoute::A16,
                             family_bit(PrefillRoute::A16));
        }
        return *r;
    }
    // A maintainer's route pin (--prefill-route, --decode-route; ops/route_pins.h): the pinned
    // family where it can compute the call, else the automatic route below. No pin: nothing here.
    // It acts before Volta's HMMA verify route (volta_fixed_route's second half), the automatic
    // choice of that band, so an A/B can pin vec or R8 against it.
    if (const auto pinned = pinned_route(problem, weight, tokens, prefill)) return *pinned;
    if (const auto r = volta_hmma_route(problem, weight, tokens)) return *r;
    return auto_route(problem, weight, tokens, prefill);
}

namespace {

// The automatic selection after the profiles, the probe and Volta's fixed routes (select_route).
Route auto_route(const Problem& problem, const Weight& weight, std::int32_t tokens, bool prefill) {
    // NVFP4 keeps the width invariance its own kernels give where a device has them (the shape
    // tables of sm_100+; ops/linear/nvfp4): on sm_89/86 the same projections are the generic
    // path's, whose speed bands split K by width (heuristic_rmma: 4, 8, 2 or 1 K-warps from 17
    // tokens), so a prefill token's bytes followed its chunk's size (Gemma 4 --a16 on an RTX 4090,
    // test_linear_gemma4_token_invariance). An A16 call of 17 tokens or more takes one summation
    // order at every width; decode and verify (T <= 16) keep their routes. W4A4 and W4A8 sum each
    // token over its own activation whatever the tile, so A4/A8 calls keep their speed tiles.
    if (problem.qtype == QType::NVFP4 && tokens >= 17 && !allows_w4a4(weight, problem.policy) &&
        !allows_w4a8(weight, problem.policy)) {
        // A weight that fits half the L2 takes the invariant profile's register MMA (the decode
        // split of K, 16 rows per CTA): its 32-token columns re-read the weight from L2, where the
        // 128x128 prefill tile left a 1024-row projection eight CTAs (RTX 4090, Gemma 4 26B
        // shapes 1024..8192 x 2112..8192 at T 17..128: 2-4x slower than the speed bands).
        // A larger weight takes the prefill profile's route (31B 16384x5376, 43008x5376,
        // 5376x21504: 0.3-0.8x the speed bands from 32 tokens).
        const auto weight_bytes =
            static_cast<std::size_t>(problem.n) * static_cast<std::size_t>(problem.k) * 9 / 16;
        if (2 * weight_bytes <= device_class().l2_bytes) {
            const Route r{Family::RegisterMma,
                          static_cast<std::uint8_t>(invariant_rmma(problem.k, tokens))};
            if (route_admits(r, weight, problem.policy)) return r;
        }
        prefill = true;
    }
    // A W4A8 problem of 17 tokens or more takes the W4A8 heuristic: route tables hold A16 schedules
    // only (compute_class keys these formats as a16 whatever the policy), which apply below that.
    // An R8 call likewise.
    // ggml's matrix kernel stands in for the heuristic only: a route table's or the dense tuner's
    // A16 schedule keeps its shape.
    const bool a8_wide = tokens >= 17 && allows_w4a8(weight, problem.policy);
    const bool vec     = vec_rule(weight, problem.policy, tokens);
    const bool a8      = a8_wide || vec || r8_call(weight, problem.policy, tokens) ||
                    mmq_verify_rule(weight, problem.policy, problem.n, tokens);
    const auto tuned   = a8 ? std::optional<Route>{} : table_route(problem, tokens, prefill);
    if (tuned) {
        // Under the prefill profile a format with a tiled codec keeps its tiled route: the tiled
        // and register-MMA families sum in different orders.
        const bool prefill_band = prefill && tokens >= 17;
        if (route_admits(*tuned, weight, problem.policy) &&
            (!prefill_band || route_is_token_invariant(*tuned)) &&
            !(prefill_band && supports(weight) && tiled_supports(codec_of(weight)) &&
              tuned->family != Family::TiledMma)) {
            return *tuned;
        }
    }
    // The dense tuner's choice at decode widths (generic_tune.cpp): an A16 schedule for an A16 call,
    // an R8 schedule for an R8 decode call; never a W4A8 call.
    if (!a8_wide) {
        // Under the vector kernel's rule any timed route that beat it may stand in.
        const bool r8_ok = r8_call(weight, problem.policy, tokens) ||
                           r8_verify_tunable(weight, problem.policy, tokens);
        if (const auto dense = tuned_dense_route(problem, tokens);
            dense && route_admits(*dense, weight, problem.policy) &&
            (dense->family == Family::R8Mma ? r8_ok
                                            : vec || !r8_call(weight, problem.policy, tokens))) {
            return *dense;
        }
    }
    const Route heuristic = heuristic_route(problem, weight, tokens, prefill);
    // Its prefill bands: a W4A8 schedule for the W4A8 heuristic (the same bytes at every tile), a
    // route of the prefill route's summation order under the prefill profile.
    if (const auto band = tuned_prefill_route(problem, tokens, heuristic);
        band && route_admits(*band, weight, problem.policy) &&
        ((band->family == Family::W4a8Mma && heuristic.family == Family::W4a8Mma) ||
         (prefill && band->family != Family::W4a8Mma && same_summation(*band, heuristic)))) {
        return *band;
    }
    return heuristic;
}

// The route pinned family `f` gives a call of 17 tokens or more, nullopt where it cannot compute
// it (the family's codec, K, alignment, policy or device).
std::optional<Route> prefill_family_route(PrefillRoute f, const Problem& p, const Weight& w,
                                          std::int32_t tokens, bool prefill) {
    switch (f) {
    case PrefillRoute::Auto:
        return std::nullopt;
    case PrefillRoute::A16: {
        // The automatic selection with BF16 activations: what an A16Only weight takes.
        Problem a16 = p;
        a16.policy  = LinearPolicy::A16Only;
        return auto_route(a16, w, tokens, prefill);
    }
    case PrefillRoute::W4a8: {
        if (!allows_w4a8(w, p.policy)) return std::nullopt;
        const auto r = w4a8_route(p, w, tokens);
        if (!r) return std::nullopt;
        // Its tuned prefill band, as the automatic W4A8 call takes it (the same bytes).
        if (const auto band = tuned_prefill_route(p, tokens, *r);
            band && band->family == Family::W4a8Mma && route_admits(*band, w, p.policy)) {
            return *band;
        }
        return r;
    }
    case PrefillRoute::I8:
        return i8_route(p, w, tokens);
    case PrefillRoute::Mmq:
        if (ggml_mmq_format(p.qtype) && (p.n % ggml_mmq::kRowMultiple) == 0 &&
            mmq_admits(w, p.policy)) {
            return Route{Family::GgmlMmq, 0};
        }
        return std::nullopt;
    case PrefillRoute::W4a4: {
        if (!allows_w4a4(w, p.policy)) return std::nullopt;
        const Route r{Family::W4a4Mma, static_cast<std::uint8_t>(
                                           heuristic_w4a4(p.n, tokens, prefill && tokens >= 17))};
        if (route_admits(r, w, p.policy)) return r;
        return std::nullopt;
    }
    }
    return std::nullopt;
}

// The route pinned family `f` gives a call of 1..16 tokens, nullopt where it cannot compute it.
std::optional<Route> decode_family_route(DecodeRoute f, const Problem& p, const Weight& w,
                                         std::int32_t tokens, bool prefill) {
    switch (f) {
    case DecodeRoute::Auto:
        return std::nullopt;
    case DecodeRoute::A16: {
        Problem a16 = p;
        a16.policy  = LinearPolicy::A16Only;
        return auto_route(a16, w, tokens, prefill);
    }
    case DecodeRoute::R8: {
        // Decode where the activation fits a CTA; the verify widths on the pre-quantized form of
        // every format that has it (the dense tuner's R8 candidates, r8_verify_tunable).
        const bool runs = tokens <= kR8DecodeTokens ? r8_call(w, p.policy, tokens)
                                                    : r8_verify_tunable(w, p.policy, tokens);
        if (!runs) return std::nullopt;
        const Route r{Family::R8Mma, static_cast<std::uint8_t>(heuristic_r8(p, tokens))};
        if (route_admits(r, w, p.policy)) return r;
        return std::nullopt;
    }
    case DecodeRoute::Vec: {
        const Route r{Family::GgmlVec, 0};
        if (vec_call(w, p.policy, tokens) && route_admits(r, w, p.policy)) return r;
        return std::nullopt;
    }
    case DecodeRoute::Mmq: {
        const Route r{Family::GgmlMmq, 0};
        if (mmq_call(w, p.policy, p.n, tokens) && route_admits(r, w, p.policy)) return r;
        return std::nullopt;
    }
    }
    return std::nullopt;
}

constexpr PrefillRoute kPinnedPrefillFamilies[] = {
    PrefillRoute::A16, PrefillRoute::W4a8, PrefillRoute::I8, PrefillRoute::Mmq, PrefillRoute::W4a4};
constexpr DecodeRoute kPinnedDecodeFamilies[] = {DecodeRoute::A16, DecodeRoute::R8,
                                                 DecodeRoute::Vec, DecodeRoute::Mmq};

std::optional<Route> pinned_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                                  bool prefill) {
    const RoutePins pins = route_pins();
    if (tokens >= kPrefillPinMinTokens && pins.prefill != PrefillRoute::Auto) {
        std::optional<Route> mine;
        std::uint32_t could = 0;
        for (const PrefillRoute f : kPinnedPrefillFamilies) {
            const auto r = prefill_family_route(f, problem, weight, tokens, prefill);
            if (!r) continue;
            could |= family_bit(f);
            if (f == pins.prefill) mine = r;
        }
        note_prefill_pin(mine.has_value(), could);
        return mine;
    }
    if (tokens >= 1 && tokens <= kDecodePinMaxTokens && pins.decode != DecodeRoute::Auto) {
        std::optional<Route> mine;
        std::uint32_t could = 0;
        for (const DecodeRoute f : kPinnedDecodeFamilies) {
            const auto r = decode_family_route(f, problem, weight, tokens, prefill);
            if (!r) continue;
            could |= family_bit(f);
            if (f == pins.decode) mine = r;
        }
        note_decode_pin(mine.has_value(), could);
        return mine;
    }
    return std::nullopt;
}

bool family_on_device(Family family) {
    return has_features(current_features(), route_requires(Route{family, 0}));
}

} // namespace

std::optional<Route> pinned_family_route(PrefillRoute family, const Problem& problem,
                                         const Weight& weight, std::int32_t tokens, bool prefill) {
    if (tokens < kPrefillPinMinTokens) return std::nullopt;
    return prefill_family_route(family, problem, weight, tokens, prefill);
}

std::optional<Route> pinned_family_route(DecodeRoute family, const Problem& problem,
                                         const Weight& weight, std::int32_t tokens, bool prefill) {
    if (tokens < 1 || tokens > kDecodePinMaxTokens) return std::nullopt;
    return decode_family_route(family, problem, weight, tokens, prefill);
}

bool pin_family_on_device(PrefillRoute family) {
    switch (family) {
    case PrefillRoute::Auto:
    case PrefillRoute::A16:
        return true;
    case PrefillRoute::W4a8:
        return w4a8_schedule_count() > 0 && family_on_device(Family::W4a8Mma);
    case PrefillRoute::I8:
        return i8_schedule_count() > 0 && family_on_device(Family::I8Mma);
    case PrefillRoute::Mmq:
        return ggml_mmq::device_supported() && family_on_device(Family::GgmlMmq);
    case PrefillRoute::W4a4:
        return w4a4_schedule_count() > 0 && family_on_device(Family::W4a4Mma);
    }
    return false;
}

bool pin_family_on_device(DecodeRoute family) {
    switch (family) {
    case DecodeRoute::Auto:
    case DecodeRoute::A16:
        return true;
    case DecodeRoute::R8:
        return r8_schedule_count() > 0 && family_on_device(Family::R8Mma);
    case DecodeRoute::Vec:
        return family_on_device(Family::GgmlVec);
    case DecodeRoute::Mmq:
        return ggml_mmq::device_supported() && family_on_device(Family::GgmlMmq);
    }
    return false;
}

bool r8_verify_tunable(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= 2 && tokens <= kR8VerifyTokens && allows_r8(w, policy) &&
           r8_verify_compiled(w.qtype);
}

bool r8_tunable(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return r8_verify_tunable(w, policy, tokens) ||
           (vec_rule(w, policy, tokens) && r8_call(w, policy, tokens));
}

std::optional<PreparedRule> prepared_rule(const ResolvedRoute& route, const Weight& weight) {
    using R = PreparedRepresentation;
    switch (route.executed.family) {
    case Family::GgmlVec:
        return PreparedRule{R::GgmlQ8_1};
    case Family::W4a8Mma:
        return PreparedRule{R::W4a8Blocks, w4a8_ld(weight.k)};
    case Family::R8Mma:
        if (!route.r8_prequantized) { return std::nullopt; } // quantized inside the kernel
        return PreparedRule{R::W4a8Blocks, weight.k};
    case Family::I8Mma:
        return PreparedRule{R::I8Blocks64};
    case Family::GgmlMmq:
        return PreparedRule{R::GgmlQ8_1Mmq, 0, static_cast<std::int32_t>(weight.qtype)};
    case Family::W4a4Mma:
        return PreparedRule{R::Nvfp4, 0, 0, true};
    case Family::RegisterMma:
    case Family::Scalar:
    case Family::TiledMma:
    case Family::StagedMma:
    case Family::Nvfp4TiledMma:
        return std::nullopt;
    }
    return std::nullopt;
}

bool prepared_rules_match(const PreparedRule& a, const PreparedRule& b) noexcept {
    return !a.weight_bound && !b.weight_bound && a.representation == b.representation &&
           a.ld == b.ld && a.variant == b.variant;
}

bool prepared_serves(const PreparedActivation* prepared, PreparedRepresentation representation,
                     const void* source, std::int32_t ldx, std::int32_t k,
                     std::int32_t tokens) noexcept {
    return prepared != nullptr && prepared->data != nullptr &&
           prepared->representation == representation && prepared->source == source &&
           prepared->ldx == ldx && prepared->k == k && prepared->tokens == tokens;
}

std::uint64_t ggml_mmq_fallback_count() noexcept {
    return mmq_fallbacks.load(std::memory_order_relaxed);
}

std::uint64_t instrumentation_joined_launches() noexcept {
    return joined_launches.load(std::memory_order_relaxed);
}

std::uint64_t route_demotion_count() noexcept {
    return route_demotions.load(std::memory_order_relaxed);
}

// Volta prefill (generic_volta_cutlass.cu): a tiled-route call of the prefill band (17 tokens on,
// kPrefillMinTokens) decodes its weight once and multiplies on CUTLASS's sm_70 GEMM. The tiled
// kernel (HMMA m8n8k4 on sm_70, generic_tiled.cuh) decodes a 128-row slice of the weight in each
// CTA, once per 128-token block: ceil(N / 128) CTAs a block, in waves of the SMs, each wave as long
// as one CTA's decode of 128 x K weights (0.37 .. 0.47 ms at K = 4096 on the V100). The decoded copy
// spreads the same decode over every SM, then reads two FP16 bytes a weight into the GEMM, so it
// costs in proportion to N and wins wherever the tile's grid leaves SMs idle or takes a second
// wave. V100 (ninferno_linear_bench, T = 32..240, the tile's time against the copy's): Q4_K 24576 x
// 4096 (192 CTAs: 2.4 waves) 1.36..2.29 ms against 1.20..1.40, 4096 x 12288 (32 CTAs) 1.07..1.31
// against 0.82..0.88, 4096 x 4096 0.36..0.44 against 0.27..0.28, Q5_K 4096 x 12288 1.13..1.37
// against 0.90..0.94. The one shape whose grid is a whole wave, 10240 x 4096 (80 CTAs on 80 SMs),
// keeps the tile ahead below 129 tokens (0.37..0.47 ms against 0.50) and loses from 129, two waves
// (0.82..0.93 against 0.52..0.54). A Qwen3.5 9B layer's four linears: 14 % less time at T = 64,
// 35 % at T = 192.
// The copy's narrowest margin above is 1.13 (24576 x 4096 below 129 tokens) at the V100's 870
// GB/s, and its time is its bytes over the DRAM rate while the tile's is its waves on the SMs (80 on
// every Volta part): a board under 870 / 1.13 GB/s (the Titan V's 653) or of unknown bandwidth
// keeps the tile below kVoltaTileMaxTokens, where two token blocks already decode the weight twice.
constexpr std::int32_t kVoltaCutlassMinTokens = 17;
constexpr std::int32_t kVoltaTileMaxTokens    = 256;
constexpr double kVoltaCopyMinDramGbps        = 870.0 / 1.13;

bool volta_cutlass_takes(std::int32_t tokens) {
    static PerDevice<bool> enabled;
    return tokens >= kVoltaCutlassMinTokens && enabled.get([](int) {
        if (!volta_cutlass_compiled()) return false;
        try {
            return !device_class().features.mma_tensor;
        } catch (const std::exception&) { return false; }
    });
}

// Whether a tiled format's call of `tokens` takes the copy rather than its tile on this board.
bool volta_copy_pays(std::int32_t tokens) {
    static PerDevice<bool> fast_dram;
    return tokens >= kVoltaTileMaxTokens || fast_dram.get([](int) {
        try {
            return device_class().dram_gbps() >= kVoltaCopyMinDramGbps;
        } catch (const std::exception&) { return false; }
    });
}

// Whether a TiledMma call of `tokens` runs on the decoded copy (and so needs its workspace): a
// CUTLASS-only format's (`only`: it has no tile of its own) from the prefill band's 17 tokens, a
// tiled format's where the copy pays.
bool volta_cutlass_runs(bool only, std::int32_t tokens) {
    return volta_cutlass_takes(tokens) && (only || volta_copy_pays(tokens));
}

// GemmArgs::spread_smem. The eight-warp decode CTA (w1x8, one token tile, 72 registers) fits
// three to an SM, and the block scheduler fills SMs that deep before it uses the rest: a call of
// at most two CTAs per SM then runs on as few as a third of the SMs' memory pipes. Reserving a
// third of an SM's shared memory per CTA holds it to two, spread over every SM. Same kernel, same
// bytes. RTX 5090 at T = 1 under PDL (a kernel micro-bench, L2-cold weights): NVFP4 q/k/v
// 4608x6656 11.8 -> 11.25 us, gate 4096x6656 11.1 -> 10.5 us; FP8 row-f32 20.2 -> 19.5 and 18.5
// -> 17.8 us. At 416 CTAs and more it loses (6656x4096 NVFP4 9.4 -> 12.6 us) and so does SwiGLU
// (two CTAs per SM already: 116 registers), which is why it is limited to thin Store calls. A
// joined call (the attention gate beside q/k/v) takes it too: Muse Glimmer 30B NVFP4 decode 79.07
// -> 79.97 t/s, and 78.93 with the chain head alone spread. Not Q8_G32: its six Muse layers side by
// side lose what they gained with it (79.99 -> 79.63 t/s). The boards it was measured on, and any
// other board by the nearest of them within measured::kCarryDistance (else off), never across
// instruction sets: what decides is how many SMs a thin call spreads over, which the boards' SM
// count, bandwidth and L2 carry. The RTX 5080 (84 SMs, same micro-bench): NVFP4 2048x4096 6.66
// -> 6.10 us, 2560x2560 4.93 -> 4.68 us, 2688x4096 7.43 -> 7.06 us; FP8 2048x4096 11.04 -> 10.20
// us, 2688x4096 13.21 -> 12.48 us; 64 CTAs (1024 rows) even. The RTX 5060 Ti (36 SMs; same
// micro-bench, NVFP4/FP8/FP8 row-f32 at 1024..6656 rows, spread against packed, three rounds): even
// on every problem, since only 1024 rows stay within two CTAs per SM there; the RTX 4080 SUPER:
// even.
namespace {
struct SpreadMeasured {
    Board board;
    bool faster;
};

constexpr SpreadMeasured kSpreadMeasured[] = {{measured::kRtx5090, true},
                                              {measured::kRtx5080, true},
                                              {measured::kRtx5060Ti, false},
                                              {measured::kRtx4080Super, false}};
static_assert(measured_boards_known(std::span(kSpreadMeasured)));

bool spread_by_board(const DeviceClass& d) {
    const SpreadMeasured* m =
        nearest_known_measured(std::span(kSpreadMeasured), board_of(d), measured::kCarryDistance);
    return m != nullptr && m->faster;
}

// The current device's gates, once per ordinal: a process may drive boards of different classes
// (an RTX 5090 beside a 4090), and each launch takes its own device's. Through PerDevice, so a
// substituted DeviceClass (ScopedDeviceClass: the selector goldens, route_query) reads its own
// board's gates, not the ones the host's GPU cached first. A device whose class cannot be read
// throws and is asked again next time (caching "none" would turn PDL and the spread off for the
// process on one transient failure).
struct DeviceGates {
    bool spread = false; // spread_smem measured faster on the nearest measured board
    bool pdl    = false; // programmatic dependent launch (pdl_prefetch)
};

const DeviceGates& device_gates() {
    static PerDevice<DeviceGates> gates;
    static const DeviceGates none{};
    // No current device (a host-only process without a substitute): no gates, as before.
    if (substituted_device_class_key() == nullptr) {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) {
            (void)cudaGetLastError();
            return none;
        }
    }
    return gates.get([](int) {
        const DeviceClass& d = device_class();
        return DeviceGates{spread_by_board(d), d.features.pdl};
    });
}

bool spread_measured() { return device_gates().spread; }

bool pdl_device() { return device_gates().pdl; }

} // namespace

int spread_smem(Route route, WeightCodec codec, Epilogue epilogue, const GemmArgs& args) {
    if (route.family != Family::RegisterMma || route.schedule != rmma_index(1, 8, 1, 1, 1) ||
        epilogue != Epilogue::Store || args.tokens > 8 ||
        !(codec == WeightCodec::Nvfp4 || codec == WeightCodec::Fp8 ||
          codec == WeightCodec::Fp8RowF32) ||
        !spread_measured()) {
        return 0;
    }
    try {
        const auto& device      = device_class();
        const std::int64_t ctas = (static_cast<std::int64_t>(args.rows) + 15) / 16;
        const std::size_t pad   = (device.smem_per_sm / 3 + 1024) / 1024 * 1024;
        if (ctas > 2 * static_cast<std::int64_t>(device.sms) || pad > kStaticSharedBytesPerBlock)
            return 0;
        return static_cast<int>(pad);
    } catch (const std::exception&) {
        return 0;
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

std::string_view overlap_refusal_name(OverlapRefusal why) noexcept {
    switch (why) {
    case OverlapRefusal::None: return "none";
    case OverlapRefusal::NotAsked: return "not_asked";
    case OverlapRefusal::Family: return "family";
    case OverlapRefusal::Epilogue: return "epilogue";
    case OverlapRefusal::PrequantizedR8: return "prequantized_r8";
    case OverlapRefusal::NoPdl: return "no_pdl";
    case OverlapRefusal::Wide: return "wide";
    case OverlapRefusal::Joined: return "joined";
    }
    return "none";
}

LaunchOverlap launch_overlap(const ResolvedRoute& resolved, Epilogue epilogue, int tokens,
                             bool join_asked, bool release_asked, bool pdl) {
    const Family family = resolved.executed.family;
    const bool r8_pre   = family == Family::R8Mma && resolved.r8_prequantized;
    // Only a register-MMA or R8 Store launch joins or releases; every other route runs in stream
    // order.
    const auto joinable = [&](bool asked) {
        if (!asked) return OverlapRefusal::NotAsked;
        if (family != Family::RegisterMma && family != Family::R8Mma) return OverlapRefusal::Family;
        return epilogue == Epilogue::Store ? OverlapRefusal::None : OverlapRefusal::Epilogue;
    };
    LaunchOverlap out;
    // The pre-quantized R8 form does not join either: its quantizer and kernel launch in stream
    // order (generic_r8.cuh kPre), so a join flag would only claim an overlap that never happens.
    out.join = joinable(join_asked);
    if (out.join == OverlapRefusal::None && r8_pre) out.join = OverlapRefusal::PrequantizedR8;
    // The pre-quantized R8 form does not release: the next call's quantizer could take the
    // workspace it still reads (its kernel ignores the flag too, generic_r8.cuh kPre). Without
    // PDL nothing starts early (run_route keeps the kernel's flag, a no-op there).
    out.release = joinable(release_asked);
    if (out.release == OverlapRefusal::None) {
        out.release = r8_pre ? OverlapRefusal::PrequantizedR8 : !pdl ? OverlapRefusal::NoPdl
                                                                     : OverlapRefusal::None;
    }
    // A decode-width register-MMA call that does not join starts its weight stream under the
    // previous kernel, where the device has programmatic dependent launch (sm_90 and up; Ada
    // launches in stream order as before); R8 does the same where its activation is quantized in
    // the kernel (not the pre-quantized form, whose quantizer runs before it).
    if (r8_pre) {
        out.prefetch = OverlapRefusal::PrequantizedR8;
    } else if (family != Family::RegisterMma && family != Family::R8Mma) {
        out.prefetch = OverlapRefusal::Family;
    } else if (out.joins()) {
        out.prefetch = OverlapRefusal::Joined;
    } else if (tokens > kPdlPrefetchMaxTokens) {
        out.prefetch = OverlapRefusal::Wide;
    } else {
        out.prefetch = pdl ? OverlapRefusal::None : OverlapRefusal::NoPdl;
    }
    return out;
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

R8Coverage r8_coverage(const Problem& problem, const Weight& w, std::int32_t tokens) {
    R8Coverage r;
    FamilyCoverage& c = r.family;
    c.declared        = r8_schedule_count();
    const bool reads  = r8_supports(w.qtype);
    for (int i = 0; i < c.declared; ++i) {
        const ScheduleRefusal why = r8_refusal(i, w, problem.policy);
        c.schedules.push_back({i, why});
        c.compiled += reads;
        c.legal += why == ScheduleRefusal::None;
    }
    c.heuristic             = heuristic_r8(problem, tokens);
    c.heuristic_refusal     = r8_refusal(c.heuristic, w, problem.policy);
    r.call                  = r8_call_refusal(w, problem.policy, tokens);
    r.prequantized_compiled = r8_verify_compiled(w.qtype);
    return r;
}

std::vector<FamilyAlternative> family_alternatives(const Problem& problem, const Weight& w,
                                                   std::int32_t tokens, bool prefill) {
    const bool band = prefill && tokens >= 17;
    // The calls heuristic_route prices (cost_route): 17 tokens or more outside the prefill profile,
    // NVFP4 apart. The times are cost_route's own, so a priced alternative compares as it would.
    std::vector<std::pair<Route, double>> priced;
    if (!band && tokens >= 17 && problem.qtype != QType::NVFP4) {
        price_routes(problem, w, tokens,
                     [&](Route r, double time) { priced.emplace_back(r, time); });
    }
    std::vector<FamilyAlternative> out;
    const auto take = [&](Family family, const FamilyCoverage& c, R8CallRefusal call) {
        FamilyAlternative a;
        a.family   = family;
        a.compiled = c.compiled;
        a.call     = call;
        for (const auto& [r, time] : priced) {
            if (r.family == family && (!a.seconds || time < *a.seconds)) {
                a.route   = r;
                a.seconds = time;
            }
        }
        if (!a.route && call == R8CallRefusal::None) {
            if (c.heuristic >= 0 && c.heuristic_refusal == ScheduleRefusal::None) {
                a.route = Route{family, static_cast<std::uint8_t>(c.heuristic)};
            } else {
                for (const ScheduleCoverage& s : c.schedules) {
                    if (s.refusal != ScheduleRefusal::None) continue;
                    a.route = Route{family, static_cast<std::uint8_t>(s.schedule)};
                    break;
                }
            }
        }
        if (!a.route) {
            a.refusal = c.heuristic >= 0      ? c.heuristic_refusal
                        : c.schedules.empty() ? ScheduleRefusal::NotDeclared
                                              : c.schedules.front().refusal;
        }
        out.push_back(a);
    };
    take(Family::RegisterMma, rmma_coverage(problem, w, tokens, band), R8CallRefusal::None);
    const R8Coverage r8 = r8_coverage(problem, w, tokens);
    take(Family::R8Mma, r8.family, r8.call);
    take(Family::W4a8Mma, w4a8_coverage(problem, w, tokens), R8CallRefusal::None);
    take(Family::I8Mma, i8_coverage(problem, w, tokens), R8CallRefusal::None);
    take(Family::TiledMma, tiled_coverage(w), R8CallRefusal::None);
    take(Family::StagedMma, staged_coverage(w), R8CallRefusal::None);
    for (const Family f :
         {Family::W4a4Mma, Family::Nvfp4TiledMma, Family::GgmlMmq, Family::GgmlVec}) {
        take(f, other_family_coverage(f, w, problem.policy), R8CallRefusal::None);
    }
    return out;
}

std::optional<double> route_cost_seconds(const Problem& problem, const Weight& w, Route route,
                                         std::int32_t tokens) {
    std::optional<double> out;
    price_routes(problem, w, tokens, [&](Route r, double time) {
        if (r.family == route.family && r.schedule == route.schedule) out = time;
    });
    return out;
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

void launch_rmma_codec(WeightCodec codec, int schedule, Epilogue epilogue, const GemmArgs& args,
                       cudaStream_t stream) {
    switch (codec) {
    case WeightCodec::Bf16:
        launch_rmma_bf16(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Fp8:
        launch_rmma_fp8(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Nvfp4:
        launch_rmma_nvfp4(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q4:
        launch_rmma_q4(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q5:
        launch_rmma_q5(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q6:
        launch_rmma_q6(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q3G128:
        launch_rmma_q3g128(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q3G64:
        launch_rmma_q3g64(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q8:
        launch_rmma_q8(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Fp8RowF32:
        launch_rmma_fp8_row_f32(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Fp8B128:
        launch_rmma_fp8_b128(schedule, epilogue, args, stream);
        return;
    case WeightCodec::U4Zp:
        launch_rmma_u4zp(schedule, epilogue, args, stream);
        return;
    case WeightCodec::U4ZpG32:
        launch_rmma_u4zp_g32(schedule, epilogue, args, stream);
        return;
    case WeightCodec::U4ZpG64:
        launch_rmma_u4zp_g64(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ4K:
        launch_rmma_ggml_q4k(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ5K:
        launch_rmma_ggml_q5k(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ6K:
        launch_rmma_ggml_q6k(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ3K:
        launch_rmma_ggml_q3k(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ2K:
        launch_rmma_ggml_q2k(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq4Xs:
        launch_rmma_ggml_iq4xs(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq3Xxs:
        launch_rmma_ggml_iq3xxs(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq3S:
        launch_rmma_ggml_iq3s(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ40:
        launch_rmma_ggml_q40(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ41:
        launch_rmma_ggml_q41(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ50:
        launch_rmma_ggml_q50(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ51:
        launch_rmma_ggml_q51(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq4Nl:
        launch_rmma_ggml_iq4nl(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq2Xxs:
        launch_rmma_ggml_iq2xxs(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq2Xs:
        launch_rmma_ggml_iq2xs(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq2S:
        launch_rmma_ggml_iq2s(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq1S:
        launch_rmma_ggml_iq1s(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlIq1M:
        launch_rmma_ggml_iq1m(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlTq10:
        launch_rmma_ggml_tq10(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlTq20:
        launch_rmma_ggml_tq20(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ10:
        launch_rmma_ggml_q10(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ20:
        launch_rmma_ggml_q20(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Mxfp4:
        launch_rmma_mxfp4(schedule, epilogue, args, stream);
        return;
    }
}

void run_route(Route route, const Call& ordered, WorkspaceArena* workspace, cudaStream_t stream,
               bool* released) {
    if (released != nullptr) *released = false;
    validate_call(ordered);
    const Weight& w = *ordered.weight;
    if (!supports(w)) throw std::invalid_argument("generic linear: unsupported weight layout");
    // An explicit route parses on every build; one whose kernels this build lacks is refused as
    // route_admits refuses it (the Ada build's W4A4).
    if (route.family == Family::W4a4Mma && route.schedule >= w4a4_schedule_count()) {
        throw std::invalid_argument("generic linear: route " + route_name(route) +
                                    " is not compiled in this build (NVFP4 MMA, sm_120a)");
    }
    // A weight stored in another column order multiplies the input gathered into that order.
    Call call = ordered;
    std::optional<WorkspaceArena::Scope> scope;
    Tensor gathered;
    if (w.k_order != nullptr) {
        if (workspace == nullptr) {
            throw std::invalid_argument(
                "generic linear: a weight stored in another column order needs caller workspace");
        }
        scope.emplace(workspace->scope());
        const int tokens = call.x->ne[1];
        gathered         = workspace->alloc(DType::BF16, {w.k, tokens});
        launch_gather_k(call.x->data, call.x->ne[0], w.k_order, w.k, tokens, gathered.data,
                        stream);
        call.x = &gathered;
    }
    GemmArgs args     = make_args(call);
    const Route asked = route;
    const ResolvedRoute resolved =
        resolve_route(route, w,
                      CallLayout{.x_aligned16    = aligned_to(args.x, 16),
                                 .ldx            = args.ldx,
                                 .tokens         = args.tokens,
                                 .rows           = call.rows,
                                 .segments       = args.segments,
                                 .epilogue       = call.epilogue,
                                 .workspace_free = workspace_free(workspace)});
    route             = resolved.executed;
    const bool r8_pre = resolved.r8_prequantized;
    // A caller that sized its workspace short of ggml's route (said below, with every other
    // demotion) is counted too (ggml_mmq_fallback_count).
    if (resolved.ggml_fallback) mmq_fallbacks.fetch_add(1, std::memory_order_relaxed);
    const bool gated     = call.epilogue == Epilogue::SwiGlu;
    const int plane_rows = gated ? 2 * call.rows : call.rows;
    if (route.family != asked.family && !same_bytes(asked, route)) {
        note_demotion(asked, route, call.rows, args.tokens, w.k, aligned_to(args.x, 16), args.ldx,
                      workspace != nullptr ? workspace->capacity() : 0);
    }
    // The launch's overlap with its neighbours (launch_overlap, #309). A joinable launch keeps its
    // release flag without PDL (the trigger is then a no-op), as before; the pre-quantized R8
    // form's is cleared, which its kernel ignored (kPre).
    const LaunchOverlap overlap = launch_overlap(resolved, call.epilogue, args.tokens,
                                                 call.join_previous, call.release_next,
                                                 pdl_device());
    args.join_previous = overlap.joins() ? 1 : 0;
    if (overlap.joins()) joined_launches.fetch_add(1, std::memory_order_relaxed);
    args.release_next  = overlap.releases() || overlap.release == OverlapRefusal::NoPdl ? 1 : 0;
    if (released != nullptr) *released = overlap.releases();
    args.pdl_prefetch = overlap.prefetches() ? 1 : 0;
    const WeightCodec codec = codec_of(w);
    args.spread_smem        = spread_smem(route, codec, call.epilogue, args);
    switch (route.family) {
    case Family::GgmlMmq: {
        auto scope       = workspace->scope();
        const auto type  = *mmq_type(w);
        const int slice  = std::min(args.tokens, kMmqPrefillSlice);
        const int bpr    = w.k / static_cast<int>(ggml_block_values(w.qtype));
        void* activation = workspace->alloc_bytes(ggml_mmq::activation_bytes(w.k, slice)).data;
        auto* plane      = static_cast<float*>(
            workspace->alloc_bytes(static_cast<std::size_t>(plane_rows) * slice * sizeof(float))
                .data);
        void* fixup        = workspace->alloc_bytes(ggml_mmq::fixup_bound(slice)).data;
        const auto rows_at = [&](int row) { return ggml_rows_at(args, w.qtype, bpr, row); };
        // A prefill call kMmqPrefillSlice tokens at a time through the one activation and plane.
        for (int t0 = 0; t0 < args.tokens; t0 += slice) {
            const auto first = static_cast<std::size_t>(t0);
            GemmArgs part    = args;
            part.tokens      = std::min(slice, args.tokens - t0);
            part.x           = static_cast<const __nv_bfloat16*>(args.x) + first * args.ldx;
            for (int seg = 0; seg < args.segments; ++seg) {
                part.out[seg] = static_cast<__nv_bfloat16*>(args.out[seg]) + first * args.ld[seg];
            }
            ggml_mmq::quantize_activation(type, static_cast<const __nv_bfloat16*>(part.x),
                                          args.ldx, w.k, part.tokens, activation, stream);
            ggml_mmq::product(type, rows_at(call.row_begin), bpr, call.rows, w.k, activation,
                              part.tokens, plane, plane_rows, fixup, stream);
            if (gated) {
                ggml_mmq::product(type, rows_at(call.row_begin + call.up_offset), bpr, call.rows,
                                  w.k, activation, part.tokens, plane + call.rows, plane_rows,
                                  fixup, stream);
            }
            launch_mmq_epilogue(call.epilogue, part, plane, plane_rows, stream);
        }
        return;
    }
    case Family::GgmlVec: {
        // Wider than the kernel's columns (vec_max_tokens): the call runs eight tokens at a time,
        // each part quantizing its own activation.
        if (args.tokens > ggml_mmq::kVecMaxColumns) {
            for (int begin = 0; begin < args.tokens; begin += ggml_mmq::kVecMaxColumns) {
                const int n     = std::min(ggml_mmq::kVecMaxColumns, args.tokens - begin);
                auto scope      = workspace->scope();
                const auto type = *mmq_type(w);
                const int bpr   = w.k / static_cast<int>(ggml_block_values(w.qtype));
                const auto* x   = static_cast<const __nv_bfloat16*>(args.x) +
                                static_cast<std::int64_t>(begin) * args.ldx;
                void* activation =
                    workspace->alloc_bytes(ggml_mmq::vec_activation_bytes(w.k, n)).data;
                ggml_mmq::vec_quantize_activation(x, args.ldx, w.k, n, activation, stream);
                ggml_mmq::VecOut out;
                out.epilogue  = gated ? ggml_mmq::VecOut::Epilogue::SwiGlu
                                : call.epilogue == Epilogue::AddResidual
                                    ? ggml_mmq::VecOut::Epilogue::AddResidual
                                    : ggml_mmq::VecOut::Epilogue::Store;
                out.gelu_gate = args.gelu_gate;
                out.relu2     = args.relu2;
                out.segments  = args.segments;
                for (int s = 0; s < args.segments; ++s) {
                    out.out[s] = static_cast<__nv_bfloat16*>(args.out[s]) +
                                 static_cast<std::int64_t>(begin) * args.ld[s];
                    out.segment_end[s] = args.segment_end[s];
                    out.ld[s]          = args.ld[s];
                }
                ggml_mmq::vec_product(
                    type, ggml_rows_at(args, w.qtype, bpr, call.row_begin),
                    gated ? ggml_rows_at(args, w.qtype, bpr, call.row_begin + call.up_offset)
                          : ggml_mmq::Planes{},
                    bpr, call.rows, w.k, activation, n, out, stream);
            }
            return;
        }
        auto scope       = workspace->scope();
        const auto type  = *mmq_type(w);
        const int tokens = args.tokens;
        const int bpr    = w.k / static_cast<int>(ggml_block_values(w.qtype));
        // The shared q8_1 where it was made from the x this call reads (after any column gather),
        // at its stride and extent; else this call's own.
        const void* activation =
            prepared_serves(call.prepared, PreparedRepresentation::GgmlQ8_1, args.x, args.ldx, w.k,
                            tokens)
                ? call.prepared->data
                : nullptr;
        if (activation == nullptr) {
            void* own = workspace->alloc_bytes(ggml_mmq::vec_activation_bytes(w.k, tokens)).data;
            ggml_mmq::vec_quantize_activation(static_cast<const __nv_bfloat16*>(args.x), args.ldx,
                                              w.k, tokens, own, stream);
            activation = own;
        }
        ggml_mmq::VecOut out;
        out.epilogue  = gated ? ggml_mmq::VecOut::Epilogue::SwiGlu
                        : call.epilogue == Epilogue::AddResidual
                            ? ggml_mmq::VecOut::Epilogue::AddResidual
                            : ggml_mmq::VecOut::Epilogue::Store;
        out.gelu_gate = args.gelu_gate;
        out.relu2     = args.relu2;
        out.segments  = args.segments;
        for (int s = 0; s < args.segments; ++s) {
            out.out[s]         = args.out[s];
            out.segment_end[s] = args.segment_end[s];
            out.ld[s]          = args.ld[s];
        }
        ggml_mmq::vec_product(
            type, ggml_rows_at(args, w.qtype, bpr, call.row_begin),
            gated ? ggml_rows_at(args, w.qtype, bpr, call.row_begin + call.up_offset)
                  : ggml_mmq::Planes{},
            bpr, call.rows, w.k, activation, tokens, out, stream);
        return;
    }
    case Family::Scalar:
        launch_scalar(codec, call.epilogue, args, stream);
        return;
    case Family::RegisterMma:
        if ((w.k % k_multiple(codec)) != 0) {
            // K off the register kernel's step runs scalar: a demotion (other bytes and speed).
            note_demotion(route, Route{Family::Scalar, 0}, call.rows, args.tokens, w.k,
                          aligned_to(args.x, 16), args.ldx,
                          workspace != nullptr ? workspace->capacity() : 0);
            launch_scalar(codec, call.epilogue, args, stream);
            return;
        }
        args.lane_local = invariant_routes_active() ? 0 : 1;
        args.add_sums   = rmma_add_sums() ? 1 : 0;
        if (!current_features().mma_tensor && volta_hmma_takes(codec, args)) {
            launch_volta_hmma(codec, call.epilogue, args, stream);
            return;
        }
        if (!current_features().mma_tensor && volta_gemv_takes(codec, args) &&
            (args.ldx % 8) == 0 && (reinterpret_cast<std::uintptr_t>(args.x) % 16) == 0) {
            launch_volta_gemv(codec, call.epilogue, args, stream);
            return;
        }
        launch_rmma_codec(codec, route.schedule, call.epilogue, args, stream);
        return;
    case Family::TiledMma:
        if ((w.k % 64) != 0) {
            throw std::invalid_argument("generic tiled: K must be whole 64-value tiles");
        }
        // The prefill band on Volta (kVoltaCutlassMinTokens): a CUTLASS-only format has no tile of
        // its own (V100, gpt-oss BF16 attention at T = 79 ran scalar, 7.3 ms a call, 360 -> 140
        // tok/s), and a tiled one's grid seldom fills the SMs' waves (volta_copy_pays).
        if (workspace != nullptr && volta_cutlass_runs(volta_cutlass_only(codec), args.tokens) &&
            launch_volta_cutlass(codec, call.epilogue, args, *workspace, stream)) {
            return;
        }
        // A call the decoded copy does not fit runs scalar: a demotion (other bytes and speed).
        if (volta_cutlass_only(codec)) {
            note_demotion(route, Route{Family::Scalar, 0}, call.rows, args.tokens, w.k,
                          aligned_to(args.x, 16), args.ldx,
                          workspace != nullptr ? workspace->capacity() : 0);
            launch_scalar(codec, call.epilogue, args, stream);
            return;
        }
        launch_tiled(route.schedule, codec, call.epilogue, args, stream);
        return;
    case Family::StagedMma:
        launch_staged(route.schedule, call.epilogue, args, stream);
        return;
    case Family::Nvfp4TiledMma:
#ifdef NINFERNO_WITHOUT_NVFP4
        throw std::invalid_argument("generic NVFP4 tiled GEMM is an sm_120a path on this build");
#else
        launch_nvfp4_tiled(nvfp4_tiled_schedule_info(route.schedule).classes, call.epilogue, args,
                           stream);
#endif
        return;
    case Family::W4a4Mma:
#ifdef NINFERNO_WITHOUT_NVFP4
        throw std::logic_error(
            "generic W4A4 needs the NVFP4 MMA (sm_120a); route selection must not offer it on Ada");
#else
    {
        if (workspace == nullptr) {
            throw std::invalid_argument("generic W4A4 requires caller workspace");
        }
        auto scope         = workspace->scope();
        const auto planes  = activation_planes(route, args.k, args.tokens);
        const DeviceSpan c = workspace->alloc_bytes(planes.codes, 256);
        const DeviceSpan s = workspace->alloc_bytes(planes.scales, 256);
        launch_w4a4(route.schedule, call.epilogue, args, w.input_scale_divisor,
                    static_cast<std::uint8_t*>(c.data), static_cast<std::uint8_t*>(s.data),
                    stream);
        return;
    }
#endif
    case Family::W4a8Mma: {
        if (workspace == nullptr) {
            throw std::invalid_argument("generic W4A8 requires caller workspace");
        }
        auto scope          = workspace->scope();
        const auto planes   = activation_planes(route, args.k, args.tokens);
        const DeviceSpan c  = workspace->alloc_bytes(planes.codes, 256);
        const DeviceSpan sc = workspace->alloc_bytes(planes.scales, 256);
        const DeviceSpan su = workspace->alloc_bytes(planes.sums, 256);
        launch_w4a8(route.schedule, codec, call.epilogue, args, static_cast<std::int8_t*>(c.data),
                    static_cast<float*>(sc.data), static_cast<float*>(su.data), kW4a8TokenMajor,
                    stream);
        return;
    }
    case Family::I8Mma: {
        if (workspace == nullptr) {
            throw std::invalid_argument("generic I8 requires caller workspace");
        }
        auto scope          = workspace->scope();
        const auto planes   = activation_planes(route, args.k, args.tokens);
        const DeviceSpan c  = workspace->alloc_bytes(planes.codes, 256);
        const DeviceSpan sc = workspace->alloc_bytes(planes.scales, 256);
        const DeviceSpan su = workspace->alloc_bytes(planes.sums, 256);
        launch_i8(route.schedule, i8_bits(w.qtype), call.epilogue, args,
                  static_cast<std::int8_t*>(c.data),
                  static_cast<float*>(sc.data), static_cast<int*>(su.data), stream);
        return;
    }
    case Family::R8Mma:
        if (r8_pre) {
            auto scope          = workspace->scope();
            const auto planes   = activation_planes(route, args.k, args.tokens);
            const DeviceSpan c  = workspace->alloc_bytes(planes.codes, 256);
            const DeviceSpan sc = workspace->alloc_bytes(planes.scales, 256);
            const DeviceSpan su = workspace->alloc_bytes(planes.sums, 256);
            auto* codes  = static_cast<std::int8_t*>(c.data);
            auto* scales = static_cast<float*>(sc.data);
            auto* sums   = static_cast<float*>(su.data);
            launch_w4a8_quantize(args, codes, scales, sums, stream);
            launch_r8(route.schedule, codec, call.epilogue, args, stream,
                      W4a8Activation{codes, scales, sums});
            return;
        }
        launch_r8(route.schedule, codec, call.epilogue, args, stream);
        return;
    }
}

void run(const Call& call, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream,
         bool* released) {
    const Problem problem{call.weight->qtype, policy, call.rows, call.weight->k, call.epilogue};
    record_problem(problem, *call.weight);
    tune_dense(problem, call, stream);
    const Route route =
        workspace_free_route(select_route(problem, *call.weight, call.x->ne[1], prefill_routes_active()),
                             problem, *call.weight, call.x->ne[1], prefill_routes_active(),
                             workspace != nullptr);
    run_route(route, call, workspace, stream, released);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena* workspace, cudaStream_t stream) {
    Call call;
    call.x               = &x;
    call.weight          = &w;
    call.rows            = w.n;
    call.epilogue        = Epilogue::Store;
    call.out[0]          = &out;
    call.segment_rows[0] = w.n;
    run(call, policy, workspace, stream);
}

void linear_relu2(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    Call call;
    call.x               = &x;
    call.weight          = &w;
    call.rows            = w.n;
    call.epilogue        = Epilogue::Store;
    call.relu2           = true;
    call.out[0]          = &out;
    call.segment_rows[0] = w.n;
    run(call, policy, workspace, stream);
}

bool linear_release_next(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                         WorkspaceArena* workspace, cudaStream_t stream) {
    Call call;
    call.x               = &x;
    call.weight          = &w;
    call.rows            = w.n;
    call.epilogue        = Epilogue::Store;
    call.out[0]          = &out;
    call.segment_rows[0] = w.n;
    call.release_next    = true;
    bool released        = false;
    run(call, policy, workspace, stream, &released);
    return released;
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena* workspace, cudaStream_t stream) {
    Call call;
    call.x               = &x;
    call.weight          = &w;
    call.rows            = w.n;
    call.epilogue        = Epilogue::AddResidual;
    call.out[0]          = &residual;
    call.segment_rows[0] = w.n;
    run(call, policy, workspace, stream);
}

namespace {

void run_gated(const Tensor& x, const Weight& gate_up, Tensor& out, LinearPolicy policy,
               WorkspaceArena* workspace, cudaStream_t stream, bool gelu) {
    if ((gate_up.n % 2) != 0) {
        throw std::invalid_argument("generic linear_swiglu: gate/up rows must be even");
    }
    Call call;
    call.x               = &x;
    call.weight          = &gate_up;
    call.rows            = gate_up.n / 2;
    call.up_offset       = gate_up.n / 2;
    call.epilogue        = Epilogue::SwiGlu;
    call.gelu_gate       = gelu;
    call.out[0]          = &out;
    call.segment_rows[0] = gate_up.n / 2;
    run(call, policy, workspace, stream);
}

} // namespace

void linear_swiglu(const Tensor& x, const Weight& gate_up, Tensor& out, LinearPolicy policy,
                   WorkspaceArena* workspace, cudaStream_t stream) {
    run_gated(x, gate_up, out, policy, workspace, stream, false);
}

void linear_geglu(const Tensor& x, const Weight& gate_up, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    run_gated(x, gate_up, out, policy, workspace, stream, true);
}

} // namespace ninferno::ops::detail::generic
