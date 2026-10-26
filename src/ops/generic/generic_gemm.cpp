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
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/r8_decode_table.h"
#include "ops/generic/route_table.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
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

// Mirrors NINFERNO_GENERIC_RMMA_SCHEDULES (generic_gemm_launch.cuh): warps_m, warps_k, m_tiles,
// n_tiles, cluster.
constexpr std::array<RmmaScheduleInfo, 20> kRmmaSchedules{{
    {1, 4, 1, 1},
    {1, 8, 1, 1},
    {2, 2, 1, 1},
    {4, 1, 1, 1},
    {4, 1, 2, 1},
    {2, 2, 1, 2},
    {4, 1, 1, 2},
    {4, 1, 2, 2},
    {4, 1, 1, 4},
    {4, 1, 2, 4},
    {4, 1, 1, 8},
    {4, 1, 2, 8},
    {2, 1, 2, 8},
    {8, 1, 1, 8},
    {1, 8, 1, 4},
    {1, 4, 1, 4},
    {1, 8, 1, 1, 8},
    {1, 4, 1, 2},
    // K over a CTA pair: decode calls of at most one CTA per SM (heuristic_rmma).
    {1, 8, 1, 1, 2},
    // K over sixteen warps of one CTA: the same calls where clusters do not exist (sm_86, sm_89).
    {1, 16, 1, 1},
}};

// Mirrors NINFERNO_GENERIC_W4A4_SCHEDULES (generic_w4a4.cu): block_m, block_n.
constexpr std::array<W4a4ScheduleInfo, 4> kW4a4Schedules{{
    {32, 64},
    {32, 128},
    {64, 128},
    {128, 128},
}};

// Mirrors NINFERNO_GENERIC_W4A8_SCHEDULES (generic_w4a8.cuh): block_m (tokens), block_n (rows),
// producer warps (0: every warp loads and multiplies).
constexpr std::array<W4a8ScheduleInfo, 6> kW4a8Schedules{{
    {32, 128, 0},
    {64, 128, 0},
    {128, 128, 0},
    {128, 64, 0},
    {64, 64, 0},
    {128, 64, 4},
}};

// Mirrors NINFERNO_GENERIC_R8_SCHEDULES (generic_r8.cuh): warps per CTA, rows per warp.
constexpr std::array<R8ScheduleInfo, 4> kR8Schedules{{
    {4, 1},
    {4, 2},
    {8, 2},
    {8, 1},
}};

// Mirrors NINFERNO_GENERIC_TILED_SCHEDULES (generic_tiled.cu): block_m (tokens), block_n (rows).
constexpr std::array<TiledScheduleInfo, 3> kTiledSchedules{{
    {128, 128},
    {64, 128},
    {128, 64},
}};

// Mirrors NINFERNO_GENERIC_STAGED_SCHEDULES (generic_staged.cu): rows, tokens, warps_m, warps_n,
// stages.
constexpr std::array<StagedScheduleInfo, 13> kStagedSchedules{{
    {128, 128, 2, 4, 3},
    {128, 64, 2, 2, 4},
    {64, 128, 1, 4, 4},
    {128, 128, 2, 2, 3},
    {256, 128, 4, 2, 2},
    {64, 64, 2, 2, 4},
    {128, 64, 2, 2, 2}, // 48 KiB: two CTAs per SM, for grids of fewer tiles than twice the SMs
    {64, 128, 1, 4, 2},
    {64, 64, 2, 2, 3},
    {128, 128, 2, 4, 2}, // 64 KiB: two CTAs per SM where an SM holds 128 KiB
    {128, 128, 2, 4, 5}, // 160 KiB, 144 KiB, 144 KiB: only where one block holds them (GA100, Orin)
    {256, 128, 4, 2, 3},
    {128, 256, 2, 4, 3},
}};

// Mirrors NINFERNO_GENERIC_NVFP4_TILED_SCHEDULES (generic_nvfp4_tiled.cu): classes (the register
// MMA's WarpsK whose order it reproduces), block_m (tokens), block_n (rows).
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

// The K multiple the register MMA takes: whole steps, except the ggml 32-value blocks, whose
// decoder reads a step's missing blocks as zero and whose groups mma_step stops at K
// (kGroupTail): whole blocks.
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
bool allows_w4a8(const Weight& w, LinearPolicy policy) {
    return allows_a8(policy) && w4a8_supports(w.qtype) && w4a8_k_admits(w.qtype, w.k) &&
           aligned_to(w.qdata, 16) &&
           (w.qtype != QType::NVFP4 || nvfp4_prefill_a8_policy(w.qtype, policy));
}

// I8: Q4 G64 with A8 allowed, K in whole 512-value runs (the weight scales of eight groups are
// one 16-byte copy per row), 16-byte code and scale planes.
bool allows_i8(const Weight& w, LinearPolicy policy) {
    const bool q4 = w.qtype == QType::Q4_G64_FP16, q5 = w.qtype == QType::Q5_G64_FP16,
               q6 = w.qtype == QType::Q6_G64_FP16;
    return allows_a8(policy) && (q4 || q5 || q6) && (w.k % 512) == 0 &&
           (w.padded_shape[1] % 512) == 0 && aligned_to(w.qdata, 16) && aligned_to(w.scales, 16) &&
           (q4 || aligned_to(w.qhigh, q5 ? 8 : 16));
}
// The code bits of an I8 weight.
int i8_bits(QType q) { return q == QType::Q5_G64_FP16 ? 5 : q == QType::Q6_G64_FP16 ? 6 : 4; }

// The formats the producer-warp W4A8 tile is compiled for (w4a8_ws_compiled, generic_w4a8.cuh).
bool w4a8_ws_supports(QType q) {
    return q == QType::GGML_IQ3_XXS || q == QType::GGML_IQ2_S || q == QType::GGML_Q4_K ||
           q == QType::GGML_Q6_K || q == QType::GGML_Q4_0 || q == QType::GGML_IQ4_XS ||
           q == QType::GGML_IQ2_XS || q == QType::GGML_IQ2_XXS || q == QType::GGML_IQ3_S;
}
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

// R8: a format with an R8 codec, whole load groups of K, 16-byte code rows, A8 allowed. A lane
// strides K by groups, so K need not fill 128-value tiles: Gemma 4 26B's dense down projection
// (K = 2112) and gpt-oss's hidden (2880) take R8 at decode instead of the A16 register MMA.
bool allows_r8(const Weight& w, LinearPolicy policy) {
    return allows_a8(policy) && r8_supports(w.qtype) && (w.k % r8_k_values(w.qtype)) == 0 &&
           aligned_to(w.qdata, 16);
}

// Whether a call of T tokens takes R8 (decode, T <= kR8DecodeTokens, and the activation fits).
bool r8_call(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    if (!allows_r8(w, policy)) return false;
    if (tokens <= kR8DecodeTokens) return r8_fits(w.qtype, w.k, tokens);
    return tokens <= kR8VerifyTokens && r8_verify_takes_shape(w.qtype, w.n, w.k);
}

// The ggml block formats ggml's matrix kernel reads (ggml_mmq.h) in the layout NInferno stores
// them: Q4_K and Q5_K blocks unchanged, the others as planes.
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

// The token band of the ggml matrix kernel's route: the verify widths between R8's
// (kR8VerifyTokens) and the W4A8 tiles' (17 tokens and up), which ran the A16 register MMA.
// The dense tuner times it there against the register MMA on the card and the model's own weight
// (generic_tune.cpp): no fixed rule holds across chips and formats. Against the heuristic's tile,
// serve C4: RTX 5090 ISTA IQ3_S +10 % at 16 tokens; Muse Q4_K_M -7..14 %, RTX 4080 SUPER ISTA
// -12 % at 8; RTX 5080 IQ3_S even. Below the band it loses to R8 (5090 C1 -11 %).
constexpr std::int32_t kMmqMinTokens = kR8VerifyTokens + 1;
constexpr std::int32_t kMmqMaxTokens = 16;
// A prefill call on ggml's matrix kernel (mmq_prefill_wins) runs this many tokens at a time: its
// FP32 plane (a 27B SwiGLU's gate and up rows: 2 x 17408 x 512 x 4 bytes = 71 MB) instead of the
// chunk's, and 512 columns of 128 give every SM of a 16 GB card whole tiles (no stream-k fixup).
constexpr std::int32_t kMmqPrefillSlice = 512;

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

bool mmq_admits(const Weight& w, LinearPolicy policy) {
    const auto type = mmq_type(w);
    if (!type || !ggml_mmq::matrix_supported(*type) || !allows_a8(policy) ||
        (w.k % ggml_mmq::kKMultiple) != 0) {
        return false;
    }
    return ggml_layout_admits(w) && ggml_mmq::device_supported();
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
// between the 5060 Ti and the 5090 did (an RTX 5080 or 5070 Ti the 5090's, an RTX 5070 or 5060 the
// 5060 Ti's); Volta and the architectures nobody measured keep the tiles.
constexpr measured::Gate kMmqPrefillPays[] = {
    {measured::kRtx5090, true},    {measured::kRtx5060Ti, false}, {measured::kRtx4090, true},
    {measured::kL4, true},         {measured::kRtx4060Ti, true},  {measured::kA10, true},
    {measured::kA100Sxm40, false}, {measured::kRtx2080Ti, true}};

bool mmq_prefill_device() {
    return measured::measured_gate(kMmqPrefillPays, std::numeric_limits<double>::max());
}

bool mmq_prefill_wins(const Problem& p, const Weight& w) {
    return mmq_prefill_device() && ggml_mmq_format(p.qtype) &&
           (p.n % ggml_mmq::kRowMultiple) == 0 && mmq_admits(w, p.policy);
}

// ggml's vector kernel: dp4a (sm_61 and up, below every device NInferno runs on), K whole blocks.
bool vec_admits(const Weight& w, LinearPolicy policy) {
    const auto type = mmq_type(w);
    if (!type || !ggml_mmq::vec_supported(*type) || !allows_a8(policy) ||
        (w.k % static_cast<std::int32_t>(ggml_block_values(w.qtype))) != 0) {
        return false;
    }
    return ggml_layout_admits(w);
}

// Without tensor mma.sync (Volta) ggml's matrix kernel is out (device_supported: sm_75 on) and the
// register-MMA tiles run on FP32 FMAs, so a call of 9..16 tokens (a deep speculative verify) runs
// the vector kernel eight columns at a time (V100, Gemma 4 31B Q4_K_XL MTP, draft window 10: the
// emulated tiles 1.0..2.6 ms a call, 67 % of the run).
std::int32_t vec_max_tokens() {
    try {
        const auto& f = device_class().features;
        return f.mma_tensor && !f.mma_slow ? ggml_mmq::kVecMaxColumns
                                           : 2 * ggml_mmq::kVecMaxColumns;
    } catch (const std::exception&) { return ggml_mmq::kVecMaxColumns; }
}

bool vec_call(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= 1 && tokens <= vec_max_tokens() && vec_admits(w, policy);
}

// The verify widths the vector kernel takes by rule. Decode (T = 1) keeps R8 and the dense tuner:
// on the RTX 2080 Ti the vector kernel lost decode by 10 % (9B Q4_K_M C1 88 -> 79 tok/s) and won
// C4 by 10 %. Its integer dot products grow with the width on the CUDA cores while the register
// MMA's do not, so with tensor mma.sync the register MMA takes the call from five tokens: in the
// dense tuner's timings
// of eight cards the vector kernel was within 4 % of the best route at T = 2..4 and behind it at
// T = 8 on all of them, the most where a card has the least compute per byte of bandwidth
// (A100: x1.5..1.6, RTX 3090 x1.4..1.5, RTX 5090 x1.1, RTX 4090 x1.03).
// Without tensor mma.sync (Volta) the register MMA runs on FP32 FMAs, so the vector kernel takes
// every width it reads (vec_max_tokens): V100, Qwen3.8 27B UD-Q4_K_XL, the register MMA at T = 5..16
// about 0.5 ms a call against 0.08 for the vector kernel at T = 4; C8 aggregate 37.3 tok/s against
// C4 86.3, DFlash2 (verify 6) C1 15.7 against plain 30.1.
constexpr std::int32_t kVecRuleMaxTokens = 4;

std::int32_t vec_rule_max_tokens() {
    // vec_max_tokens() reaches past eight columns only without tensor mma.sync.
    const std::int32_t widest = vec_max_tokens();
    return widest > ggml_mmq::kVecMaxColumns ? widest : kVecRuleMaxTokens;
}

bool vec_rule(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= 2 && tokens <= vec_rule_max_tokens() && vec_call(w, policy, tokens);
}

// What a call of ggml's kernels leaves for the arena's 256-byte alignment of its first buffer: the
// capacity bound counts it too, or a workspace sized to the bound ran the call A16 (a vector call
// of 200 rows x 8 tokens: 11520 bytes needed and given).
constexpr std::size_t kGgmlArenaSlack = 256;

std::size_t vec_workspace_bytes(std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(ggml_mmq::vec_activation_bytes(k, tokens), 256);
    return layout.peak_bytes(1);
}

// The planes of ggml weight rows from `row` of the call's parent.
ggml_mmq::Planes ggml_rows_at(const GemmArgs& args, QType qtype, int bpr, int row) {
    const auto bytes  = mmq_plane_bytes(qtype);
    const auto blocks = static_cast<std::int64_t>(row) * bpr;
    ggml_mmq::Planes p;
    p.codes = args.codes + blocks * bytes.codes;
    if (bytes.high > 0) p.high = args.high + blocks * bytes.high;
    if (bytes.scales > 0) p.scales = args.scales + blocks * bytes.scales;
    return p;
}

// Whether a call of `rows` output rows and T tokens takes the ggml matrix kernel.
bool mmq_call(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens) {
    return tokens >= kMmqMinTokens && tokens <= kMmqMaxTokens &&
           (rows % ggml_mmq::kRowMultiple) == 0 && mmq_admits(w, policy);
}

// Speculative verify (T 9..16) of the codebook formats takes ggml's matrix kernel on
// every device, ahead of the route tables (which timed A16 schedules only) and the A16 rules: its
// int8 tile decodes a weight once per call where the register MMA's dequant runs per token tile.
// End to end, ISTA IQ3_S 27B C4 MTP3 (16-token verify), every iq / K-quant band 9..16 on mmq:
// L4 79.4 -> 89.2 tok/s (+12.3 %), RTX 5070 Ti 242.7 -> 270.9 (+11.6 %); ninfer-all runs
// mul_mat_q<16> there (92.9 / 280.1). The route_tune grid, one call with L2 flushed, does not show
// it (L4 rule 1.06 of its best). Where the BF16 MMA is emulated (Turing) every mmq format takes
// it, as before.
bool bf16_mma_native();
// The codebook formats only: their register-MMA dequant looks every value up in a grid table per
// token tile, which the matrix kernel's one decode a call saves. The K-quants' affine dequant is
// cheap, and there the register MMA keeps the band (RTX 4090, cellgt ABBA: UD-Q4_K_XL 27B DFlash2
// C1 115.7 -> 111.7 and C4 271.5 -> 263.1 with them in the rule, Muse 30B Q4_K_M C4 282.7 -> 267.6).
bool mmq_verify_format(QType q) {
    switch (q) {
    case QType::GGML_IQ1_S: case QType::GGML_IQ1_M: case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS: case QType::GGML_IQ2_S: case QType::GGML_IQ3_XXS: case QType::GGML_IQ3_S:
    case QType::GGML_IQ4_XS: case QType::GGML_IQ4_NL:
        return true;
    default:
        return false;
    }
}
constexpr std::int32_t kMmqVerifyMinTokens = 9;
bool mmq_verify_rule(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens) {
    // The measured band is 9..16 tokens of several sequences (a C4 verify). One sequence keeps its
    // register MMA at every width (ops::round_lanes): a C1 MTP round of 5..8 tokens and a C1 copy
    // round of 16 lost on mmq (RTX 5070 Ti C1 117.0 -> 115.2 tok/s, all of it in the copy rounds:
    // with them off, 116.5 both ways), while the C4 verify gains (233.5 -> 249.3).
    return mmq_call(w, policy, rows, tokens) &&
           (!bf16_mma_native() || (tokens >= kMmqVerifyMinTokens && ops::round_lanes() != 1 &&
                                   mmq_verify_format(w.qtype)));
}

std::size_t mmq_workspace_bytes(std::int32_t k, std::int32_t plane_rows, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(ggml_mmq::activation_bytes(k, tokens), 256);
    (void)layout.alloc_bytes(static_cast<std::size_t>(plane_rows) * tokens * sizeof(float), 256);
    (void)layout.alloc_bytes(ggml_mmq::fixup_bound(tokens), 256);
    return layout.peak_bytes(1);
}

// Whether mmq_workspace_bytes' layout fits the rest of the arena. Its blocks are 256-aligned, so
// from the arena's next 256-byte address they land exactly as the layout counted them: an arena
// the planner sized to the route holds it with nothing to spare.
bool mmq_fits(const DeviceArena& workspace, std::size_t bytes) {
    const auto base = reinterpret_cast<std::uintptr_t>(workspace.base());
    const auto start =
        ((base + workspace.used() + 255U) & ~static_cast<std::uintptr_t>(255U)) - base;
    return start <= workspace.capacity() && bytes <= workspace.capacity() - start;
}

std::atomic<std::uint64_t> mmq_fallbacks{0};
std::atomic<std::uint64_t> route_demotions{0};

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

// The token-major grid of the 128-token W4A8 tiles (W4a8Activation::token_major) on the RTX 4090
// (sm_89), where the row-major grid streams a large matrix from DRAM once per token tile: Muse 30B
// Q4_K_M shapes (a W4A8 micro-bench, T = 1024 / 512 / 200, time against the row-major grid):
// classic 128x64 Q4_K SwiGLU 19968x6656 0.94 / 0.96 / 0.99, Store 6656x19968 0.97 / 0.97 / 0.98,
// the producer-warp tile Q6_K Store and AddResidual 6656x19968 0.92-0.96; no shape slower (Q5_K
// 6656x4096, 4096x6656 and 256x6656 within 0.5 %). Gemma 4 31B UD-IQ3_XXS (IQ3_XXS and IQ2_S on
// the producer-warp tile, the classic tile's formats besides) prefill 1865 -> 1891 t/s
// at 22.6-31.4k tokens (A B B A, every slice ahead), though per call (the micro-bench at its
// shapes) IQ2_S and IQ4_XS SwiGLU run 1.02-1.05 and IQ3_S down 1.04 against 0.93 for IQ3_XXS and
// IQ3_S SwiGLU: the product decides, every format keeps it. The order of the CTAs changes no bit.
// Every GPU, then: the RTX 5090 (gb202: Gemma 4 31B UD-Q4_K_XL prefill of a 16K prompt 2508 /
// 2484 -> 2595 / 2587 t/s, D TM TM D) gains as the 4090 did. The row-major order stays in the
// kernel for the micro-bench's comparison.
constexpr bool kW4a8TokenMajor = true;

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
bool light_ggml_step(QType q) {
    switch (q) {
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_S:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
        return true;
    default:
        return false;
    }
}

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
bool tall_tile_format(QType q) { return q == QType::GGML_Q4_K; }

bool bf16_mma_native();

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
                    return rmma_index(1, 4, 1, 1);
                }
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
    if (q != QType::GGML_IQ4_XS && q != QType::GGML_IQ2_XS && q != QType::GGML_IQ2_XXS &&
        q != QType::GGML_IQ3_S) {
        return false;
    }
    static PerDevice<bool> on;
    return on.get([](int) {
        try {
            const auto& d = device_class();
            return (d.major == 12 && d.minor == 0) || (d.major == 8 && d.minor == 9);
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
        blackwell     = d.major == 12 && d.minor == 0;
        ada           = d.major == 8 && d.minor == 9;
        ampere        = d.major == 8 && d.minor < 9;
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
        if (d.major != 8 || d.minor != 9) return false;
        const std::int64_t rows_per_block = p.epilogue == Epilogue::SwiGlu ? 64 : 128;
        const std::int64_t ctas =
            ((p.n + rows_per_block - 1) / rows_per_block) * ((tokens + 127) / 128);
        return 4 * ctas >= 3 * static_cast<std::int64_t>(d.sms);
    } catch (const std::exception&) {
        return false;
    }
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
int rules_r8(const Problem& p, std::int32_t tokens) {
    if (p.qtype == QType::Q8_G32_FP16) return r8_index(4, 1);
    // IQ1_S / IQ1_M: grid-lookup decoders of IQ2's weight, scheduled with them.
    const bool iq2 = p.qtype == QType::GGML_IQ2_XXS || p.qtype == QType::GGML_IQ2_XS ||
                     p.qtype == QType::GGML_IQ2_S || p.qtype == QType::GGML_IQ1_S ||
                     p.qtype == QType::GGML_IQ1_M;
    if (tokens >= 2) return iq2 && (tokens >= 4 || p.n >= 34816) ? r8_index(4, 2) : r8_index(4, 1);
    // Turing (the RTX 2080 Ti, tu102:68sm, ninferno_r8_schedule_bench on the IQ3 16 GB
    // artifacts' 63 problems: picks +6.25 % above the best schedules in sum) decodes the codebook
    // formats and Q3_K slower per word, and from K = 5120 eight warps of two rows win at every N
    // and epilogue (IQ3_XXS 17408x5120 w4r1 110.2 -> w8r2 84.5 us, IQ3_S 16384x5120 131.3 -> 114.2,
    // Q3_K 5120x17408 add 112.9 -> 89.0); Q6_K gains two rows per warp from N = 8192 or K = 4096
    // (the 248320x2048 lm_head 807 -> 723 us, 2816x8192 w8r2 42.3 -> w4r2 40.3).
    if (!device_class().features.mma_bf16_native) {
        const bool decode_heavy = iq2 || p.qtype == QType::GGML_IQ3_S ||
                                  p.qtype == QType::GGML_IQ3_XXS || p.qtype == QType::GGML_Q3_K;
        if (decode_heavy && p.k >= 5120) return r8_index(8, 2);
        if (p.qtype == QType::GGML_Q6_K && p.epilogue != Epilogue::SwiGlu &&
            (p.n >= 8192 || p.k >= 4096))
            return r8_index(4, 2);
    }
    if (p.qtype == QType::GGML_Q3_K && p.k >= 8192) return r8_index(4, 1);
    if (p.k >= 8192 || (iq2 && p.n >= 12288)) return r8_index(8, 2);
    if (p.epilogue != Epilogue::SwiGlu && p.k >= 5120 && device_class().major == 12) {
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
            return nearest_measured(std::span(kR8DecodeTables), board_of(device_class()),
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

int heuristic_r8(const Problem& p, std::int32_t tokens) {
    if (tokens == 1) {
        if (const auto tuned = tuned_r8(p)) return *tuned;
    }
    return rules_r8(p, tokens);
}

// NINFERNO_ROUTE_RECORD=<file>: append every distinct problem that reaches the generic path, in
// ninferno_route_tune's --shapes format, so a model's shapes come from running it.
void record_problem(const Problem& p) {
    static const char* path = std::getenv("NINFERNO_ROUTE_RECORD");
    if (path == nullptr || *path == 0) return;
    static std::mutex mutex;
    static std::set<std::tuple<int, int, int, int, int>> seen;
    const auto compute = compute_class(p.qtype, p.policy);
    std::lock_guard lock(mutex);
    if (!seen.insert({static_cast<int>(p.qtype), static_cast<int>(compute),
                      static_cast<int>(p.epilogue), p.n, p.k})
             .second) {
        return;
    }
    std::ofstream(path, std::ios::app) << qtype_name(p.qtype) << ' ' << compute_class_name(compute)
                                       << ' ' << epilogue_name(p.epilogue) << ' ' << p.n << ' '
                                       << p.k << '\n';
}

} // namespace

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
bool r8_supports(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::MXFP4:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ4_XS:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::Q8_G32_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    // The u4 zero-point formats decode as in W4A8 (W4a8U4ZpG): before, a u4 checkpoint's decode
    // stayed on the A16 register MMA wherever its other formats took R8.
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    // IQ1_S / IQ1_M decode as in W4A8 (W4a8Iq1S / W4a8Iq1M): before, their decode stayed on the
    // A16 register MMA (27B UD-IQ2_XXS on the RTX 2080 Ti, nsys: the IQ1_S rmma calls 7.2 + 4.0 +
    // 2.6 ms of a 45 ms token).
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
        return true;
    // NINFERNO-SPECIAL(arch:no-fp4-mma): without the native NVFP4 kernels decode multiplies the
    // doubled E2M1 codes on dp4a (W4a8Nvfp4, as the W4A8 prefill) instead of the BF16 register
    // MMA. sm_120a keeps its native decode.
    case QType::NVFP4:
        return !kNativeNvfp4;
    default:
        return false;
    }
}

std::int32_t r8_k_values(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
        return 64;
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
        return 128;
    default:
        return 32;
    }
}

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
bool r8_verify_supports(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q4_0:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
        return true;
    default:
        return false;
    }
}

bool r8_verify_compiled(QType qtype) {
    return r8_supports(qtype) && qtype != QType::GGML_Q3_K && qtype != QType::GGML_Q2_K;
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
    switch (qtype) {
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::MXFP4:
    case QType::Q8_G32_FP16:
    case QType::U4_G32_ZP_FP16:
        return k % 32 == 0;
    // The 64-value groups: the zero tail starts on a group boundary (a K of whole groups past the
    // last 128-value tile, e.g. gpt-oss's 2880 in a native G64 conversion).
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::U4_G64_ZP_FP16:
        return k % 64 == 0;
    default:
        return false;
    }
}

bool w4a8_supports(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::MXFP4:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ4_XS:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16: // the G128 codec with its group index (W4a8U4ZpG)
    case QType::U4_G64_ZP_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
    case QType::Q8_G32_FP16:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
        return true;
    // NINFERNO-SPECIAL(arch:no-fp4-mma): the int8 tier multiplies the doubled E2M1 codes exactly
    // (W4a8Nvfp4) on every build. Without the native NVFP4 kernels (sm_89 / sm_86 builds) it is
    // the A8 route of every NVFP4 weight; with them (sm_120a) W4A4 keeps the weights it takes
    // (allows_w4a8), and W4A8 runs the prefill of those whose policy allows A8 but not A4, which
    // the BF16 register MMA ran before.
    case QType::NVFP4:
        return true;
    default:
        return false;
    }
}
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

bool owned_decode_r8(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    return tokens <= kR8DecodeTokens && supports(weight) &&
           r8_call(weight, policy, tokens);
}

bool owned_prefill_w4a8(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    // Without tensor mma.sync the tiled route owns such a call, as in select_route.
    return tokens >= kQ8PrefillA8Tokens && prefill_routes_active() &&
           weight.qtype == QType::NVFP4 && supports(weight) && allows_w4a8(weight, policy) &&
           !volta_tiled_route(weight, policy, tokens);
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
        return d.features.mma_tensor && !d.features.mma_slow &&
               !generation_row(d.major, d.minor).fp32acc_full_rate;
    } catch (const std::exception&) {
        return false;
    }
}

bool qx_format(QType qtype) {
    return qtype == QType::Q4_G64_FP16 || qtype == QType::Q5_G64_FP16 ||
           qtype == QType::Q6_G64_FP16;
}
} // namespace

LinearPolicy prefill_a8_policy(QType qtype, LinearPolicy policy) {
    if (policy == LinearPolicy::A16Only && qx_format(qtype) && qx_prefill_a8_pays() &&
        !invariant_routes_active()) {
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
    const bool qx = pays.get([](int) { return qx_prefill_a8_pays(); });
    switch (qtype) {
    case QType::Q8_G32_FP16:
        return true;
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
        return qx;
    default:
        return false;
    }
}

bool q8_prefill_a8(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= kQ8PrefillA8Tokens && prefill_routes_active() &&
           prefill_a8_format(weight.qtype) && allows_a8(policy) && supports(weight);
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

namespace {
// The current device's features; when no device answers (a host-only caller), every feature, so
// the build's own gates decide as before.
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
} // namespace

bool volta_cutlass_takes(std::int32_t tokens);
bool volta_cutlass_runs(bool only, std::int32_t tokens);

namespace {
// The codecs the tiled route takes on this GPU: tiled_supports, and Q8 where tensor mma.sync is
// missing (Volta): elsewhere Q8 has its own kernels, on Volta those tiles run on FMAs.
bool tiled_takes(WeightCodec codec) {
    return tiled_supports(codec) || (codec == WeightCodec::Q8 && !current_features().mma_tensor);
}

// The formats the tiled kernel has no tile for (BF16, FP8, NVFP4, MXFP4, Q6_K, Q3_K, the IQ
// formats, ...). On Volta a call of kVoltaCutlassMinTokens or more of one takes the tiled route all
// the same: generic_volta_cutlass decodes any codec's rows to FP16 for CUTLASS (decode_element),
// where their W4A8 tiles ran on dp4a over warp shuffles (V100, Gemma 4 31B Q4_K_XL prefill: the
// Q6_K calls 14.3 ms each, 31 % of it) and BF16 on the staged or register MMA over FP32 FMAs.
// Narrower calls keep their routes.
bool volta_cutlass_only(WeightCodec codec) {
    return !tiled_takes(codec) && !current_features().mma_tensor;
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
} // namespace

bool volta_tiled_format(QType qtype) {
    return !current_features().mma_tensor && volta_tiled_qtype(qtype);
}

bool route_admits(Route route, const Weight& w, LinearPolicy policy) {
    if (!supports(w)) return false;
    if (!has_features(current_features(), route_requires(route))) return false;
    // A ggml block format's high (sign / bit) plane off a 16-byte boundary (a view into a packed
    // buffer; a row view of one stays aligned, 32 bytes a row and 256 values): the kernels read it
    // in vector words, all but the classic W4A8 tiles (one byte at a time) and the scalar path. The
    // 2080 Ti and A100 gates faulted (misaligned address) on an IQ3_XXS plane 4 bytes off once the
    // cost model's pick left the classic tile.
    if (w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16) &&
        route.family != Family::Scalar &&
        !(route.family == Family::W4a8Mma && route.schedule < w4a8_schedule_count() &&
          w4a8_schedule_info(route.schedule).producers == 0)) {
        return false;
    }
    switch (route.family) {
    case Family::Scalar:
        return true;
    case Family::RegisterMma: {
        if (route.schedule >= rmma_schedule_count()) return false;
        const WeightCodec codec = codec_of(w);
        if ((w.k % k_multiple(codec)) != 0) return false;
        const auto s = rmma_schedule_info(route.schedule);
        if (!rmma_schedule_launchable(route.schedule) ||
            !rmma_tile_compiled(codec, s.m_tiles, s.n_tiles, w.k)) {
            return false;
        }
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
            return aligned_to(w.qdata, 16);
        }
        return aligned_to(w.qdata, 8);
    }
    case Family::W4a4Mma:
        return route.schedule < w4a4_schedule_count() && allows_w4a4(w, policy);
    case Family::TiledMma:
        // A CUTLASS-only format (volta_cutlass_only) runs no tile of its own: schedule 0 names it.
        if (volta_cutlass_only(codec_of(w))) { return route.schedule == 0 && (w.k % 64) == 0; }
        return route.schedule < tiled_schedule_count() && tiled_takes(codec_of(w)) &&
               tiled_fits(route.schedule, codec_of(w)) && (w.k % 64) == 0 &&
               aligned_to(w.qdata, 16);
    case Family::W4a8Mma:
        return route.schedule < w4a8_schedule_count() && w4a8_fits(route.schedule, w.qtype) &&
               allows_w4a8(w, policy) &&
               (w4a8_schedule_info(route.schedule).producers == 0 || w4a8_ws_admits(w));
    case Family::StagedMma:
        return route.schedule < staged_schedule_count() && staged_fits(route.schedule) &&
               staged_device_fits(route.schedule) && codec_of(w) == WeightCodec::Bf16 &&
               (w.k % 64) == 0 && aligned_to(w.qdata, 16);
    case Family::Nvfp4TiledMma:
        // supports() holds K to whole 64-value steps; a lane reads eight code bytes at once. The
        // schedules parse on every build, but their kernels are 120a-only: elsewhere the tuner
        // and the route tables must not reach a launch that throws.
        return kNativeNvfp4 && route.schedule < nvfp4_tiled_schedule_count() &&
               w.qtype == QType::NVFP4 && aligned_to(w.qdata, 8);
    case Family::R8Mma:
        return route.schedule < r8_schedule_count() && allows_r8(w, policy);
    case Family::GgmlMmq:
        return route.schedule == 0 && mmq_admits(w, policy);
    case Family::GgmlVec:
        return route.schedule == 0 && vec_admits(w, policy);
    case Family::I8Mma:
        return route.schedule < i8_schedule_count() && allows_i8(w, policy) &&
               i8_fits(route.schedule, w.qtype);
    }
    return false;
}

// Without tensor mma.sync a one-tile register-MMA call of at most four tokens sums lane by lane
// (GemmArgs::lane_local) outside the invariant profile: a token's sums then depend on its tile.
namespace {
bool rmma_sums_lane_local(int schedule) {
    return !current_features().mma_tensor && !invariant_routes_active() &&
           rmma_schedule_info(schedule).n_tiles == 1;
}
} // namespace

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

bool same_summation(Route a, Route b) {
    if (a == b) return true;
    if (a.family == Family::I8Mma && b.family == Family::I8Mma) return true;
    const auto register_order = [](Route r) {
        return (r.family == Family::RegisterMma && r.schedule < rmma_schedule_count() &&
                rmma_schedule_info(r.schedule).warps_k == 1 && !rmma_sums_lane_local(r.schedule)) ||
               (r.family == Family::StagedMma && r.schedule < staged_schedule_count());
    };
    return register_order(a) && register_order(b);
}

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
constexpr double kI8Eff[] = {0.53, 0.46, 0.42};
static_assert(std::size(kI8Eff) == std::size(kI8Schedules));
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

// The I8 tile of a call: the one with the fewest tile-waves weighted by its tile's work over its
// share of the int8 peak (kI8Eff): the largest tile while its grid fills the SMs, a smaller one
// where it would leave them idle. Every schedule gives the same bytes, so the choice may vary
// with T.
std::optional<Route> i8_route(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    const double sets       = problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0;
    const double sms        = std::max(sm_count(), 1);
    std::optional<Route> best;
    double best_cost = 0;
    for (int i = 0; i < i8_schedule_count(); ++i) {
        const Route r{Family::I8Mma, static_cast<std::uint8_t>(i)};
        if (!route_admits(r, weight, problem.policy)) continue;
        const auto s       = kI8Schedules[i];
        const double ctas  = std::ceil(problem.n * sets / s.block_n) * std::ceil(tokens / double(s.block_m));
        const double waves = std::ceil(ctas / sms);
        const double cost  = waves * s.block_m * s.block_n / kI8Eff[i];
        if (!best || cost < best_cost) {
            best      = r;
            best_cost = cost;
        }
    }
    return best;
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
template <class Offer>
void a8_candidates(const Problem& problem, const Weight& weight, std::int32_t tokens,
                   const DeviceClass& d, const Offer& offer) {
    // BF16 operands in the MMA itself (sm_80 on) imply tensor mma.sync (device_features_for).
    if (!d.features.mma_bf16_native || !allows_a8(problem.policy)) return;
    // A device without a roofline in its facts (an unknown id whose driver reports no clock or bus
    // width) has none here: the A16 routes stand alone. The clock and the bandwidth are the
    // board's reference facts (DeviceClass::facts, B2), never the driver's clocks.
    if (d.sms <= 0 || !d.facts.has_roofline()) return;
    const GenerationRow& g = generation_row(d.major, d.minor);
    const double clk     = d.facts.reference_clock_hz();
    const double bw      = d.facts.dram_bytes_per_s();
    const double sms     = d.sms;
    const double n = problem.n, k = problem.k, t = tokens;
    const double sets    = problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0;
    const double bpw     = static_cast<double>(weight.payload_bytes) / (n * k);
    const double peak_sm = g.s8 * clk;
    const double scale   = g.w4a8_eff_scale;
    const double bw_sm   = bw / sms;
    const double max_thr = std::max(d.max_threads_per_sm, 1);
    const double max_cta = std::max(d.max_blocks_per_sm, 1);
    const double smem_sm = static_cast<double>(d.smem_per_sm);
    const auto cdiv      = [](double a, double b) { return std::ceil(a / b); };
    const auto quant     = [&](double cols) { return 1.7e-6 + cols * k * 4.3 / bw; };
    // Fraction of the int8 peak by kW4a8Schedules index: 32x128, 64x128, 128x128, 128x64, 64x64,
    // 128x64 with producer warps.
    constexpr double kW4a8Eff[6] = {0.31, 0.28, 0.24, 0.47, 0.37, 0.98};
    if (allows_w4a8(weight, problem.policy)) {
        for (int i = 0; i < w4a8_schedule_count(); ++i) {
            const auto s         = w4a8_schedule_info(i);
            const double rows_w  = s.block_n, cols = s.block_m;
            const double rows_o  = rows_w / sets;
            const double threads =
                s.producers > 0 ? 384.0 : rows_w == 64.0 && cols == 64.0 ? 128.0 : 256.0;
            const double smem    = (s.producers > 0 ? 3.0 : 2.0) * (rows_w + cols) * 160.0;
            const double occ     = std::max(
                1.0, std::min({std::floor(max_thr / threads), max_cta, std::floor(smem_sm / smem)}));
            const double ctas  = cdiv(n, rows_o) * cdiv(t, cols);
            const double q     = std::ceil(ctas / sms);
            const double warps = std::min(1.0, ctas / sms) * std::min(q, occ) * threads / 32;
            const double mem   = std::max(q * rows_o * k * bpw / bw_sm, n * k * bpw / bw);
            const double mma   = q * 2.0 * rows_w * cols * k / peak_sm;
            const double cap   = s.producers > 0 ? 1690.0 : 1390.0;
            const double chain = std::ceil(q / occ) * std::ceil(k / 128.0) * cap / clk;
            const double eff   = kW4a8Eff[std::min(i, 5)] * scale;
            offer(Route{Family::W4a8Mma, static_cast<std::uint8_t>(i)},
                  quant(t) + std::max(mem, mma / eff) / std::clamp(warps, 0.05, 1.0) + chain);
        }
    }
    // The I8 tiles (generic_i8.cu): one activation scale per 64 values and the code as the MMA's
    // unsigned operand, about six instructions per MMA against the W4A8 tiles' twelve, at kI8Eff
    // of the int8 peak; one group per pipeline stage (250 cycles of barrier and copy wait). Tokens
    // padded to 256 in the quantizer.
    if (allows_i8(weight, problem.policy)) {
        const double tp = i8_padded_tokens(tokens);
        for (int i = 0; i < i8_schedule_count(); ++i) {
            const auto s         = i8_schedule_info(i);
            const double rows_w  = s.block_n, cols = s.block_m;
            const double rows_o  = rows_w / sets;
            const double occ     = std::max(1.0, std::min({std::floor(max_thr / s.threads), max_cta,
                                                           std::floor(smem_sm / s.smem_bytes)}));
            const double ctas    = cdiv(n, rows_o) * cdiv(t, cols);
            const double q       = std::ceil(ctas / sms);
            const double warps   = std::min(1.0, ctas / sms) * std::min(q, occ) * s.threads / 32;
            const double mem     = std::max(q * rows_o * k * bpw / bw_sm, n * k * bpw / bw);
            const double mma     = q * 2.0 * rows_w * cols * k / peak_sm;
            const double chain   = std::ceil(q / occ) * std::ceil(k / 64.0) * 250.0 / clk;
            const double eff     = kI8Eff[i] * scale;
            offer(Route{Family::I8Mma, static_cast<std::uint8_t>(i)},
                  quant(tp) + std::max(mem, mma / eff) / std::clamp(warps / 8.0, 0.05, 1.0) +
                      chain);
        }
    }
    if (g.mmq_candidate && ggml_mmq_format(problem.qtype) &&
        problem.n % ggml_mmq::kRowMultiple == 0 && problem.k % ggml_mmq::kKMultiple == 0) {
        const double slices = std::ceil(t / kMmqPrefillSlice);
        const double tt     = std::min<double>(t, kMmqPrefillSlice);
        const double cols   = std::min(128.0, cdiv(tt, 8.0) * 8.0);
        const double tiles  = cdiv(n * sets, 128.0) * cdiv(tt, cols);
        const double mem    = n * k * bpw / bw;
        const double mma    = tiles / sms * 2.0 * 128.0 * cols * k / peak_sm;
        const double chain  = std::max(1.0, tiles * std::ceil(k / 862.0) / sms) * 4075.0 / clk;
        offer(Route{Family::GgmlMmq, 0},
              slices * (std::max(mem, mma / (0.44 * scale)) + chain + quant(tt) + 18.6e-6));
    }
}

// The speed-profile route of a call of 17 tokens or more from a cost model of every tile the
// build holds for the weight (register MMA, tiled, staged): a tile's time on the busiest SM is its
// weight bytes over that SM's share of the DRAM bandwidth (or the whole matrix over the whole
// bandwidth, whichever is longer) and its MMA work over the SM's tensor rate; the register MMA
// runs them in series, the pipelined tiles (staged, tiled) overlap them at 0.9 / 0.8 of the
// tensor rate; either is slowed in proportion when the SM holds fewer than 16 (register MMA) or 4
// (pipelined) warps, and pays its warps' dependent chain of K steps per residency and the sum of
// its split K (over a CTA's warps or a cluster). The device numbers come from its query (SMs,
// threads, CTAs and shared memory per SM) and its facts (reference clock, bus width x memory clock:
// DeviceClass::facts); the tensor rate per SM and clock from the
// generation row (core/generation.h: 512 on the consumer and pro boards, 2048 on the full-rate HBM
// parts, 1536 where the BF16 MMA is emulated, which also takes no pipelined tile and no 8-tile
// token strip).
// ninferno_route_tune's timings of seven boards (each tile in the engine, L2 flushed; BF16, Q4
// and Q8, T >= 17), the constants refit with the board held out: 1.095 of the best (geomean;
// RTX 4090 1.083, RTX 5090 1.068) against 1.110 for the F-lite rule (heuristic_rmma) and 1.137
// without the chain and reduction terms; the optimum of every constant is flat (200 and 500
// cycles: 100..300 and 250..1000 cost under 0.2 %). nullopt where the facts have no roofline.
std::optional<Route> cost_route(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    DeviceClass d;
    try {
        d = device_class();
    } catch (const std::exception&) { return std::nullopt; }
    // The reference clock and bandwidth of the board's facts (B2: the PCI-ID table, else the
    // driver's values on the 2^(1/16) grid), never the driver's clocks.
    if (d.sms <= 0 || !d.facts.has_roofline() || problem.n <= 0 || problem.k <= 0 ||
        weight.payload_bytes == 0) {
        return std::nullopt;
    }
    const GenerationRow& g = generation_row(d.major, d.minor);
    const bool emulated  = !d.features.mma_bf16_native;
    const double clk     = d.facts.reference_clock_hz();
    const double bw      = d.facts.dram_bytes_per_s();
    const double sms     = d.sms;
    const double peak_sm = g.bf16 * clk;
    const double bw_sm   = bw / sms;
    const double n = problem.n, k = problem.k;
    const double bpw     = static_cast<double>(weight.payload_bytes) / (n * k);
    const double max_thr = std::max(d.max_threads_per_sm, 1);
    const double max_cta = std::max(d.max_blocks_per_sm, 1);
    const double smem_sm = static_cast<double>(d.smem_per_sm);
    const auto cdiv      = [](double a, double b) { return std::ceil(a / b); };
    // SwiGLU: the weight has 2 n rows (gate and up), every output row multiplies both; `bpw` is the
    // payload over the n output rows, so the bytes of a CTA stay rows x K x bpw.
    const double sets = problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0;
    // Time of `ctas` CTAs of `rows` output rows x `cols` (K over `cluster` CTAs) of `threads`
    // threads, at most `occ` resident per SM.
    const auto cost = [&](double rows, double cols, double threads, double cluster, double occ,
                          double warps_k, bool pipelined, double eff) {
        const double ctas  = cdiv(n, rows) * cdiv(tokens, cols) * cluster;
        const double kc    = k / cluster;
        const double q     = std::ceil(ctas / sms);
        // The warps resident on a busy SM: SMs left idle by a small grid do not slow the busiest
        // one, whose bytes are already in `mem` (the fleet-wide fill charged calls of fewer CTAs
        // than SMs up to 20x).
        const double warps = std::min(q, std::max(occ, 1.0)) * threads / 32;
        const double mem   = std::max(q * rows * kc * bpw / bw_sm, n * k * bpw / bw);
        const double mma   = q * 2.0 * rows * sets * cols * kc / peak_sm;
        // The CTAs the busiest SM runs one residency after another each pay their warps'
        // dependent chain: the exposed load latency of a K step (register MMA: 16 K per warp, 200
        // cycles) or of a pipeline stage (64 K, 500 cycles: the barrier and the cp.async wait).
        const double rounds = std::ceil(q / std::max(occ, 1.0));
        if (pipelined) {
            const double chain = rounds * std::ceil(kc / 64.0) * 500.0 / clk;
            return std::max(mem, mma / eff) / std::clamp(warps / 4.0, 0.05, 1.0) + chain;
        }
        // K split over a cluster (30 cycles per FP32 element) and over the CTA's K warps (2 cycles
        // per element and extra warp, through shared memory).
        const double reduce = q * (cluster - 1) * rows * sets * cols * 30.0 / clk +
                              rounds * (warps_k - 1.0) * rows * sets * cols * 2.75 / clk;
        const double chain  = rounds * std::ceil(kc / (16.0 * warps_k)) * 150.0 / clk;
        return (mem + mma + reduce) / std::clamp(warps / 11.0, 0.05, 1.0) + chain;
    };
    std::optional<Route> best;
    double best_time = 0;
    // The A16 model reads 11 % slow against the A8 one (fitted with it, 0.81..0.89 with any board
    // held out); on the minor-0 row (A100) it reads 39 % fast (one board); a constant factor leaves
    // the order of the A16 tiles as it was.
    const double kA16Scale = g.a16_cost_scale;
    const auto offer = [&](Route r, double time) {
        if (!route_admits(r, weight, problem.policy)) return;
        if (r.family == Family::RegisterMma || r.family == Family::TiledMma ||
            r.family == Family::StagedMma) {
            time *= kA16Scale;
        }
        if (!best || time < best_time) {
            best      = r;
            best_time = time;
        }
    };
    for (int i = 0; i < rmma_schedule_count(); ++i) {
        const auto s = rmma_schedule_info(i);
        // Cluster schedules exist in the full (sm_12x) build only (rmma_schedule_launchable); a
        // substituted board without clusters is refused by route_admits (route_requires).
        if (!rmma_schedule_launchable(i)) continue;
        if (emulated && s.n_tiles == 8) continue;
        const double threads = 32.0 * s.warps_m * s.warps_k;
        const double occ     = std::min(std::floor(max_thr / threads), max_cta);
        offer(Route{Family::RegisterMma, static_cast<std::uint8_t>(i)},
              cost(16.0 * s.m_tiles * s.warps_m, 8.0 * s.n_tiles, threads, s.cluster, occ,
                   s.warps_k, false, 1));
    }
    a8_candidates(problem, weight, tokens, d, offer);
    if (emulated) return best;
    const auto pipelined_occ = [&](double threads, double stages, double rows, double cols) {
        const double tile = std::min(stages * (rows + cols) * 128.0, 99.0 * 1024.0);
        return std::min({std::floor(max_thr / threads), max_cta, std::floor(smem_sm / tile)});
    };
    for (int i = 0; i < tiled_schedule_count(); ++i) {
        const auto s = tiled_schedule_info(i);
        offer(Route{Family::TiledMma, static_cast<std::uint8_t>(i)},
              cost(s.block_n / sets, s.block_m, 256, 1, pipelined_occ(256, 3, s.block_n, s.block_m),
                   1.0, true, 0.8));
    }
    for (int i = 0; i < staged_schedule_count(); ++i) {
        const auto s         = staged_schedule_info(i);
        const double threads = 32.0 * s.warps_m * s.warps_n;
        offer(Route{Family::StagedMma, static_cast<std::uint8_t>(i)},
              cost(s.rows / sets, s.tokens, threads, 1,
                   pipelined_occ(threads, s.stages, s.rows, s.tokens), 1.0, true, 0.9));
    }
    return best;
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
    if (const auto r = volta_tiled_route(weight, problem.policy, tokens)) return *r;
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

bool r8_verify_tunable(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= 2 && tokens <= kR8VerifyTokens && allows_r8(w, policy) &&
           r8_verify_compiled(w.qtype);
}

bool r8_tunable(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return r8_verify_tunable(w, policy, tokens) ||
           (vec_rule(w, policy, tokens) && r8_call(w, policy, tokens));
}

bool ggml_mmq_call(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens) {
    return mmq_call(w, policy, rows, tokens);
}

bool ggml_mmq_format(QType qtype) {
    Weight probe;
    probe.qtype     = qtype;
    const auto type = mmq_type(probe);
    return type && ggml_mmq::matrix_supported(*type);
}

std::size_t ggml_mmq_workspace_bytes(std::int32_t k, std::int32_t plane_rows, std::int32_t tokens) {
    return mmq_workspace_bytes(k, plane_rows, tokens);
}

bool ggml_vec_format(QType qtype) {
    Weight probe;
    probe.qtype     = qtype;
    const auto type = mmq_type(probe);
    return type && ggml_mmq::vec_supported(*type);
}

std::size_t ggml_vec_workspace_bytes(std::int32_t k, std::int32_t tokens) {
    return vec_workspace_bytes(k, tokens);
}

static_assert(kGgmlVecMaxTokens == ggml_mmq::kVecMaxColumns);

bool ggml_vec_takes(const Problem& problem, const Weight& weight, std::int32_t tokens) {
    // The callers share one quantized activation of the call's tokens (kGgmlVecMaxTokens at most).
    return tokens <= kGgmlVecMaxTokens && vec_call(weight, problem.policy, tokens) &&
           select_route(problem, weight, tokens, prefill_routes_active()).family == Family::GgmlVec;
}

std::size_t ggml_vec_activation_bytes(std::int32_t k, std::int32_t tokens) {
    return ggml_mmq::vec_activation_bytes(k, tokens);
}

static_assert(kGgmlMmqMaxTokens == kMmqMaxTokens);

bool ggml_mmq_takes_tuned(const Tensor& x, const Weight& weight, LinearPolicy policy,
                          std::int32_t row, std::int32_t rows, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!mmq_call(weight, policy, rows, tokens)) return false;
    const Problem problem{weight.qtype, policy, rows, weight.k, Epilogue::Store};
    Call call;
    call.x         = &x;
    call.weight    = &weight;
    call.row_begin = row;
    call.rows      = rows;
    call.epilogue  = Epilogue::Store;
    tune_dense(problem, call, stream);
    return select_route(problem, weight, tokens, prefill_routes_active()).family == Family::GgmlMmq;
}

std::size_t ggml_mmq_gated_pair_bytes(std::int32_t k, std::int32_t rows, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    // Two activations where the halves' formats read different q8_1 layouts.
    (void)layout.alloc_bytes(ggml_mmq::activation_bytes(k, tokens), 256);
    (void)layout.alloc_bytes(ggml_mmq::activation_bytes(k, tokens), 256);
    (void)layout.alloc_bytes(static_cast<std::size_t>(2) * rows * tokens * sizeof(float), 256);
    (void)layout.alloc_bytes(ggml_mmq::fixup_bound(tokens), 256);
    return layout.peak_bytes(1) + kGgmlArenaSlack;
}

std::int32_t ggml_mmq_slice_tokens(QType qtype) {
    return ggml_mmq_format(qtype) ? kMmqPrefillSlice : kMmqMaxTokens;
}

bool ggml_mmq_gated_pair(const Tensor& x, const PairHalf& gate, const PairHalf& up,
                         std::int32_t rows, bool gelu, Tensor& out, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::int32_t k      = gate.weight->k;
    if (up.weight->k != k || x.ne[0] != k || tokens < 1 || out.ne[0] < rows ||
        out.ne[1] != tokens || x.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        !x.is_contiguous() || !out.is_contiguous()) {
        return false;
    }
    // A prefill takes the pair where a call of each half alone takes the kernel (mmq_prefill_wins
    // and the prefill route tables, as select_route decides it); the band where the tuner says.
    const auto takes = [&](const PairHalf& h) {
        if (tokens <= kMmqMaxTokens) {
            return ggml_mmq_takes_tuned(x, *h.weight, h.policy, h.row, rows, stream);
        }
        const Problem problem{h.weight->qtype, h.policy, rows, k, Epilogue::Store};
        return (rows % ggml_mmq::kRowMultiple) == 0 && mmq_admits(*h.weight, h.policy) &&
               select_route(problem, *h.weight, tokens, prefill_routes_active()).family ==
                   Family::GgmlMmq;
    };
    if (!takes(gate) || !takes(up)) return false;
    const std::int32_t slice = std::min(tokens, kMmqPrefillSlice);
    if (workspace.used() + ggml_mmq_gated_pair_bytes(k, rows, slice) > workspace.capacity()) {
        return false;
    }
    auto scope            = workspace.scope();
    const auto gate_type  = *mmq_type(*gate.weight);
    const auto up_type    = *mmq_type(*up.weight);
    const bool two        = ggml_mmq::activation_layout(up_type) !=
                     ggml_mmq::activation_layout(gate_type);
    void* gate_activation = workspace.alloc_bytes(ggml_mmq::activation_bytes(k, slice)).data;
    void* up_activation =
        two ? workspace.alloc_bytes(ggml_mmq::activation_bytes(k, slice)).data : gate_activation;
    // [gate; up] rows of one FP32 plane, the gated product in the epilogue.
    const int plane_rows = 2 * rows;
    auto* plane          = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(plane_rows) * slice * sizeof(float)).data);
    void* fixup = workspace.alloc_bytes(ggml_mmq::fixup_bound(slice)).data;
    for (std::int32_t t0 = 0; t0 < tokens; t0 += slice) {
        const std::int32_t t = std::min(slice, tokens - t0);
        const auto* xb       = static_cast<const __nv_bfloat16*>(x.data) +
                         static_cast<std::size_t>(t0) * static_cast<std::size_t>(k);
        ggml_mmq::quantize_activation(gate_type, xb, k, k, t, gate_activation, stream);
        if (two) ggml_mmq::quantize_activation(up_type, xb, k, k, t, up_activation, stream);
        const auto product = [&](const PairHalf& h, ggml_mmq::Type type, const void* activation,
                                 float* to) {
            const int bpr = h.weight->k / static_cast<int>(ggml_block_values(h.weight->qtype));
            ggml_mmq::product(type,
                              ggml_rows_at(weight_args(*h.weight), h.weight->qtype, bpr, h.row),
                              bpr, rows, k, activation, t, to, plane_rows, fixup, stream);
        };
        product(gate, gate_type, gate_activation, plane);
        product(up, up_type, up_activation, plane + rows);
        GemmArgs args;
        args.tokens         = t;
        args.rows           = rows;
        args.gelu_gate      = gelu ? 1 : 0;
        args.segments       = 1;
        args.out[0]         = static_cast<__nv_bfloat16*>(out.data) +
                      static_cast<std::size_t>(t0) * static_cast<std::size_t>(out.ne[0]);
        args.segment_end[0] = rows;
        args.ld[0]          = static_cast<int>(out.ne[0]);
        launch_mmq_epilogue(Epilogue::SwiGlu, args, plane, plane_rows, stream);
    }
    return true;
}

bool ggml_vec_takes_tuned(const Tensor& x, const Weight& weight, LinearPolicy policy,
                          std::int32_t row, std::int32_t rows, cudaStream_t stream) {
    const Problem problem{weight.qtype, policy, rows, weight.k, Epilogue::Store};
    if (!ggml_vec_takes(problem, weight, x.ne[1])) {
        // The tuner may still put the kernel in for the heuristic's route; time the call first.
        if (!vec_admits(weight, policy) || x.ne[1] < 1 || x.ne[1] > kGgmlVecMaxTokens) {
            return false;
        }
    }
    Call call;
    call.x         = &x;
    call.weight    = &weight;
    call.row_begin = row;
    call.rows      = rows;
    call.epilogue  = Epilogue::Store;
    tune_dense(problem, call, stream);
    return ggml_vec_takes(problem, weight, x.ne[1]);
}

bool ggml_vec_gated_pair(const Tensor& x, const PairHalf& gate, const PairHalf& up,
                         std::int32_t rows, bool gelu, Tensor& out, WorkspaceArena& workspace,
                         const void* q8, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::int32_t k      = gate.weight->k;
    // The output is written at a stride of out.ne[0]: a contiguous [>= rows, T].
    if (up.weight->k != k || x.ne[0] != k || tokens < 1 || tokens > kGgmlVecMaxTokens ||
        out.ne[0] < rows || out.ne[1] != tokens || x.dtype != DType::BF16 ||
        out.dtype != DType::BF16 || !x.is_contiguous() || !out.is_contiguous()) {
        return false;
    }
    // Each half's route as a call of it alone gets it: the tuner times the problem first.
    const auto takes = [&](const PairHalf& h) {
        return ggml_vec_takes_tuned(x, *h.weight, h.policy, h.row, rows, stream);
    };
    if (!takes(gate) || !takes(up)) return false;
    if (workspace.used() + ggml_vec_gated_pair_bytes(k, rows, tokens) > workspace.capacity()) {
        return false;
    }
    auto scope             = workspace.scope();
    const void* activation = q8;
    if (activation == nullptr) {
        void* own = workspace.alloc_bytes(ggml_mmq::vec_activation_bytes(k, tokens)).data;
        ggml_vec_quantize(x, own, stream);
        activation = own;
    }
    const auto rows_at = [](const PairHalf& h) {
        const int bpr = h.weight->k / static_cast<int>(ggml_block_values(h.weight->qtype));
        return ggml_rows_at(weight_args(*h.weight), h.weight->qtype, bpr, h.row);
    };
    const auto bpr = [](const PairHalf& h) {
        return h.weight->k / static_cast<int>(ggml_block_values(h.weight->qtype));
    };
    ggml_mmq::VecOut o;
    o.gelu_gate      = gelu ? 1 : 0;
    o.segments       = 1;
    o.out[0]         = out.data;
    o.ld[0]          = out.ne[0];
    o.segment_end[0] = rows;
    if (gate.weight->qtype == up.weight->qtype) {
        o.epilogue = ggml_mmq::VecOut::Epilogue::SwiGlu;
        ggml_mmq::vec_product(*mmq_type(*gate.weight), rows_at(gate), rows_at(up), bpr(gate), rows,
                              k, activation, tokens, o, stream);
        return true;
    }
    auto* plane = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(rows) * tokens * sizeof(float)).data);
    ggml_mmq::VecOut g = o;
    g.epilogue         = ggml_mmq::VecOut::Epilogue::StoreF32;
    g.f32              = plane;
    g.f32_ld           = rows;
    ggml_mmq::vec_product(*mmq_type(*gate.weight), rows_at(gate), ggml_mmq::Planes{}, bpr(gate),
                          rows, k, activation, tokens, g, stream);
    o.epilogue = ggml_mmq::VecOut::Epilogue::GateProduct;
    o.gate     = plane;
    o.gate_ld  = rows;
    ggml_mmq::vec_product(*mmq_type(*up.weight), rows_at(up), ggml_mmq::Planes{}, bpr(up), rows, k,
                          activation, tokens, o, stream);
    return true;
}

std::size_t ggml_vec_gated_pair_bytes(std::int32_t k, std::int32_t rows, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(ggml_mmq::vec_activation_bytes(k, tokens), 256);
    (void)layout.alloc_bytes(static_cast<std::size_t>(rows) * tokens * sizeof(float), 256);
    return layout.peak_bytes(1) + kGgmlArenaSlack;
}

void ggml_vec_quantize(const Tensor& x, void* out, cudaStream_t stream) {
    ggml_mmq::vec_quantize_activation(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<int>(x.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16))), x.ne[0],
        x.ne[1], out, stream);
}

std::uint64_t ggml_mmq_fallback_count() noexcept {
    return mmq_fallbacks.load(std::memory_order_relaxed);
}

std::uint64_t route_demotion_count() noexcept {
    return route_demotions.load(std::memory_order_relaxed);
}

std::size_t route_workspace_bytes(Route route, std::int32_t k, std::int32_t tokens) {
    if (route.family == Family::GgmlVec) return vec_workspace_bytes(k, tokens);
    if (route.family == Family::I8Mma) {
        const auto t      = static_cast<std::size_t>(i8_padded_tokens(tokens));
        const auto groups = static_cast<std::size_t>(k / 64);
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc_bytes(t * static_cast<std::size_t>(k), 256);
        (void)layout.alloc_bytes(t * groups * sizeof(float), 256);
        (void)layout.alloc_bytes(t * groups * sizeof(int), 256);
        return layout.peak_bytes(1);
    }
    // R8 at verify widths reads the W4A8 planes (run_route's r8_pre: every r8_verify_compiled
    // format, whichever device routes to it).
    const bool r8_verify =
        route.family == Family::R8Mma && tokens >= 2 && tokens <= kR8VerifyTokens;
    if (route.family == Family::W4a8Mma || r8_verify) {
        // R8 reads rows of K, W4A8 of whole tiles (w4a8_ld).
        const std::int32_t ld = route.family == Family::W4a8Mma ? w4a8_ld(k) : k;
        const auto t          = static_cast<std::size_t>(tokens);
        const auto blocks     = static_cast<std::size_t>(ld / 32);
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc_bytes(t * static_cast<std::size_t>(ld), 256);
        (void)layout.alloc_bytes(t * blocks * sizeof(float), 256);
        (void)layout.alloc_bytes(t * blocks * sizeof(float), 256);
        return layout.peak_bytes(1);
    }
    if (route.family != Family::W4a4Mma) return 0;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(static_cast<std::size_t>(tokens) * static_cast<std::size_t>(k / 2), 256);
    (void)layout.alloc_bytes(static_cast<std::size_t>(tokens) * static_cast<std::size_t>(k / 16),
                             256);
    return layout.peak_bytes(1);
}

// Volta prefill (generic_volta_cutlass.cu): a tiled-route call of kVoltaCutlassMinTokens or more
// decodes its weight once and multiplies on CUTLASS's sm_70 GEMM; below it one or two token blocks
// re-decode the weight less than the decoded copy costs to write.
constexpr std::int32_t kVoltaCutlassMinTokens = 256;

bool volta_cutlass_takes(std::int32_t tokens) {
    static PerDevice<bool> enabled;
    return tokens >= kVoltaCutlassMinTokens && enabled.get([](int) {
        if (!volta_cutlass_compiled()) return false;
        try {
            return !device_class().features.mma_tensor;
        } catch (const std::exception&) { return false; }
    });
}

// Whether a TiledMma call of `tokens` runs on the decoded copy (and so needs its workspace):
// from kVoltaCutlassMinTokens, and a CUTLASS-only format (`only`: it has no tile of its own) from
// the prefill band's 17 tokens, which the band's route covers.
bool volta_cutlass_runs(bool only, std::int32_t tokens) {
    return volta_cutlass_takes(tokens) ||
           (only && tokens >= 17 && volta_cutlass_takes(kVoltaCutlassMinTokens));
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

bool spread_by_board(const DeviceClass& d) {
    const SpreadMeasured* m =
        nearest_measured(std::span(kSpreadMeasured), board_of(d), measured::kCarryDistance);
    return m != nullptr && m->faster;
}

// The current device's gates, once per ordinal: a process may drive boards of different classes
// (an RTX 5090 beside a 4090), and each launch takes its own device's.
struct DeviceGates {
    bool spread = false; // spread_smem measured faster on the nearest measured board
    bool pdl    = false; // programmatic dependent launch (pdl_prefetch)
};

const DeviceGates& device_gates() {
    constexpr int kOrdinals = 64;
    static std::array<std::once_flag, kOrdinals> once;
    static std::array<DeviceGates, kOrdinals> gates;
    static const DeviceGates none{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kOrdinals) return none;
    const auto slot = static_cast<std::size_t>(device);
    // A device whose class cannot be read throws (call_once then stays open): caching "none"
    // would turn PDL and the spread off for the process on one transient failure.
    std::call_once(once[slot], [&] {
        const DeviceClass& d = device_class(device);
        gates[slot]          = {spread_by_board(d), d.features.pdl};
    });
    return gates[slot];
}

bool spread_measured() { return device_gates().spread; }

bool pdl_device() { return device_gates().pdl; }

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
} // namespace

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
    if (!aligned_to(args.x, 16) && route.family == Family::RegisterMma) {
        route = Route{Family::Scalar, 0};
    }
    if (route.family == Family::TiledMma && (!aligned_to(args.x, 16) || (args.ldx % 8) != 0)) {
        route = Route{Family::Scalar, 0};
    }
    // The NVFP4 tiled route stages activations in 16-byte copies; otherwise the register MMA of
    // the same order takes the call (or the scalar route where that cannot read x either).
    if (route.family == Family::Nvfp4TiledMma && (!aligned_to(args.x, 16) || (args.ldx % 8) != 0)) {
        const int classes = nvfp4_tiled_schedule_info(route.schedule).classes;
        route             = aligned_to(args.x, 16)
                                ? Route{Family::RegisterMma,
                            static_cast<std::uint8_t>(classes == 1 ? rmma_index(4, 1, 1, 8)
                                                                               : rmma_index(1, classes, 1, 4))}
                                : Route{Family::Scalar, 0};
    }
    // The W4A8 quantizer reads 32 BF16 values of a token as four 16-byte words, the I8 one 64.
    if ((route.family == Family::W4a8Mma || route.family == Family::I8Mma) &&
        (!aligned_to(args.x, 16) || (args.ldx % 8) != 0)) {
        route = Route{Family::Scalar, 0};
    }
    // A producer-warp W4A8 tile the weight is not admitted to (another format, an IQ3_XXS sign
    // plane off 8 bytes) runs the classic tile of its shape: the same bytes.
    if (route.family == Family::W4a8Mma && w4a8_schedule_info(route.schedule).producers > 0 &&
        !w4a8_ws_admits(w)) {
        route = w4a8_classic_twin(route);
    }
    if (route.family == Family::W4a8Mma && w4a8_schedule_info(route.schedule).producers == 0 &&
        w4a8_ws_extra(w.qtype) && w4a8_ws_admits(w)) {
        route = w4a8_ws_twin(route);
    }
    if (route.family == Family::StagedMma && (!aligned_to(args.x, 16) || (args.ldx % 8) != 0)) {
        // The register MMA with WarpsK == 1 gives the same bytes; the scalar kernel where even it
        // cannot read x, as for the register MMA.
        route = aligned_to(args.x, 16)
                    ? Route{Family::RegisterMma, static_cast<std::uint8_t>(rmma_index(4, 1, 1, 8))}
                    : Route{Family::Scalar, 0};
    }
    // R8 quantizes 32 BF16 values of a token as four 16-byte words, and holds the call's tokens in
    // shared memory; a call it cannot take (a route table or a probe naming it) runs A16.
    // At verify widths a format with the pre-quantized form needs the workspace instead of the fit.
    const bool r8_pre = route.family == Family::R8Mma && args.tokens >= 2 &&
                        args.tokens <= kR8VerifyTokens && workspace != nullptr &&
                        r8_verify_compiled(w.qtype);
    if (route.family == Family::R8Mma && (!aligned_to(args.x, 16) || (args.ldx % 8) != 0 ||
                                          !(r8_pre || r8_fits(w.qtype, args.k, args.tokens)))) {
        const Problem a16{w.qtype, LinearPolicy::A16Only, call.rows, w.k, call.epilogue};
        route = Route{Family::RegisterMma,
                      static_cast<std::uint8_t>(heuristic_rmma(a16, args.tokens, false))};
        if (!aligned_to(args.x, 16) || (w.k % k_multiple(codec_of(w))) != 0) {
            route = Route{Family::Scalar, 0};
        }
    }
    // ggml's matrix kernel needs its planes in the workspace and x in 8-byte words; a call it
    // cannot take runs A16 (the same route the heuristic gives without the band).
    const bool gated     = call.epilogue == Epilogue::SwiGlu;
    const int plane_rows = gated ? 2 * call.rows : call.rows;
    if ((route.family == Family::GgmlMmq &&
         (workspace == nullptr || !aligned_to(args.x, 16) || (args.ldx % 4) != 0 ||
          (call.rows % ggml_mmq::kRowMultiple) != 0 ||
          !mmq_fits(*workspace, mmq_workspace_bytes(w.k, plane_rows,
                                                    std::min(args.tokens, kMmqPrefillSlice))))) ||
        (route.family == Family::GgmlVec &&
         (workspace == nullptr || args.segments > ggml_mmq::kVecMaxSegments ||
          !mmq_fits(*workspace,
                    vec_workspace_bytes(w.k, std::min(args.tokens, ggml_mmq::kVecMaxColumns)))))) {
        // A caller that sized its workspace short of the route (said below, with every other
        // demotion) is counted here too (ggml_mmq_fallback_count).
        mmq_fallbacks.fetch_add(1, std::memory_order_relaxed);
        const Problem a16{w.qtype, LinearPolicy::A16Only, call.rows, w.k, call.epilogue};
        route = Route{Family::RegisterMma,
                      static_cast<std::uint8_t>(heuristic_rmma(a16, args.tokens, false))};
        if (!aligned_to(args.x, 16) || (w.k % k_multiple(codec_of(w))) != 0) {
            route = Route{Family::Scalar, 0};
        }
    }
    if (route.family != asked.family && !same_bytes(asked, route)) {
        note_demotion(asked, route, call.rows, args.tokens, w.k, aligned_to(args.x, 16), args.ldx,
                      workspace != nullptr ? workspace->capacity() : 0);
    }
    // Only a register-MMA or R8 launch joins the call before it; every other route runs in stream
    // order.
    const bool joinable =
        (route.family == Family::RegisterMma || route.family == Family::R8Mma) &&
        call.epilogue == Epilogue::Store;
    args.join_previous  = call.join_previous && joinable ? 1 : 0;
    args.release_next   = call.release_next && joinable ? 1 : 0;
    // The pre-quantized R8 form does not release (the next call's quantizer could take the
    // workspace it still reads); without PDL the trigger is a no-op.
    if (released != nullptr) {
        *released =
            args.release_next != 0 && !(route.family == Family::R8Mma && r8_pre) && pdl_device();
    }
    // A decode-width register-MMA call that does not join starts its weight stream under the
    // previous kernel (GemmArgs::pdl_prefetch), where the device has programmatic dependent launch
    // (sm_90 and up; Ada launches in stream order as before).
    // R8 does the same where its activation is quantized in the kernel (not the pre-quantized
    // form, whose quantizer runs before it).
    args.pdl_prefetch =
        (route.family == Family::RegisterMma || (route.family == Family::R8Mma && !r8_pre)) &&
                args.join_previous == 0 && args.tokens <= kPdlPrefetchMaxTokens && pdl_device()
            ? 1
            : 0;
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
        const void* activation = call.vec_activation;
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
        if (!current_features().mma_tensor && volta_gemv_takes(codec, args) &&
            (args.ldx % 8) == 0 && (reinterpret_cast<std::uintptr_t>(args.x) % 16) == 0) {
            launch_volta_gemv(codec, call.epilogue, args, stream);
            return;
        }
        switch (codec) {
        case WeightCodec::Bf16:
            launch_rmma_bf16(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Fp8:
            launch_rmma_fp8(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Nvfp4:
            launch_rmma_nvfp4(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Q4:
            launch_rmma_q4(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Q5:
            launch_rmma_q5(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Q6:
            launch_rmma_q6(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Q3G128:
            launch_rmma_q3g128(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Q3G64:
            launch_rmma_q3g64(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Q8:
            launch_rmma_q8(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Fp8RowF32:
            launch_rmma_fp8_row_f32(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Fp8B128:
            launch_rmma_fp8_b128(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::U4Zp:
            launch_rmma_u4zp(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::U4ZpG32:
            launch_rmma_u4zp_g32(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::U4ZpG64:
            launch_rmma_u4zp_g64(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ4K:
            launch_rmma_ggml_q4k(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ5K:
            launch_rmma_ggml_q5k(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ6K:
            launch_rmma_ggml_q6k(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ3K:
            launch_rmma_ggml_q3k(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ2K:
            launch_rmma_ggml_q2k(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlIq4Xs:
            launch_rmma_ggml_iq4xs(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlIq3Xxs:
            launch_rmma_ggml_iq3xxs(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlIq3S:
            launch_rmma_ggml_iq3s(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ40:
            launch_rmma_ggml_q40(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ41:
            launch_rmma_ggml_q41(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ50:
            launch_rmma_ggml_q50(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ51:
            launch_rmma_ggml_q51(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlIq4Nl:
            launch_rmma_ggml_iq4nl(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlIq2Xxs:
            launch_rmma_ggml_iq2xxs(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlIq2Xs:
            launch_rmma_ggml_iq2xs(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlIq2S:
            launch_rmma_ggml_iq2s(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlIq1S:
            launch_rmma_ggml_iq1s(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlIq1M:
            launch_rmma_ggml_iq1m(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlTq10:
            launch_rmma_ggml_tq10(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlTq20:
            launch_rmma_ggml_tq20(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ10:
            launch_rmma_ggml_q10(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::GgmlQ20:
            launch_rmma_ggml_q20(route.schedule, call.epilogue, args, stream);
            return;
        case WeightCodec::Mxfp4:
            launch_rmma_mxfp4(route.schedule, call.epilogue, args, stream);
            return;
        }
        return;
    case Family::TiledMma:
        if ((w.k % 64) != 0) {
            throw std::invalid_argument("generic tiled: K must be whole 64-value tiles");
        }
        // A CUTLASS-only format has no tile of its own, so it takes CUTLASS below
        // kVoltaCutlassMinTokens too: the prefill band's route covers the chunk's narrower calls
        // (V100, gpt-oss BF16 attention at T = 79 ran scalar, 7.3 ms a call, 360 -> 140 tok/s).
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
        const auto tokens  = static_cast<std::size_t>(args.tokens);
        const auto k       = static_cast<std::size_t>(args.k);
        const DeviceSpan c = workspace->alloc_bytes(tokens * (k / 2), 256);
        const DeviceSpan s = workspace->alloc_bytes(tokens * (k / 16), 256);
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
        const auto tokens   = static_cast<std::size_t>(args.tokens);
        const auto ld       = static_cast<std::size_t>(w4a8_ld(args.k));
        const auto blocks   = ld / 32;
        const DeviceSpan c  = workspace->alloc_bytes(tokens * ld, 256);
        const DeviceSpan sc = workspace->alloc_bytes(tokens * blocks * sizeof(float), 256);
        const DeviceSpan su = workspace->alloc_bytes(tokens * blocks * sizeof(float), 256);
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
        const auto tokens   = static_cast<std::size_t>(i8_padded_tokens(args.tokens));
        const auto k        = static_cast<std::size_t>(args.k);
        const DeviceSpan c  = workspace->alloc_bytes(tokens * k, 256);
        const DeviceSpan sc = workspace->alloc_bytes(tokens * (k / 64) * sizeof(float), 256);
        const DeviceSpan su = workspace->alloc_bytes(tokens * (k / 64) * sizeof(int), 256);
        launch_i8(route.schedule, i8_bits(w.qtype), call.epilogue, args,
                  static_cast<std::int8_t*>(c.data),
                  static_cast<float*>(sc.data), static_cast<int*>(su.data), stream);
        return;
    }
    case Family::R8Mma:
        if (r8_pre) {
            auto scope          = workspace->scope();
            const auto tokens   = static_cast<std::size_t>(args.tokens);
            const auto blocks   = static_cast<std::size_t>(args.k / 32);
            const DeviceSpan c  = workspace->alloc_bytes(tokens * static_cast<std::size_t>(args.k), 256);
            const DeviceSpan sc = workspace->alloc_bytes(tokens * blocks * sizeof(float), 256);
            const DeviceSpan su = workspace->alloc_bytes(tokens * blocks * sizeof(float), 256);
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
    record_problem(problem);
    tune_dense(problem, call, stream);
    Route route = select_route(problem, *call.weight, call.x->ne[1], prefill_routes_active());
    if ((route.family == Family::W4a4Mma || route.family == Family::W4a8Mma ||
         route.family == Family::I8Mma) &&
        workspace == nullptr) {
        route = heuristic_route(
            Problem{problem.qtype, LinearPolicy::A16Only, problem.n, problem.k, problem.epilogue},
            *call.weight, call.x->ne[1], prefill_routes_active());
    }
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
