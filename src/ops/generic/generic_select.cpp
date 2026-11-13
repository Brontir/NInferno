// The route selector of the generic path, moved whole out of generic_gemm.cpp (#620): which route
// a projection call of T tokens takes (select_route and the per-family heuristics behind it), and
// the admission of each schedule (route_admits, the *_refusal functions, the weight-side format and
// alignment tests). The launch code (generic_gemm.cpp, run_route) runs the routes chosen here; what
// the launch code alone needs from here is declared in generic_select_detail.h.

#include "ops/generic/generic_gemm.h"

#include "core/device_class.h"
#include "core/format_facts.h"
#include "core/generation.h"
#include "core/layout.h"
#include "core/measured_boards.h"
#include "core/per_device.h"
#include "core/tensor.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/arch_limits.h"
#include "ops/common/native_nvfp4.h"
#include "ops/common/route_probe.h"
#include "ops/generic/format_rules.h"
#include "ops/generic/format_traits.h"
#include "ops/generic/format_requires.h"
#include "ops/generic/codec_traits.h"
#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_ggml_internal.h"
#include "ops/generic/generic_i8_schedules.h"
#include "ops/generic/generic_schedule_tables.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/generic_select_detail.h"
#include "ops/generic/generic_select_internal.h"
#include "ops/generic/ggml_mmq/ggml_mmq.h"
#include "ops/generic/r8_decode_table.h"
#include "ops/generic/route_cost_constants.h"
#include "ops/generic/route_table.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/route_pins.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

namespace ninferno::ops::detail::generic {

namespace {

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

bool i8_format(QType qtype) { return format_traits(qtype).qx_g64; }

bool i8_planes_aligned(const Weight& w) {
    const bool q4 = w.qtype == QType::Q4_G64_FP16, q5 = w.qtype == QType::Q5_G64_FP16;
    return aligned_to(w.qdata, 16) && aligned_to(w.scales, 16) &&
           (q4 || aligned_to(w.qhigh, q5 ? 8 : 16));
}

bool i8_extent(const Weight& w) { return (w.k % 512) == 0 && (w.padded_shape[1] % 512) == 0; }

// Whether the producer-warp tile can read the weight: its format, and IQ3_XXS's sign plane (read
// 8 bytes at a time) on an 8-byte boundary.
bool w4a8_ws_admits(const Weight& w) {
    return w4a8_ws_supports(w.qtype) && (w.qtype != QType::GGML_IQ3_XXS || aligned_to(w.qhigh, 8));
}

// R8: a format with an R8 codec, whole load groups of K, 16-byte code rows, A8 allowed. A lane
// strides K by groups, so K need not fill 128-value tiles: Gemma 4 26B's dense down projection
// (K = 2112) and gpt-oss's hidden (2880) take R8 at decode instead of the A16 register MMA.
bool allows_r8(const Weight& w, LinearPolicy policy) {
    return allows_a8(policy) && r8_supports(w.qtype) && (w.k % r8_k_values(w.qtype)) == 0 &&
           aligned_to(w.qdata, 16);
}

// Whether a call of T tokens takes R8 (decode, T <= kR8DecodeTokens, and the activation fits):
// r8_call_refusal names why not.
bool r8_call(const Problem& problem, const Weight& w, std::int32_t tokens) {
    return r8_call_refusal(problem, w, tokens) == R8CallRefusal::None;
}

MmqPlaneBytes mmq_plane_bytes(QType q) {
    if (const int whole = codec_traits(q).block_bytes; whole > 0) return {whole, 0, 0};
    const auto p = ggml_plane_bytes(q);
    return {static_cast<int>(p.code), static_cast<int>(p.high), static_cast<int>(p.scale)};
}

namespace {

// The weight in the layout ggml's kernels read: whole blocks for Q4_K / Q5_K, the planes otherwise.
bool ggml_layout_admits(const Weight& w) {
    const bool planes = codec_traits(w.qtype).block_bytes == 0;
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

measured::GateAnswer mmq_prefill_device_gate() {
    try {
        return measured::mmq_prefill_gate(board_of(device_class()));
    } catch (...) { return {}; }
}

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


} // namespace

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
namespace {

// The codebook formats only: their register-MMA dequant looks every value up in a grid table per
// token tile, which the matrix kernel's one decode a call saves. The K-quants' affine dequant is
// cheap, and there the register MMA keeps the band (RTX 4090, cellgt ABBA: UD-Q4_K_XL 27B DFlash2
// C1 115.7 -> 111.7 and C4 271.5 -> 263.1 with them in the rule, Muse 30B Q4_K_M C4 282.7 ->
// 267.6).
// The formats, their cards and sources: format_rules.h (Rule::MmqVerify).
bool mmq_verify_format(QType q) { return format_rules::holds(format_rules::Rule::MmqVerify, q); }

// The band's first width, with its cards: format_rules.h.
using format_rules::kMmqVerifyMinTokens;

} // namespace

bool mmq_verify_rule(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens,
                     int lanes) {
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
    return mmq_call(w, policy, rows, tokens) && tokens >= kMmqVerifyMinTokens && lanes != 1 &&
           mmq_verify_format(w.qtype);
}

bool mmq_verify_rule(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens) {
    return mmq_verify_rule(w, policy, rows, tokens, ops::round_lanes());
}

namespace {

int sm_count() {
    try {
        return device_class().sms;
    } catch (const std::exception&) { return 1; }
}


} // namespace

// Index of the register-MMA schedule with this shape; the list is generic_gemm_launch.cuh's.
int rmma_index(int warps_m, int warps_k, int m_tiles, int n_tiles, int cluster) {
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

namespace {

// ggml formats that the decode heuristic below measured faster on the four-warp CTA once the
// eight-warp grid needs a second wave (RTX 5080, T = 1): Q4_K, Q5_K, IQ4_XS, IQ3_XXS and the Q4_0
// decoder family at n = 3840..8192, and the codebook formats IQ2_S and IQ3_S, whose
// decode rather than the weight stream bounds them, by 11-16% at n = 4096..12288 (IQ3_S
// 4096x2560 11.4 -> 9.7 us, IQ2_S 9216x2560 21.0 -> 18.6 us). Q8 (Q8_0) and Q6_K at K >= 12288
// were faster on eight warps at every size.
// The formats, their cards and sources: format_rules.h (Rule::LightStep).
bool light_ggml_step(QType q) { return format_rules::holds(format_rules::Rule::LightStep, q); }

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
// The format, its cards and source: format_rules.h (Rule::TallTile).
bool tall_tile_format(QType q) { return format_rules::holds(format_rules::Rule::TallTile, q); }

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
    } catch (const std::exception&) { return false; }
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
    } catch (const std::invalid_argument&) { return 128; }
    const bool quad_u4 =
        (codec == WeightCodec::U4ZpG32 || codec == WeightCodec::U4ZpG64) && k > 0 && (k % 128) == 0;
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
    const auto refills       = [&](int warps_k) { return (steps + warps_k - 1) / warps_k > 2; };
    // The warps a SM holds at 128 registers a thread (kRmmaMinBlocks).
    constexpr std::int64_t kSlotWarps = 65536 / (32 * 128);
    if (info.warps_k <= 4 || refills(info.warps_k) || !refills(4) || ctas * 4 < kSlotWarps * sms) {
        return index;
    }
    return rmma_index(1, 4, 1, 1);
}

} // namespace

std::int64_t rmma_step_values(QType qtype, std::int32_t k) { return rmma_k_step(qtype, k); }

std::string_view rmma_geometry_decoder(QType qtype) {
    Weight weight;
    weight.qtype = qtype;
    WeightCodec codec{};
    try {
        codec = codec_of(weight);
    } catch (const std::invalid_argument&) { return {}; }
    // The decoder each codec's launch file instantiates (generic_gemm_<format>.cu), where it has
    // a codebook or stages its activations; the Q6_K wide decoder the launcher may take has the
    // same row. The ggml 32-blocks and MXFP4 have rows for their group tail, which adds none.
    switch (codec) {
    case WeightCodec::GgmlIq1M:
        return "GgmlIq1MDecoder";
    case WeightCodec::GgmlIq1S:
        return "GgmlIq1SDecoder";
    case WeightCodec::GgmlIq2S:
        return "GgmlIq2SDecoder";
    case WeightCodec::GgmlIq2Xs:
        return "GgmlIq2XsDecoder";
    case WeightCodec::GgmlIq2Xxs:
        return "GgmlIq2XxsDecoder";
    case WeightCodec::GgmlIq3S:
        return "GgmlIq3SDecoder";
    case WeightCodec::GgmlIq3Xxs:
        return "GgmlIq3XxsDecoder";
    case WeightCodec::GgmlQ3K:
        return "GgmlQ3KDecoder";
    case WeightCodec::GgmlQ6K:
        return "GgmlQ6KDecoder";
    case WeightCodec::GgmlQ4K:
        return "GgmlQ45KQuadDecoder<4>";
    case WeightCodec::GgmlQ5K:
        return "GgmlQ45KQuadDecoder<5>";
    default:
        return {};
    }
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
// The architectures are a declared requirement (format_requires::kW4a8WsTunedArch).
bool w4a8_ws_extra(QType q) {
    // The formats, their cards and sources: format_rules.h (Rule::W4a8WsTuned).
    if (!format_rules::holds(format_rules::Rule::W4a8WsTuned, q)) return false;
    try {
        return satisfied(format_requires::kW4a8WsTunedArch, current_arch_facts());
    } catch (const std::exception&) { return false; }
}

// The build architecture is a macro of the tree (a tree compiles one architecture), read here.
ArchFacts current_arch_facts() {
#if defined(NINFERNO_BUILD_ARCH)
    constexpr int build_arch = NINFERNO_BUILD_ARCH;
#else
    constexpr int build_arch = 0;
#endif
    return arch_facts_of(device_class(), build_arch);
}

namespace {

bool w4a8_ws_wins(const Problem& p, const Weight& w) {
    if (!w4a8_ws_admits(w)) return false;
    bool blackwell = false, ada = false, ampere = false;
    try {
        const auto& d = device_class();
        blackwell     = gb20x_cc(d.major, d.minor);
        ada           = ada_cc(d.major, d.minor);
        ampere        = ampere_cc(d.major, d.minor);
    } catch (const std::exception&) { return false; }
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
    } catch (const std::exception&) { return false; }
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
// ninferno_r8_schedule_bench on the IQ3 16 GB artifacts' 63 problems) is fastest with eight
// warps of two rows on every IQ2_*/IQ3_S store and add from N = 5120 (IQ2_XXS 10240x5120 w4r2
// 43.5 -> w8r2 40.0 us, IQ2_XS 17408x5120 83.6 -> 79.4, IQ3_S 16384x5120 w4r1 100.6 -> 94.5) and
// IQ3_XXS two rows per warp from N = 5120 (16384x5120 92.8 -> 82.5, 6144x5120 36.0 -> 34.0), so
// the rule now keys on the architecture, with IQ2_XXS/XS joining IQ2_S. Q3_K at K >= 8192 stays at
// four single-row warps: Qwen3.8's 5120x17408 add is 7-10 % faster so on the 5060 Ti, the
// RTX 4080 SUPER and the RTX 4070 Ti SUPER (113.2 -> 103.0, 64.8 -> 60.3, 73.0 -> 66.3 us).

} // namespace

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
    // The architecture rules below read declared requirements (format_requires.h).
    const ArchFacts arch = current_arch_facts();
    if (satisfied(format_requires::kR8WithoutNativeBf16, arch)) {
        const bool decode_heavy = format_traits(p.qtype).decode_heavy;
        if (decode_heavy && p.k >= 5120) return r8_index(8, 2);
        // Volta (an sm_70 build, whose R8 kernel is the one-slot form: generic_r8.cuh
        // NINFERNO_R8_ONE_SLOT; a GPU without mma.sync: format_requires::kR8OneSlot; a tree
        // compiles one architecture) keeps one row per warp for Q6_K at every N
        // and K, where the two-slot form took two. With one slot a second row per warp raises the
        // registers from 137 to 213 (3 CTAs per SM to 2) and the code a cold SM fetches first from
        // 960 to 1480 instructions. GV100 with ten other kernels between launches, as in a model
        // step: 4096x12288 add w4r1 88.8 against w4r2 94.7 us, 8192x4096 store 57.9 against 63.3.
        if (p.qtype == QType::GGML_Q6_K && p.epilogue != Epilogue::SwiGlu &&
            satisfied(format_requires::kR8OneSlot, arch))
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

namespace {

// Before the rules, a decode call takes the entry for the exact problem in the table of the
// nearest measured board (r8_decode_table.h), when that board is a variant of this one: the
// codebook problems of Gemma 4 31B, Muse Glimmer 30B, Qwen3.6-35B-A3B and Qwen3.8-27B, and the
// K-quant down projection of Qwen3.5-9B (4096x12288), where another schedule measured faster on
// that board. A table carries to boards within kR8CarryDistance of it
// (board_distance: a 4090 D or a 3090 Ti takes the 4090's or the 3090's, at 0.17 and 0.14) and no
// further: the 4090's 80 entries on the 4080 SUPER (distance 1.2; 3 sweeps) won 41, lost 30 (to
// -22 %: IQ2_XS 4096x6656 w8r2), +0.1 % in the mean, where its own sweep keeps 52 that win.
constexpr double kR8CarryDistance = 0.5;

} // namespace

// The current device's R8 decode table, once per ordinal (a process may drive boards of
// different classes); null where no measured board is near enough.
// A device query that fails is not cached (PerDevice retries a make that throws): the next call
// asks again rather than leaving the device on the heuristics for the process's life.
R8TableLookup r8_decode_table() {
    static PerDevice<R8TableLookup> lookup;
    try {
        return lookup.get([](int) {
            const Board board = board_of(device_class());
            const R8DecodeTable* table =
                nearest_known_measured(std::span(kR8DecodeTables), board, kR8CarryDistance);
            return R8TableLookup{table, table ? board_distance(board, table->board) : 0.0};
        });
    } catch (const std::exception&) { return {}; }
}

std::optional<R8TableHit> tuned_r8(const Problem& p) {
    const R8TableLookup table = r8_decode_table();
    if (table.table == nullptr) return std::nullopt;
    const Epilogue epilogue = p.epilogue == Epilogue::AddResidual ? Epilogue::Store : p.epilogue;
    for (const R8DecodeEntry& e : table.table->entries) {
        if (e.qtype == p.qtype && e.epilogue == epilogue && e.n == p.n && e.k == p.k) {
            return R8TableHit{r8_index(e.warps, e.rows), table.distance, table.table->gpu};
        }
    }
    return std::nullopt;
}

bool w4a8_ws_supports(QType q) { return compiled(Variant::W4a8Ws, q); }

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
                    } catch (const std::exception&) {}
                    if (bytes > 0.0 && bytes < kRmmaQ6kFourWarpBytes) {
                        return rmma_k_split_fit(rmma_index(1, 4, 1, 1), problem, ctas, sms);
                    }
                }
                return rmma_k_split_fit(rmma_index(1, 8, 1, 1), problem, ctas, sms);
            } else if (tokens <= 16) {
                // 9..16 tokens (verify of an 8- or 16-token draft): one 8-token tile per CTA would
                // put two or three CTA columns on the token axis, each streaming and decoding the
                // whole weight. Four K-split warps with two tiles cover 16 tokens per
                // column: 1.00-1.12 of the best of seven schedules against 1.24-1.58 for w1x8 n1
                // (geomean per format, RTX 5080, Q8/Q4_0/IQ4_XS/Q4_K/Q5_K/Q6_K/IQ3_XXS at
                // 27B/31B/35B shapes; at T = 24 Q8 1.35 against 1.68). A call of at most one CTA
                // per SM through T = 16 takes the decode split of K above (two CTA columns): the
                // four-warp tiles leave SMs idle on that grid. The dense tuner's timings of nine
                // cards (sm_70 to sm_120) had the rule above lose such calls (n = 48..2048 rows:
                // K/V, gates, small outputs) by x1.4..2.1 (median per card) to the sixteen-warp
                // split; on the RTX 5090 the CTA pair or cluster of eight was 0..17 % faster still
                // (n = 48 and 1024, T = 12 and 16: 8..20 us against 16..41 for the rule).
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
        // 17408 rows, time of m1.n8 over the best): T = 17 and 32 1.48-3.21 on every problem; T =
        // 64 1.20-1.58 below 160 CTAs, 1.00 at 272. RTX 5090 (gb202:170sm), Q6_K A16 output head
        // (248320 rows): m1.n4 the best at T = 17 (1.22) and 32 (1.54), m1.n8 at 64. RTX 5060 Ti
        // (gb206:36sm, the same artifacts as the 4080 SUPER): T = 17 and 32 1.60-2.14 on every
        // problem. In the product (ninferno-perplexity at context 32, tuner off on both arms): 4080
        // SUPER Qwen3.8-27B IQ3 246.6 -> 252.9 tok/s, the 5090's Q4_K_XL even.
        const std::int64_t ctas = (static_cast<std::int64_t>(problem.n) + 63) / 64;
        const bool narrow_band  = tokens <= 32 || (tokens <= 64 && ctas < 2 * sm_count());
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
    // 17..32 tokens (verify of a 16-token draft, short prompts): 32 tokens by 128 rows wastes less
    // of the tile than 64x64 and reads each weight tile once per 128 rows. (RTX 5080, 7 GGUF
    // formats at 27B/31B/35B shapes, time over the best of seven routes, geomean per format) at T =
    // 17: 32x128 1.05-1.45 against 64x64 1.14-1.57, better for every format at 17, 20, 24, 32.
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
        if (const auto tuned = tuned_r8(p)) return tuned->schedule;
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

std::optional<ggml_mmq::Type> mmq_type(const Weight& w) { return format_facts(w.qtype).mmq; }

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

bool mmq_call(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens) {
    return tokens >= kMmqMinTokens && tokens <= kMmqMaxTokens &&
           (rows % ggml_mmq::kRowMultiple) == 0 && mmq_admits(w, policy);
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

bool r8_stage_grid() {
    try {
        return !device_class().features.mma_bf16_native;
    } catch (const std::exception&) { return false; }
}

bool r8_fits(QType qtype, std::int32_t k, std::int32_t tokens) {
    if (tokens <= 0 || tokens > kR8MaxTokens || k <= 0) return false;
    std::int64_t slots = 1;
    while (slots < tokens) slots *= 2;
    // The codec's constants as the kernel is compiled with them (codec_traits.h): a codec with a
    // min also holds each block's sum (r8_smem_bytes), and a codebook codec's grid follows the
    // activation where r8_kernel stages it (r8_stage_grid), 8-byte aligned (r8_grid_pad).
    const auto codec = codec_traits(qtype);
    std::int64_t grid =
        r8_stage_grid() ? static_cast<std::int64_t>(sizeof(std::uint32_t)) * codec.grid_words : 0;
    if (grid > 0) grid += static_cast<std::int64_t>(r8_grid_pad);
    return slots * (static_cast<std::int64_t>(k) / 32) * (codec.min ? 40 : 36) + grid <=
           kStaticSharedBytesPerBlock;
}

// Q4_0 at the verify widths (T = 2..4): R8 pays on a large matrix and loses on a small one
// (A B B A, C = 4: Gemma 4 31B QAT, every dense matrix >= 44M values, RTX 4090 +15.7 %,
// RTX 3090 +26 %; Gemma 4 26B QAT, at most 12M, -0.7 % and -2.7 %, its MTP verify -1.1 %).
// The threshold and its format, with their cards: format_rules.h (Rule::R8VerifyQ40MinValues).
using format_rules::kR8VerifyQ40MinValues;

bool r8_verify_takes_shape(QType qtype, std::int64_t n, std::int64_t k) {
    if (!format_rules::holds(format_rules::Rule::R8VerifyQ40MinValues, qtype) ||
        n * k >= kR8VerifyQ40MinValues) {
        return r8_verify_takes(qtype);
    }
    // A small Q4_0 matrix keeps the A16 register MMA where it is native (Turing: R8, as for
    // every compiled format).
    try {
        return !device_class().features.mma_bf16_native && r8_verify_compiled(qtype);
    } catch (const std::exception&) { return false; }
}

// Whether the device's BF16 MMA is native (sm_80 and up); a device that cannot be queried
// counts as native, the routes that assume nothing about it.
bool bf16_mma_native() {
    try {
        return device_class().features.mma_bf16_native;
    } catch (const std::exception&) { return true; }
}

bool r8_verify_takes(QType qtype) {
    if (r8_verify_supports(qtype)) return true;
    try {
        return !device_class().features.mma_bf16_native && r8_verify_compiled(qtype);
    } catch (const std::exception&) { return false; }
}

// The build admits a tile its architecture holds; an sm_80 build run on a later 99 KiB part
// (forward-compatible SASS) must still turn the 144-160 KiB tiles away, so the device decides too.
bool staged_device_fits(int schedule) {
    const auto s      = kStagedSchedules.at(static_cast<std::size_t>(schedule));
    // A stage of every row and token (family_facts.h: one byte count for both).
    const auto bytes =
        static_cast<std::size_t>(s.rows + s.tokens) * kStagedMmaFacts.stage_row_bytes * s.stages;
    std::size_t limit = 0;
    try {
        limit = device_class().smem_per_block_optin;
    } catch (const std::exception&) {
        return true; // no device (a CPU-only tool): staged_fits holds the build ceiling
    }
    return limit == 0 || bytes <= limit;
}

WeightCodec codec_of(const Weight& weight) {
    if (const auto codec = format_facts(weight.qtype).codec) return *codec;
    throw std::invalid_argument("generic linear: unsupported weight format");
}

bool nvfp4_prefill_a8_policy(QType qtype, LinearPolicy policy) {
    return qtype == QType::NVFP4 && allows_a8(policy) &&
           !(w4a4_schedule_count() > 0 && allows_a4(policy));
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
    case QType::GGML_Q2_0: {
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
        return w.layout == QuantLayout::BlockScaleK32E8M0 && w.scales != nullptr && (w.k % 32) == 0;
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
    const bool quad_u4 =
        (codec == WeightCodec::U4ZpG32 || codec == WeightCodec::U4ZpG64) && k > 0 && (k % 128) == 0;
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

bool tiled_takes(WeightCodec codec) {
    return tiled_supports(codec) ||
           (compiled(Variant::TiledVolta, codec) && !current_features().mma_tensor);
}

bool volta_cutlass_only(WeightCodec codec) {
    return !tiled_takes(codec) && !current_features().mma_tensor;
}

R8CallRefusal r8_call_refusal(const Problem& problem, const Weight& w, std::int32_t tokens) {
    if (!allows_r8(w, problem.policy)) return R8CallRefusal::NotAdmitted;
    if (tokens <= kR8DecodeTokens) {
        return r8_fits(w.qtype, w.k, tokens) ? R8CallRefusal::None : R8CallRefusal::DecodeFit;
    }
    if (tokens > r8_verify_widest(w.qtype)) return R8CallRefusal::VerifyWidth;
    // The matrix the call reads: a slice of a fused parent is its own rows, not the parent's.
    return r8_verify_takes_shape(w.qtype, call_weight_rows(problem), w.k)
               ? R8CallRefusal::None
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
    const bool high_off =
        w.layout == QuantLayout::GgmlPlanes && w.qhigh != nullptr && !aligned_to(w.qhigh, 16);
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
        codec == WeightCodec::U4ZpG64 || codec == WeightCodec::GgmlQ4K ||
        codec == WeightCodec::GgmlQ5K || codec == WeightCodec::GgmlQ6K ||
        codec == WeightCodec::GgmlQ3K || codec == WeightCodec::GgmlQ2K ||
        codec == WeightCodec::GgmlIq4Xs || codec == WeightCodec::GgmlIq3Xxs ||
        codec == WeightCodec::GgmlIq3S || codec == WeightCodec::GgmlQ40 ||
        codec == WeightCodec::GgmlQ41 || codec == WeightCodec::GgmlQ50 ||
        codec == WeightCodec::GgmlQ51 || codec == WeightCodec::GgmlIq4Nl ||
        is_ggml_block(w.qtype)) {
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
    // ggml_layout_admits: whole blocks where the codec reads them (codec_traits block_bytes: Q4_K /
    // Q5_K), the others as planes, each plane it reads there.
    const bool planes = codec_traits(w.qtype).block_bytes == 0;
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
        return true;      // one lane order per element, independent of T
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
    // min-term sums in FP32 (rmma_add_sums), and the prefill profile's tiles take them all from
    // the call's plane or all with the MMA with ones (rmma_group_sums and the device's facts,
    // rmma_plane_admitted: one answer for every tile and width of a format and epilogue), so the
    // WarpsK == 1 tiles of every width and the staged MMA (no min-term format) sum alike. Outside
    // them the tiles of four or more token tiles do not.
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

int i8_pick(const Problem& problem, QType qtype, std::int32_t tokens) {
    const double sets = problem.epilogue == Epilogue::SwiGlu ? 2.0 : 1.0;
    const double sms  = std::max(sm_count(), 1);
    int best          = -1;
    double best_cost  = 0;
    for (int i = 0; i < i8_schedule_count(); ++i) {
        if (!i8_fits(i, qtype)) continue;
        const auto s = kI8Schedules[i];
        const double ctas =
            std::ceil(problem.n * sets / s.block_n) * std::ceil(tokens / double(s.block_m));
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

std::optional<Route> volta_tiled_route(const Weight& weight, LinearPolicy policy,
                                       std::int32_t tokens) {
    // Without tensor mma.sync (Volta) every register-MMA, W4A8 and W4A4 tile runs on FP32 FMAs or
    // dp4a over warp shuffles (ops/common/mma.cuh), while the tiled kernel issues the tensor cores'
    // own mma.m8n8k4 (generic_tiled.cuh): a call of 17 tokens or more whose format has a tiled
    // codec takes it whatever the policy allows (f16 activations, not int8 ones). Decode and verify
    // (T <= 16) keep their routes, so the switch is at T = 17.
    if (tokens < kPrefillBandMinTokens || current_features().mma_tensor || !supports(weight))
        return std::nullopt;
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
        current_features().mma_tensor || !supports(weight) || !volta_hmma_codec(codec_of(weight)) ||
        (weight.k % volta_hmma_k_multiple(codec_of(weight))) != 0) {
        return std::nullopt;
    }
    if (tokens <= kVecRuleMaxTokens &&
        (vec_call(weight, problem.policy, tokens) || r8_call(problem, weight, tokens))) {
        return std::nullopt;
    }
    const Route r{Family::RegisterMma,
                  static_cast<std::uint8_t>(heuristic_rmma(problem, tokens, false))};
    if (route_admits(r, weight, problem.policy)) return r;
    return std::nullopt;
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
        const Route r{Family::W4a4Mma,
                      static_cast<std::uint8_t>(
                          heuristic_w4a4(p.n, tokens, prefill && tokens >= kPrefillBandMinTokens))};
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
        const bool runs = tokens <= kR8DecodeTokens ? r8_call(p, w, tokens)
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

namespace {

constexpr PrefillRoute kPinnedPrefillFamilies[] = {
    PrefillRoute::A16, PrefillRoute::W4a8, PrefillRoute::I8, PrefillRoute::Mmq, PrefillRoute::W4a4};

constexpr DecodeRoute kPinnedDecodeFamilies[] = {DecodeRoute::A16, DecodeRoute::R8,
                                                 DecodeRoute::Vec, DecodeRoute::Mmq};
} // namespace

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

bool r8_verify_tunable(const Weight& w, LinearPolicy policy, std::int32_t tokens) {
    return tokens >= 2 && tokens <= kR8VerifyTokens && allows_r8(w, policy) &&
           r8_verify_compiled(w.qtype);
}

// Volta prefill (generic_volta_cutlass.cu): a tiled-route call of the prefill band (17 tokens on,
// kPrefillBandMinTokens) decodes its weight once and multiplies on CUTLASS's sm_70 GEMM. The tiled
// kernel (HMMA m8n8k4 on sm_70, generic_tiled.cuh) decodes a 128-row slice of the weight in each
// CTA, once per 128-token block: ceil(N / 128) CTAs a block, in waves of the SMs, each wave as long
// as one CTA's decode of 128 x K weights (0.37 .. 0.47 ms at K = 4096 on the V100). The decoded
// copy spreads the same decode over every SM, then reads two FP16 bytes a weight into the GEMM, so
// it costs in proportion to N and wins wherever the tile's grid leaves SMs idle or takes a second
// wave. V100 (ninferno_linear_bench, T = 32..240, the tile's time against the copy's): Q4_K 24576 x
// 4096 (192 CTAs: 2.4 waves) 1.36..2.29 ms against 1.20..1.40, 4096 x 12288 (32 CTAs) 1.07..1.31
// against 0.82..0.88, 4096 x 4096 0.36..0.44 against 0.27..0.28, Q5_K 4096 x 12288 1.13..1.37
// against 0.90..0.94. The one shape whose grid is a whole wave, 10240 x 4096 (80 CTAs on 80 SMs),
// keeps the tile ahead below 129 tokens (0.37..0.47 ms against 0.50) and loses from 129, two waves
// (0.82..0.93 against 0.52..0.54). A Qwen3.5 9B layer's four linears: 14 % less time at T = 64,
// 35 % at T = 192.
// The copy's narrowest margin above is 1.13 (24576 x 4096 below 129 tokens) at the V100's 870
// GB/s, and its time is its bytes over the DRAM rate while the tile's is its waves on the SMs (80
// on every Volta part): a board under 870 / 1.13 GB/s (the Titan V's 653) or of unknown bandwidth
// keeps the tile below kVoltaTileMaxTokens, where two token blocks already decode the weight twice.
constexpr std::int32_t kVoltaCutlassMinTokens = kPrefillBandMinTokens;

constexpr std::int32_t kVoltaTileMaxTokens = 256;

constexpr double kVoltaCopyMinDramGbps = 870.0 / 1.13;

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
// Reads the CURRENT device class (volta_cutlass_takes: device_class(), cached per device), not a
// device argument a description receives: describe_route / describe_call set it with
// ScopedDeviceClass, and the first call on a device fixes the cached answer.
bool volta_cutlass_runs(bool only, std::int32_t tokens) {
    return volta_cutlass_takes(tokens) && (only || volta_copy_pays(tokens));
}

// Whether a TiledMma call of a format with `codec` runs CUTLASS over the decoded copy: the band
// takes it (volta_cutlass_runs), the build has the kernel, and the call has workspace that holds
// a chunk. run_route asks it before launching and route_query's description after resolving, so
// they cannot drift apart.
bool volta_cutlass_call_runs(WeightCodec codec, std::int32_t n, std::int32_t k, std::int32_t tokens,
                             bool swiglu, std::optional<std::size_t> workspace_free) {
    return workspace_free.has_value() && volta_cutlass_compiled() &&
           volta_cutlass_runs(volta_cutlass_only(codec), tokens) &&
           volta_cutlass_chunk_bytes(n, k, tokens, swiglu, *workspace_free) != 0;
}

// A TiledMma call of a CUTLASS-only format (no tile of its own) that volta_cutlass_call_runs
// refuses runs the scalar kernel: a demotion (other bytes and speed).
bool volta_tiled_runs_scalar(WeightCodec codec, std::int32_t n, std::int32_t k, std::int32_t tokens,
                             bool swiglu, std::optional<std::size_t> workspace_free) {
    return volta_cutlass_only(codec) &&
           !volta_cutlass_call_runs(codec, n, k, tokens, swiglu, workspace_free);
}

} // namespace ninferno::ops::detail::generic
