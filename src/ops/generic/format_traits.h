#pragma once

// What the generic path knows about each stored format, as one table: a row per QType, read by the
// admission predicates of generic_gemm.cpp (r8_supports, w4a8_supports, ...). A new format is one
// row here, not a case label in a dozen switches. Every field records a fact about the format's
// codec (which kernel family decodes it, its group length) or a rule that the measurements behind
// the predicate's comment fixed per format; no field is a threshold. Reading a row never queries
// the device: a predicate that also looks at the card keeps that part and reads only the format
// from here.
//
// A QType without a row (a value the enum does not name) reads as the default row: no capability,
// the 32-value R8 step.

#include "core/weight.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail::generic {

// Whether a codec exists for the format on this build: on every build, or only where the native
// NVFP4 kernels are absent (NVFP4's R8 decode stands in for them: r8_supports).
enum class Support : std::uint8_t { No, Yes, WithoutNativeFp4 };

constexpr bool admits(Support s, bool native_fp4) noexcept {
    return s == Support::Yes || (s == Support::WithoutNativeFp4 && !native_fp4);
}

// Fields are in the order the rows below initialize them.
struct FormatTraits {
    QType q = QType::BF16;
    // The Q4/Q5/Q6 G64 weights (one scale and one high plane per 64 values): the formats of the
    // I8 tile and of the int8 prefill tiers (i8_format, qx_format).
    bool qx_g64 = false;
    // The codebook formats, whose register-MMA dequant looks every value up in a grid per token
    // tile: they take MMQ at verify widths (mmq_verify_format).
    bool mmq_verify = false;
    // Formats measured faster on the four-warp ring step of the ggml-step decoder
    // (light_ggml_step).
    bool light_step = false;
    // The format whose 64x64 register-MMA tile spills where its 128x32 tile spills less.
    bool tall_tile = false;
    // The W4A8 codec (classic tile) of the format.
    Support w4a8 = Support::No;
    // The format the producer-warp W4A8 tile is compiled for (w4a8_ws_supports).
    bool w4a8_ws = false;
    // Of those, the formats a measured dense-tune choice may move to the producer-warp tile on
    // sm_89 / sm_120 (w4a8_ws_extra).
    bool w4a8_ws_tuned = false;
    // K values of the group the W4A8 tile's zero tail may start on when K is not a whole 128-value
    // tile (w4a8_k_admits): 32 for the 32-value blocks, 64 for the 64-value groups (a K of whole
    // groups past the last 128-value tile, e.g. gpt-oss's 2880 in a native G64 conversion), or 0
    // for a format that needs whole tiles.
    std::uint8_t w4a8_tail_k = 0;
    // The R8 (decode-width) codec of the format.
    Support r8 = Support::No;
    // R8 has its pre-quantized (W4A8 plane) verify form for the format (r8_verify_compiled, with
    // r8).
    bool r8_prequant = true;
    // R8 takes the format at verify widths on every card (r8_verify_supports).
    bool r8_verify = false;
    // The K values one R8 step of the format covers (r8_k_values).
    std::uint8_t r8_k_values = 32;
};

namespace format_rows {
using Q = QType;
using S = Support;
// Rows are listed in QType order. A row that carries nothing is still listed, so that a missing
// row is a test failure (test_format_traits), not a silent default.
inline constexpr std::array kFormats{
    FormatTraits{
        .q = Q::Q4_G64_FP16, .qx_g64 = true, .w4a8 = S::Yes, .w4a8_tail_k = 64, .r8 = S::Yes},
    FormatTraits{
        .q = Q::Q5_G64_FP16, .qx_g64 = true, .w4a8 = S::Yes, .w4a8_tail_k = 64, .r8 = S::Yes},
    FormatTraits{
        .q = Q::Q6_G64_FP16, .qx_g64 = true, .w4a8 = S::Yes, .w4a8_tail_k = 64, .r8 = S::Yes},
    FormatTraits{
        .q = Q::Q8_G32_FP16, .w4a8 = S::Yes, .w4a8_tail_k = 32, .r8 = S::Yes, .r8_verify = true},
    FormatTraits{.q = Q::BF16},
    FormatTraits{.q = Q::FP32},
    FormatTraits{.q = Q::INT32},
    // NINFERNO-SPECIAL(arch:no-fp4-mma): without the native NVFP4 kernels decode multiplies the
    // doubled E2M1 codes on dp4a (W4a8Nvfp4, as the W4A8 prefill) instead of the BF16 register MMA,
    // so R8 takes NVFP4 only there; sm_120a keeps its native decode. The int8 tier multiplies the
    // doubled E2M1 codes exactly on every build (W4A8: Yes). With the native kernels W4A4 keeps the
    // weights it takes (allows_w4a8), and W4A8 runs the prefill of those whose policy allows A8
    // but not A4, which the BF16 register MMA ran before.
    FormatTraits{.q = Q::NVFP4, .w4a8 = S::Yes, .r8 = S::WithoutNativeFp4},
    FormatTraits{.q = Q::FP8_E4M3FN_ROW_BF16},
    FormatTraits{.q = Q::MXFP4, .w4a8 = S::Yes, .w4a8_tail_k = 32, .r8 = S::Yes},
    FormatTraits{.q = Q::FP8_E4M3FN_ROW_F32},
    FormatTraits{.q = Q::FP8_E4M3FN_B128_F32},
    // The u4 zero-point formats decode as in W4A8 (W4a8U4ZpG): before, a u4 checkpoint's decode
    // stayed on the A16 register MMA wherever its other formats took R8. U4 G32 is the G128 codec
    // with its group index.
    FormatTraits{.q = Q::U4_G128_ZP_FP16, .w4a8 = S::Yes, .r8 = S::Yes},
    FormatTraits{.q           = Q::GGML_Q4_K,
                 .light_step  = true,
                 .tall_tile   = true,
                 .w4a8        = S::Yes,
                 .w4a8_ws     = true,
                 .r8          = S::Yes,
                 .r8_k_values = 64},
    FormatTraits{
        .q = Q::GGML_Q5_K, .light_step = true, .w4a8 = S::Yes, .r8 = S::Yes, .r8_k_values = 64},
    FormatTraits{
        .q = Q::GGML_Q6_K, .w4a8 = S::Yes, .w4a8_ws = true, .r8 = S::Yes, .r8_k_values = 128},
    FormatTraits{
        .q = Q::GGML_Q3_K, .w4a8 = S::Yes, .r8 = S::Yes, .r8_prequant = false, .r8_k_values = 128},
    FormatTraits{
        .q = Q::GGML_Q2_K, .w4a8 = S::Yes, .r8 = S::Yes, .r8_prequant = false, .r8_k_values = 128},
    FormatTraits{.q             = Q::GGML_IQ4_XS,
                 .mmq_verify    = true,
                 .light_step    = true,
                 .w4a8          = S::Yes,
                 .w4a8_ws       = true,
                 .w4a8_ws_tuned = true,
                 .r8            = S::Yes},
    FormatTraits{.q          = Q::GGML_IQ3_XXS,
                 .mmq_verify = true,
                 .light_step = true,
                 .w4a8       = S::Yes,
                 .w4a8_ws    = true,
                 .r8         = S::Yes,
                 .r8_verify  = true},
    FormatTraits{.q             = Q::GGML_IQ3_S,
                 .mmq_verify    = true,
                 .light_step    = true,
                 .w4a8          = S::Yes,
                 .w4a8_ws       = true,
                 .w4a8_ws_tuned = true,
                 .r8            = S::Yes,
                 .r8_verify     = true},
    FormatTraits{.q           = Q::GGML_Q4_0,
                 .light_step  = true,
                 .w4a8        = S::Yes,
                 .w4a8_ws     = true,
                 .w4a8_tail_k = 32,
                 .r8          = S::Yes,
                 .r8_verify   = true},
    FormatTraits{
        .q = Q::GGML_Q4_1, .light_step = true, .w4a8 = S::Yes, .w4a8_tail_k = 32, .r8 = S::Yes},
    FormatTraits{
        .q = Q::GGML_Q5_0, .light_step = true, .w4a8 = S::Yes, .w4a8_tail_k = 32, .r8 = S::Yes},
    FormatTraits{
        .q = Q::GGML_Q5_1, .light_step = true, .w4a8 = S::Yes, .w4a8_tail_k = 32, .r8 = S::Yes},
    FormatTraits{.q           = Q::GGML_IQ4_NL,
                 .mmq_verify  = true,
                 .light_step  = true,
                 .w4a8        = S::Yes,
                 .w4a8_tail_k = 32,
                 .r8          = S::Yes},
    FormatTraits{.q             = Q::GGML_IQ2_XXS,
                 .mmq_verify    = true,
                 .w4a8          = S::Yes,
                 .w4a8_ws       = true,
                 .w4a8_ws_tuned = true,
                 .r8            = S::Yes,
                 .r8_verify     = true},
    FormatTraits{.q             = Q::GGML_IQ2_XS,
                 .mmq_verify    = true,
                 .w4a8          = S::Yes,
                 .w4a8_ws       = true,
                 .w4a8_ws_tuned = true,
                 .r8            = S::Yes,
                 .r8_verify     = true},
    FormatTraits{.q          = Q::GGML_IQ2_S,
                 .mmq_verify = true,
                 .light_step = true,
                 .w4a8       = S::Yes,
                 .w4a8_ws    = true,
                 .r8         = S::Yes,
                 .r8_verify  = true},
    // IQ1_S / IQ1_M decode as in W4A8 (W4a8Iq1S / W4a8Iq1M): before, their decode stayed on the A16
    // register MMA (27B UD-IQ2_XXS on the RTX 2080 Ti, nsys: the IQ1_S rmma calls 7.2 + 4.0 +
    // 2.6 ms of a 45 ms token).
    FormatTraits{.q = Q::GGML_IQ1_S, .mmq_verify = true, .w4a8 = S::Yes, .r8 = S::Yes},
    FormatTraits{.q = Q::GGML_IQ1_M, .mmq_verify = true, .w4a8 = S::Yes, .r8 = S::Yes},
    FormatTraits{.q = Q::GGML_TQ1_0},
    FormatTraits{.q = Q::GGML_TQ2_0},
    FormatTraits{.q = Q::GGML_Q1_0},
    FormatTraits{.q = Q::GGML_Q2_0},
    FormatTraits{.q = Q::U4_G32_ZP_FP16, .w4a8 = S::Yes, .w4a8_tail_k = 32, .r8 = S::Yes},
    FormatTraits{.q = Q::U4_G64_ZP_FP16, .w4a8 = S::Yes, .w4a8_tail_k = 64, .r8 = S::Yes},
    FormatTraits{.q = Q::Q3_G128_FP16, .w4a8 = S::Yes, .r8 = S::Yes},
    FormatTraits{.q = Q::Q3_G64_FP16, .w4a8 = S::Yes, .r8 = S::Yes},
};
} // namespace format_rows

inline constexpr const auto& kFormatTraits = format_rows::kFormats;
inline constexpr FormatTraits kNoFormatTraits{};

// Whether the table names the format.
constexpr bool has_format_row(QType q) noexcept {
    for (const FormatTraits& t : kFormatTraits) {
        if (t.q == q) return true;
    }
    return false;
}

// The format's row; the default row (no capability) for a value the table does not name.
constexpr const FormatTraits& format_traits(QType q) noexcept {
    for (const FormatTraits& t : kFormatTraits) {
        if (t.q == q) return t;
    }
    return kNoFormatTraits;
}

} // namespace ninferno::ops::detail::generic
