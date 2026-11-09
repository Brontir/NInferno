#pragma once

// What the generic path knows about each stored format, as one table: a row per QType, read by the
// admission predicates of generic_gemm.cpp (r8_supports, w4a8_supports, ...). A new format is one
// row here, not a case label in a dozen switches. Reading a row never queries the device: a
// predicate that also looks at the card keeps that part and reads only the format from here.
//
// Each field says what kind of knowledge it is:
//   codec fact       true of the bytes of the format, for every build and card;
//   compiled         true of this binary (which kernels instantiate the format); to move to the
//                    launch files' lists (architecture review item 3, F4);
//   measured rule    a per-card sweep result, not derived from the format: it holds on the cards
//                    named in its comment, and must be re-checked when a kernel it was measured on
//                    changes. No field is a threshold; a measured rule is one bit per format.
//
// A QType without a row (a value the enum does not name) reads as the default row: no capability,
// the 32-value R8 step.

#include "core/weight.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

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
    // [codec fact] The Q4/Q5/Q6 G64 weights (one scale and one high plane per 64 values): the
    // formats of the I8 tile and of the int8 prefill tiers (i8_format, qx_format).
    bool qx_g64 = false;
    // [measured rule] The codebook formats, whose register-MMA dequant looks every value up in a
    // grid per token tile: they take MMQ at verify widths (mmq_verify_format). RTX 4090 cellgt ABBA
    // (UD-Q4_K_XL 27B DFlash2, Muse 30B Q4_K_M), RTX 5070 Ti, RTX 2080 Ti (IQ4_XS lost 23-38 % on
    // the register MMA there); see mmq_verify_rule in generic_gemm.cpp.
    bool mmq_verify = false;
    // [measured rule] Formats measured faster on the four-warp ring step of the ggml-step decoder
    // (light_ggml_step): RTX 5080, T = 1, n = 3840..12288 (the comment of light_ggml_step).
    bool light_step = false;
    // [measured rule] The format whose 64x64 register-MMA tile (seven boards, ptxas sm_86..sm_120)
    // spills where its 128x32 tile spills less.
    bool tall_tile = false;
    // [compiled] The W4A8 codec (classic tile) of the format.
    Support w4a8 = Support::No;
    // [compiled] The format the producer-warp W4A8 tile is compiled for (w4a8_ws_supports).
    bool w4a8_ws = false;
    // [measured rule] Of those, the formats a measured dense-tune choice may move to the
    // producer-warp tile on sm_89 / sm_120 (w4a8_ws_extra).
    bool w4a8_ws_tuned = false;
    // [codec fact] K values of the group the W4A8 tile's zero tail may start on when K is not a
    // whole 128-value tile (w4a8_k_admits): 32 for the 32-value blocks, 64 for the 64-value groups
    // (a K of whole groups past the last 128-value tile, e.g. gpt-oss's 2880 in a native G64
    // conversion), or 0 for a format that needs whole tiles.
    std::uint8_t w4a8_tail_k = 0;
    // [compiled] The R8 (decode-width) codec of the format.
    Support r8 = Support::No;
    // [compiled] R8 has its pre-quantized (W4A8 plane) verify form for the format
    // (r8_verify_compiled, with r8).
    bool r8_prequant = true;
    // [measured rule] R8 takes the format at verify widths on every card (r8_verify_supports):
    // the formats R8 verify was timed on (r8_verify_takes_shape: Q4_0 RTX 4090 / 3090 A B B A).
    bool r8_verify = false;
    // [codec fact] The K values one R8 step of the format covers (r8_k_values).
    std::uint8_t r8_k_values = 32;
    // [codec fact] A grid-lookup decoder of IQ2's weight (IQ2_XXS, IQ2_XS, IQ2_S and IQ1_S / IQ1_M,
    // which decode like them): R8 schedules them as one class (rules_r8's iq2).
    bool sched_iq2 = false;
    // [measured rule] The decoders whose per-word cost, not the weight stream, bounds R8 on a card
    // without native BF16 MMA (rules_r8): the IQ2 class, IQ3_S, IQ3_XXS and Q3_K; RTX 2080 Ti, 63
    // problems of the IQ3 16 GB artifacts (ninferno_r8_schedule_bench).
    bool decode_heavy = false;
};

namespace format_rows {
using Q = QType;
using S = Support;
// Rows are listed in QType order. A row that carries nothing is still listed: the static_assert
// below fails the build for a named QType without exactly one row.
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
    FormatTraits{.q            = Q::GGML_Q3_K,
                 .w4a8         = S::Yes,
                 .r8           = S::Yes,
                 .r8_prequant  = false,
                 .r8_k_values  = 128,
                 .decode_heavy = true},
    FormatTraits{
        .q = Q::GGML_Q2_K, .w4a8 = S::Yes, .r8 = S::Yes, .r8_prequant = false, .r8_k_values = 128},
    FormatTraits{.q             = Q::GGML_IQ4_XS,
                 .mmq_verify    = true,
                 .light_step    = true,
                 .w4a8          = S::Yes,
                 .w4a8_ws       = true,
                 .w4a8_ws_tuned = true,
                 .r8            = S::Yes},
    FormatTraits{.q            = Q::GGML_IQ3_XXS,
                 .mmq_verify   = true,
                 .light_step   = true,
                 .w4a8         = S::Yes,
                 .w4a8_ws      = true,
                 .r8           = S::Yes,
                 .r8_verify    = true,
                 .decode_heavy = true},
    FormatTraits{.q             = Q::GGML_IQ3_S,
                 .mmq_verify    = true,
                 .light_step    = true,
                 .w4a8          = S::Yes,
                 .w4a8_ws       = true,
                 .w4a8_ws_tuned = true,
                 .r8            = S::Yes,
                 .r8_verify     = true,
                 .decode_heavy  = true},
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
                 .r8_verify     = true,
                 .sched_iq2     = true,
                 .decode_heavy  = true},
    FormatTraits{.q             = Q::GGML_IQ2_XS,
                 .mmq_verify    = true,
                 .w4a8          = S::Yes,
                 .w4a8_ws       = true,
                 .w4a8_ws_tuned = true,
                 .r8            = S::Yes,
                 .r8_verify     = true,
                 .sched_iq2     = true,
                 .decode_heavy  = true},
    FormatTraits{.q            = Q::GGML_IQ2_S,
                 .mmq_verify   = true,
                 .light_step   = true,
                 .w4a8         = S::Yes,
                 .w4a8_ws      = true,
                 .r8           = S::Yes,
                 .r8_verify    = true,
                 .sched_iq2    = true,
                 .decode_heavy = true},
    // IQ1_S / IQ1_M decode as in W4A8 (W4a8Iq1S / W4a8Iq1M): before, their decode stayed on the A16
    // register MMA (27B UD-IQ2_XXS on the RTX 2080 Ti, nsys: the IQ1_S rmma calls 7.2 + 4.0 +
    // 2.6 ms of a 45 ms token).
    FormatTraits{.q            = Q::GGML_IQ1_S,
                 .mmq_verify   = true,
                 .w4a8         = S::Yes,
                 .r8           = S::Yes,
                 .sched_iq2    = true,
                 .decode_heavy = true},
    FormatTraits{.q            = Q::GGML_IQ1_M,
                 .mmq_verify   = true,
                 .w4a8         = S::Yes,
                 .r8           = S::Yes,
                 .sched_iq2    = true,
                 .decode_heavy = true},
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

// Whether the enum names the value (a compiler prints an unnamed enumerator as a cast): the rows
// are checked against the enum itself, not against a second hand-written list of it.
#if defined(__GNUC__) || defined(__clang__)
template <QType Q>
constexpr bool named_qtype() noexcept {
    constexpr std::string_view where = __PRETTY_FUNCTION__;
    const std::size_t at             = where.find("Q = ");
    return at != std::string_view::npos && where[at + 4] != '(';
}

constexpr int format_rows_of(QType q) noexcept {
    int n = 0;
    for (const FormatTraits& t : kFormatTraits) n += t.q == q ? 1 : 0;
    return n;
}

template <std::size_t... V>
constexpr bool every_named_qtype_has_one_row(std::index_sequence<V...>) noexcept {
    // Each named value exactly once; and as many rows as named values, so no stale row survives.
    const bool once =
        ((!named_qtype<static_cast<QType>(V)>() || format_rows_of(static_cast<QType>(V)) == 1) &&
         ...);
    const int named = ((named_qtype<static_cast<QType>(V)>() ? 1 : 0) + ... + 0);
    return once && static_cast<std::size_t>(named) == kFormatTraits.size();
}

// 0..255 covers the enum (its largest value is 43); a value past it would need this raised.
static_assert(every_named_qtype_has_one_row(std::make_index_sequence<256>{}),
              "format_traits.h: every named QType needs exactly one row in kFormats, and every row "
              "a named QType");
#endif

} // namespace ninferno::ops::detail::generic
