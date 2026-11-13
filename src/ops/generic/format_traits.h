#pragma once

// What the generic path knows about each stored format, as one table: a row per QType, read by the
// admission predicates of generic_routes.cpp and generic_select.cpp (r8_k_values, w4a8_k_admits,
// r8_verify_supports, ...). A new format is one row here, not a case label in a dozen switches.
// Reading a row never queries the device: a predicate that also looks at the card reads only the
// format from here, and the card's part is a declared requirement over the board's discrete facts
// (format_requires.h, #615 F5).
//
// Each field says what kind of knowledge it is:
//   codec fact       true of the bytes of the format, for every build and card;
//   measured rule    a per-card sweep result, not derived from the format: it holds on the cards
//                    named in its comment, and must be re-checked when a kernel it was measured on
//                    changes. No field is a threshold; a measured rule is one bit per format.
// What this binary compiles for a format (its R8, W4A8, producer-warp and tiled codecs) is not
// here: it is the launch files' lists, compiled_codecs.h (#615 F4).
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

// Fields are in the order the rows below initialize them.
struct FormatTraits {
    QType q = QType::BF16;
    // [codec fact] The Q4/Q5/Q6 G64 weights (one scale and one high plane per 64 values): the
    // formats of the I8 tile and of the int8 prefill tiers (i8_format, qx_format).
    bool qx_g64 = false;
    // [measured rule] The codebook formats, whose register-MMA dequant looks every value up in a
    // grid per token tile: they take MMQ at verify widths (mmq_verify_format). Evidence: ISTA IQ3_S
    // 27B C4 MTP3, 16-token verify, every iq / K-quant band on mmq: L4 79.4 -> 89.2 tok/s
    // (+12.3 %), RTX 5070 Ti 242.7 -> 270.9 (+11.6 %); RTX 4090 cellgt ABBA (UD-Q4_K_XL 27B
    // DFlash2, Muse 30B Q4_K_M) is why the K-quants are NOT in the set (their affine dequant is
    // cheap); RTX 2080 Ti (IQ4_XS lost 23-38 % on the register MMA there). See mmq_verify_rule.
    bool mmq_verify = false;
    // [measured rule] Formats measured faster on the four-warp ring step of the ggml-step decoder
    // (light_ggml_step): RTX 5080, T = 1, n = 3840..12288 (the comment of light_ggml_step).
    bool light_step = false;
    // [measured rule] The format whose 128x32 register-MMA tile was 10..20 % faster than the 64x64
    // one from 97 tokens, on all seven boards timed. [compiled fact behind it]
    // ptxas shows the 64x64 tile spilling where the 128x32 tile spills less (sm_86..sm_120); that
    // is the reason, not the evidence.
    bool tall_tile = false;
    // [measured rule] Of the formats the producer-warp W4A8 tile is compiled for
    // (w4a8_ws_supports, compiled_codecs.h), those a measured dense-tune choice may move to it on
    // sm_89 / sm_120 (w4a8_ws_extra; the architectures: format_requires::kW4a8WsTunedArch).
    bool w4a8_ws_tuned = false;
    // [codec fact] K values of the group the W4A8 tile's zero tail may start on when K is not a
    // whole 128-value tile (w4a8_k_admits): 32 for the 32-value blocks, 64 for the 64-value groups
    // (a K of whole groups past the last 128-value tile, e.g. gpt-oss's 2880 in a native G64
    // conversion), or 0 for a format that needs whole tiles.
    std::uint8_t w4a8_tail_k = 0;
    // [measured rule] R8 takes the format at verify widths on every card (r8_verify_supports).
    // Evidence by row: IQ2_S, IQ3_S, IQ3_XXS: RTX 5080 shape micro-benchmark, T = 4, quantize + R8
    // against the register MMA 0.73-0.91 on the large shapes; IQ2_XXS and IQ2_XS: not separately
    // measured (they follow their family); Q8_G32: RTX 5090 and 5070 Ti bench, RTX 4090 spec-tune;
    // Q4_0: end to end on RTX 4090 / 5090 (generic_gemm.h, r8_verify_supports comment), and the
    // Q4_0 RTX 4090 / 3090 A B B A of r8_verify_takes_shape is the shape-threshold evidence
    // (n * k >= 24M values), not the format's.
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
// Rows are listed in QType order. A row that carries nothing is still listed: the static_assert
// below fails the build for a named QType without exactly one row.
inline constexpr std::array kFormats{
    FormatTraits{.q = Q::Q4_G64_FP16, .qx_g64 = true, .w4a8_tail_k = 64},
    FormatTraits{.q = Q::Q5_G64_FP16, .qx_g64 = true, .w4a8_tail_k = 64},
    FormatTraits{.q = Q::Q6_G64_FP16, .qx_g64 = true, .w4a8_tail_k = 64},
    FormatTraits{.q = Q::Q8_G32_FP16, .w4a8_tail_k = 32, .r8_verify = true},
    FormatTraits{.q = Q::BF16},
    FormatTraits{.q = Q::FP32},
    FormatTraits{.q = Q::INT32},
    FormatTraits{.q = Q::NVFP4},
    FormatTraits{.q = Q::FP8_E4M3FN_ROW_BF16},
    FormatTraits{.q = Q::MXFP4, .w4a8_tail_k = 32},
    FormatTraits{.q = Q::FP8_E4M3FN_ROW_F32},
    FormatTraits{.q = Q::FP8_E4M3FN_B128_F32},
    FormatTraits{.q = Q::U4_G128_ZP_FP16},
    FormatTraits{.q = Q::GGML_Q4_K, .light_step = true, .tall_tile = true, .r8_k_values = 64},
    FormatTraits{.q = Q::GGML_Q5_K, .light_step = true, .r8_k_values = 64},
    FormatTraits{.q = Q::GGML_Q6_K, .r8_k_values = 128},
    FormatTraits{.q = Q::GGML_Q3_K, .r8_k_values = 128, .decode_heavy = true},
    FormatTraits{.q = Q::GGML_Q2_K, .r8_k_values = 128},
    FormatTraits{
        .q = Q::GGML_IQ4_XS, .mmq_verify = true, .light_step = true, .w4a8_ws_tuned = true},
    FormatTraits{.q            = Q::GGML_IQ3_XXS,
                 .mmq_verify   = true,
                 .light_step   = true,
                 .r8_verify    = true,
                 .decode_heavy = true},
    FormatTraits{.q             = Q::GGML_IQ3_S,
                 .mmq_verify    = true,
                 .light_step    = true,
                 .w4a8_ws_tuned = true,
                 .r8_verify     = true,
                 .decode_heavy  = true},
    FormatTraits{.q = Q::GGML_Q4_0, .light_step = true, .w4a8_tail_k = 32, .r8_verify = true},
    FormatTraits{.q = Q::GGML_Q4_1, .light_step = true, .w4a8_tail_k = 32},
    FormatTraits{.q = Q::GGML_Q5_0, .light_step = true, .w4a8_tail_k = 32},
    FormatTraits{.q = Q::GGML_Q5_1, .light_step = true, .w4a8_tail_k = 32},
    FormatTraits{.q = Q::GGML_IQ4_NL, .mmq_verify = true, .light_step = true, .w4a8_tail_k = 32},
    FormatTraits{.q             = Q::GGML_IQ2_XXS,
                 .mmq_verify    = true,
                 .w4a8_ws_tuned = true,
                 .r8_verify     = true,
                 .sched_iq2     = true,
                 .decode_heavy  = true},
    FormatTraits{.q             = Q::GGML_IQ2_XS,
                 .mmq_verify    = true,
                 .w4a8_ws_tuned = true,
                 .r8_verify     = true,
                 .sched_iq2     = true,
                 .decode_heavy  = true},
    FormatTraits{.q            = Q::GGML_IQ2_S,
                 .mmq_verify   = true,
                 .light_step   = true,
                 .r8_verify    = true,
                 .sched_iq2    = true,
                 .decode_heavy = true},
    FormatTraits{.q = Q::GGML_IQ1_S, .mmq_verify = true, .sched_iq2 = true, .decode_heavy = true},
    FormatTraits{.q = Q::GGML_IQ1_M, .mmq_verify = true, .sched_iq2 = true, .decode_heavy = true},
    FormatTraits{.q = Q::GGML_TQ1_0},
    FormatTraits{.q = Q::GGML_TQ2_0},
    FormatTraits{.q = Q::GGML_Q1_0},
    FormatTraits{.q = Q::GGML_Q2_0},
    FormatTraits{.q = Q::U4_G32_ZP_FP16, .w4a8_tail_k = 32},
    FormatTraits{.q = Q::U4_G64_ZP_FP16, .w4a8_tail_k = 64},
    FormatTraits{.q = Q::Q3_G128_FP16},
    FormatTraits{.q = Q::Q3_G64_FP16},
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

// Rows against the enum's own names (named_qtype, core/format_facts.h): no second list of it.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
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
