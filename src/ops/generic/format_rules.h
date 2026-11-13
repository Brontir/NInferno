#pragma once

// The measured per-format rules of the generic path's route selection, with their provenance
// (#615 F6, Layer C of the format review): which formats a rule takes, the cards its evidence was
// timed on, where that evidence is written and when. The format facts (core/format_facts.h,
// format_traits.h) are true of a format's bytes on every card; what a binary compiles is the launch
// files' lists (compiled_codecs.h); what is here is neither: a sweep or product result on named
// cards, which must be re-checked when a kernel it was timed on changes.
//
// The code reads these rows: `holds(rule, q)` is the format set of a rule (mmq_verify_format,
// light_ggml_step, tall_tile_format, w4a8_ws_extra, r8_verify_supports and the Q4_0 matrix-size
// gate read it), and the two thresholds below are the constants the selection compares a call
// against. Nothing here was re-measured: the rows restate what the comments at the rules' uses
// say, and the selector goldens are unchanged by them.
//
// A rule may have several rows: one per group of formats whose evidence differs (other cards,
// another run). Each row says how many cards stand behind it (principle 4 of
// docs/maintainer/deterministic-decisions.md asks two lab cards of a hardware constant):
//   TwoCards    the row's evidence names two or more cards;
//   SingleCard  it names one card: the rule still decides on every card, which is the gap a later
//               two-card measurement closes (the architecture review's item 7 starts from these);
//   Carried     no timing of the format's own: it is in the set because the record says it
//               follows another row's formats (its family), and it names no card.
// test_format_rules holds the shape: every rule has a row with cards, a TwoCards row names two
// cards and a SingleCard row exactly one, a carried row none, each format stands in one row of a
// rule, the thresholds equal the constants and their route_rules.h descriptors (boards, cards,
// basis, value), and `holds` equals the rows. Whether a row's evidence is right is read from the
// sources by a person (the `source` field says where), not checked.
//
// Most rows' evidence predates the repository's public history: the comments at the rules' uses
// already stand in its first public commit (54ef5203, NInferno 0.1.0, 2026-10-20), and the lab
// runs behind them are not in the repository. Such a row says so in its source and dates the
// evidence "before 2026-10-20"; a row a later PR or decision entry measured names it.
//
// Fields (route_rules.h's descriptor shape where it fits):
//   rule      the rule (Rule) the row is a part of;
//   basis     what the row rests on (route_rules::Basis: Measured, PlacedBetweenMeasured, ...;
//             Unknown for a carried row);
//   formats   the formats of the row;
//   value     a threshold's value (the constant the code reads), 0 for a format-set row;
//   boards    device class ids (chip:Nsm) of the cards, "none" for a carried row;
//   cards     the products, comma-separated, "none" for a carried row;
//   evidence  TwoCards, SingleCard or Carried (above);
//   source    where the evidence is written (a comment at the rule's use, a PR, a decision entry);
//   date      when the evidence was taken, as far as the record says;
//   record    the numbers, as the source states them.

#include "core/weight.h"
#include "ops/generic/route_rules.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

namespace ninferno::ops::detail::generic::format_rules {

using route_rules::Basis;

enum class Rule : std::uint8_t {
    LightStep,            // light_ggml_step: the four-warp decode step past two CTAs a SM
    TallTile,             // tall_tile_format: the 128x32 register-MMA tile from 97 tokens
    MmqVerify,            // mmq_verify_format: ggml's matrix kernel at verify widths
    W4a8WsTuned,          // w4a8_ws_extra: the producer-warp W4A8 tile on sm_89 / sm_120
    R8Verify,             // r8_verify_supports: R8 at verify widths
    R8VerifyQ40MinValues, // kR8VerifyQ40MinValues: Q4_0's verify R8 on a large matrix only
    MmqVerifyMinTokens,   // kMmqVerifyMinTokens: the first width of the mmq verify band
};

inline constexpr std::array kAllRules = {Rule::LightStep,         Rule::TallTile,
                                         Rule::MmqVerify,         Rule::W4a8WsTuned,
                                         Rule::R8Verify,          Rule::R8VerifyQ40MinValues,
                                         Rule::MmqVerifyMinTokens};

constexpr std::string_view rule_name(Rule rule) {
    switch (rule) {
    case Rule::LightStep:
        return "light_step";
    case Rule::TallTile:
        return "tall_tile";
    case Rule::MmqVerify:
        return "mmq_verify";
    case Rule::W4a8WsTuned:
        return "w4a8_ws_tuned";
    case Rule::R8Verify:
        return "r8_verify";
    case Rule::R8VerifyQ40MinValues:
        return "kR8VerifyQ40MinValues";
    case Rule::MmqVerifyMinTokens:
        return "kMmqVerifyMinTokens";
    }
    return "";
}

enum class Evidence : std::uint8_t { TwoCards, SingleCard, Carried };

constexpr std::string_view evidence_name(Evidence e) {
    switch (e) {
    case Evidence::TwoCards:
        return "two_cards";
    case Evidence::SingleCard:
        return "single_card";
    case Evidence::Carried:
        return "carried";
    }
    return "";
}

// A set of formats, one bit per QType value (every named value is below 64: QType's largest is 43;
// a row naming one past it does not compile).
struct FormatSet {
    std::uint64_t bits = 0;

    static constexpr int kCapacity = 64;

    constexpr FormatSet() = default;

    constexpr FormatSet(std::initializer_list<QType> formats) {
        for (const QType q : formats) {
            const auto v = static_cast<unsigned>(q);
            if (v >= static_cast<unsigned>(kCapacity)) {
                throw std::logic_error("format_rules: a QType past the FormatSet's 64 bits");
            }
            bits |= std::uint64_t{1} << v;
        }
    }

    [[nodiscard]] constexpr bool has(QType q) const noexcept {
        const auto v = static_cast<unsigned>(q);
        return v < static_cast<unsigned>(kCapacity) && ((bits >> v) & 1u) != 0;
    }

    [[nodiscard]] constexpr bool empty() const noexcept { return bits == 0; }
};

struct FormatRule {
    Rule rule;
    Basis basis;
    FormatSet formats;
    std::int64_t value;
    std::string_view boards;
    std::string_view cards;
    Evidence evidence;
    std::string_view source;
    std::string_view date;
    std::string_view record;
};

inline constexpr std::string_view kNone         = "none";
inline constexpr std::string_view kBeforePublic = "before 2026-10-20";

// The thresholds the selection compares a call against (generic_select.cpp reads them). Their
// route_rules.h descriptors (file format_rules.h) hold the literals; the rows below their formats
// and cards.
// The first width of the mmq verify band (mmq_verify_rule): 9..16 tokens of several sequences.
inline constexpr std::int32_t kMmqVerifyMinTokens = 9;
// Q4_0 at the verify widths (T = 2..4) takes R8 only on a matrix of at least this many values
// (r8_verify_takes_shape).
inline constexpr std::int64_t kR8VerifyQ40MinValues = 24ll << 20;

namespace rows {
using Q = QType;

// The codebook formats of mmq_verify (a grid lookup per value in the register-MMA dequant).
inline constexpr FormatSet kMmqVerifyFormats = {Q::GGML_IQ4_XS, Q::GGML_IQ3_XXS, Q::GGML_IQ3_S,
                                                Q::GGML_IQ4_NL, Q::GGML_IQ2_XXS, Q::GGML_IQ2_XS,
                                                Q::GGML_IQ2_S,  Q::GGML_IQ1_S,   Q::GGML_IQ1_M};

inline constexpr auto kTable = std::to_array<FormatRule>({
    // --- light_step: the four-warp CTA of the ggml-step decoder (light_ggml_step) ---
    {Rule::LightStep, Basis::Measured,
     FormatSet{Q::GGML_Q4_K, Q::GGML_Q5_K, Q::GGML_IQ4_XS, Q::GGML_IQ3_XXS, Q::GGML_Q4_0}, 0,
     "gb203:84sm", "RTX 5080", Evidence::SingleCard,
     "light_ggml_step and heuristic_rmma's comments (generic_select.cpp); the comment stands in "
     "the first public commit (54ef5203), the lab runs behind it predate the repository's "
     "history; docs/maintainer/decisions.md 61 (29.10) records it as measured on one board",
     kBeforePublic,
     "T = 1, past two eight-warp CTAs per SM: four warps per CTA (one wave) 2-10 % faster at "
     "n = 3840..8192 (Q4_K 4096x4096 16.0 -> 14.7 us, Q5_K 18.1 -> 16.6, Q4_0 13.3 -> 12.8, "
     "IQ4_XS 15.1 -> 14.2); at n = 2560 eight warps stay faster. DECISIONS 61 questions it: past "
     "two CTAs a SM eight warps won on four of the five other boards it timed"},
    {Rule::LightStep, Basis::Measured, FormatSet{Q::GGML_IQ2_S, Q::GGML_IQ3_S}, 0, "gb203:84sm",
     "RTX 5080", Evidence::SingleCard,
     "light_ggml_step's comment (generic_select.cpp); the comment stands in the first public "
     "commit (54ef5203), the lab runs behind it predate the repository's history",
     kBeforePublic,
     "the codebook formats whose decode, not the weight stream, bounds them: four warps 11-16 % "
     "faster at n = 4096..12288 (IQ3_S 4096x2560 11.4 -> 9.7 us, IQ2_S 9216x2560 21.0 -> 18.6 "
     "us); Q8 and Q6_K at K >= 12288 were faster on eight warps at every size"},
    {Rule::LightStep, Basis::Unknown,
     FormatSet{Q::GGML_Q4_1, Q::GGML_Q5_0, Q::GGML_Q5_1, Q::GGML_IQ4_NL}, 0, kNone, kNone,
     Evidence::Carried,
     "light_ggml_step's comment (generic_select.cpp); the comment stands in the first public "
     "commit (54ef5203), the lab runs behind it predate the repository's history",
     kBeforePublic,
     "the comment names them only as 'the Q4_0 decoder family' beside Q4_0's timing: no timing "
     "of their own is recorded"},

    // --- tall_tile: Q4_K's 128x32 register-MMA tile (tall_tile_format) ---
    {Rule::TallTile, Basis::Measured, FormatSet{Q::GGML_Q4_K}, 0, route_rules::kSpeedFitBoards,
     route_rules::kSpeedFitCards, Evidence::TwoCards,
     "tall_tile_format's comment (generic_select.cpp); the comment stands in the first public "
     "commit (54ef5203), the lab runs behind it predate the repository's history; it says 'all "
     "seven boards' without naming them, read as the speed profile's seven-board fit as "
     "route_rules.h's 'heuristic_rmma: Q4_K tall tile' reads them",
     kBeforePublic,
     "from 97 tokens the 128x32 tile was 10..20 % faster than the 64x64 one on all seven boards "
     "timed; ptxas shows the 64x64 tile spilling where the 128x32 tile spills less (sm_86.."
     "sm_120), which is the reason, not the evidence"},

    // --- mmq_verify: ggml's matrix kernel at verify widths (mmq_verify_format) ---
    {Rule::MmqVerify, Basis::Measured, FormatSet{Q::GGML_IQ3_S}, 0, "ad104:58sm (L4), gb203:70sm",
     "L4, RTX 5070 Ti", Evidence::TwoCards,
     "mmq_verify_format's and mmq_verify_rule's comments (generic_select.cpp); the comment "
     "stands in the first public commit (54ef5203), the lab runs behind it predate the "
     "repository's history",
     kBeforePublic,
     "ISTA IQ3_S 27B (an IQ3_S mix) C4 MTP3, 16-token verify, every iq and K-quant band 9..16 on "
     "mmq: L4 79.4 -> 89.2 tok/s (+12.3 %), RTX 5070 Ti 242.7 -> 270.9 (+11.6 %); the K-quants "
     "left the set on the RTX 4090 (cellgt A B B A: UD-Q4_K_XL 27B DFlash2 C1 115.7 -> 111.7, C4 "
     "271.5 -> 263.1 with them in; Muse 30B Q4_K_M C4 282.7 -> 267.6). Which other formats the "
     "mix's tensors hold is not in the record, so the other codebook formats are carried"},
    {Rule::MmqVerify, Basis::Measured, FormatSet{Q::GGML_IQ4_XS}, 0, "tu102:68sm", "RTX 2080 Ti",
     Evidence::SingleCard, "#159 (mmq_verify_rule's Turing branch, generic_select.cpp)",
     "2026-10-27",
     "where the BF16 MMA is emulated a verify of 5..8 tokens keeps mmq for the codebook formats "
     "only: off mmq the K-quants were 10-58 % faster (Gemma 4 12B Q4_K_M MTP5 116.7 -> 136.1 "
     "tok/s) while the codebook IQ4_XS lost 23-38 %; no native-MMA board timed IQ4_XS's verify "
     "on its own"},
    {Rule::MmqVerify, Basis::Unknown,
     FormatSet{Q::GGML_IQ3_XXS, Q::GGML_IQ4_NL, Q::GGML_IQ2_XXS, Q::GGML_IQ2_XS, Q::GGML_IQ2_S,
               Q::GGML_IQ1_S, Q::GGML_IQ1_M},
     0, kNone, kNone, Evidence::Carried,
     "mmq_verify_format's comment (generic_select.cpp); the comment stands in the first public "
     "commit (54ef5203), the lab runs behind it predate the repository's history",
     kBeforePublic,
     "in the set by their class (the register MMA's dequant looks every value up in a grid per "
     "token tile, which the matrix kernel's one decode a call saves), not by a timing of their "
     "own"},

    // --- w4a8_ws_tuned: the producer-warp W4A8 tile where a measured choice names it ---
    {Rule::W4a8WsTuned, Basis::Measured, FormatSet{Q::GGML_IQ4_XS, Q::GGML_IQ2_XS, Q::GGML_IQ2_XXS},
     0, "gb206:36sm", "RTX 5060 Ti", Evidence::SingleCard,
     "w4a8_ws_wins' and w4a8_ws_extra's comments (generic_select.cpp); the comment stands in "
     "the first public commit (54ef5203), the lab runs behind it predate the repository's "
     "history; route_rules.h's 'w4a8_ws_extra: sm_89 and sm_120 only'",
     kBeforePublic,
     "ISTA GSQ IQ3_S mix, 24K prefill, A B B A, greedy byte-identical: RTX 5060 Ti 981 -> 1011 "
     "t/s, RTX 4060 Ti 900 -> 901; against ninfer-all's MMQ on the RTX 5060 Ti (nsys) these "
     "classic-tile formats carried the gap (IQ4_XS 16.1 s against 10.99, IQ2_XS 2.81 / 2.32, "
     "IQ2_XXS 1.00 / 0.79); taken where the shipped dense tuner table names the tile. The RTX "
     "4060 Ti ran it neutral (900 -> 901), so it is no second confirming card"},
    {Rule::W4a8WsTuned, Basis::Measured, FormatSet{Q::GGML_IQ3_S}, 0, "ad106:34sm", "RTX 4060 Ti",
     Evidence::SingleCard,
     "w4a8_ws_wins' and w4a8_ws_extra's comments (generic_select.cpp); the comment stands in "
     "the first public commit (54ef5203), the lab runs behind it predate the repository's "
     "history",
     kBeforePublic,
     "on its shared-table codec (W4a8Iq3SPaired): the same RTX 4060 Ti prefill 901 -> 926 t/s "
     "with IQ3_S added; on the global grid it lost per call (1.01-1.08: its decode did not keep "
     "four producer warps ahead of the MMA warps)"},

    // --- r8_verify: R8 at verify widths (r8_verify_supports) ---
    {Rule::R8Verify, Basis::Measured, FormatSet{Q::GGML_IQ2_S, Q::GGML_IQ3_S, Q::GGML_IQ3_XXS}, 0,
     "gb203:84sm", "RTX 5080", Evidence::SingleCard,
     "r8_verify_supports' comment (generic_gemm.h); the comment stands in the first public "
     "commit (54ef5203), the lab runs behind it predate the repository's history",
     kBeforePublic,
     "27B / 31B / 35B dense shapes, quantize + R8 against the register MMA: T = 4 0.73-0.91 on "
     "the large shapes; IQ4_XS 0.96-1.0 there but slower in the product (Qwen3.8-27B UD-IQ4_XS, "
     "MTP draft 2/3: -1.6 % / -2.4 % tok/s), Q4_K / Q5_K 1.0-1.3, Q6_K 1.15-1.77: those keep R8 "
     "at T = 1 only"},
    {Rule::R8Verify, Basis::Measured, FormatSet{Q::Q8_G32_FP16}, 0,
     "gb202:170sm, gb203:70sm, ad102:128sm", "RTX 5090, RTX 5070 Ti, RTX 4090", Evidence::TwoCards,
     "r8_verify_widest's comment (generic_gemm.h); the comment stands in the first public "
     "commit (54ef5203), the lab runs behind it predate the repository's history",
     kBeforePublic,
     "Q8 at T = 2..8 took 0.96-1.15 of its T = 1 time on R8 against 0.99-1.24 on the register "
     "MMA (RTX 5090 and 5070 Ti, ninferno_r8_schedule_bench); RTX 4090 Qwen3.5-9B Q8_0 dflash:7 "
     "229 -> 273 tok/s"},
    {Rule::R8Verify, Basis::Measured, FormatSet{Q::GGML_Q4_0}, 0, "ad102:128sm, gb202:170sm",
     "RTX 4090, RTX 5090", Evidence::TwoCards,
     "r8_verify_supports' comment (generic_gemm.h); the comment stands in the first public "
     "commit (54ef5203), the lab runs behind it predate the repository's history",
     kBeforePublic,
     "serve --spec none at C = 4, the same binary with and without it: Gemma 4 31B QAT (all "
     "Q4_0) on the RTX 4090 135 -> 161 tok/s, Gemma 4 26B-A4B Q4_0 on the RTX 5090 710 -> 728 "
     "(its attention projections); the matrix-size gate is kR8VerifyQ40MinValues' row"},
    {Rule::R8Verify, Basis::Unknown, FormatSet{Q::GGML_IQ2_XXS, Q::GGML_IQ2_XS}, 0, kNone, kNone,
     Evidence::Carried,
     "r8_verify_supports' comment (generic_gemm.h); the comment stands in the first public "
     "commit (54ef5203), the lab runs behind it predate the repository's history",
     kBeforePublic, "'IQ2_XXS / IQ2_XS follow their family' (IQ2_S): not timed on their own"},

    // --- the thresholds ---
    {Rule::R8VerifyQ40MinValues, Basis::PlacedBetweenMeasured, FormatSet{Q::GGML_Q4_0},
     kR8VerifyQ40MinValues, "ad102:128sm, ga102:82sm", "RTX 4090, RTX 3090", Evidence::TwoCards,
     "r8_verify_takes_shape's comment (generic_select.cpp); the comment stands in the first "
     "public commit (54ef5203), the lab runs behind it predate the repository's history; #710 "
     "(2026-11-11) made the gate read the call's own rows, not a fused parent's",
     kBeforePublic,
     "Q4_0 R8 at T = 2..4 pays on a large matrix (Gemma 4 31B QAT, every dense matrix >= 44M "
     "values, C = 4: RTX 4090 +15.7 %, RTX 3090 +26 %) and loses on a small one (Gemma 4 26B "
     "QAT, at most 12M: -0.7 %, -2.7 %, its MTP verify -1.1 %); nothing between 12M and 44M "
     "measured"},
    {Rule::MmqVerifyMinTokens, Basis::Measured, kMmqVerifyFormats, kMmqVerifyMinTokens,
     "ad104:58sm (L4), gb203:70sm, ad102:128sm, tu102:68sm",
     "L4, RTX 5070 Ti, RTX 4090, RTX 2080 Ti", Evidence::TwoCards,
     "mmq_verify_rule's comment (generic_select.cpp); the comment stands in the first public "
     "commit (54ef5203), the lab runs behind it predate the repository's history; the Turing "
     "branch from #159 (2026-10-27)",
     kBeforePublic,
     "the band 9..16 is the 16-token C4 verify of the mmq_verify rows (L4 +12.3 %, RTX 5070 Ti "
     "+11.6 %; the K-quants off it on the RTX 4090); where the BF16 MMA is emulated (RTX 2080 "
     "Ti) every matrix-kernel format takes mmq from 9 tokens, the codebook formats below that "
     "too; the record names no timing that places the edge at 9 rather than at 5..8 on a "
     "native-MMA "
     "board"},
});
} // namespace rows

inline constexpr const auto& kFormatRules = rows::kTable;

// The formats of a rule: the union of its rows.
constexpr FormatSet formats_of(Rule rule) noexcept {
    FormatSet out;
    for (const FormatRule& r : kFormatRules) {
        if (r.rule == rule) out.bits |= r.formats.bits;
    }
    return out;
}

// Whether the rule takes the format.
constexpr bool holds(Rule rule, QType q) noexcept { return formats_of(rule).has(q); }

static_assert(formats_of(Rule::MmqVerifyMinTokens).bits == formats_of(Rule::MmqVerify).bits,
              "format_rules.h: kMmqVerifyMinTokens' formats are the mmq_verify rows' formats");
static_assert(formats_of(Rule::R8VerifyQ40MinValues).bits == FormatSet{QType::GGML_Q4_0}.bits &&
                  holds(Rule::R8Verify, QType::GGML_Q4_0),
              "format_rules.h: the matrix-size gate is Q4_0's, a format r8_verify takes");

} // namespace ninferno::ops::detail::generic::format_rules
