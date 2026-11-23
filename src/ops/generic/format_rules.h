#pragma once

// The measured per-format rules of the generic path's route selection, with their provenance
// (#615 F6, Layer C of the format review): which formats a rule takes, the cards its evidence was
// timed on, where that evidence is written and when. The format facts (core/format_facts.h,
// format_traits.h) are true of a format's bytes on every card; what a binary compiles is the launch
// files' lists (compiled_codecs.h); what is here is neither: a sweep or product result on named
// cards, which must be re-checked when a kernel it was timed on changes.
//
// The code reads these rows: `holds(rule, q)` is the format set of a rule (mmq_verify_format,
// light_ggml_step, tall_tile_format, w4a8_ws_extra, r8_verify_supports, r8_verify_takes, rules_r8,
// vec_rule_max_tokens and the Q4_0 matrix-size gate read it), and the two thresholds below are the
// constants the selection compares a call against. The rows up to #615 F6 restate what the
// comments at the rules' uses say; vec_cap_ga10x_gb20x is new with its own end-to-end A/Bs (#998,
// DECISIONS 96), the three Turing rules with theirs (#1038, DECISIONS 97), which move the
// selector goldens of the RTX 2080 Ti board, vec_cap_ga10x with its (#1152, DECISIONS 99),
// which moves the GA10x boards'.
// mmq_verify_turing_low with its (#1040, DECISIONS 100), which moves the RTX 2080 Ti board's
// again.
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
//   value     a threshold's value (the constant the code reads), a vec cap row's cap (the widest
//             T the vector kernel keeps, kVecCapScopes), 0 for any other format-set row;
//   boards    device class ids (chip:Nsm) of the cards, "none" for a carried row;
//   cards     the products, comma-separated, "none" for a carried row;
//   evidence  TwoCards, SingleCard or Carried (above);
//   source    where the evidence is written (a comment at the rule's use, a PR, a decision entry);
//   date      when the evidence was taken, as far as the record says;
//   record    the numbers, as the source states them.

#include "core/weight.h"
#include "ops/generic/format_requires.h"
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
    LightStep,             // light_ggml_step: the four-warp decode step past two CTAs a SM
    TallTile,              // tall_tile_format: the 128x32 register-MMA tile from 97 tokens
    MmqVerify,             // mmq_verify_format: ggml's matrix kernel at verify widths
    W4a8WsTuned,           // w4a8_ws_extra: the producer-warp W4A8 tile on sm_89 / sm_120
    R8Verify,              // r8_verify_supports: R8 at verify widths
    R8VerifyQ40MinValues,  // kR8VerifyQ40MinValues: Q4_0's verify R8 on a large matrix only
    MmqVerifyMinTokens,    // kMmqVerifyMinTokens: the first width of the mmq verify band
    VecCapGa10xGb20x,      // vec_rule_max_tokens: the vector kernel's caps on GA10x and consumer
                           // Blackwell (a cap a row, kVecCapScopes)
    R8VerifyEmulatedOff,   // r8_verify_takes: formats R8 leaves at verify widths on Turing
    R8VerifyTwoRowsTuring, // rules_r8: formats whose Turing verify R8 takes two rows a warp
    VecCapTuring,          // vec_rule_max_tokens: the vector kernel's caps on Turing
    VecCapGa10x,           // vec_rule_max_tokens: the vector kernel's caps on GA10x alone
    VecCapAda,             // vec_rule_max_tokens: the vector kernel's caps on Ada
    MmqVerifyTuringLow,    // mmq_verify_rule: the formats Turing keeps on mmq below
                           // kMmqVerifyMinTokens
};

inline constexpr std::array kAllRules = {Rule::LightStep,
                                         Rule::TallTile,
                                         Rule::MmqVerify,
                                         Rule::W4a8WsTuned,
                                         Rule::R8Verify,
                                         Rule::R8VerifyQ40MinValues,
                                         Rule::MmqVerifyMinTokens,
                                         Rule::VecCapGa10xGb20x,
                                         Rule::R8VerifyEmulatedOff,
                                         Rule::R8VerifyTwoRowsTuring,
                                         Rule::VecCapTuring,
                                         Rule::VecCapGa10x,
                                         Rule::VecCapAda,
                                         Rule::MmqVerifyTuringLow};

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
    case Rule::R8VerifyEmulatedOff:
        return "r8_verify_emulated_off";
    case Rule::R8VerifyTwoRowsTuring:
        return "r8_verify_two_rows_turing";
    case Rule::R8VerifyQ40MinValues:
        return "kR8VerifyQ40MinValues";
    case Rule::MmqVerifyMinTokens:
        return "kMmqVerifyMinTokens";
    case Rule::VecCapGa10xGb20x:
        return "vec_cap_ga10x_gb20x";
    case Rule::VecCapTuring:
        return "vec_cap_turing";
    case Rule::VecCapGa10x:
        return "vec_cap_ga10x";
    case Rule::VecCapAda:
        return "vec_cap_ada";
    case Rule::MmqVerifyTuringLow:
        return "mmq_verify_turing_low";
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
     "on its own. Turing's T 5..8 set is mmq_verify_turing_low since #1040"},
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

    // --- vec_cap_ga10x_gb20x: the vec band stops at three tokens on GA10x and GB20x (#998) ---
    {Rule::VecCapGa10xGb20x, Basis::Measured, FormatSet{Q::GGML_Q4_K}, 3,
     "ga106:28sm, ga102:82sm, gb203:70sm, gb202:170sm", "RTX 3060, RTX 3090, RTX 5070 Ti, RTX 5090",
     Evidence::TwoCards, "#998: abvecn.sh t3_* (lab/vec-minn NINFERNO_LAB_VEC_MAX_T=3)",
     "2026-11-19",
     "Qwen3.5-9B Q4_K_M, --spec off, C = 4 (T = 4 calls) capped at T = 3 against vec at T = 4: "
     "RTX 3060 +6.5 %, RTX 3090 +3.6 %, RTX 5070 Ti +11.1 %, RTX 5090 +5.3 %; C = 2 and 3 within "
     "0.1 %"},
    {Rule::VecCapGa10xGb20x, Basis::Measured, FormatSet{Q::GGML_Q6_K}, 3,
     "gb202:170sm, gb203:70sm, ga106:28sm", "RTX 5090, RTX 5070 Ti, RTX 3060", Evidence::TwoCards,
     "#998: abvecn.sh t3q6nc_5090, t3q6_5070, t3q6_3060", "2026-11-19",
     "Qwen3.5-9B Q6_K, --spec off, C = 4 capped at T = 3 against T = 4: RTX 5090 +7.27 %, RTX "
     "5070 Ti +4.52 %, RTX 3060 +15.3 %; C = 2 and 3 within 0.05 %"},
    {Rule::VecCapGa10xGb20x, Basis::Measured, FormatSet{Q::GGML_Q3_K, Q::GGML_Q5_K}, 3,
     "gb203:70sm, ga106:28sm", "RTX 5070 Ti, RTX 3060", Evidence::TwoCards,
     "#998: abvecn.sh t3q3_5070, t3q5_5070, t3q3_3060, t3q5_3060", "2026-11-19",
     "Qwen3.5-9B Q3_K_M / Q5_K_M, --spec off, C = 4 capped at T = 3 against T = 4: RTX 5070 Ti "
     "+15.7 % / +6.74 %, RTX 3060 +0.63 % / +10.76 %; C = 2 and 3 within 0.1 %"},
    {Rule::VecCapGa10xGb20x, Basis::Unknown, FormatSet{Q::GGML_Q2_K}, 3, kNone, kNone,
     Evidence::Carried, "#998", "2026-11-19",
     "Q2_K follows the other K-quants, whose super-block decoder it shares: not timed on its own. "
     "Not in the rule: IQ4_XS (RTX 5070 Ti +4.45 %, RTX 5090 -3.27 %, RTX 3060 -11.3 %), "
     "the formats R8 verifies, IQ1 and the legacy formats (not measured)"},

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
     "Ti) every matrix-kernel format takes mmq from 9 tokens, IQ4_XS / IQ4_NL below that too "
     "(mmq_verify_turing_low); the record names no timing that places the edge at 9 rather than at "
     "5..8 on a "
     "native-MMA "
     "board"},

    // --- r8_verify_emulated_off: K-quants keep the register MMA at verify widths on Turing ---
    {Rule::R8VerifyEmulatedOff, Basis::Measured,
     FormatSet{Q::GGML_Q3_K, Q::GGML_Q4_K, Q::GGML_Q6_K}, 0, "tu102:68sm", "RTX 2080 Ti",
     Evidence::SingleCard, "#1038: selcheck.sh 75 1038off / 1038fmt (route_tune, vec rule off)",
     "2026-11-19",
     "T = 3 / 4, the best R8 schedule against rmma n1: Q4_K 4096x4096 30.7 / 33.3 against 23.2 / "
     "23.5 us, Q6_K 12288x4096 126.1 / 150.5 against 87.0 / 88.1, Q3_K 4096x4096 41.9 / 43.8 "
     "against 23.9 / 24.8; R8's time grows with T, the register MMA's does not"},
    {Rule::R8VerifyEmulatedOff, Basis::Unknown, FormatSet{Q::GGML_Q5_K}, 0, kNone, kNone,
     Evidence::Carried, "#1038", "2026-11-19",
     "Q5_K follows Q4_K, whose decoder it shares: not timed on its own. The rows say T = 3 / 4; "
     "r8_verify_takes keys on the format, so R8 also leaves T = 2, where vec takes the call "
     "(VecCapTuring) unless vec declines it"},

    // --- r8_verify_two_rows_turing: two rows a warp at Turing's verify widths T = 3..4 ---
    {Rule::R8VerifyTwoRowsTuring, Basis::Measured,
     FormatSet{Q::GGML_IQ3_S, Q::GGML_IQ3_XXS, Q::GGML_IQ4_XS, Q::GGML_Q4_0}, 0, "tu102:68sm",
     "RTX 2080 Ti", Evidence::SingleCard, "#1038: selcheck.sh 75 1038fmt (route_tune)",
     "2026-11-19",
     "T = 3 / 4, r8.w4r1 against r8.w4r2: IQ3_S 12288x4096 83.1 / 93.9 against 67.5 / 72.9 us, "
     "IQ3_XXS 4096x4096 30.8 / 34.5 against 25.5 / 27.4, IQ4_XS 4096x4096 24.6 / 27.0 against "
     "24.5 / 25.3, Q4_0 12288x4096 60.2 / 68.6 against 57.9 / 59.9; at T = 2 within 1 %, where "
     "w4r1 is ahead for IQ4_XS and Q4_0. Not timed: the other R8 formats (NVFP4, MXFP4, U4 "
     "zero-point, the native Q3-Q6 groups), which keep w4r1. The shipped picks do not change "
     "today: vec takes T = 2..4 for these formats (kVecRuleMaxTokens) and r8_verify_widest is 4, "
     "so the rule decides where vec declines a call, under NINFERNO_LAB_NO_VEC_RULE and in the "
     "dense tuner's R8 candidate"},

    // --- vec_cap_turing: the vec band stops at two tokens on Turing (#1038) ---
    {Rule::VecCapTuring, Basis::Measured, FormatSet{Q::GGML_Q4_K, Q::GGML_Q6_K}, 2, "tu102:68sm",
     "RTX 2080 Ti", Evidence::SingleCard, "#1038: abvecn.sh on the RTX 2080 Ti (lab/vec-minn)",
     "2026-11-19",
     "Qwen3.5-9B, --spec off, vec at T = 2 and the register MMA at T = 3..4 against vec at "
     "T = 2..4, the real branch against master (abref, 8 alternating starts): Q4_K_M C = 2 +0.08 "
     "%, "
     "C = 3 +2.57 %, C = 4 +3.44 %; Q6_K +0.05 %, +28.5 %, +33.4 %. The vec-against-rmma edge "
     "split cards of one generation on GA10x (cap at T = 2: RTX 3090 C3 -2.0 %, RTX 3060 +1.5 %), "
     "so on another Turing board this row is the weak one"},
    {Rule::VecCapTuring, Basis::Unknown, FormatSet{Q::GGML_Q3_K, Q::GGML_Q5_K}, 2, kNone, kNone,
     Evidence::Carried, "#1038", "2026-11-19",
     "Q3_K and Q5_K follow Q4_K / Q6_K, whose register-MMA verify they share (R8VerifyEmulatedOff: "
     "Q3_K timed at kernel level, Q5_K carried): not timed end to end. Not in the rule: IQ4_XS "
     "(capped at T = 2: -0.05 / +0.93 / -0.74 %) and every format R8 verifies"},
    {Rule::VecCapTuring, Basis::Measured, FormatSet{Q::GGML_Q4_0}, 2, "tu102:68sm", "RTX 2080 Ti",
     Evidence::SingleCard,
     "#1160: abvec234.sh q40g12l2080 (lab/vec-max8 NINFERNO_LAB_VEC_MAX=2), abref q40brl2080",
     "2026-11-23",
     "Gemma 4 12B QAT Q4_0 (google), --spec off, default activations, vec to T = 2 against "
     "master's T = 2..4, 4 alternating starts: C = 4 +3.7 %, C = 3 +1.4 %, C = 2 +0.0 %; the "
     "real branch against master C = 4 +3.5 %, C = 3 +1.3 %"},

    // --- vec_cap_ga10x: the vector kernel's caps on GA10x alone (#1152) ---
    {Rule::VecCapGa10x, Basis::Measured, FormatSet{Q::GGML_Q6_K}, 2, "ga106:28sm, ga102:82sm",
     "RTX 3060, RTX 3090", Evidence::TwoCards,
     "#1152: abvecn.sh q6v1152 (RTX 3060), q6v1152g (RTX 3090) (lab/1152-vec-cap "
     "NINFERNO_LAB_VEC_MAX_T=1)",
     "2026-11-21",
     "Qwen3.5-9B Q6_K, --spec off, vec at T = 1 only against master (vec at T = 2..3), 4 "
     "alternating starts each: C = 3 RTX 3060 +10.7 %, RTX 3090 +5.0 %; C = 2 RTX 3060 +4.0 %, RTX "
     "3090 -2.3 % (the T = 2 edge splits the cards: T = 2 keeps vec); C = 4 within 0.1 %"},
    {Rule::VecCapGa10x, Basis::Measured, FormatSet{Q::GGML_Q5_K}, 2, "ga106:28sm, ga102:82sm",
     "RTX 3060, RTX 3090", Evidence::TwoCards,
     "#1152: abvecn.sh q5v1152c (RTX 3060), q5v1152g (RTX 3090) (lab/1152-vec-cap)", "2026-11-21",
     "Qwen3.5-9B Q5_K_M, --spec off, as Q6_K: C = 3 RTX 3060 +10.4 %, RTX 3090 +5.2 %; C = 2 RTX "
     "3060 +1.4 %, RTX 3090 -4.3 % (T = 2 keeps vec); C = 4 within 0.2 %. Not in the rule: Q4_K "
     "(RTX 3060 C = 3 +1.5 %, C = 2 -1.6 %), Q3_K (RTX 3060 vec ahead: C = 2 -11.6 %, C = 3 "
     "-5.7 %) and the consumer Blackwell cards (not measured at T = 3)"},
    {Rule::VecCapGa10x, Basis::Measured, FormatSet{Q::GGML_Q4_0}, 2, "ga106:28sm, ga102:82sm",
     "RTX 3060, RTX 3090", Evidence::TwoCards,
     "#1160: abvec234.sh q40g12l3060, q40g12c3l3060, q40g31l3090 (lab/vec-max8 "
     "NINFERNO_LAB_VEC_MAX), abref q40brl3060",
     "2026-11-23",
     "google's Gemma 4 QAT Q4_0, --spec off, default activations, vec to T = 2 against master's "
     "T = 2..4, 4 alternating starts: RTX 3060 (12B) C = 4 +8.6 %, C = 3 +1.4 %; RTX 3090 (31B, "
     "four streams at most) C = 4 +12.3 %, C = 3 +5.3 %; C = 2 0.0 % on both; the real branch "
     "against master on the RTX 3060 C = 4 +8.8 %, C = 3 +1.4 %. Grid (rtband T 1-8): Q4_0 vec "
     "1.10-1.48x behind rmma n1.w1x4 at T = 3..4 on most shapes of both cards"},

    // --- vec_cap_ada: the vector kernel's caps on Ada (#1160) ---
    {Rule::VecCapAda, Basis::Measured, FormatSet{Q::GGML_Q4_0}, 2,
     "ad106:34sm, ad104:60sm, ad102:128sm", "RTX 4060 Ti, RTX 4070 Ti, RTX 4090",
     Evidence::TwoCards,
     "#1160: abvec234.sh q40g12l4060, q40g12l4070, q40g12c3l4070, q40g31l4090 (lab/vec-max8 "
     "NINFERNO_LAB_VEC_MAX)",
     "2026-11-23",
     "google's Gemma 4 QAT Q4_0, --spec off, default activations, vec to T = 2 against master's "
     "T = 2..4, 4 alternating starts: C = 4 RTX 4060 Ti +0.9 %, RTX 4070 Ti +1.5 %, RTX 4090 "
     "(31B) +0.5 %; C = 3 +0.7 / +1.4 / +0.9 %; C = 2 0.0 %. Small, positive on all three. Not "
     "in the rule: the K-quants (decision 96: the RTX 4070 Ti and 4090 lost with a cap) and "
     "consumer Blackwell for Q4_0 (RTX 5070 Ti +1.1 %, RTX 5090 31B -4.0 % at C = 4: vec "
     "ahead on the 5090)"},

    // --- mmq_verify_turing_low: the formats Turing keeps on mmq at T < kMmqVerifyMinTokens ---
    {Rule::MmqVerifyTuringLow, Basis::Measured, FormatSet{Q::GGML_IQ4_XS}, 0, "tu102:68sm",
     "RTX 2080 Ti", Evidence::SingleCard,
     "#1040: abvec8m.sh iq4v1040 (lab/1040-mmq-verify-turing); #159", "2026-11-21",
     "Qwen3.5-9B IQ4_XS, --spec off, C = 5 / 6 / 8, mmq against the lab switch that leaves it: "
     "mmq ahead +28.4 / +28.9 / +28.3 % (#159: 23-38 % on MTP verify)"},
    {Rule::MmqVerifyTuringLow, Basis::Unknown, FormatSet{Q::GGML_IQ4_NL}, 0, kNone, kNone,
     Evidence::Carried, "#1040 (carried from IQ4_XS)", "2026-11-22",
     "IQ4_NL decodes as IQ4_XS does (one 16-entry table value a 4-bit code, no grid): carried "
     "from the IQ4_XS row. The grid formats left the set: RTX 2080 Ti route_tune at T 5-8, mmq "
     "1.31x (IQ1_S, 32 cells) and 1.45-1.58x (IQ2_S / IQ2_XS / IQ2_XXS, IQ3_S / IQ3_XXS) behind "
     "R8 or the register MMA; Qwen3.8-27B UD-IQ2_XXS C = 5 / 6 / 8 +5.4 / +2.5 / +1.7 % off mmq"},
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

// The vector kernel's caps (vec_rule_max_tokens): a rule a generation scope, a cap a row (`value`:
// the widest T the vector kernel keeps for the row's formats). A call takes the least cap of the
// rows that name its format under the scopes its card meets; with none it keeps
// kVecRuleMaxTokens (generic_select.cpp). A new cap is a row of a scope's rule; a new scope is a
// rule and a line here.
struct VecCapScope {
    Rule rule;
    ArchRequires arch;
};

inline constexpr std::array kVecCapScopes = {
    VecCapScope{Rule::VecCapGa10xGb20x, ArchRequires{.cc_any_of = {{{8, 6}, {12, 0}}}}},
    VecCapScope{Rule::VecCapTuring, format_requires::kR8VerifyTuring},
    VecCapScope{Rule::VecCapGa10x, ArchRequires{.cc_any_of = {{{8, 6}}}}},
    VecCapScope{Rule::VecCapAda, ArchRequires{.cc_any_of = {{{8, 9}}}}},
};

// The cap a scope's rows give format q (0: no row names it).
constexpr std::int64_t vec_cap(Rule rule, QType q) noexcept {
    for (const FormatRule& r : kFormatRules) {
        if (r.rule == rule && r.formats.has(q)) return r.value;
    }
    return 0;
}

static_assert(formats_of(Rule::MmqVerifyMinTokens).bits == formats_of(Rule::MmqVerify).bits,
              "format_rules.h: kMmqVerifyMinTokens' formats are the mmq_verify rows' formats");
static_assert((formats_of(Rule::MmqVerifyTuringLow).bits & ~formats_of(Rule::MmqVerify).bits) == 0,
              "format_rules.h: Turing's low mmq set is part of the mmq_verify rows' formats");
static_assert(formats_of(Rule::R8VerifyQ40MinValues).bits == FormatSet{QType::GGML_Q4_0}.bits &&
                  holds(Rule::R8Verify, QType::GGML_Q4_0),
              "format_rules.h: the matrix-size gate is Q4_0's, a format r8_verify takes");

} // namespace ninferno::ops::detail::generic::format_rules
