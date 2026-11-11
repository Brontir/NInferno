#pragma once

// The literal rules of the generic path's route selection (generic_gemm.h, generic_gemm.cpp,
// generic_select.cpp, which holds select_route and its heuristics, and route_authorities.cpp,
// which holds the authority cascade and the NVFP4 half-the-L2 rule): the
// thresholds a rule compares a call's shape, width or card against, each with what the source says
// it rests on: a timing on named boards, a point placed between timings, a fit, a structural fact,
// or nothing recorded (#614, PR-F4). This header declares what each rule IS (its literal) and what
// the record says it rests on (a basis class, the boards and cards, the margin the record gives,
// where it is written); it changes no behaviour. The code keeps its own literals. What
// test_route_rules enforces, and no more: a descriptor's literal stands in its source file (a
// named constant's value matches, a rule's literal text is found, whole, outside comments and
// strings, as many times as the descriptor says); every constant of the scanned files whose name
// says it bounds something (a scan by file and name, see the test) has a descriptor or is a listed
// non-rule; every descriptor has a basis class and non-empty boards, cards and tolerance; a
// Measured one names a board id (chip:Nsm), a Fitted, PlacedBetweenMeasured or ScaledFromOneBoard
// one names a board id or says none is recorded, and a Structural constant is defined from another
// constant, held by a static_assert or listed in the test with its reason. It does not check that
// a class is the right one (that is read from the source comments by a person, and can be wrong),
// that a described constant is used the way the descriptor says, or that a rule did not move (see
// docs/maintainer/route-rules.md, which lists the guard's known gaps).
//
// What the list covers: the named constants that bound a route, the threshold comparisons of
// rules_r8, w4a8_ws_wins, heuristic_rmma, heuristic_w4a8, heuristic_w4a4, staged_speed_route,
// tiled_speed_pays, nvfp4_tiled_route, the NVFP4 half-the-L2 rule (nvfp4_half_l2 and its weight
// bytes, route_authorities.cpp), the R8 table's one-token gate and the narrow register-MMA
// fallback tiles (route_authorities.cpp), the vector and R8 verify edges and spread_smem, and the
// table of boards that decides spread_smem. What it does not: see docs/maintainer/route-rules.md,
// which names the literals knowingly left out (among them the comparators at uses of named
// constants and the order of the three authority arrays, held by the selector goldens only).
//
// Fields:
//   name       the rule as the code names it (`symbol` or a short name for a literal inside one);
//   basis      what the number rests on, one of Basis below; the class is read from the source
//              comments, docs/maintainer/decisions.md and deterministic-decisions.md, and is
//              Unknown where they do not say;
//   file       the source file that holds the literal, under src/ops/generic/ (or a path from it
//              for a file beside them);
//   literal    the text of the literal as written: a constant's WHOLE definition (to its `;`, an
//              array with every element), or a comparison with its action where the action matters;
//   value      the number of a constant (the test evaluates the constant's initialiser itself,
//              integers staying integers, and holds it to this number)
//              or of a condition with ONE numeric cutoff that is written in the literal; 0 for an
//              array, a condition with several cutoffs and a condition that compares against a
//              card's own quantity (all cutoffs are then in the literal);
//   boards     the device class ids the source names for it (DeviceClass::id: chip:Nsm, one format
//              throughout), "unknown" where the comments and the record name none;
//   cards      the same as the products' names;
//   tolerance  the measured margin the record gives ("none recorded" where it gives none): what the
//              rule was held to, or what moving the literal was seen to cost;
//   kernel_hash  the identity of the kernel the rule was measured against: EMPTY for every rule.
//              The repository records one sha256 over all device sources per kernel-facts file
//              (kernel_facts.h Facts::source, tools/kernel_facts), not per kernel, and no rule
//              records which tree it was timed on; a rule re-measured later should fill this.
// Where the record is silent a field says "unknown". The source of a number is the comment at its
// use (quoted in the tolerance where it states a margin), docs/maintainer/decisions.md and
// docs/maintainer/deterministic-decisions.md.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ninferno::ops::detail::generic::route_rules {

// What a threshold rests on (the same idea as route_cost_constants.h's Provenance):
//   Structural             follows from a kernel's shape, a contract, the hardware's layout or
//                          another constant, not from a timing (a width a kernel takes, a band
//                          defined from another band's edge, a byte-consistency contract);
//   Measured               a timing or product run of the record names the value or the margin it
//                          was held to, on the boards the descriptor names;
//   PlacedBetweenMeasured  the record has measured points on both sides and the value was put
//                          between them: the line itself has no measurement;
//   Fitted                 chosen to make a rule match timings of several boards (a fit, scored on
//                          a held-out board where the comment says so), not derived from a fact;
//   ScaledFromOneBoard     one board's measured margin carried to other boards by a stated model
//                          (a rate, a ratio): no second board was measured;
//   Unknown                the record does not say what it rests on (the rule's action may be
//                          measured; this edge is not).
// A constant fitted, placed between boards or scaled names those boards, or says none are
// recorded; a Measured one names its boards.
enum class Basis : std::uint8_t {
    Structural,
    Measured,
    PlacedBetweenMeasured,
    Fitted,
    ScaledFromOneBoard,
    Unknown
};

inline constexpr std::array<Basis, 6> kAllBases = {
    Basis::Structural, Basis::Measured,           Basis::PlacedBetweenMeasured,
    Basis::Fitted,     Basis::ScaledFromOneBoard, Basis::Unknown};

constexpr std::string_view basis_name(Basis basis) {
    switch (basis) {
    case Basis::Structural:
        return "Structural";
    case Basis::Measured:
        return "Measured";
    case Basis::PlacedBetweenMeasured:
        return "PlacedBetweenMeasured";
    case Basis::Fitted:
        return "Fitted";
    case Basis::ScaledFromOneBoard:
        return "ScaledFromOneBoard";
    case Basis::Unknown:
        return "Unknown";
    }
    return "";
}

struct RouteRule {
    std::string_view name;
    Basis basis;
    std::string_view file;
    std::string_view literal;
    double value;
    std::string_view boards;
    std::string_view cards;
    std::string_view tolerance;
    std::string_view kernel_hash;
    // How many times the literal stands in its file outside comments (the test holds it): 1 where
    // the rule is written once; more where one literal serves several branches.
    std::size_t occurrences = 1;
};

inline constexpr std::string_view kNotRecorded = "none recorded";
inline constexpr std::string_view kUnknown     = "unknown";

// The boards of the speed profile's seven-board fit (ninferno_route_tune): the five lab boards the
// comments name, with their class ids, and the A10 (ga102:72sm) and the L4 (ad104:58sm).
inline constexpr std::string_view kSpeedFitBoards =
    "ga102:82sm, ad102:128sm, gb202:170sm, tu102:68sm, ga100:108sm, ga102:72sm (A10), ad104:58sm "
    "(L4)";
inline constexpr std::string_view kSpeedFitCards =
    "RTX 3090, RTX 4090, RTX 5090, RTX 2080 Ti, A100, A10, L4";

inline constexpr auto kRouteRules = std::to_array<RouteRule>({
    // --- R8's widths, the matrix kernel's band and the verify rules (generic_gemm.h) ---
    {"kR8MaxTokens", Basis::Measured, "generic_gemm.h",
     "inline constexpr std::int32_t kR8MaxTokens = 4;", 4, "gb203:84sm", "RTX 5080",
     "no four-row schedule and at most four tokens: the K-quant codecs' registers spilled (up to 2 "
     "KiB of stack a thread, w4r4 SwiGLU at eight tokens); w4r4 was the best T = 1 schedule on 2 "
     "of 70 shapes, by 1 %",
     ""},
    {"kR8DecodeTokens", Basis::Measured, "generic_gemm.h",
     "inline constexpr std::int32_t kR8DecodeTokens = 1;", 1, "gb203:84sm", "RTX 5080",
     "R8 at T = 1 only: at T = 4 it lost on 26 of 42 decode shapes of Qwen3.5-4B/9B; at T = 1 it "
     "won on all 42 (0.58-0.96 of the register MMA) and on 69 of 70 Qwen3.8-27B / Gemma 4 31B / "
     "Qwen3.6-35B-A3B shapes (0.77-0.99; Q6_K 4096x2048 1.05)",
     ""},
    {"kR8VerifyTokens", Basis::Measured, "generic_gemm.h",
     "inline constexpr std::int32_t kR8VerifyTokens = 8;", 8,
     "gb202:170sm, gb203:70sm, ad102:128sm", "RTX 5090, RTX 5070 Ti, RTX 4090",
     "Q8 at T = 2..8 took 0.96-1.15 of its T = 1 time on R8 against 0.99-1.24 on the register MMA "
     "(5090); past four tokens the other formats' R8 lost (Q4_K 1.48-1.89, IQ3_XXS 1.49-1.80, "
     "IQ4_XS 1.19-1.53); the dp4a work equals the weight read at 21 tokens (A100, H100), 37 "
     "(RTX 3060), 40 (GV100), 86 (RTX 4090); RTX 4090 Qwen3.5-9B Q8_0 dflash:7 229 -> 273 tok/s",
     ""},
    {"r8_verify_tunable: lower edge", Basis::Structural, "generic_select.cpp",
     "return tokens >= 2 && tokens <= kR8VerifyTokens && allows_r8(w, policy) &&", 2, kUnknown,
     kUnknown,
     "the verify band of R8 starts at two tokens (T = 1 is the decode rule, kR8DecodeTokens: the "
     "complement of that rule, but a bare 2 in the code, not defined from it, so the number is not "
     "tied to it by the compiler); the record names no board or margin for the lower edge. The "
     "same edge is written again at "
     "r8_prequantized (route demotion, generic_resolve.cpp) and in generic_workspace.cpp "
     "(route_workspace_bytes, the interval plan): a change of the edge is a change in four places",
     ""},
    {"kMmqMinTokens", Basis::Unknown, "generic_gemm.h",
     "inline constexpr std::int32_t kMmqMinTokens = 5;", 5,
     "unknown (decisions.md 58 names the families: Ampere, Ada, Blackwell and Turing)",
     "unknown (the same families)",
     "generic_gemm.h's comment: five tokens, past the four R8 verifies for every format but Q8 "
     "(kR8MaxTokens); a bare 5 in the code, not defined from kR8MaxTokens or held to it by a "
     "static_assert, and no timing names the edge itself; docs/maintainer/decisions.md 58 is about "
     "the vector kernel through T = 4 and the register MMA from 5 on the native-MMA boards, and "
     "Turing's codebook formats at T = 5..8 on the matrix kernel",
     ""},
    {"kMmqMaxTokens", Basis::Measured, "generic_ggml_internal.h",
     "inline constexpr std::int32_t kMmqMaxTokens = 16;", 16, "gb202:170sm, ad103:80sm, gb203:84sm",
     "RTX 5090, RTX 4080 SUPER, RTX 5080",
     "the comment is timings, and says no fixed rule holds across chips and formats (the dense "
     "tuner times it): against the heuristic's tile, serve C4: 5090 ISTA IQ3_S +10 % at 16 "
     "tokens; Muse Q4_K_M -7..14 %, 4080 SUPER ISTA -12 % at 8; 5080 IQ3_S even; below the band it "
     "loses to R8 (5090 C1 -11 %); the band ends where the W4A8 tiles' 17 tokens begin",
     ""},
    {"kGgmlMmqMaxTokens", Basis::Structural, "generic_gemm.h",
     "inline constexpr std::int32_t kGgmlMmqMaxTokens = 16;", 16, "see kMmqMaxTokens",
     "see kMmqMaxTokens",
     "held equal to kMmqMaxTokens by a static_assert in generic_ggml.cpp; its timings are "
     "kMmqMaxTokens' (the same 5..16-token band)",
     ""},
    {"kGgmlVecMaxTokens", Basis::Structural, "generic_gemm.h",
     "inline constexpr std::int32_t kGgmlVecMaxTokens = 8;", 8, kUnknown, kUnknown,
     "ggml_mmq::kVecMaxColumns, the kernel's own width; no margin recorded", ""},
    {"kMmqVerifyMinTokens", Basis::Measured, "generic_select.cpp",
     "constexpr std::int32_t kMmqVerifyMinTokens = 9;", 9,
     "ad104:58sm (L4), gb203:70sm, ad102:128sm, tu102:68sm",
     "L4, RTX 5070 Ti, RTX 4090, RTX 2080 Ti",
     "ISTA IQ3_S 27B C4 MTP3 (16-token verify): L4 79.4 -> 89.2 tok/s (+12.3 %), RTX 5070 Ti "
     "242.7 -> 270.9 (+11.6 %); the K-quants lost on the 4090 (UD-Q4_K_XL 27B DFlash2 C1 115.7 -> "
     "111.7, C4 271.5 -> 263.1) and stay on the register MMA; where the BF16 MMA is emulated "
     "(Turing, RTX 2080 Ti) every matrix-kernel format takes it from 9 tokens, the codebook "
     "formats below that too (mmq_verify_rule)",
     ""},
    {"mmq_verify_rule: single-sequence rounds", Basis::Measured, "generic_select.cpp",
     "tokens >= kMmqVerifyMinTokens && lanes != 1 &&", 1, "gb203:70sm", "RTX 5070 Ti",
     "the measured band is 9..16 tokens of several sequences (a C4 verify): one sequence keeps its "
     "register MMA at every width (lanes == 1; a caller that names no lane count reads "
     "ops::round_lanes()): a C1 MTP round of 5..8 tokens and a C1 copy round of 16 lost on mmq "
     "(RTX 5070 Ti C1 117.0 -> 115.2 tok/s, all of it in the copy rounds), while the C4 verify "
     "gains (233.5 -> 249.3)",
     ""},
    // Where `lanes` comes from: the four-argument overload reads the engine's round width at the
    // call; route_authorities.cpp's two Ask builders do the same, so a constant there would turn
    // the lane rule on or off for every call.
    {"mmq_verify_rule: lanes at the call", Basis::Structural, "generic_select.cpp",
     "return mmq_verify_rule(w, policy, rows, tokens, ops::round_lanes());", 0, kUnknown, kUnknown,
     "the lane count of the single-sequence rule above is the engine's own round width "
     "(ops::round_lanes()), not a constant: a contract of where the value comes from, with no "
     "timing of its own (the timing is the single-sequence rule's, RTX 5070 Ti)",
     ""},
    {"Ask: lanes at the builders", Basis::Structural, "route_authorities.cpp",
     "tokens, ops::round_lanes()};", 0, kUnknown, kUnknown,
     "resolve_ask and ask_below_profiles read the lane count from ops::round_lanes() (the rule "
     "above reads it from the Ask): a contract of where the value comes from, with no timing of "
     "its own (the timing is mmq_verify_rule's single-sequence rule, RTX 5070 Ti)",
     "", 2},
    {"mmq_verify_rule: lanes handed over", Basis::Structural, "route_authorities.cpp",
     "mmq_verify_rule(weight, problem.policy, problem.n, ask.tokens, ask.lanes)", 0, kUnknown,
     kUnknown,
     "the two authority sites that call the rule pass the Ask's lane count on, not a constant (a "
     "constant would turn the single-sequence rule on or off for every call): a contract of where "
     "the value comes from, with no timing of its own",
     "", 2},
    // The prefill band's edge: ONE definition (ops/route_pins.h), the widest speculative verify
    // plus one; the others below are defined from it, because their comments call the same width.
    {"kDecodePinMaxTokens", Basis::Structural, "../route_pins.h",
     "inline constexpr std::int32_t kDecodePinMaxTokens = 16;", 16, kUnknown, kUnknown,
     "the widest speculative verify (route_pins.h's words): the decode pin's band, 1..16 tokens "
     "(decode and speculative verify); structural, no margin recorded. kDenseTuneMaxTokens, "
     "kPdlPrefetchMaxTokens and kMmqMaxTokens are 16 too and are NOT defined from it: "
     "kMmqMaxTokens' comment names its own timings, the comments of the other two name none "
     "(generic_gemm.h: 'decode and speculative verify')",
     ""},
    {"kPrefillPinMinTokens", Basis::Structural, "../route_pins.h",
     "inline constexpr std::int32_t kPrefillPinMinTokens = kDecodePinMaxTokens + 1;", 17, kUnknown,
     kUnknown,
     "one past the widest speculative verify, so the decode and prefill pin bands are adjacent; "
     "the prefill route profile's first width, where a call takes one route per problem. "
     "kPrefillBandMinTokens, kQ8PrefillA8Tokens, kVoltaCutlassMinTokens and route_table.cpp's "
     "kPrefillMinTokens are defined from it: their comments call the same width",
     ""},
    {"kPrefillBandMinTokens", Basis::Structural, "generic_gemm.h",
     "inline constexpr std::int32_t kPrefillBandMinTokens = ops::kPrefillPinMinTokens;", 17,
     kUnknown, kUnknown,
     "the generic path's name for kPrefillPinMinTokens: the first width of the prefill band, one "
     "past the widest speculative verify (16 tokens); a structural width, no margin recorded. The "
     "selection, the workspace plan and the tuner read "
     "it; the tuner's grid and the selector facts' grid start at it, held by static_asserts",
     ""},
    {"kPrefillMinTokens", Basis::Structural, "../route_table.cpp",
     "constexpr std::int32_t kPrefillMinTokens = kPrefillPinMinTokens;", 17, kUnknown, kUnknown,
     "the first width a route table's prefill band may start at: 'the first call width the "
     "prefill route profile makes token-invariant', the same width as kPrefillPinMinTokens, from "
     "which it is defined",
     ""},
    {"kQ8PrefillA8Tokens", Basis::Structural, "generic_gemm.h",
     "inline constexpr std::int32_t kQ8PrefillA8Tokens = kPrefillBandMinTokens;", 17,
     "ad102:128sm, gb202:170sm", "RTX 4090, RTX 5090",
     "defined from kPrefillBandMinTokens (its comments say 'a call of 17 tokens or more inside a "
     "prefill'); the chunk-invariance contract's width, not a margin (the boards are those of the "
     "gains and the cost the comment quotes): Qwen3.6-35B prefill RTX 4090 +7.4/+7.1 %, RTX 5090 "
     "+4 %; a DFlash2 round of 17+ columns on A8 cost Qwen3.8-27B --spec auto C=1 2-3 % on the "
     "RTX 5090, so only the prefill takes it",
     ""},
    {"kDenseTuneMaxTokens", Basis::Unknown, "generic_gemm.h",
     "inline constexpr std::int32_t kDenseTuneMaxTokens = 16;", 16, kUnknown, kUnknown,
     "the decode and verify widths the dense tuner times; no margin recorded; equal to "
     "kDecodePinMaxTokens today, not defined from it",
     ""},
    {"kR8VerifyQ40MinValues", Basis::PlacedBetweenMeasured, "generic_select.cpp",
     "inline constexpr std::int64_t kR8VerifyQ40MinValues = 24ll << 20;", 25165824,
     "ad102:128sm, ga102:82sm", "RTX 4090, RTX 3090",
     "Q4_0 R8 at T = 2..4 pays on a large matrix (Gemma 4 31B QAT, every dense matrix >= 44M "
     "values, C = 4: RTX 4090 +15.7 %, RTX 3090 +26 %) and loses on a small one (Gemma 4 26B QAT, "
     "at most 12M: -0.7 %, -2.7 %, its MTP verify -1.1 %); nothing between 12M and 44M measured",
     ""},
    {"r8_verify_widest: four tokens", Basis::Measured, "generic_routes.cpp",
     "return qtype == QType::Q8_G32_FP16 ? kR8VerifyTokens : 4;", 4, "gb202:170sm, gb203:84sm",
     "RTX 5090, RTX 5080",
     "the verify form takes four tokens for every R8 format but Q8 (kR8VerifyTokens for Q8): past "
     "four tokens the other formats' R8 lost (Q4_K 1.48-1.89, IQ3_XXS 1.49-1.80, IQ4_XS 1.19-1.53 "
     "of the register MMA's 1.07-1.62, the bench on the RTX 5090); at T = 8 the median is 1.35 "
     "(RTX 5080), so no format other than Q8 takes it; a bare 4 in the code, not defined from "
     "kR8MaxTokens",
     ""},
    {"kRmmaQ6kFourWarpBytes", Basis::PlacedBetweenMeasured, "generic_select.cpp",
     "inline constexpr double kRmmaQ6kFourWarpBytes = 4.0;", 4.0,
     "ad102:128sm, ad104:60sm, ad106:34sm, gb203:70sm",
     "RTX 4090, RTX 4070 Ti, RTX 4060 Ti, RTX 5070 Ti",
     "w1x4 against w1x8: RTX 4090 (3.1 bytes a SM cycle) 0.82-0.98, 4070 Ti (3.2) 0.80-0.99, 4060 "
     "Ti (3.4) 0.87-1.00; 5070 Ti (5.2) 0.98-1.12; the 5080 and 5090 (4.4) not measured",
     ""},
    {"kRmmaLatencySwiGluBytes", Basis::PlacedBetweenMeasured, "generic_gemm_internal.h",
     "inline constexpr double kRmmaLatencySwiGluBytes = 4.5;", 4.5,
     "ad102:128sm, ad104:60sm, gb203:70sm, ga106:28sm, tu102:68sm",
     "RTX 4090 (3.1 bytes a SM cycle), RTX 4070 Ti (3.2), RTX 5070 Ti (5.2), RTX 3060 (7.2), "
     "RTX 2080 Ti (5.9)",
     "SwiGLU, T = 5..8, Qwen3.5-9B shapes: the 4090 and 4070 Ti (plain kernel at 86-91 % of DRAM) "
     "lost 8-34 % to the latency variant, the 5070 Ti and 3060 (64-74 %) gained 2-15 % and "
     "11-25 %; the threshold lies between the boards measured on either side "
     "(generic_gemm_launch.cuh). The edge is also the lower bound of ring_swiglu_four_warps "
     "(Turing, Q4_K and Q5_K): RTX 2080 Ti (5.9) gate/up 12288x4096 Q4_K 142.4 -> 122.3 us at "
     "T = 8",
     ""},
    {"kRmmaLatencyOneSetBytes", Basis::PlacedBetweenMeasured, "generic_gemm_internal.h",
     "inline constexpr double kRmmaLatencyOneSetBytes = 5.5;", 5.5,
     "ga106:28sm, tu102:68sm, gb203:70sm",
     "RTX 3060 (7.2 bytes a SM cycle), RTX 2080 Ti (5.9), RTX 5070 Ti (5.2)",
     "one row set (store, add), T = 5..8: the 3060 gained 8-21 %, the 2080 Ti 0-4 %, the 5070 "
     "Ti between +14 % (Q3_K) and -3 % (Q4_K add 4096x12288 w1x4, 40.7 -> 41.7 us), so the "
     "threshold lies above that board (generic_gemm_launch.cuh)",
     ""},
    {"rmma_latency_board: Volta floor", Basis::Measured, "generic_gemm_launch.cuh",
     "if (d.major * 10 + d.minor < 75) return false;", 75, "gv100:80sm", "V100 (GV100)",
     "Volta's register MMA is emulated on FMA (no m16n8k16): its kernel is compute-bound, and the "
     "latency variant only cost there (GV100, T 5..8: 0-5 % slower, never picked)",
     ""},
    {"rmma_latency_board: Turing SwiGLU needs the ring", Basis::Measured, "generic_gemm_launch.cuh",
     "return !(swiglu && d.major < 8 && !kRmmaAheadForm<Decoder, Schedule, E>);", 8, "tu102:68sm",
     "RTX 2080 Ti",
     "without cp.async (sm_75) SwiGLU's variant is the activation ring's form, or none: on the RTX "
     "2080 Ti the variant with neither left Q4_K SwiGLU 7-8 % slower, with the ring w1x4 took "
     "168.9 -> 122.3 us at T = 8",
     ""},
    {"kRmmaWideBytes", Basis::PlacedBetweenMeasured, "generic_variant_rules.h",
     "inline constexpr double kRmmaWideBytes = 5.5;", 5.5, "ad102:128sm, gb203:70sm, ga106:28sm",
     "RTX 4090, RTX 5070 Ti, RTX 3060",
     "A/B, T = 1..8: RTX 4090 (3.1 bytes a SM cycle) add 71 -> 59 us, 5070 Ti (5.2) every problem "
     "1-8 % faster; on the RTX 3060 (7.2) the extra shuffles cost 2-10 %",
     ""},
    // --- Volta's routes (generic_select.cpp, generic_gemm_internal.h) ---
    {"kVoltaCutlassMinTokens", Basis::Structural, "generic_select.cpp",
     "constexpr std::int32_t kVoltaCutlassMinTokens = kPrefillBandMinTokens;", 17, "gv100:80sm",
     "V100",
     "defined from kPrefillBandMinTokens (the comment at the rule: a tiled-route call of the "
     "prefill band, 17 tokens on). V100 ninferno_linear_bench, T = 32..240, the tile's time "
     "against the decoded copy's: Q4_K 24576 x 4096 1.36..2.29 ms against 1.20..1.40, 4096 x "
     "12288 1.07..1.31 against 0.82..0.88, 4096 x 4096 0.36..0.44 against 0.27..0.28, Q5_K 4096 x "
     "12288 1.13..1.37 against 0.90..0.94; the one shape whose grid is a whole wave, 10240 x "
     "4096, keeps the tile ahead below 129 tokens (0.37..0.47 ms against 0.50)",
     ""},
    {"kVoltaTileMaxTokens", Basis::Unknown, "generic_select.cpp",
     "constexpr std::int32_t kVoltaTileMaxTokens = 256;", 256, kUnknown, kUnknown,
     "none recorded: no timing places 256 (the rule, for boards under the copy's bandwidth bound, "
     "is not measured on them; on the V100 the copy's bandwidth bound already passes, "
     "so the edge decides only on boards nobody measured); the comment's reason, that two token "
     "blocks decode the weight twice, holds from 129 tokens with 128-token blocks and so does not "
     "place the edge at 256 either",
     ""},
    {"kVoltaCopyMinDramGbps", Basis::ScaledFromOneBoard, "generic_select.cpp",
     "constexpr double kVoltaCopyMinDramGbps = 870.0 / 1.13;", 870.0 / 1.13, "gv100:80sm",
     "V100 (870 GB/s); the Titan V (653) is under the bound, not measured",
     "the copy's narrowest margin is 1.13 (24576 x 4096 below 129 tokens) at 870 GB/s; the bound "
     "scales that one board's margin by the DRAM rate (the copy's time is its bytes over the "
     "rate, the tile's its waves on the SMs): one board measured, no other, and the model is the "
     "comment's, not a timing",
     ""},
    // The two comparisons of volta_copy_pays, the use of kVoltaTileMaxTokens and of the bound
    // above.
    {"volta_copy_pays: the two comparisons", Basis::ScaledFromOneBoard, "generic_select.cpp",
     "return tokens >= kVoltaTileMaxTokens || fast_dram.get([](int) { try { return "
     "device_class().dram_gbps() >= kVoltaCopyMinDramGbps; }",
     0, "gv100:80sm", "V100 (870 GB/s)",
     "a tiled format's call takes the decoded copy from kVoltaTileMaxTokens tokens, and below that "
     "only on a board whose DRAM rate is at least kVoltaCopyMinDramGbps; the comparisons are held "
     "here, the two constants have their own descriptors",
     ""},
    {"kVoltaHmmaMinTokens", Basis::Structural, "generic_gemm_internal.h",
     "inline constexpr std::int32_t kVoltaHmmaMinTokens = 1;", 1, "gv100:80sm", "V100",
     "a byte-consistency contract, not a margin: a one-token call in a lane-local call ran the "
     "FP32 GEMV and the same token in a wider call the HMMA kernel (test_moe)",
     ""},
    {"kVoltaHmmaMaxTokens", Basis::Measured, "generic_gemm_internal.h",
     "inline constexpr std::int32_t kVoltaHmmaMaxTokens = 16;", 16, "gv100:80sm", "V100",
     "the verify widths (Q4_K / Q5_K / Q6_K and Q8 G32); Quadro GV100 Qwen3.8 27B UD-Q4_K_XL: the "
     "register MMA at T = 5..16 about 0.5 ms a call against 0.08 for the vector kernel at T = 4; "
     "C8 aggregate 37.3 tok/s against C4 86.3",
     ""},
    // --- The vector kernel's verify widths ---
    {"kVecRuleMaxTokens", Basis::PlacedBetweenMeasured, "generic_select.cpp",
     "constexpr std::int32_t kVecRuleMaxTokens = 4;", 4,
     "tu102:68sm, ga100:108sm, ga102:82sm, gb202:170sm, ad102:128sm, gv100:80sm",
     "RTX 2080 Ti, A100, RTX 3090, RTX 5090, RTX 4090, V100",
     "in the dense tuner's timings of eight cards (four named in the comment: A100 x1.5..1.6, RTX "
     "3090 x1.4..1.5, RTX 5090 x1.1, RTX 4090 x1.03) the vector kernel was within 4 % of the best "
     "route at T = 2..4 and behind it at T = 8 on all of them; RTX 2080 Ti: lost decode by 10 % "
     "(9B Q4_K_M C1 88 -> 79 tok/s), won C4 by 10 %; Volta takes every width it reads instead "
     "(V100, see kVoltaHmmaMaxTokens); docs/maintainer/decisions.md 58: the band rule (vector "
     "kernel at T <= 4, register MMA from 5) is the one measured on Ampere, Ada and Blackwell, "
     "held against a price-based pick (RTX 4070 Ti, Q6_K 3840x15360, T = 5). The comment's "
     "timings are at T = 2..4 and at T = 8; none is at 5..7, so the line at 4 is placed between "
     "them (decisions.md 58 has one T = 5 call, the 4070 Ti's Q6_K down projection, on which the "
     "vector kernel was ahead of the rule's register MMA)",
     ""},
    // The rule's lower edge, from two tokens: T = 1 keeps R8 and the dense tuner.
    {"vec_rule lower edge", Basis::Measured, "generic_select.cpp",
     "return tokens >= 2 && tokens <= vec_rule_max_tokens() && vec_call(w, policy, tokens);", 2,
     "tu102:68sm", "RTX 2080 Ti",
     "decode (T = 1) keeps R8 and the dense tuner: on the RTX 2080 Ti the vector kernel lost "
     "decode by 10 % (9B Q4_K_M C1 88 -> 79 tok/s) and won C4 by 10 %",
     ""},
    // The widest call the vector kernel reads: eight columns, sixteen tokens (two passes) where the
    // BF16 MMA is emulated.
    {"vec_max_tokens: two passes without tensor MMA", Basis::Unknown, "generic_select.cpp",
     "return f.mma_tensor && !f.mma_slow ? ggml_mmq::kVecMaxColumns : 2 * "
     "ggml_mmq::kVecMaxColumns;",
     2, kUnknown, kUnknown,
     "the basis is Unknown because the one timing (V100) is at a draft window of about ten tokens: "
     "without tensor mma.sync (Volta) a call of 9..16 tokens runs the vector kernel eight columns "
     "at a time: V100, Gemma 4 31B Q4_K_XL MTP, draft window 10: the emulated register-MMA tiles "
     "1.0..2.6 ms a call, 67 % of the run; the timing is at one draft window (about ten tokens), "
     "so the sixteen-token edge itself (two passes of kVecMaxColumns) is not placed by it",
     ""},
    // --- The widths where the speed profile's heuristic choices change (generic_gemm.h) ---
    {"kRmmaMidMinTokens", Basis::Fitted, "generic_gemm.h",
     "inline constexpr std::int32_t kRmmaMidMinTokens = 33;", 33, kSpeedFitBoards, kSpeedFitCards,
     "one rule for every board, fitted on six boards and scored on the seventh: 1.116 of the best "
     "(geomean) against 1.219 for the bands it replaced; a rule per board gained 1.1 % at most",
     ""},
    {"kStreamMaxTokens", Basis::Unknown, "generic_gemm.h",
     "inline constexpr std::int32_t kStreamMaxTokens = 512;", 512, kUnknown, kUnknown,
     "the widest call of the register MMA's stream regime where the BF16 MMA is emulated (Turing, "
     "Volta; generic_gemm.h: 'through 512 where the BF16 MMA is emulated'); no timing or board is "
     "recorded for the 512 itself",
     ""},
    {"kTiledSpeedMinTokens", Basis::Unknown, "generic_gemm.h",
     "inline constexpr std::int32_t kTiledSpeedMinTokens = 33;", 33,
     "unknown (the evidence below names no board)", "unknown",
     "from 33 tokens, the same width as kRmmaMidMinTokens; the only timing in the record is at "
     "T 64..128 (Q4_K n = 1024: the tiled kernel 5..7x the register MMA's, on every board, below "
     "the kSpeedFill grid), nothing at 33",
     ""},
    {"kStagedSpeedMinTokens", Basis::Unknown, "generic_gemm.h",
     "inline constexpr std::int32_t kStagedSpeedMinTokens = 256;", 256, kUnknown, kUnknown,
     "the staged MMA's fallback from 256 tokens, the last resort of the speed route "
     "(generic_gemm.h); generic_select.cpp scores the old gate (from 256 tokens, the first tile) "
     "at 1.33..1.64 of the best on seven boards, against 1.116 for the rule that replaced it "
     "(staged_speed_route); no timing places the 256",
     ""},
    {"kSpeedFill", Basis::Fitted, "generic_gemm.h", "inline constexpr double kSpeedFill = 1.5;",
     1.5, kSpeedFitBoards, kSpeedFitCards,
     "1.0..2.0 cost under 0.5 % (the header's comment); the same board-held-out fit as "
     "kRmmaMidMinTokens",
     ""},
    {"kW4a8WideMinTokens", Basis::Measured, "generic_gemm.h",
     "inline constexpr std::int32_t kW4a8WideMinTokens = 128;", 128,
     "gb203:84sm, ad103:80sm, gb202:170sm, gb206:36sm",
     "RTX 5080, RTX 4080 SUPER, RTX 5090, RTX 5060 Ti",
     "from 128 tokens (RTX 5080 in the product, Qwen3.8-27B UD-IQ4_XS: IQ4_XS 0.82-0.97 per call "
     "against 64x64, p512 0.93x -> 0.98x llama.cpp; IQ2_S, IQ2_XS and Q6_K 0.53-0.92); below it "
     "(the 128x64 tile against 64x64): 1.10-1.56 on IQ2_*/IQ3_*/IQ4_XS (4080 SUPER), Q4_K "
     "1.05-1.09 .. Q6_K 1.17-1.20 (5090), 1.08-1.29 (5060 Ti); in the product 4080 SUPER "
     "Qwen3.8-27B IQ3 412.9 -> 472.4 tok/s, 5090 Q4_K_XL 738 -> 757",
     ""},
    {"kPdlPrefetchMaxTokens", Basis::Unknown, "generic_gemm.h",
     "inline constexpr std::int32_t kPdlPrefetchMaxTokens = 16;", 16, kUnknown, kUnknown,
     "none recorded (decode and speculative verify widths)", ""},
    // --- The prefill band, the dense tuner and the heuristics' width bands ---
    {"kPrefillBandTokens", Basis::Unknown, "generic_tune.cpp",
     "inline constexpr std::array<std::int32_t, 9> kPrefillBandTokens = {17, 32, 64, 127, 128, "
     "256, 512, 1024, 2048};",
     0, "unknown", "unknown",
     "the widths whose band takes a tuned route; an untimed width takes the next band's: 17 is "
     "a 16-token draft's verify (kPrefillBandMinTokens, held by a static_assert), 32, 64, 127 and "
     "128 straddle the heuristics' switch widths (the register MMA's prefill column at 32 and 64, "
     "W4A8's 128x64 from 65 and its tiles from 128); 256 .. 2048 no record; no margin recorded",
     ""},
    {"kMargin", Basis::Unknown, "generic_tune.cpp", "constexpr double kMargin = 0.97;", 0.97,
     kUnknown, kUnknown,
     "a tuned schedule replaces the heuristic's only where at most 97 % of its time, so noise "
     "keeps the heuristic; no measurement behind the 3 % is recorded",
     ""},
    {"kBytes", Basis::Unknown, "generic_tune.cpp", "constexpr double kBytes = 64.0 * kMiB;",
     67108864, kUnknown, kUnknown,
     "the widest prefill band whose buffers stay within 64 MiB each is tuned (an output head of "
     "152K rows: T <= 128); wider bands keep the heuristic's route; none recorded",
     ""},
    {"tune fits: FLOP bound", Basis::Unknown, "generic_tune.cpp",
     "2.0 * model_call.rows * k * t <= 2.5e11", 2.5e11, kUnknown, kUnknown,
     "the widest prefill band whose call is within 250 GFLOP is tuned; none recorded", ""},
    {"kRmmaWideSpillSlack", Basis::PlacedBetweenMeasured, "generic_variant_rules.h",
     "inline constexpr std::size_t kRmmaWideSpillSlack = 16;", 16, "ad102:128sm", "RTX 4090",
     "sm_89's one-row-set tiles keep 8 bytes of loop-invariant state in local memory (admitted); "
     "SwiGLU's 128-register tiles spill 160 bytes in the loop (RTX 4090 w1x16 at T = 6: 100.3 -> "
     "103.9 us); nothing between 8 and 160 bytes measured, so the 16 itself has no timing",
     ""},
    {"kVecMaxColumns", Basis::Structural, "ggml_mmq/ggml_mmq.h",
     "inline constexpr int kVecMaxColumns = 8;", 8, kUnknown, kUnknown,
     "ggml's vector kernel's width (after ninfer-all's ggml_vec.cu); feeds vec_rule_max_tokens "
     "without tensor mma.sync; no margin recorded",
     ""},
    {"heuristic_rmma: prefill column", Basis::PlacedBetweenMeasured, "generic_select.cpp",
     "const bool narrow_band = tokens <= 32 || (tokens <= 64 && ctas < 2 * sm_count());", 0,
     "ad103:80sm, gb202:170sm, gb206:36sm", "RTX 4080 SUPER, RTX 5090, RTX 5060 Ti",
     "time of m1.n8 over the best: T = 17 and 32 1.48-3.21 on every problem (4080 SUPER); T = 64 "
     "1.20-1.58 below 160 CTAs, 1.00 at 272, so the two-CTAs-per-SM line (160 on the 80 SMs) lies "
     "between measured grids; 5090 Q6_K output head m1.n4 best at T = 17 (1.22) and 32 (1.54); "
     "5060 Ti 1.60-2.14 at T = 17 and 32; in the product 4080 SUPER Qwen3.8-27B IQ3 246.6 -> "
     "252.9 tok/s",
     ""},
    {"heuristic_rmma: narrow through 16", Basis::Measured, "generic_select.cpp",
     "if (tokens <= 16 || (split != 8 && tokens < kRmmaMidMinTokens)) return narrow();", 16,
     "gb202:170sm", "RTX 5090",
     "the comment's own words: what ninferno_route_tune measured across the recorded problems "
     "(Qwen3.5 0.8B in four formats on an RTX 5090): at T <= 16 one 16-row M tile per CTA with K "
     "split across the warps, and from 17 tokens outside the prefill profile the speed rule; the "
     "second clause's 33 is kRmmaMidMinTokens (Fitted); no margin recorded; the 16 is also the "
     "widest speculative verify, not a separate constant",
     ""},
    {"heuristic_rmma: K split edges", Basis::Measured, "generic_select.cpp",
     "const int split = k >= 1024 ? 8 : k >= 512 ? 4 : 1;", 0, "gb202:170sm", "RTX 5090",
     "K split across as many warps as keep at least 128 K values each (1024 / 8, 512 / 4): what "
     "ninferno_route_tune measured across the recorded problems (Qwen3.5 0.8B in four formats); "
     "no margin recorded. invariant_rmma repeats the K edges (below)",
     ""},
    {"invariant_rmma: K split edges", Basis::Unknown, "generic_select.cpp",
     "if (k >= 1024) return rmma_index(1, 8, 1, n_tiles); if (k >= 512) return rmma_index(1, 4, "
     "1, n_tiles);",
     0, "unknown (the same edges in heuristic_rmma were measured on gb202:170sm)",
     "unknown (the same edges in heuristic_rmma: RTX 5090)",
     "the decode split of K for every T (the invariant profile's comment: the order depends on "
     "WarpsK alone, so a token's bytes are the same at every width): the same edges as "
     "heuristic_rmma's K split, written again as literals; the invariant profile's own comment "
     "names no timing that places them, and nothing ties the two sets of edges to each other, so "
     "this descriptor has no record of its own",
     ""},
    {"heuristic_rmma: nine to sixteen tokens", Basis::Measured, "generic_select.cpp",
     "} else if (tokens <= 16) {", 16, "gb203:84sm", "RTX 5080",
     "9..16 tokens (verify of an 8- or 16-token draft), RTX 5080, Q8/Q4_0/IQ4_XS/Q4_K/Q5_K/Q6_K/"
     "IQ3_XXS at 27B/31B/35B shapes: four K-split warps with two tiles 1.00-1.12 of the best of "
     "seven schedules against 1.24-1.58 for w1x8 n1 (geomean per format; at T = 24 Q8 1.35 "
     "against 1.68); the margin is held at 9..16, and the comment quotes a point at T = 24 "
     "beyond the edge in the same comparison, so no timing places the 16 itself (it is the widest "
     "speculative verify)",
     ""},
    {"one_wave_split: quarter of the SMs", Basis::Unknown, "generic_select.cpp",
     "return 4 * ctas <= sms ? rmma_index(1, 8, 1, 1, 8) : rmma_index(1, 8, 1, 1, 2);", 0,
     "gb203:84sm, gb202:170sm", "RTX 5080, RTX 5090",
     "a call of at most a quarter as many CTAs as SMs (the 32-row GDN a/b projections: 2 CTAs, "
     "12.6 us for 139 KB) splits K over a cluster of eight CTAs: 7.5 -> 3.1 us (RTX 5080, "
     "Qwen3.5-4B/9B and Gemma 4 12B shapes); the CTA pair for the rest on the RTX 5090, T = "
     "1..4: Q8 2048x4096 23.7 -> 17.6 us; the quarter edge itself has no margin recorded",
     ""},
    {"heuristic_rmma: speed rule token tiles", Basis::Fitted, "generic_select.cpp",
     "const int nt_cap = emulated ? 4 : 8;", 0, kSpeedFitBoards, kSpeedFitCards,
     "the widest token tile (8 tokens each) of the speed rule: 4 tiles (32 tokens) where the BF16 "
     "MMA is emulated, 8 otherwise; inside the seven-board fit of kRmmaMidMinTokens, which the "
     "comment does not break out: no record of its own",
     ""},
    {"heuristic_rmma: sixteen warps from 72 K steps", Basis::Fitted, "generic_select.cpp",
     "return ctas <= sm_count() || 2 * steps >= 9 * 16 ? rmma_index(1, 16, 1, 1) : rmma_index(1, "
     "8, 1, 1);",
     0, kSpeedFitBoards, kSpeedFitCards,
     "with one token tile, sixteen K warps where the call has at most one CTA per SM or a long K "
     "(from 72 K steps of 128 K values for the heavy formats, 32 otherwise), else eight; inside "
     "the seven-board fit of kRmmaMidMinTokens: no record of its own",
     ""},
    // The rest of the speed rule's ladder (heuristic_rmma, from 17 tokens outside the prefill
    // profile): its comment fits the rule as a whole, so the waves and token-tile edges are
    // Fitted with kSpeedFill; the Q4_K tile's margin below is measured, its edges are not.
    {"heuristic_rmma: Q4_K tall tile", Basis::Fitted, "generic_select.cpp",
     "if (!emulated && tokens >= kRmmaMidMinTokens && tall_tile_format(problem.qtype) && "
     "waves(128, 32) >= kSpeedFill) {",
     0, kSpeedFitBoards, kSpeedFitCards,
     "Q4_K: 128 rows by 32 tokens, half the activation loads per MMA of the 64x64 tile, 10..20 % "
     "faster on all seven boards from 97 tokens (the 64x64 tile spills where the 128x32 tile "
     "spills less, ptxas sm_86..sm_120); the rule applies from kRmmaMidMinTokens (33) where the "
     "grid fills kSpeedFill waves, so 33..96 tokens carry no timing of their own; both edges are "
     "Fitted constants",
     ""},
    {"heuristic_rmma: token tile ladder", Basis::Fitted, "generic_select.cpp",
     "if (tokens > kStreamMaxTokens || (!emulated && tokens >= kRmmaMidMinTokens && waves(64, 8 * "
     "nt_cap) >= kSpeedFill)) { for (int nt = nt_cap; nt > 1; nt /= 2) { if (8 * nt < 2 * "
     "std::max(8, static_cast<int>(tokens)) && waves(64, 8 * nt) >= kSpeedFill) {",
     0, kSpeedFitBoards, kSpeedFitCards,
     "four row warps of 64 rows and the widest token tile of under twice the call's tokens that "
     "fills kSpeedFill waves of the SMs; where the BF16 MMA is emulated the token tiles stop at "
     "32 and the regime starts past the stream band; the comment fits the rule as a whole on the "
     "seven boards and does not break out these edges",
     ""},
    {"heuristic_rmma: stream regime tiles", Basis::Fitted, "generic_select.cpp",
     "if (pad >= 4 && waves(16, 32) >= kSpeedFill) { return heavy ? rmma_index(1, 4, 1, 4) : "
     "rmma_index(1, 8, 1, 4); } if (pad >= 2 && waves(16, 16) >= kSpeedFill) return rmma_index(1, "
     "4, 1, 2);",
     0, kSpeedFitBoards, kSpeedFitCards,
     "one 16-row M tile per CTA with K over the warps, as many 8-token tiles (4, 2, 1) as the "
     "call's tokens use while the grid fills kSpeedFill waves of the SMs; the comment fits the "
     "rule as a whole on the seven boards and does not break out these edges",
     ""},
    {"heuristic_rmma: K values per step", Basis::Structural, "generic_select.cpp",
     "const std::int64_t step = heavy ? 128 : 32;", 0, kUnknown, kUnknown,
     "the K values of one step of the stream regime's decoder: 128 for the heavy formats (the "
     "32-value decoders, four runs a step), 32 otherwise (rmma_k_step); a count of the kernel, "
     "no margin",
     ""},
    // The one-wave edge of the K split, at both of its sites; the decode block's other edges
    // (tokens <= 8, ctas <= 2 * sms, ctas > 2 * sms) stay in the doc's list of omissions.
    {"heuristic_rmma: one wave at decode widths", Basis::PlacedBetweenMeasured,
     "generic_select.cpp",
     "const std::int64_t sms = sm_count(); if (ctas <= sms) return one_wave_split(ctas, sms); if "
     "(ctas > 2 * sms && light_ggml_step(problem.qtype) &&",
     0, "gb202:170sm, ad102:128sm", "RTX 5090, RTX 4090",
     "T <= 8: a call of at most one CTA per SM takes one_wave_split (its own descriptor). RTX 5090 "
     "(170 SMs), T = 1..4: the CTA pair gains at 128..168 CTAs (2048..2688 rows: Q8 2048x4096 "
     "23.7 -> 17.6 us, 2688x4096 25.5 -> 17.6) and 'from 176 CTAs (2816 rows) the pair only "
     "ties'; RTX 4090 (128 SMs, sixteen K warps in one CTA): Q8 1280x4096 26.3 -> 16.2 us, "
     "2048x4096 26.6 -> 20.5, and 'from 160 CTAs it loses' (Q4_K 2816x4096 17.4 -> 21.5): the "
     "one-wave line (170 and 128 CTAs) sits at or above the last measured win and below the first "
     "measured loss on both boards",
     ""},
    {"heuristic_rmma: one wave at verify widths", Basis::Unknown, "generic_select.cpp",
     "const std::int64_t sms = sm_count(); if (ctas <= sms) return one_wave_split(ctas, sms); "
     "return rmma_index(1, 4, 1, 2);",
     0,
     "unknown (nine cards of the dense tuner, sm_70 to sm_120, not named; the RTX 5090 for the "
     "CTA pair: gb202:170sm)",
     "unknown (nine cards; RTX 5090)",
     "the basis is Unknown because the record names no board for the one-wave line itself. T = "
     "9..16: a call of at most one CTA per SM takes the decode split of K: the dense tuner's "
     "timings of nine cards had the four-warp tiles lose such calls (n = 48..2048 rows) by "
     "x1.4..2.1 (median per card) to the sixteen-warp split; on the RTX 5090 the CTA pair or "
     "cluster of eight was 0..17 % faster still (n = 48 and 1024, T = 12 and 16: 8..20 us "
     "against 16..41)",
     ""},
    // --- rmma_k_split_fit: the K split of a one-tile store / add call of more CTAs than SMs ---
    {"rmma_k_split_fit: ring refills", Basis::Structural, "generic_select.cpp",
     "return (steps + warps_k - 1) / warps_k > 2;", 2,
     "ad104:60sm, ga106:28sm, gb203:70sm, tu102:68sm",
     "RTX 4070 Ti, RTX 3060, RTX 5070 Ti, RTX 2080 Ti",
     "held to one fact of the addressing, never a timing: the CTA's longest warp walks more K "
     "steps than its load ring holds (two, rmma_kernel's ring), so at least three; with two or "
     "fewer the ring never refills (Q4_K K = 4096 on sixteen warps streamed DRAM at 57..70 % of "
     "peak against 71..74 % for the best split); K = 2560 on eight warps keeps up (RTX 4070 Ti "
     "19.6 against 21.1 us, RTX 3060 32.4 against 32.9; the 5070 Ti and 2080 Ti ran four warps "
     "faster there, decisions.md 61)",
     ""},
    {"kSlotWarps", Basis::Structural, "generic_select.cpp",
     "constexpr std::int64_t kSlotWarps = 65536 / (32 * 128);", 16, kUnknown, kUnknown,
     "the warps a SM holds at 128 registers a thread (the register file over 32 lanes of 128 "
     "registers, kRmmaMinBlocks); a count, no margin",
     ""},
    {"rmma_k_split_fit: four warps", Basis::Structural, "generic_select.cpp",
     "if (info.warps_k <= 4 || refills(info.warps_k) || !refills(4) || ctas * 4 < kSlotWarps * "
     "sms) {",
     0, "gb203:70sm, gb202:170sm, ad104:60sm, ad106:34sm",
     "RTX 5070 Ti, RTX 5090, RTX 4070 Ti, RTX 4060 Ti",
     "eight warps that break the ring take four where four keep it and the grid fills their CTA "
     "slots (four CTAs a SM at four warps, kSlotWarps over four): below that grid the split trades "
     "resident warps for ring refills: K = 1024 (2048 rows, 128 CTAs) four warps against eight, "
     "RTX 5070 "
     "Ti (1.8 CTAs a SM) 20-37 % slower, RTX 5090 (0.75) 18-38 %, RTX 4070 Ti (2.1) within 3 %; "
     "the RTX 4060 Ti (3.8) ran four warps 2-16 % faster, which this leaves on eight; the "
     "last, partly filled round of resident CTAs is open (decisions.md 61)",
     ""},
    {"heuristic_w4a8: 17..32", Basis::Measured, "generic_select.cpp",
     "if (tokens <= 32) return k32x128;", 32, "gb203:84sm", "RTX 5080",
     "7 GGUF formats at 27B/31B/35B shapes, time over the best of seven routes: at T = 17 32x128 "
     "1.05-1.45 against 64x64 1.14-1.57, better for every format at 17, 20, 24, 32",
     ""},
    {"heuristic_w4a8: 33..64", Basis::Unknown, "generic_select.cpp",
     "if (tokens <= 64) return k64x64;", 64, kUnknown, kUnknown,
     "none recorded for the 64-token edge itself", ""},
    {"heuristic_w4a8: wide tile grid", Basis::PlacedBetweenMeasured, "generic_select.cpp",
     "if (2 * wide_ctas < sm_count()) return k64x64;", 0, "ad103:80sm, gb202:170sm, gb206:36sm",
     "RTX 4080 SUPER, RTX 5090, RTX 5060 Ti",
     "from 65 tokens a 128-token tile pays while its grid still gives half the SMs a CTA; the "
     "dense tuner's prefill bands (perplexity at context 2048 on Qwen3.8-27B shapes), 64x64 "
     "against 128x64 below that grid: 4080 SUPER at 1024 rows (16 CTAs at T = 128, 32 at 256) "
     "1.33-1.37 for Q4_K, Q5_K and IQ4_XS; 5090 at 5120 rows and T = 128 (80 CTAs) 1.15-1.36, at "
     "1024 rows and T = 128..512 (16..64 CTAs) 1.14-1.35 for Q4_K and Q5_K, 1.07 for Q6_K; 5060 "
     "Ti at 1024 rows and T = 128 1.19-1.50; above the line the 5090 takes 128x64 from 6144 rows "
     "(96+ CTAs on 170 SMs, T 65..127: Q4_K 1.05-1.09, Q5_K 1.07-1.17, Q6_K 1.17-1.20) while "
     "5120 rows (80 CTAs) stay on 64x64; the line (half the SMs) is not separately measured, it "
     "lies between those grids",
     ""},
    {"heuristic_w4a4: widest tile that fills the SMs", Basis::Unknown, "generic_select.cpp",
     "if (grid(128, 128) >= sms) return 3; if (grid(64, 128) >= sms) return 2; if (grid(32, 128) "
     ">= sms) return 1;",
     0, kUnknown, kUnknown,
     "outside the prefill band, the widest W4A4 tile that still gives every SM a CTA, else the "
     "narrow ones; the comment records no board or margin for the ladder (for the prefill band "
     "it records the tuner's winner on every recorded NVFP4 problem but one, within noise)",
     ""},
    {"invariant_rmma: token tiles", Basis::Unknown, "generic_select.cpp",
     "const int n_tiles = tokens <= 8 ? 1 : 4;", 8, kUnknown, kUnknown,
     "eight tokens per CTA through T = 8, where decode batches live; 32 above, so a prefill chunk "
     "reads each weight a quarter as often (a design choice; no margin recorded)",
     ""},
    {"invariant_rmma: narrow K tiles", Basis::Unknown, "generic_select.cpp",
     "return tokens <= 8 ? rmma_index(4, 1, 1, 1) : rmma_index(4, 1, 1, 8);", 8, kUnknown, kUnknown,
     "the same eight-token edge as invariant_rmma's token tiles, for K under 512: one 8-token "
     "tile through T = 8, eight tiles above (a design choice; no margin recorded)",
     ""},
    {"heuristic_w4a8: 65..127 formats", Basis::Measured, "generic_select.cpp",
     "if (tokens < kW4a8WideMinTokens) {", 0, "ad103:80sm, gb202:170sm, gb206:36sm",
     "RTX 4080 SUPER, RTX 5090, RTX 5060 Ti",
     "65..127 tokens: the formats that take a 128-token tile from 128 tokens take 128x64 already, "
     "and so do IQ2_XXS and Q3_K (their wider tiles were not measured in the product); 128x64 "
     "against 64x64 1.10-1.56 on IQ2_*, IQ3_*, IQ4_XS from 5120 rows (80+ CTAs) (4080 SUPER), "
     "Q3_K 1.09, IQ2_XXS 1.10-1.14, Q4_K 1.09 at 17408 rows; 5090 Q4_K 1.05-1.09, Q5_K 1.07-1.17, "
     "Q6_K 1.17-1.20 from 6144 rows; 5060 Ti IQ2_S, IQ2_XS, IQ3_S, IQ3_XXS 1.08-1.29 at every row "
     "count; Q4_0 and the rest keep 64x64",
     ""},
    {"rmma_wide_rule: capability floor", Basis::Structural, "generic_routes.cpp",
     "compute_capability >= 75 && occupancy &&", 75, kUnknown, kUnknown,
     "the wide-load Q6_K variant is taken from compute capability 7.5 (sm_75) up: 'not before "
     "sm_75 (Volta's register MMA is FMA-emulated, compute-bound)' (generic_variant_rules.h); no "
     "timing of its own (the bytes bound is kRmmaWideBytes)",
     ""},
    {"r8_prequantized: lower edge", Basis::Structural, "generic_resolve.cpp",
     "route.family == Family::R8Mma && call.tokens >= 2 && call.tokens <= kR8VerifyTokens &&", 2,
     kUnknown, kUnknown,
     "the verify band of R8 again (r8_verify_tunable's two tokens to kR8VerifyTokens): the "
     "pre-quantized verify form; T = 1 is the decode rule (kR8DecodeTokens; the 2 is a bare "
     "literal, the complement of that rule, not defined from it); no board or margin recorded",
     ""},
    {"kMinChunkWeightRows", Basis::Unknown, "generic_volta_cutlass.cu",
     "constexpr std::int32_t kMinChunkWeightRows = 256;", 256, "gv100:80sm", "V100 (80 SMs)",
     "the fewest weight rows a decode chunk holds: a fixed 4096 rows split the 27B gate_up "
     "(34816 x 5120) into nine GEMMs of 3.2 CTA waves each, one chunk of a third of it runs 10.8 "
     "waves; the bound itself has no margin recorded",
     ""},
    // --- Rules by a function's shape ---
    {"tiled_speed_pays: tile", Basis::Fitted, "generic_select.cpp",
     "speed_waves(n, tokens, 128, 128) >= kSpeedFill", 0,
     "unknown (the comment says every board without naming them; kSpeedFill's own fit is "
     "kSpeedFitBoards)",
     "unknown",
     "the edge is kSpeedFill (Fitted); the one timing in the record is below the fill: Q4_K n = "
     "1024 at T 64..128, the tiled kernel 5..7x the register MMA's on every board; no timing "
     "places the fill for this tile",
     ""},
    {"staged_speed_route: wave thresholds", Basis::Fitted, "generic_select.cpp",
     "for (const Pick p : {Pick{256, 128, 2, 2.0}, Pick{128, 64, 2, 1.0}, Pick{64, 64, 3, 0.5}}) {",
     0,
     "unknown (the comment says seven boards without naming them; the register-MMA rule beside "
     "it names the seven of kSpeedFitBoards)",
     "unknown (see kSpeedFitCards for the register-MMA rule's seven)",
     "BF16 timings on seven boards, with the register-MMA rule: 1.116 of the best "
     "(geomean) against 1.15..1.36 for the largest tile that filled the SMs and 1.33..1.64 for "
     "the old gate (from 256 tokens, the first tile): a 256x128 tile from two waves of the SMs, "
     "128x64 from one, 64x64 from half a wave; a wider tile than the call's grid leaves SMs idle "
     "(n = 1024 at T = 256: the 128x128 tile 2.5x the register MMA, n = 48: 6x)",
     ""},
    {"nvfp4_tiled_route: two token tiles and a wave", Basis::Measured, "generic_select.cpp",
     "if (tokens < 2 * s.block_m || ctas < sm_count()) return std::nullopt;", 0, "gb202:170sm",
     "RTX 5090",
     "for calls of two token tiles or more whose tiles give every SM at least one CTA; against the "
     "invariant register MMA (WarpsK 8) on an RTX 5090: -8..-40 % from there on the Muse Glimmer "
     "30B MLP and a synthetic [4096, 4096]; below one wave, or at one token tile, it loses (up to "
     "5x at T = 32)",
     ""},
    {"nvfp4_half_l2: half the L2", Basis::PlacedBetweenMeasured, "route_authorities.cpp",
     "if (2 * weight_bytes <= device_class().l2_bytes) {", 0, "ad102:128sm", "RTX 4090",
     "an NVFP4 weight that fits half the L2 takes the invariant profile's register MMA: RTX 4090, "
     "Gemma 4 26B shapes 1024..8192 x 2112..8192 at T 17..128, where the 128x128 prefill tile ran "
     "2-4x slower than the speed bands; a larger weight takes the prefill profile's route (31B "
     "16384x5376, 43008x5376, 5376x21504: 0.3-0.8x the speed bands from 32 tokens); measured on "
     "one board, no weight size between the two sets is recorded, so the half-L2 line itself has "
     "no measurement behind it",
     ""},
    // The other literals of route_authorities.cpp's rules. The bare 1 of the R8 table's gate and
    // the 9 / 16 below are NOT renamed to kR8DecodeTokens or a format constant: heuristic_r8 in
    // generic_select.cpp writes the same bare 1 without r8_call's tokens > 0 guard, so the spelling
    // `<= kR8DecodeTokens` would differ for a call of no tokens, and the table's width is the
    // table's own, not the decode rule's (the two are equal today).
    {"nvfp4_half_l2: weight bytes", Basis::Structural, "route_authorities.cpp",
     "static_cast<std::size_t>(problem.n) * static_cast<std::size_t>(problem.k) * 9 / 16;", 0,
     kUnknown, kUnknown,
     "the bytes of an NVFP4 weight of n x k values: nine sixteenths of a byte a value, four bits "
     "of the value plus one FP8 scale byte over sixteen (the format's layout), the left side of "
     "the half-L2 line; the layout is a fact of the format, no timing places it",
     ""},
    {"r8_schedule: the decode table's width", Basis::Structural, "route_authorities.cpp",
     "const auto hit = ask.tokens == 1 ? tuned_r8(problem) : std::nullopt;", 1, kUnknown, kUnknown,
     "the nearest measured board's decode table is consulted at one token only: its entries are "
     "T = 1 timings (r8_decode_table.h, written by tools/bench/r8_decode_table.py) and none exist "
     "for other widths; the gate follows from the table's layout, and no board places it (the "
     "table holds nine boards, none of them the RTX 5080 of the decode rule kR8DecodeTokens); "
     "the same bare 1 is written in heuristic_r8 (generic_select.cpp, the next descriptor)",
     ""},
    {"heuristic_r8: the decode table's width", Basis::Structural, "generic_select.cpp",
     "if (tokens == 1) { if (const auto tuned = tuned_r8(p)) return tuned->schedule; }", 1,
     kUnknown, kUnknown,
     "the same one-token gate as r8_schedule's, written again in the heuristic's own path; the "
     "table (r8_decode_table.h) has T = 1 entries only; no board places the gate",
     ""},
    {"rmma_narrow_fallback: narrow tiles", Basis::Unknown, "route_authorities.cpp",
     "for (const int n_tiles : {2, 1}) { const int index = rmma_index(4, 1, 1, n_tiles); const "
     "Route narrow{Family::RegisterMma, static_cast<std::uint8_t>(index)};",
     0, "gv100:80sm, tu102:68sm", "V100, RTX 2080 Ti",
     "the narrower register-MMA tiles the dense tuner picks where the BF16 MMA is emulated (m1.n2 "
     "then m1.n1 of four row warps), taken when the build does not hold the speed profile's tile: "
     "V100 and RTX 2080 Ti, BF16 N 1536 K 2048, T 17..2048; one shape on two boards, no margin "
     "recorded, and the order (n2 before n1) is not timed",
     ""},
    {"kR8CarryDistance", Basis::Unknown, "generic_select.cpp",
     "constexpr double kR8CarryDistance = 0.5;", 0.5, "ad102:128sm, ad103:80sm",
     "RTX 4090, RTX 4080 SUPER",
     "the one measured distance is the 4090's 80 entries on the 4080 SUPER (distance 1.2; 3 "
     "sweeps): won 41, lost 30 (to -22 %), +0.1 % in the mean; the comment names a 4090 D and a "
     "3090 Ti (0.17, 0.14) as boards that take their tables, as a consequence of the bound, with "
     "no timing; nothing places the 0.5",
     ""},
    {"kCarryDistance", Basis::Unknown, "../../core/measured_boards.h",
     "inline constexpr double kCarryDistance = 1.0;", 1.0, "ad102:128sm, ad103:80sm",
     "RTX 4090, RTX 4080 SUPER",
     "how far a measured choice carries (the default bound of nearest_known_measured; "
     "spread_smem's boards use it, and moe.cpp's): the "
     "comment lists variants (0.17, 0.14) and near neighbours (an A40 0.43 from the A10, an RTX "
     "5070 Ti 0.57 from the 5080) as examples and names one measured point beyond the bound, the "
     "4090's R8 decode entries on the 4080 SUPER (1.2: won 41, lost 30, to -22 %); nothing "
     "measured places the 1.0",
     ""},
    {"kSpreadMeasured", Basis::Measured, "generic_gates.cpp",
     "constexpr SpreadMeasured kSpreadMeasured[] = {{measured::kRtx5090, true}, "
     "{measured::kRtx5080, true}, {measured::kRtx5060Ti, false}, {measured::kRtx4080Super, "
     "false}};",
     0, "gb202:170sm, gb203:84sm, gb206:36sm, ad103:80sm",
     "RTX 5090, RTX 5080, RTX 5060 Ti, RTX 4080 SUPER",
     "the boards spread_smem was measured on and whether it was faster there: RTX 5090 (T = 1 "
     "under PDL, NVFP4 q/k/v 4608x6656 11.8 -> 11.25 us, gate 11.1 -> 10.5; FP8 row-f32 20.2 -> "
     "19.5) and RTX 5080 (NVFP4 2048x4096 6.66 -> 6.10 us, FP8 2048x4096 11.04 -> 10.20) faster; "
     "RTX 5060 Ti (even on every problem: only 1024 rows stay within two CTAs per SM there) and "
     "RTX 4080 SUPER (even) not; any other board takes the nearest of them within "
     "measured::kCarryDistance (else off), never across instruction sets",
     ""},
    {"spread_smem: thin calls", Basis::Unknown, "generic_gates.cpp",
     "epilogue != Epilogue::Store || args.tokens > 8 ||", 8, "gb202:170sm", "RTX 5090",
     "the eight-warp decode CTA (w1x8) spread over every SM: Store epilogue and at most eight "
     "tokens; measured at T = 1 only, SwiGLU loses (two CTAs per SM already: 116 registers); no "
     "record for the eight-token edge itself",
     ""},
    {"spread_smem: two CTAs per SM", Basis::Unknown, "generic_gates.cpp",
     "if (ctas > 2 * static_cast<std::int64_t>(device.sms) || pad > kStaticSharedBytesPerBlock)", 0,
     "gb202:170sm", "RTX 5090",
     "a call of at most two CTAs per SM: at 416 CTAs and more spreading loses (6656x4096 NVFP4 "
     "9.4 -> 12.6 us on the RTX 5090, 2.4 CTAs per SM); the two-per-SM line is the eight-warp "
     "CTA's residency (three to an SM) less one, a choice of the pad (a third of the shared "
     "memory), not a measured edge",
     ""},
    // --- w4a8_ws_wins and its neighbour ---
    {"w4a8_ws_extra: sm_89 and sm_120 only", Basis::Measured, "generic_select.cpp",
     "return gb20x_cc(d.major, d.minor) || ada_cc(d.major, d.minor);", 0, "gb206:36sm, ad106:34sm",
     "RTX 5060 Ti, RTX 4060 Ti",
     "IQ4_XS, IQ2_XS, IQ2_XXS and IQ3_S take the producer-warp tile where a measured choice names "
     "it, on sm_89 and sm_120 only: product, 24K prefill, A B B A, greedy byte-identical: RTX "
     "5060 Ti 981 -> 1011 t/s (IQ4_XS, IQ2_XS, IQ2_XXS), RTX 4060 Ti 900 -> 901 with those and 926 "
     "with IQ3_S on its shared-table codec; the comment also names sm_89 / sm_120 for IQ3_S "
     "(W4a8Iq3SPaired); other architectures keep the classic tiles, not measured",
     ""},
    {"w4a8_ws_wins: ampere", Basis::Measured, "generic_select.cpp", "if (ampere) return true;", 0,
     "ga102:82sm", "RTX 3090",
     "no format lost; one 16K prompt, D WS WS D: Muse 30B Q4_K_M 1278 / 1273 -> 1398 / 1399 t/s, "
     "Qwen3.8 27B UD-Q4_K_XL 1290 / 1289 -> 1314, Gemma 4 31B 1071 -> 1081; measured on the "
     "RTX 3090 only, taken for sm_80 and sm_86 alike",
     ""},
    {"w4a8_ws_wins: iq3xxs and iq2s", Basis::Measured, "generic_select.cpp",
     "if (ampere) return true; switch (p.qtype) { case QType::GGML_IQ3_XXS: case "
     "QType::GGML_IQ2_S: return blackwell || ada;",
     0, "gb203:84sm, gb202:170sm, ad102:128sm", "RTX 5080, RTX 5090, RTX 4090",
     "W4A8 micro-bench, T = 200 and 1024, the producer-warp tile's time against the best classic "
     "tile: RTX 5080 and RTX 5090 IQ3_XXS 0.79-0.93, IQ2_S 0.86-0.95; RTX 4090 IQ3_XXS 0.78-0.94, "
     "IQ2_S 0.89-0.93; in the product on the RTX 5080: Muse 30B (IQ3_XXS, Q6_K and Q4_K layers) "
     "prefill 2103 -> 2431 t/s, Gemma 4 31B UD-IQ3_XXS (IQ3_XXS, IQ2_S) 1696 -> 1912 t/s; only the "
     "measured architectures take it",
     ""},
    {"w4a8_ws_wins: q4_0 on ada", Basis::Measured, "generic_select.cpp",
     "case QType::GGML_Q4_K: return blackwell; case QType::GGML_Q4_0: return ada && p.n >= 5376 && "
     "p.k >= 5376;",
     5376, "ad102:128sm", "RTX 4090",
     "the smallest measured win (every problem of Gemma 4 31B QAT is 5376 or more each way); the "
     "26B QAT's 2816 loses; nothing between 2816 and 5376 measured, nor any sm_89 card with "
     "less L2 than the 4090's 72 MB (Gemma 4 31B QAT prefill 1914 -> 2149 t/s; the 26B QAT keeps "
     "64x64: 0.357 -> 0.383 s a prompt)",
     ""},
    {"w4a8_ws_wins: q6k blackwell only", Basis::Measured, "generic_select.cpp",
     "return blackwell || ada; case QType::GGML_Q6_K: return blackwell;", 0,
     "gb203:84sm, gb202:170sm, ad102:128sm", "RTX 5080, RTX 5090, RTX 4090",
     "W4A8 micro-bench, T = 200 and 1024, the producer-warp tile against the best classic tile: "
     "Q6_K 0.90-0.98 on the RTX 5080 and RTX 5090; Q6_K on sm_89 lost 2.2 % once the tiles run "
     "token-major (RTX 4090, Muse 30B Q4_K_M, 16K prompt: 2935 -> 2997 t/s without the tile; the "
     "micro-bench there has Q6_K Store 0.95-0.96 and SwiGLU 1.10)",
     ""},
    {"w4a8_ws_wins: q4k blackwell only", Basis::Measured, "generic_select.cpp",
     "case QType::GGML_Q6_K: return blackwell; case QType::GGML_Q4_K: return blackwell;", 0,
     "gb203:84sm, gb202:170sm, ad102:128sm", "RTX 5080, RTX 5090, RTX 4090",
     "W4A8 micro-bench, T = 200 and 1024, the producer-warp tile against the best classic tile: "
     "Q4_K 0.88-0.95 on the RTX 5080 and RTX 5090; on the RTX 4090 Q4_K lost (1.06-1.13) and "
     "takes the wide tile instead (w4a8_wide_q4k)",
     ""},
    {"w4a8_wide_q4k: grid bound", Basis::PlacedBetweenMeasured, "generic_select.cpp",
     "4 * ctas >= 3 * static_cast<std::int64_t>(d.sms)", 0, "ad102:128sm", "RTX 4090",
     "a grid of three quarters of the SMs or more (4 * ctas >= 3 * SMs); Muse 30B Q4_K_M "
     "micro-bench against the classic 128x64 at T = 1024 / 512 / 200: 0.86-0.98 where the grid "
     "fills three quarters of the SMs, 1.65-1.71 under it (64 CTAs); the points are 64 CTAs "
     "(4096x6656 at T = 200: 1.65) and 104 (6656x19968 at T = 200: 0.87) on 128 SMs, so the line "
     "at 96 lies between them",
     ""},
    // --- rules_r8's thresholds ---
    {"rules_r8: verify branch from two tokens", Basis::Structural, "generic_select.cpp",
     "if (tokens >= 2) return iq2 &&", 2, kUnknown, kUnknown,
     "calls of two tokens or more (the verify widths) take this branch and none of the "
     "architecture rules below it; T = 1 (decode) takes the rules; the record names no board or "
     "margin for the two-token edge",
     ""},
    {"rules_r8: iq2 two rows", Basis::Unknown, "generic_select.cpp",
     "iq2 && (tokens >= 4 || p.n >= 34816) ? r8_index(4, 2) : r8_index(4, 1);", 0, kUnknown,
     kUnknown,
     "the comment says 'within 3 % of the best schedule elsewhere' on the 27B/31B/35B verify "
     "shapes and names no board; the same shapes are the ones generic_gemm.h's verify comment "
     "measured on the RTX 5080 (gb203:84sm), which this comment does not repeat",
     ""},
    {"rules_r8: turing eight warps", Basis::Measured, "generic_select.cpp",
     "if (!device_class().features.mma_bf16_native) { const bool decode_heavy = "
     "format_traits(p.qtype).decode_heavy; if (decode_heavy && p.k >= 5120) return r8_index(8, 2);",
     5120, "tu102:68sm", "RTX 2080 Ti",
     "ninferno_r8_schedule_bench on the IQ3 16 GB artifacts' 63 problems: the picks +6.25 % above "
     "the best schedules in sum (IQ3_XXS 17408x5120 110.2 -> 84.5 us, IQ3_S 131.3 -> 114.2, Q3_K "
     "112.9 -> 89.0)",
     ""},
    {"rules_r8: turing q6k two rows", Basis::Measured, "generic_select.cpp",
     "return r8_index(4, 1); if (p.qtype == QType::GGML_Q6_K && p.epilogue != Epilogue::SwiGlu && "
     "(p.n >= 8192 || p.k >= 4096)) return r8_index(4, 2);",
     0, "tu102:68sm", "RTX 2080 Ti",
     "N >= 8192 or K >= 4096: the 248320x2048 lm_head 807 -> 723 us, 2816x8192 w8r2 42.3 -> w4r2 "
     "40.3",
     ""},
    {"rules_r8: volta one slot", Basis::Measured, "generic_select.cpp",
     "if (p.qtype == QType::GGML_Q6_K && p.epilogue != Epilogue::SwiGlu && r8_one_slot_build() && "
     "!device_class().features.mma_tensor) return r8_index(4, 1);",
     0, "gv100:80sm", "V100 (GV100)",
     "4096x12288 add w4r1 88.8 against w4r2 94.7 us, 8192x4096 store 57.9 against 63.3 (ten other "
     "kernels between launches)",
     ""},
    {"rules_r8: q3k four warps", Basis::Measured, "generic_select.cpp",
     "return r8_index(4, 2); } if (p.qtype == QType::GGML_Q3_K && p.k >= 8192) return "
     "r8_index(4, 1);",
     8192, "gb206:36sm, ad103:80sm", "RTX 5060 Ti, RTX 4080 SUPER, RTX 4070 Ti SUPER",
     "Qwen3.8's 5120x17408 add is 7-10 % faster so (113.2 -> 103.0, 64.8 -> 60.3, 73.0 -> 66.3 "
     "us); the RTX 4070 Ti SUPER's class id is not named in the comment",
     ""},
    {"rules_r8: wide K", Basis::Measured, "generic_select.cpp",
     "if (p.qtype == QType::GGML_Q3_K && p.k >= 8192) return r8_index(4, 1); if (p.k >= 8192 || "
     "(iq2 && p.n >= 12288)) return r8_index(8, 2);",
     0, "gb203:84sm", "RTX 5080",
     "K >= 8192, or N >= 12288 for IQ2: Qwen3.5-4B/9B shapes: IQ3_XXS 4096x12288 32.1 -> 27.9 us, "
     "IQ2_S 12288x4096 28.3 -> 24.9 us; the ggml blocks and K-quants follow the K rule within 2 % "
     "of their best schedule",
     ""},
    {"rules_r8: blackwell branch gate", Basis::PlacedBetweenMeasured, "generic_select.cpp",
     "if (p.k >= 8192 || (iq2 && p.n >= 12288)) return r8_index(8, 2); if (p.epilogue != "
     "Epilogue::SwiGlu && p.k >= 5120 && consumer_blackwell_major(device_class().major)) {",
     5120, "gb203:84sm, gb206:36sm, gb202:170sm", "RTX 5080, RTX 5060 Ti, RTX 5090",
     "the gate of the four Blackwell rules below: every shape their comments cite has 5120 or "
     "5376 as its K (10240x5120, 17408x5120, 16384x5120, 8192x5376, 43008x5376); on the RTX 5080 "
     "(T = 1, store epilogue) 'at K <= 4096, w4r1 is ahead at every N' and the gains start at "
     "K = 5120..6656 (5120x6144 gains 1 %), so the edge is placed between 4096 and 5120 and no "
     "timing sits between them; the RTX 5060 Ti's problems are at K = 5120",
     ""},
    {"rules_r8: blackwell iq2 from N 5120", Basis::Measured, "generic_select.cpp",
     "consumer_blackwell_major(device_class().major)) { if (iq2 && p.n >= 5120) return "
     "r8_index(8, 2);",
     5120, "gb206:36sm, gb203:84sm, gb202:170sm", "RTX 5060 Ti, RTX 5080, RTX 5090",
     "IQ2_S 10240x5120 w4r2 29.9 -> w8r2 20.1 us (still gains at 170 SMs: the 5090); IQ2_XXS "
     "10240x5120 43.5 -> 40.0, IQ2_XS "
     "17408x5120 83.6 -> 79.4 (5060 Ti)",
     ""},
    {"rules_r8: blackwell sm count", Basis::PlacedBetweenMeasured, "generic_select.cpp",
     "if (iq2 && p.n >= 5120) return r8_index(8, 2); if (device_class().sms <= 96) {", 96,
     "gb203:84sm, gb202:170sm", "RTX 5080, RTX 5090",
     "an SM-count rule whose edge lies between the measured boards (84 and 170 SMs; no board "
     "between; docs/maintainer/deterministic-decisions.md lists it among the rules resting on few "
     "boards): 10240x5120 IQ3_S w4r1 18.4 us against w8r2 20.3 at 170, IQ3_XXS 16.0 against 18.0",
     ""},
    {"rules_r8: blackwell iq3s", Basis::Measured, "generic_select.cpp",
     "if (device_class().sms <= 96) { if (p.qtype == QType::GGML_IQ3_S && p.n >= 5120) return "
     "r8_index(8, 2);",
     5120, "gb203:84sm, gb206:36sm", "RTX 5080, RTX 5060 Ti",
     "IQ3_S 8192x5376 28.5 -> 27.5 us (5080); 16384x5120 w4r1 100.6 -> 94.5 (5060 Ti)", ""},
    {"rules_r8: blackwell iq3xxs", Basis::Measured, "generic_select.cpp",
     "if (p.qtype == QType::GGML_IQ3_S && p.n >= 5120) return r8_index(8, 2); if (p.qtype == "
     "QType::GGML_IQ3_XXS && p.n >= 5120) return r8_index(4, 2); } }",
     5120, "gb203:84sm, gb206:36sm", "RTX 5080, RTX 5060 Ti",
     "IQ3_XXS 43008x5376 112.9 -> 106.6 us (5080); 16384x5120 92.8 -> 82.5 (5060 Ti)", ""},
    {"rules_r8: iq2 two rows from N 4096", Basis::Measured, "generic_select.cpp",
     "return r8_index(4, 2); } } if (iq2 && p.n >= 4096) return r8_index(4, 2); return "
     "r8_index(4, 1); }",
     4096, "gb203:84sm", "RTX 5080", "IQ2_XS 9216x2560 14.1 -> 12.8 us", ""},
    // --- ggml's vector kernel's launch (ggml_mmq/ggml_vec.cu) ---
    {"kVecWarps", Basis::Structural, "ggml_mmq/ggml_vec.cu", "constexpr int kVecWarps = 4;", 4,
     kUnknown, kUnknown,
     "the vector kernel's warps a CTA: 128 threads, the CTA size the resident-CTA count of "
     "kVecBoundedBlocks is counted for; a count of the kernel, no margin",
     ""},
    {"kVecBoundedBlocks", Basis::Structural, "ggml_mmq/ggml_vec.cu",
     "constexpr int kVecBoundedBlocks = 7;", 7, "tu102:68sm, ad102:128sm",
     "RTX 2080 Ti, RTX 4090 (for the timings; the comment also names an 82-SM card, not by "
     "product)",
     "the bounded twin of the runtime-K plain instance of 4 columns holds 7 resident CTAs per SM "
     "(__launch_bounds__' minimum): the registers of the 3-column instance, 72 against the 85 the "
     "unbounded one takes, which is 5 CTAs of 128 threads on a 64K-register SM instead of 7. The "
     "launch takes the twin only where that saves a round of row groups (a counted saving), "
     "'deliberately conservative': the twin is 11..22 % faster on a 2080 Ti even where it is not "
     "taken, and 9 % slower on a 4090 at 1024 groups with 2 rounds either way; the 7 is a "
     "register count, the other numbers are timings that do not place it",
     ""},
});

} // namespace ninferno::ops::detail::generic::route_rules
