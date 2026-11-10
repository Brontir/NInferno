#pragma once

// The literal rules of the generic path's route selection (generic_gemm.h, generic_gemm.cpp): the
// thresholds a rule compares a call's shape, width or card against, each a number someone measured
// on some boards and wrote into the code (#614, PR-F4). This header declares what each rule IS
// (its literal) and what it rests on (the boards and cards it was measured on, the margin the
// record gives, where it is written); it changes no behaviour. The code keeps its own literals:
// test_route_rules reads the sources and holds every descriptor to them (a named constant's value,
// a rule's literal text inside the file), and every bounding constant under src/ops/generic (a
// recursive scan by file and name, see the test) to a descriptor or a listed non-rule, so a rule
// cannot change, move or appear as such a constant without its record.
//
// What the list covers: the named constants that bound a route, the threshold comparisons of
// rules_r8, w4a8_ws_wins, heuristic_rmma, heuristic_w4a8, heuristic_w4a4, staged_speed_route,
// tiled_speed_pays, nvfp4_tiled_route, auto_route's NVFP4 rule, the vector and R8 verify edges and
// spread_smem, and the table of boards that decides spread_smem. What it does not: see
// docs/maintainer/route-rules.md, which names the literals knowingly left out.
//
// Fields:
//   name       the rule as the code names it (`symbol` or a short name for a literal inside one);
//   file       the source file that holds the literal, under src/ops/generic/ (or a path from it
//              for a file beside them);
//   literal    the text of the literal as written: a constant's WHOLE definition (to its `;`, an
//              array with every element), or a comparison with its action where the action matters;
//   value      the number of a constant (computed as C++ computes it, from the source's definition)
//              or of a condition with ONE numeric cutoff that is written in the literal; 0 for an
//              array, a condition with several cutoffs and a condition that compares against a
//              card's own quantity (all cutoffs are then in the literal);
//   boards     the device class ids it was measured on (DeviceClass::id: chip:Nsm, one format
//              throughout), "unknown" where the comments and the record name none;
//   cards      the same as the products' names;
//   tolerance  the measured margin the record gives ("none recorded" where it gives none): what the
//              rule was held to, or what moving the literal was seen to cost. A number FITTED to
//              measurements says so here;
//   kernel_hash  the identity of the kernel the rule was measured against: EMPTY for every rule.
//              The repository records one sha256 over all device sources per kernel-facts file
//              (kernel_facts.h Facts::source, tools/kernel_facts), not per kernel, and no rule
//              records which tree it was timed on; a rule re-measured later should fill this.
// Where the record is silent a field says "unknown": nothing here is a guess. The source of a
// number is the comment at its use (quoted in the tolerance where it states a margin), DECISIONS.md
// and docs/maintainer.

#include <array>
#include <cstddef>
#include <string_view>

namespace ninferno::ops::detail::generic::route_rules {

struct RouteRule {
    std::string_view name;
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

// The boards of the speed profile's seven-board fit (ninferno_route_tune): the six named in the
// comments with their class ids, and the A10 (ga102:72sm) and the L4 (ad104:58sm).
inline constexpr std::string_view kSpeedFitBoards =
    "ga102:82sm, ad102:128sm, gb202:170sm, tu102:68sm, ga100:108sm, ga102:72sm (A10), ad104:58sm "
    "(L4)";
inline constexpr std::string_view kSpeedFitCards =
    "RTX 3090, RTX 4090, RTX 5090, RTX 2080 Ti, A100, A10, L4";

inline constexpr auto kRouteRules = std::to_array<RouteRule>({
    // --- R8's widths, the matrix kernel's band and the verify rules (generic_gemm.h) ---
    {"kR8MaxTokens", "generic_gemm.h", "inline constexpr std::int32_t kR8MaxTokens = 4;", 4,
     "gb203:84sm", "RTX 5080",
     "no four-row schedule and at most four tokens: the K-quant codecs' registers spilled (up to 2 "
     "KiB of stack a thread, w4r4 SwiGLU at eight tokens); w4r4 was the best T = 1 schedule on 2 "
     "of 70 shapes, by 1 %",
     ""},
    {"kR8DecodeTokens", "generic_gemm.h", "inline constexpr std::int32_t kR8DecodeTokens = 1;", 1,
     "gb203:84sm", "RTX 5080",
     "R8 at T = 1 only: at T = 4 it lost on 26 of 42 decode shapes of Qwen3.5-4B/9B; at T = 1 it "
     "won on all 42 (0.58-0.96 of the register MMA) and on 69 of 70 Qwen3.8-27B / Gemma 4 31B / "
     "Qwen3.6-35B-A3B shapes (0.77-0.99; Q6_K 4096x2048 1.05)",
     ""},
    {"kR8VerifyTokens", "generic_gemm.h", "inline constexpr std::int32_t kR8VerifyTokens = 8;", 8,
     "gb202:170sm, gb203:70sm, ad102:128sm", "RTX 5090, RTX 5070 Ti, RTX 4090",
     "Q8 at T = 2..8 took 0.96-1.15 of its T = 1 time on R8 against 0.99-1.24 on the register MMA "
     "(5090); past four tokens the other formats' R8 lost (Q4_K 1.48-1.89, IQ3_XXS 1.49-1.80, "
     "IQ4_XS 1.19-1.53); the dp4a work equals the weight read at 21 tokens (A100, H100), 37 "
     "(RTX 3060), 40 (GV100), 86 (RTX 4090); RTX 4090 Qwen3.5-9B Q8_0 dflash:7 229 -> 273 tok/s",
     ""},
    {"r8_verify_tunable: lower edge", "generic_gemm.cpp",
     "return tokens >= 2 && tokens <= kR8VerifyTokens && allows_r8(w, policy) &&", 2, kUnknown,
     kUnknown,
     "the verify band of R8 starts at two tokens (T = 1 is the decode rule, kR8DecodeTokens); the "
     "record names no board or margin for the lower edge. generic_workspace.cpp reads the same "
     "edge as its own literals (route_workspace_bytes, the interval plan): a change of the edge "
     "is a change in three places",
     ""},
    {"kMmqMinTokens", "generic_gemm.h", "inline constexpr std::int32_t kMmqMinTokens = 5;", 5,
     "unknown (decisions.md 58 names the families: Ampere, Ada, Blackwell and Turing)",
     "unknown (the same families)",
     "docs/maintainer/decisions.md 58: five tokens, past the four R8 verifies for every format "
     "but Q8; the vector kernel through T = 4 on the native-MMA boards, Turing's codebook formats "
     "at T = 5..8 to the matrix kernel",
     ""},
    {"kMmqMaxTokens", "generic_ggml_internal.h",
     "inline constexpr std::int32_t kMmqMaxTokens = 16;", 16, "gb202:170sm, ad103:80sm, gb203:84sm",
     "RTX 5090, RTX 4080 SUPER, RTX 5080",
     "no fixed rule holds across chips and formats (the dense tuner times it): against the "
     "heuristic's tile, serve C4: 5090 ISTA IQ3_S +10 % at 16 tokens; Muse Q4_K_M -7..14 %, 4080 "
     "SUPER ISTA -12 % at 8; 5080 IQ3_S even; below the band it loses to R8 (5090 C1 -11 %)",
     ""},
    {"kGgmlMmqMaxTokens", "generic_gemm.h", "inline constexpr std::int32_t kGgmlMmqMaxTokens = 16;",
     16, "see kMmqMaxTokens", "see kMmqMaxTokens", "see kMmqMaxTokens (the same 5..16-token band)",
     ""},
    {"kGgmlVecMaxTokens", "generic_gemm.h", "inline constexpr std::int32_t kGgmlVecMaxTokens = 8;",
     8, kUnknown, kUnknown, "ggml_mmq::kVecMaxColumns, the kernel's own width; no margin recorded",
     ""},
    {"kMmqVerifyMinTokens", "generic_gemm.cpp", "constexpr std::int32_t kMmqVerifyMinTokens = 9;",
     9, "ad104:58sm (L4), gb203:70sm, ad102:128sm, tu102:68sm",
     "L4, RTX 5070 Ti, RTX 4090, RTX 2080 Ti",
     "ISTA IQ3_S 27B C4 MTP3 (16-token verify): L4 79.4 -> 89.2 tok/s (+12.3 %), RTX 5070 Ti "
     "242.7 -> 270.9 (+11.6 %); the K-quants lost on the 4090 (UD-Q4_K_XL 27B DFlash2 C1 115.7 -> "
     "111.7, C4 271.5 -> 263.1) and stay on the register MMA; where the BF16 MMA is emulated "
     "(Turing, RTX 2080 Ti) every matrix-kernel format takes it from 9 tokens, the codebook "
     "formats below that too (mmq_verify_rule)",
     ""},
    // The prefill band's edge: ONE definition (ops/route_pins.h), the widest speculative verify
    // plus one; the others below are defined from it, because their comments call the same width.
    {"kDecodePinMaxTokens", "../route_pins.h",
     "inline constexpr std::int32_t kDecodePinMaxTokens = 16;", 16, kUnknown, kUnknown,
     "the widest speculative verify (a 16-token draft's verify plus its anchor): the decode pin's "
     "band, 1..16 tokens; structural, no margin recorded. kDenseTuneMaxTokens, "
     "kPdlPrefetchMaxTokens and kMmqMaxTokens are 16 too and are NOT defined from it: their "
     "comments name their own measurements",
     ""},
    {"kPrefillPinMinTokens", "../route_pins.h",
     "inline constexpr std::int32_t kPrefillPinMinTokens = kDecodePinMaxTokens + 1;", 17, kUnknown,
     kUnknown,
     "one past the widest speculative verify, so the decode and prefill pin bands are adjacent; "
     "the prefill route profile's first width, where a call takes one route per problem. "
     "kPrefillBandMinTokens, kQ8PrefillA8Tokens, kVoltaCutlassMinTokens and route_table.cpp's "
     "kPrefillMinTokens are defined from it: their comments call the same width",
     ""},
    {"kPrefillBandMinTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kPrefillBandMinTokens = ops::kPrefillPinMinTokens;", 17,
     kUnknown, kUnknown,
     "the generic path's name for kPrefillPinMinTokens: the first width of the prefill band, one "
     "past the widest speculative verify (16 tokens: a 16-token draft's verify plus its anchor); a "
     "structural width, no margin recorded. The selection, the workspace plan and the tuner read "
     "it; the tuner's grid and the selector facts' grid start at it, held by static_asserts",
     ""},
    {"kPrefillMinTokens", "../route_table.cpp",
     "constexpr std::int32_t kPrefillMinTokens = kPrefillPinMinTokens;", 17, kUnknown, kUnknown,
     "the first width a route table's prefill band may start at: 'the first call width the "
     "prefill route profile makes token-invariant', the same width as kPrefillPinMinTokens, from "
     "which it is defined",
     ""},
    {"kQ8PrefillA8Tokens", "generic_gemm.h",
     "inline constexpr std::int32_t kQ8PrefillA8Tokens = kPrefillBandMinTokens;", 17, "gb202:170sm",
     "RTX 5090",
     "defined from kPrefillBandMinTokens (its comments say 'a call of 17 tokens or more inside a "
     "prefill'); the chunk-invariance contract's width, not a margin: a DFlash2 round of 17+ "
     "columns on A8 cost Qwen3.8-27B --spec auto C=1 2-3 %, so only the prefill takes it",
     ""},
    {"kDenseTuneMaxTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kDenseTuneMaxTokens = 16;", 16, kUnknown, kUnknown,
     "the decode and verify widths the dense tuner times; no margin recorded; equal to "
     "kDecodePinMaxTokens today, not defined from it",
     ""},
    {"kR8VerifyQ40MinValues", "generic_gemm.cpp",
     "inline constexpr std::int64_t kR8VerifyQ40MinValues = 24ll << 20;", 25165824,
     "ad102:128sm, ga102:82sm", "RTX 4090, RTX 3090",
     "Q4_0 R8 at T = 2..4 pays on a large matrix (Gemma 4 31B QAT, every dense matrix >= 44M "
     "values, C = 4: RTX 4090 +15.7 %, RTX 3090 +26 %) and loses on a small one (Gemma 4 26B QAT, "
     "at most 12M: -0.7 %, -2.7 %, its MTP verify -1.1 %); nothing between 12M and 44M measured",
     ""},
    {"kRmmaQ6kFourWarpBytes", "generic_gemm.cpp",
     "inline constexpr double kRmmaQ6kFourWarpBytes = 4.0;", 4.0,
     "ad102:128sm, ad104:60sm, ad106:34sm, gb203:70sm",
     "RTX 4090, RTX 4070 Ti, RTX 4060 Ti, RTX 5070 Ti",
     "w1x4 against w1x8: RTX 4090 (3.1 bytes a SM cycle) 0.82-0.98, 4070 Ti (3.2) 0.80-0.99, 4060 "
     "Ti (3.4) 0.87-1.00; 5070 Ti (5.2) 0.98-1.12; the 5080 and 5090 (4.4) not measured",
     ""},
    {"kRmmaLatencySwiGluBytes", "generic_gemm_internal.h",
     "inline constexpr double kRmmaLatencySwiGluBytes = 4.5;", 4.5,
     "ad102:128sm, ad104:60sm, gb203:70sm, ga106:28sm",
     "RTX 4090 (3.1 bytes a SM cycle), RTX 4070 Ti (3.2), RTX 5070 Ti (5.2), RTX 3060 (7.2)",
     "SwiGLU, T = 5..8, Qwen3.5-9B shapes: the 4090 and 4070 Ti (plain kernel at 86-91 % of DRAM) "
     "lost 8-34 % to the latency variant, the 5070 Ti and 3060 (64-74 %) gained 2-15 % and "
     "11-25 %; the threshold lies between the boards measured on either side "
     "(generic_gemm_launch.cuh)",
     ""},
    {"kRmmaLatencyOneSetBytes", "generic_gemm_internal.h",
     "inline constexpr double kRmmaLatencyOneSetBytes = 5.5;", 5.5,
     "ga106:28sm, tu102:68sm, gb203:70sm",
     "RTX 3060 (7.2 bytes a SM cycle), RTX 2080 Ti (5.9), RTX 5070 Ti (5.2)",
     "one row set (store, add), T = 5..8: the 3060 gained 8-21 %, the 2080 Ti 0-4 %, the 5070 "
     "Ti between +14 % (Q3_K) and -3 % (Q4_K add 4096x12288 w1x4, 40.7 -> 41.7 us), so the "
     "threshold lies above that board (generic_gemm_launch.cuh)",
     ""},
    {"kRmmaWideBytes", "generic_gemm_internal.h", "inline constexpr double kRmmaWideBytes = 5.5;",
     5.5, "ad102:128sm, gb203:70sm, ga106:28sm", "RTX 4090, RTX 5070 Ti, RTX 3060",
     "A/B, T = 1..8: RTX 4090 (3.1 bytes a SM cycle) add 71 -> 59 us, 5070 Ti (5.2) every problem "
     "1-8 % faster; on the RTX 3060 (7.2) the extra shuffles cost 2-10 %",
     ""},
    // --- Volta's routes (generic_gemm.cpp, generic_gemm_internal.h) ---
    {"kVoltaCutlassMinTokens", "generic_gemm.cpp",
     "constexpr std::int32_t kVoltaCutlassMinTokens = kPrefillBandMinTokens;", 17, "gv100:80sm",
     "V100",
     "defined from kPrefillBandMinTokens (the comment at the rule: a tiled-route call of the "
     "prefill band, 17 tokens on). V100 ninferno_linear_bench, T = 32..240, the tile's time "
     "against the decoded copy's: Q4_K 24576 x 4096 1.36..2.29 ms against 1.20..1.40, 4096 x "
     "12288 1.07..1.31 against 0.82..0.88, 4096 x 4096 0.36..0.44 against 0.27..0.28, Q5_K 4096 x "
     "12288 1.13..1.37 against 0.90..0.94; the one shape whose grid is a whole wave, 10240 x "
     "4096, keeps the tile ahead below 129 tokens (0.37..0.47 ms against 0.50)",
     ""},
    {"kVoltaTileMaxTokens", "generic_gemm.cpp", "constexpr std::int32_t kVoltaTileMaxTokens = 256;",
     256, "gv100:80sm",
     "V100 (the rule, for boards under the copy's bandwidth bound, is not measured on them)",
     "none recorded: two token blocks already decode the weight twice from 256", ""},
    {"kVoltaCopyMinDramGbps", "generic_gemm.cpp",
     "constexpr double kVoltaCopyMinDramGbps = 870.0 / 1.13;", 870.0 / 1.13, "gv100:80sm",
     "V100 (870 GB/s); the Titan V (653) is under the bound, not measured",
     "the copy's narrowest margin is 1.13 (24576 x 4096 below 129 tokens) at 870 GB/s", ""},
    {"kVoltaHmmaMinTokens", "generic_gemm_internal.h",
     "inline constexpr std::int32_t kVoltaHmmaMinTokens = 1;", 1, "gv100:80sm", "V100",
     "a byte-consistency contract, not a margin: a one-token call in a lane-local call ran the "
     "FP32 GEMV and the same token in a wider call the HMMA kernel (test_moe)",
     ""},
    {"kVoltaHmmaMaxTokens", "generic_gemm_internal.h",
     "inline constexpr std::int32_t kVoltaHmmaMaxTokens = 16;", 16, "gv100:80sm", "V100",
     "the verify widths (Q4_K / Q5_K / Q6_K and Q8 G32); Quadro GV100 Qwen3.8 27B UD-Q4_K_XL: the "
     "register MMA at T = 5..16 about 0.5 ms a call against 0.08 for the vector kernel at T = 4; "
     "C8 aggregate 37.3 tok/s against C4 86.3",
     ""},
    // --- The vector kernel's verify widths ---
    {"kVecRuleMaxTokens", "generic_gemm.cpp", "constexpr std::int32_t kVecRuleMaxTokens = 4;", 4,
     "tu102:68sm, ga100:108sm, ga102:82sm, gb202:170sm, ad102:128sm, gv100:80sm",
     "RTX 2080 Ti, A100, RTX 3090, RTX 5090, RTX 4090, V100",
     "in the dense tuner's timings of eight cards (four named in the comment: A100 x1.5..1.6, RTX "
     "3090 x1.4..1.5, RTX 5090 x1.1, RTX 4090 x1.03) the vector kernel was within 4 % of the best "
     "route at T = 2..4 and behind it at T = 8 on all of them; RTX 2080 Ti: lost decode by 10 % "
     "(9B Q4_K_M C1 88 -> 79 tok/s), won C4 by 10 %; Volta takes every width it reads instead "
     "(V100, see kVoltaHmmaMaxTokens); docs/maintainer/decisions.md 58: the band rule (vector "
     "kernel at T <= 4, register MMA from 5) is the one measured on Ampere, Ada and Blackwell, "
     "held against a price-based pick (RTX 4070 Ti, Q6_K 3840x15360, T = 5)",
     ""},
    // The rule's lower edge, from two tokens; the record names no board or margin for it.
    {"vec_rule lower edge", "generic_gemm.cpp",
     "return tokens >= 2 && tokens <= vec_rule_max_tokens() && vec_call(w, policy, tokens);", 2,
     kUnknown, kUnknown, kNotRecorded, ""},
    // --- The widths where the speed profile's heuristic choices change (generic_gemm.h) ---
    {"kRmmaMidMinTokens", "generic_gemm.h", "inline constexpr std::int32_t kRmmaMidMinTokens = 33;",
     33, kSpeedFitBoards, kSpeedFitCards,
     "FITTED: one rule for every board, fitted on six boards and scored on the seventh: 1.116 of "
     "the best (geomean) against 1.219 for the bands it replaced; a rule per board gained 1.1 % "
     "at most",
     ""},
    {"kStreamMaxTokens", "generic_gemm.h", "inline constexpr std::int32_t kStreamMaxTokens = 512;",
     512, kUnknown, kUnknown, "none recorded", ""},
    {"kTiledSpeedMinTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kTiledSpeedMinTokens = 33;", 33,
     "every board measured, not listed (Q4_K n = 1024 at T 64..128)", "unknown",
     "the tiled kernel 5..7x the register MMA's at Q4_K n = 1024, T 64..128, on every board", ""},
    {"kStagedSpeedMinTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kStagedSpeedMinTokens = 256;", 256, kUnknown, kUnknown,
     "none recorded", ""},
    {"kSpeedFill", "generic_gemm.h", "inline constexpr double kSpeedFill = 1.5;", 1.5,
     kSpeedFitBoards, kSpeedFitCards,
     "FITTED: 1.0..2.0 cost under 0.5 % (the header's comment); the same board-held-out fit as "
     "kRmmaMidMinTokens",
     ""},
    {"kW4a8WideMinTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kW4a8WideMinTokens = 128;", 128,
     "ad103:80sm, gb202:170sm, gb206:36sm, ad102:128sm",
     "RTX 4080 SUPER, RTX 5090, RTX 5060 Ti, RTX 4090",
     "below it (the 128x64 tile against 64x64): 1.10-1.56 on IQ2_*/IQ3_*/IQ4_XS (4080 SUPER), Q4_K "
     "1.05-1.09 .. Q6_K "
     "1.17-1.20 (5090), 1.08-1.29 (5060 Ti); in the product 4080 SUPER Qwen3.8-27B IQ3 412.9 -> "
     "472.4 tok/s, 5090 Q4_K_XL 738 -> 757",
     ""},
    {"kPdlPrefetchMaxTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kPdlPrefetchMaxTokens = 16;", 16, kUnknown, kUnknown,
     "none recorded (decode and speculative verify widths)", ""},
    // --- The prefill band, the dense tuner and the heuristics' width bands ---
    {"kPrefillBandTokens", "generic_tune.cpp",
     "inline constexpr std::array<std::int32_t, 9> kPrefillBandTokens = {17, 32, 64, 127, 128, "
     "256, 512, 1024, 2048};",
     0, "unknown", "unknown",
     "the widths whose band takes a tuned route; an untimed width takes the next band's: 17 is "
     "a 16-token draft's verify (kPrefillBandMinTokens, held by a static_assert), 32, 64, 127 and "
     "128 straddle the heuristics' switch widths (the register MMA's prefill column at 32 and 64, "
     "W4A8's 128x64 from 65 and its tiles from 128); 256 .. 2048 no record; no margin recorded",
     ""},
    {"kMargin", "generic_tune.cpp", "constexpr double kMargin = 0.97;", 0.97, kUnknown, kUnknown,
     "a tuned schedule replaces the heuristic's only where at most 97 % of its time, so noise "
     "keeps the heuristic; no measurement behind the 3 % is recorded",
     ""},
    {"kBytes", "generic_tune.cpp", "constexpr double kBytes = 64.0 * kMiB;", 67108864, kUnknown,
     kUnknown,
     "the widest prefill band whose buffers stay within 64 MiB each is tuned (an output head of "
     "152K rows: T <= 128); wider bands keep the heuristic's route; none recorded",
     ""},
    {"tune fits: FLOP bound", "generic_tune.cpp", "2.0 * model_call.rows * k * t <= 2.5e11", 2.5e11,
     kUnknown, kUnknown,
     "the widest prefill band whose call is within 250 GFLOP is tuned; none recorded", ""},
    {"kRmmaWideSpillSlack", "generic_gemm_internal.h",
     "inline constexpr std::size_t kRmmaWideSpillSlack = 16;", 16, "ad102:128sm", "RTX 4090",
     "sm_89's one-row-set tiles keep 8 bytes of loop-invariant state in local memory; SwiGLU's "
     "128-register tiles spill 160 bytes in the loop (RTX 4090 w1x16 at T = 6: 100.3 -> 103.9 us)",
     ""},
    {"kVecMaxColumns", "ggml_mmq/ggml_mmq.h", "inline constexpr int kVecMaxColumns = 8;", 8,
     kUnknown, kUnknown,
     "ggml's vector kernel's width (after ninfer-all's ggml_vec.cu); feeds vec_rule_max_tokens "
     "without tensor mma.sync; no margin recorded",
     ""},
    {"heuristic_rmma: prefill column", "generic_gemm.cpp",
     "const bool narrow_band = tokens <= 32 || (tokens <= 64 && ctas < 2 * sm_count());", 0,
     "ad103:80sm, gb202:170sm, gb206:36sm", "RTX 4080 SUPER, RTX 5090, RTX 5060 Ti",
     "time of m1.n8 over the best: T = 17 and 32 1.48-3.21 on every problem (4080 SUPER); T = 64 "
     "1.20-1.58 below 160 CTAs, 1.00 at 272; 5090 Q6_K output head m1.n4 best at T = 17 (1.22) "
     "and 32 (1.54); 5060 Ti 1.60-2.14 at T = 17 and 32; in the product 4080 SUPER Qwen3.8-27B "
     "IQ3 246.6 -> 252.9 tok/s",
     ""},
    {"heuristic_rmma: narrow through 16", "generic_gemm.cpp",
     "if (tokens <= 16 || (split != 8 && tokens < kRmmaMidMinTokens)) return narrow();", 16,
     kUnknown, kUnknown,
     "the verify widths keep the decode splits; none recorded beyond kRmmaMidMinTokens' fit", ""},
    {"heuristic_rmma: K split edges", "generic_gemm.cpp",
     "const int split = k >= 1024 ? 8 : k >= 512 ? 4 : 1;", 0, "gb202:170sm", "RTX 5090",
     "K split across as many warps as keep at least 128 K values each (1024 / 8, 512 / 4): what "
     "ninferno_route_tune measured across the recorded problems (Qwen3.5 0.8B in four formats); "
     "no margin recorded. invariant_rmma repeats the K edges (below)",
     ""},
    {"invariant_rmma: K split edges", "generic_gemm.cpp",
     "if (k >= 1024) return rmma_index(1, 8, 1, n_tiles); if (k >= 512) return rmma_index(1, 4, "
     "1, n_tiles);",
     0, kUnknown, kUnknown,
     "the decode split of K for every T (the invariant profile): the same edges as "
     "heuristic_rmma's K split, written again as literals; no record of its own",
     ""},
    {"heuristic_rmma: nine to sixteen tokens", "generic_gemm.cpp", "} else if (tokens <= 16) {", 16,
     "gb203:84sm", "RTX 5080",
     "9..16 tokens (verify of an 8- or 16-token draft), RTX 5080, Q8/Q4_0/IQ4_XS/Q4_K/Q5_K/Q6_K/"
     "IQ3_XXS at 27B/31B/35B shapes: four K-split warps with two tiles 1.00-1.12 of the best of "
     "seven schedules against 1.24-1.58 for w1x8 n1 (geomean per format; at T = 24 Q8 1.35 "
     "against 1.68)",
     ""},
    {"one_wave_split: quarter of the SMs", "generic_gemm.cpp",
     "return 4 * ctas <= sms ? rmma_index(1, 8, 1, 1, 8) : rmma_index(1, 8, 1, 1, 2);", 0,
     "gb203:84sm, gb202:170sm", "RTX 5080, RTX 5090",
     "a call of at most a quarter as many CTAs as SMs (the 32-row GDN a/b projections: 2 CTAs, "
     "12.6 us for 139 KB) splits K over a cluster of eight CTAs: 7.5 -> 3.1 us (RTX 5080, "
     "Qwen3.5-4B/9B and Gemma 4 12B shapes); the CTA pair for the rest on the RTX 5090, T = "
     "1..4: Q8 2048x4096 23.7 -> 17.6 us; the quarter edge itself has no margin recorded",
     ""},
    {"heuristic_rmma: speed rule token tiles", "generic_gemm.cpp",
     "const int nt_cap = emulated ? 4 : 8;", 0, kSpeedFitBoards, kSpeedFitCards,
     "the widest token tile (8 tokens each) of the speed rule: 4 tiles (32 tokens) where the BF16 "
     "MMA is emulated, 8 otherwise; inside the seven-board fit of kRmmaMidMinTokens, which the "
     "comment does not break out: no record of its own",
     ""},
    {"heuristic_rmma: sixteen warps from 72 K steps", "generic_gemm.cpp",
     "return ctas <= sm_count() || 2 * steps >= 9 * 16 ? rmma_index(1, 16, 1, 1) : rmma_index(1, "
     "8, 1, 1);",
     0, kSpeedFitBoards, kSpeedFitCards,
     "with one token tile, sixteen K warps where the call has at most one CTA per SM or a long K "
     "(from 72 K steps of 128 K values for the heavy formats, 32 otherwise), else eight; inside "
     "the seven-board fit of kRmmaMidMinTokens: no record of its own",
     ""},
    {"heuristic_w4a8: 17..32", "generic_gemm.cpp", "if (tokens <= 32) return k32x128;", 32,
     "gb203:84sm", "RTX 5080",
     "7 GGUF formats at 27B/31B/35B shapes, time over the best of seven routes: at T = 17 32x128 "
     "1.05-1.45 against 64x64 1.14-1.57, better for every format at 17, 20, 24, 32",
     ""},
    {"heuristic_w4a8: 33..64", "generic_gemm.cpp", "if (tokens <= 64) return k64x64;", 64, kUnknown,
     kUnknown, "none recorded for the 64-token edge itself", ""},
    {"heuristic_w4a8: wide tile grid", "generic_gemm.cpp",
     "if (2 * wide_ctas < sm_count()) return k64x64;", 0, "ad103:80sm, gb202:170sm, gb206:36sm",
     "RTX 4080 SUPER, RTX 5090, RTX 5060 Ti",
     "from 65 tokens a 128-token tile pays while its grid still gives half the SMs a CTA; the "
     "dense tuner's prefill bands (perplexity at context 2048 on Qwen3.8-27B shapes), 64x64 "
     "against 128x64 below that grid: 4080 SUPER at 1024 rows (16 CTAs at T = 128, 32 at 256) "
     "1.33-1.37 for Q4_K, Q5_K and IQ4_XS; 5090 at 5120 rows and T = 128 (80 CTAs) 1.15-1.36, at "
     "1024 rows and T = 128..512 (16..64 CTAs) 1.14-1.35 for Q4_K and Q5_K, 1.07 for Q6_K; 5060 "
     "Ti at 1024 rows and T = 128 1.19-1.50; the evidence quoted is for grids below the line, "
     "the line (half the SMs) is not separately measured",
     ""},
    {"heuristic_w4a4: widest tile that fills the SMs", "generic_gemm.cpp",
     "if (grid(128, 128) >= sms) return 3; if (grid(64, 128) >= sms) return 2; if (grid(32, 128) "
     ">= sms) return 1;",
     0, kUnknown, kUnknown,
     "outside the prefill band, the widest W4A4 tile that still gives every SM a CTA, else the "
     "narrow ones; the comment records no board or margin for the ladder (for the prefill band "
     "it records the tuner's winner on every recorded NVFP4 problem but one, within noise)",
     ""},
    {"invariant_rmma: token tiles", "generic_gemm.cpp", "const int n_tiles = tokens <= 8 ? 1 : 4;",
     8, kUnknown, kUnknown,
     "eight tokens per CTA through T = 8, where decode batches live; 32 above, so a prefill chunk "
     "reads each weight a quarter as often (a design choice; no margin recorded)",
     ""},
    {"kMinChunkWeightRows", "generic_volta_cutlass.cu",
     "constexpr std::int32_t kMinChunkWeightRows = 256;", 256, "gv100:80sm", "V100 (80 SMs)",
     "the fewest weight rows a decode chunk holds: a fixed 4096 rows split the 27B gate_up "
     "(34816 x 5120) into nine GEMMs of 3.2 CTA waves each, one chunk of a third of it runs 10.8 "
     "waves; the bound itself has no margin recorded",
     ""},
    // --- Rules by a function's shape ---
    {"tiled_speed_pays: tile", "generic_gemm.cpp", "speed_waves(n, tokens, 128, 128) >= kSpeedFill",
     0, "every board measured, not listed", "unknown",
     "Q4_K n = 1024 at T 64..128: the tiled kernel 5..7x on every board below the fill", ""},
    {"staged_speed_route: wave thresholds", "generic_gemm.cpp",
     "for (const Pick p : {Pick{256, 128, 2, 2.0}, Pick{128, 64, 2, 1.0}, Pick{64, 64, 3, 0.5}}) {",
     0,
     "unknown (the comment says seven boards without naming them; the register-MMA rule beside "
     "it names the seven of kSpeedFitBoards)",
     "unknown (see kSpeedFitCards for the register-MMA rule's seven)",
     "FITTED: BF16 timings on seven boards, with the register-MMA rule: 1.116 of the best "
     "(geomean) against 1.15..1.36 for the largest tile that filled the SMs and 1.33..1.64 for "
     "the old gate (from 256 tokens, the first tile): a 256x128 tile from two waves of the SMs, "
     "128x64 from one, 64x64 from half a wave; a wider tile than the call's grid leaves SMs idle "
     "(n = 1024 at T = 256: the 128x128 tile 2.5x the register MMA, n = 48: 6x)",
     ""},
    {"nvfp4_tiled_route: two token tiles and a wave", "generic_gemm.cpp",
     "if (tokens < 2 * s.block_m || ctas < sm_count()) return std::nullopt;", 0, "gb202:170sm",
     "RTX 5090",
     "for calls of two token tiles or more whose tiles give every SM at least one CTA; against the "
     "invariant register MMA (WarpsK 8) on an RTX 5090: -8..-40 % from there on the Muse Glimmer "
     "30B MLP and a synthetic [4096, 4096]; below one wave, or at one token tile, it loses (up to "
     "5x at T = 32)",
     ""},
    {"auto_route: NVFP4 half the L2", "generic_gemm.cpp",
     "if (2 * weight_bytes <= device_class().l2_bytes) {", 0, "ad102:128sm", "RTX 4090",
     "an NVFP4 weight that fits half the L2 takes the invariant profile's register MMA: RTX 4090, "
     "Gemma 4 26B shapes 1024..8192 x 2112..8192 at T 17..128, where the 128x128 prefill tile ran "
     "2-4x slower than the speed bands; a larger weight takes the prefill profile's route (31B "
     "16384x5376, 43008x5376, 5376x21504: 0.3-0.8x the speed bands from 32 tokens); measured on "
     "one board, no weight size between the two sets is recorded, so the half-L2 line itself has "
     "no measurement behind it",
     ""},
    {"kR8CarryDistance", "generic_gemm.cpp", "constexpr double kR8CarryDistance = 0.5;", 0.5,
     "ad102:128sm, ad103:80sm, ga102:82sm", "RTX 4090, RTX 4080 SUPER, RTX 3090",
     "the 4090's 80 entries on the 4080 SUPER (distance 1.2; 3 sweeps) won 41, lost 30 (to -22 %), "
     "+0.1 % in the mean; a 4090 D or a 3090 Ti (0.17, 0.14) take their tables",
     ""},
    {"kSpreadMeasured", "generic_gemm.cpp",
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
    {"spread_smem: thin calls", "generic_gemm.cpp",
     "epilogue != Epilogue::Store || args.tokens > 8 ||", 8, "gb202:170sm", "RTX 5090",
     "the eight-warp decode CTA (w1x8) spread over every SM: Store epilogue and at most eight "
     "tokens; measured at T = 1 only, SwiGLU loses (two CTAs per SM already: 116 registers); no "
     "record for the eight-token edge itself",
     ""},
    {"spread_smem: two CTAs per SM", "generic_gemm.cpp",
     "if (ctas > 2 * static_cast<std::int64_t>(device.sms) || pad > kStaticSharedBytesPerBlock)", 0,
     "gb202:170sm", "RTX 5090",
     "a call of at most two CTAs per SM: at 416 CTAs and more spreading loses (6656x4096 NVFP4 "
     "9.4 -> 12.6 us on the RTX 5090); the two-per-SM line is the eight-warp CTA's residency "
     "(three to an SM) less one, not a measured edge",
     ""},
    // --- w4a8_ws_wins and its neighbour ---
    {"w4a8_ws_wins: ampere", "generic_gemm.cpp", "if (ampere) return true;", 0, "ga102:82sm",
     "RTX 3090",
     "no format lost; one 16K prompt, D WS WS D: Muse 30B Q4_K_M 1278 / 1273 -> 1398 / 1399 t/s, "
     "Qwen3.8 27B UD-Q4_K_XL 1290 / 1289 -> 1314, Gemma 4 31B 1071 -> 1081; measured on the "
     "RTX 3090 only, taken for sm_80 and sm_86 alike",
     ""},
    {"w4a8_ws_wins: iq3xxs and iq2s", "generic_gemm.cpp",
     "case QType::GGML_IQ2_S: return blackwell || ada;", 0, kUnknown, kUnknown,
     "Blackwell and Ada take the producer-warp tile for IQ3_XXS and IQ2_S; the record at the rule "
     "names no board or margin for this line (the nearby ISTA comment says the producer-warp "
     "formats were even against ninfer-all's MMQ on an RTX 5060 Ti: IQ3_XXS 12.0 / 11.8 s, IQ2_S "
     "3.96 / 3.67 s)",
     ""},
    {"w4a8_ws_wins: q4_0 on ada", "generic_gemm.cpp", "return ada && p.n >= 5376 && p.k >= 5376;",
     5376, "ad102:128sm", "RTX 4090",
     "the smallest measured win; nothing between 2816 and 5376 measured, nor any sm_89 card with "
     "less L2 than the 4090's 72 MB (Gemma 4 31B QAT prefill 1914 -> 2149 t/s; the 26B QAT keeps "
     "64x64: 0.357 -> 0.383 s a prompt)",
     ""},
    {"w4a8_ws_wins: q6k blackwell only", "generic_gemm.cpp",
     "case QType::GGML_Q6_K: return blackwell;", 0, "gb203:84sm, gb202:170sm", "RTX 5080, RTX 5090",
     "Q6_K on sm_89 lost 2.2 % (RTX 4090, Muse 30B Q4_K_M, 16K prompt: 2935 -> 2997 t/s without "
     "the tile); the Q6_K Blackwell margin is in the micro-bench comment at the rule, not "
     "restated here",
     ""},
    {"w4a8_ws_wins: q4k blackwell only", "generic_gemm.cpp",
     "case QType::GGML_Q4_K: return blackwell;", 0, "gb203:84sm, gb202:170sm", "RTX 5080, RTX 5090",
     "the Q4_K Blackwell margin is in the micro-bench comment at the rule, not restated here; "
     "Q4_K on Ada takes the wide tile instead (w4a8_wide_q4k)",
     ""},
    {"w4a8_wide_q4k: grid bound", "generic_gemm.cpp",
     "4 * ctas >= 3 * static_cast<std::int64_t>(d.sms)", 0, "ad102:128sm", "RTX 4090",
     "a grid of three quarters of the SMs or more (4 * ctas >= 3 * SMs); Muse 30B Q4_K_M "
     "micro-bench against the classic 128x64 at T = 1024 / 512 / 200: 0.86-0.98 where the grid "
     "fills three quarters of the SMs, 1.65-1.71 under it (64 CTAs)",
     ""},
    // --- rules_r8's thresholds ---
    {"rules_r8: verify branch from two tokens", "generic_gemm.cpp",
     "if (tokens >= 2) return iq2 &&", 2, kUnknown, kUnknown,
     "calls of two tokens or more (the verify widths) take this branch and none of the "
     "architecture rules below it; T = 1 (decode) takes the rules; the record names no board or "
     "margin for the two-token edge",
     ""},
    {"rules_r8: iq2 two rows", "generic_gemm.cpp",
     "iq2 && (tokens >= 4 || p.n >= 34816) ? r8_index(4, 2) : r8_index(4, 1);", 0, kUnknown,
     kUnknown, "within 3 % of the best schedule elsewhere (27B/31B/35B verify shapes)", ""},
    {"rules_r8: turing eight warps", "generic_gemm.cpp",
     "if (decode_heavy && p.k >= 5120) return r8_index(8, 2);", 5120, "tu102:68sm", "RTX 2080 Ti",
     "ninferno_r8_schedule_bench on the IQ3 16 GB artifacts' 63 problems: the picks +6.25 % above "
     "the best schedules in sum (IQ3_XXS 17408x5120 110.2 -> 84.5 us, IQ3_S 131.3 -> 114.2, Q3_K "
     "112.9 -> 89.0)",
     ""},
    {"rules_r8: turing q6k two rows", "generic_gemm.cpp",
     "p.epilogue != Epilogue::SwiGlu && (p.n >= 8192 || p.k >= 4096)) return r8_index(4, 2);", 0,
     "tu102:68sm", "RTX 2080 Ti",
     "N >= 8192 or K >= 4096: the 248320x2048 lm_head 807 -> 723 us, 2816x8192 w8r2 42.3 -> w4r2 "
     "40.3",
     ""},
    {"rules_r8: volta one slot", "generic_gemm.cpp",
     "if (p.qtype == QType::GGML_Q6_K && p.epilogue != Epilogue::SwiGlu && r8_one_slot_build() && "
     "!device_class().features.mma_tensor) return r8_index(4, 1);",
     0, "gv100:80sm", "V100 (GV100)",
     "4096x12288 add w4r1 88.8 against w4r2 94.7 us, 8192x4096 store 57.9 against 63.3 (ten other "
     "kernels between launches)",
     ""},
    {"rules_r8: q3k four warps", "generic_gemm.cpp",
     "if (p.qtype == QType::GGML_Q3_K && p.k >= 8192) return r8_index(4, 1);", 8192,
     "gb206:36sm, ad103:80sm", "RTX 5060 Ti, RTX 4080 SUPER, RTX 4070 Ti SUPER",
     "Qwen3.8's 5120x17408 add is 7-10 % faster so (113.2 -> 103.0, 64.8 -> 60.3, 73.0 -> 66.3 "
     "us); the RTX 4070 Ti SUPER's class id is not named in the comment",
     ""},
    {"rules_r8: wide K", "generic_gemm.cpp",
     "if (p.k >= 8192 || (iq2 && p.n >= 12288)) return r8_index(8, 2);", 0, "gb203:84sm",
     "RTX 5080",
     "K >= 8192, or N >= 12288 for IQ2: Qwen3.5-4B/9B shapes: IQ3_XXS 4096x12288 32.1 -> 27.9 us, "
     "IQ2_S 12288x4096 28.3 -> 24.9 us; the ggml blocks and K-quants follow the K rule within 2 % "
     "of their best schedule",
     ""},
    {"rules_r8: blackwell branch gate", "generic_gemm.cpp",
     "if (p.epilogue != Epilogue::SwiGlu && p.k >= 5120 && "
     "consumer_blackwell_major(device_class().major)) {",
     5120, "gb206:36sm, gb203:84sm, gb202:170sm (the boards of the rules inside it)",
     "RTX 5060 Ti, RTX 5080, RTX 5090",
     "the gate of the four Blackwell rules below: every shape their comments cite has 5120 or "
     "5376 as its K (10240x5120, 17408x5120, 16384x5120, 8192x5376, 43008x5376); no measurement "
     "places the K = 5120 edge itself, and none below it on Blackwell is recorded",
     ""},
    {"rules_r8: blackwell iq2 from N 5120", "generic_gemm.cpp",
     "if (iq2 && p.n >= 5120) return r8_index(8, 2);", 5120, "gb206:36sm, gb203:84sm, gb202:170sm",
     "RTX 5060 Ti, RTX 5080, RTX 5090",
     "IQ2_S 10240x5120 w4r2 29.9 -> w8r2 20.1 us (still gains at 170 SMs: the 5090); IQ2_XXS "
     "10240x5120 43.5 -> 40.0, IQ2_XS "
     "17408x5120 83.6 -> 79.4 (5060 Ti)",
     ""},
    {"rules_r8: blackwell sm count", "generic_gemm.cpp", "if (device_class().sms <= 96) {", 96,
     "gb203:84sm, gb202:170sm", "RTX 5080, RTX 5090",
     "an SM-count rule whose edge lies between the measured boards (84 and 170 SMs; no board "
     "between; docs/maintainer/deterministic-decisions.md lists it among the rules resting on few "
     "boards): 10240x5120 IQ3_S w4r1 18.4 us against w8r2 20.3 at 170, IQ3_XXS 16.0 against 18.0",
     ""},
    {"rules_r8: blackwell iq3s", "generic_gemm.cpp",
     "if (p.qtype == QType::GGML_IQ3_S && p.n >= 5120) return r8_index(8, 2);", 5120,
     "gb203:84sm, gb206:36sm", "RTX 5080, RTX 5060 Ti",
     "IQ3_S 8192x5376 28.5 -> 27.5 us (5080); 16384x5120 w4r1 100.6 -> 94.5 (5060 Ti)", ""},
    {"rules_r8: blackwell iq3xxs", "generic_gemm.cpp",
     "if (p.qtype == QType::GGML_IQ3_XXS && p.n >= 5120) return r8_index(4, 2);", 5120,
     "gb203:84sm, gb206:36sm", "RTX 5080, RTX 5060 Ti",
     "IQ3_XXS 43008x5376 112.9 -> 106.6 us (5080); 16384x5120 92.8 -> 82.5 (5060 Ti)", ""},
    {"rules_r8: iq2 two rows from N 4096", "generic_gemm.cpp",
     "if (iq2 && p.n >= 4096) return r8_index(4, 2);", 4096, "gb203:84sm", "RTX 5080",
     "IQ2_XS 9216x2560 14.1 -> 12.8 us", ""},
});

} // namespace ninferno::ops::detail::generic::route_rules
