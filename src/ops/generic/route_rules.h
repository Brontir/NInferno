#pragma once

// The literal rules of the generic path's route selection (generic_gemm.h, generic_gemm.cpp): the
// thresholds a rule compares a call's shape, width or card against, each a number someone measured
// on some boards and wrote into the code (#614, PR-F4). This header declares what each rule IS
// (its literal) and what it rests on (the boards and cards it was measured on, the margin the
// record gives, where it is written); it changes no behaviour. The code keeps its own literals:
// test_route_rules reads the sources and holds every descriptor to them (a named constant's value,
// a rule's literal text inside the file), and every constant of the widths block of generic_gemm.h
// to a descriptor, so a rule cannot change, move or appear without its record.
//
// Fields:
//   name       the rule as the code names it (`symbol` or a short name for a literal inside one);
//   file       the source file that holds the literal, under src/ops/generic/;
//   literal    the text of the literal as written (a constant's definition, or the comparison);
//   value      its number where it is one (0 where the literal is a condition, not a number);
//   boards     the board classes it was measured on (core/board_registry ids: chip:SMs), "unknown"
//              where the comments and the record name none;
//   cards      the same as the products' names;
//   tolerance  the measured margin the record gives ("none recorded" where it gives none): what the
//              rule was held to, or what moving the literal was seen to cost;
//   kernel_hash  the identity of the kernel the rule was measured against: EMPTY for every rule.
//              The repository records one sha256 over all device sources per kernel-facts file
//              (kernel_facts.h Facts::source, tools/kernel_facts), not per kernel, and no rule
//              records which tree it was timed on; a rule re-measured later should fill this.
// Where the record is silent a field says "unknown": nothing here is a guess. The source of a
// number is the comment at its use (quoted in the tolerance where it states a margin), DECISIONS.md
// and docs/maintainer.

#include <array>
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
};

inline constexpr std::string_view kNotRecorded = "none recorded";
inline constexpr std::string_view kUnknown     = "unknown";

inline constexpr auto kRouteRules = std::to_array<RouteRule>({
    // --- The vector kernel's verify widths ---
    {"kVecRuleMaxTokens", "generic_gemm.cpp", "constexpr std::int32_t kVecRuleMaxTokens = 4;", 4,
     "tu102:68sm, ga100, ga102, gb202:170sm, ad102:128sm, gv100",
     "RTX 2080 Ti, A100, RTX 3090, RTX 5090, RTX 4090, V100",
     "the vector kernel was within 4 % of the best route at T = 2..4 and behind it at T = 8 on all "
     "eight cards of the dense tuner (A100 x1.5..1.6, RTX 3090 x1.4..1.5, RTX 5090 x1.1, RTX 4090 "
     "x1.03); RTX 2080 Ti: lost decode by 10 % (9B Q4_K_M C1 88 -> 79 tok/s), won C4 by 10 %; "
     "Volta takes every width it reads instead",
     ""},
    // --- The widths where the speed profile's heuristic choices change (generic_gemm.h) ---
    {"kRmmaMidMinTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kRmmaMidMinTokens     = 33;", 33,
     "ga102, ad102:128sm, gb202:170sm, tu102:68sm, ga100 (A10 and L4: class ids not named)",
     "RTX 3090, RTX 4090, RTX 5090, RTX 2080 Ti, A10, L4, A100",
     "one rule for every board, fitted on six boards and scored on the seventh: 1.116 of the best "
     "(geomean) against 1.219 for the bands it replaced; a rule per board gained 1.1 % at most",
     ""},
    {"kStreamMaxTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kStreamMaxTokens      = 512;", 512, kUnknown, kUnknown,
     "none recorded", ""},
    {"kTiledSpeedMinTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kTiledSpeedMinTokens  = 33;", 33, "every board measured, not "
     "listed (Q4_K n = 1024 at T 64..128)", "unknown",
     "the tiled kernel 5..7x the register MMA's at Q4_K n = 1024, T 64..128, on every board", ""},
    {"kStagedSpeedMinTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kStagedSpeedMinTokens = 256;", 256, kUnknown, kUnknown,
     "none recorded", ""},
    {"kSpeedFill", "generic_gemm.h", "inline constexpr double kSpeedFill                  = 1.5;",
     1.5, "ga102, ad102:128sm, gb202:170sm, tu102:68sm, ga100 (A10 and L4: class ids not named)",
     "RTX 3090, RTX 4090, RTX 5090, RTX 2080 Ti, A10, L4, A100",
     "1.0..2.0 cost under 0.5 % (the header's comment); the same board-held-out fit as "
     "kRmmaMidMinTokens",
     ""},
    {"kW4a8WideMinTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kW4a8WideMinTokens    = 128;", 128,
     "ad103:80sm, gb202:170sm, gb206:36sm, ad102:128sm",
     "RTX 4080 SUPER, RTX 5090, RTX 5060 Ti, RTX 4090",
     "below it (the 128x64 tile against 64x64): 1.10-1.56 on IQ2_*/IQ3_*/IQ4_XS (4080 SUPER), Q4_K 1.05-1.09 .. Q6_K "
     "1.17-1.20 (5090), 1.08-1.29 (5060 Ti); in the product 4080 SUPER Qwen3.8-27B IQ3 412.9 -> "
     "472.4 tok/s, 5090 Q4_K_XL 738 -> 757",
     ""},
    {"kPdlPrefetchMaxTokens", "generic_gemm.h",
     "inline constexpr std::int32_t kPdlPrefetchMaxTokens = 16;", 16, kUnknown, kUnknown,
     "none recorded (decode and speculative verify widths)", ""},
    // --- Rules by a function's shape ---
    {"tiled_speed_pays: tile", "generic_gemm.cpp", "speed_waves(n, tokens, 128, 128) >= kSpeedFill",
     128, "every board measured, not listed", "unknown",
     "Q4_K n = 1024 at T 64..128: the tiled kernel 5..7x on every board below the fill", ""},
    {"kR8CarryDistance", "generic_gemm.cpp", "constexpr double kR8CarryDistance = 0.5;", 0.5,
     "ad102:128sm, ad103:80sm, ga102", "RTX 4090, RTX 4080 SUPER, RTX 3090",
     "the 4090's 80 entries on the 4080 SUPER (distance 1.2; 3 sweeps) won 41, lost 30 (to -22 %), "
     "+0.1 % in the mean; a 4090 D or a 3090 Ti (0.17, 0.14) take their tables",
     ""},
    // --- w4a8_ws_wins and its neighbour ---
    {"w4a8_ws_wins: ampere", "generic_gemm.cpp", "if (ampere) return true;", 0, "ga102",
     "RTX 3090",
     "no format lost; one 16K prompt, D WS WS D: Muse 30B Q4_K_M 1278 / 1273 -> 1398 / 1399 t/s, "
     "Qwen3.8 27B UD-Q4_K_XL 1290 / 1289 -> 1314, Gemma 4 31B 1071 -> 1081; measured on the "
     "RTX 3090 only, taken for sm_80 and sm_86 alike",
     ""},
    {"w4a8_ws_wins: q4_0 on ada", "generic_gemm.cpp", "return ada && p.n >= 5376 && p.k >= 5376;",
     5376, "ad102:128sm", "RTX 4090",
     "the smallest measured win; nothing between 2816 and 5376 measured, nor any sm_89 card with "
     "less L2 than the 4090's 72 MB (Gemma 4 31B QAT prefill 1914 -> 2149 t/s; the 26B QAT keeps "
     "64x64: 0.357 -> 0.383 s a prompt)",
     ""},
    {"w4a8_ws_wins: q6k and q4k blackwell only", "generic_gemm.cpp", "return blackwell;", 0,
     "gb203:84sm, gb202:170sm", "RTX 5080, RTX 5090",
     "Q6_K on sm_89 lost 2.2 % (RTX 4090, Muse 30B Q4_K_M, 16K prompt: 2935 -> 2997 t/s without "
     "the tile); the Q4_K and Q6_K Blackwell margins are in the micro-bench comment at the rule, "
     "not restated here",
     ""},
    {"w4a8_wide_q4k: grid bound", "generic_gemm.cpp",
     "4 * ctas >= 3 * static_cast<std::int64_t>(d.sms)", 0.75, "ad102:128sm", "RTX 4090",
     "Muse 30B Q4_K_M micro-bench against the classic 128x64 at T = 1024 / 512 / 200: 0.86-0.98 "
     "where the grid fills three quarters of the SMs, 1.65-1.71 under it (64 CTAs)",
     ""},
    // --- rules_r8's thresholds ---
    {"rules_r8: iq2 two rows", "generic_gemm.cpp", "(tokens >= 4 || p.n >= 34816)", 34816,
     kUnknown, kUnknown, "within 3 % of the best schedule elsewhere (27B/31B/35B verify shapes)",
     ""},
    {"rules_r8: turing eight warps", "generic_gemm.cpp", "decode_heavy && p.k >= 5120", 5120,
     "tu102:68sm", "RTX 2080 Ti",
     "ninferno_r8_schedule_bench on the IQ3 16 GB artifacts' 63 problems: the picks +6.25 % above "
     "the best schedules in sum (IQ3_XXS 17408x5120 110.2 -> 84.5 us, IQ3_S 131.3 -> 114.2, Q3_K "
     "112.9 -> 89.0)",
     ""},
    {"rules_r8: turing q6k two rows", "generic_gemm.cpp", "(p.n >= 8192 || p.k >= 4096)", 8192,
     "tu102:68sm", "RTX 2080 Ti",
     "the 248320x2048 lm_head 807 -> 723 us, 2816x8192 w8r2 42.3 -> w4r2 40.3", ""},
    {"rules_r8: volta one slot", "generic_gemm.cpp", "r8_one_slot_build()", 0, "gv100",
     "V100 (GV100)",
     "4096x12288 add w4r1 88.8 against w4r2 94.7 us, 8192x4096 store 57.9 against 63.3 (ten other "
     "kernels between launches)",
     ""},
    {"rules_r8: q3k four warps", "generic_gemm.cpp", "p.qtype == QType::GGML_Q3_K && p.k >= 8192",
     8192, "gb206:36sm, ad103:80sm", "RTX 5060 Ti, RTX 4080 SUPER, RTX 4070 Ti SUPER",
     "Qwen3.8's 5120x17408 add is 7-10 % faster so (113.2 -> 103.0, 64.8 -> 60.3, 73.0 -> 66.3 "
     "us); the RTX 4070 Ti SUPER's class id is not named in the comment",
     ""},
    {"rules_r8: wide K", "generic_gemm.cpp", "p.k >= 8192 || (iq2 && p.n >= 12288)", 8192,
     "gb203:84sm", "RTX 5080",
     "Qwen3.5-4B/9B shapes: IQ3_XXS 4096x12288 32.1 -> 27.9 us, IQ2_S 12288x4096 28.3 -> 24.9 us; "
     "the ggml blocks and K-quants follow the K rule within 2 % of their best schedule",
     ""},
    {"rules_r8: blackwell iq2 from N 5120", "generic_gemm.cpp", "if (iq2 && p.n >= 5120)", 5120,
     "gb206:36sm, gb203:84sm", "RTX 5060 Ti, RTX 5080",
     "IQ2_S 10240x5120 w4r2 29.9 -> w8r2 20.1 us; IQ2_XXS 10240x5120 43.5 -> 40.0, IQ2_XS "
     "17408x5120 83.6 -> 79.4 (5060 Ti)",
     ""},
    {"rules_r8: blackwell sm count", "generic_gemm.cpp", "device_class().sms <= 96", 96,
     "gb203:84sm, gb202:170sm", "RTX 5080, RTX 5090",
     "an SM-count rule whose edge lies between the measured boards (84 and 170 SMs; no board "
     "between): 10240x5120 IQ3_S w4r1 18.4 us against w8r2 20.3 at 170, IQ3_XXS 16.0 against 18.0",
     ""},
    {"rules_r8: blackwell iq3s", "generic_gemm.cpp", "p.qtype == QType::GGML_IQ3_S && p.n >= 5120",
     5120, "gb203:84sm, gb206:36sm", "RTX 5080, RTX 5060 Ti",
     "IQ3_S 8192x5376 28.5 -> 27.5 us (5080); 16384x5120 w4r1 100.6 -> 94.5 (5060 Ti)", ""},
    {"rules_r8: blackwell iq3xxs", "generic_gemm.cpp",
     "p.qtype == QType::GGML_IQ3_XXS && p.n >= 5120", 5120, "gb203:84sm, gb206:36sm",
     "RTX 5080, RTX 5060 Ti",
     "IQ3_XXS 43008x5376 112.9 -> 106.6 us (5080); 16384x5120 92.8 -> 82.5 (5060 Ti)", ""},
    {"rules_r8: iq2 two rows from N 4096", "generic_gemm.cpp", "if (iq2 && p.n >= 4096)", 4096,
     "gb203:84sm", "RTX 5080", "IQ2_XS 9216x2560 14.1 -> 12.8 us", ""},
});

} // namespace ninferno::ops::detail::generic::route_rules
