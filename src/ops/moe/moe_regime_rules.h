#pragma once

// The generic MoE op's R8 regime rules, declared with their evidence (#616 step 5): how many tokens
// one R8 call of an expert bank may take before the register MMA runs instead (r8_token_limit,
// moe.cpp). No law prices R8's SIMT decode against the register MMA yet (#616: the MoE law campaign
// refuted five), so the answer is a table: a rule per generation of the generation table
// (core/generation.h) and a few per bank, each a record of what it decides, the predicate it is
// asked by, the cards it was measured on, the margin the record gives and where that is written.
// moe.cpp reads these records; it keeps no limit of its own.
//
// The labels (Label below) say what a rule rests on: Measured on two or more cards (the two-card
// principle of docs/maintainer/deterministic-decisions.md), SingleCard (the lab has one card of
// the generation), or Unmeasured (the conservative answer, with no card). A generation the table
// does not know (generation_known false: a lent row carries no evidence, #300) takes the Unknown
// rule: conservative, and labelled so. Conservative is the narrow limit, kR8NarrowTokens: up to
// four tokens R8 measured at or ahead of the register MMA on every swept board and format but
// Q6_K on two boards (the RTX 2080 Ti at four tokens, the cell recorded below; the RTX 3090 at one
// to three tokens, +1..+7 %).
// A widened limit (above kR8NarrowTokens) must be Measured, which tests/ops/test_moe_regime_rules
// holds.
//
// A Recorded rule is a measured cell the table does not act on (Disposition::Recorded): the
// measurement and why it is kept, for the regret collector and the next sweep, never read by the
// lookup.
//
// The law that was put to the Ampere sweep and refuted (DECISIONS 82): R8 decodes each weight on
// the SM's integer lanes (dp4a), the register MMA on its tensor cores, so R8's margin above four
// tokens was predicted to follow the integer lanes an SM has per DRAM byte it must consume a cycle
// (lanes / dram_bytes_per_sm_cycle: RTX 2080 Ti 10.9, RTX 4060 Ti 19.2, RTX 4090 20.5, RTX 5070 Ti
// 24.5, RTX 5090 29.2, which orders those five boards' T = 5..8 margins). The RTX 3090 (9.5) and
// 3060 (8.8) sit below the RTX 2080 Ti, and the law predicted Q6_K +40..+60 % and Q3_K +15..+30 %
// there; they measured +3..+11 % and -3..-8 %. No rule reads it; the limits stay a table.

#include "core/generation.h"
#include "core/weight.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/moe/moe_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ninferno::ops::detail::moe::regime {

// What a rule decides.
//   TokenLimit  the widest R8 call of a generation (GenerationRow::name), every bank alike
//   BankLimit   a bank whose format or layout keeps R8 narrower, on every generation; the first
//               Applied bank rule whose predicate holds answers (a rule that keeps the generation's
//               limit stops the search)
//   Cell        one measured (generation, format, width) cell, Recorded only
enum class Regime : std::uint8_t { TokenLimit, BankLimit, Cell };

// What a rule rests on: two or more cards, one card, or none (the conservative answer).
enum class Label : std::uint8_t { Measured, SingleCard, Unmeasured };

enum class Disposition : std::uint8_t { Applied, Recorded };

[[nodiscard]] constexpr std::string_view label_name(Label label) {
    switch (label) {
    case Label::Measured:
        return "measured";
    case Label::SingleCard:
        return "single_card";
    case Label::Unmeasured:
        return "unmeasured";
    }
    return "";
}

// A card a rule was measured on, by its PCI id in the board registry (core/board_registry.h) and
// its product name; pci 0 is an empty slot.
struct Card {
    std::uint32_t pci = 0;
    std::string_view name;
};

// What a bank rule's predicate reads: the bank's format and stored layout, and the board's DRAM
// bytes per SM cycle (generic::dram_bytes_per_sm_cycle; 0 when unknown, and 0 where the caller
// did not need it: only ggml_frag_v1 banks are asked about it).
struct BankFacts {
    QType format                   = QType::BF16;
    QuantLayout layout             = QuantLayout::RowSplit;
    double dram_bytes_per_sm_cycle = 0;
};

struct RegimeRule {
    std::string_view name;
    Regime regime;
    // TokenLimit and Cell: the generation (GenerationRow::name, or kUnknownGeneration); BankLimit:
    // empty (every generation).
    std::string_view generation;
    // BankLimit and Cell: whether the rule is about this bank; TokenLimit: null.
    bool (*applies)(const BankFacts&);
    // The widest R8 call the rule admits. A BankLimit rule's limit caps the generation's
    // (kR8Tokens: it keeps it); a Cell's is what the cell alone would take.
    std::int32_t limit;
    Label label;
    std::array<Card, 4> cards;
    // The measured margin the record gives, or why there is none.
    std::string_view tolerance;
    // Where the measurement is written: the bench and its shape, the tree, the record.
    std::string_view source;
    Disposition disposition = Disposition::Applied;
};

inline constexpr std::string_view kUnknownGeneration = "unknown";

inline constexpr Card kRtx2080Ti{0x1E07, "RTX 2080 Ti"};
inline constexpr Card kRtx3060{0x2503, "RTX 3060"};
inline constexpr Card kRtx3090{0x2204, "RTX 3090"};
inline constexpr Card kRtx4060Ti{0x2805, "RTX 4060 Ti"};
inline constexpr Card kRtx4090{0x2684, "RTX 4090"};
inline constexpr Card kRtx5070Ti{0x2C05, "RTX 5070 Ti"};
inline constexpr Card kRtx5090{0x2B85, "RTX 5090"};

// The sweep the generation rules rest on: moe_bench (bench/ops/moe_bench.cu), a Qwen3.6-35B-like
// layer (H 2048, F 512, E 128, k 8, published layout), the gate/up stage forced to R8 w4r1 against
// the A16 register-MMA plan (--a8 0 --plans auto) at T = 1..8, Q3_K, Q4_K, Q5_K, Q6_K and IQ3_XXS
// banks (r8t8 on master 55333b90 and the eight-token branch be7fa880, 2026-11-03).
inline constexpr std::string_view kR8T8Sweep =
    "moe_bench H 2048 F 512 E 128 k 8, published layout, R8 w4r1 gate/up against the A16 "
    "register MMA, T = 1..8, Q3_K/Q4_K/Q5_K/Q6_K/IQ3_XXS (2026-11-03, 55333b90 and be7fa880; "
    "commit 9aa7e431)";
// The same command on the two Ampere cards (13.11, #616 step 5): the tree of 2b37dfa7f with the
// ga10x rule's limit at kR8Tokens so that the forced R8 plan runs at T = 5..8, two passes each.
inline constexpr std::string_view kR8SwSweep =
    "moe_bench H 2048 F 512 E 128 k 8, published layout, R8 w4r1 gate/up against the A16 "
    "register MMA, T = 1..8, Q3_K/Q4_K/Q5_K/Q6_K/IQ3_XXS, two passes (2026-11-13, 2b37dfa7f with "
    "the ga10x limit at eight, DECISIONS 82)";
inline constexpr std::string_view kConservative =
    "none: no card of the generation was swept; conservative, the narrow limit (kR8NarrowTokens), "
    "where R8 measured at or ahead of the register MMA on the swept boards but two Q6_K cells";

inline constexpr auto kRegimeRules = std::to_array<RegimeRule>({
    // --- The widest R8 call per generation (r8_token_limit()) ---
    {"r8_token_limit: volta",
     Regime::TokenLimit,
     "volta",
     nullptr,
     kR8NarrowTokens,
     Label::Unmeasured,
     {},
     "R8 at T = 5..8 not timed; what was timed is the split: a call wider than the limit runs as "
     "calls of at most four (moe_split_tokens): V100, Gemma 4 26B experts, R8 at T = 4 214 us "
     "against the emulated register MMA's 1535 at T = 5",
     "moe.cpp moe_split_tokens"},
    {"r8_token_limit: turing",
     Regime::TokenLimit,
     "turing",
     nullptr,
     kR8NarrowTokens,
     Label::SingleCard,
     {kRtx2080Ti},
     "R8 against the register MMA at T = 5..8: Q6_K +38..+45 %, Q3_K +15..+19 % (Q4_K -3..-6 %, "
     "Q5_K -17..-21 %, IQ3_XXS -9..-11 %); one Turing card in the lab (GTX 16 is not supported)",
     kR8T8Sweep},
    {"r8_token_limit: ga100",
     Regime::TokenLimit,
     "ga100",
     nullptr,
     kR8NarrowTokens,
     Label::Unmeasured,
     {},
     kConservative,
     "no sweep"},
    // The two cards disagree above four tokens on Q4_K and Q5_K and both lose Q6_K there: a wider
    // limit needs both, so the generation keeps four.
    {"r8_token_limit: ga10x",
     Regime::TokenLimit,
     "ga10x",
     nullptr,
     kR8NarrowTokens,
     Label::Measured,
     {kRtx3090, kRtx3060},
     "R8 against the register MMA at T = 5..8: RTX 3090 Q6_K +3..+6 %, Q3_K -5..-8 %, Q4_K "
     "-3..-5 %, Q5_K -5..-9 %, IQ3_XXS -8..-10 %; RTX 3060 Q6_K +8..+11 %, Q3_K -3..-7 %, Q4_K "
     "+1..+8 %, Q5_K -1..+6 %, IQ3_XXS -6..-10 %; at T = 1..4 R8 ahead by 11..28 % on both but "
     "Q6_K (RTX 3090 +1..+7 % at T = 1..3, -1..-2 % at 4; RTX 3060 -2..+1 %); the two passes "
     "within 2 %, the RTX 3090's medians within 3 us of nsys kernel sums",
     kR8SwSweep},
    {"r8_token_limit: ada",
     Regime::TokenLimit,
     "ada",
     nullptr,
     kR8Tokens,
     Label::Measured,
     {kRtx4090, kRtx4060Ti},
     "R8 against the register MMA at T = 5..8: RTX 4090 -3..-7 % (Q6_K level), RTX 4060 Ti "
     "-0.4..-6 %",
     kR8T8Sweep},
    {"r8_token_limit: hopper",
     Regime::TokenLimit,
     "hopper",
     nullptr,
     kR8NarrowTokens,
     Label::Unmeasured,
     {},
     kConservative,
     "no sweep"},
    {"r8_token_limit: gb100",
     Regime::TokenLimit,
     "gb100",
     nullptr,
     kR8NarrowTokens,
     Label::Unmeasured,
     {},
     kConservative,
     "no sweep"},
    {"r8_token_limit: thor",
     Regime::TokenLimit,
     "thor",
     nullptr,
     kR8NarrowTokens,
     Label::Unmeasured,
     {},
     kConservative,
     "no sweep"},
    // The row spans sm_120 and sm_121 (GB10): the two cards are sm_120; sm_121 takes the row.
    {"r8_token_limit: gb20x",
     Regime::TokenLimit,
     "gb20x",
     nullptr,
     kR8Tokens,
     Label::Measured,
     {kRtx5090, kRtx5070Ti},
     "R8 against the register MMA at T = 5..8, every K-quant and IQ3_XXS: RTX 5070 Ti -7..-14 %, "
     "RTX 5090 -5..-15 %; end to end C8 Qwen3.6-35B 180.8 -> 189.0 tok/s, Gemma 4 26B 161.0 -> "
     "167.4 (RTX 5090)",
     kR8T8Sweep},
    {"r8_token_limit: unknown generation",
     Regime::TokenLimit,
     kUnknownGeneration,
     nullptr,
     kR8NarrowTokens,
     Label::Unmeasured,
     {},
     "none: a generation the table does not know (a lent row carries no evidence, #300); "
     "conservative, the narrow limit",
     "core/generation.h generation_known"},

    // --- Banks that keep R8 narrower (r8_token_limit(bank)), in the order they are asked ---
    {"r8_fp4_elements",
     Regime::BankLimit,
     {},
     +[](const BankFacts& b) { return b.format == QType::MXFP4 || b.format == QType::NVFP4; },
     kR8NarrowTokens,
     Label::Measured,
     {kRtx5070Ti, kRtx5090, kRtx4060Ti},
     "gpt-oss's shape (H 2880, F 2880, E 32, k 4), MXFP4, R8 against the register MMA at T = 5 / "
     "6 / 8: RTX 5070 Ti +11.5 / +12.3 / +13.9 %, RTX 5090 +4.9 / +5.6 / +11.2 %; end to end "
     "gpt-oss-20b MXFP4 C8 RTX 5070 Ti 109.4 -> 97.1 tok/s, RTX 4060 Ti 39.5 -> 38.7 at eight "
     "tokens; NVFP4 decodes the same elements (not timed)",
     "moe_bench --plans auto; moe.cpp r8_token_limit(bank)"},
    {"r8_frag_q4k_wide",
     Regime::BankLimit,
     {},
     +[](const BankFacts& b) {
         return b.layout == QuantLayout::GgmlFrag && b.format == QType::GGML_Q4_K;
     },
     kR8Tokens,
     Label::Measured,
     {kRtx4090, kRtx4060Ti, kRtx5090, kRtx5070Ti},
     "frag Q4_K, R8 against the register MMA at T = 5..8: RTX 4090 -5..-7 %, 4060 Ti -1..-3 %, "
     "5090 -10..-13 %, 5070 Ti -11..-13 %: the super-block takes R8 on every board",
     "moe_bench Qwen3.6-35B's shape (H 2048, F 512, E 256, k 8), frag"},
    {"r8_frag_q4_0_narrow",
     Regime::BankLimit,
     {},
     +[](const BankFacts& b) {
         return b.layout == QuantLayout::GgmlFrag && b.format == QType::GGML_Q4_0;
     },
     kR8NarrowTokens,
     Label::Measured,
     {kRtx5090, kRtx5070Ti, kRtx4090, kRtx4060Ti},
     "frag Q4_0, R8 against the register MMA at T = 5..8: RTX 5090 +2..+6 %, 5070 Ti +11..+12 % "
     "(loses on both Blackwell boards), RTX 4090 -2 %, 4060 Ti 0 (ties on Ada)",
     "moe_bench Qwen3.6-35B's shape (H 2048, F 512, E 256, k 8), frag"},
    {"r8_frag_no_roofline",
     Regime::BankLimit,
     {},
     +[](const BankFacts& b) {
         return b.layout == QuantLayout::GgmlFrag && b.dram_bytes_per_sm_cycle <= 0;
     },
     kR8NarrowTokens,
     Label::Unmeasured,
     {},
     "none: a board without a roofline (no DRAM bytes per SM cycle) cannot be placed against the "
     "rule below; conservative, the narrow limit",
     "moe.cpp r8_token_limit(bank)"},
    // The register MMA's SwiGLU tiles take their latency variant from kRmmaLatencySwiGluBytes
    // (generic_gemm_internal.h) and then hold the DRAM bus themselves; below it R8 has SM cycles
    // to spare per DRAM byte. The edge is the register MMA's rule, not one placed for R8.
    {"r8_frag_dram_bound",
     Regime::BankLimit,
     {},
     +[](const BankFacts& b) {
         return b.layout == QuantLayout::GgmlFrag &&
                b.dram_bytes_per_sm_cycle >= generic::kRmmaLatencySwiGluBytes;
     },
     kR8NarrowTokens,
     Label::Measured,
     {kRtx4090, kRtx4060Ti, kRtx5090, kRtx5070Ti},
     "frag IQ3_XXS / IQ2_S, R8 against the register MMA at T = 5..8 by DRAM bytes per SM cycle: "
     "RTX 4090 (3.1) -9..-10 / -6..-8 %, 4060 Ti (3.3) -3..-6 / -2..-4 %, 5090 (4.4) -4..-8 / "
     "-4..-8 %, 5070 Ti (5.2, narrow) -1..-4 / -1..-6 %, which turned end to end: C8 UD-Q3_K_S "
     "(frag IQ3_XXS) RTX 5070 Ti 104.9 -> 102.8 tok/s at eight tokens against four, RTX 4090 "
     "129.1 -> 134.0, 4060 Ti 45.2 -> 45.6; the narrowing side rests on the RTX 5070 Ti alone "
     "(the one board at or above the 4.5 line), the other three bound it from below",
     "moe_bench Qwen3.6-35B's shape (H 2048, F 512, E 256, k 8), frag; end to end C8"},

    // --- Measured cells the table does not act on ---
    // The RTX 2080 Ti's Q6_K at four tokens: R8 loses, but the lab has one Turing card, so the
    // limit is not narrowed for it (two cards of a generation move a route); the cause is under
    // test in #616 step 6 (the planes' layout locality), which would remove the cell.
    {"turing_q6k_t4",
     Regime::Cell,
     "turing",
     +[](const BankFacts& b) { return b.format == QType::GGML_Q6_K; },
     kR8NarrowTokens - 1,
     Label::SingleCard,
     {kRtx2080Ti},
     "Q6_K gate/up, R8 against the register MMA: T = 4 +25 % in two runs (260.3 against 208.7 us, "
     "261.8 against 208.7), T = 3 +2 / +19 %, T = 1 -9 / +10 % (the board's two clock states "
     "between runs)",
     kR8T8Sweep,
     Disposition::Recorded},
});

// The rule that answers a generation's limit: its own TokenLimit rule, or the Unknown one for a
// generation the table does not know.
[[nodiscard]] constexpr const RegimeRule& generation_rule(int major, int minor) {
    const std::string_view generation = generation_known(major, minor)
                                            ? std::string_view(generation_row(major, minor).name)
                                            : kUnknownGeneration;
    const RegimeRule* unknown         = nullptr;
    for (const RegimeRule& rule : kRegimeRules) {
        if (rule.regime != Regime::TokenLimit) continue;
        if (rule.generation == generation) return rule;
        if (rule.generation == kUnknownGeneration) unknown = &rule;
    }
    return *unknown;
}

// The first Applied bank rule whose predicate holds, or null.
[[nodiscard]] constexpr const RegimeRule* bank_rule(const BankFacts& bank) {
    for (const RegimeRule& rule : kRegimeRules) {
        if (rule.regime == Regime::BankLimit && rule.disposition == Disposition::Applied &&
            rule.applies(bank)) {
            return &rule;
        }
    }
    return nullptr;
}

// The widest R8 call over a bank on a generation, and the rule that set it: the bank rule where
// it narrows the generation's limit, else the generation's rule (whose label the answer carries).
struct R8LimitAnswer {
    std::int32_t limit           = kR8NarrowTokens;
    const RegimeRule* rule       = nullptr; // the rule that set the limit
    const RegimeRule* generation = nullptr; // the generation's rule
};

[[nodiscard]] constexpr R8LimitAnswer r8_limit_answer(int major, int minor, const BankFacts& bank) {
    const RegimeRule& generation = generation_rule(major, minor);
    const RegimeRule* narrowing  = bank_rule(bank);
    if (narrowing != nullptr && narrowing->limit < generation.limit) {
        return {narrowing->limit, narrowing, &generation};
    }
    return {generation.limit, &generation, &generation};
}

// The same for the current device and a bank (moe.cpp; r8_token_limit(bank) is its limit).
[[nodiscard]] R8LimitAnswer r8_limit_answer(const Weight& bank);

} // namespace ninferno::ops::detail::moe::regime
