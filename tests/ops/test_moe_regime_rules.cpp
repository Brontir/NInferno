// The MoE R8 regime rules (ops/moe/moe_regime_rules.h, #616 step 5), host only, no GPU:
//   - every rule's label matches its cards: Measured names two or more distinct cards, SingleCard
//     one, Unmeasured none; each card is a row of the board registry, and a generation rule's (or
//     a cell's) cards are of its generation;
//   - every rule says what it rests on (tolerance, source) and has a predicate where it needs one;
//   - every generation of the generation table has one TokenLimit rule and the table one Unknown
//     rule; an unmeasured generation and the unknown one take the conservative limit
//     (kR8NarrowTokens), and only a Measured rule widens it;
//   - the answer on substituted boards: r8_token_limit() and r8_token_limit(bank) give what the
//     record says the measured boards take (written here by hand, the independent check: eight on
//     sm_89 and sm_12x, four elsewhere; the bank cases of the comments), with the rule that set it
//     and its label; a generation the table does not know answers four, labelled unmeasured;
//   - a Recorded cell never moves an answer (the RTX 2080 Ti's Q6_K stays at four).
#include "core/board_registry.h"
#include "core/device_class.h"
#include "core/generation.h"
#include "core/weight.h"
#include "ops/moe/moe_internal.h"
#include "ops/moe/moe_regime_rules.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <set>
#include <string>
#include <string_view>

namespace {

using namespace ninferno;
namespace moe    = ninferno::ops::detail::moe;
namespace regime = ninferno::ops::detail::moe::regime;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok && ++failures <= 40) std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

std::string name_of(const regime::RegimeRule& rule) { return std::string(rule.name); }

void check_records() {
    for (const regime::RegimeRule& rule : regime::kRegimeRules) {
        const std::string name = name_of(rule);
        std::set<std::uint32_t> pcis;
        int empty_named = 0;
        for (const regime::Card& card : rule.cards) {
            if (card.pci == 0) {
                if (!card.name.empty()) ++empty_named;
                continue;
            }
            expect(!card.name.empty(), name + ": a card without a name");
            expect(pcis.insert(card.pci).second, name + ": a card listed twice");
            const board_registry::BoardRow* row = board_registry::find(card.pci);
            expect(row != nullptr,
                   name + ": card " + std::string(card.name) + " not in the registry");
            if (row == nullptr) continue;
            if (rule.regime != regime::Regime::BankLimit) {
                const bool same = generation_known(row->major, row->minor) &&
                                  generation_row(row->major, row->minor).name == rule.generation;
                expect(same, name + ": card " + std::string(card.name) + " is not of generation " +
                                 std::string(rule.generation));
            }
        }
        expect(empty_named == 0, name + ": a named card without a PCI id");
        const std::size_t cards = pcis.size();
        switch (rule.label) {
        case regime::Label::Measured:
            expect(cards >= 2, name + ": measured on fewer than two cards (" +
                                   std::to_string(cards) + "): label it single_card");
            break;
        case regime::Label::SingleCard:
            expect(cards == 1, name + ": single_card lists " + std::to_string(cards) + " cards");
            break;
        case regime::Label::Unmeasured:
            expect(cards == 0, name + ": unmeasured lists cards");
            break;
        }
        expect(!rule.tolerance.empty() && !rule.source.empty(),
               name + ": no tolerance or source recorded");
        expect(rule.limit >= 1 && rule.limit <= moe::kR8Tokens,
               name + ": limit outside 1..kR8Tokens");
        switch (rule.regime) {
        case regime::Regime::TokenLimit:
            expect(rule.applies == nullptr, name + ": a generation rule with a bank predicate");
            expect(rule.disposition == regime::Disposition::Applied,
                   name + ": a generation rule must be applied");
            expect(rule.limit == moe::kR8NarrowTokens || rule.label == regime::Label::Measured,
                   name + ": a limit wider than kR8NarrowTokens needs two cards");
            expect(rule.label != regime::Label::Unmeasured || rule.limit == moe::kR8NarrowTokens,
                   name + ": an unmeasured generation is not conservative");
            break;
        case regime::Regime::BankLimit:
            expect(rule.applies != nullptr && rule.generation.empty(),
                   name + ": a bank rule needs a predicate and no generation");
            // An unmeasured bank rule may only narrow: it cannot keep a widened limit.
            expect(rule.label != regime::Label::Unmeasured || rule.limit == moe::kR8NarrowTokens,
                   name + ": an unmeasured bank rule is not conservative");
            break;
        case regime::Regime::Cell:
            expect(rule.applies != nullptr && !rule.generation.empty(),
                   name + ": a cell needs a predicate and a generation");
            expect(rule.disposition == regime::Disposition::Recorded,
                   name + ": a cell is recorded, not applied");
            break;
        }
    }
    // One TokenLimit rule per generation of the table, and one for the unknown generation.
    std::set<std::string_view> generations;
    for (const GenerationRow& row : kGenerations) {
        int n = 0;
        for (const regime::RegimeRule& rule : regime::kRegimeRules) {
            n += rule.regime == regime::Regime::TokenLimit && rule.generation == row.name;
        }
        expect(n == 1, std::string("generation ") + row.name + ": " + std::to_string(n) +
                           " token-limit rules");
        generations.insert(row.name);
    }
    int unknown = 0;
    for (const regime::RegimeRule& rule : regime::kRegimeRules) {
        if (rule.regime != regime::Regime::TokenLimit) continue;
        if (rule.generation == regime::kUnknownGeneration) {
            ++unknown;
            expect(rule.label == regime::Label::Unmeasured && rule.limit == moe::kR8NarrowTokens,
                   "the unknown generation's rule is not conservative and unmeasured");
        } else {
            expect(generations.count(rule.generation) == 1,
                   name_of(rule) + ": a generation the table does not have");
        }
    }
    expect(unknown == 1, "the unknown generation has " + std::to_string(unknown) + " rules");
}

struct Board {
    const char* name;
    std::uint32_t pci;
    int major, minor, sms;
    bool roofline;
    int wide;            // r8_token_limit(), from the record
    regime::Label label; // the generation's label
};

// The boards of the record and one of each other generation, with the limits the comments give.
constexpr Board kBoards[] = {
    {"RTX 4090", 0x2684, 8, 9, 128, true, 8, regime::Label::Measured},
    {"RTX 4060 Ti", 0x2805, 8, 9, 34, true, 8, regime::Label::Measured},
    {"RTX 5090", 0x2B85, 12, 0, 170, true, 8, regime::Label::Measured},
    {"RTX 5070 Ti", 0x2C05, 12, 0, 70, true, 8, regime::Label::Measured},
    {"RTX 2080 Ti", 0x1E07, 7, 5, 68, true, 4, regime::Label::SingleCard},
    {"RTX 3090", 0x2204, 8, 6, 82, true, 4, regime::Label::Measured},
    {"RTX 3060", 0x2503, 8, 6, 28, true, 4, regime::Label::Measured},
    {"A100 SXM4 40GB", 0x20B0, 8, 0, 108, true, 4, regime::Label::Unmeasured},
    {"Quadro GV100", 0x1DBA, 7, 0, 80, true, 4, regime::Label::Unmeasured},
    {"sm_89 without a roofline", 0, 8, 9, 64, false, 8, regime::Label::Measured},
    {"sm_90", 0, 9, 0, 132, false, 4, regime::Label::Unmeasured},
    {"sm_100", 0, 10, 0, 148, false, 4, regime::Label::Unmeasured},
    {"sm_121", 0, 12, 1, 48, false, 8, regime::Label::Measured},
    // Not in the generation table: the unknown rule (a lent row carries no evidence).
    {"sm_61 (unknown)", 0, 6, 1, 28, false, 4, regime::Label::Unmeasured},
    {"sm_130 (unknown)", 0, 13, 0, 160, false, 4, regime::Label::Unmeasured},
};

Weight bank(QType q, QuantLayout layout) {
    Weight w;
    w.qtype  = q;
    w.layout = layout;
    return w;
}

void check_answers() {
    for (const Board& b : kBoards) {
        DeviceClass d = synthetic_device_class(b.pci, b.major, b.minor, b.sms, 64U << 20,
                                               std::size_t{16} << 30);
        d.facts       = b.roofline ? machine_facts_of(d) : MachineFacts{};
        const ScopedDeviceClass scoped(d);
        const std::string board = b.name;
        expect(moe::r8_token_limit() == b.wide, board + ": r8_token_limit() " +
                                                    std::to_string(moe::r8_token_limit()) + ", " +
                                                    std::to_string(b.wide) + " expected");
        const regime::RegimeRule& generation = regime::generation_rule(b.major, b.minor);
        expect(generation.label == b.label, board + ": the generation's label is " +
                                                std::string(regime::label_name(generation.label)));
        if (!generation_known(b.major, b.minor)) {
            expect(generation.generation == regime::kUnknownGeneration,
                   board + ": an unknown generation not answered by the unknown rule");
        }
        // A frag IQ3_XXS / IQ2_S bank narrows where the board has many DRAM bytes per SM cycle
        // (RTX 5070 Ti: 5.2) or none known; the RTX 4090, 4060 Ti and 5090 keep the generation's.
        const bool frag_iq_narrow = !b.roofline || std::string_view(b.name) == "RTX 5070 Ti";
        const int narrow          = std::min(4, b.wide);

        const struct {
            QType q;
            QuantLayout layout;
            int expected;
            const char* rule; // the rule that sets it ("" where the generation's does)
        } cases[] = {
            {QType::GGML_IQ3_XXS, QuantLayout::GgmlFrag, frag_iq_narrow ? narrow : b.wide,
             frag_iq_narrow && b.wide > 4
                 ? (b.roofline ? "r8_frag_dram_bound" : "r8_frag_no_roofline")
                 : ""},
            {QType::GGML_IQ2_S, QuantLayout::GgmlFrag, frag_iq_narrow ? narrow : b.wide, nullptr},
            {QType::GGML_Q4_0, QuantLayout::GgmlFrag, narrow,
             b.wide > 4 ? "r8_frag_q4_0_narrow" : ""},
            {QType::GGML_Q4_K, QuantLayout::GgmlFrag, b.wide, ""},
            {QType::GGML_IQ3_XXS, QuantLayout::GgmlBlocks, b.wide, ""},
            {QType::GGML_Q6_K, QuantLayout::GgmlBlocks, b.wide, ""},
            {QType::GGML_Q3_K, QuantLayout::GgmlBlocks, b.wide, ""},
            {QType::MXFP4, QuantLayout::BlockScaleK32E8M0, narrow,
             b.wide > 4 ? "r8_fp4_elements" : ""},
            {QType::NVFP4, QuantLayout::BlockScaleK16M128x4, narrow,
             b.wide > 4 ? "r8_fp4_elements" : ""},
        };

        for (const auto& c : cases) {
            const Weight w          = bank(c.q, c.layout);
            const auto answer       = regime::r8_limit_answer(w);
            const std::string label = board + " " + std::to_string(static_cast<int>(c.q)) +
                                      (c.layout == QuantLayout::GgmlFrag ? " frag" : "");
            expect(moe::r8_token_limit(w) == c.expected,
                   "R8 limit " + label + ": " + std::to_string(moe::r8_token_limit(w)) + ", " +
                       std::to_string(c.expected) + " expected");
            expect(answer.limit == moe::r8_token_limit(w), label + ": answer and limit differ");
            expect(answer.generation == &generation, label + ": not the generation's rule");
            if (c.rule != nullptr) {
                const std::string_view want =
                    *c.rule == '\0' ? generation.name : std::string_view(c.rule);
                expect(answer.rule != nullptr && answer.rule->name == want,
                       label + ": set by " +
                           (answer.rule != nullptr ? std::string(answer.rule->name) : "none") +
                           ", " + std::string(want) + " expected");
            }
        }
    }
}

} // namespace

int main() {
    check_records();
    check_answers();
    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("MoE R8 regime rules: %zu rules, each labelled by its cards; the answers of %zu "
                "boards\n",
                std::size(regime::kRegimeRules), std::size(kBoards));
    return 0;
}
