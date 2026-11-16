// The measured per-format rules (ops/generic/format_rules.h, #615 F6), read through their rows:
// every rule has rows and at least one with cards; a row's evidence tag matches its cards (TwoCards
// names two or more distinct cards, SingleCard exactly one, Carried none, and a carried row's rule
// has a row with cards); a row with cards names a device class id (chip:Nsm) for each card; every
// row has formats, a source, a date and a record; a format stands in one row of a rule; a
// format-set row has value 0 and a threshold row the constant the code reads, and that threshold's
// route_rules.h descriptor has the same value, basis, boards and cards (one record of the
// threshold, not two that can drift); `holds` equals the rows for every QType value. Which rows
// rest on one card is printed (principle 4: the list a two-card measurement works from); `--list`
// prints the table. Whether a row's evidence is right is read from its source by a person. That
// the selection reads these rows is held by ninferno_format_traits_test (the legacy predicates
// against `holds` and the engine's functions) and by the selector goldens. No GPU.

#include "core/format_facts.h"
#include "ops/generic/format_rules.h"
#include "ops/generic/route_rules.h"

#include <cstdio>
#include <iterator>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace fr = ninferno::ops::detail::generic::format_rules;
namespace rr = ninferno::ops::detail::generic::route_rules;
using ninferno::QType;

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::string str(std::string_view s) { return std::string(s); }

std::string trimmed(const std::string& s) {
    const auto a = s.find_first_not_of(' ');
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(' ');
    return s.substr(a, b - a + 1);
}

// The products of a cards field, split at the commas of its own level (a parenthesis such as
// "RTX 4090 (3.1 bytes a SM cycle)" stays with its card), each without its parenthesis.
std::vector<std::string> cards_of(std::string_view field) {
    std::vector<std::string> out;
    std::string cur;
    int depth = 0;
    for (const char c : field) {
        if (c == '(') ++depth;
        if (c == ')') --depth;
        if (c == ',' && depth == 0) {
            out.push_back(cur);
            cur.clear();
            continue;
        }
        cur += c;
    }
    out.push_back(cur);
    for (auto& card : out) {
        const auto paren = card.find('(');
        if (paren != std::string::npos) card = card.substr(0, paren);
        card = trimmed(card);
    }
    return out;
}

std::size_t board_ids_in(std::string_view field) {
    static const std::regex id(R"(\b[a-z]+[0-9]+[a-z]?:[0-9]+sm\b)");
    const std::string s(field);
    return static_cast<std::size_t>(
        std::distance(std::sregex_iterator(s.begin(), s.end(), id), std::sregex_iterator()));
}

const rr::RouteRule* route_rule(std::string_view name) {
    for (const auto& r : rr::kRouteRules) {
        if (r.name == name) return &r;
    }
    return nullptr;
}

std::string formats_text(fr::FormatSet set) {
    std::string out;
    for (int v = 0; v < fr::FormatSet::kCapacity; ++v) {
        if (!set.has(static_cast<QType>(v))) continue;
        if (!out.empty()) out += ' ';
        const std::string_view spelled = ninferno::format_facts(static_cast<QType>(v)).name;
        out += spelled.empty() ? std::to_string(v) : std::string(spelled);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    // The thresholds the code reads are the rows' values (the code reads these names).
    static_assert(fr::kMmqVerifyMinTokens == 9 && fr::kR8VerifyQ40MinValues == (24ll << 20));

    std::size_t single = 0, carried = 0;
    for (const fr::Rule rule : fr::kAllRules) {
        const std::string name = str(fr::rule_name(rule));
        std::size_t rows = 0, with_cards = 0;
        std::uint64_t seen = 0;
        for (const fr::FormatRule& r : fr::kFormatRules) {
            if (r.rule != rule) continue;
            ++rows;
            if (r.formats.empty()) fail(name + ": a row without formats");
            if ((seen & r.formats.bits) != 0) {
                fr::FormatSet twice;
                twice.bits = seen & r.formats.bits;
                fail(name + ": a format in two rows (" + formats_text(twice) + ")");
            }
            seen |= r.formats.bits;
            if (r.source.empty() || r.date.empty() || r.record.empty()) {
                fail(name + ": a row without source, date or record");
            }
            if (fr::evidence_name(r.evidence).empty())
                fail(name + ": a row without an evidence tag");
            const std::vector<std::string> cards = cards_of(r.cards);
            std::set<std::string> distinct;
            for (const auto& c : cards) {
                if (c.empty()) fail(name + ": an empty card in '" + str(r.cards) + "'");
                distinct.insert(c);
            }
            switch (r.evidence) {
            case fr::Evidence::Carried:
                ++carried;
                if (r.cards != fr::kNone || r.boards != fr::kNone) {
                    fail(name +
                         ": a carried row names cards or boards (it has no timing of its "
                         "own): '" +
                         str(r.cards) + "' / '" + str(r.boards) + "'");
                }
                if (r.basis != rr::Basis::Unknown) {
                    fail(name + ": a carried row's basis is Unknown (the record does not time it)");
                }
                break;
            case fr::Evidence::SingleCard:
                ++single;
                ++with_cards;
                if (distinct.size() != 1 || r.cards == fr::kNone) {
                    fail(name + ": tagged single_card but names '" + str(r.cards) + "'");
                }
                break;
            case fr::Evidence::TwoCards:
                ++with_cards;
                if (distinct.size() < 2) {
                    fail(name + ": tagged two_cards but names '" + str(r.cards) +
                         "' (tag it single_card, or name the second card)");
                }
                break;
            }
            if (r.evidence != fr::Evidence::Carried) {
                if (r.basis == rr::Basis::Unknown || r.basis == rr::Basis::Structural) {
                    fail(name + ": a row with cards rests on a timing (Measured, "
                                "PlacedBetweenMeasured, Fitted or ScaledFromOneBoard)");
                }
                if (board_ids_in(r.boards) < distinct.size()) {
                    fail(name + ": fewer board ids (chip:Nsm) than cards in '" + str(r.boards) +
                         "' for '" + str(r.cards) + "'");
                }
            }
            const bool threshold =
                rule == fr::Rule::R8VerifyQ40MinValues || rule == fr::Rule::MmqVerifyMinTokens;
            if (!threshold && r.value != 0) fail(name + ": a format-set row with a value");
        }
        if (rows == 0) fail(name + ": no rows");
        if (with_cards == 0) fail(name + ": no row with cards (every format carried)");
        // `holds` is the rows, for every value a QType holds in a byte's range.
        for (int v = 0; v < 256; ++v) {
            const auto q        = static_cast<QType>(v);
            const bool rows_say = v < fr::FormatSet::kCapacity && ((seen >> v) & 1u) != 0;
            if (fr::holds(rule, q) != rows_say) {
                fail(name + ": holds() disagrees with the rows for QType " + std::to_string(v));
            }
        }
    }

    // The thresholds: one row each, its value the constant the code reads, and the route_rules.h
    // descriptor of the same name with the same value, basis, boards and cards.
    struct Threshold {
        fr::Rule rule;
        std::int64_t constant;
    };

    for (const Threshold t : {Threshold{fr::Rule::R8VerifyQ40MinValues, fr::kR8VerifyQ40MinValues},
                              Threshold{fr::Rule::MmqVerifyMinTokens, fr::kMmqVerifyMinTokens}}) {
        const std::string name    = str(fr::rule_name(t.rule));
        const fr::FormatRule* row = nullptr;
        std::size_t count         = 0;
        for (const auto& r : fr::kFormatRules) {
            if (r.rule != t.rule) continue;
            row = &r;
            ++count;
        }
        if (count != 1 || row == nullptr) {
            fail(name + ": a threshold has exactly one row");
            continue;
        }
        if (row->value != t.constant) fail(name + ": the row's value is not the constant");
        const rr::RouteRule* desc = route_rule(name);
        if (desc == nullptr) {
            fail(name + ": no route_rules.h descriptor of the threshold");
            continue;
        }
        if (desc->file != "format_rules.h") {
            fail(name + ": its route_rules.h descriptor names " + str(desc->file) +
                 ", the constant is format_rules.h's");
        }
        if (desc->value != static_cast<double>(t.constant)) {
            fail(name + ": route_rules.h's value differs from the constant");
        }
        if (desc->basis != row->basis || desc->boards != row->boards || desc->cards != row->cards) {
            fail(name + ": route_rules.h's basis, boards or cards differ from the row's");
        }
    }
    if (fr::formats_of(fr::Rule::MmqVerifyMinTokens).bits !=
        fr::formats_of(fr::Rule::MmqVerify).bits) {
        fail("kMmqVerifyMinTokens: its formats are not the mmq_verify rows'");
    }

    if (argc > 1 && std::string(argv[1]) == "--list") {
        for (const auto& r : fr::kFormatRules) {
            const std::string_view rule = fr::rule_name(r.rule), ev = fr::evidence_name(r.evidence),
                                   basis = rr::basis_name(r.basis);
            std::printf("%-22.*s %-12.*s %-22.*s %-12lld [%s] %.*s | %.*s | %.*s\n",
                        static_cast<int>(rule.size()), rule.data(), static_cast<int>(ev.size()),
                        ev.data(), static_cast<int>(basis.size()), basis.data(),
                        static_cast<long long>(r.value), formats_text(r.formats).c_str(),
                        static_cast<int>(r.cards.size()), r.cards.data(),
                        static_cast<int>(r.date.size()), r.date.data(),
                        static_cast<int>(r.source.size()), r.source.data());
        }
    }
    std::printf("format rules: %zu rules, %zu rows, %zu single_card, %zu carried\n",
                fr::kAllRules.size(), fr::kFormatRules.size(), single, carried);
    for (const auto& r : fr::kFormatRules) {
        if (r.evidence != fr::Evidence::SingleCard) continue;
        const std::string_view rule = fr::rule_name(r.rule);
        std::printf("  single_card %-16.*s [%s] %.*s\n", static_cast<int>(rule.size()), rule.data(),
                    formats_text(r.formats).c_str(), static_cast<int>(r.cards.size()),
                    r.cards.data());
    }
    return failures == 0 ? 0 : 1;
}
