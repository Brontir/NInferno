// The literal rules of the generic path (ops/generic/route_rules.h), held to the sources they
// describe: every descriptor has a unique name, a literal that stands in its file outside comments
// exactly `occurrences` times (whitespace aside), a recorded boards / cards / tolerance field
// ("unknown" or "none recorded" where the record is silent, never empty) and an empty kernel hash
// (none is recorded); a named constant's value is the one the source defines.
//
// The guard over the sources: every `const` / `constexpr` definition and `#define` of a name kX
// whose name says it bounds something (Min, Max, Tokens, Fill, Distance, Values, Bytes, Gbps,
// Margin, Slack, Columns, Rows, Limit), in any .h, .cpp, .cu or .cuh under src/ops/generic
// (subdirectories included; route_cost_constants.h has its own registry and test), has a
// descriptor or an entry of kNotRules below with its reason. That is the extent of the guard:
// enumerators, a constant whose name carries none of those words, and a threshold written as a bare
// literal in an expression are found by the descriptors' own literal check only where a descriptor
// names them, not by this scan. `--list` prints the descriptors. No GPU.

#include "ops/generic/route_rules.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace rr = ninferno::ops::detail::generic::route_rules;

namespace {

const std::filesystem::path& generic_dir() {
    static const std::filesystem::path dir =
        std::filesystem::path(NINFERNO_SOURCE_DIR) / "src/ops/generic";
    return dir;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", path.string().c_str());
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The text without comments (block, then line comments), as the compiler reads it.
std::string without_comments(std::string text) {
    text = std::regex_replace(text, std::regex(R"(/\*[\s\S]*?\*/)"), "");
    return std::regex_replace(text, std::regex(R"(//[^\n]*)"), "");
}

// Runs of whitespace as one space: clang-format may realign a definition's `=`.
std::string squeezed(const std::string& text) {
    return std::regex_replace(text, std::regex(R"(\s+)"), " ");
}

std::size_t count_of(const std::string& haystack, const std::string& needle) {
    std::size_t n = 0;
    for (auto at = haystack.find(needle); at != std::string::npos;
         at      = haystack.find(needle, at + needle.size())) {
        ++n;
    }
    return n;
}

// The value of the right-hand side of a constant's definition: a number (with its integer or float
// suffix), the names of other constants, parentheses and the operators << * / + - (C++ precedence:
// shift lowest). `lookup` gives the text of a named constant's definition, or nothing.
class ConstantExpression {
public:
    using Lookup = std::function<std::optional<std::string>(const std::string&)>;

    ConstantExpression(std::string text, Lookup lookup, int depth = 0)
        : text_(std::move(text)), lookup_(std::move(lookup)), depth_(depth) {}

    // nullopt where the text is not an expression this class reads (the caller fails the rule).
    std::optional<double> value() {
        const double v = shift();
        skip();
        if (!ok_ || at_ != text_.size()) return std::nullopt;
        return v;
    }

private:
    void skip() {
        while (at_ < text_.size() && text_[at_] == ' ') ++at_;
    }

    bool take(const char* token) {
        skip();
        const std::size_t n = std::char_traits<char>::length(token);
        if (text_.compare(at_, n, token) != 0) return false;
        at_ += n;
        return true;
    }

    double shift() {
        double v = add();
        while (ok_ && take("<<")) {
            const double r = add();
            if (r < 0 || r > 62) ok_ = false;
            v = static_cast<double>(static_cast<long long>(v) << static_cast<int>(r));
        }
        return v;
    }

    double add() {
        double v = mul();
        while (ok_) {
            if (take("+"))
                v += mul();
            else if (take("-"))
                v -= mul();
            else
                break;
        }
        return v;
    }

    double mul() {
        double v = primary();
        while (ok_) {
            if (take("*"))
                v *= primary();
            else if (take("/")) {
                const double d = primary();
                if (d == 0)
                    ok_ = false;
                else
                    v /= d;
            } else
                break;
        }
        return v;
    }

    double primary() {
        skip();
        if (at_ >= text_.size()) {
            ok_ = false;
            return 0;
        }
        if (take("(")) {
            const double v = shift();
            if (!take(")")) ok_ = false;
            return v;
        }
        if (take("-")) return -primary();
        const char c = text_[at_];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            const char* begin = text_.c_str() + at_;
            char* end         = nullptr;
            const double v    = std::strtod(begin, &end);
            at_ += static_cast<std::size_t>(end - begin);
            while (at_ < text_.size() && std::strchr("uUlLfF", text_[at_]) != nullptr) ++at_;
            return v;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::size_t stop = at_;
            while (stop < text_.size() &&
                   (std::isalnum(static_cast<unsigned char>(text_[stop])) || text_[stop] == '_')) {
                ++stop;
            }
            const std::string name = text_.substr(at_, stop - at_);
            at_                    = stop;
            const auto definition  = depth_ < 8 ? lookup_(name) : std::nullopt;
            const auto v = definition ? ConstantExpression(*definition, lookup_, depth_ + 1).value()
                                      : std::nullopt;
            if (!v) {
                ok_ = false;
                return 0;
            }
            return *v;
        }
        ok_ = false;
        return 0;
    }

    std::string text_;
    Lookup lookup_;
    int depth_;
    std::size_t at_ = 0;
    bool ok_        = true;
};

// The right-hand side of `constexpr <type> <name> = <expr>;` in `text`, if it stands there.
std::optional<std::string> definition_of_constant(const std::string& text,
                                                  const std::string& name) {
    const std::regex pattern(R"(constexpr [^=;]*\b)" + name + " = ([^;]+);");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) return std::nullopt;
    return match[1].str();
}

// Names the scan finds that no selection compares a call against, with the reason.
const std::map<std::string, std::string>& not_rules() {
    static const std::map<std::string, std::string> m = {
        {"kMaxSegments", "the segmented projection's slot count (an array size)"},
        {"kMaxScratchUses", "scratch lifetime bookkeeping"},
        {"kExactMinor", "a build switch (an \"a\" / \"f\" target's minor)"},
        {"kActivationSlack", "ggml_mmq's arena sizing"},
        {"kGgmlArenaSlack", "the ggml arena's sizing"},
        {"kBlockBytes", "a codec's block size (a layout fact)"},
        {"kGroupBytes", "a codec's group size (a layout fact)"},
        {"kBufferBytes", "a kernel's shared buffer size"},
        {"kSmemBytes", "a kernel's shared memory use (counted)"},
        {"kStageBytes", "a kernel's pipeline stage size (counted)"},
        {"kRingBytes", "a kernel's load ring size (counted)"},
        {"kW4a8MaxDynamicSmem", "the architecture's dynamic shared memory limit"},
        {"kCodeRowBytes", "a codec's row size (a layout fact)"},
        {"kChunkWeightBytes", "Volta's decode chunk: 140 MiB of FP16 (a scratch size, derived in "
                              "the comment from the memory it may take)"},
        {"kCtaRows", "a kernel's rows per CTA (a tile shape)"},
        {"kWarpRows", "a kernel's rows per warp (a tile shape)"},
        {"kSetRows", "a kernel's rows per set (a tile shape)"},
        {"kHalfMax", "the largest finite FP16 value"},
        {"kHalfMin", "the smallest normal FP16 value"},
        {"kIq1Minus", "an IQ1 codebook offset (a format fact)"},
        {"kIq1MinusHigh", "an IQ1 codebook offset (a format fact)"},
        {"kIq1MinusLow", "an IQ1 codebook offset (a format fact)"},
        {"kWithMin", "a codec flag name (the format carries a min)"},
        {"kRmmaMinBlocks", "a launch bound: the registers a kernel is capped at"},
        {"kVecMaxSegments", "a kernel's argument slots (an array size)"},
        {"kMin", "a template parameter of a kernel (does the codec carry a min)"},
        {"kMinBlocks", "a launch bound: resident blocks a kernel asks for"},
        {"kMinTerm", "a kernel's own accumulation term (a codec fact)"},
        {"kRows", "a kernel's tile rows (a tile shape, counted)"},
        {"kRowsPerBlock", "a kernel's tile rows (a tile shape, counted)"},
        {"kRowsPerCta", "a kernel's tile rows (a tile shape, counted)"},
        {"kTokens", "a kernel's tile tokens (a tile shape, counted)"},
        {"kTokensPerCta", "a kernel's tile tokens (a tile shape, counted)"},
        {"kValuesPerBlock", "a codec's values per block (a layout fact)"},
    };
    return m;
}

} // namespace

int main(int argc, char** argv) {
    int failures    = 0;
    const auto fail = [&](const std::string& message) {
        std::fprintf(stderr, "%s\n", message.c_str());
        ++failures;
    };
    std::set<std::string> names;
    std::map<std::string, std::string> sources; // file -> text without comments, squeezed
    const auto source = [&](const std::string& file) -> const std::string& {
        auto it = sources.find(file);
        if (it == sources.end()) {
            it = sources.emplace(file, squeezed(without_comments(read_file(generic_dir() / file))))
                     .first;
        }
        return it->second;
    };
    for (const auto& rule : rr::kRouteRules) {
        const std::string name(rule.name);
        if (name.empty() || !names.insert(name).second) {
            fail("rule name '" + name + "' empty or listed twice");
        }
        if (rule.literal.empty()) fail(name + ": no literal");
        for (const auto field : {rule.boards, rule.cards, rule.tolerance}) {
            if (field.empty()) fail(name + ": a boards / cards / tolerance field is empty");
        }
        if (!rule.kernel_hash.empty()) {
            fail(name +
                 ": a kernel hash is set, but none is recorded anywhere to check it against");
        }
        const std::string& file = source(std::string(rule.file));
        if (file.empty()) {
            fail(name + ": cannot read " + std::string(rule.file));
            continue;
        }
        const std::string literal = squeezed(std::string(rule.literal));
        const std::size_t found   = count_of(file, literal);
        if (found != rule.occurrences) {
            fail(name + ": the literal '" + std::string(rule.literal) + "' stands " +
                 std::to_string(found) + " times outside comments in " + std::string(rule.file) +
                 ", the descriptor says " + std::to_string(rule.occurrences));
        }
        // A named constant: the number the source defines is the descriptor's. The definition may
        // be an expression (a shift, a quotient, a product with another constant): it is
        // evaluated, and one this test cannot read is a failure, never a skipped comparison.
        static const std::regex definition(R"(constexpr [^=;]*\b(k[A-Z]\w*) = ([^;]+);)");
        std::smatch match;
        if (std::regex_search(literal, match, definition)) {
            if (match[1].str() != name) { fail(name + ": its literal defines " + match[1].str()); }
            const auto lookup = [&](const std::string& other) -> std::optional<std::string> {
                if (const auto own = definition_of_constant(file, other)) return own;
                return definition_of_constant(source("../../core/units.h"), other);
            };
            const auto defined = ConstantExpression(match[2].str(), lookup).value();
            if (!defined) {
                fail(name + ": cannot evaluate the source's '" + match[2].str() + "'");
            } else if (std::fabs(*defined - rule.value) >
                       1e-12 * std::max(1.0, std::fabs(rule.value))) {
                fail(name + ": the source says " + match[2].str() + " (" +
                     std::to_string(*defined) + "), the descriptor " + std::to_string(rule.value));
            }
        } else if (literal.find("constexpr") != std::string::npos &&
                   literal.find(" = ") != std::string::npos && literal.back() == ';') {
            // A complete scalar definition that the pattern above missed; a literal that is the
            // start of an array's initializer (kPrefillBandTokens) has no `;` and is only counted.
            fail(name + ": a constant's definition the test cannot read");
        }
    }
    // The guard: every bounding constant of the selection code is a described rule or a listed
    // non-rule.
    static const std::regex bounding(
        "Min|Max|Tokens|Fill|Distance|Values|Bytes|Gbps|Margin|Slack|Columns|Rows|Limit");
    static const std::regex definition_of(
        R"(\b(?:constexpr|const)\b[^=;(){}]*?\b(k[A-Z]\w*)\s*(?:=|\{))");
    static const std::regex define_of(R"(#\s*define\s+(k[A-Z]\w*))");
    std::size_t scanned = 0;
    std::set<std::string> found_names;
    for (const auto& item : std::filesystem::recursive_directory_iterator(generic_dir())) {
        const auto ext = item.path().extension();
        if (!item.is_regular_file() ||
            (ext != ".h" && ext != ".cpp" && ext != ".cu" && ext != ".cuh")) {
            continue;
        }
        const std::string file = item.path().filename().string();
        if (file == "route_rules.h" || file == "route_cost_constants.h") continue;
        const std::string text = without_comments(read_file(item.path()));
        ++scanned;
        for (const std::regex* pattern : {&definition_of, &define_of}) {
            for (std::sregex_iterator it(text.begin(), text.end(), *pattern), stop; it != stop;
                 ++it) {
                const std::string name = (*it)[1].str();
                if (!std::regex_search(name, bounding)) continue;
                found_names.insert(name);
                if (names.count(name) == 0 && not_rules().count(name) == 0) {
                    fail(name + " (" + file +
                         ") is a bounding constant of the selection code without a descriptor "
                         "in kRouteRules (or an entry of kNotRules)");
                }
            }
        }
    }
    if (scanned < 20 || found_names.size() < 30) fail("the constant scan found too little");
    for (const auto& [name, why] : not_rules()) {
        if (found_names.count(name) == 0) fail(name + " is in kNotRules but no longer defined");
        if (names.count(name) != 0) fail(name + " is both a rule and in kNotRules");
    }
    if (argc > 1 && std::string(argv[1]) == "--list") {
        for (const auto& rule : rr::kRouteRules) {
            std::printf("%-44.*s %-10g %.*s | %.*s\n", static_cast<int>(rule.name.size()),
                        rule.name.data(), rule.value, static_cast<int>(rule.cards.size()),
                        rule.cards.data(), static_cast<int>(rule.tolerance.size()),
                        rule.tolerance.data());
        }
    }
    std::printf("route rules: %zu described, %zu non-rules listed, kernel hash recorded for none\n",
                std::size(rr::kRouteRules), not_rules().size());
    return failures == 0 ? 0 : 1;
}
