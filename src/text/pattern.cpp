#include "text/pattern.h"

#include <utf8proc/utf8proc.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ninferno::text {
namespace {

constexpr int kUnbounded   = -1;
constexpr int kMaxRepeat   = 1000;
constexpr char32_t kMaxCp  = 0x10FFFF;

constexpr std::uint32_t bit(utf8proc_category_t category) {
    return std::uint32_t{1} << static_cast<unsigned>(category);
}

constexpr std::uint32_t kLetters = bit(UTF8PROC_CATEGORY_LU) | bit(UTF8PROC_CATEGORY_LL) |
                                   bit(UTF8PROC_CATEGORY_LT) | bit(UTF8PROC_CATEGORY_LM) |
                                   bit(UTF8PROC_CATEGORY_LO);
constexpr std::uint32_t kMarks =
    bit(UTF8PROC_CATEGORY_MN) | bit(UTF8PROC_CATEGORY_MC) | bit(UTF8PROC_CATEGORY_ME);
constexpr std::uint32_t kNumbers =
    bit(UTF8PROC_CATEGORY_ND) | bit(UTF8PROC_CATEGORY_NL) | bit(UTF8PROC_CATEGORY_NO);
constexpr std::uint32_t kPunctuation =
    bit(UTF8PROC_CATEGORY_PC) | bit(UTF8PROC_CATEGORY_PD) | bit(UTF8PROC_CATEGORY_PS) |
    bit(UTF8PROC_CATEGORY_PE) | bit(UTF8PROC_CATEGORY_PI) | bit(UTF8PROC_CATEGORY_PF) |
    bit(UTF8PROC_CATEGORY_PO);
constexpr std::uint32_t kSymbols = bit(UTF8PROC_CATEGORY_SM) | bit(UTF8PROC_CATEGORY_SC) |
                                   bit(UTF8PROC_CATEGORY_SK) | bit(UTF8PROC_CATEGORY_SO);
constexpr std::uint32_t kSeparators =
    bit(UTF8PROC_CATEGORY_ZS) | bit(UTF8PROC_CATEGORY_ZL) | bit(UTF8PROC_CATEGORY_ZP);
constexpr std::uint32_t kOthers = bit(UTF8PROC_CATEGORY_CC) | bit(UTF8PROC_CATEGORY_CF) |
                                  bit(UTF8PROC_CATEGORY_CS) | bit(UTF8PROC_CATEGORY_CO) |
                                  bit(UTF8PROC_CATEGORY_CN);
// Oniguruma's Unicode \w: Letter | Mark | Decimal_Number | Connector_Punctuation.
constexpr std::uint32_t kWord =
    kLetters | kMarks | bit(UTF8PROC_CATEGORY_ND) | bit(UTF8PROC_CATEGORY_PC);

using Range = std::pair<char32_t, char32_t>;

// Oniguruma's Unicode \s: 0009-000D, 0085 and the Zs/Zl/Zp code points (= White_Space).
const std::vector<Range>& space_ranges() {
    static const std::vector<Range> ranges = {
        {0x09, 0x0D},     {0x20, 0x20},     {0x85, 0x85},     {0xA0, 0xA0},
        {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029}, {0x202F, 0x202F},
        {0x205F, 0x205F}, {0x3000, 0x3000},
    };
    return ranges;
}

char32_t simple_fold(char32_t cp) {
    return static_cast<char32_t>(
        utf8proc_tolower(utf8proc_toupper(static_cast<utf8proc_int32_t>(cp))));
}

// A set of code points: union of ranges, general categories and nested (typically
// complemented) sets, optionally complemented itself.
struct CharSet {
    bool negate = false;
    // Bracket classes inside (?i): the input also matches through its case variants.
    bool icase = false;
    std::vector<Range> ranges;
    std::uint32_t categories = 0;
    std::vector<CharSet> nested;
    std::array<std::uint64_t, 2> ascii{};

    [[nodiscard]] bool contains(char32_t cp, std::uint32_t category_bit) const {
        for (const auto& [lo, hi] : ranges) {
            if (cp >= lo && cp <= hi) { return true; }
        }
        if ((categories & category_bit) != 0) { return true; }
        for (const CharSet& inner : nested) {
            if (inner.matches(cp, category_bit)) { return true; }
        }
        return false;
    }

    [[nodiscard]] bool slow_matches(char32_t cp, std::uint32_t category_bit) const {
        bool in = contains(cp, category_bit);
        if (!in && icase) {
            for (const char32_t variant :
                 {static_cast<char32_t>(utf8proc_tolower(static_cast<utf8proc_int32_t>(cp))),
                  static_cast<char32_t>(utf8proc_toupper(static_cast<utf8proc_int32_t>(cp))),
                  simple_fold(cp)}) {
                if (variant != cp &&
                    contains(variant, bit(utf8proc_category(static_cast<utf8proc_int32_t>(
                                          variant))))) {
                    in = true;
                    break;
                }
            }
        }
        return in != negate;
    }

    // category_bit is bit(category of cp); callers compute it once per text code point.
    [[nodiscard]] bool matches(char32_t cp, std::uint32_t category_bit) const {
        if (cp < 128) { return ((ascii[cp >> 6U] >> (cp & 63U)) & 1U) != 0; }
        return slow_matches(cp, category_bit);
    }

    void finalize() {
        for (CharSet& inner : nested) { inner.finalize(); }
        ascii = {};
        for (char32_t cp = 0; cp < 128; ++cp) {
            if (slow_matches(cp, bit(utf8proc_category(static_cast<utf8proc_int32_t>(cp))))) {
                ascii[cp >> 6U] |= std::uint64_t{1} << (cp & 63U);
            }
        }
    }
};

struct Node {
    enum class Kind { kSet, kConcat, kAlt, kRepeat, kLook, kAtomic };
    Kind kind = Kind::kConcat;
    int set   = -1;
    std::vector<Node> children;
    int min         = 0;
    int max         = 0;  // kUnbounded for no upper limit
    bool greedy     = true;
    bool possessive = false;
    bool negate     = false;  // negative lookahead
};

bool nullable(const Node& node) {
    switch (node.kind) {
    case Node::Kind::kSet:
        return false;
    case Node::Kind::kConcat:
        for (const Node& child : node.children) {
            if (!nullable(child)) { return false; }
        }
        return true;
    case Node::Kind::kAlt:
        for (const Node& child : node.children) {
            if (nullable(child)) { return true; }
        }
        return false;
    case Node::Kind::kRepeat:
        return node.min == 0 || nullable(node.children[0]);
    case Node::Kind::kLook:
        return true;
    case Node::Kind::kAtomic:
        return nullable(node.children[0]);
    }
    return true;
}

std::u32string decode_pattern(std::string_view text) {
    std::u32string out;
    for (std::size_t offset = 0; offset < text.size();) {
        utf8proc_int32_t cp = 0;
        const utf8proc_ssize_t length =
            utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(text.data() + offset),
                             static_cast<utf8proc_ssize_t>(text.size() - offset), &cp);
        if (length <= 0) { throw std::invalid_argument("regex pattern is not valid UTF-8"); }
        out.push_back(static_cast<char32_t>(cp));
        offset += static_cast<std::size_t>(length);
    }
    return out;
}

struct GeneralCategory {
    std::string_view name;
    std::uint32_t bits;
};

std::uint32_t category_bits(std::string_view name, std::string_view pattern) {
    static const GeneralCategory table[] = {
        {"l", kLetters},
        {"letter", kLetters},
        {"lu", bit(UTF8PROC_CATEGORY_LU)},
        {"uppercaseletter", bit(UTF8PROC_CATEGORY_LU)},
        {"ll", bit(UTF8PROC_CATEGORY_LL)},
        {"lowercaseletter", bit(UTF8PROC_CATEGORY_LL)},
        {"lt", bit(UTF8PROC_CATEGORY_LT)},
        {"titlecaseletter", bit(UTF8PROC_CATEGORY_LT)},
        {"lm", bit(UTF8PROC_CATEGORY_LM)},
        {"modifierletter", bit(UTF8PROC_CATEGORY_LM)},
        {"lo", bit(UTF8PROC_CATEGORY_LO)},
        {"otherletter", bit(UTF8PROC_CATEGORY_LO)},
        {"lc", bit(UTF8PROC_CATEGORY_LU) | bit(UTF8PROC_CATEGORY_LL) | bit(UTF8PROC_CATEGORY_LT)},
        {"l&", bit(UTF8PROC_CATEGORY_LU) | bit(UTF8PROC_CATEGORY_LL) | bit(UTF8PROC_CATEGORY_LT)},
        {"m", kMarks},
        {"mark", kMarks},
        {"mn", bit(UTF8PROC_CATEGORY_MN)},
        {"mc", bit(UTF8PROC_CATEGORY_MC)},
        {"me", bit(UTF8PROC_CATEGORY_ME)},
        {"n", kNumbers},
        {"number", kNumbers},
        {"nd", bit(UTF8PROC_CATEGORY_ND)},
        {"nl", bit(UTF8PROC_CATEGORY_NL)},
        {"no", bit(UTF8PROC_CATEGORY_NO)},
        {"p", kPunctuation},
        {"punctuation", kPunctuation},
        {"pc", bit(UTF8PROC_CATEGORY_PC)},
        {"pd", bit(UTF8PROC_CATEGORY_PD)},
        {"ps", bit(UTF8PROC_CATEGORY_PS)},
        {"pe", bit(UTF8PROC_CATEGORY_PE)},
        {"pi", bit(UTF8PROC_CATEGORY_PI)},
        {"pf", bit(UTF8PROC_CATEGORY_PF)},
        {"po", bit(UTF8PROC_CATEGORY_PO)},
        {"s", kSymbols},
        {"symbol", kSymbols},
        {"sm", bit(UTF8PROC_CATEGORY_SM)},
        {"sc", bit(UTF8PROC_CATEGORY_SC)},
        {"sk", bit(UTF8PROC_CATEGORY_SK)},
        {"so", bit(UTF8PROC_CATEGORY_SO)},
        {"z", kSeparators},
        {"separator", kSeparators},
        {"zs", bit(UTF8PROC_CATEGORY_ZS)},
        {"zl", bit(UTF8PROC_CATEGORY_ZL)},
        {"zp", bit(UTF8PROC_CATEGORY_ZP)},
        {"c", kOthers},
        {"other", kOthers},
        {"cc", bit(UTF8PROC_CATEGORY_CC)},
        {"cf", bit(UTF8PROC_CATEGORY_CF)},
        {"cs", bit(UTF8PROC_CATEGORY_CS)},
        {"co", bit(UTF8PROC_CATEGORY_CO)},
        {"cn", bit(UTF8PROC_CATEGORY_CN)},
    };
    // Oniguruma compares property names ignoring case, spaces, hyphens and underscores.
    std::string key;
    for (const char c : name) {
        if (c == ' ' || c == '_' || c == '-') { continue; }
        key.push_back(static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
    }
    for (const GeneralCategory& entry : table) {
        if (entry.name == key) { return entry.bits; }
    }
    throw std::invalid_argument("unsupported Unicode property \\p{" + std::string(name) +
                                "} in regex " + std::string(pattern) +
                                " (only general categories are supported)");
}

// One escape or literal inside or outside a bracket class: a single code point or a set.
struct Atom {
    bool is_char = false;
    char32_t cp  = 0;
    CharSet set;
};

class Parser {
public:
    Parser(std::string_view source, std::vector<CharSet>& sets)
        : source_(source), text_(decode_pattern(source)), sets_(sets) {}

    Node parse() {
        bool icase = false;
        Node root  = parse_alternation(icase);
        if (pos_ != text_.size()) { fail("unbalanced ')'"); }
        resolve_case_folds();
        return root;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        throw std::invalid_argument("unsupported regex " + std::string(source_) + ": " + what);
    }

    [[nodiscard]] bool at_end() const { return pos_ >= text_.size(); }
    [[nodiscard]] char32_t peek(std::size_t ahead = 0) const {
        return pos_ + ahead < text_.size() ? text_[pos_ + ahead] : char32_t{0};
    }
    char32_t next() {
        if (at_end()) { fail("unexpected end of pattern"); }
        return text_[pos_++];
    }

    int add_set(CharSet set) {
        sets_.push_back(std::move(set));
        return static_cast<int>(sets_.size() - 1);
    }

    Node set_node(int set) {
        Node node;
        node.kind = Node::Kind::kSet;
        node.set  = set;
        return node;
    }

    Node literal_node(char32_t cp, bool icase) {
        CharSet set;
        set.ranges.push_back({cp, cp});
        const int index = add_set(std::move(set));
        if (icase) { folded_literals_.push_back(index); }
        return set_node(index);
    }

    // `inherited` carries an inline (?i) flag that lasts until the end of the enclosing group,
    // across later alternatives, as in Perl and Oniguruma.
    Node parse_alternation(bool& icase) {
        Node first = parse_sequence(icase);
        if (peek() != U'|' || at_end()) { return first; }
        Node alt;
        alt.kind = Node::Kind::kAlt;
        alt.children.push_back(std::move(first));
        while (!at_end() && peek() == U'|') {
            ++pos_;
            alt.children.push_back(parse_sequence(icase));
        }
        return alt;
    }

    Node parse_sequence(bool& icase) {
        Node seq;
        seq.kind = Node::Kind::kConcat;
        while (!at_end() && peek() != U'|' && peek() != U')') {
            bool inline_flags = false;
            Node atom         = parse_atom(icase, inline_flags);
            if (inline_flags) { continue; }
            parse_quantifiers(atom);
            seq.children.push_back(std::move(atom));
        }
        if (seq.children.size() == 1) { return std::move(seq.children[0]); }
        return seq;
    }

    // Parses "{m}", "{m,}", "{,n}" or "{m,n}" at pos_. Leaves pos_ unchanged and returns false
    // when the brace does not start a quantifier, in which case it is a literal.
    bool parse_braces(int& min, int& max) {
        std::size_t p = pos_ + 1;
        const auto number = [&](int& value) {
            const std::size_t begin = p;
            long long parsed        = 0;
            while (p < text_.size() && text_[p] >= U'0' && text_[p] <= U'9') {
                parsed = parsed * 10 + static_cast<long long>(text_[p] - U'0');
                if (parsed > kMaxRepeat) { fail("repeat count above 1000"); }
                ++p;
            }
            value = static_cast<int>(parsed);
            return p > begin;
        };
        int lo = 0, hi = 0;
        const bool has_lo = number(lo);
        if (p < text_.size() && text_[p] == U'}') {
            if (!has_lo) { return false; }
            min = max = lo;
            pos_      = p + 1;
            return true;
        }
        if (p >= text_.size() || text_[p] != U',') { return false; }
        ++p;
        const bool has_hi = number(hi);
        if (p >= text_.size() || text_[p] != U'}' || (!has_lo && !has_hi)) { return false; }
        min = has_lo ? lo : 0;
        max = has_hi ? hi : kUnbounded;
        if (max != kUnbounded && max < min) { fail("repeat range {m,n} with n < m"); }
        pos_ = p + 1;
        return true;
    }

    void parse_quantifiers(Node& atom) {
        while (!at_end()) {
            int min = 0, max = 0;
            bool braces = false;
            switch (peek()) {
            case U'?':
                min = 0, max = 1;
                ++pos_;
                break;
            case U'*':
                min = 0, max = kUnbounded;
                ++pos_;
                break;
            case U'+':
                min = 1, max = kUnbounded;
                ++pos_;
                break;
            case U'{':
                if (!parse_braces(min, max)) { return; }
                braces = true;
                break;
            default:
                return;
            }
            Node repeat;
            repeat.kind = Node::Kind::kRepeat;
            repeat.min  = min;
            repeat.max  = max;
            if (peek() == U'?' && !at_end()) {
                repeat.greedy = false;
                ++pos_;
            } else if (peek() == U'+' && !at_end() && !braces) {
                // Ruby syntax: ?+ *+ ++ are possessive; {n,m}+ is a nested repeat instead.
                repeat.possessive = true;
                ++pos_;
            }
            if (nullable(atom) && (max == kUnbounded || max > 1)) {
                fail("repeated sub-pattern can match the empty string");
            }
            repeat.children.push_back(std::move(atom));
            atom = std::move(repeat);
        }
    }

    Node parse_group(bool& icase, bool& inline_flags) {
        // pos_ is just past '('.
        if (peek() != U'?') {
            bool inner = icase;
            Node body  = parse_alternation(inner);
            expect_close();
            return body;
        }
        ++pos_;
        const char32_t kind = next();
        if (kind == U':') {
            bool inner = icase;
            Node body  = parse_alternation(inner);
            expect_close();
            return body;
        }
        if (kind == U'=' || kind == U'!' || kind == U'>') {
            bool inner = icase;
            Node body  = parse_alternation(inner);
            expect_close();
            Node node;
            node.kind   = kind == U'>' ? Node::Kind::kAtomic : Node::Kind::kLook;
            node.negate = kind == U'!';
            node.children.push_back(std::move(body));
            return node;
        }
        if (kind == U'<') { fail("lookbehind and named groups"); }
        // Option flags: (?i) (?-i) (?i:...) (?-i:...).
        --pos_;
        bool on        = true;
        bool new_icase = icase;
        while (!at_end() && peek() != U':' && peek() != U')') {
            const char32_t flag = next();
            if (flag == U'-') {
                on = false;
            } else if (flag == U'i') {
                new_icase = on;
            } else {
                fail("group option other than i");
            }
        }
        if (next() == U')') {
            icase        = new_icase;
            inline_flags = true;
            return Node{};
        }
        Node body = parse_alternation(new_icase);
        expect_close();
        return body;
    }

    void expect_close() {
        if (at_end() || next() != U')') { fail("missing ')'"); }
    }

    static int hex_value(char32_t c) {
        if (c >= U'0' && c <= U'9') { return static_cast<int>(c - U'0'); }
        if (c >= U'a' && c <= U'f') { return static_cast<int>(c - U'a' + 10); }
        if (c >= U'A' && c <= U'F') { return static_cast<int>(c - U'A' + 10); }
        return -1;
    }

    char32_t parse_hex(std::size_t digits) {
        char32_t value = 0;
        for (std::size_t i = 0; i < digits; ++i) {
            const int digit = hex_value(next());
            if (digit < 0) { fail("malformed hex escape"); }
            value = value * 16 + static_cast<char32_t>(digit);
        }
        return value;
    }

    // Parses the escape after a backslash (pos_ is past the backslash).
    Atom parse_escape() {
        const char32_t c = next();
        Atom atom;
        const auto character = [&](char32_t cp) {
            atom.is_char = true;
            atom.cp      = cp;
            return atom;
        };
        switch (c) {
        case U't':
            return character(U'\t');
        case U'n':
            return character(U'\n');
        case U'r':
            return character(U'\r');
        case U'f':
            return character(U'\f');
        case U'v':
            return character(U'\v');
        case U'a':
            return character(0x07);
        case U'e':
            return character(0x1B);
        case U'0':
            return character(0);
        case U'x': {
            if (peek() == U'{') {
                ++pos_;
                char32_t value     = 0;
                std::size_t digits = 0;
                while (!at_end() && peek() != U'}') {
                    const int digit = hex_value(next());
                    if (digit < 0 || ++digits > 8) { fail("malformed \\x{...} escape"); }
                    value = value * 16 + static_cast<char32_t>(digit);
                }
                if (digits == 0 || next() != U'}' || value > kMaxCp) {
                    fail("malformed \\x{...} escape");
                }
                return character(value);
            }
            return character(parse_hex(2));
        }
        case U'u':
            return character(parse_hex(4));
        case U's':
        case U'S':
            atom.set.ranges = space_ranges();
            atom.set.negate = c == U'S';
            return atom;
        case U'd':
        case U'D':
            atom.set.categories = bit(UTF8PROC_CATEGORY_ND);
            atom.set.negate     = c == U'D';
            return atom;
        case U'w':
        case U'W':
            atom.set.categories = kWord;
            atom.set.negate     = c == U'W';
            return atom;
        case U'p':
        case U'P': {
            if (next() != U'{') { fail("\\p without braces"); }
            bool negate = c == U'P';
            if (peek() == U'^') {
                negate = !negate;
                ++pos_;
            }
            std::string name;
            while (!at_end() && peek() != U'}') {
                const char32_t ch = next();
                if (ch >= 128) { fail("non-ASCII property name"); }
                name.push_back(static_cast<char>(ch));
            }
            expect_char(U'}');
            atom.set.categories = category_bits(name, source_);
            atom.set.negate     = negate;
            return atom;
        }
        default:
            break;
        }
        if ((c >= U'0' && c <= U'9') || (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z')) {
            fail(std::string("escape \\") + static_cast<char>(c) +
                 " (backreferences, anchors and other escapes)");
        }
        return character(c);
    }

    void expect_char(char32_t expected) {
        if (at_end() || next() != expected) { fail("malformed pattern"); }
    }

    // Parses a bracket class; pos_ is just past '['.
    CharSet parse_class(bool icase) {
        CharSet set;
        set.icase = icase;
        if (peek() == U'^') {
            set.negate = true;
            ++pos_;
        }
        bool first = true;
        while (true) {
            if (at_end()) { fail("missing ']'"); }
            char32_t c = peek();
            if (c == U']' && !first) {
                ++pos_;
                break;
            }
            first = false;
            if (c == U'[') { fail("nested or POSIX bracket class"); }
            if (c == U'&' && peek(1) == U'&') { fail("class intersection"); }
            Atom item;
            ++pos_;
            if (c == U'\\') {
                item = parse_escape();
            } else {
                item.is_char = true;
                item.cp      = c;
            }
            if (!item.is_char) {
                if (item.set.negate) {
                    set.nested.push_back(std::move(item.set));
                } else {
                    set.ranges.insert(set.ranges.end(), item.set.ranges.begin(),
                                      item.set.ranges.end());
                    set.categories |= item.set.categories;
                }
                continue;
            }
            char32_t lo = item.cp;
            char32_t hi = item.cp;
            if (peek() == U'-' && peek(1) != U']' && pos_ + 1 < text_.size()) {
                ++pos_;
                char32_t end = next();
                if (end == U'\\') {
                    Atom bound = parse_escape();
                    if (!bound.is_char) { fail("class range ending in a set escape"); }
                    end = bound.cp;
                } else if (end == U'[') {
                    fail("nested bracket class");
                }
                if (end < lo) { fail("reversed class range"); }
                hi = end;
            }
            set.ranges.push_back({lo, hi});
        }
        return set;
    }

    Node parse_atom(bool& icase, bool& inline_flags) {
        const char32_t c = next();
        switch (c) {
        case U'(':
            return parse_group(icase, inline_flags);
        case U'[':
            return set_node(add_set(parse_class(icase)));
        case U'.': {
            CharSet set;
            set.negate = true;
            set.ranges.push_back({U'\n', U'\n'});
            return set_node(add_set(std::move(set)));
        }
        case U'\\': {
            Atom atom = parse_escape();
            if (atom.is_char) { return literal_node(atom.cp, icase); }
            // Under (?i) a set escape such as \p{Lu} also accepts the case variants of the input.
            atom.set.icase = icase;
            return set_node(add_set(std::move(atom.set)));
        }
        case U'^':
        case U'$':
            fail("anchors");
        case U'*':
        case U'+':
        case U'?':
            fail("quantifier without a target");
        default:
            return literal_node(c, icase);
        }
    }

    // A case-insensitive literal matches every code point with the same simple fold; the
    // inverse fold needs a scan of the code space, done once for all such literals.
    void resolve_case_folds() {
        if (folded_literals_.empty()) { return; }
        std::unordered_map<char32_t, std::vector<char32_t>> classes;
        for (const int index : folded_literals_) {
            classes.emplace(simple_fold(sets_[static_cast<std::size_t>(index)].ranges[0].first),
                            std::vector<char32_t>{});
        }
        for (char32_t cp = 0; cp <= kMaxCp; ++cp) {
            if (cp == 0xD800) { cp = 0xDFFF; continue; }
            const auto it = classes.find(simple_fold(cp));
            if (it != classes.end()) { it->second.push_back(cp); }
        }
        for (const int index : folded_literals_) {
            CharSet& set = sets_[static_cast<std::size_t>(index)];
            for (const char32_t cp : classes.at(simple_fold(set.ranges[0].first))) {
                if (cp != set.ranges[0].first) { set.ranges.push_back({cp, cp}); }
            }
        }
    }

    std::string_view source_;
    std::u32string text_;
    std::size_t pos_ = 0;
    std::vector<CharSet>& sets_;
    std::vector<int> folded_literals_;
};

enum class Op : std::uint8_t { kSet, kSplit, kJump, kMatch, kLook, kAtomic };

struct Inst {
    Op op       = Op::kMatch;
    bool negate = false;
    int x       = 0;  // kSet: set index; kSplit: preferred target; kJump: target; kLook/kAtomic: continuation
    int y       = 0;  // kSplit: alternative target
};

} // namespace

struct Pattern::Program {
    std::vector<Inst> code;
    std::vector<CharSet> sets;
};

namespace {

class Compiler {
public:
    explicit Compiler(std::vector<Inst>& code) : code_(code) {}

    void emit(const Node& node) {
        switch (node.kind) {
        case Node::Kind::kSet:
            push({Op::kSet, false, node.set, 0});
            return;
        case Node::Kind::kConcat:
            for (const Node& child : node.children) { emit(child); }
            return;
        case Node::Kind::kAlt: {
            std::vector<int> exits;
            for (std::size_t i = 0; i + 1 < node.children.size(); ++i) {
                const int split = push({Op::kSplit, false, 0, 0});
                code_[static_cast<std::size_t>(split)].x = split + 1;
                emit(node.children[i]);
                exits.push_back(push({Op::kJump, false, 0, 0}));
                code_[static_cast<std::size_t>(split)].y = size();
            }
            emit(node.children.back());
            for (const int exit : exits) { code_[static_cast<std::size_t>(exit)].x = size(); }
            return;
        }
        case Node::Kind::kLook:
        case Node::Kind::kAtomic: {
            const int head = push({node.kind == Node::Kind::kLook ? Op::kLook : Op::kAtomic,
                                   node.negate, 0, 0});
            emit(node.children[0]);
            push({Op::kMatch, false, 0, 0});
            code_[static_cast<std::size_t>(head)].x = size();
            return;
        }
        case Node::Kind::kRepeat:
            emit_repeat(node);
            return;
        }
    }

private:
    int size() const { return static_cast<int>(code_.size()); }
    int push(Inst inst) {
        code_.push_back(inst);
        return size() - 1;
    }
    void set_split(int at, int body, int skip, bool greedy) {
        Inst& inst = code_[static_cast<std::size_t>(at)];
        inst.x     = greedy ? body : skip;
        inst.y     = greedy ? skip : body;
    }

    void emit_repeat(const Node& node) {
        const Node& body = node.children[0];
        if (node.possessive) {
            // X*+ is (?>X*): match greedily, then never give anything back.
            Node greedy       = node;
            greedy.possessive = false;
            const int head    = push({Op::kAtomic, false, 0, 0});
            emit_repeat(greedy);
            push({Op::kMatch, false, 0, 0});
            code_[static_cast<std::size_t>(head)].x = size();
            return;
        }
        if (node.max == kUnbounded) {
            if (node.min == 0) {
                const int loop = push({Op::kSplit, false, 0, 0});
                emit(body);
                push({Op::kJump, false, loop, 0});
                set_split(loop, loop + 1, size(), node.greedy);
                return;
            }
            for (int i = 0; i + 1 < node.min; ++i) { emit(body); }
            const int start = size();
            emit(body);
            const int split = push({Op::kSplit, false, 0, 0});
            set_split(split, start, size(), node.greedy);
            return;
        }
        for (int i = 0; i < node.min; ++i) { emit(body); }
        // X{0,k} as nested optionals (X(X(X)?)?)?: a failed step skips all later ones.
        std::vector<int> splits;
        for (int i = node.min; i < node.max; ++i) {
            splits.push_back(push({Op::kSplit, false, 0, 0}));
            emit(body);
        }
        for (const int split : splits) { set_split(split, split + 1, size(), node.greedy); }
    }

    std::vector<Inst>& code_;
};

// Decodes one code point of valid UTF-8.
inline char32_t decode_utf8(const unsigned char* p, std::size_t& length) {
    const unsigned char b0 = p[0];
    if (b0 < 0x80) {
        length = 1;
        return b0;
    }
    if (b0 < 0xE0) {
        length = 2;
        return (static_cast<char32_t>(b0 & 0x1F) << 6) | (p[1] & 0x3F);
    }
    if (b0 < 0xF0) {
        length = 3;
        return (static_cast<char32_t>(b0 & 0x0F) << 12) |
               (static_cast<char32_t>(p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    }
    length = 4;
    return (static_cast<char32_t>(b0 & 0x07) << 18) | (static_cast<char32_t>(p[1] & 0x3F) << 12) |
           (static_cast<char32_t>(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
}

// Backtracking matcher over a decoded text. The backtrack stack is shared by nested calls
// (lookahead, atomic groups); each call only unwinds the entries it pushed.
class Matcher {
public:
    Matcher(const Pattern::Program& program, const std::vector<char32_t>& cps,
            const std::vector<std::uint32_t>& categories)
        : program_(program), cps_(cps), categories_(categories),
          size_(static_cast<int>(cps.size())) {}

    // Returns the end of the preferred match of code[pc..] starting at pos, or -1.
    int run(int pc, int pos) {
        const std::size_t base = stack_.size();
        const Inst* code       = program_.code.data();
        while (true) {
            const Inst& inst = code[pc];
            bool ok          = true;
            switch (inst.op) {
            case Op::kSet:
                if (pos < size_ && program_.sets[static_cast<std::size_t>(inst.x)].matches(
                                       cps_[static_cast<std::size_t>(pos)],
                                       categories_[static_cast<std::size_t>(pos)])) {
                    ++pos;
                    ++pc;
                    continue;
                }
                ok = false;
                break;
            case Op::kSplit:
                stack_.emplace_back(inst.y, pos);
                pc = inst.x;
                continue;
            case Op::kJump:
                pc = inst.x;
                continue;
            case Op::kMatch:
                stack_.resize(base);
                return pos;
            case Op::kLook: {
                const bool found = run(pc + 1, pos) >= 0;
                if (found != inst.negate) {
                    pc = inst.x;
                    continue;
                }
                ok = false;
                break;
            }
            case Op::kAtomic: {
                const int end = run(pc + 1, pos);
                if (end >= 0) {
                    pos = end;
                    pc  = inst.x;
                    continue;
                }
                ok = false;
                break;
            }
            }
            if (!ok) {
                if (stack_.size() == base) { return -1; }
                pc  = stack_.back().first;
                pos = stack_.back().second;
                stack_.pop_back();
            }
        }
    }

private:
    const Pattern::Program& program_;
    const std::vector<char32_t>& cps_;
    const std::vector<std::uint32_t>& categories_;
    int size_ = 0;
    std::vector<std::pair<int, int>> stack_;
};

} // namespace

Pattern::Pattern(std::string_view regex) : source_(regex) {
    auto program = std::make_shared<Program>();
    Parser parser(regex, program->sets);
    const Node root = parser.parse();
    if (nullable(root)) {
        throw std::invalid_argument("unsupported regex " + std::string(regex) +
                                    ": it can match the empty string");
    }
    Compiler(program->code).emit(root);
    program->code.push_back({Op::kMatch, false, 0, 0});
    for (CharSet& set : program->sets) { set.finalize(); }
    program_ = std::move(program);
}

Pattern Pattern::literal(std::string_view text) {
    if (text.empty()) { throw std::invalid_argument("empty literal split/replace pattern"); }
    Pattern pattern;
    pattern.source_  = std::string(text);
    pattern.literal_ = std::string(text);
    return pattern;
}

std::vector<ByteSpan> Pattern::find_all(std::string_view text) const {
    std::vector<ByteSpan> spans;
    if (!literal_.empty()) {
        for (std::size_t pos = text.find(literal_); pos != std::string_view::npos;
             pos             = text.find(literal_, pos + literal_.size())) {
            spans.push_back({pos, pos + literal_.size()});
        }
        return spans;
    }

    std::vector<char32_t> cps;
    std::vector<std::uint32_t> categories;
    std::vector<std::size_t> offsets;
    cps.reserve(text.size());
    categories.reserve(text.size());
    offsets.reserve(text.size() + 1);
    const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
    for (std::size_t offset = 0; offset < text.size();) {
        std::size_t length = 1;
        const char32_t cp  = decode_utf8(bytes + offset, length);
        cps.push_back(cp);
        // ASCII never reaches the category test (sets keep a bitmap for it).
        categories.push_back(cp < 128 ? 0U
                                      : bit(utf8proc_category(static_cast<utf8proc_int32_t>(cp))));
        offsets.push_back(offset);
        offset += length;
    }
    offsets.push_back(text.size());

    Matcher matcher(*program_, cps, categories);
    const int count = static_cast<int>(cps.size());
    for (int pos = 0; pos < count;) {
        const int end = matcher.run(0, pos);
        if (end > pos) {
            spans.push_back(
                {offsets[static_cast<std::size_t>(pos)], offsets[static_cast<std::size_t>(end)]});
            pos = end;
        } else {
            ++pos;
        }
    }
    return spans;
}

} // namespace ninferno::text
