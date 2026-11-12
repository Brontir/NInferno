// The literal rules of the generic path (ops/generic/route_rules.h), held to the sources they
// describe. Every descriptor has a unique name, a recorded boards / cards / tolerance field
// ("unknown" or "none recorded" where the record is silent, never empty) and an empty kernel hash
// (none is recorded), and a basis (Structural, Measured, PlacedBetweenMeasured, Fitted,
// ScaledFromOneBoard or Unknown: the field has no default, so a descriptor without one does not
// compile; a Measured one names a board id; a Fitted, PlacedBetweenMeasured or ScaledFromOneBoard
// one names its boards or says none are recorded; a Structural constant that is a bare number in
// the code is held by a static_assert or listed in structural_roots; the count per class is
// printed; whether a class is the right one is not checked). Its literal stands in its file,
// outside comments and outside the contents of string and character literals, exactly `occurrences`
// times, whole (a number is not the prefix of a longer one, a name not the tail of a longer one),
// whatever the whitespace between its tokens (a space between two identifier characters stays one
// space) and a trailing comma before a `}`; a literal that opens with `if(` is not found behind an
// `else`. A literal that defines a constant is complete (it ends in `;`, an array carries all its
// elements), defines the descriptor's name, is that name's only definition in the file, and
// evaluates (by the test's own evaluator, integers staying integers) to `value`, or to the number
// each element of an array says; a literal that is a condition has `value` 0 or a number written in
// it.
//
// The guard over the sources: every definition of a name kX that says it bounds something (Min,
// Max, Tokens, Fill, Distance, Values, Bytes, Gbps, Margin, Slack, Columns, Rows, Limit, Measured,
// Threshold), a `const`, `constexpr`, `static` or `inline` variable or array (an initialiser by
// `=`, braces or parentheses, a later declarator of the same declaration included) or a `#define`,
// in any .h, .hpp, .hh, .cpp, .cc, .cxx, .inc, .inl, .ipp, .tpp, .cu or .cuh under src/ops/generic
// (subdirectories included; route_cost_constants.h has its own registry and test), directly in
// src/ops, and in src/core/measured_boards.h, is keyed by file and name. A described one
// is defined once in its file; any other is an entry of kNotRules with its reason and the number of
// definitions the file holds, so a second definition of a listed name, or of a described one, is a
// failure. Known limits: a name without kX camel case or without one of those words
// (kMmqPrefillSlice in generic_ggml_internal.h is one), an enumerator, a template parameter, and a
// threshold written as a bare literal in an expression are found only where a descriptor's literal
// names them; the scanner reads text, not the preprocessor, so two definitions under #if / #else
// count as two (and fail for a described name). The selector's sources are generic_gemm.h,
// generic_gemm.cpp, generic_select.cpp (select_route and its heuristics) and route_authorities.cpp
// (the authority cascade and the NVFP4 half-the-L2 rule live there): a descriptor of a literal in
// any of them names its file. Not held by this test: the comparators at uses of a named constant
// and the order of the three authority arrays (the selector goldens hold the order;
// docs/maintainer/route-rules.md lists the rest).
// `--list` prints the descriptors, `--unlisted` the guard's findings in kNotRules' form. No GPU.

#include "ops/generic/route_rules.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <climits>
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
#include <vector>

namespace rr = ninferno::ops::detail::generic::route_rules;

namespace {

const std::filesystem::path& generic_dir() {
    static const std::filesystem::path dir =
        std::filesystem::path(NINFERNO_SOURCE_DIR) / "src/ops/generic";
    return dir;
}

// nullopt where the file cannot be read: never an empty text that counts as scanned.
std::optional<std::string> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool ident_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_'; }

bool hex_char(char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; }

// The token of identifier characters (and digit separators) that ends just before `i`: what a `'`
// at `i` follows. A digit separator stands in a number, so that token starts with a digit; a
// character literal's prefix (u8, u, U, L) starts with a letter.
std::string token_before(const std::string& t, std::size_t i) {
    std::size_t s = i;
    while (s > 0 && (ident_char(t[s - 1]) || t[s - 1] == '\'')) --s;
    return t.substr(s, i - s);
}

// The run of identifier characters that ends just before `i` (a raw string's prefix).
std::string ident_before(const std::string& t, std::size_t i) {
    std::size_t s = i;
    while (s > 0 && ident_char(t[s - 1])) --s;
    return t.substr(s, i - s);
}

// The text without comments and without the contents of string and character literals, in one pass
// as the compiler reads it: a `//` or `/*` inside a literal is literal text, not a comment; a digit
// separator (1'000) inside a number is not a character literal (u8'a' is); an `R` before a quote
// opens a raw string only as the whole prefix (R, u8R, uR, UR, LR), not as the end of a name. A
// comment becomes one space (the newlines of a line comment stay); a string, character or
// raw-string literal keeps its quotes and loses what is between them, so a constant or a threshold
// quoted in a string is not code.
std::string without_comments(const std::string& t) {
    std::string out;
    out.reserve(t.size());
    const std::size_t n = t.size();
    std::size_t i       = 0;
    while (i < n) {
        const char c = t[i];
        if (c == '/' && i + 1 < n && t[i + 1] == '/') {
            while (i < n && t[i] != '\n') {
                i += (t[i] == '\\' && i + 1 < n && t[i + 1] == '\n') ? 2 : 1;
            }
            continue;
        }
        if (c == '/' && i + 1 < n && t[i + 1] == '*') {
            const std::size_t end = t.find("*/", i + 2);
            i                     = end == std::string::npos ? n : end + 2;
            out.push_back(' ');
            continue;
        }
        const std::string raw_prefix = c == '"' && i > 0 ? ident_before(t, i) : std::string();
        if (c == '"' && (raw_prefix == "R" || raw_prefix == "u8R" || raw_prefix == "uR" ||
                         raw_prefix == "UR" || raw_prefix == "LR")) { // R"delim( ... )delim"
            const std::size_t open = t.find('(', i + 1);
            if (open != std::string::npos && open - i <= 17) {
                const std::string close = ")" + t.substr(i + 1, open - i - 1) + "\"";
                const std::size_t end   = t.find(close, open);
                const std::size_t stop  = end == std::string::npos ? n : end + close.size();
                out += "\"\""; // the `R` before it is already in `out`
                i = stop;
                continue;
            }
        }
        const bool separator = c == '\'' && i > 0 && hex_char(t[i - 1]) && i + 1 < n &&
                               hex_char(t[i + 1]) &&
                               std::isdigit(static_cast<unsigned char>(token_before(t, i)[0])) != 0;
        if (c == '"' || (c == '\'' && !separator)) {
            ++i;
            while (i < n && t[i] != c && t[i] != '\n') i += (t[i] == '\\' && i + 1 < n) ? 2 : 1;
            if (i < n && t[i] == c) ++i;
            out.push_back(c);
            out.push_back(c);
            continue;
        }
        out.push_back(c);
        ++i;
    }
    return out;
}

// The text compared: whitespace is dropped where it does not separate two identifier characters
// (one space stays there, so `a b` is not `ab`), and a trailing comma before a `}` is dropped, so a
// re-wrapped brace, a realigned `=` or a clang-format comma is not a change of the rule.
std::string squeezed(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool gap = false;
    for (const char c : text) {
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            gap = true;
            continue;
        }
        if (gap && !out.empty() && ident_char(out.back()) && ident_char(c)) out.push_back(' ');
        gap = false;
        if (c == '}' && !out.empty() && out.back() == ',') out.pop_back();
        out.push_back(c);
    }
    return out;
}

std::string trimmed(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

// How many times `needle` stands in `haystack` as a whole: not inside a longer identifier or
// number (before it or after it), so `p.k >= 8192` is not found in `p.k >= 81920`; and a needle
// that opens with `if(` is not found chained behind an `else`, where it would no longer run on its
// own (an `else if (...)` in front of a described condition is a change of the rule).
std::size_t count_of(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return 0;
    const bool head     = ident_char(needle.front());
    const bool tail     = ident_char(needle.back());
    const bool opens_if = needle.compare(0, 3, "if(") == 0;
    std::size_t n       = 0;
    for (auto at = haystack.find(needle); at != std::string::npos;
         at      = haystack.find(needle, at + 1)) {
        if (head && at > 0 && ident_char(haystack[at - 1])) continue;
        if (opens_if && at >= 5 && haystack.compare(at - 5, 5, "else ") == 0) continue;
        const std::size_t after = at + needle.size();
        if (tail && after < haystack.size() &&
            (ident_char(haystack[after]) || haystack[after] == '\'' ||
             (haystack[after] == '.' && std::isdigit(static_cast<unsigned char>(needle.back())))))
            continue;
        ++n;
    }
    return n;
}

// ---- Constants as the sources define them -------------------------------------------------------

struct Definition {
    std::string name;
    std::string rhs; // the initialiser without its `=` / braces / parentheses (arrays: with braces)
    bool array = false; // `kName[...]`
};

// The index after the group that opens at `at` (one of ( [ {), or npos.
std::size_t skip_group(const std::string& t, std::size_t at) {
    int depth = 0;
    for (std::size_t i = at; i < t.size(); ++i) {
        if (std::strchr("([{", t[i]) != nullptr) ++depth;
        if (std::strchr(")]}", t[i]) != nullptr && --depth == 0) return i + 1;
    }
    return std::string::npos;
}

bool constant_name(const std::string& s) {
    return s.size() > 1 && s[0] == 'k' && std::isupper(static_cast<unsigned char>(s[1])) != 0 &&
           std::all_of(s.begin(), s.end(), ident_char);
}

// The declarators of one declaration whose type text starts at `at`, appended to `out` (keyed by
// the position of the name, so a keyword pair like `inline constexpr` reads the declaration once).
// A declarator is `kName = init`, `kName{init}`, `kName(init)` (followed by `;` or `,`) or one of
// those after `kName[...]`; after a `,` at the declaration's own level the next one is read.
void read_declaration(const std::string& t, std::size_t at,
                      std::map<std::size_t, Definition>& out) {
    const std::size_t n = t.size();
    std::size_t p       = at;
    while (p < n && std::strchr("=;{([", t[p]) == nullptr) ++p;
    while (p < n && t[p] == '(') { // decltype(...) and alignas(...) belong to the type
        std::size_t w = p;
        while (w > at && t[w - 1] == ' ') --w;
        const bool skip = (w >= at + 8 && t.compare(w - 8, 8, "decltype") == 0) ||
                          (w >= at + 7 && t.compare(w - 7, 7, "alignas") == 0);
        if (!skip) break;
        p = skip_group(t, p);
        if (p == std::string::npos) return;
        while (p < n && std::strchr("=;{([", t[p]) == nullptr) ++p;
    }
    if (p >= n || t[p] == ';') return;
    std::size_t name_end = p;
    for (;;) {
        std::size_t e = name_end;
        while (e > at && std::isspace(static_cast<unsigned char>(t[e - 1]))) --e;
        std::size_t s = e;
        while (s > at && ident_char(t[s - 1])) --s;
        const std::string name = t.substr(s, e - s);
        if (!constant_name(name)) return;
        Definition def;
        def.name = name;
        if (t[p] == '[') {
            const std::size_t after = skip_group(t, p);
            if (after == std::string::npos) return;
            def.array = true;
            p         = after;
            while (p < n && std::isspace(static_cast<unsigned char>(t[p]))) ++p;
            if (p >= n || (t[p] != '=' && t[p] != '{')) return;
        }
        std::size_t stop = 0; // the index after the initialiser
        if (t[p] == '=') {
            std::size_t i = p + 1;
            int depth     = 0;
            for (; i < n; ++i) {
                if (std::strchr("([{", t[i]) != nullptr) ++depth;
                if (std::strchr(")]}", t[i]) != nullptr) --depth;
                if (depth == 0 && (t[i] == ';' || t[i] == ',')) break;
                if (depth < 0) return;
            }
            if (i >= n) return; // no terminator: not a complete declaration
            def.rhs = trimmed(t.substr(p + 1, i - p - 1));
            if (!def.rhs.empty() && def.rhs.front() == '{') def.array = true; // std::array = {...}
            stop = i;
        } else {
            const std::size_t close = skip_group(t, p);
            if (close == std::string::npos) return;
            std::size_t i = close;
            while (i < n && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
            if (i >= n || (t[i] != ';' && t[i] != ',')) return; // a function, not a variable
            def.rhs = t[p] == '{' && def.array ? t.substr(p, close - p)
                                               : trimmed(t.substr(p + 1, close - p - 2));
            stop    = i;
        }
        out.emplace(s, def);
        if (t[stop] != ',') return;
        // The next declarator: `, *kNext = ...`.
        p = stop + 1;
        while (p < n &&
               (std::isspace(static_cast<unsigned char>(t[p])) || t[p] == '*' || t[p] == '&')) {
            ++p;
        }
        at            = p; // the next name starts here
        std::size_t q = p;
        while (q < n && ident_char(t[q])) ++q;
        if (q == p) return;
        name_end = q;
        p        = q;
        while (p < n && std::isspace(static_cast<unsigned char>(t[p]))) ++p;
        if (p >= n || std::strchr("=[{(", t[p]) == nullptr) return;
    }
}

// Every definition in `text` (comments already removed): variables and arrays after const,
// constexpr, static or inline, and #define lines, in the order of their position.
std::vector<Definition> definitions_in(const std::string& text) {
    std::map<std::size_t, Definition> found;
    static const std::regex keyword(R"(\b(?:constexpr|const|static|inline)\b)");
    for (std::sregex_iterator it(text.begin(), text.end(), keyword), stop; it != stop; ++it) {
        read_declaration(text, static_cast<std::size_t>(it->position() + it->length()), found);
    }
    static const std::regex define_of(
        R"((?:^|\n)[ \t]*#[ \t]*define[ \t]+(k[A-Z]\w*)[ \t]+([^\n]*))");
    for (std::sregex_iterator it(text.begin(), text.end(), define_of), stop; it != stop; ++it) {
        Definition def;
        def.name = (*it)[1].str();
        def.rhs  = trimmed((*it)[2].str());
        found.emplace(static_cast<std::size_t>(it->position(1)), def);
    }
    std::vector<Definition> out;
    for (auto& [position, def] : found) out.push_back(std::move(def));
    return out;
}

// ---- The value of a constant's initialiser, by the test's own evaluator -------------------------

// An integer carries the width (32 or 64 bits) and signedness of its C++ type, so `7 / 2` is 3,
// `0u - 1` wraps and an int that does not fit is refused; a floating value is a double.
struct Number {
    bool integer   = true;
    long long i    = 0; // for an unsigned 64-bit value, the bit pattern
    double d       = 0;
    int bits       = 32;
    bool unsigned_ = false;

    double as_double() const {
        if (!integer) return d;
        return unsigned_ && bits == 64 ? static_cast<double>(static_cast<unsigned long long>(i))
                                       : static_cast<double>(i);
    }
};

Number make_int(long long v, int bits, bool is_unsigned) {
    Number x;
    x.i         = v;
    x.bits      = bits;
    x.unsigned_ = is_unsigned;
    return x;
}

// A value of the type (bits, signedness) from a wide intermediate: an unsigned one wraps, a signed
// int that does not fit is nullopt (undefined in C++, refused here).
std::optional<Number> fit(long long v, int bits, bool is_unsigned) {
    if (is_unsigned) {
        if (bits == 32) {
            v = static_cast<long long>(static_cast<unsigned long long>(v) & 0xffffffffULL);
        }
        return make_int(v, bits, true);
    }
    if (bits == 32 && (v < INT_MIN || v > INT_MAX)) return std::nullopt;
    return make_int(v, bits, false);
}

// A number token of the sources: decimal, 0x, 0b and 0 octal integers with their u / l / ll
// suffixes, decimal floating literals, ' digit separators. nullopt for what C++ does not accept
// (4fu, 08, 1e400) or this reader does not (hex floats).
std::optional<Number> parse_number(std::string token) {
    token.erase(std::remove(token.begin(), token.end(), '\''), token.end());
    static const std::regex integer_token(
        R"(^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)(?:([uU](?:ll|LL|l|L)?|(?:ll|LL|l|L)[uU]?))?$)");
    static const std::regex float_token(
        R"(^(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?[fFlL]?$)");
    std::smatch m;
    if (std::regex_match(token, m, integer_token)) {
        const std::string digits = m[1].str();
        const std::string suffix = m[2].str();
        int base                 = 10;
        std::size_t skip         = 0;
        if (digits.size() > 1 && digits[0] == '0') {
            if (digits[1] == 'x' || digits[1] == 'X') {
                base = 16;
                skip = 2;
            } else if (digits[1] == 'b' || digits[1] == 'B') {
                base = 2;
                skip = 2;
            } else {
                base = 8;
                skip = 1;
            }
        }
        errno                      = 0;
        const unsigned long long u = std::strtoull(digits.c_str() + skip, nullptr, base);
        if (errno != 0) return std::nullopt;
        const bool is_u  = suffix.find_first_of("uU") != std::string::npos;
        const bool is_l  = suffix.find_first_of("lL") != std::string::npos;
        const bool plain = base == 10;
        if (!is_u && !is_l) {
            if (u <= INT_MAX) return make_int(static_cast<long long>(u), 32, false);
            if (!plain && u <= UINT_MAX) return make_int(static_cast<long long>(u), 32, true);
            if (u <= LLONG_MAX) return make_int(static_cast<long long>(u), 64, false);
            if (!plain) return make_int(static_cast<long long>(u), 64, true);
            return std::nullopt;
        }
        if (is_u && !is_l) {
            if (u <= UINT_MAX) return make_int(static_cast<long long>(u), 32, true);
            return make_int(static_cast<long long>(u), 64, true);
        }
        if (is_u) return make_int(static_cast<long long>(u), 64, true);
        if (u <= LLONG_MAX) return make_int(static_cast<long long>(u), 64, false);
        if (!plain) return make_int(static_cast<long long>(u), 64, true);
        return std::nullopt;
    }
    if (std::regex_match(token, float_token) && token.find_first_of(".eE") != std::string::npos) {
        const double v = std::strtod(token.c_str(), nullptr);
        if (!std::isfinite(v)) return std::nullopt;
        Number x;
        x.integer = false;
        x.d       = v;
        return x;
    }
    return std::nullopt;
}

// The value of an initialiser: numbers, the names of other constants (qualified or not),
// parentheses and the operators unary + -, * / %, + -, << >> (C++ precedence, shift lowest).
// Anything else (a cast, a call, a comparison, `~`, a division by zero, a shift of a float, an int
// that overflows) is a refusal: nullopt, which the caller turns into a failure, never into a guess.
class ConstantExpression {
public:
    using Lookup = std::function<std::optional<std::string>(const std::string&)>;

    ConstantExpression(std::string text, Lookup lookup, int depth = 0)
        : text_(std::move(text)), lookup_(std::move(lookup)), depth_(depth) {}

    std::optional<Number> value() {
        const Number v = shift();
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
        // `<` must not be the start of `<=`, nor `>` of `>=`; `<<` / `>>` are taken whole.
        if (n == 1 && (token[0] == '<' || token[0] == '>') && at_ + 1 < text_.size() &&
            (text_[at_ + 1] == token[0] || text_[at_ + 1] == '=')) {
            return false;
        }
        at_ += n;
        return true;
    }

    Number refuse() {
        ok_ = false;
        return {};
    }

    Number binary(char op, const Number& a, const Number& b) {
        if (!ok_) return {};
        if (!a.integer || !b.integer) {
            if (op == '<' || op == '>' || op == '%') return refuse();
            Number r;
            r.integer      = false;
            const double x = a.as_double();
            const double y = b.as_double();
            switch (op) {
            case '*':
                r.d = x * y;
                break;
            case '/':
                if (y == 0) return refuse();
                r.d = x / y;
                break;
            case '+':
                r.d = x + y;
                break;
            default:
                r.d = x - y;
                break;
            }
            return std::isfinite(r.d) ? r : refuse();
        }
        if (op == '<' || op == '>') { // the result has the left operand's type
            if (b.i < 0 || b.i >= a.bits) return refuse();
            if (op == '>') {
                const long long v =
                    a.unsigned_ ? static_cast<long long>((static_cast<unsigned long long>(a.i) &
                                                          (a.bits == 32 ? 0xffffffffULL : ~0ULL)) >>
                                                         b.i)
                                : a.i >> b.i;
                return make_int(v, a.bits, a.unsigned_);
            }
            if (!a.unsigned_ && a.i < 0) return refuse();
            const unsigned long long shifted = static_cast<unsigned long long>(a.i) << b.i;
            if (a.unsigned_) {
                const auto r = fit(static_cast<long long>(shifted), a.bits, true);
                return r ? *r : refuse();
            }
            const long long v = static_cast<long long>(shifted);
            if ((v >> b.i) != a.i) return refuse(); // overflow
            const auto r = fit(v, a.bits, false);
            return r ? *r : refuse();
        }
        const int bits     = std::max(a.bits, b.bits);
        const bool is_uns  = (a.unsigned_ && a.bits == bits) || (b.unsigned_ && b.bits == bits);
        const auto promote = [&](const Number& x) {
            // The operand in the common type: an unsigned 32-bit operand of a 64-bit type is
            // just its value; a negative signed operand of an unsigned type wraps.
            if (is_uns && bits == 32) return static_cast<long long>(static_cast<unsigned int>(x.i));
            return x.i;
        };
        const long long x = promote(a);
        const long long y = promote(b);
        if (is_uns) {
            const unsigned long long ux = static_cast<unsigned long long>(x);
            const unsigned long long uy = static_cast<unsigned long long>(y);
            unsigned long long r        = 0;
            switch (op) {
            case '*':
                r = ux * uy;
                break;
            case '+':
                r = ux + uy;
                break;
            case '-':
                r = ux - uy;
                break;
            default: {
                const unsigned long long mx = bits == 32 ? ux & 0xffffffffULL : ux;
                const unsigned long long my = bits == 32 ? uy & 0xffffffffULL : uy;
                if (my == 0) return refuse();
                r = op == '/' ? mx / my : mx % my;
            }
            }
            const auto f = fit(static_cast<long long>(r), bits, true);
            return f ? *f : refuse();
        }
        long long r = 0;
        switch (op) {
        case '+':
            if ((y > 0 && x > LLONG_MAX - y) || (y < 0 && x < LLONG_MIN - y)) return refuse();
            r = x + y;
            break;
        case '-':
            if ((y < 0 && x > LLONG_MAX + y) || (y > 0 && x < LLONG_MIN + y)) return refuse();
            r = x - y;
            break;
        case '*':
            r = static_cast<long long>(static_cast<unsigned long long>(x) *
                                       static_cast<unsigned long long>(y));
            if (x != 0 && (r / x != y || (x == -1 && y == LLONG_MIN))) return refuse();
            break;
        default:
            if (y == 0 || (x == LLONG_MIN && y == -1)) return refuse();
            r = op == '/' ? x / y : x % y; // truncating toward zero, as C++ does
        }
        const auto f = fit(r, bits, false);
        return f ? *f : refuse();
    }

    Number shift() {
        Number v = add();
        while (ok_) {
            if (take("<<"))
                v = binary('<', v, add());
            else if (take(">>"))
                v = binary('>', v, add());
            else
                break;
        }
        return v;
    }

    Number add() {
        Number v = mul();
        while (ok_) {
            if (take("+"))
                v = binary('+', v, mul());
            else if (take("-"))
                v = binary('-', v, mul());
            else
                break;
        }
        return v;
    }

    Number mul() {
        Number v = unary();
        while (ok_) {
            if (take("*"))
                v = binary('*', v, unary());
            else if (take("/"))
                v = binary('/', v, unary());
            else if (take("%"))
                v = binary('%', v, unary());
            else
                break;
        }
        return v;
    }

    Number unary() {
        skip();
        if (take("-")) {
            const Number v = unary();
            if (!v.integer) {
                Number r = v;
                r.d      = -v.d;
                return r;
            }
            return binary('-', make_int(0, v.bits, v.unsigned_), v);
        }
        if (take("+")) return unary();
        return primary();
    }

    Number primary() {
        skip();
        if (at_ >= text_.size()) return refuse();
        if (take("(")) {
            const Number v = shift();
            if (!take(")")) return refuse();
            return v;
        }
        const char c = text_[at_];
        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && at_ + 1 < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[at_ + 1])))) {
            std::size_t stop = at_;
            const bool hex   = text_.compare(at_, 2, "0x") == 0 || text_.compare(at_, 2, "0X") == 0;
            while (stop < text_.size() &&
                   (ident_char(text_[stop]) || text_[stop] == '.' || text_[stop] == '\'' ||
                    (!hex && (text_[stop] == '+' || text_[stop] == '-') &&
                     (text_[stop - 1] == 'e' || text_[stop - 1] == 'E')))) {
                ++stop;
            }
            const auto v = parse_number(text_.substr(at_, stop - at_));
            at_          = stop;
            return v ? *v : refuse();
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::size_t stop = at_;
            std::string name;
            for (;;) { // a qualified name: the last component is the constant
                std::size_t s = stop;
                while (stop < text_.size() && ident_char(text_[stop])) ++stop;
                name = text_.substr(s, stop - s);
                if (text_.compare(stop, 2, "::") == 0 && stop + 2 < text_.size() &&
                    (std::isalpha(static_cast<unsigned char>(text_[stop + 2])) ||
                     text_[stop + 2] == '_')) {
                    stop += 2;
                    continue;
                }
                break;
            }
            at_                   = stop;
            const auto definition = depth_ < 8 ? lookup_(name) : std::nullopt;
            const auto v = definition ? ConstantExpression(*definition, lookup_, depth_ + 1).value()
                                      : std::nullopt;
            return v ? *v : refuse();
        }
        return refuse();
    }

    std::string text_;
    Lookup lookup_;
    int depth_;
    std::size_t at_ = 0;
    bool ok_        = true;
};

// The elements of a braced initialiser, split at its own level's commas.
std::vector<std::string> elements_of(const std::string& braced) {
    std::vector<std::string> out;
    if (braced.size() < 2 || braced.front() != '{' || braced.back() != '}') return out;
    int depth        = 0;
    std::string item = {};
    for (std::size_t i = 1; i + 1 < braced.size(); ++i) {
        const char c = braced[i];
        if (std::strchr("([{", c) != nullptr) ++depth;
        if (std::strchr(")]}", c) != nullptr) --depth;
        if (c == ',' && depth == 0) {
            out.push_back(trimmed(item));
            item.clear();
        } else {
            item.push_back(c);
        }
    }
    if (!trimmed(item).empty()) out.push_back(trimmed(item));
    return out;
}

// The numbers written in a literal (decimal, hex, float tokens), for a condition's `value`.
std::vector<double> numbers_in(const std::string& literal) {
    std::vector<double> out;
    for (std::size_t i = 0; i < literal.size();) {
        if (std::isdigit(static_cast<unsigned char>(literal[i])) &&
            (i == 0 || !ident_char(literal[i - 1]))) {
            std::size_t stop = i;
            const bool hex   = literal.compare(i, 2, "0x") == 0 || literal.compare(i, 2, "0X") == 0;
            while (stop < literal.size() &&
                   (ident_char(literal[stop]) || literal[stop] == '.' || literal[stop] == '\'' ||
                    (!hex && (literal[stop] == '+' || literal[stop] == '-') &&
                     (literal[stop - 1] == 'e' || literal[stop - 1] == 'E')))) {
                ++stop;
            }
            if (const auto v = parse_number(literal.substr(i, stop - i)))
                out.push_back(v->as_double());
            i = stop;
        } else {
            ++i;
        }
    }
    return out;
}

// ---- The sources --------------------------------------------------------------------------------

struct Source {
    bool readable = false;
    std::string raw;      // without comments, newlines kept
    std::string squeezed; // the same, whitespace runs as one space
    std::vector<Definition> defs;
};

const Source& source_of(const std::string& file) {
    static std::map<std::string, Source> cache;
    auto it = cache.find(file);
    if (it != cache.end()) return it->second;
    Source s;
    if (const auto text = read_file(generic_dir() / file)) {
        s.readable = true;
        s.raw      = without_comments(*text);
        s.squeezed = squeezed(s.raw);
        s.defs     = definitions_in(s.raw);
    }
    return cache.emplace(file, std::move(s)).first->second;
}

std::size_t definition_count(const std::string& file, const std::string& name) {
    const Source& s = source_of(file);
    return static_cast<std::size_t>(std::count_if(
        s.defs.begin(), s.defs.end(), [&](const Definition& d) { return d.name == name; }));
}

// The files a constant's initialiser may name constants of, after the file that holds it.
const std::vector<std::string>& lookup_files() {
    static const std::vector<std::string> files = {"generic_gemm.h", "generic_gemm_internal.h",
                                                   "../route_pins.h", "../../core/units.h"};
    return files;
}

// The initialiser of the one definition of `name` in `file`, else in the shared files; nothing
// where it is defined twice in the first file that defines it (the scan reads no preprocessor).
std::optional<std::string> initialiser_of(const std::string& file, const std::string& name) {
    std::vector<std::string> order = {file};
    order.insert(order.end(), lookup_files().begin(), lookup_files().end());
    for (const std::string& f : order) {
        const Source& s         = source_of(f);
        const Definition* found = nullptr;
        std::size_t count       = 0;
        for (const Definition& d : s.defs) {
            if (d.name == name) {
                found = &d;
                ++count;
            }
        }
        if (count == 1 && !found->array) return squeezed(found->rhs);
        if (count != 0) return std::nullopt;
    }
    return std::nullopt;
}

std::optional<Number> evaluate(const std::string& file, const std::string& rhs) {
    return ConstantExpression(squeezed(rhs),
                              [&](const std::string& other) { return initialiser_of(file, other); })
        .value();
}

// Names the scan finds that no selection compares a call against, keyed by file and name, with
// the number of definitions the file holds and the reason.
struct NotRule {
    std::string_view file;
    std::string_view name;
    std::size_t count;
    std::string_view reason;
};

const std::vector<NotRule>& not_rules() {
    static const std::vector<NotRule> v = {
        {"generic_codec_traits.cu", "kRows", 1,
         "the table of codec trait rows the launch files instantiate (a registry, no bound)"},
        {"generic_gemm.cuh", "kBlockBytes", 1, "a codec's block size (a layout fact)"},
        {"generic_gemm.cuh", "kGroupBytes", 1, "a codec's group size (a layout fact)"},
        {"generic_gemm.cuh", "kIq1MinusHigh", 1, "an IQ1 codebook offset (a format fact)"},
        {"generic_gemm.cuh", "kIq1MinusLow", 1, "an IQ1 codebook offset (a format fact)"},
        {"generic_gemm.cuh", "kMinBlocks", 1, "a launch bound: resident blocks a kernel asks for"},
        {"generic_gemm.cuh", "kMinTerm", 3, "a kernel's own accumulation term (a codec fact)"},
        {"generic_gemm.cuh", "kRmmaMinBlocks", 1,
         "a launch bound: the registers a kernel is capped at"},
        {"generic_gemm.cuh", "kRowsPerCta", 1, "a kernel's tile rows (a tile shape, counted)"},
        {"generic_gemm.cuh", "kTokensPerCta", 1, "a kernel's tile tokens (a tile shape, counted)"},
        {"generic_gemm.cuh", "kWithMin", 1, "a codec flag name (the format carries a min)"},
        {"generic_gemm.h", "kMaxSegments", 1,
         "the segmented projection's slot count (an array size)"},
        {"generic_ggml_internal.h", "kGgmlArenaSlack", 1, "the ggml arena's sizing"},
        {"generic_i8.cu", "kRingBytes", 1, "a kernel's load ring size (counted)"},
        {"generic_i8.cu", "kRowsPerBlock", 1, "a kernel's tile rows (a tile shape, counted)"},
        {"generic_i8.cu", "kSmemBytes", 1, "a kernel's shared memory use (counted)"},
        {"generic_i8.cu", "kStageBytes", 1, "a kernel's pipeline stage size (counted)"},
        {"generic_r8.cuh", "kHalfMin", 1, "the smallest normal FP16 value"},
        {"generic_r8.cuh", "kRows", 2, "a kernel's tile rows (a tile shape, counted)"},
        {"generic_r8.cuh", "kRowsPerCta", 1, "a kernel's tile rows (a tile shape, counted)"},
        {"generic_shared_bytes.h", "kStagedSharedBytes", 1,
         "the staged layouts' shared memory use (counted: sizeof, static_asserted by the kernel)"},
        {"generic_shared_bytes.h", "kTiledSharedBytes", 1,
         "the tiled layouts' shared memory use (counted: sizeof, static_asserted by the kernel)"},
        {"generic_shared_bytes.h", "kTiledSharedBytesTuring", 1,
         "the tiled layouts' shared memory use on Turing (counted: sizeof, static_asserted by the "
         "kernel)"},
        {"generic_staged.cu", "kRowsPerBlock", 1, "a kernel's tile rows (a tile shape, counted)"},
        {"generic_staged.cuh", "kRows", 1, "a kernel's tile rows (a tile shape, counted)"},
        {"generic_staged.cuh", "kSetRows", 2, "a kernel's rows per set (a tile shape)"},
        {"generic_staged.cuh", "kTokens", 1, "a kernel's tile tokens (a tile shape, counted)"},
        {"generic_staged.cuh", "kWarpRows", 1, "a kernel's rows per warp (a tile shape)"},
        {"generic_tiled.cu", "kRowsPerBlock", 1, "a kernel's tile rows (a tile shape, counted)"},
        {"generic_tiled.cuh", "kBlockBytes", 1, "a codec's block size (a layout fact)"},
        {"generic_tiled.cuh", "kGroupBytes", 1, "a codec's group size (a layout fact)"},
        {"generic_tiled.cuh", "kMinTerm", 5, "a kernel's own accumulation term (a codec fact)"},
        {"generic_volta_cutlass_sizes.cpp", "kChunkWeightBytes", 1,
         "Volta's decode chunk: 140 MiB of FP16 (a scratch size, derived in the comment from the "
         "memory it may take)"},
        {"generic_volta_gemv.cu", "kBlockBytes", 1, "a codec's block size (a layout fact)"},
        {"generic_volta_hmma.cuh", "kBlockBytes", 1, "a codec's block size (a layout fact)"},
        {"generic_volta_hmma.cuh", "kCtaRows", 1, "a kernel's rows per CTA (a tile shape)"},
        {"generic_volta_hmma.cuh", "kMin", 6,
         "a template parameter of a kernel (does the codec carry a min)"},
        {"generic_volta_hmma.cuh", "kWarpRows", 1, "a kernel's rows per warp (a tile shape)"},
        {"generic_w4a4.cu", "kRowsPerBlock", 1, "a kernel's tile rows (a tile shape, counted)"},
        {"generic_w4a4.cuh", "kCodeRowBytes", 1, "a codec's row size (a layout fact)"},
        {"generic_w4a8.cuh", "kBufferBytes", 1, "a kernel's shared buffer size"},
        {"generic_w4a8.cuh", "kBytes", 3, "a kernel's layout size in bytes (counted)"},
        {"generic_w4a8.cuh", "kGroupBytes", 1, "a codec's group size (a layout fact)"},
        {"generic_w4a8.cuh", "kHalfMin", 2, "the smallest normal FP16 value"},
        {"generic_w4a8.cuh", "kIq1Minus", 1, "an IQ1 codebook offset (a format fact)"},
        {"generic_w4a8.cuh", "kMin", 22,
         "a template parameter of a kernel (does the codec carry a min)"},
        {"generic_w4a8.cuh", "kRowsPerBlock", 1, "a kernel's tile rows (a tile shape, counted)"},
        {"generic_w4a8.cuh", "kSmemBytes", 1, "a kernel's shared memory use (counted)"},
        {"generic_w4a8.cuh", "kW4a8MaxDynamicSmem", 1,
         "the architecture's dynamic shared memory limit"},
        {"generic_w4a8.cuh", "kWithMin", 1, "a codec flag name (the format carries a min)"},
        {"ggml_iq_tables.cuh", "kIq4nlValues", 1, "the IQ4_NL codebook (a format fact)"},
        {"ggml_mmq/ggml_mmq.cu", "kHalfMax", 1, "the largest finite FP16 value"},
        {"ggml_mmq/ggml_mmq.cu", "kValuesPerBlock", 1,
         "a codec's values per block (a layout fact)"},
        {"ggml_mmq/ggml_mmq.h", "kActivationSlack", 1, "ggml_mmq's arena sizing"},
        {"ggml_mmq/ggml_mmq.h", "kHalf2Bytes", 1, "a half2's size (a layout fact)"},
        {"ggml_mmq/ggml_mmq.h", "kQ8MmqBlockBytes", 1,
         "ggml's q8_1 tile block size (a layout fact)"},
        {"ggml_mmq/ggml_mmq.h", "kQ8MmqValues", 1,
         "ggml's q8_1 tile values per block (a layout fact)"},
        {"ggml_mmq/ggml_mmq.h", "kVecMaxSegments", 1, "a kernel's argument slots (an array size)"},
        {"ggml_mmq/ggml_vec.cu", "kBlockValues", 18, "a codec's values per block (a layout fact)"},
        {"ggml_mmq/ggml_vec.cu", "kHalfMax", 1, "the largest finite FP16 value"},
        {"ggml_mmq/ggml_vec.cu", "kRows", 1, "a kernel's tile rows (a tile shape, counted)"},
        {"kernel_facts.cpp", "kColumns", 1,
         "the kernel facts file's column header (a file format)"},
        {"kernel_facts.cpp", "kLoopColumns", 1,
         "the K-loop count columns of the kernel facts header (a file format)"},
        {"kernel_facts.cpp", "kPipeColumns", 1,
         "a family section's per-pipe count columns (a file format)"},
        {"kernel_facts.cpp", "kResourceColumnsLength", 1,
         "the length of the resources-only column header (a file format)"},
        {"rmma_geometry.h", "kTileBytes", 1,
         "the bytes of one accumulator tile, 4 floats a lane (a layout fact)"},
        {"scratch_lifetime.h", "kMaxScratchUses", 1, "scratch lifetime bookkeeping"},
        {"selector_facts.cpp", "kExactMinor", 2, "a build switch (an 'a' / 'f' target's minor)"},
        {"selector_facts.cpp", "kMinor", 1, "the build architecture's minor (a build switch)"},
        {"selector_facts.cpp", "kTokens", 1, "a kernel's tile tokens (a tile shape, counted)"},
        {"../weight_input.cpp", "kGemma4GateUpRows", 1,
         "a model's expert-plane rows (a layout fact of Gemma 4, not a selection threshold)"},

    };
    return v;
}

// Structural constants that are a bare number in the code and held by no static_assert, each with
// the reason the class is Structural: the test requires every other Structural constant to be
// defined from another constant or named in a static_assert of the scanned sources, so a number
// that only a comment calls structural does not pass as one.
const std::map<std::string_view, std::string_view>& structural_roots() {
    static const std::map<std::string_view, std::string_view> m = {
        {"kDecodePinMaxTokens",
         "the one definition of the widest speculative verify (a design width): "
         "kPrefillPinMinTokens is defined from it and held to 17 by a static_assert in "
         "route_pins.h"},
        {"kVoltaHmmaMinTokens",
         "a byte-consistency contract: a one-token call must run the kernel a wider call runs "
         "(test_moe)"},
        {"kSlotWarps", "the register file in warps of 128 registers a thread: 65536 / (32 * 128)"},
        {"kVecWarps", "the vector kernel's CTA size in warps, a count of the kernel"},
        {"kVecBoundedBlocks",
         "resident CTAs the 72-register instance holds on a 64K-register SM: 65536 / (128 * 72)"},
    };
    return m;
}

bool matches_bounding(const std::string& name) {
    static const std::regex bounding(
        "Min|Max|Tokens|Fill|Distance|Values|Bytes|Gbps|Margin|Slack|Columns|Rows|Limit|Measured|"
        "Threshold");
    return std::regex_search(name, bounding);
}

// ---- The evaluator's own checks -----------------------------------------------------------------

void evaluator_self_test(const std::function<void(const std::string&)>& fail) {
    const auto lookup = [](const std::string& name) -> std::optional<std::string> {
        if (name == "kEight") return std::string("8");
        if (name == "kBig") return std::string("2147483647");
        return std::nullopt;
    };
    const auto eval = [&](const std::string& text) {
        return ConstantExpression(text, lookup).value();
    };
    const auto expect = [&](const std::string& text, double want) {
        const auto v = eval(text);
        if (!v || v->as_double() != want) {
            fail("evaluator: '" + text + "' should be " + std::to_string(want) + ", is " +
                 (v ? std::to_string(v->as_double()) : std::string("refused")));
        }
    };
    const auto refused = [&](const std::string& text) {
        if (eval(text)) fail("evaluator: '" + text + "' should be refused");
    };
    expect("7 / 2", 3);
    expect("-7 / 2", -3);
    expect("-7 % 3", -1);
    expect("(1 << 20) / 3", 349525);
    expect("7.0 / 2", 3.5);
    expect("1'000'000", 1000000);
    expect("0b101", 5);
    expect("010", 8);
    expect("0x1F", 31);
    expect("0x1e+3", 33); // hex 0x1e plus 3, not an exponent
    expect("5 % 3", 2);
    expect("1024 >> 3", 128);
    expect("+4", 4);
    expect("24ll << 20", 25165824);
    expect("2.5e11", 2.5e11);
    expect("0u - 1", 4294967295.0);
    expect("ns::kEight * 2", 16);
    expect("kEight + kEight", 16);
    refused("1.5 << 2");
    refused("4fu");
    refused("08");
    refused("1e400");
    refused("1 / 0");
    refused("1.0 / 0");
    refused("kBig + 1"); // an int overflow
    refused("1 << 40");  // past the int's width
    refused("static_cast<int>(3)");
    refused("1 < 2");
    refused("kUnknown");
    refused("(1 + 2");
}

// The text reader's own checks: comments gone, the contents of string, character and raw-string
// literals gone with their quotes kept, digit separators left alone, whitespace normalised.
void text_self_test(const std::function<void(const std::string&)>& fail) {
    const auto same = [&](const std::string& in, const std::string& want) {
        const std::string got = squeezed(without_comments(in));
        if (got != squeezed(want)) {
            fail("text: '" + in + "' reads as '" + got + "', should be '" + squeezed(want) + "'");
        }
    };
    same("int a = 1; // k = 3\nint b;", "int a = 1; int b;");
    same("int a; /* \"x */ int b;", "int a; int b;");
    same(R"t(const char* s = "kFoo = 4; // no";)t", R"t(const char* s = "";)t");
    same(R"t(const char* s = "a\"b // c"; int kMaxA = 5;)t",
         R"t(const char* s = ""; int kMaxA = 5;)t");
    same(R"t(char c = '"'; int kMaxB = 6;)t", R"t(char c = ''; int kMaxB = 6;)t");
    same(R"t(char c = '\''; int kMaxC = 7;)t", R"t(char c = ''; int kMaxC = 7;)t");
    same(R"t(auto r = R"x(// "k = 1;" )x"; int kMaxD = 8;)t", R"t(auto r = R""; int kMaxD = 8;)t");
    same("int v = 1'000; // c", "int v = 1'000;");
    same("int v = 0xAB'CD'EF; // c", "int v = 0xAB'CD'EF;");
    same("auto c = u8'a'; int kMaxE = 9; // c", "auto c = u8''; int kMaxE = 9;");
    same("auto c = L'1'; int kMaxF = 9;", "auto c = L''; int kMaxF = 9;");
    same(R"t(NINFERNO_ERR"(x"; int kMaxG = 1;)t", R"t(NINFERNO_ERR""; int kMaxG = 1;)t");
    same(R"t(auto r = u8R"x(")" )x"; int kMaxH = 2;)t", R"t(auto r = u8R""; int kMaxH = 2;)t");
    if (count_of(squeezed("a; else if (x) y;"), squeezed("if (x) y;")) != 0)
        fail("text: an `else if` in front of a described condition must not match it");
    if (count_of(squeezed("a; if (x) y;"), squeezed("if (x) y;")) != 1)
        fail("text: a described condition on its own must match");
    same("a  =\n  1 ;", "a=1;");
    same("{ {1, 2},\n }", "{{1,2}}");
    if (squeezed("int  x") != "int x")
        fail("text: a space between identifiers must stay one space");
    if (squeezed("int x") == squeezed("intx")) fail("text: 'int x' must differ from 'intx'");
    for (const Definition& d :
         definitions_in(without_comments(R"t(const char* s = "constexpr int kMaxZ = 4;";)t"))) {
        if (d.name == "kMaxZ") fail("text: a definition quoted in a string is read as code");
    }
    bool seen = false;
    for (const Definition& d : definitions_in(without_comments("constexpr int kMaxY = 4; // c"))) {
        seen = seen || d.name == "kMaxY";
    }
    if (!seen) fail("text: a definition in code is not read");
}

} // namespace

int main(int argc, char** argv) {
    int failures    = 0;
    const auto fail = [&](const std::string& message) {
        std::fprintf(stderr, "%s\n", message.c_str());
        ++failures;
    };
    evaluator_self_test(fail);
    text_self_test(fail);
    std::array<std::size_t, rr::kAllBases.size()> by_basis{};
    std::set<std::string> names;
    std::set<std::pair<std::string, std::string>> described; // (file, name) of the named constants
    // Structural constants whose definition is a bare number: held by a static_assert, or listed.
    std::vector<std::pair<std::string, std::string>> structural_bare;
    for (const auto& rule : rr::kRouteRules) {
        const std::string name(rule.name);
        const std::string file_name(rule.file);
        if (name.empty() || !names.insert(name).second) {
            fail("rule name '" + name + "' empty or listed twice");
        }
        if (rule.literal.empty()) fail(name + ": no literal");
        for (const auto field : {rule.boards, rule.cards, rule.tolerance}) {
            if (field.empty()) fail(name + ": a boards / cards / tolerance field is empty");
        }
        // The basis: one of the classes (the field has no default, so a descriptor without one does
        // not compile); a fit or a value placed between boards names its boards or says none are
        // recorded; the class is the field, not a word in the prose.
        const auto basis_index = static_cast<std::size_t>(rule.basis);
        if (basis_index >= by_basis.size()) {
            fail(name + ": a basis that is none of the classes");
        } else {
            ++by_basis[basis_index];
        }
        if (rule.tolerance.find("FITTED") != std::string_view::npos) {
            fail(name + ": FITTED in the tolerance prose; the basis field says it");
        }
        // A Measured entry names the board it was measured on (a class id such as gb202:170sm);
        // a Fitted, PlacedBetweenMeasured or ScaledFromOneBoard one names its boards or says none
        // are recorded. Whether the class is the right one stays a reading of the source.
        static const std::regex board_id("[a-z]{2}[0-9]{3}:[0-9]+sm");
        const std::string boards(rule.boards);
        const bool names_board = std::regex_search(boards, board_id);
        if (rule.basis == rr::Basis::Measured && !names_board) {
            fail(name + ": Measured names no board (a class id such as gb202:170sm): a timing "
                        "on no named board is Unknown");
        }
        if (rule.basis == rr::Basis::Fitted || rule.basis == rr::Basis::PlacedBetweenMeasured ||
            rule.basis == rr::Basis::ScaledFromOneBoard) {
            if (!names_board && boards.find("unknown") == std::string::npos &&
                boards.find("none recorded") == std::string::npos &&
                boards.find("see k") == std::string::npos) {
                fail(name + ": " + std::string(rr::basis_name(rule.basis)) +
                     " names no board (a class id such as gb202:170sm) and does not say none is "
                     "recorded");
            }
        }
        if (!rule.kernel_hash.empty()) {
            fail(name +
                 ": a kernel hash is set, but none is recorded anywhere to check it against");
        }
        const Source& file = source_of(file_name);
        if (!file.readable) {
            fail(name + ": cannot read " + file_name);
            continue;
        }
        const std::string literal = squeezed(without_comments(std::string(rule.literal)));
        const std::size_t found   = count_of(file.squeezed, literal);
        if (found != rule.occurrences) {
            fail(name + ": the literal '" + std::string(rule.literal) + "' stands " +
                 std::to_string(found) + " times, whole, outside comments in " + file_name +
                 ", the descriptor says " + std::to_string(rule.occurrences));
        }
        // A literal that defines a constant: complete, the rule's own name, defined once in the
        // file, and its number the descriptor's (every element of an array, an array's value 0).
        const auto literal_defs = definitions_in(literal);
        // A literal that opens like a declaration of a kName (any const / constexpr / static /
        // inline form, an initialiser or none) or a #define of one must read as one definition.
        static const std::regex declares(
            R"(^(?:#define k[A-Z]|(?:(?:inline|static|constexpr|const) )+[^=;{(]*\bk[A-Z]))");
        const bool says_definition = std::regex_search(literal, declares);
        if (says_definition && literal_defs.size() != 1) {
            fail(name + ": a definition literal that is not one whole definition (a missing ';', "
                        "or a form the test cannot read)");
        } else if (literal_defs.size() == 1) {
            const Definition& def = literal_defs.front();
            if (def.name != name) fail(name + ": its literal defines " + def.name);
            if (definition_count(file_name, def.name) != 1) {
                fail(name + ": " + def.name + " is defined " +
                     std::to_string(definition_count(file_name, def.name)) + " times in " +
                     file_name + " (the test reads text, not the preprocessor)");
            }
            if (def.array) {
                if (rule.value != 0)
                    fail(name + ": an array has value 0, its elements are literal");
                const auto items = elements_of(def.rhs);
                if (items.empty()) fail(name + ": cannot read the array's initialiser");
                // A flat array of numbers is evaluated element by element; an array of
                // aggregates (kSpreadMeasured) is held by its whole literal alone.
                const bool aggregate = !items.empty() && items.front().front() == '{';
                for (const auto& item : items) {
                    if (!aggregate && !evaluate(file_name, item)) {
                        fail(name + ": cannot evaluate the element '" + item + "'");
                    }
                }
            } else {
                const auto defined = evaluate(file_name, def.rhs);
                if (!defined) {
                    fail(name + ": cannot evaluate the source's '" + def.rhs + "'");
                } else if (std::fabs(defined->as_double() - rule.value) >
                           1e-12 * std::max(1.0, std::fabs(rule.value))) {
                    fail(name + ": the source says " + def.rhs + " (" +
                         std::to_string(defined->as_double()) + "), the descriptor " +
                         std::to_string(rule.value));
                }
            }
            if (rule.basis == rr::Basis::Structural && !def.array) {
                static const std::regex names_constant(R"(\bk[A-Z]\w*)");
                if (!std::regex_search(def.rhs, names_constant)) {
                    structural_bare.emplace_back(file_name, name);
                }
            }
            if (constant_name(name)) described.insert({file_name, name});
        } else {
            // A condition: `value` is 0 (no single cutoff) or a number written in the literal.
            if (constant_name(name))
                fail(name + ": a constant's name, but its literal defines nothing");
            if (rule.value != 0) {
                const auto written = numbers_in(literal);
                const bool stands  = std::any_of(written.begin(), written.end(), [&](double v) {
                    return std::fabs(v - rule.value) <= 1e-12 * std::max(1.0, std::fabs(v));
                });
                if (!stands) {
                    fail(name + ": value " + std::to_string(rule.value) +
                         " is not a number the literal writes (use 0 for a condition without one "
                         "cutoff)");
                }
            }
        }
    }
    // The guard: every definition of a bounding name under src/ops/generic is a described rule or
    // a listed non-rule, by file and name, with the file's number of definitions.
    std::map<std::pair<std::string, std::string>, std::size_t> seen; // (file, name) -> definitions
    std::size_t scanned = 0;
    // Everything under src/ops/generic, and the two files beside it that hold constants the
    // descriptors point at (the pins' band edge and the route table's first prefill width).
    std::vector<std::string> scan_files = {"../../core/measured_boards.h"};
    const auto scanned_extension        = [](const std::filesystem::path& ext) {
        return ext == ".h" || ext == ".hpp" || ext == ".hh" || ext == ".cpp" || ext == ".cc" ||
               ext == ".cxx" || ext == ".inc" || ext == ".inl" || ext == ".ipp" || ext == ".tpp" ||
               ext == ".cu" || ext == ".cuh";
    };
    for (const auto& item : std::filesystem::recursive_directory_iterator(generic_dir())) {
        if (!item.is_regular_file() || !scanned_extension(item.path().extension())) continue;
        const std::string file =
            std::filesystem::relative(item.path(), generic_dir()).generic_string();
        if (file == "route_rules.h" || file == "route_cost_constants.h") continue;
        scan_files.push_back(file);
    }
    // The files directly in src/ops beside the generic directory (route pins, the route table, the
    // prefill routes, a new selector header put there).
    for (const auto& item : std::filesystem::directory_iterator(generic_dir() / "..")) {
        if (!item.is_regular_file() || !scanned_extension(item.path().extension())) continue;
        scan_files.push_back("../" + item.path().filename().generic_string());
    }
    for (const std::string& file : scan_files) {
        const Source& s = source_of(file);
        if (!s.readable) {
            fail(file + ": cannot read a scanned file");
            continue;
        }
        ++scanned;
        for (const Definition& def : s.defs) {
            if (matches_bounding(def.name)) ++seen[{file, def.name}];
        }
    }
    if (scanned < 20 || seen.size() < 30) fail("the constant scan found too little");
    // A Structural constant defined by a bare number: a static_assert of the scanned sources names
    // it, or it is listed with its reason; and a listed one is still such a constant.
    std::set<std::string> bare_names;
    for (const auto& [file, name] : structural_bare) {
        bare_names.insert(name);
        const std::regex assertion("static_assert\\s*\\([^;]*\\b" + name + "\\b");
        bool held = structural_roots().count(name) != 0;
        for (const std::string& f : scan_files) {
            if (held) break;
            held = std::regex_search(source_of(f).raw, assertion);
        }
        if (!held) {
            fail(name + ": Structural, but defined by a bare number and held by no static_assert "
                        "in the scanned sources (define it from another constant, add one, or "
                        "list it in structural_roots with its reason; otherwise it is Unknown)");
        }
    }
    for (const auto& [name, reason] : structural_roots()) {
        if (bare_names.count(std::string(name)) == 0) {
            fail(std::string(name) + " is in structural_roots but is not a Structural constant "
                                     "defined by a bare number");
        }
    }
    std::map<std::pair<std::string, std::string>, const NotRule*> listed;
    for (const NotRule& n : not_rules()) {
        const auto key = std::make_pair(std::string(n.file), std::string(n.name));
        if (!listed.emplace(key, &n).second) fail(key.first + " " + key.second + " listed twice");
        if (described.count(key) != 0) {
            fail(key.second + " (" + key.first + ") is both a rule and in kNotRules");
        }
        const auto it = seen.find(key);
        if (it == seen.end()) {
            fail(key.second + " (" + key.first + ") is in kNotRules but no longer defined");
        } else if (it->second != n.count) {
            fail(key.second + " (" + key.first + ") is defined " + std::to_string(it->second) +
                 " times, kNotRules says " + std::to_string(n.count));
        }
    }
    const bool list_unlisted = argc > 1 && std::string(argv[1]) == "--unlisted";
    for (const auto& [key, count] : seen) {
        if (described.count(key) != 0) continue; // defined once: checked with its descriptor
        if (listed.count(key) != 0) continue;
        if (list_unlisted) {
            std::printf("    {\"%s\", \"%s\", %zu, \"\"},\n", key.first.c_str(), key.second.c_str(),
                        count);
        } else {
            fail(key.second + " (" + key.first +
                 ") is a bounding constant of the selection code without a descriptor in "
                 "kRouteRules (or an entry of kNotRules)");
        }
    }
    if (argc > 1 && std::string(argv[1]) == "--list") {
        for (const auto& rule : rr::kRouteRules) {
            const std::string_view basis = rr::basis_name(rule.basis);
            std::printf("%-44.*s %-22.*s %-10g %.*s | %.*s\n", static_cast<int>(rule.name.size()),
                        rule.name.data(), static_cast<int>(basis.size()), basis.data(), rule.value,
                        static_cast<int>(rule.cards.size()), rule.cards.data(),
                        static_cast<int>(rule.tolerance.size()), rule.tolerance.data());
        }
    }
    std::printf("route rules: %zu described, %zu non-rules listed, kernel hash recorded for none\n",
                std::size(rr::kRouteRules), not_rules().size());
    std::printf("basis:");
    for (const rr::Basis b : rr::kAllBases) {
        std::printf(" %s %zu%s", std::string(rr::basis_name(b)).c_str(),
                    by_basis[static_cast<std::size_t>(b)], b == rr::kAllBases.back() ? "\n" : ",");
    }
    return failures == 0 ? 0 : 1;
}
