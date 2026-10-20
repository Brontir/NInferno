#pragma once

#include "text/byte_span.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::text {

// A regular expression in the Oniguruma subset that tokenizer.json split patterns use, matched
// by backtracking over Unicode code points with Perl/Oniguruma leftmost-first semantics.
//
// Supported: literals and escapes (\t \n \r \f \v \e \a \xHH \x{H..} \uHHHH and escaped
// punctuation), \s \S \d \D \w \W, general-category properties \p{..} / \P{..} / \p{^..}
// (L Lu Ll Lt Lm Lo LC M Mn Mc Me N Nd Nl No P.. S.. Z.. C..), '.', bracket classes with
// ranges, negation and escapes inside, groups ( ) (?: ) (?i: ) (?-i: ) (?i) (?-i),
// lookahead (?= ) (?! ), atomic groups (?> ), alternation, and quantifiers ? * + {m} {m,}
// {,n} {m,n} in greedy, lazy (?) and possessive (+) form.
//
// Semantics follow Oniguruma with UTF-8 encoding: \s is 0009-000D, 0085 and the Z categories;
// \d is Nd; \w is L, M, Nd and Pc; categories come from utf8proc. Case-insensitive literals
// match every code point with the same simple case fold (so (?i:s) also matches U+017F);
// multi-character folds such as "ss" against U+00DF are not applied.
//
// Construction throws std::invalid_argument naming any other construct (backreferences,
// lookbehind, anchors, class intersections, script properties) and patterns that could
// match the empty string, since a split over empty matches is not meaningful here.
class Pattern {
public:
    explicit Pattern(std::string_view regex);

    // A pattern matching `text` literally, as tokenizers' String split and replace patterns do.
    // `text` must be non-empty.
    static Pattern literal(std::string_view text);

    // Successive non-overlapping leftmost matches of valid UTF-8 `text`, as byte spans. Each
    // search resumes at the end of the previous match, like Oniguruma's find_iter.
    [[nodiscard]] std::vector<ByteSpan> find_all(std::string_view text) const;

    [[nodiscard]] const std::string& source() const noexcept { return source_; }

    struct Program;

private:
    Pattern() = default;

    std::string source_;
    std::string literal_;  // non-empty for literal patterns, which skip the matcher
    std::shared_ptr<const Program> program_;
};

} // namespace ninferno::text
