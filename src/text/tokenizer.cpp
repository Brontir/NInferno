#include "text/tokenizer.h"

#include "core/utf8.h"

#include "text/bpe_merge_table.h"
#include "text/pattern.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>
#include <utf8proc/utf8proc.h>

#include <algorithm>
#include <limits>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ninferno::text {
namespace {

using Json    = nlohmann::json;
namespace uni = unicode_internal;

constexpr std::int64_t kMaxTokenId = 16'000'000;

[[noreturn]] void refuse(const std::string& what) {
    throw std::invalid_argument("tokenizer.json: " + what);
}

// Tables are indexed by token id, so the ids must be about as dense as the tokens: a single entry
// with id 15,999,999 would otherwise cost two 16-million-string tables. Real vocabularies leave
// at most a few reserved gaps.
void check_id_density(std::size_t max_id_plus_one, std::size_t tokens) {
    if (max_id_plus_one > 2 * tokens + 4096) {
        refuse("token id " + std::to_string(max_id_plus_one - 1) + " is far beyond its " +
               std::to_string(tokens) + " tokens (a sparse id space)");
    }
}

// ---------------------------------------------------------------------------------------------
// UTF-8 helpers. Text inside the pipeline is always valid UTF-8: encode() validates its input
// and every transformation below produces valid UTF-8 from valid UTF-8.

// Rust's String::from_utf8_lossy: each maximal invalid subpart becomes one U+FFFD.
std::string utf8_lossy(std::string_view bytes) {
    static constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
    const auto* s       = reinterpret_cast<const unsigned char*>(bytes.data());
    const std::size_t n = bytes.size();
    std::string out;
    out.reserve(n);
    for (std::size_t i = 0; i < n;) {
        const unsigned char b = s[i];
        if (b < 0x80) {
            out.push_back(static_cast<char>(b));
            ++i;
            continue;
        }
        std::size_t need = 0;
        unsigned char lo = 0x80, hi = 0xBF;
        if (b >= 0xC2 && b <= 0xDF) {
            need = 1;
        } else if (b >= 0xE0 && b <= 0xEF) {
            need = 2;
            if (b == 0xE0) { lo = 0xA0; }
            if (b == 0xED) { hi = 0x9F; }
        } else if (b >= 0xF0 && b <= 0xF4) {
            need = 3;
            if (b == 0xF0) { lo = 0x90; }
            if (b == 0xF4) { hi = 0x8F; }
        } else {
            out += kReplacement;
            ++i;
            continue;
        }
        std::size_t j = i + 1;
        bool complete = true;
        for (std::size_t k = 0; k < need; ++k, ++j) {
            const unsigned char lower = k == 0 ? lo : 0x80;
            const unsigned char upper = k == 0 ? hi : 0xBF;
            if (j >= n || s[j] < lower || s[j] > upper) {
                complete = false;
                break;
            }
        }
        if (complete) {
            out.append(bytes.substr(i, need + 1));
        } else {
            out += kReplacement;
        }
        i = j;
    }
    return out;
}

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

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Calls fn(code point, byte offset, byte length) for each code point of valid UTF-8.
template <class Fn> void for_each_cp(std::string_view text, Fn&& fn) {
    const auto* s = reinterpret_cast<const unsigned char*>(text.data());
    for (std::size_t i = 0; i < text.size();) {
        std::size_t length = 1;
        const char32_t cp  = decode_utf8(s + i, length);
        fn(cp, i, length);
        i += length;
    }
}

// Unicode White_Space, which is both Oniguruma's and Rust regex's \s.
bool is_white_space(char32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

utf8proc_category_t category(char32_t cp) {
    return utf8proc_category(static_cast<utf8proc_int32_t>(cp));
}

// Rust regex's Unicode \w (Alphabetic, Mark, Decimal_Number, Connector_Punctuation,
// Join_Control), with Alphabetic approximated by the letter categories, Nl and the few
// symbol ranges that carry Other_Alphabetic. Only single_word added tokens use it.
bool is_word_char(char32_t cp) {
    switch (category(cp)) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
    case UTF8PROC_CATEGORY_ME:
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_PC:
        return true;
    default:
        return cp == 0x200C || cp == 0x200D || (cp >= 0x24B6 && cp <= 0x24E9) ||
               (cp >= 0x1F130 && cp <= 0x1F149) || (cp >= 0x1F150 && cp <= 0x1F169) ||
               (cp >= 0x1F170 && cp <= 0x1F189);
    }
}

// Rust char::is_numeric: general categories Nd, Nl, No.
bool is_numeric(char32_t cp) { return uni::is_number(static_cast<std::int32_t>(cp)); }

// tokenizers' Punctuation pre-tokenizer: ASCII punctuation or a P* category.
bool is_punctuation(char32_t cp) {
    if (cp < 0x80) {
        return (cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) ||
               (cp >= 0x5B && cp <= 0x60) || (cp >= 0x7B && cp <= 0x7E);
    }
    switch (category(cp)) {
    case UTF8PROC_CATEGORY_PC:
    case UTF8PROC_CATEGORY_PD:
    case UTF8PROC_CATEGORY_PS:
    case UTF8PROC_CATEGORY_PE:
    case UTF8PROC_CATEGORY_PI:
    case UTF8PROC_CATEGORY_PF:
    case UTF8PROC_CATEGORY_PO:
        return true;
    default:
        return false;
    }
}

// The single code point of a one-character JSON string.
char32_t single_char(const Json& value, const std::string& where) {
    if (!value.is_string()) { refuse(where + " must be a one-character string"); }
    const std::string text = value.get<std::string>();
    if (text.empty() || !valid_utf8(text)) { refuse(where + " must be a one-character string"); }
    std::size_t length = 0;
    const char32_t cp  = decode_utf8(reinterpret_cast<const unsigned char*>(text.data()), length);
    if (length != text.size()) { refuse(where + " must be a one-character string"); }
    return cp;
}

// ---------------------------------------------------------------------------------------------
// Byte-level alphabet (GPT-2 bytes_to_unicode): printable bytes map to themselves, the rest to
// U+0100 upwards in byte order.

struct ByteAlphabet {
    std::array<std::string, 256> encoded;  // byte -> UTF-8 of its visible character
    std::array<int, 324> decoded{};        // visible code point -> byte, or -1

    ByteAlphabet() {
        decoded.fill(-1);
        int next = 256;
        for (int b = 0; b < 256; ++b) {
            const bool visible = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
                                 (b >= 0xAE && b <= 0xFF);
            const int cp = visible ? b : next++;
            append_utf8(encoded[static_cast<std::size_t>(b)], static_cast<char32_t>(cp));
            decoded[static_cast<std::size_t>(cp)] = b;
        }
    }
};

const ByteAlphabet& byte_alphabet() {
    static const ByteAlphabet alphabet;
    return alphabet;
}

const Pattern& gpt2_pattern() {
    static const Pattern pattern(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)");
    return pattern;
}

// ---------------------------------------------------------------------------------------------
// Pieces and the split behaviours of NormalizedString::split.

struct Piece {
    std::string text;
    int token = -1;  // an added token id; such pieces are final
    // The piece starts at byte 0 of the original input (Metaspace prepend_scheme "first").
    bool at_origin = false;
};

enum class Behavior { kIsolated, kRemoved, kMergedWithPrevious, kMergedWithNext, kContiguous };

Behavior parse_behavior(const Json& value, const std::string& where) {
    const std::string name = value.is_string() ? value.get<std::string>() : std::string{};
    if (name == "Isolated") { return Behavior::kIsolated; }
    if (name == "Removed") { return Behavior::kRemoved; }
    if (name == "MergedWithPrevious") { return Behavior::kMergedWithPrevious; }
    if (name == "MergedWithNext") { return Behavior::kMergedWithNext; }
    if (name == "Contiguous") { return Behavior::kContiguous; }
    refuse(where + " has unsupported behavior " + value.dump());
}

struct Segment {
    std::size_t begin = 0;
    std::size_t end   = 0;
    bool match        = false;
};

// Segments covering [0, size) from ordered disjoint matches (tokenizers' find_matches).
std::vector<Segment> segments_from_matches(const std::vector<ByteSpan>& matches, std::size_t size,
                                           bool invert) {
    std::vector<Segment> segments;
    if (size == 0) {
        segments.push_back({0, 0, invert});
        return segments;
    }
    std::size_t prev = 0;
    for (const ByteSpan& m : matches) {
        if (prev != m.begin) { segments.push_back({prev, m.begin, invert}); }
        segments.push_back({m.begin, m.end, !invert});
        prev = m.end;
    }
    if (prev != size) { segments.push_back({prev, size, invert}); }
    return segments;
}

// Segments for a character predicate: each matching character is its own match.
template <class Pred>
std::vector<Segment> segments_from_predicate(std::string_view text, Pred&& pred) {
    std::vector<Segment> segments;
    if (text.empty()) {
        segments.push_back({0, 0, false});
        return segments;
    }
    std::size_t last = 0;
    for_each_cp(text, [&](char32_t cp, std::size_t offset, std::size_t length) {
        if (!pred(cp)) { return; }
        if (last < offset) { segments.push_back({last, offset, false}); }
        segments.push_back({offset, offset + length, true});
        last = offset + length;
    });
    if (last < text.size()) { segments.push_back({last, text.size(), false}); }
    return segments;
}

// Applies a split behaviour; the result lists the kept spans in order.
std::vector<ByteSpan> apply_behavior(const std::vector<Segment>& segments, Behavior behavior) {
    std::vector<ByteSpan> kept;
    switch (behavior) {
    case Behavior::kIsolated:
        for (const Segment& s : segments) { kept.push_back({s.begin, s.end}); }
        break;
    case Behavior::kRemoved:
        for (const Segment& s : segments) {
            if (!s.match) { kept.push_back({s.begin, s.end}); }
        }
        break;
    case Behavior::kContiguous: {
        bool previous = false;
        for (const Segment& s : segments) {
            if (s.match == previous && !kept.empty()) {
                kept.back().end = s.end;
            } else {
                kept.push_back({s.begin, s.end});
            }
            previous = s.match;
        }
        break;
    }
    case Behavior::kMergedWithPrevious: {
        bool previous = false;
        for (const Segment& s : segments) {
            if (s.match && !previous && !kept.empty()) {
                kept.back().end = s.end;
            } else {
                kept.push_back({s.begin, s.end});
            }
            previous = s.match;
        }
        break;
    }
    case Behavior::kMergedWithNext: {
        bool previous = false;
        for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
            if (it->match && !previous && !kept.empty()) {
                kept.back().begin = it->begin;
            } else {
                kept.push_back({it->begin, it->end});
            }
            previous = it->match;
        }
        std::reverse(kept.begin(), kept.end());
        break;
    }
    }
    return kept;
}

void emit_spans(const Piece& parent, const std::vector<ByteSpan>& spans, std::vector<Piece>& out) {
    for (const ByteSpan& span : spans) {
        if (span.end <= span.begin) { continue; }
        out.push_back(Piece{parent.text.substr(span.begin, span.end - span.begin), -1,
                            parent.at_origin && span.begin == 0});
    }
}

// Runs fn(piece, out) over the pieces still to be tokenized and drops empty results, like
// PreTokenizedString::split.
template <class Fn> void split_pieces(std::vector<Piece>& pieces, Fn&& fn) {
    std::vector<Piece> out;
    out.reserve(pieces.size() * 2);
    for (Piece& piece : pieces) {
        if (piece.token >= 0) {
            out.push_back(std::move(piece));
            continue;
        }
        const std::size_t first = out.size();
        fn(std::move(piece), out);
        out.erase(std::remove_if(out.begin() + static_cast<std::ptrdiff_t>(first), out.end(),
                                 [](const Piece& p) { return p.token < 0 && p.text.empty(); }),
                  out.end());
    }
    pieces = std::move(out);
}

// ---------------------------------------------------------------------------------------------
// Pipeline components.

std::optional<Pattern> parse_pattern(const Json& node, const std::string& where) {
    if (!node.is_object() || node.size() != 1) { refuse(where + ".pattern must be String or Regex"); }
    if (node.contains("String") && node["String"].is_string()) {
        return Pattern::literal(node["String"].get<std::string>());
    }
    if (node.contains("Regex") && node["Regex"].is_string()) {
        return Pattern(node["Regex"].get<std::string>());
    }
    refuse(where + ".pattern must be String or Regex");
}

std::string replace_all(std::string_view text, const Pattern& pattern, std::string_view content) {
    const std::vector<ByteSpan> matches = pattern.find_all(text);
    if (matches.empty()) { return std::string(text); }
    std::string out;
    out.reserve(text.size() + matches.size() * content.size());
    std::size_t prev = 0;
    for (const ByteSpan& m : matches) {
        out.append(text.substr(prev, m.begin - prev));
        out.append(content);
        prev = m.end;
    }
    out.append(text.substr(prev));
    return out;
}

std::string utf8proc_normalize(std::string_view text, int options, const char* form) {
    utf8proc_uint8_t* mapped = nullptr;
    const utf8proc_ssize_t result =
        utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()),
                     static_cast<utf8proc_ssize_t>(text.size()), &mapped,
                     static_cast<utf8proc_option_t>(options | UTF8PROC_STABLE));
    if (result < 0) {
        throw std::invalid_argument(std::string("failed to normalize text as ") + form + ": " +
                                    utf8proc_errmsg(result));
    }
    std::string out(reinterpret_cast<const char*>(mapped), static_cast<std::size_t>(result));
    std::free(mapped);
    return out;
}

struct NormalizerStep {
    enum class Kind { kNfc, kNfd, kNfkc, kNfkd, kLowercase, kPrepend, kReplace };
    Kind kind = Kind::kNfc;
    std::string text;  // Prepend: prefix; Replace: content
    std::optional<Pattern> pattern;

    void apply(std::string& s) const {
        switch (kind) {
        case Kind::kNfc:
            s = uni::normalize_nfc(s);
            return;
        case Kind::kNfd:
            s = utf8proc_normalize(s, UTF8PROC_DECOMPOSE, "NFD");
            return;
        case Kind::kNfkc:
            s = utf8proc_normalize(s, UTF8PROC_COMPOSE | UTF8PROC_COMPAT, "NFKC");
            return;
        case Kind::kNfkd:
            s = utf8proc_normalize(s, UTF8PROC_DECOMPOSE | UTF8PROC_COMPAT, "NFKD");
            return;
        case Kind::kLowercase: {
            // Rust char::to_lowercase: simple mappings plus the one unconditional
            // multi-character mapping, U+0130 -> "i̇".
            std::string out;
            out.reserve(s.size());
            for_each_cp(s, [&](char32_t cp, std::size_t, std::size_t) {
                if (cp == 0x130) {
                    out += "i\xCC\x87";
                } else {
                    append_utf8(out, static_cast<char32_t>(
                                         utf8proc_tolower(static_cast<utf8proc_int32_t>(cp))));
                }
            });
            s = std::move(out);
            return;
        }
        case Kind::kPrepend:
            if (!s.empty()) { s.insert(0, text); }
            return;
        case Kind::kReplace:
            s = replace_all(s, *pattern, text);
            return;
        }
    }
};

void parse_normalizer(const Json& node, std::vector<NormalizerStep>& steps) {
    if (node.is_null()) { return; }
    const std::string type = node.is_object() && node.contains("type") && node["type"].is_string()
                                 ? node["type"].get<std::string>()
                                 : std::string{};
    NormalizerStep step;
    if (type == "Sequence") {
        if (!node.contains("normalizers") || !node["normalizers"].is_array()) {
            refuse("normalizer Sequence without normalizers");
        }
        for (const Json& child : node["normalizers"]) { parse_normalizer(child, steps); }
        return;
    }
    if (type == "NFC") {
        step.kind = NormalizerStep::Kind::kNfc;
    } else if (type == "NFD") {
        step.kind = NormalizerStep::Kind::kNfd;
    } else if (type == "NFKC") {
        step.kind = NormalizerStep::Kind::kNfkc;
    } else if (type == "NFKD") {
        step.kind = NormalizerStep::Kind::kNfkd;
    } else if (type == "Lowercase") {
        step.kind = NormalizerStep::Kind::kLowercase;
    } else if (type == "Prepend") {
        if (!node.contains("prepend") || !node["prepend"].is_string()) {
            refuse("normalizer Prepend without prepend");
        }
        step.kind = NormalizerStep::Kind::kPrepend;
        step.text = node["prepend"].get<std::string>();
    } else if (type == "Replace") {
        if (!node.contains("content") || !node["content"].is_string() || !node.contains("pattern")) {
            refuse("normalizer Replace without pattern/content");
        }
        step.kind    = NormalizerStep::Kind::kReplace;
        step.text    = node["content"].get<std::string>();
        step.pattern = parse_pattern(node["pattern"], "normalizer Replace");
    } else {
        refuse("unsupported normalizer type " + (type.empty() ? node.dump() : type));
    }
    steps.push_back(std::move(step));
}

enum class PrependScheme { kAlways, kFirst, kNever };

PrependScheme parse_prepend_scheme(const Json& node, const std::string& where) {
    PrependScheme scheme = PrependScheme::kAlways;
    if (node.contains("prepend_scheme")) {
        const std::string name =
            node["prepend_scheme"].is_string() ? node["prepend_scheme"].get<std::string>() : "";
        if (name == "always") {
            scheme = PrependScheme::kAlways;
        } else if (name == "first") {
            scheme = PrependScheme::kFirst;
        } else if (name == "never") {
            scheme = PrependScheme::kNever;
        } else {
            refuse(where + " has unsupported prepend_scheme " + node["prepend_scheme"].dump());
        }
    }
    // Legacy field: add_prefix_space false must agree with a "never" scheme; the library
    // rejects it otherwise (including when prepend_scheme is absent and defaults to always).
    if (node.contains("add_prefix_space") && node["add_prefix_space"].is_boolean() &&
        !node["add_prefix_space"].get<bool>() && scheme != PrependScheme::kNever) {
        refuse(where + " add_prefix_space does not match prepend_scheme");
    }
    return scheme;
}

bool bool_field(const Json& node, const char* field, bool fallback, const std::string& where) {
    if (!node.contains(field) || node[field].is_null()) { return fallback; }
    if (!node[field].is_boolean()) { refuse(where + "." + field + " must be boolean"); }
    return node[field].get<bool>();
}

struct PreTokenizerStep {
    enum class Kind { kSplit, kByteLevel, kMetaspace, kDigits, kPunctuation };
    Kind kind = Kind::kSplit;
    std::optional<Pattern> pattern;  // Split
    Behavior behavior     = Behavior::kIsolated;
    bool invert           = false;
    bool add_prefix_space = false;  // ByteLevel
    bool use_regex        = true;   // ByteLevel
    char32_t replacement  = 0;      // Metaspace
    std::string replacement_text;
    PrependScheme scheme   = PrependScheme::kAlways;
    bool split             = true;  // Metaspace
    bool individual_digits = false;

    void apply(std::vector<Piece>& pieces) const {
        switch (kind) {
        case Kind::kSplit:
            split_pieces(pieces, [&](Piece piece, std::vector<Piece>& out) {
                const auto segments = segments_from_matches(pattern->find_all(piece.text),
                                                            piece.text.size(), invert);
                emit_spans(piece, apply_behavior(segments, behavior), out);
            });
            return;
        case Kind::kDigits:
            split_pieces(pieces, [&](Piece piece, std::vector<Piece>& out) {
                const auto segments = segments_from_predicate(piece.text, is_numeric);
                emit_spans(piece,
                           apply_behavior(segments, individual_digits ? Behavior::kIsolated
                                                                      : Behavior::kContiguous),
                           out);
            });
            return;
        case Kind::kPunctuation:
            split_pieces(pieces, [&](Piece piece, std::vector<Piece>& out) {
                const auto segments = segments_from_predicate(piece.text, is_punctuation);
                emit_spans(piece, apply_behavior(segments, behavior), out);
            });
            return;
        case Kind::kMetaspace:
            split_pieces(pieces, [&](Piece piece, std::vector<Piece>& out) {
                std::string text;
                text.reserve(piece.text.size() + replacement_text.size());
                for (const char c : piece.text) {
                    if (c == ' ') {
                        text += replacement_text;
                    } else {
                        text.push_back(c);
                    }
                }
                const bool starts = text.compare(0, replacement_text.size(), replacement_text) == 0;
                if (!starts && (scheme == PrependScheme::kAlways ||
                                (scheme == PrependScheme::kFirst && piece.at_origin))) {
                    text.insert(0, replacement_text);
                }
                piece.text = std::move(text);
                if (!split) {
                    out.push_back(std::move(piece));
                    return;
                }
                const auto segments = segments_from_predicate(
                    piece.text, [&](char32_t cp) { return cp == replacement; });
                emit_spans(piece, apply_behavior(segments, Behavior::kMergedWithNext), out);
            });
            return;
        case Kind::kByteLevel: {
            split_pieces(pieces, [&](Piece piece, std::vector<Piece>& out) {
                if (add_prefix_space && (piece.text.empty() || piece.text[0] != ' ')) {
                    piece.text.insert(0, " ");
                }
                if (!use_regex) {
                    out.push_back(std::move(piece));
                    return;
                }
                const auto segments = segments_from_matches(
                    gpt2_pattern().find_all(piece.text), piece.text.size(), false);
                emit_spans(piece, apply_behavior(segments, Behavior::kIsolated), out);
            });
            const ByteAlphabet& alphabet = byte_alphabet();
            for (Piece& piece : pieces) {
                if (piece.token >= 0) { continue; }
                std::string mapped;
                mapped.reserve(piece.text.size() * 2);
                for (const char c : piece.text) {
                    mapped += alphabet.encoded[static_cast<unsigned char>(c)];
                }
                piece.text = std::move(mapped);
            }
            return;
        }
        }
    }
};

void parse_pre_tokenizer(const Json& node, std::vector<PreTokenizerStep>& steps) {
    if (node.is_null()) { return; }
    const std::string type = node.is_object() && node.contains("type") && node["type"].is_string()
                                 ? node["type"].get<std::string>()
                                 : std::string{};
    PreTokenizerStep step;
    if (type == "Sequence") {
        if (!node.contains("pretokenizers") || !node["pretokenizers"].is_array()) {
            refuse("pre_tokenizer Sequence without pretokenizers");
        }
        for (const Json& child : node["pretokenizers"]) { parse_pre_tokenizer(child, steps); }
        return;
    }
    if (type == "Split") {
        if (!node.contains("pattern") || !node.contains("behavior")) {
            refuse("pre_tokenizer Split without pattern/behavior");
        }
        step.kind     = PreTokenizerStep::Kind::kSplit;
        step.pattern  = parse_pattern(node["pattern"], "pre_tokenizer Split");
        step.behavior = parse_behavior(node["behavior"], "pre_tokenizer Split");
        step.invert   = bool_field(node, "invert", false, "pre_tokenizer Split");
    } else if (type == "ByteLevel") {
        step.kind             = PreTokenizerStep::Kind::kByteLevel;
        step.add_prefix_space = bool_field(node, "add_prefix_space", true, "pre_tokenizer ByteLevel");
        step.use_regex        = bool_field(node, "use_regex", true, "pre_tokenizer ByteLevel");
    } else if (type == "Metaspace") {
        step.kind        = PreTokenizerStep::Kind::kMetaspace;
        step.replacement = single_char(node.value("replacement", Json()), "pre_tokenizer Metaspace.replacement");
        append_utf8(step.replacement_text, step.replacement);
        step.scheme = parse_prepend_scheme(node, "pre_tokenizer Metaspace");
        step.split  = bool_field(node, "split", true, "pre_tokenizer Metaspace");
    } else if (type == "Digits") {
        step.kind              = PreTokenizerStep::Kind::kDigits;
        step.individual_digits = bool_field(node, "individual_digits", false, "pre_tokenizer Digits");
    } else if (type == "Punctuation") {
        step.kind     = PreTokenizerStep::Kind::kPunctuation;
        step.behavior = node.contains("behavior")
                            ? parse_behavior(node["behavior"], "pre_tokenizer Punctuation")
                            : Behavior::kIsolated;
    } else {
        refuse("unsupported pre_tokenizer type " + (type.empty() ? node.dump() : type));
    }
    steps.push_back(std::move(step));
}

struct DecoderStep {
    enum class Kind { kByteLevel, kMetaspace, kReplace, kByteFallback, kFuse, kStrip };
    Kind kind = Kind::kFuse;
    std::optional<Pattern> pattern;  // Replace
    std::string content;             // Replace
    char32_t character   = 0;        // Metaspace replacement, Strip content
    PrependScheme scheme = PrependScheme::kAlways;
    std::size_t start    = 0;  // Strip
    std::size_t stop     = 0;

    void apply(std::vector<std::string>& tokens) const {
        switch (kind) {
        case Kind::kByteLevel: {
            const ByteAlphabet& alphabet = byte_alphabet();
            std::string bytes;
            for (const std::string& token : tokens) {
                const std::size_t mark = bytes.size();
                bool mapped            = true;
                for_each_cp(token, [&](char32_t cp, std::size_t, std::size_t) {
                    if (!mapped) { return; }
                    if (cp < alphabet.decoded.size() && alphabet.decoded[cp] >= 0) {
                        bytes.push_back(static_cast<char>(alphabet.decoded[cp]));
                    } else {
                        mapped = false;
                    }
                });
                // A token outside the byte alphabet contributes its raw UTF-8 bytes.
                if (!mapped) {
                    bytes.resize(mark);
                    bytes += token;
                }
            }
            tokens.assign(1, utf8_lossy(bytes));
            return;
        }
        case Kind::kMetaspace:
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                std::string out;
                for_each_cp(tokens[i], [&](char32_t cp, std::size_t, std::size_t) {
                    if (cp != character) {
                        append_utf8(out, cp);
                    } else if (i != 0 || scheme == PrependScheme::kNever) {
                        out.push_back(' ');
                    }
                });
                tokens[i] = std::move(out);
            }
            return;
        case Kind::kReplace:
            for (std::string& token : tokens) { token = replace_all(token, *pattern, content); }
            return;
        case Kind::kByteFallback: {
            std::vector<std::string> out;
            std::string pending;
            std::size_t pending_count = 0;
            const auto flush          = [&] {
                if (pending_count == 0) { return; }
                if (valid_utf8(pending)) {
                    out.push_back(pending);
                } else {
                    for (std::size_t i = 0; i < pending_count; ++i) { out.emplace_back("\xEF\xBF\xBD"); }
                }
                pending.clear();
                pending_count = 0;
            };
            for (std::string& token : tokens) {
                const int byte = byte_fallback_value(token);
                if (byte >= 0) {
                    pending.push_back(static_cast<char>(byte));
                    ++pending_count;
                } else {
                    flush();
                    out.push_back(std::move(token));
                }
            }
            flush();
            tokens = std::move(out);
            return;
        }
        case Kind::kFuse: {
            std::string joined;
            for (const std::string& token : tokens) { joined += token; }
            tokens.assign(1, std::move(joined));
            return;
        }
        case Kind::kStrip:
            for (std::string& token : tokens) {
                std::vector<std::size_t> starts;  // byte offset of each character
                std::vector<char32_t> chars;
                for_each_cp(token, [&](char32_t cp, std::size_t offset, std::size_t) {
                    chars.push_back(cp);
                    starts.push_back(offset);
                });
                std::size_t start_cut = 0;
                for (std::size_t i = 0; i < std::min(start, chars.size()); ++i) {
                    if (chars[i] != character) { break; }
                    start_cut = i + 1;
                }
                std::size_t stop_cut = chars.size();
                for (std::size_t i = 0; i < std::min(stop, chars.size()); ++i) {
                    const std::size_t index = chars.size() - i - 1;
                    if (chars[index] != character) { break; }
                    stop_cut = index;
                }
                if (stop_cut < start_cut) { stop_cut = start_cut; }
                const std::size_t begin = start_cut < chars.size() ? starts[start_cut] : token.size();
                const std::size_t end   = stop_cut < chars.size() ? starts[stop_cut] : token.size();
                token = token.substr(begin, end - begin);
            }
            return;
        }
    }

    // "<0xHH>" -> HH, as Rust u8::from_str_radix on the two characters (which accepts "+H").
    static int byte_fallback_value(std::string_view token) {
        if (token.size() != 6 || token.substr(0, 3) != "<0x" || token[5] != '>') { return -1; }
        const auto hex = [](char c) {
            if (c >= '0' && c <= '9') { return c - '0'; }
            if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
            if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
            return -1;
        };
        const int hi = hex(token[3]);
        const int lo = hex(token[4]);
        if (token[3] == '+' && lo >= 0) { return lo; }
        if (hi < 0 || lo < 0) { return -1; }
        return hi * 16 + lo;
    }
};

void parse_decoder(const Json& node, std::vector<DecoderStep>& steps) {
    if (node.is_null()) { return; }
    const std::string type = node.is_object() && node.contains("type") && node["type"].is_string()
                                 ? node["type"].get<std::string>()
                                 : std::string{};
    DecoderStep step;
    if (type == "Sequence") {
        if (!node.contains("decoders") || !node["decoders"].is_array()) {
            refuse("decoder Sequence without decoders");
        }
        for (const Json& child : node["decoders"]) { parse_decoder(child, steps); }
        return;
    }
    if (type == "ByteLevel") {
        step.kind = DecoderStep::Kind::kByteLevel;
    } else if (type == "Metaspace") {
        step.kind      = DecoderStep::Kind::kMetaspace;
        step.character = single_char(node.value("replacement", Json()), "decoder Metaspace.replacement");
        step.scheme    = parse_prepend_scheme(node, "decoder Metaspace");
    } else if (type == "Replace") {
        if (!node.contains("content") || !node["content"].is_string() || !node.contains("pattern")) {
            refuse("decoder Replace without pattern/content");
        }
        step.kind    = DecoderStep::Kind::kReplace;
        step.content = node["content"].get<std::string>();
        step.pattern = parse_pattern(node["pattern"], "decoder Replace");
    } else if (type == "ByteFallback") {
        step.kind = DecoderStep::Kind::kByteFallback;
    } else if (type == "Fuse") {
        step.kind = DecoderStep::Kind::kFuse;
    } else if (type == "Strip") {
        step.kind      = DecoderStep::Kind::kStrip;
        step.character = single_char(node.value("content", Json()), "decoder Strip.content");
        if (!node.value("start", Json()).is_number_unsigned() ||
            !node.value("stop", Json()).is_number_unsigned()) {
            refuse("decoder Strip start/stop must be non-negative integers");
        }
        step.start = node["start"].get<std::size_t>();
        step.stop  = node["stop"].get<std::size_t>();
    } else {
        refuse("unsupported decoder type " + (type.empty() ? node.dump() : type));
    }
    steps.push_back(std::move(step));
}

// ---------------------------------------------------------------------------------------------
// Byte trie over added-token contents for leftmost-longest matching.
class TokenTrie {
public:
    void insert(std::string_view content, int id) {
        if (terminal_.empty()) { terminal_.push_back(-1); }
        std::uint32_t node = 0;
        for (const char c : content) {
            const std::uint64_t key = edge(node, static_cast<unsigned char>(c));
            auto it                 = edges_.find(key);
            if (it == edges_.end()) {
                it = edges_.emplace(key, static_cast<std::uint32_t>(terminal_.size())).first;
                terminal_.push_back(-1);
            }
            node = it->second;
        }
        terminal_[node] = id;
        first_[static_cast<unsigned char>(content[0])] = true;
    }

    [[nodiscard]] bool empty() const noexcept { return terminal_.size() <= 1; }
    [[nodiscard]] bool may_start(unsigned char byte) const noexcept { return first_[byte]; }

    // Longest content starting at pos: {length, id}, or {0, -1}.
    [[nodiscard]] std::pair<std::size_t, int> longest(std::string_view text, std::size_t pos) const {
        std::pair<std::size_t, int> best{0, -1};
        std::uint32_t node = 0;
        for (std::size_t i = pos; i < text.size(); ++i) {
            const auto it = edges_.find(edge(node, static_cast<unsigned char>(text[i])));
            if (it == edges_.end()) { break; }
            node = it->second;
            if (terminal_[node] >= 0) { best = {i + 1 - pos, terminal_[node]}; }
        }
        return best;
    }

private:
    static std::uint64_t edge(std::uint32_t node, unsigned char byte) {
        return (static_cast<std::uint64_t>(node) << 8U) | byte;
    }

    std::unordered_map<std::uint64_t, std::uint32_t> edges_;
    std::vector<int> terminal_;
    std::array<bool, 256> first_{};
};

struct AddedSplit {
    std::size_t begin = 0;
    std::size_t end   = 0;
    int token         = -1;
};

int parse_id(const Json& value, const std::string& where) {
    if (!value.is_number_integer()) { refuse(where + " must be an integer id"); }
    const std::int64_t id = value.get<std::int64_t>();
    if (id < 0 || id > kMaxTokenId) { refuse(where + " id out of range"); }
    return static_cast<int>(id);
}

std::optional<std::string> config_token_name(const Json& config, const char* field) {
    if (!config.is_object() || !config.contains(field)) { return std::nullopt; }
    const Json& value = config[field];
    if (value.is_string()) { return value.get<std::string>(); }
    if (value.is_object() && value.contains("content") && value["content"].is_string()) {
        return value["content"].get<std::string>();
    }
    return std::nullopt;
}

} // namespace

// ---------------------------------------------------------------------------------------------

struct Tokenizer::Impl {
    // Model.
    std::unordered_map<std::string, int> vocab;
    // The most bytes of a piece one token covers (its vocabulary string): a piece of n bytes needs
    // at least n / max_token_bytes ids, when every byte has a token (byte-level, or a complete byte
    // fallback). 0 when a character can be dropped or fused into one unknown token.
    std::size_t max_token_bytes = 0;
    std::vector<std::string> vocab_by_id;
    std::vector<char> vocab_present;
    std::size_t vocab_entries = 0;
    BpeMergeTable merges;
    std::array<int, 0x800> small_char_ids{};  // single-code-point tokens below U+0800
    std::unordered_map<char32_t, int> char_ids;
    std::array<int, 256> byte_ids{};  // <0xHH> tokens, or -1
    bool byte_fallback = false;
    bool ignore_merges = false;
    bool fuse_unk      = false;
    bool has_unk       = false;
    int unk_id         = -1;

    // Added vocabulary.
    std::vector<AddedToken> added;  // sorted by id
    std::unordered_map<int, std::size_t> added_by_id;
    std::unordered_map<std::string, int> added_by_content;
    std::unordered_map<int, std::string> decode_form;  // normalized content of normalized tokens
    std::unordered_set<std::string> special_contents;
    TokenTrie raw_trie;
    TokenTrie normalized_trie;

    // Pipeline.
    std::vector<NormalizerStep> normalizers;
    std::vector<PreTokenizerStep> pre_tokenizers;
    std::vector<std::pair<bool, std::vector<int>>> template_single;  // {is sequence, special ids}
    bool has_template = false;
    std::vector<DecoderStep> decoders;
    bool has_decoder = false;

    std::size_t max_id_plus_one = 0;
    std::optional<int> bos, eos, pad, unk;
    std::vector<std::string> raw_bytes;  // token_bytes by id

    [[nodiscard]] int char_id(char32_t cp) const {
        if (cp < small_char_ids.size()) { return small_char_ids[cp]; }
        const auto it = char_ids.find(cp);
        return it == char_ids.end() ? -1 : it->second;
    }

    [[nodiscard]] std::optional<int> lookup(std::string_view token) const {
        if (const auto it = added_by_content.find(std::string(token)); it != added_by_content.end()) {
            return it->second;
        }
        if (const auto it = vocab.find(std::string(token)); it != vocab.end()) { return it->second; }
        return std::nullopt;
    }

    [[nodiscard]] const std::string* token_string(int id) const {
        if (const auto it = added_by_id.find(id); it != added_by_id.end()) {
            const auto form = decode_form.find(id);
            return form != decode_form.end() ? &form->second : &added[it->second].content;
        }
        if (id >= 0 && static_cast<std::size_t>(id) < vocab_by_id.size() &&
            vocab_present[static_cast<std::size_t>(id)] != 0) {
            return &vocab_by_id[static_cast<std::size_t>(id)];
        }
        return nullptr;
    }

    void normalize(std::string& text) const {
        for (const NormalizerStep& step : normalizers) { step.apply(text); }
    }

    // AddedVocabulary::find_matches: leftmost-longest matches, then single_word, lstrip and
    // rstrip adjustments applied per match without re-scanning.
    [[nodiscard]] std::vector<AddedSplit> find_added(std::string_view s, const TokenTrie& trie,
                                                     std::span<const ByteSpan> literal = {}) const {
        std::vector<AddedSplit> splits;
        if (trie.empty()) {
            splits.push_back({0, s.size(), -1});
            return splits;
        }
        std::size_t start_offset = 0;
        for (std::size_t i = 0; i < s.size();) {
            if (!trie.may_start(static_cast<unsigned char>(s[i]))) {
                ++i;
                continue;
            }
            const auto [length, id] = trie.longest(s, i);
            if (id < 0 || overlaps(literal, i, i + length)) {
                ++i;
                continue;
            }
            std::size_t start = i;
            std::size_t stop  = i + length;
            i                 = stop;
            const AddedToken& token = added[added_by_id.at(id)];
            if (token.single_word) {
                bool before_word = false, after_word = false;
                if (start > 0) {
                    std::size_t back = start - 1;
                    while (back > 0 && (static_cast<unsigned char>(s[back]) & 0xC0) == 0x80) { --back; }
                    std::size_t length_unused = 0;
                    before_word = is_word_char(decode_utf8(
                        reinterpret_cast<const unsigned char*>(s.data()) + back, length_unused));
                }
                if (stop < s.size()) {
                    std::size_t length_unused = 0;
                    after_word = is_word_char(decode_utf8(
                        reinterpret_cast<const unsigned char*>(s.data()) + stop, length_unused));
                }
                if (before_word || after_word) { continue; }
            }
            if (token.lstrip) {
                std::size_t new_start = start;
                while (new_start > 0) {
                    std::size_t back = new_start - 1;
                    while (back > 0 && (static_cast<unsigned char>(s[back]) & 0xC0) == 0x80) { --back; }
                    std::size_t length_unused = 0;
                    if (!is_white_space(decode_utf8(
                            reinterpret_cast<const unsigned char*>(s.data()) + back, length_unused))) {
                        break;
                    }
                    new_start = back;
                }
                start = std::max(new_start, start_offset);
            }
            if (token.rstrip) {
                while (stop < s.size()) {
                    std::size_t cp_length = 0;
                    if (!is_white_space(decode_utf8(
                            reinterpret_cast<const unsigned char*>(s.data()) + stop, cp_length))) {
                        break;
                    }
                    stop += cp_length;
                }
            }
            if (start_offset < start) { splits.push_back({start_offset, start, -1}); }
            splits.push_back({start, stop, id});
            start_offset = stop;
        }
        if (start_offset != s.size()) { splits.push_back({start_offset, s.size(), -1}); }
        return splits;
    }

    // BPE::merge_word followed by Word::merge_all, appending the resulting ids.
    struct Symbol {
        int id   = 0;
        int prev = -1;
        int next = -1;
        int len  = 0;
    };
    struct QueuedMerge {
        int rank   = 0;
        int pos    = 0;
        int result = 0;
    };
    struct Later {
        bool operator()(const QueuedMerge& a, const QueuedMerge& b) const {
            return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
        }
    };

    void bpe(std::string_view word, std::vector<Symbol>& symbols, std::vector<int>& out) const {
        if (ignore_merges) {
            if (const auto it = vocab.find(std::string(word)); it != vocab.end()) {
                out.push_back(it->second);
                return;
            }
        }
        symbols.clear();
        const auto add = [&](int id, int len) {
            const int index = static_cast<int>(symbols.size());
            if (!symbols.empty()) { symbols.back().next = index; }
            symbols.push_back(Symbol{id, index - 1, -1, len});
        };
        int pending_unk_len = 0;  // bytes of a pending (possibly fused) unknown symbol
        for_each_cp(word, [&](char32_t cp, std::size_t offset, std::size_t length) {
            const int id = char_id(cp);
            if (id >= 0) {
                if (pending_unk_len > 0) {
                    add(unk_id, pending_unk_len);
                    pending_unk_len = 0;
                }
                add(id, static_cast<int>(length));
                return;
            }
            if (byte_fallback) {
                bool all = true;
                for (std::size_t k = 0; k < length; ++k) {
                    all = all && byte_ids[static_cast<unsigned char>(word[offset + k])] >= 0;
                }
                if (all) {
                    // Like the library, byte tokens do not flush a pending unknown symbol.
                    for (std::size_t k = 0; k < length; ++k) {
                        add(byte_ids[static_cast<unsigned char>(word[offset + k])], 1);
                    }
                    return;
                }
            }
            if (!has_unk) { return; }  // no unk_token: the library drops the character
            if (unk_id < 0) { throw std::invalid_argument("tokenizer.json: model.unk_token is not in model.vocab"); }
            if (pending_unk_len > 0 && fuse_unk) {
                pending_unk_len += static_cast<int>(length);
            } else {
                if (pending_unk_len > 0) { add(unk_id, pending_unk_len); }
                pending_unk_len = static_cast<int>(length);
            }
        });
        if (pending_unk_len > 0) { add(unk_id, pending_unk_len); }

        if (symbols.size() > 1) {
            std::priority_queue<QueuedMerge, std::vector<QueuedMerge>, Later> queue;
            for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                if (const auto* m = merges.find(symbols[i].id, symbols[i + 1].id)) {
                    queue.push({m->rank, static_cast<int>(i), m->result});
                }
            }
            while (!queue.empty()) {
                const QueuedMerge top = queue.top();
                queue.pop();
                Symbol& current = symbols[static_cast<std::size_t>(top.pos)];
                if (current.len == 0 || current.next == -1) { continue; }
                const Symbol right = symbols[static_cast<std::size_t>(current.next)];
                // Stale entries are recognised by the merge result only, as in the library.
                const auto* target = merges.find(current.id, right.id);
                if (target == nullptr || target->result != top.result) { continue; }
                symbols[static_cast<std::size_t>(current.next)].len = 0;
                current.id = top.result;
                current.len += right.len;
                current.next = right.next;
                if (right.next >= 0) { symbols[static_cast<std::size_t>(right.next)].prev = top.pos; }
                if (current.prev >= 0) {
                    const Symbol& prev = symbols[static_cast<std::size_t>(current.prev)];
                    if (const auto* m = merges.find(prev.id, current.id)) {
                        queue.push({m->rank, current.prev, m->result});
                    }
                }
                if (current.next >= 0) {
                    const Symbol& next = symbols[static_cast<std::size_t>(current.next)];
                    if (const auto* m = merges.find(current.id, next.id)) {
                        queue.push({m->rank, top.pos, m->result});
                    }
                }
            }
        }
        for (const Symbol& symbol : symbols) {
            if (symbol.len != 0) { out.push_back(symbol.id); }
        }
    }

    [[nodiscard]] std::vector<int> encode(std::string_view text, bool add_special_tokens) const {
        if (!valid_utf8(text)) { throw std::invalid_argument("tokenizer input is not valid UTF-8"); }

        // 1. Added tokens that match the raw text.
        std::vector<Piece> pieces;
        if (!text.empty()) {
            for (const AddedSplit& split : find_added(text, raw_trie)) {
                if (split.end <= split.begin) { continue; }
                pieces.push_back(Piece{std::string(text.substr(split.begin, split.end - split.begin)),
                                       split.token, split.begin == 0});
            }
        }
        std::vector<int> ids;
        ids.reserve(text.size() / 3 + 4);
        encode_pieces(std::move(pieces), ids, true);

        // 5. Post-processing (only TemplateProcessing changes ids).
        if (!add_special_tokens || !has_template) { return ids; }
        std::vector<int> result;
        result.reserve(ids.size() + 8);
        for (const auto& [is_sequence, special] : template_single) {
            if (is_sequence) {
                result.insert(result.end(), ids.begin(), ids.end());
            } else {
                result.insert(result.end(), special.begin(), special.end());
            }
        }
        return result;
    }

    // Steps 2-4 of encode over pieces that already carry their added-token splits. Every step is
    // local to a piece, so a piece encodes the same alone as within its text.
    // budget, below SIZE_MAX, is an overflow check (EncodeOptions::overflow_check_only): once the
    // pieces left need more ids than it allows, ids is left at budget + 1 entries that are not the
    // encoding.
    void encode_pieces(std::vector<Piece> pieces, std::vector<int>& ids, bool match_normalized,
                       std::size_t budget = std::numeric_limits<std::size_t>::max()) const {
        // 2. Normalization of the remaining pieces, then added tokens that match normalized text.
        split_pieces(pieces, [&](Piece piece, std::vector<Piece>& out) {
            normalize(piece.text);
            if (normalized_trie.empty() || !match_normalized) {
                out.push_back(std::move(piece));
                return;
            }
            for (const AddedSplit& split : find_added(piece.text, normalized_trie)) {
                if (split.end <= split.begin) { continue; }
                out.push_back(Piece{piece.text.substr(split.begin, split.end - split.begin),
                                    split.token, piece.at_origin && split.begin == 0});
            }
        });
        // 3. Pre-tokenization.
        for (const PreTokenizerStep& step : pre_tokenizers) { step.apply(pieces); }

        // 4. The model, with a per-call cache of word -> ids.
        std::unordered_map<std::string, std::pair<std::uint32_t, std::uint32_t>> cache;
        std::vector<int> cached_ids;
        std::vector<Symbol> symbols;
        const bool bounded =
            budget != std::numeric_limits<std::size_t>::max() && max_token_bytes != 0;
        std::size_t rest = 0;
        if (bounded) {
            for (const Piece& piece : pieces) {
                if (piece.token < 0) rest += piece.text.size();
            }
        }
        for (const Piece& piece : pieces) {
            if (piece.token >= 0) {
                ids.push_back(piece.token);
                continue;
            }
            if (bounded) {
                if ((rest + max_token_bytes - 1) / max_token_bytes >
                    budget - std::min(budget, ids.size())) {
                    ids.resize(budget + 1, ids.empty() ? 0 : ids.back());
                    return;
                }
                rest -= piece.text.size();
            }
            if (const auto it = cache.find(piece.text); it != cache.end()) {
                ids.insert(ids.end(), cached_ids.begin() + it->second.first,
                           cached_ids.begin() + it->second.first + it->second.second);
                continue;
            }
            const std::size_t mark = ids.size();
            bpe(piece.text, symbols, ids);
            if (piece.text.size() < 256) {
                cache.emplace(piece.text,
                              std::pair<std::uint32_t, std::uint32_t>(
                                  static_cast<std::uint32_t>(cached_ids.size()),
                                  static_cast<std::uint32_t>(ids.size() - mark)));
                cached_ids.insert(cached_ids.end(), ids.begin() + static_cast<std::ptrdiff_t>(mark),
                                  ids.end());
            }
        }
    }

    [[nodiscard]] std::vector<int>
    encode_run(std::string_view run, bool at_origin, bool match_normalized,
               std::size_t budget = std::numeric_limits<std::size_t>::max()) const {
        std::vector<int> ids;
        if (run.empty()) { return ids; }
        std::vector<Piece> pieces;
        pieces.push_back(Piece{std::string(run), -1, at_origin});
        encode_pieces(std::move(pieces), ids, match_normalized, budget);
        return ids;
    }

    [[nodiscard]] BoundaryEncodedText
    encode_with_boundaries(std::string_view text, std::span<const std::size_t> byte_boundaries,
                           const EncodeOptions& options,
                           std::span<const ByteSpan> literal) const {
        if (!valid_utf8(text)) { throw std::invalid_argument("tokenizer input is not valid UTF-8"); }
        std::size_t previous_end = 0;
        for (const ByteSpan span : literal) {
            if (span.begin < previous_end || span.begin >= span.end || span.end > text.size()) {
                throw std::invalid_argument(
                    "Tokenizer literal spans must be ordered disjoint input ranges");
            }
            previous_end = span.end;
        }
        for (const std::size_t boundary : byte_boundaries) {
            if (boundary > text.size()) {
                throw std::out_of_range("Tokenizer byte boundary exceeds input text");
            }
        }
        struct Run {
            std::size_t begin = 0;
            std::size_t end   = 0;
            int token         = -1;
            std::size_t first = 0;  // token frontier before the run
            std::size_t count = 0;
            bool literal      = false;
        };
        std::vector<Run> runs;
        if (!options.parse_added_tokens) {
            if (!text.empty()) { runs.push_back(Run{0, text.size(), -1}); }
        } else if (!text.empty()) {
            for (const AddedSplit& split : find_added(text, raw_trie, literal)) {
                if (split.end > split.begin) {
                    runs.push_back(Run{split.begin, split.end, split.token});
                }
            }
        }
        BoundaryEncodedText encoded;
        encoded.input_ids.reserve(std::min(text.size() / 3 + 4, options.max_tokens));
        for (Run& run : runs) {
            if (options.overflow_check_only && encoded.input_ids.size() > options.max_tokens) {
                encoded.input_ids.resize(options.max_tokens);
                encoded.boundaries.resize(byte_boundaries.size());
                return encoded;
            }
            run.first   = encoded.input_ids.size();
            run.literal = overlaps(literal, run.begin, run.end);
            if (run.token >= 0) {
                encoded.input_ids.push_back(run.token);
            } else {
                const std::size_t budget =
                    options.overflow_check_only
                        ? options.max_tokens -
                              std::min(options.max_tokens, encoded.input_ids.size())
                        : std::numeric_limits<std::size_t>::max();
                const std::vector<int> ids = encode_run(text.substr(run.begin, run.end - run.begin),
                                                        run.begin == 0, !run.literal, budget);
                encoded.input_ids.insert(encoded.input_ids.end(), ids.begin(), ids.end());
            }
            run.count = encoded.input_ids.size() - run.first;
        }
        if (options.overflow_check_only && encoded.input_ids.size() > options.max_tokens) {
            encoded.input_ids.resize(options.max_tokens);
            encoded.boundaries.resize(byte_boundaries.size());
            return encoded;
        }
        encoded.boundaries.resize(byte_boundaries.size());
        for (std::size_t index = 0; index < byte_boundaries.size(); ++index) {
            const std::size_t boundary = byte_boundaries[index];
            const auto run = std::find_if(runs.begin(), runs.end(),
                                          [&](const Run& r) { return boundary < r.end; });
            TokenBoundaryResult& result = encoded.boundaries[index];
            if (run == runs.end()) {
                result = {encoded.input_ids.size(), encoded.input_ids.size()};
            } else if (boundary <= run->begin) {
                result = {run->first, run->first};
            } else if (run->token >= 0) {
                result = {std::nullopt, run->first};
            } else {
                const std::string_view head = text.substr(run->begin, boundary - run->begin);
                const std::string_view tail = text.substr(boundary, run->end - boundary);
                const std::vector<int> prefix = encode_run(head, run->begin == 0, !run->literal);
                const std::vector<int> suffix = encode_run(tail, false, !run->literal);
                const auto whole =
                    std::span<const int>(encoded.input_ids).subspan(run->first, run->count);
                const bool exact =
                    prefix.size() + suffix.size() == whole.size() &&
                    std::equal(prefix.begin(), prefix.end(), whole.begin()) &&
                    std::equal(suffix.begin(), suffix.end(),
                               whole.begin() + static_cast<std::ptrdiff_t>(prefix.size()));
                if (exact) {
                    result = {run->first + prefix.size(), run->first + prefix.size()};
                } else {
                    result = {std::nullopt, run->first};
                }
            }
        }
        if (encoded.input_ids.size() > options.max_tokens) {
            encoded.input_ids.resize(options.max_tokens);
        }
        return encoded;
    }

    // Token-local decoder steps over one token string (see Tokenizer::token_bytes).
    [[nodiscard]] std::string local_bytes(const std::string& token) const {
        if (!has_decoder) { return token; }
        std::string out = token;
        for (const DecoderStep& step : decoders) {
            switch (step.kind) {
            case DecoderStep::Kind::kByteLevel: {
                const ByteAlphabet& alphabet = byte_alphabet();
                std::string bytes;
                bool mapped = true;
                for_each_cp(out, [&](char32_t cp, std::size_t, std::size_t) {
                    if (!mapped) { return; }
                    if (cp < alphabet.decoded.size() && alphabet.decoded[cp] >= 0) {
                        bytes.push_back(static_cast<char>(alphabet.decoded[cp]));
                    } else {
                        mapped = false;
                    }
                });
                if (mapped) { out = std::move(bytes); }
                break;
            }
            case DecoderStep::Kind::kMetaspace: {
                std::string spaced;
                for_each_cp(out, [&](char32_t cp, std::size_t, std::size_t) {
                    if (cp == step.character) {
                        spaced.push_back(' ');
                    } else {
                        append_utf8(spaced, cp);
                    }
                });
                out = std::move(spaced);
                break;
            }
            case DecoderStep::Kind::kReplace:
                out = replace_all(out, *step.pattern, step.content);
                break;
            case DecoderStep::Kind::kByteFallback:
                if (const int byte = DecoderStep::byte_fallback_value(out); byte >= 0) {
                    out.assign(1, static_cast<char>(byte));
                }
                break;
            case DecoderStep::Kind::kFuse:
            case DecoderStep::Kind::kStrip:
                break;
            }
        }
        return out;
    }

    void load_token_bytes() {
        raw_bytes.assign(max_id_plus_one, std::string{});
        for (std::size_t id = 0; id < max_id_plus_one; ++id) {
            if (const std::string* token = token_string(static_cast<int>(id)); token != nullptr) {
                raw_bytes[id] = local_bytes(*token);
            }
        }
    }

    [[nodiscard]] std::string decode(std::span<const int> ids, bool skip_special_tokens) const {
        std::vector<std::string> tokens;
        tokens.reserve(ids.size());
        for (const int id : ids) {
            const std::string* token = token_string(id);
            if (token == nullptr) { continue; }
            // The library tests the token string, not the id, against the special contents.
            if (skip_special_tokens && special_contents.contains(*token)) { continue; }
            tokens.push_back(*token);
        }
        std::string out;
        if (!has_decoder) {
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                if (i != 0) { out.push_back(' '); }
                out += tokens[i];
            }
            return out;
        }
        for (const DecoderStep& step : decoders) { step.apply(tokens); }
        for (const std::string& token : tokens) { out += token; }
        return out;
    }

    void load_model(const Json& model);
    void load_overflow_bound();
    void load_added_tokens(const Json& root);
    void load_post_processor(const Json& node);
};

void Tokenizer::Impl::load_overflow_bound() {
    const bool byte_level =
        std::any_of(pre_tokenizers.begin(), pre_tokenizers.end(), [](const auto& step) {
            return step.kind == PreTokenizerStep::Kind::kByteLevel;
        });
    bool every_byte = true;
    for (int b = 0; b < 256; ++b) {
        const bool fallback = byte_fallback && byte_ids[static_cast<std::size_t>(b)] >= 0;
        const bool alphabet =
            byte_level && vocab.contains(byte_alphabet().encoded[static_cast<std::size_t>(b)]);
        every_byte = every_byte && (fallback || alphabet);
    }
    if (!every_byte) { return; }
    for (const auto& [token, id] : vocab) {
        max_token_bytes = std::max(max_token_bytes, token.size());
    }
}

void Tokenizer::Impl::load_model(const Json& model) {
    if (!model.is_object() || model.value("type", std::string{}) != "BPE") {
        refuse("unsupported model type " +
               (model.is_object() && model.contains("type") ? model["type"].dump() : model.dump()) +
               " (only BPE is supported)");
    }
    if (model.contains("dropout") && !model["dropout"].is_null() &&
        !(model["dropout"].is_number() && model["dropout"].get<double>() == 0.0)) {
        refuse("BPE dropout is not supported");
    }
    for (const char* field : {"continuing_subword_prefix", "end_of_word_suffix"}) {
        if (model.contains(field) && !model[field].is_null() &&
            !(model[field].is_string() && model[field].get<std::string>().empty())) {
            refuse(std::string("BPE ") + field + " is not supported");
        }
    }
    byte_fallback = bool_field(model, "byte_fallback", false, "model");
    ignore_merges = bool_field(model, "ignore_merges", false, "model");
    fuse_unk      = bool_field(model, "fuse_unk", false, "model");

    if (!model.contains("vocab") || !model["vocab"].is_object()) { refuse("model.vocab must be an object"); }
    const Json& vocab_json = model["vocab"];
    int max_id             = -1;
    vocab.reserve(vocab_json.size());
    for (const auto& item : vocab_json.items()) {
        const int id = parse_id(item.value(), "model.vocab");
        vocab[item.key()] = id;
        max_id            = std::max(max_id, id);
    }
    vocab_entries = vocab.size();
    check_id_density(static_cast<std::size_t>(max_id + 1), vocab_entries);
    vocab_by_id.assign(static_cast<std::size_t>(max_id + 1), std::string{});
    vocab_present.assign(static_cast<std::size_t>(max_id + 1), 0);
    small_char_ids.fill(-1);
    byte_ids.fill(-1);
    for (const auto& [token, id] : vocab) {
        if (vocab_present[static_cast<std::size_t>(id)] != 0) {
            refuse("model.vocab maps two tokens to id " + std::to_string(id));
        }
        vocab_present[static_cast<std::size_t>(id)] = 1;
        vocab_by_id[static_cast<std::size_t>(id)]   = token;
        if (!valid_utf8(token) || token.empty()) { continue; }
        std::size_t length = 0;
        const char32_t cp  = decode_utf8(reinterpret_cast<const unsigned char*>(token.data()), length);
        if (length == token.size()) {
            if (cp < small_char_ids.size()) {
                small_char_ids[cp] = id;
            } else {
                char_ids[cp] = id;
            }
        }
        const int byte = DecoderStep::byte_fallback_value(token);
        // merge_word looks tokens up as "<0x%02X>" with upper-case digits.
        if (byte >= 0 && token[3] != '+') {
            static constexpr char kHex[] = "0123456789ABCDEF";
            const std::string canonical  = std::string("<0x") + kHex[byte >> 4] + kHex[byte & 15] + ">";
            if (canonical == token) { byte_ids[static_cast<std::size_t>(byte)] = id; }
        }
    }
    if (model.contains("unk_token") && !model["unk_token"].is_null()) {
        if (!model["unk_token"].is_string()) { refuse("model.unk_token must be a string"); }
        has_unk       = true;
        const auto it = vocab.find(model["unk_token"].get<std::string>());
        // Like the library, a dangling unk_token only fails once a character needs it.
        unk_id = it == vocab.end() ? -1 : it->second;
    }

    if (!model.contains("merges") || !model["merges"].is_array()) { refuse("model.merges must be an array"); }
    const Json& merges_json = model["merges"];
    merges.reset(merges_json.size());
    int rank = 0;
    for (const Json& merge : merges_json) {
        std::string left, right;
        if (merge.is_string()) {
            const std::string line = merge.get<std::string>();
            if (line.starts_with("#version")) { continue; }
            const std::size_t space = line.find(' ');
            if (space == std::string::npos || line.find(' ', space + 1) != std::string::npos) {
                refuse("bad merge \"" + line + "\"");
            }
            left  = line.substr(0, space);
            right = line.substr(space + 1);
        } else if (merge.is_array() && merge.size() == 2 && merge[0].is_string() && merge[1].is_string()) {
            left  = merge[0].get<std::string>();
            right = merge[1].get<std::string>();
        } else {
            refuse("bad merge " + merge.dump());
        }
        const auto a = vocab.find(left);
        const auto b = vocab.find(right);
        const auto c = vocab.find(left + right);
        if (a == vocab.end() || b == vocab.end() || c == vocab.end()) {
            refuse("merge token out of vocabulary: " + left + " " + right);
        }
        merges.assign(a->second, b->second, rank, c->second);
        ++rank;
    }
    max_id_plus_one = static_cast<std::size_t>(max_id + 1);
}

void Tokenizer::Impl::load_added_tokens(const Json& root) {
    if (!root.contains("added_tokens") || root["added_tokens"].is_null()) { return; }
    if (!root["added_tokens"].is_array()) { refuse("added_tokens must be an array"); }
    // Ids are assigned as AddedVocabulary::add_tokens does: a content already in the model
    // vocabulary takes that id, anything else the next id from the model vocabulary size.
    // A tokenizer.json whose ids disagree would not load the way it reads, so it is refused.
    int next_id = static_cast<int>(vocab_entries);
    for (const Json& item : root["added_tokens"]) {
        if (!item.is_object() || !item.contains("content") || !item["content"].is_string() ||
            !item.contains("id")) {
            refuse("malformed added_tokens entry " + item.dump());
        }
        AddedToken token;
        token.id          = parse_id(item["id"], "added_tokens.id");
        token.content     = item["content"].get<std::string>();
        token.special     = bool_field(item, "special", false, "added_tokens");
        token.single_word = bool_field(item, "single_word", false, "added_tokens");
        token.lstrip      = bool_field(item, "lstrip", false, "added_tokens");
        token.rstrip      = bool_field(item, "rstrip", false, "added_tokens");
        token.normalized  = bool_field(item, "normalized", !token.special, "added_tokens");
        if (token.content.empty()) { continue; }
        if (!valid_utf8(token.content)) { refuse("added token is not valid UTF-8"); }
        if (added_by_content.contains(token.content)) {
            refuse("duplicate added token " + token.content);
        }
        int assigned = 0;
        if (const auto it = vocab.find(token.content); it != vocab.end()) {
            assigned = it->second;
        } else {
            assigned = next_id++;
        }
        if (assigned != token.id) {
            refuse("added token " + token.content + " declares id " + std::to_string(token.id) +
                   " but the library assigns " + std::to_string(assigned));
        }
        if (added_by_id.contains(token.id)) { refuse("duplicate added token id " + std::to_string(token.id)); }
        added_by_content.emplace(token.content, token.id);
        if (token.special) { special_contents.insert(token.content); }
        added.push_back(std::move(token));
        added_by_id.emplace(added.back().id, added.size() - 1);
    }
    std::sort(added.begin(), added.end(),
              [](const AddedToken& a, const AddedToken& b) { return a.id < b.id; });
    added_by_id.clear();
    for (std::size_t i = 0; i < added.size(); ++i) {
        const AddedToken& token = added[i];
        added_by_id.emplace(token.id, i);
        max_id_plus_one = std::max(max_id_plus_one, static_cast<std::size_t>(token.id) + 1);
        if (!token.normalized) {
            raw_trie.insert(token.content, token.id);
            continue;
        }
        std::string form = token.content;
        normalize(form);
        if (form.empty()) { refuse("added token " + token.content + " normalizes to nothing"); }
        if (form != token.content) { decode_form.emplace(token.id, form); }
        normalized_trie.insert(form, token.id);
    }
    check_id_density(max_id_plus_one, vocab_entries + added.size());
}

void Tokenizer::Impl::load_post_processor(const Json& node) {
    if (node.is_null()) { return; }
    const std::string type = node.is_object() && node.contains("type") && node["type"].is_string()
                                 ? node["type"].get<std::string>()
                                 : std::string{};
    if (type == "Sequence") {
        if (!node.contains("processors") || !node["processors"].is_array()) {
            refuse("post_processor Sequence without processors");
        }
        for (const Json& child : node["processors"]) { load_post_processor(child); }
        return;
    }
    if (type == "ByteLevel") { return; }  // only adjusts offsets
    if (type != "TemplateProcessing") {
        refuse("unsupported post_processor type " + (type.empty() ? node.dump() : type));
    }
    if (has_template) { refuse("more than one TemplateProcessing post_processor"); }
    if (!node.contains("single") || !node["single"].is_array()) {
        refuse("TemplateProcessing.single must be a list of pieces");
    }
    const Json& specials = node.value("special_tokens", Json::object());
    for (const Json& piece : node["single"]) {
        if (piece.is_object() && piece.contains("Sequence")) {
            if (piece["Sequence"].value("id", std::string{}) != "A") {
                refuse("TemplateProcessing.single refers to a sequence other than A");
            }
            template_single.push_back({true, {}});
        } else if (piece.is_object() && piece.contains("SpecialToken")) {
            const std::string name = piece["SpecialToken"].value("id", std::string{});
            if (!specials.is_object() || !specials.contains(name) ||
                !specials[name].contains("ids") || !specials[name]["ids"].is_array()) {
                refuse("TemplateProcessing special token " + name + " is not defined");
            }
            std::vector<int> ids;
            for (const Json& id : specials[name]["ids"]) {
                ids.push_back(parse_id(id, "TemplateProcessing.special_tokens"));
            }
            template_single.push_back({false, std::move(ids)});
        } else {
            refuse("malformed TemplateProcessing piece " + piece.dump());
        }
    }
    has_template = true;
}

Tokenizer::Tokenizer(std::string_view tokenizer_json, std::string_view tokenizer_config_json) {
    Json root;
    try {
        root = Json::parse(tokenizer_json);
    } catch (const Json::exception& ex) {
        refuse(std::string("malformed JSON: ") + ex.what());
    }
    if (!root.is_object()) { refuse("root must be an object"); }
    // "truncation" and "padding" are batch-encoding settings of the Python library, not part of
    // the tokenization itself. transformers switches both off for every call unless the caller
    // asks for them, and the engine never truncates or pads a prompt: the context limit is
    // enforced by the engine with an explicit refusal. They are read past and have no effect
    // (Gemma 4's tokenizer.json sets truncation {max_length 8192, LongestFirst}).
    auto impl = std::make_shared<Impl>();
    parse_normalizer(root.value("normalizer", Json()), impl->normalizers);
    parse_pre_tokenizer(root.value("pre_tokenizer", Json()), impl->pre_tokenizers);
    if (!root.contains("model")) { refuse("missing model"); }
    impl->load_model(root["model"]);
    impl->load_overflow_bound();
    impl->load_added_tokens(root);
    impl->load_post_processor(root.value("post_processor", Json()));
    const Json& decoder = root.value("decoder", Json());
    parse_decoder(decoder, impl->decoders);
    impl->has_decoder = !decoder.is_null();
    impl->load_token_bytes();

    if (!tokenizer_config_json.empty()) {
        Json config;
        try {
            config = Json::parse(tokenizer_config_json);
        } catch (const Json::exception& ex) {
            throw std::invalid_argument(std::string("tokenizer_config.json: malformed JSON: ") +
                                        ex.what());
        }
        const auto resolve = [&](const char* field) -> std::optional<int> {
            const auto name = config_token_name(config, field);
            return name ? impl->lookup(*name) : std::nullopt;
        };
        impl->bos = resolve("bos_token");
        impl->eos = resolve("eos_token");
        impl->pad = resolve("pad_token");
        impl->unk = resolve("unk_token");
    }
    impl_ = std::move(impl);
}

std::vector<int> Tokenizer::encode(std::string_view text, bool add_special_tokens) const {
    return impl_->encode(text, add_special_tokens);
}

BoundaryEncodedText Tokenizer::encode_with_boundaries(std::string_view text,
                                                      std::span<const std::size_t> byte_boundaries,
                                                      EncodeOptions options,
                                                      std::span<const ByteSpan> literal_spans) const {
    return impl_->encode_with_boundaries(text, byte_boundaries, options, literal_spans);
}

std::string_view Tokenizer::token_bytes(int id) const noexcept {
    if (id < 0 || static_cast<std::size_t>(id) >= impl_->raw_bytes.size()) { return {}; }
    return impl_->raw_bytes[static_cast<std::size_t>(id)];
}

bool Tokenizer::is_valid_token(int id) const noexcept { return impl_->token_string(id) != nullptr; }

std::string Tokenizer::decode(std::span<const int> ids, bool skip_special_tokens) const {
    return impl_->decode(ids, skip_special_tokens);
}

std::optional<int> Tokenizer::token_id(std::string_view token) const { return impl_->lookup(token); }

std::optional<std::string> Tokenizer::id_to_token(int id) const {
    const std::string* token = impl_->token_string(id);
    return token != nullptr ? std::optional<std::string>(*token) : std::nullopt;
}

std::size_t Tokenizer::vocab_size() const noexcept { return impl_->max_id_plus_one; }

std::vector<int> Tokenizer::template_prefix() const {
    std::vector<int> out;
    if (!impl_->has_template) { return out; }
    for (const auto& [is_sequence, special] : impl_->template_single) {
        if (is_sequence) { break; }
        out.insert(out.end(), special.begin(), special.end());
    }
    return out;
}

std::span<const AddedToken> Tokenizer::added_tokens() const noexcept { return impl_->added; }

bool Tokenizer::is_special_token(int id) const noexcept {
    const auto it = impl_->added_by_id.find(id);
    return it != impl_->added_by_id.end() && impl_->added[it->second].special;
}

std::optional<int> Tokenizer::bos_token_id() const noexcept { return impl_->bos; }
std::optional<int> Tokenizer::eos_token_id() const noexcept { return impl_->eos; }
std::optional<int> Tokenizer::pad_token_id() const noexcept { return impl_->pad; }
std::optional<int> Tokenizer::unk_token_id() const noexcept { return impl_->unk; }

} // namespace ninferno::text
