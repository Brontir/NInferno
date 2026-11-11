// The holdout records (core/fitted_holdouts.h) of the fitted constants, one side table. What it
// covers: the tagged constants (Provenance in route_cost_constants.h, Basis in route_rules.h), read
// from the compiled tables; every definition of the files listed in specs() below with a
// deny-by-default mode; and the untagged kernel and dispatch constants a manual sweep of src/
// found (Mode::RecordedOnly: only their keys are checked). An untagged fitted constant in any
// other file is not found by this test. The checks:
//   - every record is complete (an explicit None, or a Checked whose source names a script under
//     tools/fits/ that exists, with the fields its kind needs; an Unset one was forgotten) and
//     appears once;
//   - every key matches its DEFINITION in its file (a definition, a struct field or a function,
//     read from the file's text, not a substring of it): a renamed or removed constant fails;
//   - every constant a covered file marks or behaves as fitted has a record, or sits on that file's
//     not-fitted list below with the reason it is none: route_cost_constants.h (every definition
//     tagged Provenance::Fitted), choice_constants.h (every definition), core/generation.h (every
//     field of the row, and every field its own kind table names `fitted` or `preference` has a
//     record), core/cost_constants.h, spec/round_cost.h, spec/moe_routed_share.h and
//     ops/common/sm_wave_rules.h (every definition);
//   - the tags are read from the compiled tables, not only from the text: every entry of
//     route_cost::kRouteCostConstants whose `from` is Provenance::Fitted has a record whose
//     definition carries its name, and every record there names a Fitted entry; every row of
//     route_rules::kRouteRules whose basis is Basis::Fitted or Basis::ScaledFromOneBoard has a
//     record keyed by its name, and every record there names such a row;
//   - the Provenance and Basis tags are found by a scan of the whole source tree (any spacing
//     around `::`, and `using enum` of either enum): a file outside their home headers that uses
//     one fails, so a tagged constant cannot sit outside the compiled checks;
//   - definitions are read from the code with its comments (line and block) removed;
//   - the total is pinned, so adding or dropping a record is a decision.
// No GPU.

#include "core/fitted_holdouts.h"
#include "ops/generic/route_cost_constants.h"
#include "ops/generic/route_rules.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fit = ninferno::fitted;

constexpr std::size_t kExpectedRecords = 103;

int failures = 0;

void fail(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
}

std::string read_file(std::string_view relative) {
    std::ifstream in(std::filesystem::path(NINFERNO_SOURCE_DIR) / std::string(relative),
                     std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// The text without its comments: a `//` to the end of its line, and a `/* ... */` block (its
// newlines kept, so a line stays a line). A tag or a definition inside a comment is neither. String
// and character literals are copied as they are, so a "//" or "/*" inside one is not a comment.
std::string code_of(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        const char c = text[i];
        if (c == '"' || c == '\'') {
            const std::size_t start = i++;
            while (i < text.size() && text[i] != c && text[i] != '\n') i += text[i] == '\\' ? 2 : 1;
            i = std::min(i + 1, text.size());
            out.append(text, start, i - start);
        } else if (text.compare(i, 2, "//") == 0) {
            while (i < text.size() && text[i] != '\n') ++i;
        } else if (text.compare(i, 2, "/*") == 0) {
            const std::size_t end  = text.find("*/", i + 2);
            const std::size_t stop = end == std::string::npos ? text.size() : end + 2;
            for (; i < stop; ++i) {
                if (text[i] == '\n') out += '\n';
            }
        } else {
            out += c;
            ++i;
        }
    }
    return out;
}

// How a file's definitions are read.
enum class Mode {
    TaggedScalars, // `inline constexpr Scalar|Constant<N> kName{...};`, with their Provenance
    Definitions,   // every `[inline] constexpr <type> name[ [ ( = {` at the left margin
    RowFields,     // every field of struct GenerationRow
    RecordedOnly,  // `constexpr` definitions at any indent; only the recorded keys are checked
};

struct Definition {
    std::string name;
    std::string text; // from the definition to its end (for the rows of a table); the last one
    std::vector<std::string> blocks; // every definition of the name in the file (one, mostly)
};

std::map<std::string, Definition> definitions_of(const std::string& text, Mode mode) {
    std::map<std::string, Definition> out;
    if (mode == Mode::RowFields) {
        const std::size_t at = text.find("struct GenerationRow {");
        // The struct's closing brace stands at column 0; a "};" inside a field's initializer
        // does not end it.
        const std::size_t end = text.find("\n};", at);
        if (at == std::string::npos || end == std::string::npos) return out;
        std::istringstream lines(text.substr(at, end - at));
        // A field of any type (`int major = 0;`, `const char* name = "";`, `bool x = false;`).
        const std::regex field(R"(^\s*(?:const\s+)?[\w:<>]+\s*\*?\s+(\w+)\s*[=;\{])");
        std::string line;
        std::getline(lines, line); // the `struct GenerationRow {` line itself
        for (; std::getline(lines, line);) {
            std::smatch m;
            if (std::regex_search(line, m, field)) out[m[1]] = {m[1], line};
        }
        return out;
    }
    const std::regex tagged(R"(inline constexpr\s+(?:Scalar|Constant<\d+>)\s+(k\w+)\{)");
    const std::regex any(
        std::string(mode == Mode::RecordedOnly ? R"(^\s*(?:static\s+)?)" : "^") +
        R"((?:\[\[nodiscard\]\]\s*)?(?:inline\s+)?constexpr\s+[\w:<>,\s]+?\s(\w+)\s*[\[\(=\{;])");
    std::istringstream lines(text);
    std::vector<std::string> all;
    for (std::string line; std::getline(lines, line);) all.push_back(line);
    for (std::size_t i = 0; i < all.size(); ++i) {
        std::smatch m;
        if (mode == Mode::TaggedScalars) {
            if (!std::regex_search(all[i], m, tagged)) continue;
        } else {
            if (!std::regex_search(all[i], m, any)) continue;
        }
        const std::string name = m[1];
        std::string block      = all[i];
        std::size_t j          = i;
        while (j + 1 < all.size() && block.find("};") == std::string::npos &&
               block.find(");") == std::string::npos && block.back() != ';') {
            block += "\n" + all[++j];
        }
        Definition& d = out[name];
        d.name        = name;
        d.text        = block;
        d.blocks.push_back(block);
    }
    return out;
}

struct Spec {
    std::string_view file;
    Mode mode;
    std::set<std::string_view> not_fitted; // defined, and not a fitted constant
};

const std::vector<Spec>& specs() {
    static const std::vector<Spec> all = {
        {fit::kRouteCost, Mode::TaggedScalars, {}},
        {fit::kChoice,
         Mode::Definitions,
         {
             // counts of the graph (nsys, ncu): facts of the kernels the family launches
             "kDecodeLaunches",
             "kR8DecodeIssue",
             // a policy
             "kTransferGeneration",
             "kMoeChunkTwoCards",
             "kMoeChunkCurveGeneration", // the generation kMoeChunkShare (recorded) was read on
             "kDefaultChunk",
             "kMoeChunks",
             // the kernels' own geometry and thresholds, read off their source
             "kW4a8KquantTokens",
             "kW4a8KquantRows",
             "kW4a8Q3kTile",
             "kW4a8Q3kResident",
             "kMmqSliceTokens",
             "kMmqMinTokens",
             "kMmqTile",
             "kMmqInputBytes",
             "kMmqOutputBytes",
             "kMmqWrittenBytes",
             "kVoltaChunkBytes",
             "kVoltaCutlassMinTokens",
             "kVoltaCopyMinTokens",
             "kExpertJobSlots",
             // derived: the INT8 prompt kernel's kInt8PromptHalfCtaPercent / 100, whose record
             // (int8/prompt_fast_cost.h) holds the holdout
             "kAttentionHalfCta",
             "kFp8AttentionTiledMinWidth",
             "kFp8AttentionMaxSplits",
             "kFp8AttentionSplitKeys",
             "kFp8AttentionKeyTile",
             "kBf16AttentionQueryRows",
             "kAttentionQueryRows",
             "kAttentionTileHeadDim",
             "kAttentionTiledMinWidth",
             "kGemmaChainMinTokens",
             // measurements of accuracy (KLD against the exact reference): data, not fitted
             "kKldDefaultFamily",
             "kQuantKldFamily",
             "kKldTerms",
             "kQuantKldSource",
             "kQuantKldCards",
             "kQuantKld",
             // the benchmark's request mix and the trade (DECISIONS 73: declared conventions)
             "kRequestPrompt",
             "kRequestAnswer",
             "kMixShortRequests",
             "kMixLongRequests",
             "kLongPrompt",
             "kAccuracyWeight",
             // derived, not fitted: the rms of the decode fit's own residuals, recomputed by
             // ninferno_choice_test; and the means of the generation's lab cards and the tolerance
             "kDecodePredictionError",
             "kReferenceCards",
             "kWorthTie",
             // names of an enum's values, no number
             "route_name",
             "axis_name",
         }},
        // The row's identity; the rates are whitepaper facts except the emulated BF16 aggregate
        // (recorded); the tensor core's kind (full-rate FP32-accumulate HMMA) is a chip fact.
        {fit::kGeneration,
         Mode::RowFields,
         {"major", "minor_lo", "minor_hi", "name", "s8", "fp32acc_full_rate"}},
        {fit::kCostConstants, Mode::Definitions, {"kDp4aOpsPerSmClock"}}, // a chip fact
        {fit::kRoundCost, Mode::Definitions, {"kPricedVerifyRows"}},      // a bound, not a fit
        {fit::kMoeShare, Mode::Definitions, {}},
        {fit::kWave, Mode::Definitions, {"fp8_split_tail"}}, // a rule over its parameters
        // Kernels' own fitted launch constants: only the recorded keys are held to definitions.
        {fit::kInt8PromptCost, Mode::RecordedOnly, {}},
        {fit::kRope, Mode::RecordedOnly, {}},
        {fit::kRmsNorm, Mode::RecordedOnly, {}},
        {fit::kMoePrefill, Mode::RecordedOnly, {}},
        {fit::kGdnGating, Mode::RecordedOnly, {}},
        {fit::kRmsNormRope, Mode::RecordedOnly, {}},
        {fit::kGemma4Attention, Mode::RecordedOnly, {}},
        {fit::kQ5LinearAdd, Mode::RecordedOnly, {}},
        {fit::kGdnConv, Mode::RecordedOnly, {}},
        {fit::kQwenAttention, Mode::RecordedOnly, {}},
        {fit::kAttnInputPlan, Mode::RecordedOnly, {}},
        {fit::kW4a8, Mode::RecordedOnly, {}},
        {fit::kSparseMoePrefill, Mode::RecordedOnly, {}},
        {fit::kQwenDecode, Mode::RecordedOnly, {}},
        {fit::kGemma4Kernel, Mode::RecordedOnly, {}},
        {fit::kBf16Vector, Mode::RecordedOnly, {}},
        {fit::kQ8LinearAdd, Mode::RecordedOnly, {}},
        {fit::kCapturePlanner, Mode::RecordedOnly, {}},
        {fit::kMoeInternal, Mode::RecordedOnly, {}},
        {fit::kQ4LinearSwiglu, Mode::RecordedOnly, {}},
    };
    return all;
}

std::vector<const fit::FittedHoldout*> all_records() {
    std::vector<const fit::FittedHoldout*> out;
    for (const auto& r : fit::kRouteCostHoldouts) out.push_back(&r);
    for (const auto& r : fit::kChoiceHoldouts) out.push_back(&r);
    for (const auto& r : fit::kGenerationHoldouts) out.push_back(&r);
    for (const auto& r : fit::kSmallTEtaHoldouts) out.push_back(&r);
    for (const auto& r : fit::kSpecHoldouts) out.push_back(&r);
    for (const auto& r : fit::kWaveHoldouts) out.push_back(&r);
    for (const auto& r : fit::kRouteRuleHoldouts) out.push_back(&r);
    for (const auto& r : fit::kKernelHoldouts) out.push_back(&r);
    return out;
}

// The tags as the compiler reads them: the route cost model's Fitted constants
// (route_cost::kRouteCostConstants) and the literal rules tagged Fitted or scaled from one board
// (route_rules::kRouteRules). A tag the text patterns above miss (split over lines, a `using enum`)
// is still a Fitted entry here.
void check_compiled_tags(const std::vector<const fit::FittedHoldout*>& records,
                         const std::map<std::string, Definition>& route_cost_definitions) {
    namespace rc = ninferno::ops::detail::generic::route_cost;
    namespace rr = ninferno::ops::detail::generic::route_rules;

    // route_cost_constants.h: a record's definition carries the descriptor name of its constant.
    std::set<std::string> fitted_names;
    for (const rc::Listed& c : rc::kRouteCostConstants) {
        if (c.from == rc::Provenance::Fitted) fitted_names.insert(std::string(c.name));
    }
    if (fitted_names.empty()) fail("kRouteCostConstants lists no Fitted constant");
    std::map<std::string, int> covered;
    for (const auto* r : records) {
        if (r->file != fit::kRouteCost) continue;
        const auto it = route_cost_definitions.find(std::string(r->constant));
        if (it == route_cost_definitions.end()) continue; // reported by the text check
        bool fitted = false;
        for (const std::string& name : fitted_names) {
            if (code_of(it->second.text).find('"' + name + '"') != std::string::npos) {
                ++covered[name];
                fitted = true;
            }
        }
        if (!fitted) {
            fail(std::string(r->constant) +
                 ": recorded, but kRouteCostConstants lists it as not Provenance::Fitted");
        }
    }
    for (const std::string& name : fitted_names) {
        if (covered[name] == 0) {
            fail(name + ": Provenance::Fitted in kRouteCostConstants, but no record's definition "
                        "carries it");
        } else if (covered[name] > 1) {
            fail(name + ": recorded " + std::to_string(covered[name]) + " times");
        }
    }

    // route_rules.h: a record is keyed by the row's name.
    std::map<std::string, rr::Basis> rules;
    for (const rr::RouteRule& rule : rr::kRouteRules) rules[std::string(rule.name)] = rule.basis;
    const auto needs_record = [](rr::Basis b) {
        return b == rr::Basis::Fitted || b == rr::Basis::ScaledFromOneBoard;
    };
    std::set<std::string> recorded_rules;
    for (const auto* r : records) {
        if (r->file != fit::kRouteRules) continue;
        const std::string name(r->constant);
        const auto it = rules.find(name);
        if (it == rules.end()) {
            fail(name + ": recorded, but no row of kRouteRules has that name");
        } else if (!needs_record(it->second)) {
            fail(name + ": recorded, but its row's basis is " +
                 std::string(rr::basis_name(it->second)) + ", not Fitted or ScaledFromOneBoard");
        }
        recorded_rules.insert(name);
    }
    for (const auto& [name, basis] : rules) {
        if (needs_record(basis) && !recorded_rules.count(name)) {
            fail(name + ": a kRouteRules row tagged " + std::string(rr::basis_name(basis)) +
                 " without a record");
        }
    }
}

// The reader itself: a definition or a tag inside a comment is not one, a "//" inside a string
// literal starts no comment, and lines stay lines.
void check_comment_stripping() {
    const std::string text = "constexpr int kKept = 1; // constexpr int kLine = 2;\n"
                             "/* constexpr int kBlock = 3;\n"
                             "   constexpr int kBlockToo = 4; */ constexpr int kAfter = 5;\n"
                             "    constexpr int kIndented = 6;\n"
                             "constexpr std::string_view kUrl = \"http://x\";\n"
                             "// Provenance::Fitted\n";
    const std::string code = code_of(text);
    if (std::count(code.begin(), code.end(), '\n') != std::count(text.begin(), text.end(), '\n')) {
        fail("code_of: a comment changed the line count");
    }
    if (code.find("Provenance") != std::string::npos) fail("code_of: kept a line comment");
    if (code.find("\"http://x\"") == std::string::npos) fail("code_of: cut a string literal");
    const auto margin   = definitions_of(code, Mode::Definitions);
    const auto indented = definitions_of(code, Mode::RecordedOnly);
    for (const char* hidden : {"kLine", "kBlock", "kBlockToo"}) {
        if (margin.count(hidden) || indented.count(hidden)) {
            fail(std::string("definitions_of: read ") + hidden + " inside a comment");
        }
    }
    for (const char* seen : {"kKept", "kUrl"}) {
        if (!margin.count(seen)) fail(std::string("definitions_of: missed ") + seen);
    }
    if (!indented.count("kIndented")) fail("definitions_of (RecordedOnly): missed kIndented");
    if (margin.count("kIndented")) fail("definitions_of (Definitions): read an indented line");
}

} // namespace

int main() {
    const auto records = all_records();
    if (records.size() != kExpectedRecords) {
        fail(std::to_string(records.size()) + " records, expected " +
             std::to_string(kExpectedRecords) +
             " (a constant added or dropped: give it a record, update the count)");
    }

    std::set<std::string> seen;
    for (const auto* r : records) {
        const std::string key =
            std::string(r->file) + "|" + std::string(r->constant) + "|" + std::string(r->row);
        const std::string name(r->constant);
        if (!seen.insert(key).second) fail(name + ": recorded twice");
        if (r->holdout.kind == ninferno::HoldoutKind::Unset) {
            fail(name + ": Unset (forgotten, not declared none)");
        } else if (!r->holdout.complete()) {
            fail(name + ": its Holdout leaves a required field empty");
        }
        // A Checked record reproduces its held-out score: its source names a script under
        // tools/fits/ that exists (a score quoted from a fit whose data is not in the repository
        // is a None record).
        if (r->holdout.kind == ninferno::HoldoutKind::Checked) {
            const std::string source(r->holdout.source);
            const std::regex script(R"(tools/fits/[\w./-]+\.py)");
            std::smatch m;
            if (!std::regex_search(source, m, script)) {
                fail(name + ": Checked, but its source names no script under tools/fits/");
            } else if (read_file(m.str()).empty()) {
                fail(name + ": Checked, but its script " + m.str() + " is not in the repository");
            }
        }
    }

    check_comment_stripping();

    std::map<std::string, Definition> route_cost_definitions;
    for (const Spec& spec : specs()) {
        const std::string text = read_file(spec.file);
        if (text.empty()) {
            fail(std::string(spec.file) + ": cannot read");
            continue;
        }
        const auto defined = definitions_of(code_of(text), spec.mode);
        if (defined.empty()) fail(std::string(spec.file) + ": no definitions found");
        if (spec.file == fit::kRouteCost) route_cost_definitions = defined;

        std::set<std::string> recorded;
        for (const auto* r : records) {
            if (r->file != spec.file) continue;
            // A key is the name of a definition; a field of a row is named in words after it.
            const std::string name = std::string(r->constant).substr(0, r->constant.find(' '));
            const auto it          = defined.find(name);
            if (it == defined.end()) {
                fail(std::string(r->constant) + ": recorded, but not defined in " +
                     std::string(spec.file) + " (renamed or removed? update its record)");
                continue;
            }
            recorded.insert(name);
            // A name defined more than once in its file (two kBlock in rmsnorm.cu) is keyed
            // `name = value`, and that definition must stand: renaming it or changing its value
            // fails, and a bare key of a repeated name is ambiguous.
            const std::string key(r->constant);
            const std::size_t eq = key.find(" = ");
            if (eq != std::string::npos) {
                const std::string value = key.substr(eq + 3);
                bool found              = false;
                for (const std::string& block : it->second.blocks) {
                    const std::size_t at = block.find(name);
                    const std::size_t is = block.find('=', at);
                    if (at == std::string::npos || is == std::string::npos) continue;
                    std::string rest = block.substr(is + 1);
                    rest.erase(0, rest.find_first_not_of(" \t"));
                    if (rest.compare(0, value.size(), value) == 0 &&
                        (rest.size() == value.size() || rest[value.size()] == ';' ||
                         rest[value.size()] == ' ')) {
                        found = true;
                    }
                }
                if (!found) {
                    fail(key + ": no definition of " + name + " with that value in " +
                         std::string(spec.file));
                }
            } else if (spec.mode == Mode::RecordedOnly && it->second.blocks.size() > 1) {
                fail(key + ": defined " + std::to_string(it->second.blocks.size()) + " times in " +
                     std::string(spec.file) + "; key the record `" + name + " = <value>`");
            }
            if (!r->row.empty() &&
                !std::regex_search(it->second.text,
                                   std::regex(R"(\{\s*\")" + std::string(r->row) + R"(\")"))) {
                fail(name + ": no row \"" + std::string(r->row) + "\" in its definition");
            }
            if (spec.not_fitted.count(name)) fail(name + ": recorded and listed as not fitted");
        }
        if (spec.mode == Mode::TaggedScalars) {
            // Every tag in the file belongs to a definition read above: a Fitted constant written
            // in a form the definition pattern misses still fails.
            const std::string code = code_of(text);
            std::size_t tags       = 0;
            for (std::size_t at = code.find("Provenance::Fitted"); at != std::string::npos;
                 at             = code.find("Provenance::Fitted", at + 1)) {
                ++tags;
            }
            std::size_t tagged = 0;
            for (const auto& [name, def] : defined) {
                if (code_of(def.text).find("Provenance::Fitted") != std::string::npos) ++tagged;
            }
            if (tags != tagged) {
                fail(std::string(spec.file) + ": " + std::to_string(tags) +
                     " Provenance::Fitted tags but " + std::to_string(tagged) +
                     " tagged definitions read (a definition the pattern does not see?)");
            }
        }
        for (const auto& [name, def] : defined) {
            if (spec.mode == Mode::TaggedScalars) {
                // Only the Fitted-tagged definitions are covered; the others carry no record.
                const bool fitted =
                    code_of(def.text).find("Provenance::Fitted") != std::string::npos;
                if (fitted && recorded.count(name) == 0) fail(name + ": Fitted without a record");
                if (!fitted && recorded.count(name) != 0) fail(name + ": a record but not Fitted");
                continue;
            }
            if (spec.mode == Mode::RecordedOnly)
                continue; // the file's other definitions: not audited
            if (!recorded.count(name) && !spec.not_fitted.count(name)) {
                fail(name + ": defined in " + std::string(spec.file) +
                     " but neither recorded in core/fitted_holdouts.h nor listed not fitted here");
            }
        }
        if (spec.mode == Mode::RowFields) {
            // The row's own kind table ("What a lent row carries") names its fitted and measured
            // fields: each of them has a record, whatever the not-fitted list says.
            const std::regex kind_line(R"(^//\s+(fitted|preference)\s+([\w,\s]+?):)");
            std::istringstream lines(text);
            std::set<std::string> kinds;
            for (std::string line; std::getline(lines, line);) {
                std::smatch m;
                if (!std::regex_search(line, m, kind_line)) continue;
                kinds.insert(m[1]);
                std::istringstream names(m[2].str());
                for (std::string field; std::getline(names, field, ',');) {
                    field.erase(0, field.find_first_not_of(" \t"));
                    field.erase(field.find_last_not_of(" \t") + 1);
                    if (field.empty()) continue;
                    if (!defined.count(field)) {
                        fail(field + ": named " + m[1].str() + " in " + std::string(spec.file) +
                             "'s kind table, but not a field of the row");
                    } else if (!recorded.count(field)) {
                        fail(field + ": named " + m[1].str() + " in " + std::string(spec.file) +
                             "'s kind table, but without a record");
                    }
                }
            }
            if (kinds.size() != 2) {
                fail(std::string(spec.file) + ": the kind table's fitted and preference lines "
                                              "are not found (moved or reworded? update the test)");
            }
        }
        if (spec.mode != Mode::TaggedScalars) {
            for (const std::string_view name : spec.not_fitted) {
                if (!defined.count(std::string(name))) {
                    fail(std::string(name) + ": listed not fitted but not defined in " +
                         std::string(spec.file));
                }
            }
        }
        // A table with a record per row has one for each of its rows.
        for (const auto& [name, def] : defined) {
            std::set<std::string> rows;
            for (const auto* r : records) {
                if (r->file == spec.file && r->constant == name && !r->row.empty()) {
                    rows.insert(std::string(r->row));
                }
            }
            if (rows.empty()) continue;
            const std::regex row(R"(\{\s*\"(\w+)\")");
            std::size_t count = 0;
            for (auto it = std::sregex_iterator(def.text.begin(), def.text.end(), row);
                 it != std::sregex_iterator(); ++it) {
                ++count;
                if (!rows.count((*it)[1]))
                    fail(name + ": row " + (*it)[1].str() + " has no record");
            }
            if (count != rows.size()) fail(name + ": records and rows differ in number");
        }
    }

    check_compiled_tags(records, route_cost_definitions);

    // The Provenance::Fitted tag, wherever it is used in the source tree: every file that tags a
    // constant Fitted is one this test reads with Mode::TaggedScalars, so its Fitted constants are
    // held to their records above. A tag in any other file fails here; so does a route_rules
    // Basis::Fitted or Basis::ScaledFromOneBoard outside route_rules.h, whose rows are held to
    // their records by check_compiled_tags.
    std::set<std::string> tagged_files;
    for (const Spec& spec : specs()) {
        if (spec.mode == Mode::TaggedScalars) tagged_files.insert(std::string(spec.file));
    }
    const std::filesystem::path root(NINFERNO_SOURCE_DIR);
    std::size_t scanned = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root / "src")) {
        if (!entry.is_regular_file()) continue;
        const std::string ext = entry.path().extension().string();
        if (ext != ".h" && ext != ".hpp" && ext != ".cpp" && ext != ".cu" && ext != ".cuh" &&
            ext != ".inc") {
            continue;
        }
        ++scanned;
        const std::string relative = entry.path().lexically_relative(root).generic_string();
        const std::string code     = code_of(read_file(relative));
        // The spellings caught: the tag with any spaces around `::`, and any way to reach the
        // enum under another spelling: `using enum` (after which a bare `Fitted` tags a
        // constant), a `using Alias = ...` or a `typedef` of it. Not caught: a macro or a
        // decltype that names it. Their home headers are read through the compiled tables
        // instead.
        static const std::regex provenance_tag(R"(\bProvenance\s*::\s*Fitted\b)");
        static const std::regex provenance_using(
            R"(\b(?:using\s+enum\s+|using\s+\w+\s*=\s*|typedef\s+)[\w:\s]*\bProvenance\b)");
        static const std::regex basis_tag(R"(\bBasis\s*::\s*(Fitted|ScaledFromOneBoard)\b)");
        static const std::regex basis_using(
            R"(\b(?:using\s+enum\s+|using\s+\w+\s*=\s*|typedef\s+)[\w:\s]*\bBasis\b)");
        const bool provenance = code.find("Provenance") != std::string::npos;
        const bool basis      = code.find("Basis") != std::string::npos;
        if (provenance &&
            (std::regex_search(code, provenance_tag) ||
             std::regex_search(code, provenance_using)) &&
            !tagged_files.count(relative)) {
            fail(relative + ": tags a constant Provenance::Fitted (or aliases Provenance) but is "
                            "not covered by this test (add it to specs() with "
                            "Mode::TaggedScalars and give its constants records)");
        }
        if (basis && (std::regex_search(code, basis_tag) || std::regex_search(code, basis_using)) &&
            relative != fit::kRouteRules) {
            fail(relative +
                 ": tags a rule Basis::Fitted or ScaledFromOneBoard (or aliases Basis) outside "
                 "route_rules.h, which this test does not read");
        }
    }
    if (scanned == 0) fail("no source file found under src/");

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("fitted holdouts: %zu records across %zu files\n", records.size(), specs().size());
    return 0;
}
