// The holdout records (core/fitted_holdouts.h) of every fitted constant, one side table for the
// whole code base. The checks:
//   - every record is complete (a Checked or an explicit None with the fields its kind needs; an
//     Unset one was forgotten) and appears once;
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
//   - the Provenance::Fitted and Basis::Fitted / ScaledFromOneBoard tags are found by a scan of the
//     whole source tree: a file that uses one and is not covered here fails, so a tagged constant
//     cannot sit outside the audit;
//   - a few kernels' own fitted launch constants, untagged, have records whose keys are definitions
//     in their files (those files' other definitions are not audited);
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

constexpr std::size_t kExpectedRecords = 84;

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

// The text without its line comments (a `//` to the end of its line): a tag named in a comment is
// not a tag.
std::string code_of(const std::string& text) {
    std::string out;
    std::istringstream lines(text);
    for (std::string line; std::getline(lines, line);) {
        out += line.substr(0, line.find("//"));
        out += '\n';
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
    std::string text; // from the definition to its end (for the rows of a table)
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
        std::string(mode == Mode::RecordedOnly ? R"(^\s*)" : "^") +
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
        out[name] = {name, block};
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
        {fit::kInt8PromptLaunch, Mode::RecordedOnly, {}},
        {fit::kRope, Mode::RecordedOnly, {}},
        {fit::kRmsNorm, Mode::RecordedOnly, {}},
        {fit::kMoePrefill, Mode::RecordedOnly, {}},
        {fit::kGdnGating, Mode::RecordedOnly, {}},
        {fit::kRmsNormRope, Mode::RecordedOnly, {}},
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
            if (it->second.text.find('"' + name + '"') != std::string::npos) {
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
    }

    std::map<std::string, Definition> route_cost_definitions;
    for (const Spec& spec : specs()) {
        const std::string text = read_file(spec.file);
        if (text.empty()) {
            fail(std::string(spec.file) + ": cannot read");
            continue;
        }
        const auto defined = definitions_of(text, spec.mode);
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
        if (code.find("Provenance::Fitted") != std::string::npos && !tagged_files.count(relative)) {
            fail(relative + ": tags a constant Provenance::Fitted but is not covered by this test "
                            "(add it to specs() with Mode::TaggedScalars and give its constants "
                            "records)");
        }
        if ((code.find("Basis::Fitted") != std::string::npos ||
             code.find("Basis::ScaledFromOneBoard") != std::string::npos) &&
            relative != fit::kRouteRules) {
            fail(relative + ": tags a rule Basis::Fitted or ScaledFromOneBoard outside "
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
