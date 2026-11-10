// The holdout records (core/fitted_holdouts.h) of every fitted constant, one side table for the
// whole code base. The checks:
//   - every record is complete (a Checked or an explicit None with the fields its kind needs; an
//     Unset one was forgotten) and appears once;
//   - every key matches its DEFINITION in its file (a definition, a struct field or a function,
//   read
//     from the file's text, not a substring of it): a renamed or removed constant fails;
//   - every constant a covered file marks or behaves as fitted has a record, or sits on that file's
//     not-fitted list below with the reason it is none: route_cost_constants.h (every definition
//     tagged Provenance::Fitted), choice_constants.h (every definition), core/generation.h (the
//     double fields of the row), core/cost_constants.h, spec/round_cost.h, spec/moe_routed_share.h
//     and ops/common/sm_wave_rules.h (every definition);
//   - the total is pinned, so adding or dropping a record is a decision.
// No GPU.

#include "core/fitted_holdouts.h"

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

constexpr std::size_t kExpectedRecords = 61;

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

// How a file's definitions are read.
enum class Mode {
    TaggedScalars, // `inline constexpr Scalar|Constant<N> kName{...};`, with their Provenance
    Definitions,   // every `[inline] constexpr <type> name[ [ ( = {` at the left margin
    RowFields,     // the `double` fields of struct GenerationRow
};

struct Definition {
    std::string name;
    std::string text; // from the definition to its end (for the rows of a table)
};

std::map<std::string, Definition> definitions_of(const std::string& text, Mode mode) {
    std::map<std::string, Definition> out;
    if (mode == Mode::RowFields) {
        const std::size_t at  = text.find("struct GenerationRow {");
        const std::size_t end = text.find("};", at);
        std::istringstream lines(text.substr(at, end - at));
        const std::regex field(R"(^\s*double\s+(\w+))");
        for (std::string line; std::getline(lines, line);) {
            std::smatch m;
            if (std::regex_search(line, m, field)) out[m[1]] = {m[1], line};
        }
        return out;
    }
    const std::regex tagged(R"(inline constexpr\s+(?:Scalar|Constant<\d+>)\s+(k\w+)\{)");
    const std::regex any(R"(^(?:inline )?constexpr\s+[\w:<>]+\s+(\w+)\s*[\[\(=\{;])");
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
             "kVoltaCopyMinDramBps",
             "kExpertJobSlots",
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
         }},
        // the rates are whitepaper facts except the emulated BF16 aggregate (recorded)
        {fit::kGeneration, Mode::RowFields, {"s8"}},
        {fit::kCostConstants, Mode::Definitions, {"kDp4aOpsPerSmClock"}}, // a chip fact
        {fit::kRoundCost, Mode::Definitions, {"kPricedVerifyRows"}},      // a bound, not a fit
        {fit::kMoeShare, Mode::Definitions, {}},
        {fit::kWave, Mode::Definitions, {"fp8_split_tail"}}, // a rule over its parameters
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
    return out;
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

    for (const Spec& spec : specs()) {
        const std::string text = read_file(spec.file);
        if (text.empty()) {
            fail(std::string(spec.file) + ": cannot read");
            continue;
        }
        const auto defined = definitions_of(text, spec.mode);
        if (defined.empty()) fail(std::string(spec.file) + ": no definitions found");

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
                it->second.text.find("{\"" + std::string(r->row) + "\"") == std::string::npos) {
                fail(name + ": no row \"" + std::string(r->row) + "\" in its definition");
            }
            if (spec.not_fitted.count(name)) fail(name + ": recorded and listed as not fitted");
        }
        for (const auto& [name, def] : defined) {
            if (spec.mode == Mode::TaggedScalars) {
                // Only the Fitted-tagged definitions are covered; the others carry no record.
                const bool fitted = def.text.find("Provenance::Fitted") != std::string::npos;
                if (fitted && recorded.count(name) == 0) fail(name + ": Fitted without a record");
                if (!fitted && recorded.count(name) != 0) fail(name + ": a record but not Fitted");
                continue;
            }
            if (!recorded.count(name) && !spec.not_fitted.count(name)) {
                fail(name + ": defined in " + std::string(spec.file) +
                     " but neither recorded in core/fitted_holdouts.h nor listed not fitted here");
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
            const std::regex row(R"(\{\"(\w+)\")");
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

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("fitted holdouts: %zu records across %zu files\n", records.size(), specs().size());
    return 0;
}
