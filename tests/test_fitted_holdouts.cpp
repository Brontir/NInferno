// The holdout records (core/fitted_holdouts.h) of the fitted constants outside
// route_cost_constants.h (whose own test, test_route_cost_constants, counts its 25): the choice's
// speed model, the generation row's scales, the speculation round cost's launch time and routed
// share, and the SM-wave rule. Every record must be Checked or an explicit None with the fields its
// kind requires (an Unset one was forgotten and fails); every named constant must still be in its
// source file; the counts are pinned, so adding or dropping a record is a decision; and every table
// or scalar of choice_constants.h must be either recorded here or declared below as a fact, a count
// of the graph or a kernel, a convention or a measurement of accuracy: a new constant cannot skip
// the question "what was it fitted on, and what was left out". No GPU.

#include "core/fitted_holdouts.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

namespace {

namespace fit = ninferno::fitted;

int failures = 0;

void fail(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
}

std::string read_file(const std::string& relative) {
    std::ifstream in(std::filesystem::path(NINFERNO_SOURCE_DIR) / relative, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// What choice_constants.h defines that is not fitted, by the reason it is not.
const std::set<std::string_view>& declared_not_fitted() {
    static const std::set<std::string_view> names = {
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
        // measurements of accuracy (KLD against the exact reference): data, not fitted to speed
        "kKldDefaultFamily",
        "kQuantKldFamily",
        "kKldTerms",
        "kQuantKldSource",
        "kQuantKldCards",
        "kQuantKld",
        // the benchmark's request mix (DECISIONS 73: a convention, not Fitted)
        "kRequestPrompt",
        "kRequestAnswer",
        "kMixShortRequests",
        "kMixLongRequests",
        "kLongPrompt",
        // the means of the generation's lab cards, and the tie tolerance
        "kReferenceCards",
        "kWorthTie",
    };
    return names;
}

void check_records(const std::string& group, const ninferno::fitted::FittedHoldout* begin,
                   const ninferno::fitted::FittedHoldout* end, std::size_t expected,
                   std::set<std::string>& seen) {
    const auto count = static_cast<std::size_t>(end - begin);
    if (count != expected) {
        fail(group + ": " + std::to_string(count) + " records, expected " +
             std::to_string(expected) +
             " (a constant added or dropped: give it a record, update the count)");
    }
    std::string text;
    std::string text_file;
    for (const auto* r = begin; r != end; ++r) {
        const std::string name(r->constant);
        if (!seen.insert(std::string(r->file) + "|" + name).second) fail(name + ": recorded twice");
        if (r->holdout.kind == ninferno::HoldoutKind::Unset) {
            fail(name + ": Unset (forgotten, not declared none)");
        } else if (!r->holdout.complete()) {
            fail(name + ": its Holdout leaves a required field empty");
        }
        if (text_file != r->file) {
            text      = read_file(std::string(r->file));
            text_file = std::string(r->file);
            if (text.empty()) fail(text_file + ": cannot read");
        }
        // The first word of the name must still be in the file (a row field is named in words).
        const std::string word = name.substr(0, name.find(' '));
        if (text.find(word) == std::string::npos) {
            fail(name + ": no longer in " + text_file + " (renamed? update its record)");
        }
    }
}

} // namespace

int main() {
    std::set<std::string> seen;
    check_records("choice", std::begin(fit::kChoiceHoldouts), std::end(fit::kChoiceHoldouts), 29,
                  seen);
    check_records("generation", std::begin(fit::kGenerationHoldouts),
                  std::end(fit::kGenerationHoldouts), 4, seen);
    check_records("spec", std::begin(fit::kSpecHoldouts), std::end(fit::kSpecHoldouts), 2, seen);
    check_records("sm_wave_rules", std::begin(fit::kWaveHoldouts), std::end(fit::kWaveHoldouts), 1,
                  seen);

    // Every constant of choice_constants.h is recorded or declared not fitted.
    std::set<std::string> recorded;
    for (const auto& r : fit::kChoiceHoldouts) recorded.insert(std::string(r.constant));
    const std::string header = read_file(std::string(fit::kChoice));
    const std::regex definition(R"(^inline constexpr\s+[\w:<>]+\s+(k\w+))");
    std::set<std::string> defined;
    std::istringstream lines(header);
    for (std::string line; std::getline(lines, line);) {
        std::smatch m;
        if (std::regex_search(line, m, definition)) defined.insert(m[1]);
    }
    if (defined.size() < 60) fail("choice_constants.h: fewer constants found than expected");
    for (const std::string& name : defined) {
        if (!recorded.count(name) && !declared_not_fitted().count(name)) {
            fail(name + ": defined in choice_constants.h but neither recorded in "
                        "core/fitted_holdouts.h nor declared not fitted in this test");
        }
    }
    for (const std::string& name : recorded) {
        if (!defined.count(name)) fail(name + ": recorded but not defined in choice_constants.h");
    }
    for (const std::string_view name : declared_not_fitted()) {
        if (!defined.count(std::string(name))) {
            fail(std::string(name) + ": declared not fitted but not defined in choice_constants.h");
        }
        if (recorded.count(std::string(name))) {
            fail(std::string(name) + ": both recorded and declared not fitted");
        }
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("fitted holdouts: %zu choice, %zu generation, %zu spec, %zu wave records; "
                "%zu constants of choice_constants.h accounted for\n",
                std::size(fit::kChoiceHoldouts), std::size(fit::kGenerationHoldouts),
                std::size(fit::kSpecHoldouts), std::size(fit::kWaveHoldouts), defined.size());
    return 0;
}
