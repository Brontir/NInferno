// Speculation per measured pair (spec/pairs.h), on the lab's own spec-tune measurements
// (tests/fixtures/spec_pairs: Qwen3.5-9B with DFlash and MTP, gpt-oss-20b with DFlash, on an RTX
// 4070 Ti / 4090 (sm89) and an RTX 5070 Ti (sm120)).

#include "spec/pairs.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace sp = ninferno::spec;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

sp::MeasuredPair load(const std::string& name) {
    std::ifstream in(std::string(NINFERNO_SOURCE_DIR) + "/tests/fixtures/spec_pairs/" + name +
                     ".json");
    return sp::parse_pair(nlohmann::json::parse(in));
}

} // namespace

int main() {
    const auto ada_q4    = load("df9_89");            // 9B Q4_K_M + DFlash, RTX 4070 Ti
    const auto bw_q4     = load("edf_base_q4_120a");  // the same pair, RTX 5070 Ti
    const auto bw_q8     = load("edf_base_q8_120a");
    const auto bw_iq2    = load("edf_base_iq2_120a");
    const auto hauhau    = load("edf_hauhau_120a");   // a fine-tune of 9B, the base's DFlash
    const auto oss_bw    = load("gptoss_dflash");
    const auto oss_ada   = load("gptoss_dflash_89");
    const auto mtp_q4    = load("ev_base_q4_120a");
    const auto mtp_iq2   = load("ev_base_iq2_120a");

    // A pair's JSON reads back as it was written.
    {
        const auto again = sp::parse_pair(sp::pair_json(ada_q4));
        expect(again.target == ada_q4.target && again.generation == "sm89" &&
                   again.tune.measurements.size() == ada_q4.tune.measurements.size(),
               "a pair round-trips through its JSON");
    }
    // On the lab card's generation every measured depth counts: Ada verifies deep DFlash blocks
    // cheaply (dflash:15 thinking). Elsewhere only what carries: up to kPortableDepth drafts, one
    // request at a time.
    {
        const auto home = sp::profile_for_pair(ada_q4, "sm89");
        expect(home.thinking.config.label() == "dflash:15",
               "the lab card's generation takes the measured best depth: " +
                   home.thinking.config.label());
        const auto away = sp::profile_for_pair(ada_q4, "sm120");
        expect(away.thinking.config.enabled() && away.thinking.config.k <= sp::kPortableDepth,
               "another generation takes no deeper than the portable depth: " +
                   away.thinking.config.label());
        expect(away.load.size() == 1 && away.load[0].lanes == 2 && away.load[0].depth == 0,
               "another generation speculates for one request only");
        expect(away.thinking.floor_tokens_per_round > 1.0,
               "a choice carries its break-even for the watchdog");
    }
    // A target nobody measured: its group's depth, best on average over the pairs measured.
    {
        const std::vector<sp::MeasuredPair> others = {ada_q4, bw_q8, hauhau, bw_iq2, oss_bw};
        const auto group = sp::pair_group(others, "qwen3_5", "dflash", 4.9);
        expect(group.size() == 3, "the group: the Qwen3.5 DFlash pairs of 4 bits and more (" +
                                      std::to_string(group.size()) + ")");
        const auto choice = sp::profile_for_group(group, "dflash");
        expect(choice && choice->thinking.config.label() == "dflash:7" &&
                   choice->non_thinking.config.label() == "dflash:7",
               "9B Q4 left out: the group's DFlash depth 7");
        expect(choice && choice->drafters.size() == 1 && choice->drafters[0].status == "preliminary",
               "the group's drafter is preliminary");
        // Measured, the same target took depth 7 too (within a few percent of the group's).
        const auto own = sp::profile_for_pair(bw_q4, "sm120");
        expect(own.thinking.config.label() == "dflash:7", "its own measurement agrees: " +
                                                              own.thinking.config.label());
    }
    // Below 4 bits drafts pay less: a 2-bit target's group is the 2-bit pairs.
    {
        const std::vector<sp::MeasuredPair> mtp = {mtp_q4, mtp_iq2};
        const auto low  = sp::pair_group(mtp, "qwen3_5", "mtp", 2.7);
        const auto high = sp::pair_group(mtp, "qwen3_5", "mtp", 4.9);
        expect(low.size() == 1 && low[0]->target_bits < 4 && high.size() == 1 &&
                   high[0]->target_bits >= 4,
               "the bit class splits the group");
    }
    // gpt-oss without thinking gains 1-2 % over plain decode: at a fine-tune's acceptance it would
    // lose, so a target nobody measured decodes plainly there.
    {
        const std::vector<sp::MeasuredPair> oss = {oss_bw, oss_ada};
        const auto choice = sp::profile_for_group(sp::pair_group(oss, "dense", "dflash", 4.25),
                                                  "dflash");
        expect(choice && !choice->non_thinking.config.enabled(),
               "a group that would not pay at a fine-tune's acceptance decodes plainly");
    }
    // Nothing measured for an engine and kind: no choice.
    {
        const std::vector<sp::MeasuredPair> none = {oss_bw};
        expect(!sp::profile_for_group(sp::pair_group(none, "gemma4", "mtp", 4.5), "mtp"),
               "no pair of the group: no choice");
    }
    // The generation of a device: its chain's architecture key.
    {
        sp::GpuKey gpu;
        gpu.chain = {"ad104:60sm", "ad104", "sm89", "ad102", "*"};
        expect(sp::gpu_generation(gpu) == "sm89", "an Ada card's generation");
    }
    std::cout << (failures == 0 ? "spec pairs: ok\n" : "spec pairs: FAILED\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
