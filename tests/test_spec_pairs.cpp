// Speculation per measured pair (spec/pairs.h), on the lab's own spec-tune measurements
// (tests/fixtures/spec_pairs: Qwen3.5-9B with DFlash and MTP, gpt-oss-20b with DFlash, on an RTX
// 4070 Ti / 4090 (sm89) and an RTX 5070 Ti (sm120)).

#include "spec/manifest.h"
#include "spec/pairs.h"
#include "spec/registry.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
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
    // Deep drafts and load are a generation's: taken where two cards of the generation measured the
    // pair and chose alike. Ada verifies deep DFlash blocks cheaply (dflash:15 thinking on the
    // RTX 4070 Ti); one card is not two, so alone it gives the portable choice.
    {
        const auto one = sp::profile_for_target({&ada_q4}, "sm89");
        expect(one.thinking.config.enabled() && one.thinking.config.k <= sp::kPortableDepth,
               "one card of the generation: the portable depth: " + one.thinking.config.label());
        auto second = ada_q4; // the same measurement on a second Ada card
        second.tune.key.gpu = "ad102:128sm";
        const auto two = sp::profile_for_target({&ada_q4, &second}, "sm89");
        expect(two.thinking.config.label() == "dflash:15",
               "two cards of the generation that agree: the measured best depth: " +
                   two.thinking.config.label());
        expect(two.load.size() == 1 && two.load[0].lanes == 2 && two.load[0].depth == 0,
               "no load bounds measured: one request at a time");
        auto slower = second; // a second card on which depth 15 loses its thinking lead
        for (auto& m : slower.tune.measurements) {
            if (m.config.k == 15) { m.seconds *= 2; }
        }
        // A second card whose plain decode is 2 % faster: every round costs more plain steps (a
        // higher break-even), the order of the configurations and so the choice unchanged.
        auto costlier = second;
        for (auto& m : costlier.tune.measurements) {
            if (!m.config.enabled()) { m.seconds *= 0.98; }
        }
        const auto ab = sp::profile_for_target({&ada_q4, &costlier}, "sm89");
        const auto ba = sp::profile_for_target({&costlier, &ada_q4}, "sm89");
        expect(ab.thinking.config.label() == "dflash:15" &&
                   ab.thinking.floor_tokens_per_round == ba.thinking.floor_tokens_per_round &&
                   ab.thinking.floor_tokens_per_round > two.thinking.floor_tokens_per_round,
               "agreeing cards: the highest break-even, whatever their order");
        const auto twice = sp::profile_for_target({&ada_q4, &ada_q4}, "sm89");
        expect(twice.thinking.config.k <= sp::kPortableDepth,
               "one card's measurement twice is one card: " + twice.thinking.config.label());
        const auto split = sp::profile_for_target({&ada_q4, &slower}, "sm89");
        expect(split.thinking.config.k <= sp::kPortableDepth,
               "two cards that disagree: the portable depth: " + split.thinking.config.label());
        const auto away = sp::profile_for_target({&ada_q4, &second}, "sm120");
        expect(away.thinking.config.enabled() && away.thinking.config.k <= sp::kPortableDepth,
               "another generation takes no deeper than the portable depth: " +
                   away.thinking.config.label());
        expect(away.load.size() == 1 && away.load[0].lanes == 2 && away.load[0].depth == 0,
               "another generation speculates for one request only");
        expect(away.thinking.floor_tokens_per_round > 1.0,
               "a choice carries its break-even for the watchdog");
    }
    // The identity of weights: sorted digests, so the listing's order does not matter.
    expect(sp::weights_identity({"bb", "aa"}) == "aa,bb", "weights identity is order-free");
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
        const auto own = sp::profile_for_pair(bw_q4);
        expect(own.thinking.config.label() == "dflash:7", "its own measurement agrees: " +
                                                              own.thinking.config.label());
    }
    // Below 4 bits drafts pay less: a 2-bit target's group is the 2-bit pairs.
    {
        // The same pairs on a second card (an Ada one).
        auto q4_ada = mtp_q4, iq2_ada = mtp_iq2;
        q4_ada.tune.key.gpu = iq2_ada.tune.key.gpu = "ad104:60sm";
        q4_ada.generation = iq2_ada.generation = "sm89";
        const std::vector<sp::MeasuredPair> mtp = {mtp_q4, mtp_iq2, q4_ada, iq2_ada};
        const auto low  = sp::pair_group(mtp, "qwen3_5", "mtp", 2.7);
        const auto high = sp::pair_group(mtp, "qwen3_5", "mtp", 4.9);
        expect(low.size() == 2 && low[0]->target_bits < 4 && high.size() == 2 &&
                   high[0]->target_bits >= 4,
               "the bit class splits the group");
        // One card's measurements are no lab constant: no group from them.
        const std::vector<sp::MeasuredPair> one = {mtp_q4, mtp_iq2, mtp_q4};
        expect(sp::pair_group(one, "qwen3_5", "mtp", 4.9).empty() &&
                   sp::pair_group(one, "qwen3_5", "mtp", 2.7).empty(),
               "a group measured on one card is no group");
        // Another drafter of the same mode is not this kind's group.
        expect(sp::pair_group(mtp, "qwen3_5", "gemma4_assistant", 4.9).empty(),
               "a group is of the drafter kind, not only of its mode");
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
    // A pair another speculation engine measured: no choice from it, not in a group.
    {
        auto old           = bw_q8;
        old.tune.key.engine = "ninferno-spec-0";
        const auto p       = sp::profile_for_pair(old);
        expect(!sp::pair_current(old) && !p.overall.config.enabled() && p.disabled.has_value(),
               "an old engine version's pair decodes plainly");
        const std::vector<sp::MeasuredPair> stale = {old};
        expect(sp::pair_group(stale, "qwen3_5", "dflash", 8.5).empty(),
               "an old engine version's pair is in no group");
    }
    // What the registry's pairs give a target: its own measurement, else its group's.
    {
        auto mtp_ada         = mtp_q4; // its MTP measurement on a second card
        mtp_ada.tune.key.gpu = "ad104:60sm";
        mtp_ada.generation   = "sm89";
        const std::vector<sp::MeasuredPair> all = {ada_q4, bw_q4,  bw_q8,  bw_iq2,
                                                   hauhau, oss_bw, mtp_q4, mtp_ada};
        const auto own = sp::pair_choice(all, bw_q4.target, "qwen3_5", {"dflash"}, 4.9, "sm120");
        expect(own && own->basis == "measured" && own->profile.thinking.config.label() == "dflash:7",
               "a measured target takes its own measurement");
        const auto unknown =
            sp::pair_choice(all, "sha:not-measured", "qwen3_5", {"dflash"}, 4.9, "sm120");
        expect(unknown && unknown->basis == "preliminary" &&
                   unknown->profile.thinking.config.mode == sp::Mode::DFlash,
               "an unmeasured target takes its group's");
        // Measured with DFlash, carrying only its MTP layers: unmeasured for those.
        const auto other = sp::pair_choice(all, bw_q4.target, "qwen3_5", {"mtp"}, 4.9, "sm120");
        expect(other && other->basis == "preliminary" &&
                   other->profile.thinking.config.mode != sp::Mode::DFlash,
               "measured with another drafter: the carried drafter's group");
        expect(!sp::pair_choice(all, "sha:x", "gemma4", {"gemma4_assistant"}, 4.5, "sm120"),
               "no measurement and no group: nothing");
    }
    // One card's tunes of one target, a drafter each (MTP, then DFlash): one measurement of the
    // pair over both drafters, in either order.
    {
        auto mtp   = mtp_q4;
        mtp.target = bw_q4.target;
        for (const auto& pairs : {std::vector<sp::MeasuredPair>{mtp, bw_q4},
                                  std::vector<sp::MeasuredPair>{bw_q4, mtp}}) {
            const auto both =
                sp::pair_choice(pairs, bw_q4.target, "qwen3_5", {"mtp", "dflash"}, 4.9, "sm120");
            expect(both && both->basis == "measured" &&
                       both->profile.thinking.config.label() == "dflash:7" &&
                       both->source.find("(1 lab card(s))") != std::string::npos,
                   "one card's MTP and DFlash tunes are pooled: DFlash, the faster, chosen");
        }
    }
    // Each request at its own thinking flag's depth, or all at the default's (spec::
    // per_request_gain): on Ada the 9B's DFlash drafts 15 with thinking and 7 without. With
    // thinking on by default the requests without it lose 2 % at depth 15 (within the
    // measurements' spread: one depth serves); with it off, those with it lose 16 % at depth 7.
    {
        sp::Profile p = ada_q4.tune;
        sp::select(p);
        expect(p.thinking.config.label() == "dflash:15" && p.non_thinking.config.label() == "dflash:7",
               "Ada 9B DFlash: 15 with thinking, 7 without: " + p.thinking.config.label() + " / " +
                   p.non_thinking.config.label());
        const double on  = sp::per_request_gain(p, true);
        const double off = sp::per_request_gain(p, false);
        expect(on > 1.0 && on < 1.03, "thinking by default: the others lose under 3 %: " + std::to_string(on));
        expect(off > 1.10, "thinking off by default: the thinking requests lose over 10 %: " +
                               std::to_string(off));
        sp::Profile same = bw_q4.tune;
        sp::select(same);
        expect(sp::per_request_gain(same, true) == 1.0 && sp::per_request_gain(same, false) == 1.0,
               "one configuration for both flags: nothing to gain");
    }
    // A record of an older engine version never pools with the current one, in either order.
    {
        auto stale       = bw_q4;
        stale.tune.key.engine = "stale-engine";
        for (auto& m : stale.tune.measurements) {
            if (m.config.enabled()) { m.seconds *= 0.25; } // it would win were it pooled
        }
        const auto alone = sp::pair_choice({bw_q4}, bw_q4.target, "qwen3_5", {"dflash"}, 4.9, "sm120");
        const auto first = sp::pair_choice({stale, bw_q4}, bw_q4.target, "qwen3_5", {"dflash"}, 4.9, "sm120");
        const auto last  = sp::pair_choice({bw_q4, stale}, bw_q4.target, "qwen3_5", {"dflash"}, 4.9, "sm120");
        expect(alone && first && last && first->basis == "measured" && last->basis == "measured" &&
                   first->profile.thinking.config == alone->profile.thinking.config &&
                   last->profile.thinking.config == alone->profile.thinking.config &&
                   first->profile.thinking.tok_s == alone->profile.thinking.tok_s &&
                   last->profile.thinking.tok_s == alone->profile.thinking.tok_s,
               "a stale record leaves the choice as the current one alone makes it");
    }
    // A pair's fastest configuration the family's Engine does not run (Qwen's MTP drafts at most
    // 5): its served choices are what the Engine runs.
    {
        auto deep = mtp_q4;
        for (auto& m : deep.tune.measurements) {
            if (m.config.enabled() && m.config.k == 4) {
                m.config.k = 7;
                m.seconds *= 0.5;
            }
        }
        const auto choice = sp::pair_choice({deep}, deep.target, "qwen3_5", {"mtp"}, 4.9, "sm120");
        const auto runs   = [](const sp::Choice& c) { return !c.config.enabled() || c.config.k <= 5; };
        expect(choice && choice->profile.served &&
                   runs(choice->profile.served->overall) && runs(choice->profile.served->thinking) &&
                   runs(choice->profile.served->non_thinking),
               "the Engine's choices stand beside a faster one it does not run");
    }
    // Bits per weight from a quantization label.
    expect(sp::label_bits("Q4_K_M") == 4 && sp::label_bits("UD-IQ2_M") == 2 &&
               sp::label_bits("MXFP4") == 4 && sp::label_bits("") == 16 &&
               sp::label_bits("TQ1_0") == 1 && sp::label_bits("UD-TQ2_0") == 2,
           "quantization labels' bits");
    // The generation of a device: its chain's architecture key.
    {
        sp::GpuKey gpu;
        gpu.chain = {"ad104:60sm", "ad104", "sm89", "ad102", "*"};
        expect(sp::gpu_generation(gpu) == "sm89", "an Ada card's generation");
    }
    // A pair's profile keeps the model's published floors (a head fallback reads them), as a
    // published profile does: the registry with this pair, resolved for a planned target.
    {
        std::ifstream in(std::string(NINFERNO_SOURCE_DIR) + "/src/spec/data/drafter_registry.json");
        nlohmann::json doc = nlohmann::json::parse(in);
        doc["pairs"]       = nlohmann::json::array({sp::pair_json(ada_q4)});
        const auto path    = std::filesystem::temp_directory_path() / "ninferno_spec_pairs_registry.json";
        std::ofstream(path) << doc.dump();
        sp::Registry::use_file(path);
        const sp::ModelEntry* entry = sp::Registry::builtin().by_id("qwen3.5-9b");
        expect(entry != nullptr, "the registry has qwen3.5-9b");
        if (entry != nullptr) {
            sp::GpuKey gpu;
            gpu.chain = {"ad104:60sm", "ad104", "sm89", "ad102", "*"};
            const auto resolved = sp::resolve_planned_speculation(
                std::nullopt, entry, {"mtp", "dflash"}, gpu, "planned", ada_q4.target, "Q4_K_M",
                "qwen3_5");
            expect(resolved.basis == "measured" && resolved.floor_sources.is_array() &&
                       !resolved.floor_sources.empty(),
                   "a pair's profile keeps the model's published floors: " + resolved.basis);
        }
        std::filesystem::remove(path);
    }
    std::cout << (failures == 0 ? "spec pairs: ok\n" : "spec pairs: FAILED\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
