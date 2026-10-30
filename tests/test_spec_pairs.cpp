// Speculation per measured pair (spec/pairs.h), on the lab's own spec-tune measurements
// (tests/fixtures/spec_pairs: Qwen3.5-9B with DFlash and MTP, gpt-oss-20b with DFlash, on an RTX
// 4070 Ti / 4090 (sm89) and an RTX 5070 Ti (sm120)).

#include "core/quant_label.h"
#include "core/hub_revision.h"
#include "spec/manifest.h"
#include "spec/pairs.h"
#include "spec/registry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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
    // Another generation's measurements, no facts for this one: the median round cost per
    // configuration of every lab card that measured the target, whatever the records' order,
    // said as a transfer (tokens a round are the pair's on any card).
    {
        const auto on = [](sp::MeasuredPair p, const char* gpu, const char* gen, double scale) {
            p.tune.key.gpu = gpu;
            p.generation   = gen;
            for (auto& m : p.tune.measurements) {
                if (m.config.enabled()) { m.seconds *= scale; }
            }
            return p;
        };
        const auto a = on(ada_q4, "ad104:60sm", "sm89", 1.0);
        const auto b = on(ada_q4, "ad102:128sm", "sm89", 1.0);
        const auto c = on(ada_q4, "ga106:28sm", "sm86", 1.2);
        const auto d = on(ada_q4, "gb203:70sm", "sm120", 1.1);
        const auto e = on(ada_q4, "gv100:80sm", "sm70", 1.3);
        const auto base = sp::profile_round_costs(a.tune);
        std::string note;
        const auto turing = sp::profile_for_target({&a, &b, &c, &d, &e}, "sm75", {}, nullptr, &note);
        const auto cost   = sp::profile_round_costs(turing);
        bool median = !cost.empty();
        for (const auto& [label, r] : cost) {
            median = median && std::abs(r / base.at(label) - 1.1) < 1e-9; // 1, 1, 1.1, 1.2, 1.3
        }
        expect(median, "no facts and no lab card of the generation: the median of every lab card's R");
        expect(note == "transferred: median of 5 lab cards of other generations (no card rates)",
               "the transfer is said: " + note);
        expect(turing.thinking.config.k <= sp::kPortableDepth && turing.load.size() == 1,
               "a transfer takes the portable depths, one request at a time");
        // Every order of the records gives the same profile.
        const std::vector<const sp::MeasuredPair*> records = {&a, &b, &c, &d, &e};
        std::vector<std::size_t> idx = {0, 1, 2, 3, 4};
        const auto json = sp::to_json(turing).dump();
        bool same = true;
        do {
            std::vector<const sp::MeasuredPair*> shuffled;
            for (const auto i : idx) { shuffled.push_back(records[i]); }
            std::string n;
            same = same && sp::to_json(sp::profile_for_target(shuffled, "sm75", {}, nullptr, &n))
                                   .dump() == json && n == note;
        } while (same && std::next_permutation(idx.begin(), idx.end()));
        expect(same, "the records' order never moves the profile or its source");
        // A configuration only some cards measured: the median of those that did.
        auto partial = c;
        std::erase_if(partial.tune.measurements, [](const auto& m) { return m.config.label() == "dflash:7"; });
        const auto some = sp::profile_round_costs(sp::profile_for_target({&a, &partial, &e}, "sm75"));
        expect(some.contains("dflash:7") && std::abs(some.at("dflash:7") / base.at("dflash:7") - 1.15) < 1e-9,
               "a configuration's median over the cards that measured it (1, 1.3)");
        // The base is the record that measured the most configurations: a card that measured only
        // part of the menu (and sorts first by class id) does not drop the others' configurations.
        auto first_part = on(ada_q4, "aa000:1sm", "sm86", 1.0);
        std::erase_if(first_part.tune.measurements,
                      [](const auto& m) { return m.config.label() == "dflash:7"; });
        const auto kept = sp::profile_round_costs(sp::profile_for_target({&first_part, &e}, "sm75"));
        expect(kept.contains("dflash:7"), "a configuration the first card did not measure stays");
        // Two lab cards of the generation: their median, on the first by GPU class id; their
        // order never matters either.
        std::string home_note;
        const auto f  = on(ada_q4, "ga104:48sm", "sm86", 1.4);
        const auto ab = sp::profile_for_target({&c, &f}, "sm86", {}, nullptr, &home_note);
        const auto ba = sp::profile_for_target({&f, &c}, "sm86");
        const auto r2 = sp::profile_round_costs(ab);
        expect(!r2.empty() && std::abs(r2.begin()->second / base.at(r2.begin()->first) - 1.3) < 1e-9 &&
                   sp::to_json(ab).dump() == sp::to_json(ba).dump() &&
                   home_note == "rounds: the median of 2 lab cards of ampere",
               "two lab cards of the generation: their median R, whatever their order: " + home_note);
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
        // Two boards of one class and a record of that class naming no board, in either order:
        // the named boards stand as two cards (the unnamed record no longer hides them).
        auto board_a = mtp_q4, board_b = mtp_q4, unnamed = mtp_q4;
        board_a.tune.key.gpu = board_b.tune.key.gpu = unnamed.tune.key.gpu = "ad104:60sm";
        board_a.generation = board_b.generation = unnamed.generation = "sm89";
        board_a.lab_board = 0x2782;
        board_b.lab_board = 0x2783;
        unnamed.lab_board = 0;
        unnamed.tune.drafters.front().status = "aaa-first";
        for (const auto& order : {std::vector<sp::MeasuredPair>{unnamed, board_a, board_b},
                                  std::vector<sp::MeasuredPair>{board_b, board_a, unnamed}}) {
            const auto group = sp::pair_group(order, "qwen3_5", "mtp", 4.9);
            expect(group.size() == 2 && group[0]->lab_board != 0 && group[1]->lab_board != 0,
                   "named boards of one class stand beside a record naming none (" +
                       std::to_string(group.size()) + ")");
        }
    }
    // Three cards, no depth all of them measured (one 1-3, one 3-5, one 1 and 5): each depth on
    // two cards stands, the group speculates (it decoded plainly for want of a common label).
    {
        const auto keep = [](sp::MeasuredPair pair, const std::string& gpu,
                             std::initializer_list<const char*> labels) {
            pair.tune.key.gpu = gpu;
            std::erase_if(pair.tune.measurements, [&](const sp::Measurement& m) {
                if (!m.config.enabled()) { return false; }
                const auto label = m.config.label();
                return std::none_of(labels.begin(), labels.end(),
                                    [&](const char* l) { return label == l; });
            });
            return pair;
        };
        const std::vector<sp::MeasuredPair> apart = {
            keep(mtp_q4, "gb203:70sm", {"mtp:1", "mtp:2", "mtp:3"}),
            keep(mtp_q4, "ad104:60sm", {"mtp:3", "mtp:4", "mtp:5"}),
            keep(mtp_q4, "ad102:128sm", {"mtp:1", "mtp:5"})};
        const auto group = sp::pair_group(apart, "qwen3_5", "mtp", 4.9);
        const auto choice = sp::profile_for_group(group, "mtp");
        expect(group.size() == 3 && choice && choice->overall.config.enabled() &&
                   choice->overall.speedup() >= sp::kPreliminaryGain,
               "no depth common to every card: the group still speculates (" +
                   (choice ? choice->overall.config.label() : std::string("none")) + ")");
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
    // Not measured, but a sibling quantization of its repository was (choice/DECISIONS 48): the
    // sibling nearest in bits within its bit class, not the group's fine-tune guard.
    {
        auto q4        = mtp_q4;
        q4.target      = "sha:q4";
        q4.target_repo = "org/m-GGUF@c0ffee Q4_K_M";
        q4.target_bits = 4;
        auto q8        = q4;
        q8.target      = "sha:q8";
        q8.target_repo = "org/m-GGUF@c0ffee Q8_0";
        q8.target_bits = 8;
        auto q5xl        = q4;
        q5xl.target      = "sha:q5xl";
        q5xl.target_repo = "org/m-GGUF@c0ffee UD-Q5_K_XL";
        q5xl.target_bits = 5;
        const std::vector<sp::MeasuredPair> all = {q4, q8, bw_iq2, mtp_iq2};
        // The nearest of one card's siblings, not its first (one card's records are pooled).
        const std::vector<sp::MeasuredPair> three = {q4, q5xl, q8};
        const auto q5m = sp::pair_choice(three, "sha:q5m", "qwen3_5", {"mtp"}, 5, "sm120", {}, {},
                                         "org/m-GGUF@c0ffee");
        const auto q8x = sp::pair_choice(three, "sha:q8x", "qwen3_5", {"mtp"}, 8, "sm120", {}, {},
                                         "org/m-GGUF@c0ffee");
        expect(q5m && q5m->source.find("sibling org/m-GGUF UD-Q5_K_XL (") != std::string::npos &&
                   q8x && q8x->source.find("sibling org/m-GGUF Q8_0 (") != std::string::npos,
               "the sibling of the same bits among one card's siblings");
        const auto q5 = sp::pair_choice(all, "sha:q5", "qwen3_5", {"mtp"}, 5, "sm120", {}, {},
                                        "org/m-GGUF@c0ffee");
        expect(q5 && q5->basis == "preliminary" &&
                   q5->source.find("registry sibling org/m-GGUF Q4_K_M (") == 0 &&
                   q5->profile.overall.config.label() ==
                       sp::pair_choice(all, "sha:q4", "qwen3_5", {"mtp"}, 4, "sm120")
                           ->profile.overall.config.label(),
               "an unmeasured quantization takes its nearest measured sibling's choice");
        const auto q6 = sp::pair_choice(all, "sha:q6", "qwen3_5", {"mtp"}, 6, "sm120", {}, {},
                                        "org/m-GGUF@c0ffee");
        expect(q6 && q6->source.find("Q4_K_M, Q8_0") != std::string::npos,
               "two siblings equally near: both");
        const auto q2 = sp::pair_choice(all, "sha:q2", "qwen3_5", {"mtp"}, 2, "sm120", {}, {},
                                        "org/m-GGUF@c0ffee");
        const auto other = sp::pair_choice(all, "sha:q5", "qwen3_5", {"mtp"}, 5, "sm120", {}, {},
                                           "org/other-GGUF@c0ffee");
        expect((!q2 || q2->source.find("sibling") == std::string::npos) &&
                   (!other || other->source.find("sibling") == std::string::npos),
               "no sibling of its bit class, or of another repository: the group");
        // The nearest sibling measured with another drafter than the one attached gives way to
        // the next one measured with it, not to the group.
        auto old_q5      = q5xl;
        old_q5.drafters  = {{"mtp", "org/m-GGUF@0ld"}};
        auto with_q8     = q8;
        with_q8.drafters = {{"mtp", "org/m-GGUF@c0ffee"}};
        const std::vector<sp::MeasuredPair> drafted = {old_q5, with_q8};
        const auto q6x = sp::pair_choice(drafted, "sha:q6x", "qwen3_5", {"mtp"}, 6, "sm120", {},
                                         {{"mtp", "org/m-GGUF@c0ffee"}}, "org/m-GGUF@c0ffee");
        expect(q6x && q6x->source.find("sibling org/m-GGUF Q8_0 (") != std::string::npos,
               "a nearer sibling of another drafter: the next one of the attached drafter");
        // Another commit of the repository may hold other weights (a new version of the model):
        // no sibling from it, nor where the target's commit is not known.
        const auto moved = sp::pair_choice(all, "sha:q5", "qwen3_5", {"mtp"}, 5, "sm120", {}, {},
                                           "org/m-GGUF@beef");
        const auto unknown_commit = sp::pair_choice(all, "sha:q5", "qwen3_5", {"mtp"}, 5, "sm120",
                                                    {}, {}, "org/m-GGUF");
        const auto prefix = sp::pair_choice(all, "sha:q5", "qwen3_5", {"mtp"}, 5, "sm120", {}, {},
                                            "Org/M-GGUF@c0ffee0123");
        expect((!moved || moved->source.find("sibling") == std::string::npos) &&
                   (!unknown_commit || unknown_commit->source.find("sibling") == std::string::npos),
               "a sibling of the same commit only");
        expect(prefix && prefix->source.find("sibling Org/M-GGUF Q4_K_M (") != std::string::npos,
               "the same commit spelled in full, the repository's case aside");
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
    // thinking on by default the requests without it lose 4 % at depth 15 on the benchmark mix
    // (2 % with the sections weighing the same: the mix leans to prose, where depth 15 loses);
    // with it off, those with it lose 16 % at depth 7.
    {
        sp::Profile p = ada_q4.tune;
        sp::select(p);
        expect(p.thinking.config.label() == "dflash:15" && p.non_thinking.config.label() == "dflash:7",
               "Ada 9B DFlash: 15 with thinking, 7 without: " + p.thinking.config.label() + " / " +
                   p.non_thinking.config.label());
        const double on  = sp::per_request_gain(p, true);
        const double off = sp::per_request_gain(p, false);
        expect(on > 1.03 && on < 1.05, "thinking by default: the others lose 4 %: " + std::to_string(on));
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
    // A pair measured with another drafter of the kind (the registry replaced its repository or
    // revision) is not measured for the one attached; the same drafter, or one not recorded, is.
    {
        auto with = bw_q4;
        with.drafters = {{"dflash", "z-lab/Qwen3.5-9B-DFlash@5fc3b3d4"}};
        const auto same = sp::pair_choice({with}, with.target, "qwen3_5", {"dflash"}, 4.9, "sm120", {},
                                          {{"dflash", "z-lab/Qwen3.5-9B-DFlash@5fc3b3d474760f18"}});
        const auto other = sp::pair_choice({with}, with.target, "qwen3_5", {"dflash"}, 4.9, "sm120", {},
                                           {{"dflash", "z-lab/Qwen3.5-9B-DFlash@0123abcd"}});
        expect(same && same->basis == "measured", "the drafter measured: measured");
        expect(!other || other->basis != "measured", "another revision of the drafter: not measured");
        const auto same_revision = [](const char* a, const char* b) {
            return ninferno::HubRevision::parse(a).same_as(ninferno::HubRevision::parse(b));
        };
        expect(same_revision("Org/D@abc", "org/d@abcdef") && !same_revision("org/d@abc", "org/d@abd") &&
                   !same_revision("org/d", "org/e") && same_revision("", "org/d@x") &&
                   ninferno::HubRevision::parse("org/d@abc").id() == "org/d@abc" &&
                   ninferno::HubRevision{"org/d", ""}.id() == "org/d",
               "drafter identities");
        const auto back = sp::parse_pair(sp::pair_json(with));
        expect(back.drafters == with.drafters, "the measured drafters round-trip through the JSON");
    }
    // Bits per weight from a quantization label.
    expect(sp::label_bits("Q4_K_M") == 4 && sp::label_bits("UD-IQ2_M") == 2 &&
               sp::label_bits("MXFP4") == 4 && sp::label_bits("") == 16 &&
               sp::label_bits("TQ1_0") == 1 && sp::label_bits("UD-TQ2_0") == 2 &&
               sp::label_bits("F32") == 32 && sp::label_bits("model") == 16,
           "quantization labels' bits");
    expect(ninferno::quant_label_bits("UD-Q4_K_XL") == 4 && ninferno::quant_label_bits("bf16") == 16 &&
               ninferno::quant_label_bits("Q8_0") == 8 && !ninferno::quant_label_bits("") &&
               !ninferno::quant_label_bits("model") && !ninferno::quant_label_bits("aq4"),
           "a label's width, none where it names none");
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
        // The pair measured with DFlash and with MTP (the RTX 5070 Ti's MTP runs stand in): the
        // choice drafts with DFlash, and MTP, measured too, is attached for the drafter axis.
        auto both = ada_q4;
        for (const auto& m : mtp_q4.tune.measurements) {
            if (m.config.enabled()) { both.tune.measurements.push_back(m); }
        }
        doc["pairs"]       = nlohmann::json::array({sp::pair_json(both)});
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
            expect(resolved.drafters.contains("dflash") && resolved.drafters.contains("mtp"),
                   "every measured drafter is attached, not only the chosen one's");
        }
        std::filesystem::remove(path);
    }
    // A round's cost in a profile (what the plan reports as "round_cost"): per section a round's
    // seconds over plain decode's seconds a token, the mean over the sections.
    {
        sp::Profile p;
        const auto m = [](const char* label, const char* family, std::int64_t tokens, double seconds,
                          std::int64_t rounds) {
            sp::Measurement out;
            out.config  = sp::SpecConfig::parse(label);
            out.family  = family;
            out.tokens  = tokens;
            out.seconds = seconds;
            out.rounds  = rounds;
            return out;
        };
        p.measurements = {m("plain", "math", 100, 10.0, 0), m("plain", "code", 100, 5.0, 0),
                          m("mtp:3", "math", 300, 4.0, 20),   // 0.2 s a round over 0.1: 2
                          m("mtp:3", "code", 300, 3.0, 20),   // 0.15 over 0.05: 3
                          m("mtp:3", "prose", 300, 3.0, 20)}; // no plain decode of prose: left out
        const auto cost = sp::profile_round_costs(p);
        expect(cost.size() == 1 && cost.contains("mtp:3") && std::abs(cost.at("mtp:3") - 2.5) < 1e-9,
               "a round's cost is the mean over the sections plain decode measured too");
        expect(sp::profile_round_costs(sp::Profile{}).empty(), "no measurements: no round cost");
    }
    std::cout << (failures == 0 ? "spec pairs: ok\n" : "spec pairs: FAILED\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
