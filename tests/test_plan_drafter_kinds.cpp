// The configuration choice weighs every drafter a target carries (spec::ResolvedSpeculation::
// other_kinds), on the shipped registry's lab pairs: the Qwen3.8-27B GGUF's sibling pair
// (UD-Q4_K_S, an RTX 5070 Ti) was measured with its MTP layers alone, the DFlash2 drafter the
// target carries resolves by itself (its published profile), the drafter axis offers both, and the
// chosen one serves from its own resolution. CPU.

#include "product/plan/plan_choice.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_speculation.h"
#include "product/speculation/speculation.h"
#include "spec/pairs.h"
#include "spec/profile.h"
#include "spec/registry.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pr   = ninferno::product;
namespace spec = ninferno::spec;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) { return; }
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

bool drafts_with(const spec::Profile& profile, spec::Mode mode) {
    return spec::drafting_modes(profile).contains(mode);
}

// The drafter axis's MTP option; nullopt when it offers none.
std::optional<pr::choice::DrafterOption> mtp_option_of(const pr::DrafterAxis& axis) {
    for (const auto& o : axis.options) {
        if (o.label.rfind("mtp:", 0) == 0) { return o; }
    }
    return std::nullopt;
}

void test_every_kind_carried() {
    pr::PlanSourceFacts facts;
    facts.model.model_type        = "qwen3_5_text";
    facts.model.architecture      = "qwen35";
    facts.model.hidden            = 5120;
    facts.shape.hidden_size       = 5120;
    facts.shape.num_hidden_layers = 64;
    facts.shape.vocab_size        = 248320;
    facts.shape.builtin_mtp       = true;
    const spec::GpuKey gpu = pr::gpu_key_of(
        pr::plan_device_of_stats("NVIDIA GeForce RTX 4090", 24ULL << 30, 8, 9, false, 0x2684, 128));
    const pr::SpeculationSubject subject = pr::planned_subject(
        pr::PlannedTarget{.repo          = "unsloth/Qwen3.8-27B-GGUF",
                          .facts         = facts,
                          .gguf          = true,
                          .with_drafters = true,
                          .quant         = "UD-Q4_K_M",
                          .commit        = "4ca720788d1e01f1bff70c033e0d0028fd02e502"},
        gpu);
    check(subject.resolved && subject.resolved->source.find("registry sibling") == 0 &&
              drafts_with(subject.resolved->profile, spec::Mode::Mtp) &&
              !drafts_with(subject.resolved->profile, spec::Mode::DFlash2),
          "the sibling's pair, measured with the MTP layers alone (" +
              (subject.resolved ? subject.resolved->source : std::string("unresolved")) + ")");
    const bool dflash2 =
        subject.resolved &&
        std::any_of(subject.resolved->other_kinds.begin(), subject.resolved->other_kinds.end(),
                    [](const auto& other) { return drafts_with(other.profile, spec::Mode::DFlash2); });
    check(dflash2, "the DFlash2 drafter it carries, resolved alone");

    const pr::DrafterAxis axis = pr::drafter_axis(subject, pr::SpeculationAsk{});
    std::string offered;
    for (const auto& o : axis.options) { offered += " " + o.label; }
    const auto offers = [&](const char* label) { return offered.find(label) != std::string::npos; };
    check(offers(" mtp:") && offers(" dflash2:"),
          "the drafter axis offers MTP and DFlash2 (offers:" + offered + ")");

    // The joint verify's scales follow the load table's depth in force (#303): bounds "2:1,3:k"
    // run depth 1 at two streams and the configuration's depth k at three (the last bound at or
    // below, not the least), each priced at its own configuration; streams below the first bound
    // run k, and a scale the profile lacks leaves its streams unpriced.
    {
        pr::SpeculationSubject bounded = subject;
        spec::Profile& profile         = bounded.resolved->profile;
        const spec::SpecConfig config  = spec::pick(profile, true).config;
        check(config.enabled() && config.k >= 2, "the sibling pair drafts MTP at depth 2 or more");
        spec::SpecConfig shallow = config;
        shallow.k                = 1;
        profile.load             = {spec::LoadBound{.lanes = 2, .depth = 1},
                                    spec::LoadBound{.lanes = 3, .depth = config.k}};
        // A table given as measured: one derived from the facts would be derived again for the
        // mode (subject_of_mode), from the scales below.
        profile.load_basis = "measured";
        profile.joint_scale.clear();
        profile.joint_scale[shallow.label()] = {{2, 0.5}, {3, 0.7}};
        profile.joint_scale[config.label()]  = {{2, 0.9}, {3, 0.25}, {4, 0.2}};
        const pr::DrafterAxis with_bounds    = pr::drafter_axis(bounded, pr::SpeculationAsk{});
        const auto mtp_option =
            std::find_if(with_bounds.options.begin(), with_bounds.options.end(),
                         [](const auto& o) { return o.label.rfind("mtp:", 0) == 0; });
        check(mtp_option != with_bounds.options.end(), "the bounded profile offers MTP");
        if (mtp_option != with_bounds.options.end()) {
            const auto& joint = mtp_option->joint;
            check(
                joint.size() == 3 && joint[0].streams == 2 && joint[0].scale == 0.5 &&
                    joint[1].streams == 3 && joint[1].scale == 0.25 && joint[2].streams == 4 &&
                    joint[2].scale == 0.2,
                "the scale of each streams count is its depth in force's: 1 at two, k from three");
        }
    }
    // A default the Engine cannot run: here it runs MTP one draft shallower than the pair's
    // choice, so the profile's served choice replaces it (Profile::served). The option is what
    // runs: its gain, its load bound's depth and gain and its joint price are the served
    // configuration's, never the rejected default's; a served choice published without its
    // speeds is not measured (and not priced), whatever the rejected default measured.
    {
        pr::SpeculationSubject restricted     = subject;
        spec::Profile& profile                = restricted.resolved->profile;
        const spec::SpecConfig rejected       = spec::pick(profile, true).config;
        restricted.support.mtp_depth          = rejected.k - 1;
        const pr::SpeculationSubject mtp_only = pr::subject_of_mode(restricted, spec::Mode::Mtp);
        const spec::Choice served =
            spec::pick(pr::served_choices(mtp_only.resolved->profile), true);
        check(served.config.mode == spec::Mode::Mtp && served.config.k >= 1 &&
                  served.config.k < rejected.k && served.tok_s > 0 && served.plain_tok_s > 0,
              "the Engine's MTP window below the pair's choice serves a measured shallower one (" +
                  served.config.label() + " for " + rejected.label() + ")");
        profile.load       = {spec::LoadBound{.lanes = 2, .depth = rejected.k}};
        profile.load_basis = "measured"; // given, not derived again for the mode
        profile.joint_scale.clear();
        profile.joint_scale[rejected.label()]      = {{2, 0.3}};
        profile.joint_scale[served.config.label()] = {{2, 0.6}};
        const auto option = mtp_option_of(pr::drafter_axis(restricted, pr::SpeculationAsk{}));
        check(option && option->label == served.config.label(),
              "the option names the served configuration");
        if (option) {
            check(option->measured && option->speedup == std::max(1.0, served.speedup()),
                  "the option's gain is the served configuration's (" +
                      std::to_string(option->speedup) + " for " + std::to_string(served.speedup()) +
                      ")");
            check(option->load.size() == 1 && option->load[0].streams == 2 &&
                      option->load[0].depth == static_cast<std::uint32_t>(served.config.k) &&
                      !option->load[0].bounded && option->load[0].speedup == option->speedup,
                  "a bound at the rejected default's depth runs the served depth, at its gain");
            check(option->joint.size() == 1 && option->joint[0].streams == 2 &&
                      option->joint[0].scale == 0.6,
                  "the joint price is the served configuration's, not the rejected default's");
        }
        // Published without measurements: the served choice carries no speeds.
        spec::Choice unmeasured = served;
        unmeasured.tok_s = unmeasured.plain_tok_s = 0;
        profile.measurements.clear();
        profile.served       = spec::ServedChoices{unmeasured, unmeasured, unmeasured};
        const auto published = mtp_option_of(pr::drafter_axis(restricted, pr::SpeculationAsk{}));
        check(published && published->label == served.config.label() && !published->measured &&
                  published->speedup == 1.0 && published->joint.empty(),
              "a served choice without speeds: not measured, not priced, no borrowed gain");
    }
    const pr::SpeculationSubject chosen = pr::subject_of_mode(subject, spec::Mode::DFlash2);
    check(chosen.resolved && drafts_with(chosen.resolved->profile, spec::Mode::DFlash2) &&
              chosen.resolved->source.find("sibling") == std::string::npos,
          "DFlash2 chosen: its own resolution serves");
    const pr::SpeculationSubject mtp = pr::subject_of_mode(subject, spec::Mode::Mtp);
    check(mtp.resolved && mtp.resolved->source == subject.resolved->source,
          "MTP chosen: the sibling's pair serves");
}

// The plan JSON's drafter gain note says where the joint verify's price comes from (#303): priced
// at exactly the task's streams, the round cost's; at more streams than the last priced count, held
// from the price there (the gain does not climb with load), with that count named.
void test_gain_note_names_where_the_price_is_held_from() {
    const auto note_at =
        [](std::uint32_t streams,
           std::vector<pr::choice::DrafterOption::JointScale> joint = {{.streams = 2,
                                                                        .scale   = 0.45}},
           std::vector<pr::choice::DrafterOption::LoadGain> load = {}, bool measured = true) {
            pr::ServedChoice served;
            pr::choice::DrafterOption option;
            option.label    = "mtp:3";
            option.speedup  = 2.0;
            option.measured = measured;
            option.joint    = std::move(joint);
            option.load     = std::move(load);
            served.drafter_options.push_back(option);
            served.task.concurrency = streams;
            pr::choice::Scored scored;
            scored.candidate.drafter = 1;
            scored.drafter           = "mtp:3";
            served.choice.scored.push_back(scored);
            served.choice.chosen = 0;
            return pr::choice_json(served)["chosen"]["drafter_gain"];
        };
    const auto priced = note_at(2);
    check(priced["joint_verify_scale"] == 0.45 && !priced.contains("joint_priced_at_streams") &&
              priced["note"].get<std::string>().find("priced for the streams' rows") !=
                  std::string::npos,
          "priced at exactly the streams: the priced note");
    const auto held = note_at(3);
    check(held["joint_verify_scale"] == 0.45 && held.value("joint_priced_at_streams", 0) == 2 &&
              held["note"].get<std::string>().find("held from the price at 2 streams") !=
                  std::string::npos,
          "priced only at fewer streams: the note says it is held from that count");
    // Priced at 2 (0.5) and at 4 (0.9): the gain at 4 is held to 2's lower price, and the note
    // names 2, not the priced count 4.
    const auto least = note_at(4, {{.streams = 2, .scale = 0.5}, {.streams = 4, .scale = 0.9}});
    check(least.value("joint_priced_at_streams", 0) == 2 &&
              least["note"].get<std::string>().find("held from the price at 2 streams") !=
                  std::string::npos,
          "a count priced above a smaller count's price: the note names the smaller count");
    // An unmeasured bound at 2 streams holds a placeholder: the gain at 4 is 4's own price, and
    // the note says so (speedup_at skips it alike).
    const auto unmeasured =
        note_at(4, {{.streams = 2, .scale = 0.9}, {.streams = 4, .scale = 0.8}},
                {{.streams = 2, .speedup = 1.0, .depth = 1, .bounded = true, .measured = false},
                 {.streams = 4, .speedup = 1.6, .depth = 3, .bounded = true}});
    check(!unmeasured.contains("joint_priced_at_streams") &&
              unmeasured["note"].get<std::string>().find("priced for the streams' rows") !=
                  std::string::npos,
          "an unmeasured smaller bound: the note names this count's own price");
    // Only an unmeasured bound's placeholder below the streams, none at them: the gain is the
    // ceiling, and no joint scale is claimed.
    const auto only_placeholder =
        note_at(3, {{.streams = 2, .scale = 0.9}},
                {{.streams = 2, .speedup = 1.0, .depth = 1, .bounded = true, .measured = false}});
    check(!only_placeholder.contains("joint_verify_scale") &&
              !only_placeholder.contains("joint_priced_at_streams") &&
              only_placeholder["note"].get<std::string>().find("a ceiling") != std::string::npos,
          "only a placeholder below the streams: a ceiling, no joint_verify_scale");
    // A price at the placeholder bound's own count: the estimate counts none there (the drafter
    // is not counted), and the JSON claims none, at 2 and at 3 streams (the 2 bound holds to 3).
    const std::vector<pr::choice::DrafterOption::LoadGain> placeholder = {
        {.streams = 2, .speedup = 1.0, .depth = 2, .bounded = true, .measured = false}};
    for (const std::uint32_t streams : {2U, 3U}) {
        const auto at = note_at(
            streams, {{.streams = 2, .scale = 0.9}, {.streams = 3, .scale = 0.8}}, placeholder);
        check(at["at_concurrency"] == 1.0 && !at.contains("joint_verify_scale") &&
                  !at.contains("joint_priced_at_streams") &&
                  at["note"].get<std::string>().find("a ceiling") != std::string::npos,
              "a price at an unmeasured bound's own count: no price claimed at " +
                  std::to_string(streams) + " streams");
    }
    // An option nobody measured is not priced: the gain stays the ceiling at its own counts.
    for (const std::uint32_t streams : {2U, 3U}) {
        const auto at = note_at(streams, {{.streams = 2, .scale = 0.45}}, {}, false);
        check(at["at_concurrency"] == 2.0 && !at.contains("joint_verify_scale") &&
                  at["note"].get<std::string>().find("a ceiling") != std::string::npos,
              "an unmeasured option: no price claimed at " + std::to_string(streams) + " streams");
    }
    // The scale reported is the supplying count's own: bounds 2 (gain 1.5, depth 2) and 4 (gain
    // 1.2, depth 1), priced at 2 only: at 4 streams the gain is held from 2's 0.5 (0.75), not the
    // 0.625 that gain over the bound's own one-request gain would give.
    const auto held_scale = note_at(4, {{.streams = 2, .scale = 0.5}},
                                    {{.streams = 2, .speedup = 1.5, .depth = 2, .bounded = true},
                                     {.streams = 4, .speedup = 1.2, .depth = 1, .bounded = true}});
    check(held_scale["at_concurrency"] == 0.75 && held_scale["joint_verify_scale"] == 0.5 &&
              held_scale.value("joint_priced_at_streams", 0) == 2,
          "the scale of the count the gain is held from");
    // A scale above 1 is clamped: it is no price, and the JSON does not say it is one.
    const auto over = note_at(2, {{.streams = 2, .scale = 1.05}});
    check(over["at_concurrency"] == 2.0 && !over.contains("joint_verify_scale") &&
              over["note"].get<std::string>().find("a ceiling") != std::string::npos,
          "a scale above 1 is not reported as a price");
    // Priced at 2 (0.8) and at 3 (0.9): the gain at 3 is held from 2, and the note says this
    // count is priced, no lower, not that it is not priced. A tie (0.8 at both) is held from 2
    // alike: "no lower" is true of both, "higher" would be false of the tie.
    for (const double at_three : {0.9, 0.8}) {
        const auto held =
            note_at(3, {{.streams = 2, .scale = 0.8}, {.streams = 3, .scale = at_three}});
        check(held.value("joint_priced_at_streams", 0) == 2 && held["joint_verify_scale"] == 0.8 &&
                  held["note"].get<std::string>().find("held from the price at 2 streams") !=
                      std::string::npos &&
                  held["note"].get<std::string>().find("priced no lower") != std::string::npos &&
                  held["note"].get<std::string>().find("not priced") == std::string::npos,
              at_three > 0.8
                  ? "a count priced above the held price: the note says priced no lower"
                  : "a count priced equal to the held price: held from the smaller count, "
                    "priced no lower");
    }
    const auto unpriced_here = note_at(3);
    check(unpriced_here["note"].get<std::string>().find("this count is not priced") !=
              std::string::npos,
          "a count with no price of its own: the note says not priced");
}

// The 27B GGUF's facts, as test_every_kind_carried plans it.
pr::PlanSourceFacts qwen27b_facts() {
    pr::PlanSourceFacts facts;
    facts.model.model_type        = "qwen3_5_text";
    facts.model.architecture      = "qwen35";
    facts.model.hidden            = 5120;
    facts.shape.hidden_size       = 5120;
    facts.shape.num_hidden_layers = 64;
    facts.shape.vocab_size        = 248320;
    facts.shape.builtin_mtp       = true;
    return facts;
}

const std::string kQwen27bCommit = "4ca720788d1e01f1bff70c033e0d0028fd02e502";

// The 27B GGUF's subject on a named board: a quantization of the repository revision the lab's
// UD-Q4_K_S pair was measured at (its MTP layers on an RTX 5070 Ti), with its DFlash2 drafter.
pr::SpeculationSubject qwen27b_on(const char* board) {
    std::string error;
    const auto device = pr::plan_device_of_board(std::string(board) + ",linux,headless", &error);
    check(device.has_value(), std::string(board) + ": " + error);
    return pr::planned_subject(pr::PlannedTarget{.repo          = "unsloth/Qwen3.8-27B-GGUF",
                                                 .facts         = qwen27b_facts(),
                                                 .gguf          = true,
                                                 .with_drafters = true,
                                                 .quant         = "UD-Q4_K_M",
                                                 .commit        = kQwen27bCommit},
                               device ? pr::gpu_key_of(*device) : spec::GpuKey{});
}

// The head's own resolution (#777): a server that drafts with DFlash2 (subject_of_mode) keeps the
// kinds it does not draft with, so the model's MTP head is priced from this model's own lab
// measurement of it (head_priced_profile); the lab's pairs answer by the target's own weights or a
// sibling quantization of its revision, and only a group of other models (a fine-tune's acceptance
// is not its base's) where neither is measured, which a published profile's other kinds do not
// take (spec::resolve_other_kinds, same model only).
void test_head_resolution() {
    const auto measures_mtp = [](const spec::Profile& p) {
        return std::any_of(
            p.measurements.begin(), p.measurements.end(),
            [](const spec::Measurement& m) { return m.config.mode == spec::Mode::Mtp; });
    };
    // On the RTX 4090 (Ada, whose round-cost row the facts have) the sibling's rounds are priced on
    // the card; on the RTX 3090 (Ampere, no round-cost row) its measurements are transferred
    // unpriced: the pool rule's unpriced branch decides there.
    for (const auto& [board, priced_here] :
         std::vector<std::pair<const char*, bool>>{{"4090", true}, {"3090", false}}) {
        const pr::SpeculationSubject subject = qwen27b_on(board);
        const pr::SpeculationSubject dflash2 = pr::subject_of_mode(subject, spec::Mode::DFlash2);
        check(dflash2.resolved && drafts_with(dflash2.resolved->profile, spec::Mode::DFlash2),
              std::string(board) + ": the DFlash2 drafter's own resolution");
        const spec::Profile& priced = pr::head_priced_profile(dflash2);
        check(measures_mtp(priced) &&
                  (!priced.joint_scale.empty() && !priced.plain_step_scale.empty()) == priced_here,
              std::string(board) + ": a DFlash2 server takes the MTP head's measurements from the "
                                   "model's sibling pair, priced on the card's facts where it has "
                                   "them");
        check(priced_here ||
                  subject.resolved->source.find("no facts for ampere") != std::string::npos,
              std::string(board) + ": unpriced for want of Ampere's facts (" +
                  subject.resolved->source + ")");
        check(&pr::head_priced_profile(subject) == &subject.resolved->profile,
              std::string(board) + ": a subject that measured MTP prices the head from its own");

        // The head fallback's load table (fit_fallbacks' Head step, and the pool rule's head)
        // follows the profile that prices it (#777): on the RTX 4090 the facts derive it from
        // the model's own MTP measurement (2:3,3:0); without that measurement, and on the RTX 3090
        // whose rounds the facts do not price, it stays the built-in literal.
        const auto table = [](const pr::ServedSpeculation& head) {
            std::string out;
            for (const auto& bound : head.engine.load) {
                out += (out.empty() ? "" : ",") + std::to_string(bound.lanes) + ":" +
                       std::to_string(bound.draft_tokens);
            }
            return out + "/" + head.load_basis;
        };
        pr::SpeculationSubject bare = dflash2;
        bare.resolved->other_kinds.clear();
        const auto head      = pr::served_head_fallback(dflash2, false);
        const auto bare_head = pr::served_head_fallback(bare, false);
        check(
            head && bare_head && bare_head->load_basis == "literal" &&
                (priced_here ? table(*head) == "2:3,3:0/facts" : table(*head) == table(*bare_head)),
            std::string(board) + ": the head fallback's table from the model's MTP measurement (" +
                (head ? table(*head) : std::string("none")) + " against " +
                (bare_head ? table(*bare_head) : std::string("none")) + ")");

        // A kind resolved only to price the head beside a published profile (pricing_only, as
        // spec::resolve_other_kinds marks the same-model pair it finds there) offers no drafter
        // and is no mode a server switches to; it prices the head as before.
        pr::SpeculationSubject priced_only = dflash2;
        for (auto& other : priced_only.resolved->other_kinds) { other.pricing_only = true; }
        const auto labels = [](const pr::DrafterAxis& axis) {
            std::string out;
            for (const auto& o : axis.options) {
                out += " " + o.label + "@" + std::to_string(o.speedup);
            }
            return out;
        };
        const std::string with_kinds = labels(pr::drafter_axis(priced_only, pr::SpeculationAsk{}));
        const std::string without    = labels(pr::drafter_axis(bare, pr::SpeculationAsk{}));
        check(with_kinds == without, std::string(board) +
                                         ": a pricing-only kind is no drafter option (" +
                                         with_kinds + " against" + without + ")");
        const auto switched      = pr::subject_of_mode(priced_only, spec::Mode::Mtp);
        const auto bare_switched = pr::subject_of_mode(bare, spec::Mode::Mtp);
        check(switched.resolved && bare_switched.resolved &&
                  switched.resolved->source == bare_switched.resolved->source,
              std::string(board) + ": no server switches to a pricing-only kind");
        check(measures_mtp(pr::head_priced_profile(priced_only)),
              std::string(board) + ": a pricing-only kind prices the head");
    }

    const auto& pairs = spec::Registry::builtin().pairs();
    const auto bits   = spec::label_bits("UD-Q4_K_M");
    const auto at     = [&](const std::string& repo, const std::vector<std::string>& kinds) {
        return spec::pair_choice(pairs, "", "qwen3_5", kinds, bits, "sm86", {}, {}, repo);
    };
    const auto base = at("unsloth/Qwen3.8-27B-GGUF@" + kQwen27bCommit, {"mtp"});
    check(base && base->step == spec::PairStep::Sibling &&
              base->source.find("unsloth/Qwen3.8-27B-GGUF") != std::string::npos,
          "the base model's sibling pair answers for its MTP (" +
              (base ? base->source : std::string("none")) + ")");
    // Another revision of the repository: no sibling (another commit may hold other weights); a
    // group of other models at most, which the head's own resolution does not take.
    const auto other_commit =
        at("unsloth/Qwen3.8-27B-GGUF@0000000000000000000000000000000000000000", {"mtp"});
    check(!other_commit || other_commit->step == spec::PairStep::Group,
          "another revision: no sibling, a group at most");
    // A fine-tune's repository is not the base model's: never its sibling.
    const auto fine_tune =
        at("mradermacher/Qwen3.8-27B-Uncensored-GGUF@" + kQwen27bCommit, {"mtp"});
    check(!fine_tune || fine_tune->step == spec::PairStep::Group,
          "a fine-tune's pair is no sibling of the base model");
    // A kind no pair drafts with: nothing.
    check(!at("unsloth/Qwen3.8-27B-GGUF@" + kQwen27bCommit, {"eagle"}), "no pair: nothing");
}

// The lab's measured pool runs against the rule (#777): Qwen3.8 27B with its drafters at
// --max-context 32768, the server drafting with DFlash2 (the measured runs' dflash2:7), a pool of n
// whole contexts taken at its least (n x 32768 tokens). Each prints the rule's speed of the head's
// pool over plain decode's in each part of the mix (the chats' part is the benchmark's short chats
// at C, which the runs measured) and at the mix, beside the measured speed a request.
//   RTX 3090, C = 8: plain decode 38.3 tok/s a request in 5 contexts, MTP 3 51.1 in 4.
//   RTX 4090, C = 4: dflash2 K=7 left 3 of 4 contexts; plain 48.9, MTP 3 88.8, the head leaving
//   every context plain decode does; C = 8: plain 46.2, MTP 3 69.4.
// The drafter's own count of contexts was not recorded: every count that enters the rule. The
// 3090's rounds are not priced (Ampere has no round-cost row: "no facts for ampere"), so the rule
// before #777 decides it (four fifths: 4 of 5, the head); the 4090's are, and its price decides.
void test_pool_rule_lab_runs() {
    const auto run = [](const char* board, std::uint32_t streams, std::uint32_t plain,
                        std::uint32_t head, std::uint32_t drafter, double measured,
                        bool priced_here) {
        const pr::SpeculationSubject subject =
            pr::subject_of_mode(qwen27b_on(board), spec::Mode::DFlash2);
        const std::uint32_t context = 32768;
        const auto pool             = [&](std::uint32_t contexts) {
            return pr::PoolSize{.contexts = contexts,
                                            .tokens   = std::uint64_t{contexts} * context,
                                            .context  = context};
        };
        const pr::PoolDecision decision = pr::pool_rule(
            subject, false, drafter, streams, [&] { return pool(plain); },
            [&](const pr::ServedSpeculation&) { return std::optional<pr::PoolSize>(pool(head)); });
        const auto served           = pr::served_head_fallback(subject, false);
        const spec::Profile& priced = pr::head_priced_profile(subject);
        std::string parts;
        if (served) {
            for (const pr::MixPart& part : pr::mix_parts(streams, pool(head), pool(plain))) {
                const auto step = [&](std::uint32_t c) { return pr::plain_step_price(priced, c); };
                const auto with_head = pr::mix_steps(
                    part.head, part.requests,
                    [&](std::uint32_t c) { return pr::pool_head_gain(priced, *served, c); }, step);
                const auto plainly = pr::mix_steps(
                    part.plain, part.requests, [](std::uint32_t) { return 1.0; }, step);
                char text[96];
                std::snprintf(text, sizeof text, " %u requests on %u vs %u streams x%.3f;",
                              part.requests, part.head, part.plain,
                              with_head && plainly ? *plainly / *with_head : -1.0);
                parts += text;
            }
        }
        char line[320];
        std::snprintf(line, sizeof line,
                      "%s C %u, head %u / plain %u contexts (drafter %u): %s, mix x%.3f;%s "
                      "measured x%.3f a request",
                      board, streams, head, plain, drafter,
                      decision.keep == pr::PoolKeep::Head ? "the head" : "plain decode",
                      decision.head_price ? *decision.head_price : -1.0, parts.c_str(), measured);
        std::cout << line << '\n';
        check(decision.keep == pr::PoolKeep::Head && decision.head_price.has_value() == priced_here,
              std::string(line) + ": measured faster, the rule keeps the head" +
                  (priced_here ? " by its price" : ", unpriced, by four fifths"));
    };
    for (std::uint32_t drafter = 1; drafter < 5; ++drafter) {
        run("3090", 8, 5, 4, drafter, 51.1 / 38.3, false);
    }
    run("4090", 4, 4, 4, 3, 88.8 / 48.9, true);
    for (std::uint32_t contexts = 2; contexts <= 8; ++contexts) {
        run("4090", 8, contexts, contexts, contexts - 1, 69.4 / 46.2, true);
    }
}

} // namespace

int main() {
    test_every_kind_carried();
    test_gain_note_names_where_the_price_is_held_from();
    test_head_resolution();
    test_pool_rule_lab_runs();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "drafter kinds: ok\n";
    return 0;
}
