// The configuration choice weighs every drafter a target carries (spec::ResolvedSpeculation::
// other_kinds), on the shipped registry's lab pairs: the Qwen3.8-27B GGUF's sibling pair
// (UD-Q4_K_S, an RTX 5070 Ti) was measured with its MTP layers alone, the DFlash2 drafter the
// target carries resolves by itself (its published profile), the drafter axis offers both, and the
// chosen one serves from its own resolution. CPU.

#include "product/plan/plan_choice.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_speculation.h"
#include "product/speculation/speculation.h"
#include "spec/profile.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
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
    check(held["joint_verify_scale"] == 0.45 && held["joint_priced_at_streams"] == 2 &&
              held["note"].get<std::string>().find("held from the price at 2 streams") !=
                  std::string::npos,
          "priced only at fewer streams: the note says it is held from that count");
    // Priced at 2 (0.5) and at 4 (0.9): the gain at 4 is held to 2's lower price, and the note
    // names 2, not the priced count 4.
    const auto least = note_at(4, {{.streams = 2, .scale = 0.5}, {.streams = 4, .scale = 0.9}});
    check(least["joint_priced_at_streams"] == 2 &&
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
              held_scale["joint_priced_at_streams"] == 2,
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
        check(held["joint_priced_at_streams"] == 2 && held["joint_verify_scale"] == 0.8 &&
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

} // namespace

int main() {
    test_every_kind_carried();
    test_gain_note_names_where_the_price_is_held_from();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "drafter kinds: ok\n";
    return 0;
}
