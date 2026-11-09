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
    const auto note_at = [](std::uint32_t streams,
                            std::vector<pr::choice::DrafterOption::JointScale> joint =
                                {{.streams = 2, .scale = 0.45}}) {
        pr::ServedChoice served;
        pr::choice::DrafterOption option;
        option.label   = "mtp:3";
        option.speedup = 2.0;
        option.joint   = std::move(joint);
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
