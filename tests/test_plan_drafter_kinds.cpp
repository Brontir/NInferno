// The configuration choice weighs every drafter a target carries (spec::ResolvedSpeculation::
// other_kinds), on the shipped registry's lab pairs: the Qwen3.8-27B GGUF's sibling pair
// (UD-Q4_K_S, an RTX 5070 Ti) was measured with its MTP layers alone, the DFlash2 drafter the
// target carries resolves by itself (its published profile), the drafter axis offers both, and the
// chosen one serves from its own resolution. CPU.

#include "product/plan/plan_choice.h"
#include "product/plan/plan_device.h"
#include "product/plan/plan_speculation.h"
#include "product/speculation/speculation.h"
#include "spec/profile.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

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

    const pr::SpeculationSubject chosen = pr::subject_of_mode(subject, spec::Mode::DFlash2);
    check(chosen.resolved && drafts_with(chosen.resolved->profile, spec::Mode::DFlash2) &&
              chosen.resolved->source.find("sibling") == std::string::npos,
          "DFlash2 chosen: its own resolution serves");
    const pr::SpeculationSubject mtp = pr::subject_of_mode(subject, spec::Mode::Mtp);
    check(mtp.resolved && mtp.resolved->source == subject.resolved->source,
          "MTP chosen: the sibling's pair serves");
}

} // namespace

int main() {
    test_every_kind_carried();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "drafter kinds: ok\n";
    return 0;
}
