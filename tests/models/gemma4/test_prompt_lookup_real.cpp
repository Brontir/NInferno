// Prompt lookup (model-free speculation) and the acceptance watchdog on the real Gemma 4 26B-A4B
// target. NINFERNO_TEST_GEMMA4_DIR names the directory that holds the artifacts; without it the
// test skips.
//
//   * prompt lookup verifies host-staged drafts losslessly: every emitted token is the one plain
//     greedy decode picks on the same prefix, up to a near-tie (less than 0.5 raw logits);
//   * it drafts on a prompt whose answer repeats it (a function to complete), meets a budget
//     exactly and is deterministic across sequences in one session;
//   * the watchdog: a floor no drafter can meet trips it after its minimum rounds, the rest of
//     the sequence is plain decode (still plain decode's tokens), the event is logged once, and
//     the next sequence speculates again; a floor a working drafter clears never trips;
//   * with prompt lookup, steps without a match count as one-token rounds: on prose, where
//     lookup finds nearly nothing, the default floor turns it off (plain decode's tokens).

#include "core/device.h"
#include "models/gemma4/model.h"
#include "models/gemma4/session.h"
#include "spec/corpus.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::models::gemma4;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

const std::vector<std::int32_t>& prompt(const std::string& id) {
    const auto* tokens = spec::builtin_corpus().tokens_for("v1:9a7777576d37749e");
    if (tokens == nullptr) { throw std::runtime_error("the corpus lacks Gemma 4 prompts"); }
    return tokens->prompts.at(id);
}

SessionOptions base() {
    SessionOptions o;
    o.max_context   = 4096;
    o.prefill_chunk = 512;
    return o;
}

// The worst deficit (raw logits) of an emitted stream against plain decode teacher-forced on
// the same prefix; `differed` counts the tokens plain decode would not have chosen.
double worst_deficit(const Model& model, DeviceContext& device,
                     const std::vector<std::int32_t>& prompt_ids,
                     const std::vector<std::int32_t>& stream, std::size_t& differed) {
    Session plain(model, device, base());
    plain.prefill(prompt_ids);
    const auto cap = model.config().softcap;
    const auto raw = [&](float v) { return cap ? *cap * std::atanh(v / *cap) : v; };
    double worst   = 0;
    differed       = 0;
    for (std::size_t i = 0; i + 1 < stream.size(); ++i) {
        plain.set_pending_token(stream[i]);
        (void)plain.decode(1);
        const auto logits = plain.last_logits();
        const auto it     = std::max_element(logits.begin(), logits.end());
        const auto choice = static_cast<std::int32_t>(it - logits.begin());
        if (choice != stream[i + 1]) {
            ++differed;
            worst = std::max(worst, double(raw(*it)) - raw(logits[stream[i + 1]]));
        }
    }
    return worst;
}

void check_prompt_lookup(const Model& model, DeviceContext& device) {
    auto o         = base();
    o.ngram        = true;
    o.draft_tokens = 8;
    Session s(model, device, o);
    const auto& code = prompt("code_0");
    s.reset();
    s.prefill(code);
    const auto before = s.speculation_stats();
    const auto first  = s.speculate_until(128);
    const auto after  = s.speculation_stats();
    check(first.size() == 128, "prompt lookup meets a 128-token budget exactly");
    check(after.rounds > before.rounds && after.accepted > before.accepted,
          "prompt lookup drafts and gets drafts accepted on a completion that repeats its prompt");
    check(after.rounds - before.rounds + s.plain_steps() < 128,
          "prompt lookup takes fewer steps than tokens");
    std::size_t differed = 0;
    const auto worst     = worst_deficit(model, device, code, first, differed);
    check(worst < 0.5, "prompt lookup emits plain decode's tokens up to a near-tie (worst " +
                           std::to_string(worst) + " over " + std::to_string(differed) + ")");
    s.reset();
    s.prefill(prompt("prose_0"));
    (void)s.speculate_until(40);
    s.reset();
    s.prefill(code);
    check(s.speculate_until(128) == first, "prompt lookup: A, B, A emits A twice alike");
    s.reset();
    s.prefill(code);
    check(s.speculate_until(37).size() == 37, "prompt lookup meets a 37-token budget exactly");
    std::cout << "prompt lookup: " << (after.rounds - before.rounds) << " rounds, "
              << (after.accepted - before.accepted) << " drafts accepted, " << s.plain_steps()
              << " plain steps in the last call\n";
}

void check_watchdog(const Model& model, DeviceContext& device, const Drafter& mtp) {
    auto o             = base();
    o.drafter          = &mtp;
    o.draft_tokens     = 4;
    int events         = 0;
    o.log              = [&](const std::string& line) {
        ++events;
        std::cout << line << '\n';
    };
    o.acceptance_floor = 100.0; // no drafter emits 100 tokens per round
    Session s(model, device, o);
    const auto& math = prompt("math_0");
    s.reset();
    s.prefill(math);
    const auto before = s.speculation_stats();
    const auto stream = s.speculate_until(96);
    const auto after  = s.speculation_stats();
    check(stream.size() == 96, "a tripped sequence still meets its budget");
    check(s.speculation_latched() && events == 1, "the watchdog trips once and says so");
    check(after.rounds - before.rounds >= o.watchdog_min_rounds &&
              after.rounds - before.rounds < o.watchdog_min_rounds + 8,
          "it trips at its minimum rounds (" + std::to_string(after.rounds - before.rounds) + ")");
    check(s.plain_steps() > 0, "the rest of the sequence is plain decode");
    std::size_t differed = 0;
    const auto worst     = worst_deficit(model, device, math, stream, differed);
    check(worst < 0.5, "the tripped sequence emits plain decode's tokens up to a near-tie (worst " +
                           std::to_string(worst) + ")");
    s.reset();
    check(!s.speculation_latched() && s.plain_steps() == 0, "reset clears the verdict");

    auto ok             = o;
    ok.acceptance_floor = 1.05;
    events              = 0;
    Session fine(model, device, ok);
    fine.reset();
    fine.prefill(math);
    (void)fine.speculate_until(128);
    check(!fine.speculation_latched() && events == 0, "a working drafter clears a low floor");
}


void check_prompt_lookup_watchdog(const Model& model, DeviceContext& device) {
    auto o             = base();
    o.ngram            = true;
    o.draft_tokens     = 4;
    o.acceptance_floor = 1.2; // the fallback's default floor
    int events         = 0;
    o.log              = [&](const std::string& line) {
        ++events;
        std::cout << line << '\n';
    };
    Session s(model, device, o);
    const auto& prose = prompt("prose_1");
    s.reset();
    s.prefill(prose);
    const auto stream = s.speculate_until(160);
    check(stream.size() == 160, "a tripped prompt-lookup sequence still meets its budget");
    check(s.speculation_latched() && events == 1,
          "prompt lookup that finds nothing trips the watchdog once");
    std::size_t differed = 0;
    const auto worst     = worst_deficit(model, device, prose, stream, differed);
    check(worst < 0.5, "the tripped lookup sequence emits plain decode's tokens up to a near-tie "
                       "(worst " + std::to_string(worst) + ")");
    // Information only: whether the same floor keeps lookup on for a completion.
    events = 0;
    s.reset();
    s.prefill(prompt("code_1"));
    (void)s.speculate_until(160);
    std::cout << "prompt lookup on code_1 with floor 1.2: "
              << (s.speculation_latched() ? "turned off" : "kept on") << ", " << s.plain_steps()
              << " plain steps\n";
}

} // namespace

int main() {
    const char* dir = std::getenv("NINFERNO_TEST_GEMMA4_DIR");
    if (dir == nullptr || *dir == '\0') {
        std::cout << "skip: NINFERNO_TEST_GEMMA4_DIR is not set\n";
        return 77;
    }
    const std::filesystem::path artifacts(dir);
    try {
        DeviceContext device(0);
        const auto target = Model::load(artifacts / "gemma4_26b_a4b_nvfp4.ninferno", device,
                                        ActivationProfile::Artifact);
        const auto mtp    = Drafter::load(artifacts / "gemma4_26b_assistant_q8_sl98304r2.ninferno",
                                          device, target->config());
        const auto run    = [&](const std::string& name, const auto& body) {
            try {
                body();
            } catch (const std::exception& error) {
                check(false, name + " threw: " + error.what());
            }
        };
        run("prompt lookup", [&] { check_prompt_lookup(*target, device); });
        run("watchdog", [&] { check_watchdog(*target, device, *mtp); });
        run("prompt lookup watchdog", [&] { check_prompt_lookup_watchdog(*target, device); });
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "gemma4 real prompt lookup and watchdog: lossless, budgets exact, latched\n";
    return 0;
}
