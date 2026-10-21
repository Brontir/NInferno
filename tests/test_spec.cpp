// Speculation support without a GPU: configuration labels, vocabulary fingerprints, the acceptance
// watchdog, profile selection and publication, the shipped registry and corpus, and resolving
// what an artifact came with (spec/manifest.h) from files next to it.

#include "spec/config.h"
#include "spec/corpus.h"
#include "spec/engine_support.h"
#include "spec/fingerprint.h"
#include "spec/load_model.h"
#include "spec/load_tune.h"
#include "spec/manifest.h"
#include "spec/menu.h"
#include "spec/profile.h"
#include "spec/registry.h"
#include "spec/watchdog.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <regex>
#include <set>
#include <string>
#include <unistd.h>

namespace {

using namespace ninferno::spec;
using Json = nlohmann::json;

int check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAILED: " << what << '\n'; }
    return ok ? 0 : 1;
}

bool rejects(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

Measurement measure(const std::string& label, const std::string& family, double tok_s,
                    std::int64_t rounds = 0, std::int64_t accepted = 0, std::int64_t drafted = 0) {
    Measurement m;
    m.config   = SpecConfig::parse(label);
    m.family   = family;
    m.prompts  = 4;
    m.tokens   = 1000;
    m.seconds  = 1000.0 / tok_s;
    m.rounds   = rounds;
    m.accepted = accepted;
    m.drafted  = drafted;
    return m;
}

int test_config() {
    int failures = 0;
    for (const char* label : {"plain", "mtp:4", "mtp:8:2", "dflash:11", "dflash2:15+head",
                              "mtp:3+head", "ngram:8"}) {
        failures += check(SpecConfig::parse(label).label() == label,
                          std::string("label round trip: ") + label);
    }
    failures += check(SpecConfig::parse("mtp:8:2").adaptive(), "mtp:8:2 is adaptive");
    failures += check(!SpecConfig::parse("off").enabled(), "off is plain");
    for (const char* bad : {"mtp", "mtp:0", "mtp:4:4", "dflash:7:2", "eagle:3", "mtp:x", "mtp:17"}) {
        failures += check(rejects([&] { (void)SpecConfig::parse(bad); }),
                          std::string("rejects ") + bad);
    }
    return failures;
}

int test_fingerprint() {
    int failures = 0;
    const auto a = vocab_fingerprint_text(
        R"({"model":{"vocab":{"c":2,"a":0,"b":1}},"added_tokens":[{"id":3,"content":"<s>"},{"id":1,"content":"B"}]})");
    // The same value tools/spec/tokenize_corpus.py computes (sha256 of "0\ta\n1\tB\n2\tc\n3\t<s>\n").
    failures += check(a.value == "v1:6129e08858369376", "fingerprint value " + a.value);
    failures += check(a.tokens == 4 && a.max_id == 4, "fingerprint counts");
    const auto unigram = vocab_fingerprint_text(
        R"({"model":{"vocab":[["a",0.0],["b",-1.0],["c",-2.0]]},"added_tokens":[{"id":3,"content":"<s>"},{"id":1,"content":"B"}]})");
    failures += check(unigram.value == a.value, "a Unigram list fingerprints like the same map");
    const auto other = vocab_fingerprint_text(R"({"model":{"vocab":{"a":0,"b":1,"c":2}}})");
    failures += check(other.value != a.value, "an added token changes the fingerprint");
    failures += check(rejects([] { (void)vocab_fingerprint_text(R"({"x":1})"); }),
                      "a document without a vocabulary is rejected");
    return failures;
}

int test_watchdog() {
    int failures = 0;
    AcceptanceWatchdog off;
    failures += check(!off.observe(100, 0) && !off.tripped(), "a floor of 0 never trips");
    AcceptanceWatchdog w(WatchdogPolicy{2.0, 32, 16, 1.0});
    // 8 rounds at a time; 3 tokens per round (2 accepted drafts each).
    std::int64_t rounds = 0, accepted = 0;
    for (int i = 0; i < 8; ++i) {
        rounds += 8;
        accepted += 16;
        failures += check(!w.observe(rounds, accepted), "a good drafter does not trip");
    }
    failures += check(w.rolling_tokens_per_round() > 2.99 && w.rolling_tokens_per_round() < 3.01,
                      "rolling tokens per round");
    // Then acceptance collapses: nothing accepted. The window is 32 rounds, so it trips once
    // the last 32 rounds average below 2: after 3 slices of 8 (24 bad rounds + 8 good: 1.5).
    int slices = 0;
    bool tripped = false;
    while (!tripped && slices < 10) {
        rounds += 8;
        tripped = w.observe(rounds, accepted);
        ++slices;
    }
    failures += check(tripped && slices <= 3, "a collapse trips within the window (" +
                                                  std::to_string(slices) + " slices)");
    failures += check(w.tripped() && !w.observe(rounds + 8, accepted + 100),
                      "the verdict is latched");
    w.reset();
    failures += check(!w.tripped(), "reset clears the verdict");
    AcceptanceWatchdog early(WatchdogPolicy{2.0, 32, 16, 1.0});
    failures += check(!early.observe(8, 0), "no verdict before min_rounds");
    failures += check(early.observe(16, 0), "a verdict at min_rounds");
    // A verdict the owner held (a shared launch) is judged again on the next rounds, on the same
    // window: still below the floor it trips again, above it it holds.
    early.rearm();
    failures += check(!early.tripped() && early.observe(24, 0), "a rearmed verdict judges again");
    early.rearm();
    failures += check(!early.observe(56, 80), "a rearmed watchdog recovers above the floor");

    // A step down mid-request: a lower floor, judged only on the rounds after the restart, with
    // min_rounds counted from it. 40 rounds at 1.5 tokens per round tripped a floor of 2.
    AcceptanceWatchdog step(WatchdogPolicy{2.0, 32, 16, 1.0});
    failures += check(step.observe(40, 20), "1.5 tokens per round trips a floor of 2");
    step.restart(1.4, 40, 20);
    failures += check(!step.tripped() && step.policy().floor_tokens_per_round == 1.4,
                      "restart clears the verdict and takes the step's floor");
    failures += check(!step.observe(48, 20), "no verdict before min_rounds from the restart");
    // 16 rounds after the restart at 1.5 per round: above the new floor, though the request's
    // average over its whole history would not decide anything here.
    failures += check(!step.observe(56, 28), "the step's rounds pass its floor");
    failures += check(step.observe(72, 28), "and trip it when they fall below");

    // The default margin: a floor of 2.5 trips below 1.75 tokens per round, not at 2.4.
    AcceptanceWatchdog margin(WatchdogPolicy{2.5, 32, 16});
    failures += check(margin.policy().trip_tokens_per_round() == 1.75, "the default trip point");
    failures += check(!margin.observe(32, 45), "1.4 drafted tokens per round hold a floor of 2.5");
    failures += check(margin.observe(64, 65), "and 0.6 trip it");

    // The depth's rate counts every round since the restart, not the rolling window.
    AcceptanceWatchdog depth(WatchdogPolicy{2.0, 8, 4, 1.0});
    failures += check(depth.depth_tokens_per_round() == 0.0, "no rate before any round");
    failures += check(!depth.observe(16, 32), "3 tokens per round hold a floor of 2");
    failures += check(depth.observe(32, 32), "and a collapse trips the window of 8");
    failures += check(depth.rolling_tokens_per_round() < 1.01 &&
                          depth.depth_tokens_per_round() == 2.0,
                      "the window says 1, the depth's 32 rounds 2");
    depth.restart(1.5, 32, 32);
    failures += check(depth.depth_tokens_per_round() == 0.0, "a restart starts the rate afresh");
    failures += check(!depth.observe(48, 40) && depth.depth_tokens_per_round() == 1.5,
                      "the rate after a restart counts only its own rounds");

    // The steps a trip skips: those whose trip point (margin 0.5) is above the depth's rate.
    const std::vector<double> floors{2.6, 2.2, 1.8}; // trip points 1.8, 1.6, 1.4
    failures +=
        check(steps_that_cannot_pay(2.0, 0.5, floors) == 0, "a rate above them all skips none");
    failures += check(steps_that_cannot_pay(1.7, 0.5, floors) == 1, "1.7 skips the 1.8 step");
    failures += check(steps_that_cannot_pay(1.5, 0.5, floors) == 2, "1.5 skips two");
    failures += check(steps_that_cannot_pay(1.2, 0.5, floors) == 3, "1.2 skips all: plain");
    failures += check(steps_that_cannot_pay(1.2, 0.5, {}) == 0, "no steps, nothing to skip");
    return failures;
}

int test_select() {
    int failures = 0;
    Profile p;
    p.drafters.push_back({"assistant", "gemma4_assistant", "mtp", "", "", "", "", "", 0, 0});
    p.drafters.push_back({"dflash", "dflash", "dflash", "", "", "", "", "", 0, 0});
    p.drafters.push_back({"ngram", "ngram", "ngram", "", "", "", "", "", 0, 0});
    for (const char* f : {"math", "code", "prose", "thinking"}) {
        p.measurements.push_back(measure("plain", f, 100));
    }
    // math: dflash:7 best; code: dflash:11 best; prose: only mtp:1 beats plain; thinking: dflash:7.
    const auto add = [&](const char* label, double m, double c, double pr, double t) {
        p.measurements.push_back(measure(label, "math", m, 100, 300, 700));
        p.measurements.push_back(measure(label, "code", c, 100, 300, 700));
        p.measurements.push_back(measure(label, "prose", pr, 100, 50, 700));
        p.measurements.push_back(measure(label, "thinking", t, 100, 300, 700));
    };
    add("dflash:7", 250, 240, 90, 230);
    add("dflash:11", 240, 270, 70, 220);
    add("mtp:1", 150, 150, 120, 150);
    add("ngram:4", 101, 102, 100, 101);
    select(p);
    // Overall: seconds per token summed over the corpus sections. dflash:7 would sum lowest (1/250
    // + 1/240 + 1/90 + 1/230 = 0.0236), but it loses to plain decode on prose (90 < 100), as
    // dflash:11 does; a default runs every section, so mtp:1 (0.0283, plain 0.04) is chosen.
    failures += check(p.overall.config.label() == "mtp:1" && p.overall.drafter == "assistant",
                      "overall choice " + p.overall.config.label());
    failures += check(p.overall.floor_tokens_per_round > 1.0, "a floor for the overall choice");
    failures += check(p.non_thinking.config.label() == "mtp:1",
                      "non-thinking choice never slower than plain on a section");
    failures += check(p.thinking.config.label() == "dflash:7", "thinking choice");
    // Break-even: seconds per round over plain seconds per token. thinking dflash:7: 1000/230 s /
    // 100 rounds = 0.0435 s per round, plain 0.01 s per token -> 4.35 tokens per round.
    failures += check(std::abs(p.thinking.floor_tokens_per_round - 1000.0 / 230.0) < 1e-9,
                      "thinking floor");
    // The profile laid onto one engine (spec/menu.h): the thinking default's drafter (DFlash)
    // runs it; the MTP defaults fall back to the primary, noted.
    {
        const auto menu = build_menu(p, p.thinking, [](const Choice&) { return true; });
        failures += check(menu.mode == Mode::DFlash, "menu runs the primary's drafter");
        failures += check(menu.thinking.config.label() == "dflash:7" && menu.thinking.note.empty(),
                          "menu keeps the primary's own choice");
        failures +=
            check(menu.overall.config.label() == "dflash:7" &&
                      menu.overall.profiled.label() == "mtp:1" && !menu.overall.note.empty(),
                  "menu runs the primary for another drafter's choice, noted");
        failures += check(menu.depths() == std::vector<std::int32_t>{7}, "menu depths");
        failures += check(menu.pick(true).config.label() == "dflash:7" &&
                              menu.pick(false).config.label() == "dflash:7" &&
                              menu.pick(std::nullopt).config.label() == "dflash:7",
                          "menu pick follows spec::pick's order");
        const auto none = build_menu(p, p.thinking, [](const Choice& c) {
            return c.config.mode != Mode::DFlash;
        });
        failures += check(none.mode == Mode::None && !none.overall.config.enabled() &&
                              none.depths().empty(),
                          "an unusable primary gives a plain menu");
    }
    const auto status = [&](const std::string& name) {
        for (const auto& d : p.drafters) {
            if (d.name == name) { return d.status; }
        }
        return std::string();
    };
    failures += check(status("dflash") == "accepted" && status("assistant") == "accepted",
                      "useful drafters are accepted");
    failures += check(status("ngram") == "rejected", "a drafter within 3 % of plain is rejected");

    // A drafter that never beats plain is rejected and never chosen; no config: plain.
    Profile q;
    q.drafters.push_back({"dflash", "dflash", "dflash", "", "", "", "", "", 0, 0});
    for (const char* f : {"math", "code", "prose", "thinking"}) {
        q.measurements.push_back(measure("plain", f, 100));
        q.measurements.push_back(measure("dflash:7", f, 60, 100, 5, 700));
    }
    select(q);
    failures += check(q.drafters[0].status == "rejected" && !q.overall.config.enabled() &&
                          !q.thinking.config.enabled(),
                      "a wrong drafter is rejected and plain decode chosen");

    // Round trip through JSON, and the published form.
    const auto again = profile_from_json(to_json(p));
    failures +=
        check(again.overall.config == p.overall.config && again.overall.drafter == "assistant" &&
                  again.measurements.size() == p.measurements.size(),
              "profile JSON round trip");
    const auto published =
        published_from_profile(p, "gb202", "GPU", {"org/model"}, "test artifact");
    const auto from = profile_from_published(published);
    failures +=
        check(from.overall.config == p.overall.config && from.thinking.config.label() == "dflash:7",
              "published round trip");
    failures += check(pick(from, true).config == from.thinking.config &&
                          pick(from, false).config == from.non_thinking.config &&
                          pick(from, std::nullopt).config == from.overall.config,
                      "pick by the thinking flag");

    // The portable choices ("*" profile): the shallowest configuration within 2% of the best.
    Profile r;
    r.drafters.push_back({"assistant", "gemma4_assistant", "mtp", "", "", "", "", "", 0, 0});
    for (const char* f : {"math", "code", "prose", "thinking"}) {
        r.measurements.push_back(measure("plain", f, 100));
    }
    const auto add_r = [&](const char* label, double math, double code) {
        for (const char* f : {"math", "prose", "thinking"}) {
            r.measurements.push_back(measure(label, f, math, 100, 200, 300));
        }
        r.measurements.push_back(measure(label, "code", code, 100, 200, 300));
    };
    // math and the others: mtp:10 best at 203, mtp:3 within 2% (200); code: mtp:10 at 230, mtp:6
    // within 2% (226), mtp:3 not (200).
    add_r("mtp:3", 200, 200);
    add_r("mtp:6", 190, 226);
    add_r("mtp:10", 203, 230);
    select(r);
    failures += check(r.thinking.config.label() == "mtp:10" && r.overall.config.label() == "mtp:10",
                      "the best choices stay the fastest");
    failures += check(r.portable && r.portable->thinking.config.label() == "mtp:3" &&
                          r.portable->thinking.drafter == "assistant",
                      "portable choices: the shallowest within 2%");
    // Overall seconds per token: mtp:10 3/203 + 1/230 = 0.01913; mtp:6 3/190 + 1/226 = 0.02021
    // (5.6% more); mtp:3 3/200 + 1/200 = 0.02 (4.5% more): none within 2%.
    failures += check(r.portable->overall.config.label() == "mtp:10",
                      "a portable default with nothing shallower within 2% keeps the best");
    // The floor is the chosen configuration's own: mtp:3's 5 s per 100 rounds over plain's 0.01 s
    // per token (mtp:10's would be 4.93).
    failures += check(std::abs(r.portable->thinking.floor_tokens_per_round - 5.0) < 1e-9,
                      "a portable choice carries its own floor");
    const auto r_published = published_from_profile(r, "gb202", "GPU", {"org/model"}, "test");
    failures += check(r_published.contains("portable") &&
                          profile_from_json(to_json(r)).portable->thinking.config ==
                              r.portable->thinking.config,
                      "portable choices are published and read back");
    select(r, {.portable_within = 0.0});
    failures += check(!r.portable, "no portable choices when asked for none");

    // Mixed drafters: dflash2:11+head best at 300, dflash2:7+head at 296 and mtp:3 at 299 within
    // 2%. The portable choice keeps the best's drafter and head (dflash2:7+head), never mtp:3 (the
    // load table measured with the best would sit on another kind). An Engine that runs DFlash2 to
    // depth 7 without the head: the portable choices come from what it runs (mtp:3 then, the best
    // it runs, with nothing of its drafter shallower within 2%).
    Profile m;
    m.drafters.push_back({"assistant", "mtp", "mtp", "", "", "", "", "", 0, 0});
    m.drafters.push_back({"dflash2", "dflash2", "dflash2", "", "", "", "", "", 0, 0});
    for (const char* f : {"math", "code", "prose", "thinking"}) {
        m.measurements.push_back(measure("plain", f, 100));
        m.measurements.push_back(measure("mtp:3", f, 299, 100, 200, 300));
        m.measurements.push_back(measure("dflash2:7+head", f, 296, 100, 400, 700));
        m.measurements.push_back(measure("dflash2:11+head", f, 300, 100, 600, 1100));
    }
    select(m);
    failures += check(m.portable && m.overall.config.label() == "dflash2:11+head" &&
                          m.portable->overall.config.label() == "dflash2:7+head" &&
                          m.portable->thinking.config.label() == "dflash2:7+head",
                      "a portable choice keeps the best's drafter and proposal head");
    select(m, {.served = EngineSupport{.mtp_depth = 5, .dflash_depth = 7, .dflash2 = true}});
    failures += check(m.portable && m.portable->overall.config.label() == "mtp:3" &&
                          m.served && m.served->overall.config.label() == "mtp:3",
                      "portable choices come from what the serving Engine runs");

    // spec-reselect: a saved profile with served choices is selected again only with the serving
    // Engine named; without it the served and portable choices would take what it does not run.
    const EngineSupport runs_less{.mtp_depth = 5, .dflash_depth = 7, .dflash2 = true};
    Profile saved = m;
    failures += check(rejects([&] { (void)reselect(saved, {.gpu_class = "gb203"}); }),
                      "spec-reselect refuses served choices without the serving Engine");
    const Json again_published = reselect(saved, {.gpu_class   = "gb203",
                                                  .gpu_name    = "GPU",
                                                  .applies_to  = {"org/model"},
                                                  .measured_on = "test",
                                                  .served      = runs_less});
    failures += check(again_published.at("gpu_class") == "gb203" &&
                          again_published.at("served").at("default").at("config") == "mtp:3" &&
                          again_published.at("portable").at("default").at("config") == "mtp:3",
                      "spec-reselect publishes the served and portable choices");
    Profile unmeasured;
    failures += check(rejects([&] { (void)reselect(unmeasured, {.gpu_class = "gb203"}); }),
                      "spec-reselect needs measurements");
    return failures;
}

// What the serving Engine runs (spec/engine_support.h), and the served choices a tune records when
// its own choices need more (Gemma 4: the Session runs DFlash and adaptive depths).
int test_served() {
    int failures = 0;
    const auto gemma = engine_support("gemma4_text");
    const auto qwen  = engine_support("qwen3_5_moe_text");
    failures += check(gemma.runs(SpecConfig::parse("mtp:15")) &&
                          gemma.runs(SpecConfig::parse("plain")) &&
                          !gemma.runs(SpecConfig::parse("mtp:16")) &&
                          !gemma.runs(SpecConfig::parse("mtp:5:2")) &&
                          !gemma.runs(SpecConfig::parse("dflash:7")) &&
                          !gemma.runs(SpecConfig::parse("mtp:3+head")) &&
                          !gemma.runs(SpecConfig::parse("ngram:4")),
                      "Gemma 4's Engine runs fixed MTP windows up to 15 only");
    failures += check(qwen.runs(SpecConfig::parse("dflash:15")) &&
                          qwen.runs(SpecConfig::parse("dflash2:11+head")) &&
                          qwen.runs(SpecConfig::parse("mtp:3+head")) &&
                          !qwen.runs(SpecConfig::parse("dflash:16")) &&
                          !qwen.runs(SpecConfig::parse("mtp:8:2")),
                      "Qwen's Engine runs MTP to 5 and DFlash to 15, fixed");
    const auto dense = engine_support("dense_decoder");
    failures += check(dense.runs(SpecConfig::parse("dflash:15")) &&
                          dense.runs(SpecConfig::parse("dflash:3")) &&
                          !dense.runs(SpecConfig::parse("dflash:16")) &&
                          !dense.runs(SpecConfig::parse("dflash2:7")) &&
                          !dense.runs(SpecConfig::parse("dflash:7+head")) &&
                          !dense.runs(SpecConfig::parse("mtp:1")),
                      "the dense decoder's Engine runs DFlash to one block's drafts (15) only");
    const auto moe = engine_support("moe_decoder");
    failures += check(moe.runs(SpecConfig::parse("dflash:15")) &&
                          moe.runs(SpecConfig::parse("dflash:3")) &&
                          !moe.runs(SpecConfig::parse("dflash:16")) &&
                          !moe.runs(SpecConfig::parse("dflash2:7")) &&
                          !moe.runs(SpecConfig::parse("mtp:1")),
                      "the MoE decoder's Engine runs DFlash as the dense decoder's does");
    failures += check(!engine_support("no_such_family").any() &&
                          !engine_support("no_such_family").runs(SpecConfig::parse("dflash:3")),
                      "a family without speculation runs none");

    Profile p;
    p.drafters.push_back({"assistant", "gemma4_assistant", "mtp", "", "", "", "", "", 0, 0});
    p.drafters.push_back({"dflash", "dflash", "dflash", "", "", "", "", "", 0, 0});
    for (const char* f : {"math", "code", "prose", "thinking"}) {
        p.measurements.push_back(measure("plain", f, 100));
    }
    const auto add = [&](const char* label, double m, double c, double pr, double t) {
        p.measurements.push_back(measure(label, "math", m, 100, 300, 700));
        p.measurements.push_back(measure(label, "code", c, 100, 300, 700));
        p.measurements.push_back(measure(label, "prose", pr, 100, 50, 700));
        p.measurements.push_back(measure(label, "thinking", t, 100, 300, 700));
    };
    // The Session's best: DFlash on code and thinking, the adaptive window elsewhere. What an
    // Engine with MTP windows up to 5 runs: mtp:5 everywhere but prose, where mtp:2 wins.
    const EngineSupport mtp5{.mtp_depth = 5};
    add("dflash:11", 180, 250, 60, 230);
    add("mtp:8:2", 200, 220, 115, 210);
    add("mtp:5", 190, 210, 95, 200);
    add("mtp:2", 150, 160, 112, 150);
    select(p);
    failures += check(!p.served, "no served choices without the Engine's support");
    select(p, {.served = mtp5});
    failures +=
        check(p.thinking.config.label() == "dflash:11" && p.overall.config.label() == "mtp:8:2",
              "the tune's own choices are unchanged");
    failures += check(p.served.has_value(), "served choices when a choice needs more");
    if (p.served) {
        const auto& s = *p.served;
        failures +=
            check(s.non_thinking.config.label() == "mtp:2" && s.non_thinking.drafter == "assistant",
                  "served choices are the best the Engine runs");
        // mtp:5 loses to plain decode on prose (95 < 100), so the served default is mtp:2.
        failures += check(s.overall.config.label() == "mtp:2" &&
                              s.thinking.config.label() == "mtp:5",
                          "served defaults: " + s.overall.config.label() + ", " +
                              s.thinking.config.label());
        // The server's menu over the served choices runs the assistant at each default's depth.
        Profile served      = p;
        served.overall      = s.overall;
        served.non_thinking = s.non_thinking;
        served.thinking     = s.thinking;
        const auto menu     = build_menu(served, served.thinking, [&](const Choice& c) {
            return mtp5.runs(c.config);
        });
        // A tripped thinking request (mtp:5) steps down to the overall depth with its floor; the
        // shallowest entry has nowhere to go but plain decode.
        const auto steps = menu.step_down(menu.pick(true));
        failures += check(
            steps.size() == 1 && steps[0].depth == 2 && steps[0].floor >= menu.overall.floor &&
                menu.step_down(menu.pick(false)).empty() && menu.step_down(MenuChoice{}).empty(),
            "a menu entry steps down through the shallower menu depths");
        failures +=
            check(menu.mode == Mode::Mtp && menu.depths() == std::vector<std::int32_t>{2, 5} &&
                      menu.pick(false).config.k == 2 && menu.pick(true).config.k == 5 &&
                      menu.thinking.note.empty(),
                  "the served menu runs every default's choice");
    }
    // The tune's own choices laid onto the Engine: the primary (adaptive) cannot run, so nothing
    // speculates; the served choices exist for exactly this.
    const auto own = build_menu(p, p.thinking, [&](const Choice& c) { return mtp5.runs(c.config); });
    failures += check(own.mode == Mode::None, "the Session's choices give the Engine a plain menu");

    const auto again = profile_from_json(to_json(p));
    failures += check(again.served && again.served->thinking.config.label() == "mtp:5" &&
                          again.served->overall.config == p.served->overall.config,
                      "served choices survive the JSON round trip");
    const auto from =
        profile_from_published(published_from_profile(p, "gb202", "GPU", {}, "test artifact"));
    failures += check(from.served && from.served->thinking.config.label() == "mtp:5" &&
                          from.served->non_thinking.config.label() == "mtp:2",
                      "served choices survive publication");

    // An Engine that runs every choice needs no served section.
    Profile q = p;
    q.measurements.clear();
    for (const char* f : {"math", "code", "prose", "thinking"}) {
        q.measurements.push_back(measure("plain", f, 100));
        q.measurements.push_back(measure("mtp:3", f, 150, 100, 300, 700));
    }
    select(q, {.served = gemma});
    failures += check(!q.served && q.overall.config.label() == "mtp:3",
                      "no served section when the Engine runs every choice");
    return failures;
}

// The fit fallback (Profile::fit_fallback): what runs when the chosen drafter does not fit the
// requested context. It survives both JSON forms, and the shipped Qwen3.8-27B profile for the
// unsloth UD-IQ4_XS GGUF on gb203 (16 GB) carries DFlash2 K7 with MTP K3 as its fallback.
int test_fit_fallback() {
    int failures = 0;
    Profile p;
    p.overall = p.thinking = p.non_thinking =
        Choice{.config = SpecConfig::parse("dflash2:7"), .drafter = "dflash2", .tok_s = 190};
    p.fit_fallback     = Choice{.config = SpecConfig::parse("mtp:3"), .drafter = "mtp",
                                .tok_s = 125, .floor_tokens_per_round = 1.4};
    const auto again = profile_from_json(to_json(p));
    failures += check(again.fit_fallback && again.fit_fallback->config.label() == "mtp:3" &&
                          again.fit_fallback->drafter == "mtp",
                      "fit fallback survives the JSON round trip");
    const auto from =
        profile_from_published(published_from_profile(p, "gb203", "GPU", {}, "test artifact"));
    failures += check(from.fit_fallback && from.fit_fallback->config.label() == "mtp:3" &&
                          from.fit_fallback->floor_tokens_per_round == 1.4,
                      "fit fallback survives publication");
    Profile none = p;
    none.fit_fallback.reset();
    failures += check(!profile_from_json(to_json(none)).fit_fallback &&
                          !profile_from_published(published_from_profile(none, "gb203", "GPU", {},
                                                                         "x"))
                               .fit_fallback,
                      "no fit fallback unless one is given");

    const auto* model = Registry::builtin().by_repo("Qwen/Qwen3.8-27B");
    failures += check(model != nullptr, "registry has Qwen3.8-27B");
    failures += check(Registry::builtin().by_repo("unsloth/Qwen3.8-27B-GGUF") == model,
                      "the unsloth GGUF is a Qwen3.8-27B variant");
    if (model == nullptr) { return failures; }
    const Json profiles = model->profiles;
    const GpuKey gb203{.chain = {"gb203:84sm", "gb203", "sm120", "*"}, .memory_bytes = 16ULL << 30};
    const std::vector<std::string> both{"mtp", "dflash2"}, mtp_only{"mtp"};
    const auto* gguf = select_published(profiles, gb203, "unsloth/Qwen3.8-27B-GGUF", &both);
    failures += check(gguf != nullptr && gguf->value("gpu_class", std::string()) == "gb203",
                      "the UD-IQ4_XS GGUF takes its gb203 profile");
    if (gguf != nullptr) {
        const auto q = profile_from_published(*gguf);
        failures += check(q.overall.config.label() == "dflash2:7" && q.fit_fallback &&
                              q.fit_fallback->config.label() == "mtp:3",
                          "gb203 UD-IQ4_XS: dflash2:7, falling back to mtp:3");
    }
    const auto* other = select_published(profiles, gb203, "Qwen/Qwen3.8-27B", &both);
    failures += check(other == nullptr || other->value("gpu_class", std::string()) != "gb203",
                      "the gb203 profile applies to the unsloth GGUF only");
    const auto* no_dflash = select_published(profiles, gb203, "unsloth/Qwen3.8-27B-GGUF", &mtp_only);
    failures += check(no_dflash == nullptr || no_dflash->value("gpu_class", std::string()) != "gb203",
                      "without the DFlash2 component the gb203 profile does not apply");
    // A "*" profile published from a Blackwell tune can choose an NVFP4 drafter variant; a GPU
    // without NVFP4 (Ada, Ampere) takes the profile of the published drafter even when the variant
    // is attached, not a drafter its kernels cannot run.
    const Json star = Json::parse(R"([
        {"gpu_class": "*", "requires": ["assistant.nvfp4"], "choice": "mtp:6"},
        {"gpu_class": "*", "choice": "mtp:3"}])");
    const std::vector<std::string> attached{"assistant", "assistant.nvfp4"};
    const GpuKey ada{{"ad103:76sm", "ad103", "sm89", "*"}, 16ULL << 30, false};
    const GpuKey blackwell{{"gb205:48sm", "gb205", "sm120", "*"}, 12ULL << 30, true};
    const auto* on_ada       = select_published(star, ada, "", &attached);
    const auto* on_blackwell = select_published(star, blackwell, "", &attached);
    failures += check(on_ada != nullptr && !on_ada->contains("requires") &&
                          on_blackwell != nullptr && on_blackwell->contains("requires"),
                      "an NVFP4 drafter variant is chosen only where the GPU runs NVFP4");
    return failures;
}

// The bounds under decode load (Profile::load): the text form, both JSON forms, and malformed
// tables refused.
// The load model reproduces the depth tables the p34 grids measured best (27B, 8K, 14.10).
int test_load_model() {
    using namespace ninferno::spec;
    int failures = 0;
    failures += check(std::abs(tokens_per_round(0.75, 0.97, 0) - 1.0) < 1e-12, "E(0) = 1");
    failures += check(std::abs(tokens_per_round(0.5, 1.0, 2) - 1.75) < 1e-12, "E(2) at a 0.5, no decay");
    const double a = first_draft_acceptance(tokens_per_round(0.75, 0.97, 5), 5, 0.97);
    failures += check(std::abs(a - 0.75) < 1e-6, "first_draft_acceptance inverts tokens_per_round");
    failures += check(first_draft_acceptance(1.0, 3, 0.97) == 0.0, "no accepted drafts: a = 0");

    const std::int32_t dflash[] = {3, 5, 7};
    const std::int32_t mtp[]    = {1, 2, 3, 5};
    const RoundCost ada_dflash{.alpha = 0.0233, .beta = 0.0069, .knee = 8, .delta = 0.056, .pass_per_draft = false};
    const RoundCost ada_mtp{.alpha = 0.0233, .beta = 0.0069, .knee = 8, .delta = 0.080, .pass_per_draft = true};
    const RoundCost bw_dflash{.alpha = 0.0285, .beta = 0.0122, .knee = 12, .delta = 0.066, .pass_per_draft = false};
    const RoundCost bw_mtp{.alpha = 0.0285, .beta = 0.0122, .knee = 12, .delta = 0.066, .pass_per_draft = true};
    const auto expect = [&](const RoundCost& cost, double fa, double d, std::span<const std::int32_t> depths,
                            const char* want, const char* what) {
        const std::string got = load_label(modeled_load(cost, fa, d, depths, 8));
        return check(got == want, std::string(what) + ": " + got + " (expected " + want + ")");
    };
    failures += expect(ada_dflash, 0.75, 0.97, dflash, "2:5,4:3", "RTX 4090 DFlash2");
    failures += expect(ada_mtp, 0.77, 0.91, mtp, "1:3,5:2", "RTX 4090 MTP");
    failures += expect(bw_dflash, 0.75, 0.97, dflash, "2:5,3:3", "RTX 5090 DFlash2");
    failures += expect(bw_mtp, 0.77, 0.91, mtp, "1:3,4:2", "RTX 5090 MTP");
    // The card rates give the grids' alpha (RTX 4090: 128 SMs, 2.52 GHz, 1008 GB/s; RTX 5090: 170, 2.407, 1792).
    GpuKey ada{.chain = {"ad102"}, .sms = 128, .compute_capability = 89, .sm_clock_khz = 2520000, .dram_gbps = 1008};
    GpuKey bw{.chain = {"gb202"}, .sms = 170, .compute_capability = 120, .sm_clock_khz = 2407000, .dram_gbps = 1792};
    const auto ada_cost = round_cost_of(ada, 0.53, false);
    const auto bw_cost  = round_cost_of(bw, 0.53, false);
    failures += check(ada_cost && std::abs(ada_cost->alpha - 0.0233) < 0.001, "RTX 4090 alpha from its rates");
    failures += check(bw_cost && std::abs(bw_cost->alpha - 0.0285) < 0.001, "RTX 5090 alpha from its rates");
    if (ada_cost) { failures += expect(*ada_cost, 0.75, 0.97, dflash, "2:5,4:3", "RTX 4090 DFlash2 from its rates"); }
    if (bw_cost) { failures += expect(*bw_cost, 0.75, 0.97, dflash, "2:5,3:3", "RTX 5090 DFlash2 from its rates"); }
    GpuKey volta = ada; volta.compute_capability = 70;
    failures += check(!round_cost_of(volta, 0.53, false), "no grid for Volta: no model");
    // A free round keeps the deepest at every load: no bound.
    failures += expect(RoundCost{}, 0.75, 0.97, dflash, "none", "free rounds");
    // Drafts that are never accepted never pay: plain from one row.
    failures += expect(ada_dflash, 0.0, 0.97, dflash, "1:0", "no acceptance");
    return failures;
}

int test_load() {
    int failures = 0;
    const auto load = parse_load("2:7,3:3,5:0");
    failures += check(load.size() == 3 && load[0] == LoadBound{2, 7} &&
                          load[1] == LoadBound{3, 3} && load[2] == LoadBound{5, 0},
                      "parse_load reads lanes:depth entries");
    failures += check(load_label(load) == "2:7,3:3,5:0" && load_label({}) == "none",
                      "load_label is parse_load's inverse");
    failures += check(parse_load("none").empty() && parse_load("").empty(), "none: no bound");
    const auto wide = parse_load("2:7:block,3:3:drafts,5:0");
    failures += check(wide.size() == 3 && wide[0].width == DraftWidth::Block &&
                          wide[1].width == DraftWidth::Drafts &&
                          wide[2].width == DraftWidth::Default && wide[1].depth == 3 &&
                          load_label(wide) == "2:7:block,3:3:drafts,5:0",
                      "parse_load reads a DFlash bound's width and load_label writes it back");
    for (const char* bad : {"2:0:block", "2:3:wide", "2:3:", "2:3:drafts:block", "2::drafts"}) {
        failures += check(rejects([&] { (void)parse_load(bad); }),
                          std::string("parse_load refuses ") + bad);
    }
    // A stand-in profile's bounds on a smaller GPU: plain from ceil(sms / 10) rows.
    failures += check(load_label(stand_in_load(parse_load("3:3"), 66)) == "3:3,7:0",
                      "66 SMs: plain from 7 rows after the profile's own bounds");
    failures += check(load_label(stand_in_load(parse_load("3:3"), 80)) == "3:3,8:0",
                      "80 SMs: plain from 8 rows");
    failures += check(load_label(stand_in_load(parse_load("2:7,4:3,9:1"), 80)) == "2:7,4:3,8:0",
                      "bounds past the cap are dropped");
    failures += check(load_label(stand_in_load(parse_load("3:3,5:0"), 80)) == "3:3,5:0",
                      "a plain bound at or below the cap stands");
    failures +=
        check(load_label(stand_in_load({}, 36)) == "4:0", "36 SMs, no bounds: plain from 4");
    failures += check(load_label(stand_in_load({}, 8)) == "2:0", "the cap is at least 2 rows");
    failures +=
        check(load_label(stand_in_load(parse_load("3:3"), 0)) == "3:3", "unknown SMs: unchanged");
    for (const char* bad :
         {"2:7,2:3", "3:3,2:7", "0:3", "1:3", "2", "2:x", "2:16", "2:3,", "2:0,4:3", "3:0,5:1"}) {
        failures += check(rejects([&] { (void)parse_load(bad); }),
                          std::string("parse_load refuses ") + bad);
    }
    Profile p;
    p.overall = p.thinking = p.non_thinking =
        Choice{.config = SpecConfig::parse("dflash:5+head"), .drafter = "dflash"};
    p.load = load;
    failures += check(profile_from_json(to_json(p)).load == load,
                      "load survives the JSON round trip");
    const Json published = published_from_profile(p, "gb202", "GPU", {}, "x");
    failures += check(profile_from_published(published).load == load, "load survives publication");
    Profile widths         = p;
    widths.load            = wide;
    const Json widths_json = to_json(widths);
    failures += check(widths_json["load"][0]["draft_width"] == "block" &&
                          !widths_json["load"][2].contains("draft_width") &&
                          profile_from_json(widths_json).load == wide,
                      "a bound's width survives the JSON round trip, absent when the Engine's");
    for (const Json& entry : {Json{{"lanes", 2}, {"depth", 3}, {"draft_width", "wide"}},
                              Json{{"lanes", 2}, {"depth", 0}, {"draft_width", "drafts"}}}) {
        Json bad_width    = widths_json;
        bad_width["load"] = Json::array({entry});
        failures += check(rejects([&] { (void)profile_from_json(bad_width); }),
                          "a bound's draft_width is block or drafts, and none at depth 0");
    }
    Profile bounded       = p;
    bounded.context_limit = 32768;
    failures += check(profile_from_json(to_json(bounded)).context_limit == 32768 &&
                          profile_from_published(published_from_profile(bounded, "gb202", "GPU",
                                                                         {}, "x"))
                                  .context_limit == 32768 &&
                          !to_json(p).contains("context_limit") &&
                          profile_from_json(to_json(p)).context_limit == 0,
                      "context_limit survives the JSON round trip and publication, absent when 0");
    Profile none = p;
    none.load.clear();
    failures += check(!to_json(none).contains("load") &&
                          profile_from_json(to_json(none)).load.empty(),
                      "no load table unless one is measured");
    Json bad = to_json(p);
    bad["load"] = Json::array({Json{{"lanes", 4}, {"depth", 3}}, Json{{"lanes", 2}, {"depth", 1}}});
    failures += check(rejects([&] { (void)profile_from_json(bad); }),
                      "a load table must ascend by lanes");

    // spec-tune --load's table: each choice from the lanes after the previous concurrency, equal
    // neighbours and leading no-bounds dropped, plain for good once a concurrency is plain.
    const auto table = [](std::vector<LoadChoice> choices, std::int32_t window) {
        return load_label(load_table(choices, window));
    };
    failures += check(table({{2, 3}, {4, 3}, {8, 7}}, 11) == "2:3,5:7",
                      "load_table puts a choice after the previous concurrency and merges equals");
    failures += check(table({{2, 11}, {4, 3}, {8, 11}}, 11) == "3:3,5:11",
                      "load_table drops leading entries at the window (no bound)");
    failures += check(table({{2, 3}, {3, 0}, {4, 5}, {8, 2}}, 7) == "2:3,3:0",
                      "load_table keeps plain once a concurrency decodes plainly");
    failures += check(table({{2, 0}, {4, 5}}, 7) == "2:0", "load_table: plain from two requests");
    failures += check(rejects([&] { (void)load_table({{4, 3}, {2, 3}}, 7); }) &&
                          rejects([&] { (void)load_table({{1, 3}}, 7); }),
                      "load_table needs concurrencies ascending from 2");
    const auto widths_table = [](std::vector<LoadChoice> choices) {
        return load_label(load_table(choices, 7));
    };
    failures += check(widths_table({{2, 3, DraftWidth::Block},
                                    {4, 3, DraftWidth::Drafts},
                                    {6, 3, DraftWidth::Drafts},
                                    {8, 0, DraftWidth::Drafts}}) == "2:3:block,3:3:drafts,7:0",
                      "load_table keeps a width change at one depth and none at plain");
    failures += check(widths_table({{2, 7, DraftWidth::Block}, {4, 7}}) == "2:7:block,3:7",
                      "load_table keeps a leading entry at the window that names a width");

    // spec-tune --load's cell engines: one bound each, from two rows; none at the window.
    failures += check(load_label(load_cell_table(0, 15)) == "2:0" &&
                          load_label(load_cell_table(3, 15)) == "2:3" &&
                          load_cell_table(15, 15).empty(),
                      "load_cell_table bounds every step of two rows or more at the cell's bound");
    failures += check(rejects([&] { (void)load_cell_table(16, 15); }) &&
                          rejects([&] { (void)load_cell_table(-1, 15); }),
                      "load_cell_table refuses a bound beyond the window");
    // The engines for one bound: the widest first; what its pool does not hold is tried again on
    // an engine sized for it. Here an engine for w rows holds concurrencies up to 12 - w (a wider
    // engine keeps less for its pool), and none refuses: 8 holds up to 4 (so 8 is unheld), 6 holds
    // up to 6.
    {
        std::vector<std::uint32_t> engines;
        std::vector<std::uint32_t> measured;
        const auto unheld = measure_load_cells(
            {8, 2, 4, 6, 4}, [&](std::uint32_t widest, const std::vector<std::uint32_t>& wanted) {
                engines.push_back(widest);
                std::vector<std::uint32_t> held;
                for (const std::uint32_t c : wanted) {
                    if (c <= 12 - widest) { held.push_back(c); }
                }
                measured.insert(measured.end(), held.begin(), held.end());
                return held;
            });
        failures += check(engines == std::vector<std::uint32_t>{8, 6} &&
                              measured == std::vector<std::uint32_t>{2, 4, 6} &&
                              unheld == std::vector<std::uint32_t>{8},
                          "measure_load_cells: the widest engine first, one more for the rest");
        // An engine that does not fit at all: each concurrency is tried on its own, widest first.
        engines.clear();
        const auto none = measure_load_cells(
            {2, 4, 8}, [&](std::uint32_t widest, const std::vector<std::uint32_t>&) {
                engines.push_back(widest);
                return std::vector<std::uint32_t>{};
            });
        failures += check(engines == std::vector<std::uint32_t>{8, 4, 2} &&
                              none == std::vector<std::uint32_t>{2, 4, 8},
                          "measure_load_cells: an engine per concurrency when none fits");
    }

    // The cells' file: bound names, a round trip, and a resume refused on any other setting.
    failures += check(load_bound_name(0, 15) == "plain" && load_bound_name(15, 15) == "none" &&
                          load_bound_name(3, 15) == "3" &&
                          parse_load_bound_name("plain", 15) == 0 &&
                          parse_load_bound_name("none", 15) == 15 &&
                          parse_load_bound_name("6", 15) == 6,
                      "load bound names");
    failures += check(rejects([&] { (void)parse_load_bound_name("15", 15); }) &&
                          rejects([&] { (void)parse_load_bound_name("0", 15); }) &&
                          rejects([&] { (void)parse_load_bound_name("x3", 15); }) &&
                          rejects([&] { (void)parse_load_bound_name("", 15); }),
                      "unknown load bound names are refused");
    {
        const auto cell = [](std::uint32_t c, std::uint32_t bound, double tok_s) {
            LoadCell out{.concurrency = c, .bound = bound, .tokens = 1000};
            out.seconds = 1000.0 / tok_s;
            out.rounds  = bound == 0 ? 0 : 100;
            return out;
        };
        LoadCells kept;
        kept.settings      = Json{{"gpu", "ad102:128sm"}, {"temperature", 1.0}, {"corpus", "x"}};
        kept.window        = 15;
        kept.prefill_chunk = 256;
        kept.cells         = {cell(2, 0, 80), cell(2, 6, 254.7), cell(4, 15, 281.1)};
        kept.unheld        = {LoadCell{.concurrency = 8, .bound = 15}};
        const Json doc     = load_cells_json(kept, "", "");

        const LoadCells back = load_cells_from_json(doc, kept.settings);
        failures += check(back.window == 15 && back.prefill_chunk == 256 &&
                              back.cells.size() == 3 && back.cells[2].bound == 15 &&
                              back.cells[1].bound == 6 &&
                              std::abs(back.cells[1].tok_s() - 254.7) < 1e-9 &&
                              back.unheld.empty() && doc.at("unheld").size() == 1,
                          "load cells round trip (unheld cells are tried again, not taken over)");
        Json other   = kept.settings;
        other["gpu"] = "gb203:84sm";
        bool named   = false;
        try {
            (void)load_cells_from_json(doc, other);
        } catch (const std::runtime_error& error) {
            named = std::string(error.what()).find("gpu") != std::string::npos;
        }
        failures += check(named, "a resume measured on another GPU is refused, naming the field");
        Json more      = kept.settings;
        more["repeat"] = 2;
        failures += check(rejects([&] { (void)load_cells_from_json(doc, more); }),
                          "a resume without a setting this run has is refused");
        Json doubled = doc;
        doubled["cells"].push_back(doubled["cells"][0]);
        failures += check(rejects([&] { (void)load_cells_from_json(doubled, kept.settings); }),
                          "a resume with a cell twice is refused");
        failures +=
            check(rejects([&] { (void)load_cells_from_json(Json::object(), kept.settings); }),
                  "a file without settings is not resumed");

        // The choice per concurrency. C=2: bound 3 within 3 % of 4, the shallowest wins. C=4: plain
        // measured, every speculative bound unheld: undecided. C=6: bounds but no plain:
        // undecided. C=8: speculation within the gain: plain. C=3 not asked.
        const std::vector<LoadCell> measured = {
            cell(2, 0, 100), cell(2, 3, 198), cell(2, 4, 200), cell(2, 15, 150),
            cell(4, 0, 150), cell(6, 3, 300), cell(8, 0, 200), cell(8, 3, 204),
        };
        const LoadDecision decision =
            decide_load(measured, {8, 2, 4, 6}, {0, 3, 4, 15}, 0.03, 0.03);
        failures += check(decision.choices.size() == 2 && decision.choices[0].concurrency == 2 &&
                              decision.choices[0].depth == 3 &&
                              decision.choices[1].concurrency == 8 &&
                              decision.choices[1].depth == 0 &&
                              decision.undecided == std::vector<std::uint32_t>{4, 6},
                          "decide_load: shallowest within tolerance, gaps undecided");
        failures += check(rejects([&] { (void)decide_load(measured, {2}, {0, 3}, -0.1, 0.03); }) &&
                              rejects([&] { (void)decide_load(measured, {2}, {0, 3}, 0.03, -1); }),
                          "decide_load refuses a negative tolerance or gain");
        // A middle gap takes the next decided bound (3 up to 2 lanes, plain from 3), and nothing
        // above the widest decided one needs a cap.
        failures += check(load_label(decided_load_table(decision, 15)) == "2:3,3:0",
                          "a middle gap takes the next decided concurrency's bound");
        // Wider concurrencies asked and undecided: the bound above the widest decided is capped at
        // the shallowest decided (C=2 -> 6, C=4 -> 3; C=8 undecided: 3 from 5 lanes stays 3).
        const LoadDecision capped{.choices = {{2, 6}, {4, 3}}, .undecided = {8}};
        failures += check(load_label(decided_load_table(capped, 15)) == "2:6,3:3",
                          "a cap equal to the last bound adds nothing");
        const LoadDecision deeper{.choices = {{2, 3}, {4, 6}}, .undecided = {8}};
        failures += check(load_label(decided_load_table(deeper, 15)) == "2:3,3:6,5:3",
                          "above the widest decided, the shallowest decided bound caps the table");
        const LoadDecision widths{
            .choices   = {{2, 3, DraftWidth::Block}, {4, 6, DraftWidth::Drafts}},
            .undecided = {8}};
        failures += check(load_label(decided_load_table(widths, 15)) ==
                              "2:3:block,3:6:drafts,5:3:block",
                          "the cap above the widest decided keeps the shallowest choice's width");
        const LoadDecision none{.choices = {}, .undecided = {2, 4}};
        failures += check(decided_load_table(none, 15).empty(), "nothing decided: no table");
    }

    // A published profile without a table takes the "*" one with the same requires.
    const Json with_star = Json::array(
        {Json{{"gpu_class", "gb202"}, {"applies_to", Json::array({"a/b"})},
              {"requires", Json::array({"mtp", "dflash"})}},
         Json{{"gpu_class", "gb202"}, {"applies_to", Json::array({"a/c"})},
              {"requires", Json::array({"mtp"})}},
         Json{{"gpu_class", "*"}, {"applies_to", Json::array()},
              {"requires", Json::array({"dflash", "mtp"})},
              {"load", Json::array({Json{{"lanes", 2}, {"depth", 3}}})}},
         Json{{"gpu_class", "*"}, {"applies_to", Json::array()},
              {"requires", Json::array({"mtp"})}}});
    failures += check(select_load_fallback(with_star, with_star[0]) == &with_star[2],
                      "a profile without load takes the * table with the same requires");
    failures += check(select_load_fallback(with_star, with_star[1]) == nullptr &&
                          select_load_fallback(with_star, with_star[2]) == nullptr,
                      "no * table with the same requires, or a table of its own: nothing taken");
    // The drafter kind decides, and a sibling measured on the same GPU class comes first.
    const Json kinds = Json::parse(R"([
      {"gpu_class": "gb202", "applies_to": ["org/a"], "default": {"config": "mtp:10"}},
      {"gpu_class": "gb202", "applies_to": ["org/b"], "default": {"config": "mtp:10"},
       "load": [{"lanes": 2, "depth": 7}]},
      {"gpu_class": "*", "applies_to": [], "default": {"config": "mtp:10"},
       "load": [{"lanes": 2, "depth": 3}]},
      {"gpu_class": "gb203", "applies_to": ["org/a"], "default": {"config": "mtp:5"}},
      {"gpu_class": "gb203", "applies_to": ["org/c"], "default": {"config": "dflash2:11"}},
      {"gpu_class": "gb202", "applies_to": ["org/d"], "default": {"config": "dflash:5"}},
      {"gpu_class": "gb202", "applies_to": ["org/e"], "default": {"config": "mtp:10"},
       "requires": ["mtp"]}
    ])");
    failures += check(select_load_fallback(kinds, kinds[0]) == &kinds[1],
                      "a sibling on the same GPU class did not come before the * table");
    failures += check(select_load_fallback(kinds, kinds[3]) == &kinds[2],
                      "an MTP profile without a sibling did not take the MTP * table");
    failures += check(select_load_fallback(kinds, kinds[4]) == nullptr &&
                          select_load_fallback(kinds, kinds[5]) == nullptr,
                      "a DFlash profile took an MTP table");
    failures += check(select_load_fallback(kinds, kinds[6]) == nullptr,
                      "a same-class sibling that requires other drafters lent its table");

    // The Muse Glimmer 30B profile's table in the compiled-in registry: measured, one
    // draft deep from three lanes, one lane unbounded.
    const auto* muse = Registry::builtin().by_id("muse-glimmer-30b");
    bool muse_table = false;
    if (muse != nullptr) {
        for (const auto& profile : muse->profiles) {
            muse_table =
                muse_table ||
                (profile.value("gpu_class", std::string()) == "gb203" && profile.contains("load") &&
                 profile_from_published(profile).load == std::vector<LoadBound>{{3, 1}});
        }
    }
    failures += check(muse_table,
                      "the registry's Muse gb203 profile is not one draft deep from three lanes");
    return failures;
}

int test_select_published() {
    int failures = 0;
    const auto profiles = Json::parse(R"([
      {"gpu_class": "gb202", "applies_to": ["org/a"], "requires": ["dflash2"], "default": {"config": "dflash2:11"}},
      {"gpu_class": "gb202", "applies_to": ["org/a"], "default": {"config": "mtp:5"}},
      {"gpu_class": "gb202", "default": {"config": "mtp:3"}},
      {"gpu_class": "gb202:188sm", "applies_to": ["org/a"], "default": {"config": "mtp:6"}},
      {"gpu_class": "gb203", "mem": "<16gib", "default": {"config": "mtp:1"}},
      {"gpu_class": "gb203", "default": {"config": "mtp:7"}},
      {"gpu_class": "sm120", "default": {"config": "mtp:4"}},
      {"gpu_class": "*", "default": {"config": "mtp:2"}}
    ])");
    constexpr std::uint64_t kGiB = 1ULL << 30;
    const GpuKey rtx5090{{"gb202:170sm", "gb202", "sm120", "*"}, 31 * kGiB + kGiB / 3};
    const GpuKey pro6000{{"gb202:188sm", "gb202", "sm120", "*"}, 95 * kGiB};
    const GpuKey rtx5080{{"gb203:84sm", "gb203", "sm120", "*"}, 15 * kGiB + kGiB / 2};
    const GpuKey rtx5070ti_24{{"gb203:70sm", "gb203", "sm120", "*"}, 24 * kGiB};
    const GpuKey rtx5070{{"gb205:48sm", "gb205", "sm120", "*"}, 12 * kGiB};
    const GpuKey rtx4090{{"ad102:128sm", "ad102", "sm89", "*"}, 24 * kGiB};
    const auto label = [&](const GpuKey& gpu, std::string_view repo,
                           const std::vector<std::string>* available) {
        const auto* p = select_published(profiles, gpu, repo, available);
        return p ? (*p)["default"]["config"].get<std::string>() : std::string("none");
    };
    const std::vector<std::string> with{"mtp", "dflash2"}, without{"mtp"};
    failures += check(label(rtx5090, "org/a", &with) == "dflash2:11", "chip and repo first");
    failures += check(label(rtx5090, "ORG/A", &without) == "mtp:5", "requires skips");
    failures += check(label(rtx5090, "org/b", &with) == "mtp:3", "chip, any repo");
    failures += check(label(pro6000, "org/a", &with) == "mtp:6", "an SM variant beats its chip");
    failures += check(label(pro6000, "org/b", &with) == "mtp:3", "a variant falls back to its chip");
    failures += check(label(rtx5080, "org/b", &with) == "mtp:1", "a memory range admits");
    failures += check(label(rtx5070ti_24, "org/b", &with) == "mtp:7", "a memory range excludes");
    failures += check(label(rtx5070, "org/a", &with) == "mtp:4", "an unmeasured chip: its arch");
    failures += check(label(rtx4090, "org/a", &with) == "mtp:2", "another arch: the * fallback");
    // The profile using more of the attached drafters wins whatever the listing order.
    const auto listed = Json::parse(R"([
      {"gpu_class": "gb202", "applies_to": ["org/c"], "requires": ["mtp"], "default": {"config": "mtp:5"}},
      {"gpu_class": "gb202", "applies_to": ["org/c"], "requires": ["mtp", "dflash2"], "default": {"config": "dflash2:11"}}
    ])");
    const auto pick_listed = [&](const std::vector<std::string>* available) {
        const auto* p = select_published(listed, rtx5090, "org/c", available);
        return p ? (*p)["default"]["config"].get<std::string>() : std::string("none");
    };
    failures += check(pick_listed(&with) == "dflash2:11" && pick_listed(&without) == "mtp:5",
                      "the profile using the most attached drafters, whatever the order");
    failures += check(memory_in_range(">=24gib", 24 * kGiB) && !memory_in_range("<24gib", 24 * kGiB) &&
                          memory_in_range("<=31.5GiB", 31 * kGiB) && !memory_in_range(">16gib", 0) &&
                          !memory_in_range("24gib", 24 * kGiB) && memory_in_range("", 0),
                      "memory ranges");
    return failures;
}

int test_registry_and_corpus() {
    int failures = 0;
    const auto& registry = Registry::builtin();
    failures += check(registry.models().size() >= 4, "the registry lists the measured models");
    for (const char* repo : {"RedHatAI/gemma-4-26B-A4B-it-NVFP4", "nvidia/Gemma-4-31B-IT-NVFP4",
                             "Qwen/Qwen3.8-27B", "qwen/qwen3.6-35b-a3b"}) {
        failures += check(registry.by_repo(repo) != nullptr, std::string("registry has ") + repo);
    }
    failures += check(registry.by_repo("someone/unknown") == nullptr, "no entry for a stranger");
    // Versioned (tools/spec/registry_add.py bumps it), each profile dated and hashed.
    bool stamped = registry.version() >= 1;
    for (const auto& model : registry.models()) {
        for (const auto& profile : model.profiles) {
            stamped = stamped && profile.value("measured_at", "").size() == 10 &&
                      profile.value("revision", "").size() == 12;
        }
    }
    failures += check(stamped, "the registry has a version and every profile a date and revision");
    const auto version = [](const char* json) { return Registry::parse(json).version(); };
    failures += check(version(R"({"schema": 1, "models": []})") == 0 &&
                          version(R"({"schema": 1, "version": 7, "models": []})") == 7,
                      "a registry without a version reads as 0");
    // Muse Glimmer: the dense engine's DFlash, its drafter published only as a GGUF file.
    if (const auto* muse = registry.by_repo("unsloth/Muse-Glimmer-30B-GGUF")) {
        failures += check(muse->engine == "dense" && registry.supported("dflash", "dense") &&
                              muse->drafters.size() == 1 &&
                              muse->drafters[0].gguf == "dflash-Muse-Glimmer-30B-Q4_K_M" &&
                              drafter_json(muse->drafters[0]).value("gguf", "") ==
                                  muse->drafters[0].gguf &&
                              registry.by_lineage("meta-models/Muse-Glimmer-30B") == muse,
                          "Muse Glimmer's DFlash drafter is a GGUF file the dense engine runs");
        // The RTX 5080 speculates three drafts deep (K=3: 108.8 tok/s against plain 66.4).
        const GpuKey rtx5080{{"gb203:84sm", "gb203", "sm120", "*"}, 15ULL << 30};
        const auto* published =
            select_published(muse->profiles, rtx5080, "unsloth/Muse-Glimmer-30B-GGUF", nullptr);
        failures += check(published != nullptr &&
                              profile_from_published(*published).overall.config.label() ==
                                  "dflash:3",
                          "Muse Glimmer on the RTX 5080 drafts dflash:3");
        // From three lanes one draft deep (spec-tune --load: C=4 x1.21 and C=8 x1.05 over
        // plain, where K=3 gave x1.15 and x0.97; --spec auto C=4 +12.7 %, C=8 +13.3 %).
        failures += check(published != nullptr && profile_from_published(*published).load ==
                                                      std::vector<LoadBound>{{3, 1}},
                          "Muse Glimmer on the RTX 5080 is not one draft deep from three lanes");
        // The RTX 5080 profile is the unsloth quantizations': another repository of the lineage
        // gets none there (no gb203 profile for any repository, no "*" profile).
        failures += check(
            select_published(muse->profiles, rtx5080, "meta-models/Muse-Glimmer-30B-GGUF",
                             nullptr) == nullptr,
            "no Muse Glimmer profile for another repository on the RTX 5080");
        // The RTX 3090 has a profile of its own (spec-tune: dflash:7,
        // x2.69 over plain; under load three deep at two lanes, seven from three, three from five).
        const auto* on3090 = select_published(
            muse->profiles, GpuKey{{"ga102:82sm", "ga102", "sm86", "*"}, 24ULL << 30},
            "unsloth/Muse-Glimmer-30B-GGUF", nullptr);
        failures += check(on3090 != nullptr && on3090->value("gpu_class", "") == "ga102" &&
                              profile_from_published(*on3090).overall.config.label() == "dflash:7" &&
                              profile_from_published(*on3090).load ==
                                  std::vector<LoadBound>{{2, 3}, {3, 7}, {5, 3}},
                          "Muse Glimmer on the RTX 3090 drafts dflash:7");
        // Along the similarity chain an RTX 3070 (ga104, nobody measured it) takes the most similar
        // measured card's choices: no profile for ga104, ga103 or ga106, so the RTX 3090's, and
        // the note says whose.
        // The spec library needs no CUDA, so the chain is written out as
        // ninferno::gpu_similarity_chain("ga104", 46, 8, 6) builds it (tests/test_device.cpp).
        const GpuKey rtx3070{{"ga104:46sm", "ga104", "sm86", "ga103", "ga106", "ga102", "ga107",
                              "ga100", "*", "ad106", "ad104", "ad103", "ad107", "ad102", "tu104",
                              "tu106", "tu102"},
                             8ULL << 30,
                             false,
                             3};
        const auto* stand_in =
            select_published(muse->profiles, rtx3070, "unsloth/Muse-Glimmer-30B-GGUF", nullptr);
        failures += check(stand_in != nullptr && stand_in->value("gpu_class", "") == "ga102" &&
                              similarity_note(rtx3070, *stand_in).find("ga102 profile stands in") !=
                                  std::string::npos &&
                              similarity_note(rtx5080, *published).empty(),
                          "an unmeasured GPU takes the most similar measured GPU's profile");
        // A stand-in's floors are another GPU's: they are raised to the registry's
        // (raise_foreign_floors); the card's own profile, and a key list without own keys, keep
        // theirs.
        failures += check(measured_elsewhere(rtx3070, *stand_in) &&
                              measured_elsewhere(rtx3070, nlohmann::json{{"gpu_class", "*"}}) &&
                              !measured_elsewhere(rtx3070, nlohmann::json{{"gpu_class", "sm86"}}) &&
                              !measured_elsewhere(rtx5080, *published),
                          "a profile measured on another GPU is told apart from the card's own");
        failures += check(muse->vocab_fingerprint == "v1:49d8b46011586387" &&
                              muse->geometry.hidden_size == 6656 &&
                              muse->geometry.num_hidden_layers == 52 &&
                              muse->geometry.vocab_size == 202048,
                          "the Muse Glimmer entry records its vocabulary and geometry");
    } else {
        failures += check(false, "registry has unsloth/Muse-Glimmer-30B-GGUF");
    }
    // Nemotron 3.5 Lightning: the nemotron_h engine's MTP, its head a GGUF file of its own.
    if (const auto* nemo =
            registry.by_repo("bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF")) {
        failures += check(
            nemo->engine == "nemotron_h" && registry.supported("nemotron_h_mtp", "nemotron_h") &&
                nemo->drafters.size() == 1 && nemo->drafters[0].kind == "nemotron_h_mtp" &&
                nemo->drafters[0].gguf == "mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0" &&
                registry.by_repo("unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF") == nemo &&
                registry.by_lineage("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16") == nemo,
            "Nemotron 3.5 Lightning's MTP head is a GGUF file the nemotron_h engine runs");
        const auto choice = [&](const GpuKey& gpu, const char* repo) {
            const auto* published = select_published(nemo->profiles, gpu, repo, nullptr);
            return published ? profile_from_published(*published) : Profile{};
        };
        const GpuKey rtx5090{{"gb202:170sm", "gb202", "sm120", "*"}, 31ULL << 30};
        const GpuKey rtx4090{{"ad102:128sm", "ad102", "sm89", "*"}, 24ULL << 30};
        const auto on5090 = choice(rtx5090, "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF");
        const auto on4090 = choice(rtx4090, "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF");
        // Three drafts deep on both GPUs (spec-tune on the built-in corpus, the separate MTP head:
        // RTX 5090 599.0 tok/s against plain 414.5; RTX 4090 423.7 against 275.2).
        failures += check(on5090.overall.config.label() == "mtp:3" &&
                              on4090.overall.config.label() == "mtp:3",
                          "Nemotron 3.5 Lightning drafts mtp:3 on the RTX 5090 and RTX 4090");
        // Under load the RTX 5090 drafts two deep at two lanes (x1.31 over plain) and three from
        // three, the RTX 4090 the same (spec-tune --load 2026-10-20, C 2,4,8; before it two deep
        // from two lanes only).
        failures += check(on5090.load == std::vector<LoadBound>{{2, 2}, {3, 3}} &&
                              on4090.load == std::vector<LoadBound>{{2, 2}, {3, 3}},
                          "Nemotron 3.5 Lightning's load tables");
        // The RTX 4090's table is its own measurement: it is published, so no fallback table
        // replaces it.
        const auto* ad102 = select_published(
            nemo->profiles, rtx4090, "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", nullptr);
        failures += check(ad102 != nullptr && ad102->contains("load") &&
                              select_load_fallback(nemo->profiles, *ad102) == nullptr,
                          "the RTX 4090's load table is its own");
        // The RTX 5080 (gb203) is measured too: two deep, one deep from three lanes.
        const GpuKey rtx5080{{"gb203:84sm", "gb203", "sm120", "*"}, 15ULL << 30};
        const auto on5080 = choice(rtx5080, "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF");
        failures += check(on5080.overall.config.label() == "mtp:2" &&
                              on5080.load == std::vector<LoadBound>{{3, 1}},
                          "Nemotron 3.5 Lightning drafts mtp:2 on the RTX 5080");
        // The RTX 3090 (ga102) is measured too: three deep (x1.46 over plain), three deep at two
        // lanes.
        const GpuKey rtx3090{{"ga102:82sm", "ga102", "sm86", "*"}, 24ULL << 30};
        const auto on3090 = choice(rtx3090, "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF");
        failures += check(on3090.overall.config.label() == "mtp:3" &&
                              on3090.load == std::vector<LoadBound>{{2, 3}},
                          "Nemotron 3.5 Lightning drafts mtp:3 on the RTX 3090");
        // A GPU nobody measured, an RTX 3070 (its chain as ninferno::gpu_similarity_chain("ga104",
        // 46, 8, 6) builds it), takes the RTX 3090's profile: no ga104, ga103 or ga106 profile and
        // no "*" one. Its bounds are another card's, so they are capped to its 46 SMs: plain from
        // five rows.
        const GpuKey rtx3070{{"ga104:46sm", "ga104", "sm86", "ga103", "ga106", "ga102", "ga107",
                              "ga100", "*", "ad106", "ad104", "ad103", "ad107", "ad102", "tu104",
                              "tu106", "tu102"},
                             8ULL << 30,
                             false,
                             3,
                             46};
        const auto* nemo3070 = select_published(
            nemo->profiles, rtx3070, "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", nullptr);
        failures += check(nemo3070 != nullptr && nemo3070->value("gpu_class", "") == "ga102" &&
                              profile_from_published(*nemo3070).overall.config.label() == "mtp:3" &&
                              measured_elsewhere(rtx3070, *nemo3070) &&
                              stand_in_load(profile_from_published(*nemo3070).load, rtx3070.sms) ==
                                  std::vector<LoadBound>{{2, 3}, {5, 0}},
                          "an RTX 3070 takes the RTX 3090's Nemotron 3.5 Lightning profile, capped");
        failures += check(nemo->vocab_fingerprint == "v1:9e123547b12f55ce" &&
                              nemo->geometry.hidden_size == 2688 &&
                              nemo->geometry.num_hidden_layers == 52 &&
                              nemo->geometry.vocab_size == 131072,
                          "the Nemotron 3.5 Lightning entry records its vocabulary and geometry");
    } else {
        failures +=
            check(false, "registry has bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF");
    }
    failures += check(registry.by_lineage("google/gemma-4-26B-A4B-it") != nullptr,
                      "lineage lookup");
    // gb203 boards tune apart. The RTX 5070 Ti (70 SMs) has a profile of its SM variant
    // "gb203:70sm" for the GGUF quantizations it was measured on (unsloth UD-IQ3_XXS: mtp:2, two
    // deep from two lanes); the safetensors and NVFP4 checkpoints keep the gb203 profiles the
    // RTX 5080 measured for them (mtp:8:2, two lanes three deep), and a 5080 running a GGUF takes
    // neither the 5070 Ti's nor a gb203 one: the RTX 5060 Ti's (gb206) stands in, capped to its
    // 84 SMs. Chains as ninferno::gpu_similarity_chain builds them.
    if (const auto* g26 = registry.by_repo("google/gemma-4-26B-A4B-it")) {
        const GpuKey rtx5080{{"gb203:84sm", "gb203", "sm120", "gb205", "gb202", "gb206", "gb207",
                              "*"},
                             15ULL << 30, true, 3, 84};
        const GpuKey rtx5070ti{{"gb203:70sm", "gb203", "sm120", "gb205", "gb206", "gb202",
                                "gb207", "*"},
                               15ULL << 30, true, 3, 70};
        const auto pick = [&](const GpuKey& gpu, const char* repo) {
            return select_published(g26->profiles, gpu, repo, nullptr);
        };
        const auto label = [](const nlohmann::json* p) {
            return profile_from_published(*p).overall.config.label();
        };
        const auto load = [](const nlohmann::json* p) { return profile_from_published(*p).load; };
        const auto* google5080   = pick(rtx5080, "google/gemma-4-26B-A4B-it");
        const auto* google5070ti = pick(rtx5070ti, "google/gemma-4-26B-A4B-it");
        const auto* redhat5070ti = pick(rtx5070ti, "RedHatAI/gemma-4-26B-A4B-it-NVFP4");
        const auto* gguf5070ti   = pick(rtx5070ti, "unsloth/gemma-4-26B-A4B-it-GGUF");
        const auto* gguf5080     = pick(rtx5080, "unsloth/gemma-4-26B-A4B-it-GGUF");
        failures += check(google5080 != nullptr && google5080->value("gpu_class", "") == "gb203" &&
                              label(google5080) == "mtp:8:2" &&
                              load(google5080) == std::vector<LoadBound>{{2, 3}},
                          "Gemma 4 26B (google) on the RTX 5080: its gb203 profile, 2:3");
        failures += check(google5070ti == google5080,
                          "Gemma 4 26B (google) on the RTX 5070 Ti: the same gb203 profile");
        const auto* lender =
            redhat5070ti != nullptr ? select_load_fallback(g26->profiles, *redhat5070ti) : nullptr;
        failures += check(redhat5070ti != nullptr &&
                              redhat5070ti->value("gpu_class", "") == "gb203" &&
                              label(redhat5070ti) == "mtp:8:2" && lender == google5080,
                          "Gemma 4 26B (RedHatAI NVFP4) on the RTX 5070 Ti: its gb203 profile, "
                          "the google gb203 load table");
        failures += check(gguf5070ti != nullptr &&
                              gguf5070ti->value("gpu_class", "") == "gb203:70sm" &&
                              label(gguf5070ti) == "mtp:2" &&
                              load(gguf5070ti) == std::vector<LoadBound>{{2, 2}} &&
                              !measured_elsewhere(rtx5070ti, *gguf5070ti),
                          "Gemma 4 26B (unsloth GGUF) on the RTX 5070 Ti: gb203:70sm, mtp:2, 2:2");
        failures += check(gguf5080 != nullptr && gguf5080->value("gpu_class", "") == "gb206" &&
                              label(gguf5080) == "mtp:8:2" &&
                              load(gguf5080) == std::vector<LoadBound>{{2, 3}} &&
                              measured_elsewhere(rtx5080, *gguf5080) &&
                              stand_in_load(load(gguf5080), rtx5080.sms) ==
                                  std::vector<LoadBound>{{2, 3}, {9, 0}},
                          "Gemma 4 26B (unsloth GGUF) on the RTX 5080: the gb206 stand-in, "
                          "mtp:8:2, 2:3 capped at 9 rows");
    } else {
        failures += check(false, "registry has google/gemma-4-26B-A4B-it");
    }
    // Qwen3.8 27B on the RTX 4070 Ti (ad104, 60 SMs): the GGUF it was measured on drafts dflash2:7,
    // three deep from three lanes; the safetensors checkpoint keeps the RTX 4070 Ti SUPER's
    // profile measured for it (ad103).
    if (const auto* q38 = registry.by_repo("Qwen/Qwen3.8-27B")) {
        const GpuKey rtx4070ti{{"ad104:60sm", "ad104", "sm89", "ad103", "ad106", "ad102", "ad107",
                                "*"},
                               12ULL << 30, false, 3, 60};
        const auto* gguf = select_published(q38->profiles, rtx4070ti, "unsloth/Qwen3.8-27B-GGUF",
                                            nullptr);
        const auto* base =
            select_published(q38->profiles, rtx4070ti, "Qwen/Qwen3.8-27B", nullptr);
        failures += check(gguf != nullptr && gguf->value("gpu_class", "") == "ad104" &&
                              profile_from_published(*gguf).overall.config.label() == "dflash2:7" &&
                              profile_from_published(*gguf).load == std::vector<LoadBound>{{3, 3}},
                          "Qwen3.8 27B (unsloth GGUF) on the RTX 4070 Ti: ad104, dflash2:7, 3:3");
        failures += check(base != nullptr && base->value("gpu_class", "") == "ad103",
                          "Qwen3.8 27B (Qwen) on the RTX 4070 Ti: the ad103 profile stands in");
    } else {
        failures += check(false, "registry has Qwen/Qwen3.8-27B");
    }
    // gpt-oss-20b was tuned on an RTX 5070 Ti, published for its SM variant "gb203:70sm": the 5070 Ti
    // runs it as its own (dflash:7, two lanes three deep), an RTX 5080 (no gb203 profile) takes the
    // RTX 5090's as a stand-in (dflash:7 for the openai checkpoint, dflash:5 for the GGUFs), 2:3
    // capped at 9 rows for its 84 SMs.
    if (const auto* oss = registry.by_repo("openai/gpt-oss-20b")) {
        const GpuKey rtx5080{{"gb203:84sm", "gb203", "sm120", "gb205", "gb202", "gb206", "gb207",
                              "*"},
                             15ULL << 30, true, 3, 84};
        const GpuKey rtx5070ti{{"gb203:70sm", "gb203", "sm120", "gb205", "gb206", "gb202",
                                "gb207", "*"},
                               15ULL << 30, true, 3, 70};
        const auto label = [](const nlohmann::json* p) {
            return profile_from_published(*p).overall.config.label();
        };
        const auto load = [](const nlohmann::json* p) { return profile_from_published(*p).load; };
        const auto* own = select_published(oss->profiles, rtx5070ti, "openai/gpt-oss-20b", nullptr);
        failures += check(own != nullptr && own->value("gpu_class", "") == "gb203:70sm" &&
                              label(own) == "dflash:7" &&
                              load(own) == std::vector<LoadBound>{{2, 3}} &&
                              !measured_elsewhere(rtx5070ti, *own),
                          "gpt-oss-20b on the RTX 5070 Ti: its gb203:70sm profile, dflash:7, 2:3");
        const auto* base = select_published(oss->profiles, rtx5080, "openai/gpt-oss-20b", nullptr);
        const auto* gguf =
            select_published(oss->profiles, rtx5080, "unsloth/gpt-oss-20b-GGUF", nullptr);
        failures += check(base != nullptr && base->value("gpu_class", "") == "gb202" &&
                              label(base) == "dflash:7" && measured_elsewhere(rtx5080, *base) &&
                              stand_in_load(load(base), rtx5080.sms) ==
                                  std::vector<LoadBound>{{2, 3}, {9, 0}},
                          "gpt-oss-20b on the RTX 5080: the RTX 5090's profile stands in, "
                          "dflash:7, 2:3 capped at 9 rows");
        failures += check(gguf != nullptr && gguf->value("gpu_class", "") == "gb202" &&
                              label(gguf) == "dflash:5" &&
                              stand_in_load(load(gguf), rtx5080.sms) ==
                                  std::vector<LoadBound>{{2, 3}, {9, 0}},
                          "gpt-oss-20b (unsloth GGUF) on the RTX 5080: the RTX 5090's GGUF "
                          "profile, dflash:5");
    } else {
        failures += check(false, "registry has openai/gpt-oss-20b");
    }
    failures += check(registry.kind_for_architecture("DFlashDraftModel") == "dflash" &&
                          registry.supported("dflash2", "qwen3_5") &&
                          !registry.supported("eagle3", "gemma4"),
                      "kinds");
    const std::regex sha("^[0-9a-f]{40}$");
    for (const auto& model : registry.models()) {
        std::set<std::string> names;
        for (const auto& d : model.drafters) {
            names.insert(d.name);
            for (const auto& v : d.variants) { names.insert(d.name + "." + v); }
            failures += check(registry.supported(d.kind, model.engine),
                              model.id + ": drafter kind " + d.kind + " runs on " + model.engine);
            if (!d.repo.empty()) {
                failures += check(std::regex_match(d.revision, sha),
                                  model.id + ": " + d.repo + " is pinned to a commit");
            }
        }
        failures += check(!model.profiles.empty(), model.id + " publishes a profile");
        for (const auto& published : model.profiles) {
            // A chip (with an optional SM variant), an architecture, or "*": never a memory size.
            const auto cls = published.value("gpu_class", std::string());
            failures += check(cls == "*" ||
                                  std::regex_match(cls, std::regex("^[a-z]+[0-9]+(:[0-9]+sm)?$")),
                              model.id + ": gpu_class '" + cls + "' is a class key");
            Profile p;
            try {
                p = profile_from_published(published);
            } catch (const std::exception& e) {
                failures += check(false, model.id + ": profile does not parse: " + e.what());
                continue;
            }
            std::vector<const Choice*> choices{&p.overall, &p.non_thinking, &p.thinking};
            for (const auto* c : choices) {
                if (!c->config.enabled()) { continue; }
                failures += check(names.contains(c->drafter),
                                  model.id + ": " + c->config.label() + " names drafter '" +
                                      c->drafter + "'");
            }
            // A drafter variant (NInferno's own quantization) is required by every profile that
            // runs it, so a model without it attached takes the published drafters' profile; an
            // NVFP4 variant is chosen only where it was measured on hardware with FP4 (never for
            // unmeasured GPUs, "*", nor for the RTX 4090, ad102).
            for (const auto* s : {p.served ? &*p.served : nullptr, p.portable ? &*p.portable : nullptr}) {
                if (s == nullptr) { continue; }
                for (const auto* c : {&s->overall, &s->non_thinking, &s->thinking}) {
                    choices.push_back(c);
                }
            }
            if (p.fit_fallback) { choices.push_back(&*p.fit_fallback); }
            const auto required = published.value("requires", Json::array());
            for (const auto* c : choices) {
                const auto variant = split_variant(c->drafter).second;
                if (!c->config.enabled() || variant.empty()) { continue; }
                failures += check(std::find(required.begin(), required.end(), c->drafter) !=
                                      required.end(),
                                  model.id + " " + cls + ": a profile running " + c->drafter +
                                      " requires it");
                failures += check(variant != "nvfp4" || (cls != "*" && cls != "ad102"),
                                  model.id + " " + cls + ": no NVFP4 drafter variant here");
            }
        }
    }
    // The published Qwen3.8-27B artifact carries DFlash2: on its chip it runs the DFlash2 profile.
    {
        const GpuKey rtx5090{{"gb202:170sm", "gb202", "sm120", "*"}, 31ULL << 30};
        const std::vector<std::string> both{"mtp", "dflash2"}, mtp_only{"mtp"};
        const auto& profiles = registry.by_id("qwen3.8-27b")->profiles;
        const auto* with     = select_published(profiles, rtx5090, "Qwen/Qwen3.8-27B", &both);
        const auto* without  = select_published(profiles, rtx5090, "Qwen/Qwen3.8-27B", &mtp_only);
        const auto needs     = [](const Json* p) { return p ? p->value("requires", Json::array()) : Json(); };
        failures += check(with != nullptr && needs(with).size() == 2 && without != nullptr &&
                              needs(without).size() == 1,
                          "qwen3.8-27b: the DFlash2 profile with DFlash2 attached, MTP without");
    }
    const auto& corpus = builtin_corpus();
    for (const char* family : kFamilies) {
        int n = 0;
        for (const auto& p : corpus.prompts) { n += p.family == family; }
        failures += check(n >= 8, std::string("corpus family ") + family);
    }
    const auto* gemma = corpus.tokens_for("v1:9a7777576d37749e");
    failures += check(gemma != nullptr && gemma->prompts.size() == corpus.prompts.size() &&
                          !gemma->stops.empty(),
                      "the corpus carries Gemma 4's rendered prompts");
    return failures;
}

int test_resolve() {
    int failures = 0;
    const GpuKey rtx5090{{"gb202:170sm", "gb202", "sm120", "*"}, 32ULL << 30};
    const GpuKey pro6000{{"gb202:188sm", "gb202", "sm120", "*"}, 96ULL << 30};
    namespace fs = std::filesystem;
    const auto dir =
        fs::temp_directory_path() / ("ninferno_spec_test_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const auto artifact = dir / "model.ninferno"; // never opened: ids come from the files
    const auto write    = [](const fs::path& path, const Json& doc) {
        std::ofstream(path) << doc.dump(1);
    };
    // A manifest for another artifact is passed over; nothing else: nothing found.
    Json manifest = {{"schema", 1},
                     {"target", {{"artifact_id", "aaaa"}, {"repo", "org/a"}, {"engine", "qwen3_5"}}},
                     {"source", "registry"},
                     {"fallback", "none"},
                     {"drafters",
                      Json::array({{{"name", "mtp"}, {"kind", "mtp"}, {"component", "mtp"},
                                    {"status", "embedded"}},
                                   {{"name", "dflash2"}, {"kind", "dflash2"},
                                    {"component", "dflash2"}, {"status", "not in the artifact"}}})},
                     {"profiles", Json::parse(R"([
                        {"gpu_class": "*", "requires": ["dflash2"], "default": {"config": "dflash2:11+head", "drafter": "dflash2"}},
                        {"gpu_class": "*", "requires": ["mtp"], "default": {"config": "mtp:5+head", "drafter": "mtp", "floor_tokens_per_round": 1.6}}
                      ])")}};
    // No manifest beside the artifact (a model converted without --attach-drafters, or attached
    // beside a symlink to it): nothing found, and the note says what writes one.
    auto r = resolve_speculation(artifact, "aaaa", rtx5090);
    failures += check(!r.found() && r.notes.size() == 1 &&
                          r.notes[0].rfind("no model.ninferno.speculation.json beside the artifact",
                                           0) == 0 &&
                          r.notes[0].find("--attach-drafters") != std::string::npos,
                      "a missing manifest is not named in the notes");
    write(manifest_path(artifact), manifest);
    r = resolve_speculation(artifact, "bbbb", rtx5090);
    failures += check(!r.found() && !r.notes.empty(), "a manifest of another artifact is ignored");
    r = resolve_speculation(artifact, "aaaa", rtx5090);
    // The "*" profile was measured on no GPU in particular: its floor is at least the largest the
    // model's own profiles measured for the configuration (here its own, 1.6).
    failures += check(r.found() && r.profile.overall.config.label() == "mtp:5+head" &&
                          r.drafters.at("mtp") == "component:mtp" && r.foreign_floors &&
                          r.floor_sources.is_array() &&
                          r.profile.overall.floor_tokens_per_round == 1.6,
                      "the manifest's profile for the attached drafters");
    // A maintainer's tune for this artifact and GPU wins.
    Profile tuned;
    tuned.key.artifact_id = "aaaa";
    tuned.key.gpu         = "gb202:170sm";
    tuned.drafters.push_back({"mtp", "mtp", "mtp", "", "", "", "accepted", "", 2.0, 0.7});
    tuned.overall.config  = SpecConfig::parse("mtp:3+head");
    tuned.overall.drafter = "mtp";
    write_profile(default_profile_path(artifact), tuned);
    // Left beside the artifact, it is not read (a machine's own measurement steers nothing by
    // itself); a note names the option that reads it.
    r = resolve_speculation(artifact, "aaaa", rtx5090);
    failures += check(r.found() && !r.source.starts_with("local tune") &&
                          std::any_of(r.notes.begin(), r.notes.end(),
                                      [](const std::string& n) {
                                          return n.find("is not read") != std::string::npos &&
                                                 n.find("--spec-profile") != std::string::npos;
                                      }),
                      "a local tune beside the artifact is not read unless named");
    const auto named = default_profile_path(artifact);
    r                = resolve_speculation(artifact, "aaaa", rtx5090, named);
    failures += check(r.found() && r.source.starts_with("local tune") &&
                          r.profile.overall.config.label() == "mtp:3+head",
                      "a named local tune wins");
    r = resolve_speculation(artifact, "aaaa", rtx5090, artifact.parent_path() / "absent.json");
    failures += check(r.found() && !r.source.starts_with("local tune") &&
                          std::any_of(r.notes.begin(), r.notes.end(),
                                      [](const std::string& n) {
                                          return n.rfind("the named tune ", 0) == 0;
                                      }),
                      "an unreadable named tune is a note, and the published profile runs");
    // The tune measured no load bounds: the published ones of its drafter kind apply.
    failures += check(r.profile.load.empty(), "no load bounds without a published table");
    manifest["profiles"][1]["load"] =
        Json::array({Json{{"lanes", 2}, {"depth", 2}}, Json{{"lanes", 4}, {"depth", 0}}});
    write(manifest_path(artifact), manifest);
    r = resolve_speculation(artifact, "aaaa", rtx5090, named);
    failures += check(r.source.starts_with("local tune") &&
                          r.profile.overall.config.label() == "mtp:3+head" &&
                          r.profile.load.size() == 2 && r.profile.load[0].depth == 2 &&
                          r.profile.load[1].depth == 0 &&
                          std::any_of(r.notes.begin(), r.notes.end(),
                                      [](const std::string& n) {
                                          return n.find("the local tune measured none") !=
                                                 std::string::npos;
                                      }),
                      "a local tune takes the published load bounds of its kind");
    manifest["profiles"][1].erase("load");
    write(manifest_path(artifact), manifest);
    r = resolve_speculation(artifact, "aaaa", pro6000, named);
    failures += check(r.found() && r.source.starts_with("manifest"),
                      "a tune for another GPU class falls back to the manifest");
    // A foreign profile's floors: the largest the same model measured for exactly that
    // configuration on any GPU (another depth, head or an adaptive choice does not count), or the
    // registry median for a model measured nowhere; raised, never lowered.
    {
        const Json sources = Json::parse(R"([
            {"gpu_class": "gb202", "default": {"config": "dflash:7", "floor_tokens_per_round": 1.8}},
            {"gpu_class": "gv100", "default": {"config": "dflash:7", "floor_tokens_per_round": 2.6},
             "thinking": {"config": "dflash:5", "floor_tokens_per_round": 9.0}},
            {"gpu_class": "ad102", "default": {"config": "dflash:7:3", "floor_tokens_per_round": 9.0},
             "thinking": {"config": "mtp:7+head", "floor_tokens_per_round": 9.0}}])");
        failures += check(floor_ceiling(sources, SpecConfig::parse("dflash:7")) == 2.6 &&
                              floor_ceiling(sources, SpecConfig::parse("dflash:6")) == 0.0 &&
                              floor_ceiling(sources, SpecConfig{}) == 0.0,
                          "a configuration's floor ceiling: exactly that one, over the model's GPUs");
        ResolvedSpeculation from;
        from.foreign_floors = true;
        from.floor_sources  = sources;
        Profile foreign;
        foreign.overall.config                  = SpecConfig::parse("dflash:7");
        foreign.overall.floor_tokens_per_round  = 1.1;
        foreign.thinking.config                 = SpecConfig::parse("dflash:7");
        foreign.thinking.floor_tokens_per_round = 3.0;
        std::vector<std::string> raised;
        raise_foreign_floors(foreign, from, raised);
        ResolvedSpeculation own;
        failures += check(foreign.overall.floor_tokens_per_round == 2.6 &&
                              foreign.thinking.floor_tokens_per_round == 3.0 &&
                              raised.size() == 1 &&
                              foreign_floor(own, SpecConfig::parse("dflash:7"), 1.1) == 1.1,
                          "a foreign profile's floors rise to its model's ceiling, never fall; a "
                          "profile of this GPU keeps its own");
        ResolvedSpeculation nowhere;
        nowhere.foreign_floors = true;
        const double median    = registry_floor_median(SpecConfig::parse("mtp:5"));
        failures += check(median > 1.0 &&
                              foreign_floor(nowhere, SpecConfig::parse("mtp:5"), 0.0) == median &&
                              registry_floor_median(SpecConfig{}) == 0.0,
                          "a model measured nowhere: the registry median");
    }
    // A manifest the registry wrote, naming its entry: the profiles of the registry this binary
    // carries replace the (stale, here memory-keyed) copy taken at conversion.
    fs::remove(default_profile_path(artifact));
    manifest["entry"]           = "qwen3.8-27b";
    manifest["target"]["repo"]  = "Qwen/Qwen3.8-27B";
    manifest["profiles"]        = Json::parse(R"([
        {"gpu_class": "sm120-32gb", "default": {"config": "mtp:2+head", "drafter": "mtp"}},
        {"gpu_class": "*", "default": {"config": "mtp:2+head", "drafter": "mtp"}}])");
    write(manifest_path(artifact), manifest);
    r = resolve_speculation(artifact, "aaaa", rtx5090);
    const std::vector<std::string> attached{"mtp"};
    const auto* current = select_published(Registry::builtin().by_id("qwen3.8-27b")->profiles,
                                           rtx5090, "Qwen/Qwen3.8-27B", &attached);
    failures += check(r.found() && r.source.starts_with("registry qwen3.8-27b") &&
                          current != nullptr &&
                          r.profile.overall.config.label() ==
                              (*current)["default"]["config"].get<std::string>(),
                      "a registry manifest takes the current registry's profiles");
    manifest["source"] = "bundle";
    write(manifest_path(artifact), manifest);
    r = resolve_speculation(artifact, "aaaa", rtx5090);
    failures += check(r.found() && r.source.starts_with("manifest") &&
                          r.profile.overall.config.label() == "mtp:2+head",
                      "a bundle declaration keeps its own profiles");
    fs::remove_all(dir);
    return failures;
}

// Drafter variants: NInferno's own quantizations of a drafter, built beside it by the converter and
// chosen per GPU class by the profiles that require them.
int test_variants() {
    int failures = 0;
    const auto d = parse_drafter(Json::parse(
        R"({"name": "assistant", "kind": "gemma4_assistant", "variants": ["q8", "nvfp4"]})"));
    failures += check(d.variants == std::vector<std::string>{"q8", "nvfp4"} &&
                          drafter_json(d).at("variants").size() == 2,
                      "a drafter declares its variants");
    failures += check(split_variant("assistant.q8") == std::pair<std::string, std::string>{"assistant", "q8"} &&
                          split_variant("dflash").second.empty(),
                      "split_variant");
    failures += check(rejects([] {
                          (void)parse_drafter(Json::parse(
                              R"({"name": "assistant", "kind": "gemma4_assistant", "variants": ["q3"]})"));
                      }),
                      "an unknown variant is refused");
    failures += check(rejects([] {
                          (void)parse_drafter(Json::parse(R"({"name": "a.b", "kind": "gemma4_assistant"})"));
                      }),
                      "a declared name with a '.' is refused");
    failures += check(rejects([] {
                          (void)parse_drafter(Json::parse(
                              R"({"name": "dflash", "kind": "dflash", "component": "dflash", "variants": ["q8"]})"));
                      }),
                      "a companion component has no variants");
    const auto v = parse_drafter(Json::parse(
        R"({"name": "assistant.q8", "kind": "gemma4_assistant", "quant": "q8"})"));
    failures += check(v.quant == "q8" && drafter_json(v).at("quant") == "q8",
                      "a planned variant carries its quantization");

    // A tune that chose a variant publishes a profile that requires it.
    Profile tuned;
    tuned.overall.config      = SpecConfig::parse("mtp:8");
    tuned.overall.drafter     = "assistant.q8";
    tuned.non_thinking        = tuned.overall;
    tuned.thinking.config     = SpecConfig::parse("dflash:7");
    tuned.thinking.drafter    = "dflash.q8";
    const auto published = published_from_profile(tuned, "gb202", "RTX 5090", {}, "test");
    failures += check(published.value("requires", Json::array()) ==
                          Json::array({"assistant.q8", "dflash.q8"}),
                      "a variant profile requires its drafters: " +
                          published.value("requires", Json::array()).dump());
    Profile plain_tune = tuned;
    plain_tune.overall.drafter = plain_tune.non_thinking.drafter = "assistant";
    plain_tune.thinking.drafter          = "dflash";
    failures += check(!published_from_profile(plain_tune, "gb202", "", {}, "").contains("requires"),
                      "a profile of the published drafters requires nothing");

    // Per class: the variant's profile where the model has it attached, the published drafters'
    // profile where it has not; another class keeps its own.
    const auto profiles = Json::parse(R"([
        {"gpu_class": "gb202", "default": {"config": "mtp:3", "drafter": "assistant"}},
        {"gpu_class": "gb202", "requires": ["assistant.nvfp4"], "default": {"config": "mtp:8", "drafter": "assistant.nvfp4"}},
        {"gpu_class": "ad102", "requires": ["assistant.q8"], "default": {"config": "mtp:6", "drafter": "assistant.q8"}},
        {"gpu_class": "*", "default": {"config": "mtp:3", "drafter": "assistant"}}])");
    const GpuKey rtx5090{{"gb202:170sm", "gb202", "sm120", "*"}, 32ULL << 30};
    const GpuKey rtx4090{{"ad102:128sm", "ad102", "sm89", "*"}, 24ULL << 30};
    const std::vector<std::string> all{"assistant", "assistant.q8", "assistant.nvfp4"};
    const std::vector<std::string> published_only{"assistant"};
    const auto drafter = [](const Json* p) {
        return p ? p->at("default").value("drafter", std::string()) : std::string("none");
    };
    failures += check(drafter(select_published(profiles, rtx5090, "", &all)) == "assistant.nvfp4",
                      "gb202 takes its measured variant");
    failures += check(drafter(select_published(profiles, rtx4090, "", &all)) == "assistant.q8",
                      "ad102 takes its own measured variant");
    failures += check(drafter(select_published(profiles, rtx5090, "", &published_only)) == "assistant",
                      "without the variant attached gb202 takes the published drafter's profile");
    failures += check(drafter(select_published(profiles, rtx4090, "", &published_only)) == "assistant",
                      "without the variant attached ad102 falls back to *");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_config();
    failures += test_fingerprint();
    failures += test_watchdog();
    failures += test_select();
    failures += test_served();
    failures += test_fit_fallback();
    failures += test_load();
    failures += test_load_model();
    failures += test_select_published();
    failures += test_registry_and_corpus();
    failures += test_variants();
    failures += test_resolve();
    if (failures == 0) { std::cout << "spec: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
