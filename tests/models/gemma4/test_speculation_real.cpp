// Speculative rounds on the real Gemma 4 26B-A4B target with both drafters: the bookkeeping a
// round must keep after any acceptance, checked from outside the session. NINFERNO_TEST_GEMMA4_DIR
// names the directory that holds the artifacts; without it the test skips.
//
//   * every round opens where the previous one left: position P + accepted + 1;
//   * the emitted stream is the accepted drafts followed by the target's token at the first
//     disagreement (the bonus token after a fully accepted round);
//   * a continuation split over several calls equals one call (the pending token and the MTP
//     carry survive the boundary), and a traced continuation equals an untraced one;
//   * the adaptive depth schedule does not leak from one sequence into the next, and A, B, A in
//     one session drafts the second A exactly as the first;
//   * speculate_until meets a token budget exactly and stops right after a stop id that arrives
//     among a round's accepted drafts;
//   * the DFlash drafter's block near the end of the context budget drafts what it drafts with
//     room to spare (its keys never wrap onto committed context);
//   * at depth 1 every emitted token is the one plain greedy decode picks on the same prefix, up
//     to a near-tie of the plain logits (less than 0.5 on the raw logit scale);
//   * a session whose graphs warm_up() captured at load emits exactly what one that captures
//     them on first use emits, plain and speculative, also after a prompt shorter than the
//     warm-up's widest round.

#include "core/device.h"
#include "models/gemma4/model.h"
#include "models/gemma4/session.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

// A GSM8K question and a prose request in the Gemma 4 chat template, thinking off.
const std::vector<std::int32_t> kMath = {
    2,      105,    2364,   107,    209885, 236858, 236751, 64674,  6267,   236743, 236770, 236825,
    14318,  810,    1719,   236761, 2625,   55936,  1806,   573,    13932,  1418,   5597,   532,
    518,    1950,   90587,  573,    1116,   4690,   1418,   1719,   607,    2390,   236761, 2625,
    35025,  506,    24232,  657,    506,    14369,  236789, 2436,   6376,   573,    609,    236778,
    810,    5756,   29989,  8727,   236761, 2088,   1623,   528,    11092,  1677,   1304,   1386,
    1418,   1719,   657,    506,    14369,  236789, 2436,   236881, 61973,  2918,   684,    2918,
    532,    2583,   506,    1626,   3890,   1308,   113887, 236761, 106,    107,    105,    4368,
    107,    100,    45518,  107,    101};
const std::vector<std::int32_t> kProse = {
    2,      105,    2364,   107,    6974,   496,    236743, 236812, 236771, 236771, 236772, 3017,
    2822,   3925,   1003,   496,    63937,  64222,  1015,   15249,  496,    3618,   528,    496,
    11988,  236761, 106,    107,    105,    4368,   107,    100,    45518,  107,    101};

struct Drafters {
    const Drafter* mtp;
    const DFlashDrafter* dflash;
};

SessionOptions options(const Drafters& d, bool dflash, std::int32_t k, std::int32_t k_min = 0,
                       std::int32_t context = 4096) {
    SessionOptions o;
    o.max_context      = context;
    o.prefill_chunk    = 512;
    o.drafter          = dflash ? nullptr : d.mtp;
    o.dflash           = dflash ? d.dflash : nullptr;
    o.draft_tokens     = k;
    o.min_draft_tokens = k_min;
    return o;
}

std::vector<std::int32_t> continuation(Session& s, const std::vector<std::int32_t>& prompt,
                                       const std::vector<std::int32_t>& calls) {
    s.reset();
    s.prefill(prompt);
    std::vector<std::int32_t> out;
    for (const auto rounds : calls) {
        const auto part = s.speculate(rounds);
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

std::string label(bool dflash, std::int32_t k) {
    return std::string(dflash ? "dflash" : "mtp") + " k=" + std::to_string(k);
}

// One call of 24 rounds against three of 8, and a traced call against both; then the trace's
// own account of every round against the stream it emitted.
void check_rounds(const Model& model, DeviceContext& device, const Drafters& d, bool dflash,
                  std::int32_t k, std::int32_t k_min) {
    const auto name = label(dflash, k) + (k_min ? " adaptive" : "");
    Session one(model, device, options(d, dflash, k, k_min));
    const auto whole = continuation(one, kMath, {24});
    Session split(model, device, options(d, dflash, k, k_min));
    const auto parts = continuation(split, kMath, {8, 8, 8});
    check(whole == parts, name + ": a continuation split over three calls equals one call");

    Session traced(model, device, options(d, dflash, k, k_min));
    traced.prefill(kMath);
    std::vector<SpeculationRoundTrace> trace;
    const auto stream = traced.speculate_traced(24, trace, true);
    check(stream == whole, name + ": the traced continuation equals the untraced one");
    check(trace.size() == 24, name + ": one trace record per round");
    const auto prompt = static_cast<std::int32_t>(kMath.size());
    std::int32_t position = prompt;
    for (std::size_t r = 0; r < trace.size(); ++r) {
        const auto& t   = trace[r];
        const auto what = name + " round " + std::to_string(r);
        check(t.position == position, what + ": opens at the previous round's P + accepted + 1");
        check(t.depth >= std::max<std::int32_t>(k_min, 1) && t.depth <= k,
              what + ": depth inside the schedule's range");
        check(static_cast<std::int32_t>(t.target.size()) == t.depth + 1,
              what + ": one target argmax per verified column");
        check(static_cast<std::int32_t>(t.drafts.size()) ==
                  (dflash ? d.dflash->config().block_size - 1 : t.depth),
              what + ": the drafter's whole proposal is recorded");
        check(t.accepted >= 0 && t.accepted <= t.depth, what + ": accepted within the depth");
        check(t.draft_ms > 0 && t.forward_ms > 0 && t.head_ms > 0 && t.commit_ms > 0,
              what + ": every phase of a profiled round takes device time");
        // stream[i] holds position prompt + i: the pending token, then what the round emitted.
        const auto base = static_cast<std::size_t>(t.position - prompt);
        if (base + t.accepted + 1 >= stream.size()) { break; }
        bool drafts_emitted = true;
        for (std::int32_t i = 0; i < t.accepted; ++i) {
            drafts_emitted &= stream[base + 1 + i] == t.drafts[i];
            drafts_emitted &= t.target[i] == t.drafts[i];
        }
        check(drafts_emitted, what + ": the accepted drafts are emitted in order");
        check(t.accepted == t.depth || t.drafts[t.accepted] != t.target[t.accepted],
              what + ": acceptance stops at the first disagreement");
        check(stream[base + t.accepted + 1] == t.target[t.accepted],
              what + ": the target's token follows the accepted drafts");
        position = t.position + t.accepted + 1;
    }
}

// The adaptive schedule's depth after one sequence must not steer the next one's rounds.
void check_schedule_reset(const Model& model, DeviceContext& device, const Drafters& d) {
    // The depth of every round is compared, not only the text: the text of this prompt does not
    // depend on how its rounds are cut.
    const auto depths = [&](Session& s) {
        s.reset();
        s.prefill(kMath);
        std::vector<SpeculationRoundTrace> trace;
        const auto stream = s.speculate_traced(20, trace, false);
        std::vector<std::int32_t> out(stream);
        for (const auto& t : trace) { out.push_back(-t.depth); }
        return out;
    };
    Session used(model, device, options(d, false, 8, 2));
    (void)continuation(used, kProse, {12});
    const auto after_prose = depths(used);
    Session fresh(model, device, options(d, false, 8, 2));
    const auto from_scratch = depths(fresh);
    check(after_prose == from_scratch,
          "adaptive mtp: a sequence runs the rounds it would run in a fresh session");
}

// DFlash drafts a 16-wide block whatever its depth: close to the end of a session's context the
// block's tail lies past the last position the session admits. With a budget just large enough
// for the rounds, every round must draft exactly what it drafts with room to spare.
void check_context_edge(const Model& model, DeviceContext& device, const Drafters& d) {
    // A budget exactly filled by the rounds, large enough that every sliding ring of the target
    // and of the drafter holds its window plus the prompt (the attention ops' own requirement).
    const std::int32_t k      = 3;
    const auto prompt         = static_cast<std::int32_t>(kMath.size());
    const std::int32_t rounds = 520;
    const auto tight          = prompt + rounds * (k + 1) + 1;
    std::vector<SpeculationRoundTrace> edge, roomy;
    Session a(model, device, options(d, true, k, 0, tight));
    a.prefill(kMath);
    const auto stream_a = a.speculate_traced(rounds, edge, false);
    Session b(model, device, options(d, true, k, 0, 4096));
    b.prefill(kMath);
    const auto stream_b = b.speculate_traced(rounds, roomy, false);
    check(stream_a == stream_b, "dflash at the context edge: the same continuation");
    bool same = edge.size() == roomy.size();
    for (std::size_t r = 0; same && r < edge.size(); ++r) {
        same = edge[r].drafts == roomy[r].drafts;
    }
    check(same, "dflash at the context edge: every round drafts what it drafts with room");
}

// A token budget is met exactly, and a stop id that arrives among a round's accepted drafts ends
// the continuation right after it.
void check_stop_and_budget(const Model& model, DeviceContext& device, const Drafters& d,
                           bool dflash, std::int32_t k) {
    const auto name = label(dflash, k);
    Session s(model, device, options(d, dflash, k));
    s.prefill(kMath);
    std::vector<SpeculationRoundTrace> trace;
    const auto stream = s.speculate_traced(30, trace, false);
    const auto prompt = static_cast<std::int32_t>(kMath.size());

    s.reset();
    s.prefill(kMath);
    const auto budget = s.speculate_until(37);
    check(budget.size() == 37 && std::equal(budget.begin(), budget.end(), stream.begin()),
          name + ": a 37-token budget returns exactly the stream's first 37 tokens");
    check(s.position() == prompt + 37, name + ": the session ends at the budget");

    // The first token that is an accepted draft in the middle of its round, and appears nowhere
    // earlier in the stream: a stop id there must end the continuation right after it.
    std::int32_t stop_at = -1;
    for (const auto& t : trace) {
        if (t.accepted < 2) { continue; }
        const auto index = t.position - prompt + 1; // the round's first accepted draft
        if (index <= 0 || index >= static_cast<std::int32_t>(stream.size())) { continue; }
        if (std::find(stream.begin(), stream.begin() + index, stream[index]) ==
            stream.begin() + index) {
            stop_at = index;
            break;
        }
    }
    check(stop_at > 0, name + ": the stream has an accepted draft to stop at");
    if (stop_at <= 0) { return; }
    s.reset();
    s.prefill(kMath);
    const std::vector<std::int32_t> stops{stream[stop_at]};
    const auto stopped = s.speculate_until(200, stops);
    check(static_cast<std::int32_t>(stopped.size()) == stop_at + 1 &&
              std::equal(stopped.begin(), stopped.end(), stream.begin()),
          name + ": a stop id among accepted drafts ends the continuation right after it");
    check(s.position() == prompt + stop_at + 1, name + ": the session ends at the stop");
}

// A, then B, then A again in one session: the second A must run exactly the rounds of the first
// (the drafter keeps nothing from the sequence in between).
void check_aba(const Model& model, DeviceContext& device, const Drafters& d, bool dflash,
               std::int32_t k, std::int32_t k_min) {
    const auto name = label(dflash, k) + (k_min ? " adaptive" : "");
    Session s(model, device, options(d, dflash, k, k_min));
    const auto run = [&](const std::vector<std::int32_t>& prompt) {
        s.reset();
        s.prefill(prompt);
        std::vector<SpeculationRoundTrace> trace;
        auto out = s.speculate_traced(20, trace, false);
        for (const auto& t : trace) {
            out.push_back(-1 - t.depth);
            out.insert(out.end(), t.drafts.begin(), t.drafts.end());
        }
        return out;
    };
    const auto first = run(kMath);
    (void)run(kProse);
    const auto again = run(kMath);
    check(first == again, name + ": A, B, A: the second A drafts and emits what the first did");
}

// Largest logit and its index.
std::pair<float, std::int32_t> best(const std::vector<float>& logits) {
    const auto it = std::max_element(logits.begin(), logits.end());
    return {*it, static_cast<std::int32_t>(it - logits.begin())};
}

// Depth 1: each emitted token against plain greedy decode teacher-forced on the same prefix.
void check_depth_one(const Model& model, DeviceContext& device, const Drafters& d, bool dflash) {
    const auto name = label(dflash, 1);
    Session spec(model, device, options(d, dflash, 1));
    const auto stream = continuation(spec, kMath, {40});
    SessionOptions plain_options = options(d, false, 1);
    plain_options.drafter        = nullptr;
    Session plain(model, device, plain_options);
    plain.prefill(kMath);
    const auto cap       = model.config().softcap;
    const auto raw       = [&](float v) { return cap ? *cap * std::atanh(v / *cap) : v; };
    double worst         = 0;
    std::size_t differed = 0;
    for (std::size_t i = 0; i + 1 < stream.size(); ++i) {
        plain.set_pending_token(stream[i]);
        (void)plain.decode(1);
        const auto logits          = plain.last_logits();
        const auto [value, choice] = best(logits);
        if (choice != stream[i + 1]) {
            ++differed;
            worst = std::max(worst, double(raw(value)) - raw(logits[stream[i + 1]]));
        }
    }
    check(stream.size() > 40, name + ": forty rounds emit more than forty tokens");
    check(worst < 0.5, name + ": every emitted token is plain decode's up to a near-tie (worst " +
                           std::to_string(worst) + " over " + std::to_string(differed) + ")");
}

// warm_up() at load against capture on first use: the same tokens, for plain decode and for
// speculation, after a full prompt and after one shorter than the deepest round's k + 1 columns
// (the warm-up's keys past the prompt are rewritten before any round reads them).
void check_warm_up(const Model& model, DeviceContext& device, const Drafters& d, bool dflash,
                   std::int32_t k, std::int32_t k_min) {
    const auto name = label(dflash, k) + (k_min ? " adaptive" : "") + " warm_up";
    const std::vector<std::int32_t> short_prompt(kMath.begin(), kMath.begin() + 6);
    for (const auto* prompt : {&kMath, &short_prompt}) {
        const auto what = name + (prompt == &kMath ? "" : " (short prompt)");
        Session cold(model, device, options(d, dflash, k, k_min));
        const auto expected = continuation(cold, *prompt, {12, 12});
        Session warm(model, device, options(d, dflash, k, k_min));
        warm.warm_up();
        check(continuation(warm, *prompt, {12, 12}) == expected,
              what + ": speculation after warm_up emits what first-use capture does");
    }
    SessionOptions plain_options = options(d, false, 1);
    plain_options.drafter        = nullptr;
    Session cold(model, device, plain_options);
    cold.prefill(short_prompt);
    const auto expected = cold.decode(48);
    Session warm(model, device, plain_options);
    warm.warm_up();
    warm.prefill(short_prompt);
    check(warm.decode(48) == expected, name + ": plain decode after warm_up emits the same");
    bool refused = false;
    try {
        warm.warm_up();
    } catch (const std::logic_error&) {
        refused = true;
    }
    check(refused, name + ": warm_up after a prefill is refused");
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
        const auto dflash = DFlashDrafter::load(artifacts / "gemma4_26b_dflash_q8.ninferno", device,
                                                target->config());
        const Drafters d{mtp.get(), dflash.get()};
        // Each check runs on its own: one that throws does not hide the others.
        const auto run = [&](const std::string& name, const auto& body) {
            try {
                body();
            } catch (const std::exception& error) {
                check(false, name + " threw: " + error.what());
            }
        };
        run("mtp k=4", [&] { check_rounds(*target, device, d, false, 4, 0); });
        run("mtp adaptive", [&] { check_rounds(*target, device, d, false, 8, 2); });
        run("dflash k=7", [&] { check_rounds(*target, device, d, true, 7, 0); });
        run("dflash k=15", [&] { check_rounds(*target, device, d, true, 15, 0); });
        run("schedule reset", [&] { check_schedule_reset(*target, device, d); });
        run("context edge", [&] { check_context_edge(*target, device, d); });
        run("stop mtp", [&] { check_stop_and_budget(*target, device, d, false, 4); });
        run("stop dflash", [&] { check_stop_and_budget(*target, device, d, true, 7); });
        run("aba mtp", [&] { check_aba(*target, device, d, false, 4, 0); });
        run("aba mtp adaptive", [&] { check_aba(*target, device, d, false, 8, 2); });
        run("aba dflash", [&] { check_aba(*target, device, d, true, 7, 0); });
        run("mtp k=1", [&] { check_depth_one(*target, device, d, false); });
        run("dflash k=1", [&] { check_depth_one(*target, device, d, true); });
        run("warm_up mtp adaptive", [&] { check_warm_up(*target, device, d, false, 15, 2); });
        run("warm_up dflash", [&] { check_warm_up(*target, device, d, true, 15, 0); });
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "gemma4 real speculation: round bookkeeping, schedule reset, context edge, "
                 "depth 1 and warm-up capture hold for MTP and DFlash\n";
    return 0;
}
