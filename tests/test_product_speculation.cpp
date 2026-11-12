// The drafter a product finds for an explicit --spec without --drafter: the separate drafter
// artifact named in <artifact>.speculation.json whose kind drafts through that backend
// (product::model_drafter). No GPU; only the manifest is read. Also where load bounds move when
// their own graphs do not fit (product::snap_load_to_captured), and the pool rule's price of the
// MTP head over the benchmark's wave (product::pool_rule, #777) with its properties.

#include "models/qwen3_5/program/planning/copy_proposal.h"
#include "models/qwen3_5/program/planning/speculation_load.h"
#include "product/choice/choice_constants.h"
#include "product/speculation/speculation.h"
#include "spec/pairs.h"
#include "spec/profile.h"
#include "spec/registry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// A manifest beside `artifact` with one attached separate drafter of `kind` at `drafter`.
void write_manifest(const std::filesystem::path& artifact, const std::string& kind,
                    const std::string& drafter, const std::string& status = "attached") {
    const nlohmann::json doc = {
        {"schema", 1},
        {"target", {{"artifact_id", "0"}, {"engine", "nemotron_h"}}},
        {"source", "none"},
        {"drafters",
         nlohmann::json::array({{{"name", "mtp"},
                                 {"kind", kind},
                                 {"artifact", drafter},
                                 {"status", status}}})},
        {"profiles", nlohmann::json::array()}};
    std::ofstream(ninferno::spec::manifest_path(artifact)) << doc.dump(1) << '\n';
}

} // namespace

int main() {
    using ninferno::SpeculativeBackend;
    using ninferno::product::model_drafter;
    int failures   = 0;
    const auto dir =
        std::filesystem::temp_directory_path() /
        ("ninferno-product-speculation-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    const auto target = dir / "target.ninferno";
    const auto mtp    = [&] { return model_drafter(target, SpeculativeBackend::Mtp); };
    const auto dflash = [&] { return model_drafter(target, SpeculativeBackend::DFlash); };

    failures += check(mtp().empty(), "no manifest: no drafter");
    // The head a backend drafts with (model_draft_head): MTP and DFlash take the artifact's head
    // when it serves them; DFlash2 keeps what it was asked for; an asked-for head stays.
    {
        using ninferno::ProposalHead;
        using ninferno::product::model_draft_head;
        failures += check(model_draft_head(true, SpeculativeBackend::DFlash, ProposalHead::Full) ==
                              ProposalHead::Optimized,
                          "DFlash drafts with a head that serves it");
        failures += check(model_draft_head(false, SpeculativeBackend::DFlash, ProposalHead::Full) ==
                              ProposalHead::Full,
                          "DFlash keeps the full head without one");
        failures += check(model_draft_head(true, SpeculativeBackend::Mtp, ProposalHead::Full) ==
                              ProposalHead::Optimized,
                          "MTP drafts with a head that serves it");
        failures += check(model_draft_head(false, SpeculativeBackend::Mtp, ProposalHead::Full) ==
                              ProposalHead::Full,
                          "MTP keeps the full head without one");
        failures += check(model_draft_head(true, SpeculativeBackend::DFlash2, ProposalHead::Full) ==
                              ProposalHead::Full,
                          "DFlash2 keeps the full head");
        failures += check(model_draft_head(false, SpeculativeBackend::Mtp, ProposalHead::Optimized) ==
                              ProposalHead::Optimized,
                          "an asked-for head stays");
        failures += check(model_draft_head(target, SpeculativeBackend::DFlash,
                                           std::filesystem::path("drafter.ninferno"),
                                           ProposalHead::Full) == ProposalHead::Full,
                          "a separate DFlash drafter keeps the full head");
    }
    // A "+head" choice of a drafter binds the head only through that drafter's Use (Codex on
    // #170: DFlash + DFlash2 with a DFlash-only head chose dflash2:7+head).
    {
        using ninferno::product::proposal_head_drafters;
        namespace art = ninferno::artifact;
        art::Directory dir;
        dir.components["text"].proposal = art::Proposal{.indexed = true, .rows = 147456};
        dir.components["dflash"];
        dir.components["dflash2"];
        dir.components["mtp"];
        const auto use = [&](const std::string& input) {
            art::Use u;
            u.parameter = "proposal/head";
            u.input     = input;
            dir.uses.emplace(std::pair{u.parameter, u.input}, u);
        };
        use("dflash/final_hidden");
        failures += check(proposal_head_drafters(dir) == std::set<std::string>{"dflash"},
                          "a DFlash-only head serves DFlash alone");
        use("mtp/final_hidden");
        failures += check(proposal_head_drafters(dir) == std::set<std::string>{"dflash", "mtp"},
                          "the default head serves MTP and DFlash, not DFlash2");
        dir.components["text"].proposal.reset();
        failures += check(proposal_head_drafters(dir).empty(), "no proposal: no head");
    }
    // One mode of a published profile (no measurements): its fit fallback of that mode becomes its
    // choice, and the load table measured with the default's drafter gives way to one request at a
    // time (Codex on #117).
    {
        ninferno::product::SpeculationSubject subject;
        subject.support = ninferno::spec::engine_support("qwen3_5_text");
        ninferno::spec::ResolvedSpeculation resolved;
        resolved.source                       = "test";
        resolved.profile.overall.config       = ninferno::spec::SpecConfig::parse("dflash2:7");
        resolved.profile.thinking             = resolved.profile.overall;
        resolved.profile.non_thinking         = resolved.profile.overall;
        ninferno::spec::Choice fallback;
        fallback.config                       = ninferno::spec::SpecConfig::parse("mtp:3");
        resolved.profile.fit_fallback         = fallback;
        resolved.profile.load                 = {ninferno::spec::LoadBound{.lanes = 4, .depth = 3}};
        subject.resolved                      = resolved;
        const auto mtp = ninferno::product::subject_of_mode(subject, ninferno::spec::Mode::Mtp);
        failures += check(mtp.resolved->profile.overall.config.label() == "mtp:3" &&
                              !mtp.resolved->profile.fit_fallback &&
                              mtp.resolved->profile.load.size() == 1 &&
                              mtp.resolved->profile.load[0].lanes == 2 &&
                              mtp.resolved->profile.load[0].depth == 0,
                          "a published profile's MTP fit fallback as its MTP choice, one request at a time");
        const auto same = ninferno::product::subject_of_mode(subject, ninferno::spec::Mode::DFlash2);
        failures += check(same.resolved->profile.overall.config.label() == "dflash2:7" &&
                              same.resolved->profile.load.size() == 1 &&
                              same.resolved->profile.load[0].lanes == 4,
                          "its own mode: kept as it is");
    }
    // A load table derived from the facts (Profile::load_basis "facts") is derived again for the
    // choices one mode's reselection makes: the old table bounded the choices before it.
    {
        namespace sp = ninferno::spec;
        // Seconds per 100 tokens of each configuration on math, code, prose and thinking (plain
        // decode 1 s on each).
        const auto measured =
            [](std::initializer_list<std::pair<const char*, std::vector<double>>> rows) {
                std::vector<sp::Measurement> out;
                for (const char* family : {"math", "code", "prose", "thinking"}) {
                    out.push_back({.family = family, .tokens = 100, .seconds = 1.0});
                }
                for (const auto& [label, seconds] : rows) {
                    std::size_t i = 0;
                    for (const char* family : {"math", "code", "prose", "thinking"}) {
                        out.push_back({.config  = sp::SpecConfig::parse(label),
                                       .family  = family,
                                       .tokens  = 100,
                                       .seconds = seconds[i++]});
                    }
                }
                return out;
            };
        const auto subject_of = [](const sp::Profile& profile, const char* model_type) {
            ninferno::product::SpeculationSubject subject;
            subject.support = sp::engine_support(model_type);
            ninferno::spec::ResolvedSpeculation resolved;
            resolved.source  = "test";
            resolved.profile = profile;
            subject.resolved = resolved;
            return subject;
        };
        // (i) The same mode: overall and non-thinking mtp:3, thinking dflash:7, table 2:7,3:0.
        // MTP alone reselects the thinking choice as mtp:2, which loses at two streams
        // (1.667 x 0.4); every MTP choice gains at depth 1 (x 0.9), so the table is 2:1,3:0.
        {
            sp::Profile p;
            p.measurements = measured({{"mtp:3", {0.4, 0.4, 0.4, 0.9}},
                                       {"mtp:2", {0.5, 0.5, 0.5, 0.6}},
                                       {"mtp:1", {0.7, 0.7, 0.7, 0.7}},
                                       {"dflash:7", {0.95, 0.95, 0.95, 0.3}}});
            sp::select(p, {.served = sp::engine_support("qwen3_5_text")});
            p.joint_scale = {{"mtp:1", {{2, 0.9}}},
                             {"mtp:2", {{2, 0.4}}},
                             {"mtp:3", {{2, 0.6}}},
                             {"dflash:7", {{2, 0.8}}}};
            p.load        = sp::facts_load(p);
            p.load_basis  = "facts";
            failures += check(p.overall.config.label() == "mtp:3" &&
                                  p.thinking.config.label() == "dflash:7" &&
                                  sp::load_label(p.load) == "2:7,3:0",
                              "the profile: mtp:3 by default, dflash:7 thinking, table " +
                                  sp::load_label(p.load));
            const auto mtp =
                ninferno::product::subject_of_mode(subject_of(p, "qwen3_5_text"), sp::Mode::Mtp);
            const sp::Profile& q = mtp.resolved->profile;
            failures += check(q.thinking.config.label() == "mtp:2" &&
                                  sp::load_label(q.load) == "2:1,3:0" && q.load_basis == "facts",
                              "MTP alone: thinking " + q.thinking.config.label() +
                                  ", the table derived again: " + sp::load_label(q.load));
        }
        // (ii) Another mode than the profile's own choice: Gemma 4's Engine runs MTP only, so a
        // profile choosing dflash:7 serves mtp:3; its table (from the served choice, 2:3,3:0)
        // stays derived when MTP is chosen, not the flat 2:0 of a table measured with another
        // drafter.
        {
            sp::Profile p;
            p.measurements =
                measured({{"mtp:3", {0.4, 0.4, 0.4, 0.5}}, {"dflash:7", {0.2, 0.2, 0.2, 0.2}}});
            sp::select(p, {.served = sp::engine_support("gemma4_text")});
            p.joint_scale = {{"mtp:3", {{2, 0.6}}}, {"dflash:7", {{2, 0.8}}}};
            p.load        = sp::facts_load(p);
            p.load_basis  = "facts";
            failures +=
                check(p.overall.config.label() == "dflash:7" && p.served &&
                          p.served->overall.config.label() == "mtp:3" &&
                          sp::load_label(p.load) == "2:3,3:0",
                      "the profile: dflash:7, served mtp:3, table " + sp::load_label(p.load));
            const auto mtp =
                ninferno::product::subject_of_mode(subject_of(p, "gemma4_text"), sp::Mode::Mtp);
            const sp::Profile& q = mtp.resolved->profile;
            failures += check(q.overall.config.label() == "mtp:3" &&
                                  sp::load_label(q.load) == "2:3,3:0" && q.load_basis == "facts",
                              "MTP chosen: the served choice's derived table stays, " +
                                  sp::load_label(q.load) + " (" + q.load_basis + ")");
        }
    }
    // Every step that serves a profile carries its table's basis (the plan reports it): the
    // profile's choice, its fit fallback and a shallower draft alike.
    {
        ninferno::product::SpeculationSubject subject;
        subject.support = ninferno::spec::engine_support("qwen3_5_text");
        ninferno::spec::ResolvedSpeculation resolved;
        resolved.source                  = "test";
        resolved.profile.overall.config  = ninferno::spec::SpecConfig::parse("mtp:3");
        resolved.profile.overall.drafter = "mtp";
        resolved.profile.thinking        = resolved.profile.overall;
        resolved.profile.non_thinking    = resolved.profile.overall;
        ninferno::spec::Choice fallback  = resolved.profile.overall;
        fallback.config                  = ninferno::spec::SpecConfig::parse("mtp:1");
        resolved.profile.fit_fallback    = fallback;
        resolved.profile.load            = {ninferno::spec::LoadBound{.lanes = 2, .depth = 1}};
        resolved.profile.load_basis      = "facts";
        resolved.drafters["mtp"]         = "component:mtp";
        subject.components               = {"text", "mtp"};
        subject.resolved                 = resolved;
        const auto served = ninferno::product::served_speculation(subject, false, false);
        failures += check(served && served->drafts() && served->load_basis == "facts",
                          "the profile's choice carries its table's basis");
        if (served) {
            using ninferno::product::FallbackStep;
            // The shallower draft steps down from the fit fallback where there is one (mtp:1 has
            // nothing shallower), so it is asked of the profile without it (from mtp:3).
            auto without_fallback = subject;
            without_fallback.resolved->profile.fit_fallback.reset();
            for (const FallbackStep wanted : {FallbackStep::FitFallback, FallbackStep::Shallower}) {
                std::string basis = "unserved";
                const auto taken  = ninferno::product::fit_fallbacks(
                    wanted == FallbackStep::Shallower ? without_fallback : subject, *served, false,
                    [&](const ninferno::product::ServedSpeculation& candidate, FallbackStep at) {
                        if (at != wanted) { return false; }
                        basis = candidate.load_basis;
                        return true;
                    });
                failures += check(taken && basis == "facts" && taken->load_basis == "facts",
                                  std::string(ninferno::product::fallback_step_name(wanted)) +
                                      " carries the profile's table basis: " + basis);
            }
        }
    }
    // A step taken for memory runs under a load table checked for what it runs (#713): a
    // shallower depth, the fit fallback and the model's own MTP head.
    {
        namespace sp = ninferno::spec;
        using ninferno::product::shallower_profile;
        // Each configuration's seconds per 100 tokens on every section (plain decode 1 s).
        const auto measured = [](std::initializer_list<std::pair<const char*, double>> rows) {
            std::vector<sp::Measurement> out;
            for (const char* family : {"math", "code", "prose", "thinking"}) {
                out.push_back({.family = family, .tokens = 100, .seconds = 1.0});
                for (const auto& [label, seconds] : rows) {
                    out.push_back({.config  = sp::SpecConfig::parse(label),
                                   .family  = family,
                                   .tokens  = 100,
                                   .seconds = seconds});
                }
            }
            return out;
        };
        const auto chosen = [](const char* label, const char* drafter, double gain) {
            sp::Choice c;
            c.config      = sp::SpecConfig::parse(label);
            c.drafter     = drafter;
            c.tok_s       = gain;
            c.plain_tok_s = 1.0;
            return c;
        };
        const auto subject_of = [](const sp::Profile& profile) {
            ninferno::product::SpeculationSubject subject;
            subject.support = sp::engine_support("qwen3_5_text");
            sp::ResolvedSpeculation resolved;
            resolved.source              = "test";
            resolved.profile             = profile;
            resolved.drafters["mtp"]     = "component:mtp";
            resolved.drafters["dflash"]  = "component:dflash";
            resolved.drafters["dflash2"] = "component:dflash2";
            subject.components           = {"text", "mtp", "dflash", "dflash2"};
            subject.resolved             = resolved;
            return subject;
        };
        // The engine's bounds as lanes:draft tokens.
        const auto engine_label = [](const ninferno::product::ServedSpeculation& served) {
            std::string out;
            for (const auto& bound : served.engine.load) {
                out += (out.empty() ? "" : ",") + std::to_string(bound.lanes) + ":" +
                       std::to_string(bound.draft_tokens);
            }
            return out;
        };
        // (i) A shallower depth of a table derived from the facts: derived again at that depth's
        // own one-request gain. mtp:3 gains x2 (x 0.6 = 1.2 at two streams), mtp:2 x1.2, mtp:1
        // x1.05 (x 0.9 < 1): the default's table is 2:3,3:0. At R_1 / R_2 = 0.7 mtp:2 loses (0.84)
        // and runs plainly from two streams, where it ran under the default's 3; at 0.9 it gains
        // (1.08) and keeps depth 2 there.
        {
            sp::Profile p;
            p.overall = p.thinking = p.non_thinking = chosen("mtp:3", "mtp", 2.0);
            p.measurements  = measured({{"mtp:2", 1.0 / 1.2}, {"mtp:1", 1.0 / 1.05}});
            p.joint_scale   = {{"mtp:3", {{2, 0.6}}}, {"mtp:2", {{2, 0.7}}}, {"mtp:1", {{2, 0.9}}}};
            p.load          = sp::facts_load(p);
            p.load_basis    = "facts";
            const auto mtp3 = sp::SpecConfig::parse("mtp:3");
            failures += check(sp::load_label(p.load) == "2:3,3:0",
                              "the facts table of mtp:3: " + sp::load_label(p.load));
            const auto losing = shallower_profile(p, mtp3, 1.5);
            failures += check(losing && sp::load_label(losing->load) == "2:0" &&
                                  losing->load_basis == "facts",
                              "a shallower mtp:2 that loses at two streams decodes plainly there, "
                              "not under its parent's 3: " +
                                  (losing ? sp::load_label(losing->load) : std::string("none")));
            p.joint_scale["mtp:2"] = {{2, 0.9}};
            const auto gaining     = shallower_profile(p, mtp3, 1.5);
            failures += check(gaining && sp::load_label(gaining->load) == "2:2,3:0",
                              "a shallower mtp:2 that gains at two streams keeps depth 2 there: " +
                                  (gaining ? sp::load_label(gaining->load) : std::string("none")));
        }
        // (ii) A measured table (or a literal): a bound stays while the shallower depth runs
        // under it the depth the table's choices ran; from a bound of its parent's depth or
        // deeper nothing showed it to gain, and it decodes plainly from there. A table so cut is a
        // literal; one kept whole keeps its basis.
        {
            sp::Profile p;
            p.overall = p.thinking = p.non_thinking = chosen("mtp:3", "mtp", 2.0);
            p.load_basis                            = "measured";
            const auto shallower_load               = [&](const char* table, const char* config) {
                p.load           = sp::parse_load(table);
                const auto lower = shallower_profile(p, sp::SpecConfig::parse(config), 1.5);
                return lower ? sp::load_label(lower->load) + "/" + lower->load_basis
                                           : std::string("none");
            };
            failures += check(shallower_load("2:3,4:2,6:0", "mtp:3") == "2:0/literal",
                              "mtp:2 under a measured 2:3,4:2,6:0: plain from two streams, " +
                                  shallower_load("2:3,4:2,6:0", "mtp:3"));
            failures += check(shallower_load("2:2,4:1", "mtp:3") == "2:2,4:1/measured",
                              "mtp:2 under a measured 2:2,4:1 runs what it showed: " +
                                  shallower_load("2:2,4:1", "mtp:3"));
            failures += check(shallower_load("2:2,4:1", "mtp:2") == "2:0/literal",
                              "mtp:1 under a measured 2:2,4:1: plain from two streams, " +
                                  shallower_load("2:2,4:1", "mtp:2"));
            failures += check(shallower_load("2:2,5:3", "mtp:3") == "2:2,5:0/literal",
                              "a rising 2:2,5:3 is cut at its first unchecked bound: " +
                                  shallower_load("2:2,5:3", "mtp:3"));
            p.load_basis = "literal";
            failures += check(shallower_load("13:0", "mtp:3") == "13:0/literal",
                              "stand_in_load's cap holds for any depth: " +
                                  shallower_load("13:0", "mtp:3"));
        }
        // (iii) The fit fallback runs under a table of its own. From the facts: an mtp:1 at x1.2
        // (x 0.95 and x 0.9: 1.14 and 1.08) drafts at two and three streams and is unpriced at
        // four, whatever the default's 2:3,3:0.
        {
            sp::Profile p;
            p.overall = p.thinking = p.non_thinking = chosen("mtp:3", "mtp", 2.0);
            p.fit_fallback                          = chosen("mtp:1", "mtp", 1.2);
            p.joint_scale = {{"mtp:3", {{2, 0.6}}}, {"mtp:1", {{2, 0.95}, {3, 0.9}}}};
            p.load        = sp::facts_load(p);
            p.load_basis  = "facts";
            failures += check(sp::load_label(p.load) == "2:3,3:0",
                              "the default's facts table: " + sp::load_label(p.load));
            const auto fallback = ninferno::product::served_fit_fallback(subject_of(p), false);
            failures += check(fallback && engine_label(*fallback) == "2:1,4:0" &&
                                  fallback->load_basis == "facts",
                              "the fit fallback's own facts table: " +
                                  (fallback ? engine_label(*fallback) : std::string("none")));
        }
        // Measured with the default's drafter (DFlash2): another drafter's (MTP) is not known
        // under load, so one request speculates and two or more decode plainly.
        {
            sp::Profile p;
            p.overall = p.thinking = p.non_thinking = chosen("dflash2:7", "dflash2", 2.0);
            p.fit_fallback                          = chosen("mtp:3", "mtp", 1.6);
            p.load                                  = sp::parse_load("2:3");
            p.load_basis                            = "measured";
            const auto fallback = ninferno::product::served_fit_fallback(subject_of(p), false);
            failures += check(fallback && engine_label(*fallback) == "2:0" &&
                                  fallback->load_basis == "literal",
                              "an MTP fit fallback beside a DFlash2 table measured: 2:0, " +
                                  (fallback ? engine_label(*fallback) + "/" + fallback->load_basis
                                            : std::string("none")));
        }
        // (iii b) A shallower depth of the fit fallback's drafter (the default drafts with
        // another): under the fallback's own table, not the default's.
        {
            const auto lower_load = [&](const sp::Profile& p, const char* config) {
                const auto lower = shallower_profile(p, sp::SpecConfig::parse(config), 1.5);
                return lower ? sp::load_label(lower->load) + "/" + lower->load_basis
                             : std::string("no profile");
            };
            // Measured with DFlash2 (2:3): the MTP fallback's own table is 2:0 (another drafter),
            // and its shallower mtp:2 runs under it.
            sp::Profile p;
            p.overall = p.thinking = p.non_thinking = chosen("dflash2:7", "dflash2", 2.0);
            p.fit_fallback                          = chosen("mtp:3", "mtp", 1.6);
            p.load                                  = sp::parse_load("2:3");
            p.load_basis                            = "measured";
            failures += check(lower_load(p, "mtp:3") == "2:0/literal",
                              "the fallback's mtp:2 beside a DFlash2 table measured: " +
                                  lower_load(p, "mtp:3"));
            // No table: the fallback and its shallower depth stay unbounded, as the default.
            p.load.clear();
            failures +=
                check(lower_load(p, "mtp:3") == "none/measured",
                      "no table: the fallback's mtp:2 stays unbounded: " + lower_load(p, "mtp:3"));
            // From the facts. The fallback mtp:3 at x1.5 loses at two streams (x 0.6), mtp:1 at
            // x1.2 gains at two and three (x 0.95, x 0.9) and four is unpriced: the fallback's own
            // table is 2:1,4:0. Its mtp:2 is derived at its own gain: at x1.05 (x 0.9 = 0.945) it
            // loses, and the table is 2:1,4:0 too (at the fallback's x1.5 it would be 2:2,3:1,4:0).
            p.load_basis   = "facts";
            p.measurements = measured({{"mtp:2", 1.0 / 1.05}, {"mtp:1", 1.0 / 1.2}});
            p.joint_scale  = {
                {"mtp:3", {{2, 0.6}}}, {"mtp:2", {{2, 0.9}}}, {"mtp:1", {{2, 0.95}, {3, 0.9}}}};
            p.fit_fallback      = chosen("mtp:3", "mtp", 1.5);
            p.load              = sp::facts_load(p);
            const auto fallback = ninferno::product::served_fit_fallback(subject_of(p), false);
            failures += check(fallback && engine_label(*fallback) == "2:1,4:0",
                              "the fallback mtp:3's own facts table: " +
                                  (fallback ? engine_label(*fallback) : std::string("none")));
            failures += check(lower_load(p, "mtp:3") == "2:1,4:0/facts",
                              "the fallback's mtp:2 from the facts at its own gain: " +
                                  lower_load(p, "mtp:3"));
            // mtp:2 not measured: its gain is unknown, so the fallback's table holds while it runs
            // the depth the fallback ran under it (depth 1 from two streams), kept whole with its
            // basis; under the default's table (DFlash2's: another drafter) it would get 2:0.
            p.measurements = measured({{"mtp:1", 1.0 / 1.2}});
            failures += check(lower_load(p, "mtp:3") == "2:1,4:0/facts",
                              "the fallback's mtp:2 without its gain under the fallback's table: " +
                                  lower_load(p, "mtp:3"));
            // A fallback table the shallower depth would run deeper under is cut there, a
            // literal: the fallback mtp:3 at x2 drafts 3 at two streams (x 0.6 = 1.2), then 1
            // (2:3,3:1,4:0); mtp:2 without its gain would run 2 where the fallback ran 3, which
            // nothing showed to gain, so 2:0.
            p.fit_fallback = chosen("mtp:3", "mtp", 2.0);
            p.load         = sp::facts_load(p);
            failures += check(lower_load(p, "mtp:3") == "2:0/literal",
                              "the fallback's mtp:2 without its gain, its table cut: " +
                                  lower_load(p, "mtp:3"));
        }
        // (iii b') The derivation follows the root profile's basis, not the parent table's label
        // (#760). A facts profile whose default drafts with DFlash2 and whose fit fallback mtp:3
        // has no known gain (no tok_s, not measured): the fallback's table is cut from the
        // default's (another drafter: one request speculates, two or more decode plainly), a
        // literal 2:0. The fallback's shallower mtp:2, measured at x1.2 (x 0.95, x 0.9: 1.14 and
        // 1.08 at two and three streams), is derived from the facts at its own gain, 2:2,4:0;
        // before #760 it was cut again from the fallback's literal (2:0).
        {
            sp::Profile p;
            p.overall = p.thinking = p.non_thinking = chosen("dflash2:7", "dflash2", 2.0);
            sp::Choice no_gain                      = chosen("mtp:3", "mtp", 1.0);
            no_gain.tok_s                           = 0;
            no_gain.plain_tok_s                     = 0;
            p.fit_fallback                          = no_gain;
            p.measurements                          = measured({{"mtp:2", 1.0 / 1.2}});
            p.joint_scale                           = {{"mtp:2", {{2, 0.95}, {3, 0.9}}}};
            p.load                                  = sp::parse_load("2:3");
            p.load_basis                            = "facts";
            const auto fallback = ninferno::product::served_fit_fallback(subject_of(p), false);
            failures += check(fallback && engine_label(*fallback) == "2:0" &&
                                  fallback->load_basis == "literal",
                              "a fit fallback with no known gain, its table cut: " +
                                  (fallback ? engine_label(*fallback) + "/" + fallback->load_basis
                                            : std::string("none")));
            const auto lower = shallower_profile(p, sp::SpecConfig::parse("mtp:3"), 1.5);
            failures += check(lower && sp::load_label(lower->load) == "2:2,4:0" &&
                                  lower->load_basis == "facts",
                              "the no-gain fallback's mtp:2 from the facts at its own gain: " +
                                  (lower ? sp::load_label(lower->load) + "/" + lower->load_basis
                                         : std::string("no profile")));
        }
        // (iii c) A bound measured with the full head shows nothing for a proposal head (another
        // verify cost a round): the shallower mtp:2+head of a fit fallback mtp:3+head beside a
        // default mtp:5 measured at 2:2,4:1 decodes plainly from two streams; the default's own
        // shallower mtp:4 keeps the table.
        {
            sp::Profile p;
            p.overall = p.thinking = p.non_thinking = chosen("mtp:5", "mtp", 2.0);
            p.fit_fallback                          = chosen("mtp:3+head", "mtp", 1.6);
            p.load                                  = sp::parse_load("2:2,4:1");
            p.load_basis                            = "measured";
            const auto lower_load                   = [&](const char* config) {
                const auto lower = shallower_profile(p, sp::SpecConfig::parse(config), 1.5);
                return lower ? lower->overall.config.label() + " " + sp::load_label(lower->load) +
                                   "/" + lower->load_basis
                                               : std::string("no profile");
            };
            failures += check(lower_load("mtp:3+head") == "mtp:2+head 2:0/literal",
                              "a proposal head's shallower depth under a full head's table: " +
                                  lower_load("mtp:3+head"));
            failures +=
                check(lower_load("mtp:5") == "mtp:4 2:2,4:1/measured",
                      "the full head's shallower depth keeps its table: " + lower_load("mtp:5"));
        }
        // (iv) The model's own MTP head: from the facts where the profile prices its mtp:3 (x2,
        // x 0.6 at two streams), else the built-in literal, labelled.
        {
            sp::Profile p;
            p.overall = p.thinking = p.non_thinking = chosen("dflash:7", "dflash", 2.0);
            p.measurements                          = measured({{"mtp:3", 0.5}});
            p.joint_scale                           = {{"mtp:3", {{2, 0.6}}}};
            p.load_basis                            = "facts";
            const auto head = ninferno::product::served_head_fallback(subject_of(p), false);
            failures += check(
                head && engine_label(*head) == "2:3,3:0" && head->load_basis == "facts",
                "the head's table from the facts: " +
                    (head ? engine_label(*head) + "/" + head->load_basis : std::string("none")));
            // A measured table (a lab record at home, which prices its joint verify too) is not
            // derived again: the head keeps the built-in table, as load_for keeps a measured one.
            p.load_basis       = "measured";
            p.load             = sp::parse_load("2:0");
            const auto at_home = ninferno::product::served_head_fallback(subject_of(p), false);
            failures += check(at_home && engine_label(*at_home) == "8:2" &&
                                  at_home->load_basis == "literal",
                              "beside a measured table the head keeps the built-in one: " +
                                  (at_home ? engine_label(*at_home) + "/" + at_home->load_basis
                                           : std::string("none")));
            p.load_basis = "facts";
            p.load.clear();
            p.joint_scale.clear();
            const auto literal = ninferno::product::served_head_fallback(subject_of(p), false);
            failures += check(literal && engine_label(*literal) == "8:2" &&
                                  literal->load_basis == "literal",
                              "the head's built-in table, labelled a literal: " +
                                  (literal ? engine_label(*literal) + "/" + literal->load_basis
                                           : std::string("none")));
        }
        // (v) The pool rule (#777) prices the head's pool against plain decode's over the
        // benchmark's short-chat wave. The profile of (iv): the head's facts table 2:3,3:0, its
        // mtp:3 x2 (x 0.6 = 1.2 at two streams); a plain step of C streams costs 1 + 0.02 (C - 1)
        // one-stream steps.
        {
            namespace pr = ninferno::product;
            sp::Profile p;
            p.overall = p.thinking = p.non_thinking = chosen("dflash:7", "dflash", 2.0);
            p.measurements                          = measured({{"mtp:3", 0.5}});
            p.joint_scale                           = {{"mtp:3", {{2, 0.6}}}};
            p.load_basis                            = "facts";
            for (std::int32_t c = 2; c <= 8; ++c) { p.plain_step_scale[c] = 1.0 + 0.02 * (c - 1); }
            // The benchmark's short chat (DECISIONS 73): 2048 + 512 tokens.
            const std::uint64_t chat = pr::choice::kRequestPrompt + pr::choice::kRequestAnswer;
            // Pools of whole contexts of `context` tokens (by default one chat's: a context a
            // stream).
            const auto rule = [&](const sp::Profile& profile, std::uint32_t with_drafter,
                                  std::uint32_t streams, std::uint32_t plain,
                                  std::optional<std::uint32_t> head, std::uint64_t context = 0) {
                const std::uint64_t tokens = context == 0 ? chat : context;
                return pr::pool_rule(
                    subject_of(profile), false, with_drafter, streams,
                    [&] { return pr::PoolSize{.contexts = plain, .tokens = plain * tokens}; },
                    [&](const pr::ServedSpeculation&) -> std::optional<pr::PoolSize> {
                        if (!head) { return std::nullopt; }
                        return pr::PoolSize{.contexts = *head, .tokens = *head * tokens};
                    });
            };
            failures += check(pr::pool_slots({.contexts = 4, .tokens = 4 * 32768}) == 51 &&
                                  pr::pool_slots({.contexts = 3, .tokens = 3 * 1024}) == 3 &&
                                  pr::pool_slots({}) == 0,
                              "a pool holds its tokens' short chats, at least its contexts");
            failures += check(
                pr::mix_turns(8) == std::vector<std::uint32_t>{8, 8, 4} &&
                    pr::mix_turns(3) == std::vector<std::uint32_t>{3, 3, 3, 3, 3, 3, 2} &&
                    pr::mix_turns(20) == std::vector<std::uint32_t>{20} && pr::mix_turns(0).empty(),
                "the benchmark's wave: twenty prompts in turns of the slots");
            // As many streams as plain decode, five turns of four: the head decodes plainly
            // from three streams, so the waves take the same time, and a tie keeps it.
            const auto same = rule(p, 3, 4, 5, 4);
            failures += check(same.keep == pr::PoolKeep::Head && same.head && same.head_price &&
                                  *same.head_price == 1.0,
                              "the head holding every stream asked: kept at a tie");
            // The issue's case: four of plain decode's five contexts at eight streams, which the
            // four fifths kept, at contexts of one chat. Five turns of four streams decoding
            // plainly (5 x 1.06) against four turns of five (4 x 1.08): plain decode.
            const auto fifths = rule(p, 3, 8, 5, 4);
            failures +=
                check(fifths.keep == pr::PoolKeep::Plain && !fifths.head && fifths.head_price &&
                          std::abs(*fifths.head_price - 4.32 / 5.3) < 1e-12,
                      "four fifths of the contexts for no gain at four streams: plain");
            // The same pools of 32K contexts hold 51 and 64 chats: both serve the eight streams,
            // the same turns, a tie: the head.
            const auto roomy = rule(p, 3, 8, 5, 4, 32768);
            failures += check(roomy.keep == pr::PoolKeep::Head && roomy.head_price &&
                                  *roomy.head_price == 1.0,
                              "four of five 32K contexts hold every chat: the head");
            // One context of two: twenty answers one at a time at x2 (10 steps) against ten
            // turns of two plain streams (10 x 1.02): the head, which four fifths refused.
            const auto alone = rule(p, 1, 2, 2, 1);
            failures += check(alone.keep == pr::PoolKeep::Head && alone.head_price &&
                                  std::abs(*alone.head_price - 1.02) < 1e-12,
                              "one stream at x2 against two plain: the head");
            failures += check(rule(p, 3, 4, 5, std::nullopt).keep == pr::PoolKeep::Plain &&
                                  !rule(p, 3, 4, 5, std::nullopt).head_price,
                              "the head does not fit: plain decode");
            // Nothing priced: the head's built-in table (8:2) drafts three at four streams,
            // where no joint verify is priced. Holding as many streams as plain decode it runs as
            // the built-in default does; a stream it costs is priced by nothing: plain decode.
            sp::Profile unpriced = p;
            unpriced.joint_scale.clear();
            unpriced.plain_step_scale.clear();
            const auto bare = rule(unpriced, 3, 4, 5, 4);
            failures += check(bare.keep == pr::PoolKeep::Head && !bare.head_price,
                              "an unpriced head holding every stream: the head");
            const auto short_of = rule(unpriced, 3, 8, 5, 4);
            failures += check(short_of.keep == pr::PoolKeep::Plain && !short_of.head_price,
                              "an unpriced head a stream short: plain decode");
            failures += check(rule(unpriced, 3, 8, 5, 4, 32768).keep == pr::PoolKeep::Head,
                              "an unpriced head whose 32K contexts hold every chat: the head");
            // A gain priced below 1 at a turn, the step prices unknown: plain decode.
            failures +=
                check(!pr::pool_head_price(
                           2, 2, 2, [](std::uint32_t) { return std::optional<double>(0.9); },
                           [](std::uint32_t c) {
                               return c == 1 ? std::optional<double>(1.0) : std::nullopt;
                           })
                           .pays,
                      "a head priced below plain decode at its turns: refused");
            // The step prices unknown but both pools serving as many streams and the head
            // decoding plainly from two: no turn is slower than plain decode's, kept.
            unpriced.joint_scale = {{"mtp:3", {{2, 0.6}}}};
            unpriced.load_basis  = "facts";
            const auto stepless  = rule(unpriced, 3, 4, 5, 4);
            failures += check(stepless.keep == pr::PoolKeep::Head && !stepless.head_price,
                              "no step prices, the same streams, no turn below plain: the head");
        }
        // (vi) Properties of the pool rule's price over random tables (#777): a head the rule
        // keeps never decodes the benchmark's wave slower than plain decode's pool, run as the
        // benchmark sends it (`wave_rate`: C clients, client k sending prompts k, k + C, ... one
        // after another, a server admitting waiting requests in order while its pool has the KV
        // pages for them, each chat taking its tokens, at most a context's); where both are
        // priced the rule keeps exactly the heads that are no slower, and a head that gains more
        // is never refused where one that gains less is kept.
        {
            namespace pr       = ninferno::product;
            const auto prompts = static_cast<std::uint32_t>(pr::choice::kMixShortRequests);
            const std::uint64_t chat_tokens =
                pr::choice::kRequestPrompt + pr::choice::kRequestAnswer;

            // A pool of `tokens` KV tokens at contexts of `context`.
            struct Pool {
                std::uint64_t context = 0, tokens = 0;
            };

            const auto wave_rate = [&](std::uint32_t clients, Pool pool,
                                       const pr::StreamPrice& rate) -> std::optional<double> {
                const std::uint64_t need = std::min(chat_tokens, pool.context);
                std::uint64_t held       = 0; // the tokens the active chats hold
                std::vector<std::uint32_t> next(clients);
                std::vector<std::uint32_t> waiting; // clients whose request waits, in order
                for (std::uint32_t c = 0; c < clients; ++c) {
                    next[c] = c;
                    if (c < prompts) { waiting.push_back(c); }
                }
                std::vector<std::pair<std::uint32_t, double>> active; // client, answer left
                double time        = 0;
                std::uint32_t done = 0;
                while (done < prompts) {
                    while (held + need <= pool.tokens && !waiting.empty()) {
                        active.emplace_back(waiting.front(), 1.0);
                        waiting.erase(waiting.begin());
                        held += need;
                    }
                    if (active.empty()) { return std::nullopt; } // no chat fits the pool
                    const auto r = rate(static_cast<std::uint32_t>(active.size()));
                    if (!r || !(*r > 0)) { return std::nullopt; }
                    double least = 1.0;
                    for (const auto& a : active) { least = std::min(least, a.second); }
                    time += least / *r;
                    std::vector<std::pair<std::uint32_t, double>> left;
                    for (auto& [client, answer] : active) {
                        answer -= least;
                        if (answer > 1e-12) {
                            left.emplace_back(client, answer);
                            continue;
                        }
                        ++done;
                        held -= need;
                        next[client] += clients;
                        if (next[client] < prompts) { waiting.push_back(client); }
                    }
                    active = std::move(left);
                }
                return double(prompts) / time;
            };
            std::mt19937 rng(777);
            const auto uniform = [&](double a, double b) {
                return std::uniform_real_distribution<double>(a, b)(rng);
            };
            const auto pick = [&](int a, int b) {
                return std::uniform_int_distribution<int>(a, b)(rng);
            };
            // A plain step's price at 1..8 streams: 1 at one, never falling, at most C times.
            const auto step_prices = [&] {
                std::map<std::int32_t, double> out;
                double price = 1.0;
                for (std::int32_t c = 2; c <= 8; ++c) {
                    price  = std::min(double(c), price + uniform(0.0, 0.4));
                    out[c] = price;
                }
                return out;
            };
            const auto price_at = [](const std::map<std::int32_t, double>& prices) {
                return [&prices](std::uint32_t c) -> std::optional<double> {
                    if (c == 1) { return 1.0; }
                    const auto it = prices.find(static_cast<std::int32_t>(c));
                    if (it == prices.end()) { return std::nullopt; }
                    return it->second;
                };
            };
            int kept = 0, refused = 0, stepless = 0, wrong = 0, unsound = 0, rising = 0;
            for (int trial = 0; trial < 4000; ++trial) {
                const int k = pick(1, 5);
                // The head's gains at depths 1..k, measured on every section.
                std::vector<double> seconds;
                for (int d = 1; d <= k; ++d) { seconds.push_back(1.0 / uniform(0.5, 3.0)); }
                const auto profile_of = [&](double faster) {
                    sp::Profile priced;
                    for (const char* family : {"math", "code", "prose", "thinking"}) {
                        priced.measurements.push_back(
                            {.family = family, .tokens = 100, .seconds = 1.0});
                        for (int d = 1; d <= k; ++d) {
                            priced.measurements.push_back(
                                {.config  = sp::SpecConfig::parse("mtp:" + std::to_string(d)),
                                 .family  = family,
                                 .tokens  = 100,
                                 .seconds = seconds[d - 1] / faster});
                        }
                    }
                    return priced;
                };
                sp::Profile priced = profile_of(1.0);
                for (int d = 1; d <= k; ++d) {
                    for (std::int32_t c = 2; c <= 8; ++c) {
                        if (uniform(0, 1) < 0.85) {
                            priced.joint_scale["mtp:" + std::to_string(d)][c] = uniform(0.2, 1.3);
                        }
                    }
                }
                if (uniform(0, 1) < 0.9) { priced.plain_step_scale = step_prices(); }
                // Any table: bounds by ascending lanes from 2, depths never rising, ending
                // anywhere (at plain decode or not).
                pr::ServedSpeculation head;
                head.menu.primary.config = sp::SpecConfig::parse("mtp:" + std::to_string(k));
                std::uint32_t lanes      = static_cast<std::uint32_t>(pick(2, 4));
                int depth                = k;
                for (int bounds = pick(0, 3); bounds > 0 && depth > 0; --bounds) {
                    depth = pick(0, depth);
                    head.engine.load.push_back(
                        {.lanes = lanes, .draft_tokens = static_cast<std::uint32_t>(depth)});
                    lanes += static_cast<std::uint32_t>(pick(1, 3));
                }
                const auto streams = static_cast<std::uint32_t>(pick(2, 8));
                // Pools of whole contexts, and up to a context's tokens beyond them.
                const std::uint64_t contexts[] = {1024, 2048, 2560, 4096, 8192, 32768};
                const std::uint64_t context    = contexts[pick(0, 5)];
                const auto pool_of             = [&](int whole) {
                    return Pool{.context = context,
                                            .tokens  = whole * context +
                                          static_cast<std::uint64_t>(pick(0, int(context) - 1))};
                };
                const Pool head_pool  = pool_of(pick(1, 9));
                const Pool plain_pool = pool_of(pick(1, 10));
                const auto slots_of   = [](const Pool& pool) {
                    return pr::pool_slots(
                        {.contexts = static_cast<std::uint32_t>(pool.tokens / pool.context),
                           .tokens   = pool.tokens});
                };
                const std::uint32_t with   = slots_of(head_pool);
                const std::uint32_t plain  = slots_of(plain_pool);
                const pr::StreamPrice gain = [&](std::uint32_t c) {
                    return pr::pool_head_gain(priced, head, c);
                };
                const pr::StreamPrice step = [&](std::uint32_t c) {
                    return pr::plain_step_price(priced, c);
                };
                const pr::HeadPrice price = pr::pool_head_price(streams, with, plain, gain, step);
                const auto head_rate      = wave_rate(streams, head_pool, [&](std::uint32_t c) {
                    const auto g = gain(c);
                    const auto s = step(c);
                    return g && s ? std::optional<double>(*g / *s) : std::nullopt;
                });
                const auto plain_rate     = wave_rate(streams, plain_pool, [&](std::uint32_t c) {
                    const auto s = step(c);
                    return s ? std::optional<double>(1.0 / *s) : std::nullopt;
                });
                (price.pays ? kept : refused) += 1;
                if (head_rate && plain_rate) {
                    const double ratio = *head_rate / *plain_rate;
                    if (price.pays && ratio < 1 - 1e-12) { ++unsound; }
                    if (std::abs(ratio - 1) > 1e-9 && price.pays != (ratio > 1)) { ++wrong; }
                    if (!price.over_plain || std::abs(*price.over_plain - ratio) > 1e-9) {
                        ++wrong;
                    }
                } else if (price.pays) {
                    // Kept unpriced: both pools serve as many streams, no gain the wave meets is
                    // below 1, and with every one of them priced it holds at any step price.
                    ++stepless;
                    std::set<std::uint32_t> met; // the active streams the head's wave meets
                    (void)wave_rate(streams, head_pool, [&](std::uint32_t c) {
                        met.insert(c);
                        return std::optional<double>(1.0);
                    });
                    std::set<std::uint32_t> met_plain;
                    (void)wave_rate(streams, plain_pool, [&](std::uint32_t c) {
                        met_plain.insert(c);
                        return std::optional<double>(1.0);
                    });
                    if (met != met_plain) { ++unsound; }
                    bool all_priced = true;
                    for (const std::uint32_t c : met) {
                        const auto g = gain(c);
                        if (g && *g < 1.0) { ++unsound; }
                        all_priced = all_priced && g.has_value();
                    }
                    for (int draw = 0; all_priced && draw < 20; ++draw) {
                        const auto prices = step_prices();
                        const auto at     = price_at(prices);
                        const auto h      = wave_rate(streams, head_pool, [&](std::uint32_t c) {
                            const auto g = gain(c);
                            return g ? std::optional<double>(*g / *at(c)) : std::nullopt;
                        });
                        const auto q      = wave_rate(streams, plain_pool, [&](std::uint32_t c) {
                            return std::optional<double>(1.0 / *at(c));
                        });
                        if (!h || !q || *h < *q * (1 - 1e-12)) { ++unsound; }
                    }
                }
                // Monotone in the head's gain: every depth faster by the same factor.
                const sp::Profile faster_profile = [&] {
                    sp::Profile out      = profile_of(uniform(1.0, 2.0));
                    out.joint_scale      = priced.joint_scale;
                    out.plain_step_scale = priced.plain_step_scale;
                    return out;
                }();
                const pr::HeadPrice faster = pr::pool_head_price(
                    streams, with, plain,
                    [&](std::uint32_t c) { return pr::pool_head_gain(faster_profile, head, c); },
                    [&](std::uint32_t c) { return pr::plain_step_price(faster_profile, c); });
                if (price.pays && !faster.pays) { ++rising; }
            }
            std::cout << "pool rule properties: " << kept << " kept (" << stepless << " unpriced), "
                      << refused << " refused\n";
            failures += check(kept > 400 && refused > 400 && stepless > 0,
                              "the random tables reach both answers and the unpriced case");
            failures += check(unsound == 0, "a kept head below plain decode at the mix: " +
                                                std::to_string(unsound));
            failures += check(wrong == 0, "the rule against the wave's own price: " +
                                              std::to_string(wrong) + " disagree");
            failures += check(rising == 0, "a faster head refused where a slower one is kept: " +
                                               std::to_string(rising));
        }
    }
    // Per request, a default that decodes plainly loads the other flag's drafter: those requests
    // speculate, the default's decode plainly (Codex on #117).
    {
        ninferno::product::SpeculationSubject subject;
        subject.support = ninferno::spec::engine_support("qwen3_5_text");
        ninferno::spec::ResolvedSpeculation resolved;
        resolved.source                    = "test";
        resolved.profile.thinking.config     = ninferno::spec::SpecConfig::parse("mtp:4");
        resolved.profile.thinking.drafter    = "mtp";
        resolved.profile.overall           = resolved.profile.thinking;
        resolved.drafters["mtp"]           = "component:mtp";
        subject.components                 = {"text", "mtp"};
        subject.resolved                   = resolved;
        const auto fixed = ninferno::product::served_speculation(subject, false, false);
        const auto each  = ninferno::product::served_speculation(subject, false, true);
        failures += check(fixed && fixed->menu.mode == ninferno::spec::Mode::None,
                          "fixed at the plain default: no drafter");
        failures += check(each && each->menu.mode == ninferno::spec::Mode::Mtp &&
                              each->menu.thinking.config.enabled() &&
                              !each->menu.non_thinking.config.enabled(),
                          "per request: MTP loaded for thinking, the default plain");
    }
    // Nothing shallower than one draft, nothing for plain decode, nothing without a profile.
    {
        using ninferno::product::served_shallower;
        const auto dflash_at = [](std::int32_t k) {
            ninferno::spec::SpecConfig c;
            c.mode = ninferno::spec::Mode::DFlash;
            c.k    = k;
            return c;
        };
        const auto subject = ninferno::product::speculation_subject(target, 0);
        failures += check(!served_shallower(subject, true, dflash_at(1), 1.5) &&
                              !served_shallower(subject, true, ninferno::spec::SpecConfig{}, 0) &&
                              !served_shallower(subject, true, dflash_at(3), 1.5),
                          "served_shallower ran below depth 1, for plain, or without a profile");
    }
    // A server whose choice does not fit runs the deepest depth that fits, with the same drafter:
    // one draft shallower, then depth 1, then halving between them, and the depth it returns is the
    // one built last; nothing when depth 1 does not fit.
    {
        using ninferno::product::deepest_fitting_shallower;
        using ninferno::product::ServedSpeculation;
        std::vector<std::int32_t> offered;
        const auto shallower = [](const ninferno::spec::SpecConfig& config, double floor) {
            std::optional<ServedSpeculation> out;
            if (config.k <= 1) { return out; }
            out.emplace();
            out->menu.primary.config   = config;
            out->menu.primary.config.k = config.k - 1;
            out->menu.primary.floor    = floor;
            return out;
        };
        const auto fits_at_most = [&](std::int32_t depth) {
            return [&offered, depth](const ServedSpeculation& candidate) {
                offered.push_back(candidate.menu.primary.config.k);
                return candidate.menu.primary.config.k <= depth;
            };
        };
        ninferno::spec::MenuChoice tried;
        tried.config.mode = ninferno::spec::Mode::Mtp;
        tried.config.k    = 6;
        tried.floor       = 1.4;

        const auto three = deepest_fitting_shallower(tried, shallower, fits_at_most(3));
        failures += check(three && three->menu.primary.config.k == 3 &&
                              three->menu.primary.config.mode == ninferno::spec::Mode::Mtp &&
                              offered == std::vector<std::int32_t>{5, 1, 3, 4, 3},
                          "deepest_fitting_shallower: 5 no, 1 and 3 fit, 4 no, 3 built again");
        offered.clear();
        const auto five = deepest_fitting_shallower(tried, shallower, fits_at_most(5));
        failures += check(five && five->menu.primary.config.k == 5 &&
                              offered == std::vector<std::int32_t>{5},
                          "deepest_fitting_shallower: a near miss takes one build");
        offered.clear();
        failures += check(!deepest_fitting_shallower(tried, shallower, fits_at_most(0)) &&
                              offered == std::vector<std::int32_t>{5, 1},
                          "deepest_fitting_shallower: depth 1 refused, then nothing");
        // From a pick at depth 16 (mtp:15 + 1): a handful of builds, not one per depth.
        tried.config.k = 16;
        for (std::int32_t fitting = 1; fitting <= 15; ++fitting) {
            offered.clear();
            const auto deepest = deepest_fitting_shallower(tried, shallower, fits_at_most(fitting));
            failures += check(deepest && deepest->menu.primary.config.k == fitting &&
                                  offered.back() == fitting && offered.size() <= 7,
                              "deepest_fitting_shallower from depth 16: depth " +
                                  std::to_string(fitting) + " in at most 7 builds, built last");
        }
    }
    // One draft shallower than a depth that does not fit: the profile's drafter for that mode, and
    // a watchdog floor for the shallower depth from the floors measured around it.
    {
        using ninferno::product::shallower_floor;
        using ninferno::product::shallower_profile;
        const auto near = [](double a, double b) { return a > b - 1e-9 && a < b + 1e-9; };
        failures += check(near(shallower_floor({{2, 1.52}, {7, 2.14}}, 6), 1.52 + 0.8 * 0.62) &&
                              near(shallower_floor({{7, 2.14}, {6, 1.9}}, 6), 1.9) &&
                              near(shallower_floor({{7, 2.14}}, 6), 1.0 + 1.14 * 6.0 / 7.0) &&
                              near(shallower_floor({{2, 1.5}}, 4), 2.0) &&
                              shallower_floor({{7, 0.0}}, 6) == 0.0 &&
                              shallower_floor({{7, 1.0}}, 6) == 0.0 &&
                              near(shallower_floor({{2, 0.9}, {7, 2.14}}, 6),
                                   1.0 + 1.14 * 6.0 / 7.0) &&
                              shallower_floor({}, 6) == 0.0,
                          "shallower_floor: measured, interpolated, scaled, none; floors of 1 or "
                          "less are no measurement");

        // A request that names its own depth (ninferno.draft_tokens): the floor measured at that
        // depth, else interpolated; the menu's shallower depths as its step-down; 0 is plain.
        using ninferno::product::request_speculation_at;
        ninferno::spec::SpeculationMenu menu;
        menu.mode                   = ninferno::spec::Mode::Mtp;
        menu.primary.config         = {.mode = ninferno::spec::Mode::Mtp, .k = 5};
        menu.primary.floor          = 2.4;
        menu.thinking               = menu.primary;
        menu.overall.config         = {.mode = ninferno::spec::Mode::Mtp, .k = 2};
        menu.overall.floor          = 1.5;
        menu.non_thinking           = menu.overall;
        const auto at5              = request_speculation_at(menu, 5, 9.0);
        const auto at3              = request_speculation_at(menu, 3, 9.0);
        const auto at7              = request_speculation_at(menu, 7, 9.0);
        const auto at0              = request_speculation_at(menu, 0, 9.0);
        ninferno::spec::SpeculationMenu plain;
        const auto plain3           = request_speculation_at(plain, 3, 1.7);
        failures += check(at5.draft_tokens == 5 && near(at5.acceptance_floor, 2.4) &&
                              at5.step_down.size() == 1 && at5.step_down[0].draft_tokens == 2 &&
                              near(at5.step_down[0].acceptance_floor, 1.5),
                          "a menu depth keeps its own floor and steps down to the shallower one");
        failures += check(at3.draft_tokens == 3 &&
                              near(at3.acceptance_floor, shallower_floor({{5, 2.4}, {2, 1.5}}, 3)) &&
                              at3.acceptance_floor < at5.acceptance_floor &&
                              at3.step_down.size() == 1 && at3.step_down[0].draft_tokens == 2,
                          "a depth between menu depths takes the interpolated floor, not the "
                          "deeper choice's");
        failures += check(at7.draft_tokens == 7 && at7.acceptance_floor > at5.acceptance_floor &&
                              at7.step_down.size() == 2 && at7.step_down[0].draft_tokens == 5,
                          "a deeper depth scales the floor up and steps down through every menu depth");
        failures += check(at0.draft_tokens == 0 && at0.step_down.empty(), "depth 0 is plain decode");
        failures += check(plain3.draft_tokens == 3 && near(plain3.acceptance_floor, 1.7) &&
                              plain3.step_down.empty(),
                          "a plain menu (explicit --spec) gives the engine's floor");

        const auto choice = [](ninferno::spec::Mode mode, std::int32_t k, const char* drafter,
                               double floor) {
            ninferno::spec::Choice c;
            c.config.mode            = mode;
            c.config.k               = k;
            c.drafter                = drafter;
            c.floor_tokens_per_round = floor;
            return c;
        };
        using ninferno::spec::Mode;
        ninferno::spec::Profile profile;
        profile.overall = profile.non_thinking = choice(Mode::DFlash, 7, "dflash", 2.14);
        profile.thinking                       = choice(Mode::DFlash, 5, "dflash", 1.9);
        profile.fit_fallback                   = choice(Mode::Mtp, 2, "mtp", 1.3);
        profile.served                         = ninferno::spec::ServedChoices{};

        const ninferno::spec::SpecConfig tried = profile.overall.config;
        const auto dflash6                     = shallower_profile(profile, tried, 2.14);
        failures +=
            check(dflash6 && dflash6->overall.config.k == 6 &&
                      dflash6->overall.drafter == "dflash" &&
                      near(dflash6->overall.floor_tokens_per_round, 1.9 + 0.5 * 0.24) &&
                      dflash6->thinking.config.k == 6 && dflash6->non_thinking.config.k == 6 &&
                      !dflash6->fit_fallback && !dflash6->served,
                  "DFlash k7 serves k6 everywhere with the floor between k5 and k7");

        // The fit fallback's MTP at k3 serves k2 with the fallback's drafter and its measured
        // floor.
        const ninferno::spec::SpecConfig mtp3 = choice(Mode::Mtp, 3, "", 0).config;
        const auto mtp2                       = shallower_profile(profile, mtp3, 1.6);
        failures += check(mtp2 && mtp2->overall.config.mode == Mode::Mtp &&
                              mtp2->overall.config.k == 2 &&
                              mtp2->overall.drafter == "mtp" &&
                              near(mtp2->overall.floor_tokens_per_round, 1.3),
                          "MTP k3 serves k2 with the fallback's drafter and floor");
        failures += check(!shallower_profile(profile, choice(Mode::DFlash2, 4, "", 0).config, 2) &&
                              !shallower_profile(profile, choice(Mode::DFlash, 1, "", 0).config,
                                                 1.5) &&
                              !shallower_profile(profile, ninferno::spec::SpecConfig{}, 0),
                          "no shallower profile for a mode it lacks, depth 1, or plain decode");
    }

    for (const char* kind : {"nemotron_h_mtp", "gemma4_assistant"}) {
        write_manifest(target, kind, "/drafters/head.ninferno");
        failures += check(mtp() == "/drafters/head.ninferno",
                          std::string("an attached ") + kind + " drafter drafts for --spec mtp");
        failures += check(dflash().empty(),
                          std::string("an attached ") + kind + " drafter is no DFlash drafter");
    }
    write_manifest(target, "nemotron_h_mtp", "/drafters/head.ninferno", "incompatible: vocabulary");
    failures += check(mtp().empty(), "an incompatible drafter is not used");
    write_manifest(target, "dflash", "/drafters/dflash.ninferno");
    failures += check(dflash() == "/drafters/dflash.ninferno" && mtp().empty(),
                      "a dflash drafter drafts for --spec dflash only");

    // Load bounds at depths the menu does not capture move to captured ones when the Engine
    // cannot afford their graphs: the deepest within the bound, else the shallowest captured.
    {
        ninferno::SpeculativeOptions options;
        options.backend              = SpeculativeBackend::DFlash;
        options.draft_tokens         = 7;
        options.request_draft_tokens = {2, 5, 11};
        options.load                 = {{.lanes = 2, .draft_tokens = 3},
                                        {.lanes = 3, .draft_tokens = 0},
                                        {.lanes = 4, .draft_tokens = 1},
                                        {.lanes = 5, .draft_tokens = 7}};
        failures +=
            check(ninferno::product::snap_load_to_captured(options, 8), "uncaptured bounds move");
        const auto& load = options.load;
        failures += check(load[0].draft_tokens == 2 && load[1].draft_tokens == 0 &&
                              load[2].draft_tokens == 2 && load[3].draft_tokens == 7,
                          "bounds move to 2, stay plain, move up to 2, stay at 7");
        failures += check(!ninferno::product::snap_load_to_captured(options, 8),
                          "captured bounds do not move");

        // A bound wider than the server's concurrency plans no graphs: it stays, and nothing moved
        // means no retry of the Engine.
        options.load = {{.lanes = 2, .draft_tokens = 5}, {.lanes = 5, .draft_tokens = 3}};
        failures += check(!ninferno::product::snap_load_to_captured(options, 4) &&
                              options.load[1].draft_tokens == 3,
                          "a bound beyond max_concurrency moved");
        failures += check(ninferno::product::snap_load_to_captured(options, 5) &&
                              options.load[1].draft_tokens == 2,
                          "a bound within max_concurrency did not move");
    }

    // A profile's table on the Engine: depths clamped to the engine's window, none on a plain one.
    {
        using ninferno::product::engine_load;
        const std::vector<ninferno::spec::LoadBound> table{{2, 7}, {3, 3}, {5, 0}};
        const auto clamped = engine_load(table, 5);
        failures += check(clamped.size() == 3 && clamped[0].lanes == 2 &&
                              clamped[0].draft_tokens == 5 && clamped[1].draft_tokens == 3 &&
                              clamped[2].lanes == 5 && clamped[2].draft_tokens == 0,
                          "engine_load clamps depths to the window and keeps lanes");
        failures += check(engine_load(table, 0).empty(), "a plain engine takes no bounds");
        const auto widths = engine_load(ninferno::spec::parse_load("2:7:block,3:3:drafts,5:0"), 7);
        failures += check(widths.size() == 3 &&
                              widths[0].draft_width == ninferno::DFlashDraftWidth::Block &&
                              widths[1].draft_width == ninferno::DFlashDraftWidth::Drafts &&
                              widths[2].draft_width == ninferno::DFlashDraftWidth::Default,
                          "engine_load carries a bound's draft width");

        // --spec auto drops a width its speculation does not run, with a note, rather than refuse
        // (the Engine refuses a width on an MTP round, and Qwen 3.5 refuses block).
        using ninferno::DFlashDraftWidth;
        using ninferno::product::fit_draft_widths;
        const auto named = [](SpeculativeBackend backend) {
            ninferno::SpeculativeOptions options;
            options.backend     = backend;
            options.draft_width = DFlashDraftWidth::Block;
            options.load = engine_load(ninferno::spec::parse_load("2:3:block,4:2:drafts"), 7);
            return options;
        };
        using ninferno::spec::engine_support;
        const ninferno::spec::EngineSupport dense_family = engine_support("dense_decoder");
        const ninferno::spec::EngineSupport qwen_family  = engine_support("qwen3_5_text");

        auto kept = named(SpeculativeBackend::DFlash);
        failures += check(dense_family.dflash_block_width &&
                              fit_draft_widths(kept, dense_family).empty() &&
                              kept.draft_width == DFlashDraftWidth::Block &&
                              kept.load[0].draft_width == DFlashDraftWidth::Block &&
                              kept.load[1].draft_width == DFlashDraftWidth::Drafts,
                          "a dense DFlash drafter runs both widths");
        auto qwen_block       = named(SpeculativeBackend::DFlash2);
        const auto qwen_notes = fit_draft_widths(qwen_block, qwen_family);
        failures += check(!qwen_family.dflash_block_width && qwen_notes.size() == 2 &&
                              qwen_block.draft_width == DFlashDraftWidth::Default &&
                              qwen_block.load[0].draft_width == DFlashDraftWidth::Default &&
                              qwen_block.load[1].draft_width == DFlashDraftWidth::Drafts &&
                              qwen_block.load[0].draft_tokens == 3,
                          "Qwen 3.5 drops block and keeps drafts, with notes");
        auto mtp_options = named(SpeculativeBackend::Mtp);
        failures += check(fit_draft_widths(mtp_options, dense_family).size() == 2 &&
                              mtp_options.draft_width == DFlashDraftWidth::Default &&
                              !ninferno::models::qwen3_5::detail::names_draft_width(mtp_options),
                          "an MTP pick or fit fallback drops every width");

        // A published table plain from three requests (Muse Glimmer 30B's gb203 table until
        // 2026-09-29) reaches SpeculativeOptions::load: two rows speculate at the profile's depth,
        // three and more decode plainly, and no speculative launch is planned wider than two rows.
        const std::vector<ninferno::spec::LoadBound> muse{{.lanes = 3, .depth = 0}};
        ninferno::SpeculativeOptions options;
        options.backend      = SpeculativeBackend::DFlash;
        options.draft_tokens = 3;
        options.load         = engine_load(muse, ninferno::product::speculative_window(options));
        namespace plan       = ninferno::models::qwen3_5::detail;
        failures += check(options.load.size() == 1 && options.load[0].lanes == 3 &&
                              options.load[0].draft_tokens == 0 &&
                              plan::speculation_load_valid(options),
                          "the Muse table reaches SpeculativeOptions::load as 3:0");
        constexpr auto unbounded = std::numeric_limits<std::uint32_t>::max();
        failures += check(plan::speculation_load_bound(options.load, 1) == unbounded &&
                              plan::speculation_load_bound(options.load, 2) == unbounded &&
                              plan::speculation_load_bound(options.load, 3) == 0 &&
                              plan::speculation_load_bound(options.load, 8) == 0,
                          "Muse: below the first entry no bound, from three rows plain");
        failures += check(plan::speculative_batch_limit(options.load, 8) == 2 &&
                              plan::speculative_batch_limit(options.load, 2) == 2 &&
                              plan::speculative_batch_limit({}, 8) == 8,
                          "no speculative launch from a bound of 0's lanes on");
        options.load = {{.lanes = 2, .draft_tokens = 3}, {.lanes = 4, .draft_tokens = 0}};
        failures += check(plan::speculation_load_bound(options.load, 3) == 3 &&
                              plan::speculation_load_valid(options),
                          "a step between two entries takes the lower one's bound");
        // The DFlash drafter's width per round: the whole block for one row, K + 1 columns from
        // two, unless a load bound or the options name one.
        const auto width = [&](DFlashDraftWidth forced, std::size_t rows) {
            return plan::dflash_draft_width(forced, options.load, rows);
        };
        failures += check(width(DFlashDraftWidth::Default, 1) == DFlashDraftWidth::Block &&
                              width(DFlashDraftWidth::Default, 2) == DFlashDraftWidth::Drafts &&
                              width(DFlashDraftWidth::Default, 8) == DFlashDraftWidth::Drafts &&
                              !plan::names_draft_width(options),
                          "the default draft width: block at one row, drafts from two");
        options.load = {{.lanes = 2, .draft_tokens = 3, .draft_width = DFlashDraftWidth::Block},
                        {.lanes = 4, .draft_tokens = 3}};
        failures +=
            check(width(DFlashDraftWidth::Default, 1) == DFlashDraftWidth::Block &&
                      width(DFlashDraftWidth::Default, 3) == DFlashDraftWidth::Block &&
                      width(DFlashDraftWidth::Default, 4) == DFlashDraftWidth::Drafts &&
                      width(DFlashDraftWidth::Drafts, 1) == DFlashDraftWidth::Drafts &&
                      width(DFlashDraftWidth::Block, 8) == DFlashDraftWidth::Block &&
                      plan::names_draft_width(options) && plan::speculation_load_valid(options),
                  "a bound's draft width applies to its rows; the options' to every round");
        options.load        = {};
        options.draft_width = DFlashDraftWidth::Drafts;
        failures += check(plan::names_draft_width(options), "a forced draft width is named");
        options.draft_width = DFlashDraftWidth::Default;
        for (const std::vector<ninferno::SpeculationLoad>& bad :
             {std::vector<ninferno::SpeculationLoad>{{.lanes = 1, .draft_tokens = 2}},
              std::vector<ninferno::SpeculationLoad>{{.lanes = 2, .draft_tokens = 0},
                                                   {.lanes = 3, .draft_tokens = 2}},
              std::vector<ninferno::SpeculationLoad>{{.lanes = 2, .draft_tokens = 4}}}) {
            options.load = bad;
            failures += check(!plan::speculation_load_valid(options),
                              "one-lane, after-plain and wider-than-window bounds are refused");
        }

        // The batches each launch window runs: Qwen3.6 on gb203, menu 2,5,7,11 with 2:3 at C=8
        // (depth 3 from two lanes). The deeper windows run one row only; 2 every batch, 3 from 2.
        const auto batches = [](std::initializer_list<std::uint32_t> list) {
            std::uint32_t mask = 0;
            for (const std::uint32_t batch : list) { mask |= 1U << (batch - 1U); }
            return mask;
        };
        const std::vector<std::uint32_t> qwen =
            plan::launch_window_batches({2, 3, 5, 7, 11}, {1, 2, 1, 1, 1},
                                        {{.lanes = 2, .draft_tokens = 3}}, 8, 8);
        failures += check(qwen == std::vector<std::uint32_t>{batches({1, 2, 3, 4, 5, 6, 7, 8}),
                                                             batches({2, 3, 4, 5, 6, 7, 8}),
                                                             batches({1}), batches({1}),
                                                             batches({1})},
                          "2:3 at C=8: windows 5, 7 and 11 run one row only");
        // Unbounded: every window every batch from its own first.
        const std::vector<std::uint32_t> unbounded_masks =
            plan::launch_window_batches({2, 5}, {1, 1}, {}, 4, 4);
        failures += check(unbounded_masks == std::vector<std::uint32_t>{batches({1, 2, 3, 4}),
                                                                        batches({1, 2, 3, 4})},
                          "without bounds every window runs every batch");
        // Gemma 4 31B on gb202 (2:7,3:3,5:5, menu 3,7): a launch of 3 or 4 rows in a step of 5
        // rows or more runs depth 5, whose graphs start at 5 rows, so it takes window 7; a bound
        // of 0 at 7 lanes ends every window at 6 rows.
        const std::vector<std::uint32_t> gemma = plan::launch_window_batches(
            {3, 5, 7}, {1, 5, 1},
            {{.lanes = 2, .draft_tokens = 7}, {.lanes = 3, .draft_tokens = 3},
             {.lanes = 5, .draft_tokens = 5}, {.lanes = 7, .draft_tokens = 0}},
            8, 6);
        failures += check(gemma == std::vector<std::uint32_t>{batches({1, 2, 3, 4, 5, 6}),
                                                              batches({5, 6}),
                                                              batches({1, 2, 3, 4})},
                          "a non-monotone table keeps the windows its bounds reach at each batch");
        failures += check(plan::window_runs_batch(batches({2}), 2) &&
                              !plan::window_runs_batch(batches({2}), 1) &&
                              !plan::window_runs_batch(batches({2}), 0),
                          "a window mask holds its batches only");

        // Every speculative launch the decode step can make finds its window, and that window's
        // mask holds the launch's batch (launch_mtp throws otherwise): menus x load tables x
        // concurrency, each step of `rows` rows launching `batch` of them (the rest plain) at
        // every depth up to the step's bound (the engine's window when unbounded). The window is
        // checked against its definition, not against launch_window_of's own loop: windows
        // ascend, it covers the depth and the batch, and no narrower one does. The layouts do not
        // depend on the backend (the options below say MTP), so the launches up to the bound hold
        // for MTP and DFlash alike. Deeper launches come from DFlash only: a row still appending
        // the features of a wider round launches deeper than the bound (up to the engine's window,
        // ProgramImpl::launch_depth); such a launch still finds a window, whose graphs may leave
        // it out (it then runs without one, ungraphed_rounds). MTP rows never exceed the bound.
        const auto covers = [](const plan::LaunchWindowShape& shape, std::uint32_t depth,
                               std::uint32_t batch) {
            return shape.drafts >= depth && shape.min_batch <= batch;
        };
        const auto launches_hold = [&](const ninferno::SpeculativeOptions& speculative,
                                       std::uint32_t concurrency) {
            const auto shapes = plan::launch_window_shapes(speculative, concurrency);
            std::vector<std::uint32_t> depths;
            std::vector<std::uint32_t> firsts;
            for (const plan::LaunchWindowShape& shape : shapes) {
                if (!depths.empty() && shape.drafts <= depths.back()) { return -1; }
                depths.push_back(shape.drafts);
                firsts.push_back(shape.min_batch);
            }
            // The window a launch takes, by its definition.
            const auto narrowest = [&](std::size_t w, std::uint32_t depth, std::uint32_t batch) {
                bool ok = covers(shapes[w], depth, batch);
                for (std::size_t v = 0; v < w; ++v) { ok = ok && !covers(shapes[v], depth, batch); }
                return ok;
            };
            const std::uint32_t limit =
                shapes.empty() ? 0U : plan::speculative_batch_limit(speculative.load, concurrency);
            const auto masks =
                plan::launch_window_batches(depths, firsts, speculative.load, concurrency, limit);
            std::uint32_t launches = 0;
            for (std::uint32_t rows = 1; rows <= concurrency; ++rows) {
                const std::uint32_t bound =
                    shapes.empty()
                        ? 0U
                        : std::min(plan::speculation_load_bound(speculative.load, rows),
                                   depths.back());
                for (std::uint32_t batch = 1; batch <= std::min(rows, limit); ++batch) {
                    for (std::uint32_t depth = 1; depth <= bound; ++depth) {
                        const auto w = plan::launch_window_of(shapes, depth, batch);
                        if (!w || !narrowest(*w, depth, batch) ||
                            !plan::window_runs_batch(masks.at(*w), batch)) {
                            return -1;
                        }
                        ++launches;
                    }
                    // DFlash rows appending a wider round's features (not MTP): past the bound.
                    for (std::uint32_t depth = bound + 1; bound > 0 && depth <= depths.back();
                         ++depth) {
                        const auto w = plan::launch_window_of(shapes, depth, batch);
                        if (!w || !narrowest(*w, depth, batch)) { return -1; }
                    }
                }
            }
            return static_cast<int>(launches);
        };
        using Load                    = std::vector<ninferno::SpeculationLoad>;
        const std::vector<Load> loads = {
            {},
            {{.lanes = 2, .draft_tokens = 3}},
            {{.lanes = 2, .draft_tokens = 3}, {.lanes = 3, .draft_tokens = 0}},
            {{.lanes = 2, .draft_tokens = 7},
             {.lanes = 3, .draft_tokens = 3},
             {.lanes = 5, .draft_tokens = 5},
             {.lanes = 7, .draft_tokens = 0}},
            {{.lanes = 3, .draft_tokens = 1},
             {.lanes = 4, .draft_tokens = 2},
             {.lanes = 6, .draft_tokens = 0}},
            {{.lanes = 2, .draft_tokens = 0}},
            {{.lanes = 2, .draft_tokens = 5}, {.lanes = 4, .draft_tokens = 2}}};
        const std::vector<std::vector<std::uint32_t>> menus = {{7}, {2, 5, 7, 11}, {3, 7}, {1, 4}};
        int walked                                          = 0;
        bool held                                           = true;
        for (const auto& menu : menus) {
            for (const Load& load : loads) {
                for (std::uint32_t concurrency = 1; concurrency <= 8; ++concurrency) {
                    ninferno::SpeculativeOptions speculative;
                    speculative.request_draft_tokens.assign(menu.begin(), menu.end() - 1);
                    speculative.backend      = SpeculativeBackend::Mtp;
                    speculative.draft_tokens = menu.back();
                    speculative.load         = load;
                    if (!plan::speculation_load_valid(speculative)) { continue; }
                    const int launches = launches_hold(speculative, concurrency);
                    held               = held && launches >= 0;
                    walked += launches > 0 ? 1 : 0;
                }
            }
        }
        failures += check(held && walked > 100,
                          "a launch's window does not run its batch (" + std::to_string(walked) +
                              " layouts walked)");
        // One request: every depth from one row, every window runs batch 1 only.
        ninferno::SpeculativeOptions single;
        single.backend              = SpeculativeBackend::DFlash;
        single.draft_tokens         = 7;
        single.request_draft_tokens = {3};
        single.load                 = {{.lanes = 2, .draft_tokens = 3}};
        failures += check(launches_hold(single, 1) == 7 &&
                              plan::launch_window_shapes(single, 1).size() == 2 &&
                              plan::launch_window_batches({3, 7}, {1, 1}, single.load, 1, 1) ==
                                  std::vector<std::uint32_t>{batches({1}), batches({1})},
                          "C=1: both windows run one row, every depth finds one");
        // Plain decode: no windows, no masks, no launch.
        const ninferno::SpeculativeOptions none;
        failures += check(plan::launch_window_shapes(none, 4).empty() &&
                              plan::launch_window_batches({}, {}, {}, 4, 4).empty() &&
                              !plan::launch_window_of(std::vector<plan::LaunchWindowShape>{},
                                                      1, 1) &&
                              launches_hold(none, 4) == 0,
                          "an empty window list plans and finds nothing");

        // Stopped rows ride at the width of the window the launch takes: menu 3,7 with 4:5 at C=8
        // (depth 5 captured from four rows). A drafting row at depth 5 in a launch of three rows
        // takes window 7 (width 8), in one of four window 5 (width 6).
        ninferno::SpeculativeOptions riding;
        riding.backend              = SpeculativeBackend::Mtp;
        riding.draft_tokens         = 7;
        riding.request_draft_tokens = {3};
        riding.load                 = {{.lanes = 4, .draft_tokens = 5}};
        const auto shapes           = plan::launch_window_shapes(riding, 8);
        const auto three            = plan::launch_window_of(shapes, 5, 3);
        const auto four             = plan::launch_window_of(shapes, 5, 4);
        failures += check(shapes.size() == 3 && shapes[1].drafts == 5 &&
                              shapes[1].min_batch == 4 && three == 2U && four == 1U &&
                              !plan::launch_window_of(shapes, 8, 4),
                          "depth 5 takes window 7 below four rows and window 5 from four");
        failures += check(plan::stopped_rows_ride(shapes, *three, 2, 16) &&
                              !plan::stopped_rows_ride(shapes, *three, 2, 15) &&
                              plan::stopped_rows_ride(shapes, *four, 2, 12) &&
                              !plan::stopped_rows_ride(shapes, *four, 3, 16),
                          "stopped rows ride within the columns of the launch's own window");

        // Copy rounds (ngram_draft_tokens): MTP 3 at C=4 with 2:2 adds the copy window 15 last,
        // from one row; a depth-3 row still takes window 3, a 15-token copy the copy window. Under
        // the bound of 2 from two rows only batch 1 runs the copy window. DFlash has no copy
        // window.
        ninferno::SpeculativeOptions copies;
        copies.backend            = SpeculativeBackend::Mtp;
        copies.draft_tokens       = 3;
        copies.load               = {{.lanes = 2, .draft_tokens = 2}};
        copies.ngram_draft_tokens = 15;
        const auto copy_shapes    = plan::launch_window_shapes(copies, 4);
        failures += check(copy_shapes.size() == 3 && copy_shapes[0].drafts == 2 &&
                              copy_shapes[1].drafts == 3 && copy_shapes[2].drafts == 15 &&
                              copy_shapes[2].min_batch == 1 &&
                              plan::launch_window_of(copy_shapes, 3, 1) == 1U &&
                              plan::launch_window_of(copy_shapes, 15, 1) == 2U,
                          "the copy window comes last, from one row");
        failures +=
            check(plan::launch_window_batches({2, 3, 15}, {2, 1, 1}, copies.load, 4, 4).back() ==
                      batches({1}),
                  "a load bound below the copy window keeps it to the rows it allows");
        ninferno::SpeculativeOptions dflash_copies = copies;
        dflash_copies.backend                    = SpeculativeBackend::DFlash;
        failures += check(plan::launch_window_shapes(dflash_copies, 4).back().drafts == 3,
                          "a DFlash engine has no copy window");

        // The copy proposal: the continuation of the most recent occurrence of the longest
        // recurring suffix, at most `limit` tokens; nothing below the minimum match.
        {
            using ninferno::models::qwen3_5::detail::copy_proposal;
            std::int32_t out[16] = {};
            // 1 2 3 4 5 6 | 9 | 1 2 3 4 5 6 | 7 8 | 3 4 5 6: suffix 3 4 5 6 recurs twice; the
            // most recent occurrence of the longest match (4) continues with 7 8 3 4 5 6.
            const std::vector<std::int32_t> h{1, 2, 3, 4, 5, 6, 9, 1, 2, 3,
                                              4, 5, 6, 7, 8, 3, 4, 5, 6};
            const std::size_t n = copy_proposal(h, 3, 4, out);
            failures += check(n == 4 && out[0] == 7 && out[1] == 8 && out[2] == 3 && out[3] == 4,
                              "a copy proposal follows the most recent longest match");
            // A longer match earlier wins over a shorter recent one: x 1 2 3 4 a | 4 b | 1 2 3 4.
            const std::vector<std::int32_t> l{0, 1, 2, 3, 4, 50, 4, 60, 1, 2, 3, 4};
            failures += check(copy_proposal(l, 2, 3, out) == 3 && out[0] == 50 && out[1] == 4 &&
                                  out[2] == 60,
                              "the longest recurring suffix decides the copy");
            failures +=
                check(copy_proposal(l, 5, 3, out) == 0 && copy_proposal(l, 2, 0, out) == 0 &&
                          copy_proposal(std::vector<std::int32_t>{7}, 1, 3, out) == 0,
                      "no copy below the minimum match, without a limit or a history");
        }
    }

    std::filesystem::remove_all(dir);
    if (failures == 0) { std::cout << "product speculation: ok\n"; }
    return failures == 0 ? 0 : 1;
}
