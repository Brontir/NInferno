#include "spec/pairs.h"

#include "artifact/reader.h"
#include "core/ascii.h"
#include "core/hub_revision.h"
#include "core/quant_label.h"
#include "spec/engine_support.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace ninferno::spec {

namespace {

using Json = nlohmann::json;

// The defaults a profile carries, each over its corpus sections (spec/profile.h select()).
struct Default {
    Choice Profile::*choice;
    std::vector<std::string> families;
};
const std::vector<Default>& defaults() {
    static const std::vector<Default> out = {
        {&Profile::overall, {"math", "code", "prose", "thinking"}},
        {&Profile::non_thinking, {"math", "code", "prose"}},
        {&Profile::thinking, {"thinking"}},
    };
    return out;
}

// The speculation mode a drafter kind drafts in.
Mode mode_of_kind(std::string_view kind) {
    if (kind == "mtp" || kind == "gemma4_assistant" || kind == "nemotron_h_mtp") {
        return Mode::Mtp;
    }
    if (kind == "dflash") { return Mode::DFlash; }
    if (kind == "dflash2") { return Mode::DFlash2; }
    throw std::invalid_argument("not a drafter kind: " + std::string(kind));
}

// One label over one section, its repeated measurements summed.
struct Sum {
    double tokens = 0, seconds = 0, rounds = 0, drafted = 0, accepted = 0;
};

// A pair's measurements by (label, family).
using Sums = std::map<std::pair<std::string, std::string>, Sum>;
Sums sums(const MeasuredPair& pair) {
    Sums out;
    for (const auto& m : pair.tune.measurements) {
        auto& s = out[{m.config.label(), m.family}];
        s.tokens += double(m.tokens);
        s.seconds += m.seconds;
        s.rounds += double(m.rounds);
        s.drafted += double(m.drafted);
        s.accepted += double(m.accepted);
    }
    return out;
}

// How one label does over a default's sections on one pair (each section one token's weight, as
// select() weighs them). nullopt when a section was not measured.
struct Over {
    double speed            = 0; // plain decode's cost over this label's
    double tokens_per_round = 1;
    double acceptance       = 0;
    double floor            = 0; // a round's seconds over plain's seconds per token
    // The speed were each section's drafted tokens accepted at kFineTuneAcceptance of its rate:
    // a section's rounds cost the seconds measured and yield rounds + kFineTuneAcceptance x
    // accepted tokens, then the sections weigh as above.
    double fine_tune_speed = 0;
};
std::optional<Over> over(const Sums& table, const std::string& label,
                         const std::vector<std::string>& families) {
    double cost = 0, plain_cost = 0, seconds = 0, rounds = 0, drafted = 0, accepted = 0;
    double fine_tune_cost = 0;
    for (const auto& f : families) {
        const auto m = table.find({label, f});
        const auto p = table.find({"plain", f});
        if (m == table.end() || p == table.end() || m->second.tokens <= 0 ||
            p->second.tokens <= 0) {
            return std::nullopt;
        }
        const double section_cost = m->second.seconds / m->second.tokens;
        cost += section_cost;
        // The section's rounds cost the seconds measured and yield, a draft accepted at
        // kFineTuneAcceptance of its rate, rounds + kFineTuneAcceptance x accepted tokens.
        fine_tune_cost +=
            m->second.rounds > 0
                ? m->second.seconds / (m->second.rounds + kFineTuneAcceptance * m->second.accepted)
                : section_cost;
        plain_cost += p->second.seconds / p->second.tokens;
        seconds += m->second.seconds;
        rounds += m->second.rounds;
        drafted += m->second.drafted;
        accepted += m->second.accepted;
    }
    if (cost <= 0 || rounds <= 0) { return std::nullopt; }
    Over out;
    out.speed            = plain_cost / cost;
    out.tokens_per_round = (accepted + rounds) / rounds;
    out.acceptance       = drafted > 0 ? accepted / drafted : 0.0;
    out.floor            = (seconds / rounds) / (plain_cost / double(families.size()));
    out.fine_tune_speed  = plain_cost / fine_tune_cost;
    return out;
}

Choice plain_choice() {
    Choice c;
    c.tok_s       = 1.0;
    c.plain_tok_s = 1.0;
    return c;
}

} // namespace

nlohmann::json pair_json(const MeasuredPair& pair) {
    Json out{{"target", pair.target},       {"target_repo", pair.target_repo},
             {"engine", pair.engine},       {"target_bits", pair.target_bits},
             {"generation", pair.generation}, {"tune", to_json(pair.tune)}};
    if (!pair.drafters.empty()) { out["drafters"] = pair.drafters; }
    return out;
}

MeasuredPair parse_pair(const nlohmann::json& doc) {
    MeasuredPair out;
    try {
        out.target      = doc.at("target").get<std::string>();
        out.target_repo = doc.value("target_repo", std::string());
        out.engine      = doc.at("engine").get<std::string>();
        out.target_bits = doc.at("target_bits").get<double>();
        out.generation  = doc.at("generation").get<std::string>();
        out.tune        = profile_from_json(doc.at("tune"));
        if (const auto it = doc.find("drafters"); it != doc.end()) {
            out.drafters = it->get<std::map<std::string, std::string>>();
        }
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("speculation pair: ") + error.what());
    }
    if (out.target.empty() || out.engine.empty() || out.generation.empty()) {
        throw std::runtime_error("speculation pair: target, engine and generation are required");
    }
    return out;
}

bool pair_current(const MeasuredPair& pair) { return pair.tune.key.engine == kSpecEngineVersion; }

std::string gpu_generation(const GpuKey& gpu) {
    for (const auto& key : gpu.chain) {
        if (key.size() > 2 && key.starts_with("sm") &&
            std::all_of(key.begin() + 2, key.end(), [](unsigned char c) {
                return std::isdigit(c) || std::isalpha(c);
            }) &&
            std::isdigit(static_cast<unsigned char>(key[2]))) {
            return key;
        }
    }
    return {};
}

namespace {

Profile not_current(const MeasuredPair& pair) {
    Profile plain;
    plain.overall = plain.thinking = plain.non_thinking = plain_choice();
    plain.disabled = RuntimeDisable{"measured by speculation engine " + pair.tune.key.engine +
                                        ", this build runs " + kSpecEngineVersion,
                                    "registry"};
    return plain;
}

// select() over the pair's measurements: every depth and its load bounds at home, else the
// portable ones.
Profile selected(const MeasuredPair& pair, bool home, const SelectOptions& options) {
    Profile p = pair.tune;
    if (!home) {
        std::erase_if(p.measurements,
                      [](const Measurement& m) { return m.config.k > kPortableDepth; });
        p.load = {LoadBound{.lanes = 2, .depth = 0}};
        if (p.fit_fallback && p.fit_fallback->config.k > kPortableDepth) {
            p.fit_fallback.reset();
        }
    }
    p.served.reset();
    p.portable.reset();
    SelectOptions select_options    = options;
    select_options.portable_within = 0;
    select(p, select_options);
    return p;
}

} // namespace

std::string weights_identity(std::vector<std::string> sha256s) {
    if (sha256s.empty() ||
        std::any_of(sha256s.begin(), sha256s.end(), [](const auto& d) { return d.empty(); })) {
        return {};
    }
    std::sort(sha256s.begin(), sha256s.end());
    std::string out;
    for (const auto& s : sha256s) { out += (out.empty() ? "" : ",") + s; }
    return out;
}

std::optional<std::string> artifact_weights(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    const auto& provenance = reader.directory().provenance;
    const auto inputs      = provenance.find("inputs");
    if (inputs == provenance.end() || !inputs->is_array()) { return std::nullopt; }
    for (const auto& input : *inputs) {
        if (!input.is_object() || input.value("role", std::string()) != "base") { continue; }
        std::vector<std::string> digests;
        for (const auto& file : input.value("files", Json::array())) {
            const auto digest = file.value("sha256", std::string());
            if (digest.empty()) { return std::nullopt; }
            digests.push_back(digest);
        }
        if (digests.empty()) { return std::nullopt; }
        return weights_identity(std::move(digests));
    }
    return std::nullopt;
}

Profile profile_for_pair(const MeasuredPair& pair, const SelectOptions& options) {
    return pair_current(pair) ? selected(pair, false, options) : not_current(pair);
}

Profile profile_for_target(const std::vector<const MeasuredPair*>& measured,
                           std::string_view generation, const SelectOptions& options) {
    std::vector<const MeasuredPair*> current, home;
    std::set<std::string> home_cards; // two cards, not two records of one card
    for (const auto* pair : measured) {
        if (!pair_current(*pair)) { continue; }
        current.push_back(pair);
        if (!generation.empty() && pair->generation == generation &&
            home_cards.insert(pair->tune.key.gpu).second) {
            home.push_back(pair);
        }
    }
    if (current.empty()) {
        return measured.empty() ? Profile{} : not_current(*measured.front());
    }
    if (home.size() >= 2) {
        std::vector<Profile> each;
        for (const auto* pair : home) { each.push_back(selected(*pair, true, options)); }
        const auto same = [&](Choice Profile::*c) {
            return std::all_of(each.begin(), each.end(), [&](const Profile& p) {
                return (p.*c).config == (each.front().*c).config;
            });
        };
        if (same(&Profile::overall) && same(&Profile::thinking) && same(&Profile::non_thinking)) {
            Profile out = each.front();
            // The watchdog's break-even: the highest any agreeing card measured (the order of the
            // records never moves it).
            for (Choice Profile::*c : {&Profile::overall, &Profile::thinking, &Profile::non_thinking}) {
                for (const Profile& p : each) {
                    (out.*c).floor_tokens_per_round =
                        std::max((out.*c).floor_tokens_per_round, (p.*c).floor_tokens_per_round);
                }
            }
            const bool loads = std::all_of(each.begin(), each.end(), [&](const Profile& p) {
                return p.load == each.front().load;
            });
            if (!loads || out.load.empty()) { out.load = {LoadBound{.lanes = 2, .depth = 0}}; }
            return out;
        }
    }
    return selected(home.empty() ? *current.front() : *home.front(), false, options);
}

std::vector<const MeasuredPair*> pair_group(const std::vector<MeasuredPair>& pairs,
                                            std::string_view engine, std::string_view kind,
                                            double target_bits) {
    const Mode mode = mode_of_kind(kind);
    std::vector<const MeasuredPair*> low, high;
    for (const auto& pair : pairs) {
        if (pair.engine != engine || !pair_current(pair)) { continue; }
        // That drafter kind, not another of its mode (MTP layers, Gemma's assistant and
        // Nemotron's MTP block all propose as MTP).
        const bool kind_of = std::any_of(pair.tune.drafters.begin(), pair.tune.drafters.end(),
                                         [&](const DrafterRecord& d) { return d.kind == kind; });
        const bool drafts  = std::any_of(
            pair.tune.measurements.begin(), pair.tune.measurements.end(),
            [&](const Measurement& m) { return m.config.mode == mode; });
        if (!kind_of || !drafts) { continue; }
        auto& into = pair.target_bits < 4 ? low : high;
        // One card's record of a target once.
        const bool again = std::any_of(into.begin(), into.end(), [&](const MeasuredPair* p) {
            return p->target == pair.target && p->tune.key.gpu == pair.tune.key.gpu;
        });
        if (!again) { into.push_back(&pair); }
    }
    // A group's choice is a lab constant for targets nobody measured: fitted on two cards or more.
    const auto enough = [](const std::vector<const MeasuredPair*>& group) {
        std::set<std::string> cards;
        for (const auto* p : group) { cards.insert(p->tune.key.gpu); }
        return cards.size() >= kGroupCards;
    };
    const bool below  = target_bits < 4;
    const auto& same  = below ? low : high;
    const auto& other = below ? high : low;
    if (enough(same)) { return same; }
    if (same.empty() && enough(other)) { return other; }
    return {};
}

namespace {

// A group's pair with its measurements by (label, family).
struct GroupMember {
    const MeasuredPair* pair;
    Sums table;
};

// How a label does over a group: the pairs that measured it over these sections and the mean of
// their gains at a fine-tune's acceptance (Over::fine_tune_speed); nullopt when fewer than
// kGroupCards cards measured it.
struct LabelGain {
    double gain = 0;
    std::vector<Over> each;
};
std::optional<LabelGain> label_gain(const std::vector<GroupMember>& members, const std::string& label,
                                    const std::vector<std::string>& families) {
    LabelGain out;
    std::set<std::string> cards;
    for (const auto& m : members) {
        if (const auto o = over(m.table, label, families)) {
            out.each.push_back(*o);
            out.gain += o->fine_tune_speed;
            cards.insert(m.pair->tune.key.gpu);
        }
    }
    if (cards.size() < kGroupCards) { return std::nullopt; }
    out.gain /= double(out.each.size());
    return out;
}

} // namespace

std::optional<Profile> profile_for_group(const std::vector<const MeasuredPair*>& group,
                                         std::string_view kind, const EngineSupport* support) {
    if (group.empty()) { return std::nullopt; }
    const Mode mode = mode_of_kind(kind);
    std::vector<GroupMember> members;
    for (const auto* pair : group) { members.push_back({pair, sums(*pair)}); }

    Profile out;
    out.key.engine = kSpecEngineVersion;
    for (const auto* pair : group) {
        for (const auto& d : pair->tune.drafters) {
            if (d.kind == kind && out.drafters.empty()) {
                DrafterRecord r = d;
                r.status        = "preliminary";
                r.reason        = "no measurement of this pair: the group of " +
                           std::to_string(group.size()) + " measured pairs";
                out.drafters.push_back(std::move(r));
            }
        }
    }
    const std::string drafter = out.drafters.empty() ? std::string(kind) : out.drafters[0].name;

    for (const auto& d : defaults()) {
        Choice& choice = out.*(d.choice);
        choice         = plain_choice();
        // The labels of this kind up to kPortableDepth any pair measured over these sections.
        std::set<std::string> labels;
        for (const auto& m : members) {
            for (const auto& [key, sum] : m.table) {
                const auto config = SpecConfig::parse(key.first);
                if (config.mode == mode && config.k <= kPortableDepth &&
                    (support == nullptr || support->runs(config))) {
                    labels.insert(key.first);
                }
            }
        }
        // The label a target nobody measured gains most with even when its acceptance is a
        // fine-tune's (label_gain): no plain decode for want of a label every pair measured, or
        // for the group's weakest pair (choice/DECISIONS 49).
        std::string chosen;
        LabelGain best;
        for (const auto& label : labels) {
            if (auto g = label_gain(members, label, d.families); g && g->gain > best.gain) {
                chosen = label;
                best   = std::move(*g);
            }
        }
        if (chosen.empty() || best.gain < kPreliminaryGain) { continue; }
        double tokens = 0, acceptance = 0, floor = 0;
        for (const auto& o : best.each) {
            tokens += o.tokens_per_round / double(best.each.size());
            acceptance += o.acceptance / double(best.each.size());
            floor = std::max(floor, o.floor);
        }
        choice.config                 = SpecConfig::parse(chosen);
        choice.drafter                = drafter;
        choice.tok_s                  = best.gain; // over plain decode's 1, at a fine-tune's acceptance
        choice.tokens_per_round       = tokens;
        choice.acceptance             = acceptance;
        choice.floor_tokens_per_round = floor;
    }
    out.load = {LoadBound{.lanes = 2, .depth = 0}};
    return out;
}

double label_bits(std::string_view label) { return quant_label_bits(label).value_or(16); }

std::string artifact_quant(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    return reader.directory().provenance.value("quant", std::string());
}

namespace {

// A drafter kind's mode; nullopt for a kind no pair drafts with.
std::optional<Mode> try_mode_of_kind(std::string_view kind) {
    try {
        return mode_of_kind(kind);
    } catch (const std::invalid_argument&) { return std::nullopt; }
}

// Drafts pay alike within a bit class: below 4 bits, or 4 and above.
bool same_bit_class(double a, double b) { return (a < 4) == (b < 4); }

// A pair's target_repo, "org/name@commit QUANT": its repository revision and its quantization.
std::string_view repo_revision_of(const MeasuredPair& pair) {
    return std::string_view(pair.target_repo).substr(0, pair.target_repo.find(' '));
}
std::string quant_of(const MeasuredPair& pair) {
    const auto space = pair.target_repo.rfind(' ');
    return space == std::string::npos ? pair.target_repo : pair.target_repo.substr(space + 1);
}

// What pair_choice is asked: the target, its engine family and bit class, the drafters it carries
// (their kinds and modes, and the identity of each one attached).
struct PairQuery {
    std::string_view target, engine;
    const std::vector<std::string>& kinds;
    double target_bits;
    const std::map<std::string, std::string>& attached;
    std::set<Mode> modes;
};

// A pair's measurements over the drafters the query's target carries only, of the current engine
// version (pair_current): nullopt where it is stale or measured other drafters only.
std::optional<MeasuredPair> carried_of(const MeasuredPair& pair, const PairQuery& query) {
    if (!pair_current(pair)) { return std::nullopt; }
    MeasuredPair p = pair;
    // A kind measured with another drafter than the one attached: not measured for it.
    std::set<Mode> carried = query.modes;
    for (const auto& [kind, id] : query.attached) {
        const auto measured = p.drafters.find(kind);
        if (measured == p.drafters.end() ||
            HubRevision::parse(measured->second).same_as(HubRevision::parse(id))) {
            continue;
        }
        if (const auto mode = try_mode_of_kind(kind)) { carried.erase(*mode); }
    }
    std::erase_if(p.tune.measurements, [&](const Measurement& m) {
        return m.config.enabled() && !carried.contains(m.config.mode);
    });
    std::erase_if(p.tune.drafters, [&](const DrafterRecord& d) {
        return std::find(query.kinds.begin(), query.kinds.end(), d.kind) == query.kinds.end();
    });
    if (std::none_of(p.tune.measurements.begin(), p.tune.measurements.end(),
                     [](const Measurement& m) { return m.config.enabled(); })) {
        return std::nullopt;
    }
    return p;
}

// One card's tunes (spec-tune measures one drafter at a time) are one measurement of the pair:
// their runs pooled, so its choice weighs every drafter the card measured. A load table or fit
// fallback is the one drafter's it was measured with: kept only when the tunes agree.
std::vector<MeasuredPair> pool_cards(std::vector<MeasuredPair> measured) {
    std::vector<MeasuredPair> cards;
    for (auto& p : measured) {
        const auto same = std::find_if(cards.begin(), cards.end(), [&](const MeasuredPair& c) {
            return c.tune.key.gpu == p.tune.key.gpu && c.generation == p.generation;
        });
        if (same == cards.end()) {
            cards.push_back(std::move(p));
            continue;
        }
        Profile& into = same->tune;
        into.measurements.insert(into.measurements.end(), p.tune.measurements.begin(),
                                 p.tune.measurements.end());
        for (const auto& d : p.tune.drafters) {
            const auto known = std::find_if(into.drafters.begin(), into.drafters.end(),
                                            [&](const DrafterRecord& r) { return r.name == d.name; });
            if (known == into.drafters.end()) { into.drafters.push_back(d); }
        }
        if (into.load != p.tune.load) { into.load.clear(); }
        if (!into.fit_fallback || !p.tune.fit_fallback ||
            into.fit_fallback->config != p.tune.fit_fallback->config) {
            into.fit_fallback.reset();
        }
    }
    return cards;
}

// The target's own measurements (carried_of each), one per card.
std::vector<MeasuredPair> own_cards(const std::vector<MeasuredPair>& pairs, const PairQuery& query) {
    std::vector<MeasuredPair> own;
    for (const auto& pair : pairs) {
        if (pair.target != query.target) { continue; }
        if (auto p = carried_of(pair, query)) { own.push_back(std::move(*p)); }
    }
    return pool_cards(std::move(own));
}

// The measured quantizations of the target's repository revision (`repo_at_commit`,
// org/name@commit) nearest to it in bits per weight within its bit class, each measured with a
// drafter the target carries (a nearer one measured with another drafter gives way to the next):
// their measurements one per card, and their quantization labels. Each sibling's own bits count,
// before one card's records are pooled (a pooled record keeps its first pair's).
struct Siblings {
    std::vector<MeasuredPair> cards;
    std::set<std::string> quants;
};
Siblings nearest_siblings(const std::vector<MeasuredPair>& pairs, const PairQuery& query,
                          std::string_view repo_at_commit) {
    std::vector<std::pair<const MeasuredPair*, MeasuredPair>> carried;
    double nearest = std::numeric_limits<double>::infinity();
    for (const auto& pair : pairs) {
        const auto revision = repo_revision_of(pair);
        const auto at       = revision.find('@');
        // The same repository (case aside) at the same commit (a prefix of it).
        if (at == std::string_view::npos || at + 1 == revision.size() ||
            !HubRevision::parse(revision).same_as(HubRevision::parse(repo_at_commit)) ||
            pair.engine != query.engine ||
            pair.target == query.target || !same_bit_class(pair.target_bits, query.target_bits)) {
            continue;
        }
        if (auto p = carried_of(pair, query)) {
            nearest = std::min(nearest, std::abs(pair.target_bits - query.target_bits));
            carried.emplace_back(&pair, std::move(*p));
        }
    }
    Siblings out;
    std::vector<MeasuredPair> at_nearest;
    for (auto& [pair, p] : carried) {
        if (std::abs(pair->target_bits - query.target_bits) != nearest) { continue; }
        out.quants.insert(quant_of(*pair));
        at_nearest.push_back(std::move(p));
    }
    out.cards = pool_cards(std::move(at_nearest));
    return out;
}

std::vector<const MeasuredPair*> pointers(const std::vector<MeasuredPair>& pairs) {
    std::vector<const MeasuredPair*> out;
    for (const auto& p : pairs) { out.push_back(&p); }
    return out;
}

} // namespace

std::optional<PairChoice> pair_choice(const std::vector<MeasuredPair>& pairs,
                                      std::string_view target, std::string_view engine,
                                      const std::vector<std::string>& kinds, double target_bits,
                                      std::string_view generation, const SelectOptions& options,
                                      const std::map<std::string, std::string>& attached,
                                      std::string_view repo) {
    PairQuery query{.target = target, .engine = engine, .kinds = kinds,
                    .target_bits = target_bits, .attached = attached, .modes = {}};
    for (const auto& kind : kinds) {
        if (const auto mode = try_mode_of_kind(kind)) { query.modes.insert(*mode); }
    }
    // What the serving Engine of the family runs: the choices are selected over it (a pair's
    // fastest configuration may be one it does not run, its served choice then stands).
    // An engine family it does not know leaves the selection as asked.
    SelectOptions served = options;
    if (!served.served && engine_family_support(engine).any()) {
        served.served = engine_family_support(engine);
    }
    const EngineSupport* support = served.served ? &*served.served : nullptr;

    // 1. Its own measurements.
    if (const auto own = own_cards(pairs, query); !own.empty()) {
        return PairChoice{.profile = profile_for_target(pointers(own), generation, served),
                          .basis   = "measured",
                          .source  = "registry pair " + own.front().target_repo + " (" +
                                    std::to_string(own.size()) + " lab card(s))"};
    }
    // 2. Not measured, but another quantization of the same model was: the sibling nearest in
    // bits (the weights the same and quantized otherwise, so its drafters are accepted alike: no
    // fine-tune's guard). Of the same repository revision only: another commit may hold other
    // weights (a new version of the model, not another quantization of it).
    if (const auto at = repo.find('@'); at != std::string_view::npos && at + 1 < repo.size()) {
        if (const auto siblings = nearest_siblings(pairs, query, repo); !siblings.cards.empty()) {
            std::string named;
            for (const auto& q : siblings.quants) { named += (named.empty() ? "" : ", ") + q; }
            return PairChoice{.profile = profile_for_target(pointers(siblings.cards), generation, served),
                              .basis   = "preliminary",
                              .source  = "registry sibling " + std::string(repo.substr(0, at)) +
                                        " " + named + " (" + std::to_string(siblings.cards.size()) +
                                        " lab card(s); this quantization not measured)"};
        }
    }
    // 3. Nobody measured it: the group of each kind it carries, the fastest overall.
    std::optional<PairChoice> best;
    for (const auto& kind : kinds) {
        if (!try_mode_of_kind(kind)) { continue; }
        const auto group = pair_group(pairs, engine, kind, target_bits);
        auto profile     = profile_for_group(group, kind, support);
        if (!profile) { continue; }
        if (!best || profile->overall.speedup() > best->profile.overall.speedup()) {
            best = PairChoice{.profile = std::move(*profile),
                              .basis   = "preliminary",
                              .source  = "registry group " + std::string(engine) + " + " + kind +
                                        " (" + std::to_string(group.size()) +
                                        " measured pairs; this one not measured)"};
        }
    }
    return best;
}

} // namespace ninferno::spec
