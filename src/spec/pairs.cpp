#include "spec/pairs.h"

#include <algorithm>
#include <cctype>
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
std::map<std::pair<std::string, std::string>, Sum> sums(const MeasuredPair& pair) {
    std::map<std::pair<std::string, std::string>, Sum> out;
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
};
std::optional<Over> over(const std::map<std::pair<std::string, std::string>, Sum>& table,
                         const std::string& label, const std::vector<std::string>& families) {
    double cost = 0, plain_cost = 0, seconds = 0, rounds = 0, drafted = 0, accepted = 0;
    for (const auto& f : families) {
        const auto m = table.find({label, f});
        const auto p = table.find({"plain", f});
        if (m == table.end() || p == table.end() || m->second.tokens <= 0 ||
            p->second.tokens <= 0) {
            return std::nullopt;
        }
        cost += m->second.seconds / m->second.tokens;
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
    return Json{{"target", pair.target},       {"target_repo", pair.target_repo},
                {"engine", pair.engine},       {"target_bits", pair.target_bits},
                {"generation", pair.generation}, {"tune", to_json(pair.tune)}};
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

Profile profile_for_pair(const MeasuredPair& pair, std::string_view generation,
                         const SelectOptions& options) {
    if (!pair_current(pair)) {
        Profile plain;
        plain.overall = plain.thinking = plain.non_thinking = plain_choice();
        plain.disabled = RuntimeDisable{"measured by speculation engine " + pair.tune.key.engine +
                                            ", this build runs " + kSpecEngineVersion,
                                        "registry"};
        return plain;
    }
    Profile p    = pair.tune;
    const bool home = !generation.empty() && generation == pair.generation;
    if (!home) {
        // Away from the lab card's generation, only what carries: one request, at most
        // kPortableDepth drafts a round.
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

std::vector<const MeasuredPair*> pair_group(const std::vector<MeasuredPair>& pairs,
                                            std::string_view engine, std::string_view kind,
                                            double target_bits) {
    const Mode mode = mode_of_kind(kind);
    std::vector<const MeasuredPair*> low, high;
    for (const auto& pair : pairs) {
        if (pair.engine != engine || !pair_current(pair)) { continue; }
        const bool drafts = std::any_of(
            pair.tune.measurements.begin(), pair.tune.measurements.end(),
            [&](const Measurement& m) { return m.config.mode == mode; });
        if (!drafts) { continue; }
        (pair.target_bits < 4 ? low : high).push_back(&pair);
    }
    const bool below = target_bits < 4;
    auto& same       = below ? low : high;
    return same.empty() ? (below ? high : low) : same;
}

std::optional<Profile> profile_for_group(const std::vector<const MeasuredPair*>& group,
                                         std::string_view kind) {
    if (group.empty()) { return std::nullopt; }
    const Mode mode = mode_of_kind(kind);
    std::vector<std::map<std::pair<std::string, std::string>, Sum>> tables;
    for (const auto* pair : group) { tables.push_back(sums(*pair)); }

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
        // The labels of this kind every pair measured over these sections, up to kPortableDepth.
        std::set<std::string> labels;
        for (const auto& [key, sum] : tables.front()) {
            const auto config = SpecConfig::parse(key.first);
            if (config.mode == mode && config.k <= kPortableDepth) { labels.insert(key.first); }
        }
        std::map<std::string, std::vector<Over>> by_label;
        for (const auto& label : labels) {
            std::vector<Over> each;
            for (const auto& table : tables) {
                if (const auto o = over(table, label, d.families)) { each.push_back(*o); }
            }
            if (each.size() == tables.size()) { by_label[label] = std::move(each); }
        }
        if (by_label.empty()) { continue; }
        // Each pair's best over these labels (at least plain decode's speed).
        std::vector<double> best(tables.size(), 1.0);
        for (const auto& [label, each] : by_label) {
            for (std::size_t i = 0; i < each.size(); ++i) { best[i] = std::max(best[i], each[i].speed); }
        }
        // The label with the highest mean speed relative to each pair's best.
        std::string chosen;
        double chosen_mean = -1;
        for (const auto& [label, each] : by_label) {
            double mean = 0;
            for (std::size_t i = 0; i < each.size(); ++i) { mean += each[i].speed / best[i]; }
            mean /= double(each.size());
            if (mean > chosen_mean) {
                chosen_mean = mean;
                chosen      = label;
            }
        }
        // It must pay on every pair even at a fine-tune's acceptance.
        const auto& each = by_label[chosen];
        double worst = std::numeric_limits<double>::infinity(), speed = worst, tokens = worst;
        double acceptance = worst, floor = 0;
        for (const auto& o : each) {
            const double price = o.tokens_per_round / o.speed; // a round in plain steps
            worst      = std::min(worst, (1.0 + kFineTuneAcceptance * (o.tokens_per_round - 1.0)) / price);
            speed      = std::min(speed, o.speed);
            tokens     = std::min(tokens, o.tokens_per_round);
            acceptance = std::min(acceptance, o.acceptance);
            floor      = std::max(floor, o.floor);
        }
        if (worst < kPreliminaryGain) { continue; }
        choice.config                 = SpecConfig::parse(chosen);
        choice.drafter                = drafter;
        choice.tok_s                  = speed; // the group's least gain, over plain decode's 1
        choice.tokens_per_round       = tokens;
        choice.acceptance             = acceptance;
        choice.floor_tokens_per_round = floor;
    }
    out.load = {LoadBound{.lanes = 2, .depth = 0}};
    return out;
}

} // namespace ninferno::spec
