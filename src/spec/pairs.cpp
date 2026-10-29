#include "spec/pairs.h"

#include "artifact/reader.h"
#include "core/ascii.h"
#include "core/hub_revision.h"
#include "core/quant_label.h"
#include "spec/engine_support.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
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

// How one label does over a default's sections on one pair (each section by its share of the mix,
// section_weight, as select() weighs them). nullopt when a section was not measured.
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
    double cost = 0, plain_cost = 0, rounds = 0, drafted = 0, accepted = 0;
    double fine_tune_cost = 0, round_cost = 0, weight = 0;
    for (const auto& f : families) {
        const auto m = table.find({label, f});
        const auto p = table.find({"plain", f});
        if (m == table.end() || p == table.end() || m->second.tokens <= 0 ||
            p->second.tokens <= 0) {
            return std::nullopt;
        }
        const double w            = section_weight(f);
        const double section_cost = m->second.seconds / m->second.tokens;
        cost += w * section_cost;
        // The section's rounds cost the seconds measured and yield, a draft accepted at
        // kFineTuneAcceptance of its rate, rounds + kFineTuneAcceptance x accepted tokens.
        fine_tune_cost +=
            w * (m->second.rounds > 0 ? m->second.seconds / (m->second.rounds +
                                                             kFineTuneAcceptance * m->second.accepted)
                                      : section_cost);
        plain_cost += w * p->second.seconds / p->second.tokens;
        if (m->second.rounds > 0) { round_cost += w * m->second.seconds / m->second.rounds; }
        weight += w;
        rounds += m->second.rounds;
        drafted += m->second.drafted;
        accepted += m->second.accepted;
    }
    if (cost <= 0 || rounds <= 0) { return std::nullopt; }
    Over out;
    out.speed            = plain_cost / cost;
    out.tokens_per_round = (accepted + rounds) / rounds;
    out.acceptance       = drafted > 0 ? accepted / drafted : 0.0;
    out.floor            = (round_cost / weight) / (plain_cost / weight);
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
    if (pair.round_facts) { out["round_facts"] = round_facts_json(*pair.round_facts); }
    if (pair.lab_board != 0) {
        char id[16];
        std::snprintf(id, sizeof id, "0x%04X", pair.lab_board);
        out["lab_board"] = id;
    }
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
        if (const auto it = doc.find("round_facts"); it != doc.end()) {
            out.round_facts = parse_round_facts(*it);
        }

        if (const auto it = doc.find("lab_board"); it != doc.end()) {
            const auto text = it->get<std::string>();
            char* end       = nullptr;
            const auto id   = std::strtoul(text.c_str(), &end, 16);
            if (text.size() < 3 || !text.starts_with("0x") || end == nullptr || *end != '\0' ||
                id == 0 || id > 0xFFFF) {
                throw std::runtime_error("speculation pair: lab_board \"" + text +
                                         "\" is no PCI product id (0x2782)");
            }
            out.lab_board = static_cast<std::uint32_t>(id);
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

std::map<std::string, double> round_costs(const Profile& tune);

// Each round of the measurements priced at `estimate` (by configuration label, plain decode steps
// a round): its seconds become its rounds x that price x plain decode's seconds a token in its
// section; its tokens (E) stay measured. A configuration the estimate does not price, or a section
// plain decode did not measure, is left out. The fit fallback scales by the estimate over the lab
// card's own price.
void price_by_facts(Profile& p, const RoundCostEstimate& estimate) {
    std::map<std::string, std::pair<double, double>> plain; // by section: seconds, tokens
    for (const auto& m : p.measurements) {
        if (!m.config.enabled() && m.tokens > 0) {
            plain[m.family].first += m.seconds;
            plain[m.family].second += double(m.tokens);
        }
    }
    const auto measured = round_costs(p);
    std::erase_if(p.measurements, [&](const Measurement& m) {
        if (!m.config.enabled()) { return false; }
        const auto s = plain.find(m.family);
        return m.rounds <= 0 || !estimate.cost.contains(m.config.label()) || s == plain.end() ||
               s->second.first <= 0;
    });
    for (auto& m : p.measurements) {
        if (!m.config.enabled()) { continue; }
        const auto& s = plain.at(m.family);
        m.seconds = double(m.rounds) * estimate.cost.at(m.config.label()) * s.first / s.second;
    }
    if (p.fit_fallback && p.fit_fallback->config.enabled()) {
        const auto label = p.fit_fallback->config.label();
        const auto e     = estimate.cost.find(label);
        const auto lab   = measured.find(label);
        if (e != estimate.cost.end() && lab != measured.end() && lab->second > 0) {
            const double r = e->second / lab->second;
            p.fit_fallback->tok_s /= r;
            p.fit_fallback->floor_tokens_per_round *= r;
        } else {
            p.fit_fallback.reset();
        }
    }
}

// select() over the pair's measurements: every depth and its load bounds at home, else the
// portable ones, each round's seconds scaled by `round_costs` (by configuration label: what a round
// costs on the card asked for over the measuring one), or priced by `facts` (price_by_facts). With
// costs given, a configuration without one is left out: its round's cost here is unknown, and at
// the measuring card's price it would look cheaper than the ones costed.
Profile selected(const MeasuredPair& pair, bool home, const SelectOptions& options,
                 const RoundCostRatios& round_costs = {},
                 const RoundCostEstimate* facts     = nullptr) {
    Profile p = pair.tune;
    if (facts != nullptr && facts->known) {
        price_by_facts(p, *facts);
    } else if (round_costs.known) {
        const auto cost = [&](const SpecConfig& c) {
            const auto r = round_costs.ratio.find(c.label());
            return r != round_costs.ratio.end() ? std::optional<double>{r->second} : std::nullopt;
        };
        std::erase_if(p.measurements,
                      [&](const Measurement& m) { return m.config.enabled() && !cost(m.config); });
        for (auto& m : p.measurements) {
            if (m.config.enabled()) { m.seconds *= *cost(m.config); }
        }
        if (p.fit_fallback && p.fit_fallback->config.enabled()) {
            if (const auto r = cost(p.fit_fallback->config)) {
                p.fit_fallback->tok_s /= *r;
                p.fit_fallback->floor_tokens_per_round *= *r;
            } else {
                p.fit_fallback.reset();
            }
        }
    }
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

namespace {

// A round's cost per configuration label in one tune: in each section plain decode measured too,
// its seconds per round over plain decode's seconds per token (repeats pooled); the mean over the
// sections.
std::map<std::string, double> round_costs(const Profile& tune) {
    using Pooled = std::pair<double, double>;           // seconds, and tokens or rounds
    std::map<std::string, Pooled> plain;                // by section
    std::map<std::pair<std::string, std::string>, Pooled> rounds; // by label and section
    for (const auto& m : tune.measurements) {
        if (!m.config.enabled()) {
            if (m.tokens > 0) {
                plain[m.family].first += m.seconds;
                plain[m.family].second += static_cast<double>(m.tokens);
            }
        } else if (m.rounds > 0) {
            auto& r = rounds[{m.config.label(), m.family}];
            r.first += m.seconds;
            r.second += static_cast<double>(m.rounds);
        }
    }
    std::map<std::string, Pooled> sums; // by label: the sections' costs, and their count
    for (const auto& [key, r] : rounds) {
        const auto p = plain.find(key.second);
        if (p == plain.end() || p->second.first <= 0) { continue; }
        auto& s = sums[key.first];
        s.first += (r.first / r.second) / (p->second.first / p->second.second);
        s.second += 1;
    }
    std::map<std::string, double> out;
    for (const auto& [label, s] : sums) { out[label] = s.first / s.second; }
    return out;
}

} // namespace

std::map<std::string, double> profile_round_costs(const Profile& profile) {
    return round_costs(profile);
}

namespace {

// Per configuration label, the median over `pairs` of each one's round cost (round_costs; an even
// count: the mean of the middle two).
std::map<std::string, double> median_round_costs(const std::vector<const MeasuredPair*>& pairs) {
    std::map<std::string, std::vector<double>> each;
    for (const auto* pair : pairs) {
        for (const auto& [label, cost] : round_costs(pair->tune)) { each[label].push_back(cost); }
    }
    std::map<std::string, double> out;
    for (auto& [label, v] : each) {
        std::sort(v.begin(), v.end());
        const std::size_t n = v.size();
        out[label]          = n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
    }
    return out;
}

// What scales `base`'s rounds to `wanted` costs, by configuration label (selected() leaves out a
// configuration without one).
RoundCostRatios ratios_to(const MeasuredPair& base, const std::map<std::string, double>& wanted) {
    RoundCostRatios out;
    const auto have = round_costs(base.tune);
    for (const auto& [label, cost] : wanted) {
        if (const auto h = have.find(label); h != have.end() && h->second > 0) {
            out.ratio[label] = cost / h->second;
        }
    }
    out.known = !out.ratio.empty();
    return out;
}

// The record whose rounds carry `costs`: the one that measured the most of its configurations
// (a configuration the base did not measure would be left out), then `preferred` (the card's own),
// then the first in `pairs`' order (by GPU class id).
const MeasuredPair& base_of(const std::vector<const MeasuredPair*>& pairs,
                            const std::map<std::string, double>& costs,
                            const MeasuredPair* preferred) {
    const auto covered = [&](const MeasuredPair* p) {
        const auto have = round_costs(p->tune);
        return std::count_if(costs.begin(), costs.end(),
                             [&](const auto& c) { return have.contains(c.first); });
    };
    const MeasuredPair* best = nullptr;
    long best_n              = -1;
    for (const auto* p : pairs) {
        const long n = static_cast<long>(covered(p));
        if (n > best_n || (n == best_n && p == preferred)) {
            best   = p;
            best_n = n;
        }
    }
    return *best;
}

// One record per card (its GPU class id), the first in `pairs`' order.
std::vector<const MeasuredPair*> one_per_card(const std::vector<const MeasuredPair*>& pairs) {
    std::vector<const MeasuredPair*> out;
    std::set<std::string> cards;
    for (const auto* pair : pairs) {
        if (cards.insert(pair->tune.key.gpu).second) { out.push_back(pair); }
    }
    return out;
}

// A generation's name for a source ("sm75" -> "turing"); the key itself when unnamed.
std::string generation_name(std::string_view generation) {
    static const std::map<std::string, std::string, std::less<>> names = {
        {"sm70", "volta"},  {"sm75", "turing"}, {"sm80", "ampere"},     {"sm86", "ampere"},
        {"sm87", "ampere"}, {"sm89", "ada"},    {"sm90", "hopper"},     {"sm100", "blackwell"},
        {"sm103", "blackwell"}, {"sm120", "blackwell"}, {"sm121", "blackwell"}};
    const auto it = names.find(generation);
    return it != names.end() ? it->second : std::string(generation);
}

} // namespace

RoundCostEstimate round_cost_estimate(const MeasuredPair& pair, const GpuKey& gpu,
                                      const ProposalRows* head_rows) {
    RoundCostEstimate out;
    if (!pair.round_facts) { return out; }
    const CardRates card = card_rates(gpu);
    const CardRates home = card_rates_of_board(pair.lab_board);
    if (!card.known() || !home.known()) { return out; }
    std::string raised;
    for (const auto& [label, measured] : round_costs(pair.tune)) {
        const auto config = SpecConfig::parse(label);
        if (!config.enabled() || config.k > kPortableDepth) { continue; }
        // The lab drafted a "+head" configuration through its artifact's proposal head, any other
        // with the full head (spec-tune: ProposalHead::Full); a start drafts through the head
        // wherever its artifact's serves the drafter (product model_draft_head), whatever the label.
        const ProposalRows lab = config.proposal_head ? pair.tune.proposal_rows : ProposalRows{};
        const auto here  = round_cost(*pair.round_facts, config, card.kappa(),
                                      head_rows != nullptr ? *head_rows : lab);
        const auto there = round_cost(*pair.round_facts, config, home.kappa(), lab);
        if (!here || !there || *there <= 0) { continue; }
        double cost = *here;
        // What the bytes do not show, as the lab card measured it: scales the estimate up only.
        if (const double excess = measured / *there; excess > 1 + kRoundCostAgreement) {
            cost *= excess;
            char pct[48];
            std::snprintf(pct, sizeof pct, "%s +%.0f %%", label.c_str(), 100 * (excess - 1));
            raised += (raised.empty() ? "" : ", ") + std::string(pct);
        }
        out.cost[label] = cost;
    }
    out.known = !out.cost.empty();
    if (out.known) {
        char eta[96];
        std::snprintf(eta, sizeof eta, "%s, eta %.3f%s", card.generation.c_str(), card.eta,
                      card.eta_cards < 2 ? " from one lab card" : "");
        out.note = "rounds priced from the weights' bytes and this card's rates (" +
                   std::string(eta) + ")";
        if (!raised.empty()) {
            out.note += "; conservative where the lab card measured rounds dearer than their "
                        "bytes: " +
                        raised;
        }
    }
    return out;
}

Profile profile_for_target(const std::vector<const MeasuredPair*>& measured,
                           std::string_view generation, const SelectOptions& options,
                           const GpuKey* gpu, std::string* note) {
    // The records in an order of their own (GPU class id, then content): the registry's order
    // never decides which card's measurement is taken.
    std::vector<const MeasuredPair*> ordered = measured;
    std::stable_sort(ordered.begin(), ordered.end(), [](const MeasuredPair* a, const MeasuredPair* b) {
        if (a->tune.key.gpu != b->tune.key.gpu) { return a->tune.key.gpu < b->tune.key.gpu; }
        return pair_json(*a).dump() < pair_json(*b).dump();
    });
    std::vector<const MeasuredPair*> current, home;
    std::set<std::string> home_cards; // two cards, not two records of one card
    for (const auto* pair : ordered) {
        if (!pair_current(*pair)) { continue; }
        current.push_back(pair);
        if (!generation.empty() && pair->generation == generation &&
            home_cards.insert(pair->tune.key.gpu).second) {
            home.push_back(pair);
        }
    }
    if (current.empty()) {
        return ordered.empty() ? Profile{} : not_current(*ordered.front());
    }
    const MeasuredPair* own_pair = nullptr;
    // The card itself measured it: its tokens a round, at its generation's median round cost when
    // another lab card of the generation measured it too (below). Its own keys are its class id and chip
    // (a tune's key.gpu is either; the architecture is every card of it, not this one), and where
    // the pair names its lab board, that board's rates too: another board of the class (an RTX
    // 3060 8 GB beside the 12 GB lab card, an RTX 4500 Ada beside the 4070 Ti) is priced by the
    // facts where its generation has them.
    if (gpu != nullptr && !gpu->chain.empty()) {
        const std::size_t own = std::min<std::size_t>(gpu->own_keys == 0 ? 1 : gpu->own_keys, 2);
        const CardRates here  = card_rates(*gpu);
        const auto same_board = [&](const MeasuredPair& pair) {
            const CardRates lab = card_rates_of_board(pair.lab_board);
            if (lab.dram_bytes_per_s <= 0 || here.dram_bytes_per_s <= 0) { return true; }
            return std::abs(here.dram_bytes_per_s / lab.dram_bytes_per_s - 1) < 0.01 &&
                   std::abs(here.int8_ops_per_s / lab.int8_ops_per_s - 1) < 0.01;
        };
        for (const auto* pair : home) {
            for (std::size_t i = 0; i < own && i < gpu->chain.size() && own_pair == nullptr; ++i) {
                if (pair->tune.key.gpu == gpu->chain[i] && same_board(*pair)) { own_pair = pair; }
            }
        }
    }
    // Another card's: its tokens a round (the pair's, on any card), its rounds priced for this card
    // by the facts. A pair with facts is taken over another card's measurement without them even
    // of this generation: R is computed for this card either way, and a raw measurement of another
    // card is the very transfer the facts replace. This generation's records first (every one of
    // them, a card's second record too), then the others; within each by GPU class id, so the
    // registry's order never decides.
    // Ahead of the generation's agreeing lab cards too: their deeper depths and load table are
    // theirs, not a card that measured nothing (kPortableDepth bounds an estimate).
    if (gpu != nullptr && own_pair == nullptr) {
        std::vector<const MeasuredPair*> order = current;
        std::stable_sort(order.begin(), order.end(), [&](const MeasuredPair* a, const MeasuredPair* b) {
            const bool ha = a->generation == generation, hb = b->generation == generation;
            return ha != hb ? ha : a->tune.key.gpu < b->tune.key.gpu;
        });
        for (const auto* pair : order) {
            const ProposalRows* rows = options.proposal_rows ? &*options.proposal_rows : nullptr;
            if (const auto estimate = round_cost_estimate(*pair, *gpu, rows); estimate.known) {
                if (note != nullptr) { *note = estimate.note; }
                return selected(*pair, false, options, {}, &estimate);
            }
        }
    }
    // This generation's lab cards: their median round cost per configuration, on the record that
    // measured the most configurations (the card's own among equals; tokens a round are the pair's
    // on any card). Two or more that choose alike keep every measured depth and their load bounds
    // when equal, as long as the median keeps their choice.
    if (home.size() >= 2) {
        const auto costs         = median_round_costs(home);
        const MeasuredPair& base = base_of(home, costs, own_pair);
        const auto ratios        = ratios_to(base, costs);
        std::vector<Profile> each;
        for (const auto* pair : home) { each.push_back(selected(*pair, true, options)); }
        const auto same = [&](Choice Profile::*c) {
            return std::all_of(each.begin(), each.end(), [&](const Profile& p) {
                return (p.*c).config == (each.front().*c).config;
            });
        };
        if (note != nullptr) {
            *note = "rounds: the median of " + std::to_string(home.size()) + " lab cards of " +
                    generation_name(generation);
        }
        const auto choices = {&Profile::overall, &Profile::thinking, &Profile::non_thinking};
        if (std::all_of(choices.begin(), choices.end(), same)) {
            Profile out = selected(base, true, options, ratios);
            const bool kept = std::all_of(choices.begin(), choices.end(), [&](Choice Profile::*c) {
                return (out.*c).config == (each.front().*c).config;
            });
            if (kept) {
                // The watchdog's break-even: the highest any agreeing card measured.
                for (Choice Profile::*c : choices) {
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
        return selected(base, false, options, ratios);
    }
    if (own_pair != nullptr) { return selected(*own_pair, false, options); }
    if (!home.empty()) { return selected(*home.front(), false, options); }
    // No lab card of this generation and no facts: the median round cost per configuration of
    // every lab card that measured the target, said as a transfer.
    const auto away  = one_per_card(current);
    const auto costs = median_round_costs(away);
    if (note != nullptr) {
        const std::string why =
            gpu == nullptr               ? std::string("no card rates")
            : card_rates(*gpu).known()   ? std::string("no round facts for this target")
            : generation.empty()         ? std::string("no facts for this card")
                                         : "no facts for " + generation_name(generation);
        *note = "transferred: median of " + std::to_string(away.size()) +
                " lab cards of other generations (" + why + ")";
    }
    const MeasuredPair& base = base_of(away, costs, nullptr);
    return selected(base, false, options, ratios_to(base, costs));
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

std::optional<Mode> drafter_kind_mode(std::string_view kind) {
    try {
        return mode_of_kind(kind);
    } catch (const std::invalid_argument&) { return std::nullopt; }
}

namespace {

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
        if (const auto mode = drafter_kind_mode(kind)) { carried.erase(*mode); }
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
        // Each record's facts price its own drafters' rounds: the pooled pair carries every mode's
        // (the same weights, so one target), whatever the records' order.
        if (p.round_facts) {
            if (!same->round_facts) {
                same->round_facts = p.round_facts;
            } else {
                for (const auto& [mode, d] : p.round_facts->drafters) {
                    same->round_facts->drafters.emplace(mode, d);
                }
            }
        }
        if (same->lab_board == 0) { same->lab_board = p.lab_board; }
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
                                      std::string_view repo, const GpuKey* gpu) {
    PairQuery query{.target = target, .engine = engine, .kinds = kinds,
                    .target_bits = target_bits, .attached = attached, .modes = {}};
    for (const auto& kind : kinds) {
        if (const auto mode = drafter_kind_mode(kind)) { query.modes.insert(*mode); }
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
    const auto priced = [](const std::string& note) { return note.empty() ? note : "; " + note; };
    if (const auto own = own_cards(pairs, query); !own.empty()) {
        std::string note;
        auto profile = profile_for_target(pointers(own), generation, served, gpu, &note);
        return PairChoice{.profile = std::move(profile),
                          .basis   = "measured",
                          .source  = "registry pair " + own.front().target_repo + " (" +
                                    std::to_string(own.size()) + " lab card(s))" + priced(note)};
    }
    // 2. Not measured, but another quantization of the same model was: the sibling nearest in
    // bits (the weights the same and quantized otherwise, so its drafters are accepted alike: no
    // fine-tune's guard). Of the same repository revision only: another commit may hold other
    // weights (a new version of the model, not another quantization of it).
    if (const auto at = repo.find('@'); at != std::string_view::npos && at + 1 < repo.size()) {
        if (const auto siblings = nearest_siblings(pairs, query, repo); !siblings.cards.empty()) {
            std::string named;
            for (const auto& q : siblings.quants) { named += (named.empty() ? "" : ", ") + q; }
            std::string note;
            auto profile = profile_for_target(pointers(siblings.cards), generation, served, gpu,
                                              &note);
            return PairChoice{.profile = std::move(profile),
                              .basis   = "preliminary",
                              .source  = "registry sibling " + std::string(repo.substr(0, at)) +
                                        " " + named + " (" + std::to_string(siblings.cards.size()) +
                                        " lab card(s); this quantization not measured)" +
                                        priced(note)};
        }
    }
    // 3. Nobody measured it: the group of each kind it carries, the fastest overall.
    std::optional<PairChoice> best;
    for (const auto& kind : kinds) {
        if (!drafter_kind_mode(kind)) { continue; }
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
