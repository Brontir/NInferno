#include "spec/profile.h"
#include "core/ascii.h"

#include "spec/registry.h"

#include "core/cache_file.h"
#include "core/units.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace ninferno::spec {

std::set<Mode> drafting_modes(const Profile& profile) {
    std::set<Mode> out;
    for (const Measurement& m : profile.measurements) {
        if (m.config.enabled()) { out.insert(m.config.mode); }
    }
    if (profile.measurements.empty()) {
        for (const Choice* c : {&profile.overall, &profile.thinking, &profile.non_thinking}) {
            if (c->config.enabled()) { out.insert(c->config.mode); }
        }
        if (profile.fit_fallback && profile.fit_fallback->config.enabled()) {
            out.insert(profile.fit_fallback->config.mode);
        }
    }
    return out;
}
namespace {

using Json = nlohmann::json;

Measurement& add_into(Measurement& total, const Measurement& m) {
    total.prompts += m.prompts;
    total.tokens += m.tokens;
    total.seconds += m.seconds;
    total.rounds += m.rounds;
    total.drafted += m.drafted;
    total.accepted += m.accepted;
    total.compared += m.compared;
    total.identical += m.identical;
    return total;
}

// Measurements summed per (configuration label, family).
std::map<std::pair<std::string, std::string>, Measurement>
aggregate(const std::vector<Measurement>& all) {
    std::map<std::pair<std::string, std::string>, Measurement> out;
    for (const auto& m : all) {
        auto& total = out[{m.config.label(), m.family}];
        if (total.prompts == 0 && total.tokens == 0) {
            total.config = m.config;
            total.family = m.family;
        }
        add_into(total, m);
    }
    return out;
}

double seconds_per_token(const Measurement& m) {
    return m.tokens > 0 ? m.seconds / double(m.tokens) : 0.0;
}

// A section's seconds per token weighted by its share of the mix (section_weight).
double weighted_cost(const Measurement& m) { return section_weight(m.family) * seconds_per_token(m); }

Choice plain_choice(double plain_tok_s) {
    Choice c;
    c.tok_s       = plain_tok_s;
    c.plain_tok_s = plain_tok_s;
    return c;
}

Json choice_json(const Choice& c) {
    return Json{{"config", c.config.label()},
                {"drafter", c.drafter},
                {"tok_s", c.tok_s},
                {"plain_tok_s", c.plain_tok_s},
                {"speedup", c.speedup()},
                {"tokens_per_round", c.tokens_per_round},
                {"acceptance", c.acceptance},
                {"floor_tokens_per_round", c.floor_tokens_per_round}};
}

Choice choice_from(const Json& j) {
    Choice c;
    c.config                 = SpecConfig::parse(j.at("config").get<std::string>());
    c.drafter                = j.value("drafter", std::string());
    c.tok_s                  = j.value("tok_s", 0.0);
    c.plain_tok_s            = j.value("plain_tok_s", 0.0);
    c.tokens_per_round       = j.value("tokens_per_round", 1.0);
    c.acceptance             = j.value("acceptance", 0.0);
    c.floor_tokens_per_round = j.value("floor_tokens_per_round", 0.0);
    return c;
}

Json served_json(const ServedChoices& s) {
    return Json{{"default", choice_json(s.overall)},
                {"non_thinking", choice_json(s.non_thinking)},
                {"thinking", choice_json(s.thinking)}};
}

// A published or local "served" (or "portable") object; absent, or not an object: none.
std::optional<ServedChoices> served_from(const Json& doc, const char* key = "served") {
    const auto it = doc.find(key);
    if (it == doc.end() || !it->is_object()) { return std::nullopt; }
    ServedChoices s;
    s.overall      = choice_from(it->at("default"));
    s.non_thinking = it->contains("non_thinking") ? choice_from((*it)["non_thinking"]) : s.overall;
    s.thinking     = it->contains("thinking") ? choice_from((*it)["thinking"]) : s.overall;
    return s;
}

// A published or local "load" table; absent: none.
std::vector<LoadBound> load_from(const Json& doc) {
    std::vector<LoadBound> out;
    const auto it = doc.find("load");
    if (it == doc.end()) { return out; }
    if (!it->is_array()) { throw std::runtime_error("spec profile: \"load\" is not a list"); }
    for (const auto& entry : *it) {
        LoadBound bound{.lanes = entry.at("lanes").get<std::int32_t>(),
                        .depth = entry.at("depth").get<std::int32_t>()};
        if (const auto width = entry.find("draft_width"); width != entry.end()) {
            const auto named = draft_width_from(width->get<std::string>());
            if (!named) {
                throw std::runtime_error("spec profile: a load bound's \"draft_width\" is block or "
                                         "drafts, got " + width->dump());
            }
            bound.width = *named;
        }
        out.push_back(bound);
    }
    // Validated as the text form is.
    return parse_load(load_label(out));
}

Json load_json(const std::vector<LoadBound>& load) {
    Json out = Json::array();
    for (const auto& b : load) {
        Json bound{{"lanes", b.lanes}, {"depth", b.depth}};
        if (b.width != DraftWidth::Default) { bound["draft_width"] = draft_width_name(b.width); }
        out.push_back(std::move(bound));
    }
    return out;
}

// A published or local "fit_fallback" choice; absent, or not an object: none.
std::optional<Choice> fit_fallback_from(const Json& doc) {
    const auto it = doc.find("fit_fallback");
    if (it == doc.end() || !it->is_object()) { return std::nullopt; }
    return choice_from(*it);
}

} // namespace

double section_weight(std::string_view family) {
    if (family == "code") { return 7.0; }
    if (family == "math") { return 4.0; }
    if (family == "prose") { return 12.0; }
    if (family == "thinking") { return 1.0; }
    return 0.0;
}

void select(Profile& profile, const SelectOptions& options) {
    const auto table = aggregate(profile.measurements);
    const auto get   = [&](const std::string& label, const std::string& family) -> const Measurement* {
        const auto it = table.find({label, family});
        return it == table.end() || it->second.tokens == 0 ? nullptr : &it->second;
    };
    std::set<std::string> families, labels;
    for (const auto& [key, m] : table) {
        labels.insert(key.first);
        if (key.first == "plain") { families.insert(key.second); }
    }
    const double gain = 1.0 + options.min_gain;

    // Drafters first: a mode whose drafter was rejected offers no configuration. A drafter is
    // judged on every section at once, each weighing the same (a wrong drafter loses on all of
    // them; one that loses only on prose still serves thinking), never on the one section it suits.
    std::set<std::string> rejected_modes;
    for (auto& d : profile.drafters) {
        double best = 0, drafted = 0, accepted = 0;
        bool measured = false;
        for (const auto& label : labels) {
            if (label == "plain" || mode_name(SpecConfig::parse(label).mode) != d.mode) {
                continue;
            }
            double cost = 0, plain_cost = 0;
            bool complete = !families.empty();
            for (const auto& f : families) {
                const auto* m = get(label, f);
                if (m == nullptr) {
                    complete = false;
                    continue;
                }
                measured = true;
                drafted += double(m->drafted);
                accepted += double(m->accepted);
                cost += seconds_per_token(*m);
                plain_cost += seconds_per_token(*get("plain", f));
            }
            if (complete && cost > 0) { best = std::max(best, plain_cost / cost); }
        }
        d.best_speedup = best;
        d.acceptance   = drafted > 0 ? accepted / drafted : 0.0;
        if (!measured) {
            d.status = "untested";
            d.reason = "no configuration measured within the time budget";
        } else if (best >= gain) {
            d.status = "accepted";
            d.reason.clear();
        } else {
            d.status = "rejected";
            std::ostringstream why;
            why.precision(3);
            why << "no configuration beat plain decode by " << options.min_gain * 100
                << "% over the mix (best x" << best << ", acceptance " << d.acceptance << ")";
            d.reason = why.str();
            rejected_modes.insert(d.mode);
        }
    }
    const auto usable = [&](const std::string& label) {
        if (label == "plain") { return true; }
        return !rejected_modes.contains(mode_name(SpecConfig::parse(label).mode));
    };

    // The defaults, each over its sections of the mix and the labels `allowed` admits. With
    // `within` above 0, the shallowest configuration (the fewest drafts, then the fastest) whose
    // speed is within that fraction of the best instead, with the best's drafter and proposal
    // head: another drafter would move the load table and the fit fallback measured with the
    // best's off the kind they were measured for (spec::select_load_fallback matches the kind).
    const auto choose = [&](const std::function<bool(const std::string&)>& allowed,
                            double within) {
        ServedChoices out;
        // Whether `a` is the shallower of two candidates within reach of the best (then faster).
        const auto shallower = [](std::int32_t a_depth, double a_rate, std::int32_t b_depth,
                                  double b_rate) {
            return a_depth != b_depth ? a_depth < b_depth : a_rate > b_rate;
        };
        const auto same_drafter = [](const SpecConfig& a, const SpecConfig& b) {
            return a.mode == b.mode && a.proposal_head == b.proposal_head;
        };
        // A default over corpus sections, each weighted by its share of the mix (section_weight).
        const auto group = [&](const std::vector<std::string>& wanted) {
            std::vector<std::string> members;
            for (const auto& f : wanted) {
                if (families.contains(f)) { members.push_back(f); }
            }
            Choice best;
            if (members.empty()) { return best; }
            double plain_cost = 0, weight = 0;
            for (const auto& f : members) {
                plain_cost += weighted_cost(*get("plain", f));
                weight += section_weight(f);
            }
            best = plain_choice(weight / plain_cost);
            struct Candidate {
                SpecConfig config;
                double cost = 0, seconds = 0, rounds = 0, drafted = 0, accepted = 0;
            };
            std::vector<Candidate> candidates;
            for (const auto& label : labels) {
                if (label == "plain" || !usable(label) || !allowed(label)) { continue; }
                Candidate c{.config = SpecConfig::parse(label)};
                bool complete = true;
                for (const auto& f : members) {
                    const auto* m = get(label, f);
                    // A default runs every section it covers: one that loses to plain decode on
                    // any of them is not a candidate, however much it wins on the others.
                    if (m == nullptr || m->tok_s() < get("plain", f)->tok_s()) {
                        complete = false;
                        break;
                    }
                    c.cost += weighted_cost(*m);
                    c.seconds += m->seconds;
                    c.rounds += double(m->rounds);
                    c.drafted += double(m->drafted);
                    c.accepted += double(m->accepted);
                }
                if (complete) { candidates.push_back(c); }
            }
            // The cheapest over plain decode's cost less the gain (the last of equals).
            double best_cost        = plain_cost / gain;
            const Candidate* chosen = nullptr;
            for (const Candidate& c : candidates) {
                if (c.cost <= best_cost) {
                    best_cost = c.cost;
                    chosen    = &c;
                }
            }
            if (chosen == nullptr) { return best; }
            if (within > 0) {
                const Candidate* cheapest = chosen;
                const double reach = std::min(best_cost / (1.0 - within), plain_cost / gain);
                for (const Candidate& c : candidates) {
                    if (c.cost <= reach && same_drafter(c.config, cheapest->config) &&
                        shallower(c.config.k, 1.0 / c.cost, chosen->config.k,
                                  1.0 / chosen->cost)) {
                        chosen = &c;
                    }
                }
            }
            best.config           = chosen->config;
            best.tok_s            = weight / chosen->cost;
            best.tokens_per_round = chosen->rounds > 0
                                        ? (chosen->accepted + chosen->rounds) / chosen->rounds
                                        : 1.0;
            best.acceptance = chosen->drafted > 0 ? chosen->accepted / chosen->drafted : 0.0;
            best.floor_tokens_per_round =
                chosen->rounds > 0
                    ? (chosen->seconds / chosen->rounds) / (plain_cost / weight)
                    : 0.0;
            return best;
        };
        out.overall      = group({"math", "code", "prose", "thinking"});
        out.non_thinking = group({"math", "code", "prose"});
        out.thinking     = group({"thinking"});

        // Name the drafter each choice runs.
        const auto name_drafter = [&](Choice& c) {
            c.drafter.clear();
            if (!c.config.enabled()) { return; }
            for (const auto& d : profile.drafters) {
                if (d.mode == mode_name(c.config.mode)) {
                    c.drafter = d.name.empty() ? d.kind : d.name;
                    return;
                }
            }
            c.drafter = mode_name(c.config.mode);
        };
        name_drafter(out.overall);
        name_drafter(out.non_thinking);
        name_drafter(out.thinking);
        return out;
    };

    const ServedChoices all = choose([](const std::string&) { return true; }, 0.0);
    profile.overall         = all.overall;
    profile.non_thinking    = all.non_thinking;
    profile.thinking        = all.thinking;
    profile.served.reset();
    if (options.served) {
        const EngineSupport support = *options.served;
        const auto runs             = [&](const Choice& c) { return support.runs(c.config); };
        const bool every = runs(all.overall) && runs(all.non_thinking) && runs(all.thinking);
        if (!every) {
            profile.served = choose(
                [&](const std::string& label) { return support.runs(SpecConfig::parse(label)); },
                0.0);
        }
    }
    // The portable choices, over what the serving Engine runs when it is known.
    profile.portable.reset();
    if (options.portable_within > 0) {
        profile.portable = choose(
            [&](const std::string& label) {
                return !options.served || options.served->runs(SpecConfig::parse(label));
            },
            options.portable_within);
    }
}

Json to_json(const Profile& p) {
    Json drafters = Json::array();
    for (const auto& d : p.drafters) {
        drafters.push_back(Json{{"name", d.name},
                                {"kind", d.kind},
                                {"mode", d.mode},
                                {"artifact", d.artifact},
                                {"artifact_id", d.artifact_id},
                                {"repo", d.repo},
                                {"status", d.status},
                                {"reason", d.reason},
                                {"best_speedup", d.best_speedup},
                                {"acceptance", d.acceptance}});
    }
    Json measurements = Json::array();
    for (const auto& m : p.measurements) {
        measurements.push_back(Json{{"config", m.config.label()},
                                    {"family", m.family},
                                    {"prompts", m.prompts},
                                    {"tokens", m.tokens},
                                    {"seconds", m.seconds},
                                    {"tok_s", m.tok_s()},
                                    {"rounds", m.rounds},
                                    {"drafted", m.drafted},
                                    {"accepted", m.accepted},
                                    {"tokens_per_round", m.tokens_per_round()},
                                    {"acceptance", m.acceptance()},
                                    {"compared", m.compared},
                                    {"identical", m.identical}});
    }
    Json out{{"schema", 1},
             {"key",
              {{"artifact_id", p.key.artifact_id},
               {"drafters", p.key.drafters},
               {"gpu", p.key.gpu},
               {"engine", p.key.engine}}},
             {"artifact", p.artifact},
             {"created", p.created},
             {"settings", p.settings},
             {"tune_seconds", p.tune_seconds},
             {"default", choice_json(p.overall)},
             {"non_thinking", choice_json(p.non_thinking)},
             {"thinking", choice_json(p.thinking)},
             {"drafters", drafters},
             {"skipped", p.skipped},
             {"measurements", measurements}};
    if (p.served) { out["served"] = served_json(*p.served); }
    if (p.portable) { out["portable"] = served_json(*p.portable); }
    if (p.fit_fallback) { out["fit_fallback"] = choice_json(*p.fit_fallback); }
    if (!p.load.empty()) { out["load"] = load_json(p.load); }
    if (p.context_limit != 0) { out["context_limit"] = p.context_limit; }
    if (p.disabled) {
        out["disabled"] = Json{{"reason", p.disabled->reason}, {"when", p.disabled->when}};
    }
    return out;
}

Profile profile_from_json(const Json& doc) {
    if (doc.value("schema", 0) != 1) { throw std::runtime_error("spec profile: unknown schema"); }
    Profile p;
    const auto& key = doc.at("key");
    p.key.artifact_id = key.value("artifact_id", std::string());
    p.key.drafters    = key.value("drafters", std::string());
    p.key.gpu         = key.value("gpu", std::string());
    p.key.engine      = key.value("engine", std::string());
    p.artifact        = doc.value("artifact", std::string());
    p.created         = doc.value("created", std::string());
    p.settings        = doc.value("settings", std::string());
    p.tune_seconds    = doc.value("tune_seconds", 0.0);
    p.overall         = choice_from(doc.at("default"));
    p.non_thinking    = choice_from(doc.at("non_thinking"));
    p.thinking        = choice_from(doc.at("thinking"));
    for (const auto& d : doc.value("drafters", Json::array())) {
        DrafterRecord r;
        r.name         = d.value("name", std::string());
        r.kind         = d.value("kind", std::string());
        r.mode         = d.value("mode", std::string());
        r.artifact     = d.value("artifact", std::string());
        r.artifact_id  = d.value("artifact_id", std::string());
        r.repo         = d.value("repo", std::string());
        r.status       = d.value("status", std::string());
        r.reason       = d.value("reason", std::string());
        r.best_speedup = d.value("best_speedup", 0.0);
        r.acceptance   = d.value("acceptance", 0.0);
        p.drafters.push_back(std::move(r));
    }
    for (const auto& m : doc.value("measurements", Json::array())) {
        Measurement x;
        x.config    = SpecConfig::parse(m.at("config").get<std::string>());
        x.family    = m.value("family", std::string());
        x.prompts   = m.value("prompts", std::int64_t{0});
        x.tokens    = m.value("tokens", std::int64_t{0});
        x.seconds   = m.value("seconds", 0.0);
        x.rounds    = m.value("rounds", std::int64_t{0});
        x.drafted   = m.value("drafted", std::int64_t{0});
        x.accepted  = m.value("accepted", std::int64_t{0});
        x.compared  = m.value("compared", std::int64_t{0});
        x.identical = m.value("identical", std::int64_t{0});
        p.measurements.push_back(std::move(x));
    }
    p.skipped = doc.value("skipped", std::vector<std::string>{});
    p.served   = served_from(doc);
    p.portable = served_from(doc, "portable");
    p.fit_fallback = fit_fallback_from(doc);
    p.load         = load_from(doc);
    p.context_limit = doc.value("context_limit", 0U);
    if (const auto it = doc.find("disabled"); it != doc.end() && it->is_object()) {
        p.disabled = RuntimeDisable{it->value("reason", std::string()),
                                    it->value("when", std::string())};
    }
    return p;
}

std::filesystem::path default_profile_path(const std::filesystem::path& artifact) {
    return std::filesystem::path(artifact.string() + ".spec_profile.json");
}

std::optional<Profile> read_profile(const std::filesystem::path& path, std::string* error) {
    std::ifstream stream(path);
    if (!stream) {
        if (error) { *error = "no profile at " + path.string(); }
        return std::nullopt;
    }
    try {
        return profile_from_json(Json::parse(stream));
    } catch (const std::exception& e) {
        if (error) { *error = path.string() + ": " + e.what(); }
        return std::nullopt;
    }
}

void write_json_file(const std::filesystem::path& path, const Json& doc) {
    write_file_atomic(path, doc.dump(1) + '\n');
}

void write_profile(const std::filesystem::path& path, const Profile& profile) {
    write_json_file(path, to_json(profile));
}

std::string key_mismatch(const ProfileKey& recorded, const ProfileKey& expected) {
    if (recorded.artifact_id != expected.artifact_id) { return "artifact id"; }
    if (recorded.drafters != expected.drafters) { return "drafters"; }
    if (recorded.gpu != expected.gpu) { return "GPU"; }
    if (recorded.engine != expected.engine) { return "engine version"; }
    return {};
}

std::string drafter_key(std::vector<std::string> ids) {
    std::sort(ids.begin(), ids.end());
    std::string out;
    for (const auto& id : ids) { out += (out.empty() ? "" : ",") + id; }
    return out;
}

const Choice& pick(const Profile& profile, std::optional<bool> thinking) {
    static const Choice plain;
    if (profile.disabled) { return plain; }
    if (thinking) { return *thinking ? profile.thinking : profile.non_thinking; }
    return profile.overall;
}

double per_request_gain(const Profile& profile, bool thinking_default) {
    const Choice& fixed = pick(profile, thinking_default);
    const Choice& own   = pick(profile, !thinking_default);
    if (fixed.config.label() == own.config.label()) { return 1.0; }
    const auto table = aggregate(profile.measurements);
    const std::vector<std::string> sections =
        thinking_default ? std::vector<std::string>{"math", "code", "prose"}
                         : std::vector<std::string>{"thinking"};
    // A section's seconds per token at a configuration; 0 when not measured.
    const auto cost = [&](const SpecConfig& config) {
        double out = 0;
        for (const auto& f : sections) {
            const auto it = table.find({config.enabled() ? config.label() : std::string("plain"), f});
            if (it == table.end() || it->second.tokens == 0) { return 0.0; }
            out += weighted_cost(it->second);
        }
        return out;
    };
    const double at_fixed = cost(fixed.config);
    const double at_own   = cost(own.config);
    return at_fixed > 0 && at_own > 0 ? at_fixed / at_own : 1.0;
}

bool memory_in_range(std::string_view range, std::uint64_t memory_bytes) {
    if (range.empty()) { return true; }
    if (memory_bytes == 0) { return false; }
    const auto op_end = range.find_first_not_of("<>=");
    const auto op     = range.substr(0, op_end);
    auto rest         = std::string(ascii_lower(std::string(range.substr(op_end))));
    if (!rest.ends_with("gib")) { return false; }
    rest.resize(rest.size() - 3);
    char* end          = nullptr;
    const double gib   = std::strtod(rest.c_str(), &end);
    if (end == rest.c_str() || *end != '\0') { return false; }
    const double value = to_gib(memory_bytes);
    if (op == ">=") { return value >= gib; }
    if (op == ">") { return value > gib; }
    if (op == "<=") { return value <= gib; }
    if (op == "<") { return value < gib; }
    return false;
}

const Json* select_published(const Json& profiles, const GpuKey& gpu, std::string_view repo,
                             const std::vector<std::string>* available) {
    if (!profiles.is_array()) { return nullptr; }
    const auto applies = [&](const Json& p, bool any) {
        if (!memory_in_range(p.value("mem", std::string()), gpu.memory_bytes)) { return false; }
        for (const auto& need : p.value("requires", Json::array())) {
            if (!gpu.nvfp4 && need.is_string() &&
                split_variant(need.get<std::string>()).second == "nvfp4") {
                return false;
            }
        }
        if (available != nullptr) {
            for (const auto& need : p.value("requires", Json::array())) {
                if (!need.is_string() ||
                    std::find(available->begin(), available->end(), need.get<std::string>()) ==
                        available->end()) {
                    return false;
                }
            }
        }
        const auto list = p.value("applies_to", Json::array());
        if (list.empty()) { return any; }
        if (any) { return false; } // a repository-specific profile is not "any"
        for (const auto& item : list) {
            if (item.is_string() && !repo.empty() &&
                ascii_lower(item.get<std::string>()) == ascii_lower(std::string(repo))) {
                return true;
            }
        }
        return false;
    };
    // Within one class and repository level, the profile that uses the most of the attached
    // drafters: an artifact that carries DFlash2 takes the profile measured with it, whatever
    // the order the registry lists the profiles in (ties: the first listed).
    for (const auto& cls : gpu.chain) {
        for (const bool any : {false, true}) {
            const Json* best       = nullptr;
            std::size_t best_needs = 0;
            for (const auto& p : profiles) {
                if (p.value("gpu_class", std::string()) != cls || !applies(p, any)) { continue; }
                const auto needs = p.value("requires", Json::array()).size();
                if (best == nullptr || needs > best_needs) {
                    best       = &p;
                    best_needs = needs;
                }
            }
            if (best != nullptr) { return best; }
        }
    }
    return nullptr;
}

std::string similarity_note(const GpuKey& gpu, const Json& published) {
    const auto cls = published.value("gpu_class", std::string());
    if (cls == "*" || gpu.own_keys == 0) { return {}; }
    const auto own_end =
        gpu.chain.begin() + static_cast<std::ptrdiff_t>(std::min(gpu.own_keys, gpu.chain.size()));
    if (std::find(gpu.chain.begin(), own_end, cls) != own_end) { return {}; }
    return "the " + cls + " profile stands in for " +
           (gpu.chain.empty() ? std::string("this GPU") : gpu.chain.front()) +
           ": no profile of its own class; " + cls + " is the most similar GPU measured";
}

bool measured_elsewhere(const GpuKey& gpu, const Json& published) {
    if (gpu.own_keys == 0) { return false; }
    const auto cls = published.value("gpu_class", std::string());
    if (cls == "*") { return true; }
    const auto own_end =
        gpu.chain.begin() + static_cast<std::ptrdiff_t>(std::min(gpu.own_keys, gpu.chain.size()));
    return std::find(gpu.chain.begin(), own_end, cls) == own_end;
}

std::vector<LoadBound> stand_in_load(std::vector<LoadBound> load, int sms) {
    if (sms <= 0) { return load; }
    const std::int32_t cap = std::max<std::int32_t>(2, (sms + 9) / 10);
    for (const LoadBound& bound : load) {
        if (bound.depth == 0 && bound.lanes <= cap) { return load; }
    }
    std::vector<LoadBound> out;
    for (const LoadBound& bound : load) {
        if (bound.lanes < cap) { out.push_back(bound); }
    }
    out.push_back(LoadBound{cap, 0});
    return out;
}

const Json* select_load_fallback(const Json& profiles, const Json& selected) {
    if (!profiles.is_array() || selected.contains("load")) { return nullptr; }
    const auto requires_of = [](const Json& p) {
        std::vector<std::string> out;
        for (const auto& item : p.value("requires", Json::array())) {
            if (item.is_string()) { out.push_back(item.get<std::string>()); }
        }
        std::sort(out.begin(), out.end());
        return out;
    };
    // The drafter kind of a profile's default configuration ("mtp:5+head" -> "mtp").
    const auto kind_of = [](const Json& p) {
        const auto defaults = p.value("default", Json::object());
        const std::string config =
            defaults.is_object() ? defaults.value("config", std::string()) : std::string();
        return config.substr(0, config.find(':'));
    };
    const auto wanted = requires_of(selected);
    const auto kind   = kind_of(selected);
    const auto gpu    = selected.value("gpu_class", std::string());
    const Json* star  = nullptr;
    for (const auto& p : profiles) {
        if (&p == &selected || !p.contains("load") || kind_of(p) != kind) { continue; }
        const auto cls = p.value("gpu_class", std::string());
        if (cls == gpu && gpu != "*" && requires_of(p) == wanted) { return &p; }
        if (star == nullptr && cls == "*" && p.value("applies_to", Json::array()).empty() &&
            requires_of(p) == wanted) {
            star = &p;
        }
    }
    return star;
}

Profile profile_from_published(const Json& published) {
    Profile p;
    p.overall      = choice_from(published.at("default"));
    p.non_thinking = published.contains("non_thinking") ? choice_from(published["non_thinking"])
                                                        : p.overall;
    p.thinking = published.contains("thinking") ? choice_from(published["thinking"]) : p.overall;
    p.settings = published.value("measured_on", std::string());
    p.served   = served_from(published);
    p.fit_fallback = fit_fallback_from(published);
    p.load         = load_from(published);
    p.context_limit = published.value("context_limit", 0U);
    return p;
}

Json published_from_profile(const Profile& profile, const std::string& gpu_class,
                            const std::string& gpu_name, const std::vector<std::string>& applies_to,
                            const std::string& measured_on) {
    Json out{{"gpu_class", gpu_class},
             {"gpu", gpu_name},
             {"applies_to", applies_to},
             {"measured_on", measured_on},
             {"source", "spec-tune " + profile.created + " (" + profile.key.engine + ", " +
                            std::to_string(static_cast<int>(profile.tune_seconds)) + " s)"},
             {"default", choice_json(profile.overall)},
             {"non_thinking", choice_json(profile.non_thinking)},
             {"thinking", choice_json(profile.thinking)}};
    if (profile.served) { out["served"] = served_json(*profile.served); }
    if (profile.portable) { out["portable"] = served_json(*profile.portable); }
    if (profile.fit_fallback) { out["fit_fallback"] = choice_json(*profile.fit_fallback); }
    if (!profile.load.empty()) { out["load"] = load_json(profile.load); }
    if (profile.context_limit != 0) { out["context_limit"] = profile.context_limit; }
    // A choice of a drafter variant ("<name>.<variant>": NInferno's own quantization, built beside
    // the published drafter) requires it: where the model has it not attached, select_published
    // takes the class's profile of the published drafters, never one naming a drafter it lacks.
    Json required = Json::array();
    const auto require = [&](const Choice& c) {
        if (c.config.enabled() && !split_variant(c.drafter).second.empty() &&
            std::find(required.begin(), required.end(), c.drafter) == required.end()) {
            required.push_back(c.drafter);
        }
    };
    const auto require_all = [&](const auto& choices) {
        require(choices.overall);
        require(choices.non_thinking);
        require(choices.thinking);
    };
    require_all(profile);
    if (profile.served) { require_all(*profile.served); }
    if (profile.portable) { require_all(*profile.portable); }
    if (profile.fit_fallback) { require(*profile.fit_fallback); }
    if (!required.empty()) { out["requires"] = std::move(required); }
    return out;
}

std::string_view draft_width_name(DraftWidth width) noexcept {
    switch (width) {
    case DraftWidth::Block:
        return "block";
    case DraftWidth::Drafts:
        return "drafts";
    default:
        return "";
    }
}

std::optional<DraftWidth> draft_width_from(std::string_view name) noexcept {
    if (name == "block") { return DraftWidth::Block; }
    if (name == "drafts") { return DraftWidth::Drafts; }
    return std::nullopt;
}

std::vector<LoadBound> parse_load(std::string_view text) {
    std::vector<LoadBound> out;
    if (text.empty() || text == "none") { return out; }
    const auto refuse = [&] {
        return std::invalid_argument("speculation load: expected LANES:DEPTH[:block|drafts],... "
                                     "with lanes ascending from 2, depths in [0,15], none but 0 "
                                     "after a 0 and no width at 0, got '" +
                                     std::string(text) + "'");
    };
    const auto number = [&](std::string_view item) {
        std::int32_t value = 0;
        if (item.empty() || item.size() > 3) { throw refuse(); }
        for (const char c : item) {
            if (c < '0' || c > '9') { throw refuse(); }
            value = value * 10 + (c - '0');
        }
        return value;
    };
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end   = std::min(text.find(',', start), text.size());
        const auto item  = text.substr(start, end - start);
        const auto colon = item.find(':');
        if (colon == std::string_view::npos) { throw refuse(); }
        const auto second = item.find(':', colon + 1);
        LoadBound bound{.lanes = number(item.substr(0, colon)),
                        .depth = number(item.substr(colon + 1, second - colon - 1))};
        if (second != std::string_view::npos) {
            const auto width = draft_width_from(item.substr(second + 1));
            if (!width || bound.depth == 0) { throw refuse(); }
            bound.width = *width;
        }
        if (bound.lanes < 2 || bound.depth > 15 ||
            (!out.empty() && (bound.lanes <= out.back().lanes ||
                              (out.back().depth == 0 && bound.depth != 0)))) {
            throw refuse();
        }
        out.push_back(bound);
        start = end + 1;
    }
    return out;
}

std::vector<LoadBound> load_table(const std::vector<LoadChoice>& choices, std::int32_t window) {
    std::vector<LoadBound> out;
    std::int32_t previous = 1;
    for (const LoadChoice& choice : choices) {
        if (choice.concurrency <= previous) {
            throw std::invalid_argument("spec-tune --load: concurrencies must ascend from 2");
        }
        // Plain rows draft at no width.
        const LoadBound bound{.lanes = previous + 1,
                              .depth = choice.depth,
                              .width = choice.depth == 0 ? DraftWidth::Default : choice.width};
        previous = choice.concurrency;
        if (out.empty() ? bound.depth == window && bound.width == DraftWidth::Default
                        : (out.back().depth == bound.depth && out.back().width == bound.width) ||
                              out.back().depth == 0) {
            continue;
        }
        out.push_back(bound);
    }
    return out;
}

std::string load_label(const std::vector<LoadBound>& load) {
    if (load.empty()) { return "none"; }
    std::string out;
    for (const auto& b : load) {
        out += (out.empty() ? "" : ",") + std::to_string(b.lanes) + ":" + std::to_string(b.depth);
        if (b.width != DraftWidth::Default) { out += ":" + std::string(draft_width_name(b.width)); }
    }
    return out;
}

Json reselect(Profile& profile, const ReselectOptions& options) {
    if (profile.measurements.empty()) {
        throw std::invalid_argument("spec-reselect: the profile keeps no measurements");
    }
    if (options.gpu_class.empty()) {
        throw std::invalid_argument("spec-reselect: no GPU class");
    }
    if (profile.served && !options.served) {
        throw std::invalid_argument(
            "spec-reselect: the saved profile has served choices (the serving Engine runs less "
            "than the tune measured): name the Engine's model type (--engine)");
    }
    select(profile, {.served = options.served});
    return published_from_profile(profile, options.gpu_class, options.gpu_name,
                                  options.applies_to, options.measured_on);
}

} // namespace ninferno::spec
