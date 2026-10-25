#include "product/speculation/speculation.h"

#include "artifact/reader.h"
#include "spec/fingerprint.h"
#include "spec/gpu_identity.h"
#include "spec/manifest.h"
#include "spec/registry.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace ninferno::product {

SpeculativeBackend backend_of(spec::Mode mode) noexcept {
    switch (mode) {
    case spec::Mode::Mtp:
        return SpeculativeBackend::Mtp;
    case spec::Mode::DFlash:
        return SpeculativeBackend::DFlash;
    case spec::Mode::DFlash2:
        return SpeculativeBackend::DFlash2;
    default:
        return SpeculativeBackend::None;
    }
}

bool has_proposal_head(const std::filesystem::path& artifact) {
    try {
        const artifact::Reader reader(artifact);
        const auto& components = reader.directory().components;
        const auto text        = components.find("text");
        return text != components.end() && text->second.proposal.has_value();
    } catch (const std::exception&) {
        return false;
    }
}

double watchdog_floor(double measured) noexcept { return measured > 1.0 ? measured : 1.2; }

namespace {

// What the decision fingerprint names the resolved speculation by (SpeculativeOptions::
// decision_sources): the profile's file and the drafter registry, by content.
std::map<std::string, std::string> decision_sources_of(const spec::ResolvedSpeculation& resolved) {
    std::map<std::string, std::string> out;
    const bool local = resolved.source.starts_with("local tune ");
    out["speculation.profile"] =
        resolved.sha256.empty()
            ? std::string(local               ? "local tune unreadable"
                          : resolved.found() ? "built-in default"
                                             : "none")
            : std::string(local ? "local tune" : "published") + " sha256:" + resolved.sha256;
    // A registry file only when --drafter-registry named one (spec::Registry::use_file), else the
    // compiled-in one.
    try {
        out["drafter_registry"] = std::string(spec::Registry::file().empty() ? "builtin" : "file") +
                                  " sha256:" + spec::Registry::builtin().source_sha256();
    } catch (const std::exception&) { out["drafter_registry"] = "unreadable"; }
    return out;
}

// The floor a configuration runs at under a profile measured elsewhere (spec::foreign_floor).
// Where the artifact has no proposal head a "+head" choice drafts with the full head, a dearer
// round, so the full head's ceiling counts too, under a profile of this GPU as well
// (spec::full_head_floor); `notes` (when given) gets one note for that raise.
double ran_floor(const spec::ResolvedSpeculation& resolved, const spec::SpecConfig& config,
                 bool head_here, double own, std::vector<std::string>* notes) {
    double floor = spec::foreign_floor(resolved, config, own);
    if (config.proposal_head && !head_here) {
        const double headless = spec::full_head_floor(resolved, config, own);
        if (headless > floor && notes) {
            char text[96];
            std::snprintf(text, sizeof(text), "%.2f -> %.2f", floor, headless);
            std::string note = config.label() + ": watchdog floor " + text +
                               " tokens a round (the full head's: the artifact has no proposal "
                               "head)";
            if (std::find(notes->begin(), notes->end(), note) == notes->end()) {
                notes->push_back(std::move(note));
            }
        }
        floor = std::max(floor, headless);
    }
    return floor;
}

bool drafter_present(const spec::Choice& choice, const spec::ResolvedSpeculation& resolved) {
    return backend_of(choice.config.mode) != SpeculativeBackend::None &&
           resolved.drafters.contains(choice.drafter.empty() ? spec::mode_name(choice.config.mode)
                                                             : choice.drafter);
}

// The drafter a choice runs when it ships as an artifact of its own (Gemma 4's assistant, the
// NemotronH MTP head): its path. Empty when the drafter is a component of the target (Qwen's MTP
// head and DFlash drafters).
std::filesystem::path separate_drafter(const spec::Choice& choice,
                                       const spec::ResolvedSpeculation& resolved) {
    const auto it = resolved.drafters.find(
        choice.drafter.empty() ? spec::mode_name(choice.config.mode) : choice.drafter);
    if (it == resolved.drafters.end() || it->second.starts_with("component:")) { return {}; }
    return it->second;
}

// The speculation backend a manifest drafter kind drafts for (NemotronH's MTP head is a separate
// drafter artifact that drafts through the Mtp backend, like Gemma 4's assistant).
SpeculativeBackend backend_of_kind(std::string_view kind) noexcept {
    if (kind == "mtp" || kind == "gemma4_assistant" || kind == "nemotron_h_mtp") {
        return SpeculativeBackend::Mtp;
    }
    if (kind == "dflash") { return SpeculativeBackend::DFlash; }
    if (kind == "dflash2") { return SpeculativeBackend::DFlash2; }
    return SpeculativeBackend::None;
}

// A model whose drafter is built in but which has no profile for this GPU (none tuned here, none
// published along its similarity chain, no manifest at all): its drafter's default rather than
// plain decode, with the watchdog floors raised to the registry's (raise_foreign_floors, as for a
// profile measured on another GPU) so a configuration that does not pay here stops. MTP 3 (with the proposal head when
// the artifact has one), else DFlash2 or DFlash 7. (RTX 5090: Qwen3.8 27B IQ3_S converted
// from the HF GGUF with MTP and DFlash2 decoded 93 tok/s plain under --spec auto, 192 with MTP 3
// and its head.) Nothing when the artifact carries no drafter.
void default_for_builtin_drafter(const std::set<std::string>& components, bool proposal_head,
                                 spec::ResolvedSpeculation& resolved) {
    spec::Choice choice;
    if (components.contains("mtp")) {
        choice.config = {.mode = spec::Mode::Mtp, .k = 3};
        // From eight rows on, two drafts: Qwen3.6 35B on the RTX 5090 at C=8 (both
        // orders) 1250-1258 tok/s at depth 2 against 1131-1158 at depth 3.
        resolved.profile.load = {{.lanes = 8, .depth = 2}};
    } else if (components.contains("dflash2")) {
        choice.config = {.mode = spec::Mode::DFlash2, .k = 7};
        // From two rows on, three drafts: Qwen3.8 27B Q3 (HF GGUF conversion) on the RTX 4080
        // SUPER, no profile for the chip (both orders): C=4 depth 7 131 tok/s, plain 153,
        // depth 5 155-160, depth 3 214-218; C=2 depth 7 149-158, depth 3 164-173.
        resolved.profile.load = {{.lanes = 2, .depth = 3}};
    } else if (components.contains("dflash")) {
        choice.config = {.mode = spec::Mode::DFlash, .k = 7};
    } else {
        return;
    }
    choice.config.proposal_head       = proposal_head;
    choice.drafter                    = spec::mode_name(choice.config.mode);
    resolved.drafters[choice.drafter] = "component:" + choice.drafter;
    resolved.profile.overall          = choice;
    resolved.profile.thinking         = choice;
    resolved.profile.non_thinking     = choice;
    resolved.foreign_floors           = true;
    resolved.basis                    = "builtin";
    resolved.source                   = "the built-in " + choice.drafter + " drafter's default (" +
                      choice.config.label() + ", untuned on this GPU: spec-tune measures one)";
    spec::raise_foreign_floors(resolved.profile, resolved, resolved.notes);
}

std::set<std::string> artifact_components(const std::filesystem::path& artifact) {
    std::set<std::string> components;
    try {
        const artifact::Reader reader(artifact);
        for (const auto& [name, component] : reader.directory().components) {
            components.insert(name);
        }
    } catch (const std::exception&) {}
    return components;
}

std::optional<spec::ResolvedSpeculation> resolve(const std::filesystem::path& artifact,
                                                 const spec::GpuKey& gpu,
                                                 const std::filesystem::path& local_profile) {
    std::string id;
    try {
        id = spec::artifact_id_hex(artifact);
    } catch (const std::exception&) {
        if (!local_profile.empty()) {
            throw std::runtime_error("refusing the named speculation tune " +
                                     local_profile.string() + ": the artifact's id is unreadable");
        }
        return std::nullopt;
    }
    auto resolved = spec::resolve_speculation(artifact, id, gpu, local_profile);
    spec::require_named_tune(resolved);
    if (!resolved.found()) {
        default_for_builtin_drafter(artifact_components(artifact), has_proposal_head(artifact),
                                    resolved);
    }
    return resolved;
}

std::optional<spec::ResolvedSpeculation> resolve(const std::filesystem::path& artifact, int device,
                                                 const std::filesystem::path& local_profile) {
    return resolve(artifact, spec::gpu_identity(device).key, local_profile);
}

// The choices a product of the Engine runs: the profile's served choices when its tune recorded
// them (it chose configurations the Engine cannot run), else its own.
spec::Profile engine_choices(const spec::Profile& profile, std::vector<std::string>* notes) {
    if (!profile.served) { return profile; }
    spec::Profile out  = profile;
    out.overall        = profile.served->overall;
    out.non_thinking   = profile.served->non_thinking;
    out.thinking       = profile.served->thinking;
    if (notes) {
        notes->push_back("the profile's served choices: the tune's best configurations the "
                         "Engine runs (it chose " +
                         profile.overall.config.label() + " by default)");
    }
    return out;
}

} // namespace

spec::EngineSupport engine_support(const std::filesystem::path& artifact) {
    try {
        const artifact::Reader reader(artifact);
        const auto& components = reader.directory().components;
        const auto text        = components.find("text");
        if (text == components.end() || !text->second.config.is_object()) { return {}; }
        return spec::engine_support(text->second.config.value("model_type", std::string()));
    } catch (const std::exception&) {
        return {};
    }
}

SpeculativeOptions with_given_settings(SpeculativeOptions picked, const SpeculativeOptions& given) {
    picked.watchdog_window     = given.watchdog_window;
    picked.watchdog_min_rounds = given.watchdog_min_rounds;
    picked.shared_trip_floor   = given.shared_trip_floor;
    if (picked.backend == SpeculativeBackend::Mtp) { // copy rounds ride MTP
        picked.ngram_draft_tokens = given.ngram_draft_tokens;
        picked.ngram_min_match    = given.ngram_min_match;
    }
    return picked;
}

std::optional<std::uint32_t> profile_depth(const SpeculationSubject& subject,
                                           SpeculativeBackend backend, bool thinking,
                                           std::string* source) {
    const auto& resolved = subject.resolved;
    if (!resolved || !resolved->found() || backend == SpeculativeBackend::None) { return std::nullopt; }
    const spec::Profile profile = engine_choices(resolved->profile, nullptr);
    const spec::Choice& choice  = spec::pick(profile, thinking);
    if (!choice.config.enabled() || choice.config.k <= 0 || backend_of(choice.config.mode) != backend ||
        !subject.support.runs(choice.config)) {
        return std::nullopt;
    }
    if (source) { *source = resolved->source + ": " + choice.config.label(); }
    return static_cast<std::uint32_t>(choice.config.k);
}

double shallower_floor(const std::vector<std::pair<int, double>>& floors, int depth) {
    const std::pair<int, double>* below = nullptr;
    const std::pair<int, double>* above = nullptr;
    for (const auto& point : floors) {
        if (point.second <= 1.0 || point.first <= 0) { continue; }
        if (point.first == depth) { return point.second; }
        if (point.first < depth && (below == nullptr || point.first > below->first)) {
            below = &point;
        }
        if (point.first > depth && (above == nullptr || point.first < above->first)) {
            above = &point;
        }
    }
    if (below != nullptr && above != nullptr) {
        const double t = double(depth - below->first) / double(above->first - below->first);
        return below->second + t * (above->second - below->second);
    }
    const auto* nearest = above != nullptr ? above : below;
    if (nearest == nullptr) { return 0.0; }
    return 1.0 + (nearest->second - 1.0) * double(depth) / double(nearest->first);
}

namespace {

// A server's speculation for `profile`'s choices, with `primary` as the engine's default.
ServedSpeculation served_of(const spec::Profile& profile, const spec::Choice& primary,
                            const spec::ResolvedSpeculation& resolved,
                            const SpeculationSubject& subject, bool per_request,
                            ServedSpeculation out) {
    const spec::EngineSupport& can = subject.support;
    out.menu = spec::build_menu(profile, primary, [&](const spec::Choice& choice) {
        return drafter_present(choice, resolved) && can.runs(choice.config);
    });
    if (out.menu.mode == spec::Mode::None) { return out; }
    const bool head         = subject.proposal_head;
    out.engine.backend      = backend_of(out.menu.mode);
    if (primary.config.mode == out.menu.mode) {
        out.engine.drafter = separate_drafter(primary, resolved);
    }
    out.engine.draft_tokens = static_cast<std::uint32_t>(out.menu.primary.config.k);
    out.engine.proposal_head =
        out.menu.proposal_head && head ? ProposalHead::Optimized : ProposalHead::Full;
    if (out.menu.proposal_head && !head) {
        out.notes.push_back("the artifact has no proposal head; drafting with the full head");
    }
    // A profile measured elsewhere: every entry's and depth's floor at least foreign_floor's there
    // (a shallower-to-fit choice interpolates between raised floors, which can fall below its own;
    // --spec-select thinking serves each entry per request).
    const bool head_here = subject.proposal_head;
    for (spec::MenuChoice* entry :
         {&out.menu.primary, &out.menu.thinking, &out.menu.non_thinking, &out.menu.overall}) {
        entry->floor = ran_floor(resolved, entry->config, head_here, entry->floor, &out.notes);
    }
    out.engine.acceptance_floor = watchdog_floor(out.menu.primary.floor);
    // The step-down depths and floors are the menu's shallower entries' (SpeculationMenu::
    // step_down), raised above at their own mode, depth and head (all ran_floor keys on): nothing
    // to raise again.
    out.engine.step_down = request_speculation(out.menu, out.menu.primary).step_down;
    if (per_request) {
        for (const std::int32_t depth : out.menu.depths()) {
            if (depth != out.menu.primary.config.k) {
                out.engine.request_draft_tokens.push_back(static_cast<std::uint32_t>(depth));
            }
        }
    } else {
        // A step runs only as shallow as a captured launch window.
        for (const auto& step : out.engine.step_down) {
            out.engine.request_draft_tokens.push_back(step.draft_tokens);
        }
    }
    out.engine.load = engine_load(profile.load, speculative_window(out.engine));
    out.engine.context_limit = profile.context_limit;
    out.engine.decision_sources = decision_sources_of(resolved);
    for (std::string& note : fit_draft_widths(out.engine, can)) {
        out.notes.push_back(std::move(note));
    }
    return out;
}

// The profile with every choice replaced by its fit fallback.
spec::Profile fallback_profile(const spec::Profile& profile) {
    spec::Profile out = profile;
    out.overall = out.thinking = out.non_thinking = *profile.fit_fallback;
    out.served.reset();
    return out;
}

} // namespace

SpeculationSubject speculation_subject(const std::filesystem::path& artifact, int device,
                                       const std::filesystem::path& local_profile) {
    return speculation_subject(artifact, spec::gpu_identity(device).key, local_profile);
}

SpeculationSubject speculation_subject(const std::filesystem::path& artifact,
                                       const spec::GpuKey& gpu,
                                       const std::filesystem::path& local_profile) {
    SpeculationSubject out;
    out.artifact      = artifact;
    out.resolved      = resolve(artifact, gpu, local_profile);
    out.components    = artifact_components(artifact);
    out.proposal_head = has_proposal_head(artifact);
    out.support       = engine_support(artifact);
    return out;
}

void apply_builtin_default(SpeculationSubject& subject) {
    if (subject.resolved && !subject.resolved->found()) {
        default_for_builtin_drafter(subject.components, subject.proposal_head, *subject.resolved);
    }
}

SpeculationSubject subject_of_mode(SpeculationSubject subject, spec::Mode mode) {
    if (!subject.resolved) { return subject; }
    spec::Profile& p = subject.resolved->profile;
    const spec::Mode tuned = p.overall.config.mode;
    if (p.measurements.empty()) {
        // A published profile: its fit fallback of that mode (another drafter, measured to fit
        // where the default does not) as every choice.
        if (tuned == mode || !p.fit_fallback || p.fit_fallback->config.mode != mode) {
            return subject;
        }
        p.overall = p.thinking = p.non_thinking = *p.fit_fallback;
        p.fit_fallback.reset();
        p.served.reset();
        p.portable.reset();
    } else {
        std::erase_if(p.measurements, [&](const spec::Measurement& m) {
            return m.config.enabled() && m.config.mode != mode;
        });
        p.served.reset();
        p.portable.reset();
        spec::SelectOptions options;
        options.portable_within = 0;
        options.served          = subject.support; // the choices the Engine runs
        spec::select(p, options);
    }
    // A load table was measured with the default's drafter: another kind's is not known, so one
    // request speculates and two or more decode plainly (as for a pair measured elsewhere).
    if (tuned != mode && !p.load.empty()) {
        p.load = {spec::LoadBound{.lanes = 2, .depth = 0}};
    }
    return subject;
}

std::optional<ServedSpeculation> served_speculation(const SpeculationSubject& subject,
                                                    bool thinking, bool per_request) {
    const auto& resolved = subject.resolved;
    if (!resolved || !resolved->found()) { return std::nullopt; }
    ServedSpeculation out;
    out.source                  = resolved->source;
    out.basis                   = resolved->basis;
    out.notes                   = resolved->notes;
    const spec::Profile profile = engine_choices(resolved->profile, &out.notes);
    // Per request, a default that decodes plainly still loads the drafter the other flag's
    // choice drafts with: its requests decode plainly at their own (plain) entry of the menu.
    const spec::Choice& own   = spec::pick(profile, thinking);
    const spec::Choice& other = spec::pick(profile, !thinking);
    const spec::Choice& primary =
        per_request && !own.config.enabled() && other.config.enabled() ? other : own;
    return served_of(profile, primary, *resolved, subject, per_request, std::move(out));
}

std::optional<ServedSpeculation> served_speculation(const std::filesystem::path& artifact,
                                                    int device, bool thinking, bool per_request,
                                                    const std::filesystem::path& local_profile) {
    return served_speculation(speculation_subject(artifact, device, local_profile), thinking,
                              per_request);
}

std::optional<spec::Profile> shallower_profile(const spec::Profile& profile,
                                               const spec::SpecConfig& config, double floor) {
    if (!config.enabled() || config.k <= 1) { return std::nullopt; }
    // The drafter the profile runs this mode with.
    std::vector<const spec::Choice*> choices{&profile.overall, &profile.thinking,
                                             &profile.non_thinking};
    if (profile.fit_fallback) { choices.push_back(&*profile.fit_fallback); }
    const auto same = std::find_if(choices.begin(), choices.end(), [&](const spec::Choice* c) {
        return c->config.mode == config.mode;
    });
    if (same == choices.end()) { return std::nullopt; }
    spec::Choice shallower = **same;
    shallower.config       = config;
    shallower.config.k     = config.k - 1;
    // Its watchdog floor (a round's cost in plain-decode tokens) at the shallower depth, from the
    // floors the profile measured for this mode and head at other depths and the tried depth's own
    // (a proposal head drafts at another cost per round).
    std::vector<std::pair<int, double>> floors{{config.k, floor}};
    for (const spec::Choice* choice : choices) {
        if (choice->config.mode == config.mode &&
            choice->config.proposal_head == config.proposal_head) {
            floors.emplace_back(choice->config.k, choice->floor_tokens_per_round);
        }
    }
    shallower.floor_tokens_per_round = shallower_floor(floors, shallower.config.k);
    spec::Profile out                = profile;
    out.overall                      = shallower;
    out.thinking                     = shallower;
    out.non_thinking                 = shallower;
    out.served.reset();
    out.fit_fallback.reset();
    return out;
}

std::optional<ServedSpeculation> served_shallower(const SpeculationSubject& subject,
                                                  bool per_request, const spec::SpecConfig& config,
                                                  double floor) {
    if (!config.enabled() || config.k <= 1) { return std::nullopt; }
    const auto& resolved = subject.resolved;
    if (!resolved || !resolved->found()) { return std::nullopt; }
    const auto profile = shallower_profile(resolved->profile, config, floor);
    if (!profile) { return std::nullopt; }
    ServedSpeculation out;
    out.source =
        resolved->source + " (" + profile->overall.config.label() + ", shallower to fit)";
    out.basis = resolved->basis;
    out.notes = resolved->notes;
    out = served_of(*profile, profile->overall, *resolved, subject, per_request, std::move(out));
    if (out.menu.mode == spec::Mode::None) { return std::nullopt; }
    return out;
}

std::optional<ServedSpeculation> deepest_fitting_shallower(
    spec::MenuChoice tried,
    const std::function<std::optional<ServedSpeculation>(const spec::SpecConfig&, double)>&
        shallower,
    const std::function<bool(const ServedSpeculation&)>& fits) {
    const std::int32_t deepest = tried.config.k - 1;
    if (!tried.config.enabled() || deepest < 1) { return std::nullopt; }
    std::optional<ServedSpeculation> best; // the deepest candidate known to fit
    bool best_built = false;               // whether the last `fits` call built `best`
    // Whether the candidate at `depth` fits; nullopt when there is none (no drafter for the mode).
    const auto probe = [&](std::int32_t depth) -> std::optional<bool> {
        spec::SpecConfig config = tried.config;
        config.k                = depth + 1;
        // The tried depth's floor is a measurement only there: elsewhere the profile's floors.
        auto candidate = shallower(config, config.k == tried.config.k ? tried.floor : 0.0);
        if (!candidate) { return std::nullopt; }
        best_built = fits(*candidate);
        if (best_built) { best = std::move(candidate); }
        return best_built;
    };
    const auto first = probe(deepest);
    if (!first || *first) { return best; }
    if (deepest == 1) { return std::nullopt; }
    const auto one = probe(1);
    if (!one || !*one) { return std::nullopt; }
    // Depth `fit` fits and `refused` does not: halve the depths between them.
    std::int32_t fit = 1, refused = deepest;
    while (refused - fit > 1) {
        const std::int32_t middle = fit + (refused - fit) / 2;
        const auto fits_there     = probe(middle);
        if (!fits_there) { break; }
        if (*fits_there) {
            fit = middle;
        } else {
            refused = middle;
        }
    }
    // A deeper candidate was refused after `best` was built: build `best` again.
    if (!best_built && !fits(*best)) { return std::nullopt; }
    return best;
}

spec::Profile served_choices(const spec::Profile& profile) { return engine_choices(profile, nullptr); }

bool per_request_pays(const SpeculationSubject& subject, bool thinking) {
    const auto& resolved = subject.resolved;
    if (!resolved || !resolved->found() || resolved->profile.disabled) { return false; }
    return spec::per_request_gain(engine_choices(resolved->profile, nullptr), thinking) >=
           1.0 + kPerRequestMargin;
}

const char* fallback_step_name(FallbackStep step) noexcept {
    switch (step) {
    case FallbackStep::FitFallback:
        return "fit_fallback";
    case FallbackStep::Shallower:
        return "shallower";
    case FallbackStep::Head:
        return "head";
    }
    return "fit_fallback";
}

std::optional<ServedSpeculation>
fit_fallbacks(const SpeculationSubject& subject, const ServedSpeculation& refused, bool per_request,
              const std::function<bool(const ServedSpeculation&, FallbackStep)>& fits) {
    spec::MenuChoice tried = refused.menu.primary;
    if (auto fallback = served_fit_fallback(subject, per_request)) {
        tried = fallback->menu.primary;
        if (fits(*fallback, FallbackStep::FitFallback)) { return fallback; }
    }
    if (auto deepest = deepest_fitting_shallower(
            tried,
            [&](const spec::SpecConfig& config, double floor) {
                return served_shallower(subject, per_request, config, floor);
            },
            [&](const ServedSpeculation& candidate) {
                return fits(candidate, FallbackStep::Shallower);
            })) {
        return deepest;
    }
    if (auto head = served_head_fallback(subject, per_request);
        head && fits(*head, FallbackStep::Head)) {
        return head;
    }
    return std::nullopt;
}

PoolDecision
pool_rule(const SpeculationSubject& subject, bool per_request, std::uint32_t with_drafter,
          std::uint32_t concurrency, const std::function<std::uint32_t()>& plain_contexts,
          const std::function<std::optional<std::uint32_t>(const ServedSpeculation&)>& head_contexts) {
    PoolDecision out;
    if (with_drafter == 0 || with_drafter >= concurrency) { return out; }
    out.plain = plain_contexts();
    if (out.plain <= with_drafter) { return out; }
    out.keep = PoolKeep::Plain;
    if (auto head = served_head_fallback(subject, per_request)) {
        const auto contexts = head_contexts(*head);
        if (contexts && 5 * static_cast<std::uint64_t>(*contexts) >=
                            4 * static_cast<std::uint64_t>(std::min(out.plain, concurrency))) {
            out.keep = PoolKeep::Head;
            out.head = std::move(head);
        }
    }
    return out;
}

void require_named_tune(const std::filesystem::path& artifact, int device,
                        const std::filesystem::path& local_profile) {
    if (!local_profile.empty()) { (void)resolve(artifact, device, local_profile); }
}

std::string tune_refusal(const std::filesystem::path& artifact, int device,
                         const std::filesystem::path& tune) {
    std::string id;
    try {
        id = spec::artifact_id_hex(artifact);
    } catch (const std::exception& e) {
        return tune.string() + ": the artifact's id is unreadable (" + e.what() + ")";
    }
    const auto gpu = spec::gpu_identity(device).key;
    return spec::tune_refusal(tune, id, gpu.chain.empty() ? std::string("*") : gpu.chain.front());
}

std::vector<std::string> speculation_notes(const std::filesystem::path& artifact, int device,
                                           const std::filesystem::path& local_profile) {
    const auto resolved = resolve(artifact, device, local_profile);
    if (!resolved) { return {"the artifact's id is unreadable"}; }
    if (resolved->found()) { return {}; }
    return resolved->notes;
}

std::optional<ServedSpeculation> served_fit_fallback(const SpeculationSubject& subject,
                                                     bool per_request) {
    const auto& resolved = subject.resolved;
    if (!resolved || !resolved->found() || !resolved->profile.fit_fallback) { return std::nullopt; }
    ServedSpeculation out;
    out.source = resolved->source + " (fit fallback " +
                 resolved->profile.fit_fallback->config.label() + ")";
    out.basis                   = resolved->basis;
    out.notes  = resolved->notes;
    const spec::Profile profile = fallback_profile(resolved->profile);
    out = served_of(profile, profile.overall, *resolved, subject, per_request, std::move(out));
    if (out.menu.mode == spec::Mode::None) { return std::nullopt; }
    return out;
}

std::optional<ServedSpeculation> served_head_fallback(const SpeculationSubject& subject,
                                                      bool per_request) {
    const auto& resolved = subject.resolved;
    if (!resolved || !resolved->found() ||
        engine_choices(resolved->profile, nullptr).overall.config.mode == spec::Mode::Mtp) {
        return std::nullopt;
    }
    spec::ResolvedSpeculation head;
    head.floor_sources = resolved->floor_sources; // the model's own MTP floors, where it has them
    default_for_builtin_drafter(subject.components, subject.proposal_head, head);
    if (!head.found() || head.profile.overall.config.mode != spec::Mode::Mtp) {
        return std::nullopt;
    }
    ServedSpeculation out;
    out.source = resolved->source + " (fit fallback: the built-in " +
                 head.profile.overall.config.label() + ")";
    out.basis  = "builtin";
    out.notes  = resolved->notes;
    out = served_of(head.profile, head.profile.overall, head, subject, per_request, std::move(out));
    if (out.menu.mode == spec::Mode::None) { return std::nullopt; }
    return out;
}

std::filesystem::path model_drafter(const std::filesystem::path& artifact,
                                    SpeculativeBackend backend) {
    const auto manifest = spec::read_manifest(spec::manifest_path(artifact));
    if (!manifest || backend == SpeculativeBackend::None) { return {}; }
    for (const auto& drafter : manifest->drafters) {
        if (drafter.usable() && !drafter.artifact.empty() && drafter.component.empty() &&
            backend_of_kind(drafter.kind) == backend) {
            return drafter.artifact;
        }
    }
    return {};
}

RequestSpeculation request_speculation(const spec::SpeculationMenu& menu,
                                       const spec::MenuChoice& choice) {
    if (!choice.config.enabled()) { return RequestSpeculation{}; }
    RequestSpeculation out{.draft_tokens     = static_cast<std::uint32_t>(choice.config.k),
                           .acceptance_floor = watchdog_floor(choice.floor)};
    for (const auto& step : menu.step_down(choice)) {
        out.step_down.push_back(SpeculationStep{.draft_tokens = static_cast<std::uint32_t>(step.depth),
                                                .acceptance_floor = watchdog_floor(step.floor)});
    }
    return out;
}

std::uint32_t speculative_window(const SpeculativeOptions& options) noexcept {
    if (options.backend == SpeculativeBackend::None) { return 0; }
    std::uint32_t window = options.draft_tokens;
    for (const std::uint32_t k : options.request_draft_tokens) { window = std::max(window, k); }
    return window;
}

bool snap_load_to_captured(SpeculativeOptions& options, std::uint32_t max_concurrency) {
    std::vector<std::uint32_t> captured = options.request_draft_tokens;
    captured.push_back(options.draft_tokens);
    std::erase(captured, 0U);
    if (captured.empty()) { return false; }
    std::sort(captured.begin(), captured.end());
    bool moved = false;
    for (SpeculationLoad& bound : options.load) {
        if (bound.draft_tokens == 0 || bound.lanes > max_concurrency ||
            std::binary_search(captured.begin(), captured.end(), bound.draft_tokens)) {
            continue;
        }
        const auto above = std::upper_bound(captured.begin(), captured.end(), bound.draft_tokens);
        bound.draft_tokens = above == captured.begin() ? captured.front() : *std::prev(above);
        moved              = true;
    }
    return moved;
}

std::vector<SpeculationLoad> engine_load(const std::vector<spec::LoadBound>& load,
                                         std::uint32_t window) {
    std::vector<SpeculationLoad> out;
    if (window == 0) { return out; }
    for (const auto& bound : load) {
        out.push_back(SpeculationLoad{
            .lanes        = static_cast<std::uint32_t>(bound.lanes),
            .draft_tokens = std::min(static_cast<std::uint32_t>(bound.depth), window),
            .draft_width  = bound.width == spec::DraftWidth::Block    ? DFlashDraftWidth::Block
                            : bound.width == spec::DraftWidth::Drafts ? DFlashDraftWidth::Drafts
                                                                      : DFlashDraftWidth::Default});
    }
    return out;
}

std::vector<std::string> fit_draft_widths(SpeculativeOptions& options,
                                          const spec::EngineSupport& support) {
    std::vector<std::string> notes;
    const bool dflash = options.backend == SpeculativeBackend::DFlash ||
                        options.backend == SpeculativeBackend::DFlash2;
    // Whether `width` stays: any width of a DFlash round, Block only where the family drafts it.
    const auto runs = [&](DFlashDraftWidth width) {
        return width == DFlashDraftWidth::Default ||
               (dflash && (width == DFlashDraftWidth::Drafts || support.dflash_block_width));
    };
    const std::string why =
        dflash ? "this model's DFlash rounds draft only the anchor and the drafts"
               : "the served speculation is not DFlash";
    if (!runs(options.draft_width)) {
        const bool block = options.draft_width == DFlashDraftWidth::Block;
        notes.push_back(std::string("draft width ") + (block ? "block" : "drafts") +
                        " dropped: " + why);
        options.draft_width = DFlashDraftWidth::Default;
    }
    bool dropped = false;
    for (SpeculationLoad& bound : options.load) {
        if (!runs(bound.draft_width)) {
            bound.draft_width = DFlashDraftWidth::Default;
            dropped           = true;
        }
    }
    if (dropped) { notes.push_back("the load bounds' draft widths dropped: " + why); }
    return notes;
}

} // namespace ninferno::product
