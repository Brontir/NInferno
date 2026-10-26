#include "spec/manifest.h"

#include "artifact/reader.h"
#include "core/sha256.h"
#include "spec/fingerprint.h"
#include "spec/json_fields.h"
#include "spec/registry.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>

namespace ninferno::spec {
namespace {

using Json = nlohmann::json;

// Every drafter name a profile's choices reference.
std::vector<std::string> referenced(const Profile& p) {
    std::vector<std::string> out;
    const auto add = [&](const Choice& c) {
        if (c.config.enabled() && !c.drafter.empty() &&
            std::find(out.begin(), out.end(), c.drafter) == out.end()) {
            out.push_back(c.drafter);
        }
    };
    add(p.overall);
    add(p.non_thinking);
    add(p.thinking);
    if (p.served) { // a server's choices may use a drafter the profile's own do not
        add(p.served->overall);
        add(p.served->non_thinking);
        add(p.served->thinking);
    }
    if (p.fit_fallback) { add(*p.fit_fallback); } // runs when the chosen drafter does not fit
    return out;
}

// The drafters a profile's measurements drafted with that its choices do not name: a measured
// pair's other modes, which the configuration choice weighs too (product::subject_of_mode: a 9B
// MTP GGUF measured with DFlash and MTP drafts with MTP where DFlash does not fit the context).
// Attached where present, never noted where absent (no choice asks for them).
std::vector<std::string> measured_only(const Profile& p) {
    const std::vector<std::string> named = referenced(p);
    std::vector<std::string> out;
    for (const auto& d : p.drafters) {
        const bool measured =
            std::any_of(p.measurements.begin(), p.measurements.end(), [&](const Measurement& m) {
                return m.config.enabled() && mode_name(m.config.mode) == d.mode;
            });
        if (measured && !d.name.empty() && std::find(named.begin(), named.end(), d.name) == named.end() &&
            std::find(out.begin(), out.end(), d.name) == out.end()) {
            out.push_back(d.name);
        }
    }
    return out;
}

// `published` (one of `profiles`) as this GPU runs it, into `p`: the watchdog floors of a profile
// measured on another GPU raised to the registry's (raise_foreign_floors), another profile's load
// bounds where it measured none, and a stand-in's bounds capped to this GPU. false, with a note, when it does not parse.
bool take_published(const Json& profiles, const Json& published, const GpuKey& gpu,
                    ResolvedSpeculation& out, Profile& p) {
    out.foreign_floors = measured_elsewhere(gpu, published);
    out.basis          = published.value("gpu_class", std::string()) == "*" ? "generic"
                         : out.foreign_floors                               ? "similar"
                                                                            : "own";
    if (auto note = similarity_note(gpu, published); !note.empty()) {
        out.notes.push_back(std::move(note));
    }
    try {
        p = profile_from_published(published);
        if (const auto* from = select_load_fallback(profiles, published)) {
            p.load                  = profile_from_published(*from).load;
            const auto cls          = from->value("gpu_class", std::string());
            const auto others       = from->value("applies_to", Json::array());
            const std::string whose = cls == "*"
                                          ? std::string("the \"*\" profile")
                                          : "the " + cls + " profile" +
                                                (others.empty() || !others[0].is_string()
                                                     ? std::string()
                                                     : " of " + others[0].get<std::string>());
            out.notes.push_back("load bounds " + load_label(p.load) + " from " + whose +
                                ": none measured for this profile");
        }
    } catch (const std::exception& e) {
        out.notes.push_back(std::string("the published profile does not parse: ") + e.what());
        return false;
    }
    if (out.foreign_floors) {
        out.floor_sources = profiles;
        raise_foreign_floors(p, out, out.notes);
        const auto capped = stand_in_load(p.load, gpu.sms);
        if (capped != p.load) {
            out.notes.push_back("load bounds " + load_label(p.load) + " -> " + load_label(capped) +
                                ": the profile was measured on another "
                                "GPU, plain from " +
                                std::to_string(capped.back().lanes) + " rows on this one's " +
                                std::to_string(gpu.sms) + " multiprocessors");
            p.load = capped;
        }
    }
    return true;
}

// The profiles `manifest` serves from: the registry this binary carries when the manifest came
// from it (profiles are re-measured and republished; a model converted earlier takes them without
// being converted again), else the manifest's own copy. Where they come from into `origin`, and
// into `digest` the SHA-256 of what the profiles were read from (ResolvedSpeculation::sha256).
const Json& manifest_profiles(const SpeculationManifest& manifest,
                              const std::filesystem::path& artifact, std::string& origin,
                              std::string* digest = nullptr) {
    origin = "manifest " + manifest_path(artifact).string();
    if (manifest.source == "registry" && !manifest.entry.empty()) {
        if (const auto* entry = Registry::builtin().by_id(manifest.entry)) {
            origin = "registry " + entry->id + " for " + manifest_path(artifact).string();
            if (digest) { *digest = Registry::builtin().source_sha256(); }
            return entry->profiles;
        }
    }
    if (digest) { *digest = file_sha256_hex(manifest_path(artifact)).value_or(""); }
    return manifest.profiles;
}

std::string published_source(const std::string& origin, const Json& published) {
    // The profile's revision (tools/spec/registry_add.py: a hash of its content) names the table a
    // server runs.
    const auto revision = published.value("revision", std::string());
    return origin + " (" + published.value("gpu_class", std::string()) +
           (revision.empty() ? std::string() : ", revision " + revision) + ")";
}

// The registry entry of the model an artifact is, found by what it is (Registry::nearest: its
// engine, vocabulary and geometry) and only at the entry's own shape: a requantization or an
// undeclared fine-tune of the entry's model, every drafter of which applies. The artifact's
// component names into `components`. nullptr: nothing to read, or no entry of its shape.
const ModelEntry* entry_by_shape(const std::filesystem::path& artifact,
                                 std::set<std::string>& components) {
    try {
        const artifact::Reader reader(artifact);
        for (const auto& [name, component] : reader.directory().components) {
            components.insert(name);
        }
        const auto found = reader.directory().components.find("text");
        if (found == reader.directory().components.end() || !found->second.config.is_object()) {
            return nullptr;
        }
        const Json& config = found->second.config;
        std::string architecture;
        if (const auto a = config.find("architectures");
            a != config.end() && a->is_array() && !a->empty() && (*a)[0].is_string()) {
            architecture = (*a)[0].get<std::string>();
        }
        const auto nested = config.find("text_config");
        const Json& scope = nested != config.end() && nested->is_object() ? *nested : config;
        const Geometry shape{json_unsigned(scope, "hidden_size"),
                             json_unsigned(scope, "num_hidden_layers"),
                             json_unsigned(scope, "vocab_size")};
        std::string fingerprint;
        if (const auto fp = artifact_vocab_fingerprint(artifact)) { fingerprint = fp->value; }
        const auto* entry = Registry::builtin().nearest(
            engine_for_architecture(architecture), shape, fingerprint, components.contains("mtp"));
        if (entry == nullptr || shape_distance(entry->geometry, shape) != 0.0) { return nullptr; }
        return entry;
    } catch (const std::exception&) { return nullptr; }
}

} // namespace

const ManifestDrafter* SpeculationManifest::drafter(std::string_view name) const {
    for (const auto& d : drafters) {
        if (d.name == name) { return &d; }
    }
    return nullptr;
}

std::filesystem::path manifest_path(const std::filesystem::path& artifact) {
    return std::filesystem::path(artifact.string() + ".speculation.json");
}

SpeculationManifest manifest_from_json(const Json& doc) {
    if (doc.value("schema", 0) != 1) { throw std::runtime_error("speculation manifest: unknown schema"); }
    SpeculationManifest m;
    const auto target   = doc.value("target", Json::object());
    m.target_artifact_id = json_text(target, "artifact_id");
    m.repo               = json_text(target, "repo");
    m.engine             = json_text(target, "engine");
    m.vocab_fingerprint  = json_text(target, "vocab_fingerprint");
    m.source             = json_text(doc, "source");
    m.entry              = json_text(doc, "entry");
    m.matched_by         = json_text(doc, "matched_by");
    m.fallback           = json_text(doc, "fallback");
    for (const auto& d : doc.value("drafters", Json::array())) {
        ManifestDrafter x;
        x.name        = json_text(d, "name");
        x.kind        = json_text(d, "kind");
        x.repo        = json_text(d, "repo");
        x.revision    = json_text(d, "revision");
        x.artifact    = json_text(d, "artifact");
        x.artifact_id = json_text(d, "artifact_id");
        x.component   = json_text(d, "component");
        x.status      = json_text(d, "status");
        m.drafters.push_back(std::move(x));
    }
    m.profiles = doc.value("profiles", Json::array());
    return m;
}

std::optional<SpeculationManifest> read_manifest(const std::filesystem::path& path,
                                                 std::string* error) {
    std::ifstream stream(path);
    if (!stream) {
        if (error) { *error = "no manifest at " + path.string(); }
        return std::nullopt;
    }
    try {
        return manifest_from_json(Json::parse(stream));
    } catch (const std::exception& e) {
        if (error) { *error = path.string() + ": " + e.what(); }
        return std::nullopt;
    }
}

namespace {

// Every choice object under `j` (a "config" label beside a "floor_tokens_per_round") of exactly
// `config`: mode, depth and proposal head.
void collect_floors(const Json& j, const SpecConfig& config, std::vector<double>& floors) {
    if (j.is_object()) {
        const auto label = j.find("config");
        const auto floor = j.find("floor_tokens_per_round");
        if (label != j.end() && label->is_string() && floor != j.end() && floor->is_number()) {
            try {
                const auto c = SpecConfig::parse(label->get<std::string>());
                if (c.mode == config.mode && c.k == config.k &&
                    c.proposal_head == config.proposal_head && floor->get<double>() > 1.0) {
                    floors.push_back(floor->get<double>());
                }
            } catch (const std::exception&) {
            }
        }
        for (const auto& [key, value] : j.items()) { collect_floors(value, config, floors); }
    } else if (j.is_array()) {
        for (const auto& value : j) { collect_floors(value, config, floors); }
    }
}

void raise_choice(Choice& choice, const char* name, const ResolvedSpeculation& resolved,
                  std::vector<std::string>& notes) {
    if (!choice.config.enabled()) { return; }
    const double raised = foreign_floor(resolved, choice.config, choice.floor_tokens_per_round);
    if (raised <= choice.floor_tokens_per_round) { return; }
    const bool own_model = resolved.floor_sources.is_array() &&
                           floor_ceiling(resolved.floor_sources, choice.config) > 0.0;
    char text[96];
    std::snprintf(text, sizeof(text), "%.2f -> %.2f", choice.floor_tokens_per_round, raised);
    notes.push_back(std::string(name) + " " + choice.config.label() + ": watchdog floor " + text +
                    " tokens a round (" +
                    (own_model ? "the largest this model measured for it on any GPU"
                               : "the registry's median for it: this model measured it nowhere") +
                    "; the profile was not measured on this GPU)");
    choice.floor_tokens_per_round = raised;
}

} // namespace

double floor_ceiling(const Json& profiles, const SpecConfig& config) {
    if (!config.enabled()) { return 0.0; }
    std::vector<double> floors;
    collect_floors(profiles, config, floors);
    return floors.empty() ? 0.0 : *std::max_element(floors.begin(), floors.end());
}

double registry_floor_median(const SpecConfig& config) {
    if (!config.enabled()) { return 0.0; }
    std::vector<double> floors;
    for (const auto& model : Registry::builtin().models()) {
        collect_floors(model.profiles, config, floors);
    }
    if (floors.empty()) { return 0.0; }
    std::sort(floors.begin(), floors.end());
    const auto n = floors.size();
    return n % 2 == 1 ? floors[n / 2] : 0.5 * (floors[n / 2 - 1] + floors[n / 2]);
}

namespace {

// `own` raised to the model's ceiling for `config` (floor_ceiling over resolved.floor_sources), or
// to the registry median for a configuration the model measured nowhere.
double model_floor(const ResolvedSpeculation& resolved, const SpecConfig& config, double own) {
    const double model = resolved.floor_sources.is_array()
                             ? floor_ceiling(resolved.floor_sources, config)
                             : 0.0;
    return std::max(own, model > 0.0 ? model : registry_floor_median(config));
}

} // namespace

double foreign_floor(const ResolvedSpeculation& resolved, const SpecConfig& config, double own) {
    if (!resolved.foreign_floors || !config.enabled()) { return own; }
    return model_floor(resolved, config, own);
}

double full_head_floor(const ResolvedSpeculation& resolved, const SpecConfig& config, double own) {
    const double floor = foreign_floor(resolved, config, own);
    if (!config.enabled() || !config.proposal_head) { return floor; }
    SpecConfig full    = config;
    full.proposal_head = false;
    return std::max(floor, model_floor(resolved, full, own));
}

void raise_foreign_floors(Profile& profile, const ResolvedSpeculation& resolved,
                          std::vector<std::string>& notes) {
    raise_choice(profile.overall, "default", resolved, notes);
    raise_choice(profile.thinking, "thinking", resolved, notes);
    raise_choice(profile.non_thinking, "non-thinking", resolved, notes);
    if (profile.fit_fallback) {
        raise_choice(*profile.fit_fallback, "fit fallback", resolved, notes);
    }
    for (auto* served : {&profile.served, &profile.portable}) {
        if (!served->has_value()) { continue; }
        raise_choice((*served)->overall, "served default", resolved, notes);
        raise_choice((*served)->thinking, "served thinking", resolved, notes);
        raise_choice((*served)->non_thinking, "served non-thinking", resolved, notes);
    }
}

namespace {

// Why `tuned` (read from `path`) does not serve `artifact_id` on `gpu_class`: a drafter replaced
// since the tune (or unreadable), then a key for another artifact, GPU or drafters. Empty: it serves.
std::vector<std::string> tune_refusals(const Profile& tuned, const std::filesystem::path& path,
                                       const std::string& artifact_id,
                                       const std::string& gpu_class) {
    std::vector<std::string> out;
    ProfileKey expected;
    expected.artifact_id = artifact_id;
    expected.gpu         = gpu_class;
    std::vector<std::string> ids;
    for (const auto& d : tuned.drafters) {
        if (d.artifact_id.empty()) { continue; } // built in, a component, or model-free
        ids.push_back(d.artifact_id);
        try {
            if (artifact_id_hex(d.artifact) != d.artifact_id) {
                out.push_back(path.string() + ": drafter " + d.artifact +
                              " was replaced since the tune");
            }
        } catch (const std::exception& e) {
            out.push_back(path.string() + ": drafter " + d.artifact + ": " + e.what());
        }
    }
    expected.drafters   = drafter_key(ids);
    const auto mismatch = key_mismatch(tuned.key, expected);
    if (!mismatch.empty()) { out.push_back(path.string() + " was tuned for another " + mismatch); }
    return out;
}

} // namespace

std::string tune_refusal(const std::filesystem::path& tune, const std::string& artifact_id,
                         const std::string& gpu_class) {
    std::string why;
    const auto tuned = read_profile(tune, &why);
    if (!tuned) { return tune.string() + ": " + why; }
    const auto refusals =
        tune_refusals(*tuned, tune, artifact_id, gpu_class.empty() ? tuned->key.gpu : gpu_class);
    return refusals.empty() ? std::string() : refusals.front();
}

void require_named_tune(const ResolvedSpeculation& resolved) {
    if (resolved.named_tune_refused.empty()) { return; }
    throw std::runtime_error("refusing the named speculation tune: " + resolved.named_tune_refused +
                             " (name none to run the model's published profile)");
}

namespace {

// 2. The published profile for this GPU from a manifest (manifest_profiles: the registry's, or
//    the manifest's own copy for a bundle declaration or a registry entry this binary no longer
//    knows), over its usable drafters.
// The manifest's drafters a profile's choices name, into out.drafters (an artifact path, or
// "component:<name>" inside the target); one not attached, or changed since, is noted.
void attach_drafters(const SpeculationManifest& manifest, const Profile& p,
                     ResolvedSpeculation& out) {
    // `said`: a choice names it, so its absence is noted.
    const auto attach = [&](const std::string& name, bool said) {
        const auto* d = manifest.drafter(name);
        if (d == nullptr || !d->usable()) {
            if (said) {
                out.notes.push_back("profile drafter '" + name + "' is not attached" +
                                    (d ? " (" + d->status + ")" : std::string()));
            }
            return;
        }
        if (!d->artifact.empty()) {
            try {
                if (artifact_id_hex(d->artifact) != d->artifact_id) {
                    out.notes.push_back("drafter " + d->artifact + " changed since attach");
                    return;
                }
            } catch (const std::exception& e) {
                out.notes.push_back("drafter " + d->artifact + ": " + e.what());
                return;
            }
            out.drafters[name] = d->artifact;
        } else {
            out.drafters[name] = "component:" + (d->component.empty() ? name : d->component);
        }
    };
    for (const auto& name : referenced(p)) { attach(name, true); }
    for (const auto& name : measured_only(p)) { attach(name, false); }
}

// 2. The lab's measurement of this target with the drafters it carries (spec/pairs.h, by the
//    weights it was converted from), else its group's preliminary choice. False when the registry
//    has neither for it, or the checkpoint declares its own (a bundle: it overrides the registry):
//    the published profiles answer.
// The drafters a manifest attaches from a repository of their own, by kind: repo@revision.
std::map<std::string, std::string> attached_identities(const SpeculationManifest& manifest) {
    std::map<std::string, std::string> out;
    for (const auto& d : manifest.drafters) {
        if (d.usable() && !d.repo.empty()) {
            out[d.kind] = d.repo + (d.revision.empty() ? "" : "@" + d.revision);
        }
    }
    return out;
}

bool resolve_pair_profile(const SpeculationManifest& manifest,
                          const std::filesystem::path& artifact, const GpuKey& gpu,
                          ResolvedSpeculation& out) {
    const auto& registry = Registry::builtin();
    if (registry.pairs().empty() || manifest.source == "bundle") { return false; }
    std::optional<std::string> weights;
    std::string quant;
    try {
        weights = artifact_weights(artifact);
        quant   = artifact_quant(artifact);
    } catch (const std::exception& e) {
        out.notes.push_back(std::string("the artifact's source inputs do not read: ") + e.what());
        return false;
    }
    if (!weights) {
        out.notes.push_back("the artifact records no source inputs: no lab pair names its weights");
    }
    std::vector<std::string> kinds;
    for (const auto& d : manifest.drafters) {
        if (d.usable()) { kinds.push_back(d.kind); }
    }
    auto choice = pair_choice(registry.pairs(), weights.value_or(""), manifest.engine, kinds,
                              label_bits(quant), gpu_generation(gpu), {},
                              attached_identities(manifest), manifest.repo);
    if (!choice) { return false; }
    attach_drafters(manifest, choice->profile, out);
    // The model's own floors bound a fallback the pair does not cover (the head fallback).
    std::string origin;
    out.floor_sources = manifest_profiles(manifest, artifact, origin);
    out.source  = std::move(choice->source);
    out.basis   = std::move(choice->basis);
    out.sha256  = registry.source_sha256();
    out.profile = std::move(choice->profile);
    return true;
}

// 2, without a manifest: the lab's measurement of this target's weights over the drafters it
//    carries as components (its built-in MTP layers, a DFlash drafter built in), else its group's,
//    as a planned target resolves (resolve_planned_speculation). False when the registry has
//    neither.
// The hub repository (org/name) of an artifact converted into the hub cache
// (<cache>/models--org--name/ninferno/<commit>/<file>), "" elsewhere: without a manifest naming it,
// the repository whose sibling quantizations the lab measured (pair_choice).
std::string hub_repo_of_artifact(const std::filesystem::path& artifact) {
    const auto commit_dir = artifact.parent_path();
    if (commit_dir.parent_path().filename() != "ninferno") { return {}; }
    const std::string dir = commit_dir.parent_path().parent_path().filename().string();
    if (!dir.starts_with("models--")) { return {}; }
    std::string repo = dir.substr(8);
    const auto separator = repo.find("--");
    if (separator == std::string::npos) { return {}; }
    repo.replace(separator, 2, "/");
    return repo;
}

bool resolve_entry_pair(const ModelEntry& entry, const std::set<std::string>& components,
                        const std::filesystem::path& artifact, const GpuKey& gpu,
                        ResolvedSpeculation& out) {
    const auto& registry = Registry::builtin();
    if (registry.pairs().empty()) { return false; }
    std::optional<std::string> weights;
    std::string quant;
    try {
        weights = artifact_weights(artifact);
        quant   = artifact_quant(artifact);
    } catch (const std::exception&) {
        return false;
    }
    std::vector<std::string> kinds;
    for (const char* kind : {"mtp", "dflash", "dflash2"}) {
        if (components.contains(kind)) { kinds.emplace_back(kind); }
    }
    auto choice = pair_choice(registry.pairs(), weights.value_or(""), entry.engine, kinds,
                              label_bits(quant), gpu_generation(gpu), {}, {},
                              hub_repo_of_artifact(artifact));
    if (!choice) { return false; }
    for (const auto& names : {referenced(choice->profile), measured_only(choice->profile)}) {
        for (const auto& name : names) {
            if (components.contains(name)) { out.drafters[name] = "component:" + name; }
        }
    }
    out.floor_sources = entry.profiles;
    out.source        = std::move(choice->source);
    out.basis         = std::move(choice->basis);
    out.sha256        = registry.source_sha256();
    out.profile       = std::move(choice->profile);
    return true;
}

void resolve_manifest_profile(const SpeculationManifest& manifest,
                              const std::filesystem::path& artifact, const GpuKey& gpu,
                              ResolvedSpeculation& out) {
    const std::string gpu_class = gpu.chain.empty() ? std::string("*") : gpu.chain.front();
    std::vector<std::string> available;
    for (const auto& d : manifest.drafters) {
        if (d.usable()) { available.push_back(d.name); }
    }
    std::string origin;
    std::string digest;
    const Json* profiles  = &manifest_profiles(manifest, artifact, origin, &digest);
    out.floor_sources     = *profiles; // also for a built-in default the caller falls to
    const auto* published = select_published(*profiles, gpu, manifest.repo, &available);
    if (published == nullptr) {
        out.notes.push_back(origin + " has no profile for " + gpu_class +
                            " along its similarity chain");
        return;
    }
    Profile p;
    if (!take_published(*profiles, *published, gpu, out, p)) { return; }
    attach_drafters(manifest, p, out);
    out.source  = published_source(origin, *published);
    out.sha256  = std::move(digest);
    out.profile = std::move(p);
}

// 3. No manifest: the registry entry of the model the artifact is (by its shape), over the
//    drafters it carries as components.
void resolve_entry_profile(const ModelEntry& entry, const std::set<std::string>& components,
                           const GpuKey& gpu, ResolvedSpeculation& out) {
    const std::string gpu_class = gpu.chain.empty() ? std::string("*") : gpu.chain.front();
    // A GGUF converted with its MTP layers or DFlash drafter built in, never attached: the
    // entry's profile rather than the built-in drafter's untuned default.
    std::vector<std::string> available;
    for (const char* name : {"mtp", "dflash", "dflash2"}) {
        if (components.contains(name)) { available.emplace_back(name); }
    }
    out.floor_sources     = entry.profiles;
    const auto* published = select_published(entry.profiles, gpu, {}, &available);
    if (published == nullptr) {
        out.notes.push_back("registry " + entry.id + " (by the artifact's shape) has no " +
                            "profile for " + gpu_class +
                            " along its similarity chain over the built-in drafters");
        return;
    }
    Profile p;
    if (!take_published(entry.profiles, *published, gpu, out, p)) { return; }
    // Only the drafters the artifact carries: a profile whose drafter ships as an artifact of
    // its own (gpt-oss-20b's DFlash) decodes plainly here rather than failing to load it.
    for (const auto& name : referenced(p)) {
        if (components.contains(name)) {
            out.drafters[name] = "component:" + name;
        } else {
            out.notes.push_back("profile drafter '" + name +
                                "' is neither attached nor built into the artifact");
        }
    }
    out.source  = published_source("registry " + entry.id + " by the artifact's shape", *published);
    out.sha256  = Registry::builtin().source_sha256();
    out.profile = std::move(p);
}

} // namespace

ResolvedSpeculation resolve_speculation(const std::filesystem::path& artifact,
                                        const std::string& artifact_id,
                                        const GpuKey& gpu,
                                        const std::filesystem::path& local_profile) {
    const std::string gpu_class = gpu.chain.empty() ? std::string("*") : gpu.chain.front();
    ResolvedSpeculation out;
    std::optional<SpeculationManifest> manifest;
    std::string unread; // why there is no manifest to read, when a local tune does not serve either
    {
        std::string why;
        manifest = read_manifest(manifest_path(artifact), &why);
        if (manifest && manifest->target_artifact_id != artifact_id) {
            out.notes.push_back(manifest_path(artifact).string() +
                                " belongs to another artifact (id " +
                                manifest->target_artifact_id + ")");
            manifest.reset();
        } else if (!manifest && !std::filesystem::exists(manifest_path(artifact))) {
            // A converted model without its drafters attached (or attached beside another path
            // to the same file, such as a symlink): nothing names its registry entry.
            unread = "no " + manifest_path(artifact).filename().string() +
                     " beside the artifact: ninferno-convert --attach-drafters --target-artifact "
                     "<artifact> --model-id <repo> writes it from the registry";
        } else if (!manifest) {
            unread = why;
        }
        if (manifest) { out.fallback = manifest->fallback.empty() ? "none" : manifest->fallback; }
    }

    // 1. A maintainer's own tune, only when named (--spec-profile): one left beside the artifact is
    //    not read, so a machine's own measurement never steers what it runs by itself.
    const auto& tuned_path = local_profile;
    // Status that cannot be read (a looping symlink, no permission) only drops the note.
    std::error_code beside_error;
    if (tuned_path.empty() && std::filesystem::exists(default_profile_path(artifact), beside_error)) {
        out.notes.push_back(default_profile_path(artifact).string() +
                            " is not read: a local tune applies only when named "
                            "(--spec-profile " +
                            default_profile_path(artifact).string() + ")");
    }
    // A named tune that does not serve: a note, and the first reason the entry point refuses it
    // with (require_named_tune), since it was named.
    const auto refuse = [&](std::string reason) {
        if (out.named_tune_refused.empty()) { out.named_tune_refused = reason; }
        out.notes.push_back(std::move(reason));
    };
    std::string why;
    if (auto tuned = tuned_path.empty() ? std::nullopt : read_profile(tuned_path, &why)) {
        const auto refusals = tune_refusals(*tuned, tuned_path, artifact_id, gpu_class);
        for (const auto& reason : refusals) { refuse(reason); }
        if (refusals.empty()) {
            out.source  = "local tune " + tuned_path.string();
            out.basis   = "own";
            out.sha256  = file_sha256_hex(tuned_path).value_or("");
            out.profile = std::move(*tuned);
            for (const auto& d : out.profile.drafters) {
                const auto name = d.name.empty() ? d.kind : d.name;
                if (d.status != "accepted") { continue; }
                out.drafters[name] = d.artifact_id.empty() ? "component:" + name : d.artifact;
            }
            // A tune measures its depths at one request; the published load bounds of its kind
            // (deeper drafts cost more rounds as requests share the GPU) still apply.
            if (out.profile.load.empty() && manifest) {
                std::string origin;
                const Json& profiles = manifest_profiles(*manifest, artifact, origin);
                std::vector<std::string> tuned_drafters;
                for (const auto& [name, path] : out.drafters) { tuned_drafters.push_back(name); }
                const auto* published =
                    select_published(profiles, gpu, manifest->repo, &tuned_drafters);
                ResolvedSpeculation scratch;
                Profile p;
                if (published != nullptr && take_published(profiles, *published, gpu, scratch, p) &&
                    !p.load.empty() && p.overall.config.mode == out.profile.overall.config.mode) {
                    out.profile.load = std::move(p.load);
                    out.notes.push_back("load bounds " + load_label(out.profile.load) + " from " +
                                        published_source(origin, *published) +
                                        ": the local tune measured none");
                }
            }
            // The model's own floors bound a fallback the tune does not cover (the head fallback).
            if (manifest) {
                std::string origin;
                out.floor_sources = manifest_profiles(*manifest, artifact, origin);
            }
            return out;
        }
    } else if (!tuned_path.empty()) {
        refuse("the named tune " + tuned_path.string() + ": " + why);
    }
    if (!unread.empty()) { out.notes.push_back(unread); }

    // 2. The lab's measurement of this pair, else its group's (resolve_pair_profile); then the
    //    published profile for this GPU (manifest_profiles: the registry's, or the manifest's
    //    own copy for a bundle declaration or a registry entry this binary no longer knows).
    if (manifest && resolve_pair_profile(*manifest, artifact, gpu, out)) { return out; }
    if (manifest) {
        resolve_manifest_profile(*manifest, artifact, gpu, out);
    } else if (std::set<std::string> components;
               const auto* entry = entry_by_shape(artifact, components)) {
        if (resolve_entry_pair(*entry, components, artifact, gpu, out)) { return out; }
        resolve_entry_profile(*entry, components, gpu, out);
    }
    return out;
}

ResolvedSpeculation resolve_planned_speculation(const std::optional<SpeculationManifest>& manifest,
                                                const ModelEntry* entry,
                                                const std::set<std::string>& components,
                                                const GpuKey& gpu,
                                                const std::filesystem::path& label,
                                                std::string_view weights, std::string_view quant,
                                                std::string_view engine, std::string_view repo) {
    ResolvedSpeculation out;
    const auto& registry = Registry::builtin();
    // A checkpoint's own declaration (a bundle) overrides the registry, its pairs too.
    if (!registry.pairs().empty() && !(manifest && manifest->source == "bundle")) {
        std::vector<std::string> kinds;
        if (manifest) {
            for (const auto& d : manifest->drafters) {
                if (d.usable()) { kinds.push_back(d.kind); }
            }
        } else {
            for (const char* kind : {"mtp", "dflash", "dflash2"}) {
                if (components.contains(kind)) { kinds.emplace_back(kind); }
            }
        }
        const std::string family = manifest ? manifest->engine : std::string(engine);
        const auto attached = manifest ? attached_identities(*manifest)
                                       : std::map<std::string, std::string>{};
        const std::string_view of =
            manifest && !manifest->repo.empty() ? std::string_view(manifest->repo) : repo;
        if (auto choice = pair_choice(registry.pairs(), weights, family, kinds, label_bits(quant),
                                      gpu_generation(gpu), {}, attached, of)) {
            // The model's own floors bound a fallback the pair does not cover (the head
            // fallback), as for a published profile.
            if (manifest) {
                out.fallback = manifest->fallback.empty() ? "none" : manifest->fallback;
                attach_drafters(*manifest, choice->profile, out);
                std::string origin;
                out.floor_sources = manifest_profiles(*manifest, label, origin);
            } else {
                for (const auto& names : {referenced(choice->profile), measured_only(choice->profile)}) {
                    for (const auto& name : names) {
                        if (components.contains(name)) { out.drafters[name] = "component:" + name; }
                    }
                }
                if (entry != nullptr) { out.floor_sources = entry->profiles; }
            }
            out.source  = std::move(choice->source);
            out.basis   = std::move(choice->basis);
            out.sha256  = registry.source_sha256();
            out.profile = std::move(choice->profile);
            return out;
        }
    }
    if (manifest) {
        out.fallback = manifest->fallback.empty() ? "none" : manifest->fallback;
        resolve_manifest_profile(*manifest, label, gpu, out);
    } else if (entry != nullptr) {
        resolve_entry_profile(*entry, components, gpu, out);
    } else {
        out.notes.push_back("no drafter manifest would be written and no registry entry has this "
                            "model's shape");
    }
    return out;
}

} // namespace ninferno::spec
