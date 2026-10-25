#include "convert/drafters.h"

#include "core/ascii.h"
#include "convert/files.h"
#include "core/cache_file.h"
#include "core/clock.h"
#include "core/sha256.h"
#include "core/sys.h"
#include "convert/cache.h"

#include "artifact/reader.h"
#include "convert/convert.h"
#include "convert/dense.h"
#include "convert/ggml_quants.h"
#include "convert/gguf.h"
#include "convert/gguf_model.h"
#include "convert/gguf_resources.h"
#include "convert/safetensors.h"
#include "spec/fingerprint.h"
#include "spec/json_fields.h"
#include "spec/manifest.h"
#include "spec/profile.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifndef NINFERNO_SOURCE_DIR
#    define NINFERNO_SOURCE_DIR ""
#endif

namespace ninferno::convert {
namespace {

namespace fs = std::filesystem;
using Json   = nlohmann::json;

constexpr const char* kDeclarationFile = "ninferno_speculation.json";

void log_line(const std::function<void(const std::string&)>& log, const std::string& line) {
    if (log) { log(line); }
}

constexpr const char* kUnvalidated =
    "the dense engine drafts for the models it was validated on (a registry entry for the dense "
    "engine whose geometry the target has); --drafter-artifact runs a drafter anyway";

// Every dimension the entry records is the target's.
bool fits_geometry(const spec::Geometry& entry, const spec::Geometry& target) {
    const auto fits = [](std::uint64_t mine, std::uint64_t theirs) {
        return mine == 0 || mine == theirs;
    };
    return fits(entry.hidden_size, target.hidden_size) &&
           fits(entry.num_hidden_layers, target.num_hidden_layers) &&
           fits(entry.vocab_size, target.vocab_size);
}

bool complete_geometry(const spec::Geometry& g) {
    return g.hidden_size != 0 && g.num_hidden_layers != 0 && g.vocab_size != 0;
}

std::string geometry_label(const spec::Geometry& g) {
    return "hidden " + std::to_string(g.hidden_size) + ", " + std::to_string(g.num_hidden_layers) +
           " layers, vocab " + std::to_string(g.vocab_size);
}

// The scope holding a model's text geometry: text_config when present.
const Json& text_scope(const Json& config) {
    const auto it = config.find("text_config");
    return it != config.end() && it->is_object() ? *it : config;
}

std::string first_architecture(const Json& config) {
    const auto it = config.find("architectures");
    if (it == config.end() || !it->is_array() || it->empty() || !(*it)[0].is_string()) {
        return {};
    }
    return (*it)[0].get<std::string>();
}

std::string repo_name(const std::string& repo) {
    const auto slash = repo.find('/');
    return slash == std::string::npos ? repo : repo.substr(slash + 1);
}

// base_model entries of a README's YAML front matter (a string or a "- item" list).
std::vector<std::string> readme_base_models(const fs::path& readme) {
    std::vector<std::string> out;
    std::ifstream stream(readme);
    std::string line;
    if (!std::getline(stream, line) || line.rfind("---", 0) != 0) { return out; }
    const auto trim = [](std::string s) {
        while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '"' ||
                              s.back() == '\'')) {
            s.pop_back();
        }
        while (!s.empty() && (s.front() == ' ' || s.front() == '"' || s.front() == '\'')) {
            s.erase(0, 1);
        }
        return s;
    };
    bool in_list = false;
    while (std::getline(stream, line)) {
        if (line.rfind("---", 0) == 0) { break; }
        if (line.rfind("base_model:", 0) == 0) {
            const auto value = trim(line.substr(11));
            if (!value.empty()) {
                out.push_back(value);
            } else {
                in_list = true;
            }
            continue;
        }
        if (in_list) {
            const auto t = trim(line);
            if (t.rfind("- ", 0) == 0) {
                out.push_back(trim(t.substr(2)));
            } else {
                in_list = false;
            }
        }
    }
    return out;
}

// The config of a converted drafter artifact's text component (the converter keeps the HF keys
// the checks read).
Json artifact_config(const fs::path& artifact) {
    const artifact::Reader reader(artifact);
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) {
        throw std::runtime_error(artifact.string() + " has no text component");
    }
    return text->second.config;
}

std::set<std::string> artifact_components(const fs::path& artifact) {
    std::set<std::string> names;
    const artifact::Reader reader(artifact);
    for (const auto& [name, component] : reader.directory().components) { names.insert(name); }
    return names;
}

// The frontend resources of the target artifact, written into `dir` (a drafter checkpoint
// without them borrows the target's, as the converter's --resource does).
std::vector<std::pair<std::string, fs::path>> target_resources(const fs::path& target_artifact,
                                                              const fs::path& dir) {
    std::vector<std::pair<std::string, fs::path>> out;
    const artifact::Reader reader(target_artifact);
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) { return out; }
    fs::create_directories(dir);
    for (const auto& [role, handle] : text->second.resources) {
        const auto bytes = reader.read_object(handle);
        const auto path  = dir / role;
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        out.emplace_back(role, path);
    }
    return out;
}

// The fetch of a drafter's weights: the whole checkpoint, or its one GGUF file.
FetchOptions drafter_fetch(const PlannedDrafter& d, const AttachOptions& options) {
    auto fetch = options.fetch;
    fetch.include.clear();
    fetch.quant    = d.decl.gguf;
    fetch.revision = d.decl.revision;
    return fetch;
}

// A GGUF drafter's metadata as the checkpoint config llama.cpp converted it from, and its
// tokenizer's fingerprint, from the file's first bytes: a drafter the target refuses is never
// downloaded whole. False when the mapping needs tensor bytes the head does not hold.
bool gguf_head_config(const PlannedDrafter& d, const AttachOptions& options, Json& config,
                      std::string& fingerprint, std::string& commit, std::string& file) {
    constexpr std::uint64_t kFirstHead = 16ULL << 20; // Muse's DFlash metadata: about 9 MiB
    constexpr std::uint64_t kLastHead  = 256ULL << 20;
    const auto fetch                   = drafter_fetch(d, options);
    GgufHead head;
    std::optional<GgufFile> parsed;
    for (std::uint64_t want = kFirstHead; !parsed; want *= 4) {
        head = fetch_gguf_head(d.decl.repo, fetch, want);
        try {
            parsed = parse_gguf_head(reinterpret_cast<const std::byte*>(head.bytes.data()),
                                     head.bytes.size(), head.size, head.path);
        } catch (const GgufTruncated&) {
            if (head.bytes.size() >= head.size || want >= kLastHead) { throw; }
        }
    }
    commit = head.commit;
    file   = fs::path(head.path).filename().string();
    std::map<std::string, TensorInfo> tensors;
    for (const auto& tensor : parsed->tensors) {
        TensorInfo info;
        info.name       = tensor.name;
        info.shape      = tensor.shape();
        info.bytes      = tensor.bytes;
        info.ggml_type  = tensor.type;
        info.dtype_name = ggml::type_name(tensor.type);
        tensors.emplace(tensor.name, std::move(info));
    }
    const StoredLookup lookup = [&](const std::string& name) -> const TensorInfo* {
        const auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    };
    const StoredRead read = [](const TensorInfo& tensor, std::uint64_t, std::uint64_t, void*) {
        throw std::runtime_error(tensor.name + ": its bytes are not in the metadata");
    };
    try {
        config = Json::parse(map_gguf_model(*parsed, lookup, read, head.path).config.dump());
    } catch (const std::exception& error) {
        log_line(options.log, "  " + file + ": the metadata alone does not map (" + error.what() +
                                  "); reading the whole file");
        return false;
    }
    const auto rebuilt = gguf_frontend_resources(*parsed);
    if (const auto it = rebuilt.files.find("tokenizer.json"); it != rebuilt.files.end()) {
        fingerprint = spec::vocab_fingerprint_text(it->second).value;
    }
    return true;
}

// Fetches a drafter's config.json (and tokenizer.json when it has one) and checks it. A GGUF-only
// drafter has neither: its metadata stands for the checkpoint config (and the tokenizer rebuilt
// from it) llama.cpp converted it from, read from the file's first bytes.
void check_remote(const DrafterTarget& target, PlannedDrafter& d, const AttachOptions& options) {
    Json config;
    std::string fingerprint;
    if (!d.decl.gguf.empty()) {
        std::string file;
        if (!gguf_head_config(d, options, config, fingerprint, d.commit, file)) {
            const auto fetched = fetch_model(d.decl.repo, drafter_fetch(d, options));
            d.commit           = fetched.commit;
            file               = fetched.model.filename().string();
            const Checkpoint gguf(fetched.model);
            config = Json::parse(gguf.config().dump());
            if (const auto it = gguf.rebuilt_resources().find("tokenizer.json");
                it != gguf.rebuilt_resources().end()) {
                fingerprint = spec::vocab_fingerprint_text(it->second).value;
            }
        }
        d.checks.clear();
        d.checks.push_back("note GGUF " + file + ": its metadata stands in for config.json");
    } else {
        auto fetch     = options.fetch;
        fetch.include  = {"config.json", "tokenizer.json"};
        fetch.quant.clear();
        fetch.revision = d.decl.revision;
        const auto small = fetch_model(d.decl.repo, fetch);
        d.commit         = small.commit;
        config           = Json::parse(read_file(small.snapshot / "config.json"));
        if (fs::exists(small.snapshot / "tokenizer.json")) {
            fingerprint = spec::vocab_fingerprint_file(small.snapshot / "tokenizer.json").value;
        }
        d.checks.clear();
    }
    check_drafter(target, config, fingerprint, d);
}

} // namespace

nlohmann::json DrafterPlan::to_json() const {
    Json drafters_json = Json::array();
    for (const auto& d : drafters) {
        auto item          = spec::drafter_json(d.decl);
        item["supported"]  = d.supported;
        item["compatible"] = d.compatible;
        item["checks"]     = d.checks;
        if (!d.reason.empty()) { item["reason"] = d.reason; }
        if (!d.status.empty()) { item["status"] = d.status; }
        if (!d.artifact.empty()) { item["artifact"] = d.artifact; }
        if (!d.artifact_id.empty()) { item["artifact_id"] = d.artifact_id; }
        if (!d.commit.empty()) { item["commit"] = d.commit; }
        drafters_json.push_back(std::move(item));
    }
    return Json{{"schema", 1},
                {"target",
                 {{"repo", target.repo},
                  {"lineage", target.lineage},
                  {"engine", target.engine},
                  {"architecture", target.architecture},
                  {"hidden_size", target.geometry.hidden_size},
                  {"num_hidden_layers", target.geometry.num_hidden_layers},
                  {"vocab_size", target.geometry.vocab_size},
                  {"vocab_fingerprint", target.fingerprint},
                  {"builtin_mtp", target.builtin_mtp},
                  {"gguf_source", target.gguf_source},
                  {"gguf_mapper", target.gguf_mapper}}},
                {"source", source},
                {"entry", entry},
                {"matched_by", matched_by},
                {"drafters", drafters_json},
                {"profiles", profiles},
                {"fallback", target.engine == "gemma4" ? "ngram" : "none"},
                {"notes", notes},
                {"stale_target", stale_target},
                {"created", utc_now()}};
}

namespace {

bool mtp_tensor(const std::string& name) {
    return name.rfind("mtp.", 0) == 0 || name.find(".mtp.") != std::string::npos;
}

} // namespace

bool index_lists_mtp(const Json& index) {
    // Held by name: items() of a temporary outlives it in a range-for.
    const auto weights = index.value("weight_map", Json::object());
    for (const auto& [name, file] : weights.items()) {
        if (mtp_tensor(name)) { return true; }
    }
    return false;
}

bool sharded_checkpoint_mtp(const Json& config, const Json& index) {
    return index_lists_mtp(index) ||
           spec::json_unsigned(text_scope(config), "num_nextn_predict_layers") > 0 ||
           spec::json_unsigned(config, "num_nextn_predict_layers") > 0;
}

bool checkpoint_mtp(const fs::path& dir, const Json& config) {
    const auto& text = text_scope(config);
    // MTP layers are what the weights hold where their listing is here (the index, or the one
    // weight file's header), not what the configuration declares: a fine-tune saved without them
    // (XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B: mtp_num_hidden_layers 1, no mtp.* tensor) has none.
    // Without a listing, the declaration. (NextN layers are numbered past the decoder's and keep
    // their declaration.)
    bool mtp = spec::json_unsigned(text, "mtp_num_hidden_layers") > 0;
    if (fs::exists(dir / "model.safetensors.index.json")) {
        try {
            mtp = index_lists_mtp(Json::parse(read_file(dir / "model.safetensors.index.json")));
        } catch (const std::exception&) {} // unreadable: the declaration
    } else if (fs::exists(dir / "model.safetensors")) {
        try {
            const Checkpoint weights(dir);
            bool listed = false;
            for (const auto& [name, tensor] : weights.tensors()) {
                if (mtp_tensor(name)) { listed = true; }
            }
            mtp = listed;
        } catch (const std::exception&) {} // unreadable here: the declaration
    }
    return mtp || spec::json_unsigned(text, "num_nextn_predict_layers") > 0 ||
           spec::json_unsigned(config, "num_nextn_predict_layers") > 0 ||
           fs::exists(dir / "model_mtp.safetensors");
}

DrafterTarget describe_checkpoint(const fs::path& dir, std::string repo) {
    DrafterTarget t;
    t.checkpoint      = dir;
    const auto config            = Json::parse(read_file(dir / "config.json"));
    const auto& text  = text_scope(config);
    t.architecture    = first_architecture(config);
    t.engine                     = spec::engine_for_architecture(t.architecture);
    // A checkpoint the dense converter reads (gpt-oss, Qwen3 MoE, Llama, ...) is served by the
    // dense engine; Nemotron-H keeps its own.
    if (t.engine.empty() && is_dense_architecture(t.architecture) &&
        !t.architecture.starts_with("NemotronH")) {
        t.engine = "dense";
    }
    t.geometry.hidden_size       = spec::json_unsigned(text, "hidden_size");
    t.geometry.num_hidden_layers = spec::json_unsigned(text, "num_hidden_layers");
    t.geometry.vocab_size        = spec::json_unsigned(text, "vocab_size");
    t.sliding_window             = spec::json_unsigned(text, "sliding_window");
    t.builtin_mtp = checkpoint_mtp(dir, config);
    if (fs::exists(dir / "tokenizer.json")) {
        t.fingerprint = spec::vocab_fingerprint_file(dir / "tokenizer.json").value;
    }
    if (fs::exists(dir / kDeclarationFile)) {
        t.declaration = Json::parse(read_file(dir / kDeclarationFile));
    }
    t.repo    = std::move(repo);
    t.lineage = readme_base_models(dir / "README.md");
    return t;
}

DrafterTarget describe_artifact(const fs::path& path, std::string repo) {
    DrafterTarget t;
    t.artifact = path;
    const artifact::Reader reader(path);
    const auto& components = reader.directory().components;
    for (const auto& [name, component] : components) { t.components.push_back(name); }
    const auto text = components.find("text");
    if (text == components.end()) {
        throw std::runtime_error(path.string() + " has no text component");
    }
    const auto& config = text->second.config;
    const auto& scope  = text_scope(config);
    t.architecture     = first_architecture(config);
    t.engine                     = spec::engine_for_architecture(t.architecture);
    t.geometry.hidden_size       = spec::json_unsigned(scope, "hidden_size");
    t.geometry.num_hidden_layers = spec::json_unsigned(scope, "num_hidden_layers");
    t.geometry.vocab_size        = spec::json_unsigned(scope, "vocab_size");
    t.sliding_window             = spec::json_unsigned(scope, "sliding_window");
    t.builtin_mtp                = components.contains("mtp");
    if (const auto fp = spec::artifact_vocab_fingerprint(path)) { t.fingerprint = fp->value; }
    // Converted from a GGUF: the mapping records its version, or (before the versions) the source
    // path is a .gguf file (a directory of GGUF shards is not told apart; describe_gguf_artifact
    // knows it).
    const auto& provenance = reader.directory().provenance;
    if (provenance.is_object()) {
        t.gguf_mapper = provenance.value("gguf_mapper", std::string());
        std::string base;
        if (const auto sources = provenance.find("sources");
            sources != provenance.end() && sources->is_object()) {
            if (const auto it = sources->find("base"); it != sources->end() && it->is_object()) {
                base = it->value("path", std::string());
            }
        }
        const auto extension = ascii_lower(fs::path(base).extension().string());
        t.gguf_source = !t.gguf_mapper.empty() || extension == ".gguf";
    }
    t.repo = std::move(repo);
    return t;
}

DrafterTarget describe_gguf_artifact(const fs::path& artifact, std::string repo,
                                     const fs::path& gguf) {
    DrafterTarget t = describe_artifact(artifact, std::move(repo));
    t.gguf_source   = true;
    if (const auto base = gguf_base_repository(gguf)) { t.lineage.push_back(*base); }
    return t;
}

DrafterPlan plan_drafters(DrafterTarget target, const spec::Registry& registry) {
    DrafterPlan plan;
    plan.target = std::move(target);
    const auto& t = plan.target;
    if (t.engine.empty()) {
        plan.notes.push_back("architecture '" + t.architecture +
                             "' has no NInferno engine family: no drafter can run");
    }
    std::optional<spec::ModelEntry> entry;
    if (t.declaration) {
        entry           = spec::parse_model_entry(*t.declaration);
        plan.source     = "bundle";
        plan.matched_by = "declaration";
    } else {
        const spec::ModelEntry* found = t.repo.empty() ? nullptr : registry.by_repo(t.repo);
        if (found) {
            plan.matched_by = "repo";
        } else {
            for (const auto& base : t.lineage) {
                found = registry.by_repo(base);
                if (!found) { found = registry.by_lineage(base); }
                if (found) {
                    plan.matched_by = "lineage";
                    plan.notes.push_back("matched through the declared base model " + base);
                    break;
                }
            }
        }
        if (found && plan.matched_by == "lineage" && !fits_geometry(found->geometry, t.geometry)) {
            // A descendant of another size (a distilled or pruned model) is not the entry's.
            plan.notes.push_back("the entry " + found->id + " is for " +
                                 geometry_label(found->geometry) + ", the target is " +
                                 geometry_label(t.geometry) + ": not taken");
            found = nullptr;
            plan.matched_by.clear();
        }
        // No repository or lineage names it: the nearest registered model by what it is. Of
        // another shape, only its choices for the MTP layers carry over (Registry::nearest).
        bool mtp_only = false;
        if (!found) {
            found = registry.nearest(t.engine, t.geometry, t.fingerprint, t.builtin_mtp);
            if (found) {
                plan.matched_by = "similar";
                mtp_only        = !fits_geometry(found->geometry, t.geometry);
                std::ostringstream distance;
                distance << std::fixed << std::setprecision(2)
                         << spec::shape_distance(found->geometry, t.geometry);
                plan.notes.push_back(
                    mtp_only ? "no registry entry names this model: " + found->id + " (" +
                                   geometry_label(found->geometry) + ", shape distance " +
                                   distance.str() + ") is the nearest " + t.engine +
                                   " model registered, and its choices for the MTP layers stand in"
                             : "no registry entry names this model: it has the engine, vocabulary "
                               "and geometry of " +
                                   found->id + ", whose drafters and profiles it takes");
            }
        }
        if (found) {
            entry = *found;
            if (mtp_only) {
                std::erase_if(entry->drafters,
                              [](const spec::DrafterDecl& decl) { return decl.kind != "mtp"; });
            }
            plan.source = "registry";
        }
    }
    const auto add_builtin = [&](spec::DrafterDecl decl) {
        PlannedDrafter d;
        d.decl       = std::move(decl);
        d.supported  = registry.supported("mtp", t.engine);
        d.compatible = d.supported && t.builtin_mtp;
        if (!t.builtin_mtp) {
            d.reason = "declared MTP layers are absent from the checkpoint";
        } else if (!d.supported) {
            d.reason = "the " + t.engine + " engine runs no MTP layers";
        }
        d.checks.push_back((d.compatible ? "ok " : "FAIL ") +
                           (d.reason.empty() ? std::string("MTP layers in the checkpoint") : d.reason));
        plan.drafters.push_back(std::move(d));
    };
    bool builtin_added = false;
    if (entry) {
        plan.entry    = entry->id;
        plan.profiles = entry->profiles;
        if (!entry->engine.empty() && !t.engine.empty() && entry->engine != t.engine) {
            plan.notes.push_back("the entry is for the " + entry->engine + " engine, the target is " +
                                 t.engine);
        }
        if (!entry->vocab_fingerprint.empty() && !t.fingerprint.empty() &&
            entry->vocab_fingerprint != t.fingerprint) {
            plan.notes.push_back("the target's vocabulary " + t.fingerprint + " is not the entry's " +
                                 entry->vocab_fingerprint +
                                 ": every drafter is checked against the target's own");
        }
        for (const auto& decl : entry->drafters) {
            if (decl.kind == "mtp") {
                add_builtin(decl);
                builtin_added = true;
                continue;
            }
            PlannedDrafter d;
            d.decl      = decl;
            d.taps_only = plan.source == "bundle";
            if (d.decl.recipe.empty() && d.decl.component.empty()) {
                for (const auto& k : registry.kinds()) {
                    if (k.name == decl.kind) { d.decl.recipe = k.recipe; }
                }
            }
            d.entry_geometry = entry->geometry;
            d.supported      = registry.supported(decl.kind, t.engine);

            const bool validated = plan.source == "registry" && entry->engine == "dense" &&
                                   complete_geometry(entry->geometry) &&
                                   fits_geometry(entry->geometry, t.geometry);
            if (d.supported && t.engine == "dense" && !validated) {
                d.supported   = false;
                d.unvalidated = true;
                d.reason      = kUnvalidated;
                d.checks.push_back("FAIL " + d.reason);
            } else if (!d.supported) {
                d.reason = "the " + t.engine + " engine runs no " + decl.kind + " drafter";
                d.checks.push_back("FAIL " + d.reason);
            } else {
                d.checks.push_back("note checked when fetched");
            }
            // NInferno's own quantizations, built from the same download beside it: each one more
            // drafter "<name>.<variant>" the profiles may name.
            const auto variants = std::exchange(d.decl.variants, {});
            plan.drafters.push_back(d);
            for (const auto& variant : variants) {
                auto v       = d;
                v.decl.name  = decl.name + "." + variant;
                v.decl.quant = variant;
                plan.drafters.push_back(std::move(v));
            }
        }
    }
    if (t.builtin_mtp && !builtin_added) {
        spec::DrafterDecl decl;
        decl.name      = "mtp";
        decl.kind      = "mtp";
        decl.component = "mtp";
        add_builtin(decl);
    }
    // A target converted wrong takes no drafter at all (DrafterPlan::stale_target). A Nemotron-H
    // GGUF converted before nemotron_h_moe/2 counted the GGUF's MTP block as a last trunk layer
    // when the file carried one (unsloth's fold it into block_count; bartowski's do not): whatever
    // repository it is said to be, such a conversion is taken only at a depth a nemotron_h model
    // of the registry has at its hidden size. A repository names the model, not the depth: a
    // target of a listed repository whose layer count is not the entry's was converted by such a
    // mapping, or is another model.
    const auto listed_depth = [&] {
        return std::any_of(registry.models().begin(), registry.models().end(),
                           [&](const spec::ModelEntry& m) {
                               return m.engine == "nemotron_h" &&
                                      m.geometry.hidden_size == t.geometry.hidden_size &&
                                      m.geometry.num_hidden_layers == t.geometry.num_hidden_layers;
                           });
    };
    if (t.engine == "nemotron_h" && t.gguf_source && t.gguf_mapper.empty() && !listed_depth()) {
        plan.stale_target =
            "the target has " + std::to_string(t.geometry.num_hidden_layers) +
            " layers, no nemotron_h model of the registry has that depth at hidden size " +
            std::to_string(t.geometry.hidden_size) +
            ", and it was converted from a GGUF before nemotron_h_moe/2 (it records no "
            "provenance.gguf_mapper), which counted an MTP block as a trunk layer: reconvert "
            "the target (not needed if it was converted from a GGUF without an MTP block)";
    } else if (entry && plan.matched_by == "repo" && entry->geometry.num_hidden_layers != 0 &&
               t.geometry.num_hidden_layers != 0 &&
               entry->geometry.num_hidden_layers != t.geometry.num_hidden_layers) {
        plan.stale_target = "the target has " + std::to_string(t.geometry.num_hidden_layers) +
                            " layers (num_hidden_layers), the model of " + entry->id + " has " +
                            std::to_string(entry->geometry.num_hidden_layers) +
                            ": reconvert the target, or --model-id names another model";
    }
    if (!plan.stale_target.empty()) {
        plan.notes.push_back(plan.stale_target);
        for (auto& d : plan.drafters) {
            d.stale_target = plan.stale_target;
            d.supported    = false;
            d.compatible   = false;
            d.reason       = plan.stale_target;
            d.checks.push_back("FAIL " + plan.stale_target);
        }
    }
    if (plan.source.empty()) { plan.source = t.builtin_mtp ? "builtin" : "none"; }
    return plan;
}

void check_drafter(const DrafterTarget& target, const Json& config,
                   const std::string& drafter_fingerprint, PlannedDrafter& d,
                   const spec::Registry& registry) {
    auto& checks = d.checks;
    bool ok      = true;
    d.reason.clear();
    const auto fail = [&](const std::string& what) {
        checks.push_back("FAIL " + what);
        if (d.reason.empty()) { d.reason = what; }
        ok = false;
    };
    const auto arch = first_architecture(config);
    const auto kind = registry.kind_for_architecture(arch);
    if (!kind) {
        fail("architecture '" + arch + "' is not a known drafter");
    } else if (!d.decl.kind.empty() && d.decl.kind != *kind) {
        fail("architecture " + arch + " is a " + *kind + " drafter, declared " + d.decl.kind);
    } else {
        d.decl.kind = *kind;
        checks.push_back("ok architecture " + arch + " (" + *kind + ")");
    }
    d.supported = kind && registry.supported(*kind, target.engine) && !d.unvalidated &&
                  d.stale_target.empty();
    if (!d.stale_target.empty()) {
        fail(d.stale_target);
    } else if (kind && d.unvalidated) {
        fail(kUnvalidated);
    } else if (kind && !d.supported) {
        fail("the " + (target.engine.empty() ? std::string("target's") : target.engine) +
             " engine runs no " + *kind + " drafter");
    }
    const auto& scope = text_scope(config);
    const auto vocab  = spec::json_unsigned(config, "vocab_size")
                            ? spec::json_unsigned(config, "vocab_size")
                            : spec::json_unsigned(scope, "vocab_size");
    if (!drafter_fingerprint.empty() && !target.fingerprint.empty()) {
        if (drafter_fingerprint == target.fingerprint) {
            checks.push_back("ok vocabulary fingerprint " + drafter_fingerprint);
        } else {
            fail("vocabulary fingerprint " + drafter_fingerprint + " differs from the target's " +
                 target.fingerprint);
        }
    } else if (vocab != 0) {
        if (vocab == target.geometry.vocab_size) {
            checks.push_back("ok vocab size " + std::to_string(vocab) +
                             " (no tokenizer.json: fingerprint not compared)");
        } else {
            fail("vocab size " + std::to_string(vocab) + " != target " +
                 std::to_string(target.geometry.vocab_size));
        }
    } else if (d.decl.kind != "gemma4_assistant") {
        fail("no vocab_size in config.json");
    }
    const auto expect = [&](const std::string& what, std::uint64_t mine, std::uint64_t theirs) {
        if (mine == theirs) {
            checks.push_back("ok " + what + " " + std::to_string(mine));
        } else {
            fail(what + " " + std::to_string(mine) + " != target " + std::to_string(theirs));
        }
    };
    if (d.decl.kind == "gemma4_assistant") {
        expect("backbone_hidden_size", spec::json_unsigned(config, "backbone_hidden_size"),
               target.geometry.hidden_size);
        if (config.value("use_ordered_embeddings", false)) {
            fail("use_ordered_embeddings drafters (Gemma 4 E-models) are not supported");
        }
        const auto window = spec::json_unsigned(scope, "sliding_window");
        if (window != 0 && target.sliding_window != 0) {
            expect("sliding_window", window, target.sliding_window);
        }
    } else if (d.decl.kind == "nemotron_h_mtp") {
        expect("hidden_size", spec::json_unsigned(config, "hidden_size"),
               target.geometry.hidden_size);
        // The head is one attention+MoE block; its depths stay the target's.
        expect("num_hidden_layers", spec::json_unsigned(config, "num_hidden_layers"), 1);
    } else if (d.decl.kind == "dflash" || d.decl.kind == "dflash2") {
        expect("hidden_size", spec::json_unsigned(config, "hidden_size"),
               target.geometry.hidden_size);
        if (config.contains("num_target_layers") || (d.decl.gguf.empty() && !d.taps_only)) {
            expect("num_target_layers", spec::json_unsigned(config, "num_target_layers"),
                   target.geometry.num_hidden_layers);
        } else if (d.taps_only) {
            checks.push_back("note num_target_layers unrecorded: the taps alone are checked");
        } else if (d.entry_geometry.num_hidden_layers != 0) {
            // llama.cpp's DFlash GGUF records the taps only: the target has the layers of the
            // model the registry entry publishes the drafter for (the taps are checked below).
            expect("num_target_layers unrecorded (a GGUF drafter): the entry's num_hidden_layers",
                   d.entry_geometry.num_hidden_layers, target.geometry.num_hidden_layers);
        } else {
            fail("num_target_layers unrecorded (a GGUF drafter) and no entry geometry to check");
        }
        const auto draft = config.value("dflash_config", Json::object());
        const auto taps  = draft.contains("target_layer_ids")
                               ? draft["target_layer_ids"]
                               : config.value("target_layer_ids", Json());
        bool taps_ok = taps.is_array() && !taps.empty();
        for (const auto& tap : taps.is_array() ? taps : Json::array()) {
            taps_ok = taps_ok && tap.is_number_integer() && tap.get<std::int64_t>() >= 0 &&
                      tap.get<std::uint64_t>() < target.geometry.num_hidden_layers;
        }
        if (taps_ok) {
            checks.push_back("ok target_layer_ids " + taps.dump());
        } else {
            fail("target_layer_ids " + taps.dump() + " outside the target's " +
                 std::to_string(target.geometry.num_hidden_layers) + " layers");
        }
        const auto mask = draft.value("mask_token_id", std::int64_t{-1});
        if (mask < 0 || static_cast<std::uint64_t>(mask) >= target.geometry.vocab_size) {
            fail("mask_token_id " + std::to_string(mask) + " outside the vocabulary");
        }
        const auto block = spec::json_unsigned(config, "block_size")
                               ? spec::json_unsigned(config, "block_size")
                               : spec::json_unsigned(draft, "block_size");
        if (block == 0 || block > 16) {
            fail("block_size " + std::to_string(block) + " outside [1, 16]");
        } else {
            checks.push_back("ok block_size " + std::to_string(block));
        }
        // A scaled rope is resolved at conversion (the block-mask drafters: gpt-oss's YaRN) or
        // refused there (the Gemma 4 drafters).
        if (config.contains("rope_scaling") && !config["rope_scaling"].is_null()) {
            checks.push_back("note rope_scaling " + config["rope_scaling"].value("rope_type",
                                                                                std::string("?")) +
                             ": resolved into the drafter's frequency table at conversion");
        }
    }
    d.compatible = ok && d.supported;
}

std::optional<fs::path> resolve_ranking(const std::string& name, const fs::path& repository) {
    if (name.empty()) { return std::nullopt; }
    // The given source tree (the working directory for ninferno-convert), then the rankings
    // installed beside this binary (<exe dir>/../share/ninferno/rankings, as the route tables),
    // then the tree it was built from: the rankings the registry names ship in tools/spec/rankings.
    const auto in_tree = [&](const fs::path& root) -> std::optional<fs::path> {
        if (root.empty()) { return std::nullopt; }
        for (const auto& candidate : {root / "tools/spec/rankings" / name,
                                      root / "tools/freq_corpus/fixtures/ranking" / name}) {
            if (fs::exists(candidate)) { return candidate; }
        }
        return std::nullopt;
    };
    if (auto found = in_tree(repository)) { return found; }
    std::error_code ec;
    if (const auto exe = sys::self_exe(); !exe.empty()) {
        const auto installed = exe.parent_path().parent_path() / "share" / "ninferno" / "rankings" / name;
        if (fs::exists(installed, ec)) { return installed; }
    }
    return in_tree(fs::path(NINFERNO_SOURCE_DIR));
}

std::string ranking_label(const std::string& name, const fs::path& path) {
    return name + " sha256:" + ninferno::sha256_hex(read_file(path, "ranking"));
}

std::vector<std::pair<std::string, fs::path>> fetch_companions(DrafterPlan& plan,
                                                               const AttachOptions& options) {
    std::vector<std::pair<std::string, fs::path>> out;
    for (auto& d : plan.drafters) {
        if (d.decl.kind == "mtp" || d.decl.component.empty() || !d.supported) { continue; }
        try {
            check_remote(plan.target, d, options);
            if (!d.compatible) {
                d.status = "incompatible: " + d.reason;
                log_line(options.log, "drafter " + d.decl.repo + " rejected: " + d.reason);
                continue;
            }
            auto fetch = options.fetch;
            fetch.include.clear();
            fetch.quant.clear();
            fetch.revision = d.decl.revision;
            log_line(options.log, "companion drafter " + d.decl.repo + " -> component " +
                                      d.decl.component);
            const auto fetched = fetch_model(d.decl.repo, fetch);
            d.commit           = fetched.commit;
            d.status           = "fetched";
            out.emplace_back(d.decl.component, fetched.snapshot);
        } catch (const std::exception& error) {
            d.status = std::string("failed: ") + error.what();
            log_line(options.log, "  " + d.status);
        }
    }
    return out;
}

DrafterPlan attach_drafters(DrafterPlan plan, const fs::path& target_artifact,
                            const AttachOptions& options) {
    const auto components = artifact_components(target_artifact);
    // A given drafter the model declares nowhere (no bundle, no registry entry yet, e.g. the
    // NemotronH MTP sidecar converted by hand): attached under its given name, its kind read from
    // its own architecture and checked against the target like a declared one (check_drafter
    // below decides whether the target's engine runs it). A target converted wrong (stale_target)
    // takes none of them either.
    const auto& stale = plan.stale_target;
    for (const auto& given : options.drafter_artifacts) {
        const auto& name    = given.first;
        const bool declared = std::any_of(
            plan.drafters.begin(), plan.drafters.end(),
            [&](const PlannedDrafter& d) { return d.decl.name == name; });
        if (declared) { continue; }
        PlannedDrafter d;
        d.decl.name    = name;
        d.supported    = stale.empty();
        d.stale_target = stale;
        d.reason       = stale;
        plan.notes.push_back("drafter " + name +
                             " is given (--drafter-artifact), not declared by the model");
        plan.drafters.push_back(std::move(d));
    }
    for (auto& d : plan.drafters) {
        if (d.decl.kind == "mtp") {
            if (d.compatible) {
                d.status = components.contains("mtp")
                               ? "embedded"
                               : "absent from the artifact (convert it with --components text,mtp)";
            } else {
                d.status = "unsupported: " + d.reason;
            }
            continue;
        }
        if (!d.supported && !(d.unvalidated && options.drafter_artifacts.contains(d.decl.name))) {
            d.status = "unsupported: " + d.reason;
            continue;
        }
        if (!d.decl.component.empty()) {
            if (components.contains(d.decl.component)) {
                if (d.status.empty() || d.status == "fetched") {
                    // The converter pinned the component's hidden size, vocabulary and target
                    // layer count to the target's when it built it in (and dropped those keys);
                    // what the stored config still says is checked here.
                    d.checks.clear();
                    const artifact::Reader reader(target_artifact);
                    auto config = reader.directory().components.at(d.decl.component).config;
                    for (const auto& [key, value] :
                         {std::pair{"hidden_size", plan.target.geometry.hidden_size},
                          std::pair{"vocab_size", plan.target.geometry.vocab_size},
                          std::pair{"num_target_layers", plan.target.geometry.num_hidden_layers}}) {
                        if (!config.contains(key)) { config[key] = value; }
                    }
                    if (!config.contains("block_size")) { config["block_size"] = 16; }
                    check_drafter(plan.target, config, "", d);
                    d.checks.push_back("note component " + d.decl.component +
                                       " built into the artifact (geometry pinned at conversion)");
                    d.status = d.compatible ? "embedded" : "incompatible: " + d.reason;
                }
            } else if (d.status.empty()) {
                d.status = "not in the artifact: convert the target with --components text,mtp," +
                           d.decl.component + " (ninferno-convert --fetch does)";
            }
            continue;
        }
        try {
            const auto local = options.drafter_artifacts.find(d.decl.name);
            if (local != options.drafter_artifacts.end()) {
                d.checks.clear();
                d.taps_only = true; // given: no entry vouches for its layer count
                if (d.unvalidated) {
                    d.unvalidated = false; // given: the user runs it
                    d.checks.push_back(
                        "note given for a model the dense engine was not validated on");
                }
                const auto fp = spec::artifact_vocab_fingerprint(local->second);
                check_drafter(plan.target, artifact_config(local->second), fp ? fp->value : "", d);
                if (!d.compatible) {
                    d.status = "incompatible: " + d.reason;
                    continue;
                }
                d.artifact    = fs::absolute(local->second).string();
                d.artifact_id = spec::artifact_id_hex(local->second);
                d.status      = "attached";
                log_line(options.log, "drafter " + d.decl.name + ": " + d.artifact + " (given)");
                continue;
            }
            if (d.decl.repo.empty()) {
                d.status = "failed: no repository declared";
                continue;
            }
            check_remote(plan.target, d, options);
            if (!d.compatible) {
                d.status = "incompatible: " + d.reason;
                log_line(options.log, "drafter " + d.decl.repo + " rejected: " + d.reason);
                continue;
            }
            const auto fetch = drafter_fetch(d, options);
            log_line(options.log, "drafter " + d.decl.name + ": " + d.decl.repo +
                                      (d.decl.revision.empty() ? "" : "@" + d.decl.revision.substr(0, 12)));
            const auto fetched = fetch_model(d.decl.repo, fetch);
            d.commit           = fetched.commit;
            const auto dir =
                fetched.snapshot.parent_path().parent_path() / "ninferno" / fetched.commit;
            fs::create_directories(dir);
            ConvertOptions convert_options;
            convert_options.model         = fetched.model;
            convert_options.recipe        = d.decl.recipe;
            convert_options.drafter_quant       = options.conversion.drafter_quant.empty()
                                                      ? d.decl.quant
                                                      : options.conversion.drafter_quant;
            convert_options.drafter_quant_declared = options.conversion.drafter_quant.empty();
            convert_options.threads             = options.conversion.threads;
            convert_options.max_file_bytes      = options.conversion.max_file_bytes;
            convert_options.round_tiny_scales   = options.conversion.round_tiny_scales;
            convert_options.maintainer_quantize = options.conversion.maintainer_quantize;
            convert_options.log                 = options.log;
            std::string suffix;
            if (d.decl.proposal_rows) {
                if (const auto ranking = resolve_ranking(d.decl.ranking, options.repository)) {
                    convert_options.proposal      = true;
                    convert_options.proposal_rows = d.decl.proposal_rows;
                    convert_options.ranking       = *ranking;
                    // By name and content, not by where it lay: the drafter's bytes are the same
                    // on every machine that converts it.
                    convert_options.ranking_label = ranking_label(d.decl.ranking, *ranking);
                    suffix = "_sl" + std::to_string(d.decl.proposal_rows);
                    d.checks.push_back("ok ranking " + convert_options.ranking_label + " from " +
                                       ranking->string());
                } else {
                    d.checks.push_back("note ranking " + d.decl.ranking +
                                       " not found (tools/spec/rankings, share/ninferno/rankings): "
                                       "full drafter head");
                }
            }
            // Frontend files the drafter checkpoint lacks come from the target.
            std::vector<std::string> missing;
            for (const char* role : {"tokenizer.json", "tokenizer_config.json",
                                     "chat_template.jinja", "generation_config.json"}) {
                if (!fs::exists(fetched.snapshot / role)) { missing.emplace_back(role); }
            }
            if (!missing.empty()) {
                for (auto& [role, path] :
                     target_resources(target_artifact, dir / "target-resources")) {
                    if (std::find(missing.begin(), missing.end(), role) != missing.end()) {
                        convert_options.resources.emplace_back(role, path);
                    }
                }
            }
            // A GGUF-only drafter by its file (its repository holds other models too).
            const std::string stem = d.decl.gguf.empty() ? repo_name(d.decl.repo)
                                                         : fetched.model.stem().string();
            const std::string quant = d.decl.quant.empty() ? "" : "." + d.decl.quant;
            convert_options.out = dir / (stem + "." + d.decl.recipe + quant + suffix + ".ninferno");
            if (fs::exists(convert_options.out) &&
                fs::exists(convert_options.out.string() + ".conversion.json")) {
                d.status = "cached";
            } else {
                const auto result = convert(convert_options);
                log_line(options.log, "  wrote " + convert_options.out.string() + " in " +
                                          std::to_string(result.seconds).substr(0, 5) + " s");
                d.status = "converted";
            }
            d.artifact    = convert_options.out.string();
            d.artifact_id = spec::artifact_id_hex(convert_options.out);
            if (const auto fp = spec::artifact_vocab_fingerprint(convert_options.out);
                fp && !plan.target.fingerprint.empty()) {
                if (fp->value != plan.target.fingerprint) {
                    d.compatible = false;
                    d.reason     = "the converted drafter's vocabulary " + fp->value +
                               " differs from the target's " + plan.target.fingerprint;
                    d.status = "incompatible: " + d.reason;
                } else {
                    d.checks.push_back("ok converted drafter's vocabulary fingerprint " + fp->value);
                }
            }
        } catch (const std::exception& error) {
            d.status = std::string("failed: ") + error.what();
            log_line(options.log, "  " + d.status);
        }
    }
    // A failed drafter is listed as failed: the profiles that require it are passed over for the
    // others (spec::select_published), and drafters_incomplete() has the next run try again.
    auto manifest = plan.to_json();
    manifest["target"]["artifact"]    = fs::absolute(target_artifact).string();
    manifest["target"]["artifact_id"] = spec::artifact_id_hex(target_artifact);
    // The registry the drafters were planned from: another one (an update that names a new
    // drafter) has drafters_incomplete() plan them again.
    manifest["registry_sha256"] = spec::Registry::builtin().source_sha256();
    const auto path = spec::manifest_path(target_artifact);
    write_file_atomic(path, manifest.dump(1) + '\n');
    log_line(options.log, "speculation manifest " + path.string());
    return plan;
}

bool drafters_incomplete(const fs::path& target_artifact) {
    const auto path = spec::manifest_path(target_artifact);
    if (!fs::exists(path)) { return true; }
    try {
        const auto manifest = Json::parse(read_file(path));
        if (manifest.value("registry_sha256", std::string()) !=
            spec::Registry::builtin().source_sha256()) {
            return true;
        }
        const auto& drafters = manifest.at("drafters");
        return std::any_of(drafters.begin(), drafters.end(), [](const Json& d) {
            return !d.contains("quant") && d.value("status", "").starts_with("failed");
        });
    } catch (const std::exception&) { return true; }
}

} // namespace ninferno::convert
