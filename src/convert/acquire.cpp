#include "convert/acquire.h"

#include "artifact/reader.h"

#include "core/ascii.h"
#include "core/quant_label.h"
#include "convert/files.h"
#include "convert/options_key.h"

#include "convert/cache.h"
#include "convert/checks.h"
#include "convert/dense.h"
#include "convert/pyjson.h"
#include "convert/safetensors.h"
#include "convert/support.h"
#include "core/cache_file.h"
#include "core/host_memory.h"
#include "core/sha256.h"
#include "spec/manifest.h"

#include "core/sys.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <stdexcept>
#include <string_view>

namespace ninferno::convert {
namespace {

namespace fs = std::filesystem;

// A Vision projector's place in the order a start takes them (mmproj-*.gguf): BF16 first (the
// published encoder's own type, kept exactly), then F32, then F16 (rounded to a narrower range),
// then any; -1 for another file.
int projector_rank(std::string_view file_name) {
    const std::string name = ascii_lower(file_name);
    if (!name.starts_with("mmproj") || !name.ends_with(".gguf")) { return -1; }
    return name.find("bf16") != std::string::npos  ? 0
           : name.find("f32") != std::string::npos ? 1
           : name.find("f16") != std::string::npos ? 2
                                                   : 3;
}

// A GGUF model's Vision projector. A hub model's is the one its repository's listing ranks first
// (best_projector), fetched or already in the snapshot: the plan without the weights charges that
// one, whatever an earlier run left beside the model. A local model's, or a hub snapshot's served
// offline (no listing), is the best one beside the model file. nullopt when there is none.
std::optional<fs::path> gguf_projector(const FetchResult& fetched, const FetchOptions& fetch,
                                       const std::function<void(const std::string&)>& log) {
    if (fetched.repo.empty() || fetched.offline) { return gguf_projector_beside(fetched.model); }
    auto listing     = fetch;
    listing.revision = fetched.commit;
    std::vector<std::string> names;
    for (const auto& quant : list_quants(fetched.repo + "@" + fetched.commit, listing).quants) {
        for (const auto& file : quant.files) { names.push_back(file.path); }
    }
    const auto parts = best_projector(names);
    if (parts.empty()) { return std::nullopt; }
    auto small     = fetch;
    small.include  = parts; // every part of a split projector: the converter reads them all
    small.quant.clear();
    small.revision = fetched.commit;
    const auto files = fetch_model(fetched.repo, small);
    for (const auto& file : files.files) {
        if (file.path == parts.front()) {
            if (log) { log("vision projector " + fetched.repo + "/" + file.path); }
            return files.snapshot / file.path;
        }
    }
    throw std::runtime_error(fetched.repo + ": the vision projector " + parts.front() +
                             " is listed but was not fetched");
}

constexpr const char* kTokenizerFiles[] = {"tokenizer.json", "tokenizer_config.json",
                                           "chat_template.jinja", "generation_config.json"};

void say(const AcquireOptions& options, const std::string& line) {
    if (options.log) { options.log(line); }
}

// A fetched GGUF without tokenizer files beside it: the ones its metadata cannot rebuild come
// from its base model, as overrides. The GGUF's own chat template and end-of-sequence ids win
// (a quantized instruct model often names its pretrained base, whose template differs); the
// base supplies what the metadata keeps only approximately, the Hugging Face tokenizer itself.
void base_tokenizer(const FetchResult& fetched, const FetchOptions& fetch, ConvertOptions& options) {
    if (!fs::is_regular_file(fetched.model)) { return; }
    std::map<std::string, std::string> rebuilt;
    try {
        rebuilt = Checkpoint(fetched.model).rebuilt_resources();
    } catch (const std::exception&) {}
    std::vector<std::string> wanted;
    for (const char* role : kTokenizerFiles) {
        bool given = false;
        for (const auto& [name, _] : options.resources) { given = given || name == role; }
        const bool beside = fs::exists(fetched.model.parent_path() / role);
        const bool keep   = std::string_view(role) == "chat_template.jinja" ||
                          std::string_view(role) == "generation_config.json";
        if (!given && !beside && !(keep && rebuilt.contains(role))) { wanted.emplace_back(role); }
    }
    if (wanted.empty()) { return; }
    const auto base = gguf_base_repository(fetched.model);
    if (!base) { return; }
    auto options_for_base    = fetch;
    options_for_base.include = {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                                "generation_config.json", "config.json"};
    options_for_base.quant.clear();
    options_for_base.revision.clear();
    options_for_base.before_download = {};
    try {
        const auto files = fetch_model(*base, options_for_base);
        std::string taken;
        for (const auto& role : wanted) {
            const auto path = files.snapshot / role;
            if (fs::exists(path)) {
                options.resources.emplace_back(role, path);
                taken += (taken.empty() ? "" : ", ") + role;
            }
        }
        if (fetch.log && !taken.empty()) {
            fetch.log("tokenizer files from the base model " + *base + ": " + taken);
        }
    } catch (const std::exception& error) {
        if (fetch.log) {
            fetch.log(std::string("base model ") + *base + " unavailable (" + error.what() +
                      "); tokenizer files are rebuilt from the GGUF metadata");
        }
    }
}

// Whether the conversion keeps a checkpoint's MTP layers: as components_auto is set below (not
// when the caller named the components or asked for no drafters), or --components names mtp.
bool keeps_mtp(const AcquireOptions& options, const ConvertOptions& convert_options) {
    return (!options.components_given && options.drafters != Drafters::Off) ||
           converts_mtp(convert_options);
}

// What config.json alone says before any weight is downloaded: the recipe the checkpoint takes
// (throws for an unsupported architecture) and, for a dense decoder, whether the family can
// represent it (throws naming the field). "" when the repository has no config.json (a GGUF).
std::string preflight(const std::string& repo, const std::string& revision,
                      const AcquireOptions& options) {
    auto small            = options.fetch;
    small.include         = {"config.json"};
    small.quant.clear();
    small.revision        = revision;
    small.before_download = {};
    small.log             = {};
    FetchResult config;
    try {
        config = fetch_model(repo, small);
    } catch (const std::exception& error) {
        const std::string what = error.what();
        if (what.find("no file matches") != std::string::npos) { return {}; } // GGUF repository
        throw;
    }
    if (!fs::is_regular_file(config.snapshot / "config.json")) { return {}; }
    std::ifstream stream(config.snapshot / "config.json");
    const OJson parsed = OJson::parse(stream);
    // The snapshot is shared with every file fetched from this revision before: a GGUF
    // repository's other quantizations sit beside its config.json, and reading the directory
    // refused a second one ("holds several GGUF models") or took the recipe from another file.
    // The recipe is read from config.json alone (the conversion reads the chosen file's own).
    // Without a usable temporary directory the snapshot is read as before.
    std::error_code temp_error;
    fs::path alone = fs::temp_directory_path(temp_error);
    if (!temp_error) {
        alone /= "ninferno-preflight-" + std::to_string(sys::pid()) + "-" +
                 std::to_string(reinterpret_cast<std::uintptr_t>(&parsed));
    }
    struct RemoveAlone {
        fs::path path;
        ~RemoveAlone() {
            std::error_code ignored;
            if (!path.empty()) { fs::remove_all(path, ignored); }
        }
    } remove_alone{temp_error ? fs::path() : alone};
    if (!temp_error) { fs::create_directories(alone, temp_error); }
    if (!temp_error) {
        fs::copy_file(config.snapshot / "config.json", alone / "config.json",
                      fs::copy_options::overwrite_existing, temp_error);
    }
    RecipeChoice choice;
    try {
        choice = choose_recipe(temp_error ? config.snapshot : alone, GgufRequest::Model,
                               keeps_mtp(options, options.convert));
        if (is_dense_architecture(declared_architecture(parsed))) {
            (void)dense_text_config(parsed, true);
        }
    } catch (const std::exception& error) {
        throw std::runtime_error(repo + " cannot be converted (config.json, nothing else was "
                                        "downloaded): " +
                                 error.what());
    }
    return options.convert.recipe.empty() ? choice.recipe : options.convert.recipe;
}

// Refuses a download the cache's file system cannot hold.
void check_space(const FetchPlan& plan) {
    if (plan.missing_bytes == 0) { return; }
    fs::create_directories(plan.root);
    std::error_code error;
    const auto space = fs::space(plan.root, error);
    if (error) { return; }
    const std::uint64_t free = space.available;
    if (plan.missing_bytes > free) {
        throw std::runtime_error(
            plan.repo + "@" + plan.commit.substr(0, 12) + " needs " +
            format_gib(plan.missing_bytes) + " more in the hub cache " + plan.root.string() +
            ", which has " + format_gib(free) +
            " free (its artifact needs about as much again): free space there, or point "
            "HF_HUB_CACHE at a larger disk");
    }
}

struct LockFile {
    int fd = -1;

    explicit LockFile(const fs::path& path) {
        fd = sys::open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) { throw std::runtime_error("cannot open " + path.string()); }
        if (sys::lock(fd, true) != 0) {
            sys::close(fd);
            throw std::runtime_error("cannot lock " + path.string());
        }
    }

    ~LockFile() { sys::close(fd); }

    LockFile(const LockFile&)            = delete;
    LockFile& operator=(const LockFile&) = delete;
};

bool complete(const fs::path& artifact, bool drafters) {
    return fs::exists(artifact) && fs::exists(artifact.string() + ".conversion.json") &&
           (!drafters || fs::exists(spec::manifest_path(artifact)));
}

// Whether the start attaches the model's separate drafters: asked for by name or by the automatic
// choice (not --no-drafters, and for a GGUF model not by default).
bool wants_drafters(Drafters mode) { return mode == Drafters::Attach || mode == Drafters::Require; }

// A reused artifact's drafters whose file here was written again since attach take its new id
// (drafters.h, refresh_changed_drafters): nothing is fetched, whatever the start asks of drafters.
std::string required_text(const AcquireOptions& options);

void refresh_drafters(const AcquireOptions& options, const fs::path& artifact) {
    if (options.drafters == Drafters::Off) { return; }
    for (const auto& name : refresh_changed_drafters(artifact)) {
        say(options, "drafter " + name +
                         ": its file was written again since attach; the "
                         "manifest takes it as it is now");
    }
    // A drafter the user gave that the refresh could not take (its file gone, unreadable or
    // refused by the checks) is served without, and no plan restores it: a start that requires
    // its drafters stops instead. (Another drafter is planned and attached again.)
    if (options.drafters == Drafters::Require) {
        auto changed = given_drafters_changed(artifact);
        // Given again now (--drafter-artifact NAME=FILE): the attach takes the new file.
        std::erase_if(changed, [&](const std::string& name) {
            return options.drafter_artifacts.contains(name);
        });
        if (!changed.empty()) {
            throw std::runtime_error("drafter " + changed.front() + " of " + artifact.string() +
                                     " (given with --drafter-artifact) is gone or no longer "
                                     "passes the checks since it was attached (" +
                                     required_text(options) + ")");
        }
    }
}

// How to fetch a drafter ahead of time, for the warning: its repository, the commit the registry
// pins (drafter_fetch asks for it) and the GGUF file it names.
struct FetchAdvice {
    std::string repo;
    std::string revision;
    std::string gguf;
};

// The model is served without drafters it attached nothing of. Under Attach (the automatic
// choice drafts, the user gave no flag) this is a warning that says what it costs, and, when
// `advice` is given (the drafter's fetch failed, so a download or a network cures it), how to get
// the drafter; otherwise a plain line, as a checkpoint's drafters always were.
void not_attached(const AcquireOptions& options, const std::string& what, std::string why,
                  bool plural, const FetchAdvice* advice = nullptr) {
    while (!why.empty() && (why.back() == '.' || why.back() == ' ')) { why.pop_back(); }
    if (options.drafters != Drafters::Attach) {
        say(options, what + ": " + why);
        return;
    }
    std::string line = "warning: " + what + ": " + why + ". The model is served without " +
                       (plural ? "them" : "it") +
                       ", with speculative decoding off: replies are slower.";
    if (advice != nullptr && !advice->repo.empty()) {
        const bool offline = why.find("HF_HUB_OFFLINE") != std::string::npos;
        line += std::string(offline ? " Unset HF_HUB_OFFLINE and start again"
                                    : " Go online and start again") +
                ", or fetch the drafter ahead of time: hf download " + advice->repo;
        if (!advice->revision.empty()) { line += " --revision " + advice->revision; }
        if (!advice->gguf.empty()) { line += " --include \"" + advice->gguf + "*\""; }
        line += ".";
    }
    say(options, line);
}

// Why a Require start fails without its drafter. --with-drafters says so and that the flag can be
// left out; a --spec backend names the drafter it runs, and the ways out.
std::string required_text(const AcquireOptions& options) {
    if (options.required_by.empty() || options.required_by == "--with-drafters") {
        return "--with-drafters requires every drafter; without the flag the model is served "
               "without it";
    }
    return "the selected " + options.required_by +
           " needs its drafter: go online or fetch it ahead of time, or start with --spec none "
           "(or --spec mtp, where the model has MTP layers)";
}

// Every drafter planned for the model that failed to be fetched, converted or attached, or that
// does not fit the target. Under --with-drafters (Require) that is an error. Under Attach (the
// automatic choice drafts) it is a warning, and the model is served without the drafter.
void require_drafters(const DrafterPlan& plan, const AcquireOptions& options) {
    if (!wants_drafters(options.drafters)) { return; }
    for (const auto& d : plan.drafters) {
        if (!(d.status.starts_with("failed") || d.status.starts_with("incompatible") ||
              d.status.starts_with("not in the artifact"))) {
            continue;
        }
        if (options.drafters == Drafters::Require) {
            throw std::runtime_error("drafter " + d.decl.name + " " + d.status + " (" +
                                     required_text(options) + ")");
        }
        // Only a failed fetch is cured by the network or a copy in the cache; a full disk, a
        // conversion error or a drafter that does not fit the model is not.
        const FetchAdvice advice{d.decl.repo, d.decl.revision, d.decl.gguf};
        not_attached(options, "drafter " + d.decl.name, d.status, false,
                     d.fetch_failed ? &advice : nullptr);
    }
}

} // namespace

void check_free_space(const FetchPlan& plan) { check_space(plan); }

void check_attached_drafters(const DrafterPlan& plan, const AcquireOptions& options) {
    require_drafters(plan, options);
}

bool drafters_enabled(Drafters mode, bool gguf) {
    return mode == Drafters::Default ? !gguf : mode != Drafters::Off;
}

DrafterAction decide_drafters(Drafters mode, bool gguf, bool download_present,
                              const ManifestState& manifest, bool components_given) {
    using Kind            = DrafterActionKind;
    const bool incomplete = drafters_incomplete(manifest);
    if (!download_present) {
        // A plan the checkpoint's own declaration made cannot be made again without that
        // download: kept as it is, and a drafter of it that failed fails --with-drafters.
        if (manifest.bundle_failed && mode == Drafters::Require) {
            return {Kind::Fail, "failed (or its file no longer passes the checks) and its download "
                                "was removed: fetch the model again"};
        }
        // A checkpoint's companion the artifact lacks is converted into it, which needs the
        // download too. A GGUF's is missing when its components were named without it: fetching
        // the model again does not change that.
        if (manifest.missing_companion && !gguf && mode == Drafters::Require) {
            return {Kind::Fail,
                    "is not in the artifact and its download was removed: fetch the model again"};
        }
        // Said under the default too (a checkpoint's drafters are on by default), so a start that
        // serves without it never does so silently.
        if (manifest.bundle_failed &&
            (mode == Drafters::Attach || (mode == Drafters::Default && !gguf))) {
            return {Kind::Warn, "it failed (or its file no longer passes the checks) and its "
                                "download was removed (fetch the model again to retry it)"};
        }
        // Planned again from the artifact (a checkpoint's by default, a GGUF's when asked for).
        if (mode != Drafters::Off && incomplete && !manifest.bundle_planned &&
            (!gguf || wants_drafters(mode))) {
            return {Kind::Attach, {}};
        }
        return {};
    }
    if (!drafters_enabled(mode, gguf)) { return {}; }
    // A GGUF's manifest is missing or incomplete; a checkpoint's artifact with no manifest was
    // converted without its drafters.
    if (gguf ? incomplete : !manifest.exists) {
        return {gguf ? (components_given ? Kind::Attach : Kind::CompanionsThenAttach)
                     : Kind::Reconvert,
                {}};
    }
    if (!gguf && incomplete) { return {Kind::ReplanFromCheckpoint, {}}; }
    return {};
}

DrafterPlan plan_checkpoint_drafters(const std::filesystem::path& checkpoint,
                                     const std::string& repo, bool components_given,
                                     ConvertOptions& convert_options, const AttachOptions& attach) {
    auto plan = plan_drafters(describe_checkpoint(checkpoint, repo));
    // Built-in MTP layers and companion drafters are components of the target itself.
    std::string components = convert_options.components;
    const auto add         = [&](const std::string& name) {
        if (("," + components + ",").find("," + name + ",") == std::string::npos) {
            components += "," + name;
        }
    };
    for (const auto& d : plan.drafters) {
        if (d.decl.kind == "mtp" && d.compatible && !components_given) { add("mtp"); }
    }
    bool companion = false;
    for (const auto& [component, snapshot] : fetch_companions(plan, attach)) {
        add(component);
        convert_options.sources.emplace_back(component, snapshot);
        companion = true;
    }
    // --drafter-quant is the drafters' format: a target that takes no companion leaves it to
    // its separate drafters (AttachOptions::conversion) instead of refusing it.
    const bool separate =
        std::any_of(plan.drafters.begin(), plan.drafters.end(),
                    [](const auto& d) { return d.decl.kind != "mtp" && d.decl.component.empty(); });
    if (!companion && separate) { convert_options.drafter_quant.clear(); }
    // A companion the registry declares quantized (its "quant", as for a separate drafter) is
    // converted so unless --drafter-quant says otherwise.
    for (const auto& d : plan.drafters) {
        if (convert_options.drafter_quant.empty() && d.status == "fetched" &&
            !d.decl.component.empty() && !d.decl.quant.empty()) {
            convert_options.drafter_quant          = d.decl.quant;
            convert_options.drafter_quant_declared = true;
        }
    }
    convert_options.components = components;
    return plan;
}

std::optional<ConvertedName> converted_name(const fs::path& artifact) {
    if (artifact.extension() != ".ninferno") { return std::nullopt; }
    const auto base = artifact.stem().string(); // <stem>[.vision].<recipe>
    const auto dot  = base.rfind('.');
    if (dot == std::string::npos) { return std::nullopt; }
    ConvertedName out;
    out.stem   = base.substr(0, dot);
    out.recipe = base.substr(dot + 1);
    out.vision = out.stem.ends_with(".vision");
    if (out.vision) { out.stem.resize(out.stem.size() - std::string_view(".vision").size()); }
    out.label = gguf_quant_label(out.stem + ".gguf");
    return out;
}

std::optional<fs::path> converted_artifact(const fs::path& commit_dir, const std::string& quant,
                                           const std::string& recipe) {
    const int width = quant_width_tag(quant);
    std::vector<std::pair<fs::path, ConvertedName>> found;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(commit_dir, ec)) {
        const auto path = entry.path();
        const auto name = converted_name(path);
        if (!name || !complete(path, false) || (!recipe.empty() && name->recipe != recipe)) {
            continue;
        }
        const bool matches =
            quant.empty() ||
            (width > 0 ? quant_label_bits(name->label) == width
                       : ascii_lower(name->stem).find(ascii_lower(quant)) != std::string::npos);
        if (matches) { found.emplace_back(path, *name); }
    }
    // A label that is part of another ("Q4_K_M" of "UD-Q4_K_M") picks the conversion whose own
    // label it is, as the fetch picks the GGUF model (select_files: an exact label wins).
    if (found.size() > 1 && width == 0 && !quant.empty()) {
        decltype(found) exact;
        for (const auto& f : found) {
            if (ascii_lower(f.second.label) == ascii_lower(quant)) { exact.push_back(f); }
        }
        if (exact.size() == 1) { found = exact; }
    }
    if (found.size() != 1) { return std::nullopt; }
    return found.front().first;
}

// The commit a kept conversion belongs to: its directory's name, or its parent's for one in
// options-<hash>/.
static std::string commit_of(const fs::path& artifact) {
    const auto dir = artifact.parent_path();
    return dir.filename().string().rfind("options-", 0) == 0 ? dir.parent_path().filename().string()
                                                             : dir.filename().string();
}

std::optional<fs::path> kept_artifact(const fs::path& cache, const std::string& repo,
                                      const std::string& revision, const std::string& quant,
                                      const std::string& recipe, const std::string& key) {
    const auto root   = cache / hub_repo_directory(repo);
    const auto commit = local_commit(root, revision);
    if (!commit) { return std::nullopt; }
    std::error_code ec;
    if (fs::exists(root / "snapshots" / *commit, ec)) { return std::nullopt; } // the usual path
    return converted_artifact(conversion_directory(root / "ninferno" / *commit, key), quant,
                              recipe);
}

namespace {

// Whether a conversion carries drafters a start serves: its manifest (of this artifact) names one
// other than its own MTP layers that is present (spec::ManifestDrafter::present), or its report
// lists a component beside the text, MTP and vision ones.
bool conversion_carries_drafters(const fs::path& artifact) {
    std::ifstream stream(artifact.string() + ".conversion.json");
    try {
        const OJson record = OJson::parse(stream);
        if (const auto manifest = spec::read_manifest(spec::manifest_path(artifact));
            manifest && manifest->target_artifact_id == record.value("artifact_id", std::string())) {
            for (const auto& d : manifest->drafters) {
                if (d.kind != "mtp" && d.present()) { return true; }
            }
        }
        const OJson components = record.value("components", OJson::object());
        for (const auto& [component, _] : components.items()) {
            if (component != "text" && component != "mtp" && component != "vision") {
                return true;
            }
        }
    } catch (const std::exception&) {}
    return false;
}

} // namespace

std::optional<fs::path> cached_conversion_with_drafters(const fs::path& cache,
                                                        const std::string& repo,
                                                        const std::string& revision,
                                                        const std::string& quant,
                                                        const std::string& recipe, bool vision) {
    if (quant.empty()) { return std::nullopt; }
    const auto root   = cache / hub_repo_directory(repo);
    const auto commit = local_commit(root, revision);
    if (!commit) { return std::nullopt; }
    // A conversion of an older revision is converted again where its download is still there
    // (acquire_model: cache_current), text only without --with-drafters; kept where it is gone.
    std::error_code gone;
    const bool download = fs::exists(root / "snapshots" / *commit, gone);
    // Complete conversions of exactly this label, the recipe and the --vision kind the start
    // converts ([0] text, [1] --vision), each with whether the start serves it with drafters.
    std::vector<std::pair<fs::path, bool>> found[2];
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(root / "ninferno" / *commit, ec)) {
        const auto path = entry.path();
        const auto name = converted_name(path);
        if (!name || !complete(path, false) || (!recipe.empty() && name->recipe != recipe) ||
            ascii_lower(name->label) != ascii_lower(quant)) {
            continue;
        }
        found[name->vision].emplace_back(
            path, (!download || cache_serves(path)) && conversion_carries_drafters(path));
    }
    auto& kept = found[vision && !found[1].empty() ? 1 : 0];
    if (kept.empty() ||
        !std::all_of(kept.begin(), kept.end(), [](const auto& k) { return k.second; })) {
        return std::nullopt;
    }
    std::sort(kept.begin(), kept.end());
    return kept.front().first;
}

bool cached_conversion_has_drafters(const fs::path& cache, const std::string& repo,
                                    const std::string& revision, const std::string& quant,
                                    const std::string& recipe, bool vision) {
    return cached_conversion_with_drafters(cache, repo, revision, quant, recipe, vision)
        .has_value();
}

std::string conversions_stamp(const fs::path& cache, const std::string& repo) {
    const fs::path root = cache / hub_repo_directory(repo);
    std::string out;
    std::error_code ec;
    // The refs: a ref that moves to another kept commit selects another conversion.
    for (auto it = fs::recursive_directory_iterator(root / "refs", ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        std::error_code e;
        if (!it->is_regular_file(e)) { continue; }
        std::ifstream in(it->path());
        std::string commit;
        in >> commit;
        out += it->path().lexically_relative(root / "refs").generic_string() + "=" + commit + ";";
    }
    // The commits downloaded: a conversion whose download is gone is kept as it is, one whose
    // download is there serves only at its own revision (cache_serves).
    ec.clear();
    for (auto it = fs::directory_iterator(root / "snapshots", ec);
         !ec && it != fs::directory_iterator(); it.increment(ec)) {
        out += "snapshot " + it->path().filename().string() + ";";
    }
    ec.clear();
    for (auto it = fs::recursive_directory_iterator(root / "ninferno", ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        std::error_code e;
        if (!it->is_regular_file(e)) { continue; }
        out += it->path().filename().string() + ":" + std::to_string(it->file_size(e)) + ":" +
               std::to_string(it->last_write_time(e).time_since_epoch().count()) + ";";
        if (!it->path().filename().string().ends_with(".speculation.json")) { continue; }
        const auto manifest = spec::read_manifest(it->path());
        if (!manifest) { continue; }
        for (const auto& d : manifest->drafters) {
            if (d.artifact.empty()) { continue; }
            std::error_code gone;
            const auto size = fs::file_size(d.artifact, gone);
            out += d.artifact + (gone ? std::string(":gone")
                                      : ":" + std::to_string(size) + ":" +
                                            std::to_string(fs::last_write_time(d.artifact, gone)
                                                               .time_since_epoch()
                                                               .count())) +
                   ";";
        }
    }
    return std::to_string(std::hash<std::string>{}(out));
}

void accept_converted(const fs::path& commit_dir, const std::string& ref,
                      const std::string& hub_commit) {
    const OJson record = {{"ref", ref}, {"hub_commit", hub_commit}};
    write_file_atomic(commit_dir / "accepted.json", record.dump(1) + "\n");
}

bool converted_accepted(const fs::path& commit_dir, const std::string& ref,
                        const std::string& hub_commit) {
    std::ifstream stream(commit_dir / "accepted.json");
    if (!stream) { return false; }
    try {
        const auto record = OJson::parse(stream);
        return record.value("ref", std::string()) == ref &&
               record.value("hub_commit", std::string()) == hub_commit;
    } catch (const std::exception&) { return false; }
}

bool is_repository_spec(const std::string& text) {
    static const std::regex pattern(R"([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(@[^\s/]+)?)");
    return std::regex_match(text, pattern) && !text.ends_with(".ninferno") && !text.ends_with(".gguf");
}

bool recipe_has_vision(const std::string& recipe) {
    return recipe.starts_with("qwen3_") ||
           (recipe.starts_with("gemma4") && recipe.find("assistant") == std::string::npos &&
            recipe.find("dflash") == std::string::npos);
}

std::vector<std::string> best_projector(const std::vector<std::string>& files) {
    // A split projector's parts (<name>-00001-of-0000N.gguf) are one projector, named by its group.
    static const std::regex split(R"((.*)-\d{5}-of-\d{5}\.gguf)", std::regex::icase);
    const auto group = [](const std::string& file) {
        std::smatch match;
        return std::regex_match(file, match, split) ? match[1].str() : file;
    };
    std::optional<std::string> best;
    int best_rank = 99;
    for (const auto& file : files) {
        const int r = projector_rank(fs::path(file).filename().string());
        if (r >= 0 && (r < best_rank || (r == best_rank && file < *best))) {
            best      = file;
            best_rank = r;
        }
    }
    std::vector<std::string> parts;
    if (!best) { return parts; }
    for (const auto& file : files) {
        if (group(file) == group(*best)) { parts.push_back(file); }
    }
    std::sort(parts.begin(), parts.end());
    return parts;
}

std::optional<std::filesystem::path> gguf_projector_beside(const std::filesystem::path& gguf) {
    std::optional<fs::path> best;
    int best_rank = 99;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(gguf.parent_path(), ec)) {
        const int r = projector_rank(entry.path().filename().string());
        if (r >= 0 && r < best_rank && entry.is_regular_file(ec)) {
            best      = entry.path();
            best_rank = r;
        }
    }
    return best;
}

// The description lines "--vocab q8" and "--name x" as "--vocab q8, --name x".
std::string options_line_of(const std::string& described) {
    std::string line;
    for (const char c : described) {
        if (c == '\n') {
            line += ", ";
        } else {
            line += c;
        }
    }
    if (line.size() >= 2) { line.resize(line.size() - 2); }
    return line;
}

namespace {

// Whether the resource `role` given as `path` is the snapshot's own file.
bool snapshot_resource(const fs::path& snapshot, const std::string& role, const fs::path& path) {
    std::error_code ec;
    return !snapshot.empty() && fs::is_directory(snapshot, ec) &&
           fs::is_regular_file(snapshot / role, ec) &&
           content_identity(snapshot / role) == content_identity(path);
}

} // namespace

std::string snapshot_resources(const ConvertOptions& options, const fs::path& snapshot) {
    auto resources = options.resources;
    std::sort(resources.begin(), resources.end());
    std::string lines;
    for (const auto& [role, path] : resources) {
        if (snapshot_resource(snapshot, role, path)) {
            lines += role + " " + content_identity(path) + '\n';
        }
    }
    return lines;
}

std::string content_options_key(const AcquireOptions& options, std::string* described,
                                const OptionsKeyContext& context, std::string* identity) {
    const ConvertOptions defaults;
    const auto& c = options.convert;
    std::vector<OptionPart> parts;
    const auto add    = [&](const std::string& name, const std::string& value,
                         const std::string& shown) { parts.push_back({name, value, shown}); };
    const auto simple = [&](const std::string& name, const std::string& value) {
        add(name, value, value);
    };
    // The vocabulary the dense and MoE recipes keep is the checkpoint's own: "native" is the
    // default spelled out.
    if (!c.vocab.empty() && c.vocab != "native") { simple("--vocab", c.vocab); }
    if (!c.drafter_quant.empty() && !c.drafter_quant_declared) {
        simple("--drafter-quant", c.drafter_quant);
    }
    if (c.name) { simple("--name", *c.name); }
    auto resources = c.resources;
    std::sort(resources.begin(), resources.end());
    for (const auto& [role, path] : resources) {
        // A resource that is the snapshot's own file (the template the model ships) changes
        // nothing.
        if (snapshot_resource(context.snapshot, role, path)) { continue; }
        add("--resource " + role, content_identity(path), utf8_file_name(path));
    }
    auto sources = c.sources;
    std::sort(sources.begin(), sources.end());
    for (const auto& [name, path] : sources) {
        add("--source " + name, content_identity(path), utf8_file_name(path));
    }
    if (options.components_given) { simple("--components", c.components); }
    if (c.proposal) { simple("--proposal", ""); }
    if (c.proposal_rows != defaults.proposal_rows) {
        simple("--proposal-rows", std::to_string(c.proposal_rows));
    }
    if (!c.proposal_default) { simple("--no-proposal", ""); }
    if (!c.ranking.empty()) {
        add("--ranking", content_identity(c.ranking), utf8_file_name(c.ranking));
    }
    if (c.max_file_bytes != defaults.max_file_bytes) {
        simple("--max-file-bytes", std::to_string(c.max_file_bytes));
    }
    if (c.maintainer_quantize) { simple("--maintainer-quantize", ""); }
    if (c.round_tiny_scales) { simple("--round-tiny-scales", ""); }
    if (c.expert_layout != defaults.expert_layout) {
        simple("--layout", c.expert_layout == ExpertLayout::Frag ? "frag" : "published");
    }
    if (options.drafters == Drafters::Off) { simple("--no-drafters", ""); }
    for (const auto& [name, path] : options.drafter_artifacts) {
        add("--drafter-artifact " + name, content_identity(path), utf8_file_name(path));
    }
    if (!options.model_id.empty() && options.model_id != context.repo) {
        simple("--model-id", options.model_id);
    }
    if (!options.base_tokenizer_files) { simple("--no-base-tokenizer", ""); }
    // The same files in any order are the same selection.
    auto include = options.fetch.include;
    std::sort(include.begin(), include.end());
    include.erase(std::unique(include.begin(), include.end()), include.end());
    for (const auto& pattern : include) { simple("--include", pattern); }
    return options_key(parts, described, identity);
}

std::optional<fs::path> kept_for_options(const fs::path& cache, const std::string& repo,
                                         const std::string& revision, const AcquireOptions& options,
                                         const OptionsKeyContext& context) {
    const auto& c = options.convert;
    if (auto kept = kept_artifact(cache, repo, revision, options.fetch.quant, c.recipe,
                                  content_options_key(options, nullptr, context))) {
        return kept;
    }
    // With the snapshot gone, a --resource that was its own file cannot be told from an option:
    // the conversions made without some of them are taken where they recorded those very files.
    const auto& resources = c.resources;
    if (context.snapshot.empty() && !resources.empty() && resources.size() <= 8) {
        for (std::size_t mask = 1; mask < (std::size_t{1} << resources.size()); ++mask) {
            AcquireOptions fewer = options;
            fewer.convert.resources.clear();
            std::vector<std::string> dropped;
            for (std::size_t i = 0; i < resources.size(); ++i) {
                if ((mask >> i) & 1U) {
                    dropped.push_back(resources[i].first + " " +
                                      content_identity(resources[i].second));
                } else {
                    fewer.convert.resources.push_back(resources[i]);
                }
            }
            const auto kept = kept_artifact(cache, repo, revision, options.fetch.quant, c.recipe,
                                            content_options_key(fewer, nullptr, context));
            if (!kept) { continue; }
            std::ifstream in(kept->string() + ".snapshot-resources");
            std::set<std::string> recorded;
            for (std::string line; std::getline(in, line);) { recorded.insert(line); }
            if (std::all_of(dropped.begin(), dropped.end(),
                            [&](const std::string& line) { return recorded.contains(line); })) {
                return kept;
            }
        }
    }
    // A model that ships no chat template is converted with the caller's as its resource, which
    // makes the conversion an optioned one only once the model is fetched (acquire_model); the
    // key before the fetch does not know. Looked for under that key too.
    if (options.chat_template.empty()) { return std::nullopt; }
    AcquireOptions baked = options;
    baked.convert.resources.emplace_back("chat_template.jinja", options.chat_template);
    return kept_artifact(cache, repo, revision, options.fetch.quant, c.recipe,
                         content_options_key(baked, nullptr, context));
}

namespace {

// A model that does not ship a chat template is converted with the caller's as its resource
// (#583): one known to ship none, and one whose answer is unknown (a GGUF naming a base
// repository, whose template may be missing or unreachable: base_tokenizer then fetches no other).
// A GGUF's chat_template.jinja beside it is its template. The conversion after the fetch and the
// refetch lookup of the kept one ask it the same way.
void bake_chat_template(ConvertOptions& convert, const AcquireOptions& options,
                        const fs::path& model) {
    if (options.chat_template.empty()) { return; }
    const bool ships =
        local_support_facts(model).chat_template.value_or(false) ||
        (fs::is_regular_file(model) && fs::exists(model.parent_path() / "chat_template.jinja"));
    if (!ships) { convert.resources.emplace_back("chat_template.jinja", options.chat_template); }
}

} // namespace

fs::path conversion_directory(const fs::path& commit_dir, const std::string& key) {
    return key.empty() ? commit_dir : commit_dir / ("options-" + key);
}

AcquiredModel acquire_model(const std::string& spec, AcquireOptions options) {
    AcquiredModel out;
    // MovedRef::Refetch: the old conversion goes once the new commit's artifact is there: the
    // artifact with its sidecars, and its commit's directory when nothing but its own drafters
    // (named in its speculation manifest) is left there (another quantization of that commit
    // stays).
    struct RemoveReplaced {
        fs::path artifact;
        const AcquiredModel& out;
        const AcquireOptions& options;
        ~RemoveReplaced() {
            if (artifact.empty() || std::uncaught_exceptions() > 0 || out.artifact.empty() ||
                out.artifact == artifact) {
                return;
            }
            auto dir                 = artifact.parent_path();
            const auto drafter_files = manifest_drafter_files(artifact);
            remove_artifact(artifact);
            std::error_code ec;
            // Whether a directory holds a conversion other than the replaced one's drafters.
            // Recursive: a conversion with options of its own lives in options-<hash>/ below, and
            // may be serving; it stays, with the commit it belongs to. A conversion still being
            // written below has no .ninferno file yet.
            // The walk takes its errors in `ec` (increment, not operator++): this destructor
            // must not throw.
            const auto others_in = [&](const fs::path& where) {
                for (auto it = fs::recursive_directory_iterator(where, ec);
                     !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
                    // By path, not by name: a conversion with options of its own below has the
                    // same file name as the one replaced.
                    std::error_code e;
                    if (it->path().extension() == ".ninferno" &&
                        !drafter_files.contains(fs::weakly_canonical(it->path(), e))) {
                        return true;
                    }
                }
                return ec ? true : conversion_in_progress(where);
            };
            bool others = others_in(dir);
            if (!others) { fs::remove_all(dir, ec); }
            // A replaced conversion with options of its own: its commit's directory is looked at
            // next, as App::delete_artifact does (its lock, accepted-ref records, drafters).
            if (dir.filename().string().rfind("options-", 0) == 0) {
                dir = dir.parent_path();
                if (!others && !others_in(dir)) { fs::remove_all(dir, ec); }
            }
            say(options, "removed the old conversion " + artifact.string() +
                             (others ? " (other conversions of that commit stay)" : ""));
        }
    } replaced{{}, out, options};
    ConvertOptions& convert_options = options.convert;
    // A conversion with options of its own is kept apart from the model's default one (#553): a
    // later run with other options never takes it, nor replaces it.
    const bool url = spec.starts_with("http://") || spec.starts_with("https://");
    std::string options_described;
    std::string options_identity; // what tells this list from another (options_key)
    OptionsKeyContext key_context;
    if (!url) { key_context.repo = split_spec(spec, options.fetch).first; }
    std::string variant_key = convert_options.out.empty()
                                  ? content_options_key(options, &options_described, key_context)
                                  : std::string();
    std::string preflight_recipe;
    std::string pinned; // org/name@commit when weights of a moved ref stay (see below)
    std::string pinned_quant; // the exact quantization of the GGUF model there
    if (!url && !options.fetch_only) {
        const auto [repo, revision] = split_spec(spec, options.fetch);
        // The download removed, its conversion kept: the one with these options is found that way
        // (the default's directory, or its options-<hash>/).
        if (options.reuse && convert_options.out.empty() &&
            (is_hugging_face_hub(options.fetch.hub))) {
            const auto cache =
                options.fetch.cache.empty() ? default_hub_cache() : options.fetch.cache;
            auto kept = kept_for_options(cache, repo, revision, options, key_context);
            // A branch or a tag may have moved since its weights came here, whether their download
            // was converted and removed (the kept artifact) or is still in the cache. They stay
            // unless the user asks for the new commit (a download of many gigabytes): the start
            // says so (MovedRef::Notice), or the user kept them for good (Accept), or asked for
            // the new ones (Refetch). The answer is the same whether the hub can be reached or
            // not: unreachable, it is only unchecked. Weights kept with their download start
            // pinned to their commit, so the fetch below finds them in the cache.
            const std::string ref   = revision.empty() ? std::string("main") : revision;
            const std::string label = repo + "@" + ref;
            const fs::path repo_dir = cache / hub_repo_directory(repo);
            std::string held_commit; // the commit of the weights here, when the ref names one
            if (kept) {
                held_commit = commit_of(*kept);
            } else if (const auto commit = local_commit(repo_dir, revision)) {
                // Only a snapshot holding the model this start asks for (its quantization, every
                // shard, the projector a --vision start takes): another quantization of the old
                // commit is no reason to pin to it.
                if (snapshot_holds_model(repo_dir / "snapshots" / *commit, options.fetch, repo,
                                         options.vision)) {
                    held_commit = *commit;
                }
            }
            if (!held_commit.empty() && revision != held_commit) {
                const fs::path commit_dir = repo_dir / "ninferno" / held_commit;
                std::optional<std::string> now;
                bool checked = false;
                try {
                    now = current_commit(spec, options.fetch);
                    if (!now) {
                        // Asked for the new commit: no hub to download it from is an error.
                        if (options.moved_ref == MovedRef::Refetch) {
                            throw std::runtime_error("HF_HUB_OFFLINE is set");
                        }
                        say(options, "HF_HUB_OFFLINE: " + label + " not checked against the hub");
                    }
                    checked = true;
                } catch (const std::exception& error) {
                    if (options.moved_ref == MovedRef::Refetch) {
                        throw std::runtime_error("--refetch: cannot ask the hub what " + label +
                                                 " names now (" + error.what() +
                                                 "); the weights of " +
                                                 held_commit.substr(0, 12) + " are kept");
                    }
                    say(options, "warning: could not ask the hub whether " + label +
                                     " moved (" + error.what() + "); serving the weights of " +
                                     held_commit.substr(0, 12));
                }
                bool refetch = false;
                if (now && *now != held_commit) {
                    const std::string moved =
                        label + " is now " + now->substr(0, 12) + "; the weights here are of " +
                        held_commit.substr(0, 12) +
                        (kept ? " (converted, download removed)" : " (downloaded)");
                    if (options.moved_ref == MovedRef::Refetch) {
                        say(options, moved + ": downloading the new commit (--refetch)");
                        // The old conversion goes once the new one is there (unless the caller
                        // keeps it: --keep-replaced); a download kept stays in the hub cache.
                        if (!options.keep_replaced) {
                            if (kept) {
                                replaced.artifact = *kept;
                            } else {
                                // The conversion of the model the old snapshot serves (of several
                                // a commit may hold, the one this start's selection takes).
                                // Its exact label: a width selector (--quant 4bit) or none
                                // matches several.
                                auto quant = snapshot_quant_label(
                                    repo_dir / "snapshots" / held_commit, options.fetch, repo);
                                if (quant.empty()) { quant = options.fetch.quant; }
                                // In this start's own directory of the commit: a conversion
                                // with options replaces its optioned one (options-<key>/),
                                // never the default one beside it, which may be serving. Its key
                                // as it was written, once that snapshot was known (below, after
                                // the fetch): the caller's template a resource where the model
                                // ships none, a resource that is the snapshot's own file none.
                                const auto held_snapshot = repo_dir / "snapshots" / held_commit;
                                // The model that snapshot serves, chosen as the fetch chooses it
                                // (a GGUF file of it, or the directory), and asked as the fetch's
                                // model is asked below.
                                FetchOptions held_fetch = options.fetch;
                                held_fetch.quant        = quant;
                                // Where that choice fails (an ambiguous label), the conversion
                                // is not looked up: a key asked of the wrong model could name
                                // the default conversion another server serves. It stays.
                                const auto chosen = snapshot_model(held_snapshot, held_fetch, repo);
                                const fs::path held_model = chosen.value_or(held_snapshot);
                                AcquireOptions as_written = options;
                                bake_chat_template(as_written.convert, options, held_model);
                                OptionsKeyContext written_context = key_context;
                                if (fs::is_directory(held_model)) {
                                    written_context.snapshot = held_model;
                                }
                                const std::string written_key =
                                    content_options_key(as_written, nullptr, written_context);
                                if (auto old =
                                        !chosen ? std::nullopt
                                                : converted_artifact(
                                                      conversion_directory(commit_dir, written_key),
                                                      quant, convert_options.recipe)) {
                                    replaced.artifact = *old;
                                }
                            }
                        }
                        kept.reset();
                        refetch = true;
                    } else if (options.moved_ref == MovedRef::Accept) {
                        std::error_code ec;
                        fs::create_directories(commit_dir, ec);
                        accept_converted(commit_dir, ref, *now);
                        say(options, moved + ": kept, now and later (--accept-converted)");
                    } else if (converted_accepted(commit_dir, ref, *now)) {
                        say(options, moved + ": kept, as accepted earlier");
                    } else {
                        const std::string notice =
                            "warning: " + moved + ". Serving these weights; --accept-converted "
                            "keeps them without this notice, --refetch downloads the new commit "
                            "and removes these";
                        say(options, notice);
                        if (options.on_moved_ref) { options.on_moved_ref(notice); }
                    }
                }
                // Weights kept with their download: the start below runs at their commit (the
                // fetch finds them in the cache, with the hub or without it).
                // The ref named the commit here when asked, or moved and the weights here stay:
                // the start runs at that commit either way, so a move after the question
                // downloads nothing unannounced.
                if (!kept && !refetch) {
                    pinned = repo + "@" + held_commit;
                    // Moved: the model held there, not what a width selector would take of the
                    // commit's whole listing (a larger one of that width not downloaded).
                    if (!checked || !now || *now != held_commit) {
                        pinned_quant = snapshot_quant_label(
                            repo_dir / "snapshots" / held_commit, options.fetch, repo);
                    }
                }
            }
            if (kept) {
                const auto base      = kept->stem().string();
                out.artifact         = *kept;
                out.recipe           = base.substr(base.rfind('.') + 1);
                out.reused           = true;
                out.fetched.repo     = repo;
                out.fetched.commit   = commit_of(*kept);
                out.fetched.snapshot = kept->parent_path();
                out.fetched.model    = *kept;
                out.fetched.offline  = true;
                say(options, "artifact " + kept->string() +
                                 " (converted earlier; its download was removed)");
                // cache_current stamps a current one, so only a defective artifact's report
                // is read on every start.
                if (!cache_current(*kept) && unquantized_drafter_attention(*kept)) {
                    say(options, "warning: this cached artifact predates conversion revision " +
                                     std::to_string(kCacheRevision) +
                                     " (its DFlash drafter is BF16: it runs the generic "
                                     "projections, slower than Q8), and its download is gone: "
                                     "delete " +
                                     kept->string() + " or fetch the model again");
                }
                // Its separate drafters planned from another registry, or one failed: planned
                // again from the artifact (a checkpoint's by default, a GGUF's when asked for), as
                // when its download was there. A companion it lacks stays reported missing: the
                // artifact cannot be converted again without that download.
                // A plan the checkpoint's own declaration (ninferno_speculation.json) made cannot
                // be made again without that download: kept as it is, and a drafter of it that
                // failed fails --with-drafters.
                refresh_drafters(options, *kept);
                const auto state = options.drafters == Drafters::Off ? ManifestState()
                                                                     : read_manifest_state(*kept);
                auto action = decide_drafters(options.drafters, state.gguf_source, false, state,
                                              options.components_given);
                std::optional<DrafterTarget> target;
                std::string target_error;
                if (action.kind == DrafterActionKind::Attach) {
                    // The plan is made from the artifact. A GGUF the manifest does not record
                    // (none, or one written before it did) is told by the artifact's provenance.
                    try {
                        target = reused_target(describe_artifact(*kept, options.model_id.empty()
                                                                            ? repo
                                                                            : options.model_id),
                                               state);
                        if (target->gguf_source && !state.gguf_source) {
                            action = decide_drafters(options.drafters, true, false, state,
                                                     options.components_given);
                        }
                    } catch (const std::exception& error) { target_error = error.what(); }
                }
                if (action.kind == DrafterActionKind::Fail) {
                    throw std::runtime_error("a drafter of " + kept->string() + " " +
                                             action.message + " (" + required_text(options) + ")");
                }
                if (action.kind == DrafterActionKind::Warn) {
                    not_attached(options, "a drafter of " + kept->string(), action.message, false);
                }
                if (action.kind == DrafterActionKind::Attach) {
                    try {
                        if (!target) { throw std::runtime_error(target_error); }
                        AttachOptions attach;
                        attach.fetch                 = options.fetch;
                        attach.fetch.before_download = [](const FetchPlan& plan) {
                            check_space(plan);
                        };
                        attach.log               = options.log;
                        attach.repository        = options.repository;
                        attach.drafter_artifacts = options.drafter_artifacts;
                        attach.conversion        = convert_options;
                        out.drafters = attach_drafters(plan_drafters(*target), *kept, attach);
                        require_drafters(*out.drafters, options);
                    } catch (const std::exception& error) {
                        if (options.drafters == Drafters::Require) { throw; }
                        not_attached(options, "drafters not attached", error.what(), true);
                    }
                }
                return out;
            }
        }
        // The commit the start runs at: the one the weights here are of when pinned.
        const auto at = pinned.empty() ? revision : split_spec(pinned, options.fetch).second;
        // Whether the engine runs the model at all, from what the hub shows before a download
        // (convert/support.h). A hub not asked (offline, ModelScope) or not answering leaves it
        // to the conversion.
        std::optional<SupportFacts> facts;
        try {
            auto ask     = options.fetch;
            ask.revision = at;
            if (!pinned_quant.empty()) { ask.quant = pinned_quant; }
            facts = hub_support_facts(repo, ask);
        } catch (const std::exception&) {}
        if (facts) {
            // A template given (--resource chat_template.jinja=FILE) is the template.
            if (!options.chat_template.empty() ||
                std::any_of(options.convert.resources.begin(), options.convert.resources.end(),
                            [](const auto& r) { return r.first == "chat_template.jinja"; })) {
                facts->chat_template = true;
            }
            const auto support = model_support(*facts);
            if (support.verdict == Verdict::Unsupported || support.verdict == Verdict::NeedsToken) {
                throw std::runtime_error(repo + " cannot be converted (nothing was downloaded): " +
                                         support.reason);
            }
        }
        preflight_recipe = preflight(repo, at, options);
        if (!preflight_recipe.empty()) {
            say(options, "recipe " + preflight_recipe + " (from config.json)");
        }
    }
    auto fetch            = options.fetch;
    const auto caller     = options.before_download;
    fetch.before_download = [&](const FetchPlan& plan) {
        check_space(plan);
        if (caller) { caller(plan, preflight_recipe); }
    };
    if (!pinned_quant.empty()) { fetch.quant = pinned_quant; }
    out.fetched = fetch_model(pinned.empty() ? spec : pinned, fetch);
    const FetchResult& fetched = out.fetched;
    say(options, "fetched " + fetched.repo + "@" + fetched.commit.substr(0, 12) + " -> " +
                     fetched.model.string());
    if (fetched.offline && !is_convertible(fetched.model)) {
        throw std::runtime_error(
            "the hub is unreachable and the cached snapshot " + fetched.snapshot.string() +
            " is incomplete (no weights): connect to the hub once to finish the download");
    }
    if (options.fetch_only) { return out; }
    convert_options.model = fetched.model;
    // Known now: a resource that is the snapshot's own file is no option (the key before the
    // fetch counted it).
    if (convert_options.out.empty()) {
        bake_chat_template(convert_options, options, fetched.model);
        if (fs::is_directory(fetched.model)) { key_context.snapshot = fetched.model; }
        variant_key =
            content_options_key(options, &options_described, key_context, &options_identity);
    }
    if (options.base_tokenizer_files) { base_tokenizer(fetched, fetch, convert_options); }

    // Beside the hub cache's snapshots, per commit (a URL: beside the file); the file is named
    // <model stem>.<recipe>.ninferno once the recipe is known.
    const bool default_out = convert_options.out.empty();
    const auto directory =
        conversion_directory(fetched.commit.empty() ? fetched.snapshot
                                                    : fetched.snapshot.parent_path().parent_path() /
                                                          "ninferno" / fetched.commit,
                             variant_key);
    const std::string stem = fs::is_regular_file(fetched.model) ? fetched.model.stem().string()
                                                                : std::string("model");
    RecipeChoice choice;
    if (convert_options.recipe.empty()) {
        choice                 = choose_recipe(convert_options.model, GgufRequest::Model,
                                               keeps_mtp(options, convert_options));
        convert_options.recipe = choice.recipe;
        for (auto& [name, path] : choice.sources) {
            bool given = false;
            for (const auto& [existing, _] : convert_options.sources) {
                given = given || existing == name;
            }
            if (!given) { convert_options.sources.emplace_back(name, path); }
        }
    }
    out.recipe      = convert_options.recipe;
    bool vision     = false;
    if (options.vision && recipe_has_vision(out.recipe)) {
        if (fs::is_directory(fetched.model)) {
            vision = checkpoint_has_vision(fetched.model);
        } else {
            // A GGUF model's Vision encoder is its projector file (mmproj), converted with it.
            bool given = false;
            for (const auto& [name, _] : convert_options.sources) { given = given || name == "vision"; }
            if (!given) {
                // Pinned to the weights here: the projector here too (snapshot_holds_model found
                // one whole), not a better one the old commit lists.
                if (const auto projector = pinned.empty()
                                               ? gguf_projector(fetched, fetch, options.log)
                                               : gguf_projector_beside(fetched.model)) {
                    convert_options.sources.emplace_back("vision", *projector);
                    given = true;
                }
            }
            vision = given;
        }
    }
    if (options.vision) {
        if (vision) {
            if (("," + convert_options.components + ",").find(",vision,") == std::string::npos) {
                convert_options.components += ",vision";
            }
        } else {
            say(options,
                "no Vision encoder converted: " +
                    std::string(fs::is_directory(fetched.model)
                                    ? "this checkpoint publishes none NInferno runs"
                                    : "no vision projector (mmproj-*.gguf) beside this GGUF model "
                                      "or in its repository"));
        }
    }
    if (default_out) {
        convert_options.out =
            directory / (stem + (vision ? ".vision." : ".") + out.recipe + ".ninferno");
    }
    out.artifact = convert_options.out;
    // "--vocab q8, --resource chat_template.jinja custom.jinja": the options that made this
    // conversion its own, for the log and for <artifact>.options.
    const std::string options_line      = options_line_of(options_described);
    const std::string converted_earlier = variant_key.empty()
                                              ? " (converted earlier)"
                                              : " (converted earlier with " + options_line + ")";
    if (!variant_key.empty()) {
        say(options, "conversion options of their own, kept apart from the default conversion: " +
                         options_line);
    }
    // A GGUF model carries no checkpoint to plan from: its drafters are planned for the converted
    // artifact (its text component's architecture and geometry, its tokenizer), once converted.
    // They are attached when asked for (a separate download, 1.6 GB for Muse Glimmer's DFlash);
    // a checkpoint's by default.
    const bool gguf        = !fs::is_directory(fetched.model);
    const bool drafters_on = drafters_enabled(options.drafters, gguf);

    fs::create_directories(convert_options.out.parent_path());
    AttachOptions attach;
    attach.fetch                 = fetch;
    attach.fetch.before_download = [](const FetchPlan& plan) { check_space(plan); };
    attach.log                   = options.log;
    attach.repository            = options.repository;
    attach.drafter_artifacts     = options.drafter_artifacts;
    attach.conversion            = convert_options;
    // The MTP layers a model carries are part of it: kept unless the caller named the components
    // or asked for no drafters (as ninferno-convert does).
    if (!options.components_given && options.drafters != Drafters::Off) {
        convert_options.components_auto = true;
    }
    const std::string model_id = options.model_id.empty() ? fetched.repo : options.model_id;
    // The companions convert_gguf_companions could not fetch or convert in. The plan made from
    // the artifact afterwards has them missing; this carries the failure into it, so the manifest
    // lists them as failed (the next run tries again) and one warning names the cause.
    std::vector<PlannedDrafter> companion_failures;
    // The records a later start reads beside the conversion it serves: its options for the user
    // and their identity, and the resources it took as the snapshot's own (kept_for_options). A
    // note: none is needed to serve it.
    const auto record_options = [&] {
        try {
            if (!variant_key.empty()) {
                write_file_atomic(out.artifact.string() + ".options", options_described);
                write_file_atomic(out.artifact.string() + ".options-id", options_identity + '\n');
            }
            if (const auto own = snapshot_resources(convert_options, key_context.snapshot);
                !own.empty()) {
                write_file_atomic(out.artifact.string() + ".snapshot-resources", own);
            }
        } catch (const std::exception&) {}
    };
    // Whatever fails here, the model itself is served as it was before drafters were planned.
    const auto attach_to_artifact = [&] {
        try {
            const auto target = describe_gguf_artifact(out.artifact, model_id, fetched.model);
            auto plan         = plan_drafters(target);
            carry_failures(plan, companion_failures);
            out.drafters = attach_drafters(std::move(plan), out.artifact, attach);
            require_drafters(*out.drafters, options);
        } catch (const std::exception& error) {
            // Asked for by name (--with-drafters): a drafter that cannot be attached is an error,
            // not a model served slower without it.
            if (options.drafters == Drafters::Require) { throw; }
            not_attached(options, "drafters not attached", error.what(), true);
        }
    };
    // A GGUF model's companion drafters (Qwen3.8's DFlash2) are components of the artifact, planned
    // from its converted text: they are fetched and the model is converted again with them (in
    // `quant`, else the registry's). If that fails, it is converted as before, without them.
    const auto convert_gguf_companions = [&](const ConvertOptions& text_only,
                                             const std::string& quant) {
        std::vector<std::pair<std::string, fs::path>> companions;
        std::string declared_quant;
        std::vector<PlannedDrafter> fetched_companions;
        try {
            auto plan =
                plan_drafters(describe_gguf_artifact(out.artifact, model_id, fetched.model));
            const auto& have = plan.target.components;
            if (std::any_of(plan.drafters.begin(), plan.drafters.end(), [&](const auto& d) {
                    return !d.decl.component.empty() && d.decl.kind != "mtp" &&
                           std::find(have.begin(), have.end(), d.decl.component) != have.end();
                })) {
                return; // converted with its companion already
            }
            companions = fetch_companions(plan, attach);
            if (options.drafters == Drafters::Require) { require_drafters(plan, options); }
            for (const auto& d : plan.drafters) {
                if (d.status == "fetched" && !d.decl.component.empty()) {
                    declared_quant = d.decl.quant;
                    fetched_companions.push_back(d);
                } else if (d.status.starts_with("failed") || d.status.starts_with("incompatible")) {
                    companion_failures.push_back(d); // warned by attach_to_artifact
                }
            }
        } catch (const std::exception& error) {
            if (options.drafters == Drafters::Require) { throw; }
            not_attached(options, "companion drafters not planned", error.what(), true);
        }
        if (companions.empty()) { return; }
        ConvertOptions with = text_only;
        for (const auto& [component, snapshot] : companions) {
            with.sources.emplace_back(component, snapshot);
        }
        with.drafter_quant          = quant.empty() ? declared_quant : quant;
        with.drafter_quant_declared = quant.empty();
        say(options, "converting " + fetched.repo + " again with its companion drafters");
        remove_artifact(out.artifact);
        try {
            out.conversion = convert(with);
        } catch (const std::exception& error) {
            if (options.drafters == Drafters::Require) { throw; }
            if (options.drafters == Drafters::Attach) {
                // Said by attach_to_artifact, which finds them missing from the artifact.
                for (auto d : fetched_companions) {
                    d.status       = std::string("failed: ") + error.what();
                    d.fetch_failed = false;
                    companion_failures.push_back(std::move(d));
                }
            } else {
                say(options, std::string("companion drafters not converted in: ") + error.what());
            }
            remove_artifact(out.artifact);
            out.conversion = convert(text_only);
        }
        stamp_revision(out.artifact);
    };
    std::optional<LockFile> lock;
    if (default_out) { lock.emplace(convert_options.out.parent_path() / ".convert.lock"); }
    if (!options.reuse && default_out && fs::exists(out.artifact)) {
        // The kept artifact is keyed by commit, quant and recipe: a flag that changes what is
        // written replaces it.
        say(options, "converting " + fetched.repo + " again: a flag changes what is written");
        remove_artifact(out.artifact);
    }
    if (options.reuse) {
        if (!variant_key.empty() && complete(out.artifact, false)) {
            // The options this conversion was made with, as it recorded them: another list under
            // the same name is not this conversion.
            // By identity (a file by its content, wherever it lies now); a conversion that recorded
            // none (made before) by its list.
            std::ifstream recorded(out.artifact.string() + ".options");
            const std::string text((std::istreambuf_iterator<char>(recorded)),
                                   std::istreambuf_iterator<char>());
            std::ifstream recorded_id(out.artifact.string() + ".options-id");
            std::string id;
            const bool other = recorded_id && std::getline(recorded_id, id)
                                   ? id != options_identity
                                   : recorded && text != options_described;
            if (other) {
                say(options, "converting " + fetched.repo +
                                 " again: its cached artifact was made with other options (" +
                                 options_line_of(text) + ")");
                remove_artifact(out.artifact);
            }
        }
        if (complete(out.artifact, false) && !cache_current(out.artifact)) {
            say(options, "converting " + fetched.repo + " again: its cached artifact is conversion " +
                             "revision " + std::to_string(cached_revision(out.artifact)) +
                             ", this NInferno makes revision " + std::to_string(kCacheRevision));
            remove_artifact(out.artifact);
        }
        if (complete(out.artifact, false)) {
            refresh_drafters(options, out.artifact);
            const auto action = decide_drafters(options.drafters, gguf, true,
                                                options.drafters == Drafters::Off
                                                    ? ManifestState()
                                                    : read_manifest_state(out.artifact),
                                                options.components_given);
            if (action.kind == DrafterActionKind::Attach ||
                action.kind == DrafterActionKind::CompanionsThenAttach) {
                // A GGUF's, converted before its drafters could be attached, or one failed: its
                // companions are converted in and the others attached now.
                out.reused = true;
                say(options, "artifact " + out.artifact.string() + converted_earlier);
                if (action.kind == DrafterActionKind::CompanionsThenAttach) {
                    auto text_only          = convert_options;
                    text_only.drafter_quant = std::string();
                    convert_gguf_companions(text_only, convert_options.drafter_quant);
                }
                attach_to_artifact();
                record_options();
                return out;
            }
            if (action.kind == DrafterActionKind::Reconvert) {
                // A checkpoint's artifact with no manifest was converted without its drafters
                // (--no-drafters): its MTP layers and companions are missing, so it is replaced.
                say(options, "converting " + fetched.repo +
                                 " again: converted earlier without its drafters");
                remove_artifact(out.artifact);
            } else {
                // Its separate drafters planned from another registry (one whose kinds convert
                // them otherwise), or one failed: planned again from the checkpoint (its
                // declaration and lineage), as at first. A companion that plan adds is converted
                // into the artifact, so the artifact is converted again.
                std::optional<DrafterPlan> replanned;
                std::string missing;
                std::string replan_error;
                if (action.kind == DrafterActionKind::ReplanFromCheckpoint) {
                    try {
                        replanned = plan_drafters(describe_checkpoint(fetched.model, model_id));
                        const artifact::Reader reader(out.artifact);
                        for (const PlannedDrafter& d : replanned->drafters) {
                            if (d.supported && !d.decl.component.empty() &&
                                !reader.directory().components.count(d.decl.component)) {
                                missing = d.decl.component;
                            }
                        }
                    } catch (const std::exception& error) {
                        replan_error = error.what();
                    }
                }
                if (missing.empty()) {
                    out.reused = true;
                    say(options, "artifact " + out.artifact.string() + converted_earlier);
                    try {
                        if (!replan_error.empty()) { throw std::runtime_error(replan_error); }
                        if (replanned) {
                            out.drafters = attach_drafters(*replanned, out.artifact, attach);
                            require_drafters(*out.drafters, options);
                        }
                    } catch (const std::exception& error) {
                        if (options.drafters == Drafters::Require) { throw; }
                        not_attached(options, "drafters not attached", error.what(), true);
                    }
                    record_options();
                    return out;
                }
                say(options, "converting " + fetched.repo + " again: its drafter component " +
                                 missing + " is not in the artifact");
                remove_artifact(out.artifact);
            }
        }
        // A report without its artifact is what an interrupted conversion leaves behind.
        std::error_code ignored;
        fs::remove(out.artifact.string() + ".conversion.json", ignored);
    }

    if (drafters_on && !fetched.commit.empty()) {
        // What the model says about its drafters: a bundle declaration, and the model card's
        // base_model lineage (small files, at the same commit).
        auto small     = attach.fetch;
        small.include  = {"README.md", "ninferno_speculation.json"};
        small.quant.clear();
        small.revision = fetched.commit;
        try {
            (void)fetch_model(fetched.repo, small);
        } catch (const std::exception&) {} // neither file: nothing declared
    }
    // A GGUF target is converted first without companions: --drafter-quant is its drafters' and
    // its companions' (convert_gguf_companions).
    const std::string drafter_quant = convert_options.drafter_quant;
    if (drafters_on && gguf) { convert_options.drafter_quant.clear(); }
    if (drafters_on && !gguf) {
        out.drafters = plan_checkpoint_drafters(fetched.model, model_id, options.components_given,
                                                convert_options, attach);
    }
    say(options, "recipe " + out.recipe + (choice.note.empty() ? "" : " (" + choice.note + ")"));
    out.conversion = convert(convert_options);
    stamp_revision(out.artifact);
    if (out.drafters) {
        out.drafters = attach_drafters(std::move(*out.drafters), out.artifact, attach);
        require_drafters(*out.drafters, options);
    } else if (drafters_on && gguf) {
        if (!options.components_given) { convert_gguf_companions(convert_options, drafter_quant); }
        attach_to_artifact();
    }
    record_options();
    return out;
}

} // namespace ninferno::convert
