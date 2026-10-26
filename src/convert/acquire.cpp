#include "convert/acquire.h"

#include "core/ascii.h"
#include "core/quant_label.h"
#include "convert/files.h"

#include "convert/cache.h"
#include "convert/checks.h"
#include "convert/dense.h"
#include "convert/pyjson.h"
#include "convert/safetensors.h"
#include "convert/support.h"
#include "core/cache_file.h"
#include "core/host_memory.h"
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
    return (!options.components_given && options.with_drafters != false) ||
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

// Under --with-drafters every drafter planned for the model is required: one that failed to be
// fetched, converted or attached, or that does not fit the target, is an error, not a model served
// without it.
void require_drafters(const DrafterPlan& plan, const AcquireOptions& options) {
    if (options.with_drafters != true) { return; }
    for (const auto& d : plan.drafters) {
        if (d.status.starts_with("failed") || d.status.starts_with("incompatible")) {
            throw std::runtime_error("drafter " + d.decl.name + " " + d.status +
                                     " (--with-drafters requires every drafter; without the "
                                     "flag the model is served without it)");
        }
    }
}

} // namespace

void check_free_space(const FetchPlan& plan) { check_space(plan); }

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

std::optional<fs::path> kept_artifact(const fs::path& cache, const std::string& repo,
                                      const std::string& revision, const std::string& quant,
                                      const std::string& recipe) {
    const auto root   = cache / hub_repo_directory(repo);
    const auto commit = local_commit(root, revision);
    if (!commit) { return std::nullopt; }
    std::error_code ec;
    if (fs::exists(root / "snapshots" / *commit, ec)) { return std::nullopt; } // the usual path
    return converted_artifact(root / "ninferno" / *commit, quant, recipe);
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
            const auto dir = artifact.parent_path();
            std::string manifest;
            if (std::ifstream in(spec::manifest_path(artifact)); in) {
                manifest.assign(std::istreambuf_iterator<char>(in), {});
            }
            remove_artifact(artifact);
            std::error_code ec;
            bool others = false;
            for (const auto& entry : fs::directory_iterator(dir, ec)) {
                const auto name = entry.path().filename().string();
                if (entry.path().extension() == ".ninferno" &&
                    manifest.find(name) == std::string::npos) {
                    others = true;
                }
            }
            if (!others) { fs::remove_all(dir, ec); }
            say(options, "removed the old conversion " + artifact.string() +
                             (others ? " (other conversions of that commit stay)" : ""));
        }
    } replaced{{}, out, options};
    ConvertOptions& convert_options = options.convert;
    const bool url = spec.starts_with("http://") || spec.starts_with("https://");
    std::string preflight_recipe;
    std::string pinned; // org/name@commit when weights of a moved ref stay (see below)
    std::string pinned_quant; // the exact quantization of the GGUF model there
    if (!url && !options.fetch_only) {
        const auto [repo, revision] = split_spec(spec, options.fetch);
        if (options.reuse && convert_options.out.empty() &&
            (is_hugging_face_hub(options.fetch.hub))) {
            const auto cache =
                options.fetch.cache.empty() ? default_hub_cache() : options.fetch.cache;
            auto kept = kept_artifact(cache, repo, revision, options.fetch.quant,
                                      convert_options.recipe);
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
                held_commit = kept->parent_path().filename().string();
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
                                if (auto old = converted_artifact(commit_dir, quant,
                                                                  convert_options.recipe)) {
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
                out.fetched.commit   = kept->parent_path().filename().string();
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
            if (std::any_of(options.convert.resources.begin(), options.convert.resources.end(),
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
    if (options.base_tokenizer_files) { base_tokenizer(fetched, fetch, convert_options); }

    // Beside the hub cache's snapshots, per commit (a URL: beside the file); the file is named
    // <model stem>.<recipe>.ninferno once the recipe is known.
    const bool default_out = convert_options.out.empty();
    const auto directory =
        fetched.commit.empty()
            ? fetched.snapshot
            : fetched.snapshot.parent_path().parent_path() / "ninferno" / fetched.commit;
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
    // A GGUF model carries no checkpoint to plan from: its drafters are planned for the converted
    // artifact (its text component's architecture and geometry, its tokenizer), once converted.
    // They are attached when asked for (a separate download, 1.6 GB for Muse Glimmer's DFlash);
    // a checkpoint's by default.
    const bool gguf        = !fs::is_directory(fetched.model);
    const bool drafters_on = options.with_drafters.value_or(!gguf);

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
    if (!options.components_given && options.with_drafters != false) {
        convert_options.components_auto = true;
    }
    const std::string model_id = options.model_id.empty() ? fetched.repo : options.model_id;
    // Whatever fails here, the model itself is served as it was before drafters were planned.
    const auto attach_to_artifact = [&] {
        try {
            const auto target = describe_gguf_artifact(out.artifact, model_id, fetched.model);
            out.drafters      = attach_drafters(plan_drafters(target), out.artifact, attach);
            require_drafters(*out.drafters, options);
        } catch (const std::exception& error) {
            // Asked for by name (--with-drafters): a drafter that cannot be attached is an error,
            // not a model served slower without it.
            if (options.with_drafters == true) { throw; }
            say(options, std::string("drafters not attached: ") + error.what());
        }
    };
    // A GGUF model's companion drafters (Qwen3.8's DFlash2) are components of the artifact, planned
    // from its converted text: they are fetched and the model is converted again with them (in
    // `quant`, else the registry's). If that fails, it is converted as before, without them.
    const auto convert_gguf_companions = [&](const ConvertOptions& text_only,
                                             const std::string& quant) {
        std::vector<std::pair<std::string, fs::path>> companions;
        std::string declared_quant;
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
            require_drafters(plan, options);
            for (const auto& d : plan.drafters) {
                if (d.status == "fetched" && !d.decl.component.empty()) {
                    declared_quant = d.decl.quant;
                }
            }
        } catch (const std::exception& error) {
            if (options.with_drafters == true) { throw; }
            say(options, std::string("companion drafters not planned: ") + error.what());
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
            if (options.with_drafters == true) { throw; }
            say(options, std::string("companion drafters not converted in: ") + error.what());
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
        if (complete(out.artifact, false) && !cache_current(out.artifact)) {
            say(options, "converting " + fetched.repo + " again: its cached artifact is conversion " +
                             "revision " + std::to_string(cached_revision(out.artifact)) +
                             ", this NInferno makes revision " + std::to_string(kCacheRevision));
            remove_artifact(out.artifact);
        }
        if (complete(out.artifact, false)) {
            const bool attach_again = gguf ? drafters_incomplete(out.artifact)
                                           : !fs::exists(spec::manifest_path(out.artifact));
            if (drafters_on && attach_again) {
                if (gguf) {
                    // Converted before its drafters could be attached, or one failed: its
                    // companions are converted in and the others attached now.
                    out.reused = true;
                    say(options, "artifact " + out.artifact.string() + " (converted earlier)");
                    if (!options.components_given) {
                        auto text_only          = convert_options;
                        text_only.drafter_quant = std::string();
                        convert_gguf_companions(text_only, convert_options.drafter_quant);
                    }
                    attach_to_artifact();
                    return out;
                }
                // A checkpoint's artifact with no manifest was converted without its drafters
                // (--no-drafters): its MTP layers and companions are missing, so it is replaced.
                say(options, "converting " + fetched.repo +
                                 " again: converted earlier without its drafters");
                remove_artifact(out.artifact);
            } else {
                out.reused = true;
                say(options, "artifact " + out.artifact.string() + " (converted earlier)");
                return out;
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
    return out;
}

} // namespace ninferno::convert
