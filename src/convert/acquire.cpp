#include "convert/acquire.h"
#include "core/ascii.h"
#include "convert/files.h"

#include "convert/cache.h"
#include "convert/checks.h"
#include "convert/dense.h"
#include "convert/pyjson.h"
#include "convert/safetensors.h"
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

// A GGUF model's Vision projector (mmproj-*.gguf): beside the model file (a local directory, or
// the hub snapshot when it was fetched before), else fetched from the model's repository at the
// same commit. BF16 first (the published encoder's own type, kept exactly), then F32, then F16
// (rounded to a narrower range). nullopt when there is none.
int projector_rank(std::string_view file_name) {
    const std::string name = ascii_lower(file_name);
    if (!name.starts_with("mmproj") || !name.ends_with(".gguf")) { return -1; }
    return name.find("bf16") != std::string::npos  ? 0
           : name.find("f32") != std::string::npos ? 1
           : name.find("f16") != std::string::npos ? 2
                                                   : 3;
}

std::optional<fs::path> gguf_projector(const FetchResult& fetched, const FetchOptions& fetch,
                                       const std::function<void(const std::string&)>& log) {
    if (auto beside = gguf_projector_beside(fetched.model); beside || fetched.repo.empty()) {
        return beside;
    }
    std::error_code ec;
    for (const char* pattern : {"mmproj-BF16.gguf", "mmproj-F32.gguf", "mmproj-F16.gguf",
                                "mmproj*.gguf"}) {
        auto small     = fetch;
        small.include  = {pattern};
        small.quant.clear();
        small.revision = fetched.commit;
        try {
            const auto files = fetch_model(fetched.repo, small);
            for (const auto& file : files.files) {
                const auto path = files.snapshot / file.path;
                if (projector_rank(path.filename().string()) >= 0 &&
                    fs::is_regular_file(path, ec)) {
                    if (log) { log("vision projector " + fetched.repo + "/" + file.path); }
                    return path;
                }
            }
        } catch (const std::exception& error) {
            const std::string what = error.what();
            if (what.find("no file matches") == std::string::npos) { throw; }
        }
    }
    return std::nullopt;
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

std::optional<fs::path> kept_artifact(const fs::path& cache, const std::string& repo,
                                      const std::string& revision, const std::string& quant,
                                      const std::string& recipe) {
    const auto root = cache / hub_repo_directory(repo);
    std::ifstream ref(root / "refs" / (revision.empty() ? std::string("main") : revision));
    std::string commit;
    if (!(ref >> commit) || commit.find_first_of("/\\.") != std::string::npos) {
        return std::nullopt;
    }
    std::error_code ec;
    if (fs::exists(root / "snapshots" / commit, ec)) { return std::nullopt; } // the usual path
    const int width = quant_width_tag(quant);
    std::vector<fs::path> found;
    for (const auto& entry : fs::directory_iterator(root / "ninferno" / commit, ec)) {
        const auto path = entry.path();
        if (path.extension() != ".ninferno" || !complete(path, false)) { continue; }
        const auto base = path.stem().string(); // <stem>.<recipe>
        const auto dot  = base.rfind('.');
        if (dot == std::string::npos) { continue; }
        const auto stem = base.substr(0, dot);
        if (!recipe.empty() && base.substr(dot + 1) != recipe) { continue; }
        const bool matches =
            quant.empty() ||
            (width > 0 ? quant_label_bits(gguf_quant_label(stem + ".gguf")) == width
                       : ascii_lower(stem).find(ascii_lower(quant)) != std::string::npos);
        if (matches) { found.push_back(path); }
    }
    if (found.size() != 1) { return std::nullopt; }
    return found.front();
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
    if (!url && !options.fetch_only) {
        const auto [repo, revision] = split_spec(spec, options.fetch);
        if (options.reuse && convert_options.out.empty() &&
            (is_hugging_face_hub(options.fetch.hub))) {
            const auto cache =
                options.fetch.cache.empty() ? default_hub_cache() : options.fetch.cache;
            auto kept = kept_artifact(cache, repo, revision, options.fetch.quant,
                                      convert_options.recipe);
            // A branch or a tag may have moved since its download was converted and removed. The
            // converted weights stay unless the user asks for the new commit (a download of many
            // gigabytes): the start says so (MovedRef::Notice), or the user kept them for good
            // (Accept), or asked for the new ones (Refetch). The answer is the same whether the
            // hub can be reached or not: unreachable, it is only unchecked.
            const std::string ref = revision.empty() ? std::string("main") : revision;
            if (kept && revision != kept->parent_path().filename().string()) {
                const auto commit_dir         = kept->parent_path();
                const std::string kept_commit = commit_dir.filename().string();
                const std::string label       = repo + "@" + ref;
                std::optional<std::string> now;
                try {
                    now = current_commit(spec, options.fetch);
                    if (!now) {
                        say(options, "HF_HUB_OFFLINE: " + label + " not checked against the hub");
                    }
                } catch (const std::exception& error) {
                    say(options, "warning: could not ask the hub whether " + label +
                                     " moved (" + error.what() + "); serving its conversion of " +
                                     kept_commit.substr(0, 12));
                }
                if (now && *now != kept_commit) {
                    const std::string moved = label + " is now " + now->substr(0, 12) +
                                              "; the converted weights are of " +
                                              kept_commit.substr(0, 12) + " (download removed)";
                    if (options.moved_ref == MovedRef::Refetch) {
                        say(options, moved + ": downloading the new commit (--refetch)");
                        replaced.artifact = *kept;
                        kept.reset();
                    } else if (options.moved_ref == MovedRef::Accept) {
                        accept_converted(commit_dir, ref, *now);
                        say(options, moved + ": kept, now and later (--accept-converted)");
                    } else if (converted_accepted(commit_dir, ref, *now)) {
                        say(options, moved + ": kept, as accepted earlier");
                    } else {
                        const std::string notice =
                            "warning: " + moved + ". Serving the converted weights; "
                            "--accept-converted keeps them without this notice, --refetch "
                            "downloads the new commit and removes these";
                        say(options, notice);
                        if (options.on_moved_ref) { options.on_moved_ref(notice); }
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
        preflight_recipe = preflight(repo, revision, options);
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
    out.fetched = fetch_model(spec, fetch);
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
                if (const auto projector = gguf_projector(fetched, fetch, options.log)) {
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
