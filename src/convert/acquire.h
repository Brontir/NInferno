#pragma once

// A model named by its hub id, made ready to load: fetched into the hub cache, converted with the
// official recipe for its architecture into <cache>/models--org--name/ninferno/<commit>/, with the
// drafters the checkpoint or the official registry declares attached. ninferno-convert --fetch and
// ninferno-serve <org/name> share it.
//
//   preflight  before the weights are downloaded, config.json alone says whether the architecture
//              is supported and which recipe converts it (an unsupported model is refused before
//              gigabytes move), and the free space of the cache is checked against the download;
//              the caller may refuse too (before_download), e.g. weights larger than its device;
//   reuse      with `reuse`, an artifact converted by an earlier run for the same commit and
//              recipe (and its drafter manifest) is loaded as is: nothing is converted twice, and
//              with the hub unreachable the cached snapshot's artifact still serves;
//   locking    one process converts a commit at a time (a lock file beside the artifacts); the
//              next one finds the artifact;
//   kept       with `reuse`, an artifact whose download was removed after it was converted
//              (ninferno-ui removes the original once the model loaded: blobs/ and snapshots/ go,
//              refs/ and ninferno/ stay) is loaded at the commit the cache's ref names. Taken when
//              exactly one complete artifact there matches the quantization asked for (any, a
//              label, or a width: "4bit"). The hub is asked (one listing request) whether the ref
//              still names that commit; where it moved, `moved_ref` decides (a download is many
//              gigabytes, so the converted weights stay unless the user asks otherwise), and a
//              hub that cannot be reached leaves them, said.
//
// Nothing is searched for on the machine: only the hub cache entry of the named repository.

#include "convert/convert.h"
#include "convert/drafters.h"
#include "convert/fetch.h"

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>

namespace ninferno::convert {

// What a start does with a kept artifact whose ref the hub now names at another commit (the
// branch moved after the model was converted and its download removed).
enum class MovedRef {
    Notice,  // keep the converted weights for this start, after a notice (on_moved_ref)
    Accept,  // keep them now and later: recorded beside them (accept_converted), no more notice
             // until the ref moves again
    Refetch, // download the commit the ref names now, convert it, then remove the old conversion
};

struct AcquireOptions {
    FetchOptions fetch;
    // The conversion: recipe ("" for the official one), out ("" for
    // <cache>/.../ninferno/<commit>/<stem>.<recipe>.ninferno), components, resources, sources, ...
    // model is filled in here.
    ConvertOptions convert;
    bool components_given = false;    // convert.components was named by the caller
    std::optional<bool> with_drafters; // default: a checkpoint's, not a GGUF model's
    std::string model_id;             // the registry's id of the model (default: the repository)
    std::map<std::string, std::filesystem::path> drafter_artifacts;
    std::filesystem::path repository; // source tree for shortlist rankings
    bool base_tokenizer_files = true; // a GGUF without tokenizer files takes its base model's
    bool fetch_only           = false;
    // A checkpoint that publishes a Vision encoder the converter keeps (Qwen3.5 / 3.6 / 3.8) is
    // converted with it, into its own artifact (<stem>.vision.<recipe>.ninferno): ninferno-serve
    // --vision. A GGUF model takes its projector (mmproj-*.gguf, BF16 first) from beside it or
    // from its repository. Other models are converted as without it (the Engine then says why).
    bool vision               = false;
    bool reuse                = false; // load an earlier run's artifact for the same commit
    MovedRef moved_ref        = MovedRef::Notice;
    // MovedRef::Refetch leaves the replaced conversion in place for the caller to remove once the
    // new one serves (ninferno-ui switches a running model over first): --keep-replaced.
    bool keep_replaced        = false;
    // Called with the notice of MovedRef::Notice once it is logged; a console counts down here
    // before the start goes on with the converted weights.
    std::function<void(const std::string& notice)> on_moved_ref;
    // Called before the weights are downloaded, with the recipe chosen from config.json ("" for a
    // GGUF or a URL, whose recipe is known only once fetched); throwing stops the acquisition.
    std::function<void(const FetchPlan&, const std::string& recipe)> before_download;
    std::function<void(const std::string&)> log;
};

struct AcquiredModel {
    FetchResult fetched;
    std::filesystem::path artifact; // empty with fetch_only
    std::string recipe;
    bool reused = false;     // an earlier run's artifact
    std::optional<ConvertResult> conversion;
    std::optional<DrafterPlan> drafters; // the attached drafters (unset: none asked for)
};

// The "kept" case above: the artifact under `cache` (the hub cache root) for `repo` at the commit
// its ref `revision` ("" for main) names, when that commit's snapshot is gone and exactly one
// complete artifact there matches `quant` (and `recipe` when given); nullopt otherwise.
[[nodiscard]] std::optional<std::filesystem::path>
kept_artifact(const std::filesystem::path& cache, const std::string& repo,
              const std::string& revision, const std::string& quant, const std::string& recipe);

// The one complete conversion in `commit_dir` (<cache>/models--org--name/ninferno/<commit>)
// that matches `quant` (any, a label, or a width: "4bit") and `recipe` (any when empty); nullopt
// when none or several do.
[[nodiscard]] std::optional<std::filesystem::path>
converted_artifact(const std::filesystem::path& commit_dir, const std::string& quant,
                   const std::string& recipe);

// Whether the conversion of `repo`'s GGUF quantization `quant` at the commit `revision` names (a
// commit, or a ref: "" for main) under `cache` carries drafters: a component from a repository of
// its own (a Qwen DFlash companion), or a speculation manifest of this artifact beside it with a
// drafter still usable (not failed, its artifact there), as an earlier --with-drafters wrote them.
// A start reuses that conversion, flag or not, and serves them: a plan of these weights counts them
// (product::PlannedTarget::with_drafters). The conversion is the one the start takes: of `recipe`
// (any when empty), a --vision one with `vision` (the text one when none is: a GGUF model without
// a projector converts without it); several left (two recipes, none named) answer what all of them
// carry. That conversion (the first by path when several recipes all carry them), for a plan to
// resolve as the start does; nullopt where none carries them.
[[nodiscard]] std::optional<std::filesystem::path>
cached_conversion_with_drafters(const std::filesystem::path& cache, const std::string& repo,
                                const std::string& revision, const std::string& quant,
                                const std::string& recipe = {}, bool vision = false);
// Whether cached_conversion_with_drafters finds one.
[[nodiscard]] bool cached_conversion_has_drafters(const std::filesystem::path& cache,
                                                  const std::string& repo,
                                                  const std::string& revision,
                                                  const std::string& quant,
                                                  const std::string& recipe = {},
                                                  bool vision                = false);
// A stamp of everything under `cache` that cached_conversion_with_drafters reads for `repo`: its
// refs, which commits' downloads are there, its conversions (names, sizes, times) and the separate
// drafters their manifests attach (another repository's, removed or replaced apart from them): a
// plan that depends on what a conversion carries is kept until the stamp changes.
[[nodiscard]] std::string conversions_stamp(const std::filesystem::path& cache,
                                            const std::string& repo);

// A conversion's file name in a commit's directory, <gguf stem>[.vision].<recipe>.ninferno:
// the GGUF's stem, whether it is a --vision conversion, its recipe, and the GGUF's quantization
// label (gguf_quant_label); nullopt for another name.
struct ConvertedName {
    std::string stem;
    bool vision = false;
    std::string recipe;
    std::string label;
};
[[nodiscard]] std::optional<ConvertedName> converted_name(const std::filesystem::path& artifact);

// The record that the user keeps the conversion in `commit_dir` (<cache>/models--org--name/
// ninferno/<commit>) for the ref `ref` while the hub names `hub_commit` (accepted.json there), and
// whether it is there: a start then serves it without a notice, until the ref moves again.
void accept_converted(const std::filesystem::path& commit_dir, const std::string& ref,
                      const std::string& hub_commit);
[[nodiscard]] bool converted_accepted(const std::filesystem::path& commit_dir,
                                      const std::string& ref, const std::string& hub_commit);

// `spec` is org/name[@revision] on the options' hub, or an http(s) URL of one file. Throws naming
// what failed and what to do about it.
[[nodiscard]] AcquiredModel acquire_model(const std::string& spec, AcquireOptions options);

// Of a repository's file names, the Vision projector a start takes under --vision (mmproj-*.gguf):
// BF16 first, then F32, then F16, then any; ties to the first name. Every part of it (one file, or
// a split projector's parts), first part first: the start fetches them and the plan without the
// weights charges them. Empty when there is none.
[[nodiscard]] std::vector<std::string> best_projector(const std::vector<std::string>& files);

// The Vision projector beside a GGUF model file (mmproj-*.gguf in its directory): BF16 first,
// then F32, then F16; nullopt when there is none.
[[nodiscard]] std::optional<std::filesystem::path>
gguf_projector_beside(const std::filesystem::path& gguf);

// Whether a recipe's model family converts a Vision encoder (Qwen3.5 / 3.6 / 3.8, Gemma 4 31B and
// 26B-A4B; a checkpoint without one, or a tower the converter refuses, says so when converted).
[[nodiscard]] bool recipe_has_vision(const std::string& recipe);

// The drafters of a local checkpoint (plan_drafters), with what they add to its conversion: the
// built-in MTP layers (unless the caller named the components) and the companion drafters,
// fetched, as components and sources of `convert_options`.
[[nodiscard]] DrafterPlan plan_checkpoint_drafters(const std::filesystem::path& checkpoint,
                                                   const std::string& repo, bool components_given,
                                                   ConvertOptions& convert_options,
                                                   const AttachOptions& attach);

// Whether `text` names a hub repository (org/name[@revision]) rather than a path or a URL.
[[nodiscard]] bool is_repository_spec(const std::string& text);

// Refuses a download (fetch_model's before_download) the hub cache's file system cannot hold.
void check_free_space(const FetchPlan& plan);

} // namespace ninferno::convert
