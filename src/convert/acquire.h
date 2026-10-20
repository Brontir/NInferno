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
//              refs/ and ninferno/ stay) is loaded at the commit the cache's ref names, without
//              asking the hub: fetching would download the weights again only to find the
//              artifact. Taken when exactly one complete artifact there matches the quantization
//              asked for (any, a label, or a width: "4bit").
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

// `spec` is org/name[@revision] on the options' hub, or an http(s) URL of one file. Throws naming
// what failed and what to do about it.
[[nodiscard]] AcquiredModel acquire_model(const std::string& spec, AcquireOptions options);

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
