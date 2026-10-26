#pragma once

// Model fetching: a repository on a hub (Hugging Face, ModelScope) or one file URL, downloaded
// into a Hugging Face-layout cache and handed to the converter.
//
//   hub cache  <root>/models--<org>--<name>/{blobs/<digest>, snapshots/<commit>/<path> -> blob,
//              refs/<revision>}, the layout huggingface_hub writes, so either tool reuses the
//              other's files (root: default_hub_cache() below);
//   files      only what a conversion reads: config, safetensors shards and index, the frontend
//              resources; or one GGUF model (a chosen quantization and its split parts);
//   transfer   HTTP range requests over N parallel streams per file (fixed, or adapted by
//              measuring throughput while ramping 1, 2, 4, ... streams), each stream aborted and
//              resumed when its speed stays under a floor (a stalled connection), the partial
//              file resumed from the bytes already written (a progress record per chunk, or the
//              prefix another downloader left), the Hugging Face 302's X-Linked-Etag (the
//              LFS sha256; the CDN's own ETag is not it) verified before an atomic rename;
//   auth       $HF_TOKEN (or $HUGGING_FACE_HUB_TOKEN) / $MODELSCOPE_API_TOKEN, sent to the hub
//              only, never to the CDN a download redirects to, never logged;
//   offline    under $HF_HUB_OFFLINE, the snapshot the revision resolved to before, as it lies in
//              the cache; a listing that fails falls back to it only for a commit (a branch or a
//              tag may have moved: the error says so).

#include "core/hub_revision.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace ninferno::convert {

// The files a fetch is about to download, reported before the first byte is transferred.
struct FetchPlan {
    std::string repo, commit;
    std::filesystem::path root;      // the hub cache root the files go to
    std::uint64_t total_bytes   = 0; // every chosen file
    std::uint64_t missing_bytes = 0; // the chosen files not yet in the snapshot
    std::vector<std::string> files;  // repository paths
};

// The hubs FetchOptions::hub names ("hf" and "ms" are their short names).
[[nodiscard]] inline bool is_hugging_face_hub(std::string_view hub) noexcept {
    return hub == "huggingface" || hub == "hf";
}

[[nodiscard]] inline bool is_modelscope_hub(std::string_view hub) noexcept {
    return hub == "modelscope" || hub == "ms";
}

struct FetchOptions {
    std::string hub = "huggingface"; // "huggingface" or "modelscope"
    std::string endpoint;            // default: $HF_ENDPOINT / $MODELSCOPE_ENDPOINT or the hub's
    std::string revision;            // default: "main" (Hugging Face), "master" (ModelScope)
    std::vector<std::string> include; // file globs; empty: what the converter needs
    std::string quant;                // GGUF: case-insensitive part of the file name, e.g. Q4_K_M
    std::filesystem::path cache;      // hub cache root (see above)
    unsigned streams      = 0;        // parallel ranges per file; 0: adapt
    unsigned max_streams  = 8;
    std::uint64_t chunk_bytes   = 32ULL << 20;
    std::uint64_t min_speed     = 64ULL << 10; // bytes per second a stream must keep ...
    unsigned stall_seconds      = 20;          // ... over this many seconds
    unsigned retries            = 10;          // per chunk
    bool verify                 = true;        // digest check before publishing a file
    std::function<void(const std::string&)> log;
    // Called once the files are chosen, before any download; throwing stops the fetch (a caller
    // that knows the files cannot be used, or cannot be stored, says so before the transfer).
    std::function<void(const FetchPlan&)> before_download;
};

struct FetchedFile {
    std::string path;      // repository path
    std::uint64_t bytes   = 0;
    double seconds        = 0.0; // transfer time (0 when cached)
    unsigned streams      = 0;   // streams the transfer settled on
    bool cached           = false;
    bool verified         = false;
};

struct FetchResult {
    std::filesystem::path snapshot; // directory holding the chosen files
    std::filesystem::path model;    // what to convert: the snapshot, or its GGUF file
    std::string repo, commit;
    std::vector<FetchedFile> files;
    // The hub was not asked (unreachable, or $HF_HUB_OFFLINE): the snapshot is what the cache
    // held for the revision, complete or not.
    bool offline = false;
};

// The hub cache root: $HF_HUB_CACHE, $HF_HOME/hub, $XDG_CACHE_HOME/huggingface/hub or
// ~/.cache/huggingface/hub.
[[nodiscard]] std::filesystem::path default_hub_cache();

// The Hugging Face token the hub is asked with, as huggingface_hub finds it: $HF_TOKEN (or
// $HUGGING_FACE_HUB_TOKEN), else the file `hf auth login` writes (hub_token_path()). `source`
// names where it came from ("" with no token).
struct HubToken {
    std::string value;
    std::string source;
};
[[nodiscard]] HubToken hub_token();
// $HF_TOKEN_PATH, else $HF_HOME/token, else ~/.cache/huggingface/token.
[[nodiscard]] std::filesystem::path hub_token_path();
// Writes the token where `hf auth login` keeps it (read by its owner only); "" removes it.
void save_hub_token(const std::string& token);
// The account a Hugging Face token belongs to (/api/whoami-v2); throws naming why the hub refused
// it (an invalid token: HTTP 401).
[[nodiscard]] std::string hub_account(const std::string& token, const FetchOptions& options);

// The commit `revision` ("" for main) names in a hub cache's repository directory `repo_dir`
// (<cache>/models--org--name): a commit id itself, else the commit its ref file (refs/<revision>)
// records; nullopt when none is recorded or what is recorded is no commit id.
[[nodiscard]] std::optional<std::string> local_commit(const std::filesystem::path& repo_dir,
                                                      const std::string& revision);

// org/name[@revision] -> (org/name, revision); without @, the options' revision ("": the hub's
// default branch). Throws for anything that is not one org and one name.
[[nodiscard]] std::pair<std::string, std::string> split_spec(const std::string& spec,
                                                             const FetchOptions& options);

// `spec` is "org/name" or "org/name@revision" on the options' hub, or an http(s) URL of one file
// (cached under <root>/urls/). Throws naming what failed; a failed download leaves its partial
// file to resume.
[[nodiscard]] FetchResult fetch_model(const std::string& spec, const FetchOptions& options);

// Whether the cached `snapshot` holds the whole model fetch_model() would choose for `options`
// (options.quant's GGUF model, else the weights; every shard a safetensors index names): what a
// start may serve from it without the hub. With `projector` (--vision), a GGUF model's Vision
// projector (mmproj-*.gguf, every part) too, and a checkpoint's Vision encoder (vision_config).
[[nodiscard]] bool snapshot_holds_model(const std::filesystem::path& snapshot,
                                        const FetchOptions& options, const std::string& repo,
                                        bool projector = false);

// The quantization label of the GGUF model fetch_model() would choose from the cached `snapshot`
// for `options` ("" for safetensors weights or none): which of several conversions of a commit is
// that model's.
[[nodiscard]] std::string snapshot_quant_label(const std::filesystem::path& snapshot,
                                               const FetchOptions& options, const std::string& repo);

// The first bytes of the GGUF model fetch_model() would choose for `spec` (options.revision,
// options.quant), to read its metadata before the whole file is downloaded: `bytes` of it by an
// HTTP range request (all of it when smaller), or the cached file's. A revision pinned to a commit
// must resolve to it. Offline, the cached snapshot's file for options.quant, as fetch_model().
struct GgufHead {
    std::string repo, commit;
    std::string path;       // the file's repository path
    std::uint64_t size = 0; // the whole file's bytes
    std::string bytes;      // its first bytes
    bool cached  = false;   // read from the hub cache
    bool offline = false;   // the hub was not asked
    // The model's parts (a split GGUF's -0000N-of-0000M files, in their order): this is `part`
    // of them. Offline, the cached first part only (1).
    std::size_t parts = 1;
};

[[nodiscard]] GgufHead fetch_gguf_head(const std::string& spec, const FetchOptions& options,
                                       std::uint64_t bytes, std::size_t part = 0);

// The Hugging Face repository a GGUF file names as its base model
// (gguf_metadata_base_repository of its metadata), if any.
[[nodiscard]] std::optional<std::string> gguf_base_repository(const std::filesystem::path& gguf);

// Whether the GGUF model a fetch takes for `options.quant` carries MTP (NextN) layers, by its
// metadata (gguf_metadata_carries_mtp of the first MiB of its file, then of 8 MiB where that ended
// first): a repository's quantizations differ (unsloth drops them from its smallest). nullopt when
// it cannot be read.
[[nodiscard]] std::optional<bool> gguf_carries_mtp(const std::string& spec,
                                                   const FetchOptions& options);
// The same for each of `quants` (FetchOptions::quant each), read together (a few at a time): a
// repository's table plans every one.
[[nodiscard]] std::map<std::string, std::optional<bool>>
gguf_quants_carry_mtp(const std::string& spec, const FetchOptions& options,
                      const std::vector<std::string>& quants);

// One file: `url` (resolved through redirects) into `target`, `size` bytes (0: unknown),
// checked against `sha256` (lowercase hex) or `git_sha1` when given. Exposed for tests.
struct DownloadRequest {
    std::string label; // name for logs and errors (default: the target's file name)
    std::string url;
    std::filesystem::path target;
    std::uint64_t size = 0;
    std::string sha256;
    std::string git_sha1;
    std::vector<std::string> headers; // e.g. Authorization, sent to `url` itself only
    // Re-resolves an expired signed URL (403/410 from the CDN); empty: retry the same URL.
    std::function<std::string()> refresh;
};

[[nodiscard]] FetchedFile download_file(const DownloadRequest& request, const FetchOptions& options);

// The SHA-256 (lowercase hex) of a file's bytes, read in 16 MiB blocks. Throws when it cannot
// be read.
[[nodiscard]] std::string file_sha256(const std::filesystem::path& path);

// The quantization label a GGUF file name ends with, as --quant takes it:
// "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -> "UD-Q4_K_M", "m-Q6_K-00001-of-00002.gguf" -> "Q6_K";
// "" when the name ends with none.
[[nodiscard]] std::string gguf_quant_label(const std::string& file_name);

// Whether a fetch takes the repository file `path` by its --include globs (none: what the
// converter needs, so true).
[[nodiscard]] bool include_selects(const FetchOptions& options, const std::string& path);

// One model a repository offers, as a listing shows it: a GGUF quantization (its split parts
// together), a safetensors checkpoint (quant ""), an artifact in the engine's own format (.ninferno,
// run as published), or a sidecar (MTP/DFlash/EAGLE drafter, vision
// projector) that is not a model by itself.
struct RepoQuant {
    std::string quant;  // as --quant takes it: "UD-Q4_K_M", "MXFP4"; "" for safetensors weights
    std::string format; // gguf | safetensors | ninferno
    std::uint64_t bytes = 0;
    int bits            = 0; // from the label (Q4_K_M -> 4, BF16 -> 16); 0 when it says none
    bool drafter        = false;
    std::vector<std::string> files;
    std::vector<std::string> sha256; // each file's LFS digest, as files lists them ("": in git)
    std::vector<std::uint64_t> sizes; // each GGUF file's bytes, as files lists them
    // Of `bytes`, the safetensors weights' own part for the MTP layers (model_mtp.safetensors):
    // fetched with the model, loaded only under MTP (PlanModelFacts::mtp_bytes).
    std::uint64_t mtp_bytes = 0;
    std::string commit; // the commit the listing was read at ("" unknown)
};

// The commit `spec`'s revision names on the hub now (one listing request); nullopt under
// $HF_HUB_OFFLINE. Throws when the hub cannot be reached.
[[nodiscard]] std::optional<std::string> current_commit(const std::string& spec,
                                                        const FetchOptions& options);

// The GGUF model `quant` names among `groups` (each model's split stem and its bytes), as the
// fetch picks it: a width ("4bit") the largest model of that width; else the models whose stem
// holds the label, "Q4_K_M" matching "UD-Q4_K_M", an exact label when several do. Throws naming
// the models when none or several match.
[[nodiscard]] std::string gguf_group_for_quant(const std::map<std::string, std::uint64_t>& groups,
                                               const std::string& quant, const std::string& repo);
// The listing's GGUF quantization the fetch takes for --quant `quant` (gguf_group_for_quant over
// list_quants' models): the plan without weights plans the files the start downloads. A split
// model the listing gives a part at a time (no quantization label in its names) is one entry of
// all its parts, named by its group.
[[nodiscard]] RepoQuant listed_quant_for(const std::vector<RepoQuant>& quants,
                                         const std::string& quant, const std::string& repo);

// The models of `spec` (org/name[@revision]) with their sizes, from one listing request: nothing
// is downloaded. GGUF quantizations in the listing's order, then the safetensors weights.
[[nodiscard]] std::vector<RepoQuant> list_quants(const std::string& spec,
                                                 const FetchOptions& options);

// A model the hub's search found: its id and what the hub says of it.
struct HubModel {
    std::string id; // org/name
    std::uint64_t downloads = 0;
    std::uint64_t likes     = 0;
    std::string updated;  // ISO time of the last commit
    std::string pipeline; // text-generation, image-text-to-text, ...
    bool gguf = false;    // tagged gguf
    // What the hub read of the weights: a GGUF repository's general.architecture ("qwen35moe")
    // and context length, else config.json's architectures[0] ("Gemma4ForConditionalGeneration")
    // or model_type; "" and 0 when it says none.
    std::string architecture;
    std::int64_t context = 0;
    std::string library; // the weights' library: transformers, gguf, mlx, ... ("" unsaid)
    // The search's answer for the model as the hub gave it (Hugging Face; null on ModelScope): the
    // facts convert::search_support_facts reads.
    nlohmann::json card;
};

// The hub's model search, at most `limit` (1 to 100), largest first by `sort`: downloads, likes,
// lastModified, createdAt or trendingScore. Hugging Face (/api/models?search=) orders it itself
// and an empty query lists the GGUF repositories; ModelScope (openapi/v1/models) answers in its
// relevance order (its own orders mix in unrelated models), sorted here, its trending as
// downloads and the architecture its model_type tag names. Throws naming what failed (an unknown
// order too).
[[nodiscard]] std::vector<HubModel> search_models(const std::string& query,
                                                  const FetchOptions& options, int limit,
                                                  const std::string& sort = "downloads");

// The body of an http(s) `url` (redirects followed; HF_TOKEN goes to the Hugging Face
// host only, for a private repository), at most `max_bytes`.
// Throws naming what failed.
[[nodiscard]] std::string fetch_text(const std::string& url, std::size_t max_bytes);

// The repository's config.json (org/name[@revision]), parsed; one small request. Throws naming
// what failed (absent, not JSON); ModelScope is not read.
[[nodiscard]] nlohmann::json fetch_config(const std::string& spec, const FetchOptions& options);

// What Hugging Face says of a repository before any file is downloaded: its model card fields
// (/api/models/<repo>/revision/<rev>): pipeline_tag, library_name, gated, the part of config.json
// the hub reads (architectures, model_type, quantization_config, tokenizer_config), gguf (the
// architecture and chat template of a GGUF model), safetensors (parameters by dtype) and
// siblings. nullopt on another hub or under $HF_HUB_OFFLINE. Throws when the hub cannot answer.
[[nodiscard]] std::optional<nlohmann::json> fetch_model_card(const std::string& spec,
                                                             const FetchOptions& options);

// The base models a repository's model card names, in its order (cardData.base_model: its
// README's YAML front matter, what the converter's drafter plan reads from the downloaded README;
// a merge names several), none when it names none, offline, or the hub does not answer. The plan of
// a checkpoint not downloaded yet matches the registry by them as the converter will (a fine-tune
// finds its base's entry).
[[nodiscard]] std::vector<std::string> hub_base_models(const std::string& spec,
                                                       const FetchOptions& options);

// The quantization a config.json declares when the converter has no reader for it (formats.cpp:
// quant_config and the matrix sources it picks): "bitsandbytes", "hqq", compressed-tensors with
// transforms or sparse packings; "" when it reads it (unquantized, FP8, AWQ, GPTQ,
// compressed-tensors, ModelOpt, AutoRound, MXFP4).
[[nodiscard]] std::string unread_quantization(const nlohmann::json& config);

// Another small JSON file of the repository the same way (a native repository's
// artifact-manifest.json).
[[nodiscard]] nlohmann::json fetch_repo_json(const std::string& spec, const FetchOptions& options,
                                             const std::string& file);

// A width tag in place of a quantization name: "4bit", "4-bit", "8bits" -> 4, 4, 8 (1 to 16); 0
// when `quant` is a name. --quant 4bit takes the largest GGUF model of that width.
[[nodiscard]] int quant_width_tag(const std::string& quant);

} // namespace ninferno::convert
