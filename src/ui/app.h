#pragma once

// ninferno-ui's HTTP side: the page, the JSON API it calls, and the chat proxy to a running model.
//
//   GET  /                 the page (compiled in; no external scripts, fonts or images)
//   GET  /api/state        GPUs (NVML), running processes, UI settings: polled by the page
//   GET  /api/catalog?gpu  recommended models with fit, speed and context for that GPU
//   GET  /api/quants?repo&gpu   a hub repository's quantizations with sizes and fit (one listing)
//   GET  /api/plan?repo&gpu&quant&ctx&par&kv&spec  the plan without the weights of one or every
//                          quantization and the largest that fits (ninferno-dry-plan/1)
//   GET  /api/local        models already converted (hub cache, --models-dir): architecture,
//                          size, quantization, what they take, the original kept beside them
//   GET  /api/search?q     the catalog's models and Hugging Face's search for q
//   POST /api/delete       {artifact}: a converted model (and its original when nothing else
//                          of that repository is left)
//   POST /api/delete_original {repo}: a converted hub model's downloaded original
//   GET  /api/log?id&n     the last lines of a process's output
//   POST /api/launch       {model, quant?, gpu, max_context?, spec?, precision?, spec_profile?}
//                          -> ninferno-serve (spec_profile: the tune beside the model, by request)
//   POST /api/tune         {artifact, gpu}: spec-tune on that GPU writes a tune beside the model
//   POST /api/prepare      {model, quant?} -> download and convert only (--prepare-only)
//   POST /api/stop         {id}
//   POST /api/update       {action: install | restart}: install the newest release (the updater
//                          runs as a process); restart into it (the models stop)
//   /chat/<id>/...         the chat window (llama-ui's bundle) of a running model; its ./props is
//                          answered here, its ./v1/chat/completions forwarded to the model with
//                          the served model id filled in (streams; chat_ui.h)
//
// Every request passes access.h's Host check; every POST its Origin check and must be JSON (a
// form post from another site cannot be, without a CORS preflight that is never granted).

#include "ui/access.h"
#include "ui/catalog.h"
#include "ui/gpu_stats.h"
#include "ui/processes.h"
#include "product/plan/plan_quants.h"
#include "product/update/update_check.h"

#include <chrono>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace httplib {
class Server;
}

namespace ninferno::ui {

// The first model id of an OpenAI /v1/models body ("" when it names none).
[[nodiscard]] std::string first_model_id(const std::string& models_body);
// A chat request with "model" set to `served` when it has none (the page sends none; ninferno-serve
// requires the id it serves). Throws std::invalid_argument when the body is not a JSON object.
[[nodiscard]] std::string with_model(const std::string& chat_body, const std::string& served);

struct UiOptions {
    AccessPolicy access;
    std::filesystem::path serve_binary;            // ninferno-serve
    std::filesystem::path hub_cache;               // empty: convert::default_hub_cache()
    std::vector<std::filesystem::path> model_dirs; // more places holding .ninferno artifacts
    bool offline = false;                          // no hub listings (tests, air-gapped machines)
    // llama-ui's bundle for the chat window (chat_ui.h); empty: the page has no chat window.
    std::filesystem::path chat_ui;
    // Keep a hub model's downloaded original after its conversion (default: removed once the
    // converted model loaded, as nothing reads it again; ninferno-ui starts the converted file).
    bool keep_originals = false;
    // The catalog's feed: an http(s) URL of a catalog.json (src/ui/data/catalog.json's schema),
    // read at start and every 6 hours and kept in <hub cache>/ninferno-ui/catalog.json, which serves
    // until a read succeeds; empty: the built-in catalog only.
    std::string catalog_url;
    // This build's version: a release version is checked against the release manifest
    // (product/update/update_check.h) and a newer release offered on the page.
    std::string version;
    // The command that installs the newest release (ninferno-update beside ninferno-ui; on
    // Windows NInferno-Setup.exe --update); "--yes" is appended. Empty: the page only says that
    // a release is out.
    std::vector<std::string> updater;
};

// The launch command for a model: what /api/launch runs. Exposed for the tests.
struct LaunchRequest {
    std::string model; // hub id, hub id:QUANT, or a local path
    std::string quant; // overrides a :QUANT suffix
    int gpu          = 0;
    int port         = 0;
    long max_context = 0;  // 0: automatic
    long max_context_cap = 0;  // with an automatic context: its ceiling (--max-context-cap)
    int max_concurrency = 0;  // requests at once; 0: ninferno-serve's default
    std::string spec;      // "" or auto | none | mtp | dflash | dflash2
    std::string precision; // "" or exact | balanced | fast
    std::string kv_dtype;  // "" or bf16 | int8 | fp8 | nvfp4 | k8v4 (serve's --kv-dtype)
    std::string hub;       // "" (Hugging Face) or modelscope: where a hub id is fetched
    // Name the tune beside the artifact (<artifact>.spec_profile.json, /api/tune's) for this start
    // (--spec-profile). Never by itself: a measurement of this machine runs only when asked for,
    // and asked for, one that would not serve on the request's GPU refuses the start.
    bool spec_profile = false;
    bool prepare_only = false;
    // A prepare of a hub model whose branch moved after its conversion: download the new commit,
    // convert it and remove the old conversion (ninferno-serve --refetch).
    bool refetch = false;
    // The conversion recipe (ninferno-serve --recipe; "": the official one): a refetch keeps the
    // replaced artifact's.
    std::string recipe;
};

// Throws std::invalid_argument on a value outside the allowed set, and when spec_profile is set but
// the model is not started by its .ninferno path or the tune beside it does not serve on `gpu`
// (tune_refusal_on). `gpu`: what NVML says of the GPU the request runs on (nullptr: unknown).
[[nodiscard]] LaunchSpec make_launch(const UiOptions& options, const LaunchRequest& request,
                                     const GpuStat* gpu = nullptr);

// Why the tune at `tune` (of the artifact `artifact_id`) would not serve on `gpu`; empty: nothing
// here refuses it. The GPU class the server keys a tune by ("gb202:170sm", spec::gpu_identity) from
// the PCI id, the compute capability and the SM count. On an unknown GPU, or one without an SM
// count (boards of one chip differ in it; a MIG slice reports none), only the tune itself is
// checked: ninferno-serve names its GPU's class from CUDA and refuses to start with a named tune
// of another.
[[nodiscard]] std::string tune_refusal_on(const std::filesystem::path& tune,
                                          const std::string& artifact_id, const GpuStat* gpu);

// The GPU a launch on `gpu` runs on (CUDA_VISIBLE_DEVICES in PCI order: NVML's index); none named
// (gpu < 0): the only GPU there is, else none (nullptr: unknown).
[[nodiscard]] const GpuStat* launch_gpu(const std::vector<GpuStat>& stats, int gpu);

// Speculation calibration of a converted model on one GPU (spec-tune --quick): the Qwen/dense
// engines through `ninferno spec-tune`, Gemma 4 through `ninferno-gemma4 ... spec-tune`.
struct TuneRequest {
    std::string artifact;
    std::string engine; // convert::describe_artifact's: gemma4 | qwen3_5 | ""
    int gpu = 0;
};

// Throws std::invalid_argument when `artifact` is not an existing .ninferno file.
[[nodiscard]] LaunchSpec make_tune(const UiOptions& options, const TuneRequest& request);

// "org/name:Q4_K_M" -> ("org/name", "Q4_K_M"); a path or a name without a tag is returned whole.
[[nodiscard]] std::pair<std::string, std::string> split_model_tag(const std::string& model);

// What an artifact's header says of it (cached per path and modification time).
struct ArtifactFacts {
    bool readable = false;
    bool drafter  = false; // an MTP sidecar, a DFlash block, an assistant: attached, not run alone
    bool vision   = false; // a vision component: takes images
    bool mtp      = false; // carries its MTP layers
    std::string architecture;
};

[[nodiscard]] ArtifactFacts read_artifact_facts(const std::filesystem::path& artifact);

// A converted file's name parts: "Qwen3.6-35B-A3B-UD-Q4_K_M.qwen3_5_gguf.ninferno" ->
// stem "Qwen3.6-35B-A3B-UD-Q4_K_M", recipe "qwen3_5_gguf"; quant "UD-Q4_K_M" (the GGUF label,
// else NVFP4/MXFP4/FP8/Q8/BF16 named in it); params "35B-A3B" from `name`.
struct ArtifactName {
    std::string stem, recipe, quant;
};

[[nodiscard]] ArtifactName artifact_name(const std::string& file_name);
[[nodiscard]] std::string params_label(const std::string& name);
// "Qwen3_5ForCausalLM" -> "qwen3_5"; "Gemma4ForConditionalGeneration" -> "gemma4".
[[nodiscard]] std::string short_architecture(const std::string& architecture);

// Whether `path` may be deleted through the UI: a .ninferno file under the hub cache's
// models--*/ninferno/ or under one of `model_dirs`.
[[nodiscard]] bool deletable_artifact(const std::filesystem::path& path,
                                      const std::filesystem::path& hub_root,
                                      const std::vector<std::filesystem::path>& model_dirs);

class App {
public:
    App(UiOptions options, std::unique_ptr<GpuSource> gpus);
    ~App();

    // Binds and serves until stop(); false when the address cannot be bound.
    bool run();
    void stop();
    // The routes, for tests that drive the server on an ephemeral port.
    void install(httplib::Server& server);

    // Hub listing cache: fills sizes for the catalog's sources in the background.
    void prefetch_listings();
    [[nodiscard]] SizeTable sizes() const;
    // Reads the catalog's feed now (UiOptions::catalog_url); true when it was read and is in use.
    bool refresh_catalog();
    // The catalog in use: the feed's, its cached copy, or the built-in one.
    [[nodiscard]] std::shared_ptr<const Catalog> catalog() const;
    // Set by a restart request (/api/update {action: restart}): run() returned so that the
    // installed version starts in place of this one.
    [[nodiscard]] bool restart_requested() const { return restart_requested_; }

private:
    // The release on offer and the update under way, for /api/state.
    nlohmann::json update_json();
    // Starts the updater (once at a time); its process id.
    std::uint64_t install_update();
    void restart();
    nlohmann::json state_json();
    nlohmann::json catalog_json(int gpu, const product::LaunchAsk& launch = {});
    nlohmann::json quants_json(const std::string& repo, int gpu,
                               const std::string& hub           = "huggingface",
                               const product::LaunchAsk& launch = {});
    // The plan without the weights of a repository's quantization (all of them when `quant` is
    // empty, with the largest that fits) on GPU `gpu` for what a launch asks: the
    // "ninferno-dry-plan/1" document (product/plan/plan_quants.h), as ninferno-serve <repo> --plan
    // --no-load --json would print it for an idle card of that GPU. Throws when the repository's
    // shape cannot be read.
    nlohmann::json plan_json(const std::string& repo, const std::string& hub, int gpu,
                             const std::string& quant, const product::LaunchAsk& launch);
    // The facts of a repository's model (kv_json reads them from the same header); nullopt
    // offline or unreadable.
    std::optional<product::PlanSourceFacts>
    plan_facts(const std::string& repo, const nlohmann::json& listing, const std::string& hub);
    // The architecture the repository's main weights name, read from them (the head of its
    // smallest GGUF model, else config.json), with how far the engine runs it.
    nlohmann::json arch_json(const std::string& repo, const std::string& hub);
    nlohmann::json listing_of(const std::string& repo, const std::string& hub);
    nlohmann::json local_json();
    // The converted hub models whose download was removed and whose branch the hub now names at
    // another commit: what each start of them would warn about (ninferno-serve keeps the
    // converted weights unless asked). One listing request per repository and branch, at most
    // hourly; none with --offline.
    nlohmann::json moved_json();
    // The user's answer for one such conversion: "accept" keeps it as valid now and later
    // (convert::accept_converted), "refetch" starts a prepare per converted quantization that
    // downloads the new commit and removes the old conversion. The prepares' ids.
    nlohmann::json answer_moved(const std::string& repo, const std::string& ref,
                                const std::string& commit, const std::string& action);
    nlohmann::json search_json(const std::string& query, const std::string& sort = "downloads",
                               const std::string& hub = "huggingface");
    nlohmann::json kv_json(const std::string& repo, const nlohmann::json& listing,
                           const std::string& hub = "huggingface");
    // What --spec auto runs for `repo` on GPU `gpu` (the registry's profile for its class), with
    // the drafter's weights and KV the memory estimate adds.
    nlohmann::json speculation_json(const std::string& repo, int gpu, const nlohmann::json& kv,
                                    const nlohmann::json& listing);
    // Bytes freed; throws std::invalid_argument for what may not be deleted.
    std::uint64_t delete_artifact(const std::filesystem::path& artifact);
    std::uint64_t delete_original(const std::string& repo);
    [[nodiscard]] std::filesystem::path hub_root() const;
    [[nodiscard]] ArtifactFacts facts(const std::filesystem::path& artifact);
    // A converted model of `repo` (and `quant`, when named) already here, or empty.
    [[nodiscard]] std::filesystem::path local_artifact(const std::string& repo,
                                                       const std::string& quant);
    void remove_converted_originals();
    void watch_health();
    void install_chat(httplib::Server& server);

    UiOptions options_;
    std::unique_ptr<GpuSource> gpus_;
    ProcessPool processes_;
    std::unique_ptr<httplib::Server> server_;
    mutable std::mutex sizes_mutex_;
    // Which Hugging Face token the caches below hold answers for (under sizes_mutex_): a new one
    // counts up, and a request started before it keeps nothing.
    std::uint64_t hub_generation_ = 0;
    [[nodiscard]] std::uint64_t hub_generation() const {
        std::lock_guard lock(sizes_mutex_);
        return hub_generation_;
    }
    SizeTable sizes_;
    std::map<std::string, nlohmann::json> listings_; // repo -> quant list
    // Artifact path -> (modification time, what its header says).
    std::mutex kinds_mutex_;
    std::map<std::string, std::pair<std::filesystem::file_time_type, ArtifactFacts>> kinds_;
    std::map<std::string, nlohmann::json> kv_; // repo -> KvModel json
    // hub_key -> the plan facts kv_json read with it (product/plan/plan_facts.h)
    std::map<std::string, product::PlanSourceFacts> plan_facts_;
    // plan_json's documents by their inputs (the listing and the GPU's facts do not change;
    // bounded: plan_json clears it beyond kMaxKeptPlans)
    std::map<std::string, nlohmann::json> plans_;
    // hub_key -> the config.json kv_json read (a checkpoint's converted size reads it)
    std::map<std::string, nlohmann::json> configs_;
    // hub_key -> when a header or config read last failed (tried again after a minute)
    std::map<std::string, std::chrono::steady_clock::time_point> kv_failed_;
    std::map<std::string, nlohmann::json> archs_;    // arch_json, by hub_key
    std::map<std::string, std::string> base_models_; // repo -> its card's base_model ("-": none)
    std::map<std::string, std::pair<std::int64_t, nlohmann::json>> searches_; // q -> (time, hits)
    std::set<std::uint64_t> originals_done_; // processes already handled
    std::mutex moved_mutex_;
    // repo@ref -> (when asked, the commit the hub names; "" when it could not be asked)
    std::map<std::string, std::pair<std::int64_t, std::string>> hub_commits_;
    mutable std::mutex catalog_mutex_;
    std::shared_ptr<const Catalog> catalog_;
    std::string catalog_source_; // "built in", or the feed and when it was read
    std::atomic<bool> stopping_{false};
    std::atomic<bool> restart_requested_{false};
    std::mutex update_mutex_;
    std::optional<update::Release> release_; // the manifest's answer, once read
    std::uint64_t update_job_ = 0;            // the updater's process (0: none started)
    std::thread health_thread_, prefetch_thread_, catalog_thread_, update_thread_;
};

} // namespace ninferno::ui
