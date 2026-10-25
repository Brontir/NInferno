#pragma once

#include "ninferno/types.h"
#include "product/logging/logging.h"
#include "spec/profile.h"
#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens                    = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    bool help_requested    = false;
    bool version_requested = false;
    // The model as named on the command line: a .ninferno artifact, a Hugging Face checkpoint
    // directory or GGUF model (converted once into its .ninferno-cache), or a hub id
    // org/name[@revision] (fetched, converted and cached; see serve/model_source.h). The artifact
    // it resolves to is artifact_path.
    std::string model_source;
    std::string artifact_path;
    std::string recipe;        // --recipe: the conversion of a hub model ("" for the official)
    std::string quant;         // --quant: the GGUF quantization of a hub model
    bool prepare_only = false; // --prepare-only: fetch and convert, then exit
    // --plan: print the memory plan ladder and the rung the free memory picks, then exit without
    // serving (GenerationService::plan); --json prints it as JSON.
    bool plan_only = false;
    bool plan_json = false;
    // --no-load (with --plan): plan without loading or downloading the weights and without a GPU,
    // from the model's header or config (serve/plan_offline.h); --plan-device NAME plans for a
    // named board of the PCI-ID table instead of the CUDA device.
    bool plan_no_load = false;
    std::string plan_device;
    // --with-drafters: attach the drafters the registry names to a GGUF hub model too (fetched
    // and converted beside it); a checkpoint's are attached without it.
    bool with_drafters = false;
    // A hub model whose branch moved after it was converted and its download removed: false
    // keeps the converted weights after a notice (a terminal counts down first);
    // --accept-converted keeps them without one from now on; --refetch downloads the new commit,
    // converts it and removes the old conversion. Refused together.
    bool accept_converted = false;
    bool refetch          = false;
    // --keep-replaced (with --refetch): the old conversion stays for the caller to remove once the
    // new one serves (ninferno-ui switches a running model over first).
    bool keep_replaced = false;
    // --round-tiny-scales: the conversion keeps an integer matrix whose only inexact binary16
    // scales lie below binary16's normal range, with those scales rounded (ninferno-convert's
    // flag; the report counts them). Off: such a matrix is refused, naming the flag.
    bool round_tiny_scales = false;
    std::filesystem::path chat_template_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact metadata.name
    std::string request_log_jsonl;                // empty => structured request logging disabled
    // 0 (no --max-context): automatic, the context of the plan ladder's rung the free Device
    // memory picks (EngineOptions::max_context_automatic); the Engine reports the rung.
    std::uint32_t max_context          = 0;
    // --max-context-cap N: the context stays automatic (the free memory picks the rung, stepping
    // down on a busy card as without it) but at most N tokens. The quantization a GGUF hub
    // repository is fetched at without --quant sets it to the context each of several streams
    // was planned to keep (serve/plan_offline.h), as ninferno-ui's launch passes it. 0: none.
    std::uint32_t max_context_cap = 0;
    // Unset --kv-capacity: automatic with an automatic context, else max_context tokens.
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::automatic();
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    // Ten minutes: agents send several 20-40K-token prompts at once, and a prompt waits for the
    // prefills ahead of it (30 s turned the third of three such requests away on a 4060 Ti).
    std::uint32_t pending_timeout_ms   = 600000;
    std::uint32_t prefill_chunk        = 1024;
    // --prefill-chunk given: the Engine keeps it. Omitted: 1024 is the widest, and the Engine may
    // narrow it when the automatic context would stay below its floor
    // (EngineOptions::prefill_chunk_automatic).
    bool prefill_chunk_explicit        = false;
    std::uint32_t kv_lease_window      = 4096; // EngineOptions::kv_lease_window (--kv-lease-window)
    std::filesystem::path context_cost_presets;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    // SIGTERM/SIGINT drain: /health and new requests answer 503 while requests in flight finish,
    // for at most this long; then the server stops. A second signal stops at once.
    std::uint32_t drain_timeout_ms = 60000;
    // /health answers 503 "stalled" while the Engine has work and has made no progress for this
    // long (a hung device wait, a wedged executor); 0 disables the check.
    std::uint32_t health_stall_ms = 60000;
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    // --kv-dtype auto (or omitted): the model family chooses (EngineOptions::kv_cache_automatic).
    std::optional<KvCacheStorage> kv_cache;
    KvCacheStorage sliding_kv_cache        = KvCacheStorage::BFloat16; // --sliding-kv-dtype
    // --state-dtype (omitted: EngineOptions::recurrent_state_automatic).
    std::optional<RecurrentStateStorage> recurrent_state;
    // --activations exact|fast (ninferno/types.h): BF16 activations on every quantized linear, or
    // the width-dependent routes. The default is the product's one (kDefaultActivationMode).
    ActivationMode activations = kDefaultActivationMode;
    // Whether --activations and --sliding-kv-dtype were given: the configuration choice
    // (product/choice/choice.h) picks what the user left automatic.
    bool activations_given = false;
    bool sliding_kv_given  = false;
    float rope_yarn_factor     = 1.0F; // --rope-yarn-factor (EngineOptions::rope_yarn_factor)
    SpeculativeOptions speculative;
    // --spec auto (the default without an explicit --spec, --draft-tokens, --drafter or
    // --lm-head-draft): the speculation the artifact came with for this GPU (its profile, see
    // product/speculation/speculation.h), resolved when the service starts.
    bool speculation_from_profile = false;
    // The profile was asked for by default rather than by --spec auto: a model without one for
    // this GPU decodes plain without a warning.
    bool speculation_by_default = false;
    // The configuration choice's speculation mode (product::StartAxes::drafter) for --spec auto:
    // the profile chosen again for it (product::subject_of_mode). nullopt: the profile's own.
    std::optional<spec::Mode> speculation_mode;
    // An explicit --spec without --draft-tokens: the service takes the depth the model's profile
    // runs that backend with on this GPU, else product::default_draft_tokens.
    bool draft_tokens_from_model = false;
    // --spec-select fixed | thinking: with a profile, fixed runs the engine's default
    // configuration (the default thinking flag's) for every request; thinking runs each request at
    // the profile's default for its own thinking flag, at its own depth in one engine. Not given:
    // per request where the profile's measurements say it pays (product::per_request_pays).
    bool per_request_speculation = false;
    bool per_request_given       = false; // --spec-select given
    // --spec-load LANES:DEPTH,...|none: the speculation's bounds under decode load
    // (SpeculativeOptions::load) instead of the profile's table (spec::Profile::load); with an
    // explicit --spec, its only bounds. Absent: the profile's table (none with an explicit --spec).
    std::optional<std::vector<spec::LoadBound>> speculation_load;
    // --spec-profile PATH: a maintainer's own tune (`ninferno spec-tune`'s output) in place of the
    // published profile. Only read when named: a tune left beside the artifact is not.
    std::filesystem::path spec_profile;
    ContextCacheOptions context_cache;
    bool enable_vision      = false;
    bool use_cuda_graph     = true;
    bool allow_prefix_reuse = true;
    bool concurrent_prefill = false; // --concurrent-prefill
    // OpenAI requests: a shared-prefix candidate at the end of the leading system/developer run.
    bool auto_system_shared_prefix = true;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    std::optional<std::uint32_t> default_thinking_budget;
    // --repetition-stop N: end an output whose last N tokens repeat one short period (0: off).
    std::uint32_t repetition_stop = 0;
    // --repetition-thinking conclude: a loop --repetition-stop finds inside the thinking closes the
    // thinking (once) and the answer follows, instead of ending the output.
    bool repetition_concludes_thinking = false;
    int default_max_tokens = kDefaultMaxTokens;
    // --max-request-messages N: a request with more messages (Responses: input Items) is a 400.
    std::size_t max_request_messages = kDefaultMaxRequestMessages;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // --usage-chunk-choice: emit the streaming usage chunk with a zero-delta choice instead of the
    // OpenAI-conformant empty choices array. Strict client parsers (GitHub Copilot) reject the
    // empty array as "Response contained no choices"; the extra choice is inert for other clients.
    bool usage_chunk_choice = false;
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy                 = false; // --greedy: force temperature 0 (exact argmax)
    DecodePacingOptions pacing;          // --decode-steps, --decode-wait, --cuda-sync
    MaintainerOptions maintainer;        // --route-table ... --debug-poison-skip
    // --drafter-registry FILE: the drafter registry instead of the compiled-in one
    // (spec::Registry::use_file, applied before the model's speculation is resolved).
    std::filesystem::path drafter_registry;
    product::LogLevel log_level = product::LogLevel::Info;

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name);
std::string serve_usage_text(const char* argv0);

} // namespace ninferno::serve
