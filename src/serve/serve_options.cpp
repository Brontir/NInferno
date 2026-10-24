#include "serve/serve_options.h"
#include "core/command_line.h"
#include "product/activation_options.h"
#include "product/engine_flags.h"
#include "product/numeric_options.h"
#include "product/sampling_options.h"
#include "product/speculative_options.h"
#include "serve/request_validation.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninferno::serve {
namespace {

using product::parse_float_in;
using product::parse_kv_capacity;
using product::parse_kv_dtype;
using product::parse_nonnegative_int;
using product::parse_u64;

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 + " --version\n       " + argv0 +
           " <model.ninferno | checkpoint-dir | model.gguf | org/name[@revision]> "
           "[--recipe NAME] [--quant LABEL] [--prepare-only] [--plan [--json] [--no-load "
           "[--plan-device NAME]]] [--with-drafters] [--accept-converted | --refetch] "
           "[--host H] [--port N] [--api-key KEY | --api-key-file PATH] "
           "[--model-id ID] [--max-context N] [--max-context-cap N] [--kv-capacity N|auto] "
           "[--kv-auto-headroom-mib N] "
           "[--max-concurrency N] "
           "[--max-pending-requests N] [--pending-timeout-ms N] "
           "[--prefill-chunk N] [--kv-lease-window N] [--log-stats-interval-ms N] "
           "[--drain-timeout-ms N] "
           "[--health-stall-ms N] [--device N] "
           "[--context-cost-presets FILE] "
           "[--max-request-mib N] [--max-request-messages N] "
           "[--media-cache-mib N] [--media-live-mib N] "
           "[--media-preprocess-threads N] "
           "[--device-state-slots N] [--host-state-slots N] [--host-kv-mib N] "
           "[--max-private-continuations N] [--max-shared-prefixes N] "
           "[--max-long-anchors-per-continuation N] "
           "[--request-log-jsonl FILE] "
           "[--response-store-max-records N] [--response-store-max-mib N] "
           "[--kv-dtype auto|bf16|int8|fp8|nvfp4|k8v4] [--sliding-kv-dtype bf16|fp8] "
           "[--state-dtype f32|bf16|f16] "
           "[--activations exact|fast|int8] [--rope-yarn-factor F] "
           "[--spec none|mtp|dflash|dflash2 --draft-tokens N] "
           "[--ngram-draft-tokens N [--ngram-min-match M]] "
           "[--drafter DRAFTER.ninferno] [--draft-width block|drafts] "
           "[--spec auto [--spec-select fixed|thinking]] [--spec-load LANES:DEPTH,...|none] "
           "[--spec-profile TUNE.json] "
           "[--spec-shared-floor F] "
           "[--default-max-tokens N] [--default-thinking-budget N] [--repetition-stop N] "
           "[--repetition-thinking stop|conclude] "
           "[--vision] [--no-cuda-graph] [--no-prefix-reuse] [--concurrent-prefill] "
           "[--chat-template FILE] [--lm-head-draft] [--no-thinking] [--preserve-thinking] "
           "[--no-auto-system-shared-prefix] [--cors] [--usage-chunk-choice] "
           "[--temperature F] [--top-p F] [--top-k N] [--min-p F] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n" +
           std::string(product::kEngineFlagsUsage) +
           "       [--drafter-registry FILE]\n"
           "       serves OpenAI Responses/Chat Completions and Anthropic Messages endpoints\n"
           "       the model: a .ninferno artifact; a Hugging Face checkpoint directory or GGUF\n"
           "       model (converted once into <dir>/.ninferno-cache/); or a hub id "
           "org/name[@rev],\n"
           "       fetched into the Hugging Face cache ($HF_HUB_CACHE, $HF_HOME/hub,\n"
           "       $XDG_CACHE_HOME/huggingface/hub, ~/.cache/huggingface/hub), converted\n"
           "       with the official recipe (dense and MoE decoders as published; --recipe names\n"
           "       another) into <cache>/models--org--name/ninferno/<commit>/ with the drafters "
           "the\n"
           "       model or the official registry declares, and reused by later starts; nothing\n"
           "       else on the machine is searched. $HF_TOKEN authorizes gated models,\n"
           "       $HF_HUB_OFFLINE=1 uses the cache only. The served model id is the hub id.\n"
           "       --prepare-only fetches and converts, then exits (no GPU).\n"
           "       --plan prints the memory plan ladder these options give (every rung's bytes),\n"
           "       the rung the free GPU memory picks now and what would get more, then exits\n"
           "       without serving (the weights are loaded to size the rungs, then released);\n"
           "       --json prints it as JSON. --no-load plans without loading or downloading\n"
           "       the weights and without a GPU, from the model's header or config (an\n"
           "       estimate: the start is the authority); for a GGUF hub repository without\n"
           "       --quant it lists every quantization and the largest that fits.\n"
           "       --plan-device NAME plans for a named board instead (\"RTX 4090\", 0x2684,\n"
           "       then ,headless or ,display and ,linux or ,wddm).\n"
           "       Without --quant a GGUF hub repository is fetched at the largest\n"
           "       quantization that fits --max-context x --max-concurrency on this GPU (an\n"
           "       idle card of its class), and refused before the download when none does.\n"
           "       --with-drafters attaches a GGUF hub model's drafters too (a separate\n"
           "       download); a checkpoint's are attached without it.\n"
           "       A hub model whose branch moved after it was converted and its download\n"
           "       removed is served from its converted weights, after a notice (and a 10 s\n"
           "       count-down on a terminal): --accept-converted keeps them without the notice\n"
           "       from now on, --refetch downloads the new commit, converts it and removes\n"
           "       the old conversion.\n"
           "       --default-max-tokens defaults to " +
           std::to_string(kDefaultMaxTokens) +
           " when omitted\n"
           "       --max-request-mib defaults to 384 and is enforced before JSON parsing\n"
           "       --max-request-messages defaults to " +
           std::to_string(kDefaultMaxRequestMessages) +
           "; a request with more messages (Responses: input Items) is a 400\n"
           "       --media-cache-mib defaults to 1024; 0 disables retained media reuse\n"
           "       --media-live-mib defaults to 2048 and bounds all live BF16 patch payloads\n"
           "       --media-preprocess-threads defaults to 0 (auto, at most 16 workers)\n"
           "       --request-log-jsonl appends full-precision server/request records\n"
           "       --model-id overrides the name the server reports (the artifact metadata.name, "
           "else\n"
           "         its file name; a Hugging Face id when serving one)\n"
           "       Responses state is process-local and bounded to 1024 records / 256 MiB by "
           "default\n"
           "       --log-stats-interval-ms defaults to 5000; 0 disables periodic throughput logs\n"
           "       SIGTERM/SIGINT drain: /health and new requests get 503 while requests in "
           "flight finish, at most --drain-timeout-ms (default 60000); a second signal stops "
           "at once\n"
           "       --health-stall-ms (default 60000; 0 off): /health is 503 while the engine has "
           "work and makes no progress that long\n"
           "       --vision enables media and loads the fixed Vision GPU allocations\n"
           "       --max-context-cap N keeps the context automatic (the free memory picks the "
           "rung, a busy card steps down) but at most N tokens; not with --max-context. A GGUF "
           "repository fetched without --quant for several slots sets it to the context each "
           "slot was planned to keep of its own\n"
           "       --max-context defaults to automatic: the free GPU memory picks the first rung "
           "that fits of a declared plan ladder (the same for the same model, options and card "
           "type): the model's own context (max_position_embeddings) for every slot, then the "
           "cached device states down to one per slot, the KV pool down to one context, with a "
           "Host tier (the default) the cached device states down to none, the context down by "
           "steps of 2^(-1/16) to 8192 tokens at chunk 1024, then survival rungs served with a "
           "warning and without the Windows reserve (8192 tokens at chunk 1024, the chunk "
           "halving to 128, the remaining cached states, shorter contexts); more free memory "
           "never selects less; the startup log names the rung (plan | rung k of N) and --plan "
           "prints the ladder\n"
           "       --prefill-chunk defaults to 1024, halved down to 128 by the ladder's survival "
           "rungs; an explicit value is kept\n"
           "       --kv-lease-window N: a request reserves KV for its prompt and N output tokens "
           "(at least one speculative round) instead of its whole max_tokens, and extends it as "
           "it decodes, releasing idle cached context when the pool is short; when one still "
           "cannot grow, the request admitted last leaves its lane and resumes from its text "
           "once another has finished (a request that cannot: media, penalties, a thinking "
           "budget, a constraint, waits while others run, and ends with its output limit only "
           "when every running request waits) (default 4096; 0 reserves the whole output)\n"
           "       --kv-capacity defaults to auto with an automatic context, to --max-context "
           "with one slot, and with several to auto within --max-concurrency x --max-context "
           "(at least one context); auto takes the ladder rung's pool (from every slot's "
           "context down to one), each rung counting every runtime byte (KV, cached states, "
           "workspace, CUDA graphs; a drafter's weights are loaded first), and leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB (--kv-auto-headroom-mib) for the CUDA driver's own allocations (under WDDM, "
           "native Windows or WSL 2, where Windows shares the card and pages it out to Host "
           "memory when full, a 32nd of it, at most 1 GiB, an 8th, at most 4 GiB, on a card "
           "driving the display; an 8th, at least 4 GiB, on a GPU sharing the host's memory; "
           "only a survival rung or a plan that fits no rung with it gives it up, with a "
           "warning; an explicit --kv-capacity, and an "
           "explicit --max-context at --max-concurrency 1, which makes the pool that context, "
           "keep none)\n"
           "       --no-prefix-reuse disables compatible-prefix caching (enabled by default)\n"
           "       --concurrent-prefill admits waiting requests while others prefill; the\n"
           "       prefilling requests take chunks in turn (a later request's first token\n"
           "       comes sooner, an earlier long prompt's later; output unchanged)\n"
           "       context cache defaults: device-state=2x max-concurrency, private=max(2x,+8) "
           "concurrency, "
           "shared=max(max-concurrency,7), anchors=2; Host state=max(8, 2x private) slots, Host "
           "KV=8192 MiB, "
           "both scaled down together to what the host can pin unless --host-state-slots or "
           "--host-kv-mib is given\n"
           "       --device-state-slots is extra checkpoint capacity beyond active lanes; "
           "--host-kv-mib uses MiB\n"
           "       --default-thinking-budget caps model-origin thinking for enabled requests; "
           "control tokens count toward the request output limit\n"
           "       --repetition-stop N ends an output whose last N tokens repeat one period of at "
           "most " +
           std::to_string(kRepetitionMaximumPeriod) + " tokens " +
           std::to_string(kRepetitionMinimumRepeats) +
           " or more times (finish_reason length); 0, the default, is off\n"
           "       --preserve-thinking retains closed-turn assistant reasoning in later prompts\n"
           "       --usage-chunk-choice gives the streamed usage chunk a zero-delta choice so\n"
           "       strict parsers that reject choices:[] accept the terminal usage record\n"
           "       sampler defaults come from the loaded model and resolved thinking mode; "
           "server flags and request fields override individual values.\n"
           "       --greedy forces temperature 0 (exact argmax).\n"
           "       --activations exact keeps BF16 activations on every quantized linear: an\n"
           "       answer does not depend on batch neighbours, prompt cuts, the prefix cache or\n"
           "       speculation; fast encodes NVFP4 activations in four bits on wide calls; int8\n"
           "       (Qwen) also gives int8 activations to the sites the artifact kept BF16 (heads\n"
           "       and embeddings excepted), which may move quality.\n"
           "       --spec auto (the default without --spec, --draft-tokens, --drafter or\n"
           "       --lm-head-draft; --spec none decodes plain) runs the speculation the model\n"
           "       came with for this GPU (its tuned profile): one drafter at the profile's\n"
           "       default, tuned on the whole mixed corpus, for every request\n"
           "       (--spec-select fixed, the default); --spec-select thinking runs each\n"
           "       request at the profile's default for its own thinking flag.\n"
           "       --spec-load bounds speculation under decode load instead of the profile's\n"
           "       table: 2:7,3:3,5:0 caps a step of 2+ rows at 7 drafts per row, of 3+ rows\n"
           "       at 3, and turns speculation off from 5 rows (none: no bound); 2:7:drafts\n"
           "       also sets the DFlash draft width of launches of 2+ rows (--draft-width).\n"
           "       --spec-profile TUNE.json: a tune of this model on this GPU (ninferno\n"
           "       spec-tune's output) in place of the published profile; one beside the\n"
           "       artifact is read only when named here.\n"
           "       A named tune that does not serve (unreadable, or another artifact, GPU\n"
           "       class or drafters) refuses start-up.\n"
           "       --spec-shared-floor F: a request below its break-even floor keeps drafting\n"
           "       while another request of its step drafts and it emits F tokens per round or\n"
           "       more (default 1.4; 1 or less: it stops at its floor).\n"
           "       --draft-width: the columns a DFlash drafter computes each round: block (every\n"
           "       column it was trained on) or drafts (the anchor and the round's drafts only);\n"
           "       by default block for a round of one request, drafts from two (any case).\n"
           "       Under --spec auto a width the served speculation does not run is dropped.\n";
}

// The key file's contents without its trailing line break or spaces (as an editor or `echo` leaves
// them); a key that is empty or holds a line break within it is refused.
namespace {

std::string read_api_key_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::invalid_argument("--api-key-file: cannot read " + path); }
    std::string key{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    while (!key.empty() &&
           (key.back() == '\n' || key.back() == '\r' || key.back() == ' ' || key.back() == '\t')) {
        key.pop_back();
    }
    if (key.empty()) { throw std::invalid_argument("--api-key-file: " + path + " holds no key"); }
    if (key.find_first_of("\r\n") != std::string::npos) {
        throw std::invalid_argument("--api-key-file: " + path + " holds more than one line");
    }
    return key;
}

} // namespace

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool kv_capacity_explicit        = false;
    std::optional<std::size_t> kv_auto_headroom_bytes;
    bool context_capacity_explicit = false;
    if (help_requested(argc, argv)) {
        options.help_requested = true;
        return options;
    }
    if (version_requested(argc, argv)) {
        options.version_requested = true;
        return options;
    }
    if (argc < 2) {
        throw std::invalid_argument("a model (artifact, checkpoint or hub id) is required");
    }
    options.model_source       = argv[1];
    options.artifact_path      = argv[1];
    bool host_tier_explicit    = false;
    bool spec_given            = false; // --spec, any value
    const char* api_key_source = nullptr;
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        // A flag's value as a count from 0 to INT_MAX, or as MiB in bytes.
        const auto count = [&](const char* flag) {
            return static_cast<std::uint32_t>(parse_nonnegative_int(require_value(flag), flag + 2));
        };
        const auto mib = [&](const char* flag, bool allow_zero) {
            return product::parse_mib(require_value(flag), flag + 2, allow_zero);
        };
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            if (api_key_source != nullptr) {
                throw std::invalid_argument(std::string("--api-key and ") + api_key_source +
                                            " name the key twice");
            }
            api_key_source  = "--api-key";
            options.api_key = require_value("--api-key");
            // An empty key (an unset variable expanded on the command line) would serve without
            // authentication while the operator believes it protected.
            if (options.api_key.empty()) {
                throw std::invalid_argument(
                    "--api-key must not be empty; omit it to serve without authentication");
            }
        } else if (arg == "--api-key-file") {
            if (api_key_source != nullptr) {
                throw std::invalid_argument(std::string("--api-key-file and ") + api_key_source +
                                            " name the key twice");
            }
            api_key_source  = "--api-key-file";
            options.api_key = read_api_key_file(require_value("--api-key-file"));
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = count("--max-context");
            if (options.max_context == 0) {
                throw std::invalid_argument("--max-context must be positive");
            }
        } else if (arg == "--max-context-cap") {
            options.max_context_cap = count("--max-context-cap");
            if (options.max_context_cap == 0) {
                throw std::invalid_argument("--max-context-cap must be positive");
            }
        } else if (arg.starts_with("--moe-offload")) {
            // MoE expert offload (--moe-offload, --moe-offload-compute, --moe-offload-threads)
            // was removed; its options are refused by name rather than as unknown.
            throw std::invalid_argument(
                arg.substr(0, arg.find('=')) +
                " was removed (MoE expert offload is not supported in this version)");
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--kv-auto-headroom-mib") {
            kv_auto_headroom_bytes = mib("--kv-auto-headroom-mib", true);
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = count("--max-concurrency");
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = count("--max-pending-requests");
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = count("--pending-timeout-ms");
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk          = count("--prefill-chunk");
            options.prefill_chunk_explicit = true;
        } else if (arg == "--kv-lease-window") {
            options.kv_lease_window = count("--kv-lease-window");
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = count("--log-stats-interval-ms");
        } else if (arg == "--drain-timeout-ms") {
            options.drain_timeout_ms = count("--drain-timeout-ms");
        } else if (arg == "--health-stall-ms") {
            options.health_stall_ms = count("--health-stall-ms");
        } else if (arg == "--max-request-mib") {
            options.max_request_bytes = mib("--max-request-mib", false);
        } else if (arg == "--max-request-messages") {
            const std::uint64_t messages =
                parse_u64(require_value("--max-request-messages"), "max-request-messages");
            if (messages == 0 || messages > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--max-request-messages is out of range");
            }
            options.max_request_messages = static_cast<std::size_t>(messages);
        } else if (arg == "--media-cache-mib") {
            options.media_cache_bytes = mib("--media-cache-mib", true);
        } else if (arg == "--media-live-mib") {
            options.media_live_bytes = mib("--media-live-mib", false);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--device-state-slots") {
            options.context_cache.device_state_slots = count("--device-state-slots");
            context_capacity_explicit                = true;
        } else if (arg == "--host-state-slots") {
            options.context_cache.host_state_slots = count("--host-state-slots");
            context_capacity_explicit              = true;
            host_tier_explicit                     = true;
        } else if (arg == "--host-kv-mib") {
            options.context_cache.host_kv_capacity_bytes = mib("--host-kv-mib", true);
            context_capacity_explicit                    = true;
            host_tier_explicit                           = true;
        } else if (arg == "--max-private-continuations") {
            options.context_cache.max_private_continuations = count("--max-private-continuations");
            context_capacity_explicit                       = true;
        } else if (arg == "--max-shared-prefixes") {
            options.context_cache.max_shared_prefixes = count("--max-shared-prefixes");
            context_capacity_explicit                 = true;
        } else if (arg == "--max-long-anchors-per-continuation") {
            options.context_cache.max_long_anchors_per_continuation =
                count("--max-long-anchors-per-continuation");
            context_capacity_explicit = true;
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            options.response_store_max_bytes = mib("--response-store-max-mib", false);
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--kv-dtype") {
            const std::string value = require_value("--kv-dtype");
            options.kv_cache =
                value == "auto" ? std::nullopt : std::optional(parse_kv_dtype(value.c_str()));
        } else if (arg == "--sliding-kv-dtype") {
            const std::string value = require_value("--sliding-kv-dtype");
            if (value != "bf16" && value != "fp8") {
                throw std::invalid_argument("--sliding-kv-dtype must be bf16 or fp8");
            }
            options.sliding_kv_cache = parse_kv_dtype(value.c_str());
            options.sliding_kv_given = true;
        } else if (arg == "--state-dtype") {
            const std::string value = require_value("--state-dtype");
            const auto storage      = parse_recurrent_state_storage(value);
            if (!storage) { throw std::invalid_argument("--state-dtype must be f32, bf16 or f16"); }
            options.recurrent_state = *storage;
        } else if (arg == "--activations") {
            options.activations = product::parse_activation_mode(require_value("--activations"));
            options.activations_given = true;
        } else if (arg == "--rope-yarn-factor") {
            options.rope_yarn_factor =
                parse_float_in(require_value("--rope-yarn-factor"), "rope-yarn-factor", 1.0F, 64.0F);
        } else if (arg == "--spec") {
            const std::string value = require_value("--spec");
            spec_given              = true;
            if (value == "auto") {
                options.speculation_from_profile = true;
            } else {
                options.speculative.backend = product::parse_speculative_backend(value);
            }
        } else if (arg == "--spec-select") {
            const std::string value = require_value("--spec-select");
            if (value != "thinking" && value != "fixed") {
                throw std::invalid_argument("--spec-select must be fixed or thinking");
            }
            options.per_request_speculation = value == "thinking";
        } else if (arg == "--spec-load") {
            options.speculation_load = spec::parse_load(require_value("--spec-load"));
        } else if (arg == "--spec-profile") {
            options.spec_profile = require_value("--spec-profile");
        } else if (arg == "--spec-shared-floor") {
            options.speculative.shared_trip_floor = parse_float_in(
                require_value("--spec-shared-floor"), "spec-shared-floor", 0.0f, 16.0f);
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = count("--draft-tokens");
        } else if (arg == "--ngram-draft-tokens") {
            options.speculative.ngram_draft_tokens = count("--ngram-draft-tokens");
        } else if (arg == "--ngram-min-match") {
            options.speculative.ngram_min_match = count("--ngram-min-match");
            if (options.speculative.ngram_min_match == 0) {
                throw std::invalid_argument("--ngram-min-match must be at least 1");
            }
        } else if (arg == "--drafter") {
            options.speculative.drafter = require_value("--drafter");
        } else if (arg == "--draft-width") {
            options.speculative.draft_width =
                product::parse_draft_width(require_value("--draft-width"));
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--repetition-stop") {
            const std::uint64_t window =
                parse_u64(require_value("--repetition-stop"), "repetition-stop");
            if (window != 0 &&
                (window < kRepetitionMinimumWindow || window > kRepetitionMaximumWindow)) {
                throw std::invalid_argument("--repetition-stop must be 0 (off) or in [" +
                                            std::to_string(kRepetitionMinimumWindow) + "," +
                                            std::to_string(kRepetitionMaximumWindow) + "]");
            }
            options.repetition_stop = static_cast<std::uint32_t>(window);
        } else if (arg == "--repetition-thinking") {
            const std::string mode = require_value("--repetition-thinking");
            if (mode != "stop" && mode != "conclude") {
                throw std::invalid_argument("--repetition-thinking must be stop or conclude");
            }
            options.repetition_concludes_thinking = mode == "conclude";
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--concurrent-prefill") {
            options.concurrent_prefill = true;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--chat-template") {
            options.chat_template_path = require_value("--chat-template");
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--no-auto-system-shared-prefix") {
            options.auto_system_shared_prefix = false;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--usage-chunk-choice") {
            options.usage_chunk_choice = true;
        } else if (product::parse_sampling_flag(
                       arg, [&] { return require_value(arg.c_str()); },
                       options.sampling_overrides)) {
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (product::parse_engine_flag(
                       arg, [&] { return require_value(arg.c_str()); }, options.pacing,
                       options.maintainer)) {
        } else if (arg == "--drafter-registry") {
            options.drafter_registry = require_value("--drafter-registry");
        } else if (arg == "--recipe") {
            options.recipe = require_value("--recipe");
        } else if (arg == "--quant") {
            options.quant = require_value("--quant");
        } else if (arg == "--prepare-only") {
            options.prepare_only = true;
        } else if (arg == "--plan") {
            options.plan_only = true;
        } else if (arg == "--json") {
            options.plan_json = true;
        } else if (arg == "--no-load") {
            options.plan_no_load = true;
        } else if (arg == "--plan-device") {
            options.plan_device = require_value("--plan-device");
        } else if (arg == "--with-drafters") {
            options.with_drafters = true;
        } else if (arg == "--accept-converted") {
            options.accept_converted = true;
        } else if (arg == "--refetch") {
            options.refetch = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(require_value("--log-level"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    // Without --kv-capacity: an automatic context takes an automatic KV capacity (the rest of the
    // free memory). An explicit one is each request's limit: one slot takes exactly its tokens;
    // more slots take an automatic capacity, which the Engine bounds by the context's pages below
    // and every slot's full context above (--max-concurrency x --max-context when that fits, else
    // the most that does), so a server does not run fewer requests than it admits.
    if (options.max_context != 0 && options.max_context_cap != 0) {
        throw std::invalid_argument("--max-context-cap caps an automatic context: pass it or "
                                    "--max-context, not both");
    }
    if (!kv_capacity_explicit) {
        options.kv_capacity = options.max_context == 0 || options.max_concurrency > 1
                                  ? KvCapacityPolicy::automatic()
                                  : KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    // The automatic pool's reserve for the CUDA driver's own allocations. Only an
    // automatic capacity has one: with an explicit one the option would do nothing, so it fails.
    if (kv_auto_headroom_bytes) {
        if (options.kv_capacity.mode != KvCapacityMode::Automatic) {
            throw std::invalid_argument(
                "--kv-auto-headroom-mib needs an automatic KV capacity (--kv-capacity auto)");
        }
        options.kv_capacity.automatic_headroom_bytes = *kv_auto_headroom_bytes;
    }
    if (options.plan_json && !options.plan_only) {
        throw std::invalid_argument("--json prints the --plan dry run: it needs --plan");
    }
    if (options.accept_converted && options.refetch) {
        throw std::invalid_argument("--accept-converted keeps the converted weights, --refetch "
                                    "replaces them: name one");
    }
    if (options.plan_no_load && !options.plan_only) {
        throw std::invalid_argument("--no-load plans without the weights: it needs --plan");
    }
    if (!options.plan_device.empty() && !options.plan_no_load) {
        throw std::invalid_argument("--plan-device names a board to plan for: it needs --plan "
                                    "--no-load");
    }
    if (options.plan_only && options.prepare_only) {
        throw std::invalid_argument("--plan and --prepare-only exclude each other");
    }
    // The default Host tier is a default: the Engine fits it to what the host can pin.
    options.context_cache.host_tier_automatic = !host_tier_explicit;
    if (!options.allow_prefix_reuse) {
        if (context_capacity_explicit) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with context-cache capacity options");
        }
        options.context_cache.disable();
    }
    // No --spec and nothing that configures an explicit speculation: the model's profile, as
    // --spec auto (and as the CLI does).
    if (!spec_given && options.speculative.backend == SpeculativeBackend::None &&
        options.speculative.draft_tokens == 0 && options.speculative.drafter.empty() &&
        options.speculative.proposal_head == ProposalHead::Full) {
        options.speculation_from_profile = true;
        options.speculation_by_default   = true;
    }
    if (options.speculation_load && !options.speculation_from_profile &&
        options.speculative.backend == SpeculativeBackend::None) {
        throw std::invalid_argument("--spec-load bounds speculation: it needs --spec");
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    if (options.speculation_from_profile &&
        (options.speculative.backend != SpeculativeBackend::None ||
         options.speculative.draft_tokens != 0 || !options.speculative.drafter.empty() ||
         options.speculative.proposal_head != ProposalHead::Full)) {
        throw std::invalid_argument(
            "--spec auto takes its drafter, depths and head from the model's profile");
    }
    // An explicit --spec without --draft-tokens drafts at the depth the model's profile runs that
    // backend with on this GPU, else at product::default_draft_tokens (resolved at startup).
    options.draft_tokens_from_model = !options.speculation_from_profile &&
                                      options.speculative.backend != SpeculativeBackend::None &&
                                      options.speculative.draft_tokens == 0;
    product::validate_speculative_cli_options(options.speculative, options.draft_tokens_from_model);
    product::validate_draft_width(options.speculative, options.speculation_from_profile);
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    // Without a flag the key may come from the environment, which, unlike the command line, other
    // users cannot list.
    if (api_key_source == nullptr) {
        if (const char* key = std::getenv("NINFERNO_API_KEY"); key != nullptr) {
            if (*key == '\0') {
                throw std::invalid_argument("NINFERNO_API_KEY is set but empty; unset it to serve "
                                            "without authentication");
            }
            options.api_key = key;
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_name.empty()) {
        throw std::logic_error("loaded artifact model name must not be empty");
    }
    return std::string(artifact_model_name);
}

} // namespace ninferno::serve
