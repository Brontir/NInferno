#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;
inline constexpr std::size_t kMaximumExplicitPromptCacheMarkers  = 4;
// Explicit markers plus the engine's automatic tool/leading-instruction/full-prompt candidates;
// one request's shared-prefix opportunities never exceed this (the frontend reserves this many).
inline constexpr std::size_t kMaximumPreparedPromptCacheCandidatesPerRequest = 7;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes    = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes     = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes      = 2ULL << 30;
inline constexpr std::uint32_t kDefaultHostStateSlots    = 8;
inline constexpr std::size_t kDefaultHostKvCapacityBytes = 8ULL << 30;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    Nvfp4Group16,
    Fp8KeyNvfp4Value,
};

// The --kv-dtype spelling of each storage, as every tool parses and prints it.
inline constexpr std::string_view kKvCacheStorageNames = "bf16|int8|fp8|nvfp4|k8v4";

[[nodiscard]] constexpr std::string_view kv_cache_storage_name(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

// The storage's layout for reports: the int8 and fp8 storages with their scale grouping, the
// others as their --kv-dtype spelling.
[[nodiscard]] constexpr std::string_view kv_cache_storage_layout(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::Int8Group64:
        return "int8-group64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-row256";
    default:
        return kv_cache_storage_name(storage);
    }
}

// How the linear-attention layers keep their recurrent state between calls (--state-dtype).
// - FP32: --activations exact, and the families without an automatic choice.
// - BF16: half the bytes of every state slot and cached state image (Qwen 3.5 GDN layers). The
//   recurrence still runs in FP32 registers and rounds only the state it stores, so the rounding
//   does not feed back within a call; across calls it does, which is why it is opt-in.
// - FP16: the same bytes as BF16 with an 11-bit significand instead of 8:
//   eight times finer rounding where the state stays inside FP16's range,
//   which the decayed GDN state does (stores saturate at 65504). The automatic choice of the
//   Qwen 3.5 families under --activations fast (EngineOptions::recurrent_state_automatic): over
//   a long stepwise decode it stays closer to FP32 prefill logits than BF16 does, and speeds up
//   batched decode by a few percent.
enum class RecurrentStateStorage : std::uint8_t {
    FP32,
    BF16,
    FP16,
};

[[nodiscard]] constexpr std::string_view
recurrent_state_storage_name(RecurrentStateStorage storage) noexcept {
    switch (storage) {
    case RecurrentStateStorage::BF16:
        return "bf16";
    case RecurrentStateStorage::FP16:
        return "f16";
    case RecurrentStateStorage::FP32:
        break;
    }
    return "f32";
}

// The storage a --state-dtype spelling names; nullopt for any other text.
[[nodiscard]] constexpr std::optional<RecurrentStateStorage>
parse_recurrent_state_storage(std::string_view text) noexcept {
    if (text == "f32") { return RecurrentStateStorage::FP32; }
    if (text == "bf16") { return RecurrentStateStorage::BF16; }
    if (text == "f16") { return RecurrentStateStorage::FP16; }
    return std::nullopt;
}

// The storage a --kv-dtype spelling names; nullopt for any other text.
[[nodiscard]] constexpr std::optional<KvCacheStorage>
parse_kv_cache_storage(std::string_view text) noexcept {
    for (const KvCacheStorage storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        if (text == kv_cache_storage_name(storage)) { return storage; }
    }
    return std::nullopt;
}

// How quantized linears treat their activations (--activations exact|fast|int8): the user's
// choice of 16, automatic (down to 4) or 8 bits.
// - Exact: BF16 activations on every quantized linear at every width. A request's tokens do not
//   depend on its batch neighbours, where its prompt is cut, the prefix cache or speculation
//   (greedy agrees up to BF16 logit ties). A model whose kernels have no BF16-activation route
//   refuses to load under Exact rather than run another one.
// - Fast: each weight's recorded policy with the width-dependent routes: wide calls may encode
//   activations in four (NVFP4) or eight bits, so the answer can depend on the load.
enum class ActivationMode : std::uint8_t {
    Fast,
    Exact,
    // Eight bits: every quantized linear is allowed int8 activations where its route has them
    // and none takes four, whatever the artifact recorded (a head or embedding recorded A16Only
    // keeps BF16). A W4A16 NVFP4 checkpoint thus runs W4A8 where Fast runs W4A4. May move
    // quality, so never the default.
    Int8,
};

// The one place the default is chosen.
inline constexpr ActivationMode kDefaultActivationMode = ActivationMode::Fast;

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

[[nodiscard]] constexpr std::string_view kv_capacity_mode_name(KvCapacityMode mode) noexcept {
    return mode == KvCapacityMode::Automatic ? "auto" : "explicit";
}

// What an automatic KV capacity leaves unplanned: the CUDA driver's own Device allocations that
// no plan models (streams, events, the code of kernels loaded on first launch, graph executables
// beyond their allowance). Everything else the runtime takes is in the Program's reservation (KV,
// cached StateImages, workspace, CUDA graph allowance) or was loaded before the free memory was
// measured (weights, a drafter). Measured with the reservation exact (explicit capacity, no
// headroom), the driver took up to about 20 MiB beyond the reservation while starting, and
// nothing more while serving (long prefills, decode, speculation, prefix reuse).
inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 32ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;
    // Kept back beside the headroom when the free memory allows, never refused for: an automatic
    // pool gives up its pages for it down to the minimum reservation (a GPU shared through WDDM,
    // runtime/engine/kv_capacity.h with_paravirtual_headroom).
    std::size_t best_effort_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes, 0};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

// How many columns of its block a DFlash drafter computes in a round of the serving Engine
// (SpeculativeOptions and SpeculationLoad draft_width; ninferno-dense's and ninferno-gemma4's
// Sessions always draft the whole block). The drafts are the same K either way, but the block
// attention is bidirectional, so the width changes what the drafter proposes: its acceptance, and
// with it a sampled request's tokens, as any other drafter change would; a greedy request's tokens
// are plain decode's at every width.
enum class DFlashDraftWidth : std::uint8_t {
    // The engine's choice: Block for a launch of one row, Drafts for a launch of two rows or more
    // (a family that drafts only its K + 1 columns keeps Drafts at every width).
    Default,
    // Every column of the block the drafter was trained on (its block_size).
    Block,
    // The anchor and the round's K drafts only (K + 1 columns).
    Drafts,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
};

// A shallower depth a request moves to when its acceptance watchdog trips at the depth above it,
// with that depth's own floor (a profile's break-even point for it).
struct SpeculationStep {
    std::uint32_t draft_tokens = 0;
    double acceptance_floor    = 0.0;
};

// A bound on speculation under decode load: a decode step of at least `lanes` rows verifies at
// most `draft_tokens` drafts per row, and at 0 its rows decode plainly (SpeculativeOptions::load).
// A DFlash round whose launch holds at least `lanes` rows drafts at `draft_width` (Default: the
// engine's choice for them) unless SpeculativeOptions::draft_width names one for every round. The
// width follows the launch's rows (its captured batch: one graph, one width), where the depth
// follows the step's; they differ when rows of the step decode plainly beside the launch.
// (draft_width made the struct 12 bytes, from 8.)
struct SpeculationLoad {
    std::uint32_t lanes          = 0;
    std::uint32_t draft_tokens   = 0;
    DFlashDraftWidth draft_width = DFlashDraftWidth::Default;
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    // The default K of a request: MTP 1..5; DFlash and DFlash2 1..15 (query width K+1).
    std::uint32_t draft_tokens = 0;
    // Further depths a request may choose (ExecutionOptions::speculation), in the same range.
    // The engine captures its speculative CUDA graphs for draft_tokens and each of these at
    // start; its window is the largest. A decode step verifies its speculating rows in one launch
    // at the smallest captured depth that covers the deepest of them; every row drafts at that
    // launch's width.
    std::vector<std::uint32_t> request_draft_tokens;
    // Copy rounds (prompt lookup beside the MTP drafter; 0: off; kCopyWindowAuto, the default: a
    // window of 15 on a family whose MTP round runs them, else off): before a round, a request
    // whose last ngram_min_match or more tokens recur earlier in its context (prompt and output)
    // verifies the up to ngram_draft_tokens tokens that followed their most recent longest
    // occurrence instead of its MTP drafts, when there are more of them, in a launch window of its
    // own (ngram_draft_tokens drafts, deeper than every draft depth). Its drafter still proposes
    // its depth for the next round. The verify is the ordinary one, so the output is unchanged.
    // Copy rounds feed no watchdog, and run only where every MTP draft agrees with the copy.
    // Qwen 3.5/3.6 and Gemma 4 run them; other families refuse an explicit nonzero value. They pay
    // most where the output repeats its context (summaries, edits of a given text).
    static constexpr std::uint32_t kCopyWindowAuto = 0xffffffffU;
    std::uint32_t ngram_draft_tokens = kCopyWindowAuto;
    std::uint32_t ngram_min_match    = 12;
    ProposalHead proposal_head = ProposalHead::Full;
    // The DFlash drafter's width in every round (Block or Drafts), or Default: per round, the width
    // of the load bound that applies to its launch's rows (SpeculationLoad::draft_width), else the
    // engine's choice for them. A DFlash family that drafts only K + 1 columns refuses Block.
    DFlashDraftWidth draft_width = DFlashDraftWidth::Default;
    // The drafter artifact of a family whose drafter ships separately from the target (Gemma 4's
    // MTP assistant). Empty: the drafter the target artifact carries.
    std::filesystem::path drafter;
    // Acceptance watchdog: a request whose tokens emitted per round over its last
    // watchdog_window rounds fall below the trip point stops drafting for the
    // rest of the request, while the other requests of its batches keep speculating. While another
    // row of a step drafts, it verifies without drafts in that step's speculative launch (fallback
    // steps: one launch, not a second, ordinary one beside it) while the step's stopped rows
    // verify at most 16 columns in all (their count times the launch's width); beyond that, or
    // from the first step where no row drafts, it decodes plainly, through the plain decode
    // graphs a speculative engine also captures (plain steps). Its SpeculativeStats report the
    // trip (watchdog_tripped). The floor is the configuration's break-even point (a profile's
    // floor, see spec-tune); the trip point sits halfway between one token a round and it,
    // 1 + (acceptance_floor - 1) / 2, so a request trips only well below the break-even. 1 or
    // less: off.
    double acceptance_floor           = 0.0;
    // Before decoding plainly, a tripped request steps down through these shallower depths, deepest
    // first, each judged against its own floor from the round it starts (SpeculativeStats
    // step_downs); plain decode follows a trip at the last one. Each depth is below the one before
    // it (the first below draft_tokens); a launch covers it with the smallest captured window that
    // does. Empty: plain at once.
    std::vector<SpeculationStep> step_down;
    // The load bound (a tuned profile's "load" table), by ascending lanes from 2. A decode step of
    // n rows applies the entry with the largest lanes <= n (below the first entry: no bound). A
    // bound K caps every speculating row of the step at K drafts: the step verifies at most K + 1
    // columns per row, whatever the rows' own depths. A bound of 0 turns the step's speculating
    // rows to plain decode for the rest of their requests (their drafter state does not follow
    // plain steps, as after a watchdog trip). Rounds verified below a request's own depth do not
    // feed its watchdog, whose floor is that depth's. Each nonzero bound is at most the engine's
    // window and is a captured launch window of its own, from its lanes on; no nonzero bound
    // follows a 0, and from a 0's lanes on no speculative graph or round workspace is planned.
    // Empty: a step verifies at the depth of its deepest row.
    std::vector<SpeculationLoad> load;
    // The context bound (a tuned profile's "context_limit"): a request whose prompt holds more
    // tokens decodes plainly. A round's drafter and verify read the whole context while plain
    // decode pays for it once per token, so past some length the rounds cost more than their
    // tokens save. 0: none.
    std::uint32_t context_limit = 0;
    std::uint32_t watchdog_window     = 32;
    std::uint32_t watchdog_min_rounds = 16;
    // The acceptance floor is a break-even against plain decode of one request. A request whose
    // watchdog trips while another row of its step keeps drafting (the step's speculative launch
    // runs anyway; stopped, it would verify its columns there without drafts) keeps drafting as
    // long as it emits at least this many tokens per round, judged again after each round
    // (SpeculativeStats::shared_trip_deferrals). 1 or less: it stops at its floor.
    double shared_trip_floor = 1.4;
    // Where the product's speculation choices above came from, for the decision fingerprint
    // (runtime/engine/decision_fingerprint.h, section "sources"): "speculation.profile" and
    // "drafter_registry", each "<kind> sha256:<hex>", and "spec_fit_fallback" when a capacity
    // refusal made --spec auto fall back (src/serve/generation_service.cpp). The Engine reads no
    // profile itself; the product code that resolved one (src/product/speculation) fills this.
    // Never read by a decision.
    std::map<std::string, std::string> decision_sources;
};

enum class StartupPhase : std::uint8_t {
    EngineStartup,
    CudaInitialize,
    ArtifactInspect,
    TargetPlan,
    WeightsMaterialize,
    WeightsStagingPin,
    TargetFinalize,
    FrontendInitialize,
    ProgramInitialize,
    HostStatePin,
    HostKvPin,
    CudaGraphPrepare,
    KernelWarmUp,
    EngineFinalize,
};

enum class StartupStatus : std::uint8_t {
    Begin,
    Progress,
    Complete,
    Failed,
};

enum class StartupProgressUnit : std::uint8_t {
    None,
    Bytes,
};

struct StartupEvent {
    StartupPhase phase                = StartupPhase::EngineStartup;
    StartupStatus status              = StartupStatus::Begin;
    StartupProgressUnit progress_unit = StartupProgressUnit::None;
    std::uint64_t current             = 0;
    std::uint64_t total               = 0;
    std::uint64_t elapsed_ns          = 0;
};

struct StartupObserver {
    // Startup diagnostics never participate in Engine control flow. Callback exceptions are
    // ignored by the publishing boundary so a logging failure cannot invalidate model startup.
    std::function<void(const StartupEvent& event)> callback;
};

struct ContextCacheOptions {
    // Engine resolves every optional once at construction. With C=max_concurrency, the enabled
    // defaults are H=2C (or fewer, see below), R=8, Host KV=8 GiB, P=max(2C, C+8), S=max(C,7)
    // and L=2; Engine::options() returns those effective values.
    bool enabled = true;
    // Extra Device checkpoint StateImage slots H. Total Device StateImage capacity is C + H.
    // Unset: two cached states per active lane (H = 2C), or as many of them as fit in the free
    // Device memory beside the KV capacity policy; none at all when the Host tier holds the
    // StateImages instead.
    std::optional<std::uint32_t> device_state_slots;
    // Host StateImages and Host KV bytes are independently configured pinned-memory capacities.
    std::uint32_t host_state_slots     = kDefaultHostStateSlots;
    std::size_t host_kv_capacity_bytes = kDefaultHostKvCapacityBytes;
    // false: the two are kept, and construction refuses with the numbers when the host cannot pin
    // them. true: they are defaults the Engine scales down together, when the host cannot pin them
    // (a share of MemAvailable minus the headroom kept for the rest of the host, see
    // core/host_memory.h), and Engine::options() / MemorySummary report what it chose.
    bool host_tier_automatic = false;
    // Bounded private/shared logical catalogs and per-continuation long-anchor count.
    std::optional<std::uint32_t> max_private_continuations;
    std::optional<std::uint32_t> max_shared_prefixes;
    std::optional<std::uint32_t> max_long_anchors_per_continuation;

    // No cross-request context: nothing is retained and no Host memory is pinned for it (a
    // product that runs one request, or ninferno-serve --no-prefix-reuse).
    void disable() noexcept {
        enabled                = false;
        host_state_slots       = 0;
        host_kv_capacity_bytes = 0;
    }
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    EnginePurpose purpose              = EnginePurpose::Generation;
    int device                         = 0;    // CUDA device ordinal
    std::uint32_t max_context          = 2048; // Logical ceiling of one request or score window.
    // true: the Engine chooses max_context (then a cap when nonzero): the model's own position
    // capacity when a plan with it fits the free Device memory, else the longest whole number of
    // KV pages (64 tokens) below it, from 1024 up, that fits (runtime/engine/kv_capacity.h),
    // from the free memory measured once at construction. Engine::options() and MemorySummary
    // report the choice.
    bool max_context_automatic         = false;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    // Requests allowed to wait behind the running ones (submit refuses past it), and the
    // longest a request waits for admission, counted from submit. The long default suits agent
    // clients that queue long prompts.
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 600000;
    // Admit waiting requests while others prefill; the prefilling requests take chunks in turn.
    bool concurrent_prefill = false;
    std::uint32_t prefill_chunk        = 1024;
    // true: prefill_chunk is the widest chunk, and an automatic max_context that stays below its
    // floor (runtime/engine/kv_capacity.h: 8192 tokens, or the model's own when smaller) with it
    // is sized again with half the chunk, then a quarter (not below 256): a narrower chunk takes
    // less workspace and, on a model with sliding-window layers, less StateImage per state. The
    // first chunk that reaches the floor is kept, else the one with the longest context.
    // Engine::options() reports the chunk chosen.
    bool prefill_chunk_automatic       = false;
    // Output tokens of KV a request reserves past its prompt at admission (0: its whole output
    // budget). A request then extends its reservation as it decodes, releasing idle cached
    // context when the pool is short; when that frees no page, the request admitted last leaves
    // its lane and resumes from its text later (docs/reference/serving.md, --kv-lease-window). A
    // client asking for far more output than it uses (an agent's max_tokens 32000) then no longer
    // holds a whole context out of the pool while it answers in a few hundred tokens.
    // ninferno-serve passes 4096.
    std::uint32_t kv_lease_window      = 0;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    // true: kv_cache is ignored and the family of the artifact chooses it (FamilyEntry::kv_cache):
    // Int8Group64 for Qwen 3.5, FP8 rows (Fp8E4M3Row256) for Gemma 4 and the dense and MoE
    // decoders (BFloat16 below sm_89), BFloat16 for Nemotron-H. Engine::options() reports the
    // choice. Narrower keys and values cut the KV traffic of decode (faster the longer the
    // context), and held BF16's accuracy on long-context retrieval and reasoning evaluations.
    bool kv_cache_automatic = false;
    // --state-dtype f32|bf16|f16 (RecurrentStateStorage): the Qwen 3.5 GDN layers' recurrent state.
    // The other families keep FP32 and refuse BF16.
    RecurrentStateStorage recurrent_state = RecurrentStateStorage::FP32;
    // true: recurrent_state is ignored; FP32 under --activations exact, else the family's choice
    // (FamilyEntry::recurrent_state: FP16 for Qwen 3.5). Engine::options() reports the choice.
    bool recurrent_state_automatic = false;
    // The sliding-window layers' rings (a sequence's StateImage): BFloat16, or Fp8E4M3Row256 (E4M3
    // codes with an FP32 scale per row, about half the bytes; Gemma 4 and dense decoders with
    // sliding-window layers, refused elsewhere). Opt-in: Gemma 4
    // 26B/31B next-token NLL moves by less than its standard error with every layer in FP8
    // (docs/serving.md, FP8 keys and values).
    KvCacheStorage sliding_kv_cache    = KvCacheStorage::BFloat16;
    ActivationMode activations         = kDefaultActivationMode;
    // --rope-yarn-factor F: Qwen 3.5/3.6 rotate with YaRN (transformers' "yarn" rope over the
    // checkpoint's own context as original_max_position_embeddings) and admit F times that
    // context. 1, the default: the checkpoint's RoPE. Other families refuse any other value.
    float rope_yarn_factor = 1.0F;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false; // load the vision tower (image and video input)
    bool use_cuda_graph                    = true;  // decode through captured CUDA graphs
    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    StartupObserver startup_observer;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;
    // Where the presets come from, for the server to state: "generation_config.json" (what the
    // checkpoint recommends), "transformers defaults" (a generation config that names no sampling),
    // or the family's own documented presets.
    std::string source;
    // What the checkpoint recommends that is not applied, and why (e.g. a repetition_penalty).
    std::vector<std::string> notes;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// The sampler keeps at most this many candidates; a request's top_k is in [0, kMaximumTopK] (0:
// no limit of its own, so kMaximumTopK) and a resolved one in [1, kMaximumTopK].
inline constexpr std::int32_t kMaximumTopK = 64;

// The closed range a sampling field accepts (OpenAI's; the Anthropic dialect narrows temperature
// to [0,1]). The request parsers, the sampling flags and Engine resolution share them.
struct SamplingRange {
    float low  = 0.0F;
    float high = 0.0F;

    [[nodiscard]] constexpr bool contains(double value) const noexcept {
        return value >= static_cast<double>(low) && value <= static_cast<double>(high);
    }
};

inline constexpr SamplingRange kTemperatureRange{0.0F, 2.0F};
inline constexpr SamplingRange kProbabilityRange{0.0F, 1.0F}; // top_p and min_p
inline constexpr SamplingRange kPenaltyRange{-2.0F, 2.0F};    // presence and frequency penalties

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

// StopPolicy::repetition_window: a loop is one period of at most kRepetitionMaximumPeriod tokens
// repeated at least kRepetitionMinimumRepeats times; a window is 0 (off) or in
// [kRepetitionMinimumWindow, kRepetitionMaximumWindow].
inline constexpr std::uint32_t kRepetitionMaximumPeriod  = 64;
inline constexpr std::uint32_t kRepetitionMinimumRepeats = 4;
inline constexpr std::uint32_t kRepetitionMinimumWindow  = 16;
inline constexpr std::uint32_t kRepetitionMaximumWindow  = 65536;

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    // Also stop at the checkpoint's own end tokens (its end of sequence and end of turn). A
    // constrained output keeps them whatever this says: its grammar ends there.
    bool include_model_defaults = true;
    // The text of the stop token that ended the output stays in it (raw output); off, held back.
    bool publish_stop_token     = false;
    // Ends the output (FinishReason::Repetition) once its last `repetition_window` tokens repeat
    // one period of at most kRepetitionMaximumPeriod tokens at least kRepetitionMinimumRepeats
    // times: a model caught in a loop. 0: off.
    std::uint32_t repetition_window = 0;
    // A loop found while the thinking phase is still open closes the thinking instead (the same
    // canonical control suffix a thinking budget injects, once) and the answer follows; a loop in
    // the answer, or where the template has no control suffix or the output limit leaves no room
    // for it, still ends the output. Needs repetition_window.
    bool repetition_concludes_thinking = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
};

// A request's speculation in a speculative engine: its draft depth, from 1 up to the engine's
// window (SpeculativeOptions: the largest of draft_tokens and request_draft_tokens), or 0 for
// plain decode; and its acceptance watchdog floor in tokens per round (1 or less: off).
struct RequestSpeculation {
    std::uint32_t draft_tokens = 0;
    double acceptance_floor    = 0.0;
    std::vector<SpeculationStep> step_down; // as SpeculativeOptions::step_down, below draft_tokens
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    // The output budget in tokens. 0 generates nothing: the request finishes at once with
    // FinishReason::OutputLimit.
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
    // Absent: the engine's draft_tokens and acceptance_floor.
    std::optional<RequestSpeculation> speculation;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds only Qwen's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
};

// ---- Constrained decoding (structured output) -------------------------------------------------

enum class OutputConstraintKind : std::uint8_t {
    JsonObject,    // any JSON object
    JsonSchema,    // definition: JSON Schema document (the supported subset is enforced exactly)
    Regex,         // definition: regular expression the whole constrained text must match
    Choice,        // choices: the constrained text equals one of these strings
    Grammar,       // definition: EBNF grammar (GBNF-compatible) with root rule "root"
    StructuralTag, // definition: XGrammar structural-tag JSON (model markup such as tool calls)
    ToolCalls,     // tools: the model's tool-call markup; the Engine renders it for its model
};

enum class ToolCallRequirement : std::uint8_t {
    Optional, // text and/or any number of calls
    Required, // at least one call
    Named,    // exactly the call to named_tool, nothing else
};

struct ToolCallConstraintTool {
    std::string name;
    // JSON Schema of the arguments. Enforced when strict; otherwise the arguments are free text in
    // the model's markup.
    std::string parameters_json;
    bool strict = false;
};

struct OutputConstraintSpec {
    OutputConstraintKind kind = OutputConstraintKind::JsonObject;
    std::string definition;
    std::vector<std::string> choices;
    // ToolCalls only.
    std::vector<ToolCallConstraintTool> tools;
    ToolCallRequirement tool_requirement = ToolCallRequirement::Optional;
    std::string named_tool;
    bool parallel_tool_calls = true;
    // ToolCalls and StructuralTag: a JSON Schema the answer may match instead of the markup
    // (response_format beside tools). The output is then either the calls or that JSON, nothing
    // else; empty means no such alternative.
    std::string answer_schema;
    // JSON kinds only: forbid insignificant whitespace. Otherwise a bounded whitespace run is
    // admitted between JSON tokens.
    bool compact_json = false;
    // The constrained text starts after a reasoning section. A short leading whitespace run, which
    // the output presentation strips from content, is admitted before it.
    bool follows_reasoning = false;
    // Special tokens are visible in the output (OutputOptions::preserve_special_tokens or raw), so
    // their bytes take part in matching. Otherwise the constraint never admits a special token.
    bool special_tokens_visible = false;
};

// A compiled constraint: immutable, shareable across requests, and bound to one Engine's
// tokenizer.
class OutputConstraint;

// Raised by Engine::compile_output_constraint. code() is a stable machine-readable reason such as
// "invalid_schema", "unsupported_schema", "invalid_regex", "invalid_grammar", "invalid_choice",
// "invalid_structural_tag", "constraint_too_large", "constraint_compile_timeout" or
// "constraint_compiler_busy".
class OutputConstraintError final : public std::invalid_argument {
public:
    OutputConstraintError(std::string code, std::string message)
        : std::invalid_argument(std::move(message)), code_(std::move(code)) {}

    [[nodiscard]] const std::string& code() const noexcept { return code_; }

private:
    std::string code_;
};

struct OutputConstraintOptions {
    // Null means unconstrained. The constraint applies to the content after the reasoning section
    // when the prompt starts in reasoning, and from the first generated token otherwise.
    std::shared_ptr<const OutputConstraint> compiled;
};

struct OutputConstraintStats {
    bool active = false;
    // The committed content is a complete sentence of the constraint.
    bool complete = false;
    // The output cannot be presented as conforming: a committed token was rejected, the device
    // could not apply a mask, or the matcher failed. violation names the cause.
    bool violated = false;
    std::string violation;
    std::uint64_t masked_positions = 0;
    double mask_host_seconds       = 0.0;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
    OutputConstraintOptions constraint;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

enum class ImageResizePolicy : std::uint8_t {
    Downsize,
    RejectOversized,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Model-origin structured output. Protocol adapters own any wire-level call identifier.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
};

// Terminal interpretation of model-origin tool-call markup. Parameter schemas guide JSON
// normalization but do not validate the call; only a structure/identity failure can return a
// complete marker region to ordinary content.
enum class ToolCallParseFallbackReason : std::uint8_t {
    None,
    MalformedStructure,
    DuplicateParameter,
    InvalidToolName,
    UndeclaredTool, // Llama 3's JSON form only; elsewhere an undeclared name is a call
    TrailingContent,
};

[[nodiscard]] inline constexpr const char*
tool_call_parse_fallback_reason_name(ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ToolCallParseFallbackReason::None:
        return "none";
    case ToolCallParseFallbackReason::MalformedStructure:
        return "malformed_structure";
    case ToolCallParseFallbackReason::DuplicateParameter:
        return "duplicate_parameter";
    case ToolCallParseFallbackReason::InvalidToolName:
        return "invalid_tool_name";
    case ToolCallParseFallbackReason::UndeclaredTool:
        return "undeclared_tool";
    case ToolCallParseFallbackReason::TrailingContent:
        return "trailing_content";
    }
    return "malformed_structure";
}

struct ToolCallParseDiagnostics {
    bool marker_seen                            = false;
    std::uint32_t structured_call_count         = 0;
    std::uint32_t empty_arguments_omitted       = 0;
    std::uint32_t schema_mismatch_arguments     = 0;
    std::uint32_t repeated_parameters           = 0; // a repeated parameter kept its last value
    ToolCallParseFallbackReason fallback_reason = ToolCallParseFallbackReason::None;
    // The calls came from a thinking span the model ended with its stop token without closing
    // it: the content channel carried none.
    bool recovered_from_reasoning = false;
    // The model ended the turn with its stop token inside a thinking span it never closed and no
    // call came from it: the reply has reasoning and no content.
    bool ended_inside_reasoning = false;
    // The output limit cut the last call off; the complete calls before it were kept and the
    // unfinished call's markup dropped.
    std::uint32_t cut_calls_dropped = 0;
    // Calls published with a name the request did not declare; the client answers them.
    std::uint32_t undeclared_calls = 0;
    // Bytes of text the model wrote after its complete calls, returned as content after them.
    std::uint32_t trailing_content_bytes = 0;
    // Qwen XML calls written as bare <function=NAME> blocks without <tool_call>, taken because
    // every name is declared.
    std::uint32_t unwrapped_calls = 0;
    // A last call whose </function> the stop token cut off after a closed parameter, closed.
    std::uint32_t closed_calls = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const ToolCallParseDiagnostics&, const ToolCallParseDiagnostics&) noexcept = default;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::string_view reasoning_effort_name(ReasoningEffort effort) noexcept {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return {};
}

[[nodiscard]] constexpr std::optional<ReasoningEffort>
parse_reasoning_effort(std::string_view value) noexcept {
    for (const ReasoningEffort effort :
         {ReasoningEffort::None, ReasoningEffort::Minimal, ReasoningEffort::Low,
          ReasoningEffort::Medium, ReasoningEffort::High, ReasoningEffort::XHigh,
          ReasoningEffort::Max}) {
        if (value == reasoning_effort_name(effort)) { return effort; }
    }
    return std::nullopt;
}

enum class PromptContinuationMode : std::uint8_t {
    NewAssistantTurn,
    ContinueFinalAssistant,
};

struct PromptOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // JSON object of template parameters. Unset typed fields leave template defaults intact.
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
    // False (tool_choice none) keeps the declarations in the prompt, so the conversation's cached
    // prefix stays the same, but no output is read back as a call.
    bool tool_calls_enabled = true;
};

enum class CacheRetentionHint : std::uint8_t {
    Default,
    LiveSession,
    Disposable,
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class SharedCandidateEvidence : std::uint8_t {
    None               = 0,
    ExplicitBoundary   = 1U << 0U,
    RequestedAutomatic = 1U << 1U,
    DefaultAutomatic   = 1U << 2U,
    EngineStructural   = 1U << 3U,
    EngineObserved     = 1U << 4U,
};

[[nodiscard]] constexpr SharedCandidateEvidence operator|(SharedCandidateEvidence left,
                                                          SharedCandidateEvidence right) noexcept {
    return static_cast<SharedCandidateEvidence>(static_cast<std::uint8_t>(left) |
                                                static_cast<std::uint8_t>(right));
}

constexpr SharedCandidateEvidence& operator|=(SharedCandidateEvidence& left,
                                              SharedCandidateEvidence right) noexcept {
    left = left | right;
    return left;
}

[[nodiscard]] constexpr bool has_shared_candidate_evidence(SharedCandidateEvidence value,
                                                           SharedCandidateEvidence evidence) {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(evidence)) != 0;
}

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    MessagePartBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence   = SharedCandidateEvidence::ExplicitBoundary;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;
    // For MessagePartBoundary, after_message_count identifies the containing message using a
    // one-based count and this value identifies the number of serialized parts within it.
    std::uint32_t after_message_part_count = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

struct ContextCacheHints {
    std::optional<std::string> session_key;
    CacheRetentionHint retention = CacheRetentionHint::Default;
    std::vector<PromptCacheMarker> markers;
    // Protocols with their own automatic/explicit write policy disable the Engine's structural
    // candidates. Exact reads from already-published shared prefixes remain enabled.
    bool allow_engine_automatic_shared_prefixes = true;
    // Advance the named session lineage when session_key is present. This does not require an
    // anonymous content-matched source to be retained.
    bool update_session_index = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    ThinkingBudgetCapacityInsufficient,
    MediaBudgetExceeded,
    InvalidMedia,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
    // The model computed a NaN or an infinity for the request (its final hidden state); the
    // request fails alone and the server keeps serving the others.
    NumericalFailure,
};

// The free Device memory cannot hold what the EngineOptions ask for: the KV capacity of the
// requested max_context with the model, its drafter and the cached Device states. what() carries
// the refusal's numbers. An Engine construction that throws it left nothing allocated behind, so a
// caller may construct again with options that need less (a smaller context, another drafter).
class CapacityError final : public std::invalid_argument {
public:
    explicit CapacityError(const std::string& message) : std::invalid_argument(message) {}
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    bool starts_in_reasoning    = false;
    // The model may open a reasoning section itself (its prompt does not already hold one).
    bool reasoning_open_allowed = false;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
    // The chat template defines a thinking-budget control (text that ends reasoning early), so a
    // thinking budget can be enforced; gpt-oss's harmony template defines none.
    bool thinking_budget_control = false;

    // The request thinks: its prompt starts the reasoning (Qwen), or leaves the model to open it
    // (Gemma 4 with thinking on, gpt-oss). It selects the thinking sampler preset and speculation.
    [[nodiscard]] constexpr bool thinking() const noexcept {
        return starts_in_reasoning || reasoning_open_allowed;
    }
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
    // The output repeated one short period (StopPolicy::repetition_window).
    Repetition,
};

// The request log's spelling of each reason; the operational log and the CLI print it with spaces
// or dashes in place of the underscores.
[[nodiscard]] constexpr std::string_view finish_reason_name(FinishReason reason) noexcept {
    switch (reason) {
    case FinishReason::None:
        return "none";
    case FinishReason::OutputLimit:
        return "output_limit";
    case FinishReason::ContextCapacity:
        return "context_capacity";
    case FinishReason::StopToken:
        return "stop_token";
    case FinishReason::StopString:
        return "stop_string";
    case FinishReason::Cancelled:
        return "cancelled";
    case FinishReason::Repetition:
        return "repetition";
    }
    return "unknown";
}

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputDelta, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

// Cumulative prompt frontier published only after the corresponding Program work has completed.
// The Engine owns the request-relative clock and accounting; protocol adapters choose names and
// units for the wire representation.
struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

// Cumulative timing snapshot at one stable output-commit boundary. Generated tokens count model
// and Engine-injected tokens accepted into the sequence, independently of whether the Frontend has
// enough visible bytes to publish an OutputDelta for that boundary.
struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputDelta delta)                 = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Observation affects only request publication. It never changes model execution, output
// semantics, scheduling, or cache selection. Live observations require a Streaming consumer;
// phase timings may also be retained for an Aggregate terminal response.
struct GenerationObservationOptions {
    bool phase_timings   = false;
    bool live_timings    = false;
    bool prompt_progress = false;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    // Request wall phases at committed model-state boundaries. Prompt begins when admission and
    // its exact reuse choice are published and ends at the first accepted output token. Generation
    // spans the first through last accepted output token and therefore has N-1 token intervals.
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                   = 0.0;
    double engine_boundary_exposed_seconds      = 0.0;
    double program_submit_exposed_seconds       = 0.0;
    double program_post_exposed_seconds         = 0.0;
    double engine_commit_output_exposed_seconds = 0.0;
    double engine_maintenance_exposed_seconds   = 0.0;
    double device_wait_exposed_seconds          = 0.0;
    double decode_host_exposed_seconds          = 0.0;
    double decode_device_wait_exposed_seconds   = 0.0;
    std::uint64_t prefill_units                 = 0;
    std::uint64_t decode_rounds                 = 0;
    std::uint64_t control_units                 = 0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0; // the request's draft depth
    // Rounds this request verified in a launch wider than its depth (a batch with a deeper
    // request). The row then drafts at the launch's width (its columns are verified anyway), so
    // accepted_per_position may grow past draft_window.
    std::uint64_t wide_rounds     = 0;
    // Rounds whose DFlash drafter computed only the anchor and its drafts, not its whole block
    // (DFlashDraftWidth::Drafts, SpeculativeOptions::draft_width).
    std::uint64_t narrow_rounds   = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
    // Rounds this request verified below its depth because the step's load bound capped it
    // (SpeculativeOptions::load); they do not feed its watchdog.
    std::uint64_t capped_rounds = 0;
    // Rounds this request verified a copy proposal in (SpeculativeOptions::ngram_draft_tokens), and
    // the drafts they accepted; both are also counted in rounds and accepted_tokens.
    std::uint64_t copy_rounds          = 0;
    std::uint64_t copy_accepted_tokens = 0;
    // Rounds this request verified in a launch run without its CUDA graph: a DFlash window the
    // load bounds leave out at the launch's batch, which a row still appending the features of a
    // wider round from before its step grew needs (SpeculativeOptions::load).
    std::uint64_t ungraphed_rounds = 0;
    // Tokens this request decoded plainly in a speculative engine, one per step (after its
    // watchdog tripped and no row of a step drafted, or after a load bound of 0 turned it plain:
    // load_plain). A request that
    // ran to its limit generated 1 + rounds + accepted_tokens + fallback_steps + plain_steps
    // tokens.
    std::uint64_t plain_steps = 0;
    bool load_plain           = false;
    // Its prompt held more tokens than the context bound (SpeculativeOptions::context_limit):
    // a request that would have drafted decoded plainly from its first token.
    bool context_plain = false;
    // The acceptance watchdog's verdict (SpeculativeOptions::acceptance_floor): the round it
    // tripped at and the rolling tokens per round it saw. From then on the request drafts no
    // more (fallback_steps beside drafting rows, else plain_steps).
    bool watchdog_tripped            = false;
    std::uint64_t watchdog_round     = 0;
    double watchdog_tokens_per_round = 0.0;
    // Verdicts below the floor that kept the request drafting because another row of its step
    // kept drafting (the step's speculative launch runs anyway) while it still emitted at least
    // 1.4 tokens per round; the watchdog judged again after the next round.
    std::uint64_t shared_trip_deferrals = 0;
    // Trips that moved the request to a shallower depth (SpeculativeOptions::step_down) instead of
    // plain decode, and the depth it speculated at last (draft_window when it never stepped).
    std::uint32_t step_downs        = 0;
    std::uint32_t final_draft_window = 0;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    // Model-origin tokens accepted while thinking remained open, including the token that closes
    // it; counted for every response that starts in thinking, with or without a budget.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
    // The suffix was injected because the thinking repeated itself
    // (StopPolicy::repetition_concludes_thinking), not because the budget ran out.
    bool loop_concluded = false;
};

enum class PrefixReusePath : std::uint8_t {
    Root,
    PrivateEndpoint,
    PrivateTurnClosure,
    PrivateResponseReplay,
    PrivateLongAnchor,
    SharedStablePrefix,
};

// Why bounded pressure planning stopped for the materialization decision committed to one request.
enum class MaterializationStopReason : std::uint8_t {
    NoPressure,
    QueueExhausted,
    TargetBudget,
    ExpansionCapacity,
    PlanningBudget,
    InsufficientExpectedGain,
    WorkBudget,
};

[[nodiscard]] inline constexpr const char*
materialization_stop_reason_name(MaterializationStopReason reason) noexcept {
    switch (reason) {
    case MaterializationStopReason::NoPressure:
        return "no_pressure";
    case MaterializationStopReason::QueueExhausted:
        return "queue_exhausted";
    case MaterializationStopReason::TargetBudget:
        return "target_budget";
    case MaterializationStopReason::ExpansionCapacity:
        return "expansion_capacity";
    case MaterializationStopReason::PlanningBudget:
        return "planning_budget";
    case MaterializationStopReason::InsufficientExpectedGain:
        return "insufficient_expected_gain";
    case MaterializationStopReason::WorkBudget:
        return "work_budget";
    }
    return "no_pressure";
}

enum class MaterializationSearchPhase : std::uint8_t {
    None,
    Setup,
    Construction,
    Assessment,
    Expansion,
    Refinement,
};

[[nodiscard]] inline constexpr const char*
materialization_search_phase_name(MaterializationSearchPhase phase) noexcept {
    switch (phase) {
    case MaterializationSearchPhase::None:
        return "none";
    case MaterializationSearchPhase::Setup:
        return "setup";
    case MaterializationSearchPhase::Construction:
        return "construction";
    case MaterializationSearchPhase::Assessment:
        return "assessment";
    case MaterializationSearchPhase::Expansion:
        return "expansion";
    case MaterializationSearchPhase::Refinement:
        return "refinement";
    }
    return "none";
}

struct MaterializationDiagnostics {
    std::uint64_t predicted_now_ns           = 0;
    std::uint64_t predicted_future_loss_ns   = 0;
    std::uint64_t predicted_total_ns         = 0;
    std::uint32_t targets_evaluated          = 0;
    std::uint64_t projection_work            = 0;
    std::uint64_t planning_elapsed_ns        = 0;
    std::uint64_t search_elapsed_ns          = 0;
    MaterializationStopReason stop_reason    = MaterializationStopReason::NoPressure;
    bool budget_exhausted                    = false;
    std::uint32_t selected_degradation_units = 0;
    bool selected_maximal_fallback           = false;

    std::uint64_t initial_predicted_total_ns = 0;
    std::optional<std::uint64_t> first_improvement_ns;
    std::uint32_t incumbent_improvements         = 0;
    std::uint64_t search_work                    = 0;
    std::uint64_t search_granted_units           = 0;
    std::uint32_t search_renewals                = 0;
    bool search_discovery_used                   = false;
    MaterializationSearchPhase search_stop_phase = MaterializationSearchPhase::None;
    bool search_boundary_limited                 = false;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationDiagnostics&,
               const MaterializationDiagnostics&) noexcept = default;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;
    MaterializationDiagnostics materialization;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    SpeculativeStats speculative;
    ThinkingBudgetStats thinking;
    OutputConstraintStats constraint;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

// Logical regions within the one physical workspace allocation. These byte values describe
// layout and live extents and must not be added to workspace.capacity_bytes.
struct VisionWorkspaceMemorySummary {
    std::uint32_t aggregate_prompt_tokens = 0;
    std::uint32_t max_item_tokens         = 0;
    std::size_t general_capacity_bytes    = 0;
    std::size_t encode_peak_bytes         = 0;
    std::size_t handoff_offset_bytes      = 0;
    std::size_t handoff_capacity_bytes    = 0;
    std::size_t handoff_active_bytes      = 0;
    std::size_t handoff_peak_bytes        = 0;
};

// The longest automatic context a lower max_concurrency would fit (MemorySummary::
// lower_concurrency_fits): what an operator can trade concurrency for when the asked one fits
// only a context below the automatic floor.
struct ConcurrencyFit {
    std::uint32_t max_concurrency = 0;
    std::uint32_t max_context     = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    // An automatic max_context (EngineOptions::max_context_automatic) and the model's own position
    // capacity it started from: max_context below it is what the free memory allowed (or the cap).
    bool max_context_automatic                = false;
    std::uint32_t max_context_native          = 0;
    // The prefill chunk the Program runs (EngineOptions::prefill_chunk_automatic may narrow it).
    std::uint32_t prefill_chunk               = 0;
    // With max_concurrency above 1 and an automatic context below its floor (8192, or the model's
    // own when smaller): for each lower concurrency that fits more, the longest context it fits.
    std::vector<ConcurrencyFit> lower_concurrency_fits;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::optional<VisionWorkspaceMemorySummary> vision_workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    // The speculative graphs each launch window takes of that allowance: its draft depth and the
    // speculative batches it is captured for (a depth only a load bound asks runs from the bound's
    // lanes; no window runs from a bound of 0's lanes on).
    struct SpeculativeGraphWindow {
        std::uint32_t draft_tokens    = 0;
        std::uint32_t first_batch     = 0;
        std::uint32_t last_batch      = 0;
        std::size_t allowance_bytes   = 0;
    };
    std::vector<SpeculativeGraphWindow> speculative_graph_windows;
    std::size_t kv_payload_bytes                  = 0;
    // Cached Device StateImages of the context cache beyond the active ones; automatic when the
    // count was not configured (then chosen from [1, default] by the free memory).
    std::uint32_t cached_device_states            = 0;
    std::uint32_t cached_device_states_default    = 0;
    bool cached_device_states_automatic           = false;
    std::uint32_t host_state_capacity_slots       = 0;
    std::uint32_t host_state_occupied_slots       = 0;
    std::size_t host_kv_capacity_bytes            = 0;
    // An automatic Host tier (ContextCacheOptions::host_tier_automatic): the defaults it started
    // from, the pinning budget it was fitted to, and whether it had to be scaled down.
    bool host_tier_automatic                      = false;
    bool host_tier_reduced                        = false;
    std::uint32_t host_state_slots_default        = 0;
    std::size_t host_kv_capacity_default_bytes    = 0;
    std::size_t host_tier_budget_bytes            = 0;
    std::size_t host_kv_occupied_bytes            = 0;
    // The kernel warm-up construction ran before it measured the free memory for an automatic fit
    // (the plan's kernels load their code and grow the per-thread local memory for good at their
    // first launch): the prefill widths it ran (1..N), the probe Programs that captured every
    // decode graph class, its time, the free Device memory it took for good, and the CUDA stack
    // limit (cudaLimitStackSize, per resident thread) before and after it, with the warm-up step
    // that last raised it (empty when none did).
    bool kernel_warm_up                           = false;
    std::uint32_t kernel_warm_up_prefill_widths   = 0;
    // The widths the warm-up ran (one per width class of 1..prefill_widths), ascending.
    std::vector<std::uint32_t> kernel_warm_up_widths_run;
    std::uint32_t kernel_warm_up_graph_passes     = 0;
    double kernel_warm_up_seconds                 = 0.0;
    std::size_t kernel_warm_up_lasting_bytes      = 0;
    std::size_t stack_limit_before_bytes          = 0;
    std::size_t stack_limit_bytes                 = 0;
    std::string stack_limit_source;
    // Device memory pinning the context cache's Host tier takes (the driver's page tables for the
    // pinned bytes), measured by the warm-up and reserved in the fit.
    std::size_t host_pin_device_bytes             = 0;
    std::size_t graph_returned_bytes              = 0;
};

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// device_wait_ns is blocked wall time and is intentionally excluded from their sum. Detail values
// are subsets of a top-level phase and must not be added to Host-active time again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns      = 0;
    std::uint64_t program_submit_ns       = 0;
    std::uint64_t program_post_ns         = 0;
    std::uint64_t engine_commit_output_ns = 0;
    std::uint64_t engine_maintenance_ns   = 0;
    std::uint64_t device_wait_ns          = 0;

    std::uint64_t decode_host_ns         = 0;
    std::uint64_t decode_device_wait_ns  = 0;
    std::uint64_t prefill_host_ns        = 0;
    std::uint64_t prefill_device_wait_ns = 0;
    std::uint64_t control_host_ns        = 0;
    std::uint64_t control_device_wait_ns = 0;
    std::uint64_t prefill_units          = 0;
    std::uint64_t control_units          = 0;

    std::uint64_t admission_policy_ns           = 0;
    std::uint64_t context_progress_ns           = 0;
    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t admission_policy_invocations  = 0;
    std::uint64_t context_progress_invocations  = 0;
    std::uint64_t stats_publication_invocations = 0;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    RuntimeHostWorkStats host_work;
    // Actual prompt tokens evaluated by prefill; reused checkpoint-prefix tokens are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t capture_pending_requests  = 0;
    std::uint32_t terminal_pending_requests = 0;
    std::uint64_t active_captures_completed = 0;
    std::uint64_t active_captures_aborted   = 0;

    std::uint64_t root_selections                    = 0;
    std::uint64_t private_endpoint_selections        = 0;
    std::uint64_t private_turn_closure_selections    = 0;
    std::uint64_t private_response_replay_selections = 0;
    std::uint64_t private_long_anchor_selections     = 0;
    std::uint64_t shared_stable_prefix_selections    = 0;
    std::uint64_t reused_prompt_tokens               = 0;
    std::uint32_t last_selected_frontier_tokens      = 0;

    std::uint64_t state_moves     = 0;
    std::uint64_t state_forks     = 0;
    std::uint64_t state_restores  = 0;
    std::uint64_t state_d2h_count = 0;
    std::uint64_t state_h2d_count = 0;
    std::uint64_t state_d2d_count = 0;
    std::uint64_t state_d2h_bytes = 0;
    std::uint64_t state_h2d_bytes = 0;
    std::uint64_t state_d2d_bytes = 0;
    double state_d2h_seconds      = 0.0;
    double state_h2d_seconds      = 0.0;
    double state_d2d_seconds      = 0.0;

    std::uint64_t main_kv_d2h_pages    = 0;
    std::uint64_t main_kv_h2d_pages    = 0;
    std::uint64_t main_kv_d2d_pages    = 0;
    std::uint64_t main_kv_d2h_bytes    = 0;
    std::uint64_t main_kv_h2d_bytes    = 0;
    std::uint64_t main_kv_d2d_bytes    = 0;
    double main_kv_d2h_seconds         = 0.0;
    double main_kv_h2d_seconds         = 0.0;
    double main_kv_d2d_seconds         = 0.0;
    std::uint64_t backend_kv_d2h_pages = 0;
    std::uint64_t backend_kv_h2d_pages = 0;
    std::uint64_t backend_kv_d2d_pages = 0;
    std::uint64_t backend_kv_d2h_bytes = 0;
    std::uint64_t backend_kv_h2d_bytes = 0;
    std::uint64_t backend_kv_d2d_bytes = 0;
    double backend_kv_d2h_seconds      = 0.0;
    double backend_kv_h2d_seconds      = 0.0;
    double backend_kv_d2d_seconds      = 0.0;

    std::uint64_t pressure_spill_pages                 = 0;
    std::uint64_t partial_tail_cow_pages               = 0;
    std::uint32_t device_state_occupied_slots          = 0;
    std::uint32_t host_state_occupied_slots            = 0;
    std::uint32_t device_main_kv_occupied_pages        = 0;
    std::uint32_t device_backend_kv_occupied_pages     = 0;
    std::size_t host_kv_occupied_bytes                 = 0;
    std::uint64_t pressure_private_owners_degraded     = 0;
    std::uint64_t pressure_private_owners_evicted      = 0;
    std::uint64_t pressure_shared_owners_degraded      = 0;
    std::uint64_t pressure_shared_owners_evicted       = 0;
    std::uint64_t pressure_checkpoints_dropped         = 0;
    std::uint64_t pressure_searches                    = 0;
    std::uint64_t pressure_search_budget_exhaustions   = 0;
    std::uint64_t pressure_maximal_fallback_selections = 0;
    std::uint32_t shared_active_references             = 0;
    std::uint64_t historical_fork_hits                 = 0;
    double actual_context_transfer_seconds             = 0.0;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    External,
    // A measured machine of the same architecture, the nearest by SM count, its prefill scaled by
    // the SM ratio (ContextCostSummary::similar_to names it).
    SimilarMachine,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::External:
        return "external";
    case ContextCostPresetSource::SimilarMachine:
        return "similar-machine";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string prefill_signature;
    std::filesystem::path preset_path;
    std::string similar_to; // the machine a SimilarMachine source stands in with ("" otherwise)
};

// Static facts about the loaded model, independent of the current request context and memory
// layout. The Engine derives the model identity (model_id) and the dimension facts
// (vocab_size, embedding_size, native_context) from the loaded model, and measures the
// parameters, weight bytes, and weights profile from the artifact's tensor inventory. Serving
// renders these into the OpenAI-compatible /v1/models model object and its llama.cpp-compatible
// `meta` field.
struct ModelMetadata {
    std::string model_id;             // Artifact model name (directory metadata "name").
    std::string weights_id;           // Encoded formats of the artifact tensors (meta ftype).
    std::uint64_t vocab_size     = 0; // Tokenizer token domain (meta n_vocab).
    std::uint64_t embedding_size = 0; // Model embedding width (meta n_embd).
    std::uint64_t native_context = 0; // Model native/training context (meta n_ctx_train).
    std::uint64_t parameters     = 0; // Total logical weight elements (meta n_params).
    std::uint64_t weight_bytes   = 0; // Encoded weight payload bytes (meta size).
    // The activation widths the checkpoint's publisher validated its kept matrices with (bits ->
    // matrices, 0: named nowhere; empty: not said) and how many the fast mode runs below that width.
    std::map<int, std::uint64_t> published_activations;
    std::uint64_t below_published_activations      = 0; // under --activations fast
    std::uint64_t below_published_activations_int8 = 0; // under --activations int8
    // The decision fingerprint (LoadSummary::decision_fingerprint) and its plan manifest.
    std::string decision_fingerprint;
    std::string decision_manifest;
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::string cuda_sync_mode;
    std::vector<std::string> weight_formats;
    std::string prefill_signature;
    std::string gpu_class;   // the device class the route data is keyed by, e.g. "gb202:170sm"
    std::string route_table; // what the per-GPU-class route table is: loaded, absent, or refused
    // The device facts guessed or unreported, with the conservative choice taken (device_fact_notes).
    std::vector<std::string> device_facts;
    // The load-time route tuners: one line per tuned MoE layer shape (ops::tune_moe) and per dense
    // problem the generic path's tuner timed during the warm-up.
    std::vector<std::string> moe_tune;
    // A speculative drafter's load notes (the dense decoder's DFlash feature projection: split by
    // tap, or whole and why); empty without one.
    std::string drafter;
    // Steps an ordinary decode round runs, and the machine fact that default follows: the
    // process's CPU bandwidth quota (0: none). Both enter the decision fingerprint.
    std::uint32_t decode_steps_per_round = 1;
    std::uint64_t cpu_quota_us           = 0;
    std::uint64_t cpu_quota_period_us    = 0;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t device_object_count    = 0;
    std::size_t host_object_count      = 0;
    ContextCostSummary context_cost;
    ModelMetadata metadata;
    // The decision fingerprint (docs/maintainer/deterministic-decisions.md, Fingerprint):
    // "nf1-" + 16 hex digits of SHA-256 over decision_canonical, the canonical text of every input
    // the engine's decisions read (device facts, build, requested options, source files, decision
    // environment variables, the resolved plan, taint); decision_manifest, "nm1-" + 16 hex digits,
    // covers the resolved plan and the observed sizes (decision_observed). decision_taint lists
    // the pillar violations present ("<what>=<detail>"), each also in the canonical text. Both are
    // "unreadable" when an input could not be read; the start goes on.
    std::string decision_fingerprint;
    std::string decision_manifest;
    std::string decision_canonical;
    std::vector<std::string> decision_taint;
    // The sizes startup took from the free memory (the "observed" section): in the manifest, not in
    // the fingerprint, until plan levels exist.
    std::string decision_observed;
    // The artifact's id (16 bytes, lowercase hex), as the load read it.
    std::string artifact_id;
};

} // namespace ninferno
