#pragma once

// The plan without the weights (ninferno-serve --plan --no-load, ninferno-ui /api/plan, the quant a
// hub GGUF repository is fetched at): the startup plan ladder of plan_ladder.h, built and selected
// by the same functions a start runs (plan_axes, plan_ladder_inputs, plan_margins, plan_ladder,
// prune_plan_ladder, select_plan_rung), from facts only:
//
//   the model   PlanModelFacts: the shape a GGUF header, a config.json or an artifact's text
//               config names (attention, sliding and recurrent layers, the widths a prefill chunk
//               holds), the family's model_type and the weight bytes (the artifact's tensors, a
//               GGUF's tensor table, else the listed file sizes: WeightBasis says which);
//   the device  PlanDeviceFacts: the VRAM class, the compute capability (the automatic KV codec)
//               and the discrete facts the class reserve reads (WDDM, a display, a GPU sharing the
//               host's memory), from the CUDA device or from a named board of core/data/
//               gpu_facts.json (a what-if);
//   the options EngineOptions, as the server builds them.
//
// Each rung's bytes are a start's where the facts name the model's state whole
// (PlanModelFacts::offline_bytes, estimate_basis "family_planner"): the family planner's own byte
// functions (models/qwen3_5/program/planning/sequence_bytes.h) lay out the same persistent state
// (the paged KV of the full layers and of the MTP layer, the recurrent states and their replay
// records, the sliding rings, a DFlash drafter's KV and ring, the Program's buffers) and plan the
// same CUDA graph allowance a start plans for the same options. What a start measures and the dry
// run cannot: the workspace (its Op scratch the bound weights' routes size: plan_workspace_estimate
// here), and the free memory after the weights (and after the load-time route tuner has freed its
// buffers: whatever the tuner leaves resident, the documented exception) where the dry run takes
// either the free memory now less the weights or an idle card's (the VRAM less a declared driver
// and desktop share). Facts that do not name the state (estimate_basis "approximate", and why)
// take plan_rung_estimate's bytes. The started plan is the authority; the dry run says "estimate".
// Pure: no CUDA, no files, no clock.

#include "models/qwen3_5/program/planning/state_shape.h"
#include "ninferno/types.h"
#include "runtime/engine/plan_ladder.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::runtime {

// ---- The seams a start and the dry run share (construct_or_plan reads them too).

// What the ladder of a start is built from: the artifact's context, the options, and the device's
// discrete facts. Nothing measured.
struct PlanAxes {
    std::uint32_t context_ceiling    = 0; // N: the asked context, or the model's own under the caps
    bool context_automatic           = true;
    std::uint32_t concurrency        = 1;
    bool kv_automatic                = true;
    std::uint32_t kv_explicit_tokens = 0;       // an explicit --kv-capacity (kv_automatic false)
    std::uint64_t headroom_bytes     = 0;       // an automatic pool's base margin
    std::optional<std::uint32_t> states_pinned; // --device-state-slots (0 with the cache off)
    std::uint32_t states_minimum = 1;           // 0 when a Host tier keeps StateImages
    std::uint32_t prefill_chunk  = 1024;
    bool chunk_automatic         = true;
    std::uint64_t class_reserve  = 0; // plan_class_reserve of the device
    bool speculation             = false;
    std::uint32_t round_tokens   = 1; // the widest decode round: draft window + 1
};

// The widest decode round a request asks for, the axes' round_tokens: the deepest launch window
// (the draft depth and every per-request one) and its target token. A start's warm-up and the
// ladder's floor read it; an MTP copy window is the family planner's (sequence_planning_inputs)
// and not part of it.
[[nodiscard]] std::uint32_t plan_round_tokens(const SpeculativeOptions& speculative) noexcept;

// Whether a start's plan has an automatic axis (the context, the chunk, the KV pool, or the cached
// states), so it measures the kernel warm-up and resets the CUDA stack limit before it reads the
// free memory (model_instance.cpp, automatic_fit).
[[nodiscard]] bool plan_automatic_fit(const EngineOptions& options) noexcept;

// The context ceiling of a start: an explicit --max-context, else the model's own, under an
// automatic context's cap (a nonzero max_context with max_context_automatic) and an explicit KV
// capacity (which bounds one sequence).
[[nodiscard]] std::uint32_t plan_context_ceiling(const EngineOptions& options,
                                                 std::uint32_t native_context);

// Whether a Host tier keeps StateImages: what the options and the platform configure (an
// automatic tier's default slots, none on a GPU sharing the host's memory), never what the host can
// pin now.
[[nodiscard]] bool plan_host_states_configured(const ContextCacheOptions& cache,
                                               std::uint32_t default_host_state_slots,
                                               bool integrated);

// The axes of `options` (normalized: normalize_engine_options) for a model of `native_context`.
[[nodiscard]] PlanAxes plan_axes(const EngineOptions& options, std::uint32_t native_context,
                                 bool host_states_configured, std::uint64_t class_reserve);

// The ladder inputs of a concurrency: `pins` false gives an advice ladder (every sizing axis
// automatic but the chunk, which `chunk_pinned` keeps).
[[nodiscard]] PlanLadderInputs plan_ladder_inputs(const PlanAxes& axes, std::uint32_t concurrency,
                                                  bool pins, bool chunk_pinned);

// What the selection keeps beside a rung of the served ladder, and of an advice ladder.
[[nodiscard]] PlanMargins plan_margins(const PlanAxes& axes);
[[nodiscard]] PlanMargins plan_advice_margins(const PlanAxes& axes);

// The longest context a concurrency's advice ladder (every sizing axis automatic; the chunk kept
// when `chunk_kept`) fits, nullopt: none.
using PlanLongestContextFn =
    std::function<std::optional<std::uint32_t>(std::uint32_t concurrency, bool chunk_kept)>;

// Each concurrency below `concurrency` that fits a longer context than `served_context` (nullopt:
// nothing fits at the served one), the highest first, until one fits the `ceiling`.
[[nodiscard]] std::vector<ConcurrencyFit>
plan_lower_concurrency_fits(std::uint32_t concurrency, std::optional<std::uint32_t> served_context,
                            std::uint32_t ceiling, bool chunk_kept,
                            const PlanLongestContextFn& longest);

// What fits instead when no rung of the served ladder does (a start's refusal advice, the dry
// run's fits_instead): the longest context at the asked `concurrency` when the context, the pool or
// the states were asked for (`pinned`), then each lower concurrency that fits longer
// (plan_lower_concurrency_fits); with an explicit --prefill-chunk (`chunk_pinned`) nothing fits
// with, what the narrower chunks fit (`narrowed`).
struct PlanFitsInstead {
    std::vector<ConcurrencyFit> fits;
    bool narrowed = false;
};

[[nodiscard]] PlanFitsInstead plan_fits_instead(std::uint32_t concurrency, std::uint32_t ceiling,
                                                bool pinned, bool chunk_pinned,
                                                const PlanLongestContextFn& longest);

// The family's automatic storages, by model_type: the KV codec and the recurrent state under
// --activations fast. One table for the Engine (family_registry.h automatic_kv_cache,
// automatic_recurrent_state), the plan without the weights and the configuration choice.
struct PlanFamilyStorage {
    KvCacheStorage kv               = KvCacheStorage::BFloat16;
    RecurrentStateStorage recurrent = RecurrentStateStorage::FP32;
};

// The family's own, whatever the GPU. nullopt: no family declares the model_type.
[[nodiscard]] std::optional<PlanFamilyStorage> family_storage(std::string_view model_type);

// A family's KV codec on a GPU of `compute_capability` (major * 10 + minor): FP8 rows fall back to
// BF16 where the chip has no E4M3 instructions (the cvt the codec decodes with; DeviceFeatures::
// mma_fp8 marks the same sm_89 instruction set, a fact of the chip).
[[nodiscard]] KvCacheStorage kv_storage_on(KvCacheStorage family_kv, int compute_capability);

// family_storage with its KV codec on a GPU of `compute_capability` (kv_storage_on).
[[nodiscard]] std::optional<PlanFamilyStorage> plan_family_storage(std::string_view model_type,
                                                                   int compute_capability);

// ---- Facts.

enum class WeightBasis : std::uint8_t {
    Artifact,    // the converted artifact's tensor bytes
    GgufTensors, // a GGUF's tensor table (every tensor's type and shape read from its header)
    FileSize,    // the listed file sizes ("estimated from file size": header and padding counted)
    Unknown,
};

[[nodiscard]] std::string_view weight_basis_name(WeightBasis basis) noexcept;

// The GEMM an MoE's routed banks prefill on (product/plan expert_prefill_of: the banks' formats and
// their stored layout, as ops/moe w4a8_banks_admit decides it). Unknown: dense, or not known.
enum class ExpertPrefill : std::uint8_t {
    Unknown,
    Int8,  // the grouped W4A8 reads every bank: int8 activations on the int8 MMA (dp4a without one)
    A16,   // a bank it does not read (a codebook i-quant in fragment order): the 16-bit register MMA
    Mxfp4, // MXFP4 banks: int8 as Int8, or the block-scaled MMA where the build has it (sm_120a)
};

// One linear a token runs in prefill (DECISIONS 66): N output rows, K deep, `per_token` times (its
// layers; a routed expert's, its picks a token across the layers).
struct PlanLinear {
    double n         = 0;
    double k         = 0;
    double per_token = 0;
    bool expert      = false; // a routed expert's bank (its grid counts jobs, not tokens)
    bool gated       = false; // a gated FFN's gate and up: two matrices of n / 2 rows, one input
};

struct PlanModelFacts {
    std::string model_type;   // the family (qwen3_5_text, gemma4_text, dense_decoder, ...); ""
    std::string architecture; // as the source names it (GGUF general.architecture, HF class)
    std::uint32_t native_context = 0;
    // Full-attention layers (their KV grows with the context): count, KV heads, key and value
    // widths. A model whose layers differ in KV heads counts the sum in full_kv_heads_total.
    std::uint32_t full_layers         = 0;
    std::uint32_t full_kv_heads_total = 0; // summed over the full layers
    std::uint32_t full_key_dim        = 0;
    std::uint32_t full_value_dim      = 0;
    // Sliding-window layers (a ring of window + chunk - 1 rows per sequence).
    std::uint32_t sliding_layers         = 0;
    std::uint32_t sliding_kv_heads_total = 0;
    std::uint32_t sliding_key_dim        = 0;
    std::uint32_t sliding_value_dim      = 0;
    std::uint32_t window                 = 0;
    // The layers that attend, their own KV or an earlier layer's (a KV-shared layer reads one
    // before it: Gemma 4 E2B / E4B); the widths above sum over these.
    std::uint32_t full_attending_layers    = 0;
    std::uint32_t sliding_attending_layers = 0;
    // Recurrent layers (Gated DeltaNet, Mamba2): the state matrix elements of one sequence, all
    // layers (stored in the --state-dtype), and the convolution window bytes (BF16).
    std::uint32_t recurrent_layers         = 0;
    std::uint64_t recurrent_state_elements = 0;
    std::uint64_t recurrent_conv_bytes     = 0;
    // The widths one prefill row holds: hidden, the widest feed-forward a token runs, the vocab.
    std::uint32_t hidden     = 0;
    std::uint32_t ffn_active = 0;
    std::uint32_t vocab      = 0;
    double params            = 0; // every weight (0: unknown)
    // The weights' elements as a whole GGUF tensor table counts them (0: unknown): the
    // denominator of bits_per_weight, exact where the geometry's parameter count above is an
    // estimate (Gemma 4's runs 18 % over its tensors).
    double weight_elements = 0;
    // `bytes` of weights over the weights: weight_elements where counted, else params; 0 when
    // neither is known. What the quantization rule's floors and the accuracy curve read.
    [[nodiscard]] double bits_per_weight(double bytes) const noexcept {
        const double of = weight_elements > 0 ? weight_elements : params;
        return of > 0 ? bytes * 8 / of : 0;
    }
    // The weights the Device holds (the target's; a separate drafter's apart).
    std::uint64_t weight_bytes  = 0;
    WeightBasis weight_basis    = WeightBasis::Unknown;
    std::uint64_t drafter_bytes = 0; // a separate drafter's weights, when one is planned
    // The share of weight_bytes a start uploads only for its speculation: the MTP layers (a GGUF's
    // NextN blocks, an artifact's mtp component: loaded under MTP only) and the model's own DFlash
    // drafter (an artifact's dflash / dflash2 component: under DFlash only). 0: none or unknown.
    std::uint64_t mtp_bytes         = 0;
    std::uint64_t own_drafter_bytes = 0;
    // The proposal head's (an artifact's proposal/head and proposal/token_ids), loaded only when
    // a run drafts with it (ProposalHead::Optimized; MTP and DFlash promote to it, product::
    // model_draft_head). 0: none or unknown.
    std::uint64_t proposal_bytes = 0;
    // The drafters it serves (a Use on mtp/final_hidden, dflash/final_hidden).
    bool proposal_serves_mtp    = false;
    bool proposal_serves_dflash = false;
    // The output head's stored bytes (a whole GGUF tensor table's output.weight, or the token
    // table it is tied to), 0 when unknown: what a proposal head gathered from it is priced by.
    std::uint64_t head_bytes = 0;
    // An untied model's token embedding table's stored bytes (a GGUF's token_embd.weight beside
    // its output.weight), 0 when tied or unknown: a decode step never reads it whole.
    std::uint64_t embedding_bytes = 0;
    // The parsed GGUF header's token_embd.weight bytes, tied or not (a split part 0's table cannot
    // say; product::complete_split_gguf_head settles it), 0: none in that table.
    std::uint64_t token_table_bytes = 0;
    // The routed experts' stored bytes (a whole GGUF table's *_exps tensors: the trunk's, and the
    // MTP layers' apart), 0 when dense or unknown: a pass reads them at its rows' routed share.
    std::uint64_t routed_expert_bytes = 0, mtp_routed_expert_bytes = 0;
    // The trunk's linears by shape (empty: the geometry did not name them).
    std::vector<PlanLinear> linears;
    // Their prefill GEMM (the trunk's banks; the most of their bytes where layers differ).
    ExpertPrefill expert_prefill = ExpertPrefill::Unknown;
    // Where a whole GGUF's tensor data starts (its tensor table ends just before), 0: not a GGUF.
    std::uint64_t gguf_data_offset = 0;
    // A drafter's full-attention KV per token (DFlash's own layers), 0 when none or unknown: the
    // approximate estimate's (the family planner reads state.drafter).
    std::uint64_t drafter_kv_bytes_per_token = 0;
    // The vision encoder's weights, on the Device under --vision (0: none or unknown).
    std::uint64_t vision_bytes = 0;
    // The persistent state the family's planner lays out (models::StateShape), when the facts name
    // it whole: the rung bytes are then the family planner's (estimate_basis "family_planner").
    // Its drafter: a DFlash drafter's own state (an artifact's dflash component, or a separate
    // drafter's config the caller read); a DFlash plan without one is approximate. offline_bytes
    // false: plan_rung_estimate's bytes, `offline_reason` why.
    bool offline_bytes = false;
    std::string offline_reason;
    models::StateShape state;

    // Whether the shape is enough to plan: a context and some attention.
    [[nodiscard]] bool known() const noexcept {
        return native_context != 0 && full_layers + sliding_layers != 0;
    }
};

struct PlanDeviceFacts {
    std::string name;
    std::string source; // "GPU 0" | "named board" ...
    std::uint64_t total_bytes   = 0;
    int compute_capability      = 0;
    std::uint32_t pci_device_id = 0;     // PCI product id (0: unknown): the speculation profile key
    int sms                     = 0;     // multiprocessors (0: unknown)
    bool paravirtual            = false; // WDDM: native Windows or WSL 2
    bool drives_display         = false;
    bool integrated             = false;
    // The free Device memory now, before any weights (a live device); nullopt: an idle card,
    // plan_idle_free_bytes.
    std::optional<std::uint64_t> free_bytes;
};

// What a CUDA context and the driver take of an idle card before any weights (declared, not
// measured: 448-600 MiB with lazy loading on the lab's cards), and what a desktop the card drives
// holds (Linux 256 MiB; Windows and WSL 2 1.5 GiB: 1.0-1.9 measured on the RTX 5090 desktop).
inline constexpr std::uint64_t kPlanIdleContextBytes        = std::uint64_t{512} << 20;
inline constexpr std::uint64_t kPlanIdleLinuxDesktopBytes   = std::uint64_t{256} << 20;
inline constexpr std::uint64_t kPlanIdleWindowsDesktopBytes = std::uint64_t{1536} << 20;
[[nodiscard]] std::uint64_t plan_idle_free_bytes(const PlanDeviceFacts& device) noexcept;

// A named board's memory as the driver reports it: 97.5 % of the marketed VRAM (an RTX 5090's
// 32 GiB reads 31.36 GiB under Linux, 31.84 under WDDM: the low side, rounded down).
[[nodiscard]] std::uint64_t plan_board_total_bytes(double vram_gib) noexcept;

// ---- The estimate of a rung's bytes.

// Declared, not measured: a CUDA graph set per launch configuration (16 MiB a lane: DFlash at C=8
// takes about 130 MiB a depth set) and the Program's own fixed buffers.
inline constexpr std::uint64_t kPlanGraphBytesPerLane = std::uint64_t{16} << 20;
inline constexpr std::uint64_t kPlanFixedRuntimeBytes = std::uint64_t{64} << 20;

struct PlanRungBytes {
    std::uint64_t kv        = 0; // the pool's pages (the MTP layer's and the drafter's beside)
    std::uint64_t states    = 0; // recurrent states of the active lanes and the cached states
    std::uint64_t rings     = 0; // sliding rings of the same sequences (a drafter's among them)
    std::uint64_t workspace = 0; // the prefill chunk's activations, the decode logits (estimated)
    std::uint64_t graphs    = 0; // the CUDA graph allowance
    // The rest of the persistent layout: page tables, replay records, the drafter's features, the
    // Program's round and sampling buffers, alignment (the approximate estimate: a declared share).
    std::uint64_t fixed = 0;
    // One StateImage as the Host tier pins it (not in total(): the pin's page tables are).
    std::uint64_t host_state_image = 0;

    [[nodiscard]] std::uint64_t total() const noexcept {
        return kv + states + rings + workspace + graphs + fixed;
    }
};

// Bytes per KV token of the full layers at a codec (the paged storage layouts of
// core/paged_kv_storage.h, any head width); a ring's per token at its codec.
[[nodiscard]] std::uint64_t plan_kv_bytes_per_token(const PlanModelFacts& model,
                                                    KvCacheStorage storage);
[[nodiscard]] std::uint64_t plan_ring_bytes_per_token(const PlanModelFacts& model,
                                                      KvCacheStorage storage);
// One sequence's recurrent state at a --state-dtype.
[[nodiscard]] std::uint64_t plan_state_bytes(const PlanModelFacts& model,
                                             RecurrentStateStorage storage);

// The approximate estimate (facts without the state's shape): the full layers' KV pages (and a
// drafter's per token), the recurrent states and rings of the active and cached sequences,
// plan_workspace_estimate, a declared graph allowance and fixed share.
[[nodiscard]] PlanRungBytes plan_rung_estimate(const PlanModelFacts& model, const PlanRung& rung,
                                               const EngineOptions& options,
                                               std::uint32_t concurrency);

// The workspace of a rung (both bases: a start sizes it over the bound weights' routes): about
// twelve hidden-wide BF16 rows a prefill token and three of the widest feed-forward it runs, and
// FP32 logits of every decode column.
[[nodiscard]] std::uint64_t plan_workspace_estimate(const PlanModelFacts& model,
                                                    const PlanRung& rung,
                                                    const EngineOptions& options,
                                                    std::uint32_t concurrency);

// A rung's bytes by the family planner's byte functions (model.offline_bytes): the persistent
// layout and the CUDA graph allowance a start plans for `options` (normalized) at the rung's
// context, cached states, chunk and `concurrency`, its pages clamped to the plan's range as a
// start clamps an automatic pool's (`kv_explicit`: kept), and plan_workspace_estimate. Throws
// std::invalid_argument (or std::overflow_error) for a plan the family cannot lay out.
[[nodiscard]] PlanRungBytes plan_rung_bytes(const PlanModelFacts& model, const PlanRung& rung,
                                            const EngineOptions& options, std::uint32_t concurrency,
                                            bool kv_explicit);

// ---- The dry run.

struct DryRunRung {
    PlanRung rung;
    PlanRungBytes bytes;
    std::uint64_t needed = 0; // free after the weights it needs (every margin; survival: Soft)
};

struct DryRunPlan {
    bool planned        = false; // the facts were enough to plan (else `reason`)
    bool fits           = false; // a rung was selected
    std::size_t rung    = 0;     // 0-based index into `ladder` (when fits)
    PlanReserve reserve = PlanReserve::Kept;
    std::vector<DryRunRung> ladder;
    PlanAxes axes;
    PlanMargins margins;
    KvCacheStorage kv_cache          = KvCacheStorage::BFloat16;
    RecurrentStateStorage recurrent  = RecurrentStateStorage::FP32;
    std::uint64_t total_bytes        = 0;
    std::uint64_t free_bytes         = 0; // before the weights
    bool free_measured               = false;
    // The target's (its MTP layers under MTP only, its own drafter under DFlash only), a separate
    // drafter's, the vision encoder's under --vision.
    std::uint64_t weight_bytes       = 0;
    std::uint64_t free_after_weights = 0;
    // Where the rung bytes come from: "family_planner" (the family planner's byte functions; the
    // workspace estimated) or "approximate" (plan_rung_estimate), and why.
    std::string estimate_basis = "approximate";
    std::string estimate_reason;
    // When nothing fits (or the facts do not plan): why, in plain words; and what fits instead,
    // as a start's refusal advises it (plan_fits_instead: the asked concurrency first when an axis
    // was pinned, then each lower one that fits longer).
    std::string reason;
    // A refusal before the memory (the start refuses the options whatever fits): its code
    // ("activations_not_run"), "" when none. The plan then does not fit and `reason` says why.
    std::string refused_code;
    std::vector<ConcurrencyFit> fits_instead;

    // The rung selected, or null.
    [[nodiscard]] const DryRunRung* chosen() const noexcept {
        return fits && rung < ladder.size() ? &ladder[rung] : nullptr;
    }

    // Whole contexts the selected rung's pool holds (tenths), 0 when none fits.
    [[nodiscard]] double pool_contexts() const noexcept;
    // Whether every lane holds a whole context of its own on the selected rung (the pool did not
    // step below C contexts), not a survival rung.
    [[nodiscard]] bool private_contexts() const noexcept;
};

// The plan of `options` for a model and a device from facts (see the header comment). `options`
// as the server builds them (kv_cache_automatic, recurrent_state_automatic, context-cache defaults
// as GenerationService sets them; normalized here as normalize_engine_options would for the
// family). Never throws for a plan that does not fit; throws std::invalid_argument for options the
// Engine refuses (validate_engine_options' checks the dry run can make).
[[nodiscard]] DryRunPlan dry_run_plan(const PlanModelFacts& model, const PlanDeviceFacts& device,
                                      EngineOptions options);

} // namespace ninferno::runtime
