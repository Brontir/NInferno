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
// What a start measures and the dry run cannot: each rung's bytes come from the family's
// SequencePlanner over the bound model at a start and from plan_rung_estimate here (KV pages,
// recurrent states, sliding rings, the prefill workspace and a declared CUDA graph allowance), and
// the free memory is measured after the weights (and after the load-time route tuner has freed its
// buffers: whatever the tuner leaves resident, the documented exception) where the dry run takes
// either the free memory now less the weights or an idle card's (the VRAM less a declared driver
// and desktop share). The started plan is the authority; the dry run says "estimate". Pure: no
// CUDA, no files, no clock.

#include "ninferno/types.h"
#include "runtime/engine/plan_ladder.h"

#include <cstdint>
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

// The family's automatic storages, by model_type (family_registry.cpp's table; a CPU test checks
// the two agree): the KV codec on a GPU of `compute_capability` and the recurrent state under
// --activations fast. nullopt: no family declares the model_type.
struct PlanFamilyStorage {
    KvCacheStorage kv               = KvCacheStorage::BFloat16;
    RecurrentStateStorage recurrent = RecurrentStateStorage::FP32;
};

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
    // The weights the Device holds (the target's; a separate drafter's apart).
    std::uint64_t weight_bytes  = 0;
    WeightBasis weight_basis    = WeightBasis::Unknown;
    std::uint64_t drafter_bytes = 0; // a separate drafter's weights, when one is planned
    // A drafter's full-attention KV per token (DFlash's own layers), 0 when none or unknown.
    std::uint64_t drafter_kv_bytes_per_token = 0;

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
    std::uint64_t kv        = 0; // the pool's pages (the drafter's KV beside)
    std::uint64_t states    = 0; // recurrent states of the active lanes and the cached states
    std::uint64_t rings     = 0; // sliding rings of the same sequences
    std::uint64_t workspace = 0; // the prefill chunk's activations, the decode logits
    std::uint64_t graphs    = 0; // the CUDA graph allowance
    std::uint64_t fixed     = 0;

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

[[nodiscard]] PlanRungBytes plan_rung_estimate(const PlanModelFacts& model, const PlanRung& rung,
                                               const EngineOptions& options,
                                               std::uint32_t concurrency);

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
    std::uint64_t weight_bytes       = 0; // target and separate drafter
    std::uint64_t free_after_weights = 0;
    // When nothing fits (or the facts do not plan): why, in plain words; and the longest
    // automatic context each concurrency up to the asked one fits (the asked one first).
    std::string reason;
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

// ---- The quant rule (G3): the largest quantization that fits.

struct QuantCandidate {
    std::string label;                  // as --quant takes it
    std::uint64_t bytes    = 0;         // its weights
    int label_bits         = 0;         // from the label (Q4_K_M: 4), 0 when it says none
    double bits_per_weight = 0;         // bytes * 8 / params (0: params unknown)
    bool supported         = true;      // the engine serves it (at most 8 bits; not a drafter)
    std::optional<double> quality_loss; // published KLD against the full-precision model
};

struct QuantFit {
    bool fits   = false; // every lane a whole context of the asked length, not a survival rung
    bool shared = false; // it fits only as one pool the lanes share (fewer than C contexts)
    std::uint32_t context = 0;
};

// The largest quantization: among the supported candidates that fit (private contexts first, then
// those that fit only shared), the most bits per weight (the label's bits when the parameters are
// unknown), within 1 % of the bytes a tie broken by the published quality loss where both have
// one (lower first), then by bytes. nullopt: none fits. `fits[i]` is candidate i's.
[[nodiscard]] std::optional<std::size_t>
largest_fitting_quant(const std::vector<QuantCandidate>& candidates,
                      const std::vector<QuantFit>& fits);

// Whether `a` ranks above `b` by the rule's order (more bits per weight, ties as above): the
// order the quant table lists.
[[nodiscard]] bool quant_ranks_above(const QuantCandidate& a, const QuantCandidate& b) noexcept;

// The fit the rule reads from a dry run: a rung whose pool holds C private contexts (or a shared
// one) of the asked context (any with an automatic one), not a survival rung.
[[nodiscard]] QuantFit quant_fit_of(const DryRunPlan& plan);

} // namespace ninferno::runtime
