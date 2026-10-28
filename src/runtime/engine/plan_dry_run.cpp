#include "runtime/engine/plan_dry_run.h"

#include "core/device_class.h"
#include "models/qwen3_5/program/planning/sequence_bytes.h"
#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace ninferno::runtime {

namespace {

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

// One K or V vector of `width` elements at a codec, with its scales (core/paged_kv_storage.h's
// layouts, for any width: the Engine refuses a codec a head width has no kernel for).
std::uint64_t vector_bytes(KvCacheStorage storage, std::uint64_t width, bool key) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return 2 * width;
    case KvCacheStorage::Int8Group64:
        return width + 2 * ceil_div(width, 64);
    case KvCacheStorage::Fp8E4M3Row256:
        return width + 2;
    case KvCacheStorage::Nvfp4Group16:
        return width / 2 + ceil_div(width, 16);
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return key ? width + 2 : width / 2 + ceil_div(width, 16);
    }
    return 2 * width;
}

// The widest round's logits columns the workspace estimate counts: the deepest draft depth, or an
// MTP copy window (resolved: models::qwen3_5::detail::resolve_copy_window) deeper still, and the
// anchor. Not the axes' round_tokens (plan_round_tokens), which a start's warm-up and the
// ladder's floor read as the draft window + 1.
std::uint64_t workspace_round_columns(const SpeculativeOptions& speculative) {
    std::uint32_t window = speculative.draft_tokens;
    for (const std::uint32_t k : speculative.request_draft_tokens) { window = std::max(window, k); }
    if (speculative.backend == SpeculativeBackend::Mtp &&
        speculative.ngram_draft_tokens != SpeculativeOptions::kCopyWindowAuto) {
        window = std::max(window, speculative.ngram_draft_tokens);
    }
    return window + 1U;
}

std::string gib_text(std::uint64_t bytes) {
    const std::uint64_t hundredths = (bytes * 100ULL + kGiB - 1ULL) >> 30;
    const std::string cents        = std::to_string(hundredths % 100);
    return std::to_string(hundredths / 100) + "." + (cents.size() < 2 ? "0" : "") + cents + " GiB";
}

} // namespace

// ---- Seams.

bool plan_automatic_fit(const EngineOptions& options) noexcept {
    return options.max_context_automatic || options.prefill_chunk_automatic ||
           options.kv_capacity.mode == KvCapacityMode::Automatic ||
           (options.context_cache.enabled && !options.context_cache.device_state_slots.has_value());
}

std::uint32_t plan_round_tokens(const SpeculativeOptions& speculative) noexcept {
    std::uint32_t window = speculative.draft_tokens;
    for (const std::uint32_t k : speculative.request_draft_tokens) { window = std::max(window, k); }
    return window + 1U;
}

std::uint32_t plan_context_ceiling(const EngineOptions& options, std::uint32_t native_context) {
    if (!options.max_context_automatic) { return options.max_context; }
    if (native_context == 0) { throw std::logic_error("the model reports no position capacity"); }
    std::uint32_t ceiling = native_context;
    if (options.max_context != 0) { ceiling = std::min(ceiling, options.max_context); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit) {
        ceiling = std::min(ceiling, options.kv_capacity.explicit_tokens);
    }
    return ceiling;
}

bool plan_host_states_configured(const ContextCacheOptions& cache,
                                 std::uint32_t default_host_state_slots, bool integrated) {
    return cache.enabled &&
           (cache.host_tier_automatic ? default_host_state_slots != 0 && !integrated
                                      : cache.host_state_slots != 0);
}

PlanAxes plan_axes(const EngineOptions& options, std::uint32_t native_context,
                   bool host_states_configured, std::uint64_t class_reserve) {
    PlanAxes axes;
    axes.context_ceiling    = plan_context_ceiling(options, native_context);
    axes.context_automatic  = options.max_context_automatic;
    axes.concurrency        = options.max_concurrency;
    axes.kv_automatic       = options.kv_capacity.mode == KvCapacityMode::Automatic;
    axes.kv_explicit_tokens = axes.kv_automatic ? 0U : options.kv_capacity.explicit_tokens;
    axes.headroom_bytes     = options.kv_capacity.automatic_headroom_bytes;
    axes.states_pinned   = options.context_cache.enabled ? options.context_cache.device_state_slots
                                                         : std::optional<std::uint32_t>(0U);
    axes.states_minimum  = host_states_configured ? 0U : 1U;
    axes.prefill_chunk   = options.prefill_chunk;
    axes.chunk_automatic = options.prefill_chunk_automatic;
    axes.class_reserve   = class_reserve;
    axes.speculation     = options.speculative.backend != SpeculativeBackend::None;
    axes.round_tokens       = plan_round_tokens(options.speculative);
    return axes;
}

PlanMargins plan_margins(const PlanAxes& axes) {
    const bool automatic_fit = axes.context_automatic || axes.chunk_automatic ||
                               axes.kv_automatic || !axes.states_pinned.has_value();
    return PlanMargins{
        .base      = axes.kv_automatic ? axes.headroom_bytes : 0,
        .reserve   = axes.kv_automatic ? axes.class_reserve : 0,
        .allowance = automatic_fit ? kPlanWarmUpAllowanceBytes : 0,
    };
}

PlanMargins plan_advice_margins(const PlanAxes& axes) {
    return PlanMargins{
        .base      = axes.kv_automatic ? axes.headroom_bytes : kDefaultKvCapacityHeadroomBytes,
        .reserve   = axes.class_reserve,
        .allowance = kPlanWarmUpAllowanceBytes,
    };
}

PlanLadderInputs plan_ladder_inputs(const PlanAxes& axes, std::uint32_t concurrency, bool pins,
                                    bool chunk_pinned) {
    const PlanMargins margins = pins ? plan_margins(axes) : plan_advice_margins(axes);
    return PlanLadderInputs{
        .context_ceiling  = axes.context_ceiling,
        .context_pinned   = pins && !axes.context_automatic,
        .concurrency      = concurrency,
        .kv_tokens_pinned = pins && !axes.kv_automatic
                                ? std::optional<std::uint32_t>(axes.kv_explicit_tokens)
                                : std::nullopt,
        .states_pinned    = axes.states_pinned,
        .states_default   = automatic_cached_device_states(concurrency),
        .states_minimum   = axes.states_minimum,
        .prefill_chunk    = axes.prefill_chunk,
        .chunk_pinned     = chunk_pinned,
        // A class reserve beside the pool (a discrete fact: the driver model, the display, an
        // integrated GPU): the survival range then has its rungs without it.
        .class_reserve = margins.reserve > margins.base,
        // The last-resort rung only without speculation: a drafter's plan that fits nothing
        // longer is refused, and ninferno-serve's --spec auto tries the profile's fit fallback,
        // shallower drafts and the MTP head, then plain decode, which hold more.
        .last_resort_context =
            !axes.speculation ? std::max(kAutomaticContextGrain,
                                         (axes.round_tokens + 2U + kAutomaticContextGrain - 1U) /
                                             kAutomaticContextGrain * kAutomaticContextGrain)
                              : 0U,
    };
}

std::vector<ConcurrencyFit> plan_lower_concurrency_fits(std::uint32_t concurrency,
                                                        std::optional<std::uint32_t> served_context,
                                                        std::uint32_t ceiling, bool chunk_kept,
                                                        const PlanLongestContextFn& longest) {
    std::vector<ConcurrencyFit> out;
    for (std::uint32_t lower = concurrency - 1U; lower >= 1U && concurrency >= 1U; --lower) {
        const std::optional<std::uint32_t> fit = longest(lower, chunk_kept);
        if (fit && (!served_context || *fit > *served_context)) {
            out.push_back({.max_concurrency = lower, .max_context = *fit});
            served_context = fit;
        }
        if (served_context && *served_context >= ceiling) { break; }
    }
    return out;
}

PlanFitsInstead plan_fits_instead(std::uint32_t concurrency, std::uint32_t ceiling, bool pinned,
                                  bool chunk_pinned, const PlanLongestContextFn& longest) {
    const auto what_fits = [&](bool chunk_kept) {
        std::vector<ConcurrencyFit> out;
        if (pinned) {
            if (const auto here = longest(concurrency, chunk_kept)) {
                out.push_back({.max_concurrency = concurrency, .max_context = *here});
            }
        }
        if (concurrency > 1) {
            const std::vector<ConcurrencyFit> lower = plan_lower_concurrency_fits(
                concurrency,
                out.empty() ? std::nullopt : std::optional<std::uint32_t>(out.back().max_context),
                ceiling, chunk_kept, longest);
            out.insert(out.end(), lower.begin(), lower.end());
        }
        return out;
    };
    PlanFitsInstead out{.fits = what_fits(chunk_pinned)};
    if (out.fits.empty() && chunk_pinned) {
        // An explicit --prefill-chunk nothing fits with: what the narrower chunks fit.
        out.fits     = what_fits(false);
        out.narrowed = !out.fits.empty();
    }
    return out;
}

namespace {

// The families' automatic storages: the one table (family_registry.cpp's Engine reads it through
// family_storage too). Qwen 3.5 / 3.6 keep INT8 K/V groups and an FP16 recurrent state under
// --activations fast; Gemma 4 keeps its full layers' pool in FP8 rows (the sliding rings stay
// BF16: RTX 5090 long1 26B at 218K decodes 173 -> 221 t/s, 31B at 98K 55 -> 63); the dense
// decoders keep FP8 rows (E4M3 and a scale per row): half the K/V a long decode step reads
// (gpt-oss ~24.6 KB a context token in BF16). Nemotron-H keeps the defaults.
struct FamilyStorageRow {
    std::string_view model_type;
    PlanFamilyStorage storage;
};

constexpr FamilyStorageRow kFamilyStorage[] = {
    {"qwen3_5_text", {KvCacheStorage::Int8Group64, RecurrentStateStorage::FP16}},
    {"qwen3_5_moe_text", {KvCacheStorage::Int8Group64, RecurrentStateStorage::FP16}},
    {"gemma4_text", {KvCacheStorage::Fp8E4M3Row256, RecurrentStateStorage::FP32}},
    {"dense_decoder", {KvCacheStorage::Fp8E4M3Row256, RecurrentStateStorage::FP32}},
    {"moe_decoder", {KvCacheStorage::Fp8E4M3Row256, RecurrentStateStorage::FP32}},
    {"nemotron_h", {KvCacheStorage::BFloat16, RecurrentStateStorage::FP32}},
};

} // namespace

std::optional<PlanFamilyStorage> family_storage(std::string_view model_type) {
    for (const FamilyStorageRow& row : kFamilyStorage) {
        if (row.model_type == model_type) { return row.storage; }
    }
    return std::nullopt;
}

KvCacheStorage kv_storage_on(KvCacheStorage family_kv, int compute_capability) {
    // FP8 rows halve the K/V bytes a long step reads only where the GPU converts E4M3 in hardware
    // (the E4M3 instructions, DeviceFeatures::mma_fp8: sm_89 on); below it the attention widens
    // every code in software and the step slows down (RTX 3090, gpt-oss-20b at 88K: 136 -> 91
    // t/s, short requests 233 -> 208).
    const bool e4m3 = device_features_for(compute_capability / 10, compute_capability % 10).mma_fp8;
    return family_kv == KvCacheStorage::Fp8E4M3Row256 && !e4m3 ? KvCacheStorage::BFloat16
                                                               : family_kv;
}

std::optional<PlanFamilyStorage> plan_family_storage(std::string_view model_type,
                                                     int compute_capability) {
    std::optional<PlanFamilyStorage> out = family_storage(model_type);
    if (out) { out->kv = kv_storage_on(out->kv, compute_capability); }
    return out;
}

// ---- Facts.

std::string_view weight_basis_name(WeightBasis basis) noexcept {
    switch (basis) {
    case WeightBasis::Artifact:
        return "artifact";
    case WeightBasis::GgufTensors:
        return "gguf_tensors";
    case WeightBasis::FileSize:
        return "file_size";
    case WeightBasis::Unknown:
        break;
    }
    return "unknown";
}

std::uint64_t plan_idle_free_bytes(const PlanDeviceFacts& device) noexcept {
    std::uint64_t taken = kPlanIdleContextBytes;
    if (device.drives_display) {
        taken += device.paravirtual ? kPlanIdleWindowsDesktopBytes : kPlanIdleLinuxDesktopBytes;
    }
    return device.total_bytes > taken ? device.total_bytes - taken : 0;
}

std::uint64_t plan_board_total_bytes(double vram_gib) noexcept {
    if (!(vram_gib > 0)) { return 0; }
    const long double bytes = static_cast<long double>(vram_gib) * 0.975L * kGiB;
    // Down to whole MiB.
    return static_cast<std::uint64_t>(bytes) >> 20 << 20;
}

// ---- Estimate.

std::uint64_t plan_kv_bytes_per_token(const PlanModelFacts& model, KvCacheStorage storage) {
    return static_cast<std::uint64_t>(model.full_kv_heads_total) *
           (vector_bytes(storage, model.full_key_dim, true) +
            vector_bytes(storage, model.full_value_dim, false));
}

std::uint64_t plan_ring_bytes_per_token(const PlanModelFacts& model, KvCacheStorage storage) {
    return static_cast<std::uint64_t>(model.sliding_kv_heads_total) *
           (vector_bytes(storage, model.sliding_key_dim, true) +
            vector_bytes(storage, model.sliding_value_dim, false));
}

std::uint64_t plan_state_bytes(const PlanModelFacts& model, RecurrentStateStorage storage) {
    const std::uint64_t element = storage == RecurrentStateStorage::FP32 ? 4 : 2;
    return model.recurrent_state_elements * element + model.recurrent_conv_bytes;
}

PlanRungBytes plan_rung_estimate(const PlanModelFacts& model, const PlanRung& rung,
                                 const EngineOptions& options, std::uint32_t concurrency) {
    PlanRungBytes out;
    const std::uint64_t chunk = std::min(rung.prefill_chunk, rung.max_context);
    out.kv                    = static_cast<std::uint64_t>(rung.kv_pages) * kAutomaticContextGrain *
             (plan_kv_bytes_per_token(model, options.kv_cache) + model.drafter_kv_bytes_per_token);
    const std::uint64_t sequences = static_cast<std::uint64_t>(concurrency) + rung.cached_states;
    out.states                    = sequences * plan_state_bytes(model, options.recurrent_state);
    if (model.sliding_layers != 0 && model.window != 0) {
        const std::uint64_t rows =
            std::min<std::uint64_t>(model.window + chunk - 1, rung.max_context);
        out.rings = sequences * rows * plan_ring_bytes_per_token(model, options.sliding_kv_cache);
    }
    out.workspace = plan_workspace_estimate(model, rung, options, concurrency);
    if (options.use_cuda_graph) {
        std::set<std::uint32_t> depths;
        if (options.speculative.backend != SpeculativeBackend::None) {
            depths.insert(options.speculative.draft_tokens);
            for (const std::uint32_t k : options.speculative.request_draft_tokens) {
                depths.insert(k);
            }
        }
        out.graphs = kPlanGraphBytesPerLane * concurrency * (1 + depths.size());
    }
    out.fixed = kPlanFixedRuntimeBytes;
    out.host_state_image =
        plan_state_bytes(model, options.recurrent_state) +
        (model.sliding_layers != 0 ? static_cast<std::uint64_t>(model.window) *
                                         plan_ring_bytes_per_token(model, options.sliding_kv_cache)
                                   : 0);
    return out;
}

std::uint64_t plan_workspace_estimate(const PlanModelFacts& model, const PlanRung& rung,
                                      const EngineOptions& options, std::uint32_t concurrency) {
    const std::uint64_t chunk = std::min(rung.prefill_chunk, rung.max_context);
    const std::uint64_t round = workspace_round_columns(options.speculative);
    if (model.hidden == 0) { return std::uint64_t{256} << 20; }
    // About twelve hidden-wide BF16 rows a token (norms, residual, attention projections) and
    // three of the widest feed-forward it runs (catalog.h's estimate, checked on the RTX 4090
    // log of Qwen3.6 35B: 79 MiB at chunk 1024), and FP32 logits of every decode column.
    return chunk * (12ULL * model.hidden + 3ULL * model.ffn_active) * 2ULL +
           static_cast<std::uint64_t>(concurrency) * round * model.vocab * 4ULL;
}

PlanRungBytes plan_rung_bytes(const PlanModelFacts& model, const PlanRung& rung,
                              const EngineOptions& options, std::uint32_t concurrency,
                              bool kv_explicit) {
    namespace detail = models::qwen3_5::detail;
    if (!model.offline_bytes) {
        throw std::logic_error("plan_rung_bytes needs the facts of the model's state");
    }
    // The options a start plans this rung with (construct_or_plan, planned_at), its copy window
    // resolved by the family's rule as the start's planner resolves it (sequence_planning_inputs).
    const bool copy_rounds                   = detail::family_copy_rounds(model.state.family);
    EngineOptions planned                    = options;
    planned.max_context                      = rung.max_context;
    planned.max_concurrency                  = concurrency;
    planned.prefill_chunk                    = rung.prefill_chunk;
    planned.context_cache.device_state_slots = rung.cached_states;
    planned.speculative = detail::resolve_copy_window(planned.speculative, copy_rounds);
    const detail::SequencePlanningInputs inputs =
        detail::sequence_planning_inputs(planned, copy_rounds, 0, false);
    const detail::SequencePageRange range =
        detail::sequence_page_range(inputs.capacity, inputs.max_concurrency);
    // An automatic pool's pages are the planner's curve's (a start clamps them the same way); an
    // explicit --kv-capacity outside it is refused.
    const std::uint32_t pages =
        kv_explicit ? rung.kv_pages : std::clamp(rung.kv_pages, range.minimum, range.maximum);
    if (pages < range.minimum || pages > range.maximum) {
        throw std::invalid_argument(
            "kv_capacity is outside the usable range for max_context and max_concurrency");
    }
    const auto plan                       = detail::sequence_plan_shell(inputs, pages);
    const detail::PersistentLayout layout = detail::plan_persistent_layout(model.state, *plan);
    detail::plan_graph_allowance(*plan, false);

    PlanRungBytes out;
    out.kv             = layout.kv_payload_bytes;
    const auto& images = layout.state_images;
    if (images.linear) {
        for (const auto& region : images.linear->conv) { out.states += region.bytes; }
        for (const auto& region : images.linear->recurrent) { out.states += region.bytes; }
    }
    out.states += images.continuation_hidden.region.bytes;
    for (const auto& group : images.rings) {
        for (const auto* planes : {&group.k, &group.v, &group.k_scale, &group.v_scale}) {
            for (const auto& plane : *planes) { out.rings += plane.region.bytes; }
        }
    }
    if (images.dflash_local) {
        for (const auto* planes : {&images.dflash_local->k, &images.dflash_local->v}) {
            for (const auto& plane : *planes) { out.rings += plane.region.bytes; }
        }
    }
    const std::uint64_t held = out.kv + out.states + out.rings;
    if (held > layout.bytes) { throw std::logic_error("the persistent layout's regions overlap"); }
    out.fixed            = layout.bytes - held;
    out.graphs           = plan->graph_allowance_bytes;
    // The workspace's round logits over the resolved window's columns, as the start's are.
    out.workspace        = plan_workspace_estimate(model, rung, planned, concurrency);
    out.host_state_image = images.host.image_bytes;
    return out;
}

// ---- Dry run.

double DryRunPlan::pool_contexts() const noexcept {
    const DryRunRung* c = chosen();
    if (c == nullptr) { return 0; }
    const std::uint32_t pages = plan_context_pages(c->rung.max_context);
    return pages == 0 ? 0 : static_cast<double>(c->rung.kv_pages) / pages;
}

bool DryRunPlan::private_contexts() const noexcept {
    const DryRunRung* c = chosen();
    if (c == nullptr || c->rung.survival) { return false; }
    return static_cast<std::uint64_t>(c->rung.kv_pages) >=
           static_cast<std::uint64_t>(plan_context_pages(c->rung.max_context)) * axes.concurrency;
}

namespace {

// normalize_engine_options' share the plan reads, for the family the facts name.
EngineOptions normalized_for_plan(EngineOptions options, const PlanModelFacts& model,
                                  int compute_capability) {
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_context == 0 && !options.max_context_automatic) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    if (options.prefill_chunk == 0) {
        throw std::invalid_argument("prefill chunk must be nonzero");
    }
    const std::optional<PlanFamilyStorage> family =
        plan_family_storage(model.model_type, compute_capability);
    if (options.recurrent_state_automatic) {
        options.recurrent_state           = options.activations == ActivationMode::Exact || !family
                                                ? RecurrentStateStorage::FP32
                                                : family->recurrent;
        options.recurrent_state_automatic = false;
    }
    if (options.kv_cache_automatic) {
        options.kv_cache           = family ? family->kv : KvCacheStorage::BFloat16;
        options.kv_cache_automatic = false;
    }
    ContextCacheOptions& cache = options.context_cache;
    if (!cache.enabled) {
        cache.device_state_slots     = 0;
        cache.host_state_slots       = 0;
        cache.host_kv_capacity_bytes = 0;
        return options;
    }
    const std::uint32_t concurrency = options.max_concurrency;
    const std::uint32_t private_continuations =
        cache.max_private_continuations.value_or(std::max(2U * concurrency, concurrency + 8U));
    if (cache.host_tier_automatic) {
        cache.host_state_slots = std::max(cache.host_state_slots, 2U * private_continuations);
    }
    return options;
}

} // namespace

DryRunPlan dry_run_plan(const PlanModelFacts& model, const PlanDeviceFacts& device,
                        EngineOptions options) {
    DryRunPlan out;
    out.total_bytes   = device.total_bytes;
    out.free_measured = device.free_bytes.has_value();
    out.free_bytes    = device.free_bytes.value_or(plan_idle_free_bytes(device));
    // The weights a start uploads for these options: the MTP layers only under MTP, the model's
    // own DFlash drafter only under DFlash (the family load reads only the component the
    // speculation names).
    const SpeculativeBackend backend = options.speculative.backend;
    const std::uint64_t unloaded =
        (backend != SpeculativeBackend::Mtp ? model.mtp_bytes : 0) +
        (backend != SpeculativeBackend::DFlash && backend != SpeculativeBackend::DFlash2
             ? model.own_drafter_bytes
             : 0);
    out.weight_bytes = (model.weight_bytes > unloaded ? model.weight_bytes - unloaded : 0) +
                       model.drafter_bytes + (options.enable_vision ? model.vision_bytes : 0);
    out.free_after_weights =
        out.free_bytes > out.weight_bytes ? out.free_bytes - out.weight_bytes : 0;
    if (!model.known()) {
        out.reason = "the model's shape is unknown (no context length or attention geometry in "
                     "its header or config): the plan needs the start";
        return out;
    }
    options       = normalized_for_plan(std::move(options), model, device.compute_capability);
    out.kv_cache  = options.kv_cache;
    out.recurrent = options.recurrent_state;

    // The bytes' basis: the family planner's byte functions when the facts name the state whole
    // (and, for a DFlash plan, its drafter's).
    const bool masked_draft = options.speculative.backend == SpeculativeBackend::DFlash ||
                              options.speculative.backend == SpeculativeBackend::DFlash2;
    bool family_planner = model.offline_bytes;
    if (!family_planner) {
        out.estimate_reason = model.offline_reason.empty()
                                  ? "the facts do not name the model's state"
                                  : model.offline_reason;
    } else if (masked_draft && !model.state.drafter) {
        family_planner = false;
        out.estimate_reason =
            "the DFlash drafter's shape is unknown: its KV and ring are estimated";
    }
    if (family_planner) {
        // What the family refuses before any plan (its storage, its drafter).
        try {
            models::qwen3_5::detail::validate_state_storage(model.state, options);
            if (masked_draft && model.state.family == models::StateFamily::Qwen35 &&
                (options.speculative.backend == SpeculativeBackend::DFlash2) !=
                    model.state.drafter->dflash2) {
                throw std::invalid_argument(
                    "selected masked draft backend is not supported by this target");
            }
        } catch (const std::invalid_argument& refusal) {
            out.reason = refusal.what();
            return out;
        }
        out.estimate_basis  = "family_planner";
        out.estimate_reason = "the family planner's persistent state and graph allowance; the "
                              "workspace estimated (its Op scratch the bound weights' routes size)";
    }

    ContextCacheOptions& cache = options.context_cache;
    // An automatic Host tier on a GPU sharing the host's memory is off (construct_or_plan).
    const std::uint32_t default_host_slots = cache.host_state_slots;
    if (cache.enabled && cache.host_tier_automatic && device.integrated) {
        cache.host_state_slots       = 0;
        cache.host_kv_capacity_bytes = 0;
    }
    const bool host_states =
        plan_host_states_configured(cache, default_host_slots, device.integrated);
    const std::uint64_t class_reserve = plan_class_reserve(
        device.total_bytes, device.paravirtual, device.drives_display, device.integrated);
    out.axes    = plan_axes(options, model.native_context, host_states, class_reserve);
    out.margins = plan_margins(out.axes);

    // The Host tier's pinning (2 MiB of page tables per pinned GiB), from a rung's StateImage as
    // the planner's host_state_image_bytes gives it.
    const auto pin_bytes_of = [&](const PlanRungBytes& b) -> std::uint64_t {
        if (!cache.enabled) { return 0; }
        const std::uint64_t pinned =
            cache.host_kv_capacity_bytes +
            (cache.host_state_slots != 0
                 ? b.host_state_image * static_cast<std::uint64_t>(cache.host_state_slots)
                 : 0ULL);
        return pinned >> 9;
    };
    const auto bytes_of = [&](const PlanRung& rung, std::uint32_t concurrency,
                              bool kv_explicit) -> std::optional<PlanRungBytes> {
        if (!family_planner) { return plan_rung_estimate(model, rung, options, concurrency); }
        return plan_rung_bytes(model, rung, options, concurrency, kv_explicit);
    };

    struct Built {
        std::vector<PlanRung> rungs;
        std::vector<std::optional<PlanRungBytes>> bytes;
    };

    // A ladder of a concurrency (`pins`: the served one), as a start builds it: a shape the family
    // cannot lay out is the user's option at the top rung of the served ladder (refused), and is
    // skipped anywhere else.
    const auto build = [&](std::uint32_t concurrency, bool pins, bool chunk_pinned) {
        const bool kv_explicit = pins && !out.axes.kv_automatic;
        const std::vector<PlanRung> candidates =
            plan_ladder(plan_ladder_inputs(out.axes, concurrency, pins, chunk_pinned));
        const auto bytes_at = [&](const PlanRung& rung, bool top) -> std::optional<PlanRungBytes> {
            try {
                return bytes_of(rung, concurrency, kv_explicit);
            } catch (const std::invalid_argument&) {
                if (top) { throw; }
            } catch (const std::overflow_error&) {
                if (top) { throw; }
            }
            return std::nullopt;
        };
        std::vector<std::optional<PlanRungBytes>> candidate_bytes;
        candidate_bytes.reserve(candidates.size());
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            candidate_bytes.push_back(bytes_at(candidates[i], pins && i == 0));
        }
        Built built;
        built.rungs = prune_plan_ladder(candidates, [&](std::size_t index) {
            return candidate_bytes[index]
                       ? std::optional<std::uint64_t>(candidate_bytes[index]->total())
                       : std::nullopt;
        });
        for (std::size_t i = 0; i < built.rungs.size(); ++i) {
            built.bytes.push_back(bytes_at(built.rungs[i], pins && i == 0));
        }
        return built;
    };
    const auto cost_of = [&](const Built& built) {
        return PlanRungCostFn(
            [&built, &pin_bytes_of](std::size_t index) -> std::optional<PlanRungCost> {
                const std::optional<PlanRungBytes>& b = built.bytes.at(index);
                if (!b) { return std::nullopt; }
                return PlanRungCost{.bytes      = b->total(),
                                    .held_bytes = b->total() - b->graphs,
                                    .pin_bytes  = pin_bytes_of(*b)};
            });
    };

    const bool chunk_pinned = !out.axes.chunk_automatic;
    Built served;
    try {
        served = build(out.axes.concurrency, true, chunk_pinned);
    } catch (const std::exception& refusal) { // the family refuses the asked shape
        out.reason = refusal.what();
        return out;
    }
    const PlanRungCostFn cost = cost_of(served);
    for (std::size_t index = 0; index < served.rungs.size(); ++index) {
        const PlanRung& rung = served.rungs[index];
        const std::optional<PlanRungCost> c = cost(index);
        out.ladder.push_back(DryRunRung{
            .rung   = rung,
            .bytes  = served.bytes[index].value_or(PlanRungBytes{}),
            .needed = c ? plan_needed_bytes(*c, out.margins,
                                            rung.survival ? PlanReserve::Soft : PlanReserve::Kept)
                        : 0});
    }
    out.planned = true;
    const std::optional<PlanSelection> selected =
        select_plan_rung(served.rungs, cost, out.free_after_weights, out.margins);
    if (selected) {
        out.fits    = true;
        out.rung    = selected->rung;
        out.reserve = selected->reserve;
        return out;
    }
    // Nothing fits: why, and what fits instead as a start's refusal advises it.
    const PlanMargins advice = plan_advice_margins(out.axes);
    const bool pinned =
        !out.axes.context_automatic || !out.axes.kv_automatic || out.axes.states_pinned.has_value();
    out.fits_instead =
        plan_fits_instead(
            out.axes.concurrency, out.axes.context_ceiling, pinned, chunk_pinned,
            [&](std::uint32_t concurrency, bool chunk_kept) -> std::optional<std::uint32_t> {
                const Built ladder = build(concurrency, false, chunk_kept);
                const std::optional<PlanSelection> fit =
                    select_plan_rung(ladder.rungs, cost_of(ladder), out.free_after_weights, advice);
                if (!fit) { return std::nullopt; }
                return ladder.rungs[fit->rung].max_context;
            })
            .fits;
    const DryRunRung& last = out.ladder.back();
    if (out.weight_bytes >= out.free_bytes) {
        out.reason = "the weights (" + gib_text(out.weight_bytes) + ") do not fit the " +
                     gib_text(out.free_bytes) + " the card has free";
    } else {
        const std::optional<PlanRungCost> low = cost(served.rungs.size() - 1);
        const std::uint64_t tier =
            low ? plan_needed_bytes(*low, out.margins,
                                    out.margins.allowance != 0 ? PlanReserve::WarmUp
                                                               : PlanReserve::Soft)
                : 0;
        out.reason = "beside the weights (" + gib_text(out.weight_bytes) + ") the smallest plan, " +
                     describe_plan_rung(last.rung, out.axes.concurrency) + ", needs " +
                     gib_text(tier) + "; " + gib_text(out.free_after_weights) + " is free, " +
                     gib_text(tier > out.free_after_weights ? tier - out.free_after_weights : 0) +
                     " more is needed";
    }
    return out;
}

} // namespace ninferno::runtime
