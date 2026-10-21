#include "runtime/engine/plan_dry_run.h"

#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <cmath>
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

std::uint64_t round_tokens_of(const SpeculativeOptions& speculative) {
    std::uint32_t window = speculative.draft_tokens;
    for (const std::uint32_t k : speculative.request_draft_tokens) { window = std::max(window, k); }
    return window + 1U;
}

std::string gib_text(std::uint64_t bytes) {
    const std::uint64_t hundredths = (bytes * 100ULL + kGiB - 1ULL) >> 30;
    const std::string cents        = std::to_string(hundredths % 100);
    return std::to_string(hundredths / 100) + "." + (cents.size() < 2 ? "0" : "") + cents + " GiB";
}

} // namespace

// ---- Seams.

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
    axes.round_tokens    = static_cast<std::uint32_t>(round_tokens_of(options.speculative));
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

std::optional<PlanFamilyStorage> plan_family_storage(std::string_view model_type,
                                                     int compute_capability) {
    PlanFamilyStorage out;
    if (model_type == "qwen3_5_text" || model_type == "qwen3_5_moe_text") {
        out.kv        = KvCacheStorage::Int8Group64;
        out.recurrent = RecurrentStateStorage::FP16;
    } else if (model_type == "gemma4_text" || model_type == "dense_decoder" ||
               model_type == "moe_decoder") {
        out.kv = KvCacheStorage::Fp8E4M3Row256;
    } else if (model_type != "nemotron_h") {
        return std::nullopt;
    }
    // FP8 rows fall back to BF16 below sm_89 (family_registry.h automatic_kv_cache).
    if (out.kv == KvCacheStorage::Fp8E4M3Row256 && compute_capability < 89) {
        out.kv = KvCacheStorage::BFloat16;
    }
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
    const std::uint64_t round = round_tokens_of(options.speculative);
    if (model.hidden != 0) {
        // About twelve hidden-wide BF16 rows a token (norms, residual, attention projections) and
        // three of the widest feed-forward it runs (catalog.h's estimate, checked on the RTX 4090
        // log of Qwen3.6 35B: 79 MiB at chunk 1024), and FP32 logits of every decode column.
        out.workspace = chunk * (12ULL * model.hidden + 3ULL * model.ffn_active) * 2ULL +
                        static_cast<std::uint64_t>(concurrency) * round * model.vocab * 4ULL;
    } else {
        out.workspace = std::uint64_t{256} << 20;
    }
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
    out.weight_bytes  = model.weight_bytes + model.drafter_bytes;
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

    const std::uint64_t image_bytes =
        plan_state_bytes(model, options.recurrent_state) +
        (model.sliding_layers != 0 ? static_cast<std::uint64_t>(model.window) *
                                         plan_ring_bytes_per_token(model, options.sliding_kv_cache)
                                   : 0);
    const std::uint64_t pin_bytes =
        cache.enabled ? (cache.host_kv_capacity_bytes +
                         static_cast<std::uint64_t>(cache.host_state_slots) * image_bytes) >>
                            9
                      : 0;

    struct Built {
        std::vector<PlanRung> rungs;
        std::vector<PlanRungBytes> bytes;
    };

    const auto build = [&](std::uint32_t concurrency, bool pins, bool chunk_pinned) {
        const std::vector<PlanRung> candidates =
            plan_ladder(plan_ladder_inputs(out.axes, concurrency, pins, chunk_pinned));
        std::vector<PlanRungBytes> candidate_bytes;
        candidate_bytes.reserve(candidates.size());
        for (const PlanRung& rung : candidates) {
            candidate_bytes.push_back(plan_rung_estimate(model, rung, options, concurrency));
        }
        Built built;
        built.rungs = prune_plan_ladder(candidates, [&](std::size_t index) {
            return std::optional<std::uint64_t>(candidate_bytes[index].total());
        });
        for (const PlanRung& rung : built.rungs) {
            built.bytes.push_back(plan_rung_estimate(model, rung, options, concurrency));
        }
        return built;
    };
    const auto cost_of = [&](const Built& built) {
        return PlanRungCostFn(
            [&built, pin_bytes](std::size_t index) -> std::optional<PlanRungCost> {
                const PlanRungBytes& b = built.bytes.at(index);
                return PlanRungCost{
                    .bytes = b.total(), .held_bytes = b.total() - b.graphs, .pin_bytes = pin_bytes};
            });
    };

    const bool chunk_pinned   = !out.axes.chunk_automatic;
    const Built served        = build(out.axes.concurrency, true, chunk_pinned);
    const PlanRungCostFn cost = cost_of(served);
    for (std::size_t index = 0; index < served.rungs.size(); ++index) {
        const PlanRung& rung = served.rungs[index];
        const PlanRungCost c = *cost(index);
        out.ladder.push_back(
            DryRunRung{.rung   = rung,
                       .bytes  = served.bytes[index],
                       .needed = plan_needed_bytes(
                           c, out.margins, rung.survival ? PlanReserve::Soft : PlanReserve::Kept)});
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
    // Nothing fits: why, and the longest automatic context each concurrency fits.
    const PlanMargins advice = plan_advice_margins(out.axes);
    for (std::uint32_t concurrency = out.axes.concurrency; concurrency >= 1; --concurrency) {
        const Built ladder = build(concurrency, false, chunk_pinned);
        if (const std::optional<PlanSelection> fit =
                select_plan_rung(ladder.rungs, cost_of(ladder), out.free_after_weights, advice)) {
            const std::uint32_t context = ladder.rungs[fit->rung].max_context;
            if (out.fits_instead.empty() || context > out.fits_instead.back().max_context) {
                out.fits_instead.push_back(
                    {.max_concurrency = concurrency, .max_context = context});
            }
        }
    }
    const DryRunRung& last = out.ladder.back();
    if (out.weight_bytes >= out.free_bytes) {
        out.reason = "the weights (" + gib_text(out.weight_bytes) + ") do not fit the " +
                     gib_text(out.free_bytes) + " the card has free";
    } else {
        const std::uint64_t tier =
            plan_needed_bytes(*cost(served.rungs.size() - 1), out.margins,
                              out.margins.allowance != 0 ? PlanReserve::WarmUp : PlanReserve::Soft);
        out.reason = "beside the weights (" + gib_text(out.weight_bytes) + ") the smallest plan, " +
                     describe_plan_rung(last.rung, out.axes.concurrency) + ", needs " +
                     gib_text(tier) + "; " + gib_text(out.free_after_weights) + " is free, " +
                     gib_text(tier > out.free_after_weights ? tier - out.free_after_weights : 0) +
                     " more is needed";
    }
    return out;
}

// ---- The quant rule.

namespace {

double rank_bits(const QuantCandidate& q) noexcept {
    return q.bits_per_weight > 0 ? q.bits_per_weight : static_cast<double>(q.label_bits);
}

bool near_bytes(std::uint64_t a, std::uint64_t b) noexcept {
    const double hi = static_cast<double>(std::max(a, b));
    const double lo = static_cast<double>(std::min(a, b));
    return hi > 0 && (hi - lo) / hi <= 0.01;
}

} // namespace

bool quant_ranks_above(const QuantCandidate& a, const QuantCandidate& b) noexcept {
    if (near_bytes(a.bytes, b.bytes)) {
        if (a.quality_loss && b.quality_loss && *a.quality_loss != *b.quality_loss) {
            return *a.quality_loss < *b.quality_loss;
        }
        if (a.bytes != b.bytes) { return a.bytes > b.bytes; }
        return a.label < b.label;
    }
    const double x = rank_bits(a), y = rank_bits(b);
    if (x != y) { return x > y; }
    if (a.bytes != b.bytes) { return a.bytes > b.bytes; }
    return a.label < b.label;
}

std::optional<std::size_t> largest_fitting_quant(const std::vector<QuantCandidate>& candidates,
                                                 const std::vector<QuantFit>& fits) {
    if (fits.size() != candidates.size()) {
        throw std::invalid_argument("largest_fitting_quant: a fit for every candidate");
    }
    for (const bool private_only : {true, false}) {
        std::optional<std::size_t> best;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (!candidates[i].supported) { continue; }
            const bool ok = private_only ? fits[i].fits : fits[i].shared;
            if (!ok) { continue; }
            if (!best || quant_ranks_above(candidates[i], candidates[*best])) { best = i; }
        }
        if (best) { return best; }
    }
    return std::nullopt;
}

QuantFit quant_fit_of(const DryRunPlan& plan) {
    QuantFit out;
    const DryRunRung* chosen = plan.chosen();
    if (chosen == nullptr || chosen->rung.survival) { return out; }
    out.context = chosen->rung.max_context;
    out.fits    = plan.private_contexts();
    out.shared  = !out.fits;
    return out;
}

} // namespace ninferno::runtime
