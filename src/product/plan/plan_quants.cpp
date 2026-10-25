#include "product/plan/plan_quants.h"

#include "product/plan/plan_choice.h"

#include "convert/fetch.h"
#include "core/ascii.h"

#include "product/numeric_options.h"
#include "product/speculative_options.h"
#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace ninferno::product {

namespace {

using Json            = nlohmann::json;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

std::string grouped(std::uint64_t value) {
    std::string digits = std::to_string(value);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) {
        digits.insert(static_cast<std::size_t>(at), ",");
    }
    return digits;
}

std::string gib(std::uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof text, "%.2f GiB", static_cast<double>(bytes) / kGiB);
    return text;
}

// Rounded to a step (0.1, 0.01, ...): by the step's inverse, so 61.4 prints as 61.4.
double round_to(double x, double step) {
    const double inverse = std::round(1.0 / step);
    return inverse >= 1 ? std::round(x * inverse) / inverse : std::round(x / step) * step;
}

std::string context_text(const EngineOptions& options) {
    return options.max_context_automatic ? std::string("auto") : grouped(options.max_context);
}

std::string class_text(const runtime::PlanDeviceFacts& device) {
    return std::to_string(card_gb(device)) + " GB";
}

std::string ascii_upper(std::string text) {
    for (char& c : text) {
        if (c >= 'a' && c <= 'z') { c = static_cast<char>(c - 'a' + 'A'); }
    }
    return text;
}

struct Unsupported {
    std::string code;
    std::string text;
};

// Why a listed quantization is not planned (an empty code when it is).
Unsupported unsupported(const QuantListing& q, double bits_per_weight,
                        const runtime::PlanDeviceFacts& device) {
    if (q.drafter) { return {"drafter_file", "a drafter or sidecar file, not a model"}; }
    if (q.format == "safetensors") {
        if (q.converted_bytes) { return {}; }
        return {"converted_size_unknown",
                "converted at start by its recipe: its size after conversion is planned at the "
                "start"};
    }
    if (q.bits > 8 || (q.bits == 0 && bits_per_weight > 9.0)) {
        return {"above_8_bits", "above 8 bits per weight: the engine serves 8 bits or fewer"};
    }
    // NVFP4 weights run on cards with FP4 tensor cores (compute capability 10.0 and up).
    if (ascii_upper(q.quant).find("NVFP4") != std::string::npos && device.compute_capability != 0 &&
        device.compute_capability < 100) {
        return {"nvfp4_card", "NVFP4 weights need an RTX 50 / Blackwell card"};
    }
    return {};
}

// The probe contexts own_context_of tries, ascending (the model's own last).
std::vector<std::uint32_t> probe_contexts(std::uint32_t native) {
    std::vector<std::uint32_t> out;
    for (const std::uint32_t c : {8192U, 12288U, 16384U, 24576U, 32768U, 49152U, 65536U, 98304U,
                                  131072U, 196608U, 262144U, 393216U, 524288U, 786432U, 1048576U}) {
        if (c < native) { out.push_back(c); }
    }
    out.push_back(native);
    return out;
}

// The inputs at a pinned context (ninferno-serve --max-context X with these streams: an
// automatic pool with several, launch_options' rule).
DryRunInputs at_context(const DryRunInputs& inputs, std::uint32_t context) {
    DryRunInputs out                  = inputs;
    out.options.max_context           = context;
    out.options.max_context_automatic = false;
    out.options.kv_capacity           = out.options.max_concurrency > 1
                                            ? KvCapacityPolicy::automatic()
                                            : KvCapacityPolicy::explicit_capacity(context);
    return out;
}

// The inputs of a launch with its context capped at `context` and still automatic (ninferno-serve
// --max-context-cap X: the free memory picks the rung under that ceiling, the step-down rungs
// kept), the way a row's launch_context starts.
DryRunInputs under_context_cap(const DryRunInputs& inputs, std::uint32_t context) {
    DryRunInputs out                  = inputs;
    out.options.max_context           = context;
    out.options.max_context_automatic = true;
    return out;
}

bool own_plan(const runtime::DryRunPlan& plan) {
    const runtime::DryRunRung* chosen = plan.chosen();
    return chosen != nullptr && !chosen->rung.survival && plan.private_contexts();
}

std::string reserve_name(runtime::PlanReserve reserve) {
    return std::string(runtime::plan_reserve_name(reserve));
}

} // namespace

LaunchPlanOptions launch_options(const LaunchAsk& launch) {
    LaunchPlanOptions out;
    EngineOptions& o          = out.options;
    o.max_context             = launch.max_context;
    o.max_context_automatic   = launch.max_context == 0;
    o.max_concurrency         = launch.max_concurrency == 0 ? 1U : launch.max_concurrency;
    o.max_pending_requests    = 16;
    o.pending_timeout_ms      = 600000;
    o.prefill_chunk           = 1024;
    o.prefill_chunk_automatic = true;
    o.kv_lease_window         = 4096;
    // ninferno-ui's launch: exact takes a BF16 KV cache unless one is named (make_launch).
    const std::string kv        = launch.kv_dtype == "auto" ? std::string() : launch.kv_dtype;
    o.kv_cache_automatic        = kv.empty() && !launch.exact;
    o.kv_cache                  = o.kv_cache_automatic ? KvCacheStorage::BFloat16
                                  : kv.empty()         ? KvCacheStorage::BFloat16
                                                       : parse_kv_dtype(kv);
    o.recurrent_state           = RecurrentStateStorage::FP32;
    o.recurrent_state_automatic = true;
    o.activations               = launch.exact ? ActivationMode::Exact : kDefaultActivationMode;
    // ninferno-serve: an automatic context or several slots take an automatic pool, one slot at
    // an explicit context exactly its tokens.
    o.kv_capacity                       = o.max_context_automatic || o.max_concurrency > 1
                                              ? KvCapacityPolicy::automatic()
                                              : KvCapacityPolicy::explicit_capacity(o.max_context);
    o.context_cache                     = ContextCacheOptions{};
    o.context_cache.host_tier_automatic = true;
    SpeculationAsk& ask                 = out.ask;
    if (launch.spec.empty() || launch.spec == "auto") {
        ask.automatic = true;
        // A launch passes no --spec-select: the server decides per request by the profile.
        ask.per_request_from_profile = true;
    } else {
        ask.automatic          = false;
        ask.given.backend      = parse_speculative_backend(launch.spec);
        ask.depth_from_profile = ask.given.backend != SpeculativeBackend::None;
    }
    o.speculative = ask.given;
    return out;
}

DryRunInputs inputs_for(const DryRunInputs& inputs, const QuantListing& quant) {
    DryRunInputs out = inputs;
    if (inputs.subject_of) {
        if (const SpeculationSubject* own = inputs.subject_of(quant)) { out.subject = own; }
    }
    return out;
}

PlannedSpeculation plan_model(const PlanSourceFacts& facts, const DryRunInputs& inputs) {
    return plan_with_speculation(facts, inputs.device, inputs.options, inputs.subject, inputs.ask,
                                 inputs.drafter_bytes, inputs.drafter_state);
}

runtime::QuantAsk quant_ask_of(const DryRunInputs& inputs, std::uint32_t native_context) {
    // An automatic context under --max-context-cap: the targets never pass the cap.
    const bool automatic = inputs.options.max_context_automatic;
    std::uint32_t native = native_context;
    if (automatic && inputs.options.max_context != 0) {
        native =
            native == 0 ? inputs.options.max_context : std::min(native, inputs.options.max_context);
    }
    return runtime::QuantAsk{.automatic   = automatic,
                             .context     = automatic ? 0U : inputs.options.max_context,
                             .concurrency = std::max(1U, inputs.options.max_concurrency),
                             .native      = native};
}

std::uint32_t card_gb(const runtime::PlanDeviceFacts& device) noexcept {
    constexpr std::uint64_t kOne = std::uint64_t{1} << 30;
    return static_cast<std::uint32_t>((device.total_bytes + kOne - 1) / kOne);
}

PlanReasonCode plan_reason_code(const runtime::DryRunPlan& plan) {
    PlanReasonCode out;
    if (!plan.planned) {
        out.code = "shape_unknown";
        return out;
    }
    if (const runtime::DryRunRung* chosen = plan.chosen()) {
        if (chosen->rung.survival) {
            out.code   = "survival_only";
            out.params = {{"context", chosen->rung.max_context},
                          {"reserve_bytes", plan.axes.class_reserve}};
        }
        return out;
    }
    if (plan.weight_bytes >= plan.free_bytes) {
        out.code   = "weights_too_big";
        out.params = {{"weights_bytes", plan.weight_bytes}, {"free_bytes", plan.free_bytes}};
        return out;
    }
    const std::uint64_t needed = plan.ladder.empty() ? 0 : plan.ladder.back().needed;
    out.code                   = "smallest_plan_short";
    out.params                 = {
        {"weights_bytes", plan.weight_bytes},
        {"needed_bytes", needed},
        {"free_after_weights_bytes", plan.free_after_weights},
        {"short_bytes", needed > plan.free_after_weights ? needed - plan.free_after_weights : 0},
        {"context", plan.ladder.empty() ? 0U : plan.ladder.back().rung.max_context},
        {"concurrency", plan.axes.concurrency}};
    return out;
}

AttachedDrafterMemory
attached_drafter_memory(const SpeculationSubject& subject, const std::vector<QuantListing>& listing,
                        std::uint32_t target_hidden, const convert::FetchOptions* fetch) {
    AttachedDrafterMemory out;
    for (const spec::DrafterDecl& d : subject.attached_drafters) {
        std::uint64_t bytes = 0;
        if (!d.gguf.empty()) {
            // A GGUF drafter file beside the target's quantizations.
            for (const QuantListing& q : listing) {
                if (q.drafter && ascii_lower(q.quant).find(ascii_lower(d.gguf)) != std::string::npos) {
                    bytes = std::max(bytes, q.bytes);
                }
            }
        } else if (!d.repo.empty() && fetch != nullptr) {
            const std::string spec = d.repo + (d.revision.empty() ? "" : "@" + d.revision);
            try {
                for (const auto& q : convert::list_quants(spec, *fetch)) {
                    if (!q.drafter) { bytes = std::max(bytes, q.bytes); }
                }
            } catch (const std::exception&) {}
            if (!out.state && (d.kind == "dflash" || d.kind == "dflash2")) {
                try {
                    out.state = plan_drafter_state_of_config(convert::fetch_config(spec, *fetch),
                                                             target_hidden);
                } catch (const std::exception&) {}
            }
        }
        out.bytes = std::max(out.bytes, bytes);
    }
    return out;
}

std::optional<std::uint64_t> checkpoint_converted_bytes(const Json& config, std::uint64_t bytes) {
    const Json& text       = config.contains("text_config") && config["text_config"].is_object()
                                 ? config["text_config"]
                                 : config;
    const std::string type = text.value("model_type", config.value("model_type", std::string()));
    const bool qwen        = type.rfind("qwen3_5", 0) == 0 || type.rfind("qwen3_6", 0) == 0;
    const bool moe         = text.contains("num_experts") && !text["num_experts"].is_null();
    // An encoded checkpoint (a quantization_config) runs as published by the recipes that keep it.
    if (!qwen || !moe || config.contains("quantization_config")) { return bytes; }
    // qwen3_6_35b_a3b: routed experts gate/up Q4, down Q5 (Q6 on three layers), the rest Q8; a
    // group of 32 weights carries a BF16 scale: Q4 4.5, Q5 5.5, Q8 8.5 bits per weight.
    const double layers  = text.value("num_hidden_layers", 0.0);
    const double experts = text.value("num_experts", 0.0);
    const double hidden  = text.value("hidden_size", 0.0);
    const double inter   = text.value("moe_intermediate_size", 0.0);
    const double params  = static_cast<double>(bytes) / 2.0; // BF16
    const double routed  = layers * experts * 3.0 * hidden * inter;
    if (routed <= 0 || routed >= params) { return std::nullopt; }
    const double bits = routed * (2.0 * 4.5 + 5.5) / 3.0 + (params - routed) * 8.5;
    return static_cast<std::uint64_t>(bits / 8.0);
}

std::uint32_t own_context_of(const PlanSourceFacts& facts, const DryRunInputs& inputs,
                             const PlannedSpeculation& served) {
    const runtime::DryRunPlan& plan = served.plan;
    if (!plan.planned) { return 0; }
    if (own_plan(plan)) { return plan.chosen()->rung.max_context; }
    // One stream under an automatic context starts as planned here (no launch context caps it):
    // a probe at a pinned context takes an explicit capacity, which keeps no class reserve, so it
    // would call own a context the start serves on a survival rung without the reserve.
    if (!inputs.options.max_context_automatic || inputs.options.max_concurrency <= 1) { return 0; }
    // An automatic context: the longest pinned context that keeps every stream's own (the plan
    // is monotone in the context: a shorter one fits whenever a longer one does).
    const std::vector<std::uint32_t> probes = probe_contexts(facts.model.native_context);
    std::size_t lo = 0, hi = probes.size(); // [lo, hi): the first not own
    std::uint32_t best = 0;
    while (lo < hi) {
        const std::size_t mid            = lo + (hi - lo) / 2;
        const DryRunInputs pinned        = at_context(inputs, probes[mid]);
        const PlannedSpeculation planned = plan_model(facts, pinned);
        if (own_plan(planned.plan)) {
            best = planned.plan.chosen()->rung.max_context;
            lo   = mid + 1;
        } else {
            hi = mid;
        }
    }
    return best;
}

bool holds_own_context(const PlanSourceFacts& facts, const DryRunInputs& inputs,
                       std::uint32_t context) {
    return own_plan(plan_model(facts, at_context(inputs, context)).plan);
}

QuantPlan plan_quants(const PlanSourceFacts& shape, const std::vector<QuantListing>& listing,
                      const DryRunInputs& inputs) {
    QuantPlan out;
    const runtime::QuantAsk ask = quant_ask_of(inputs, shape.model.native_context);
    std::vector<runtime::QuantCandidate> candidates;
    std::vector<runtime::QuantFit> fits;
    for (const QuantListing& q : listing) {
        if (q.drafter) { continue; }
        QuantRow row;
        row.listing      = q;
        row.converted    = q.format == "safetensors" && q.converted_bytes.has_value();
        row.weight_bytes = q.converted_bytes.value_or(q.bytes);
        row.basis        = q.format == "ninferno" ? runtime::WeightBasis::Artifact
                                                  : runtime::WeightBasis::FileSize;
        // One size source per quantization: the shape's own file keeps the header's tensor
        // bytes; every other the listed file sizes.
        if (!row.converted && shape.model.weight_bytes == q.bytes &&
            shape.model.weight_basis != runtime::WeightBasis::Unknown) {
            row.basis = shape.model.weight_basis;
        }
        row.candidate.label      = q.quant;
        row.candidate.bytes      = row.weight_bytes;
        row.candidate.label_bits = q.bits;
        row.candidate.bits_per_weight =
            shape.model.params > 0
                ? static_cast<double>(row.weight_bytes) * 8.0 / shape.model.params
                : 0.0;
        const Unsupported why   = unsupported(q, row.candidate.bits_per_weight, inputs.device);
        row.unsupported_code    = why.code;
        row.why_unsupported     = why.text;
        row.candidate.supported = why.code.empty();
        if (row.candidate.supported) {
            const PlanSourceFacts facts = with_weights(shape, row.weight_bytes, row.basis);
            const DryRunInputs own      = inputs_for(inputs, q);
            row.planned                 = plan_model(facts, own);
            row.fit                     = runtime::quant_fit_of(row.planned.plan,
                                                                own_context_of(facts, own, row.planned), ask);
            if (!ask.automatic) {
                row.launch_context = ask.context;
            } else if (ask.concurrency > 1 && row.fit.state == runtime::QuantFitState::Own &&
                       !own_plan(row.planned.plan)) {
                row.launch_context = row.fit.own_context;
                // The plan the launch serves under that cap (the one the row shows: the ladder,
                // the speculation, fits_now), not the uncapped pool the streams would share.
                row.planned = plan_model(facts, under_context_cap(own, row.launch_context));
            }
        } else {
            row.fit.state = why.code == "converted_size_unknown"
                                ? runtime::QuantFitState::Unknown
                                : runtime::QuantFitState::Unsupported;
        }
        if (row.planned.plan.planned) {
            // The speed of a typical request at the storages this row's plan serves with.
            const PlanSourceFacts facts = with_weights(shape, row.weight_bytes, row.basis);
            choice::ModelFacts model     = choice_model_of(facts);
            const choice::CardFacts card = choice_card_of(inputs.gpu);
            choice::QuantOption q;
            q.label           = row.listing.quant;
            q.weight_bytes    = static_cast<double>(row.weight_bytes);
            q.bits_per_weight = row.candidate.bits_per_weight;
            q.nvfp4           = q.label.find("NVFP4") != std::string::npos ||
                                q.label.find("nvfp4") != std::string::npos;
            // The ternary and two-bit GGML blocks keep BF16 activations (kept_block_policy).
            for (const char* a16 : {"TQ1_0", "TQ2_0", "Q1_0", "Q2_0"}) {
                if (q.label.find(a16) != std::string::npos) { q.int8_route = false; }
            }
            choice::Candidate c;
            c.kv          = row.planned.plan.kv_cache;
            c.state       = row.planned.plan.recurrent;
            c.ring        = inputs.options.sliding_kv_cache;
            c.activations = inputs.options.activations;
            // The drafter this row's plan serves, at the gain the choice weighs it with: the
            // row's speed is of what the row would run.
            const SpeculativeBackend served = row.planned.speculation.engine.backend;
            if (const DryRunInputs own = inputs_for(inputs, row.listing);
                served != SpeculativeBackend::None && own.subject != nullptr) {
                const DrafterAxis axis = drafter_axis(*own.subject, own.ask);
                for (std::size_t i = 0; i < axis.served.size(); ++i) {
                    if (axis.served[i].engine.backend == served) {
                        model.drafters       = axis.options;
                        c.drafter            = i + 1;
                        // Only a measured gain is counted (speed_of); the estimate names it.
                        if (axis.options[i].measured) { row.estimate_drafter = axis.options[i].label; }
                        break;
                    }
                }
            }
            choice::Task task;
            task.concurrency    = std::max<std::uint32_t>(inputs.options.max_concurrency, 1);
            const std::uint32_t context =
                ask.context != 0 ? ask.context : runtime::quant_own_targets(ask).front();
            row.estimate_depth = std::max<std::uint32_t>(512, context / 4);
            row.estimate = choice::speed_of(model, card, q, c, task.concurrency, row.estimate_depth);
            row.request_seconds = choice::request_seconds(model, card, q, c, task);
        }
        candidates.push_back(row.candidate);
        fits.push_back(row.fit);
        out.rows.push_back(std::move(row));
    }
    out.pick = runtime::largest_fitting_quant(candidates, fits, ask);
    // A GPU generation that decides weighs the quantizations by worth within the rule's tier: the
    // rows that keep the same own context per stream on the same side of the floor, each at its own
    // choice (product/plan/plan_choice.h), a tie to the rule's order (quant_ranks_above). Only
    // where every supported row's bits place it on the quantization curve (an unknown one has no
    // accuracy term and would look exact). Elsewhere the rule's largest stands.
    const bool bits_known = std::all_of(out.rows.begin(), out.rows.end(), [](const QuantRow& row) {
        return !row.candidate.supported || row.candidate.bits_per_weight > 0; // what the choice reads
    });
    if (out.pick && bits_known &&
        choice::weighs_quantizations(choice_model_of(shape), choice_card_of(inputs.gpu)) &&
        (out.pick->tier == runtime::QuantPickTier::Own ||
         out.pick->tier == runtime::QuantPickTier::OwnStepped ||
         out.pick->tier == runtime::QuantPickTier::OwnBelowFloor)) {
        const ChoicePins& pins = inputs.pins;
        // Every supported row on the tier's side of the floor, scored at its own choice: a narrower
        // KV codec may bring a row the rule's resolved options left short to the target. Rows go in
        // the order of the best worth their weights could reach (no memory planned: a bound); the
        // memory is planned (one plan a step) only while a row's bound can still beat the best.
        std::vector<std::pair<double, std::size_t>> bounds;
        for (std::size_t i = 0; i < out.rows.size(); ++i) {
            const QuantRow& row = out.rows[i];
            if (!row.candidate.supported ||
                out.pick->below_floor != runtime::below_quality_floor(row.candidate)) {
                continue;
            }
            const auto bound = choose_at_weights(with_weights(shape, row.weight_bytes, row.basis),
                                                 inputs_for(inputs, row.listing), pins,
                                                 row.listing.quant, OwnContext::Unbounded);
            if (bound && bound->choice.chosen) {
                bounds.emplace_back(bound->choice.scored[*bound->choice.chosen].worth, i);
            }
        }
        std::stable_sort(bounds.begin(), bounds.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        std::optional<std::size_t> best;
        double best_worth           = 0;
        std::uint32_t best_context  = 0;
        choice::Scored best_scored;
        for (const auto& [bound, i] : bounds) {
            if (best && bound < best_worth - choice::kWorthTie) { break; }
            const QuantRow& row = out.rows[i];
            const auto served = choose_at_weights(with_weights(shape, row.weight_bytes, row.basis),
                                                  inputs_for(inputs, row.listing), pins,
                                                  row.listing.quant, OwnContext::Steps);
            if (!served || !served->choice.chosen ||
                served->choice.context < out.pick->target_context) {
                continue;
            }
            const double worth = served->choice.scored[*served->choice.chosen].worth;
            if (!best || worth > best_worth + choice::kWorthTie ||
                (std::abs(worth - best_worth) <= choice::kWorthTie &&
                 runtime::quant_ranks_above(row.candidate, out.rows[*best].candidate))) {
                best         = i;
                best_worth   = worth;
                best_context = served->choice.context;
                best_scored  = served->choice.scored[*served->choice.chosen];
            }
        }
        if (best && *best != out.pick->index) {
            // The largest at its own choice: what the trade was against.
            const QuantRow& largest = out.rows[out.pick->index];
            if (const auto at = choose_at_weights(with_weights(shape, largest.weight_bytes, largest.basis),
                                                  inputs_for(inputs, largest.listing), pins,
                                                  largest.listing.quant, OwnContext::Steps);
                at && at->choice.chosen && best_scored.decision_seconds > 0) {
                const choice::Scored& l = at->choice.scored[*at->choice.chosen];
                out.trade = QuantPlan::Trade{.time_ratio = l.decision_seconds / best_scored.decision_seconds,
                                             .kld_delta  = l.accuracy.kld - best_scored.accuracy.kld};
            }
            out.largest_index = out.pick->index;
            out.pick->index   = *best;
            out.by_worth      = true;
            // The launch keeps the own context per stream the row's choice reached (several
            // streams under an automatic context: never a pool they silently share), as the rule's
            // pick does: its preliminary plan was made at the family's defaults.
            QuantRow& row = out.rows[*best];
            if (ask.automatic && ask.concurrency > 1 && best_context != 0) {
                row.launch_context = best_context;
            }
            row.fit.state       = runtime::QuantFitState::Own;
            row.fit.own_context = std::max(row.fit.own_context, best_context);
        }
    }
    if (out.pick) {
        out.chosen                        = out.pick->index;
        out.rows[out.pick->index].largest = true;
    }
    out.rule = quant_rule_line(out, inputs);
    return out;
}

namespace {

// The worth's trade in words: "Q8_0: 26 % more time per request for 0.0030 less KLD; lambda 25:
// 0.001 KLD is worth 2.5 % of a request's time".
std::string trade_words(const std::optional<QuantPlan::Trade>& trade) {
    if (!trade) { return ""; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(0) << ": " << std::abs(trade->time_ratio - 1) * 100 << " % "
        << (trade->time_ratio >= 1 ? "more" : "less") << " time per request for " << std::setprecision(4)
        << std::abs(trade->kld_delta) << (trade->kld_delta <= 0 ? " less" : " more") << " KLD; lambda "
        << std::setprecision(0) << choice::kAccuracyWeight << ": 0.001 KLD is worth " << std::setprecision(1)
        << choice::kAccuracyWeight * 0.1 << " % of a request's time";
    return out.str();
}

} // namespace

std::string quant_rule_line(QuantPlan& plan, const DryRunInputs& inputs) {
    const std::uint32_t streams = std::max(1U, inputs.options.max_concurrency);
    const std::string stream_text =
        std::to_string(streams) + (streams == 1 ? " stream" : " streams");
    const std::string where = "this " + inputs.device.name + " (" + class_text(inputs.device) + ")";
    const bool automatic    = inputs.options.max_context_automatic;
    // "auto context", "auto context (at most 49,152)" under --max-context-cap, or the pinned one.
    const std::string context_words =
        !automatic ? grouped(inputs.options.max_context)
        : inputs.options.max_context != 0
            ? "auto context (at most " + grouped(inputs.options.max_context) + ")"
            : std::string("auto context");
    QuantPlan& mutable_plan = plan;
    Json& params            = mutable_plan.rule_params;
    params                  = {{"concurrency", streams},
                               {"card_gb", card_gb(inputs.device)},
                               {"automatic", automatic},
                               {"context", automatic ? 0U : inputs.options.max_context},
                               {"context_cap", automatic ? inputs.options.max_context : 0U}};
    if (plan.pick) {
        const QuantRow& row         = plan.rows[plan.pick->index];
        const runtime::QuantPick& k = *plan.pick;
        const std::uint32_t context = row.launch_context != 0    ? row.launch_context
                                      : row.fit.own_context != 0 ? row.fit.own_context
                                                                 : row.fit.context;
        mutable_plan.rule_code      = plan.by_worth ? "best_worth" : "largest_fits";
        params["quant"]             = row.listing.quant;
        if (plan.by_worth && plan.largest_index) {
            params["largest"] = plan.rows[*plan.largest_index].listing.quant;
            if (plan.trade) {
                params["time_ratio"]      = round_to(plan.trade->time_ratio, 0.001);
                params["kld_delta"]       = round_to(plan.trade->kld_delta, 0.0001);
                params["accuracy_weight"] = choice::kAccuracyWeight;
            }
        }
        params["tier"]              = std::string(runtime::quant_pick_tier_name(k.tier));
        params["planned_context"]   = context;
        params["below_floor"]       = k.below_floor;
        params["low_quality"]       = k.low_quality;
        std::string line =
            row.listing.quant +
            (plan.by_worth && plan.largest_index
                 ? ": the best balance of speed and accuracy (the largest that fits is " +
                       plan.rows[*plan.largest_index].listing.quant + trade_words(plan.trade) + ") on "
                 : ": the largest that fits on ") +
            where + " at " + context_words + " × " + stream_text;
        switch (k.tier) {
        case runtime::QuantPickTier::Own:
        case runtime::QuantPickTier::OwnStepped:
        case runtime::QuantPickTier::OwnBelowFloor:
            line += ", " + grouped(context) + (streams == 1 ? "" : " for each stream on its own");
            if (k.tier == runtime::QuantPickTier::OwnStepped && automatic) {
                line += " (no larger quantization holds " +
                        grouped(runtime::quant_own_targets(
                                    quant_ask_of(inputs, row.planned.plan.axes.context_ceiling))
                                    .front()) +
                        " each)";
            }
            break;
        case runtime::QuantPickTier::Shared:
        case runtime::QuantPickTier::SharedBelowFloor:
            line += ": the " + std::to_string(streams) + " streams share one pool of " +
                    grouped(row.fit.context) +
                    " tokens (no quantization gives each its own context)";
            break;
        case runtime::QuantPickTier::Desktop:
            line += ": only by giving up the memory kept for the desktop (the screen may stutter)";
            break;
        }
        if (k.low_quality) {
            line += "; below 3 bits per weight: the answers lose quality";
        } else if (k.below_floor) {
            line += "; below 3.5 bits per weight: nothing larger fits";
        }
        return line;
    }
    // Nothing fits: the smallest supported quantization's reason and what fits instead.
    const QuantRow* smallest = nullptr;
    for (const QuantRow& row : plan.rows) {
        if (row.candidate.supported &&
            (smallest == nullptr || row.candidate.bytes < smallest->candidate.bytes)) {
            smallest = &row;
        }
    }
    const std::string ask = context_words + " × " + stream_text;
    if (smallest == nullptr) {
        mutable_plan.rule_code = "none_served";
        return "no quantization of this repository is served here (" + ask + ")";
    }
    mutable_plan.rule_code       = "none_fits";
    const runtime::DryRunPlan& p = smallest->planned.plan;
    const PlanReasonCode reason  = plan_reason_code(p);
    params["smallest"]           = smallest->listing.quant;
    params["reason_code"]        = reason.code;
    params["reason_params"]      = reason.params;
    Json instead                 = Json::array();
    for (const ConcurrencyFit& fit : p.fits_instead) {
        instead.push_back({{"concurrency", fit.max_concurrency}, {"context", fit.max_context}});
    }
    params["fits_instead"] = std::move(instead);
    std::string line       = "no quantization fits " + ask + " on " + where + "; the smallest, " +
                       smallest->listing.quant + ": " +
                       (p.reason.empty() ? std::string("does not fit") : p.reason);
    for (const ConcurrencyFit& fit : p.fits_instead) {
        line += "; --max-concurrency " + std::to_string(fit.max_concurrency) + " fits " +
                grouped(fit.max_context) + " tokens";
    }
    return line;
}

Json plan_json(const runtime::DryRunPlan& plan, bool ladder) {
    const auto bytes_json = [](const runtime::PlanRungBytes& b) {
        return Json{{"kv", b.kv},         {"states", b.states},
                    {"rings", b.rings},   {"workspace", b.workspace},
                    {"graphs", b.graphs}, {"fixed", b.fixed},
                    {"total", b.total()}};
    };
    const auto rung_json = [&](const runtime::DryRunRung& r) {
        return Json{{"context", r.rung.max_context},
                    {"kv_tokens",
                     static_cast<std::uint64_t>(r.rung.kv_pages) * runtime::kAutomaticContextGrain},
                    {"cached_states", r.rung.cached_states},
                    {"prefill_chunk", r.rung.prefill_chunk},
                    {"step", std::string(runtime::plan_step_name(r.rung.step))},
                    {"survival", r.rung.survival},
                    {"bytes", bytes_json(r.bytes)},
                    {"needed_bytes", r.needed}};
    };
    Json out                    = {{"planned", plan.planned},
                                   {"fits", plan.fits},
                                   {"estimate", true},
                                   {"kv_dtype", std::string(kv_cache_storage_name(plan.kv_cache))},
                                   {"state_dtype", plan.recurrent == RecurrentStateStorage::FP32 ? "fp32"
                                                   : plan.recurrent == RecurrentStateStorage::BF16 ? "bf16"
                                                                                                   : "f16"},
                                   {"rungs", plan.ladder.size()},
                                   {"weights_bytes", plan.weight_bytes},
                                   {"free_bytes", plan.free_bytes},
                                   {"free_after_weights_bytes", plan.free_after_weights},
                                   {"class_reserve_bytes", plan.axes.class_reserve},
                                   {"base_margin_bytes", plan.margins.base},
                                   {"allowance_bytes", plan.margins.allowance},
                                   {"context_ceiling", plan.axes.context_ceiling},
                                   {"estimate_basis", plan.estimate_basis},
                                   {"estimate_reason", plan.estimate_reason},
                                   {"reason", plan.reason}};
    const PlanReasonCode reason = plan_reason_code(plan);
    out["reason_code"]          = reason.code;
    out["reason_params"]        = reason.params;
    Json instead = Json::array();
    for (const ConcurrencyFit& fit : plan.fits_instead) {
        instead.push_back({{"concurrency", fit.max_concurrency}, {"context", fit.max_context}});
    }
    out["fits_instead"] = std::move(instead);
    if (const runtime::DryRunRung* chosen = plan.chosen()) {
        out["rung"]             = plan.rung + 1;
        out["reserve"]          = reserve_name(plan.reserve);
        out["reserve_kept"]     = plan.reserve == runtime::PlanReserve::Kept;
        out["chosen"]           = rung_json(*chosen);
        out["context"]          = chosen->rung.max_context;
        out["pool_contexts"]    = round_to(plan.pool_contexts(), 0.1);
        out["private_contexts"] = plan.private_contexts();
        out["description"]      = runtime::describe_plan_rung(chosen->rung, plan.axes.concurrency);
    } else {
        out["rung"]             = 0;
        out["reserve"]          = "";
        out["reserve_kept"]     = false;
        out["chosen"]           = nullptr;
        out["context"]          = 0;
        out["pool_contexts"]    = 0;
        out["private_contexts"] = false;
        out["description"]      = "";
    }
    if (ladder) {
        Json rungs = Json::array();
        for (const runtime::DryRunRung& r : plan.ladder) { rungs.push_back(rung_json(r)); }
        out["ladder"] = std::move(rungs);
    }
    return out;
}

Json planned_json(const PlannedSpeculation& planned, bool ladder) {
    Json out           = plan_json(planned.plan, ladder);
    out["speculation"] = to_json(planned.speculation);
    return out;
}

Json device_json(const runtime::PlanDeviceFacts& device, const runtime::DryRunPlan& plan) {
    return {{"name", device.name},
            {"source", device.source},
            {"total_bytes", device.total_bytes},
            {"class_bytes", device.total_bytes / runtime::kPlanVramClassGrainBytes *
                                runtime::kPlanVramClassGrainBytes},
            {"compute_capability", device.compute_capability},
            {"pci_device_id", device.pci_device_id},
            {"wddm", device.paravirtual},
            {"display", device.drives_display},
            {"integrated", device.integrated},
            {"card_gb", card_gb(device)},
            {"free_basis", device.free_bytes ? "measured" : "idle_card"},
            {"free_bytes", plan.free_bytes},
            {"class_reserve_bytes",
             runtime::plan_class_reserve(device.total_bytes, device.paravirtual,
                                         device.drives_display, device.integrated)},
            {"desktop_bytes", !device.drives_display ? std::uint64_t{0}
                              : device.paravirtual   ? runtime::kPlanIdleWindowsDesktopBytes
                                                     : runtime::kPlanIdleLinuxDesktopBytes}};
}

Json model_json(const PlanSourceFacts& facts, const std::string& source, const std::string& kind) {
    const runtime::PlanModelFacts& m = facts.model;
    return {{"source", source},
            {"kind", kind},
            {"architecture", m.architecture},
            {"family", m.model_type},
            {"native_context", m.native_context},
            {"params", m.params},
            {"shape_known", m.known()},
            {"weights_bytes", m.weight_bytes},
            {"weights_basis", std::string(runtime::weight_basis_name(m.weight_basis))},
            {"kv_bytes_per_token",
             {{"bf16", runtime::plan_kv_bytes_per_token(m, KvCacheStorage::BFloat16)},
              {"int8", runtime::plan_kv_bytes_per_token(m, KvCacheStorage::Int8Group64)},
              {"fp8", runtime::plan_kv_bytes_per_token(m, KvCacheStorage::Fp8E4M3Row256)},
              {"nvfp4", runtime::plan_kv_bytes_per_token(m, KvCacheStorage::Nvfp4Group16)},
              {"k8v4", runtime::plan_kv_bytes_per_token(m, KvCacheStorage::Fp8KeyNvfp4Value)}}},
            {"window", m.window},
            {"builtin_mtp", facts.shape.builtin_mtp}};
}

Json ask_json(const EngineOptions& options, const SpeculationAsk& ask, bool kv_automatic) {
    return {{"context", options.max_context_automatic ? Json("auto") : Json(options.max_context)},
            {"concurrency", options.max_concurrency},
            {"kv_dtype", kv_automatic ? std::string("auto")
                                      : std::string(kv_cache_storage_name(options.kv_cache))},
            {"spec", ask.automatic ? std::string("auto")
                     : ask.given.backend == SpeculativeBackend::None
                         ? std::string("none")
                         : std::string(speculative_backend_name(ask.given.backend)) + ":" +
                               std::to_string(ask.given.draft_tokens)}};
}

Json quant_plan_json(const QuantPlan& plan) {
    Json rows = Json::array();
    for (const QuantRow& row : plan.rows) {
        Json item    = {{"quant", row.listing.quant},
                        {"format", row.listing.format},
                        {"bits", row.listing.bits},
                        {"bits_per_weight", round_to(row.candidate.bits_per_weight, 0.01)},
                        {"weights_bytes", row.weight_bytes},
                        {"weights_basis", row.converted
                                              ? std::string("converted")
                                              : std::string(runtime::weight_basis_name(row.basis))},
                        {"supported", row.candidate.supported},
                        {"unsupported_code", row.unsupported_code},
                        {"why_unsupported", row.why_unsupported},
                        {"state", std::string(runtime::quant_fit_state_name(row.fit.state))},
                        {"own_context", row.fit.own_context},
                        {"launch_context", row.launch_context},
                        {"below_floor",
                         row.candidate.supported && runtime::below_quality_floor(row.candidate)},
                        {"low_quality",
                         row.candidate.supported && runtime::quant_rank_bits(row.candidate) > 0 &&
                             runtime::quant_rank_bits(row.candidate) < runtime::kPlanLowQualityBits},
                        {"drafter_bytes", row.planned.plan.weight_bytes > row.weight_bytes
                                              ? row.planned.plan.weight_bytes - row.weight_bytes
                                              : 0},
                        {"fits", row.fit.state == runtime::QuantFitState::Own},
                        {"fits_shared", row.fit.state == runtime::QuantFitState::Shared},
                        {"largest", row.largest}};
        item["plan"] = row.planned.plan.planned ? planned_json(row.planned, false) : Json(nullptr);
        if (row.estimate) {
            item["estimate"] = {{"decode_tok_s", round_to(row.estimate->decode_per_stream, 0.1)},
                                {"prefill_tok_s", round_to(row.estimate->prefill, 1.0)},
                                {"depth", row.estimate_depth},
                                {"drafter", row.estimate_drafter},
                                {"request_seconds", round_to(row.request_seconds, 0.001)},
                                {"prompt", choice::kRequestPrompt},
                                {"answer", choice::kRequestAnswer},
                                {"transferred", row.estimate->transferred}};
        } else {
            item["estimate"] = nullptr;
        }
        rows.push_back(std::move(item));
    }
    Json pick = nullptr;
    if (plan.pick) {
        const QuantRow& row = plan.rows[plan.pick->index];
        pick                = {{"quant", row.listing.quant},
                               {"tier", std::string(runtime::quant_pick_tier_name(plan.pick->tier))},
                               {"target_context", plan.pick->target_context},
                               {"context", plan.rule_params.value("planned_context", 0U)},
                               {"concurrency", plan.rule_params.value("concurrency", 1U)},
                               {"card_gb", plan.rule_params.value("card_gb", 0U)},
                               {"below_floor", plan.pick->below_floor},
                               {"low_quality", plan.pick->low_quality}};
    }
    return {{"quants", std::move(rows)},
            {"chosen", plan.chosen ? Json(plan.rows[*plan.chosen].listing.quant) : Json(nullptr)},
            {"pick", std::move(pick)},
            {"rule", plan.rule},
            {"rule_code", plan.rule_code},
            {"rule_params", plan.rule_params}};
}

Json dry_plan_document(const PlanSourceFacts& facts, const std::string& source,
                       const std::string& kind, const DryRunInputs& inputs,
                       const PlannedSpeculation& planned, const QuantPlan* quants) {
    Json out = {{"schema", "ninferno-dry-plan/1"},
                {"estimate", true},
                {"model", model_json(facts, source, kind)},
                {"device", device_json(inputs.device, planned.plan)},
                {"ask", ask_json(inputs.options, inputs.ask, inputs.options.kv_cache_automatic)},
                {"plan", planned_json(planned, true)}};
    if (quants != nullptr) {
        Json q        = quant_plan_json(*quants);
        out["quants"] = std::move(q["quants"]);
        out["chosen"]      = std::move(q["chosen"]);
        out["pick"]        = std::move(q["pick"]);
        out["rule"]        = std::move(q["rule"]);
        out["rule_code"]   = std::move(q["rule_code"]);
        out["rule_params"] = std::move(q["rule_params"]);
    } else {
        out["quants"]      = Json::array();
        out["chosen"]      = nullptr;
        out["pick"]        = nullptr;
        out["rule"]        = "";
        out["rule_code"]   = "";
        out["rule_params"] = Json::object();
    }
    return out;
}

std::string render_dry_plan_text(const PlanSourceFacts& facts, const std::string& source,
                                 const DryRunInputs& inputs, const PlannedSpeculation& planned,
                                 const QuantPlan* quants) {
    std::ostringstream out;
    const runtime::DryRunPlan& plan        = planned.plan;
    const runtime::PlanDeviceFacts& device = inputs.device;
    out << "plan without the weights (an estimate; the start is the authority)\n";
    out << "model  | " << source << " (" << facts.model.architecture << ", family "
        << (facts.model.model_type.empty() ? std::string("none") : facts.model.model_type)
        << ", own context " << grouped(facts.model.native_context) << ")\n";
    out << "device | " << device.name << " (" << device.source << ", " << class_text(device)
        << (device.paravirtual ? ", WDDM" : "") << (device.drives_display ? ", display" : "")
        << "), " << (device.free_bytes ? "free now " : "idle card ") << gib(plan.free_bytes)
        << "\n";
    out << "ask    | context " << context_text(inputs.options) << " × "
        << inputs.options.max_concurrency << " streams, KV "
        << (inputs.options.kv_cache_automatic ? "auto"
                                              : kv_cache_storage_name(inputs.options.kv_cache))
        << " (" << kv_cache_storage_name(plan.kv_cache) << "), spec "
        << ask_json(inputs.options, inputs.ask, false)["spec"].get<std::string>() << "\n";
    if (quants != nullptr) {
        out << "quants | (largest that fits first; * the rule's pick)\n";
        std::vector<const QuantRow*> order;
        for (const QuantRow& row : quants->rows) { order.push_back(&row); }
        std::stable_sort(order.begin(), order.end(), [](const QuantRow* a, const QuantRow* b) {
            return runtime::quant_ranks_above(a->candidate, b->candidate);
        });
        for (const QuantRow* row : order) {
            char line[256];
            std::snprintf(line, sizeof line, "  %c %-14s %5.2f bpw %9s  ", row->largest ? '*' : ' ',
                          row->listing.quant.c_str(), row->candidate.bits_per_weight,
                          gib(row->listing.bytes).c_str());
            out << line;
            switch (row->fit.state) {
            case runtime::QuantFitState::Unsupported:
                out << "not served: " << row->why_unsupported;
                break;
            case runtime::QuantFitState::Unknown:
                out << "not planned: "
                    << (row->why_unsupported.empty() ? row->planned.plan.reason
                                                     : row->why_unsupported);
                break;
            case runtime::QuantFitState::Own:
                out << "fits, "
                    << grouped(row->fit.own_context != 0 ? row->fit.own_context : row->fit.context)
                    << (inputs.options.max_concurrency > 1 ? " each" : "");
                if (row->launch_context != 0 && inputs.options.max_context_automatic) {
                    out << " (--max-context " << row->launch_context << ")";
                }
                break;
            case runtime::QuantFitState::Shared:
                out << "fits as one shared pool of " << grouped(row->fit.context);
                break;
            case runtime::QuantFitState::Desktop:
                out << "fits only without the desktop's reserve";
                break;
            case runtime::QuantFitState::No:
                out << "does not fit";
                break;
            }
            out << " ("
                << (row->converted ? std::string_view("converted")
                                   : runtime::weight_basis_name(row->basis))
                << ")\n";
        }
        out << "quant  | " << quants->rule << "\n";
    }
    if (!plan.planned) {
        out << "plan   | " << plan.reason << "\n";
        return out.str();
    }
    for (std::size_t i = 0; i < plan.ladder.size(); ++i) {
        const runtime::DryRunRung& r = plan.ladder[i];
        out << (plan.fits && i == plan.rung ? "  > " : "    ") << "rung " << (i + 1) << " "
            << runtime::plan_step_name(r.rung.step) << (r.rung.survival ? " (survival)" : "")
            << ": " << runtime::describe_plan_rung(r.rung, plan.axes.concurrency) << "; "
            << gib(r.bytes.total()) << ", needs " << gib(r.needed) << "\n";
    }
    if (plan.fits) {
        out << "plan   | rung " << (plan.rung + 1) << " of " << plan.ladder.size() << " ("
            << reserve_name(plan.reserve) << "): "
            << runtime::describe_plan_rung(plan.ladder[plan.rung].rung, plan.axes.concurrency)
            << (plan.private_contexts() ? "" : "; the streams share one pool") << "\n";
    } else {
        out << "plan   | does not fit: " << plan.reason << "\n";
        for (const ConcurrencyFit& fit : plan.fits_instead) {
            out << "         --max-concurrency " << fit.max_concurrency << " fits "
                << grouped(fit.max_context) << " tokens\n";
        }
    }
    const SpeculationReport& s = planned.speculation;
    out << "spec   | " << s.ask << " -> " << s.label;
    if (!s.source.empty()) { out << " (" << s.basis << ": " << s.source << ")"; }
    out << "\n";
    if (s.engine.backend != SpeculativeBackend::None) {
        out << "         load " << spec::load_label(s.load) << ", at "
            << inputs.options.max_concurrency << " streams depth " << s.depth_at_concurrency
            << "\n";
    }
    if (s.with_thinking && s.without_thinking) {
        const auto text = [](const SpeculationReport::ThinkingVariant& v) {
            std::ostringstream o;
            o << v.label;
            if (v.speedup > 0) { o << " x" << std::fixed << std::setprecision(2) << v.speedup; }
            return o.str();
        };
        out << "         thinking " << text(*s.with_thinking) << ", without "
            << text(*s.without_thinking)
            << (s.per_request ? " (each request its own)" : " (one for every request)") << "\n";
    }
    for (const SpeculationFallbackStep& f : s.fallbacks) {
        out << "         " << (f.served ? "> " : "  ") << f.step << " " << f.label << ": "
            << (f.fits ? "fits" : "does not fit") << (f.note.empty() ? "" : " (" + f.note + ")")
            << "\n";
    }
    return out.str();
}

} // namespace ninferno::product
