#include "product/choice/choice.h"

#include "core/generation.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace ninferno::product::choice {

namespace {

constexpr double kQuantFloorBits = 3.5; // runtime/engine/quant_rule.h kPlanQualityFloorBits

std::string_view generation_of(const CardFacts& card) noexcept {
    return generation_row(card.major, card.minor).name;
}

// The generation row's tensor rate per SM and clock for a route (Fp4: twice the int8 rate).
double peak(const CardFacts& card, Route route) noexcept {
    const GenerationRow& row = generation_row(card.major, card.minor);
    const double rate = route == Route::Bf16 ? row.bf16 : route == Route::Fp4 ? 2 * row.s8 : row.s8;
    return card.sms * card.clock_hz * rate;
}

const DecodeConstants& decode_constants(const ModelFacts& model) noexcept {
    return model.moe() ? kMoeDecode : kDenseDecode;
}

// The prefill efficiency of (kind, generation, route): its own row, else the transfer generation's,
// else the dense row of either; `transferred` when not its own.
double prefill_eta(bool moe, std::string_view generation, Route route, bool& transferred) noexcept {
    const Route base = route == Route::Fp4 ? Route::Int8 : route;
    transferred      = route == Route::Fp4;
    for (const std::string_view gen : {generation, kTransferGeneration}) {
        for (const bool kind : {moe, false}) {
            for (const PrefillEta& row : kPrefillEta) {
                if (row.moe == kind && row.generation == gen && row.route == base) {
                    transferred = transferred || gen != generation || kind != moe;
                    return row.eta;
                }
            }
        }
    }
    transferred = true;
    return kPrefillEta[2].eta;
}

double exact_factor(std::string_view generation, bool& transferred) noexcept {
    for (const std::string_view gen : {generation, kTransferGeneration}) {
        for (const RouteFactor& row : kExactComputeFactor) {
            if (row.generation == gen) {
                transferred = transferred || gen != generation;
                return row.factor;
            }
        }
    }
    transferred = true;
    return 1.0;
}

double kv_read_factor(std::string_view generation, KvCacheStorage kv, bool& transferred) noexcept {
    if (kv == KvCacheStorage::Int8Group64) { return 1.0; } // the reference codec
    for (const std::string_view gen : {generation, kTransferGeneration}) {
        for (const KvReadFactor& row : kKvReadFactor) {
            if (row.generation == gen && row.kv == kv) {
                transferred = transferred || gen != generation;
                return row.factor;
            }
        }
    }
    transferred = true;
    return 1.0;
}

const KldTerm* measured_term(Axis axis, std::string_view value) noexcept {
    for (const KldTerm& term : kKldTerms) {
        if (term.axis == axis && term.value == value) { return &term; }
    }
    return nullptr;
}

// The worst measured term of an axis: an unmeasured value never counts as more accurate.
double worst_term(Axis axis) noexcept {
    double worst = 0;
    for (const KldTerm& term : kKldTerms) {
        if (term.axis == axis) { worst = std::max(worst, term.kld); }
    }
    return worst;
}

std::string_view state_name(RecurrentStateStorage state) noexcept {
    return recurrent_state_storage_name(state);
}

bool qwen_family(const ModelFacts& model) noexcept {
    return model.family == "qwen3_5_text" || model.family == "qwen3_5_moe_text";
}

std::string activation_name(ActivationMode mode) {
    return mode == ActivationMode::Exact ? "exact" : mode == ActivationMode::Int8 ? "int8" : "fast";
}

// The order among candidates of equal standing: the larger worth, then the longer own context,
// then the fewer bytes of weights, then the enumeration order.
bool better(const Scored& a, std::size_t ia, const Scored& b, std::size_t ib,
            const std::vector<QuantOption>& quants) {
    if (std::abs(a.worth - b.worth) > kWorthTie) { return a.worth > b.worth; }
    if (a.own_context != b.own_context) { return a.own_context > b.own_context; }
    const double wa = quants[a.candidate.quant].weight_bytes;
    const double wb = quants[b.candidate.quant].weight_bytes;
    if (wa != wb) { return wa < wb; }
    return ia < ib;
}

// The quantization rule's order: more bits per weight, then more bytes, then the listing.
bool larger_quant(const QuantOption& a, std::size_t ia, const QuantOption& b, std::size_t ib) {
    if (a.bits_per_weight != b.bits_per_weight) { return a.bits_per_weight > b.bits_per_weight; }
    if (a.weight_bytes != b.weight_bytes) { return a.weight_bytes > b.weight_bytes; }
    return ia < ib;
}

std::string fixed(double value, int digits) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(digits) << value;
    return out.str();
}

} // namespace

Coupled coupled_storages(const ModelFacts& model, KvCacheStorage /*kv*/, ActivationMode activations,
                         const Pins& pins) noexcept {
    Coupled out{KvCacheStorage::BFloat16, RecurrentStateStorage::FP32};
    out.ring = pins.ring.value_or(KvCacheStorage::BFloat16);
    if (model.recurrent) {
        out.state = pins.state.value_or(activations == ActivationMode::Exact
                                            ? RecurrentStateStorage::FP32
                                            : RecurrentStateStorage::FP16);
    }
    return out;
}

Route route_of(const QuantOption& quant, ActivationMode activations, const CardFacts& card) noexcept {
    if (activations == ActivationMode::Exact) { return Route::Bf16; }
    if (quant.nvfp4) {
        return activations == ActivationMode::Fast && card.fp4 ? Route::Fp4 : Route::Int8;
    }
    return quant.int8_route ? Route::Int8 : Route::Bf16;
}

Speed speed_of(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
               const Candidate& candidate, std::uint32_t concurrency, double context) {
    Speed out;
    const DecodeConstants& k  = decode_constants(model);
    const std::string_view gen = generation_of(card);
    const Route route          = route_of(quant, candidate.activations, card);
    const double rows          = std::max<std::uint32_t>(concurrency, 1);
    const double params        = model.active_params > 0 ? model.active_params : model.params;

    // The weights a step reads: all but an untied input table's rows; an MoE's routed experts as
    // rows routed apart add them.
    double weights = quant.weight_bytes - model.input_table_params * quant.bits_per_weight / 8;
    if (model.moe() && model.params > 0) {
        const double routed = std::clamp(model.routed_expert_params / model.params, 0.0, 1.0);
        const double pick   = static_cast<double>(model.experts_per_token) / model.experts;
        const double apart  = 1 - std::pow(1 - pick, rows);
        weights *= (1 - routed) + routed * (pick + kMoeRoutedShare * (apart - pick));
    }
    bool transferred = false;
    const double kv  = model.kv_bytes_per_token ? model.kv_bytes_per_token(candidate.kv) : 0;
    const double kkv = kv_read_factor(gen, candidate.kv, transferred);
    const double state = model.state_bytes ? model.state_bytes(candidate.state) : 0;
    const double ring  = model.ring_bytes_per_token ? model.ring_bytes_per_token(candidate.ring) : 0;
    const double window_rows = model.window > 0 ? std::min<double>(context, model.window) : 0;
    const double bytes = weights + rows * context * kv * kkv + 2 * rows * state + rows * window_rows * ring;

    double compute = rows * 2 * params / (k.eta_compute * peak(card, Route::Int8));
    if (route == Route::Bf16) {
        compute *= exact_factor(gen, transferred);
    } else if (route == Route::Fp4) {
        compute *= 0.5; // twice the int8 rate (not measured)
        transferred = true;
    }
    const double step     = k.step_seconds + bytes / (k.eta_dram * card.dram_bytes_per_s) + compute;
    out.decode_per_stream = step > 0 ? 1 / step : 0;

    bool prefill_transferred = false;
    const double eta = prefill_eta(model.moe(), gen, route, prefill_transferred);
    out.prefill      = eta * peak(card, route) / (2 * params);
    out.transferred  = transferred || prefill_transferred;
    return out;
}

double request_seconds(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                       const Candidate& candidate, const Task& task) {
    const double middle = task.prompt + task.answer / 2.0;
    const Speed speed   = speed_of(model, card, quant, candidate, task.concurrency, middle);
    if (speed.prefill <= 0 || speed.decode_per_stream <= 0) { return INFINITY; }
    return task.prompt / speed.prefill + task.answer / speed.decode_per_stream;
}

Accuracy accuracy_of(const ModelFacts& model, const QuantOption& quant, const Candidate& candidate,
                     const CardFacts& card) {
    Accuracy out;
    const bool other_family = model.family != kKldFamily;
    const auto add = [&](Axis axis, std::string value, double kld, bool measured, std::string source) {
        out.terms.push_back({axis, std::move(value), kld, measured, other_family && kld > 0,
                             std::move(source)});
    };
    // The quantization: no measured curve yet (choice_constants.h).
    add(Axis::Quant, quant.label, 0, false, "no measured curve: the quant rule picks");
    out.quant_known = false;

    const std::string kv{kv_cache_storage_name(candidate.kv)};
    if (candidate.kv == KvCacheStorage::BFloat16) {
        add(Axis::Kv, kv, 0, true, "the reference");
    } else if (const KldTerm* term = measured_term(Axis::Kv, kv)) {
        add(Axis::Kv, kv, term->kld, true, std::string(term->source));
    } else {
        add(Axis::Kv, kv, worst_term(Axis::Kv), false, "not measured: the axis's worst");
    }
    if (model.window > 0) {
        const std::string ring{kv_cache_storage_name(candidate.ring)};
        if (candidate.ring == KvCacheStorage::BFloat16) {
            add(Axis::Ring, ring, 0, true, "the reference");
        } else {
            const KldTerm* proxy = measured_term(Axis::Kv, ring);
            add(Axis::Ring, ring, proxy ? proxy->kld : worst_term(Axis::Kv), false,
                "not measured: the KV codec's term stands in");
        }
    }
    if (model.recurrent) {
        const std::string state{state_name(candidate.state)};
        if (candidate.state == RecurrentStateStorage::FP32) {
            add(Axis::State, state, 0, true, "the reference");
        } else if (const KldTerm* term = measured_term(Axis::State, state)) {
            add(Axis::State, state, term->kld, true, std::string(term->source));
        } else {
            add(Axis::State, state, worst_term(Axis::State), false, "not measured: the axis's worst");
        }
    }
    const Route route = route_of(quant, candidate.activations, card);
    if (route == Route::Bf16) {
        add(Axis::Activations, "bf16", 0, true, "the reference");
    } else if (route == Route::Int8) {
        const KldTerm* term = measured_term(Axis::Activations, "int8");
        add(Axis::Activations, "int8", term->kld, true, std::string(term->source));
    } else {
        add(Axis::Activations, "fp4", worst_term(Axis::Activations), false,
            "not measured: the axis's worst");
    }
    double sum = 0;
    for (const Term& term : out.terms) { sum += term.kld * term.kld; }
    out.kld = std::sqrt(sum);
    return out;
}

Allowed allowed_axes(const ModelFacts& model, const CardFacts& /*card*/) {
    Allowed out;
    if (qwen_family(model)) {
        out.kv = {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                  KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Nvfp4Group16};
        out.activations = {ActivationMode::Fast, ActivationMode::Exact, ActivationMode::Int8};
    } else {
        out.kv          = {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256};
        out.activations = {ActivationMode::Fast, ActivationMode::Exact};
    }
    return out;
}

Choice choose(const ModelFacts& model, const CardFacts& card, const std::vector<QuantOption>& quants,
              const Allowed& allowed, const Pins& pins, const Task& task,
              const OwnContextFn& own_context) {
    Choice out;
    std::vector<std::size_t> quant_ids;
    if (pins.quant) {
        quant_ids.push_back(*pins.quant);
    } else {
        for (std::size_t i = 0; i < quants.size(); ++i) {
            if (quants[i].supported) { quant_ids.push_back(i); }
        }
    }
    const std::vector<KvCacheStorage> kvs =
        pins.kv ? std::vector<KvCacheStorage>{*pins.kv} : allowed.kv;
    const std::vector<ActivationMode> modes =
        pins.activations ? std::vector<ActivationMode>{*pins.activations} : allowed.activations;
    for (const std::size_t q : quant_ids) {
        for (const KvCacheStorage kv : kvs) {
            for (const ActivationMode mode : modes) {
                // --activations exact refuses a rounded recurrent state the user pinned.
                if (mode == ActivationMode::Exact && model.recurrent && pins.state &&
                    *pins.state != RecurrentStateStorage::FP32) {
                    continue;
                }
                Scored s;
                const Coupled coupled = coupled_storages(model, kv, mode, pins);
                s.candidate   = {q, kv, coupled.ring, coupled.state, mode};
                s.own_context = own_context ? own_context(s.candidate) : 0;
                s.fits        = s.own_context >= task.context && task.context > 0;
                s.speed       = speed_of(model, card, quants[q], s.candidate, task.concurrency,
                                         task.prompt + task.answer / 2.0);
                s.accuracy    = accuracy_of(model, quants[q], s.candidate, card);
                s.seconds     = request_seconds(model, card, quants[q], s.candidate, task);
                s.worth       = -std::log(s.seconds) - task.accuracy_weight * s.accuracy.kld;
                out.scored.push_back(std::move(s));
            }
        }
    }
    // The quantization: the rule's (the largest that fits, not below the floor bits unless none
    // above it fits), then the best worth among its candidates. With a measured quant curve the
    // worth alone would decide (choice_constants.h).
    for (const bool above_floor : {true, false}) {
        std::optional<std::size_t> quant;
        for (const Scored& s : out.scored) {
            const QuantOption& q = quants[s.candidate.quant];
            if (!s.fits || (q.bits_per_weight >= kQuantFloorBits) != above_floor) { continue; }
            if (!quant || larger_quant(q, s.candidate.quant, quants[*quant], *quant)) {
                quant = s.candidate.quant;
            }
        }
        if (!quant) { continue; }
        for (std::size_t i = 0; i < out.scored.size(); ++i) {
            const Scored& s = out.scored[i];
            if (!s.fits || s.candidate.quant != *quant) { continue; }
            if (!out.chosen || better(s, i, out.scored[*out.chosen], *out.chosen, quants)) {
                out.chosen = i;
            }
        }
        const Scored& c = out.scored[*out.chosen];
        out.why = quants[*quant].label + ": the largest quantization whose every stream holds " +
                  std::to_string(task.context) + " tokens" +
                  (above_floor ? "" : " (below " + fixed(kQuantFloorBits, 1) + " bits per weight)") +
                  "; KV " + std::string(kv_cache_storage_name(c.candidate.kv)) + ", activations " +
                  activation_name(c.candidate.activations) + ": the best worth at it (" +
                  fixed(c.seconds, 2) + " s a " + std::to_string(task.prompt) + "+" +
                  std::to_string(task.answer) + "-token request, KLD " + fixed(c.accuracy.kld, 4) + ")";
        return out;
    }
    out.why = "no candidate holds " + std::to_string(task.context) + " tokens for each of " +
              std::to_string(task.concurrency) + " streams";
    return out;
}

std::vector<Alternative> explain(const Choice& choice, const std::vector<QuantOption>& quants) {
    std::vector<Alternative> out;
    if (!choice.chosen) { return out; }
    const Scored& chosen = choice.scored[*choice.chosen];
    const Candidate& c   = chosen.candidate;
    const auto same_but = [&](const Candidate& x, Axis axis) {
        return (axis == Axis::Quant || x.quant == c.quant) && (axis == Axis::Kv || x.kv == c.kv) &&
               (axis == Axis::Activations || x.activations == c.activations);
    };
    for (const Axis axis : {Axis::Quant, Axis::Kv, Axis::Activations}) {
        for (const Scored& s : choice.scored) {
            if (!same_but(s.candidate, axis)) { continue; }
            Alternative alt;
            alt.axis   = axis;
            alt.scored = &s;
            alt.value  = axis == Axis::Quant ? quants[s.candidate.quant].label
                         : axis == Axis::Kv  ? std::string(kv_cache_storage_name(s.candidate.kv))
                                             : activation_name(s.candidate.activations);
            if (&s == &chosen) {
                alt.why_not = "";
            } else if (!s.fits) {
                alt.why_not = "does not hold the asked context (" + std::to_string(s.own_context) +
                              " each)";
            } else if (axis == Axis::Quant) {
                alt.why_not = quants[s.candidate.quant].bits_per_weight >
                                      quants[c.quant].bits_per_weight
                                  ? "does not fit"
                                  : "smaller: the rule takes the largest that fits";
            } else {
                alt.why_not = "worth " + fixed(s.worth - chosen.worth, 3) + " against the chosen";
            }
            out.push_back(std::move(alt));
        }
    }
    return out;
}

} // namespace ninferno::product::choice
