#include "product/choice/choice.h"

#include "core/generation.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <vector>

namespace ninferno::product::choice {

namespace {

constexpr double kQuantFloorBits = 3.5; // runtime/engine/quant_rule.h kPlanQualityFloorBits

std::string_view generation_of(const CardFacts& card) noexcept {
    return generation_row(card.major, card.minor).name;
}

// The generation row's tensor rate per SM and clock for a route (Fp4: twice the int8 rate). A
// generation without an int8 MMA (Volta) takes its BF16 rate for the int8 route's compute.
double peak(const CardFacts& card, Route route) noexcept {
    const GenerationRow& row = generation_row(card.major, card.minor);
    const double s8          = row.s8 > 0 ? row.s8 : row.bf16;
    const double rate        = route == Route::Bf16 ? row.bf16 : route == Route::Fp4 ? 2 * s8 : s8;
    return card.sms * card.clock_hz * rate;
}

const DecodeConstants& decode_constants(const ModelFacts& model) noexcept {
    return model.moe() ? kMoeDecode : kDenseDecode;
}

const PrefillEta* prefill_row(bool moe, std::string_view generation, Route route) noexcept {
    for (const PrefillEta& row : kPrefillEta) {
        if (row.moe == moe && row.generation == generation && row.route == route) { return &row; }
    }
    return nullptr;
}

// The prefill efficiency of (kind, generation, route): its own row, else its kind's on the
// transfer generation; a route its kind has no row for takes its kind's int8 row times the dense
// rows' ratio of the two routes (the route's cost is the kernels', not the model kind's); else the
// dense row. `transferred` when not its own.
double prefill_eta(bool moe, std::string_view generation, Route route, bool& transferred) noexcept {
    const Route base = route == Route::Fp4 ? Route::Int8 : route;
    transferred      = route == Route::Fp4;
    for (const std::string_view gen : {generation, kTransferGeneration}) {
        if (const PrefillEta* row = prefill_row(moe, gen, base)) {
            transferred = transferred || gen != generation;
            return row->eta;
        }
    }
    for (const std::string_view gen : {generation, kTransferGeneration}) {
        const PrefillEta* own_int8 = prefill_row(moe, gen, Route::Int8);
        if (own_int8 == nullptr) { continue; }
        for (const std::string_view dense_gen : {gen, generation, kTransferGeneration}) {
            const PrefillEta* dense_route = prefill_row(false, dense_gen, base);
            const PrefillEta* dense_int8  = prefill_row(false, dense_gen, Route::Int8);
            if (dense_route != nullptr && dense_int8 != nullptr) {
                transferred = true;
                return own_int8->eta * dense_route->eta / dense_int8->eta;
            }
        }
    }
    for (const std::string_view gen : {generation, kTransferGeneration}) {
        if (const PrefillEta* row = prefill_row(false, gen, base)) {
            transferred = true;
            return row->eta;
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

bool quant_curve() noexcept { return !kQuantKldSource.empty(); }

// An MoE's prefill share at a chunk (kMoeChunkShare, linear in log2 of the tokens each expert takes).
double chunk_share(const ModelFacts& model, std::uint32_t chunk) noexcept {
    const double x = static_cast<double>(chunk) * model.experts_per_token / std::max(model.experts, 1);
    const std::size_t n = std::size(kMoeChunkShare);
    if (x <= kMoeChunkShare[0].tokens_per_expert) { return kMoeChunkShare[0].share; }
    for (std::size_t i = 1; i < n; ++i) {
        const ChunkShare& lo = kMoeChunkShare[i - 1];
        const ChunkShare& hi = kMoeChunkShare[i];
        if (x <= hi.tokens_per_expert) {
            const double t = std::log2(x / lo.tokens_per_expert) /
                             std::log2(hi.tokens_per_expert / lo.tokens_per_expert);
            return lo.share + t * (hi.share - lo.share);
        }
    }
    return kMoeChunkShare[n - 1].share;
}

// The interpolated quantization term at `bits` (choice_constants.h kQuantKld), and the measured
// point when `label` is one. log KLD is linear in bits between the two measured points around
// `bits`; past the measured range, the nearest pair's slope; at or above the reference's bits, 0.
double quant_kld(double bits, std::string_view label, bool& measured) {
    measured = false;
    double reference_bits = 0;
    std::vector<const QuantKldPoint*> points;
    for (const QuantKldPoint& p : kQuantKld) {
        if (!label.empty() && p.label == label) {
            measured = true;
            return p.kld;
        }
        if (p.kld <= 0) {
            reference_bits = std::max(reference_bits, p.bits_per_weight);
        } else {
            points.push_back(&p);
        }
    }
    if (bits >= reference_bits || points.empty()) { return 0; }
    if (points.size() == 1) { return points.front()->kld; }
    std::sort(points.begin(), points.end(),
              [](const QuantKldPoint* a, const QuantKldPoint* b) { return a->bits_per_weight < b->bits_per_weight; });
    std::size_t i = 0;
    while (i + 2 < points.size() && points[i + 1]->bits_per_weight <= bits) { ++i; }
    const QuantKldPoint& lo = *points[i];
    const QuantKldPoint& hi = *points[i + 1];
    const double t = (bits - lo.bits_per_weight) / (hi.bits_per_weight - lo.bits_per_weight);
    return std::exp(std::log(lo.kld) + t * (std::log(hi.kld) - std::log(lo.kld)));
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

Scored score(const ModelFacts& model, const CardFacts& card, const std::optional<Reference>& ref,
             const QuantOption& quant, const Candidate& c, const Task& task,
             const OwnContextFn& own_context) {
    Scored s;
    s.candidate   = c;
    s.own_context = own_context ? own_context(c) : 0;
    s.speed       = speed_of(model, card, quant, c, task.concurrency, task.prompt + task.answer / 2.0);
    s.accuracy    = accuracy_of(model, quant, c, card);
    s.seconds     = request_seconds(model, card, quant, c, task);
    if (ref) {
        s.decision_seconds = request_seconds(model, ref->card, quant, c, task);
        s.worth = -std::log(s.decision_seconds) - task.accuracy_weight * s.accuracy.kld;
    }
    return s;
}

} // namespace

Coupled coupled_storages(const ModelFacts& model, KvCacheStorage kv, ActivationMode activations,
                         const Pins& pins) noexcept {
    Coupled out{KvCacheStorage::BFloat16, RecurrentStateStorage::FP32};
    const bool narrow_kv = kv != KvCacheStorage::BFloat16;
    out.ring = pins.ring.value_or(model.window > 0 && model.fp8_ring && narrow_kv
                                      ? KvCacheStorage::Fp8E4M3Row256
                                      : KvCacheStorage::BFloat16);
    if (model.recurrent) {
        out.state = pins.state.value_or(activations == ActivationMode::Exact || !narrow_kv
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
    out.decode_per_stream = step > 0 && std::isfinite(step) ? 1 / step : 0;

    bool prefill_transferred = false;
    const double eta = prefill_eta(model.moe(), gen, route, prefill_transferred);
    out.prefill      = eta * peak(card, route) / (2 * params);
    if (model.moe() && model.experts_per_token > 0) {
        out.prefill *= chunk_share(model, candidate.chunk) / chunk_share(model, kDefaultChunk);
    }
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

std::optional<Reference> reference_of(const CardFacts& card) {
    const std::string_view gen = generation_of(card);
    for (const ReferenceCard& r : kReferenceCards) {
        if (r.generation != gen) { continue; }
        Reference out;
        out.card = {card.major, card.minor, r.sms, r.clock_hz, r.dram_bytes_per_s, card.fp4};
        out.decides = r.kv_two_cards && r.exact_two_cards;
        if (!out.decides) {
            out.why = std::string(gen) + "'s KV and exact-route factors rest on one lab card";
        }
        return out;
    }
    return std::nullopt;
}

Accuracy accuracy_of(const ModelFacts& model, const QuantOption& quant, const Candidate& candidate,
                     const CardFacts& card) {
    Accuracy out;
    const bool other_family = model.family != kKldFamily;
    const auto add = [&](Axis axis, std::string value, double kld, bool measured, std::string source) {
        out.terms.push_back({axis, std::move(value), kld, measured, other_family && kld > 0,
                             std::move(source)});
    };
    // The quantization: the measured curve (choice_constants.h), or none (the quant rule picks).
    if (quant_curve()) {
        bool measured   = false;
        const double q  = quant_kld(quant.bits_per_weight, quant.label, measured);
        add(Axis::Quant, quant.label, q, measured && !other_family,
            measured ? std::string(kQuantKldSource) : "interpolated by bits per weight on " + std::string(kQuantKldSource));
        out.quant_known = true;
    } else {
        add(Axis::Quant, quant.label, 0, false, "no measured curve: the quant rule picks");
        out.quant_known = false;
    }

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
        const std::string state{recurrent_state_storage_name(candidate.state)};
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

Allowed allowed_axes(const ModelFacts& model, KvCacheStorage automatic_kv) {
    Allowed out;
    out.automatic_kv = automatic_kv;
    if (qwen_family(model)) {
        out.kv = {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                  KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Nvfp4Group16};
        out.activations = {ActivationMode::Fast, ActivationMode::Exact, ActivationMode::Int8};
    } else {
        out.kv          = {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256};
        out.activations = {ActivationMode::Fast, ActivationMode::Exact};
    }
    if (std::find(out.kv.begin(), out.kv.end(), automatic_kv) == out.kv.end()) {
        out.kv.push_back(automatic_kv);
    }
    return out;
}

Choice choose(const ModelFacts& model, const CardFacts& card, const std::vector<QuantOption>& quants,
              const Allowed& allowed, const Pins& pins, const Task& task,
              const OwnContextFn& own_context) {
    Choice out;
    const std::optional<Reference> ref = reference_of(card);
    out.by_worth = ref && ref->decides;
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
    // The chunk: pinned, else an MoE's measured widths (only the default eligible until the shares
    // rest on two lab cards), else the default.
    std::vector<std::uint32_t> chunks;
    if (pins.chunk) {
        chunks = {*pins.chunk};
    } else if (model.moe()) {
        chunks.assign(std::begin(kMoeChunks), std::end(kMoeChunks));
    } else {
        chunks = {kDefaultChunk};
    }
    const bool chunk_decides          = kMoeChunkTwoCards;
    const std::uint32_t default_chunk = pins.chunk.value_or(kDefaultChunk);
    for (const std::size_t q : quant_ids) {
        for (const KvCacheStorage kv : kvs) {
            for (const ActivationMode mode : modes) {
                // --activations exact refuses a rounded recurrent state the user pinned.
                if (mode == ActivationMode::Exact && model.recurrent && pins.state &&
                    *pins.state != RecurrentStateStorage::FP32) {
                    continue;
                }
                const Coupled coupled = coupled_storages(model, kv, mode, pins);
                for (const std::uint32_t chunk : chunks) {
                    out.scored.push_back(score(model, card, ref, quants[q],
                                               {q, kv, coupled.ring, coupled.state, mode, chunk},
                                               task, own_context));
                }
            }
        }
    }
    // The family's defaults: the automatic KV codec and --activations fast, unless pinned.
    const KvCacheStorage default_kv          = pins.kv.value_or(allowed.automatic_kv);
    const ActivationMode default_activations = pins.activations.value_or(ActivationMode::Fast);
    // A candidate the selection may take: any under the worth; only the family's defaults else.
    const auto eligible = [&](const Scored& s) {
        const bool chunk_ok = (out.by_worth && chunk_decides) || s.candidate.chunk == default_chunk;
        return chunk_ok && (out.by_worth || (s.candidate.kv == default_kv &&
                                             s.candidate.activations == default_activations));
    };
    // Each stream's own context in turn (a pinned one, or Auto's steps), above the floor bits at
    // every step before any quantization below them (the quant rule's tiers).
    std::optional<std::size_t> chosen;
    std::uint32_t settled = 0;
    bool above_floor_settled = true;
    for (const bool above_floor : {true, false}) {
        for (const std::uint32_t context : task.contexts) {
            std::optional<std::size_t> quant;
            // With a measured curve and a reference card the worth weighs the quantizations too:
            // the best eligible candidate of any quantization at or (in the second pass) below the
            // floor. Else the quant rule's: the largest.
            const bool by_quant_worth = out.by_worth && quant_curve();
            for (const Scored& s : out.scored) {
                const QuantOption& q = quants[s.candidate.quant];
                if (!eligible(s) || s.own_context < context ||
                    (q.bits_per_weight >= kQuantFloorBits) != above_floor) {
                    continue;
                }
                if (by_quant_worth) {
                    quant = s.candidate.quant; // any: the worth below ranks across them
                } else if (!quant || larger_quant(q, s.candidate.quant, quants[*quant], *quant)) {
                    quant = s.candidate.quant;
                }
            }
            if (!quant) { continue; }
            for (std::size_t i = 0; i < out.scored.size(); ++i) {
                const Scored& s = out.scored[i];
                if (!eligible(s) || s.own_context < context) { continue; }
                if ((quants[s.candidate.quant].bits_per_weight >= kQuantFloorBits) != above_floor) { continue; }
                if (!by_quant_worth && s.candidate.quant != *quant) { continue; }
                if (!chosen || better(s, i, out.scored[*chosen], *chosen, quants)) { chosen = i; }
            }
            if (chosen) {
                settled             = context;
                above_floor_settled = above_floor;
                break;
            }
        }
        if (chosen) { break; }
    }
    if (!chosen) {
        out.why = "no candidate holds " + std::to_string(task.contexts.empty() ? 0 : task.contexts.back()) +
                  " tokens for each of " + std::to_string(task.concurrency) + " streams";
        return out;
    }
    out.chosen  = chosen;
    out.context = settled;
    for (Scored& s : out.scored) { s.fits = s.own_context >= settled; }
    const Scored& c = out.scored[*chosen];
    const bool quant_by_worth = out.by_worth && quant_curve();
    out.why = quants[c.candidate.quant].label +
              (quant_ids.size() == 1 ? " (the one quantization): every stream holds "
               : quant_by_worth      ? ": the best worth among the quantizations whose every stream holds "
                                     : ": the largest quantization whose every stream holds ") +
              std::to_string(settled) + " tokens" +
              (above_floor_settled ? "" : " (below " + fixed(kQuantFloorBits, 1) + " bits per weight)") +
              "; KV " + std::string(kv_cache_storage_name(c.candidate.kv)) + ", activations " +
              activation_name(c.candidate.activations) +
              (out.by_worth ? ": the best worth on the " + std::string(generation_of(card)) +
                                  " reference card (" + fixed(c.decision_seconds, 2) + " s a " +
                                  std::to_string(task.prompt) + "+" + std::to_string(task.answer) +
                                  "-token request there, KLD " + fixed(c.accuracy.kld, 4) + ")"
                            : ": the family's defaults (" +
                                  (ref ? ref->why : std::string(generation_of(card)) + " has no reference card") +
                                  ")");
    // The coupled storages' other values beside the chosen (to explain them; never chosen).
    std::vector<Candidate> variants;
    if (model.recurrent && !pins.state) {
        for (const RecurrentStateStorage st :
             {RecurrentStateStorage::FP32, RecurrentStateStorage::FP16, RecurrentStateStorage::BF16}) {
            if (st == c.candidate.state ||
                (c.candidate.activations == ActivationMode::Exact && st != RecurrentStateStorage::FP32)) {
                continue;
            }
            Candidate v = c.candidate;
            v.state     = st;
            variants.push_back(v);
        }
    }
    if (model.window > 0 && model.fp8_ring && !pins.ring) {
        for (const KvCacheStorage ring : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256}) {
            if (ring == c.candidate.ring) { continue; }
            Candidate v = c.candidate;
            v.ring      = ring;
            variants.push_back(v);
        }
    }
    for (const Candidate& v : variants) {
        Scored s  = score(model, card, ref, quants[v.quant], v, task, own_context);
        s.variant = true;
        s.fits    = s.own_context >= settled;
        out.scored.push_back(std::move(s));
    }
    return out;
}

std::vector<Alternative> explain(const Choice& choice, const std::vector<QuantOption>& quants) {
    std::vector<Alternative> out;
    if (!choice.chosen) { return out; }
    const Scored& chosen = choice.scored[*choice.chosen];
    const Candidate& c   = chosen.candidate;
    const auto value_of = [&](const Scored& s, Axis axis) -> std::string {
        switch (axis) {
        case Axis::Quant: return quants[s.candidate.quant].label;
        case Axis::Kv: return std::string(kv_cache_storage_name(s.candidate.kv));
        case Axis::Ring: return std::string(kv_cache_storage_name(s.candidate.ring));
        case Axis::State: return std::string(recurrent_state_storage_name(s.candidate.state));
        case Axis::Activations: return activation_name(s.candidate.activations);
        case Axis::Chunk: return std::to_string(s.candidate.chunk);
        }
        return "";
    };
    // The chosen candidate but for `axis`: a main axis keeps the other main axes (its coupled
    // storages follow it: exact takes the FP32 state), a coupled one keeps everything else.
    const auto same_but = [&](const Candidate& x, Axis axis) {
        const bool main = (axis == Axis::Quant || x.quant == c.quant) && (axis == Axis::Kv || x.kv == c.kv) &&
                          (axis == Axis::Activations || x.activations == c.activations) &&
                          (axis == Axis::Chunk || x.chunk == c.chunk);
        if (axis != Axis::State && axis != Axis::Ring) { return main; }
        return main && (axis == Axis::State || x.state == c.state) && (axis == Axis::Ring || x.ring == c.ring);
    };
    for (const Axis axis : {Axis::Quant, Axis::Kv, Axis::Activations, Axis::Chunk, Axis::State, Axis::Ring}) {
        const bool coupled = axis == Axis::State || axis == Axis::Ring;
        for (const Scored& s : choice.scored) {
            if (s.variant != coupled && &s != &chosen) { continue; }
            if (!same_but(s.candidate, axis)) { continue; }
            if (&s == &chosen) {
                // The chosen value of an axis is listed only when the axis has another value beside.
                const bool any = std::any_of(choice.scored.begin(), choice.scored.end(), [&](const Scored& v) {
                    return &v != &chosen && v.variant == coupled && same_but(v.candidate, axis);
                });
                if (!any) { continue; }
            }
            Alternative alt;
            alt.axis   = axis;
            alt.scored = &s;
            alt.value  = value_of(s, axis);
            if (&s == &chosen) {
                // the chosen value
            } else if (!s.fits) {
                alt.why_code = "no_fit";
                alt.why_not  = "does not hold the asked context (" + std::to_string(s.own_context) + " each)";
            } else if (axis == Axis::Quant) {
                alt.why_code = "quant_rule";
                alt.why_not  = quants[s.candidate.quant].bits_per_weight > quants[c.quant].bits_per_weight
                                   ? "does not fit"
                                   : "smaller: the rule takes the largest that fits";
            } else if (coupled) {
                alt.why_code = "coupled";
                alt.why_not  = "follows the KV codec and the activations (pin it with its flag)";
            } else if (axis == Axis::Chunk && !kMoeChunkTwoCards) {
                alt.why_code = "default";
                alt.why_not  = "the default chunk stands (the MoE chunk shares rest on one lab card)";
            } else if (!choice.by_worth) {
                alt.why_code = "default";
                alt.why_not  = "the family's default stands (no decision constants for this generation)";
            } else {
                alt.why_code = "lower_worth";
                alt.why_not  = "worth " + fixed(s.worth - chosen.worth, 3) + " against the chosen";
            }
            out.push_back(std::move(alt));
        }
    }
    return out;
}

} // namespace ninferno::product::choice
