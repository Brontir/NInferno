#include "product/choice/choice.h"

#include "core/generation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <vector>

namespace ninferno::product::choice {

std::string_view generation_of(const CardFacts& card) noexcept {
    return generation_row(card.major, card.minor).name;
}

namespace {

// The generation row's tensor rate per SM and clock for a route (Fp4: twice the int8 rate). A
// generation without an int8 MMA (Volta) takes its BF16 rate for the int8 route's compute.
double peak(const CardFacts& card, Route route) noexcept {
    const GenerationRow& row = generation_row(card.major, card.minor);
    const double s8          = row.s8 > 0 ? row.s8 : row.bf16;
    const double rate        = route == Route::Bf16 ? row.bf16 : route == Route::Fp4 ? 2 * s8 : s8;
    return card.sms * card.clock_hz * rate;
}

// A decode step's int8 compute rate: the int8 MMA's; a generation without one (Volta) takes its
// prefill through the tiled f16 MMA (peak) but decodes on dp4a, 4 int8 multiply-adds a lane each
// clock on its 64 lanes per SM: 512 operations a SM-clock.
double decode_int8_peak(const CardFacts& card) noexcept {
    const GenerationRow& row = generation_row(card.major, card.minor);
    return row.s8 > 0 ? peak(card, Route::Int8) : card.sms * card.clock_hz * kDp4aOpsPerSmClock;
}

// The generation's own row of a per-generation table, else kTransferGeneration's (`transferred`).
template <class Row, std::size_t N>
const Row* generation_row_of(const Row (&rows)[N], std::string_view generation, bool& transferred) noexcept {
    for (const std::string_view gen : {generation, kTransferGeneration}) {
        for (const Row& row : rows) {
            if (row.generation == gen) {
                transferred = transferred || gen != generation;
                return &row;
            }
        }
    }
    transferred = true;
    return nullptr;
}

// A row of kKernelCost or kRowsEta that rests on one card prices only its boards' estimate: marked.
template <class Row, std::size_t N>
const Row* measured_row_of(const Row (&rows)[N], std::string_view generation, bool& transferred) noexcept {
    const Row* row = generation_row_of(rows, generation, transferred);
    if (row != nullptr && row->cards < 2) { transferred = true; }
    return row;
}

double dram_eta(std::string_view generation) noexcept {
    for (const DramEta& row : kDramEtaOf) {
        if (row.generation == generation) { return row.eta; }
    }
    return kDramEta;
}

// The share of the DRAM rate the weight reads of `rows` rows of a weight class reach (kRowsEta:
// linear in log2 rows between the generation's points, kDramEta's at one row, the last point's
// beyond), scaled by the generation's own DRAM share. A generation without rows takes Ada's, a class
// without rows the K-quants' (transferred either way); a row of one card marks it too.
double rows_eta(std::string_view generation, WeightClass weights, double rows, bool& transferred) noexcept {
    double lo_rows = 1, lo_eta = kDramEta;
    std::string_view gen = generation;
    const auto has = [&](std::string_view g, WeightClass w) {
        return std::any_of(std::begin(kRowsEta), std::end(kRowsEta),
                           [&](const RowsEta& r) { return r.generation == g && r.weights == w; });
    };
    if (!has(gen, WeightClass::KQuant)) {
        gen         = kTransferGeneration;
        transferred = transferred || rows > 1;
    }
    WeightClass cls = weights;
    if (!has(gen, cls)) {
        cls         = WeightClass::KQuant;
        transferred = true;
    }
    const double scale = dram_eta(generation) / kDramEta;
    for (const RowsEta& r : kRowsEta) {
        if (r.generation != gen || r.weights != cls) { continue; }
        if (r.cards < 2 && rows > 1) { transferred = true; } // one card: its boards' estimate only
        if (rows <= r.rows) {
            const double t = std::log2(rows / lo_rows) / std::log2(r.rows / lo_rows);
            return scale * (lo_eta + t * (r.eta - lo_eta));
        }
        lo_rows = r.rows;
        lo_eta  = r.eta;
    }
    return scale * lo_eta;
}

// The kernels a step of `rows` rows launches (kDecodeLaunches by the family's layers): the
// family's row for the class; else the family's K-quant row, Qwen 3.5's row for the class, Qwen 3.5's
// K-quant row, in that order (transferred).
struct StepKernels {
    double gemv  = 0;
    double small = 0;
};
StepKernels step_kernels(const ModelFacts& model, WeightClass weights, double rows, bool& transferred) noexcept {
    const auto find = [](std::string_view family, WeightClass w) -> const DecodeLaunches* {
        for (const DecodeLaunches& r : kDecodeLaunches) {
            if (r.family == family && r.weights == w) { return &r; }
        }
        return nullptr;
    };
    const DecodeLaunches* row = find(model.family, weights);
    if (row == nullptr) {
        transferred = true;
        row         = find(model.family, WeightClass::KQuant);
    }
    if (row == nullptr) { row = find(kDecodeLaunches[0].family, weights); }
    if (row == nullptr) { row = &kDecodeLaunches[0]; }
    if (model.layers == 0) { transferred = true; } // unknown layers: the census model's
    // The layers' share scales with the model's layers; the once-a-step kernels do not.
    const double scale = model.layers > 0 ? model.layers / row->layers : 1.0;
    const int at       = rows <= 1 ? 0 : rows <= 4 ? 1 : 2;
    return {row->fixed_gemv + (row->gemv - row->fixed_gemv) * scale,
            row->fixed_small + (row->small[at] - row->fixed_small) * scale};
}

const PrefillEta* prefill_row(bool moe, std::string_view generation, Route route) noexcept {
    for (const PrefillEta& row : kPrefillEta) {
        if (row.moe == moe && row.generation == generation && row.route == route) { return &row; }
    }
    return nullptr;
}

bool has_prefill_rows(std::string_view generation) noexcept {
    return std::any_of(std::begin(kPrefillEta), std::end(kPrefillEta),
                       [&](const PrefillEta& row) { return row.generation == generation; });
}

// The generation whose prefill rows one without its own takes: kTransferGeneration, unless another
// generation with rows runs the same BF16 tensor rate per SM (the same tensor path: Volta's composed
// one is Turing's) and the transfer generation does not.
std::string_view prefill_stand_in(std::string_view generation) noexcept {
    const auto row_of = [](std::string_view name) -> const GenerationRow* {
        for (const GenerationRow& row : kGenerations) {
            if (row.name == name) { return &row; }
        }
        return nullptr;
    };
    const GenerationRow* own      = row_of(generation);
    const GenerationRow* transfer = row_of(kTransferGeneration);
    if (own != nullptr && (transfer == nullptr || transfer->bf16 != own->bf16)) {
        for (const GenerationRow& row : kGenerations) {
            if (row.name != generation && row.bf16 == own->bf16 && has_prefill_rows(row.name)) {
                return row.name;
            }
        }
    }
    return kTransferGeneration;
}

// The prefill efficiency of (kind, generation, route): its own row, else its kind's on the
// transfer generation; a route its kind has no row for takes its kind's int8 row times the dense
// rows' ratio of the two routes (the route's cost is the kernels', not the model kind's); else the
// dense row; another generation's: prefill_stand_in's. `transferred` when not its own.
double prefill_eta(bool moe, std::string_view generation, Route route, bool& transferred) noexcept {
    const Route base            = route == Route::Fp4 ? Route::Int8 : route;
    const std::string_view away = prefill_stand_in(generation);
    transferred                 = transferred || route == Route::Fp4;
    for (const std::string_view gen : {generation, away}) {
        if (const PrefillEta* row = prefill_row(moe, gen, base)) {
            transferred = transferred || gen != generation;
            return row->eta;
        }
    }
    for (const std::string_view gen : {generation, away}) {
        const PrefillEta* own_int8 = prefill_row(moe, gen, Route::Int8);
        if (own_int8 == nullptr) { continue; }
        for (const std::string_view dense_gen : {gen, generation, away}) {
            const PrefillEta* dense_route = prefill_row(false, dense_gen, base);
            const PrefillEta* dense_int8  = prefill_row(false, dense_gen, Route::Int8);
            if (dense_route != nullptr && dense_int8 != nullptr) {
                transferred = true;
                return own_int8->eta * dense_route->eta / dense_int8->eta;
            }
        }
    }
    for (const std::string_view gen : {generation, away}) {
        if (const PrefillEta* row = prefill_row(false, gen, base)) {
            transferred = true;
            return row->eta;
        }
    }
    transferred = true;
    return kPrefillEta[2].eta;
}

double moe_exact_factor(std::string_view generation, bool& transferred) noexcept {
    const RouteFactor* row = generation_row_of(kMoeExactFactor, generation, transferred);
    return row != nullptr ? row->factor : 1.0;
}

double exact_eta(std::string_view generation, bool& transferred) noexcept {
    const RouteEta* row = generation_row_of(kExactComputeEta, generation, transferred);
    return row != nullptr ? row->eta : kExactComputeEta[0].eta;
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

// The measured term of (axis, value) of `family` on `generation`: its own row, else the model
// constant's.
const KldTerm* measured_term(Axis axis, std::string_view value, std::string_view generation,
                             std::string_view family) noexcept {
    const KldTerm* constant = nullptr;
    for (const KldTerm& term : kKldTerms) {
        if (term.family != family || term.axis != axis || term.value != value) { continue; }
        if (term.generation == generation) { return &term; }
        if (term.generation.empty()) { constant = &term; }
    }
    return constant;
}

bool family_measured(std::string_view family) noexcept {
    return std::any_of(std::begin(kKldTerms), std::end(kKldTerms),
                       [&](const KldTerm& t) { return t.family == family; });
}

// The worst measured term of an axis: an unmeasured value never counts as more accurate.
double worst_term(Axis axis) noexcept {
    double worst = 0;
    for (const KldTerm& term : kKldTerms) {
        if (term.axis == axis) { worst = std::max(worst, term.kld); }
    }
    return worst;
}

double attention_eta(std::string_view generation, bool& transferred) noexcept {
    for (const std::string_view gen : {generation, prefill_stand_in(generation)}) {
        for (const GenerationEta& row : kPrefillAttentionEta) {
            if (row.generation == gen) {
                transferred = transferred || gen != generation;
                return row.eta;
            }
        }
    }
    transferred = true;
    return kPrefillAttentionEta[0].eta;
}

// The seconds a prompt of n tokens spends in attention over itself: a token at position t attends
// over t positions on the full layers and min(t, window) on the sliding ones, 2 flops each per
// width, at the BF16 tensor rate.
double attention_seconds(const ModelFacts& model, const CardFacts& card, double n,
                         bool& transferred) noexcept {
    if (n <= 0) { return 0; }
    const double w       = model.window;
    const double full    = n * n / 2;
    const double sliding = w <= 0 || n <= w ? n * n / 2 : n * w - w * w / 2;
    const double flops   = 2 * (model.full_attention_width * full + model.sliding_attention_width * sliding);
    return flops / (attention_eta(generation_of(card), transferred) * peak(card, Route::Bf16));
}

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
// `bits`; below the measured range, the lowest pair's slope; above it, a straight line to 0 at the
// reference's bits; at or above them, 0. The reference's own label is 0 on any model (the curve is
// against it).
double quant_kld(double bits, std::string_view label, bool same_family, bool& measured) {
    measured = false;
    double reference_bits = 0;
    std::vector<const QuantKldPoint*> points;
    for (const QuantKldPoint& p : kQuantKld) {
        if (p.kld <= 0 && p.label == label) {
            measured = same_family && std::abs(p.bits_per_weight - bits) < 0.05;
            return 0;
        }
        // A label's point is the measured file's: the same family at the same bits (another model's
        // Q4_K_M has other bits per weight and takes the interpolation).
        if (same_family && !label.empty() && p.label == label &&
            std::abs(p.bits_per_weight - bits) < 0.05) {
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
    if (const QuantKldPoint& top = *points.back(); bits > top.bits_per_weight) {
        return top.kld * (reference_bits - bits) / (reference_bits - top.bits_per_weight);
    }
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

// The order among candidates: the larger worth, then the longer own context, then the
// enumeration order.
bool better(const Scored& a, std::size_t ia, const Scored& b, std::size_t ib) {
    if (std::abs(a.worth - b.worth) > kWorthTie) { return a.worth > b.worth; }
    if (a.own_context != b.own_context) { return a.own_context > b.own_context; }
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
    if (c.drafter > 0 && c.drafter <= model.drafters.size()) { s.drafter = model.drafters[c.drafter - 1].label; }
    s.own_context = own_context ? own_context(c) : 0;
    s.speed       = speed_of(model, card, quant, c, task.concurrency, task.prompt + task.answer / 2.0,
                             task.prompt);
    s.accuracy    = accuracy_of(model, quant, c, card);
    s.seconds     = request_seconds(model, card, quant, c, task);
    if (ref) {
        s.decision_seconds = request_seconds(model, ref->card, quant, c, task);
        s.worth = -std::log(s.decision_seconds) - task.accuracy_weight * s.accuracy.kld;
    }
    return s;
}

} // namespace

bool is_qwen_family(const ModelFacts& model) noexcept { return qwen_family(model); }

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

namespace {

// The linears' prefill rate (tokens a second) with chunks of `filled` tokens: an MoE's chunk fills
// its experts only as far as its tokens reach (its share of the saturated rate).
double linear_prefill(const ModelFacts& model, const CardFacts& card, Route route,
                      std::uint32_t filled, bool& transferred) noexcept {
    const double params = model.active_params > 0 ? model.active_params : model.params;
    if (params <= 0) { return 0; }
    double rate = prefill_eta(model.moe(), generation_of(card), route, transferred) *
                  peak(card, route) / (2 * params);
    if (model.moe() && model.experts_per_token > 0) {
        rate *= chunk_share(model, filled);
    }
    return rate;
}

} // namespace

Speed speed_of(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
               const Candidate& candidate, std::uint32_t concurrency, double context,
               std::uint32_t prompt) {
    Speed out;
    const std::string_view gen = generation_of(card);
    const Route route          = route_of(quant, candidate.activations, card);
    const double rows          = std::max<std::uint32_t>(concurrency, 1);
    const double params        = model.active_params > 0 ? model.active_params : model.params;

    bool transferred = false;
    const double kv  = model.kv_bytes_per_token ? model.kv_bytes_per_token(candidate.kv) : 0;
    const double kkv = kv_read_factor(gen, candidate.kv, transferred);
    const double state = model.state_bytes ? model.state_bytes(candidate.state) : 0;
    const double ring  = model.ring_bytes_per_token ? model.ring_bytes_per_token(candidate.ring) : 0;
    const double window_rows = model.window > 0 ? std::min<double>(context, model.window) : 0;
    // A row's own reads: its KV over the context, its recurrent state (read and written), its ring.
    const double row_bytes = context * kv * kkv + 2 * state + window_rows * ring;
    // The exact route's compute beyond the fast route's (choice_constants.h kExactComputeEta).
    const double exact = route == Route::Bf16 && !model.moe() ? rows * 2 * params /
                                                    (exact_eta(gen, transferred) * peak(card, Route::Int8))
                                              : 0;
    if (route == Route::Fp4) { transferred = true; } // decodes as the int8 route (not measured)

    double step = 0;
    if (model.moe()) {
        // An MoE (kMoeDecode, fitted on 35B-A3B): the weights a step reads all but an untied input
        // table's rows at the average bits, its routed experts as rows routed apart add them.
        const DecodeConstants& k = kMoeDecode;
        double weights           = quant.weight_bytes - model.input_table_params * quant.bits_per_weight / 8;
        if (model.params > 0) {
            const double routed = std::clamp(model.routed_expert_params / model.params, 0.0, 1.0);
            const double pick   = static_cast<double>(model.experts_per_token) / model.experts;
            const double apart  = 1 - std::pow(1 - pick, rows);
            weights *= (1 - routed) + routed * (pick + kMoeRoutedShare * (apart - pick));
        }
        const double compute = rows * 2 * params /
                               (k.eta_compute * (route == Route::Int8 ? decode_int8_peak(card) : peak(card, Route::Int8))) *
                               (route == Route::Fp4 ? 0.5 : 1.0); // FP4: twice the int8 rate (not measured)
        // The exact route at kMoeExactFactor times its compute (as fitted; kExactComputeEta is the
        // dense route's, measured on 9B).
        const double factor = route == Route::Bf16 ? moe_exact_factor(gen, transferred) : 1.0;
        step = k.step_seconds + (weights + rows * row_bytes) / (k.eta_dram * card.dram_bytes_per_s) + compute * factor;
    } else {
        // A dense model, from facts (choice_constants.h): the weights a step reads (not the input
        // table, of which it gathers one row, nor the MTP layers or own drafter), at the share of the
        // DRAM rate C rows' reads reach, the rows' own reads, the launches; no less than the rows'
        // flops at the prefill route's rate.
        const double table   = model.input_table_bytes > 0 ? model.input_table_bytes
                                                           : model.input_table_params * quant.bits_per_weight / 8;
        const double weights = std::max(0.0, quant.weight_bytes - table);
        // A generation without a row: the one whose launches it shares (Volta: Turing's, neither
        // has PDL nor Ada's L2), else Ada's; transferred either way.
        const KernelCost* cost = measured_row_of(kKernelCost, gen == "volta" ? std::string_view("turing") : gen,
                                                 transferred);
        if (gen == "volta") { transferred = true; }
        if (quant.weights == WeightClass::Other) { transferred = true; } // no census of its routes
        const StepKernels kernels = step_kernels(model, quant.weights, rows, transferred);
        const double launches = cost != nullptr ? kernels.gemv * cost->gemv + kernels.small * cost->small : 0;
        const double reads = weights / (rows_eta(gen, quant.weights, rows, transferred) * card.dram_bytes_per_s) +
                             rows * row_bytes / (dram_eta(gen) * card.dram_bytes_per_s);
        bool prefill_transferred = false;
        const double flops_rate  = linear_prefill(model, card, route, kDefaultChunk, prefill_transferred) * 2 * params;
        const double flops       = flops_rate > 0 ? rows * 2 * params / flops_rate : 0;
        step = std::max(reads + launches + exact, flops);
    }
    out.decode_per_stream = step > 0 && std::isfinite(step) ? 1 / step : 0;
    // A drafter's measured gain over plain decode, while its load table lets these many streams
    // speculate (a gain no measurement gives is not counted: DrafterOption::measured).
    if (candidate.drafter > 0 && candidate.drafter <= model.drafters.size()) {
        const DrafterOption& d = model.drafters[candidate.drafter - 1];
        if (d.measured && (d.plain_from_streams == 0 || rows < d.plain_from_streams)) {
            out.decode_per_stream *= d.speedup;
        }
    }

    bool prefill_transferred = false;
    const std::uint32_t filled = prompt > 0 ? std::min(candidate.chunk, prompt) : candidate.chunk;
    out.prefill = linear_prefill(model, card, route, filled, prefill_transferred);
    if (prompt > 0 && out.prefill > 0) {
        const double attention = attention_seconds(model, card, prompt, prefill_transferred);
        out.prefill            = prompt / (prompt / out.prefill + attention);
    }
    out.transferred = transferred || prefill_transferred;
    return out;
}

namespace {

// A label's tokens (split at '-', '_', '.', ' '), upper-cased.
std::vector<std::string> label_tokens(std::string_view label) {
    std::vector<std::string> out(1);
    for (const char ch : label) {
        if (ch == '-' || ch == '_' || ch == '.' || ch == ' ') {
            if (!out.back().empty()) { out.emplace_back(); }
        } else {
            out.back() += static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        }
    }
    if (out.back().empty()) { out.pop_back(); }
    return out;
}

} // namespace

WeightClass weight_class_of(const std::vector<std::string>& formats, std::string_view label) {
    if (!formats.empty()) {
        bool k = false, q8 = false, iq4 = false, other = false;
        for (const std::string& f : formats) {
            if (f == "ggml_iq4_xs" || f == "ggml_iq4_nl") {
                iq4 = true;
            } else if (f == "ggml_q4_k" || f == "ggml_q5_k" || f == "ggml_q6_k") {
                k = true;
            } else if (f == "q8_g32_fp16") {
                q8 = true;
            } else if (f != "bf16" && f != "f16" && f != "ggml_f16" && f != "fp32" && f != "int32") {
                other = true;
            }
        }
        return other ? WeightClass::Other
               : iq4 ? WeightClass::IQ4
               : k   ? WeightClass::KQuant
               : q8  ? WeightClass::Q8
                     : WeightClass::Other;
    }
    // A label: Q8 / Q8_0 (not Q8_KV), or a Q4_K / Q5_K / Q6_K mix (Q4_K_M, UD-Q5_K_XL, Q6_K).
    const std::vector<std::string> t = label_tokens(label);
    for (std::size_t i = 0; i < t.size(); ++i) {
        const bool next_k = i + 1 < t.size() && t[i + 1] == "K";
        // Not another engine's repacked layouts (Q4_K_R4, Q8_K_R8): their routes are not ours.
        const bool repacked = i + 2 < t.size() && t[i + 2].size() >= 2 && t[i + 2][0] == 'R' &&
                              std::isdigit(static_cast<unsigned char>(t[i + 2][1]));
        if ((t[i] == "Q4" || t[i] == "Q5" || t[i] == "Q6") && next_k && !repacked) { return WeightClass::KQuant; }
        if (t[i] == "IQ4" && i + 1 < t.size() && (t[i + 1] == "XS" || t[i + 1] == "NL") &&
            !(i + 2 < t.size() && t[i + 2].size() >= 2 && t[i + 2][0] == 'R')) {
            return WeightClass::IQ4;
        }
        if (t[i] == "Q8") {
            // Q8, Q8_0; unsloth's Q8_K_XL / _L / _M / _S (Q8_0 with 16-bit tensors), not Q8_KV or Q8_K_R8.
            const std::string next  = i + 1 < t.size() ? t[i + 1] : "";
            const std::string after = i + 2 < t.size() ? t[i + 2] : "";
            if (next.empty() || next == "0" ||
                (next == "K" && (after == "XL" || after == "L" || after == "M" || after == "S"))) {
                return WeightClass::Q8;
            }
        }
    }
    return WeightClass::Other;
}

double worth_band(const CardFacts& card) noexcept {
    const std::string_view gen = generation_of(card);
    for (const PredictionError& e : kDecodePredictionError) {
        if (e.generation == gen) { return e.rms; }
    }
    return 0;
}

double request_seconds(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                       const Candidate& candidate, const Task& task) {
    const double middle = task.prompt + task.answer / 2.0;
    const Speed speed   = speed_of(model, card, quant, candidate, task.concurrency, middle, task.prompt);
    if (speed.prefill <= 0 || speed.decode_per_stream <= 0) { return INFINITY; }
    // The prompt's linears run as whole chunks and a final partial one, each at its own rate (an
    // MoE's prefill depends on how full a chunk is); its attention runs over the whole prompt.
    const Route route         = route_of(quant, candidate.activations, card);
    const std::uint32_t chunk = std::max<std::uint32_t>(candidate.chunk, 1);
    const std::uint32_t whole = task.prompt / chunk * chunk;
    const std::uint32_t rest  = task.prompt - whole;
    bool transferred          = false;
    double prefill_seconds    = attention_seconds(model, card, task.prompt, transferred);
    if (whole > 0) { prefill_seconds += whole / linear_prefill(model, card, route, chunk, transferred); }
    if (rest > 0) { prefill_seconds += rest / linear_prefill(model, card, route, rest, transferred); }
    return prefill_seconds + task.answer / speed.decode_per_stream;
}

bool accuracy_decides(std::string_view generation, std::string_view family, std::string* why) {
    const auto say = [&](std::string text) {
        if (why) { *why = std::move(text); }
        return false;
    };
    if (!family_measured(family)) {
        return say("no accuracy terms were measured on " + std::string(family));
    }
    for (const KldTerm& term : kKldTerms) {
        if (term.family == family && term.axis == Axis::Kv && term.cards < 2) {
            return say("the KV codecs' accuracy terms rest on one lab card");
        }
    }
    const KldTerm* act = measured_term(Axis::Activations, "int8", generation, family);
    if (act == nullptr) {
        return say("the int8 activations' accuracy on " + std::string(generation) + " is not measured");
    }
    if (act->cards < 2) {
        return say("the int8 activations' accuracy on " + std::string(generation) + " rests on one lab card");
    }
    return true;
}

std::optional<Reference> reference_of(const CardFacts& card, const Gates& gates) {
    const std::string_view gen = generation_of(card);
    for (const ReferenceCard& r : kReferenceCards) {
        if (r.generation != gen) { continue; }
        Reference out;
        out.card          = {card.major, card.minor, r.sms, r.clock_hz, r.dram_bytes_per_s, card.fp4};
        out.decides       = r.kv_two_cards && r.exact_two_cards;
        out.chunk_decides = gates.moe_chunk;
        if (!out.decides) { out.why = std::string(gen) + "'s KV and exact-route factors rest on one lab card"; }
        return out;
    }
    return std::nullopt;
}

bool decides_switches(const ModelFacts& model, const CardFacts& card, const Gates& gates) {
    const auto ref = reference_of(card, gates);
    // A gate opens the terms a family has; it never lends another family's.
    if (!ref || !ref->decides || !family_measured(model.family)) { return false; }
    return gates.accuracy ? *gates.accuracy : accuracy_decides(generation_of(card), model.family);
}

bool weighs_quantizations(const ModelFacts& model, const CardFacts& card, const Gates& gates) {
    return decides_switches(model, card, gates) && model.family == kQuantKldFamily &&
           (gates.accuracy ? *gates.accuracy : kQuantKldCards >= 2);
}

Accuracy accuracy_of(const ModelFacts& model, const QuantOption& quant, const Candidate& candidate,
                     const CardFacts& card) {
    Accuracy out;
    const bool other_family    = !family_measured(model.family);
    const bool other_curve     = model.family != kQuantKldFamily;
    const std::string_view gen = generation_of(card);
    // A family without terms of its own borrows the default family's (an estimate to show only).
    const std::string_view family = other_family ? kKldDefaultFamily : std::string_view(model.family);
    const auto add = [&](Axis axis, std::string value, double kld, bool measured, std::string source) {
        const bool transferred = (axis == Axis::Quant ? other_curve : other_family) && kld > 0;
        out.terms.push_back({axis, std::move(value), kld, measured, transferred, std::move(source)});
    };
    // The quantization: the measured curve (choice_constants.h).
    // Unknown bits (0) have no place on the curve: no term, and the quantization stays the rule's
    // (plan_quants weighs the quantizations only where every row's bits are known).
    bool measured  = false;
    const double q = quant.bits_per_weight > 0
                         ? quant_kld(quant.bits_per_weight, quant.label, !other_curve, measured)
                         : 0.0;
    add(Axis::Quant, quant.label, q, measured,
        quant.bits_per_weight <= 0 ? std::string("unknown bits per weight: not weighed")
        : measured ? std::string(kQuantKldSource)
                   : "interpolated by bits per weight on " + std::string(kQuantKldSource));

    const std::string kv{kv_cache_storage_name(candidate.kv)};
    if (candidate.kv == KvCacheStorage::BFloat16) {
        add(Axis::Kv, kv, 0, true, "the reference");
    } else if (const KldTerm* term = measured_term(Axis::Kv, kv, gen, family)) {
        add(Axis::Kv, kv, term->kld, true, std::string(term->source));
    } else {
        add(Axis::Kv, kv, worst_term(Axis::Kv), false, "not measured: the axis's worst");
    }
    if (model.window > 0) {
        const std::string ring{kv_cache_storage_name(candidate.ring)};
        if (candidate.ring == KvCacheStorage::BFloat16) {
            add(Axis::Ring, ring, 0, true, "the reference");
        } else if (const KldTerm* term = measured_term(Axis::Ring, ring, gen, family)) {
            add(Axis::Ring, ring, term->kld, true, std::string(term->source));
        } else {
            const KldTerm* proxy = measured_term(Axis::Kv, ring, gen, family);
            add(Axis::Ring, ring, proxy ? proxy->kld : worst_term(Axis::Kv), false,
                "not measured: the KV codec's term stands in");
        }
    }
    if (model.recurrent) {
        const std::string state{recurrent_state_storage_name(candidate.state)};
        if (candidate.state == RecurrentStateStorage::FP32) {
            add(Axis::State, state, 0, true, "the reference");
        } else if (const KldTerm* term = measured_term(Axis::State, state, gen, family)) {
            add(Axis::State, state, term->kld, true, std::string(term->source));
        } else {
            add(Axis::State, state, worst_term(Axis::State), false, "not measured: the axis's worst");
        }
    }
    const Route route = route_of(quant, candidate.activations, card);
    if (route == Route::Bf16) {
        add(Axis::Activations, "bf16", 0, true, "the reference");
    } else if (route == Route::Int8) {
        // The generation's own row, else the model constant's; none: the axis's worst.
        const KldTerm* term = measured_term(Axis::Activations, "int8", gen, family);
        if (term != nullptr) {
            add(Axis::Activations, "int8", term->kld, true, std::string(term->source));
        } else {
            add(Axis::Activations, "int8", worst_term(Axis::Activations), false,
                "not measured on " + std::string(gen) + ": the axis's worst");
        }
    } else {
        add(Axis::Activations, "fp4", worst_term(Axis::Activations), false,
            "not measured: the axis's worst");
    }
    // The switches' terms were measured against one quantization and combine as the root of the
    // sum of their squares (the product default: measured 0.0040, root sum square 0.0044); how the
    // quantization's own error combines with them is not measured, so it adds on top (the
    // conservative side: a coarse quantization never makes a switch's error look free).
    double switches = 0;
    double own      = 0;
    for (const Term& term : out.terms) {
        if (term.axis == Axis::Quant) {
            own += term.kld;
        } else {
            switches += term.kld * term.kld;
        }
    }
    out.kld = own + std::sqrt(switches);
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

Choice choose(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
              const Allowed& allowed, const Pins& pins, const Task& task,
              const OwnContextFn& own_context, const Gates& gates) {
    Choice out;
    if (pins.activations == ActivationMode::Exact && model.recurrent && pins.state &&
        *pins.state != RecurrentStateStorage::FP32) {
        out.why = "--activations exact refuses the rounded recurrent state pinned";
        return out;
    }
    const std::optional<Reference> ref = reference_of(card, gates);
    out.by_worth       = decides_switches(model, card, gates);
    out.chunk_by_speed = ref && ref->chunk_decides && model.moe() && !pins.chunk;
    const std::vector<KvCacheStorage> kvs =
        pins.kv ? std::vector<KvCacheStorage>{*pins.kv} : allowed.kv;
    const std::vector<ActivationMode> modes =
        pins.activations ? std::vector<ActivationMode>{*pins.activations} : allowed.activations;
    // The chunk: pinned, else an MoE's measured widths (only the default eligible unless the
    // shares decide), else the default.
    std::vector<std::uint32_t> chunks;
    if (pins.chunk) {
        chunks = {*pins.chunk};
    } else if (model.moe()) {
        chunks.assign(std::begin(kMoeChunks), std::end(kMoeChunks));
    } else {
        chunks = {kDefaultChunk};
    }
    const std::uint32_t default_chunk = pins.chunk.value_or(kDefaultChunk);
    // The drafter: pinned (--spec given), else plain decode and each the model carries.
    std::vector<std::size_t> drafters;
    if (pins.drafter) {
        drafters.push_back(*pins.drafter);
    } else {
        for (std::size_t d = 0; d <= model.drafters.size(); ++d) { drafters.push_back(d); }
    }
    for (const KvCacheStorage kv : kvs) {
        for (const ActivationMode mode : modes) {
            // --activations exact refuses a rounded recurrent state the user pinned.
            if (mode == ActivationMode::Exact && model.recurrent && pins.state &&
                *pins.state != RecurrentStateStorage::FP32) {
                continue;
            }
            const Coupled coupled = coupled_storages(model, kv, mode, pins);
            for (const std::size_t drafter : drafters) {
                for (const std::uint32_t chunk : chunks) {
                    out.scored.push_back(score(model, card, ref, quant,
                                               {kv, coupled.ring, coupled.state, mode, chunk, drafter},
                                               task, own_context));
                }
            }
        }
    }
    // The family's defaults: the automatic KV codec and --activations fast, unless pinned.
    const KvCacheStorage default_kv          = pins.kv.value_or(allowed.automatic_kv);
    const ActivationMode default_activations = pins.activations.value_or(ActivationMode::Fast);
    // A candidate the selection may take: any under the worth; only the family's defaults else.
    // The chunk is free where its shares decide (it has no accuracy term: the worth among chunks is
    // their speed on the reference card), else the default.
    // Where the worth does not decide, the drafter is the server's resolved profile's.
    // A drafter is weighed only where every option's gain is measured: one published without its
    // speeds would weigh a gain of 1 and lose to plain decode for its memory alone.
    const std::size_t default_drafter = pins.drafter.value_or(model.default_drafter);
    out.drafter_by_worth = out.by_worth && !pins.drafter &&
                           std::all_of(model.drafters.begin(), model.drafters.end(),
                                       [](const DrafterOption& d) { return d.measured; });
    const auto eligible = [&](const Scored& s) {
        const bool chunk_ok   = out.chunk_by_speed || s.candidate.chunk == default_chunk;
        const bool drafter_ok = out.drafter_by_worth || s.candidate.drafter == default_drafter;
        return chunk_ok && drafter_ok &&
               (out.by_worth || (s.candidate.kv == default_kv &&
                                 s.candidate.activations == default_activations));
    };
    // Each stream's own context in turn (a pinned one, or Auto's steps): the first that holds an
    // eligible candidate takes its best.
    std::optional<std::size_t> chosen;
    std::uint32_t settled = 0;
    for (const std::uint32_t context : task.contexts) {
        for (std::size_t i = 0; i < out.scored.size(); ++i) {
            const Scored& s = out.scored[i];
            if (!eligible(s) || s.own_context < context) { continue; }
            if (!chosen || better(s, i, out.scored[*chosen], *chosen)) { chosen = i; }
        }
        if (chosen) {
            settled = context;
            break;
        }
    }
    if (!chosen) {
        out.why = "no candidate holds " + std::to_string(task.contexts.empty() ? 0 : task.contexts.back()) +
                  " tokens for each of " + std::to_string(task.concurrency) + " streams";
        return out;
    }
    // Why the family's defaults stand, where they do.
    std::string defaults_why;
    if (!ref) {
        defaults_why = std::string(generation_of(card)) + " has no reference card";
    } else if (!ref->decides) {
        defaults_why = ref->why;
    } else if (accuracy_decides(generation_of(card), model.family, &defaults_why)) {
        // The terms rest on two cards (no reason written): a closed Gates::accuracy kept them out.
        defaults_why = "the accuracy gate is closed";
    }
    out.chosen  = chosen;
    out.context = settled;
    for (Scored& s : out.scored) { s.fits = s.own_context >= settled; }
    const Scored& c = out.scored[*chosen];
    out.why = quant.label + ": every stream holds " + std::to_string(settled) + " tokens; KV " + std::string(kv_cache_storage_name(c.candidate.kv)) + ", activations " +
              activation_name(c.candidate.activations) +
              (out.by_worth ? ": the best worth on the " + std::string(generation_of(card)) +
                                  " reference card (" + fixed(c.decision_seconds, 2) + " s a " +
                                  std::to_string(task.prompt) + "+" + std::to_string(task.answer) +
                                  "-token request there, KLD " + fixed(c.accuracy.kld, 4) + ")"
                            : ": the family's defaults (" + defaults_why + ")") +
              (out.chunk_by_speed ? "; chunk " + std::to_string(c.candidate.chunk) +
                                        ": the fastest prefill on the reference card"
                                  : "") +
              (model.drafters.empty() ? std::string() : "; speculation " + c.drafter);
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
        Scored s  = score(model, card, ref, quant, v, task, own_context);
        s.variant = true;
        s.fits    = s.own_context >= settled;
        out.scored.push_back(std::move(s));
    }
    return out;
}

std::vector<Alternative> explain(const Choice& choice) {
    std::vector<Alternative> out;
    if (!choice.chosen) { return out; }
    const Scored& chosen = choice.scored[*choice.chosen];
    const Candidate& c   = chosen.candidate;
    const auto value_of = [&](const Scored& s, Axis axis) -> std::string {
        switch (axis) {
        case Axis::Kv: return std::string(kv_cache_storage_name(s.candidate.kv));
        case Axis::Ring: return std::string(kv_cache_storage_name(s.candidate.ring));
        case Axis::State: return std::string(recurrent_state_storage_name(s.candidate.state));
        case Axis::Activations: return activation_name(s.candidate.activations);
        case Axis::Chunk: return std::to_string(s.candidate.chunk);
        case Axis::Drafter: return s.drafter;
        case Axis::Quant: break; // one quantization: not an axis of the choice
        }
        return "";
    };
    // The chosen candidate but for `axis`: a main axis keeps the other main axes (its coupled
    // storages follow it: exact takes the FP32 state), a coupled one keeps everything else.
    const auto same_but = [&](const Candidate& x, Axis axis) {
        const bool main = (axis == Axis::Kv || x.kv == c.kv) &&
                          (axis == Axis::Activations || x.activations == c.activations) &&
                          (axis == Axis::Chunk || x.chunk == c.chunk) &&
                          (axis == Axis::Drafter || x.drafter == c.drafter);
        if (axis != Axis::State && axis != Axis::Ring) { return main; }
        return main && (axis == Axis::State || x.state == c.state) && (axis == Axis::Ring || x.ring == c.ring);
    };
    for (const Axis axis : {Axis::Kv, Axis::Activations, Axis::Drafter, Axis::Chunk, Axis::State,
                            Axis::Ring}) {
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
                // A faster drafter the asked context and streams leave out: said with its gain.
                if (axis == Axis::Drafter && s.speed.decode_per_stream > chosen.speed.decode_per_stream) {
                    alt.why_not += ", though it decodes x" +
                                   fixed(s.speed.decode_per_stream / chosen.speed.decode_per_stream, 2) +
                                   " as fast";
                }
            } else if (coupled) {
                alt.why_code = "coupled";
                alt.why_not  = "follows the KV codec and the activations (pin it with its flag)";
            } else if (axis == Axis::Drafter && !choice.drafter_by_worth) {
                alt.why_code = choice.by_worth ? "unmeasured" : "default";
                alt.why_not  = choice.by_worth
                                   ? "the speculation profile's choice stands (its gain is not measured on this card)"
                                   : "the speculation profile's choice stands (no decision constants for this generation)";
            } else if (axis == Axis::Chunk ? !choice.chunk_by_speed : !choice.by_worth) {
                alt.why_code = "default";
                alt.why_not  = axis == Axis::Chunk
                                   ? "the default chunk stands (no measured chunk shares decide here)"
                                   : "the family's default stands (no decision constants for this generation)";
            } else if (std::abs(s.worth - chosen.worth) <= kWorthTie && s.own_context != chosen.own_context) {
                alt.why_code = "tie_context";
                alt.why_not  = "the same worth: the chosen keeps more context per stream";
            } else if (std::abs(s.worth - chosen.worth) <= kWorthTie) {
                alt.why_code = "tie_order";
                alt.why_not  = axis == Axis::Activations
                                   ? "the same in every respect (fast and int8 take one route on these weights): the listing's order"
                                   : "the same in every respect for this request: the listing's order";
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
