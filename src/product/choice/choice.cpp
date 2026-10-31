#include "product/choice/choice.h"

#include "core/generation.h"
#include "core/measured_boards.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <iomanip>
#include <span>
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

// An MoE's routed banks' share of the DRAM rate at `rows` concurrent rows (kMoeRoutedEta: linear
// in log2 rows from kDramEta at one row), scaled by the generation's own DRAM share; a generation
// without rows takes Ada's, a row of one card marks it (transferred either way).
double moe_routed_eta(std::string_view generation, double rows, bool& transferred) noexcept {
    std::string_view gen = generation;
    if (std::none_of(std::begin(kMoeRoutedEta), std::end(kMoeRoutedEta),
                     [&](const MoeRoutedEta& r) { return r.generation == gen; })) {
        gen         = kTransferGeneration;
        transferred = transferred || rows > 1;
    }
    const double scale = dram_eta(generation) / kDramEta;
    double lo_rows = 1, lo_eta = kDramEta;
    for (const MoeRoutedEta& r : kMoeRoutedEta) {
        if (r.generation != gen) { continue; }
        if (r.cards < 2 && rows > 1) { transferred = true; }
        if (rows <= r.rows) {
            const double t = std::log2(rows / lo_rows) / std::log2(r.rows / lo_rows);
            return scale * (lo_eta + t * (r.eta - lo_eta));
        }
        lo_rows = r.rows;
        lo_eta  = r.eta;
    }
    return scale * lo_eta;
}

// The kernels a step of `rows` rows launches (kDecodeLaunches by the family's layers): an MoE's
// family's MoE row (its graph whatever the quantization); else the family's row for the class; else
// the family's K-quant row, the family's other row, Qwen 3.5's row for the class, Qwen 3.5's
// first row, in that order (transferred).
struct StepKernels {
    double gemv  = 0;
    double small = 0;
};
// A row's kernels at the model's layers: the layers' share scales; the once-a-step kernels do not.
StepKernels kernels_of(const ModelFacts& model, const DecodeLaunches& row, double rows, bool& transferred) noexcept {
    if (model.layers == 0) { transferred = true; } // unknown layers: the census model's
    const double scale = model.layers > 0 ? model.layers / row.layers : 1.0;
    const int at       = rows <= 1 ? 0 : rows <= 4 ? 1 : 2;
    return {row.fixed_gemv + (row.gemv - row.fixed_gemv) * scale,
            row.fixed_small + (row.small[at] - row.fixed_small) * scale};
}

StepKernels step_kernels(const ModelFacts& model, WeightClass weights, double rows, bool& transferred) noexcept {
    const auto find = [](std::string_view family, WeightClass w) -> const DecodeLaunches* {
        for (const DecodeLaunches& r : kDecodeLaunches) {
            if (!r.moe && r.family == family && r.weights == w) { return &r; } // a dense model's rows
        }
        return nullptr;
    };
    if (model.moe()) {
        for (const DecodeLaunches& r : kDecodeLaunches) {
            if (r.moe && r.family == model.family) { return kernels_of(model, r, rows, transferred); }
        }
        transferred = true; // an MoE without its graph's census: a dense row's launches
    }
    const DecodeLaunches* row = find(model.family, weights);
    if (row == nullptr) {
        transferred = true;
        row         = find(model.family, WeightClass::KQuant);
    }
    for (const DecodeLaunches& r : kDecodeLaunches) {
        if (row == nullptr && !r.moe && r.family == model.family) { row = &r; }
    }
    if (row == nullptr) { row = find(kDecodeLaunches[0].family, weights); }
    if (row == nullptr) { row = &kDecodeLaunches[0]; }
    return kernels_of(model, *row, rows, transferred);
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
// dense row, an MoE's times its kind's ratio to the dense row on the transfer generation (the
// grouped GEMMs' cost against the dense ones' is the kernels', as above); another generation's:
// prefill_stand_in's. `transferred` when not its own.
double prefill_eta(bool moe, std::string_view generation, Route route, bool& transferred) noexcept {
    const Route base            = route == Route::Fp4 ? Route::Int8 : route;
    const std::string_view away = prefill_stand_in(generation);
    transferred                 = transferred || route == Route::Fp4;
    // A generation's own rows first, its route's or its int8 row at the dense rows' ratio (Volta's
    // exact route: its int8 row times Turing's ratio, not Turing's exact row); then the stand-in's.
    for (const std::string_view gen : {generation, away}) {
        if (const PrefillEta* row = prefill_row(moe, gen, base)) {
            transferred = transferred || gen != generation || row->cards < 2;
            return row->eta;
        }
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
    const PrefillEta* transfer_moe   = prefill_row(true, kTransferGeneration, Route::Int8);
    const PrefillEta* transfer_dense = prefill_row(false, kTransferGeneration, Route::Int8);
    const double kind = moe && transfer_moe != nullptr && transfer_dense != nullptr
                            ? transfer_moe->eta / transfer_dense->eta
                            : 1.0;
    if (moe && kind != 1.0) {
        // The dense model's efficiency on this route, as a dense model resolves it (its own int8
        // row at the dense ratio before a stand-in's route row), at the kind's ratio.
        bool dense_transferred = false;
        transferred            = true;
        return prefill_eta(false, generation, route, dense_transferred) * kind;
    }
    for (const std::string_view gen : {generation, away}) {
        if (const PrefillEta* row = prefill_row(false, gen, base)) {
            transferred = true;
            return row->eta * kind;
        }
    }
    transferred = true;
    return kPrefillEta[2].eta;
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
                transferred = transferred || gen != generation || row.cards < 2;
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
        // A point on fewer cards than kQuantKldCards is neither the label's value nor a curve
        // point: an output-affecting choice rests on two cards (AGENTS.md). The reference is 0 by
        // definition.
        if (p.kld > 0 && p.cards < kQuantKldCards) { continue; }
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

// A dense model's int8 route on the int8 tiles (Q8 weights): the tile's clock count at the model's
// hidden width (2 x 64 x 64 operations a tile over c0 + K c1 SM-clocks, against the s8 rate a
// SM-clock), Turing's measured row, else the generation's int8 row, transferred.
double tiles_eta(const CardFacts& card, double hidden, bool& transferred) noexcept {
    const std::string_view generation = generation_of(card);
    for (const TilesEta& row : kPrefillTilesEta) {
        if (row.generation != generation) { continue; }
        transferred = transferred || row.cards < 2; // one card: its boards' estimate only
        const double s8 = generation_row(card.major, card.minor).s8;
        if (row.c1 > 0 && hidden > 0 && s8 > 0) {
            return 2.0 * 64 * 64 / ((row.c0 / hidden + row.c1) * s8);
        }
        if (row.eta > 0) { return row.eta; }
        break;
    }
    transferred = true;
    return prefill_eta(false, generation, Route::Int8, transferred);
}

// A chunk of `tokens` on the int8 route by its tiles and the small kernels beside them (DECISIONS 66:
// prefill_linears_seconds, prefill_small_seconds), seconds; 0 where either has no row (the shares
// below stand). One card's census of the small kernels prices only its own boards.
double tiled_chunk_seconds(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                           Route route, double tokens, bool& transferred) noexcept;

// The linears' prefill rate (tokens a second) with chunks of `filled` tokens: by the tiles where
// they have a row (`tiles`; decode's compute bound keeps the shares), else the generation's share of
// the peak, an MoE's chunk filling its experts only as far as its tokens reach (its share of the
// saturated rate).
double linear_prefill(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                      Route route, std::uint32_t filled, bool& transferred, bool tiles = true) noexcept {
    if (tiles && filled > 0) {
        const double seconds = tiled_chunk_seconds(model, card, quant, route, filled, transferred);
        if (seconds > 0) { return filled / seconds; }
    }
    const double params = model.active_params > 0 ? model.active_params : model.params;
    if (params <= 0) { return 0; }
    const std::string_view gen = generation_of(card);
    const double eta           = route == Route::Int8 && quant.weights == WeightClass::Q8 && !model.moe()
                                     ? tiles_eta(card, model.hidden, transferred)
                                     : prefill_eta(model.moe(), gen, route, transferred);
    double rate = eta * peak(card, route) / (2 * params);
    if (model.moe() && model.experts_per_token > 0) {
        rate *= chunk_share(model, filled);
    }
    return rate;
}

// A label's tokens (weight_class_of reads them too).
std::vector<std::string> label_tokens(std::string_view label);

// A dense model's Q8_0 / K-quant linears on Volta's CUTLASS route (kVoltaPrefillCost); 0 for another
// class or a K-quant label without its format.
double volta_linears_seconds(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                             double tokens, bool* transferred) noexcept {
    const VoltaPrefillCost& cost = kVoltaPrefillCost;
    double decode                = 0;
    if (quant.weights == WeightClass::Q8) {
        decode = cost.q8;
    } else if (quant.weights == WeightClass::KQuant) {
        const std::vector<std::string> t = label_tokens(quant.label);
        for (std::size_t i = 0; i + 1 < t.size(); ++i) {
            if (t[i + 1] != "K") { continue; }
            if (t[i] == "Q4") { decode = cost.q4k; }
            if (t[i] == "Q5") { decode = cost.q5k; }
            if (t[i] == "Q6") { decode = cost.q6k; }
        }
    }
    if (decode <= 0) { return 0; }
    // CUTLASS is built for sm_70 alone (CMakeLists.txt): sm_72's tree runs the tiles and the scalar path.
    if (card.major != 7 || card.minor != 0 || tokens < kVoltaCutlassMinTokens) { return 0; }
    if (decode != cost.q6k && tokens < kVoltaCopyMinTokens && card.dram_bytes_per_s < kVoltaCopyMinDramBps) {
        return 0;
    }
    if (transferred != nullptr && cost.cards < 2) { *transferred = true; }
    const double sms = card.sms, slots = 2 * card.sms;
    double clocks = 0, bytes = 0;
    for (const LinearShape& l : model.linears) {
        const double step = std::max(256.0, std::floor(kVoltaChunkBytes / (l.k * 2) / 128) * 128);
        for (double r0 = 0; r0 < l.n; r0 += step) {
            const double rows  = std::min(step, l.n - r0);
            const double tiles = std::ceil(tokens / 128) * std::ceil(rows / 128);
            clocks += l.per_token * (std::ceil(tiles / slots) * 2 * l.k * cost.tile_step + rows * l.k * decode / sms);
        }
        bytes += l.per_token * tokens * l.k * 4; // the cast to FP16
        if (l.gated) {
            bytes += l.per_token * tokens * l.n / 2 * 10;
        } else if (l.n != model.hidden) { // a fused projection's epilogue (a plain one the GEMM stores)
            bytes += l.per_token * tokens * l.n * 6;
        }
    }
    return clocks / card.clock_hz + bytes / (dram_eta("volta") * card.dram_bytes_per_s);
}

// A dense model's K-quant linears on ggml's matrix kernel (kMmqTileCost) where the card's nearest
// measured board takes it, else on the W4A8 tiles (kW4a8KquantCost); 0 where the generation has no
// row.
double mmq_linears_seconds(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                           double tokens, bool* transferred) noexcept {
    const ::ninferno::Board board{card.major, card.minor, card.sms, card.l2_bytes / (1 << 20),
                                card.dram_bytes_per_s / 1e9};
    const bool mmq             = measured::mmq_prefill_pays(board);
    const std::string_view gen = generation_of(card);
    const MmqTileCost* cost    = nullptr;
    for (const MmqTileCost& row : mmq ? std::span<const MmqTileCost>(kMmqTileCost) : std::span<const MmqTileCost>(kW4a8KquantCost)) {
        if (row.generation == gen) { cost = &row; }
    }
    if (cost == nullptr) { return 0; }
    // The label's K-quant format (Q4_K_M: Q4_K), its bits a weight past that format's toward Q6_K.
    struct Format {
        std::string_view name;
        double bits, step;
    };
    const Format formats[] = {{"Q4", 4.5, cost->q4}, {"Q5", 5.5, cost->q5}, {"Q6", 6.5625, cost->q6}};
    const std::vector<std::string> tokens_of = label_tokens(quant.label);
    const Format* format = nullptr;
    for (std::size_t i = 0; i + 1 < tokens_of.size(); ++i) {
        for (const Format& f : formats) {
            if (tokens_of[i] == f.name && tokens_of[i + 1] == "K") { format = &f; }
        }
    }
    if (format == nullptr) { return 0; }
    if (transferred != nullptr && cost->cards < 2) { *transferred = true; }
    const double toward = format->bits < 6.5625
                              ? std::clamp((quant.bits_per_weight - format->bits) / (6.5625 - format->bits), 0.0, 1.0)
                              : 0;
    const double sms = card.sms, step = format->step + (cost->q6 - format->step) * toward;
    double clocks = 0, bytes = 0;
    // The W4A8 tiles from 128 tokens (kW4a8WideMinTokens; a narrower call takes the 64-row, 32-row or
    // vector kernels, not read: the share) on sm_120 (sm_121 runs Q4_K / Q6_K on the classic tile).
    if (!mmq && (tokens < kW4a8KquantTokens || card.major != 12 || card.minor != 0)) { return 0; }
    if (!mmq) { // the W4A8 tiles: a call's grid of 128-token x 64-row tiles, a tile a CTA in rounds
        for (const LinearShape& l : model.linears) {
            const double tiles = std::ceil(l.n / kW4a8KquantRows) * std::ceil(tokens / kW4a8KquantTokens);
            clocks += l.per_token * std::ceil(tiles / sms) * (cost->c0 + l.k * step);
        }
        return clocks / card.clock_hz;
    }
    for (const LinearShape& l : model.linears) {
        // A gated pair runs as two grids of n / 2 rows; its SwiGLU writes one row of the two.
        const double grids = l.gated ? 2 : 1, n = l.n / grids;
        for (double done = 0; done < tokens; done += kMmqSliceTokens) {
            const double t = std::min(kMmqSliceTokens, tokens - done);
            double columns = kMmqTile; // ggml's narrowest column tile with the fewest tiles
            for (const double j : {16.0, 32.0, 64.0}) {
                if (std::ceil(t / j) == std::ceil(t / kMmqTile)) {
                    columns = j;
                    break;
                }
            }
            const double tiles = std::ceil(n / kMmqTile) * std::ceil(t / columns);
            const double waves = std::ceil(tiles / sms);
            const double k_step = l.k * step * (kMmqTileFixed + (1 - kMmqTileFixed) * columns / kMmqTile);
            if (100 * tiles >= 90 * sms * waves) {
                clocks += grids * l.per_token * waves * (cost->c0 + k_step);
            } else { // stream-k: the SMs share the tiles' K steps; the fixup adds the split tiles
                clocks += grids * l.per_token * (cost->c0 + tiles * k_step / sms);
                bytes += grids * l.per_token * sms * kMmqTile * columns * 4 * 2;
            }
            bytes += l.per_token * t * (grids * l.k * kMmqInputBytes + l.n * kMmqOutputBytes + n * kMmqWrittenBytes);
        }
    }
    return clocks / card.clock_hz + bytes / card.dram_bytes_per_s;
}

} // namespace

double prefill_linears_seconds(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                                double tokens, bool* transferred) noexcept {
    if (model.linears.empty() || tokens <= 0 || card.sms <= 0 || card.clock_hz <= 0 ||
        card.dram_bytes_per_s <= 0) {
        return 0;
    }
    const bool moe = model.moe() && model.experts_per_token > 0;
    const std::string_view gen = generation_of(card);
    const auto cost = [gen](TileKernel kernel) -> const TileCost* {
        for (const TileCost& row : kTileCost) {
            if (row.generation == gen && row.kernel == kernel) { return &row; }
        }
        return nullptr;
    };
    // The banks' GEMM (product/plan expert_prefill_of): MXFP4 takes the block-scaled MMA on
    // Blackwell, Ada converts it to int8. The linears beside them take the classic int8 tiles when
    // their weights are Q8_0; a K-quant's take a tile by format and card (generic_gemm.cpp
    // w4a8_ws_wins: Q6_K on Blackwell the 128 x 64 producer-warp tile), not yet priced.
    // A dense model's linears likewise: the int8 tiles for Q8_0 weights (q8_prefill_a8).
    TileKernel banks = TileKernel::Int8;
    switch (moe ? quant.expert_banks : ExpertBanks::Int8) {
    case ExpertBanks::Int8: banks = TileKernel::Int8; break;
    case ExpertBanks::A16: banks = TileKernel::A16; break;
    case ExpertBanks::Mxfp4: banks = gen == "gb20x" ? TileKernel::Mx : TileKernel::Int8; break;
    case ExpertBanks::Unknown: return 0;
    }
    if (!moe && gen == "volta") { return volta_linears_seconds(model, card, quant, tokens, transferred); }
    if (!moe && quant.weights == WeightClass::KQuant) { return mmq_linears_seconds(model, card, quant, tokens, transferred); }
    if ((moe ? quant.rest_weights : quant.weights) != WeightClass::Q8) { return 0; }
    const TileCost* rest = cost(TileKernel::Int8);
    const TileCost* bank = cost(banks);
    if (rest == nullptr || bank == nullptr) { return 0; }
    if (transferred != nullptr && (rest->cards < 2 || (moe && bank->cards < 2))) { *transferred = true; }
    const double picks = moe ? model.experts_per_token : 1;
    const double slots = std::ceil(tokens) * picks;
    const double jobs  = moe ? std::ceil(slots / kExpertJobSlots) + std::min<double>(slots, model.experts) : 0;

    const double sms = card.sms;
    double seconds = 0, hidden = 0, moe_layers = 0;
    for (const LinearShape& l : model.linears) {
        const TileCost& c  = l.expert ? *bank : *rest;
        const double tile  = (c.c0 + l.k * c.c1) / card.clock_hz;
        const double cols  = std::ceil(l.n / 64);
        // A grid's rounds of SMs x resident CTAs, each `resident` tiles' time; a grid of no more CTAs
        // than SMs runs one a SM alone (kLoneTile).
        const double resident = std::max(c.resident, 1);
        const auto waves      = [&](double ctas) {
            return resident > 1 && ctas <= sms ? kLoneTile : std::ceil(ctas / (sms * resident)) * resident;
        };
        if (l.expert && moe) {
            // A layer's bank is one call over its jobs; the gate and up come first (K the hidden).
            const double layers = l.per_token / picks;
            if (hidden == 0) { hidden = l.k, moe_layers = layers; }
            seconds += layers * waves(jobs * cols) * tile;
        } else {
            seconds += l.per_token * waves(std::ceil(tokens / 64) * cols) * tile;
        }
    }
    // The routing gathers every pick's token row into its expert's run and scatters the result
    // back: 16-bit rows of the hidden width, read and written, a layer.
    seconds += moe_layers * tokens * picks * hidden * 2 * 2 / (dram_eta(gen) * card.dram_bytes_per_s);
    return seconds;
}

double prefill_small_seconds(const ModelFacts& model, const CardFacts& card, double tokens) noexcept {
    if (tokens <= 0 || card.sms <= 0 || card.clock_hz <= 0 || card.dram_bytes_per_s <= 0) { return 0; }
    const std::string_view gen = generation_of(card);
    for (const SmallKernelCost& row : kSmallKernelCost) {
        if (row.generation != gen) { continue; }
        const double scan  = tokens * model.recurrent_state_elements * row.scan_clocks / (card.sms * card.clock_hz);
        const double bytes = tokens * model.layers * model.hidden * row.pass_bytes;
        return scan + bytes / card.dram_bytes_per_s;
    }
    return 0;
}

namespace {

// CTAs of `seconds` each, dealt in issue order to `slots` places, each to the one that frees first:
// when the last frees.
double deal(const std::vector<double>& ctas, int slots) {
    std::vector<double> free(static_cast<std::size_t>(std::max(slots, 1)), 0.0);
    for (const double t : ctas) {
        std::pop_heap(free.begin(), free.end(), std::greater<>());
        free.back() += t;
        std::push_heap(free.begin(), free.end(), std::greater<>());
    }
    return *std::max_element(free.begin(), free.end());
}

// Gemma 4's attention by its CTAs (kGemmaAttentionCost); a layer type whose tile was not read keeps
// its share, as attention_seconds prices it.
double gemma_attention_tiled_seconds(const ModelFacts& model, const CardFacts& card, double prompt,
                                     std::uint32_t chunk, bool* transferred) {
    const std::string_view gen = generation_of(card);
    const GemmaAttentionCost* cost = nullptr;
    for (const GemmaAttentionCost& row : kGemmaAttentionCost) {
        if (row.generation == gen) { cost = &row; }
    }
    if (cost == nullptr || chunk == 0) { return 0; }
    if (transferred != nullptr && cost->cards < 2) { *transferred = true; }
    bool share_transferred = false;
    const double share     = attention_eta(gen, share_transferred) * peak(card, Route::Bf16);
    const bool chains      = generation_row(card.major, card.minor).attention_chain;
    const double window    = model.window;
    double seconds         = 0;
    const auto layer_type = [&](bool sliding) {
        const double own      = sliding ? model.sliding_layers : model.full_layers;
        const double attend   = sliding ? model.sliding_attending_layers : model.full_attending_layers;
        const double layers   = attend > 0 ? attend : own;
        const double width  = sliding ? model.sliding_attention_width : model.full_attention_width;
        const double dim    = sliding ? model.sliding_head_dim : model.full_head_dim;
        const double kv     = sliding ? model.sliding_kv_heads : model.full_kv_heads;
        if (layers <= 0 || width <= 0) { return; }
        const double heads = dim > 0 ? std::round(width / (layers * 2 * dim)) : 0;
        const double group = kv > 0 ? heads / kv : 0;
        const bool read    = sliding ? dim == 256 && group >= 1 && group <= 32 : dim == 512 && group == 16;
        if (!read) {
            // attention_seconds' pairs up to x tokens: x^2 / 2, past the window x w - w^2 / 2.
            const auto pairs = [&](double x) {
                return sliding && window > 0 && x > window ? x * window - window * window / 2 : x * x / 2;
            };
            seconds += 2 * width * pairs(prompt) / share;
            if (transferred != nullptr && share_transferred) { *transferred = true; }
            return;
        }
        const double tokens  = std::floor(32 / group); // a block's tokens (block_rows / G)
        const double per_key = (sliding ? cost->sliding : cost->full) / card.clock_hz;
        const double slots   = card.sms * (sliding ? 2.0 : 1.0);
        std::vector<double> ctas;
        for (double prefix = 0; prefix < prompt; prefix += chunk) {
            const double n      = std::min<double>(chunk, prompt - prefix);
            const double blocks = std::ceil(n / tokens) + 1;
            ctas.clear();
            for (double h = 0; h < kv; ++h) {
                for (double b = blocks - 1; b >= 0; --b) {
                    if (b * tokens >= n) { continue; } // the spare block past the rows: empty
                    double keys = prefix + std::min((b + 1) * tokens, n);
                    if (sliding && window > 0) { keys = std::min(keys, window); }
                    ctas.push_back(keys * per_key);
                }
            }
            // The launch's grid counts the spare block (launch.cu chain_pays), and a call of a decode's
            // width never chains (chain_plan).
            const double grid  = blocks * kv;
            const double waves = std::ceil(grid / slots);
            if (!sliding && chains && n > kGemmaChainMinTokens && grid < 0.9 * waves * slots) {
                double work = 0;
                for (const double t : ctas) { work += t; }
                seconds += layers * work / card.sms; // persistent: the work spread over the SMs
            } else {
                seconds += layers * deal(ctas, static_cast<int>(slots));
            }
        }
    };
    layer_type(true);
    layer_type(false);
    return seconds;
}

} // namespace

namespace {

// The INT8 prompt kernel's layout of a call (int8/template_launch.cuh int8_prompt_fast_layout): its
// leading 64-row slots, every slot when `all`.
struct Int8PromptLayout {
    bool all  = false;
    int slots = 0;
};

// The makespan the kernel weighs a layout by, in hundredths of a 128-row CTA (int8_prompt_fast_makespan).
long long int8_prompt_makespan(int wide, int narrow, int sms) {
    std::vector<long long> free(static_cast<std::size_t>(sms), 100LL * (wide / sms));
    for (int sm = 0; sm < wide % sms; ++sm) { free[static_cast<std::size_t>(sm)] += 100; }
    long long end = *std::max_element(free.begin(), free.end());
    for (int i = 0; i < narrow; ++i) {
        const auto sm = std::min_element(free.begin(), free.end());
        *sm += 72;
        end = std::max(end, *sm);
    }
    return end;
}

Int8PromptLayout int8_prompt_layout(int width, int heads, int sms) {
    const auto up = [](int a, int b) { return (a + b - 1) / b; };
    Int8PromptLayout best;
    long long best_cost = int8_prompt_makespan(up(width, 128) * heads, 0, sms);
    const int most      = std::min(up(width, 64), 2 * up(2 * sms, heads));
    for (int slots = 1; slots <= most; ++slots) {
        const long long cost = int8_prompt_makespan(up(std::max(width - slots * 64, 0), 128) * heads, slots * heads, sms);
        if (cost < best_cost) {
            best_cost  = cost;
            best.slots = slots;
        }
    }
    if (72LL * up(up(width, 64) * heads, sms) <= best_cost) { return {true, 0}; }
    return best;
}

// The FP8-KV prompt kernel's chunk of `width` tokens after `prefix` over one layer of `heads`
// heads, and its merge (kFp8AttentionTileCost); seconds.
double fp8_attention_chunk_seconds(const Fp8AttentionTileCost& cost, const CardFacts& card, double heads,
                                   double prefix, double width) {
    const auto up      = [](double a, double b) { return std::ceil(a / b); };
    const double tiles = up(width, 128);
    // mxfp8_tiled_partition: the split count with the fewest waves a split, fewer on a tie.
    const double ctas = heads * tiles;
    double target = 1, waves = up(ctas, card.sms);
    for (double splits = 2; splits <= kFp8AttentionMaxSplits; ++splits) {
        const double next = up(ctas * splits, card.sms);
        if (next * target < waves * splits) {
            target = splits;
            waves  = next;
        }
    }
    const double visible = prefix + width;
    const double active  = std::min(up(visible, kFp8AttentionSplitKeys), target);
    const double keys    = up(visible, kFp8AttentionKeyTile); // the call's key tiles
    std::vector<double> grid;
    for (double z = 0; z < active; ++z) {
        const double first = std::floor(z * keys / active), end = std::floor((z + 1) * keys / active);
        for (double h = 0; h < heads; ++h) {
            for (double x = 0; x < tiles; ++x) {
                const double last   = std::floor((prefix + std::min(x * 128 + 128, width) - 1) / kFp8AttentionKeyTile);
                const double blocks = std::max(0.0, std::min(end, last + 1) - first);
                grid.push_back(cost.cta + blocks * kFp8AttentionKeyTile * cost.key);
            }
        }
    }
    // The merge (causal_tiled_merge.cuh, after every call): each row's partial of every active split
    // (a CTA with no keys stores an empty one), its 256 FP32 sums and the maximum and the sum, in;
    // the BF16 row out.
    const double bytes = width * heads * (active * (256 * 4 + 8) + 256 * 2);
    const bool held    = card.l2_bytes > 0 && bytes < card.l2_bytes;
    return deal(grid, card.sms) / card.clock_hz +
           bytes / (card.dram_bytes_per_s * (held ? kAttentionMergeL2Share : kAttentionMergeDramShare));
}

} // namespace

double prefill_attention_tiled_seconds(const ModelFacts& model, const CardFacts& card, KvCacheStorage kv,
                                       double prompt, std::uint32_t chunk, bool* transferred) {
    if (prompt > 0 && card.sms > 0 && card.clock_hz > 0 && model.family == "gemma4_text") {
        return gemma_attention_tiled_seconds(model, card, prompt, chunk, transferred);
    }
    const bool bf16 = kv == KvCacheStorage::BFloat16;
    const bool fp8  = kv == KvCacheStorage::Fp8E4M3Row256;
    if (prompt <= 0 || chunk == 0 || (kv != KvCacheStorage::Int8Group64 && !bf16 && !fp8) || model.full_layers == 0 ||
        model.full_head_dim != kAttentionTileHeadDim || model.sliding_attention_width > 0 ||
        model.family.rfind("qwen3_5", 0) != 0 || card.sms <= 0 || card.clock_hz <= 0) {
        return 0;
    }
    const std::string_view gen = generation_of(card);
    if (fp8) {
        const Fp8AttentionTileCost* cost = nullptr;
        for (const Fp8AttentionTileCost& row : kFp8AttentionTileCost) {
            if (row.generation == gen) { cost = &row; }
        }
        const double heads =
            std::round(model.full_attention_width / (model.full_layers * 2 * model.full_head_dim));
        // A call of at most kFp8AttentionTiledMinWidth tokens takes the grouped kernels: the share.
        if (cost == nullptr || heads < 1 || card.dram_bytes_per_s <= 0 || prompt <= kFp8AttentionTiledMinWidth) {
            return 0;
        }
        if (transferred != nullptr && cost->cards < 2) { *transferred = true; }
        bool share_transferred = false;
        const double share     = attention_eta(gen, share_transferred) * peak(card, Route::Bf16);
        double seconds         = 0;
        for (double prefix = 0; prefix < prompt; prefix += chunk) {
            const double width = std::min<double>(chunk, prompt - prefix);
            if (width <= kFp8AttentionTiledMinWidth) {
                seconds += 2 * model.full_attention_width * (width * prefix + width * width / 2) / share;
                if (transferred != nullptr && share_transferred) { *transferred = true; }
                continue;
            }
            seconds += model.full_layers * fp8_attention_chunk_seconds(*cost, card, heads, prefix, width);
        }
        return seconds;
    }
    const AttentionTileCost* cost = nullptr;
    for (const AttentionTileCost& row : bf16 ? std::span(kBf16AttentionTileCost) : std::span(kAttentionTileCost)) {
        if (row.generation == gen) { cost = &row; }
    }
    if (cost == nullptr) { return 0; }
    // INT8 keys: 128 rows a CTA issued longest first; BF16 keys: 64 rows in plain launch order.
    const double rows  = bf16 ? kBf16AttentionQueryRows : kAttentionQueryRows;
    const double heads = std::round(model.full_attention_width / (model.full_layers * 2 * model.full_head_dim));
    if (heads < 1) { return 0; }
    // A call of at most kAttentionTiledMinWidth tokens takes the grouped kernels (int8/plan.cpp: the
    // keys split over the SMs), not this one: its share of the BF16 peak, as attention_seconds.
    if (prompt <= kAttentionTiledMinWidth) { return 0; }
    if (transferred != nullptr && cost->cards < 2) { *transferred = true; }
    bool share_transferred = false;
    const double share     = attention_eta(gen, share_transferred) * peak(card, Route::Bf16);
    double clocks = 0, seconds = 0;
    for (double prefix = 0; prefix < prompt; prefix += chunk) {
        const double width  = std::min<double>(chunk, prompt - prefix);
        // BF16 keys of Qwen 3.5's group-8 geometry (16 heads over 2) take 128-row tiles past an 8K
        // prefix (bf16/plan.cpp Tiled128): not read, the share.
        const bool tiled128 = bf16 && heads == 16 && model.full_kv_heads == 2 && width >= 1024 &&
                              std::fmod(width, 128) == 0 && prefix >= 8192;
        if (width <= kAttentionTiledMinWidth || tiled128) {
            seconds += model.full_layers * 2 * (model.full_attention_width / model.full_layers) *
                       (width * prefix + width * width / 2) / share;
            if (transferred != nullptr && share_transferred) { *transferred = true; }
            continue;
        }
        // The kernel's issue order, each CTA to the SM that frees first; a CTA of the rows up to `last`.
        std::vector<double> ctas;
        const auto cta = [&](double last, double scale) {
            return scale * (cost->cta + prefix * cost->prefix + std::min(last, width) * cost->in_chunk);
        };
        if (bf16) {
            const double blocks = std::ceil(width / rows);
            for (double h = 0; h < heads; ++h) {
                for (double b = 0; b < blocks; ++b) { ctas.push_back(cta((b + 1) * rows, 1)); }
            }
        } else { // longest first: the last row block of every head, then the one before
            const Int8PromptLayout layout = int8_prompt_layout(static_cast<int>(width), static_cast<int>(heads), card.sms);
            const double half   = rows / 2;
            const double narrow = layout.all ? std::ceil(width / half) : layout.slots;
            const double wide   = layout.all ? 0 : std::ceil(std::max(width - narrow * half, 0.0) / rows);
            for (double s = wide - 1; s >= 0; --s) {
                for (double h = 0; h < heads; ++h) { ctas.push_back(cta(narrow * half + (s + 1) * rows, 1)); }
            }
            for (double s = narrow - 1; s >= 0; --s) {
                for (double h = 0; h < heads; ++h) { ctas.push_back(cta((s + 1) * half, kAttentionHalfCta)); }
            }
        }
        clocks += deal(ctas, card.sms);
    }
    return model.full_layers * clocks / card.clock_hz + seconds;
}

namespace {

double tiled_chunk_seconds(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                           Route route, double tokens, bool& transferred) noexcept {
    if (route != Route::Int8) { return 0; }
    bool tiles_transferred = false;
    const double linears   = prefill_linears_seconds(model, card, quant, tokens, &tiles_transferred);
    const double small     = linears > 0 ? prefill_small_seconds(model, card, tokens) : 0;
    if (small <= 0) { return 0; }
    transferred = transferred || tiles_transferred;
    const std::string_view gen = generation_of(card);
    for (const SmallKernelCost& row : kSmallKernelCost) {
        if (row.generation == gen) { transferred = transferred || row.cards < 2; }
    }
    return linears + small;
}

} // namespace

namespace {

// The calls a prefill runs after the prompt's chunks (ModelFacts::prompt_tail_calls): each a few
// tokens through every layer, a plain decode step's reads at the prompt's depth (DECISIONS 62:
// two such calls on Qwen3.5 9B, 2 x the weights over the DRAM rate predicted 22.5 / 11.3 / 12.7 ms
// on the RTX 4070 Ti / 4090 / 5070 Ti, the profiled tail 20.0 / 10.4 / 12.6).
double tail_calls_seconds(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                          const Candidate& candidate, double prompt) {
    if (model.prompt_tail_calls == 0) { return 0; }
    Candidate plain = candidate;
    plain.drafter   = 0;
    const double rate = speed_of(model, card, quant, plain, 1, prompt, 0).decode_per_stream;
    return rate > 0 ? model.prompt_tail_calls / rate : 0;
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
    const double exact = route == Route::Bf16 ? rows * 2 * params /
                                                    (exact_eta(gen, transferred) * peak(card, Route::Int8))
                                              : 0;
    if (route == Route::Fp4) { transferred = true; } // decodes as the int8 route (not measured)

    // A model from facts (choice_constants.h): the weights a step reads (not the input table, of
    // which it gathers one row, nor the MTP layers or own drafter plain decode leaves; an MoE's
    // routed banks at the share of them its rows route to, DECISIONS 65), at the share of the DRAM
    // rate C rows' reads reach, the rows' own reads, the launches; no less than the rows' flops at
    // the prefill route's rate.
    const double table = model.input_table_bytes > 0 ? model.input_table_bytes
                                                     : model.input_table_params * quant.bits_per_weight / 8;
    double weights     = std::max(0.0, quant.weight_bytes - table);
    // An MoE's routed banks: the experts any of the rows routes to, each read once, at their own
    // share of the DRAM rate (kMoeRoutedEta). Concurrent rows are separate requests and route
    // independently (1 - (1 - k/E)^C of the banks; the speculation round's correlated rows keep
    // kMoeRoutedShare, spec/round_cost.cpp).
    double routed_read = 0;
    if (model.moe() && model.params > 0) {
        const double routed = quant.routed_expert_bytes > 0
                                  ? std::min(weights, quant.routed_expert_bytes)
                                  : weights * std::clamp(model.routed_expert_params / model.params, 0.0, 1.0);
        const double pick   = std::clamp(static_cast<double>(model.experts_per_token) / model.experts, 0.0, 1.0);
        const double apart  = 1 - std::pow(1 - pick, rows);
        weights -= routed;
        routed_read = routed * apart;
    }
    // A generation without a row: the one whose launches it shares (Volta: Turing's, neither has
    // PDL nor Ada's L2), else Ada's; transferred either way.
    const KernelCost* cost = measured_row_of(kKernelCost, gen == "volta" ? std::string_view("turing") : gen,
                                             transferred);
    if (gen == "volta") { transferred = true; }
    // A dense model of another class: no census of its routes. An MoE's rest reads at the K-quant
    // rows, the rows its routed banks' share was measured against (kMoeRoutedEta).
    if (quant.weights == WeightClass::Other && !model.moe()) { transferred = true; }
    const WeightClass rest = model.moe() ? WeightClass::KQuant : quant.weights;
    const StepKernels kernels = step_kernels(model, quant.weights, rows, transferred);
    const double launches = cost != nullptr ? kernels.gemv * cost->gemv + kernels.small * cost->small : 0;
    double reads = weights / (rows_eta(gen, rest, rows, transferred) * card.dram_bytes_per_s) +
                   rows * row_bytes / (dram_eta(gen) * card.dram_bytes_per_s);
    if (routed_read > 0) { reads += routed_read / (moe_routed_eta(gen, rows, transferred) * card.dram_bytes_per_s); }
    bool flops_transferred = false;
    // Decode never takes the int8 tiles (q8_prefill_a8 is a prefill route): MMQ's rate.
    QuantOption decode_quant = quant;
    decode_quant.weights     = WeightClass::KQuant;
    const double flops_rate  = linear_prefill(model, card, decode_quant, route, kDefaultChunk, flops_transferred, false) * 2 * params;
    const double flops       = flops_rate > 0 ? rows * 2 * params / flops_rate : 0;
    const double step        = std::max(reads + launches + exact, flops);
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
    out.prefill = linear_prefill(model, card, quant, route, filled, prefill_transferred);
    if (prompt > 0 && out.prefill > 0) {
        // Exact activations run every route invariant (ScopedInvariantRoutes): attention on the grouped
        // kernels at every width, never the prompt kernel.
        const bool exact       = candidate.activations == ActivationMode::Exact;
        const double tiled     = exact ? 0 : prefill_attention_tiled_seconds(model, card, candidate.kv, prompt, candidate.chunk,
                                                                   &prefill_transferred);
        const double attention = tiled > 0 ? tiled : attention_seconds(model, card, prompt, prefill_transferred);
        out.prefill            = prompt / (prompt / out.prefill + attention +
                                tail_calls_seconds(model, card, quant, candidate, prompt));
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
        if (t[i] == "IQ4" && i + 1 < t.size() && (t[i + 1] == "XS" || t[i + 1] == "NL") && !repacked) {
            return WeightClass::IQ4;
        }
        if (t[i] == "Q8") {
            // Q8, Q8_0; unsloth's Q8_K_XL / _L / _M / _S (Q8_0 with 16-bit tensors), not Q8_KV or Q8_K_R8.
            const std::string next  = i + 1 < t.size() ? t[i + 1] : "";
            const std::string after = i + 2 < t.size() ? t[i + 2] : "";
            if (repacked) { continue; }
            if (next.empty() || next == "0" ||
                (next == "K" && (after == "XL" || after == "L" || after == "M" || after == "S"))) {
                return WeightClass::Q8;
            }
        }
    }
    return WeightClass::Other;
}

bool class_measured(const CardFacts& card, WeightClass weights) noexcept {
    const std::string_view gen = generation_of(card);
    return std::any_of(std::begin(kRowsEta), std::end(kRowsEta), [&](const RowsEta& r) {
        return r.generation == gen && r.weights == weights && r.cards >= 2;
    });
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
    const double tiled        = candidate.activations == ActivationMode::Exact
                                    ? 0
                                    : prefill_attention_tiled_seconds(model, card, candidate.kv, task.prompt, chunk,
                                                                      &transferred);
    double prefill_seconds    = tiled > 0 ? tiled : attention_seconds(model, card, task.prompt, transferred);
    prefill_seconds += tail_calls_seconds(model, card, quant, candidate, task.prompt);
    if (whole > 0) { prefill_seconds += whole / linear_prefill(model, card, quant, route, chunk, transferred); }
    if (rest > 0) { prefill_seconds += rest / linear_prefill(model, card, quant, route, rest, transferred); }
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
        out.card          = {.major = card.major, .minor = card.minor, .sms = r.sms, .clock_hz = r.clock_hz,
                             .dram_bytes_per_s = r.dram_bytes_per_s, .fp4 = card.fp4};
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
