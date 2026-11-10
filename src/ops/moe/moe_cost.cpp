#include "ops/moe/moe_cost.h"

#include "core/generation.h"
#include "ninferno/ops/linear.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ninferno::ops::detail::moe {

namespace {

// The int8 activation of the R8 gate/up and the W4A8 tier as the consumer reads it: 32 codes and a
// scale a 32-value block (36 bytes), 40 with the block's sum for a format with a min
// (moe_format_has_min, the definition r8_admits' shared-memory bound reads); BF16 otherwise. The
// quantizer itself writes all three (codes, FP32 scale, FP32 sum: 40 bytes a block) whatever the
// format (kQuantizedBlockBytes).
constexpr double kInt8ActivationBytesPerValue       = 36.0 / 32.0;
constexpr double kInt8SummedActivationBytesPerValue = 40.0 / 32.0;
constexpr double kQuantizedBlockBytes               = 40.0;
constexpr std::int32_t kQuantizeBlock               = 32;
constexpr double kBf16BytesPerValue                 = 2.0;
constexpr double kFp32BytesPerValue                 = 4.0;

// Whether the stage reads int8 codes on `kind`: the W4A8 tier (its quantize pass wrote them) and an
// R8 gate/up whose x the router's CTAs quantized. An R8 down stage reads the BF16 intermediate (it
// quantizes it inside the kernel), and so does an R8 gate/up of a padded x (it quantizes in the
// kernel, moe.cpp r8_admits).
bool reads_int8(const MoeStageShape& shape, MoeStageKernel kind) {
    return kind == MoeStageKernel::W4a8 ||
           (kind == MoeStageKernel::R8 && shape.stage == Stage::GateUp && shape.router_quantizes_x);
}

// The bytes a value of the int8 activation takes when read.
double int8_read_bytes_per_value(const MoeStageShape& shape) {
    return moe_format_has_min(shape.qtype) ? kInt8SummedActivationBytesPerValue
                                           : kInt8ActivationBytesPerValue;
}

const Weight& bank_of(const MoeWeights& weights, Stage stage) {
    return stage == Stage::Down ? weights.down : weights.gate_up;
}

} // namespace

std::optional<MoeBoardFacts> moe_board_facts(const DeviceClass& device) {
    if (!device.facts.has_roofline() || device.sms <= 0) return std::nullopt;
    const GenerationRow& row = generation_row(device.major, device.minor);
    MoeBoardFacts out;
    out.dram_bytes_per_s      = device.facts.dram_bytes_per_s();
    out.sms                   = device.sms;
    out.clock_hz              = device.facts.reference_clock_hz();
    out.bf16_ops_per_clock_sm = row.bf16;
    out.s8_ops_per_clock_sm   = row.s8;
    out.bf16_native           = device.features.mma_bf16_native;
    out.generation_known      = generation_known(device.major, device.minor);
    return out;
}

double expected_experts_touched(std::int32_t experts, std::int32_t top_k, std::int32_t tokens) {
    if (experts <= 0 || top_k <= 0 || tokens <= 0) return 0.0;
    const double miss = std::max(0.0, 1.0 - static_cast<double>(top_k) / experts);
    return experts * (1.0 - std::pow(miss, static_cast<double>(tokens)));
}

std::optional<MoeStagePrice> price_moe_stage(const MoeStageShape& shape, MoeStageKernel kind,
                                             std::int32_t tokens, const MoeBoardFacts& board) {
    if (shape.experts <= 0 || shape.top_k <= 0 || shape.rows <= 0 || shape.cols <= 0 ||
        tokens <= 0 || board.dram_bytes_per_s <= 0 || board.sms <= 0 || board.clock_hz <= 0) {
        return std::nullopt;
    }
    const double assignments = static_cast<double>(tokens) * shape.top_k;
    const double touched     = expected_experts_touched(shape.experts, shape.top_k, tokens);
    const bool int8          = reads_int8(shape, kind);
    // The values of a BF16 activation row (x has the router's width, which a padded gate/up's K
    // exceeds); the codes the kernel reads are over the K it walks.
    const double x_cols = static_cast<double>(shape.x_cols > 0 ? shape.x_cols : shape.cols);
    // The W4A8 tier's activation planes are at K rounded up to 128 (moe_w4a8_ld, a zero tail): the
    // quantizer writes them whole and the kernel reads them whole. An R8 gate/up reads codes over
    // K.
    const double code_cols = kind == MoeStageKernel::W4a8
                                 ? static_cast<double>(moe_w4a8_ld(shape.cols))
                                 : static_cast<double>(shape.cols);
    const double act_read =
        int8 ? code_cols * int8_read_bytes_per_value(shape) : x_cols * kBf16BytesPerValue;
    // A gate and up stage reads the T rows of x, which a token's k assignments share; a down stage
    // reads the T x k rows of the activation.
    const double act_rows =
        shape.stage == Stage::GateUp ? static_cast<double>(tokens) : assignments;
    MoeStagePrice out;
    // The W4A8 stage first quantizes its activation: reads the BF16 rows, writes the codes, scales
    // and sums of the whole plane (the kernel then reads the codes: act_read).
    if (kind == MoeStageKernel::W4a8) {
        const double blocks = code_cols / kQuantizeBlock;
        out.quantize_bytes =
            act_rows * (x_cols * kBf16BytesPerValue + blocks * kQuantizedBlockBytes);
    }
    out.bytes = shape.bank_bytes * touched / shape.experts + act_rows * act_read +
                assignments * shape.out_rows * kBf16BytesPerValue + out.quantize_bytes;
    out.dram_seconds = out.bytes / board.dram_bytes_per_s;
    const double ops = 2.0 * assignments * shape.rows * shape.cols;
    double rate      = 0.0; // ops per second the kind's units reach
    switch (kind) {
    case MoeStageKernel::R8:
        out.compute_priced = false;
        out.compute_note   = "R8's dp4a issue work has no term in the law";
        break;
    case MoeStageKernel::W4a8:
        if (shape.mx8) {
            out.compute_priced = false;
            out.compute_note   = "the block-scaled MXFP8 MMA has no rate fact of its own";
            break;
        }
        if (!board.generation_known) {
            out.compute_priced = false;
            out.compute_note   = "the generation has no rate row of its own (a lent row)";
            break;
        }
        rate = board.s8_ops_per_clock_sm * board.sms * board.clock_hz;
        if (rate <= 0) return std::nullopt; // no int8 tensor cores on this generation
        break;
    case MoeStageKernel::RegisterMma:
    case MoeStageKernel::VoltaHmma:
        if (!board.generation_known) {
            out.compute_priced = false;
            out.compute_note   = "the generation has no rate row of its own (a lent row)";
            break;
        }
        if (!board.bf16_native) {
            out.compute_priced = false;
            out.compute_note   = "no native BF16 MMA: the generation's rate is a fitted aggregate";
            break;
        }
        rate = board.bf16_ops_per_clock_sm * board.sms * board.clock_hz;
        if (rate <= 0) return std::nullopt;
        break;
    }
    out.compute_seconds = rate > 0 ? ops / rate : 0.0;
    out.seconds         = std::max(out.dram_seconds, out.compute_seconds);
    return out;
}

std::optional<MoeStageShape> moe_stage_shape(const MoeWeights& weights, const MoeConfig& config,
                                             Stage stage) {
    if (stage != Stage::GateUp && stage != Stage::Down) return std::nullopt;
    const Weight& bank         = bank_of(weights, stage);
    const std::int32_t experts = static_cast<std::int32_t>(weights.router.n);
    if (experts <= 0 || bank.n <= 0 || bank.n % experts != 0) return std::nullopt;
    MoeStageShape s;
    s.stage   = stage;
    s.qtype   = bank.qtype;
    s.mx8     = moe_mx8_banks(weights);
    s.experts = experts;
    s.top_k   = config.top_k;
    s.rows    = static_cast<std::int32_t>(bank.n / experts);
    s.cols    = static_cast<std::int32_t>(bank.k);
    // The gate/up's x is the router's input: its width is the router's, and where the bank's K is
    // the same (x is not zero-padded) the router's CTAs quantize it for an R8 gate/up (r8_admits).
    // The down stage's activation has the stage's own K.
    s.x_cols = stage == Stage::GateUp ? static_cast<std::int32_t>(weights.router.k) : s.cols;
    s.router_quantizes_x = stage == Stage::GateUp && bank.k == weights.router.k;
    // The kernel walks the logical K; a grouped format stores K padded (padded_shape[1], a
    // multiple of 128) and its tail is never read: price the bytes of the logical K.
    const double stored_k = static_cast<double>(bank.padded_shape[1]);
    s.bank_bytes =
        static_cast<double>(bank.payload_bytes) *
        (stored_k > static_cast<double>(bank.k) ? static_cast<double>(bank.k) / stored_k : 1.0);
    // A gate and an up row give one output (SwiGLU); a non-gated expert's one row gives one.
    const std::int32_t sets = config.activation == MoeActivation::Relu2 ? 1 : 2;
    s.out_rows              = stage == Stage::GateUp ? s.rows / sets : s.rows;
    return s;
}

const char* moe_candidate_status_name(MoeCandidate::Status status) {
    switch (status) {
    case MoeCandidate::Status::Selected:
        return "selected";
    case MoeCandidate::Status::Inadmissible:
        return "inadmissible";
    case MoeCandidate::Status::Dominated:
        return "dominated";
    case MoeCandidate::Status::PricedBelow:
        return "priced-below";
    case MoeCandidate::Status::Tied:
        return "tied";
    case MoeCandidate::Status::Unpriced:
        return "unpriced";
    }
    return "?";
}

const char* moe_kernel_name(MoeStageKernel kind) {
    switch (kind) {
    case MoeStageKernel::R8:
        return "r8";
    case MoeStageKernel::W4a8:
        return "w4a8";
    case MoeStageKernel::RegisterMma:
        return "rmma";
    case MoeStageKernel::VoltaHmma:
        return "volta-hmma";
    }
    return "?";
}

MoeCandidate::Status rank_against_selected(const std::optional<MoeStagePrice>& price,
                                           const std::optional<MoeStagePrice>& selected,
                                           std::string& why) {
    if (!price || !selected) {
        why = "no roofline or no rate for the kind";
        return MoeCandidate::Status::Unpriced;
    }
    if (!price->compute_priced || !selected->compute_priced) {
        why = !price->compute_priced ? price->compute_note
                                     : "the selected kind: " + selected->compute_note;
        return MoeCandidate::Status::Unpriced;
    }
    if (std::abs(price->seconds - selected->seconds) <= 1e-12 * selected->seconds) {
        why = "the same price as the selected kind: the law has no term to order them";
        return MoeCandidate::Status::Tied;
    }
    // The margin between the two prices: a gap of the order of a missing term (launch, tile or
    // wave effects, none priced here) does not order the kinds, and the reader sees how wide it is.
    char margin[96];
    std::snprintf(margin, sizeof margin, "%.2f us against the selected kind's %.2f us (%+.2f %%)",
                  price->seconds * 1e6, selected->seconds * 1e6,
                  (price->seconds - selected->seconds) / selected->seconds * 100.0);
    why = margin;
    return price->seconds < selected->seconds ? MoeCandidate::Status::PricedBelow
                                              : MoeCandidate::Status::Dominated;
}

std::vector<MoeCandidate> moe_candidates(const MoeWeights& weights, const MoeConfig& config,
                                         std::int32_t tokens, bool prefill, Stage stage) {
    std::vector<MoeCandidate> out;
    if (stage != Stage::GateUp && stage != Stage::Down) return out;
    const Weight& bank            = bank_of(weights, stage);
    const MoePlanQuery query      = moe_plan_query(weights, config, tokens, prefill);
    const MoeStageRoutes routes   = stage_routes(weights, query.plan);
    const MoeStageKernel selected = (stage == Stage::Down ? routes.down : routes.gate_up).kernel;
    const DeviceClass& device     = device_class();
    const auto board              = moe_board_facts(device);
    const auto shape              = moe_stage_shape(weights, config, stage);

    const auto admitted = [&](MoeStageKernel kind, std::string& why) {
        switch (kind) {
        case MoeStageKernel::RegisterMma:
            return true; // every expert format runs on it (plan_compiled aside: a schedule's tile)
        case MoeStageKernel::R8:
            if (r8_admits(weights, bank, tokens)) return true;
            if (!allows_a8(weights.expert_policy)) {
                why = "the expert policy allows no int8 activations";
            } else if (r8_bank_refusal(bank) != BankRefusal::None) {
                why = bank_refusal_name(r8_bank_refusal(bank));
            } else {
                why = "tokens past the bank's R8 limit, or the activation does not fit a CTA";
            }
            return false;
        case MoeStageKernel::W4a8: {
            // The layer-wide rule (with_w4a8): a bank's own refusal, or an NVFP4 bank refused for
            // the A4 policy on either bank, which keeps both stages off the tier. The calls it is
            // CHOSEN for (T >= 17 in the prefill profile, T >= 64 in the speed profile, a job width
            // of whole W4A8 tiles: MoeCall::w4a8_prefill, speed_w4a8) are rules of the cascade,
            // not admission: the stage runs at any T once the banks admit it, so it is listed and
            // priced here at every T. (Volta's HMMA below is different: its threshold is part of
            // the predicate production asks.)
            const MoePlan allowed = with_w4a8(MoePlan{}, weights, true);
            if (stage == Stage::Down ? allowed.w4a8_down : allowed.w4a8_gate_up) return true;
            const BankRefusal refusal = w4a8_bank_refusal(weights, bank);
            why                       = refusal != BankRefusal::None
                                            ? bank_refusal_name(refusal)
                                            : "the layer-wide A4 veto: an NVFP4 bank is refused for the A4 policy, both "
                                              "stages stay off the tier";
            return false;
        }
        case MoeStageKernel::VoltaHmma:
            // The production predicate (select_plan's Volta authority): the build's features, both
            // banks, the fragment layout rule, K and the geometry, from kMoeVoltaHmmaMinTokens.
            if (moe_volta_hmma_admitted(weights, config, tokens)) { return true; }
            why = "needs a build without mma.sync, T >= " + std::to_string(kMoeVoltaHmmaMinTokens) +
                  " and both banks of a format with an f16 fragment (K and the geometry in whole "
                  "groups)";
            return false;
        }
        return false;
    };
    const auto priced = [&](MoeStageKernel kind) -> std::optional<MoeStagePrice> {
        if (!board || !shape) return std::nullopt;
        return price_moe_stage(*shape, kind, tokens, *board);
    };
    // `seconds` only for a full price; the DRAM floor of one without its compute term is kept
    // apart.
    const auto fill_price = [](MoeCandidate& c, const std::optional<MoeStagePrice>& price) {
        if (!price) return;
        c.dram_seconds   = price->dram_seconds;
        c.compute_priced = price->compute_priced;
        if (price->compute_priced) c.seconds = price->seconds;
    };
    const auto selected_price = priced(selected);
    for (const MoeStageKernel kind : {MoeStageKernel::R8, MoeStageKernel::W4a8,
                                      MoeStageKernel::RegisterMma, MoeStageKernel::VoltaHmma}) {
        MoeCandidate c;
        c.kernel = kind;
        if (kind == selected) {
            // The plan runs it: admitted by construction, whatever a predicate here would say.
            c.status = MoeCandidate::Status::Selected;
            fill_price(c, priced(kind));
            out.push_back(std::move(c));
            continue;
        }
        if (!admitted(kind, c.why)) {
            c.status = MoeCandidate::Status::Inadmissible;
            out.push_back(std::move(c));
            continue;
        }
        const auto price = priced(kind);
        fill_price(c, price);
        c.status = rank_against_selected(price, selected_price, c.why);
        out.push_back(std::move(c));
    }
    return out;
}

std::optional<MoeLayerPrice> moe_layer_price(const MoeWeights& weights, const MoeConfig& config,
                                             const MoePlan& plan, std::int32_t tokens,
                                             const MoeBoardFacts& board) {
    const MoeStageRoutes routes = stage_routes(weights, plan);
    MoeLayerPrice out;
    std::int32_t hidden = 0;
    for (const Stage stage : {Stage::GateUp, Stage::Down}) {
        const auto shape = moe_stage_shape(weights, config, stage);
        if (!shape) return std::nullopt;
        const auto price = price_moe_stage(
            *shape, (stage == Stage::Down ? routes.down : routes.gate_up).kernel, tokens, board);
        if (!price) return std::nullopt;
        out.seconds += price->seconds;
        out.dram_seconds += price->dram_seconds;
        out.compute_binds = out.compute_binds ||
                            (price->compute_priced && price->compute_seconds > price->dram_seconds);
        if (!price->compute_priced && out.compute_priced) {
            out.compute_priced = false;
            out.compute_note   = price->compute_note;
        }
        if (stage == Stage::Down) hidden = shape->rows;
    }
    // The router: its weights, the activation and the logits, once.
    double extra_bytes = static_cast<double>(weights.router.payload_bytes) +
                         static_cast<double>(tokens) * weights.router.k * kBf16BytesPerValue +
                         static_cast<double>(tokens) * weights.router.n * kFp32BytesPerValue;
    // The separate combine where the down stage does not fuse it (an R8 down stage's last CTAs
    // combine in its epilogue unless the layer has a shared expert or the row grid exceeds the
    // ticket array: the production predicate): reads the T x k slot rows of BF16 and writes T rows.
    if (routes.down.kernel != MoeStageKernel::R8 ||
        !moe_r8_down_fuses_combine(weights, config, plan.down_r8)) {
        extra_bytes +=
            (static_cast<double>(tokens) * config.top_k + tokens) * hidden * kBf16BytesPerValue;
    }
    out.seconds += extra_bytes / board.dram_bytes_per_s;
    out.dram_seconds += extra_bytes / board.dram_bytes_per_s;
    return out;
}

std::optional<double> moe_layer_dram_floor_seconds(const MoeWeights& weights,
                                                   const MoeConfig& config, std::int32_t tokens,
                                                   const MoeBoardFacts& board) {
    if (tokens <= 0 || board.dram_bytes_per_s <= 0) return std::nullopt;
    double bytes = static_cast<double>(weights.router.payload_bytes) +
                   static_cast<double>(tokens) * weights.router.k * kBf16BytesPerValue;
    for (const Stage stage : {Stage::GateUp, Stage::Down}) {
        const auto shape = moe_stage_shape(weights, config, stage);
        if (!shape) return std::nullopt;
        // At least top-k experts' weights are read, whichever experts the T tokens pick.
        bytes += shape->bank_bytes * shape->top_k / shape->experts;
    }
    return bytes / board.dram_bytes_per_s;
}

} // namespace ninferno::ops::detail::moe
