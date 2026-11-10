#include "ops/moe/moe_cost.h"

#include "core/generation.h"

#include <algorithm>
#include <cmath>

namespace ninferno::ops::detail::moe {

namespace {

// The int8 activation of the R8 and W4A8 tiers as stored: 36 bytes a 32 values (the codes, a scale
// and a block sum, moe_internal.h), BF16 otherwise.
constexpr double kInt8ActivationBytesPerValue = 36.0 / 32.0;
constexpr double kBf16BytesPerValue           = 2.0;

bool int8_activations(MoeStageKernel kind) {
    return kind == MoeStageKernel::R8 || kind == MoeStageKernel::W4a8;
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
    const double act_value =
        int8_activations(kind) ? kInt8ActivationBytesPerValue : kBf16BytesPerValue;
    MoeStagePrice out;
    out.bytes = shape.bank_bytes * touched / shape.experts + assignments * shape.cols * act_value +
                assignments * shape.out_rows * kBf16BytesPerValue;
    out.dram_seconds = out.bytes / board.dram_bytes_per_s;
    const double ops = 2.0 * assignments * shape.rows * shape.cols;
    double rate      = 0.0; // ops per second the kind's units reach
    switch (kind) {
    case MoeStageKernel::R8:
        out.compute_priced = false; // dp4a issue work: no term
        break;
    case MoeStageKernel::W4a8:
        rate = board.s8_ops_per_clock_sm * board.sms * board.clock_hz;
        if (rate <= 0) return std::nullopt; // no int8 tensor cores on this generation
        break;
    case MoeStageKernel::RegisterMma:
    case MoeStageKernel::VoltaHmma:
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
    s.stage      = stage;
    s.bank_bytes = static_cast<double>(bank.payload_bytes);
    s.experts    = experts;
    s.top_k      = config.top_k;
    s.rows       = static_cast<std::int32_t>(bank.n / experts);
    s.cols       = static_cast<std::int32_t>(bank.k);
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
            why = r8_bank_refusal(bank) != BankRefusal::None
                      ? bank_refusal_name(r8_bank_refusal(bank))
                      : "tokens past the bank's R8 limit, or the activation does not fit a CTA";
            return false;
        case MoeStageKernel::W4a8: {
            const BankRefusal refusal = w4a8_bank_refusal(weights, bank);
            if (refusal == BankRefusal::None) return true;
            why = bank_refusal_name(refusal);
            return false;
        }
        case MoeStageKernel::VoltaHmma:
            if (!device.features.mma_tensor && tokens >= 17 &&
                moe_volta_hmma_codec(generic::codec_of(bank))) {
                return true;
            }
            why = "needs a board without mma.sync, T >= 17 and a codec with an f16 fragment";
            return false;
        }
        return false;
    };
    const auto priced = [&](MoeStageKernel kind) -> std::optional<MoeStagePrice> {
        if (!board || !shape) return std::nullopt;
        return price_moe_stage(*shape, kind, tokens, *board);
    };
    const auto selected_price = priced(selected);
    for (const MoeStageKernel kind : {MoeStageKernel::R8, MoeStageKernel::W4a8,
                                      MoeStageKernel::RegisterMma, MoeStageKernel::VoltaHmma}) {
        MoeCandidate c;
        c.kernel = kind;
        if (!admitted(kind, c.why)) {
            c.status = MoeCandidate::Status::Inadmissible;
            out.push_back(std::move(c));
            continue;
        }
        const auto price = priced(kind);
        if (price) c.seconds = price->seconds;
        if (kind == selected) {
            c.status = MoeCandidate::Status::Selected;
        } else if (!price || !selected_price) {
            c.status = MoeCandidate::Status::Unpriced;
            c.why    = "no roofline or no rate for the kind";
        } else if (!price->compute_priced || !selected_price->compute_priced) {
            c.status = MoeCandidate::Status::Unpriced;
            c.why    = "R8's issue-rate work has no term in the law: not comparable";
        } else {
            c.status = price->seconds < selected_price->seconds ? MoeCandidate::Status::PricedBelow
                                                                : MoeCandidate::Status::Dominated;
        }
        out.push_back(std::move(c));
    }
    return out;
}

std::optional<double> moe_layer_seconds(const MoeWeights& weights, const MoeConfig& config,
                                        const MoePlan& plan, std::int32_t tokens,
                                        const MoeBoardFacts& board) {
    const MoeStageRoutes routes = stage_routes(weights, plan);
    double seconds              = 0;
    for (const Stage stage : {Stage::GateUp, Stage::Down}) {
        const auto shape = moe_stage_shape(weights, config, stage);
        if (!shape) return std::nullopt;
        const auto price = price_moe_stage(
            *shape, (stage == Stage::Down ? routes.down : routes.gate_up).kernel, tokens, board);
        if (!price) return std::nullopt;
        seconds += price->seconds;
    }
    // The router: its weights, the activation and the logits, once.
    const double router_bytes =
        static_cast<double>(weights.router.payload_bytes) +
        static_cast<double>(tokens) * weights.router.k * kBf16BytesPerValue +
        static_cast<double>(tokens) * weights.router.n * 4.0;
    return seconds + router_bytes / board.dram_bytes_per_s;
}

} // namespace ninferno::ops::detail::moe
