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
// The MXFP8 quantizer (launch_moe_mx8_quantize): 32 E4M3 codes and a 4-byte scale, no sum.
constexpr double kMx8QuantizedBlockBytes = 36.0;
constexpr std::int32_t kQuantizeBlock    = 32;
constexpr double kBf16BytesPerValue      = 2.0;
constexpr double kF16BytesPerValue       = 2.0;
constexpr double kFp32BytesPerValue      = 4.0;

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
    // The block-scaled MMA reads codes and scales only, whatever the qtype's min.
    return moe_format_has_min(shape.qtype) && !shape.mx8 ? kInt8SummedActivationBytesPerValue
                                                         : kInt8ActivationBytesPerValue;
}

const Weight& bank_of(const MoeWeights& weights, Stage stage) {
    return stage == Stage::Down ? weights.down : weights.gate_up;
}

// The ops per second of the BF16 MMA (the register MMA, Volta's HMMA: a stage or the router), 0
// with the reason in `note` where the law has no fact for it, nullopt where the rate is zero.
std::optional<double> bf16_mma_rate(const MoeBoardFacts& board, std::string& note) {
    if (!board.generation_known) {
        note = "the generation has no rate row of its own (a lent row)";
        return 0.0;
    }
    if (!board.bf16_native) {
        // Volta's rate is a fitted aggregate; Turing's is the spec rate since #781, but its
        // kernels' BF16-to-f16 rounding has no term in this law yet (moe_cost.h).
        note = "no native BF16 MMA: Volta's rate is a fitted aggregate, Turing's BF16-to-f16 "
               "rounding has no term here";
        return 0.0;
    }
    const double rate = board.bf16_ops_per_clock_sm * board.sms * board.clock_hz;
    if (rate <= 0) return std::nullopt;
    return rate;
}

double optional_bytes(const Tensor& t) {
    return t.data != nullptr ? static_cast<double>(t.bytes()) : 0.0;
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
    out.fp32_fma_per_clock_sm = row.fp32_fma;
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
    // The live values of x (the router's width, which a padded gate/up's K exceeds): what the W4A8
    // quantizer reads; the kernels read their activation over the K they walk.
    const double x_cols = static_cast<double>(shape.x_cols > 0 ? shape.x_cols : shape.cols);
    // The W4A8 quantizer reads x in whole 32-value blocks (a block starting below the width is
    // loaded whole: quantize_x_task, moe_mx8_quantize_kernel). A padded gate/up's x whose width is
    // not whole blocks would overrun each row into the next (the last row past x), and production
    // refuses the stage there (w4a8_bank_refusal, KGroup): no price either. Every x the stage
    // admits is whole blocks, so x_cols is what the quantizer reads. (The bank's own K in whole
    // blocks is a bank refusal, which this price, like the others, does not model.)
    if (kind == MoeStageKernel::W4a8 && shape.stage == Stage::GateUp &&
        x_cols < static_cast<double>(shape.cols) &&
        static_cast<std::int32_t>(x_cols) % kQuantizeBlock != 0) {
        return std::nullopt;
    }
    // The W4A8 tier's activation planes are at K rounded up to 128 (moe_w4a8_ld, a zero tail): the
    // quantizer writes them whole and the kernel reads them whole. An R8 gate/up reads codes over
    // K.
    const double code_cols = kind == MoeStageKernel::W4a8
                                 ? static_cast<double>(moe_w4a8_ld(shape.cols))
                                 : static_cast<double>(shape.cols);
    // A BF16 stage reads the row at the bank's K, the zero tail of a padded gate/up included: the
    // padded copy (moe.cpp, "A zero-padded bank reads x widened with zeros") has ldx = g.input and
    // the kernels walk a.w.k; Volta's f16 rows are g.input wide too (launch_volta_x_f16). The
    // pre-pass that widens x is priced apart (widen_bytes, below).
    const double act_read = int8 ? code_cols * int8_read_bytes_per_value(shape)
                                 : static_cast<double>(shape.cols) * kBf16BytesPerValue;
    // A gate and up stage reads the T rows of x, which a token's k assignments share; a down stage
    // reads the T x k rows of the activation.
    const double act_rows =
        shape.stage == Stage::GateUp ? static_cast<double>(tokens) : assignments;
    MoeStagePrice out;
    // The W4A8 stage first quantizes its activation: reads the BF16 rows, writes the codes, scales
    // and sums of the whole plane (the kernel then reads the codes: act_read).
    if (kind == MoeStageKernel::W4a8) {
        const double blocks = code_cols / kQuantizeBlock;
        // launch_moe_mx8_quantize writes 32 code bytes and one 4-byte scale a block, never the
        // sum; launch_moe_w4a8_quantize writes the sum too.
        const double block_bytes = shape.mx8 ? kMx8QuantizedBlockBytes : kQuantizedBlockBytes;
        out.quantize_bytes       = act_rows * (x_cols * kBf16BytesPerValue + blocks * block_bytes);
    }
    // The pass that widens x before a gate/up stage, run on every call (nothing caches it): it
    // reads the T rows of x at the router's width and writes them at the bank's K.
    //   - A padded x on the register MMA or R8: moe.cpp:1740-1749 (`g.input != g.hidden` and the
    //     stage neither W4A8 nor Volta's) builds the [T][g.input] BF16 copy in ws.padded_x, a
    //     cudaMemcpy2DAsync of the g.hidden live values and a cudaMemset2DAsync of the zero tail;
    //     the kernel then reads that copy (act_read).
    //   - Volta's HMMA, padded or not: moe.cpp:1773 launch_volta_x_f16(x, g.hidden, t, g.hidden,
    //     g.input, ...) converts the g.hidden BF16 values of each row to an f16 row of g.input
    //     (moe_volta_hmma.cu volta_x_f16_kernel, the tail written as zeros).
    // The W4A8 stage reads x through its quantize pass instead (quantize_bytes).
    const bool padded_x = x_cols < static_cast<double>(shape.cols);
    if (shape.stage == Stage::GateUp) {
        if (kind == MoeStageKernel::VoltaHmma) {
            out.widen_bytes =
                act_rows * (x_cols * kBf16BytesPerValue + shape.cols * kF16BytesPerValue);
        } else if (padded_x &&
                   (kind == MoeStageKernel::RegisterMma || kind == MoeStageKernel::R8)) {
            out.widen_bytes =
                act_rows * (x_cols * kBf16BytesPerValue + shape.cols * kBf16BytesPerValue);
        }
    }
    out.bytes = (shape.bank_bytes + shape.bias_bytes) * touched / shape.experts +
                act_rows * act_read + assignments * shape.out_rows * kBf16BytesPerValue +
                out.quantize_bytes + out.widen_bytes;
    out.dram_seconds = out.bytes / board.dram_bytes_per_s;
    // The K the kind's MMAs walk: the W4A8 kernel runs k_tiles = act.ld / 128 tiles of 128 columns
    // (moe_w4a8.cuh, moe_w4a8_kernel: its K loops), the zero tail of the last tile included (K =
    // 704 issues 768 columns: code_cols); the other kinds walk the bank's K (code_cols too).
    const double ops = 2.0 * assignments * shape.rows * code_cols;
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
    case MoeStageKernel::VoltaHmma: {
        const auto bf16 = bf16_mma_rate(board, out.compute_note);
        if (!bf16) return std::nullopt;
        rate               = *bf16;
        out.compute_priced = rate > 0;
        break;
    }
    }
    out.compute_seconds = rate > 0 ? ops / rate : 0.0;
    // The pre-pass (the W4A8 quantize pass, the pass that widens x) is a kernel of its own,
    // launched before the grouped MMA on the same stream, so the two run one after the other: the
    // stage takes the pre-pass's DRAM time plus the MMA kernel's max(its own DRAM time, its
    // compute). A compute-bound MMA does not hide the pre-pass. Without a compute term the stage
    // stands at its DRAM time, the pre-pass included.
    const double prepass_bytes    = out.quantize_bytes + out.widen_bytes;
    out.prepass_seconds           = prepass_bytes / board.dram_bytes_per_s;
    const double mma_dram_seconds = (out.bytes - prepass_bytes) / board.dram_bytes_per_s;
    out.compute_binds             = out.compute_priced && out.compute_seconds > mma_dram_seconds;
    out.seconds = out.compute_binds ? out.prepass_seconds + out.compute_seconds : out.dram_seconds;
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
    // The bank's optional bias ([E x rows], FP32 or BF16): every kind adds bias[row] to its output
    // rows through moe_bias (moe_gemm.cuh, the register MMA's epilogue; moe_r8.cuh; moe_w4a8.cuh;
    // moe_volta_hmma.cu), so a call reads the biases of the rows of the experts it touches.
    s.bias_bytes = optional_bytes(stage == Stage::Down ? weights.down_bias : weights.gate_up_bias);
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
                                             const MoePlan& plan, MoeEpilogue epilogue,
                                             std::int32_t tokens, const MoeBoardFacts& board) {
    if (tokens <= 0 || board.dram_bytes_per_s <= 0) return std::nullopt;
    const MoeStageRoutes routes = stage_routes(weights, plan);
    // The router's CTAs quantize x for an R8 gate/up on this call (the production rule), and the
    // gate/up then reads those codes.
    const bool router_quantizes = moe_router_quantizes_x(weights, config, plan, tokens);
    MoeLayerPrice out;
    // One priced kernel (a stage, its pre-pass included, or the router) into the layer's: its own
    // seconds (price_moe_stage's sequential sum for a stage), its DRAM seconds, whether its compute
    // binds, and the flag with the first reason where its compute is not priced.
    const auto add = [&](double seconds, double dram_seconds, bool compute_binds,
                         bool compute_priced, const std::string& note) {
        out.seconds += seconds;
        out.dram_seconds += dram_seconds;
        out.compute_binds = out.compute_binds || compute_binds;
        if (!compute_priced && out.compute_priced) {
            out.compute_priced = false;
            out.compute_note   = note;
        }
    };
    std::int32_t hidden = 0;
    for (const Stage stage : {Stage::GateUp, Stage::Down}) {
        auto shape = moe_stage_shape(weights, config, stage);
        if (!shape) return std::nullopt;
        if (stage == Stage::GateUp) shape->router_quantizes_x = router_quantizes;
        const auto price = price_moe_stage(
            *shape, (stage == Stage::Down ? routes.down : routes.gate_up).kernel, tokens, board);
        if (!price) return std::nullopt;
        add(price->seconds, price->dram_seconds, price->compute_binds, price->compute_priced,
            price->compute_note);
        if (stage == Stage::Down) hidden = shape->rows;
    }
    // The router kernel (run, step 1; moe_router_kernel): its weights, the T rows of x, its bias
    // (args.bias, added to each logit), the FP32 logits it writes; where it quantizes x for an R8
    // gate/up its CTAs also load the BF16 rows of x a second time and write the codes, scales and
    // sums of each row's plane at K rounded up to 128 (quantize_x_task over T x xq_ld / 32 blocks,
    // 40 bytes a block, the blocks past K zero).
    const double t       = tokens;
    const double experts = static_cast<double>(weights.router.n);
    const double width   = static_cast<double>(weights.router.k);
    double router_bytes  = static_cast<double>(weights.router.payload_bytes) +
                          t * width * kBf16BytesPerValue + optional_bytes(weights.router_bias) +
                          t * experts * kFp32BytesPerValue;
    if (router_quantizes) {
        // The quantize tasks read x through GemmArgs::xq (moe_gemm.cuh, moe_router_simt.cu), apart
        // from the MMA's read through args.x: each block that starts below the width is loaded
        // whole, so T x the width in whole 32-value blocks, BF16.
        router_bytes += t * std::ceil(width / kQuantizeBlock) * kQuantizeBlock * kBf16BytesPerValue;
        router_bytes += t * (moe_w4a8_ld(static_cast<std::int32_t>(width)) / kQuantizeBlock) *
                        kQuantizedBlockBytes;
    }
    // Its compute: 2 x T x E x H on the unit the kernel uses. The register MMA and Volta's HMMA run
    // the BF16 MMA (bf16_mma_rate, the stages' rule); the SIMT router (T <= kRouterSimtMaxTokens)
    // one FP32 fmaf per weight and token (moe_router_simt.cu), two operations an FMA at the
    // generation's FP32 FMA issue rate (GenerationRow::fp32_fma; its BF16 unpacking and the
    // reduction are not counted, so the term stays a lower bound).
    std::string router_note;
    double router_rate = 0.0;
    if (moe_router_kernel(weights, config, plan, tokens) == MoeRouterKernel::Simt) {
        if (!board.generation_known) {
            router_note = "the generation has no rate row of its own (a lent row)";
        } else {
            router_rate = 2.0 * board.fp32_fma_per_clock_sm * board.sms * board.clock_hz;
            if (router_rate <= 0) router_note = "no FP32 FMA rate for the generation";
        }
    } else {
        const auto rate = bf16_mma_rate(board, router_note);
        if (!rate) return std::nullopt;
        router_rate = *rate;
    }
    // One kernel with no pre-pass: max(DRAM, compute) where the compute is priced, else its DRAM
    // time.
    const double router_ops     = 2.0 * t * experts * width;
    const double router_dram    = router_bytes / board.dram_bytes_per_s;
    const bool router_priced    = router_rate > 0;
    const double router_compute = router_priced ? router_ops / router_rate : 0.0;
    const bool router_binds     = router_priced && router_compute > router_dram;
    add(router_binds ? router_compute : router_dram, router_dram, router_binds, router_priced,
        "the router: " + router_note);
    // The selection (select / select_plan, or the SIMT router's last CTA) reads the logits back,
    // and the optional selection bias and expert scale ([E] FP32 each).
    double extra_bytes = t * experts * kFp32BytesPerValue + optional_bytes(weights.selection_bias) +
                         optional_bytes(weights.expert_scale);
    // The combine: reads the T x k slot rows of BF16 and writes T rows. The separate combine_kernel
    // and the R8 down stage's fused tail (moe_r8.cuh fused_combine_tail, where the down stage fuses
    // it: moe_r8_down_fuses_combine) move the same bytes: the tail reads each token's k slot rows
    // of y (__ldcg) and writes its out row. The y rows were written by the down stage just before;
    // the law has no L2 term for a tensor written and read back (the gate/up's intermediate read by
    // the down stage is priced the same way), so both are counted as DRAM bytes, the fused tail's
    // as the separate kernel's.
    extra_bytes += (t * config.top_k + t) * hidden * kBf16BytesPerValue;
    // An AddResidual call (out += MoE(x), out holding the residual on entry) reads the T x H BF16
    // destination before the combine writes it, whichever combine runs: the separate one
    // (moe_route.cu combine_kernel, `residual = *(out + h)` under CombineArgs::add_residual) and
    // the R8 down stage's fused tail (moe_r8.cuh fused_combine_tail, `acc += *out`) alike. A Store
    // call only writes it.
    if (epilogue == MoeEpilogue::AddResidual) {
        extra_bytes += static_cast<double>(tokens) * hidden * kBf16BytesPerValue;
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
                   optional_bytes(weights.router_bias) +
                   static_cast<double>(tokens) * weights.router.k * kBf16BytesPerValue;
    for (const Stage stage : {Stage::GateUp, Stage::Down}) {
        const auto shape = moe_stage_shape(weights, config, stage);
        if (!shape) return std::nullopt;
        // At least top-k experts' weights (and biases) are read, whichever experts the T tokens
        // pick.
        bytes += (shape->bank_bytes + shape->bias_bytes) * shape->top_k / shape->experts;
    }
    return bytes / board.dram_bytes_per_s;
}

} // namespace ninferno::ops::detail::moe
