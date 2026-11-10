// The MoE stage price (ops/moe/moe_cost.h, #616 step 4): the law's own terms on small cases, the
// candidate list on a layer of three generations' boards, and the replica: every MoE row of the
// sweeps on disk (routes/<class>.json "moe": rows [T, default us, chosen us]; "evidence_us" comes
// from the MoE sweep of the bench, "evidence_decode_us" from the load-time MoE tuner, the minimum
// over the candidates' medians), priced by moe_layer_price on that board's facts with the table's
// own plan, predicted against measured. Only the boards whose table holds that board's own sweep
// are replayed: RTX 4060 Ti, RTX 5090, RTX 5080, RTX 5060 Ti (the RTX 5070 Ti table's MoE section
// is the RTX 5080's sweep, so it is not a source). Every row of those tables is either replayed or
// counted under the reason it was not, the counts are printed, an unknown expert format fails the
// test, and the number of replayed rows per board is asserted exactly (a renamed field or a new
// format cannot empty the replica silently).
//
// What the replica prices is the table's plan, the kind the sweep measured. It is not always the
// kind production selects today: on the RTX 5090 and RTX 5080 prefill rows select_plan takes the
// grouped W4A8 (the prefill-w4a8 authority) while the table holds the register MMA, so the kind
// production runs there is verified by no sweep, nor are the W4A8 and Volta HMMA prices anywhere.
// No row of a replayed table has a format with a min, an R8 stage priced by its compute (R8 has no
// compute term), or the int8 rate in its binding term.
//
// The replica prints its rows twice per board: the rows whose every stage has a compute term
// ("fully priced") and the rows in which a stage has none and stands at its DRAM floor
// ("DRAM-only"), with the number of rows where the roofline (the compute term) binds.
//
// Checks on the replica, none of them a fact of a chip or a published number:
//   - at T = 1 predicted <= measured, with no tolerance: there the price's weight term is k
//     experts, which no routing goes below, so the price is a lower bound of the call (bytes at the
//     board's peak bandwidth, the unit rates). At T > 1 the weight term is the expected coverage
//     E (1 - (1 - k/E)^T) under uniform routing: an expectation, which a skewed router (fewer
//     experts) or its spread can put above a measured time, so no bound is asserted on it;
//   - at every T a strict bound (moe_layer_dram_floor_seconds: k experts' weights of each bank, the
//     router's weights and x read once; derived at strict_floor_checks) is at most measured, with
//     no tolerance. The rows come from the bench sweep (evidence_us) and the load-time tuner
//     (evidence_decode_us); both flush the L2 where the device has room (tune_timer.h), which is
//     what keeps a weight read from being served by it;
//   - predicted / measured is at least kRecordedMinRatio of the board: a RECORDED BASELINE of the
//     current price (the smallest ratio of the run that set it, rounded down), not a derived
//     figure. It stops the price from collapsing unseen. The distance of the ratio from 1 is the
//     named absent terms (header of moe_cost.h), none of them quantified here;
//   - the rank agreement of predicted and measured times across the rows of a board (concordant /
//     (concordant + discordant) over the pairs the prediction orders; 0.5 is chance, this is not
//     Kendall's tau) is above kMinConcordance, a CHOSEN margin (the worst board today is above it
//     by a wide gap). It is high mostly because both times grow with T.
// Host only, no GPU; it changes no route (the moe plan goldens are the check of that).
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/weight.h"
#include "ninferno/ops/moe.h"
#include "ops/generic/selector_facts.h"
#include "ops/moe/moe_cost.h"
#include "ops/moe/moe_internal.h"
#include "runtime/engine/interleave_cost.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
namespace moe     = ninferno::ops::detail::moe;
namespace qw      = ninferno::test::quantized_weight;
using Json        = nlohmann::json;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

const generic::FactsBoard& board_named(const std::string& name) {
    for (const generic::FactsBoard& b : generic::kFactsBoards) {
        if (name == b.name) return b;
    }
    throw std::runtime_error("no facts board " + name);
}

std::optional<QType> qtype_of(const std::string& s) {
    static const std::map<std::string, QType> names = {
        {"nvfp4", QType::NVFP4},         {"mxfp4", QType::MXFP4},
        {"iq3s", QType::GGML_IQ3_S},     {"iq2s", QType::GGML_IQ2_S},
        {"iq4xs", QType::GGML_IQ4_XS},   {"iq4nl", QType::GGML_IQ4_NL},
        {"iq3xxs", QType::GGML_IQ3_XXS}, {"q8", QType::Q8_G32_FP16}};
    const auto it = names.find(s);
    return it == names.end() ? std::nullopt : std::optional<QType>(it->second);
}

// A layer's weights as descriptors (no payload), the way the plan goldens build them.
struct Built {
    ops::MoeWeights weights;
    ops::MoeConfig config;
    std::optional<qw::PackedWeight> gate_up, down;
};

std::unique_ptr<Built> build(QType gate_up, QType down, std::int32_t experts, std::int32_t top_k,
                             std::int32_t hidden, std::int32_t inter, bool relu2) {
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    auto b              = std::make_unique<Built>();
    const auto options  = [](QType q) {
        qw::PatternedWeightOptions o;
        o.descriptor_only = true;
        if (q == QType::NVFP4) o.weight_scale_divisor = o.input_scale_divisor = 1.0F;
        return o;
    };
    const std::int32_t sets = relu2 ? 1 : 2;
    // A hidden size that is not whole blocks of 256 is a padded GGUF: gate/up columns to the next
    // 256 (Nemotron-H, H 2688, K 2816), as the plan goldens build it.
    const std::int32_t pad = hidden % 256 == 0 ? 0 : (hidden + 255) / 256 * 256 - hidden;
    b->gate_up.emplace(qw::make_patterned_weight(gate_up, experts * sets * inter, hidden + pad, 7U,
                                                 options(gate_up)));
    b->down.emplace(qw::make_patterned_weight(down, experts * hidden, inter, 9U, options(down)));
    b->config.top_k = top_k;
    if (relu2) b->config.activation = ops::MoeActivation::Relu2;
    Weight r{};
    r.qtype  = QType::BF16;
    r.layout = QuantLayout::Contiguous;
    r.qdata = r.payload = payload;
    r.payload_bytes     = static_cast<std::size_t>(experts) * hidden * 2;
    r.n = r.shape[0] = r.padded_shape[0] = experts;
    r.k = r.shape[1] = r.padded_shape[1] = hidden;
    r.ndim                               = 2;
    b->weights.router                    = r;
    b->weights.gate_up                   = b->gate_up->device_weight(payload);
    b->weights.down                      = b->down->device_weight(payload);
    b->weights.expert_policy             = ops::LinearPolicy::AllowA8;
    return b;
}

// The law's own terms on shapes whose answers are read off by hand.
void law_checks() {
    check(std::abs(moe::expected_experts_touched(256, 8, 1) - 8.0) < 1e-9,
          "one token touches top-k experts");
    check(moe::expected_experts_touched(256, 8, 4096) > 255.99, "a wide call touches them all");
    const double two = moe::expected_experts_touched(128, 8, 2);
    check(two > 8.0 && two < 16.0, "two tokens touch between k and 2k");
    check(moe::expected_experts_touched(0, 8, 1) == 0.0 &&
              moe::expected_experts_touched(8, 8, 0) == 0.0,
          "no experts or no tokens: none");

    const moe::MoeBoardFacts board{1e12, 100, 2e9, 64.0, 2048.0};
    moe::MoeStageShape s;
    s.bank_bytes         = 1e9;
    s.experts            = 100;
    s.top_k              = 10;
    s.rows               = 1000;
    s.out_rows           = 500;
    s.cols               = 1000;
    s.router_quantizes_x = true; // an unpadded x: the router's CTAs wrote the R8 codes
    const auto decode    = moe::price_moe_stage(s, moe::MoeStageKernel::RegisterMma, 1, board);
    check(decode && decode->dram_seconds > decode->compute_seconds,
          "a one-token stage is bandwidth-bound");
    const double bank_part = 1e9 * 10 / 100; // k of E experts' bytes at T = 1
    check(decode && decode->bytes >= bank_part && decode->bytes < bank_part * 1.01,
          "its bytes are the touched experts' (+ the small activation and output)");
    const auto wide = moe::price_moe_stage(s, moe::MoeStageKernel::RegisterMma, 4096, board);
    check(wide && wide->compute_seconds > wide->dram_seconds, "a wide stage is compute-bound");
    check(wide && wide->seconds == wide->compute_seconds, "seconds is the larger term");
    const auto r8 = moe::price_moe_stage(s, moe::MoeStageKernel::R8, 4096, board);
    check(r8 && !r8->compute_priced && r8->seconds == r8->dram_seconds,
          "R8's work has no term: its price is the DRAM floor, said so");
    const auto w4 = moe::price_moe_stage(s, moe::MoeStageKernel::W4a8, 4096, board);
    check(w4 && w4->compute_seconds < wide->compute_seconds, "the int8 tier computes faster");
    check(!moe::price_moe_stage(s, moe::MoeStageKernel::W4a8, 1,
                                moe::MoeBoardFacts{1e12, 100, 2e9, 1536.0, 0.0}),
          "no int8 tensor cores: no W4A8 price");
    check(!moe::price_moe_stage(moe::MoeStageShape{}, moe::MoeStageKernel::R8, 1, board),
          "an empty shape has no price");

    // The terms by hand, exactly: at T = 1 a gate/up stage touches 10 of 100 experts (1e8 bytes),
    // reads one row of x (1000 values, BF16) and writes 10 assignments x 500 rows (BF16).
    const auto close = [](double a, double b) { return std::abs(a - b) <= 1e-9 * std::abs(b); };
    check(decode && close(decode->bytes, 1e8 + 1000 * 2.0 + 10 * 500 * 2.0),
          "decode bytes: touched experts + x once + the outputs");
    check(decode && close(decode->dram_seconds, decode->bytes / 1e12), "DRAM seconds = bytes / bw");
    // 2 x 4096 x 10 assignments x 1000 rows x 1000 cols over 64 x 100 x 2e9 ops/s.
    check(wide && close(wide->compute_seconds, 2.0 * 4096 * 10 * 1000 * 1000 / (64.0 * 100 * 2e9)),
          "compute seconds = 2 x assignments x rows x cols / (rate x SMs x clock)");
    check(wide && w4 && close(wide->compute_seconds / w4->compute_seconds, 2048.0 / 64.0),
          "the int8 tier's compute time is the rate ratio, no other factor");
    // A down stage reads its T x k activation rows (10 x as many as a gate/up's x).
    moe::MoeStageShape down = s;
    down.stage              = moe::Stage::Down;
    const auto down_price = moe::price_moe_stage(down, moe::MoeStageKernel::RegisterMma, 1, board);
    check(down_price && close(down_price->bytes, 1e8 + 10 * 1000 * 2.0 + 10 * 500 * 2.0),
          "a down stage reads one activation row per assignment");
    // The int8 activation as read: 36 bytes a 32 values, 40 with a block sum (a format with a min).
    moe::MoeStageShape k4 = s;
    k4.qtype              = QType::GGML_Q4_K;
    const auto a8_plain   = moe::price_moe_stage(s, moe::MoeStageKernel::W4a8, 1, board);
    const auto a8_min     = moe::price_moe_stage(k4, moe::MoeStageKernel::W4a8, 1, board);
    // The W4A8 stage first quantizes (reads the BF16 row, writes codes + scale + sum: 40 bytes a
    // 32-value block whatever the format, of the plane at K rounded up to 128: 1000 values are a
    // plane of 1024, 32 blocks) and then reads the whole plane (36 bytes a 32 values, 40 with a
    // block sum).
    const double quantize_bytes = 1000 * 2.0 + 32 * 40.0;
    check(a8_plain && a8_min && close(a8_min->bytes - a8_plain->bytes, 1024 * 4.0 / 32.0),
          "a format with a min adds the block sum to the codes the kernel reads, not to the write");
    const double out_bytes = 10 * 500 * 2.0;
    check(a8_plain && close(a8_plain->bytes, 1e8 + quantize_bytes + 1024 * 36.0 / 32.0 + out_bytes),
          "W4A8 bytes at T = 1: experts + quantize pass (BF16 in, 40-byte blocks out) + codes + "
          "outputs");
    check(a8_plain && close(a8_plain->quantize_bytes, quantize_bytes) && a8_min &&
              close(a8_min->quantize_bytes, quantize_bytes),
          "the quantize pass is read BF16 + written blocks, once per x row, the same for every "
          "format");
    check(a8_min && close(a8_min->bytes, 1e8 + quantize_bytes + 1024 * 40.0 / 32.0 + out_bytes),
          "W4A8 bytes with a min at T = 1: the codes are read at 40/32 bytes a value");
    // R8: the gate/up reads the codes the router wrote, the down stage reads the BF16 intermediate.
    const auto r8_gate_up = moe::price_moe_stage(s, moe::MoeStageKernel::R8, 1, board);
    const auto r8_down    = moe::price_moe_stage(down, moe::MoeStageKernel::R8, 1, board);
    check(r8_gate_up && close(r8_gate_up->bytes, 1e8 + 1000 * 36.0 / 32.0 + out_bytes),
          "an R8 gate/up reads int8 codes (36/32 bytes a value)");
    check(r8_down && close(r8_down->bytes, 1e8 + 10 * 1000 * 2.0 + out_bytes),
          "an R8 down stage reads the BF16 intermediate, not int8 codes");
    // A padded gate/up (the router's width 800 under a bank K of 1000): x is not quantized by the
    // router. On R8 and the register MMA the widening pass first copies x into a zero-padded BF16
    // row (reads 800 values, writes 1000: (800 + 1000) x 2 bytes a token), and the stage reads that
    // copy at the bank's K of 1000 values; the W4A8 quantize pass walks 800 values (25 blocks) and
    // the kernel reads the codes over the bank's K, with no widening pass.
    moe::MoeStageShape padded = s;
    padded.x_cols             = 800;
    padded.router_quantizes_x = false;
    const double widen_padded = (800 + 1000) * 2.0;
    const auto r8_padded      = moe::price_moe_stage(padded, moe::MoeStageKernel::R8, 1, board);
    check(r8_padded && close(r8_padded->widen_bytes, widen_padded) &&
              close(r8_padded->bytes, 1e8 + widen_padded + 1000 * 2.0 + out_bytes),
          "an R8 gate/up of a padded x pays the widening pass and reads BF16 x over the bank's K "
          "(the zero tail included)");
    const auto rmma_padded =
        moe::price_moe_stage(padded, moe::MoeStageKernel::RegisterMma, 1, board);
    check(rmma_padded && close(rmma_padded->widen_bytes, widen_padded) &&
              close(rmma_padded->bytes, 1e8 + widen_padded + 1000 * 2.0 + out_bytes),
          "the register MMA pays the widening pass and reads the padded copy of x over the bank's "
          "K, not the router's width");
    const auto w4_padded = moe::price_moe_stage(padded, moe::MoeStageKernel::W4a8, 1, board);
    check(w4_padded && close(w4_padded->quantize_bytes, 800 * 2.0 + 32 * 40.0) &&
              w4_padded->widen_bytes == 0.0 &&
              close(w4_padded->bytes, 1e8 + 800 * 2.0 + 32 * 40.0 + 1024 * 36.0 / 32.0 + out_bytes),
          "the W4A8 quantize pass reads the router's width, writes and the kernel reads the plane "
          "of the bank K rounded up to 128; no widening pass");
    // The widening pass runs per token (T rows), only on a gate/up stage, and not for an unpadded
    // x on the register MMA or R8 (`decode`, `r8_gate_up` above have no such term).
    const auto rmma_padded3 =
        moe::price_moe_stage(padded, moe::MoeStageKernel::RegisterMma, 3, board);
    check(rmma_padded3 && close(rmma_padded3->widen_bytes, 3 * widen_padded),
          "the widening pass reads and writes each of the T rows of x");
    check(decode && decode->widen_bytes == 0.0 && r8_gate_up && r8_gate_up->widen_bytes == 0.0,
          "an unpadded x on the register MMA or R8 is not widened");
    moe::MoeStageShape padded_down = padded;
    padded_down.stage              = moe::Stage::Down;
    for (const moe::MoeStageKernel kind :
         {moe::MoeStageKernel::RegisterMma, moe::MoeStageKernel::R8,
          moe::MoeStageKernel::VoltaHmma}) {
        const auto p = moe::price_moe_stage(padded_down, kind, 1, board);
        check(p && p->widen_bytes == 0.0, "a down stage has no widening pass");
    }
    // Volta's HMMA converts x to f16 rows of the bank's K on every call, padded or not: reads the
    // router's width in BF16 and writes the bank's K in f16 (2 bytes), a row per token.
    const auto volta       = moe::price_moe_stage(s, moe::MoeStageKernel::VoltaHmma, 1, board);
    const auto volta_pad   = moe::price_moe_stage(padded, moe::MoeStageKernel::VoltaHmma, 2, board);
    const double volta_one = (1000 + 1000) * 2.0;
    check(volta && close(volta->widen_bytes, volta_one) &&
              close(volta->bytes, 1e8 + volta_one + 1000 * 2.0 + out_bytes),
          "Volta's f16 pass on an unpadded x: reads 1000 BF16 values, writes 1000 f16 values");
    check(volta_pad && close(volta_pad->widen_bytes, 2 * widen_padded),
          "Volta's f16 pass on a padded x: reads 800, writes 1000 values a token, T = 2");
    // The MXFP8 quantizer writes 32 code bytes and a 4-byte scale a block (36), no sum
    // (launch_moe_mx8_quantize), also for a qtype with a min; the block-scaled MMA reads the same
    // 36 bytes a block.
    moe::MoeStageShape mx_padded = padded;
    mx_padded.mx8                = true;
    moe::MoeStageShape mx_min    = mx_padded;
    mx_min.qtype                 = QType::GGML_Q4_K;
    const double mx_quantize     = 800 * 2.0 + 32 * 36.0;
    for (const moe::MoeStageShape& m : {mx_padded, mx_min}) {
        const auto mxp = moe::price_moe_stage(m, moe::MoeStageKernel::W4a8, 1, board);
        check(mxp && close(mxp->quantize_bytes, mx_quantize) &&
                  close(mxp->bytes, 1e8 + mx_quantize + 1024 * 36.0 / 32.0 + out_bytes),
              "an MXFP8 stage writes 36-byte blocks (no sum) and reads 36 bytes a 32 values");
    }
    // A down stage of K = 704 (Gemma 4 26B) runs on a plane of 768: 24 blocks, written and read.
    moe::MoeStageShape gemma = down;
    gemma.cols               = 704;
    const auto w4_gemma      = moe::price_moe_stage(gemma, moe::MoeStageKernel::W4a8, 1, board);
    check(w4_gemma && close(w4_gemma->quantize_bytes, 10 * (704 * 2.0 + 24 * 40.0)) &&
              close(w4_gemma->bytes,
                    1e8 + 10 * (704 * 2.0 + 24 * 40.0) + 10 * 768 * 36.0 / 32.0 + out_bytes),
          "a K = 704 W4A8 down stage quantizes and reads a plane of 768 values per assignment");
    // Past T = 1 a gate/up stage still reads each token's row of x once (not once in all, not once
    // per assignment): T rows; a down stage one row per assignment.
    const auto two_gate_up = moe::price_moe_stage(s, moe::MoeStageKernel::RegisterMma, 2, board);
    const auto two_down    = moe::price_moe_stage(down, moe::MoeStageKernel::RegisterMma, 2, board);
    const double touched2  = moe::expected_experts_touched(100, 10, 2);
    check(two_gate_up &&
              close(two_gate_up->bytes, 1e9 * touched2 / 100 + 2 * 1000 * 2.0 + 20 * 500 * 2.0),
          "T = 2 gate/up: the touched experts, two rows of x, 20 assignments' outputs");
    check(two_down &&
              close(two_down->bytes, 1e9 * touched2 / 100 + 20 * 1000 * 2.0 + 20 * 500 * 2.0),
          "T = 2 down: one activation row per assignment");

    // Where the law has no rate fact the compute term is absent and says why.
    moe::MoeStageShape mx = s;
    mx.mx8                = true;
    const auto mx8        = moe::price_moe_stage(mx, moe::MoeStageKernel::W4a8, 4096, board);
    check(mx8 && !mx8->compute_priced && !mx8->compute_note.empty() &&
              mx8->seconds == mx8->dram_seconds,
          "the block-scaled MXFP8 stage is not priced at the int8 rate");
    check(w4 && w4->compute_priced && w4->compute_note.empty(), "the plain W4A8 stage is priced");
    // A generation the table has no row of is lent another's rates: not a fact, not priced.
    moe::MoeBoardFacts lent = board;
    lent.generation_known   = false;
    for (const moe::MoeStageKernel kind :
         {moe::MoeStageKernel::RegisterMma, moe::MoeStageKernel::VoltaHmma,
          moe::MoeStageKernel::W4a8}) {
        const auto p = moe::price_moe_stage(s, kind, 4096, lent);
        check(p && !p->compute_priced && !p->compute_note.empty() && p->seconds == p->dram_seconds,
              "a lent generation row does not price the compute");
    }
    moe::MoeBoardFacts emulated = board;
    emulated.bf16_native        = false;
    for (const moe::MoeStageKernel kind :
         {moe::MoeStageKernel::RegisterMma, moe::MoeStageKernel::VoltaHmma}) {
        const auto p = moe::price_moe_stage(s, kind, 4096, emulated);
        check(p && !p->compute_priced && !p->compute_note.empty() && p->seconds == p->dram_seconds,
              "no native BF16 MMA: the fitted aggregate rate does not price the compute");
    }

    // The ranking of two prices: an unpriced side is not ranked, equal prices tie.
    std::string why;
    using Status     = moe::MoeCandidate::Status;
    const auto cheap = moe::price_moe_stage(s, moe::MoeStageKernel::RegisterMma, 4096, board);
    const auto same  = moe::price_moe_stage(s, moe::MoeStageKernel::VoltaHmma, 4096, board);
    // (Compute binds both at T = 4096: Volta's f16 pass adds DRAM bytes only, below the compute.)
    check(cheap && same && same->bytes > cheap->bytes &&
              moe::rank_against_selected(same, cheap, why) == Status::Tied && !why.empty(),
          "two kinds of equal price are a tie, not one dominated");
    check(cheap && w4 && moe::rank_against_selected(w4, cheap, why) == Status::PricedBelow &&
              moe::rank_against_selected(cheap, w4, why) == Status::Dominated,
          "a cheaper kind is priced below, a dearer one dominated");
    const auto emulated_a =
        moe::price_moe_stage(s, moe::MoeStageKernel::RegisterMma, 4096, emulated);
    const auto emulated_b = moe::price_moe_stage(s, moe::MoeStageKernel::VoltaHmma, 4096, emulated);
    check(emulated_a && emulated_b &&
              moe::rank_against_selected(emulated_a, emulated_b, why) == Status::Unpriced &&
              !why.empty(),
          "Volta's HMMA and the emulated register MMA are not ranked by the fitted rate");
    check(moe::rank_against_selected(std::nullopt, cheap, why) == Status::Unpriced,
          "no price, no rank");
    // The tie tolerance is numerical (1e-12 of the price): a gap of 1e-4 is a verdict with its
    // margin in `why`, a gap of 1e-13 is a tie.
    moe::MoeStageShape wider = s, same_bytes = s;
    wider.bank_bytes      = s.bank_bytes * (1.0 + 1e-4);
    same_bytes.bank_bytes = s.bank_bytes * (1.0 + 1e-13);
    const auto base       = moe::price_moe_stage(s, moe::MoeStageKernel::RegisterMma, 1, board);
    const auto dearer     = moe::price_moe_stage(wider, moe::MoeStageKernel::RegisterMma, 1, board);
    const auto equal = moe::price_moe_stage(same_bytes, moe::MoeStageKernel::RegisterMma, 1, board);
    check(base && dearer && moe::rank_against_selected(dearer, base, why) == Status::Dominated &&
              why.find("us") != std::string::npos && why.find('%') != std::string::npos,
          "a gap of 1e-4 is not a tie, and the verdict carries its margin");
    check(base && equal && moe::rank_against_selected(equal, base, why) == Status::Tied,
          "a gap of 1e-13 is a tie");
}

// The stage shape prices the bytes of the logical K, not the stored (padded) K.
void shape_checks() {
    const DeviceClass scoped_class = generic::board_device_class(board_named("rtx4090"));
    const ScopedDeviceClass scoped(scoped_class);
    // Q8 g32 stores K padded to 128: the Gemma 4 26B down K of 704 is stored at 768.
    const auto layer   = build(QType::Q8_G32_FP16, QType::Q8_G32_FP16, 8, 2, 2048, 704, false);
    const Weight& down = layer->weights.down;
    check(down.k == 704 && down.padded_shape[1] == 768,
          "the fixture stores the Q8 down bank at K = 768 for a logical K of 704");
    const auto shape = moe::moe_stage_shape(layer->weights, layer->config, moe::Stage::Down);
    check(shape && shape->cols == 704 &&
              std::abs(shape->bank_bytes -
                       static_cast<double>(down.payload_bytes) * 704.0 / 768.0) < 1e-6,
          "the down bank is priced at the bytes of K = 704 of the 768 stored");
    const auto gate_up = moe::moe_stage_shape(layer->weights, layer->config, moe::Stage::GateUp);
    check(gate_up &&
              gate_up->bank_bytes == static_cast<double>(layer->weights.gate_up.payload_bytes),
          "a bank stored at its logical K is priced at its stored bytes");

    // The widening pass on the two padded layers of the replayed tables: Nemotron's hidden 2688
    // under a gate/up K of 2816 (relu^2), gpt-oss's 2880 under 3072. A token's row is read at the
    // router's width and written at the bank's K, 2 bytes a value: (2688 + 2816) x 2 = 11008 and
    // (2880 + 3072) x 2 = 11904 bytes. The register MMA and R8 pay it (the padded BF16 copy), and
    // so does Volta's HMMA (its f16 pass), W4A8 does not (its quantize pass reads x instead).
    const moe::MoeBoardFacts board{1e12, 100, 2e9, 64.0, 2048.0};

    struct Padded {
        QType q;
        std::int32_t experts, top_k, hidden, inter, k;
        bool relu2;
        double per_token;
    };

    for (const Padded& p : {Padded{QType::GGML_IQ3_XXS, 128, 6, 2688, 2048, 2816, true, 11008.0},
                            Padded{QType::MXFP4, 32, 4, 2880, 2880, 3072, false, 11904.0}}) {
        const auto padded_layer = build(p.q, p.q, p.experts, p.top_k, p.hidden, p.inter, p.relu2);
        const auto up =
            moe::moe_stage_shape(padded_layer->weights, padded_layer->config, moe::Stage::GateUp);
        const std::string tag = " (hidden " + std::to_string(p.hidden) + ")";
        check(up && up->x_cols == p.hidden && up->cols == p.k && !up->router_quantizes_x,
              "the padded gate/up's x is the router's width under the bank's K" + tag);
        if (!up) continue;
        for (const std::int32_t tokens : {1, 4}) {
            for (const moe::MoeStageKernel kind :
                 {moe::MoeStageKernel::RegisterMma, moe::MoeStageKernel::R8,
                  moe::MoeStageKernel::VoltaHmma, moe::MoeStageKernel::W4a8}) {
                const auto price  = moe::price_moe_stage(*up, kind, tokens, board);
                const double want = kind == moe::MoeStageKernel::W4a8 ? 0.0 : tokens * p.per_token;
                check(price && std::abs(price->widen_bytes - want) <= 1e-9 * std::max(1.0, want),
                      std::string("the widening pass of ") + moe::moe_kernel_name(kind) + tag +
                          ", T = " + std::to_string(tokens) + ": " +
                          std::to_string(price ? price->widen_bytes : -1.0) + " vs " +
                          std::to_string(want));
            }
        }
        const auto down_shape =
            moe::moe_stage_shape(padded_layer->weights, padded_layer->config, moe::Stage::Down);
        const auto down_price =
            down_shape
                ? moe::price_moe_stage(*down_shape, moe::MoeStageKernel::RegisterMma, 4, board)
                : std::nullopt;
        check(down_price && down_price->widen_bytes == 0.0,
              "the down stage of a padded layer is not widened" + tag);
    }
}

// The layer price by hand, at several T, on the default plan (both stages on the register MMA), on
// R8 stages and on a mixed plan: the router's weights, x and logits, the separate combine where the
// down stage is not R8, the gate/up outputs halved for SwiGLU and not for a non-gated expert. The
// flags are asserted in both polarities.
double logical_bytes(const Weight& w) {
    return static_cast<double>(w.payload_bytes) * static_cast<double>(w.k) /
           static_cast<double>(std::max<std::int64_t>(w.padded_shape[1], w.k));
}

// The bytes by hand of one layer call: `gate_up_int8` the gate/up reads the codes of the router's
// CTAs (R8), `combine` the separate combine is run (the down stage is not R8).
double layer_bytes_by_hand(const ops::MoeWeights& w, std::int32_t experts, std::int32_t top_k,
                           std::int32_t tokens, bool relu2, bool gate_up_int8, bool combine) {
    const double t            = tokens;
    const double touched      = moe::expected_experts_touched(experts, top_k, tokens);
    const double gate_up_rows = static_cast<double>(w.gate_up.n) / experts;
    const double out_rows     = relu2 ? gate_up_rows : gate_up_rows / 2;
    const double hidden_rows  = static_cast<double>(w.down.n) / experts;
    const double x_bytes      = gate_up_int8 ? static_cast<double>(w.gate_up.k) * 36.0 / 32.0
                                             : static_cast<double>(w.router.k) * 2;
    double bytes              = logical_bytes(w.gate_up) * touched / experts + t * x_bytes +
                   t * top_k * out_rows * 2 // gate/up: experts, x rows, outputs
                   + logical_bytes(w.down) * touched / experts +
                   t * top_k * static_cast<double>(w.down.k) * 2 +
                   t * top_k * hidden_rows * 2 // down: experts, activation rows, outputs
                   + static_cast<double>(w.router.payload_bytes) +
                   t * static_cast<double>(w.router.k) * 2 +
                   t * static_cast<double>(w.router.n) * 4; // router: weights, x, FP32 logits
    if (combine) bytes += (t * top_k + t) * hidden_rows * 2;
    return bytes;
}

void layer_checks() {
    const DeviceClass scoped_class = generic::board_device_class(board_named("rtx4090"));
    const ScopedDeviceClass scoped(scoped_class);
    const moe::MoeBoardFacts board{1e12, 100, 2e9, 64.0, 2048.0};
    moe::MoeBoardFacts tiny_rate    = board; // a unit rate so low that compute binds
    tiny_rate.bf16_ops_per_clock_sm = 0.001;
    moe::MoeBoardFacts lent         = board;
    lent.generation_known           = false;
    const auto matches              = [](double got, double want) {
        return std::abs(got - want) <= 1e-9 * want;
    };
    for (const bool relu2 : {false, true}) {
        const std::int32_t experts = 8, top_k = 2, hidden = 2048, inter = 704;
        const auto layer =
            build(QType::Q8_G32_FP16, QType::Q8_G32_FP16, experts, top_k, hidden, inter, relu2);
        const ops::MoeWeights& w = layer->weights;
        const std::string tag    = std::string(" (relu2 ") + (relu2 ? "yes" : "no") + ")";

        // The default plan: both stages on the register MMA, the combine is separate.
        // (At T = 32 this board's compute passes its DRAM time: priced, not bandwidth-bound.)
        for (const std::int32_t tokens : {1, 2, 4, 32}) {
            const bool narrow = tokens <= 4;
            const auto price =
                moe::moe_layer_price(w, layer->config, moe::MoePlan{}, tokens, board);
            check(price && price->compute_priced && price->compute_binds == !narrow &&
                      price->compute_note.empty(),
                  "a register MMA layer is fully priced, bandwidth-bound while narrow, T = " +
                      std::to_string(tokens) + tag);
            if (!price) continue;
            const double expected =
                layer_bytes_by_hand(w, experts, top_k, tokens, relu2, false, true);
            check(matches(price->dram_seconds * board.dram_bytes_per_s, expected),
                  "the layer's bytes by hand, T = " + std::to_string(tokens) + tag + ": " +
                      std::to_string(price->dram_seconds * board.dram_bytes_per_s) + " vs " +
                      std::to_string(expected));
            check(narrow ? matches(price->seconds, price->dram_seconds)
                         : price->seconds > price->dram_seconds,
                  "seconds are the DRAM seconds while bandwidth-bound, more once compute binds, T "
                  "= " +
                      std::to_string(tokens) + tag);
        }

        // A rate so low that compute binds: priced, binds, and seconds exceed the DRAM seconds.
        const auto slow = moe::moe_layer_price(w, layer->config, moe::MoePlan{}, 32, tiny_rate);
        check(slow && slow->compute_priced && slow->compute_binds &&
                  slow->seconds > slow->dram_seconds,
              "a layer whose compute exceeds its DRAM time says so" + tag);
        // A lent generation row: not priced, with the reason.
        const auto lent_price = moe::moe_layer_price(w, layer->config, moe::MoePlan{}, 2, lent);
        check(lent_price && !lent_price->compute_priced && !lent_price->compute_note.empty() &&
                  !lent_price->compute_binds &&
                  matches(lent_price->seconds, lent_price->dram_seconds),
              "a layer on a lent generation row is unpriced, at its DRAM figure" + tag);

        // R8 on both stages: no compute term (flag false, note), the gate/up reads codes, the down
        // stage the BF16 intermediate, and no separate combine (the R8 down CTAs combine).
        moe::MoePlan r8;
        r8.gate_up_r8 = r8.down_r8 = 0;
        const auto r8_price        = moe::moe_layer_price(w, layer->config, r8, 2, board);
        check(r8_price && !r8_price->compute_priced && !r8_price->compute_note.empty() &&
                  !r8_price->compute_binds && matches(r8_price->seconds, r8_price->dram_seconds),
              "an R8 layer is unpriced: its DRAM figure alone, with a note" + tag);
        check(r8_price && matches(r8_price->dram_seconds * board.dram_bytes_per_s,
                                  layer_bytes_by_hand(w, experts, top_k, 2, relu2, true, false)),
              "an R8 layer's bytes by hand: codes for the gate/up, no separate combine" + tag);

        // Mixed: R8 gate/up and a register MMA down stage (10 of the 75 replayed rows): the combine
        // is separate, the layer is unpriced because of its R8 stage.
        moe::MoePlan mixed;
        mixed.gate_up_r8       = 0;
        const auto mixed_price = moe::moe_layer_price(w, layer->config, mixed, 4, board);
        check(mixed_price && !mixed_price->compute_priced && !mixed_price->compute_note.empty() &&
                  matches(mixed_price->dram_seconds * board.dram_bytes_per_s,
                          layer_bytes_by_hand(w, experts, top_k, 4, relu2, true, true)),
              "a mixed layer (R8 gate/up, register MMA down) carries the combine, unpriced" + tag);
        // The other way: a register MMA gate/up and an R8 down stage. Each stage takes its own
        // kernel: the R8 down stage leaves the layer unpriced and its CTAs fuse the combine.
        moe::MoePlan r8_down;
        r8_down.down_r8       = 0;
        const auto down_price = moe::moe_layer_price(w, layer->config, r8_down, 4, board);
        check(down_price && !down_price->compute_priced && !down_price->compute_note.empty() &&
                  matches(down_price->dram_seconds * board.dram_bytes_per_s,
                          layer_bytes_by_hand(w, experts, top_k, 4, relu2, false, false)),
              "a mixed layer (register MMA gate/up, R8 down) is unpriced and has no combine" + tag);
    }
}

// The R8 down stage fuses the combine only while its row grid fits the ticket array (and the layer
// has no shared expert); past that the separate combine runs and is priced. A hidden size of 2^20
// has a grid of rows (hidden / rows per CTA) well past 1024 tickets.
void combine_fusion_checks() {
    const DeviceClass scoped_class = generic::board_device_class(board_named("rtx4090"));
    const ScopedDeviceClass scoped(scoped_class);
    const moe::MoeBoardFacts board{1e12, 100, 2e9, 64.0, 2048.0};
    moe::MoePlan r8;
    r8.gate_up_r8 = r8.down_r8 = 0;
    const std::int32_t experts = 8, top_k = 2;
    for (const std::int32_t hidden : {2048, 1 << 20}) {
        const auto layer =
            build(QType::Q8_G32_FP16, QType::Q8_G32_FP16, experts, top_k, hidden, 256, false);
        const bool fuses = moe::moe_r8_down_fuses_combine(layer->weights, layer->config, 0);
        check(fuses == (hidden == 2048),
              "an R8 down stage fuses the combine at hidden 2048 and not past the ticket array, "
              "hidden " +
                  std::to_string(hidden));
        const auto price = moe::moe_layer_price(layer->weights, layer->config, r8, 2, board);
        check(price && std::abs(price->dram_seconds * board.dram_bytes_per_s -
                                layer_bytes_by_hand(layer->weights, experts, top_k, 2, false, true,
                                                    !fuses)) <=
                           1e-9 * price->dram_seconds * board.dram_bytes_per_s,
              "the separate combine is priced exactly where the R8 down stage does not fuse it, "
              "hidden " +
                  std::to_string(hidden));
    }
}

// The strict bound by hand. A call of T tokens each picking top-k distinct experts touches at
// least k experts of each bank (all tokens may pick the same k), so it reads at least k of E
// experts' weights of the gate/up and of the down bank, the router's weights, and the T rows of x
// at least once; everything else (a written intermediate, the logits, the activations in between,
// every expert past k) is left out, since a write into a clean L2 may not reach DRAM and a read of
// it may be served by it. The bound depends on T only through x.
void strict_floor_checks() {
    const DeviceClass scoped_class = generic::board_device_class(board_named("rtx4090"));
    const ScopedDeviceClass scoped(scoped_class);
    const moe::MoeBoardFacts board{1e12, 100, 2e9, 64.0, 2048.0};
    const std::int32_t experts = 8, top_k = 2, hidden = 2048, inter = 704;
    const auto layer =
        build(QType::Q8_G32_FP16, QType::Q8_G32_FP16, experts, top_k, hidden, inter, false);
    const ops::MoeWeights& w = layer->weights;
    for (const std::int32_t tokens : {1, 2, 32, 512}) {
        const auto floor = moe::moe_layer_dram_floor_seconds(w, layer->config, tokens, board);
        const double want =
            (logical_bytes(w.gate_up) * top_k / experts + logical_bytes(w.down) * top_k / experts +
             static_cast<double>(w.router.payload_bytes) +
             static_cast<double>(tokens) * w.router.k * 2) /
            board.dram_bytes_per_s;
        check(floor && std::abs(*floor - want) <= 1e-9 * want,
              "the strict bound by hand, T = " + std::to_string(tokens));
        const auto price = moe::moe_layer_price(w, layer->config, moe::MoePlan{}, tokens, board);
        check(floor && price && *floor <= price->dram_seconds,
              "the strict bound is below the expected-coverage price, T = " +
                  std::to_string(tokens));
    }
    check(!moe::moe_layer_dram_floor_seconds(w, layer->config, 0, board), "no tokens: no bound");
}

// format_has_min is production's: r8_admits bounds the down stage's quantized activation in shared
// memory at 36 bytes a 32-value block, 40 with a min. A down bank of K = 40192 (whole 256-value
// blocks) at T = 1 needs 1256 blocks x 36 = 45216 bytes (fits 49152) without a min, x 40 = 50240
// (does not) with one, so r8_admits is true exactly for a format without a min. Formats the
// fixture cannot build, or R8 refuses for another reason, are counted, not silently skipped.
void format_min_binding_checks() {
    const DeviceClass scoped_class = generic::board_device_class(board_named("rtx4090"));
    const ScopedDeviceClass scoped(scoped_class);
    int tested = 0, with_min = 0, without_min = 0, not_built = 0, refused = 0;
    for (const QType q :
         {QType::GGML_Q4_0, QType::GGML_Q4_1, QType::GGML_Q5_0, QType::GGML_Q5_1, QType::GGML_Q4_K,
          QType::GGML_Q5_K, QType::GGML_Q6_K, QType::GGML_IQ4_NL, QType::GGML_IQ4_XS,
          QType::GGML_IQ3_S, QType::GGML_IQ3_XXS, QType::GGML_IQ2_S, QType::GGML_IQ2_XXS,
          QType::Q8_G32_FP16, QType::MXFP4, QType::NVFP4}) {
        std::unique_ptr<Built> layer;
        try {
            layer = build(q, q, 2, 1, 256, 40192, false);
        } catch (const std::invalid_argument&) {
            ++not_built;
            continue;
        }
        if (ops::detail::moe::r8_bank_refusal(layer->weights.down) !=
            ops::detail::moe::BankRefusal::None) {
            ++refused;
            continue;
        }
        // The formats whose codec keeps a min (Q4_1, Q5_1, Q4_K, Q5_K), as a literal list: neither
        // the function under test nor the price's own flag.
        const bool has_min = q == QType::GGML_Q4_1 || q == QType::GGML_Q5_1 ||
                             q == QType::GGML_Q4_K || q == QType::GGML_Q5_K;
        const bool admitted = moe::r8_admits(layer->weights, layer->weights.down, 1);
        check(admitted == !has_min, "format " + std::to_string(static_cast<int>(q)) +
                                        ": r8_admits at K = 40192 is true exactly without a min");
        check(moe::moe_format_has_min(q) == has_min,
              "format " + std::to_string(static_cast<int>(q)) +
                  ": moe_format_has_min is the literal list of the formats with a min");
        ++tested;
        (has_min ? with_min : without_min)++;
    }
    std::printf("format_has_min binding: %d formats tested (%d with a min, %d without), %d the "
                "fixture cannot build, %d refused by R8\n",
                tested, with_min, without_min, not_built, refused);
    check(with_min == 4 && without_min == 12 && not_built == 0 && refused == 0,
          "the binding test covers the four formats with a min and the twelve without");
}

// The expert-coverage law has several copies; this one is bound to the runtime's (fixed point,
// runtime/engine/interleave_cost.h) on a grid, so that they cannot drift apart unseen.
void coverage_binding_checks() {
    for (const std::uint32_t experts : {8U, 32U, 64U, 128U, 256U, 512U}) {
        for (const std::uint32_t top_k : {1U, 2U, 4U, 8U, 22U}) {
            if (top_k > experts) continue;
            for (const std::uint32_t tokens : {0U, 1U, 2U, 3U, 7U, 16U, 64U, 300U}) {
                const double here = moe::expected_experts_touched(
                    static_cast<std::int32_t>(experts), static_cast<std::int32_t>(top_k),
                    static_cast<std::int32_t>(tokens));
                const double there =
                    static_cast<double>(
                        ninferno::runtime::touched_expert_fraction_q32(experts, top_k, tokens)) /
                    4294967296.0 * experts;
                check(std::abs(here - there) <= 1e-6 * experts,
                      "expert coverage E=" + std::to_string(experts) +
                          " k=" + std::to_string(top_k) + " T=" + std::to_string(tokens) + ": " +
                          std::to_string(here) + " vs the runtime's " + std::to_string(there));
            }
        }
    }
}

// The candidates of a layer on boards of three generations: the statuses the law gives.
void candidate_checks() {
    const auto layer = build(QType::GGML_IQ3_XXS, QType::GGML_IQ4_XS, 256, 8, 2048, 512, false);
    for (const char* name : {"rtx4090", "rtx2080ti", "v100"}) {
        const generic::FactsBoard& fb = board_named(name);
        const DeviceClass d           = generic::board_device_class(fb);
        const ScopedDeviceClass scoped(d);
        for (const std::int32_t t : {1, 4, 64}) {
            for (const moe::Stage stage : {moe::Stage::GateUp, moe::Stage::Down}) {
                const auto candidates =
                    moe::moe_candidates(layer->weights, layer->config, t, false, stage);
                int selected = 0;
                for (const auto& c : candidates) {
                    selected += c.status == moe::MoeCandidate::Status::Selected;
                    if (c.status == moe::MoeCandidate::Status::Inadmissible) {
                        check(!c.why.empty() && !c.seconds,
                              std::string(name) + ": an inadmissible candidate says why");
                    }
                    const bool ranked = c.status == moe::MoeCandidate::Status::Dominated ||
                                        c.status == moe::MoeCandidate::Status::PricedBelow ||
                                        c.status == moe::MoeCandidate::Status::Tied;
                    if (ranked) {
                        check(c.kernel != moe::MoeStageKernel::R8,
                              "R8 is never ranked by a law that does not price it");
                        // Volta and Turing have no native BF16 MMA: the generation's BF16 rate is
                        // a fitted aggregate there, so no kind is ranked by the law on them.
                        check(d.features.mma_bf16_native,
                              std::string(name) + ": a kind ranked without a native BF16 MMA");
                    }
                    if (c.status == moe::MoeCandidate::Status::Unpriced) {
                        check(!c.why.empty(), std::string(name) + ": an unpriced kind says why");
                    }
                    if (ranked && c.status != moe::MoeCandidate::Status::Tied) {
                        check(c.why.find("us") != std::string::npos &&
                                  c.why.find('%') != std::string::npos,
                              std::string(name) + ": a ranked kind carries its margin");
                    }
                    // A price to sort by only where it is a full one; a floor is kept apart.
                    check(c.seconds.has_value() == (c.compute_priced && c.dram_seconds.has_value()),
                          std::string(name) + ": `seconds` is set exactly for a full price");
                    if (c.kernel == moe::MoeStageKernel::R8 &&
                        c.status != moe::MoeCandidate::Status::Inadmissible) {
                        check(!c.compute_priced && !c.seconds && c.dram_seconds,
                              "an R8 candidate carries its DRAM floor only");
                    }
                }
                check(selected == 1, std::string(name) + " T=" + std::to_string(t) +
                                         ": exactly one candidate is the selected kind");
                check(candidates.size() == 4, "one entry per kind");
            }
        }
    }
}

// W4A8 admission follows the layer-wide rule of with_w4a8 (the exported stage flags), whichever
// bank is asked. The veto itself (an NVFP4 bank under an A4 policy keeps both stages off the tier)
// exists only in a build with the W4A4 kernels (sm_120a); where this build has none the check
// compares the candidates with with_w4a8's flags on layers that do not trigger it, and says so.
void w4a8_admission_checks() {
    const DeviceClass d = generic::board_device_class(board_named("rtx4090"));
    const ScopedDeviceClass scoped(d);
    int vetoed = 0;
    for (const auto& [gate_up, down, policy] :
         {std::tuple{QType::GGML_Q4_K, QType::NVFP4, ops::LinearPolicy::AllowA4},
          std::tuple{QType::GGML_Q4_K, QType::NVFP4, ops::LinearPolicy::AllowA8},
          std::tuple{QType::GGML_IQ3_XXS, QType::GGML_IQ4_XS, ops::LinearPolicy::AllowA8},
          std::tuple{QType::GGML_IQ3_XXS, QType::GGML_IQ4_XS, ops::LinearPolicy::A16Only}}) {
        const auto layer             = build(gate_up, down, 128, 8, 2048, 512, false);
        layer->weights.expert_policy = policy;
        const auto allowed           = moe::with_w4a8(moe::MoePlan{}, layer->weights, true);
        for (const moe::Stage stage : {moe::Stage::GateUp, moe::Stage::Down}) {
            const bool expected =
                stage == moe::Stage::Down ? allowed.w4a8_down : allowed.w4a8_gate_up;
            for (const auto& c :
                 moe::moe_candidates(layer->weights, layer->config, 64, true, stage)) {
                if (c.kernel != moe::MoeStageKernel::W4a8) continue;
                const bool admitted = c.status != moe::MoeCandidate::Status::Inadmissible;
                check(admitted == expected,
                      "W4A8 candidate admission equals with_w4a8's stage flag");
                if (!admitted) {
                    check(!c.why.empty(), "an inadmissible W4A8 candidate says why");
                    if (!allowed.w4a8_gate_up && !allowed.w4a8_down &&
                        moe::w4a8_bank_refusal(layer->weights, stage == moe::Stage::Down
                                                                   ? layer->weights.down
                                                                   : layer->weights.gate_up) ==
                            moe::BankRefusal::None) {
                        ++vetoed;
                        check(c.why.find("veto") != std::string::npos,
                              "a bank refused only by the layer-wide veto says so");
                    }
                }
            }
        }
    }
    std::printf("w4a8 admission: %d stage(s) vetoed layer-wide in this build%s\n", vetoed,
                vetoed == 0 ? " (no W4A4 kernels here: the veto itself is not exercised)" : "");
}

// The thresholds of the header: the per-board recorded_min_ratio is a RECORDED BASELINE (see the
// header), kMinConcordance a chosen margin. The upper checks (T = 1, strict bound) have no
// tolerance.
constexpr double kMinConcordance = 0.8;
// The tables carry no activation field; the one non-gated (relu^2) layer is Nemotron's.
constexpr std::int32_t kRelu2Hidden = 2688;

// One replayed row: the layer's price, the measured time, and what the price holds.
struct Row {
    double predicted, measured;
    bool compute_priced; // every stage had its compute term (else a stage stands at its DRAM floor)
    bool compute_binds;  // some stage's compute time exceeded its DRAM time
    std::int32_t tokens;
    double strict; // moe_layer_dram_floor_seconds: the bound that holds for any routing
};

double geomean_ratio(const std::vector<Row>& rows) {
    double sum = 0;
    for (const Row& r : rows) sum += std::log(r.predicted / r.measured);
    return std::exp(sum / static_cast<double>(rows.size()));
}

struct Spread {
    std::size_t count = 0;
    double geomean = 0, lo = 0, hi = 0;
};

Spread spread_of(const std::vector<Row>& rows, bool compute_priced) {
    std::vector<Row> subset;
    for (const Row& r : rows) {
        if (r.compute_priced == compute_priced) subset.push_back(r);
    }
    Spread out;
    out.count = subset.size();
    if (subset.empty()) return out;
    out.geomean = geomean_ratio(subset);
    out.lo      = 1e9;
    for (const Row& r : subset) {
        out.lo = std::min(out.lo, r.predicted / r.measured);
        out.hi = std::max(out.hi, r.predicted / r.measured);
    }
    return out;
}

// The sweeps on disk, one board at a time.
void replica(const std::filesystem::path& routes_dir) {
    struct Source {
        const char* file;
        const char* board;
        std::size_t expected_rows; // the rows of its table the replica replays, exactly
        // RECORDED BASELINE: the smallest predicted / measured of the run that set it, rounded
        // down (not derived; see the header). The absent terms of moe_cost.h are what it measures.
        // It guards the lowest row of the board only, and is to be RE-RECORDED when the tables are
        // re-measured (a re-measurement moves the rows, not the law).
        double recorded_min_ratio;
    };

    const Source sources[] = {{"ad106.json", "rtx4060ti", 19, 0.51},
                              {"gb202.json", "rtx5090", 9, 0.22},
                              {"gb203.json", "rtx5080", 23, 0.38},
                              {"gb206.json", "rtx5060ti", 24, 0.39}};
    std::printf("replica: moe_layer_price (bytes law + roofline) against the sweeps in "
                "routes/*.json\n");
    std::printf("predicted / measured per board. full: a compute term in every stage; dram-only: "
                "a stage without one (R8, block-scaled MXFP8, no native BF16 MMA) stands at its "
                "DRAM floor\n");
    std::printf("%-10s %-9s %5s %8s %8s %8s\n", "board", "rows", "n", "geomean", "min", "max");
    for (const Source& source : sources) {
        std::ifstream in(routes_dir / source.file);
        check(static_cast<bool>(in), std::string("cannot read ") + source.file);
        const Json table    = Json::parse(in);
        const DeviceClass d = generic::board_device_class(board_named(source.board));
        const ScopedDeviceClass scoped(d);
        const auto facts = moe::moe_board_facts(d);
        check(facts.has_value(), std::string(source.board) + ": the board has a roofline");
        if (!facts) continue;
        std::vector<Row> rows;
        // The rows of the table not replayed, by reason.
        std::map<std::string, int> skipped;
        for (const Json& e : table.value("moe", Json::array())) {
            const std::string format = e.at("format").get<std::string>();
            const std::string down   = e.value("down_format", format);
            const auto q             = qtype_of(format);
            const auto dq            = qtype_of(down);
            if (!q || !dq) {
                check(false,
                      std::string(source.board) +
                          ": an expert format the test cannot build: " + (q ? down : format));
                continue;
            }
            const std::int32_t experts = e.at("experts"), top_k = e.at("top_k"),
                               hidden = e.at("hidden"), inter = e.at("inter");
            // The tables carry no activation field: Nemotron mixer (hidden 2688) is the one
            // non-gated (relu^2) layer of them.
            const bool relu2 = hidden == kRelu2Hidden;
            std::unique_ptr<Built> layer;
            try {
                layer = build(*q, *dq, experts, top_k, hidden, inter, relu2);
            } catch (const std::invalid_argument& error) {
                ++skipped[std::string("layer the fixture cannot pack: ") + error.what()];
                continue;
            }
            const moe::MoeRouteKey key = moe::route_key(layer->weights, layer->config);
            for (const auto& [field, prefill] :
                 {std::pair{"evidence_us", true}, std::pair{"evidence_decode_us", false}}) {
                if (!e.contains(field)) continue;
                for (const Json& row : e.at(field)) {
                    const std::int32_t t   = row.at(0);
                    const double chosen_us = row.at(2).get<double>();
                    const auto plan        = moe::table_plan(key, t, prefill);
                    if (chosen_us <= 0) {
                        ++skipped["no measured time"];
                        continue;
                    }
                    if (!plan) {
                        ++skipped["no table plan at that T"];
                        continue;
                    }
                    const auto price =
                        moe::moe_layer_price(layer->weights, layer->config, *plan, t, *facts);
                    if (!price) {
                        ++skipped["no price"];
                        continue;
                    }
                    const auto strict =
                        moe::moe_layer_dram_floor_seconds(layer->weights, layer->config, t, *facts);
                    if (!strict) {
                        ++skipped["no strict bound"];
                        continue;
                    }
                    rows.push_back({price->seconds, chosen_us * 1e-6, price->compute_priced,
                                    price->compute_binds, t, *strict});
                    // A padded x: the row's ratio with the widening pass and without it (the
                    // gate/up stage at max(DRAM less the pass, compute)), printed for the reader.
                    if (layer->weights.gate_up.k != layer->weights.router.k) {
                        const auto up =
                            moe::moe_stage_shape(layer->weights, layer->config, moe::Stage::GateUp);
                        const moe::MoeStageKernel up_kind =
                            moe::stage_routes(layer->weights, *plan).gate_up.kernel;
                        const auto up_price =
                            up ? moe::price_moe_stage(*up, up_kind, t, *facts) : std::nullopt;
                        if (up_price) {
                            const double dram_without =
                                up_price->dram_seconds -
                                up_price->widen_bytes / facts->dram_bytes_per_s;
                            const double without =
                                price->seconds - up_price->seconds +
                                std::max(dram_without, up_price->compute_seconds);
                            std::printf("%-10s padded x %s %s T %d (%s): predicted / measured "
                                        "%.4f, without the widening pass %.4f (%.0f bytes)\n",
                                        source.board, format.c_str(), moe::moe_kernel_name(up_kind),
                                        t, field, price->seconds / (chosen_us * 1e-6),
                                        without / (chosen_us * 1e-6), up_price->widen_bytes);
                        }
                    }
                }
            }
        }
        for (const auto& [reason, count] : skipped) {
            std::printf("%-10s skipped %d row(s): %s\n", source.board, count, reason.c_str());
        }
        check(rows.size() == source.expected_rows,
              std::string(source.board) + ": replayed " + std::to_string(rows.size()) +
                  " rows, expected " + std::to_string(source.expected_rows));
        if (rows.empty()) {
            std::printf("%-10s (no rows)\n", source.board);
            continue;
        }
        const Spread full = spread_of(rows, true), floor_only = spread_of(rows, false);
        std::size_t binding = 0;
        double lo = 1e9, hi = 0;
        for (const Row& r : rows) {
            binding += r.compute_binds;
            lo = std::min(lo, r.predicted / r.measured);
            hi = std::max(hi, r.predicted / r.measured);
        }
        const auto print = [&](const char* label, const Spread& sp) {
            if (sp.count == 0) {
                std::printf("%-10s %-9s %5d\n", source.board, label, 0);
            } else {
                std::printf("%-10s %-9s %5zu %8.3f %8.3f %8.3f\n", source.board, label, sp.count,
                            sp.geomean, sp.lo, sp.hi);
            }
        };
        print("full", full);
        print("dram-only", floor_only);
        long concordant = 0, discordant = 0, ties = 0;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            for (std::size_t j = i + 1; j < rows.size(); ++j) {
                const double dp = rows[i].predicted - rows[j].predicted;
                const double dm = rows[i].measured - rows[j].measured;
                if (std::abs(dp) <= 1e-12 * std::abs(rows[i].predicted)) {
                    ++ties;
                } else if (dm == 0) {
                    continue;
                } else if ((dp > 0) == (dm > 0)) {
                    ++concordant;
                } else {
                    ++discordant;
                }
            }
        }
        const double concordance =
            concordant + discordant > 0
                ? static_cast<double>(concordant) / static_cast<double>(concordant + discordant)
                : 0.0;
        std::printf("%-10s roofline binds in %zu of %zu rows; all rows: min %.3f max %.3f; "
                    "concordance %.3f; prediction ties %ld\n",
                    source.board, binding, rows.size(), lo, hi, concordance, ties);
        // T = 1: the price is a lower bound (k experts, whatever the router picks). T > 1: the
        // strict bound is, the price is only an expectation (the header).
        double hi_t1 = 0, hi_strict = 0;
        std::size_t t1_rows = 0;
        for (const Row& r : rows) {
            if (r.tokens == 1) {
                hi_t1 = std::max(hi_t1, r.predicted / r.measured);
                ++t1_rows;
            }
            hi_strict = std::max(hi_strict, r.strict / r.measured);
        }
        std::printf("%-10s T = 1 rows: %zu, max predicted / measured %.3f; strict bound / measured "
                    "max %.3f\n",
                    source.board, t1_rows, hi_t1, hi_strict);
        std::printf("%-10s strict bound / measured per row (T: ratio):", source.board);
        for (const Row& r : rows) std::printf(" %d:%.3f", r.tokens, r.strict / r.measured);
        std::printf("\n");
        check(t1_rows >= 1, std::string(source.board) + ": the table has a T = 1 row");
        check(hi_t1 <= 1.0, std::string(source.board) +
                                ": a T = 1 row's predicted time exceeds its measured "
                                "time (" +
                                std::to_string(hi_t1) + ")");
        check(hi_strict <= 1.0, std::string(source.board) +
                                    ": the strict bound exceeds a measured time (" +
                                    std::to_string(hi_strict) + ")");
        check(lo >= source.recorded_min_ratio,
              std::string(source.board) +
                  ": predicted / measured fell below the recorded baseline (" + std::to_string(lo) +
                  " < " + std::to_string(source.recorded_min_ratio) + ")");
        check(concordance > kMinConcordance,
              std::string(source.board) + ": the prediction orders the rows no better than " +
                  std::to_string(kMinConcordance) + " of the pairs (" +
                  std::to_string(concordance) + ")");
    }
}

} // namespace

int main() {
    try {
        law_checks();
        shape_checks();
        layer_checks();
        strict_floor_checks();
        combine_fusion_checks();
        format_min_binding_checks();
        candidate_checks();
        w4a8_admission_checks();
        coverage_binding_checks();
        replica(std::filesystem::path(NINFERNO_ROUTES_DIR));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "moe cost: %s\n", error.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "moe cost: %d checks failed\n", g_failures);
        return 1;
    }
    std::printf("moe cost: PASS\n");
    return 0;
}
