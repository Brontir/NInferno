#include "spec/load_model.h"

#include "core/generation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ninferno::spec {

namespace {

// The fitted constants: audit1018/x3/x3_fit.py, form M3, least squares over the 116 non-adaptive
// spec-tune floors of src/spec/data/drafter_registry.json at 3db81deeb. In sample the fit errs by
// 9.8 % rms in x3_fit.py over the 116, and 9.23 % in test_spec over the 87 it prices (Ada and
// Blackwell; these constants rounded, the clocks and bandwidths of the reference table); held out
// (re-run 21.10; x3_fit.py's M3 never had knee or beta) it errs over the 87 priced floors by
// 20.5 % rms with a card left out of the fit and 17.4 % with a model left out (21.0 %, 18.2 % over
// all 116). Refit them
// with the script when the registry gains floors. knee and beta come from one card's measured
// grid per generation (27B Q4_G64, 8K, several rows) and stay here as its record: round_cost_of
// leaves them out (a lab constant needs two cards), which only the floors deeper than the knee
// (depth 8+ on Ada, 12+ on Blackwell) notice. The fit is of one row only (the floors are C1):
// modeled_load decodes plainly past RoundCost::fitted_rows until a multi-row fit exists.
//
// The tensor rate per SM and clock is the chip's fact from core/generation.h (GenerationRow::bf16:
// 512 on the consumer parts, 2048 on the A100, the fitted 1536 where Volta and Turing emulate the
// BF16 MMA), so inv_eta and kquant are in units of that rate: the fit's V100 11.45 and RTX 2080 Ti
// 0.3337 at their FP16 tensor rates (1024, 512) are 17.18 and 1.001 here, and its k/IQ-quant 0.46
// is 0.69 and 1.38 there (0.46 x 1536 / 1024, x 1536 / 512): the same alpha.
struct Generation {
    int cc;
    double inv_eta; // verify column cost over the generation row's tensor rate (Volta: its route)
    double kquant;  // more for GGUF k/IQ-quant weights (the k-quant route), same unit
    std::int32_t knee; // 0: no grid measured one
    double beta;
    int lab_cards; // cards of the generation whose floors the fit saw
};

constexpr Generation kGenerations[] = {
    {70, 17.18, 0.69, 0, 0.0, 1},       // V100
    {75, 1.001, 1.38, 0, 0.0, 1},       // RTX 2080 Ti
    {80, 2.793, 0.46, 24, 0.0045, 1},   // A100
    {86, 0.9557, 0.46, 32, 0.0155, 1},  // RTX 3090
    {89, 0.3600, 0.46, 8, 0.0069, 5},   // RTX 4090, 4070 Ti, 4070 Ti SUPER, 4060 Ti, L4
    {120, 0.4582, 0.46, 12, 0.0122, 4}, // RTX 5090, 5080, 5070 Ti, 5060 Ti
};

double tensor_rate(int cc) noexcept { return generation_row(cc / 10, cc % 10).bf16; }

constexpr double kMoeRead     = 0.3019; // of the experts independent routing would add
constexpr std::uint64_t kProposalHeadRows = 131072;
struct DrafterKind {
    std::string_view kind;
    double cost; // per pass, over the drafter's size relative to the target's active weights
    bool pass_per_draft;
};

constexpr DrafterKind kDrafterKinds[] = {
    // Cards behind each cost (two or more that differ, by the lab rule): mtp 4, gemma4_assistant
    // 9, nemotron_h_mtp 3 (RTX 3090, 4090, 5090), dflash 10, dflash2 11.
    {"mtp", 0.7232, true},    {"gemma4_assistant", 67.09, true}, {"nemotron_h_mtp", 1.429, true},
    {"dflash", 283.1, false}, {"dflash2", 329.2, false},
};

// The generation x family pairs the fit stands behind: measured on two or more cards, and held out
// within kClassMargin (10 %) both with a card left out and with a model left out (x3_fit.py's
// pred_m3_loc and pred_m3_lom in points.csv, rms by cc and engine; re-run 21.10 without knee or
// beta, the same). Only sm120 dense passes (7.1 %, 6.8 %; its worst single point 10.4 % card-out);
// sm89 dense (11.0 % card-out), sm120 gemma4 (11.3 %, 16.1 %), sm89 qwen3_5 (12.2 %, 25.5 %), sm89
// gemma4 (16.1 %), sm120 qwen3_5 (31.5 %) miss it, and the nemotron_h pairs have one card each. The
// sm120 dense points are all one target, quantization and drafter kind: gpt-oss-20b (an MoE, the
// registry's "dense" engine), MXFP4, DFlash, on the RTX 5060 Ti, 5070 Ti and 5090. So the key is
// what the points cover: the generation, the family, the target's shape as far as the facts tell it
// (its routed experts, 32, top-4, and hidden size, 2880: gpt-oss-20b), the weight format and the
// drafter kind. A dense target, another MoE, another weight format (BF16, NVFP4, Q8, Q4_0, a GGUF
// k-quant) or another drafter on sm120 dense stays unsure.
struct Tested {
    int cc;
    std::string_view family;
    int experts; // routed experts (0: dense)
    int experts_per_token;
    std::uint64_t hidden_size;
    std::string_view weight_format; // the one quantized format (format_name spelling)
    std::string_view drafter;
};

constexpr Tested kTested[] = {
    {120, "dense", 32, 4, 2880, "mxfp4", "dflash"},
};

ModelFit fit_of(int cc, const TargetFacts& target, std::string_view drafter) {
    const std::string& family = target.family;
    const bool moe            = target.experts > 0;
    // The quantized format of the set: one, beside unquantized ones only; "" when the set is empty
    // or holds two quantized formats (not a covered pair either way).
    std::string quantized;
    int quantized_count = 0;
    for (const std::string& f : target.weight_formats) {
        if (f == "bf16" || f == "fp32" || f == "int32") { continue; }
        quantized = f;
        ++quantized_count;
    }
    if (quantized_count != 1) { quantized.clear(); }
    std::string formats;
    for (const std::string& f : target.weight_formats) {
        formats += (formats.empty() ? "" : "+") + f;
    }
    const std::string arch = "sm" + std::to_string(cc);
    for (const Tested& t : kTested) {
        if (t.cc == cc && t.family == family && t.experts == target.experts &&
            t.experts_per_token == target.experts_per_token &&
            t.hidden_size == target.hidden_size && !quantized.empty() &&
            t.weight_format == quantized && t.drafter == drafter) {
            return {};
        }
    }
    return {false, arch + " " + family + (moe ? " moe" : "") + " " +
                       (formats.empty() ? std::string("unknown format") : formats) + " " +
                       std::string(drafter) +
                       ": not held out within the 10 % margin by the fit's points"};
}

} // namespace

std::optional<RoundCost> round_cost_of(const GpuKey& gpu, const TargetFacts& target,
                                       const DrafterFacts& drafter) {
    // Every fact the price reads is checked here, before any generation, mma_slow or provenance
    // outcome: a missing or malformed one is nullopt on every GPU, never read as "absent".
    const auto positive = [](double x) { return std::isfinite(x) && x > 0; };
    if (gpu.sms <= 0 || gpu.sm_clock_khz <= 0 || !positive(gpu.dram_gbps) ||
        gpu.compute_capability <= 0 || !positive(target.bytes_per_weight) ||
        !positive(target.active_params) || target.hidden_size == 0 ||
        target.num_hidden_layers == 0) {
        return std::nullopt;
    }
    // MoE facts come together: experts, experts per token in [1, experts] and the routed experts'
    // parameters; a dense target has none of them.
    if (target.experts < 0 || target.experts_per_token < 0 ||
        !std::isfinite(target.routed_expert_params)) {
        return std::nullopt;
    }
    if (target.experts > 0) {
        if (target.experts_per_token <= 0 || target.experts_per_token > target.experts ||
            !positive(target.routed_expert_params)) {
            return std::nullopt;
        }
    } else if (target.experts_per_token != 0 || target.routed_expert_params != 0) {
        return std::nullopt;
    }
    // An unsupported drafter kind has no model on any GPU, mma_slow or not.
    const DrafterKind* kind = nullptr;
    for (const DrafterKind& k : kDrafterKinds) {
        if (k.kind == drafter.kind) { kind = &k; }
    }
    if (kind == nullptr) { return std::nullopt; }
    // The drafter's size: every shape check that yields nullopt runs here, before any generation,
    // mma_slow or provenance outcome, so a malformed target has no model on every GPU.
    const double h = static_cast<double>(target.hidden_size);
    double size = h * h / target.active_params; // a DFlash block or an assistant: about h^2 a layer
    if (kind->kind == "mtp" || kind->kind == "nemotron_h_mtp") {
        // One target block and the head over the target's active weights.
        const std::uint64_t rows = drafter.proposal_head ? kProposalHeadRows : target.vocab_size;
        if (rows == 0) { return std::nullopt; }
        size = (target.active_params / static_cast<double>(target.num_hidden_layers) +
                static_cast<double>(rows) * h) /
               target.active_params;
    }
    if (gpu.mma_slow) {
        // TU116 / TU117 and the GTX 16 boards: the verify MMA runs on the FP16 units, a cost no
        // floor measured. Not priced as the generation's tensor-core boards: unpriced, labelled.
        return RoundCost{.fit = ModelFit{false, "mma emulated"}, .priced = false};
    }
    const Generation* gen = nullptr;
    for (const Generation& g : kGenerations) {
        if (g.cc == gpu.compute_capability) { gen = &g; }
    }
    // A lab constant needs two or more cards that differ: a generation the fit saw on one card
    // (V100, RTX 2080 Ti, A100, RTX 3090: their constants stay in the table as the fit's record) or
    // never saw is unpriced, and its caller takes the conservative fallback.
    if (gen == nullptr) {
        return RoundCost{.fit = ModelFit{false, "no lab card"}, .priced = false};
    }
    if (gen->lab_cards < 2) {
        return RoundCost{.fit = ModelFit{false, "one lab card"}, .priced = false};
    }
    const Generation& priced = *gen;
    const double rate        = tensor_rate(gpu.compute_capability);

    const double flops   = rate * gpu.sms * gpu.sm_clock_khz * 1e3;
    const double inv_eta = priced.inv_eta + (target.kquant ? priced.kquant : 0.0);
    RoundCost cost{.alpha = inv_eta * 2.0 * gpu.dram_gbps * 1e9 / (flops * target.bytes_per_weight),
                   // knee and beta: one card's grid per generation; left out (see kGenerations)
                   .beta           = 0.0,
                   .knee           = 0,
                   .delta          = kind->cost * size,
                   .pass_per_draft = kind->pass_per_draft,
                   .fit            = fit_of(gpu.compute_capability, target, kind->kind)};
    if (target.experts > 0) { // the MoE facts were checked together above
        cost.moe_pick  = static_cast<double>(target.experts_per_token) / target.experts;
        cost.moe_bytes = kMoeRead * target.routed_expert_params / target.active_params;
    }
    // Rates the PCI-ID table does not vouch for (an unlisted board, a laptop, a MIG slice): priced,
    // never reliable, so callers take the conservative fallback.
    if (gpu.facts_extrapolated) { cost.fit = ModelFit{false, "extrapolated device facts"}; }
    return cost;
}

double round_time(const RoundCost& cost, std::int32_t rows, std::int32_t depth) {
    if (rows < 1 || depth < 0) { throw std::invalid_argument("round_time: rows >= 1, depth >= 0"); }
    if (!cost.priced) {
        throw std::invalid_argument("round_time: an unpriced cost (" + cost.fit.reason + ")");
    }
    const double w      = static_cast<double>(rows) * (depth + 1);
    const double passes = depth == 0 ? 0.0 : (cost.pass_per_draft ? depth : 1.0);
    // Experts w columns routed independently read over the r of one: E (1 - (1 - r/E)^w) - r.
    const double experts =
        cost.moe_bytes > 0 ? cost.moe_bytes * (1.0 - std::pow(1.0 - cost.moe_pick, w) - cost.moe_pick)
                           : 0.0;
    return 1.0 + cost.alpha * (w - 1.0) + cost.beta * std::max(0.0, w - cost.knee) +
           passes * cost.delta + experts;
}

double tokens_per_round(double a, double d, std::int32_t depth) {
    double tokens = 1.0, accepted = 1.0;
    for (std::int32_t i = 0; i < depth; ++i) {
        double p = a;
        for (std::int32_t j = 0; j < i; ++j) { p *= d; }
        accepted *= p;
        tokens += accepted;
    }
    return tokens;
}

double first_draft_acceptance(double tokens, std::int32_t depth, double d) {
    if (depth < 1 || tokens <= 1.0) { return 0.0; }
    double lo = 0.0, hi = 0.999;
    if (tokens_per_round(hi, d, depth) <= tokens) { return hi; }
    for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        (tokens_per_round(mid, d, depth) < tokens ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

std::vector<LoadBound> modeled_load(const RoundCost& cost, double a, double d,
                                    std::span<const std::int32_t> depths,
                                    std::int32_t max_rows) {
    std::vector<std::int32_t> candidates(depths.begin(), depths.end());
    candidates.push_back(0);
    std::sort(candidates.begin(), candidates.end());
    // An unpriced cost (mma_slow): plain from one row, never round_time.
    if (!cost.priced) { return {LoadBound{.lanes = 1, .depth = 0}}; }
    const auto best_at = [&](std::int32_t rows, std::int32_t ceiling) {
        std::int32_t best = 0;
        double speed      = 1.0 / round_time(cost, rows, 0);
        for (const std::int32_t k : candidates) {
            if (k == 0 || k > ceiling) { continue; }
            const double s = tokens_per_round(a, d, k) / round_time(cost, rows, k);
            if (s > speed) { speed = s; best = k; }
        }
        return best;
    };
    const std::int32_t deepest = candidates.back();
    // An unreliable fit: no deeper than kUnsureDepth at any load, as class_depths' fallback below
    // 8K.
    std::int32_t previous =
        best_at(1, cost.fit.reliable ? deepest : std::min(deepest, kUnsureDepth));
    std::vector<LoadBound> out;
    if (previous != deepest) { out.push_back(LoadBound{.lanes = 1, .depth = previous}); }
    for (std::int32_t rows = 2; rows <= max_rows; ++rows) {
        // Past the rows the cost was fitted at (the facts-only fit: one), no model: plain.
        if (rows > cost.fitted_rows) {
            if (previous != 0) { out.push_back(LoadBound{.lanes = rows, .depth = 0}); }
            break;
        }
        const std::int32_t depth = best_at(rows, previous);
        if (depth != previous) {
            out.push_back(LoadBound{.lanes = rows, .depth = depth});
            previous = depth;
        }
        if (depth == 0) { break; }
    }
    return out;
}

std::array<ClassDepth, kContextClasses>
class_depths(const std::optional<RoundCost>& cost, double a, double d,
             std::span<const std::int32_t> captured,
             const std::array<ClassFactor, kContextClasses>& factors, std::int32_t factor_depth,
             double margin) {
    std::vector<std::int32_t> depths;
    for (const std::int32_t k : captured) {
        if (k > 0) { depths.push_back(k); }
    }
    std::sort(depths.begin(), depths.end());
    depths.erase(std::unique(depths.begin(), depths.end()), depths.end());
    // The conservative choice where the model or the factor is unknown: the deepest captured depth
    // up to kUnsureDepth while the prompt is under kContextBounds[1] (8K), plain beyond.
    std::int32_t shallow = 0;
    for (const std::int32_t k : depths) {
        if (k <= kUnsureDepth) { shallow = k; }
    }
    // `cap`: no deeper than Short's pick where the model made one, so a longer prompt never
    // speculates deeper than a short one on the fallback.
    const auto unsure = [&](ContextClass c, const std::string& why, std::int32_t cap) {
        const bool near          = c == ContextClass::Short || c == ContextClass::Mid;
        const std::int32_t depth = near ? std::min(shallow, cap) : 0;
        return ClassDepth{.depth    = depth,
                          .estimate = false,
                          .reason   = "estimate: none (" + why + "): " +
                                    (near ? "depth " + std::to_string(depth) + " up to 8K"
                                          : std::string("plain past 8K"))};
    };
    constexpr std::int32_t kNoCap = std::numeric_limits<std::int32_t>::max();

    std::array<ClassDepth, kContextClasses> out{};
    for (std::size_t i = 0; i < kContextClasses; ++i) {
        const auto c = static_cast<ContextClass>(i);
        if (!cost) {
            out[i] = unsure(c, "no cost model for this card, target or drafter", kNoCap);
            continue;
        }
        // An unpriced cost (mma_slow) is never reliable; it never reaches round_time.
        if (!cost->priced || !cost->fit.reliable) {
            out[i] = unsure(c, cost->fit.reason, kNoCap);
            continue;
        }
        // Short is the baseline the factors are relative to: 1 by definition, so an unknown Short
        // factor never lowers the depth by itself. Only the other classes fall back.
        const bool baseline = c == ContextClass::Short;
        if (!baseline && !factors[i].value) {
            out[i] =
                unsure(c, std::string(class_name(c)) + " acceptance unknown: " + factors[i].reason,
                       out[0].depth);
            continue;
        }
        const double factor = baseline ? 1.0 : *factors[i].value;
        if (depths.empty() || factor_depth < 1) {
            out[i] = ClassDepth{.depth = 0, .estimate = true, .reason = "no captured depth: plain"};
            continue;
        }
        // The factor scales the accepted drafts per round at the depth it was measured at.
        const double drafts = (tokens_per_round(a, d, factor_depth) - 1.0) * factor;
        const double a_c    = baseline ? a : first_draft_acceptance(1.0 + drafts, factor_depth, d);
        // A longer prompt never speculates deeper than a short one: Mid, Long and Huge choose among
        // the captured depths up to Short's pick, even with a factor above 1.
        if (!baseline && out[0].depth == 0) {
            out[i] = ClassDepth{.depth = 0, .estimate = true, .reason = "plain: short is plain"};
            continue;
        }
        const std::int32_t ceiling =
            baseline ? std::numeric_limits<std::int32_t>::max() : out[0].depth;
        std::int32_t best   = 0;
        double speed        = 0; // over plain decode's
        for (const std::int32_t k : depths) {
            if (k > ceiling) { continue; }
            const double s =
                tokens_per_round(a_c, d, k) / round_time(*cost, 1, k) * round_time(*cost, 1, 0);
            if (s > speed) {
                speed = s;
                best  = k;
            }
        }
        char ratio[32];
        std::snprintf(ratio, sizeof ratio, "%.2f", speed);
        if (speed > 1.0 + margin) {
            out[i] =
                ClassDepth{.depth    = best,
                           .estimate = true,
                           .reason = "depth " + std::to_string(best) + " at " + ratio + "x plain"};
        } else {
            out[i] = ClassDepth{.depth    = 0,
                                .estimate = true,
                                .reason   = std::string("plain: the best captured depth gives ") +
                                          ratio + "x, not past the margin"};
        }
    }
    return out;
}

} // namespace ninferno::spec
