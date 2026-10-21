#include "spec/load_model.h"

#include "core/generation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string_view>

namespace ninferno::spec {

namespace {

// The fitted constants: audit1018/x3/x3_fit.py, form M3, least squares over the 116 non-adaptive
// spec-tune floors of src/spec/data/drafter_registry.json at 3db81deeb (9.8 % rms; 21 % on a card
// left out of the fit, 18 % on a model left out). Refit them with the script when the registry
// gains floors. knee and beta come from the measured grids instead (27B Q4_G64, 8K, several rows):
// at one row the floors never reach a knee.
//
// The tensor rate per SM and clock is the chip's fact from core/generation.h (GenerationRow::bf16:
// 512 on the consumer parts, 2048 on the A100, the fitted 1536 where Volta and Turing emulate the
// BF16 MMA), so inv_eta is in units of that rate: the fit's V100 11.45 and RTX 2080 Ti 0.3337 at
// their FP16 tensor rates (1024, 512) are 17.18 and 1.001 here, the same alpha.
struct Generation {
    int cc;
    double inv_eta; // verify column cost over the generation row's tensor rate (Volta: its route)
    std::int32_t knee; // 0: no grid measured one
    double beta;
    int lab_cards; // cards of the generation whose floors the fit saw
};

constexpr Generation kGenerations[] = {
    {70, 17.18, 0, 0.0, 1},       // V100
    {75, 1.001, 0, 0.0, 1},       // RTX 2080 Ti
    {80, 2.793, 24, 0.0045, 1},   // A100
    {86, 0.9557, 32, 0.0155, 1},  // RTX 3090
    {89, 0.3600, 8, 0.0069, 5},   // RTX 4090, 4070 Ti, 4070 Ti SUPER, 4060 Ti, L4
    {120, 0.4582, 12, 0.0122, 4}, // RTX 5090, 5080, 5070 Ti, 5060 Ti
};

double tensor_rate(int cc) noexcept { return generation_row(cc / 10, cc % 10).bf16; }

constexpr double kKquantExtra = 0.46;   // GGUF k/IQ-quant verify, in units of the tensor peak's
constexpr double kMoeRead     = 0.3019; // of the experts independent routing would add
constexpr std::uint64_t kProposalHeadRows = 131072;
struct DrafterKind {
    std::string_view kind;
    double cost; // per pass, over the drafter's size relative to the target's active weights
    bool pass_per_draft;
};
constexpr DrafterKind kDrafterKinds[] = {
    {"mtp", 0.7232, true},
    {"gemma4_assistant", 67.09, true},
    {"nemotron_h_mtp", 1.429, true},
    {"dflash", 283.1, false},
    {"dflash2", 329.2, false},
};
// The generation x family pairs the fit measured on two or more cards with at most 15 % rms error
// on a card left out of the fit (x3_fit.py's leave-card-out by cc and engine). sm89 gemma4 (16 %)
// and sm120 qwen3_5 (32 %, the RTX 5090 left out) miss it.
struct Tested {
    int cc;
    std::string_view family;
};
constexpr Tested kTested[] = {
    {89, "dense"}, {89, "nemotron_h"}, {89, "qwen3_5"},
    {120, "dense"}, {120, "gemma4"}, {120, "nemotron_h"},
};

ModelFit fit_of(const Generation* g, int cc, const std::string& family) {
    const std::string arch = "sm" + std::to_string(cc);
    if (g == nullptr) {
        return {false, arch + ": a generation the fit never measured, priced as its costliest"};
    }
    if (g->lab_cards < 2) { return {false, arch + ": fewer than two lab cards measured"}; }
    for (const Tested& t : kTested) {
        if (t.cc == cc && t.family == family) { return {}; }
    }
    return {false, arch + " " + family + ": not within 15 % on a card left out of the fit"};
}

} // namespace

std::optional<RoundCost> round_cost_of(const GpuKey& gpu, const TargetFacts& target,
                                       const DrafterFacts& drafter) {
    if (gpu.sms <= 0 || gpu.sm_clock_khz <= 0 || gpu.dram_gbps <= 0 ||
        target.bytes_per_weight <= 0 || target.active_params <= 0 || target.hidden_size == 0 ||
        target.num_hidden_layers == 0) {
        return std::nullopt;
    }
    const DrafterKind* kind = nullptr;
    for (const DrafterKind& k : kDrafterKinds) {
        if (k.kind == drafter.kind) { kind = &k; }
    }
    if (kind == nullptr) { return std::nullopt; }
    const Generation* gen = nullptr;
    for (const Generation& g : kGenerations) {
        if (g.cc == gpu.compute_capability) { gen = &g; }
    }
    // A generation the table lacks: the costliest tensor-core generation measured (Volta's own
    // route aside), per unit of tensor rate, at its own generation row's rate; no knee.
    const double rate = tensor_rate(gpu.compute_capability);
    Generation priced{gpu.compute_capability, 0, 0, 0.0, 0};
    for (const Generation& g : kGenerations) {
        if (g.cc != 70) {
            priced.inv_eta = std::max(priced.inv_eta, g.inv_eta / tensor_rate(g.cc) * rate);
        }
    }
    if (gen != nullptr) { priced = *gen; }

    const double flops   = rate * gpu.sms * gpu.sm_clock_khz * 1e3;
    const double inv_eta = priced.inv_eta + (target.kquant ? kKquantExtra : 0.0);
    const double h = static_cast<double>(target.hidden_size);
    double size = h * h / target.active_params; // a DFlash block or an assistant: about h^2 a layer
    if (kind->kind == "mtp" || kind->kind == "nemotron_h_mtp") {
        // One target block and the head over the target's active weights.
        const std::uint64_t rows = drafter.proposal_head ? kProposalHeadRows : target.vocab_size;
        if (rows == 0) { return std::nullopt; }
        size = (target.active_params / static_cast<double>(target.num_hidden_layers) +
                static_cast<double>(rows) * h) / target.active_params;
    }
    RoundCost cost{.alpha = inv_eta * 2.0 * gpu.dram_gbps * 1e9 / (flops * target.bytes_per_weight),
                   .beta  = priced.beta,
                   .knee  = priced.knee,
                   .delta = kind->cost * size,
                   .pass_per_draft = kind->pass_per_draft,
                   .fit = fit_of(gen, gpu.compute_capability, target.family)};
    if (target.experts > 0 && target.experts_per_token > 0 && target.routed_expert_params > 0) {
        cost.moe_pick  = static_cast<double>(target.experts_per_token) / target.experts;
        cost.moe_bytes = kMoeRead * target.routed_expert_params / target.active_params;
    }
    return cost;
}

double round_time(const RoundCost& cost, std::int32_t rows, std::int32_t depth) {
    if (rows < 1 || depth < 0) { throw std::invalid_argument("round_time: rows >= 1, depth >= 0"); }
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
    std::int32_t previous      = best_at(1, deepest);
    std::vector<LoadBound> out;
    if (previous != deepest) { out.push_back(LoadBound{.lanes = 1, .depth = previous}); }
    for (std::int32_t rows = 2; rows <= max_rows; ++rows) {
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
    const auto unsure = [&](ContextClass c, const std::string& why) {
        const bool near = c == ContextClass::Short || c == ContextClass::Mid;
        return ClassDepth{.depth    = near ? shallow : 0,
                          .estimate = false,
                          .reason   = "estimate: none (" + why + "): " +
                                    (near ? "depth " + std::to_string(shallow) + " up to 8K"
                                          : std::string("plain past 8K"))};
    };

    std::array<ClassDepth, kContextClasses> out{};
    for (std::size_t i = 0; i < kContextClasses; ++i) {
        const auto c = static_cast<ContextClass>(i);
        if (!cost) {
            out[i] = unsure(c, "no cost model for this card, target or drafter");
            continue;
        }
        if (!cost->fit.reliable) {
            out[i] = unsure(c, "cost model unsure: " + cost->fit.reason);
            continue;
        }
        if (!factors[i].value) {
            out[i] =
                unsure(c, std::string(class_name(c)) + " acceptance unknown: " + factors[i].reason);
            continue;
        }
        if (depths.empty() || factor_depth < 1) {
            out[i] = ClassDepth{.depth = 0, .estimate = true, .reason = "no captured depth: plain"};
            continue;
        }
        // The factor scales the accepted drafts per round at the depth it was measured at.
        const double drafts = (tokens_per_round(a, d, factor_depth) - 1.0) * *factors[i].value;
        const double a_c    = first_draft_acceptance(1.0 + drafts, factor_depth, d);
        std::int32_t best   = 0;
        double speed        = 0; // over plain decode's
        for (const std::int32_t k : depths) {
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
