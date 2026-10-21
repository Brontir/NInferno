#pragma once

// The speculation load model: which draft depth pays at each number of rows decoding together,
// from the card's cost of a verify round and the drafter's acceptance, rather than a measured
// table per GPU class (decisions from card properties; measured tables are only
// overrides). Pure functions, tested without a GPU.
//
// A round of B rows at depth k verifies w = B(k+1) columns. Relative to a plain step of one row
// (grids on the RTX 4090, RTX 5090 and V100, 27B, per-request decode speed while in flight):
//   t(B, k) / t0 = 1 + alpha (w - 1) + beta max(0, w - knee) + passes * delta
//                    + moe_bytes ((1 - (1 - pick)^w) - pick),
// passes k for an MTP round (one drafter pass per draft), 1 for a DFlash round (one block). The
// last term is an MoE target's: w columns routed apart read more experts than one does.
// Worst error 5.6-7.8 % over plain, MTP 1/2/3/5 and DFlash2 3/5/7 at C = 1, 2, 4, 8.
//
// round_cost_of prices a card and a model from facts alone (the chip's rates, the artifact's shape,
// quantization and experts, the drafter's kind), never a timing. At one row t(1, k) / t0 is the
// registry's floor (spec-tune's seconds per round over seconds per plain token); the constants are
// fitted on those floors (audit1018/x3): 9.94 % rms in sample (test_spec; 9.8 % in x3_fit.py before
// rounding), 21 % with a card held out, 18 % with a model held out.
//
// One row only: the floors are measured at C1, so round_cost_of's constants (alpha, delta, the MoE
// term) are fitted at rows = 1; beta and the knee come from older multi-row grids. A multi-row use
// (modeled_load at rows > 1) waits for a multi-row fit and must not take this one's word for it.
//
// The i-th draft of a round is accepted, given the earlier ones, with probability a d^(i-1): the
// acceptance falls along the block (RTX 4090 grid: DFlash2 a 0.75, d 0.97; MTP a 0.77, d 0.91 give
// the measured tokens per round at every depth to 0.02). A round of k drafts then yields
// E(k) = 1 + sum_{j=1..k} prod_{i=1..j} a d^(i-1) tokens, and a row's speed is E(k) / t(B, k).

#include "spec/context_class.h"
#include "spec/profile.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninferno::spec {

// Whether the fit stands behind a prediction. Unreliable where the floors do not test it: a
// generation with fewer than two lab cards measured, a generation and model family whose error on
// a card or a model left out of the fit passes kClassMargin (10 %), a family the generation never
// measured, a generation the table lacks (then priced as its costliest measured one), or a board
// without tensor cores (unpriced, "mma emulated"). A caller should not trust the depth it picks
// there: decode plainly, or no deeper than 3.
struct ModelFit {
    bool reliable = true;
    std::string reason; // why not ("" when reliable)
};

struct RoundCost {
    double alpha = 0;      // per verify column beyond the first
    double beta  = 0;      // per column beyond `knee`
    std::int32_t knee = 0;
    double delta = 0;      // per drafter pass
    bool pass_per_draft = true; // MTP: k passes; DFlash: one per round
    // An MoE target: the routed experts' weights a step can read on top of a plain step's, over the
    // weights a plain step reads, times the share of them that routing actually adds (0: dense),
    // and one column's chance to pick an expert (experts per token / experts).
    double moe_bytes = 0;
    double moe_pick  = 0;
    ModelFit fit;
    // false: no price at all (fit unreliable, its reason says why: "mma emulated" on a board
    // without tensor cores); round_time refuses it, so the conservative fallback is the only use.
    bool priced = true;
};

// What the artifact says about the target (no measurement).
struct TargetFacts {
    double active_params    = 0; // weights one token reads (an MoE: shared plus its routed experts)
    double bytes_per_weight = 0; // 0.53: Q4_G64
    bool kquant             = false; // GGUF k- or IQ-quant weights: verified by the k-quant route
    std::uint64_t hidden_size = 0, num_hidden_layers = 0, vocab_size = 0;
    // An MoE target: routed experts, experts per token, the parameters of all routed experts (0: dense).
    int experts = 0, experts_per_token = 0;
    double routed_expert_params = 0;
    std::string family; // the registry's engine ("gemma4", "qwen3_5", "nemotron_h", "dense")
    // The weight formats the artifact holds, spelt as artifact::format_name spells a QType
    // ("mxfp4", "nvfp4", "q4_g64_fp16", "q8_g32_fp16", "bf16", "ggml_q4_k", "ggml_iq3_xxs", ...):
    // the set LoadSummary::weight_formats collects. A covered pair needs every format covered: the
    // set is the covered quantized format plus only unquantized ones (bf16, fp32, int32: norms,
    // embeddings, router). Empty (unknown), another quantized format beside it, or none of it:
    // unsure.
    std::vector<std::string> weight_formats;
};

struct DrafterFacts {
    std::string kind;           // the registry's kind: mtp, gemma4_assistant, nemotron_h_mtp, dflash, dflash2
    bool proposal_head = false; // +head: Qwen's 131072-row proposal head instead of the vocabulary
};

// The round cost of `gpu` for `target` drafted by `drafter`, from facts: a verify column costs 2
// flops a weight at the tensor cores' peak, a plain step reads every active weight once, so
// alpha = (1/eta + kquant) 2 dram / (rate * sms * clock * bytes_per_weight), rate the generation
// row's tensor rate per SM and clock (core/generation.h), sms, clock and dram the board's facts
// (GpuKey from DeviceClass::facts), 1/eta per generation (V100: about 11 at its FP16 tensor rate,
// its verify route) and kquant more for GGUF k/IQ quants (0.46 at the fit's rate: 0.69 on Volta,
// 1.38 on Turing in the generation row's unit). An mma_slow board: an unpriced cost labelled "mma
// emulated".
// delta is the drafter's pass, a cost per kind times its size (MTP: one target block and the head;
// the others: hidden_size^2) over the target's active weights. An MoE target reads 0.30 of the
// experts independent routing of the w columns would add. The knee is where the verify GEMM takes
// its next tile (RTX 4090 8 columns, RTX 5090 12, A100 24, RTX 3090 32; measured grids, none on
// Volta or Turing). nullopt: unknown rates or shape, or a drafter kind the fit has no cost for.
[[nodiscard]] std::optional<RoundCost> round_cost_of(const GpuKey& gpu, const TargetFacts& target,
                                                     const DrafterFacts& drafter);

// The relative time of a round of `rows` rows at `depth` drafts (0: plain decode). Throws
// std::invalid_argument for an unpriced cost: check RoundCost::priced (class_depths and
// modeled_load do).
[[nodiscard]] double round_time(const RoundCost& cost, std::int32_t rows, std::int32_t depth);

// E(depth) for first-draft acceptance `a` and decay `d` along the block.
[[nodiscard]] double tokens_per_round(double a, double d, std::int32_t depth);

// The first-draft acceptance that gives `tokens` per round at `depth` with decay `d` (a profile
// measures one depth); clamped to [0, 0.999].
[[nodiscard]] double first_draft_acceptance(double tokens, std::int32_t depth, double d);

// The depth of `depths` (or 0, plain decode) that maximizes a row's speed at each row count
// 1..max_rows, as a load table: an entry {lanes, depth} bounds every step of `lanes` rows or more
// at `depth` (parse_load's "lanes:depth"); one row runs the deepest that pays, unbounded when that
// is the deepest of `depths`. A depth never rises again once it fell (a row turned plain does not
// resume). An unpriced cost: plain from one row ("1:0"); an unreliable fit (RoundCost::fit): no
// deeper than kUnsureDepth at any load. The load table knows no prompt length: it is not
// class_depths' rule, which under the same fit runs at most kUnsureDepth below 8K and plain past
// it. A caller that has both takes the lower depth of the two, so past 8K an unsure fit decodes
// plainly whatever this table allows. Past one row this uses the one-row fit beyond what it was
// fitted on (see the header comment).
[[nodiscard]] std::vector<LoadBound> modeled_load(const RoundCost& cost, double a, double d,
                                                  std::span<const std::int32_t> depths,
                                                  std::int32_t max_rows);

// Context classes (idea 32, step 3): the draft depth each prompt-length class should run at one
// row, from the round cost and the class's acceptance factor (spec/context_class.h). Lab
// constants, not timings: a depth must beat plain decode by kClassMargin, the cost model's rms
// error over the registry's floors in sample (9.94 %), and where anything is unknown the choice is
// kUnsureDepth at most below 8K and plain beyond.
inline constexpr double kClassMargin       = 0.10;
inline constexpr std::int32_t kUnsureDepth = 3;

struct ClassDepth {
    std::int32_t depth = 0; // a captured depth, or 0: plain decode
    // Whether the cost model and the class factor chose it (false: the conservative rule).
    bool estimate = false;
    std::string reason; // always set; the conservative rule's starts "estimate: none ("
};

// For each class: among the captured depths (`captured`, the graphs the engine holds; 0s
// ignored), the one that maximizes tokens_per_round(a_c, d, k) / round_time(cost, 1, k), when it
// beats plain by (1 + margin); otherwise 0. a_c is the first-draft acceptance whose accepted drafts
// per round at `factor_depth` (the depth the factors were measured at) are the class factor times
// those of `a`. Mid, Long and Huge choose only among the captured depths up to Short's pick, even
// with a factor above 1: a longer prompt never speculates deeper than a short one. A class whose
// factor is unknown under a reliable cost takes the conservative depth, also capped at Short's
// pick. Short is the baseline (factor 1 by
// definition, whatever `factors` says of it). No cost, an unreliable or unpriced fit
// (RoundCost::fit, "mma emulated") or an unknown factor of another class: the deepest captured
// depth up to kUnsureDepth for Short and Mid, 0 for Long and Huge, labelled "estimate: none
// (<why>)". No caller yet: routing a request by class is the next step.
[[nodiscard]] std::array<ClassDepth, kContextClasses>
class_depths(const std::optional<RoundCost>& cost, double a, double d,
             std::span<const std::int32_t> captured,
             const std::array<ClassFactor, kContextClasses>& factors, std::int32_t factor_depth,
             double margin = kClassMargin);

} // namespace ninferno::spec
