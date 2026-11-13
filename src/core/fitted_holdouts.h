#pragma once

// The holdout records (core/holdout.h) of the fitted constants, in one place: the route cost model
// (ops/generic/route_cost_constants.h), the choice's speed model (product/choice/
// choice_constants.h), the generation row's fitted scales and measured preferences
// (core/generation.h), the speculation round cost (spec/round_cost.h, spec/moe_routed_share.h),
// the one fitted SM-wave rule (ops/common/sm_wave_rules.h), the generic path's literal rules
// tagged Basis::Fitted or ScaledFromOneBoard (ops/generic/route_rules.h), and the kernels' and
// dispatch's untagged launch constants a sweep of src/ found (kKernelHoldouts: a constexpr literal
// whose comment says measured, tuned, optimum, fastest, swept or fitted; the sweep is not a test,
// and untagged constants it missed have no record). A threshold written as a literal inside a
// function is invisible to the test: the measured ones of the plans swept in #718 are named
// constants with records, and the literals left there are coverage bounds (a compiled tile's width,
// a kernel's range, the prefill class's first width) or carry no stated measurement (the #718 PR
// lists them). A record per constant or per table of one fit: the table's rows are one fit over
// the same data, so one record says what it was fitted on and what, if anything, was left out; the
// per-row card counts stay in the tables (`cards`).
//
// Checked only where a script and data in the repository reproduce the held-out score of the
// shipped value (none today: a held-out score quoted from a fit whose data is not in the
// repository is a None record that quotes it).
//
// The records are a side table keyed by the name of the definition in its file, never a field of
// the constant: values are not touched here, the records say where each value comes from, so that
// a fitted constant without a held-out check is declared so, not forgotten.
// tests/test_fitted_holdouts.cpp says what it checks.
// Sources are the comments at the constants and the DECISIONS (docs/maintainer/decisions.md) they
// name.

#include "core/holdout.h"

#include <string_view>

namespace ninferno::fitted {

struct FittedHoldout {
    std::string_view file;     // the source file, relative to the repository root
    std::string_view constant; // the name in that file (a table, a scalar or a GenerationRow field)
    Holdout holdout;
    std::string_view row =
        {}; // one row of a table with a record of its own (kSmallTEta: the generation)
};

inline constexpr std::string_view kRouteCost     = "src/ops/generic/route_cost_constants.h";
inline constexpr std::string_view kChoice        = "src/product/choice/choice_constants.h";
inline constexpr std::string_view kGeneration    = "src/core/generation.h";
inline constexpr std::string_view kRoundCost     = "src/spec/round_cost.h";
inline constexpr std::string_view kCostConstants = "src/core/cost_constants.h";
inline constexpr std::string_view kMoeShare      = "src/spec/moe_routed_share.h";
inline constexpr std::string_view kWave          = "src/ops/common/sm_wave_rules.h";
inline constexpr std::string_view kRouteRules    = "src/ops/generic/route_rules.h";
// Kernels' and dispatch's own fitted launch constants (no tag in their files; the test checks each
// key is a definition there, it does not audit the files' other definitions).
inline constexpr std::string_view kInt8PromptCost =
    "src/ops/softmax_attention/dense/causal_cache/int8/prompt_fast_cost.h";
inline constexpr std::string_view kGemma4Attention = "src/ops/softmax_attention/gemma4/launch.cu";
inline constexpr std::string_view kQ5LinearAdd     = "src/ops/linear_add/q5/q5_linear_add_plan.cpp";
inline constexpr std::string_view kGdnConv         = "src/ops/gdn_input_proj/gdn_projected_conv.cu";
inline constexpr std::string_view kRmsNormRopeBound = "src/ops/rmsnorm_rope/rmsnorm_rope.cpp";
inline constexpr std::string_view kAttnInputPlan =
    "src/ops/attn_input_proj/bf16/bf16_attn_input_plan.h";
inline constexpr std::string_view kW4a8 = "src/ops/generic/generic_w4a8.cuh";
inline constexpr std::string_view kSparseMoePrefill =
    "src/ops/sparse_moe/prefill/sparse_moe_prefill.h";
inline constexpr std::string_view kQwenDecode   = "src/models/qwen3_5/program/decode.cpp";
inline constexpr std::string_view kGemma4Kernel = "src/ops/softmax_attention/gemma4/kernel.cuh";
inline constexpr std::string_view kBf16Vector   = "src/ops/common/bf16_vector.cuh";
inline constexpr std::string_view kQ8LinearAdd  = "src/ops/linear_add/q8/q8_linear_add_plan.cpp";
inline constexpr std::string_view kCapturePlanner =
    "src/runtime/engine/context_cache/capture_planner.h";
inline constexpr std::string_view kMoeInternal = "src/ops/moe/moe_internal.h";
inline constexpr std::string_view kQ4LinearSwiglu =
    "src/ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cu";
inline constexpr std::string_view kRope    = "src/ops/launcher/rope.cu";
inline constexpr std::string_view kRmsNorm = "src/ops/launcher/rmsnorm.cu";
inline constexpr std::string_view kMoePrefill =
    "src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu";
inline constexpr std::string_view kGdnGating =
    "src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp";
inline constexpr std::string_view kRmsNormRope = "src/ops/rmsnorm_rope/launch.h";
// The plans' measured thresholds and per-chip band tables (#718): thresholds written as literals
// inside a function are invisible to the test, so the measured ones are named there.
inline constexpr std::string_view kDynamicConvPrepare =
    "src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.cpp";
inline constexpr std::string_view kQ4Q5GdnInput =
    "src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp";
inline constexpr std::string_view kQ4Q5AttnInput =
    "src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp";
inline constexpr std::string_view kBf16CausalPlan =
    "src/ops/softmax_attention/dense/causal_cache/bf16/plan.cpp";
inline constexpr std::string_view kInt8CausalPlan =
    "src/ops/softmax_attention/dense/causal_cache/int8/plan.cpp";
inline constexpr std::string_view kNvfp4AttnInput =
    "src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp";
inline constexpr std::string_view kNvfp4LinearAdd =
    "src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp";
inline constexpr std::string_view kNvfp4LinearSwiglu =
    "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp";
inline constexpr std::string_view kFp8LinearAdd = "src/ops/linear_add/fp8/fp8_linear_add_plan.cpp";
inline constexpr std::string_view kSparseMoeSmallT =
    "src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp";

// --- route_cost_constants.h: the A16 / A8 refits and the per-generation modelling constants ---
// The A16 tiles: one leave-one-board-out refit of the model's constants.
inline constexpr Holdout h_a16_refit = Holdout::quoted(
    "ninferno_route_tune timings of seven boards (each tile in the engine, L2 flushed; BF16, Q4 "
    "and Q8, T >= 17); the record does not name all seven",
    "a held-out score quoted from the original fit, whose timings and refit are not in the "
    "repository: each board held out in turn, geomean time over the best tile 1.095 of the best "
    "(RTX 4090 1.083, RTX 5090 1.068) against 1.110 for the F-lite rule and 1.137 without the "
    "chain "
    "and reduction terms",
    "generic_cost.cpp price_routes comment",
    "one aggregate for the constants of the A16 model, no per-constant error; measured at 0.1.0 "
    "(54ef5203), before #499 re-priced the K-quant register MMA on sm_86 / sm_120");
// The same refit, for the two constants whose flatness was also scanned: the record of h_a16_refit
// with its own note.
inline constexpr Holdout h_a16_refit_flat = [] {
    Holdout h = h_a16_refit;
    h.note    = "the optimum is flat: this constant at 0.5..1.5x (register-MMA step) or 0.5..2x "
                "(pipeline stage) of its fit costs under 0.2 %; measured at 0.1.0 (54ef5203), "
                "before #499 re-priced the K-quant register MMA on sm_86 / sm_120";
    return h;
}();
// kMinWarpFill is used by the A16 model and by the W4A8 and I8 tiles, so it rests on both refits.
// Like its siblings it is named by neither comment, only covered as one of "the constants".
inline constexpr Holdout h_min_warp_fill = Holdout::quoted(
    "A16: ninferno_route_tune timings of seven boards (BF16, Q4, Q8, T >= 17); A8: --a8 timings of "
    "RTX 4090, RTX 5090, RTX 4060 Ti, L4, A10, A100",
    "held-out scores quoted from the two original fits, whose timings and refits are not in the "
    "repository: each board held out in turn, A16 1.095 of the best (4090 1.083, 5090 1.068), A8 "
    "1.06..1.09 on the consumer boards (A100 1.13 in sample only); no error for this floor alone",
    "generic_cost.cpp price_routes comment; generic_gemm.cpp route_tile comment",
    "the A16 figures were measured at 0.1.0 (54ef5203), before #499 re-priced the K-quant register "
    "MMA on sm_86 / sm_120");
// The A8 tiles (W4A8, I8, ggml MMQ) and their activation quantizer.
inline constexpr Holdout h_a8_refit = Holdout::quoted(
    "ninferno_route_tune --a8 and --a8 --epilogue timings of RTX 4090, RTX 5090, RTX 4060 Ti, L4, "
    "A10, A100 (six formats, n 48..28672, T 17..2048, quantizer in the sample; SwiGLU and "
    "AddResidual on the A100)",
    "a held-out score quoted from the original fit, whose timings and refit are not in the "
    "repository: each board held out in turn, time over the best route of either class 1.06..1.09 "
    "on the consumer boards (the A100, the minor-0 row's one board, 1.13 in sample only)",
    "generic_gemm.cpp route_tile comment",
    "one aggregate for the A8 constants; the record gives no per-constant error. Turing (priced "
    "since #781) carries them from these generations, no Turing board in the fit: on the RTX 2080 "
    "Ti A5 sweep the W4A8 tiles read 0.70..0.74 and ggml MMQ 0.58..0.75 of the measured time "
    "(band geomeans), not refit; OPEN: MMQ's dequantize on 64 CUDA-core lanes a SM on Turing "
    "against 128 on Ampere and Ada");
// The register MMA of the K-quant and codebook decoders (#499, #301).
inline constexpr Holdout h_rmma_heavy_warps = Holdout::none(
    "RTX 3060 + RTX 3090 and RTX 5090 + RTX 5070 Ti, cost-model replica",
    "6 and 8 warps give the same regret on both generations' cards; two warps for each of the four "
    "schedulers is the least that keeps them fed, the step latency that would derive it is not "
    "measured",
    "#499 Constants table");
inline constexpr Holdout h_l2_ga10x = Holdout::none(
    "RTX 3060 + RTX 3090 together (12/16/24/32/48 tried; replica geomean regret 3060 1.068, 3090 "
    "1.098; master 1.165, 1.174)",
    "after the refit on both cards neither is held out; a third GA10x board takes the value as a "
    "transfer",
    "#499 Constants table and gate; core/generation.h rmma_terms_measured",
    "the RTX 3090 was out of sample only for the first version of the terms: production geomean "
    "1.136 -> 1.091 (90 cases, 2 passes x 7 samples, worst 1.83 -> 1.83)");
inline constexpr Holdout h_l2_gb20x = Holdout::none(
    "RTX 5090 + RTX 5070 Ti together (replica geomean regret 5090 1.047, 5070 Ti 1.041; master "
    "1.135, 1.080; 16 gives 1.048 / 1.043, 32 gives 1.064 / 1.050)",
    "after the refit on both cards neither is held out; the 24 comes from the replica and was not "
    "re-timed on the cards",
    "#499 Constants table and gate; core/generation.h rmma_terms_measured",
    "the RTX 5070 Ti was held out for the first version of the terms only");
inline constexpr Holdout h_spill_accesses = Holdout::none(
    "a modelling assumption (one store and one load of a spilled byte a step), not a fit to data",
    "the spill term as a whole (with or without) moves the regret by under 0.5 % where it applies; "
    "no scan of this value was run",
    "#499 Constants table");

// The decode model's rows (kDramEta, kKernelCost, kRowsEta): fitted on the census and the steps of
// the fixtures. The 200 dense decode steps at 1..8 rows that were predicted before they were
// measured (2026-10-31) are in the fixture now and the rows were refit on them, so nothing is
// left out today: None, with the old prediction kept in the note.
inline constexpr Holdout h_decode_model = Holdout::none(
    "decode_census.json and decode_measured.json: Qwen3.5-9B Q4_K_M and Q8_0, Gemma 4 12B Q4_K_M, "
    "1/2/4/8 streams, seven cards (master f68eca34; the first census 03afc418, 2026-10-28)",
    "the 200 dense decode steps once predicted before they were measured are part of the fitted "
    "fixture now: no step is left out",
    "tests/test_choice.cpp (dense step comment); DECISIONS 60, 72",
    "the prediction as it stood (2026-10-31): 6.0 % rms, 1 row 3.0 %, 8 rows -6 % on average (the "
    "estimate slower), worst the RTX 2080 Ti's Q6_K at 2 and 4 rows (+25, +20 %); one aggregate "
    "for the three tables");

// The graph-node launch time t0 (core/cost_constants.h kLaunchNode, one row; its provenance tag
// route_cost::kLaunchNodeAda): the Ada census's intercept, read by the speculation round cost.
inline constexpr Holdout h_launch_node = Holdout::none(
    "the Ada census: Qwen3.5-9B Q4_K_M plain decode on the RTX 4090, 4070 Ti and 4060 Ti, every "
    "kernel within 2 %: t0 1.5 us without PDL",
    "the three cards are the fit; a generation without a row reads 0 (launches then only break a "
    "tie): Ampere and Turing graph gaps were measured (RTX 2080 Ti 0.38 us a kernel, RTX 3060 ~0) "
    "but no intercept was fitted; gb20x's two cards cannot tell 0.5 from 1.5 us",
    "core/cost_constants.h at LaunchNode; spec/round_cost.h at LaunchTime");

inline constexpr FittedHoldout kRouteCostHoldouts[] = {
    {kRouteCost, "kRmmaStepCycles", h_a16_refit_flat},
    {kRouteCost, "kRmmaSaturatingWarps", h_a16_refit},
    {kRouteCost, "kRmmaHeavySaturatingWarps", h_rmma_heavy_warps},
    {kRouteCost, "kL2BytesPerSmClockGa10x", h_l2_ga10x},
    {kRouteCost, "kL2BytesPerSmClockGb20x", h_l2_gb20x},
    {kRouteCost, "kSpillAccessesPerStep", h_spill_accesses},
    {kRouteCost, "kClusterReduceCycles", h_a16_refit},
    {kRouteCost, "kWarpReduceCycles", h_a16_refit},
    {kRouteCost, "kPipelinedStageCycles", h_a16_refit_flat},
    {kRouteCost, "kPipelinedSaturatingWarps", h_a16_refit},
    {kRouteCost, "kStagedTensorFraction", h_a16_refit},
    {kRouteCost, "kTiledTensorFraction", h_a16_refit},
    {kRouteCost, "kMinWarpFill", h_min_warp_fill},
    {kRouteCost, "kLaunchNodeAda", h_launch_node},
    {kRouteCost, "kQuantLaunchSeconds", h_a8_refit},
    {kRouteCost, "kQuantBytesPerValue", h_a8_refit},
    {kRouteCost, "kW4a8Efficiency", h_a8_refit},
    {kRouteCost, "kW4a8StepCycles", h_a8_refit},
    {kRouteCost, "kW4a8ProducerStepCycles", h_a8_refit},
    {kRouteCost, "kI8Efficiency", h_a8_refit},
    {kRouteCost, "kI8StageCycles", h_a8_refit},
    {kRouteCost, "kI8SaturatingWarps", h_a8_refit},
    {kRouteCost, "kMmqEfficiency", h_a8_refit},
    {kRouteCost, "kMmqChainK", h_a8_refit},
    {kRouteCost, "kMmqChainCycles", h_a8_refit},
    {kRouteCost, "kMmqSliceSeconds", h_a8_refit},
};

inline constexpr FittedHoldout kChoiceHoldouts[] = {
    {kChoice, "kDramEta", h_decode_model},
    {kChoice, "kDramEtaOf",
     Holdout::none("the Quadro GV100's HBM2 with ECC: 660 of 870 GB/s (PR #168's census, one card)",
                   "one card: no board was left out, and its boards' estimate only is priced by it",
                   "choice_constants.h at DramEta")},
    {kChoice, "kKernelCost", h_decode_model},
    {kChoice, "kRowsEta", h_decode_model},
    {kChoice, "kWeightSmRate",
     Holdout::none(
         "the Quadro GV100's census of one row (Qwen3.5 9B and Gemma 4 12B Q4_K_M, the cool decode "
         "steps less the rows' own bytes and the launches, fit_decode.py): one card",
         "one card, an estimate only (Volta has no kReferenceCards row): nothing left out",
         "DECISIONS 72")},
    {kChoice, "kR8IssueRate",
     Holdout::none(
         "the mean over the formats and cards whose R8 kernel time less the launch exceeds 1.08 x "
         "its DRAM time: RTX 3090 2.97 and 3060 2.77 (ncu counts, route-tune --a8-candidates, "
         "master 75b7209e, 2026-11-06)",
         "the two GA10x cards are the whole sample; Blackwell (1.92, 2.43) and Turing (1.75-2.85) "
         "disagree with it, so the reads there stay at the DRAM rate: not a law yet",
         "choice_constants.h at R8IssueRate")},
    {kChoice, "kMoeRoutedEta",
     Holdout::none(
         "the MoE step rows of speed_measured.json: Ada RTX 4090 (UD-Q4_K_M, UD-Q3_K_S) and RTX "
         "4060 Ti (UD-Q3_K_S); Blackwell RTX 5070 Ti (UD-IQ3_XXS, UD-Q3_K_S) and 5090 (UD-Q3_K_S); "
         "the mean over the generation's cards (tools/fits/speed_model/fit_moe_decode.py); one "
         "kernel pair, R8 gate/up over the register-MMA down (the op's route at 2-8 rows there)",
         "none left out; the check is the same rows: MoE decode 33 points rms 4.5 % worst 13.5 %; "
         "Ada's 8-row point rests on one card; no other kernel pair measured (a step the op runs "
         "on another pair takes these rows, transferred)",
         "DECISIONS 65, 79; tests/test_choice.cpp")},
    {kChoice, "kPrefillEta",
     Holdout::none(
         "tools/fits/prefill/fit_prefill.py on tools/fits/data/prefill/rows.json: 82 points, "
         "Qwen3.5 "
         "9B, Qwen3.8 27B, Gemma 4 12B / 26B-A4B / 31B, Qwen3.6 35B-A3B at ~2K / 7K / 21K-token "
         "prompts on the RTX 2080 Ti, 3060, 4060 Ti, 4070 Ti, 4090, 5070 Ti, 5090, and the 22K "
         "points "
         "of speed_measured.json (2026-10-27); Turing's int8 row: one census (RTX 2080 Ti, "
         "2026-10-29); the Quadro GV100 is left out of the fit",
         "fitted and checked on the same points: from 7K tokens on dense 6.4 % rms (worst 18.7 %), "
         "MoE 8.3 % (18.9 %); at 2K the estimate runs up to 30 % above a measured prompt rate (a "
         "request's fixed time is not modeled); the exact (BF16) rows were fitted before the "
         "attention term and keep that fit's time",
         "choice_constants.h at the file head and at PrefillEta")},
    {kChoice, "kPrefillTilesEta",
     Holdout::none(
         "the census of one ~1.8K-token prompt (nsys; the RTX 4090, 4070 Ti, RTX 3060 and RTX 5070 "
         "Ti "
         "among them; Turing's one card)",
         "no board left out; Ampere takes Ada's int8 row (the K step is about 12 SM-clocks on "
         "every "
         "generation with the int8 MMA: Ada 12.1, 4070 Ti 12.8, RTX 3060 12.5, Blackwell 11.6)",
         "DECISIONS 66")},
    {kChoice, "kLoneTile",
     Holdout::none(
         "the Gated DeltaNet b/a projection of Qwen3.5 9B (16 CTAs, K 4096; the census's 48 calls "
         "a "
         "prompt) on six cards: RTX 4090, 4060 Ti, 4070 Ti, RTX 3060, 5070 Ti, 5090 (1.28-1.35)",
         "the value is the middle of the six cards' ratios; none left out",
         "choice_constants.h at kLoneTile; DECISIONS 66")},
    {kChoice, "kTileCost",
     Holdout::none(
         "the census of one ~1.8K-token prompt (nsys, Qwen3.6-35B-A3B's and gpt-oss-20b's calls): "
         "the RTX 4090 without PDL, the RTX 5070 Ti on a build with pdl::supported() false",
         "one prompt per card; Ampere takes Ada's int8 row; the rows' `cards` count the cards "
         "whose "
         "census reads them (one for most): a one-card row prices only its boards' estimate",
         "DECISIONS 66")},
    {kChoice, "kMmqTileCost",
     Holdout::quoted(
         "Qwen3.5 9B's Q3_K_M, Q4_K_M (the MTP file), Q5_K_M and Q6_K 7K-token prompts (nsys, the "
         "calls by format and grid), each generation's cards together: Ada RTX 4060 Ti and 4070 "
         "Ti, Ampere RTX 3060, Turing RTX 2080 Ti, Blackwell RTX 5090",
         "a held-out score quoted from the census, whose per-call kernel times are not in the "
         "repository: Q4_K_S (a format the rows were not read on) predicted against measured "
         "kernel time per call Ada -3 .. +8 %, Ampere 0 %, Turing -2 .. +2 %, Blackwell -5 .. +7 % "
         "(in sample -8 .. +7 %); Q4_1 on Gemma 4 31B's 2013-token prefill at Q4_K's step x0.98 "
         "(RTX 4090) and x1.02 (RTX 3090)",
         "DECISIONS 69; choice_constants.h at MmqTileCost",
         "tests/test_choice.cpp prefills_by_format holds the whole-prompt estimate against "
         "tests/fixtures/choice/prefill_measured.json, Q4_K_S rows included (a different metric); "
         "Q3_K is read as Q6_K where no row of its own exists; the IQ formats have kMmqIqStep")},
    {kChoice, "kMmqIqStep",
     Holdout::none(
         "Qwen3.8 27B's UD-IQ3_XXS and UD-IQ2_XXS 2052-token prefills on the RTX 3090 (nsys, "
         "07.11) "
         "and the RTX 4090's census of the same files, each format against IQ4_XS in the same "
         "calls",
         "no card or model left out: the two files are a repeat of one read on the same card (they "
         "agree within 3 %: IQ2_XXS 37.2 / 37.0 ms a G weight, IQ3_XXS 39.2 / 38.8), not a "
         "holdout",
         "DECISIONS 69 (07.11 census); choice_constants.h at MmqIqStep",
         "the one prediction made beforehand, at Q3_K's step (49.25 +- 15 %, IQ3_XXS) and 45 "
         "(IQ2_XXS), missed: IQ3_XXS's grid-table decode is cheaper than Q3_K's masks; IQ3_S read "
         "in one file only")},
    {kChoice, "kW4a8KquantCost",
     Holdout::none(
         "the RTX 5070 Ti's census of Qwen3.5 9B's Q4_K_M, Q4_K_S, Q5_K_M and Q6_K (and Q3_K_M for "
         "the classic tile) at 7K tokens: one card, the K-4096 and K-12288 calls agree within 2 %, "
         "Q3_K's three grids' rounds within 3 %",
         "one card: an estimate only; no board left out", "choice_constants.h at kW4a8KquantCost")},
    {kChoice, "kSmallKernelCost",
     Holdout::none(
         "the census of Qwen3.5 9B's prefill, the linears, attention and tail calls apart (seven "
         "cards: scan 0.040-0.064 SM-clocks a token and state element on Ada and Blackwell, bytes "
         "26-35 a pass; RTX 3060 0.051 / 61; RTX 2080 Ti 0.196 / 88; the Quadro GV100 0.381 / 56, "
         "DECISIONS 70)",
         "one card a generation for Ampere, Turing and Volta: transferred; Ada and Blackwell's "
         "rows "
         "are means over their cards, none left out",
         "DECISIONS 66, 70")},
    {kChoice, "kVoltaPrefillCost",
     Holdout::none(
         "the Quadro GV100's census of Qwen3.5 9B's Q8_0, Q6_K and Q4_K_M at 7K tokens: tile step "
         "52.4, decodes 0.69 / 0.99 / 1.11 / 0.94 SM-clocks",
         "one card (the GEMM's calls -4 .. 0 %, the decodes -2 .. +1 %): transferred",
         "DECISIONS 70")},
    {kChoice, "kVoltaCopyMinDramBps",
     Holdout::none(
         "ninferno_linear_bench on the V100 (T = 32..240, the Q4_K and Q5_K tile against the "
         "decoded copy): the copy's narrowest margin, 1.13 (24576 x 4096 below 129 tokens), at the "
         "board's 870 GB/s; the bound is that rate over the margin",
         "one board: none left out; a Volta part under the bound (the Titan V's 653 GB/s) keeps "
         "the tile without a measurement of its own",
         "generic_gemm.cpp at kVoltaCopyMinDramGbps (the same bound, the device side)")},
    {kChoice, "kAttentionTileCost",
     Holdout::none(
         "the census of a 21K-token prompt in 21 1024-token chunks (nsys), each generation's cards "
         "together: RTX 5090 and 5070 Ti, RTX 4060 Ti and 4070 Ti, RTX 3060",
         "none left out: the fit's own residuals (gb20x -1 .. +2 %, Ada -8 .. +9 %, Ampere -0.4 .. "
         "+0.8 %); the 5070 Ti's first chunk shows the CTA's own cost",
         "DECISIONS 68")},
    {kChoice, "kFp8AttentionTileCost",
     Holdout::none(
         "the census of 7K and 21K prompts (1024-token chunks), each generation's cards together: "
         "RTX 4060 Ti and 4070 Ti, RTX 5090 and 5070 Ti, RTX 3060",
         "none left out: the fit's residuals (Ada -3 .. +3 %, Blackwell -6 .. +6 %, Ampere -2 .. "
         "+3 %)",
         "DECISIONS 68")},
    {kChoice, "kBf16AttentionTileCost",
     Holdout::quoted(
         "the census of a 7K-token prompt (1024-token chunks): RTX 5090, RTX 4060 Ti, RTX 3060",
         "a held-out score quoted from the census, whose attention times are not in the "
         "repository: the RTX 5070 Ti (not read for the row), predicted against measured attention "
         "time 0 .. +1 % (in sample: RTX 5090 -1 .. +2 %, 4060 Ti -2 .. +1 %, 3060 0 %)",
         "DECISIONS 68; choice_constants.h at kBf16AttentionTileCost")},
    {kChoice, "kAttentionMergeDramShare",
     Holdout::none("the 21K census's merge kernels where their bytes outgrow the L2: 0.85 .. 1.04 "
                   "of the DRAM "
                   "rate (RTX 4060 Ti 0.86, 4070 Ti 0.85, 3060 0.92, 5070 Ti 1.03)",
                   "the value is read off the four cards; none left out",
                   "choice_constants.h at the merge")},
    {kChoice, "kAttentionMergeL2Share",
     Holdout::none(
         "the same census where the L2 holds the bytes: 2.4 .. 4.7 of the DRAM rate (RTX 5090 2.4 "
         "-3.2 at 7K, 1.6 in the late chunks of 21K; the 4070 Ti's and 5070 Ti's first chunk "
         "4.7 / 3.2)",
         "a spread of 1.6 .. 4.7: the value 3.0 is a middle of it, none left out",
         "choice_constants.h at the merge")},
    {kChoice, "kGemmaAttentionCost",
     Holdout::none(
         "Gemma 4 12B Q4_K_M's 7K and 21K prompts (nsys, a layer's median a 1024-token chunk): RTX "
         "4060 Ti, RTX 3060, RTX 5070 Ti and 5090, RTX 2080 Ti, Quadro GV100",
         "none left out: the fit's residuals (sliding -3 .. +3 % to -16 .. +3 %, full -13 .. +9 "
         "%); "
         "Ada, Ampere, Turing and Volta rest on one card",
         "DECISIONS 68")},
    {kChoice, "kPrefillAttentionEta",
     Holdout::none(
         "fit_prefill.py's points (the same 82 as kPrefillEta); Turing's row the census of the RTX "
         "2080 Ti's prefill (0.167 / 0.143); Volta's the Quadro GV100's census (Gemma 0.097, Qwen "
         "0.139 at 2863 tokens: one share cannot hold both)",
         "fitted and checked on the same points; Turing and Volta one card each",
         "choice_constants.h at GenerationEta")},
    {kChoice, "kExactComputeEta",
     Holdout::none(
         "run_sw.sh, 9B at a 22K prompt: exact decode 61.7 against 85.3 tok/s on the RTX 2080 Ti, "
         "69.9 against 80.6 on the 4070 Ti, 127.9 against 150.2 on the 4090; Ada the mean of the "
         "4070 Ti's 0.0272 and the 4090's 0.0214",
         "two Ada cards averaged, Turing one: none left out", "choice_constants.h at RouteEta")},
    {kChoice, "kKvReadFactor",
     Holdout::none(
         "Ada and Blackwell BF16 and FP8, Blackwell K8V4: the nsys census of the decode attention "
         "kernel (#618 step 1, Qwen3.5-9B at 7.8K and 21.9K tokens, against INT8 on the same "
         "card), the mean of two cards each (RTX 4070 Ti / 4090, RTX 5070 Ti / 5090: within 2.3 "
         "%); Ada's K8V4 and NVFP4 "
         "run_sw.sh at a 22K prompt (1.01 / 1.04, 1.25 / 1.22); Turing run_sw.sh on one card",
         "none left out", "choice_constants.h at KvReadFactor")},
    {kChoice, "kKvSmRate",
     Holdout::none(
         "Turing: Qwen3.5 9B's int8 kernel and Gemma 4 12B's BF16 one on the RTX 2080 Ti; Volta: "
         "the Quadro GV100's census at 7K and 82K (Gemma 4 12B's BF16 full layer 707 us a call for "
         "153 MB); Ampere, Ada and Blackwell: the census of #618 step 1 (Qwen3.5-9B, the RTX 3090 "
         "and 3060, 4070 Ti and 4090, 5070 Ti and 5090: the mean of two cards, within 2.7 %; "
         "Blackwell's NVFP4 6.4 % apart, so a row of one card's standing)",
         "one card a Turing, Volta or Blackwell row, two an Ampere or Ada one: nothing left out",
         "choice_constants.h at KvSmRate")},
    {kChoice, "kMoeChunkShare",
     Holdout::none(
         "run_chunk.sh at a 22K prompt, 2026-10-23/24: Gemma 4 26B on the RTX 4070 Ti and 4090, "
         "Qwen3.6 35B on the 4090; the table is the mean of the three curves",
         "the three curves lie within 6 % of the mean: no curve was left out of it",
         "choice_constants.h at ChunkShare")},
};

inline constexpr FittedHoldout kGenerationHoldouts[] = {
    {kGeneration, "bf16 (the emulated rows)",
     Holdout::none("Volta's BF16 rate: the fitted aggregate 1536 (the unit route_tune's fit used, "
                   "where the BF16 MMA is emulated in FP32 FMAs); Turing's row holds the spec "
                   "rate 512 since #781",
                   "one aggregate for one generation; no board left out",
                   "core/generation.h at the rate comment")},
    {kGeneration, "w4a8_eff_scale",
     Holdout::none(
         "ninferno_route_tune timings of the RTX 4090, 5090, 4060 Ti, L4, A10 and A100 (18.10): "
         "the ga100 row takes the A100's 0.46; the hopper and gb100 rows, with no board of their "
         "own in the fit, hold none (#618 step 12)",
         "one board for the row that differs from 1.0: nothing left out, and not carried to a lent "
         "row or to Hopper / GB100 (fitted_route_scales: nullopt); Turing's 1.0 is the consumer "
         "rows' value with no "
         "Turing board in the fit (#781: W4A8 at 0.70..0.74 of the RTX 2080 Ti's measured time)",
         "core/generation.h comment (fitted constants)")},
    {kGeneration, "a16_cost_scale",
     Holdout::none(
         "the same route_tune timings: 1.11 on the consumer rows, 1.64 on the A100's, none on the "
         "hopper and gb100 rows (#618 step 12) (with the busy-SM fill it moved 0.89 -> 1.11, "
         "which also prices the K-quant and codebook dequant: 5070 Ti ISTA T 17..32 1.20 -> 1.04)",
         "the A100 is one board; the consumer value is checked on the same cards (and a 5070 Ti "
         "before / after case)",
         "core/generation.h; generic_cost.cpp price_routes comment")},
    {kGeneration, "mmq_tile_fixed",
     Holdout::none(
         "DECISIONS 74's forced-J grids: the grid regret over the widths is least at 0.15-0.25 on "
         "the RTX 3060 and 3090 (0.20 for GA10x), 0.05-0.15 on the RTX 5090 (0.10 for GB20x), Ada "
         "joined by its own grid (RTX 4070 Ti and 4090: 0.20); the default 0.43 where a row has "
         "none (the RTX 4060 Ti's 64-column Q4_K tiles: 0.716)",
         "a measured preference per generation, each over the boards that ran its grid; Turing was "
         "not run over the widths; not carried by a lent row",
         "DECISIONS 74; core/generation.h at mmq_work_tiles")},
    // The measured preferences: a yes or no per row, each set by an A/B on the row's boards.
    {kGeneration, "mmq_candidate",
     Holdout::none(
         "the full-rate rows (ga100, hopper, gb100) off, where ggml's matrix kernel's "
         "microbenchmark win and the 24K prefill A/B (-15 %) disagree; the other rows keep the "
         "kernel a candidate the cost model prices",
         "the A/B that turned it off is its only check, on the full-rate rows' one board (the "
         "A100); none left out; not carried by a lent row (mmq_candidate_measured)",
         "core/generation.h at mmq_candidate")},
    {kGeneration, "attention_chain",
     Holdout::none(
         "gb20x: RTX 5090 (31B, T = 4096, -2.2 %) and RTX 5080 (26B, T = 1024 at 16K keys, +5.5 "
         "%); Ada off: the chained tile 25-35 % slower even in whole waves (RTX 4090, RTX 4080 "
         "SUPER)",
         "the A/Bs that set the values are their only check, none left out; the other "
         "generations are unmeasured and keep the plain launch (attention_chain_measured)",
         "core/generation.h at attention_chain; ops/softmax_attention/gemma4/launch.cu at "
         "chain_pays")},
    {kGeneration, "mmq_work_tiles",
     Holdout::none(
         "DECISIONS 74's forced-J grids and A B A B runs: GA10x (RTX 3060, 3090), Ada (RTX 4070 "
         "Ti, 4090), GB20x (RTX 5090); Turing was not run over the widths and keeps the fewest "
         "tiles",
         "the grids and A/Bs that set the rows are their check; no board left out; not carried "
         "by a lent row (mmq_work_tiles_measured)",
         "DECISIONS 74; core/generation.h at mmq_work_tiles",
         "the RTX 3060's mix was predicted before it was measured (-7.8 %, measured -8.2 %, at f "
         "0.43), the RTX 4070 Ti's (-5.9 %, measured -4.4 %)")},
    {kGeneration, "mmq_launch_law",
     Holdout::none(
         "Turing's law (kTuringMmqLaw): tools/bench/mmq_launch_microbench.cu on one RTX 2080 Ti, "
         "11.11, Q4_K 4096 x 4096 and 4096 x 8192, the medians of two passes; the unit clocks and "
         "the stream-k ratio are measured, launch_seconds the mean of two nsys fixup times "
         "(5.0 and 1.6 us), a fitted summary",
         "one card, carried to the generation as a transfer (DECISIONS 78; no second Turing card "
         "is measured): nothing left out; the other rows have no law (null: the tile law)",
         "core/generation.h at kTuringMmqLaw")},
};

inline constexpr FittedHoldout kSmallTEtaHoldouts[] = {
    {kRoundCost, "kSmallTEta",
     Holdout::none(
         "registry spec-tune rounds of Qwen targets, k 4..7, RTX 4060 Ti, RTX 4070 Ti and RTX 4090",
         "the shipped 0.160 is fit_eta.py's fit over every card of today's registry (#689), so no "
         "card is held out of it; item 3 holds each card out of a refit, R measured / R facts: RTX "
         "4090 +1.2 % (rms 2.8 %), RTX 4070 Ti +2.0 % (4.6 %), RTX 4060 Ti -3.9 % (5.6 %), a "
         "hold-out of the procedure, not of the shipped row",
         "tools/fits/round_cost/fit_eta.py; tests/test_round_cost_eta_fit.py",
         "gpt-oss rounds ran +16..19 % over their facts in older runs (DECISIONS 53)"),
     "ada"},
    {kRoundCost, "kSmallTEta",
     Holdout::none(
         "registry spec-tune rounds of Qwen targets, k 4..7, RTX 5070 Ti and RTX 5090",
         "the shipped 0.160 is fit_eta.py's fit over both cards of today's registry (#689), so no "
         "card is held out of it; item 3 holds each card out of a refit, R measured / R facts: RTX "
         "5090 +3.8 % (rms 7.1 %), RTX 5070 Ti +3.2 % (8.3 %), a hold-out of the procedure, not of "
         "the shipped row",
         "tools/fits/round_cost/fit_eta.py; tests/test_round_cost_eta_fit.py"),
     "gb20x"},
};

inline constexpr FittedHoldout kSpecHoldouts[] = {
    {kCostConstants, "kLaunchNode", h_launch_node},
    {kRoundCost, "kLaunchTime",
     Holdout::none(
         "the Ada census: Qwen3.5-9B Q4_K_M plain decode on the RTX 4090, 4070 Ti and 4060 Ti, "
         "every kernel within 2 %: eta_dram 0.946 (the census's t0 is kLaunchNode's record)",
         "the three cards are the fit; no generation without a row is priced by it",
         "spec/round_cost.h at LaunchTime")},
    {kMoeShare, "kMoeRoutedShare",
     Holdout::none(
         "the drafter registry's speculation floors (decisions.md 53: the round cost's B(r)): rows "
         "routed apart read 0.30 of the experts independent routing would add",
         "fitted and checked on the same floors; no pair was left out",
         "docs/maintainer/decisions.md 53; spec/moe_routed_share.h")},
};

inline constexpr FittedHoldout kWaveHoldouts[] = {
    {kWave, "grid_passes_wave",
     Holdout::quoted(
         "the RTX 5090 (170 SMs): Gemma 4's NVFP4 2048x5376 A4 prefill, where the TMA route's grid "
         "(16 row tiles) stays under one wave and loses to the 128-token MMA tile until T > 1280",
         "a held-out check quoted from the comment, whose timings are not in the repository: the "
         "RTX 5070 Ti (70 SMs, not fitted on), TMA ahead from T = 641, 28 % faster there, equal "
         "from 768; other boards are extrapolated until measured",
         "ops/common/sm_wave_rules.h comment",
         "ninferno_sm_wave_rules_test holds the rule's arithmetic, not the timings")},
};

// --- route_rules.h: the rows tagged Basis::Fitted or Basis::ScaledFromOneBoard, keyed by the
// row's name. The boards are the row's (kSpeedFitBoards: the speed profile's seven-board fit).
// The speed rule's held-out score (one rule for every board, fitted on six boards and scored on the
// seventh), which the rows of kRmmaMidMinTokens and kSpeedFill state: quoted, not reproducible.
inline constexpr Holdout h_speed_rule = Holdout::quoted(
    "BF16 timings of the speed profile's seven boards (RTX 3090, RTX 4090, RTX 5090, RTX 2080 Ti, "
    "A100, A10, L4)",
    "a held-out score quoted from the original fit, whose timings are not in the repository: each "
    "board in turn left out of the fit and scored, geomean time over the best route 1.116 of the "
    "best against 1.219 for the bands it replaced; a rule per board gained 1.1 % at most",
    "route_rules.h at kRmmaMidMinTokens; generic_gemm.h; generic_gemm.cpp");
// The other edges of the same rule: the comment fits the rule as a whole and breaks out no error
// for them.
inline constexpr Holdout h_speed_rule_edge = Holdout::none(
    "inside the speed rule's seven-board fit (RTX 3090, RTX 4090, RTX 5090, RTX 2080 Ti, A100, "
    "A10, "
    "L4)",
    "the record scores the rule as a whole (a quoted 1.116 of the best, board held out: "
    "kRmmaMidMinTokens's record) and does not break out this edge: no held-out error of its own",
    "route_rules.h at the row");
inline constexpr Holdout h_volta_copy_bound = Holdout::none(
    "the V100 (870 GB/s): the decoded copy's narrowest margin, 1.13 (24576 x 4096 below 129 "
    "tokens), scaled to other boards by the DRAM rate (the comment's model, not a timing)",
    "one board measured, no other: the Titan V (653 GB/s) is under the bound without a measurement",
    "route_rules.h at kVoltaCopyMinDramGbps; generic_gemm.cpp",
    "the same bound as choice_constants.h kVoltaCopyMinDramBps (the estimate's copy)");

inline constexpr FittedHoldout kRouteRuleHoldouts[] = {
    {kRouteRules, "kVoltaCopyMinDramGbps", h_volta_copy_bound},
    {kRouteRules, "volta_copy_pays: the two comparisons", h_volta_copy_bound},
    {kRouteRules, "kRmmaMidMinTokens", h_speed_rule},
    {kRouteRules, "kSpeedFill",
     Holdout::quoted(
         "BF16 timings of the speed profile's seven boards (RTX 3090, RTX 4090, RTX 5090, RTX 2080 "
         "Ti, A100, A10, L4)",
         "the rule's held-out 1.116 of the best is quoted from the original fit, whose timings are "
         "not in the repository; the fill's own figure (1.0..2.0 cost under 0.5 %) is an in-sample "
         "sensitivity, not a held-out error",
         "route_rules.h at kSpeedFill; generic_gemm.h")},
    {kRouteRules, "heuristic_rmma: speed rule token tiles", h_speed_rule_edge},
    {kRouteRules, "heuristic_rmma: sixteen warps from 72 K steps", h_speed_rule_edge},
    {kRouteRules, "heuristic_rmma: Q4_K tall tile", h_speed_rule_edge},
    {kRouteRules, "heuristic_rmma: token tile ladder", h_speed_rule_edge},
    {kRouteRules, "heuristic_rmma: stream regime tiles", h_speed_rule_edge},
    {kRouteRules, "tiled_speed_pays: tile",
     Holdout::none(
         "kSpeedFill's fit (the speed profile's seven boards); the one timing of the tile "
         "is below the fill (Q4_K n = 1024 at T 64..128)",
         "no timing places the fill for this tile; the boards are not named: none left "
         "out",
         "route_rules.h at tiled_speed_pays")},
    {kRouteRules, "staged_speed_route: wave thresholds",
     Holdout::none("BF16 timings on seven boards with the register-MMA rule (the comment does not "
                   "name them): 1.116 of the best against 1.15..1.36 and 1.33..1.64 for the older "
                   "gates",
                   "the record does not say a board was held out for these thresholds",
                   "route_rules.h at staged_speed_route")},
};

// --- Kernels' own launch constants, fitted on lab boards.
inline constexpr FittedHoldout kKernelHoldouts[] = {
    {kInt8PromptCost, "kInt8PromptHalfCtaPercent",
     Holdout::none(
         "one RTX 5090 timing at 64K context: a four-warp CTA, or an eight-warp CTA of 64 "
         "rows, sweeps in about 0.72 of a full eight-warp CTA's time",
         "one card: none left out; every board takes the figure", "int8/prompt_fast_cost.h",
         "the one definition: the kernel's layout (template_launch.cuh) and the choice's estimate "
         "(choice.cpp, choice_constants.h kAttentionHalfCta) read it")},
    {kRope, "kLargeBlockMaxTokens",
     Holdout::none("the RTX 5090 (its wave, 170 SMs x six CTAs: 1020) and the RTX 5080 (the "
                   "256-thread block equal or faster through 1020 tokens)",
                   "fitted on both boards: none left out; every board takes the bound",
                   "rope.cu at kLargeBlockMaxTokens")},
    {kRmsNorm, "kRmsPrefetchBlocks",
     Holdout::none("grid-size sweeps of both gated shapes: the crossing between 176 and 192 blocks "
                   "on the RTX 5090; the prefetch 1.5-4.8 % faster from 88 through 168 blocks on "
                   "the RTX 5080",
                   "fitted on both boards: none left out; every board takes the count",
                   "rmsnorm.cu at kRmsPrefetchBlocks")},
    {kMoePrefill, "kPrefillMaxBlocks",
     Holdout::none("the RTX 5090 (32 CTAs an SM: 5440) and the RTX 5080 (32 an SM, 2688, 1.2-1.5 % "
                   "slower on the Q4/Q5 routed prefill at T = 1024-4096, equal on NVFP4)",
                   "fitted on both boards: none left out; every board takes the budget",
                   "sparse_moe_prefill_kernels.cu at kPrefillMaxBlocks")},
    {kGdnGating, "k27Routes",
     Holdout::none("the RTX 5090: the band edges (token counts) hold the grid near 192 CTAs",
                   "one board: none left out; every board takes the edges",
                   "bf16_gdn_gating_proj_plan.cpp at k27Routes")},
    {kGdnGating, "k35Routes",
     Holdout::none("the RTX 5090: the same token-count edges hold the grid near 256 CTAs",
                   "one board: none left out; every board takes the edges",
                   "bf16_gdn_gating_proj_plan.cpp at k35Routes")},
    {kRmsNormRope, "kTextHeadsPerBlock",
     Holdout::none("one sm_120a board: the optimum at three heads a block, the plateau flat from "
                   "two to nine",
                   "one board: none left out; every board takes the value",
                   "rmsnorm_rope/launch.h at kTextHeadsPerBlock")},
    {kGemma4Attention, "kDecodeMaxWarpRowsD128",
     Holdout::none("Gemma 4 drafter decode at T = 16, 32K keys: four warp rows 2-5 % faster than "
                   "two on the 64-head drafter, equal on the 32-head one, eight no faster",
                   "the comment names no board: none left out; every board takes the count",
                   "gemma4/launch.cu at kDecodeMaxWarpRowsD128")},
    {kGemma4Attention, "kDecodeMinSplitKeysD128",
     Holdout::none("the RTX 5090: the 64-key floor of a D <= 128 decode split",
                   "one board: none left out; every board takes the floor",
                   "gemma4/launch.cu at kDecodeMinSplitKeysD128")},
    {kGemma4Attention, "kDecodeSplitsPerGroupD128",
     Holdout::none("gpt-oss-20b plain decode at 69K keys: RTX 5090 177.6 -> 221.6 tok/s against "
                   "16 splits; +8 % long on the RTX 4090 and 5080",
                   "fitted on the three boards: none left out; every board takes the count",
                   "gemma4/launch.cu at the D<=128 splits comment")},
    // Two kBlock definitions stand in rmsnorm.cu (128 and 512): the key names this one's value.
    {kRmsNorm, "kBlock = 128",
     Holdout::none("the plain per-head D128 path's four rows a CTA (128 threads): the measured "
                   "lower envelope from T = 1 through the 1024-token prefill point",
                   "the comment names no board: none left out; every board takes the block",
                   "rmsnorm.cu at the D128 plain path",
                   "the file's other kBlock (512) is not "
                   "recorded")},
    {kQ5LinearAdd, "kNarrowTailCols",
     Holdout::none("the measured narrow schedules of a short remainder past whole 512-token waves "
                   "of the collective MMA route",
                   "the comment names no board or margin: none left out",
                   "q5_linear_add_plan.cpp at kNarrowTailCols")},
    {kGdnConv, "kNarrowThreads",
     Holdout::none("the compiled Qwen3.6 35B-A3B instance it replaced: at T = 4, 160 blocks of 64 "
                   "threads faster than 40 of 256 on 170 SMs (the RTX 5090)",
                   "one board and one width: the rule (wide_blocks < SMs -> 64 threads) extends "
                   "it to every width and geometry unmeasured",
                   "gdn_projected_conv.cu at launch")},
    // The fused text qk-norm + RoPE's default bound, kept while the law's excess is uncalibrated
    // (route_cost::kPendingFits fused_text_excess_cycles, #617 G).
    {kRmsNormRopeBound, "kFusedTextDefaultMaximumTokens",
     Holdout::none("the RTX 5090: the fused form wins by 22 to 52 % through 256 tokens and loses "
                   "by up to 22 % at 1024; the bound a doubling below the crossover",
                   "one board: none left out; every board takes the bound",
                   "rmsnorm_rope/rmsnorm_rope.cpp at kFusedTextDefaultMaximumTokens")},
    {kAttnInputPlan, "kBf16AttnInputSmallTMinTokens",
     Holdout::none("the attention epilogue's own measured crossover (with "
                   "kBf16AttnInputSmallTMaxTokens)",
                   "the comment names no board or margin: none left out",
                   "bf16_attn_input_plan.h")},
    {kAttnInputPlan, "kBf16AttnInputSmallTMaxTokens",
     Holdout::none("the attention epilogue's own measured crossover (with "
                   "kBf16AttnInputSmallTMinTokens)",
                   "the comment names no board or margin: none left out",
                   "bf16_attn_input_plan.h")},
    {kGdnGating, "kFusedSimt27SpeedMaxCols",
     Holdout::none("the fused 27B norm + gating SIMT kernel's measured speed frontier (42 columns)",
                   "the comment names no board or margin: none left out",
                   "bf16_gdn_gating_proj_plan.cpp at kFusedSimt27SpeedMaxCols")},
    {kW4a8, "kW4a8Scales4",
     Holdout::none("the W4A8 micro-bench per tile, IQ3_XXS / IQ3_S / Q4_K / Q6_K (+ IQ4_XS, Q4_0, "
                   "Q8) at Qwen3.6-27B and Muse 30B shapes: RTX 5080 and RTX 4090",
                   "the two boards are the A/B: none left out",
                   "generic_w4a8.cuh at the classic kernel's schedule choices",
                   "covers kW4a8DecodeMid, defined on the same line")},
    {kSparseMoePrefill, "kSparseMoePrefillNvfp4Min",
     Holdout::none("a cold sweep of T over [2, 46] through the operator benchmark: small-T ahead "
                   "through T = 8, at most 4.5 % behind to T = 12, the prefill route +9 % at T = "
                   "16",
                   "the comment names no board: none left out",
                   "sparse_moe_prefill.h at kSparseMoePrefillNvfp4Min")},
    {kMoePrefill, "kGateUpDeepJobsNum",
     Holdout::none("server prefill swept over the threshold 1.5, 1.5625, 1.625, 1.75 and 2.0: "
                   "fastest at 7/4 (0.10 to 0.17 points against 2/1 below the wide-plan bound)",
                   "the comment names no board: none left out",
                   "sparse_moe_prefill_kernels.cu at kGateUpDeepJobsNum",
                   "covers kGateUpDeepJobsDen (the ratio's denominator)")},
    {kQwenDecode, "kRideColumns",
     Holdout::none("measured verify costs: Qwen3.8 27B T = 16 at 1.25x a decode step, Qwen3.6 "
                   "35B-A3B T = 4 at 1.46x",
                   "the budget follows from those two costs; riding against latching plain is "
                   "not measured: none left out",
                   "qwen3_5/program/decode.cpp at kRideColumns")},
    {kGemma4Kernel, "kGemma4ChainMinTiles",
     Holdout::none("the RTX 5090, Gemma 4 26B / 31B at T = 1024..2048: 16-tile spans lost 13-16 % "
                   "at base 0 (a span's prologue costs about two tiles)",
                   "one board: none left out; every board takes the floor",
                   "gemma4/kernel.cuh at kGemma4ChainMinTiles")},
    {kBf16Vector, "kBf16x8CacheSizedMaxElements",
     Holdout::none("the RTX 5090: 16-byte packs win while a BF16 activation stays cache-sized; "
                   "AddBias and GELU take the BF16x2 streams above it",
                   "one board: none left out; every board takes the boundary",
                   "ops/common/bf16_vector.cuh at kBf16x8CacheSizedMaxElements")},
    {kQ8LinearAdd, "kK4096DecodeMaxT",
     Holdout::none("Qwen3.6-35B decode: the GEMV at T = 8 (DFlash depth 7, --spec auto C=1) cost "
                   "5 % on an RTX 4090 and 1 % on an RTX 5090, while T <= 4 keeps plain C=4 "
                   "decode's gain",
                   "fitted on the two boards: none left out; every board takes the width",
                   "q8_linear_add_plan.cpp at kK4096DecodeMaxT")},
    {kCapturePlanner, "kNominalTargetNs",
     Holdout::none("one measurement on the RTX 5090 (Qwen3.6-35B C=8): up to 4096 targets in "
                   "39-43 ms a capture, about 10 us a target; the nominal cost is its mean",
                   "one board, one measurement: none left out; it awaits calibration",
                   "context_cache/capture_planner.h at kNominalTargetNs")},
    {kMoeInternal, "kR8NarrowTokens",
     Holdout::none("moe_bench (H 2048, F 512, E 128, k 8), R8 against the register MMA at T = "
                   "5..8: RTX 5090 / 5070 Ti -5..-15 %, RTX 4090 -3..-7 %, RTX 4060 Ti -0.4..-6 % "
                   "(so kR8Tokens on sm_89 and sm_12x); RTX 2080 Ti Q6_K +38..+45 %, Q3_K "
                   "+15..+19 % (so four)",
                   "the boards that ran the A/B are its only check: none left out; Ampere (RTX "
                   "3090, 3060, 13.11) keeps four, its two cards disagreeing above four tokens; "
                   "the A100, Hopper and datacenter Blackwell are unmeasured and keep four "
                   "(ops/moe/moe_regime_rules.h)",
                   "ops/moe/moe_internal.h at kR8NarrowTokens")},
    {kQ4LinearSwiglu, "kNarrowTailCols",
     Holdout::none("a remainder of at most 40 columns measured cheaper on the narrow routes than "
                   "a second, almost empty 128-wide tile",
                   "the comment names no board or margin: none left out",
                   "q4_linear_swiglu_gemm_mma.cu at the C128 tail launch")},

    // --- The plans' measured thresholds and per-chip band tables (#718). A band table is the
    // fit of the chip it is keyed to (kMeasured in its file): that chip is its only check.
    {kDynamicConvPrepare, "kR16MaxTokens",
     Holdout::none("complete cold-cache Graph timings: split-8 R16 favoured through 48 columns",
                   "the comment names no board or margin: none left out",
                   "bf16_dynamic_grouped_conv_prepare_plan.cpp at kR16MaxTokens")},
    {kDynamicConvPrepare, "kR32NarrowMaxTokens",
     Holdout::none("the same cold-cache Graph timings: R32's 32-column tile through 96 columns, "
                   "its 64-column tile above",
                   "the comment names no board or margin: none left out",
                   "bf16_dynamic_grouped_conv_prepare_plan.cpp at kR32NarrowMaxTokens")},

    // Q4/Q5 GDN input (Qwen 27B shapes): the sliced-K small-T bands, cold against the column route.
    {kQ4Q5GdnInput, "kGb202Bands",
     Holdout::none("the RTX 5090: R16 W4 S2 from T = 5 (x1.39 / 1.68 / 1.30 at T = 8 / 13 / 24); "
                   "swept again after the K permutation, R16 W8 at 9..16 and R32 W4 S2 at 24 "
                   "and 32; MTP3 at 24K 170.0 -> 178.7 tok/s",
                   "one board: none left out", "q4_q5_gdn_input_plan.cpp at the sliced-K comment")},
    {kQ4Q5GdnInput, "kAd103Bands",
     Holdout::none("the RTX 4070 Ti SUPER (AD103): R16 W8 from T = 5 (x1.11 / 1.25 at 5 / 8), "
                   "R32 W8 at 28..32 (x1.47 / 1.51)",
                   "one board, keyed to the RTX 4080 SUPER (the same AD103): none left out",
                   "q4_q5_gdn_input_plan.cpp at the sliced-K comment")},
    {kQ4Q5GdnInput, "kAd102Bands",
     Holdout::none("the RTX 4090: R16 W8 from T = 9 (T = 8 loses, x0.94); faster cold from T = 5 "
                   "after the K permutation but not in the server, so kept; MTP3 at 24K 107.6 -> "
                   "113.6 tok/s with the verify widths 2..4",
                   "one board: none left out", "q4_q5_gdn_input_plan.cpp at the sliced-K comment")},
    {kQ4Q5GdnInput, "kGa102Bands",
     Holdout::none("the RTX 3090: R16 W8 from T = 2 after the K permutation (87 / 90 / 94 us at "
                   "T = 2 / 5 / 8 against 104 / 164 / 159), the column route from T = 25",
                   "one board: none left out", "q4_q5_gdn_input_plan.cpp at the sliced-K comment")},
    {kQ4Q5GdnInput, "kTu102Bands",
     Holdout::none("the RTX 2080 Ti: R32 W8 for T = 9..16 (x1.69 at 13), R16 W4 S2 to 24 (x1.22), "
                   "then the column route",
                   "one board: none left out", "q4_q5_gdn_input_plan.cpp at the sliced-K comment")},
    {kQ4Q5GdnInput, "kFallbackBands",
     Holdout::none("the bands every measured native-BF16 board gains on (R16 W8 for T = 9..24): "
                   "the RTX 5090, 4090, 4070 Ti SUPER and 3090, not the RTX 2080 Ti (emulated "
                   "BF16 MMA, bands of its own)",
                   "a card of an unmeasured architecture takes them unmeasured: none left out",
                   "q4_q5_gdn_input_plan.cpp at the sliced-K comment")},
    {kQ4Q5GdnInput, "kMmaGb202Bands",
     Holdout::none("ninferno_gdn_input_proj_bench, 27B form, warm graph, on the RTX 5090: the "
                   "sliced-K bands lead the small-T MMA at every width (1.04..1.28x)",
                   "one board: none left out", "q4_q5_gdn_input_plan.cpp at kMmaGb202Bands")},
    {kQ4Q5GdnInput, "kMmaAd102Bands",
     Holdout::none("the same bench on the RTX 4090: small-T MMA leads at T = 2..8 (twice the "
                   "grouped route at 5..8) and from 16, the sliced-K band at 10..12",
                   "one board: none left out", "q4_q5_gdn_input_plan.cpp at kMmaGb202Bands")},
    {kQ4Q5GdnInput, "kMmaGa102Bands",
     Holdout::none("the same bench on the RTX 3090: small-T MMA leads at every width "
                   "(1.14..1.85x)",
                   "one board: none left out", "q4_q5_gdn_input_plan.cpp at kMmaGb202Bands")},
    {kQ4Q5GdnInput, "kMmaFallbackBands",
     Holdout::none("the RTX 4090 and RTX 3090, where small-T MMA leads (the RTX 5090 alone "
                   "prefers the sliced-K bands)",
                   "an unmeasured board takes small-T MMA unmeasured: none left out",
                   "q4_q5_gdn_input_plan.cpp at kMmaGb202Bands")},
    {kQ4Q5GdnInput, "kT1MmaGa102Bands",
     Holdout::none("the RTX 3090 decode census at 19.5K: the fused-convolution GEMVs 59.1 + 24.6 "
                   "us a layer against small-T MMA 55.0 + 17.1 and a 2.6 us convolution",
                   "one board: none left out", "q4_q5_gdn_input_plan.cpp at kT1MmaGa102Bands")},
    {kQ4Q5GdnInput, "kT1MmaAd103Bands",
     Holdout::none("the RTX 4080 SUPER at T = 1: 81.1 against 85.0 us", "one board: none left out",
                   "q4_q5_gdn_input_plan.cpp at kT1MmaAd103Bands")},

    // Q4/Q5 attention input (Qwen 27B shapes): the small-T bands (variants 1..4 sliced-K, 5 the
    // small-T MMA), cold against the column route, and the retuned column routes past them.
    {kQ4Q5AttnInput, "kGb202Bands",
     Holdout::none("the RTX 5090: R16 W4 S2 from T = 5 (x1.30 / 1.75 / 1.31 at T = 8 / 13 / 24); "
                   "after the K permutation from T = 2 (33 against 41..45 us at T = 2..4) and "
                   "R32 W4 S2 at 25..32 (64..66 against 84..86)",
                   "one board: none left out",
                   "q4_q5_attn_input_plan.cpp at the sliced-K comment")},
    {kQ4Q5AttnInput, "kGb203Bands",
     Holdout::none("the RTX 5080: variant 4 53 us at T = 2..8 (the column route 61..78), variant 5 "
                   "59..63 us at 10..16, R16 W4 S2 (variant 2) at 17..24",
                   "one board: none left out",
                   "q4_q5_attn_input_plan.cpp at kGb203Bands and the sliced-K comment",
                   "the table differs from its comment past 24: it runs R32 W8 (variant 4) at "
                   "T = 25..32, where the comment has R16 W4 S2 to 28 (x1.26) and R32 W8 at 32 "
                   "(x1.32); the rows at 25..28 rest on no stated timing")},
    {kQ4Q5AttnInput, "kAd103Bands",
     Holdout::none("the RTX 4080 SUPER: variant 5 69..77 us at T = 2..16 against 75..82 (variant "
                   "1); the RTX 4070 Ti SUPER (AD103): R16 W8 from T = 5 (1.20 / 1.41 / 2.36 / "
                   "1.94 at 5 / 12 / 13 / 24)",
                   "the two AD103 boards are its only check: none left out",
                   "q4_q5_attn_input_plan.cpp at kGb203Bands and the sliced-K comment")},
    {kQ4Q5AttnInput, "kAd102Bands",
     Holdout::none("the RTX 4090: variant 5 53..57 us at T = 2..16 against 58..77 (variant 1) and "
                   "67..74 (variant 4), variant 4 from 20; the DFlash2 round census at 19.5K",
                   "one board: none left out",
                   "q4_q5_attn_input_plan.cpp at the sliced-K comment")},
    {kQ4Q5AttnInput, "kGa102Bands",
     Holdout::none("the RTX 3090: variant 5, a DFlash2 round census at 19.5K, 34.1 + 26.3 us at "
                   "T = 6 against 49.3 + 34.4 for R16 W8",
                   "one board: none left out",
                   "q4_q5_attn_input_plan.cpp at the sliced-K comment")},
    {kQ4Q5AttnInput, "kGa100Bands",
     Holdout::none("the A100 (cold, graph): variant 5 at T = 2..16 (57..88 us against 76..95 for "
                   "variant 1) and 28..32, variant 1 at 20..24 (114 / 120 against 122 / 122)",
                   "one board: none left out", "q4_q5_attn_input_plan.cpp at kGa100Bands")},
    {kQ4Q5AttnInput, "kTu102Bands",
     Holdout::none("the RTX 2080 Ti: R16 W4 S2 for T = 9..24 (1.59 / 1.20)",
                   "one board: none left out",
                   "q4_q5_attn_input_plan.cpp at the sliced-K comment")},
    {kQ4Q5AttnInput, "kFallbackMmaBands",
     Holdout::none("the pattern the A100, RTX 3090, 4060 Ti and 4090 share on the native BF16 MMA "
                   "(variant 5 1.3..2.5x the column route at T = 2..8 on each)",
                   "a chip with no measured row or sibling takes it unmeasured: none left out",
                   "q4_q5_attn_input_plan.cpp at kFallbackMmaBands")},
    {kQ4Q5AttnInput, "kFallbackBands",
     Holdout::none("where the MMA is emulated (Volta, Turing): the column route and variant 1 "
                   "(R16 W8 for T = 9..20), the GDN input plan's fallback narrowed",
                   "the comment names no board for the narrowing: none left out",
                   "q4_q5_attn_input_plan.cpp at kFallbackMmaBands")},
    {kQ4Q5AttnInput, "kPairR32C64S3MaxCols",
     Holdout::none("timings of the commit that retuned the variable-width attention projections "
                   "(the paired R32 C64 S3 route through 104 columns)",
                   "the timings and boards are not in the repository: none left out",
                   "q4_q5_attn_input_plan.cpp at kPairR32C64S3MaxCols")},
    {kQ4Q5AttnInput, "kMixedR64C128ResumeCols",
     Holdout::none("the same retune: the paired R32 C64 S4 route from 129 columns to 192, the "
                   "R64 C128 tile again from 193",
                   "the timings and boards are not in the repository: none left out",
                   "q4_q5_attn_input_plan.cpp at kPairR32C64S3MaxCols")},

    // Q5 LinearAdd: the small-T MMA bands, bench/ops/q5_linear_add_bench (cold L2, graph) against
    // the column routes.
    {kQ5LinearAdd, "kSmallTGa102Bands",
     Holdout::none("the RTX 3090: small-T MMA at every width 2..32",
                   "one board; the comment gives no figures for it: none left out",
                   "q5_linear_add_plan.cpp at the small-T MMA comment")},
    {kQ5LinearAdd, "kSmallTGa100Bands",
     Holdout::none("the A100 (cold, graph): 1.03 against 1.00 at T = 2..4, 0.71..0.94 at 5..24, "
                   "1.01..1.04 at 28..32; the RTX 3090's bands lost 27B MTP3 C4 4 % in the server",
                   "one board: none left out", "q5_linear_add_plan.cpp at kSmallTGa100Bands")},
    {kQ5LinearAdd, "kSmallTAd102Bands",
     Holdout::none("the RTX 4090 (27B down/out): 67.6 against 74.8..90.1 us at T = 2..8 (K = "
                   "17408), equal at 10..16, behind the sliced-K from 20",
                   "one board: none left out", "q5_linear_add_plan.cpp at kSmallTGa100Bands")},
    {kQ5LinearAdd, "kSmallTGb202Bands",
     Holdout::none("the RTX 5090: equal to T = 16, behind from T = 20 (34.8 against 26.6 us at "
                   "K = 6144, 88 against 61 at K = 17408): off",
                   "one board: none left out", "q5_linear_add_plan.cpp at kSmallTGa100Bands")},
    {kQ5LinearAdd, "kSmallTGb203Bands",
     Holdout::none("the RTX 5080: 71.7 against 88..90 us at T = 5..8 (K = 17408), 106 against "
                   "119..127 at 20..24, equal at 28..32",
                   "one board: none left out", "q5_linear_add_plan.cpp at kSmallTGa100Bands")},
    {kQ5LinearAdd, "kSmallTAd103Bands",
     Holdout::none("the RTX 4080 SUPER: 91 against 135..139 us at T = 5..8 and 139 against "
                   "173..178 at 20..24 (K = 17408); behind the row split at T = 2..4",
                   "one board: none left out", "q5_linear_add_plan.cpp at kSmallTGa100Bands")},

    // The BF16 and INT8 causal attention plans.
    {kBf16CausalPlan, "kTiled128MinWidth",
     Holdout::none("the 128-row prefill tiles, measured only for Qwen3.5 group 8 (16 query, 2 KV "
                   "heads)",
                   "the comment names no board or timing: none left out; other geometries keep "
                   "the 64-row tiles",
                   "causal_cache/bf16/plan.cpp at kTiled128MinWidth")},
    {kBf16CausalPlan, "kTiled128MinPastKeys",
     Holdout::none("the same measurement of the 128-row tiles (Qwen3.5 group 8)",
                   "the comment names no board or timing: none left out",
                   "causal_cache/bf16/plan.cpp at kTiled128MinWidth")},
    {kBf16CausalPlan, "kLongCtaSmDivisor",
     Holdout::none("upstream's tuning on the RTX 5090: 85 CTAs per tile on 170 SMs",
                   "one board: none left out; every board takes half its SMs",
                   "causal_cache/bf16/plan.cpp at kLongCtaSmDivisor")},
    {kBf16CausalPlan, "kSmallQueryMaxWaves",
     Holdout::none("the plan's cap on small-query partitions: up to six waves",
                   "the comment names no board or timing for the cap: none left out",
                   "causal_cache/bf16/plan.cpp at kLongCtaSmDivisor")},
    {kInt8CausalPlan, "kTwoCtasPerSmMaxWidth",
     Holdout::none("the wave rules, measured on four and two KV heads",
                   "the comment names no board or timing: none left out; other geometries take "
                   "the rules of their KV head count",
                   "causal_cache/int8/plan.cpp at kTwoCtasPerSmMaxWidth")},

    // NVFP4 and FP8 activation routes.
    {kNvfp4AttnInput, "kNvfp4AttnInputFirstA4Tokens",
     Holdout::none("the A16 / A4 selection of the commit that chose the measured NVFP4 activation "
                   "routes",
                   "its timings and boards are not in the repository: none left out",
                   "nvfp4_attn_input_plan.cpp at kNvfp4AttnInputFirstA4Tokens")},
    {kNvfp4LinearAdd, "kNvfp4AddVerifyExactMaxTokens",
     Holdout::none("the RTX 5090, Qwen3.8-27B NVFP4 (nsys census, MTP3 at 24K; serve MTP3 C1 +5 "
                   "%, DFlash 5 C1 +6 %); at T = 16 the exact route lost 5 %",
                   "one board: none left out; every board takes the band",
                   "nvfp4_linear_add_plan.cpp at nvfp4_verify_exact")},
    {kNvfp4LinearAdd, "kNvfp4AddFirstA4TokensK6144",
     Holdout::none("the commit that chose the measured NVFP4 activation routes (0ba16ed) set it "
                   "at 7; the edge moved to 17 in 54ef520 with no stated timing",
                   "no timing or board in the repository: none left out",
                   "nvfp4_linear_add_plan.cpp at kNvfp4AddFirstA4Tokens")},
    {kNvfp4LinearAdd, "kNvfp4AddFirstA4Tokens",
     Holdout::none("the A16 / A4 selection of the commit that chose the measured NVFP4 activation "
                   "routes (K = 17408)",
                   "its timings and boards are not in the repository: none left out",
                   "nvfp4_linear_add_plan.cpp at kNvfp4AddFirstA4Tokens")},
    {kNvfp4LinearSwiglu, "kNvfp4SwiGluVerifyExactMaxTokens",
     Holdout::none("the RTX 5090, Qwen3.8-27B NVFP4: SwiGLU 34816x5120 at T = 4 64.8 us against "
                   "the small-T fused 69.2; at T = 16 the exact route lost 5 %",
                   "one board: none left out; every board takes the band",
                   "nvfp4_linear_swiglu_plan.cpp at nvfp4_verify_exact")},
    {kNvfp4LinearSwiglu, "kNvfp4SwiGluFusedA16MaxTokens",
     Holdout::none("the fused A16 routes kept through T = 4 under AllowA4, A4 above",
                   "the comment names no board or timing: none left out",
                   "nvfp4_linear_swiglu_plan.cpp at kNvfp4SwiGluFusedA16MaxTokens")},
    {kFp8LinearAdd, "kFp8A8FirstTokensK6144",
     Holdout::none("the measured A16 / A8 speed frontier at K = 6144 (T = 17)",
                   "the comment names no board or margin: none left out",
                   "fp8_linear_add_plan.cpp at resolve_route")},
    {kFp8LinearAdd, "kFp8A8FirstTokensK17408",
     Holdout::none("the measured A16 / A8 speed frontier at K = 17408 (T = 20)",
                   "the comment names no board or margin: none left out",
                   "fp8_linear_add_plan.cpp at resolve_route")},

    {kSparseMoeSmallT, "kMtpPaths1MaxTokens",
     Holdout::none("the commit that tuned sparse MoE verification through T = 16 (the Q8 MTP "
                   "profile's D3 schedule)",
                   "no figures for this edge in the repository: none left out",
                   "sparse_moe_small_t_plan.cpp at kMtpPaths1MaxTokens")},
    {kQ8LinearAdd, "kK4096DecodeCtasPerTwoSms",
     Holdout::none("Qwen3.6-35B's K = 4096 decode GEMV (2048 rows): r8 the fastest of r4 / r8 / "
                   "r16 on the RTX 5090 (256 CTAs on 170 SMs), r8 on the RTX 4090, r16 on the RTX "
                   "3090 (128 CTAs on 82 SMs)",
                   "the three boards are its fit: none left out; a new card takes the rule by its "
                   "SM count",
                   "q8_linear_add_plan.cpp at k4096_decode_rows",
                   "the RTX 5090 sits at the edge: 2 x 256 CTAs = 512 against 3 x 170 SMs = 510")},
};

} // namespace ninferno::fitted
