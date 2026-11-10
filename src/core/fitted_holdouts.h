#pragma once

// The holdout records (core/holdout.h) of every fitted constant, in one place: the route cost model
// (ops/generic/route_cost_constants.h), the choice's speed model (product/choice/
// choice_constants.h), the generation row's fitted scales (core/generation.h), the speculation
// round cost (spec/round_cost.h, spec/moe_routed_share.h) and the one fitted SM-wave rule
// (ops/common/sm_wave_rules.h). A record per constant or per table of one fit: the table's rows
// are one fit over the same data, so one record says what it was fitted on and what, if anything,
// was left out; the per-row card counts stay in the tables (`cards`).
//
// The records are a side table keyed by the name of the definition in its file, never a field of
// the constant: values are not touched here, the records say where each value comes from, so that
// a fitted constant without a held-out check is declared so, not forgotten.
// tests/test_fitted_holdouts.cpp matches every key to its definition in the file (not a substring),
// requires every Fitted-marked definition of the covered files to have a record or to sit on its
// not-fitted list, fails on an Unset or incomplete record, and pins one total count.
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

inline constexpr std::string_view kChoice        = "src/product/choice/choice_constants.h";
inline constexpr std::string_view kGeneration    = "src/core/generation.h";
inline constexpr std::string_view kRoundCost     = "src/spec/round_cost.h";
inline constexpr std::string_view kCostConstants = "src/core/cost_constants.h";
inline constexpr std::string_view kMoeShare      = "src/spec/moe_routed_share.h";
inline constexpr std::string_view kWave          = "src/ops/common/sm_wave_rules.h";

// --- route_cost_constants.h: the A16 / A8 refits and the per-generation modelling constants ---
// The A16 tiles: one leave-one-board-out refit of the model's constants.
inline constexpr Holdout h_a16_refit = Holdout::checked(
    "ninferno_route_tune timings of seven boards (each tile in the engine, L2 flushed; BF16, Q4 "
    "and "
    "Q8, T >= 17); the record does not name all seven",
    "each of the seven boards in turn: the constants refit with that board held out",
    "geomean time over the best tile",
    "1.095 of the best (RTX 4090 1.083, RTX 5090 1.068) against 1.110 for the F-lite rule and "
    "1.137 "
    "without the chain and reduction terms; the worst case is not recorded",
    "generic_cost.cpp price_routes comment",
    "one aggregate for the constants of the A16 model; the record gives no per-constant error; "
    "measured at 0.1.0 (54ef5203), before #499 re-priced the K-quant register MMA on sm_86 / "
    "sm_120");
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
inline constexpr Holdout h_min_warp_fill = Holdout::checked(
    "A16: ninferno_route_tune timings of seven boards (BF16, Q4, Q8, T >= 17); A8: --a8 timings of "
    "RTX 4090, RTX 5090, RTX 4060 Ti, L4, A10, A100",
    "each board in turn: the constants refit with that board held out (A100 in sample only)",
    "A16: geomean time over the best tile; A8: time over the best route of either class",
    "A16 1.095 of the best (4090 1.083, 5090 1.068); A8 1.06..1.09 on the consumer boards, A100 "
    "1.13 in sample; no error for this floor on its own",
    "generic_cost.cpp price_routes comment; generic_gemm.cpp route_tile comment",
    "the A16 figures were measured at 0.1.0 (54ef5203), before #499 re-priced the K-quant register "
    "MMA on sm_86 / sm_120");
// The A8 tiles (W4A8, I8, ggml MMQ) and their activation quantizer.
inline constexpr Holdout h_a8_refit = Holdout::checked(
    "ninferno_route_tune --a8 and --a8 --epilogue timings of RTX 4090, RTX 5090, RTX 4060 Ti, L4, "
    "A10, A100 (six formats, n 48..28672, T 17..2048, quantizer in the sample; SwiGLU and "
    "AddResidual on the A100)",
    "each board in turn: the constants refit with that board held out (the minor-0 row, A100, has "
    "one board and no held-out value)",
    "time over the best route of either class (A16 or A8)",
    "1.06..1.09 of the best on the consumer boards; A100 1.13 in sample only",
    "generic_gemm.cpp route_tile comment",
    "one aggregate for the A8 constants; the record gives no per-constant error");
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
         "the mean over the generation's cards (lab/speed_model/fit_moe_decode.py)",
         "none left out; the check is the same rows: MoE decode 33 points rms 4.5 % worst 13.5 %; "
         "Ada's 8-row point rests on one card",
         "DECISIONS 65; tests/test_choice.cpp")},
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
         "the RTX 4090 without PDL, the RTX 5070 Ti on a lab build with pdl::supported() false",
         "one prompt per card; Ampere takes Ada's int8 row; the rows' `cards` count the cards "
         "whose "
         "census reads them (one for most): a one-card row prices only its boards' estimate",
         "DECISIONS 66")},
    {kChoice, "kMmqTileCost",
     Holdout::checked(
         "Qwen3.5 9B's Q3_K_M, Q4_K_M (the MTP file), Q5_K_M and Q6_K 7K-token prompts (nsys, the "
         "calls by format and grid), each generation's cards together: Ada RTX 4060 Ti and 4070 "
         "Ti, "
         "Ampere RTX 3060, Turing RTX 2080 Ti, Blackwell RTX 5090",
         "Q4_K_S, which the rows were not read on: another format on the same cards, so a holdout "
         "of the format, not of a card; and Q4_1 on Gemma 4 31B's 2013-token prefill (RTX 4090, "
         "RTX 3090), predicted beforehand within 10 % of Q4_K's",
         "predicted against measured kernel time per call",
         "Q4_K_S: Ada -3 .. +8 %, Ampere 0 %, Turing -2 .. +2 %, Blackwell -5 .. +7 % (in sample "
         "-8 .. +7 %); Q4_1 at Q4_K's step x0.98 (RTX 4090) and x1.02 (RTX 3090) of the census",
         "DECISIONS 69; choice_constants.h at MmqTileCost",
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
     Holdout::checked(
         "the census of a 7K-token prompt (1024-token chunks): RTX 5090, RTX 4060 Ti, RTX 3060",
         "the RTX 5070 Ti, which the row was not read on",
         "predicted against measured attention time",
         "0 .. +1 % on the RTX 5070 Ti (in sample: RTX 5090 -1 .. +2 %, 4060 Ti -2 .. +1 %, 3060 0 "
         "%)",
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
         "run_sw.sh, 9B at a 22K prompt, against INT8 on the same card: Turing's one card; Ada the "
         "mean of the RTX 4070 Ti and 4090 (BF16 1.03 / 1.09, FP8 1.03 / 1.08, K8V4 1.01 / 1.04, "
         "NVFP4 1.25 / 1.22: within 6 %)",
         "none left out", "choice_constants.h at KvReadFactor")},
    {kChoice, "kKvSmRate",
     Holdout::none(
         "Turing: Qwen3.5 9B's int8 kernel and Gemma 4 12B's BF16 one on the RTX 2080 Ti; Volta: "
         "the "
         "Quadro GV100's census at 7K and 82K (Gemma 4 12B's BF16 full layer 707 us a call for 153 "
         "MB)",
         "one card a row: nothing left out", "choice_constants.h at KvSmRate")},
    {kChoice, "kMoeChunkShare",
     Holdout::none(
         "run_chunk.sh at a 22K prompt, 2026-10-23/24: Gemma 4 26B on the RTX 4070 Ti and 4090, "
         "Qwen3.6 35B on the 4090; the table is the mean of the three curves",
         "the three curves lie within 6 % of the mean: no curve was left out of it",
         "choice_constants.h at ChunkShare")},
};

inline constexpr FittedHoldout kGenerationHoldouts[] = {
    {kGeneration, "bf16 (the emulated rows)",
     Holdout::none("Turing's and Volta's BF16 rate: the fitted aggregate 1536 (the unit "
                   "route_tune's fit used, "
                   "where the BF16 MMA is emulated)",
                   "one aggregate for two generations; no board left out",
                   "core/generation.h at the rate comment")},
    {kGeneration, "w4a8_eff_scale",
     Holdout::none(
         "ninferno_route_tune timings of the RTX 4090, 5090, 4060 Ti, L4, A10 and A100 (18.10): "
         "the "
         "full-rate rows (ga100, hopper, gb100) take the A100's 0.46, the one board of them",
         "one board for the rows that differ from 1.0: nothing left out, and not carried to a lent "
         "row (fitted_route_scales: nullopt)",
         "core/generation.h comment (fitted constants)")},
    {kGeneration, "a16_cost_scale",
     Holdout::none(
         "the same route_tune timings: 1.11 on the consumer rows, 1.64 on the A100's (with the "
         "busy-SM fill it moved 0.89 -> 1.11, which also prices the K-quant and codebook dequant: "
         "5070 Ti ISTA T 17..32 1.20 -> 1.04)",
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
};

inline constexpr FittedHoldout kSmallTEtaHoldouts[] = {
    {kRoundCost, "kSmallTEta",
     Holdout::checked(
         "registry spec-tune rounds of Qwen targets, k 4..7, RTX 4070 Ti and RTX 4090 together",
         "each card's rounds predicted from the other card's eta alone (fit_eta.py item 3)",
         "R measured / R facts",
         "RTX 4070 Ti from the 4090 -3..+4 %, RTX 4090 from the 4070 Ti +1..+9 % (DECISIONS 53); "
         "+0.1 % and +0.2 % on average, rms 3.7 % and 2.0 % (DECISIONS 56, for scale)",
         "DECISIONS 53 and 56",
         "gpt-oss rounds run +16..19 % over their facts, its runs drafted BF16 (round_cost.h "
         "header comment, not DECISIONS 53 / 56)"),
     "ada"},
    {kRoundCost, "kSmallTEta",
     Holdout::checked(
         "registry spec-tune rounds of Qwen targets, k 4..7 (the held-out check covers k <= 7), "
         "RTX 5070 Ti and RTX 5090 together",
         "each card's rounds predicted from the other card's eta alone (fit_eta.py item 3)",
         "R measured / R facts",
         "RTX 5070 Ti from the 5090 (0.184) +4.3 % on average, rms 8.4 %; RTX 5090 from the 5070 "
         "Ti "
         "(0.162) +1.4 %, rms 4.4 %",
         "DECISIONS 56",
         "the plan's pick on the other card's pairs equals the held-out card's measured best or is "
         "within 1.5 % of it (gpt-oss 4 %)"),
     "gb20x"},
};

inline constexpr FittedHoldout kSpecHoldouts[] = {
    {kRoundCost, "kLaunchTime",
     Holdout::none(
         "the Ada census: Qwen3.5-9B Q4_K_M plain decode on the RTX 4090, 4070 Ti and 4060 Ti, "
         "every kernel within 2 %: t0 1.5 us without PDL, eta_dram 0.946",
         "the three cards are the fit; no generation without a row is priced by it (Ampere and "
         "Turing graph gaps were measured, RTX 2080 Ti 0.38 us a kernel, RTX 3060 ~0, but no "
         "intercept was fitted; gb20x's two cards cannot tell 0.5 from 1.5 us)",
         "choice_constants.h and round_cost.h at LaunchTime")},
    {kMoeShare, "kMoeRoutedShare",
     Holdout::none(
         "the drafter registry's speculation floors (decisions.md 53: the round cost's B(r)): rows "
         "routed apart read 0.30 of the experts independent routing would add",
         "fitted and checked on the same floors; no pair was left out",
         "docs/maintainer/decisions.md 53; spec/moe_routed_share.h")},
};

inline constexpr FittedHoldout kWaveHoldouts[] = {
    {kWave, "grid_passes_wave",
     Holdout::checked(
         "the RTX 5090 (170 SMs): Gemma 4's NVFP4 2048x5376 A4 prefill, where the TMA route's grid "
         "(16 row tiles) stays under one wave and loses to the 128-token MMA tile until T > 1280",
         "the RTX 5070 Ti (70 SMs), which the rule was not fitted on",
         "TMA against MMA tile prefill time by T",
         "TMA from T = 641 on the 70 SMs, 28 % faster there, equal from 768; other boards are "
         "extrapolated until measured",
         "ops/common/sm_wave_rules.h comment; ninferno_sm_wave_rules_test")},
};

} // namespace ninferno::fitted
