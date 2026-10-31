#pragma once

// The constants of the configuration choice (choice.h), each with where it comes from. The one
// place they live: a new measurement changes a row here, nothing else.
//
// Speed. A decode step of C rows at context n takes
//   t0 + (W(C) + C n b_kv k_kv + 2 C s + C min(n, window) b_ring) / (eta_dram BW)
//      + C 2 P / (eta_compute peak_s8) f_route
// and a prefill of n tokens takes n 2 P / (eta_prefill share peak_route) + F_att(n) / (eta_attention
// peak_bf16) seconds (choice.cpp attention_seconds, linear_prefill): W(C) the weights C rows read
// (an MoE's routed experts as concurrent rows, routed independently, add them: DECISIONS 65),
// b_kv the full layers' KV bytes a token, s a lane's recurrent state (read and written), b_ring a
// sliding ring's bytes a token, P the active parameters (plan_facts.cpp: the layers a token runs
// and the output head), BW the board's DRAM rate, peak the generation row's tensor rate times SMs
// times the reference clock (core/generation.h), F_att(n) the attention flops of the prompt over
// itself (full layers over every earlier token, sliding ones over the window), share an MoE chunk's
// share of the saturated rate (kMoeChunkShare; 1 for a dense model). k_kv and f_route are what a KV
// codec's read and the exact route cost on a generation beyond their bytes and flops.
//
// Fitted (scratchpad fit_choice.py, 2026-10-23) on 49 decode and 15 prefill points of Qwen3.5-9B
// Q4_K_M, Qwen3.8-27B UD-IQ3_XXS / UD-Q4_K_XL and Qwen3.6-35B-A3B UD-IQ3_XXS / UD-Q4_K_M on the
// RTX 2080 Ti, 4070 Ti, 4090 and 5070 Ti (run_calib.sh, run_sw.sh, docs/performance/runs):
// decode 5.2 % rms (worst 15.5 %). The prefill: refitted 2026-10-27 with its attention term
// (choice-notes/lab/scripts/fit_prefill.py on choice-notes/data/prefill/rows.json: 82 points, Qwen3.5
// 9B, Qwen3.8 27B, Gemma 4 12B / 26B-A4B / 31B, Qwen3.6 35B-A3B at ~2K / 7K / 21K-token prompts on the
// RTX 2080 Ti, 3060, 4060 Ti, 4070 Ti, 4090, 5070 Ti, 5090, and the 22K points of
// tests/fixtures/choice/speed_measured.json): from 7K tokens on, dense 6.4 % rms (worst 18.7 %), MoE
// 8.3 % (18.9 %); a request's fixed time (10-190 ms: HTTP, a decode step) is not modeled, so at 2K the
// estimate runs up to 30 % above a measured prompt rate. The Quadro GV100 (Volta) is left out: its
// Gemma 4 attention (heads of 512) runs 4 times slower than its Qwen one, the two disagree.
//
// Accuracy. KLD against the exact reference (--activations exact, BF16 KV, FP32 state) on the
// fixed corpus (ninferno-perplexity --quick, eval/corpora/perplexity-1m), one term per switch;
// the switches' terms combine as the root of the sum of their squares (the product default
// measured 0.0040, the root sum square of its terms 0.0044, their plain sum 0.0067); the
// quantization's own term adds on top (its combination with them is not measured).
//
// The trade. One number, kAccuracyWeight: a configuration's worth is -ln(T) - lambda KLD, T the
// seconds of a typical request (kRequestPrompt tokens in, kRequestAnswer out). lambda 25 makes
// 0.001 KLD worth 2.5 % of the request's time: where two configurations cost about the same, the
// more accurate wins; where one is much faster, it wins. The exact route (0.0040 KLD better than
// the product default on 9B) thus wins only where it costs under about 11 % of a request's time
// (it costs 26 % on the RTX 4070 Ti, 38 % on the 5070 Ti at 2048 + 512 tokens). At lambda 50 the
// 4070 Ti's two were within 0.01 of each other: no clear "about the same".

#include "core/cost_constants.h"
#include "ninferno/types.h"

#include <cstdint>
#include <string_view>

namespace ninferno::product::choice {

// The activation route a configuration's linears take (the peak its compute runs at).
enum class Route : std::uint8_t {
    Int8, // --activations fast or int8 on integer weights (every GGUF format: AllowA8)
    Bf16, // --activations exact, or weights with no int8 route
    Fp4,  // --activations fast on NVFP4 weights where FP4 runs natively (W4A4)
};

[[nodiscard]] constexpr std::string_view route_name(Route route) noexcept {
    return route == Route::Int8 ? "int8" : route == Route::Bf16 ? "bf16" : "fp4";
}

// ---- Speed.

// Decode (2026-10-28; an MoE's since 2026-10-30, DECISIONS 65). The fitted share of the int8
// peak a step's flops reached (kDenseDecode) is replaced by measured per-generation efficiencies of
// the kernels a step runs. A step of C rows at context n reads
//   W + C (n b_kv k_kv + 2 s + min(n, window) b_ring)
// bytes (W the weights it reads: every layer and the output head, not the input table a token
// gathers one row of, nor the MTP layers or own drafter plain decode leaves; of an MoE's routed banks,
// the share of its experts the C rows route to) and takes
//   t(C) = W / (eta_rows(C) BW) + C (n b_kv k_kv + 2 s + min(n, window) b_ring) / (eta_dram BW)
//          + N_gemv t_gemv + N_small(C) t_small,
// and no less than its rows' flops at the prefill route's rate (a wide batch is compute-bound).
// N_gemv and N_small(C) are the kernels the family's graph launches a step (kDecodeLaunches, nsys
// counts, facts of the graph); t_gemv and t_small what one costs beyond its bytes (kKernelCost, the
// census's measured residual); eta_rows(C) the share of the DRAM rate C rows' weight reads reach on
// the weight class's routes (kRowsEta, measured). Measured against tests/fixtures/choice/
// decode_measured.json (Qwen3.5-9B Q4_K_M and Q8_0, Gemma 4 12B Q4_K_M, 1/2/4/8 streams, seven cards,
// master f68eca34, 2026-10-31; first master 03afc418, 2026-10-28) and the census
// tests/fixtures/choice/decode_census.json (likewise);
// choice-notes/lab/speed_model/fit_decode.py derives every row from the two.
// The share of the DRAM rate a weight read long enough to saturate it reaches: per kernel 0.946-0.952
// on GDDR6, 6X and 7 (nsys census of Qwen3.5-9B decode, 28.10: RTX 4090, 4070 Ti, 4060 Ti, 5070 Ti;
// its R8 GEMVs and the Q6_K output head). Volta's row: the Quadro GV100's HBM2 with ECC tops out at
// 660 of its 870 GB/s (PR #168's census, one card): the share its reads reach.
struct DramEta {
    std::string_view generation;
    double eta;
    int cards;
};
inline constexpr double kDramEta = 0.946;
inline constexpr DramEta kDramEtaOf[] = {{"volta", 660.0 / 870.0, 1}};

// The weight class of a quantization: which decode routes its weights take.
enum class WeightClass : std::uint8_t {
    KQuant, // ggml's K-quant mixes (Q4_K, Q5_K, Q6_K, with Q8_0 and 16-bit tensors: Q4_K_M ... Q6_K,
            // UD-Q4_K_XL, UD-Q5_K_XL): R8 at one row, ggml's vec kernel at 2-4, register MMA / MMQ at 5-8
    Q8,     // Q8_0 (q8_g32_fp16): R8 at every width
    IQ4,    // ggml's IQ4_XS / IQ4_NL mixes (with Q4_K/Q5_K/Q8_0 tensors): R8 at one row, vec / MMQ beyond
    Other,  // a format whose decode routes no census measured (other codebook IQ*, Q4_0/Q4_1, Q2_K,
            // Q3_K, FP4): its speed is transferred (the K-quants' rows) and it decides nothing
};

// The kernels a family's decode graph launches a step, by weight class, at 1, 2-4 and 5-8 rows
// (nsys, master 03afc418; another size of the family scales them by its layers): the weight-reading
// ones (R8, register MMA, ggml's vec and MMQ, the output head) and the small ones (norms, the mixers'
// control, convolution and recurrent update, attention, the activations' quantize ahead of a vec or
// MMQ call, sampling). One row: Qwen3.5-9B 145 + 167 = 312 (9.75 a layer), its Q8_0 177 + 191 = 368
// (11.5), Gemma 4 12B 213 + 249 = 462 (9.6), the same on every card. At 2-8 rows Ada's counts (the
// reference generation's routes; the 5-8-row K-quant routes launch 167-215 small kernels by card).
// Once a step, not a layer (the census's positions outside the layers): the output head's GEMV, and
// the input gather, the final norm and the sampling (Qwen 3.5: the scatter, top-k, sample, copy and
// finite check, 7; Gemma 4 also its embedding scale, logit soft-cap and two scalar sets, 9). Another
// size of the family scales only the rest by its layers. The MoE rows (nsys, master 289a63af, RTX 5070
// Ti, 60 decode steps of one row): Qwen3.6-35B-A3B UD-Q3_K_S 311 + 267 = 578 a step (the router,
// select, combine and the routed banks' two calls a layer; the Gated DeltaNet layers' projections
// read weights too), gpt-oss-20b MXFP4 97 + 199 = 296; the graph's count in the lab tunes (spec-tune
// graph_kernels) is the same, 577 and 295; Gemma 4 26B-A4B UD-Q4_K_M (RTX 4090, master 4c5b8165)
// 181 + 219 = 400 (its dense MLP beside the experts in every layer). At 2-8 rows the one-row count
// (not measured). A family or
// class without a row takes the family's K-quant row, else the family's other row, else Qwen 3.5's
// (transferred).
struct DecodeLaunches {
    std::string_view family;
    WeightClass weights;
    double layers;   // the census model's
    double gemv;     // the weight-reading kernels a step (the same at every row count)
    double small[3]; // the small ones at 1, 2-4 and 5-8 rows
    double fixed_gemv;  // of them once a step (the output head)
    double fixed_small; // of the small ones once a step (gather, final norm, sampling)
    bool moe = false;   // an MoE's graph (router, select, combine): its family's every quantization
};
inline constexpr DecodeLaunches kDecodeLaunches[] = {
    {"qwen3_5_text", WeightClass::KQuant, 32, 145, {167, 295, 215}, 1, 7},
    {"qwen3_5_text", WeightClass::Q8, 32, 177, {191, 367, 367}, 1, 7},
    {"gemma4_text", WeightClass::KQuant, 48, 213, {249, 557, 345}, 1, 9},
    {"qwen3_5_moe_text", WeightClass::Other, 40, 311, {267, 267, 267}, 1, 6, true},
    {"moe_decoder", WeightClass::Other, 24, 97, {199, 199, 199}, 1, 6, true},
    {"gemma4_text", WeightClass::Other, 30, 181, {219, 219, 219}, 1, 9, true},
    {"qwen3_5_text", WeightClass::IQ4, 32, 161, {167, 295, 167}, 1, 7}, // RTX 4060 Ti's census
};

// What a kernel costs a step beyond its bytes, by generation: the census's measured residual, one row,
// the three models (decode_census.json): a weight-reading kernel's time less its weights at kDramEta,
// a small one's less the state and KV bytes it reads, the mean over every kernel of the class. Where
// the census explains it: Ada's plain R8 projections 1.5 us of launch and ramp plus the graph's
// 0.29 us gap (the median 1.9-2.2 us); the SwiGLU and residual epilogues 2.2-6 us and the Q8_0 R8 on
// the 4090 3.5 us raise the mean to 2.34; its norms 1.7-1.9 us (four times their bytes), the attention
// at a short context 6.6. Blackwell's PDL hides the plain projections' launch (0.2 us median: their
// weights are read during the norm) but not the epilogues', and its small kernels carry the prefetch.
// Turing: the projections 4.2-8.3 us, one card. A row of one card prices only its own boards'
// estimate (marked transferred; no decision reads it: the choice prices on a reference card's
// generation; DECISIONS 60); a generation without a row takes Ada's (transferred). Ampere (RTX 3060,
// one card): its projections 3.1-5.5 us, no PDL, no Ada L2.
struct KernelCost {
    std::string_view generation;
    double gemv;  // seconds a weight-reading kernel beyond its bytes
    double small; // seconds a small kernel beyond its bytes
    int cards;
};
inline constexpr KernelCost kKernelCost[] = {
    {"ada", 2.35e-6, 1.94e-6, 3},    // RTX 4060 Ti, 4070 Ti, 4090
    {"gb20x", 1.90e-6, 2.29e-6, 2},  // RTX 5070 Ti, 5090
    {"ga10x", 4.72e-6, 2.55e-6, 1},  // RTX 3060
    {"turing", 8.56e-6, 4.30e-6, 1}, // RTX 2080 Ti
};

// The share of the DRAM rate C rows' weight reads reach on a weight class's routes (C = 1: kDramEta;
// linear in log2 C between points; beyond the last, the last's): measured, the mean over the class's
// models and the generation's cards, the launches at kDecodeLaunches and kKernelCost. Ada K-quants
// 0.88-0.92 / 0.85-0.89 / 0.74-0.85 at 2 / 4 / 8 rows (Gemma 4's shapes the lowest), Q8 0.92-0.95 /
// 0.90-0.93 / 0.84-0.89; Blackwell K-quants 0.92-0.93 / 0.78-0.81 / 0.71-0.80. Turing: one card. A
// class without rows (WeightClass::Other) takes the K-quants' and is marked transferred.
struct RowsEta {
    std::string_view generation;
    WeightClass weights;
    double rows;
    double eta;
    int cards;
};
inline constexpr RowsEta kRowsEta[] = {
    {"ada", WeightClass::KQuant, 2, 0.906, 3},    {"ada", WeightClass::KQuant, 4, 0.875, 3},
    {"ada", WeightClass::KQuant, 8, 0.795, 3},    {"ada", WeightClass::Q8, 2, 0.936, 3},
    {"ada", WeightClass::Q8, 4, 0.915, 3},        {"ada", WeightClass::Q8, 8, 0.869, 3},
    {"ada", WeightClass::IQ4, 2, 0.927, 2},       {"ada", WeightClass::IQ4, 4, 0.885, 2},
    {"ada", WeightClass::IQ4, 8, 0.773, 2}, // RTX 4060 Ti, 4070 Ti: IQ4_XS, IQ4_NL
    {"gb20x", WeightClass::KQuant, 2, 0.924, 2},  {"gb20x", WeightClass::KQuant, 4, 0.795, 2},
    {"gb20x", WeightClass::KQuant, 8, 0.753, 2},  {"gb20x", WeightClass::Q8, 2, 0.951, 2},
    {"gb20x", WeightClass::Q8, 4, 0.913, 2},      {"gb20x", WeightClass::Q8, 8, 0.869, 2},
    {"ga10x", WeightClass::KQuant, 2, 0.891, 1},  {"ga10x", WeightClass::KQuant, 4, 0.770, 1},
    {"ga10x", WeightClass::KQuant, 8, 0.761, 1},  {"ga10x", WeightClass::Q8, 2, 0.936, 1},
    {"ga10x", WeightClass::Q8, 4, 0.903, 1},      {"ga10x", WeightClass::Q8, 8, 0.790, 1},
    {"turing", WeightClass::KQuant, 2, 0.870, 1}, {"turing", WeightClass::KQuant, 4, 0.721, 1},
    {"turing", WeightClass::KQuant, 8, 0.688, 1}, {"turing", WeightClass::Q8, 2, 0.942, 1},
    {"turing", WeightClass::Q8, 4, 0.881, 1},     {"turing", WeightClass::Q8, 8, 0.678, 1},
};

// An MoE's routed banks at C concurrent rows (DECISIONS 65): the experts any of the rows routes to,
// each for a row or two, many small reads apart from the dense GEMVs kRowsEta measures. Their share
// of the DRAM rate (C = 1: kDramEta; linear in log2 C between points; beyond the last, the last's):
// the MoE step rows of speed_measured.json (the server's windows of exactly C rows, eight different
// tasks) less every other term of the step, the mean over the generation's cards and quantizations
// (lab/speed_model/fit_moe_decode.py). A generation without rows takes Ada's (transferred).
struct MoeRoutedEta {
    std::string_view generation;
    double rows;
    double eta;
    int cards;
};
inline constexpr MoeRoutedEta kMoeRoutedEta[] = {
    // Ada: RTX 4090 (UD-Q4_K_M, UD-Q3_K_S), 4060 Ti (UD-Q3_K_S, 7 rows its most: no 8-row point).
    {"ada", 2, 0.928, 2}, {"ada", 4, 0.911, 2}, {"ada", 8, 0.806, 1},
    // Blackwell: RTX 5070 Ti (UD-IQ3_XXS, UD-Q3_K_S), 5090 (UD-Q3_K_S).
    {"gb20x", 2, 0.918, 2}, {"gb20x", 4, 0.740, 2}, {"gb20x", 8, 0.634, 2},
};

// The model's measured error on a generation's dense decode estimate: the rms of ln(predicted /
// measured) over decode_measured.json's rows of the generation's cards under the rows above (derived,
// not tuned: ninferno_choice_test recomputes it). Two quantizations whose worths (-ln T - lambda KLD)
// differ by less are a tie: the lower KLD wins (DECISIONS 60).
struct PredictionError {
    std::string_view generation;
    double rms;
};
inline constexpr PredictionError kDecodePredictionError[] = {{"ada", 0.0259}, {"gb20x", 0.0171}};

// kDp4aOpsPerSmClock (a generation without an int8 MMA decodes its int8 route at it, choice.cpp
// decode_int8_peak): core/cost_constants.h, a chip fact shared with spec/round_cost.h.

// A row per generation (core/generation.h's names) and kind; a generation without its own row
// takes kTransferGeneration's, marked as a transfer (its prefill: a generation's with the same BF16
// tensor rate per SM where one has rows, choice.cpp prefill_stand_in).
inline constexpr std::string_view kTransferGeneration = "ada";

struct PrefillEta {
    bool moe;
    std::string_view generation;
    Route route;
    double eta;
    int cards; // lab cards it rests on: one card prices its own boards' estimate, marked transferred
};

// The linears' share of the route's peak (fit_prefill.py). The exact rows (BF16) were fitted on 9B
// at a 22K prompt without the attention term; they keep that fit's time with attention now apart.
// Turing's BF16 row kept that time against the attention's former 0.069: with the census's 0.160
// below it prices an exact prefill too fast at long prompts until a 22K point refits it.
inline constexpr PrefillEta kPrefillEta[] = {
    // Turing's int8 row: the nsys census of one Qwen3.5-9B Q4_K_M prefill on the RTX 2080 Ti
    // (2026-10-29, master 03afc418, 2K and 8K prompts: the step less its attention kernel, 0.182 /
    // 0.187; one card). The facts behind it against the RTX 4070 Ti's 0.29 (the same census there; Ada's row is 0.306):
    // ggml's MMQ runs the same launch on both (256 threads, 255 registers, 57.8 KB shared: one block,
    // 8 warps a SM) yet takes 1.9x the SM-clocks per operation on Turing: its dequantize and scale
    // math run on the CUDA cores, 64 FP32/INT32 lanes a SM on Turing against 128 on Ampere and Ada
    // (x0.5), and the 2080 Ti sustains ~1.9 GHz against its 1.545 GHz boost while Ada's run ~4 %
    // above theirs (x1.21): 0.175 from the facts, 0.185 measured. The former 0.241 (fitted at 22K
    // with attention apart) put a 2K prefill 27 % fast (the user kit, #210).
    {false, "turing", Route::Int8, 0.185, 1},
    {false, "turing", Route::Bf16, 0.073, 1}, // RTX 2080 Ti, 9B exact (stale: see above)
    // Volta's int8 route is not Turing's: without an int8 MMA its prefill linears dequantize each
    // chunk's weights to FP16 (decode_rows_kernel) and multiply them on the HMMA tiles (CUTLASS
    // sm70, the m8n8k4 FP16 MMA: 1024 operations a SM-clock at FP32 accumulation, the row below
// relative to this table's 1536 as every row), where the
    // Turing row it took before (MMQ on the int8 MMA) put the Quadro GV100's 2048-token prefill
    // 22 .. 28 % slow (user kits, master 59933708). The census of the GV100's prefill (nsys of the
    // server, 962 / 1912 / 2862-token prompts twice, master 59933708; kernel time without the
    // attention kernels and the tail calls, against this table's 1536 at the board's 1627 MHz
    // boost): Qwen3.5-9B Q4_K_M 0.235, Gemma 4 12B Q4_K_M 0.261 (the CUTLASS kernel alone at 0.70 of
    // the HMMA rate at the 1.28 GHz the board holds at its power limit). One card: transferred.
    // Its exact route has no row: the int8 row times Turing's exact-to-int8 ratio (unmeasured).
    {false, "volta", Route::Int8, 0.248, 1},
    {false, "ada", Route::Int8, 0.306, 3},    // RTX 4060 Ti, 4070 Ti, 4090: 9B, 27B, Gemma 12B, 31B
    {false, "ada", Route::Bf16, 0.317, 1},    // RTX 4070 Ti, 9B exact
    {false, "gb20x", Route::Int8, 0.324, 2},  // RTX 5070 Ti, 5090: 9B, 27B, Gemma 12B
    {false, "ga10x", Route::Int8, 0.363, 1},  // RTX 3060: 9B, Gemma 12B (one card)
    {true, "ada", Route::Int8, 0.210, 2},     // RTX 4060 Ti, 4090: Gemma 26B-A4B, 35B-A3B
    {true, "gb20x", Route::Int8, 0.205, 2},   // RTX 5070 Ti, 5090: Gemma 26B-A4B, 35B-A3B
};

// A dense model's int8 route on Q8 weights prefills on the generic int8 tiles (the W4A8 kernel's
// Q8 codec, ops/generic q8_prefill_a8), not ggml's MMQ that kPrefillEta's int8 rows measured. Since
// the 64-token tiles run token-major (PR #249) a tile no longer streams its weights from DRAM once
// per 64 tokens and the call is bound by the tile's own work (DECISIONS 67): a 64 x 64 tile takes
// c0 + K c1 SM-clocks, c1 about 12 clocks a K step on every generation with an int8 MMA (decoding
// the weights on the CUDA cores, a third of the tensor peak), c0 the tile's start. A linear of hidden
// width K then reaches 2 x 64 x 64 / ((c0 / K + c1) x the s8 rate) of the int8 peak (choice.cpp
// tiles_eta, at the model's own hidden width): Ada c0 3352, c1 12.15 clocks (the RTX 4090's nsys
// census), Blackwell 1226, 11.62 (the RTX 5070 Ti's), Ampere its RTX 3060's c1 12.5 with Ada's c0
// (one card, transferred). At K 4096 that is 0.308 / 0.336 / 0.300; measured (sm_bench.py prefill,
// master f8d4399d, Qwen3.5-9B Q8_0 at 3K-14K prompts, attention apart): RTX 4060 Ti 0.325, 4070 Ti
// 0.317, 4090 0.301, 5070 Ti 0.357.
// Turing's int8 tile is another kernel without a clock count: its row keeps the share fitted on the
// master 03afc418 points (one card, transferred; within 4 % of the RTX 2080 Ti's points after
// PR #249, which rose 1.11x), and a generation without an int8 MMA keeps its int8 row (Volta).
struct TilesEta {
    std::string_view generation;
    double c0, c1; // the tile's start and K step, SM-clocks (0: no clock count, `eta` stands)
    double eta;    // the measured share where there is no clock count
    int cards;
};
inline constexpr TilesEta kPrefillTilesEta[] = {{"ada", 3352, 12.15, 0, 3},
                                                {"gb20x", 1226, 11.62, 0, 2},
                                                {"ga10x", 3352, 12.5, 0, 1},
                                                {"turing", 0, 0, 0.147, 1}};

// A prefill linear's tile (DECISIONS 66): a CTA of 64 tokens (an MoE expert's: a job of up to 64 of
// its slots) x 64 output rows, the whole K deep, takes c0 + K c1 SM-clocks (the card's reference
// clock gives the seconds); a call runs in rounds of them (TileCost::resident). From the census of one
// ~1.8K-token prompt (nsys, the RTX 4090 without PDL at 2520 MHz, the RTX 5070 Ti at 2452 MHz on a
// lab build with pdl::supported() false): Qwen3.6-35B-A3B's and gpt-oss-20b's calls. The int8
// tile's K step is about 12 SM-clocks on every generation that has the int8 MMA (Ada 12.1, the RTX
// 4070 Ti 12.8, Ampere's RTX 3060 12.5, Blackwell 11.6: decoding the weights on the CUDA cores
// bounds it, a third of the tensor peak), so Ampere takes Ada's int8 row.
enum class TileKernel : std::uint8_t {
    Int8, // the int8 tiles: w4a8_kernel, moe_w4a8_kernel (Q4_K, Q8_0, MXFP4 converted on Ada)
    A16,  // the 16-bit register MMA of an MoE's IQ3 banks in fragment layout
    Mx,   // the block-scaled kind::mxf8f6f4 MMA of MXFP4 banks (sm_120a)
};
// A SM holds `resident` of a kernel's CTAs at once (its registers and shared memory against the SM's:
// the int8 tiles' 128 threads at 177-255 registers and 40 KB on Ampere, Ada and Blackwell, and Ada's
// 16-bit register MMA's at 255 registers and 1-2 KB, two;
// Turing's 64 KB of shared memory, one), so a grid runs in rounds of SMs x resident CTAs, each round
// `resident` tiles' time: the RTX 5090's N 4096 calls at a 1024-token chunk (1024 CTAs) take 4 rounds
// of 340, 8 tiles' time, not 6.02 waves of 170 (DECISIONS 66): the last round's 4 CTAs, one a SM, took
// a whole round, a tile alone on a SM no faster than two sharing it. A row without a read keeps one.
// A grid of no more CTAs than SMs runs one a SM, alone: it takes kLoneTile tiles' time, between the
// one a tile's c0 + K c1 gives (read on two CTAs sharing a SM, each at half its rate) and the round's
// two. Read on Qwen3.5 9B's Gated DeltaNet b/a projection (16 CTAs, K 4096; the census's 48 calls a
// prompt) against its tile's time: RTX 4090 27.5 / 21.1 us, 4060 Ti 28.4 / 21.0, 4070 Ti 26.1 /
// 20.4, RTX 3060 39.3 / 30.7, 5070 Ti 25.5 / 19.9, 5090 27.1 / 20.3: 1.28-1.35 on three generations.
inline constexpr double kLoneTile = 1.3;
struct TileCost {
    std::string_view generation;
    TileKernel kernel;
    double c0, c1; // SM-clocks a tile, and a K step
    int cards;     // the cards whose census reads it (one: its boards' estimate only)
    int resident;  // CTAs a SM holds at once
};
inline constexpr TileCost kTileCost[] = {
    {"ada", TileKernel::Int8, 3352, 12.15, 3, 2},  {"ga10x", TileKernel::Int8, 3352, 12.5, 1, 2},
    {"gb20x", TileKernel::Int8, 1226, 11.62, 1, 2}, {"ada", TileKernel::A16, 2344, 17.9, 1, 2},
    {"gb20x", TileKernel::A16, 785, 14.96, 1, 1},   {"gb20x", TileKernel::Mx, 1226, 8.34, 1, 1},
};

// ggml's matrix kernel (MMQ, mul_mat_q) on a dense model's K-quant linears where it takes their
// prefill (measured::mmq_prefill_pays; DECISIONS 69): a call runs kMmqSliceTokens tokens at a time;
// a tile is 128 weight rows x 128 tokens over the whole K (one CTA a SM, 256 threads at 254
// registers), a slice's grid ceil(N / 128) x ceil(t / 128) tiles. A grid that fills its last wave to
// 90 % runs a tile a CTA in rounds of SMs; else stream-k: the SMs share the tiles' K steps evenly and
// a fixup kernel adds the split tiles (ggml mmq.cuh). A tile takes c0 + K c1 SM-clocks, c1 by the
// weights' format (its block's decode: Q3_K's high-bit mask and 6-bit scales cost as much as Q6_K's
// six bits): read on Qwen3.5 9B's Q3_K_M, Q4_K_M (the MTP file), Q5_K_M and Q6_K 7K-token prompts
// (nsys, the calls by format and grid against the GGUF's tensor types), each generation's cards
// together; checked on its Q4_K_S, which they were not read on:
//   Ada (RTX 4060 Ti, the 4070 Ti)  c0 2000, Q4_K 42.75, Q5_K 45.25, Q6_K 50.75: -5 .. +3 %, Q4_K_S -3 .. +8 %
//   Ampere (RTX 3060)               c0 4000, 41.5, 43.5, 48.75: -1 .. +1 %, Q4_K_S 0 %
//   Turing (RTX 2080 Ti)            c0 0, 71.75, 74.5, 77.75: -8 .. +4 %, Q4_K_S -2 .. +2 %
//   Blackwell (RTX 5090, stream-k)  c0 3000, 45.5, 49.5, 59: -7 .. +7 %, Q4_K_S -5 .. +7 %
// (Q3_K read as Q6_K: Ada 51.5, Ampere 49.25, Turing 91.75, Blackwell 60.5; its mixes are not a
// K-quant class here, DECISIONS 69.) A mix's tiles take its label's format (Q4_K_M: Q4_K), its bits
// a weight past that format's moving them toward Q6_K's (the mixes raise ffn_down, attn_v and the
// like to Q5_K / Q6_K; the bits are the whole file's, its token table and head among them). A slice
// of t tokens takes ggml's narrowest column tile with the fewest tiles (16, 32, 64 or 128: ggml_mmq
// kColumnTiles); a J-column tile's K step is (kMmqTileFixed + (1 - kMmqTileFixed) J / 128) of a
// 128-column one (the RTX 4060 Ti's 64-column Q4_K tiles: 0.716).
struct MmqTileCost {
    std::string_view generation;
    double c0;
    double q4, q5, q6; // SM-clocks a K step of a 128-column tile, by format
    double q3;         // Q3_K's (a Q3_K mix's calls in the census; 0: not read). The W4A8 row's: a
                       // round of two 64 x 64 tiles a SM (kW4a8Q3kTile), not a 128-column tile
    double iq4xs;      // IQ4_XS's (an IQ4_XS file's calls in the census; 0: not read)
    int cards;
};
inline constexpr MmqTileCost kMmqTileCost[] = {{"ada", 2000, 42.75, 45.25, 50.75, 51.5, 38.3, 2},
                                               {"ga10x", 4000, 41.5, 43.5, 48.75, 49.25, 36.6, 1},
                                               {"turing", 0, 71.75, 74.5, 77.75, 91.75, 59.6, 1},
                                               {"gb20x", 3000, 45.5, 49.5, 59, 60.5, 42.9, 1}};
inline constexpr double kMmqTileFixed = 0.43;
// Where MMQ does not pay (measured::kMmqPrefillPays: the RTX 5070 Ti as the 5060 Ti) a dense
// K-quant prefills on the W4A8 tiles (generic_gemm.cpp: Q4_K and Q6_K the producer-warp 128 x 64
// tile on Blackwell, w4a8_ws_wins; Q5_K the classic 128 x 64): a tile is 128 tokens x 64 weight rows
// (a gated pair's 32 gate and 32 up rows) over the whole K, one CTA a SM (86-92 KiB, 61 KiB with 256
// threads), a call's grid ceil(N / 64) x ceil(t / 128) tiles in rounds of SMs, the activations'
// quantization and the epilogue the small kernels'. Read on the RTX 5070 Ti's census of Qwen3.5 9B's
// Q4_K_M, Q4_K_S, Q5_K_M and Q6_K at 7K tokens (the calls by format and grid, as MMQ's), c0 within
// a few K step: Q4_K 23, Q5_K 28, Q6_K 29.5 SM-clocks a K step (the K-4096 and K-12288 calls agree
// within 2 %) — a 128 x 128 MMQ tile's on the RTX 5090 per element (45.5 = 2 x 22.75 for Q4_K).
inline constexpr MmqTileCost kW4a8KquantCost[] = {{"gb20x", 0, 23, 28, 29.5, 37.4, 20.2, 1}};
inline constexpr double kW4a8KquantTokens = 128;
inline constexpr double kW4a8KquantRows   = 64;
// Q3_K there runs the classic tile (W4a8Q3K, W4a8Schedule<64, 64, 2, 2>): 64 tokens x 64 weight rows,
// 128 threads at 185 registers and 40 KiB, two CTAs a SM; a grid of ceil(N / 64) x ceil(t / 64)
// tiles in rounds of 2 x SMs, a round 37.4 SM-clocks a K step (`q3` above: the RTX 5070 Ti's census
// of Qwen3.5 9B's Q3_K_M at 7K, its three grids' rounds within 3 %: 62.2 .. 64.3 us at K 4096).
inline constexpr double kW4a8Q3kTile     = 64;
inline constexpr double kW4a8Q3kResident = 2;
inline constexpr double kMmqSliceTokens = 512; // generic_gemm.cpp kMmqPrefillSlice
inline constexpr double kMmqMinTokens   = 17;  // generic_gemm.cpp: MMQ from the prefill band (vector / R8 below)
inline constexpr double kMmqTile        = 128;
// Beside the tiles, every grid quantizes its input to q8_1 (a 16-bit element read, 1.125 bytes
// written: a gated pair's two grids each, the census's quantize calls one a grid) and the epilogue
// turns the FP32 output into 16 bits (4 bytes read, 2 written; a gated pair's SwiGLU writes one row
// of the two), at the DRAM rate (the census: RTX 3060, 2080 Ti and 5090 within 8 %; Ada's L2 holds
// part of them). The small kernels' passes (kSmallKernelCost, read on the int8 route) also count
// that route's one quantization an input, which MMQ does not run: not taken out. A stream-k call's
// fixup reads and writes a tile's FP32 partial a SM.
inline constexpr double kMmqInputBytes   = 3.125; // an input element, a grid
inline constexpr double kMmqOutputBytes  = 4;     // read, an output element of the tiles
inline constexpr double kMmqWrittenBytes = 2;     // written, an element of the call's output

// A prefill's small kernels (DECISIONS 66, the census of Qwen3.5 9B Q8_0's 1819-token prompt outside
// its linears, attention and tail calls, seven cards). The Gated DeltaNet's chunked scan is a count
// of SM-clocks a token and state element (12.6 M elements over its 24 layers): RTX 4060 Ti 0.049,
// 4070 Ti 0.064, 4090 0.062; RTX 5070 Ti 0.040, 5090 0.057; RTX 3060 0.051; Turing's scan has no
// tensor path, the RTX 2080 Ti 0.196. The rest (the norms, the activations' quantization, the
// convolution, the gates) is bytes a token, layer and element of the hidden width over the DRAM
// rate: some 15 passes over a 16-bit row, 30 bytes (RTX 4060 Ti 27, 4070 Ti 26, 4090 30, 5070 Ti 33,
// 5090 35), where a chunk's row of activations (1024 x 8 KB) stays in the L2 of 24-96 MB; Ampere's
// 3-6 MB L2 writes it out and reads it back, twice the bytes (the RTX 3060's 61); Turing's kernels
// besides (the RTX 2080 Ti's 88). One card a generation: transferred.
struct SmallKernelCost {
    std::string_view generation;
    double scan_clocks;  // SM-clocks a token and recurrent state element
    double pass_bytes;   // bytes a token, layer and element of the hidden width
    int cards;
};
inline constexpr SmallKernelCost kSmallKernelCost[] = {
    {"ada", 0.058, 30, 3}, {"gb20x", 0.048, 30, 2}, {"ga10x", 0.051, 60, 1}, {"turing", 0.196, 88, 1},
    // The Quadro GV100's census of Qwen3.5 9B Q8_0 at 7K tokens (DECISIONS 70): the scan's chunk
    // recurrence and its preparation 267 ms, the rest (the convolution, norms, gates, the a / b
    // projections on R8, conversions) 80 ms.
    {"volta", 0.381, 56, 1}};

// Volta's dense prefill (generic_volta_cutlass.cu, DECISIONS 70): a call casts its activation to FP16,
// decodes its weight rows to FP16 in chunks of kVoltaChunkBytes (gate and up rows together), multiplies
// each chunk on CUTLASS's sm_70 GEMM (128 x 128 x 32 tiles of tokens x weight rows, two CTAs a SM,
// rounds of SMs x 2 each two tiles' time) and stores through its epilogue (FP32 in, 16 bits out; a
// gated pair's SwiGLU reads both halves, a single plain projection of the hidden width is stored by
// the GEMM itself). Read on the Quadro GV100's census of Qwen3.5 9B's Q8_0, Q6_K and Q4_K_M (the MTP
// file) at 7K tokens: a tile's K step 52.4 SM-clocks (the GEMM's calls -4 .. 0 %, format-blind: one
// GEMM for every format); a weight element's decode Q8_0 0.69, Q4_K 0.99, Q5_K 1.11, Q6_K 0.94
// SM-clocks (-2 .. +1 %: the K-quants' decode is compute, Q8_0's near its bytes); the cast (4 bytes an
// input element) and the epilogue (6, a SwiGLU pair 10 an output) at the DRAM rate (they ran at 0.75
// of it: -21 .. -27 % on a tenth of the time). One card: transferred.
struct VoltaPrefillCost {
    double tile_step;              // SM-clocks a K step of a 128 x 128 tile
    double q8, q4k, q5k, q6k;      // SM-clocks a weight element's decode
    int cards;
};
inline constexpr VoltaPrefillCost kVoltaPrefillCost{52.4, 0.69, 0.99, 1.11, 0.94, 1};
inline constexpr double kVoltaChunkBytes = 4096.0 * 17408 * 2; // generic_volta_cutlass.cu kChunkWeightBytes
// A tiled format's call (Q8_0, Q4_K, Q5_K: their own tile beside the copy) takes the copy from this
// many tokens, or at any width on a board of this much DRAM (generic_gemm.cpp volta_copy_pays); a
// shorter call on a slower board runs the tile, not read: the share prices it.
inline constexpr double kVoltaCutlassMinTokens = 17; // volta_cutlass_takes: the prefill band
inline constexpr double kVoltaCopyMinTokens   = 256;   // kVoltaTileMaxTokens
inline constexpr double kVoltaCopyMinDramBps  = 870e9 / 1.13; // kVoltaCopyMinDramGbps

// An MoE's expert grid: the host launches the bound on the jobs (ops/moe max_jobs: a job is an
// expert's run of up to kExpertJobSlots of the chunk's slots, token x pick; at most one partial job
// an expert), ceil(slots / 64) + min(slots, experts) rows of tiles, before the routing is known.
inline constexpr double kExpertJobSlots = 64; // ops/moe kJobColumns at prefill widths

// The INT8-KV prompt kernel (ops/softmax_attention int8/prompt_fast.cuh, sm_80 on, a head width of
// 256 and int8 keys and values: Qwen 3.5's full layers): a CTA of 128 query rows of one head over
// every key before its last row, one CTA a SM (98 KiB of shared memory), issued longest first (the
// last row block of every head, then the one before) to the SM that frees first. Where 128-row CTAs
// would leave the last wave mostly idle the kernel lays the call out otherwise
// (int8_prompt_fast_layout): its first rows as 64-row CTAs issued last, or 64-row CTAs throughout,
// each at kAttentionHalfCta of a full one (the kernel's own figure). A CTA takes `cta` SM-clocks,
// `in_chunk` a key of its own chunk (the causal diagonal) and `prefix` a key before the chunk.
// Read on the census of a 21K-token prompt in 21 1024-token chunks (nsys; Blackwell on a lab build
// without PDL), each generation's cards together: RTX 5090 and 5070 Ti 195 / 152 and 20800 a CTA
// (-1 .. +2 %; the 5070 Ti's first chunk, 128 CTAs over 70 SMs, shows the CTA's own cost), RTX
// 4060 Ti and 4070 Ti (two 64-row slots) 328 / 188 (-8 .. +9 %: the 4060 Ti's first chunk +9 %,
// the 4070 Ti's -8 %, no CTA cost both agree on), RTX 3060 378 / 190 (-0.4 .. +0.8 %).
struct AttentionTileCost {
    std::string_view generation;
    double in_chunk, prefix; // SM-clocks a key of a CTA
    int cards;
    double cta = 0;          // SM-clocks a CTA besides its keys
};
inline constexpr AttentionTileCost kAttentionTileCost[] = {
    {"ada", 328, 188, 2}, {"ga10x", 378, 190, 1}, {"gb20x", 195, 152, 2, 20800}};
inline constexpr double kAttentionHalfCta = 0.72; // a 64-row CTA against a 128-row one
// The FP8-KV prompt kernel (ops/softmax_attention common/mxfp8_tiled_mma.cuh, `--kv-dtype fp8`, a
// call wider than kFp8AttentionTiledMinWidth): a CTA of 128 query rows of one head and one split of
// the call's keys (99 KiB, one a SM), the grid row tiles x heads x splits in launch order. The
// splits: mxfp8_tiled_partition's target (1 .. 8, the fewest waves a split) capped at one a 512 keys
// the call's last row sees; split s owns the 64-key tiles [s T / a, (s + 1) T / a) of those keys, a
// CTA the ones up to its last row. A CTA takes `cta` SM-clocks (its query rows' load and rotation,
// paid by a CTA with no keys too) and `key` a key. Read on the census of 7K and 21K prompts (1024-
// token chunks; Blackwell on a lab build without PDL), each generation's cards together: RTX 4060
// Ti and 4070 Ti 182 / 35000 (-3 .. +3 %), RTX 5090 and 5070 Ti 138 / 30000 (-6 .. +6 %: the 5090
// slower, the 5070 Ti faster than one pair gives), RTX 3060 284 / 75000 (-2 .. +3 %).
struct Fp8AttentionTileCost {
    std::string_view generation;
    double key, cta; // SM-clocks
    int cards;
};
inline constexpr Fp8AttentionTileCost kFp8AttentionTileCost[] = {
    {"ada", 182, 35000, 2}, {"ga10x", 284, 75000, 1}, {"gb20x", 138, 30000, 2}};
inline constexpr double kFp8AttentionTiledMinWidth = 80; // fp8/plan.cpp kGroupedPrefillMaxWidth
inline constexpr int kFp8AttentionMaxSplits        = 8;  // kMxfp8TiledMaxSplits
inline constexpr double kFp8AttentionSplitKeys     = 512; // the keys a split at the least
inline constexpr double kFp8AttentionKeyTile       = 64;
// The merge after every call reads each row's FP32 partial of every active split (a CTA with no
// keys stores an empty one) and writes the BF16 row: at 0.85 .. 1.04 of the DRAM rate where its
// bytes outgrow the board's L2 (the 21K census: RTX 4060 Ti 0.86, 4070 Ti 0.85, 3060 0.92, 5070 Ti
// 1.03), at 2.4 .. 4.7 of it where the L2 holds them (RTX 5090 2.4 .. 3.2 at 7K, its 21K prompt's
// late chunks 1.6; the 4070 Ti's and 5070 Ti's first chunk 4.7 / 3.2). A board without its L2 in
// the table (gpu_facts.json l2_mib) takes the DRAM rate.
inline constexpr double kAttentionMergeDramShare = 0.9;
inline constexpr double kAttentionMergeL2Share   = 3.0;
// The BF16-KV prompt kernel (ops/softmax_attention bf16/tiled_mma.cuh: `--kv-dtype bf16`): a CTA of 64
// query rows of one head (98 KiB, one a SM), issued in plain launch order (a head's row blocks
// first to last, then the next head). Read on the census of a 7K-token prompt (1024-token chunks):
// RTX 5090 144 / 120 (-1 .. +2 %), RTX 4060 Ti 168 / 144 (-2 .. +1 %), RTX 3060 168 / 142 (0 %);
// checked on the RTX 5070 Ti: 0 .. +1 %.
inline constexpr AttentionTileCost kBf16AttentionTileCost[] = {
    {"ada", 168, 144, 1}, {"ga10x", 168, 142, 1}, {"gb20x", 144, 120, 2}};
inline constexpr double kBf16AttentionQueryRows = 64;
inline constexpr double kAttentionQueryRows = 128; // a CTA's query rows (8 warps x 16)
inline constexpr double kAttentionTileHeadDim = 256;
// A call of at most this many tokens takes the grouped kernels instead (int8/plan.cpp
// kGroupedPrefillMaxWidth: the keys split over the SMs), priced by the share.
inline constexpr double kAttentionTiledMinWidth = 256;

// Gemma 4's attention kernel (ops/softmax_attention/gemma4, BF16 or FP8 planes; DECISIONS 68). A CTA
// is a row block of one KV head: 32 packed rows, 32 / group tokens (block_rows / G), the grid one
// block more than the rows need, issued longest first within a head. The sliding layers' D=256
// tile holds two CTAs a SM (40 KiB; Volta 44 with its scratch), over at most the window's keys; the full layers' D=512 tile
// (group 16: the 12B; a group-8 model's full layers from 1024 tokens take the wide tile, not read)
// one (80 KiB), over every key before its rows; where the generation chains (GenerationRow::
// attention_chain) a full layer's grid that fills its last wave under 90 % runs persistent, its work
// spread over the SMs. A CTA takes `sliding` / `full` SM-clocks a key, sharing its SM as it runs.
// Read on Gemma 4 12B Q4_K_M's 7K and 21K prompts (nsys, a layer's median a 1024-token chunk; the
// RTX 5090 on a lab build without PDL): RTX 4060 Ti 186 / 250 (sliding -3 .. +3 %, full -6 .. +8 %),
// RTX 3060 180 / 255 (-2 .. 0 %, -8 .. +7 %), RTX 5070 Ti and 5090 157 / 222 (-8 .. +7 %, -13 .. +9
// %); RTX 2080 Ti 430 / 575 (sliding -9 .. +4 %: its first chunk the -9 %, two CTAs a SM as on the
// others; full -2 .. +5 %; one card); Quadro GV100 615 / 805 (sliding -16 .. +3 %: its first chunk the
// -16 %, two CTAs a SM by registers and shared memory; full -1 .. +1 %; one card). A short last chunk whose last round holds one CTA a SM runs
// faster than this deals it (+14 .. +80 % there: the lone CTA of kLoneTile, not modelled here).
struct GemmaAttentionCost {
    std::string_view generation;
    double sliding, full; // SM-clocks a key of a CTA
    int cards;
};
inline constexpr double kGemmaChainMinTokens = 16; // gemma4/launch.h kGemma4DecodeMaxTokens: no chain
inline constexpr GemmaAttentionCost kGemmaAttentionCost[] = {
    {"ada", 186, 250, 1}, {"ga10x", 180, 255, 1}, {"gb20x", 157, 222, 2}, {"turing", 430, 575, 1},
    {"volta", 615, 805, 1}};

// The attention a prefill runs over its prompt (choice.cpp attention_seconds: the full layers over
// every earlier token, the sliding ones over the window) reaches this share of the BF16 tensor peak
// (fit_prefill.py, the same points; Turing's BF16 row is the composed aggregate, core/generation.h).
struct GenerationEta {
    std::string_view generation;
    double eta;
    int cards; // as PrefillEta::cards
};

// Turing's row: the census of the RTX 2080 Ti's prefill (int8_kv_tiled_mma_kernel: 111 ms of a 6.8K
// prompt, 11.5 ms of a 2K one: 0.167 / 0.143; one card); the former 0.069 was fitted together with
// the int8 row at 22K and priced the attention 2.4x slow while the linears ran fast.
// Volta's row: the same census (Quadro GV100), against the BF16 rate above: Gemma 4 12B's attention
// (gemma4_attention_kernel on the m8n8k4 FP16 MMA, 256- and 512-wide heads) 0.097 at 2863 tokens,
// Qwen3.5-9B's (sm70_d256 split kernels) 0.139. One share cannot hold both; the row takes Gemma's,
// where the attention is the larger part of a prefill (8 % against 2 %: Qwen's 2048-token prefill
// priced 0.6 % slow by it).
inline constexpr GenerationEta kPrefillAttentionEta[] = {
    {"ada", 0.742, 3}, {"gb20x", 0.740, 2}, {"turing", 0.160, 1}, {"ga10x", 0.497, 1},
    {"volta", 0.100, 1}};

// What the exact route's compute adds to a decode step beyond the fast route's: its rows' flops
// (C 2 P) at this share of the int8 tensor peak, its FP32 state's bytes apart (run_sw.sh, 9B at a
// 22K prompt, against the fast route at the same KV: exact decode 61.7 against 85.3 tok/s on the
// RTX 2080 Ti, 69.9 against 80.6 on the 4070 Ti, 127.9 against 150.2 on the 4090: 4.41, 1.79 and
// 1.11 ms a step, the FP32 state's extra bytes at the old 0.90 DRAM share; at kDramEta the three
// move by under 0.3 %). Ada: the mean of the 4070 Ti's 0.0272 and the 4090's 0.0214.
struct RouteEta {
    std::string_view generation;
    double eta;
};

inline constexpr RouteEta kExactComputeEta[] = {{"turing", 0.0165}, {"ada", 0.0243}};

// What reading a KV codec costs over its bytes (run_sw.sh, 9B at a 22K prompt, against INT8 on the
// same card): Turing reads BF16 and FP8 without native support. Ada: the mean of the RTX 4070 Ti
// and the 4090 (BF16 1.03 / 1.09, FP8 1.03 / 1.08, K8V4 1.01 / 1.04, NVFP4 1.25 / 1.22: within 6 %).
struct KvReadFactor {
    std::string_view generation;
    KvCacheStorage kv;
    double factor;
};

inline constexpr KvReadFactor kKvReadFactor[] = {
    {"turing", KvCacheStorage::BFloat16, 3.38},      {"turing", KvCacheStorage::Fp8E4M3Row256, 2.26},
    {"turing", KvCacheStorage::Fp8KeyNvfp4Value, 3.12}, {"turing", KvCacheStorage::Nvfp4Group16, 3.75},
    {"ada", KvCacheStorage::BFloat16, 1.06},         {"ada", KvCacheStorage::Fp8E4M3Row256, 1.055},
    {"ada", KvCacheStorage::Fp8KeyNvfp4Value, 1.025}, {"ada", KvCacheStorage::Nvfp4Group16, 1.235},
};

// A decode attention kernel whose KV read the SMs bound below the DRAM rate: the bytes it reads a
// SM-clock (at the board's reference clock), the family's kernel on the generation. A row's KV takes
// the longer of its time at the DRAM rate and at SMs x clock x this, in place of kKvReadFactor's
// codec factor (read on another family's kernel). The rate is per reference clock: a board holding
// another share of its reference clock than the 2080 Ti's 1.15 (a power-capped one) moves off it.
// One card: the rows feed the estimate a Turing or Volta board shows (transferred), never a decision
// (neither has a kReferenceCards row: their KV codec, activations and quantization keep the family's
// defaults).
// A row holds for the geometry it was read on (a full layer's head width, query and KV heads: the
// grouped kernels' rows and tiles follow the group); another geometry keeps the codec's factor. From the nsys census of one
// stream's decode at 7K and 82K (mdcen.sh, 2026-10-31, the RTX 2080 Ti: 68 SMs, 1545 MHz reference,
// 1770 held): the kernel's time per call over the KV bytes the 75K tokens between add, Qwen3.5 9B's
// int8 grouped kernel (int8_kv_grouped_mma) 423 us for 159 MB, Gemma 4 12B's BF16 full layer
// (gemma4_attention_kernel, 512-wide heads) 545 us for 153 MB: 376 and 281 GB/s, 61 and 46 % of the
// card's 616. The same kernels on the RTX 3060 (360 GB/s, 28 SMs) and the RTX 4070 Ti reach 92 .. 94 %
// of their DRAM rate: their generations have no row.
struct KvSmRate {
    std::string_view generation;
    std::string_view family;
    KvCacheStorage kv;
    double head_dim, heads, kv_heads; // a full layer's
    double bytes_per_sm_clock;
    int cards;
};

inline constexpr KvSmRate kKvSmRate[] = {
    {"turing", "qwen3_5_text", KvCacheStorage::Int8Group64, 256, 16, 4, 3.58, 1}, // Qwen3.5 9B
    {"turing", "gemma4_text", KvCacheStorage::BFloat16, 512, 16, 1, 2.67, 1},     // Gemma 4 12B
    // The Quadro GV100 (80 SMs, 1627 MHz reference; mdcenv.sh, the census at 7K and 82K): Gemma 4 12B's
    // BF16 full layer 707 us a call for 153 MB, 216 GB/s, 25 % of its 870 (Qwen3.5's int8 kernel,
    // int8_kv_key_major_kt, reaches 643 GB/s, 97 % of the 660 its HBM2 holds: no row).
    {"volta", "gemma4_text", KvCacheStorage::BFloat16, 512, 16, 1, 1.66, 1},
};

// The prefill chunk. An MoE's prefill grows with the tokens each expert takes from a chunk (chunk x
// experts per token / experts), one curve across models and cards: its share of the saturated rate
// (run_chunk.sh at a 22K prompt, 2026-10-23/24; decode unchanged at every chunk):
//   tokens per expert       8      16     32     64     128
//   Gemma 4 26B, 4070 Ti    -      0.617  0.788  0.939  0.999   (4314 ... 6994 tok/s)
//   Gemma 4 26B, 4090       -      0.664  0.848  0.984  1.0     (8645 ... 13026)
//   Qwen3.6 35B, 4090       0.409  0.625  0.802  0.932  1.0     (6287 ... 15389)
// The mean below; the three within 6 % of it. kPrefillEta's MoE rows are the saturated rate's. A dense model's prefill moves under 5 % from 512 tokens on (Gemma 4 12B, Qwen3.8
// 27B): its chunk stays the default.
struct ChunkShare {
    double tokens_per_expert;
    double share;
};

inline constexpr ChunkShare kMoeChunkShare[] = {
    {8, 0.409}, {16, 0.635}, {32, 0.813}, {64, 0.952}, {128, 1.0}};
// The shares decide an MoE's chunk on a generation with a reference card (by speed alone: a chunk has
// no accuracy term, so the accuracy terms' cards do not hold it).
inline constexpr bool kMoeChunkTwoCards = true; // RTX 4070 Ti and 4090
inline constexpr std::uint32_t kDefaultChunk = 1024;
inline constexpr std::uint32_t kMoeChunks[] = {1024, 2048, 4096};

// ---- Accuracy.

enum class Axis : std::uint8_t { Quant, Kv, Ring, State, Activations, Chunk, Drafter };

[[nodiscard]] constexpr std::string_view axis_name(Axis axis) noexcept {
    switch (axis) {
    case Axis::Quant: return "quant";
    case Axis::Kv: return "kv";
    case Axis::Ring: return "ring";
    case Axis::State: return "state";
    case Axis::Activations: return "activations";
    case Axis::Chunk: return "chunk";
    case Axis::Drafter: return "drafter";
    }
    return "?";
}

// One measured switch: its KLD and top-1 agreement against the exact reference
// (ninferno-perplexity --quick: every call a prefill of 17 tokens or more), on `generation` ("": a
// model constant, the same on each generation it was measured on), and the lab cards behind it. A
// term on an axis the worth decides (KV, activations) decides only on two cards (principle 4);
// the coupled storages' terms (state, ring) only price the candidates they follow.
struct KldTerm {
    std::string_view family; // the model_type it was measured on
    Axis axis;
    std::string_view value; // the flag's spelling (--kv-dtype int8, --state-dtype f16, ...)
    std::string_view generation;
    double kld;
    double top1;
    int cards;
    std::string_view source;
};

// The family a model without terms of its own borrows them from, for the estimates shown only (the
// worth never decides on borrowed terms: each family's switches move it differently, Qwen 3.5 MoE's
// INT8 KV costs 3.4 times dense's).
inline constexpr std::string_view kKldDefaultFamily = "qwen3_5_text";
// The family the quantization curve was measured on: the worth weighs the quantizations for it;
// another family the switches decide for keeps the quant rule's largest.
inline constexpr std::string_view kQuantKldFamily = "qwen3_5_text";

inline constexpr KldTerm kKldTerms[] = {
    // Qwen 3.5 dense (Qwen3.5-9B Q4_K_M). The KV codecs: a model constant. RTX 5070 Ti (2026-10-23)
    // and RTX 4070 Ti (2026-10-24), the mean: int8 0.00099 / 0.00092, fp8 0.00300 / 0.00260, k8v4
    // 0.00532 / 0.00535, nvfp4 0.00981 / 0.00983 (within 13 %, the order the same).
    {"qwen3_5_text", Axis::Kv, "int8", "", 0.00095, 0.9914, 2, "Qwen3.5-9B Q4_K_M, RTX 5070 Ti and RTX 4070 Ti"},
    {"qwen3_5_text", Axis::Kv, "fp8", "", 0.00280, 0.9861, 2, "Qwen3.5-9B Q4_K_M, RTX 5070 Ti and RTX 4070 Ti"},
    {"qwen3_5_text", Axis::Kv, "k8v4", "", 0.00534, 0.9777, 2, "Qwen3.5-9B Q4_K_M, RTX 5070 Ti and RTX 4070 Ti"},
    {"qwen3_5_text", Axis::Kv, "nvfp4", "", 0.00982, 0.9710, 2, "Qwen3.5-9B Q4_K_M, RTX 5070 Ti and RTX 4070 Ti"},
    // The int8 activations (fast = int8 on these weights): a model constant once ggml's MMQ
    // prefill sums its q8_1 blocks from the codes (ops/generic/ggml_mmq/ggml_mmq.cu). Before that
    // fix the boards that prefill on MMQ (RTX 4090, 4070 Ti, 2080 Ti) lost 7.6 times as much on
    // the min-term formats (Q4_K_M 0.0302 / 0.0294 / 0.0297, top-1 95.8 %); with it the RTX 4090
    // 0.00400 (98.44 %), the RTX 5070 Ti (W4A8 tiles) 0.00388 (98.51 %); Q6_K 0.00394 on the 4090,
    // Q8_0 0.0051 / 0.0049 on the 4090 / 5070 Ti.
    {"qwen3_5_text", Axis::Activations, "int8", "", 0.00394, 0.9848, 2, "Qwen3.5-9B Q4_K_M, RTX 5070 Ti and RTX 4090 (code-sum MMQ), 2026-10-24"},
    // The state's own share under fast: the root of (fast with the state)^2 - (fast with FP32)^2,
    // 0.003981 and 0.004220 against 0.00388 (top-1 98.49 % and 98.47 %). On Ada under its 0.0294
    // fast term the share is within the run's noise (f16 0.0015, bf16 below zero).
    {"qwen3_5_text", Axis::State, "f16", "", 0.00089, 0.9849, 1, "Qwen3.5-9B Q4_K_M under fast, RTX 5070 Ti, 2026-10-23"},
    {"qwen3_5_text", Axis::State, "bf16", "", 0.00166, 0.9847, 1, "Qwen3.5-9B Q4_K_M under fast, RTX 5070 Ti, 2026-10-23"},
    // Qwen 3.5 MoE (Qwen3.6-35B-A3B UD-IQ3_XXS, with the MMQ fix), RTX 4090 and RTX 5070 Ti
    // (2026-10-24), the mean: int8 0.00320 / 0.00338, fp8 0.00427 / 0.00426, k8v4 0.00695 / 0.00701,
    // nvfp4 0.00923 / 0.00936, fast 0.00496 / 0.00484 (within 6 %); the state's share under fast
    // 0.00117 / 0.00135.
    {"qwen3_5_moe_text", Axis::Kv, "int8", "", 0.00329, 0.9786, 2, "Qwen3.6-35B-A3B UD-IQ3_XXS, RTX 4090 and RTX 5070 Ti"},
    {"qwen3_5_moe_text", Axis::Kv, "fp8", "", 0.00426, 0.9751, 2, "Qwen3.6-35B-A3B UD-IQ3_XXS, RTX 4090 and RTX 5070 Ti"},
    {"qwen3_5_moe_text", Axis::Kv, "k8v4", "", 0.00698, 0.9665, 2, "Qwen3.6-35B-A3B UD-IQ3_XXS, RTX 4090 and RTX 5070 Ti"},
    {"qwen3_5_moe_text", Axis::Kv, "nvfp4", "", 0.00930, 0.9608, 2, "Qwen3.6-35B-A3B UD-IQ3_XXS, RTX 4090 and RTX 5070 Ti"},
    {"qwen3_5_moe_text", Axis::Activations, "int8", "", 0.00490, 0.9735, 2, "Qwen3.6-35B-A3B UD-IQ3_XXS, RTX 4090 and RTX 5070 Ti (code-sum MMQ)"},
    {"qwen3_5_moe_text", Axis::State, "f16", "", 0.00126, 0.9737, 2, "Qwen3.6-35B-A3B UD-IQ3_XXS under fast, RTX 4090 and RTX 5070 Ti"},
    // Dense decoders (Qwen3-8B Q4_K_M, causal scoring of the dense family, det/score-all-families),
    // two Ada cards, RTX 4070 Ti and RTX 4090, and the RTX 5070 Ti (2026-10-24/25): FP8 KV 0.004947
    // on both Ada cards (the exact route's bytes), 0.004945 on Blackwell (top-1 97.54 / 97.58 %);
    // fast 0.001254 / 0.001236 on Ada, 0.001215 on Blackwell (98.58 / 98.61 %): model constants
    // (within 3 %), on two cards of the generation that decides (Ada). Together 0.00546 /
    // 0.00540 against the root sum square 0.0051: the switches combine as the Qwen terms do. No
    // sliding window here: FP8 rings of a windowed dense model (gpt-oss) are not covered by these.
    {"dense_decoder", Axis::Kv, "fp8", "", 0.00495, 0.9756, 2, "Qwen3-8B Q4_K_M, RTX 4070 Ti and RTX 4090 (and RTX 5070 Ti), 2026-10-25"},
    {"dense_decoder", Axis::Activations, "int8", "", 0.00124, 0.9860, 2, "Qwen3-8B Q4_K_M, RTX 4070 Ti and RTX 4090 (and RTX 5070 Ti), 2026-10-25"},
};

// The quantization's own term: KLD against Q8_0 under the exact reference route (ninferno-perplexity
// --quick, the same corpus), by bits per weight. Another quantization or model takes the
// interpolation in log KLD over bits per weight between the two nearest points (below them, the
// lowest pair's slope; above Q6_K, a straight line to 0 at Q8_0's bits), marked as not measured; at
// or above Q8_0's bits, and any model's Q8_0, it is the reference (0).
struct QuantKldPoint {
    double bits_per_weight;
    double kld;
    std::string_view label;
    int cards; // lab cards the point is measured on; below kQuantKldCards it informs, never decides
};

inline constexpr std::string_view kQuantKldSource =
    "Qwen3.5-9B GGUF against Q8_0, RTX 5070 Ti (2026-10-23) and RTX 4070 Ti (2026-10-24), the mean; "
    "UD-Q5_K_XL, Q5_K_S, UD-Q4_K_XL, Q4_K_S and IQ4_NL on the RTX 4060 Ti and RTX 4070 Ti (2026-10-29)";
inline constexpr int kQuantKldCards = 2;
// Bits per weight as the plan counts them: the weight bytes over the GGUF tensor table's
// elements (8 953 803 264 for Qwen3.5-9B); the two cards'
// KLD (within 2 % at every point, the same file's identical to six digits on the RTX 4060 Ti and
// 4070 Ti against each card's own reference: the exact route's bytes do not depend on the card;
// unsloth's MTP repository quantizes UD-Q4_K_XL / IQ4_NL differently, 0.0178 / 0.0259), perplexity
// and top-1 agreement beside (perplexity alone misorders: Q3_K_M 5.029 below IQ4_XS 5.073, KLD
// 0.068 above 0.028).
inline constexpr QuantKldPoint kQuantKld[] = {
    {8.516, 0.00000, "Q8_0", 2},       // ppl 4.9651, the reference
    {6.665, 0.00629, "Q6_K", 2},       // 0.00623 / 0.00635, ppl 4.9714, top-1 0.9765
    {6.016, 0.00940, "UD-Q5_K_XL", 2}, // 0.009399 on both (29.10); the curve gives 0.00985 at its bits
    {5.878, 0.01061, "Q5_K_M", 2},     // 0.01058 / 0.01063, ppl 4.9665, top-1 0.9686
    {5.674, 0.01103, "Q5_K_S", 2},     // 0.011029 on both (29.10), top-1 0.9670
    {5.321, 0.01828, "UD-Q4_K_XL", 2}, // 0.018277 on both (29.10), top-1 0.9559
    {5.080, 0.01925, "Q4_K_M", 2},     // 0.01930 / 0.01920 (0.019188 on both, 29.10), ppl 4.9997, top-1 0.9512
    {4.810, 0.02248, "Q4_K_S", 2},     // 0.022480 on both (29.10), top-1 0.9462
    {4.789, 0.02867, "IQ4_NL", 2},     // 0.028670 on both (29.10), top-1 0.9408
    {4.617, 0.02845, "IQ4_XS", 2},     // 0.02831 / 0.02859, ppl 5.0727, top-1 0.9407
    {4.174, 0.06812, "Q3_K_M", 2},     // 0.06796 / 0.06827, ppl 5.0292, top-1 0.9040
    {3.584, 0.10710, "UD-IQ3_XXS", 2}, // 0.10682 / 0.10737, ppl 5.2870, top-1 0.8733
    {3.259, 0.16145, "UD-IQ2_M", 2},   // 0.16138 / 0.16153, ppl 5.5968, top-1 0.8420
    {2.855, 0.33460, "UD-IQ2_XXS", 2}, // 0.33445 / 0.33475, ppl 6.4693, top-1 0.7684
};

// ---- The trade.

inline constexpr double kAccuracyWeight      = 25.0; // lambda: 0.001 KLD = 2.5 % of a request's time
inline constexpr std::uint32_t kRequestPrompt = 2048;
inline constexpr std::uint32_t kRequestAnswer = 512;
// No fixed time per request: a mixed workload (8 sampled example tasks, 96-512 answer tokens) left
// 0.135 s a request unexplained on the RTX 4090's host and none on the 4070 Ti's (with it the 4070
// Ti's estimates went from +3 % to +8 %): it is the host's (launch and HTTP latency), which no GPU
// fact predicts. Measured against that workload, one stream: the 4070 Ti within 5 % (Gemma 4 12B,
// a family the fit never saw, +5 %); the 4090 2-7 % fast on the larger models and 15 % fast on the
// fastest (Qwen3.5-4B, Qwen3.6-35B-A3B: 240-250 tok/s). Turing decodes the IQ2 codebook formats
// 19 % slower than estimated (Qwen3.8-27B UD-IQ2_XXS on the RTX 2080 Ti; the same file on the 4070
// Ti within 3 %): the estimate shown there is optimistic, and Turing does not decide.
// Several streams: the compute of a step of C rows depends on the weight format beyond its bytes
// (a codebook format's decode is cheaper per row than its bits suggest): at 4-8 streams the fit
// leaves Qwen3.8-27B UD-IQ3_XXS on the RTX 5070 Ti 14-15 % slow and UD-Q4_K_XL on the 4090 3-10 %
// fast; Qwen3.5-9B Q4_K_M stays within 4 % at 2-4 streams. Not the format alone: Qwen3.5-9B in
// eight formats (Q8_0 ... UD-IQ2_XXS, RTX 5070 Ti, 1-8 streams, run_fmt 2026-10-24) stays within
// 10 % at every stream count (UD-IQ3_XXS 0.93 at 8), so no per-format factor.

// ---- What a decision may read (docs/maintainer/deterministic-decisions.md).
//
// The KV codec and the activation precision change the output bytes (tier B1): they read only
// discrete facts, the artifact and the options, never a board's clock or bandwidth. So the choice
// prices its candidates on the generation's reference card below, never on the board it runs on:
// every board of a generation decides the same, and a board's own facts only feed the estimates it
// shows (speed_of on CardFacts). A reference card is the mean of the generation's lab cards (SMs,
// reference clock, DRAM bytes per SM and clock). Its constants decide only where they rest on two
// lab cards of the generation (principle 4): the KV read factors and the exact-route factor each
// name whether they do. A generation without such a row, or whose factors rest on one card, keeps
// the family's defaults (the engine's automatic KV codec, --activations fast).
struct ReferenceCard {
    std::string_view generation;
    int sms;
    double clock_hz;
    double dram_bytes_per_s;
    bool kv_two_cards;    // kKvReadFactor's rows of the generation fitted on two lab cards
    bool exact_two_cards; // kExactComputeEta's row likewise
    std::string_view cards;
};

inline constexpr ReferenceCard kReferenceCards[] = {
    // RTX 4070 Ti (60 SMs, 2610 MHz, 504 GB/s: 3.22 B a SM-clock) and RTX 4090 (128, 2520,
    // 1008: 3.13); the KV and exact factors measured on both (2026-10-23).
    {"ada", 94, 2.565e9, 3.17 * 94 * 2.565e9, true, true, "RTX 4070 Ti, RTX 4090"},
};

// Two configurations whose worth differs by less than this are a tie (then: the longer own context,
// then the fewer bytes of weights, then the smaller quantization label, then the listing's order).
// Between quantizations the band is wider: kDecodePredictionError (the lower KLD wins inside it).
inline constexpr double kWorthTie = 1e-9;

// With the quantization curve, a generation that decides weighs the quantizations by worth too; one
// that does not keeps the quant rule's pick (runtime/engine/quant_rule.h: the largest that fits, not
// below its floor bits unless none does).

} // namespace ninferno::product::choice
