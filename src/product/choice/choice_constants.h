#pragma once

// The constants of the configuration choice (choice.h), each with where it comes from. The one
// place they live: a new measurement changes a row here, nothing else.
//
// Speed. A decode step of C rows at context n takes
//   t0 + (W(C) + C n b_kv k_kv + 2 C s + C min(n, window) b_ring) / (eta_dram BW)
//      + C 2 P / (eta_compute peak_s8) f_route
// and a prefill of n tokens takes n 2 P / (eta_prefill share peak_route) + F_att(n) / (eta_attention
// peak_bf16) seconds (choice.cpp attention_seconds, linear_prefill): W(C) the weights C rows read
// (an MoE's routed experts as rows routed apart add them, kMoeRoutedShare of independent routing),
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
#include "spec/moe_routed_share.h"
#include "ninferno/types.h"

#include <cstdint>
#include <string_view>

namespace ninferno::product::choice {

using ::ninferno::spec::kMoeRoutedShare; // spec/moe_routed_share.h, a fitted calibration

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

struct DecodeConstants {
    double eta_dram     = 0; // the share of the DRAM rate a step's reads reach
    double eta_compute  = 0; // the share of the int8 tensor peak a step's flops reach
    double step_seconds = 0; // a step's fixed time (launches, routing)
};

inline constexpr DecodeConstants kDenseDecode{0.90, 0.054, 0.0};
inline constexpr DecodeConstants kMoeDecode{0.72, 0.022, 0.0008};

// kDp4aOpsPerSmClock (a generation without an int8 MMA decodes its int8 route at it, choice.cpp
// decode_int8_peak): core/cost_constants.h, a chip fact; kMoeRoutedShare: spec/moe_routed_share.h,
// a fitted calibration. Both shared with spec/round_cost.h.

// A row per generation (core/generation.h's names) and kind; a generation without its own row
// takes kTransferGeneration's, marked as a transfer (its prefill: a generation's with the same BF16
// tensor rate per SM where one has rows, choice.cpp prefill_stand_in).
inline constexpr std::string_view kTransferGeneration = "ada";

struct PrefillEta {
    bool moe;
    std::string_view generation;
    Route route;
    double eta;
};

// The linears' share of the route's peak (fit_prefill.py). The exact rows (BF16) were fitted on 9B
// at a 22K prompt without the attention term; they keep that fit's time with attention now apart.
inline constexpr PrefillEta kPrefillEta[] = {
    {false, "turing", Route::Int8, 0.241}, // RTX 2080 Ti: 9B, Gemma 12B (one card)
    {false, "turing", Route::Bf16, 0.073}, // RTX 2080 Ti, 9B exact
    {false, "ada", Route::Int8, 0.306},    // RTX 4060 Ti, 4070 Ti, 4090: 9B, 27B, Gemma 12B, 31B
    {false, "ada", Route::Bf16, 0.317},    // RTX 4070 Ti, 9B exact
    {false, "gb20x", Route::Int8, 0.324},  // RTX 5070 Ti, 5090: 9B, 27B, Gemma 12B
    {false, "ga10x", Route::Int8, 0.363},  // RTX 3060: 9B, Gemma 12B (one card)
    {true, "ada", Route::Int8, 0.210},     // RTX 4060 Ti, 4090: Gemma 26B-A4B, 35B-A3B
    {true, "gb20x", Route::Int8, 0.205},   // RTX 5070 Ti, 5090: Gemma 26B-A4B, 35B-A3B
};

// The attention a prefill runs over its prompt (choice.cpp attention_seconds: the full layers over
// every earlier token, the sliding ones over the window) reaches this share of the BF16 tensor peak
// (fit_prefill.py, the same points; Turing's BF16 row is the composed aggregate, core/generation.h).
struct GenerationEta {
    std::string_view generation;
    double eta;
};

inline constexpr GenerationEta kPrefillAttentionEta[] = {
    {"ada", 0.742}, {"gb20x", 0.740}, {"turing", 0.069}, {"ga10x", 0.497}};

// What the exact route's compute costs a decode step over the int8 route's flops at the int8 peak,
// its FP32 state's bytes apart (run_sw.sh, 9B at a 22K prompt: exact decode 61.7 against 85.3
// tok/s on the RTX 2080 Ti; 69.9 against 80.6 on the 4070 Ti, 127.9 against 150.2 on the 4090).
// Ada: the mean of the 4070 Ti's 2.98 and the 4090's 3.52, the two within 20 %.
struct RouteFactor {
    std::string_view generation;
    double factor;
};

inline constexpr RouteFactor kExactComputeFactor[] = {{"turing", 4.28}, {"ada", 3.25}};

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
};

inline constexpr std::string_view kQuantKldSource =
    "Qwen3.5-9B GGUF against Q8_0, RTX 5070 Ti (2026-10-23) and RTX 4070 Ti (2026-10-24), the mean";
inline constexpr int kQuantKldCards = 2;
// Bits per weight as the plan counts them: the weight bytes over the GGUF tensor table's
// elements (8 953 803 264 for Qwen3.5-9B); the two cards'
// KLD (within 2 % at every point: the exact route's bytes do not depend on the card), perplexity
// and top-1 agreement beside (perplexity alone misorders: Q3_K_M 5.029 below IQ4_XS 5.073, KLD
// 0.068 above 0.028).
inline constexpr QuantKldPoint kQuantKld[] = {
    {8.516, 0.00000, "Q8_0"},       // ppl 4.9651, the reference
    {6.665, 0.00629, "Q6_K"},       // 0.00623 / 0.00635, ppl 4.9714, top-1 0.9765
    {5.878, 0.01061, "Q5_K_M"},     // 0.01058 / 0.01063, ppl 4.9665, top-1 0.9686
    {5.080, 0.01925, "Q4_K_M"},     // 0.01930 / 0.01920, ppl 4.9997, top-1 0.9512
    {4.617, 0.02845, "IQ4_XS"},     // 0.02831 / 0.02859, ppl 5.0727, top-1 0.9407
    {4.174, 0.06812, "Q3_K_M"},     // 0.06796 / 0.06827, ppl 5.0292, top-1 0.9040
    {3.584, 0.10710, "UD-IQ3_XXS"}, // 0.10682 / 0.10737, ppl 5.2870, top-1 0.8733
    {3.259, 0.16145, "UD-IQ2_M"},   // 0.16138 / 0.16153, ppl 5.5968, top-1 0.8420
    {2.855, 0.33460, "UD-IQ2_XXS"}, // 0.33445 / 0.33475, ppl 6.4693, top-1 0.7684
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
    bool exact_two_cards; // kExactComputeFactor's row likewise
    std::string_view cards;
};

inline constexpr ReferenceCard kReferenceCards[] = {
    // RTX 4070 Ti (60 SMs, 2610 MHz, 504 GB/s: 3.22 B a SM-clock) and RTX 4090 (128, 2520,
    // 1008: 3.13); the KV and exact factors measured on both (2026-10-23).
    {"ada", 94, 2.565e9, 3.17 * 94 * 2.565e9, true, true, "RTX 4070 Ti, RTX 4090"},
};

// Two configurations whose worth differs by less than this are a tie (then: the longer own context,
// then the fewer bytes of weights, then the smaller quantization label, then the listing's order).
inline constexpr double kWorthTie = 1e-9;

// With the quantization curve, a generation that decides weighs the quantizations by worth too; one
// that does not keeps the quant rule's pick (runtime/engine/quant_rule.h: the largest that fits, not
// below its floor bits unless none does).

} // namespace ninferno::product::choice
