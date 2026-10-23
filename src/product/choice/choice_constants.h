#pragma once

// The constants of the configuration choice (choice.h), each with where it comes from. The one
// place they live: a new measurement changes a row here, nothing else.
//
// Speed. A decode step of C rows at context n takes
//   t0 + (W(C) + C n b_kv k_kv + 2 C s + C min(n, window) b_ring) / (eta_dram BW)
//      + C 2 P / (eta_compute peak_s8) f_route
// and a prefill runs eta_prefill peak_route / (2 P) tokens a second: W(C) the weights C rows read
// (an MoE's routed experts as rows routed apart add them, kMoeRoutedShare of independent routing),
// b_kv the full layers' KV bytes a token, s a lane's recurrent state (read and written), b_ring a
// sliding ring's bytes a token, P the active parameters (plan_facts.cpp: the layers a token runs
// and the output head), BW the board's DRAM rate, peak the generation row's tensor rate times SMs
// times the reference clock (core/generation.h). k_kv and f_route are what a KV codec's read and
// the exact route cost on a generation beyond their bytes and flops.
//
// Fitted (scratchpad fit_choice.py, 2026-10-23) on 49 decode and 15 prefill points of Qwen3.5-9B
// Q4_K_M, Qwen3.8-27B UD-IQ3_XXS / UD-Q4_K_XL and Qwen3.6-35B-A3B UD-IQ3_XXS / UD-Q4_K_M on the
// RTX 2080 Ti, 4070 Ti, 4090 and 5070 Ti (run_calib.sh, run_sw.sh, docs/performance/runs):
// decode 5.2 % rms (worst 15.5 %), prefill 2.0 % rms (worst 5.4 %), in sample.
//
// Accuracy. KLD against the exact reference (--activations exact, BF16 KV, FP32 state) on the
// fixed corpus (ninferno-perplexity --quick, eval/corpora/perplexity-1m), one term per switch;
// terms combine as the root of the sum of their squares (the product default measured 0.0040,
// the root sum square of its terms 0.0044, their plain sum 0.0067).
//
// The trade. One number, kAccuracyWeight: a configuration's worth is -ln(T) - lambda KLD, T the
// seconds of a typical request (kRequestPrompt tokens in, kRequestAnswer out). lambda 25 makes
// 0.001 KLD worth 2.5 % of the request's time: where two configurations cost about the same, the
// more accurate wins; where one is much faster, it wins. The exact route (0.0040 KLD better than
// the product default on 9B) thus wins only where it costs under about 11 % of a request's time
// (it costs 26 % on the RTX 4070 Ti, 38 % on the 5070 Ti at 2048 + 512 tokens). At lambda 50 the
// 4070 Ti's two were within 0.01 of each other: no clear "about the same".

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

struct DecodeConstants {
    double eta_dram     = 0; // the share of the DRAM rate a step's reads reach
    double eta_compute  = 0; // the share of the int8 tensor peak a step's flops reach
    double step_seconds = 0; // a step's fixed time (launches, routing)
};

inline constexpr DecodeConstants kDenseDecode{0.90, 0.054, 0.0};
inline constexpr DecodeConstants kMoeDecode{0.72, 0.022, 0.0008};

// spec/load_model.h: rows routed apart read 0.30 of the experts independent routing would add.
inline constexpr double kMoeRoutedShare = 0.30;

// A row per generation (core/generation.h's names) and kind; a generation without its own row
// takes kTransferGeneration's, marked as a transfer.
inline constexpr std::string_view kTransferGeneration = "ada";

struct PrefillEta {
    bool moe;
    std::string_view generation;
    Route route;
    double eta;
};

inline constexpr PrefillEta kPrefillEta[] = {
    {false, "turing", Route::Int8, 0.175}, // RTX 2080 Ti, 9B Q4_K_M, 22K prompt
    {false, "turing", Route::Bf16, 0.067}, // RTX 2080 Ti, 9B exact
    {false, "ada", Route::Int8, 0.270},    // RTX 4070 Ti 9B, RTX 4090 27B
    {false, "ada", Route::Bf16, 0.305},    // RTX 4070 Ti, 9B exact
    {false, "gb20x", Route::Int8, 0.296},  // RTX 5070 Ti 9B, 27B
    {true, "ada", Route::Int8, 0.118},     // RTX 4090, 35B-A3B
    {true, "gb20x", Route::Int8, 0.118},   // RTX 5070 Ti, 35B-A3B
};

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

// ---- Accuracy.

enum class Axis : std::uint8_t { Quant, Kv, Ring, State, Activations };

[[nodiscard]] constexpr std::string_view axis_name(Axis axis) noexcept {
    switch (axis) {
    case Axis::Quant: return "quant";
    case Axis::Kv: return "kv";
    case Axis::Ring: return "ring";
    case Axis::State: return "state";
    case Axis::Activations: return "activations";
    }
    return "?";
}

// One measured switch: its KLD and top-1 agreement against the exact reference.
struct KldTerm {
    Axis axis;
    std::string_view value; // the flag's spelling (--kv-dtype int8, --state-dtype f16, ...)
    double kld;
    double top1;
    std::string_view source;
};

inline constexpr std::string_view kKldFamily = "qwen3_5_text"; // the family the terms were measured on

inline constexpr KldTerm kKldTerms[] = {
    {Axis::Kv, "int8", 0.00099, 0.9914, "Qwen3.5-9B Q4_K_M, RTX 5070 Ti, 2026-10-23"},
    {Axis::Kv, "fp8", 0.00300, 0.9861, "Qwen3.5-9B Q4_K_M, RTX 5070 Ti, 2026-10-23"},
    {Axis::Kv, "k8v4", 0.00532, 0.9777, "Qwen3.5-9B Q4_K_M, RTX 5070 Ti, 2026-10-23"},
    {Axis::Kv, "nvfp4", 0.00981, 0.9710, "Qwen3.5-9B Q4_K_M, RTX 5070 Ti, 2026-10-23"},
    {Axis::Activations, "int8", 0.00388, 0.9851, "Qwen3.5-9B Q4_K_M, RTX 5070 Ti, 2026-10-23 (fast = int8)"},
    {Axis::State, "f16", 0.0018, 0.981, "Qwen3.8-27B 16K stepwise decode (docs/reference/serving.md)"},
    {Axis::State, "bf16", 0.0052, 0.970, "Qwen3.8-27B 16K stepwise decode (docs/reference/serving.md)"},
};

// The quantization's own term: KLD against Q8_0 under the exact reference route (ninferno-perplexity
// --quick, the same corpus), by bits per weight. Another quantization or model takes the
// interpolation in log KLD over bits per weight between the two nearest points (past the ends, the
// nearest pair's slope), marked as not measured; at or above Q8_0's bits it is the reference (0).
// Empty: no curve, and the quantization is the quant rule's pick.
struct QuantKldPoint {
    double bits_per_weight;
    double kld;
    std::string_view label;
};

inline constexpr std::string_view kQuantKldSource = "";
inline constexpr QuantKldPoint kQuantKld[] = {{0, 0, ""}}; // filled from the measured curve

// ---- The trade.

inline constexpr double kAccuracyWeight      = 25.0; // lambda: 0.001 KLD = 2.5 % of a request's time
inline constexpr std::uint32_t kRequestPrompt = 2048;
inline constexpr std::uint32_t kRequestAnswer = 512;
// A request's fixed time beside its prefill and decode (HTTP, the template, tokenization, the first
// step's setup): fitted on the mixed workload of the RTX 4090 (15 runs of 8 sampled tasks: 9.4 % ->
// 3.6 % rms). One card: it enters the estimates shown, never a decision (principle 4).
inline constexpr double kRequestFixedSeconds = 0.135;

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
// then the fewer bytes of weights, then the listing's order).
inline constexpr double kWorthTie = 1e-9;

// Until a measured KLD curve over the quantizations is wired in, the quantization is the rule's
// pick (runtime/engine/quant_rule.h: the largest that fits, not below its floor bits unless none
// does); the other axes are chosen by worth at that quantization.

} // namespace ninferno::product::choice
