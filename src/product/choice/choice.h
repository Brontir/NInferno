#pragma once

// The configuration choice: one rule for every place that picks how a model is served when the
// user leaves it to the engine (ninferno-serve without the flags, --plan --no-load, ninferno-ui's
// /api/plan). From facts only (the model's shape, the card's rates, the user's task), no
// measurement on the user's machine:
//
//   memory     the own context each stream holds at a configuration (the plan without the
//              weights: runtime/engine/plan_dry_run.h, through OwnContextFn);
//   speed      the seconds of a typical request at it (decode and prefill, choice_constants.h);
//   accuracy   its KLD against the exact reference (choice_constants.h's measured terms).
//
// The candidates are the quantizations x KV codecs x activation modes the model, the card and the
// user's pins allow; the ring and the recurrent state follow the KV codec and the activations
// (coupled_storages). A candidate fits when each stream holds the asked context of its own; among
// those that fit, the largest worth -ln(T) - lambda KLD wins (choice_constants.h: the trade). The
// result keeps every candidate scored, so a caller shows what each other value of an axis would
// change (explain).
//
// Pure: no CUDA, no files, no clock.

#include "product/choice/choice_constants.h"

#include "ninferno/types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::product::choice {

// ---- Facts.

struct CardFacts {
    int major = 0, minor = 0;   // compute capability
    int sms = 0;                // multiprocessors
    double clock_hz = 0;        // reference SM clock
    double dram_bytes_per_s = 0;
    bool fp4 = false;           // FP4 tensor cores (NVFP4 weights run W4A4)
};

struct ModelFacts {
    std::string family;               // model_type (qwen3_5_text, gemma4_text, ...)
    double params = 0;                // every weight
    double active_params = 0;         // the layers a token runs and the output head
    int experts = 0, experts_per_token = 0;
    double routed_expert_params = 0;  // all routed experts' weights (0: dense)
    double input_table_params = 0;    // an untied input embedding table (a step reads one row; 0 when tied)
    // Per-token and per-lane bytes of the state a step reads, by storage
    // (runtime::plan_kv_bytes_per_token, plan_ring_bytes_per_token, plan_state_bytes).
    std::function<std::uint64_t(KvCacheStorage)> kv_bytes_per_token;
    std::function<std::uint64_t(KvCacheStorage)> ring_bytes_per_token; // 0: no sliding layers
    std::function<std::uint64_t(RecurrentStateStorage)> state_bytes;    // 0: no recurrent layers
    std::uint32_t window = 0;         // the sliding window (0: none)
    bool recurrent = false;           // Gated DeltaNet layers (a --state-dtype to choose)
    [[nodiscard]] bool moe() const noexcept { return experts > 1 && routed_expert_params > 0; }
};

// One quantization of the model (a listing row).
struct QuantOption {
    std::string label;          // as --quant takes it
    double bits_per_weight = 0;
    double weight_bytes    = 0; // what the Device holds
    bool supported         = true;
    bool int8_route        = true;  // its linears take int8 activations (every GGUF format: AllowA8)
    bool nvfp4             = false; // NVFP4 weights
};

// What the user asks: the context each stream holds of its own, the streams, the request shape.
struct Task {
    std::uint32_t context     = 32768;
    std::uint32_t concurrency = 1;
    std::uint32_t prompt      = kRequestPrompt;
    std::uint32_t answer      = kRequestAnswer;
    double accuracy_weight    = kAccuracyWeight;
};

// What the user pinned with flags; the choice keeps it.
struct Pins {
    std::optional<std::size_t> quant; // an index into the options
    std::optional<KvCacheStorage> kv;
    std::optional<KvCacheStorage> ring;
    std::optional<RecurrentStateStorage> state;
    std::optional<ActivationMode> activations;
};

// The default storages the engine couples to the KV codec and the activations: the sliding rings
// in BF16 (types.h sliding_kv_cache) and the recurrent state in FP16 under fast, FP32 under exact
// (model_instance.cpp). Pinned values stay.
struct Coupled {
    KvCacheStorage ring;
    RecurrentStateStorage state;
};
[[nodiscard]] Coupled coupled_storages(const ModelFacts& model, KvCacheStorage kv,
                                       ActivationMode activations, const Pins& pins) noexcept;

struct Candidate {
    std::size_t quant = 0;
    KvCacheStorage kv = KvCacheStorage::BFloat16;
    KvCacheStorage ring = KvCacheStorage::BFloat16;
    RecurrentStateStorage state = RecurrentStateStorage::FP32;
    ActivationMode activations = ActivationMode::Fast;
};

// ---- The three estimates.

// The route a candidate's linears take on this card.
[[nodiscard]] Route route_of(const QuantOption& quant, ActivationMode activations,
                             const CardFacts& card) noexcept;

struct Speed {
    double decode_per_stream = 0; // tokens a second each stream decodes at the context asked
    double prefill           = 0; // prompt tokens a second
    bool transferred         = false; // a constant came from another generation (kTransferGeneration)
};

// Decode at `context` tokens in the window with `concurrency` rows, and prefill.
[[nodiscard]] Speed speed_of(const ModelFacts& model, const CardFacts& card,
                             const QuantOption& quant, const Candidate& candidate,
                             std::uint32_t concurrency, double context);

// One request of `prompt` tokens in and `answer` out on one stream of `concurrency`: the prompt at
// the prefill rate, the answer at the decode rate at the middle of its window.
[[nodiscard]] double request_seconds(const ModelFacts& model, const CardFacts& card,
                                     const QuantOption& quant, const Candidate& candidate,
                                     const Task& task);

struct Term {
    Axis axis;
    std::string value;
    double kld = 0;
    bool measured = true;    // false: a stand-in (the axis's worst measured term, or a proxy)
    bool transferred = false; // measured on another family (kKldFamily)
    std::string source;
};

struct Accuracy {
    double kld = 0; // root sum square of the terms
    std::vector<Term> terms;
    bool quant_known = false; // a KLD for the quantization itself (else the quant rule picks it)
};

[[nodiscard]] Accuracy accuracy_of(const ModelFacts& model, const QuantOption& quant,
                                   const Candidate& candidate, const CardFacts& card);

// ---- The choice.

struct Scored {
    Candidate candidate;
    std::uint32_t own_context = 0; // the longest context each stream holds of its own (0: none)
    bool fits = false;             // own_context >= the task's
    Speed speed;
    Accuracy accuracy;
    double seconds = 0; // request_seconds
    double worth   = 0; // -ln(seconds) - lambda kld
};

// The own context per stream a candidate holds (the plan without the weights), 0 when none.
using OwnContextFn = std::function<std::uint32_t(const Candidate&)>;

struct Choice {
    std::vector<Scored> scored;        // every candidate, in enumeration order
    std::optional<std::size_t> chosen; // into scored; nullopt: nothing fits
    std::string why;                   // one line: what decided it
};

// The axes' values the model and the card allow (a family's KV codecs, int8 activations only on
// Qwen 3.5/3.6), before the pins.
struct Allowed {
    std::vector<KvCacheStorage> kv;
    std::vector<ActivationMode> activations;
};
[[nodiscard]] Allowed allowed_axes(const ModelFacts& model, const CardFacts& card);

[[nodiscard]] Choice choose(const ModelFacts& model, const CardFacts& card,
                            const std::vector<QuantOption>& quants, const Allowed& allowed,
                            const Pins& pins, const Task& task, const OwnContextFn& own_context);

// What each other value of an axis would change against the chosen candidate (the others kept).
struct Alternative {
    Axis axis;
    std::string value;
    const Scored* scored = nullptr; // nullptr: not a candidate (pinned away or not allowed)
    std::string why_not;            // "" for the chosen value
};
[[nodiscard]] std::vector<Alternative> explain(const Choice& choice,
                                               const std::vector<QuantOption>& quants);

} // namespace ninferno::product::choice
