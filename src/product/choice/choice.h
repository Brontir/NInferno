#pragma once

// The configuration choice: one rule for every place that picks how a model is served when the
// user leaves it to the engine (ninferno-serve without the flags, --plan --no-load, ninferno-ui's
// /api/plan). From facts only (the model's shape, the card's generation, the user's task), no
// measurement on the user's machine:
//
//   memory     the own context each stream holds at a configuration (the plan without the
//              weights: runtime/engine/plan_dry_run.h, through OwnContextFn);
//   speed      the seconds of a typical request at it (decode and prefill, choice_constants.h);
//   accuracy   its KLD against the exact reference (choice_constants.h's measured terms).
//
// A drafter is one more axis (choice/SPECULATION.md S3): plain decode or one of the drafters the
// model carries and the serving Engine runs, each at the configuration chosen for it (the lab's
// measured pair, its group's, a published profile), its memory in the plan and its gain in the
// decode speed; it changes no output (greedy verification), so no accuracy term.
//
// The choice is made at one quantization (a listing row's weights): its candidates are the KV
// codecs x activation modes x drafters x prefill chunks (an MoE's) the model, the card and the
// user's pins allow; the ring and the recurrent state follow the KV codec and the activations
// (coupled_storages), and their other values are scored beside the chosen one to explain them. A
// candidate fits when each stream holds the asked context of its own. Where the generation's
// constants and the accuracy terms rest on two lab cards (Gates), the largest worth
// -ln(T) - lambda KLD wins, T priced on the generation's reference card (choice_constants.h: what a
// decision may read); else the family's defaults stand. An MoE's chunk is the fastest on the
// reference card where its shares rest on two lab cards (it has no accuracy term). Every candidate
// keeps the estimates on the board itself for a caller to show (explain). Which quantization a
// listing serves is the quant rule's, weighed by the worth of each row's own choice where the
// quantization curve decides (product/plan/plan_quants.cpp, weighs_quantizations).
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
    double clock_hz = 0;        // reference SM clock (the PCI-ID table's)
    double dram_bytes_per_s = 0;
    bool fp4 = false;           // FP4 tensor cores (NVFP4 weights run W4A4)
};

// One way to decode beside plain decode: a drafter the model carries at the configuration chosen
// for it, and its gain.
struct DrafterOption {
    std::string label;     // the configuration, as --spec-config takes it ("mtp:4", "dflash:7")
    std::string drafter;   // the drafter's name in the artifact or manifest
    double speedup = 1;    // decode of one stream against plain decode's (the tune's choice)
    // From this many streams a round of every stream decodes plainly (the load table's first
    // bound at depth 0); 0: any number speculates.
    std::uint32_t plain_from_streams = 0;
    std::string basis;     // where the gain comes from (measured, preliminary, the registry's)
    // Whether a measurement gives the gain (a profile of this card or a similar one, a lab pair);
    // false for a choice published without its speeds (the "*" profile): its gain is unknown, so
    // the drafter axis is not weighed and the profile's choice stands (never plain decode for a
    // gain of 1 nobody measured).
    bool measured = true;
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
    std::function<std::uint64_t(KvCacheStorage)> ring_bytes_per_token; // unset: no sliding layers
    std::function<std::uint64_t(RecurrentStateStorage)> state_bytes;    // unset: no recurrent layers
    std::uint32_t window = 0;         // the sliding window (0: none)
    // A token's attention: heads x (key + value) summed over the full layers (over every earlier
    // token) and the sliding ones (over at most the window); 2 flops each per position.
    double full_attention_width = 0, sliding_attention_width = 0;
    bool recurrent = false;           // Gated DeltaNet layers (a --state-dtype to choose)
    bool fp8_ring  = false;           // its sliding rings take FP8 (Gemma 4, dense decoders)
    // The drafters it carries, beside plain decode (Candidate::drafter 0); and the one the server
    // takes where the worth does not decide (its resolved profile's: an index into the candidates'
    // drafters, 0 for plain).
    std::vector<DrafterOption> drafters;
    std::size_t default_drafter = 0;
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

// What the user asks: the context each stream holds of its own (the contexts in turn: a pinned
// one, or Auto's targets runtime::quant_own_targets gives), the streams, the request shape.
struct Task {
    std::vector<std::uint32_t> contexts = {32768};
    std::uint32_t concurrency = 1;
    std::uint32_t prompt      = kRequestPrompt;
    std::uint32_t answer      = kRequestAnswer;
    double accuracy_weight    = kAccuracyWeight;
};

// What the user pinned with flags; the choice keeps it.
struct Pins {
    std::optional<KvCacheStorage> kv;
    std::optional<KvCacheStorage> ring;
    std::optional<RecurrentStateStorage> state;
    std::optional<ActivationMode> activations;
    std::optional<std::uint32_t> chunk; // --prefill-chunk
    std::optional<std::size_t> drafter; // --spec given (0: none); a Candidate::drafter
};

// The storages that follow the KV codec and the activations (the owner's rule, 23.10): the sliding
// rings in the KV cache's kind (FP8 under any narrower KV codec where the family's rings take FP8,
// else BF16), the recurrent state one step finer than the KV cache (FP32 under BF16 KV, FP16 under
// a narrower one; FP32 under exact, which refuses a rounded state). Pinned values stay.
struct Coupled {
    KvCacheStorage ring;
    RecurrentStateStorage state;
};
[[nodiscard]] Coupled coupled_storages(const ModelFacts& model, KvCacheStorage kv,
                                       ActivationMode activations, const Pins& pins) noexcept;

struct Candidate {
    KvCacheStorage kv = KvCacheStorage::BFloat16;
    KvCacheStorage ring = KvCacheStorage::BFloat16;
    RecurrentStateStorage state = RecurrentStateStorage::FP32;
    ActivationMode activations = ActivationMode::Fast;
    std::uint32_t chunk = kDefaultChunk; // the widest prefill chunk (the ladder may narrow it)
    std::size_t drafter = 0; // 0: plain decode; i: ModelFacts::drafters[i - 1]
};

// ---- The three estimates.

// The route a candidate's linears take on this card.
[[nodiscard]] Route route_of(const QuantOption& quant, ActivationMode activations,
                             const CardFacts& card) noexcept;

struct Speed {
    double decode_per_stream = 0; // tokens a second each stream decodes at the context given
    double prefill           = 0; // prompt tokens a second
    bool transferred         = false; // a constant came from another generation (kTransferGeneration)
};

// Decode at `context` tokens in the window with `concurrency` rows, and prefill of a `prompt`
// (0: long enough to fill every chunk), on `card`.
[[nodiscard]] Speed speed_of(const ModelFacts& model, const CardFacts& card,
                             const QuantOption& quant, const Candidate& candidate,
                             std::uint32_t concurrency, double context, std::uint32_t prompt = 0);

// One request of `prompt` tokens in and `answer` out on one stream of `concurrency`: the prompt at
// the prefill rate, the answer at the decode rate at the middle of its window.
[[nodiscard]] double request_seconds(const ModelFacts& model, const CardFacts& card,
                                     const QuantOption& quant, const Candidate& candidate,
                                     const Task& task);

// Which measured constant sets rest on two lab cards (choice_constants.h) and so may decide. The
// defaults are the constants'; a caller never sets them (a test opens or closes them to check what
// they gate).
struct Gates {
    // The accuracy terms the worth decides by (a family's KV codecs' and activations', the
    // quantization curve): nullopt, from their cards (accuracy_decides, kQuantKldCards).
    std::optional<bool> accuracy;
    bool moe_chunk = kMoeChunkTwoCards; // the MoE chunk shares
};

// Whether the switches' terms the worth decides by (the KV codecs' and the int8 activations') were
// measured on `family` and rest on two lab cards on `generation`; `why` says which does not.
[[nodiscard]] bool accuracy_decides(std::string_view generation, std::string_view family,
                                    std::string* why = nullptr);

// The reference card a decision prices on for a board's generation (choice_constants.h), whether
// the generation's speed factors decide (both factor sets on two lab cards), and whether an MoE's
// chunk does (its shares on two). nullopt: no reference card.
struct Reference {
    CardFacts card;
    bool decides = false;
    bool chunk_decides = false;
    std::string why; // "" when the factors decide, else why the family's defaults stand
};
[[nodiscard]] std::optional<Reference> reference_of(const CardFacts& card, const Gates& gates = {});
// Whether the worth decides the KV codec and the activations for `model` on this card: its
// generation's factors decide and its family's switch terms rest on two cards (accuracy_decides);
// else the family's defaults stand.
[[nodiscard]] bool decides_switches(const ModelFacts& model, const CardFacts& card,
                                    const Gates& gates = {});
// Whether the worth also weighs a listing's quantizations (plan_quants, each row at its own
// choice): the curve was measured on its family (kQuantKldFamily); else the quant rule's largest
// stands.
[[nodiscard]] bool weighs_quantizations(const ModelFacts& model, const CardFacts& card,
                                        const Gates& gates = {});

struct Term {
    Axis axis;
    std::string value;
    double kld = 0;
    bool measured = true;    // false: a stand-in (the axis's worst measured term, or a proxy)
    bool transferred = false; // measured on another family (kKldDefaultFamily, kQuantKldFamily)
    std::string source;
};

struct Accuracy {
    double kld = 0; // the quantization's term plus the root sum square of the switches'
    std::vector<Term> terms;
};

[[nodiscard]] Accuracy accuracy_of(const ModelFacts& model, const QuantOption& quant,
                                   const Candidate& candidate, const CardFacts& card);

// ---- The choice.

struct Scored {
    Candidate candidate;
    std::uint32_t own_context = 0; // the longest context each stream holds of its own (0: none)
    bool fits = false;             // own_context >= the context the choice settled on
    Speed speed;                   // on the board (an estimate to show)
    Accuracy accuracy;
    double seconds = 0;            // request_seconds on the board (an estimate to show)
    double decision_seconds = 0;   // request_seconds on the reference card (0: none)
    double worth = 0;              // -ln(decision_seconds) - lambda kld (0 without a reference)
    bool variant = false;          // a coupled storage's other value beside the chosen (not eligible)
    std::string drafter = "plain"; // its drafter's configuration (DrafterOption::label)
};

// The own context per stream a candidate holds (the plan without the weights), 0 when none.
using OwnContextFn = std::function<std::uint32_t(const Candidate&)>;

struct Choice {
    std::vector<Scored> scored;        // every candidate in enumeration order, then the variants
    std::optional<std::size_t> chosen; // into scored; nullopt: nothing fits
    std::uint32_t context = 0;         // the own context per stream the choice settled on
    bool by_worth = false;             // the worth decided the KV codec and activations (else the defaults)
    bool chunk_by_speed = false;       // an MoE's chunk was the fastest (else the default stood)
    // The worth decided the drafter (else the profile's choice stood: no decision constants, or a
    // drafter whose gain no measurement gives).
    bool drafter_by_worth = false;
    std::string why;                   // one line: what decided it
};

// The axes' values the model and the card allow (a family's KV codecs, int8 activations only on
// Qwen 3.5/3.6), before the pins, and the family's automatic KV codec on this card
// (runtime::plan_family_storage).
struct Allowed {
    std::vector<KvCacheStorage> kv;
    std::vector<ActivationMode> activations;
    KvCacheStorage automatic_kv = KvCacheStorage::BFloat16;
};
[[nodiscard]] Allowed allowed_axes(const ModelFacts& model, KvCacheStorage automatic_kv);

[[nodiscard]] Choice choose(const ModelFacts& model, const CardFacts& card, const QuantOption& quant,
                            const Allowed& allowed, const Pins& pins, const Task& task,
                            const OwnContextFn& own_context, const Gates& gates = {});

// What each other value of an axis would change against the chosen candidate (the others kept).
struct Alternative {
    Axis axis;
    std::string value;
    const Scored* scored = nullptr;
    std::string why_not;  // "" for the chosen value
    std::string why_code; // "" (chosen), no_fit, lower_worth, tie_context, tie_order, default, unmeasured, coupled
};
[[nodiscard]] std::vector<Alternative> explain(const Choice& choice);

} // namespace ninferno::product::choice
