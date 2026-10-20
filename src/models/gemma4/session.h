#pragma once

// One sequence running on a loaded target: prefill, scoring, decode and greedy speculative decode.
// The session owns every piece of mutable state (KV pools, workspace, captured graphs); the models
// it reads stay immutable.

#include "core/device.h"
#include "models/gemma4/model.h"
#include "models/speculation/draft_target.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninferno::models::gemma4 {

// The widest verify a round runs: a DFlash block (16) or at most 15 drafts and the pending token.
inline constexpr std::int32_t kMaxVerifyColumns = 16;

struct SessionOptions {
    std::int32_t max_context   = 32768;
    std::int32_t prefill_chunk = 2048;
    // FP8 keys and values (E4M3 with a scale per row) instead of BF16: about half the KV bytes,
    // so fit_session finds a longer context. The MTP drafter reads the same planes.
    bool kv_fp8 = false;
    // FP8 on the full layers alone, the sliding rings in BF16 (ninferno-serve's Gemma 4 default).
    bool kv_fp8_full = false;
    // Columns of one head pass in teacher-forced scoring (score()); the session's head slab holds
    // vocab x score_columns BF16 (128 MiB for Gemma 4 at 256). A session that never scores can
    // keep kMaxVerifyColumns, the widest verify, and give the difference to its context. Scores
    // are compared at the default: the head's route follows its column count.
    std::int32_t score_columns = 256;
    // temperature <= 0 is greedy. The engine's sampler keeps at most kMaximumTopK (64) candidates
    // before top-p.
    float temperature       = 0;
    float top_p             = 1;
    std::int32_t top_k      = 20;
    unsigned long long seed = 0;
    // Speculative decoding: `draft_tokens` proposals per verified round, from Google's MTP
    // drafter, z-lab's DFlash drafter or prompt lookup (at most one). Greedy only: a round accepts the longest
    // drafted prefix the target's argmax agrees with. DFlash always drafts a whole block and
    // verifies its first draft_tokens.
    const Drafter* drafter      = nullptr;
    const DFlashDrafter* dflash = nullptr;
    std::int32_t draft_tokens   = 4;
    // Nonzero and below draft_tokens: the depth adapts per round within
    // [min_draft_tokens, draft_tokens] (deeper by two after a fully accepted round, one shallower
    // otherwise). Zero: every round drafts draft_tokens.
    std::int32_t min_draft_tokens = 0;
    // Prompt lookup, the model-free fallback (instead of a drafter): each round drafts the tokens
    // that followed the most recent earlier occurrence of the context's last ngram_max ..
    // ngram_min tokens, at most draft_tokens of them; a step without such an occurrence is a
    // plain decode step. Driven by speculate_until().
    bool ngram               = false;
    std::int32_t ngram_min   = 2;
    std::int32_t ngram_max   = 4;
    // Acceptance watchdog of speculate_until(): once the tokens emitted per round over the last
    // watchdog_window rounds of a sequence fall below acceptance_floor (tokens per round; 1 or
    // less: off), the sequence continues with plain decode until the next prefill, and `log`
    // is told once. With prompt lookup, a step without a match counts as a round of one token.
    double acceptance_floor          = 0;
    std::int32_t watchdog_window     = 32;
    std::int32_t watchdog_min_rounds = 16;
    std::function<void(const std::string&)> log;
};

// The rounds' counters and an audit's record of one round (models/speculation/draft_target.h).
using speculation::SpeculationRoundTrace;
using speculation::SpeculationStats;

// One sequence, one stream. Prefill runs in chunks; decode runs one token per step through a
// captured CUDA graph whose inputs (token, position) live on the device.
class Session {
public:
    Session(const Model& model, DeviceContext& device, SessionOptions options);
    ~Session();
    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;

    [[nodiscard]] std::int32_t position() const noexcept { return position_; }

    // Starts a new sequence: the next prefill begins at position 0, and speculation's adaptive
    // depth schedule starts over from the middle of its range.
    void reset() noexcept;

    // Captures every CUDA graph the session can replay: the plain decode step and, with a
    // drafter, the round of every draft depth it can run. Without it, the first decode() or
    // speculate() captures them inside that request's decode, and a process's first request
    // decodes slower than the ones after it. Call it once, before the first prefill (at load);
    // it writes keys and values at the first positions, which that prefill rewrites.
    void warm_up();

    // Replaces the sampling parameters read by prefill and by every later decode step, including
    // replays of an already captured graph.
    void set_sampling(float temperature, float top_p, std::int32_t top_k, unsigned long long seed);

    // Runs the prompt through every layer, appending its keys and values; writes the greedy next
    // token to the decode input. Returns the number of prompt tokens consumed.
    void prefill(std::span<const std::int32_t> tokens);

    // Log-probability of tokens[i+1] after tokens[0..i], for every i in the span, computed chunk by
    // chunk with the soft-capped head (the teacher-forced perplexity numerator).
    // When `predictions` is given it receives the argmax of the same logits, one per scored token.
    // When `logits` is given it receives those soft-capped logits themselves (vocab_size values
    // per scored token, in scoring order), for distribution-level audits against a reference.
    [[nodiscard]] std::vector<float> score(std::span<const std::int32_t> tokens,
                                           std::vector<std::int32_t>* predictions = nullptr,
                                           std::vector<float>* logits             = nullptr);

    // Greedy decode of `count` tokens after a prefill. Returns them in order.
    [[nodiscard]] std::vector<std::int32_t> decode(std::int32_t count);

    // Speculative greedy decode with the session's drafter: `rounds` draft-and-verify rounds.
    // Returns every token emitted, starting with the one the prefill drew; each round emits
    // between 1 and draft_tokens + 1 tokens. Requires SessionOptions::drafter or ::dflash.
    [[nodiscard]] std::vector<std::int32_t> speculate(std::int32_t rounds);
    [[nodiscard]] SpeculationStats speculation_stats() const;

    // Speculative greedy decode of at most `max_tokens` tokens, stopping after the first token in
    // `stops`, wherever it lands in a round (among the accepted drafts or as the target's own
    // token). Returns the tokens in order, the prefill's draw first, like decode(): each was
    // consumed, so its keys and values are in the cache. Rounds that emitted past the stop or the
    // budget leave the session at the returned tokens' end: the next call must be a prefill
    // (a new turn), which rewrites every piece of state the extra tokens touched.
    [[nodiscard]] std::vector<std::int32_t>
    speculate_until(std::int32_t max_tokens, std::span<const std::int32_t> stops = {});

    // speculate() for audits: the same rounds, each followed by a host synchronization that
    // appends its trace. With `profile`, the replayed graphs carry event records at the phase
    // boundaries and the trace holds their device times. The emitted tokens equal speculate()'s.
    [[nodiscard]] std::vector<std::int32_t>
    speculate_traced(std::int32_t rounds, std::vector<SpeculationRoundTrace>& trace, bool profile);

    // Replaces the pending token the next decode step consumes (teacher-forced audits).
    void set_pending_token(std::int32_t token);

    // Whether the watchdog stopped speculation for the current sequence, and how many tokens
    // the current sequence's speculate_until() calls decoded plainly (after the watchdog tripped,
    // or prompt-lookup steps without a match).
    [[nodiscard]] bool speculation_latched() const noexcept;
    [[nodiscard]] std::int64_t plain_steps() const noexcept;

    // Final-position logits after the last prefill or decode step (soft-capped), for audits.
    [[nodiscard]] std::vector<float> last_logits();

    // Device bytes owned by the session (KV pools plus workspace).
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;

    // Device bytes of the target's KV planes alone.
    [[nodiscard]] std::uint64_t kv_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::int32_t position_ = 0;
};

// Sizing a session to the device. The context and the prefill chunk are chosen from the memory
// that is free now, by range logic on the actual numbers, never from a memory-size class: a
// session makes its own allocations up front (workspace, KV pools, a drafter's rings), so a trial
// session either fits or fails to allocate, and the CUDA graphs it captures later (decode, one per
// speculation depth) need a reserve on top.
struct SessionFitRequest {
    std::int32_t context           = 0;     // a fixed context; 0 chooses one
    std::int32_t chunk             = 0;     // a fixed prefill chunk; 0 chooses one
    std::int32_t max_context       = 32768; // the largest context worth choosing
    std::int32_t preferred_context = 8192;  // a narrower chunk is taken to reach this much
};

struct SessionFit {
    std::int32_t context         = 0;
    std::int32_t chunk           = 0;
    std::uint64_t free_bytes     = 0; // device memory free before the session
    std::uint64_t reserve_bytes  = 0; // left free for the session's graphs
    std::int32_t trials          = 0; // trial sessions constructed
};

// The widest chunk of 2048, 1024, 512, 256, 128 whose largest fitting context (a multiple of 1024,
// at most max_context, above the sliding window plus the chunk) reaches min(preferred_context,
// max_context), else the chunk that leaves the most context; then the largest context at that
// chunk. A fixed context or chunk is kept and
// only checked. `options` carries the drafter configuration the session will run (drafter or
// dflash, draft_tokens, min_draft_tokens); its context and chunk are ignored. Throws
// std::runtime_error naming the free memory when nothing fits.
[[nodiscard]] SessionFit fit_session(const Model& model, DeviceContext& device,
                                     SessionOptions options, const SessionFitRequest& request);

} // namespace ninferno::models::gemma4
