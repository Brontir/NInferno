#pragma once

// One sequence running on a loaded dense decoder: prefill, scoring, decode and greedy speculative
// decode. The session owns every piece of mutable state (KV caches, workspace, the captured
// graphs); the models it reads stay immutable.

#include "core/device.h"
#include "models/dense/model.h"
#include "models/speculation/dflash.h"
#include "models/speculation/draft_target.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninferno::models::dense {

struct SessionOptions {
    std::int32_t max_context   = 32768;
    std::int32_t prefill_chunk = 2048;
    // FP8 keys and values (E4M3 with a scale per row) instead of BF16: half the KV bytes.
    bool kv_fp8 = false;
    // temperature <= 0 is greedy. The engine's sampler keeps at most kMaximumTopK (64) candidates
    // before top-p.
    float temperature       = 0;
    float top_p             = 1;
    std::int32_t top_k      = 20;
    unsigned long long seed = 0;
    // Decode each step eagerly, without the captured graph, under the attention envelope of the
    // step's own frontier (position + 1 keys) instead of the context's: the way ninferno-serve
    // --no-cuda-graph decodes, for a parity check against it. Slower; never the default.
    bool eager_decode = false;
    // Speculative decoding (models/speculation): `draft_tokens` proposals per verified round from
    // a DFlash drafter or prompt lookup (at most one). Greedy only: a round accepts the longest
    // drafted prefix the target's argmax agrees with. DFlash always drafts a whole block and
    // verifies its first draft_tokens.
    const speculation::DFlashDrafter* dflash = nullptr;
    std::int32_t draft_tokens                = 4;
    // Prompt lookup instead of a drafter, driven by speculate_until().
    bool ngram             = false;
    std::int32_t ngram_min = 2;
    std::int32_t ngram_max = 4;
    // speculate_until()'s acceptance watchdog (1 or less: off); see speculation::DriverOptions.
    double acceptance_floor          = 0;
    std::int32_t watchdog_window     = 32;
    std::int32_t watchdog_min_rounds = 16;
    std::function<void(const std::string&)> log;
};

// The widest verify a round runs: a DFlash block (16) or at most 15 drafts and the pending token,
// inside the attention's decode route.
inline constexpr std::int32_t kMaxVerifyColumns = 16;

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

    // Starts a new sequence: the next prefill begins at position 0, and speculation's watchdog
    // starts over.
    void reset() noexcept;

    // Captures every CUDA graph the session can replay: the plain decode step and, with a
    // drafter, the round of every draft depth it can run. Call it once, before the first prefill;
    // it writes keys and values at the first positions, which that prefill rewrites.
    void warm_up();

    // Replaces the sampling parameters read by prefill and every later decode step.
    void set_sampling(float temperature, float top_p, std::int32_t top_k, unsigned long long seed);

    // Runs the prompt through every layer, appending its keys and values; draws the next token.
    void prefill(std::span<const std::int32_t> tokens);

    // Log-probability of tokens[i+1] after tokens[0..i] for every i (teacher-forced), and the
    // argmax of the same logits when `predictions` is given.
    [[nodiscard]] std::vector<float> score(std::span<const std::int32_t> tokens,
                                           std::vector<std::int32_t>* predictions = nullptr);

    // Decodes `count` tokens after a prefill; returns the tokens each step consumed, the
    // prefill's draw first.
    [[nodiscard]] std::vector<std::int32_t> decode(std::int32_t count);

    // Speculative greedy decode with the session's drafter: `rounds` draft-and-verify rounds.
    // Returns every token emitted, the prefill's draw first; each round emits between 1 and
    // draft_tokens + 1 tokens. Requires SessionOptions::dflash.
    [[nodiscard]] std::vector<std::int32_t> speculate(std::int32_t rounds);

    // speculate() for audits: each round followed by a host synchronization that appends its
    // trace; with `profile`, the device times of the round's phases. The tokens equal speculate()'s.
    [[nodiscard]] std::vector<std::int32_t>
    speculate_traced(std::int32_t rounds, std::vector<SpeculationRoundTrace>& trace, bool profile);

    // Speculative greedy decode of at most `max_tokens` tokens, stopping after the first token in
    // `stops`; tokens in order, the prefill's draw first, like decode(). Rounds that emitted past
    // the stop or the budget leave the session at the returned tokens' end: the next call must be
    // a prefill (a new turn). Requires SessionOptions::dflash or ::ngram.
    [[nodiscard]] std::vector<std::int32_t>
    speculate_until(std::int32_t max_tokens, std::span<const std::int32_t> stops = {});

    [[nodiscard]] SpeculationStats speculation_stats() const;
    [[nodiscard]] bool speculation_latched() const noexcept;
    [[nodiscard]] std::int64_t plain_steps() const noexcept;

    // Final-position logits after the last prefill or decode step, for audits.
    [[nodiscard]] std::vector<float> last_logits();

    // Device bytes owned by the session (KV caches plus workspace).
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;

    // Device bytes of the KV caches alone.
    [[nodiscard]] std::uint64_t kv_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::int32_t position_ = 0;
};

} // namespace ninferno::models::dense
