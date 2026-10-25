#pragma once

// Greedy speculative decoding on any target (models/speculation/draft_target.h): rounds of drafted
// tokens verified by one target pass over k+1 columns. Each round is captured once per draft
// depth as a CUDA graph and replayed; the depth is fixed or adapts round by round.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "models/speculation/draft_round.h"
#include "models/speculation/draft_target.h"
#include "models/speculation/ngram_round.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninferno::models::speculation {

struct DecoderOptions {
    std::int32_t max_context      = 0;
    std::int32_t draft_tokens = 0; // every round's depth
};

class Decoder {
public:
    // `round` drafts every round; `work` is the session's workspace, sized for the target's verify
    // and whatever the drafter carves from it. Both outlive the decoder.
    Decoder(DraftTarget& target, DeviceContext& device, DeviceArena& work,
            const DecoderOptions& options, DecodeState state, DraftRound& round);

    // `rounds` rounds from the pending token at `position`. Returns every token emitted, the
    // pending one first; `position` leaves at the new pending token's, which is not returned.
    // With `trace`, every round is followed by a host synchronization that appends its record;
    // `profile` replays graphs that also time the round's phases (see SpeculationRoundTrace).
    [[nodiscard]] std::vector<std::int32_t>
    speculate(std::int32_t rounds, std::int32_t& position,
              std::vector<SpeculationRoundTrace>* trace = nullptr, bool profile = false);

    [[nodiscard]] SpeculationStats stats() const;

    // Prompt lookup: one round verifying the host's `drafts` (1..draft_tokens) after the pending
    // token at `position`. Returns the tokens consumed (the pending one first, then the accepted
    // drafts) and sets `pending` to the new pending token; `position` advances past the consumed
    // ones. One host synchronization. Requires a staged round (NgramRound).
    [[nodiscard]] std::vector<std::int32_t> verify_drafts(NgramRound& lookup,
                                                          std::span<const std::int32_t> drafts,
                                                          std::int32_t& position,
                                                          std::int32_t& pending);

    // Captures the round of every depth the session can run (a session's warm-up): its depth,
    // and for a staged round every draft count up to it. The rounds open at the device's pending
    // token and position, which the caller sets.
    void capture_rounds();

    [[nodiscard]] std::int32_t deepest() const noexcept { return deepest_; }

private:
    // One round: the draft phase, one target verify over k+1 columns, acceptance. `accept` false
    // leaves every piece of session state untouched (the pre-capture warm-up).
    void round(bool accept, std::int32_t k);

    // The round of depth k as a graph, captured on first use; the profiled family carries event
    // records at the phase boundaries.
    DecodeGraphExecutable& round_graph(std::int32_t k, bool profiled = false);

    // Host copy of the round that just ran (after a synchronization).
    SpeculationRoundTrace read_round(std::int32_t position, std::int32_t k, bool profiled);

    // Copies the emitted tokens and the new position back after the replayed rounds.
    std::vector<std::int32_t> collect(std::int32_t& position);

    // Phase-boundary events recorded by the profiled graphs: round start, end of the draft
    // phase, of the target forward, of its head and argmax, and round end.
    static constexpr int kPhaseEvents = 5;
    struct PhaseEvents {
        cudaEvent_t e[kPhaseEvents] = {};
        PhaseEvents();
        ~PhaseEvents();
        PhaseEvents(const PhaseEvents&)            = delete;
        PhaseEvents& operator=(const PhaseEvents&) = delete;
    };
    void mark(int phase);

    DraftTarget& target_;
    DeviceContext& device_;
    DeviceArena& work_;
    DraftRound& round_;
    std::int32_t max_context_;
    std::int32_t vocab_size_;
    std::int32_t hidden_size_;
    std::int32_t deepest_; // every round's draft depth
    DecodeState state_;
    DeviceBuffer verify_positions_; // I32 [k+1]: P..P+k
    DeviceBuffer target_ids_;       // I32 [k+1]: the target's argmax of every verified column
    DeviceBuffer stats_;            // I64 [3]: rounds, drafted, accepted
    std::vector<DecodeGraphExecutable> round_graphs_;    // by draft depth k, captured on first use
    std::vector<DecodeGraphExecutable> profiled_graphs_; // the same with phase events
    std::optional<PhaseEvents> events_;                  // created by the first profiled capture
    bool marking_ = false;                               // round() records phase events
};

} // namespace ninferno::models::speculation
