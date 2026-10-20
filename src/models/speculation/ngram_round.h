#pragma once

// The draft phase of a prompt-lookup (n-gram) round: no drafter model. The host finds the
// longest earlier occurrence of the context's last few tokens and writes the tokens that followed
// it into a pinned buffer before the round's graph is launched; the round copies them into the
// verify batch after the pending token. Verification is the ordinary greedy one, so the output is
// what plain decode would emit (up to the verify pass's own near-tie arithmetic).

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/speculation/draft_round.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninferno::models::speculation {

class NgramRound final : public DraftRound {
public:
    NgramRound(DeviceContext& device, std::int32_t max_drafts);

    // Opens a round from the device-side position P of the pending `token` and copies the k
    // drafts staged by stage() behind it. Captured into the round's graph: the copy reads the
    // pinned buffer when the graph is replayed.
    void draft(const std::int32_t* position, const std::int32_t* token,
               std::int32_t* verify_positions, std::int32_t k) override;

    // I32 [columns]: the pending token, then the drafts.
    [[nodiscard]] Tensor ids(std::int32_t columns) const override;

    [[nodiscard]] bool staged() const noexcept override { return true; }

    // Writes the drafts the next launched round verifies (at most max_drafts).
    void stage(std::span<const std::int32_t> drafts);

private:
    DeviceContext& device_;
    std::int32_t max_drafts_;
    DeviceBuffer draft_ids_;  // I32 [max_drafts + 1]
    PinnedHostBuffer staged_; // I32 [max_drafts]
};

// Prompt lookup over a token history: the continuation after the most recent earlier occurrence
// of the history's longest suffix of length in [min_n, max_n] (longest first), at most `limit`
// tokens. Empty when no suffix recurs.
[[nodiscard]] std::vector<std::int32_t> ngram_lookup(std::span<const std::int32_t> history,
                                                     std::int32_t min_n, std::int32_t max_n,
                                                     std::int32_t limit);

} // namespace ninferno::models::speculation
