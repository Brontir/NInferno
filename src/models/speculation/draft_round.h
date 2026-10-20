#pragma once

// The draft phase of a speculative round, as the rounds (models/speculation/decoder.h) drive it:
// each drafter (a family's MTP assistant, the DFlash drafter, prompt lookup) opens a round from
// the device's pending token and position, leaves its drafts after that token in ids(), and may
// ask the verify forward for the tapped layers' features or keep the state behind the new
// pending token.

#include "core/tensor.h"
#include "models/speculation/draft_target.h"

#include <cstdint>
#include <optional>

namespace ninferno::models::speculation {

class DraftRound {
public:
    virtual ~DraftRound() = default;

    // Opens a round from the device-side position P of the pending `token` (the verify batch at
    // `verify_positions` covers P..P+k) and drafts after it into ids().
    virtual void draft(const std::int32_t* position, const std::int32_t* token,
                       std::int32_t* verify_positions, std::int32_t k) = 0;

    // I32 [columns]: the pending token, then the drafts; the verify batch's ids.
    [[nodiscard]] virtual Tensor ids(std::int32_t columns) const = 0;

    // Drafts one draft phase of depth k leaves in ids(): k, or a whole block's.
    [[nodiscard]] virtual std::int32_t drafted(std::int32_t k) const { return k; }

    // The capture the verify forward over `columns` must make, if any.
    [[nodiscard]] virtual std::optional<FeatureTap> tap(std::int32_t columns) const {
        (void)columns;
        return std::nullopt;
    }

    // After the verify forward, before acceptance: the verified columns' features at `pos`.
    virtual void commit(const FeatureTap& tap, const Tensor& pos) {
        (void)tap;
        (void)pos;
    }

    // BF16 [H, 1] that acceptance fills with the target's final-normed state of the column that
    // predicted the new pending token (the MTP drafter's first input); empty when unused.
    [[nodiscard]] virtual Tensor carry() const { return Tensor{}; }

    // Rounds verify drafts the host staged, of any count up to the deepest (prompt lookup).
    [[nodiscard]] virtual bool staged() const noexcept { return false; }
};

} // namespace ninferno::models::speculation
