#pragma once

// One speculative-decoding configuration, as the tuner measures it and a profile records it:
//
//   plain            no speculation
//   mtp:K            an autoregressive drafter (Qwen's built-in MTP head, Gemma 4's assistant),
//                    K drafts per round
//   mtp:K:K0         the same with the depth adapting per round within [K0, K] (Gemma 4)
//   dflash:K         a block drafter (DFlash), the first K drafts of its block verified
//   dflash2:K        DFlash2 (Qwen companion)
//   ngram:K          prompt lookup: drafts copied from an earlier occurrence of the context's
//                    suffix, at most K per round (no drafter model)
//
// A label optionally ends in "+head" (Qwen's optimized proposal head, --lm-head-draft).

#include <cstdint>
#include <string>
#include <string_view>

namespace ninferno::spec {

enum class Mode : std::uint8_t { None, Mtp, DFlash, DFlash2, Ngram };

struct SpecConfig {
    Mode mode          = Mode::None;
    std::int32_t k     = 0;
    bool proposal_head = false;

    [[nodiscard]] constexpr bool enabled() const noexcept { return mode != Mode::None; }
    [[nodiscard]] std::string label() const;
    // Throws std::invalid_argument naming the text when it is not a label above.
    [[nodiscard]] static SpecConfig parse(std::string_view label);

    friend bool operator==(const SpecConfig&, const SpecConfig&) = default;
};

[[nodiscard]] const char* mode_name(Mode mode) noexcept;

} // namespace ninferno::spec
