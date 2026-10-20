#pragma once

#include "runtime/contract/request.h"

#include <cstdint>
#include <cstdlib>

namespace ninferno::runtime {

class GenerationBudget {
public:
    GenerationBudget(std::uint32_t effective_tokens, FinishReason limit_reason)
        : remaining_(effective_tokens), limit_reason_(limit_reason) {
        if (limit_reason_ != FinishReason::OutputLimit &&
            limit_reason_ != FinishReason::ContextCapacity) {
            std::abort();
        }
    }

    [[nodiscard]] std::uint32_t remaining() const noexcept { return remaining_; }

    [[nodiscard]] FinishReason limit_reason() const noexcept { return limit_reason_; }

    void commit(std::uint32_t tokens) noexcept {
        if (tokens > remaining_) { std::abort(); }
        remaining_ -= tokens;
    }

    // Narrows the licensed output to a bounded completion (a settled KV lease): the request then
    // ends with the limit reason this budget already carries.
    void cap_remaining(std::uint32_t limit) noexcept {
        if (limit < remaining_) { remaining_ = limit; }
    }

private:
    std::uint32_t remaining_   = 0;
    FinishReason limit_reason_ = FinishReason::None;
};

} // namespace ninferno::runtime
