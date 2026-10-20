#pragma once

// What the CLI sessions of the dense and Gemma 4 families share around their Decoder: the bounds
// of a speculate() call and the steps the SequenceDriver runs on the session.

#include "core/arena.h"
#include "core/device.h"
#include "models/speculation/decoder.h"
#include "models/speculation/driver.h"
#include "models/speculation/ngram_round.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::models::speculation {

// Refuses `rounds` rounds of `draft_tokens` drafts each when they could consume more than one
// call's output (`max_decode` tokens) or run past the session context from `position`.
inline void check_speculate_rounds(const char* family, std::int32_t rounds,
                                   std::int32_t draft_tokens, std::int64_t position,
                                   std::int64_t max_context, std::int32_t max_decode) {
    const std::int64_t most = std::int64_t(rounds) * (draft_tokens + 1) + 1;
    if (rounds <= 0 || most > max_decode) {
        throw std::invalid_argument(std::string(family) + " speculate: rounds out of range");
    }
    if (position + most > max_context) {
        throw std::invalid_argument(std::string(family) +
                                    " speculate: exceeds the session context");
    }
}

// The driver's steps on a session: its own rounds and plain decode, prompt-lookup rounds through
// its Decoder, and its position and pending token.
template <class Session>
class SessionSteps final : public Steps {
public:
    SessionSteps(Session& session, Decoder& decoder, NgramRound* lookup, std::int32_t& position,
                 const DeviceBuffer& pending, const DeviceContext& device)
        : session_(session), decoder_(decoder), lookup_(lookup), position_(position),
          pending_(pending), device_(device) {}

    std::vector<std::int32_t> speculate(std::int32_t rounds) override {
        return session_.speculate(rounds);
    }

    std::vector<std::int32_t> decode(std::int32_t count) override { return session_.decode(count); }

    std::vector<std::int32_t> verify_drafts(std::span<const std::int32_t> drafts,
                                            std::int32_t& pending) override {
        return decoder_.verify_drafts(*lookup_, drafts, position_, pending);
    }

    std::int32_t pending_token() override {
        std::int32_t token = 0;
        CUDA_CHECK(cudaMemcpyAsync(&token, pending_.p, sizeof(token), cudaMemcpyDeviceToHost,
                                   device_.stream));
        device_.synchronize();
        return token;
    }

    std::int32_t position() const override { return position_; }

    void set_position(std::int32_t position) override { position_ = position; }

    SpeculationStats stats() const override { return decoder_.stats(); }

private:
    Session& session_;
    Decoder& decoder_;
    NgramRound* lookup_; // null when the session has no prompt lookup
    std::int32_t& position_;
    const DeviceBuffer& pending_;
    const DeviceContext& device_;
};

} // namespace ninferno::models::speculation
