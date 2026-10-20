#pragma once

// Token ids a model session reads: every id must name a row of the embedding table. A tokenizer
// may define added tokens past it (Gemma 3 1B's <image_soft_token>), which the model cannot read,
// and an id past the table reads beyond it.

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace ninferno::models {

inline void check_token_ids(std::span<const std::int32_t> tokens, std::int32_t vocab,
                            const char* what) {
    for (const auto t : tokens) {
        if (t < 0 || t >= vocab) {
            throw std::invalid_argument(std::string(what) + ": token id " + std::to_string(t) +
                                        " is outside the vocabulary of " + std::to_string(vocab));
        }
    }
}

} // namespace ninferno::models
