#pragma once

#include "ninferno/types.h"

#include <cstdint>
#include <vector>

namespace ninferno {

// A causal score's per-target distribution detail (Engine::score_tokens), for measuring a quantized
// model against a reference (ninferno-perplexity --kld): the reference records each position's most
// probable tokens, the measured model reads its log-probabilities of the same tokens. Both are
// laid out [target][k]; an unlisted slot holds id -1 and log-probability -inf.
struct ScoreDetailRequest {
    std::uint32_t top_k   = 0;      // 0 .. 64: the top_k most probable tokens per target
    std::uint32_t given_k = 0;      // 0 .. 64: ids per target in given_ids
    std::vector<TokenId> given_ids; // [targets][given_k], each a vocabulary id or -1
    // 0: the scored targets run in prefill chunks. N: the context before the first target runs in
    // chunks and the targets N tokens per call, as decode (1) or a speculative round (N) runs them,
    // so state rounding that compounds per call shows in the score.
    std::uint32_t step_tokens = 0;
};

struct ScoreDetail {
    std::vector<TokenId> top_ids;      // [targets][top_k], by probability descending
    std::vector<float> top_logprobs;   // [targets][top_k]
    std::vector<float> given_logprobs; // [targets][given_k]
};

} // namespace ninferno
