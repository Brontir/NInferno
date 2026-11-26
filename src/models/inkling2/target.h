#pragma once

// The Inkling 2 forward over one sequence (pass.h over the sequence's own planes): every visit's
// KV planes (a ring on a sliding visit) and convolution state, for the reference comparison and
// the standalone app. This form runs plain launches (no CUDA graph); the Program runs the same
// Pass over its paged pool, rings and state images.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"
#include "models/inkling2/pass.h"
#include "ninferno/ops/gemma4_attention.h"
#include "ninferno/ops/inkling2.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninferno::models::inkling2 {

class Target {
public:
    // max_context bounds every position + 1; max_tokens bounds a call's tokens (a prefill chunk).
    Target(const Model& model, std::int32_t max_context, std::int32_t max_tokens);

    // Clears every visit's state: the next call starts a new sequence at position 0.
    void reset();

    // Runs `ids` at positions [position(), position() + T) and returns the soft-capped logits of
    // the last token (vocab_size values, host). Advances the position by T.
    std::vector<float> forward(std::span<const std::int32_t> ids, cudaStream_t stream);

    // The visit schedule (slot and physical layer of each visit, in order).
    [[nodiscard]] const std::vector<Visit>& visits() const noexcept { return pass_.visits(); }

    [[nodiscard]] std::int32_t position() const noexcept { return position_; }

    // Optional: called with every visit's output stream (BF16 [H, T]), the first visit's
    // intermediate tensors and the final hidden state of the last token, for comparisons with a
    // reference (`name` as llama.cpp's callback names them: "l_out", ..., "result_norm"; visit -1
    // for all but "l_out").
    std::function<void(const char* name, std::int32_t visit, const Tensor& value)> observe;

private:
    struct Slot {
        DeviceBuffer k, v, conv; // KV planes [D, capacity, Hkv]; conv state FP32 [C, 3]
        std::int32_t capacity = 0;
    };

    const Model& model_;
    const Config& c_;
    Pass pass_;
    std::vector<Slot> slots_;
    std::vector<ops::Gemma4KvLayer> kv_;
    std::vector<ops::Inkling2ConvState> conv_;
    std::int32_t max_context_ = 0, max_tokens_ = 0, position_ = 0;
    DeviceArena work_;
    DeviceBuffer ids_, positions_, logits_;
};

} // namespace ninferno::models::inkling2
