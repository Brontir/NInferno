#pragma once

// The Inkling 2 forward over one sequence: every visit's KV planes and convolution state, the
// residual blocks and the loop's messages (docs/maintainer/inkling2-model.md). This first form
// runs plain launches (no CUDA graph) for exactness; the decode graph comes after the family is
// qualified against llama.cpp.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"

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
    [[nodiscard]] const std::vector<Visit>& visits() const noexcept { return visits_; }

    [[nodiscard]] std::int32_t position() const noexcept { return position_; }

    // Optional: called with every visit's output stream (BF16 [H, T]) and the final hidden state,
    // for comparisons with a reference (`name` as llama.cpp's callback names them: "l_out",
    // "result_norm"; visit -1 for the latter).
    std::function<void(const char* name, std::int32_t visit, const Tensor& value)> observe;

private:
    struct Slot {
        DeviceBuffer k, v, conv; // KV planes [D, capacity, Hkv]; conv state FP32 [C, 3]
        std::int32_t capacity = 0;
    };

    Tensor state(std::int32_t slot, std::int32_t offset, std::int32_t channels);
    Tensor stream_buffer(); // a fresh [H, max_tokens] BF16 buffer of the pool, for blocks

    const Model& model_;
    const Config& c_;
    std::vector<Visit> visits_;
    std::vector<Slot> slots_;
    std::int32_t max_context_ = 0, max_tokens_ = 0, position_ = 0;
    DeviceArena work_;
    std::vector<DeviceBuffer> pool_; // block snapshots and the loop's bank outputs
    std::size_t pool_next_ = 0;
    DeviceBuffer ids_, positions_, logits_;
};

} // namespace ninferno::models::inkling2
