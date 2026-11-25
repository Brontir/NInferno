#pragma once

// The Inkling 2 forward over bindings (docs/maintainer/inkling2-model.md; llama.cpp inkling2.cpp
// build_layer / build_output and inkling2-loop.cpp's schedule, op by op): B sequences of T tokens
// each, column b * T + t, every visit's keys, values and convolution state where the caller keeps
// them (Target: one sequence's own planes; the Program: its paged pool, rings and state images).

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"
#include "ninferno/ops/gemma4_attention.h"
#include "ninferno/ops/inkling2.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninferno::models::inkling2 {

// One call's state, per visit slot (Visit::slot): its keys and values (a full visit's from the
// paged pool or a slot plane, a sliding visit's from a ring a batch row: gemma4_attention.h's
// layouts, cache.rows naming each batch row's binding), and its convolution state (the slot's
// FP32 [3, C] planes, C = Config::conv_channels(): K, V, the attention output, the MoE output at
// channel offsets 0, KV, 2 KV, 2 KV + H; Inkling2ConvState::slots naming each batch row's plane).
struct Bindings {
    std::span<const ops::Gemma4KvLayer> kv;
    std::span<const ops::Inkling2ConvState> conv;
    std::int32_t batch = 1;
};

class Pass {
public:
    explicit Pass(const Model& model);

    // tokens, positions: device I32 [T * B] (positions sequential within each batch row). Appends
    // every visit's keys and values, advances its convolution state, and returns the final-normed
    // hidden of every column, BF16 [H, T * B], in `work` (with the call's stream planes: the
    // caller's scope around the call frees them).
    Tensor forward(DeviceArena& work, const Tensor& tokens, const Tensor& positions,
                   const Bindings& bound, cudaStream_t s);

    // logits = softcap(output_head(hidden)): BF16 [vocab_rows, columns].
    void head(DeviceArena& work, const Tensor& hidden, Tensor& logits, cudaStream_t s) const;

    [[nodiscard]] const std::vector<Visit>& visits() const noexcept { return visits_; }

    [[nodiscard]] std::int32_t slots() const noexcept { return slots_; }

    // An upper bound of the work arena one call of `columns` takes (its temporaries, the MoE's
    // workspace, the projections' own).
    [[nodiscard]] static std::size_t work_bytes(const Model& model, std::int32_t columns);

    // Optional: every visit's output stream ("l_out", BF16 [H, T * B]) and, on the first visit,
    // its intermediate tensors under llama.cpp's names (visit -1), for a comparison with a
    // reference dump. The stream is synchronized before each call.
    std::function<void(const char* name, std::int32_t visit, const Tensor& value)> observe;

private:
    const Model& model_;
    const Config& c_;
    std::vector<Visit> visits_;
    std::int32_t slots_ = 0, max_blocks_ = 0, max_bank_ = 0;
};

} // namespace ninferno::models::inkling2
