#pragma once

// The text config of an Inkling 2 artifact (the converter's inkling2_text) and the decoder's
// visit schedule: which physical layer each of the 48 layer visits runs, which state slot (KV
// planes and convolution state) it owns, and which pass of the loop it belongs to.
// docs/maintainer/inkling2-model.md states the mathematics.

#include "artifact/schema.h"

#include <cstdint>
#include <vector>

namespace ninferno::models::inkling2 {

enum class LayerKind : std::uint8_t { Sliding, Full };

struct Config {
    std::int32_t hidden_size     = 0; // H
    std::int32_t vocab_rows      = 0; // the table's and head's rows (padded)
    std::int32_t vocab_size      = 0; // the tokenizer's tokens: the logits kept
    std::int32_t layers          = 0; // physical layers
    std::int32_t max_positions   = 0;
    std::int32_t query_heads     = 0;
    std::int32_t kv_heads        = 0;
    std::int32_t head_dim        = 0;
    std::int32_t sliding_window  = 0;
    std::int32_t rel_width       = 0; // d_rel
    std::int32_t rel_extent      = 0; // the full layers' bias table rows
    std::int32_t conv_taps       = 0; // 4
    std::int32_t block_size      = 0; // layers a residual block
    std::int32_t experts         = 0;
    std::int32_t experts_per_tok = 0;
    std::int32_t expert_width    = 0; // F
    float rms_norm_eps           = 0;
    float qk_clip                = 0;
    float rel_clip               = 0;
    float log_scale_floor        = 0;
    float log_scale_alpha        = 0;
    float route_scale            = 0;
    float situ_beta              = 0; // gate soft cap
    float situ_linear_beta       = 0; // up soft cap
    float softcap                = 0; // final logits
    // The loop: layers [core_start, core_end) run `passes` times; passes == 1 means none.
    std::int32_t core_start = 0;
    std::int32_t core_end   = 0;
    std::int32_t passes     = 1;
    std::int32_t gate_rank  = 0;
    std::vector<LayerKind> kinds;

    [[nodiscard]] bool looped() const noexcept { return passes > 1; }
    [[nodiscard]] std::int32_t core_layers() const noexcept { return core_end - core_start; }
    [[nodiscard]] std::int32_t visits() const noexcept { return layers + (passes - 1) * core_layers(); }
    [[nodiscard]] std::int32_t extent(std::int32_t layer) const noexcept {
        return kinds[static_cast<std::size_t>(layer)] == LayerKind::Full ? rel_extent : sliding_window;
    }
    // Channels of one visit's convolution state row: K, V, the attention output, the expert output.
    [[nodiscard]] std::int32_t conv_channels() const noexcept { return 2 * kv_heads * head_dim + 2 * hidden_size; }
};

[[nodiscard]] Config parse_config(const artifact::Json& text_config);

// One layer visit of the decoder, in execution order.
struct Visit {
    std::int32_t layer = 0; // the physical layer whose weights it runs
    std::int32_t slot  = 0; // its KV planes and convolution state (llama.cpp's cache slot il)
    std::int32_t pass  = 0; // 0 outside the loop core and in its first pass
};

// layers [0, core_start) once, the core `passes` times (slots layers + (pass - 1) * core + i
// for passes after the first), then [core_end, layers) once.
[[nodiscard]] std::vector<Visit> visit_schedule(const Config& config);

} // namespace ninferno::models::inkling2
