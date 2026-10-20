#pragma once

// The instance config of a NemotronH decoder (model_type "nemotron_h") as the converter writes it
// (src/convert/dense.cpp, resolve_nemotron_h): one mixer per block, x += mixer(RMSNorm(x)), each
// block a Mamba2 mixer, a NoPE attention or a relu^2 MoE; a final RMSNorm and the head.

#include "artifact/schema.h"
#include "ninferno/ops/mamba2.h"
#include "ninferno/ops/moe.h"

#include <cstdint>
#include <vector>

namespace ninferno::models::nemotron_h {

enum class Block : std::uint8_t { Mamba, Attention, Moe };

struct MambaSpec {
    ops::Mamba2Geometry geometry;
    bool conv_bias = true;

    [[nodiscard]] std::int32_t inner() const noexcept {
        return geometry.heads * geometry.head_dim;
    }
    // x, B and C: the convolution's channels.
    [[nodiscard]] std::int32_t conv_channels() const noexcept {
        return inner() + 2 * geometry.groups * geometry.state;
    }
    // z, xBC and dt: in_proj's rows.
    [[nodiscard]] std::int32_t in_rows() const noexcept {
        return inner() + conv_channels() + geometry.heads;
    }
    [[nodiscard]] std::int32_t norm_group() const noexcept { return inner() / geometry.groups; }
};

struct MoeSpec {
    std::int32_t experts      = 0;
    std::int32_t input        = 0; ///< the routed up bank's K: hidden, or zero-padded above it
    std::int32_t inter        = 0;
    std::int32_t shared_inter = 0;
    ops::MoeConfig op;
};

struct Config {
    std::int32_t hidden_size   = 0;
    std::int32_t vocab_size    = 0;
    std::int32_t layers        = 0;
    std::int32_t max_positions = 0;
    float rms_norm_eps         = 0;
    std::int32_t query_heads   = 0;
    std::int32_t kv_heads      = 0;
    std::int32_t head_dim      = 0;
    float attention_scale      = 1;
    MambaSpec mamba;
    MoeSpec moe;
    std::vector<Block> blocks;

    [[nodiscard]] std::uint32_t count(Block kind) const noexcept;
};

[[nodiscard]] Config parse_config(const artifact::Json& text_config);

// The instance config of a drafter artifact's text component (model_type "nemotron_h_mtp"): the
// head's attention geometry and MoE, as the converter wrote it (resolve_nemotron_h_mtp). Only the
// fields the pairing checks read; the head has no Mamba2 layers.
struct MtpConfig {
    std::int32_t hidden_size   = 0;
    std::int32_t vocab_size    = 0;
    std::int32_t max_positions = 0;
    float rms_norm_eps         = 0;
    float layer_norm_eps       = 0;
    std::int32_t query_heads   = 0;
    std::int32_t kv_heads      = 0;
    std::int32_t head_dim      = 0;
    MoeSpec moe;
};

[[nodiscard]] MtpConfig parse_mtp_config(const artifact::Json& text_config);

} // namespace ninferno::models::nemotron_h
