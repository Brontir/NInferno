#pragma once

// The immutable loaded Inkling 2 model: its config and the parameters prepared from the artifact
// (docs/maintainer/inkling2-model.md). Execution state (KV planes, convolution states, workspace)
// lives in Target.

#include "core/device.h"
#include "core/tensor.h"
#include "models/inkling2/config.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/weight_input.h"

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

namespace ninferno::models {
struct ArtifactBacking;
} // namespace ninferno::models

namespace ninferno::models::inkling2 {

using Projection = ops::SingleProjectionWeight;

// A looped layer's controller for one of q, k, v, ffn: z = down [n(c); n(m)], gain = up_g z,
// forget = up_f z.
struct Controller {
    Projection down, gain, forget;
};

struct Layer {
    LayerKind kind = LayerKind::Sliding;
    // Sliding: input_norm feeds q, k and v; full: one norm each.
    Tensor input_norm, q_norm, k_norm, v_norm;
    Projection query, key, value, gate, relative, sink, output;
    Projection key_taps, value_taps;                    // full layers only
    Tensor query_clip, relative_clip, relative_bias;    // FP32 [Hq, D], [Hq, Dr], [extent, Dr]
    Tensor key_conv, value_conv, output_conv, ffn_conv; // FP32 [C, 4]
    Tensor attention_query;                             // FP32 [m, H] (m 1 or 3); absent on layer 0
    Tensor ffn_query;                                   // FP32 [1, H]
    Tensor ffn_norm;
    ops::MoeWeights moe;
    ops::MoeConfig moe_config;
    // Looped layers only.
    Tensor message_query; // FP32 [4, H]
    std::array<Controller, 4> control{};
};

struct Weights {
    Projection token_embedding; // read by ops::embedding
    Tensor embedding_norm, final_norm, output_query;
    Projection output_head;
    std::vector<Layer> layers;
};

class Model {
public:
    static std::unique_ptr<Model> load(const std::filesystem::path& path, DeviceContext& device);
    ~Model();

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] const Weights& weights() const noexcept { return weights_; }

private:
    Model() = default;
    Config config_;
    Weights weights_;
    std::unique_ptr<ArtifactBacking> backing_;
};

} // namespace ninferno::models::inkling2
