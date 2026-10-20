#pragma once

// A loaded dense decoder: its config and every parameter in the form its Op consumes. Immutable;
// execution state lives in Session.

#include "core/device.h"
#include "core/expert_residency.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/dense/config.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/weight_input.h"
#include "ninferno/types.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::models {
struct ArtifactBacking; // models/common/loading.h
} // namespace ninferno::models

namespace ninferno::models::dense {

using Projection = ops::SingleProjectionWeight;

// A layer's parameters. Norm and bias tensors a configuration does not use stay empty.
struct Layer {
    Tensor attention_norm, attention_post_norm, ffn_norm, ffn_post_norm;
    Tensor query_norm, key_norm;
    Tensor query_bias, key_bias, value_bias, output_bias;
    // One plane whose rows are [q; k; v], or [q; k] with v apart (a GGUF layer whose v takes a
    // wider type), or three apart when the artifact could not pack them. With an output gate the
    // plane may hold its rows after k: [q; k; gate; v] or [q; k; gate] (gate_in_qkv).
    std::optional<Projection> qkv;
    Projection query, key, value; // value: beside a [q; k] qkv too
    Projection output;
    // The attention output gate W_g [Hq*D, H] (attention_output_gate) apart, else empty.
    std::optional<Projection> attention_gate;
    bool gate_in_qkv = false;
    // One plane whose rows are [gate; up], or two apart when the artifact could not pack them.
    std::optional<Projection> gate_up;
    Projection gate, up;
    Projection down;
    // A MoE decoder's routed feed-forward (every stacked bank one complete plane), and a layer's
    // attention sink logits (FP32 [Hq]).
    std::optional<ops::MoeWeights> moe;
    Tensor sinks;
};

struct Weights {
    Weight token_embedding; // the embedding multiplier is folded in at conversion
    Projection output_head; // the logit divisor is folded in at conversion
    Tensor final_norm;
    // Gain-1 BF16 vectors for the norms that have no weight ([H]: the embedding norm; [D]: an
    // unscaled q/k norm); empty when the configuration has none.
    Tensor unit_gain_hidden, unit_gain_head;
    std::vector<Layer> layers;
    std::vector<Tensor> inv_freq;      // FP32 [rotary_dim/2] per rope group, on the device
    std::vector<Tensor> long_inv_freq; // a LongRoPE group's long mode (else empty), likewise
    // MoE expert offload: where each MoE call reads its expert bank (null when off), and the
    // CPU that computes the offloaded layers' experts in decode calls (null: the GPU does).
    ExpertResidency* experts       = nullptr;
    ops::HostExperts* host_experts = nullptr;
};

// Which activation policy the projections run: each Use's recorded one (W4A4 where an NVFP4
// checkpoint was calibrated for it) or BF16 activations everywhere.
enum class ActivationProfile : std::uint8_t { Artifact, A16 };

class Model {
public:
    static std::unique_ptr<Model> load(const std::filesystem::path& path, DeviceContext& device,
                                       ActivationProfile profile,
                                       const MoeOffloadOptions& offload = {});
    ~Model();

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] const Weights& weights() const noexcept { return weights_; }
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;
    [[nodiscard]] double upload_seconds() const noexcept;
    // Routed experts left in the device arena (MoE expert offload off, or layers it did not take).
    [[nodiscard]] std::uint64_t device_expert_bytes() const noexcept;
    [[nodiscard]] std::uint32_t device_expert_layers() const noexcept;
    [[nodiscard]] std::uint64_t host_to_device_bytes() const noexcept;
    // MoE expert offload's CPU decode path: whether every offloaded layer's experts are in a
    // format the CPU computes (else why), and the executor the Engine attaches once the groups
    // are placed (before anything runs).
    [[nodiscard]] bool host_experts_readable(std::string* why) const;
    void attach_host_experts(ops::HostExperts* host) noexcept { weights_.host_experts = host; }

private:
    Model() = default;
    std::unique_ptr<ArtifactBacking> backing_;
    Config config_;
    Weights weights_;
};

} // namespace ninferno::models::dense
