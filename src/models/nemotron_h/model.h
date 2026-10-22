#pragma once

// A loaded NemotronH decoder: its config and every parameter in the form its Op consumes.
// Immutable; the recurrent and attention state lives in the Program's pools.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/nemotron_h/config.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/weight_input.h"
#include "ninferno/types.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace ninferno::models {
struct ArtifactBacking; // models/common/loading.h
} // namespace ninferno::models

namespace ninferno::models::nemotron_h {

using Projection = ops::SingleProjectionWeight;

struct MambaWeights {
    Projection in_proj;  // [z; xBC; dt] rows, [2*inner + 2*G*N + H, hidden]
    Tensor conv_weight;  // BF16 [4, C]
    Tensor conv_bias;    // BF16 [C] (empty without)
    Tensor dt_bias;      // FP32 [H]
    Tensor a_log;        // FP32 [H]
    Tensor d;            // FP32 [H]
    Tensor norm;         // BF16 [inner]
    Projection out_proj; // [hidden, inner]
};

struct AttentionWeights {
    std::optional<Projection> qkv; // [q; k; v] packed, else the three apart
    Projection query, key, value;
    Projection output;
};

struct Layer {
    Block kind = Block::Mamba;
    Tensor norm;
    std::uint32_t index = 0; // place among the layers of its kind (state slot plane, KV plane)
    std::optional<MambaWeights> mamba;
    std::optional<AttentionWeights> attention;
    std::optional<ops::MoeWeights> moe;
};

// The MTP (NextN) head of the published drafter: one block of a NoPE attention and a MoE sub-layer
// over the fusion of the next token's embedding and the previous hidden. The embedding and the
// LM head are the target's.
struct MtpWeights {
    Tensor enorm;                // BF16 [H]: RMSNorm of the embedding
    Tensor hnorm;                // BF16 [H]: RMSNorm of the previous hidden
    Projection eh_proj;          // [H, 2H], input rows [norm(e); norm(h)]
    Tensor attn_norm;            // BF16 [H]
    AttentionWeights attention;  // NoPE, its own keys and values
    Tensor post_attention_norm;  // BF16 [H], before the MoE
    ops::MoeWeights moe;         // the trunk's MoE math (Config::moe.op)
    Tensor final_norm;           // BF16 [H]: LayerNorm weight, no bias
    float rms_norm_eps   = 0;    // the drafter's: enorm, hnorm, attn_norm, post_attention_norm
    float layer_norm_eps = 0;    // the drafter's layer_norm_epsilon: final_norm (as llama.cpp)
};

struct Weights {
    Weight token_embedding;
    Projection output_head;
    Tensor final_norm;
    std::vector<Layer> layers;
};

// As dense::ActivationProfile: the recorded policies, BF16 everywhere, or eight or four bits
// allowed everywhere (a head recorded BF16 keeps it).
enum class ActivationProfile : std::uint8_t { Artifact, A16, A8, A4 };

[[nodiscard]] constexpr ActivationMode activation_mode_of(ActivationProfile profile) noexcept {
    switch (profile) {
    case ActivationProfile::Artifact:
        return ActivationMode::Auto;
    case ActivationProfile::A16:
        return ActivationMode::Bits16;
    case ActivationProfile::A8:
        return ActivationMode::Bits8;
    case ActivationProfile::A4:
        return ActivationMode::Bits4;
    }
    return ActivationMode::Bits16;
}

class Model {
public:
    static std::unique_ptr<Model> load(const std::filesystem::path& path, DeviceContext& device,
                                       ActivationProfile profile);
    ~Model();

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] const Weights& weights() const noexcept { return weights_; }
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;
    [[nodiscard]] double upload_seconds() const noexcept;
    [[nodiscard]] std::uint64_t host_to_device_bytes() const noexcept;

private:
    Model() = default;
    std::unique_ptr<ArtifactBacking> backing_;
    Config config_;
    Weights weights_;
};

// The drafter artifact of the MTP head (a nemotron-h-mtp GGUF converted by itself): its config
// and every parameter of the head in the form its Op consumes, validated against the target it
// pairs with. The embedding and the LM head stay the target's; nothing of them is loaded here.
class MtpDrafter {
public:
    // Binds a drafter artifact and checks it fits `target`: hidden width, attention geometry and
    // the MoE (experts, top-k, routing, relu^2, shared expert) the trunk computes.
    static std::unique_ptr<MtpDrafter> load(const std::filesystem::path& path,
                                            DeviceContext& device, const Config& target);
    ~MtpDrafter();

    [[nodiscard]] const MtpWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] std::uint64_t device_bytes() const noexcept;
    [[nodiscard]] std::uint64_t host_to_device_bytes() const noexcept;

private:
    MtpDrafter() = default;
    std::unique_ptr<ArtifactBacking> backing_;
    MtpWeights weights_;
};

} // namespace ninferno::models::nemotron_h
