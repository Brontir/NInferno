#pragma once

// The immutable loaded models of the family: a Gemma 4 target and its two kinds of drafter. Each
// owns its materialized artifact and the parameters prepared from it; nothing here is mutated by
// execution, which lives in Session.

#include "core/device.h"
#include "models/gemma4/config.h"
#include "models/gemma4/weights.h"
#include "ninferno/types.h"
#include "models/speculation/dflash.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace ninferno::models {
// Everything a loaded model's weights point into (models/common/loading.h): its materialized
// artifact and the device buffers derived from it.
struct ArtifactBacking;
} // namespace ninferno::models

namespace ninferno::models::gemma4 {

using models::ArtifactBacking;

// Which activation profile the NVFP4 projections may use.
// - `Artifact` honours each Use's recorded policy (W4A4 where the checkpoint was calibrated for
//   it), the expert bank's prefill route included.
// - `A16` keeps the activations of every dense projection (attention, dense MLP, head) in BF16;
//   the 26B-A4B expert bank keeps its recorded W4A4 prefill route, so on that model this profile
//   is mixed.
// - `A16Full` is `A16` plus W4A16 experts: no activation is encoded to four bits anywhere.
// On a model without an expert bank `A16Full` equals `A16`. Decode and verify widths (T <= 16)
// already run the expert bank in BF16 under every profile.
enum class ActivationProfile : std::uint8_t { Artifact, A16, A16Full };

class Model {
public:
    // `vision`: also bind the artifact's vision tower (its "vision" component; refused when it
    // has none). Without it the tower is not loaded at all.
    static std::unique_ptr<Model> load(const std::filesystem::path& path, DeviceContext& device,
                                       ActivationProfile profile,
                                       const MoeOffloadOptions& offload = {}, bool vision = false);
    ~Model();

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const Weights& weights() const noexcept { return weights_; }
    [[nodiscard]] const std::optional<VisionConfig>& vision_config() const noexcept {
        return vision_config_;
    }
    [[nodiscard]] const std::optional<VisionWeights>& vision_weights() const noexcept {
        return vision_weights_;
    }

    // The token embedding of `ids` into BF16 [H, T]: the table rows, times the embedding scale
    // when the table does not carry it folded in.
    void embed(const Tensor& ids, Tensor& out, cudaStream_t stream) const;

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
    std::optional<VisionConfig> vision_config_;
    std::optional<VisionWeights> vision_weights_;
};

class Drafter {
public:
    // Binds a drafter artifact and checks it fits `target`: the backbone width, query and KV
    // heads, head size, rotary pairs and theta, vocabulary and window: the attention geometry it
    // reads from the target's cache.
    static std::unique_ptr<Drafter> load(const std::filesystem::path& path, DeviceContext& device,
                                         const Config& target);
    ~Drafter();

    [[nodiscard]] const DrafterConfig& config() const noexcept { return config_; }

    [[nodiscard]] const DrafterWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] std::uint64_t device_bytes() const noexcept;

private:
    Drafter() = default;
    std::unique_ptr<ArtifactBacking> backing_;
    DrafterConfig config_;
    DrafterWeights weights_;
};

// z-lab's DFlash drafter: the one every target runs (models/speculation/dflash.h).
using speculation::DFlashConfig;
using speculation::DFlashDrafter;
using speculation::DFlashWeights;

} // namespace ninferno::models::gemma4
