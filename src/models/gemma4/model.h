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
// - `A8` allows eight-bit activations on every projection and expert (--activations 8): calibrated
//   NVFP4 experts take the generic MoE's W4A8 prefill instead of the native four-bit Gemma4Moe.
// - `A4` allows four bits everywhere (--activations 4); experts as under `Artifact`.
// A head recorded BF16 keeps it under `A8` and `A4` (models::effective_policy).
// On a model without an expert bank `A16Full` equals `A16`. Decode and verify widths (T <= 16)
// already run the expert bank in BF16 under every profile.
enum class ActivationProfile : std::uint8_t { Artifact, A16, A16Full, A8, A4 };

// The --activations mode a profile prepares its dense projections under.
[[nodiscard]] constexpr ActivationMode activation_mode_of(ActivationProfile profile) noexcept {
    switch (profile) {
    case ActivationProfile::Artifact:
        return ActivationMode::Auto;
    case ActivationProfile::A16:
    case ActivationProfile::A16Full:
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
    // `vision`: also bind the artifact's vision tower (its "vision" component; refused when it
    // has none). Without it the tower is not loaded at all.
    static std::unique_ptr<Model> load(const std::filesystem::path& path, DeviceContext& device,
                                       ActivationProfile profile, bool vision = false);
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
    [[nodiscard]] std::uint64_t host_to_device_bytes() const noexcept;

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
