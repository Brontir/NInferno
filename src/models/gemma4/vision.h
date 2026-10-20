#pragma once
// Gemma 4's vision tower on device: one image's patches to its soft tokens in the text width
// (modeling_gemma4.py Gemma4VisionModel and Gemma4MultimodalEmbedder). The patches come from the
// frontend already scaled to [-1, 1]; the encoder borrows one caller-owned workspace.
#include "core/device.h"
#include "core/tensor.h"
#include "models/gemma4/config.h"
#include "models/gemma4/weights.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninferno::models::gemma4 {

// One image as the encoder takes it: BF16 patches [3 p^2, width * rows] in raster order (row by
// row), each patch's values in (y, x, channel) order; width and rows multiples of the pooling.
struct VisionImage {
    std::span<const std::uint16_t> patches;
    std::int32_t width = 0; // patches per row
    std::int32_t rows  = 0;
};

class VisionEncoder {
public:
    VisionEncoder(const VisionConfig& config, const VisionWeights& weights);

    // Workspace for one image of up to `max_patches` patches (the output is the caller's).
    [[nodiscard]] std::size_t workspace_bytes(std::int32_t max_patches) const;
    // Soft tokens of an image of `patches` patches.
    [[nodiscard]] std::int32_t soft_tokens(std::int32_t patches) const noexcept {
        return patches / (config_.pooling * config_.pooling);
    }

    // output: contiguous BF16 [text hidden, soft_tokens(patches)].
    void encode(const VisionImage& image, Tensor& output, DeviceSpan workspace,
                cudaStream_t stream) const;

private:
    const VisionConfig& config_;
    const VisionWeights& weights_;
};

} // namespace ninferno::models::gemma4
