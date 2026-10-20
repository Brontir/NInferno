#pragma once
#include "models/qwen3_5/program/internal.h"

#include "models/qwen3_5/program/vision_control.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninferno::models::qwen3_5::execution {

// One request's Vision encoding across its prefill calls, as the Program drives it: a family with
// an encoder implements it (Qwen's VisionPrefillSession, Gemma 4's) and reads its own form back
// in its prefill_multimodal_chunk.
class VisionPrefill {
public:
    virtual ~VisionPrefill() = default;
    // Host patch payloads of the items encoded so far may be dropped.
    virtual void release_encoded_media_payloads() noexcept = 0;
    // The encoded item's handoff region is no longer read.
    virtual void retire_handoff() noexcept = 0;
    [[nodiscard]] virtual double elapsed_seconds() const = 0;
    [[nodiscard]] virtual std::size_t active_handoff_bytes() const noexcept = 0;
    // Where a prefill unit from `cursor` that would end at `end` ends instead: a family that runs
    // each image as one call (Gemma 4) ends the unit before an image the boundary would cut, so the
    // image starts the next unit whole; one whose encoder hands an image over in parts keeps `end`.
    [[nodiscard]] virtual std::uint32_t unit_end(std::uint32_t cursor,
                                                 std::uint32_t end) const noexcept {
        (void)cursor;
        return end;
    }
};

} // namespace ninferno::models::qwen3_5::execution

namespace ninferno::models::qwen3_5::detail {

struct VisionUseSpan {
    std::uint32_t begin               = 0;
    std::uint32_t end                 = 0;
    std::uint32_t prepared_item_index = 0;
    std::uint32_t control_index       = 0;
};

struct VisionPrefillPlan {
    std::shared_ptr<const qwen3_5::VisionControlPlan> control_plan;
    std::shared_ptr<const qwen3_5::VisionControl> control;
    std::vector<VisionUseSpan> uses;
    std::size_t max_merged_count = 0;
};

} // namespace ninferno::models::qwen3_5::detail
