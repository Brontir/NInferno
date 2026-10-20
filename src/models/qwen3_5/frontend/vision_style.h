#pragma once

#include <cstdint>

namespace ninferno::models::qwen3_5 {

// Which model's Vision encoder the prompt feeds: its resize, patch layout and prompt tokens.
//   Qwen:   Qwen2-VL's smart resize to 32-px multiples, [c, t=2, y, x] patches of 1536 values in
//           2 x 2 merge blocks, <|image_pad|> / <|video_pad|> runs inside the template's
//           <|vision_start|> ... <|vision_end|>, MRoPE positions.
//   Gemma4: Gemma4ImageProcessor's aspect-preserving resize to 48-px multiples within 280 soft
//           tokens, [y, x, c] patches of 768 values in raster order, the template's <|image|>
//           expanded to <|image> + one <|image|> per soft token + <image|>, text positions.
//           Images only.
enum class VisionStyle : std::uint8_t {
    Qwen,
    Gemma4,
};

} // namespace ninferno::models::qwen3_5
