#pragma once

#include "ninferno/types.h"

#include <filesystem>
#include <string>

namespace ninferno::product {

[[nodiscard]] PromptInput prompt_from_text(std::string text, std::optional<bool> enable_thinking);
[[nodiscard]] PromptInput prompt_from_messages(const std::filesystem::path& path,
                                               std::optional<bool> enable_thinking,
                                               bool vision_enabled);

} // namespace ninferno::product
