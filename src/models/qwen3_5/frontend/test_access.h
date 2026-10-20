#pragma once

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

namespace ninferno::models::qwen3_5 {

class FrontendTestAccess {
public:
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace ninferno::models::qwen3_5
