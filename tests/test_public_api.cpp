#include "ninferno/engine.h"

#include <type_traits>

static_assert(std::is_move_constructible_v<ninferno::PreparedPrompt>);
static_assert(!std::is_copy_constructible_v<ninferno::PreparedPrompt>);
static_assert(std::is_move_constructible_v<ninferno::Engine>);
static_assert(!std::is_copy_constructible_v<ninferno::Engine>);

int main() {
    const ninferno::EngineOptions options;
    return options.enable_vision ? 1 : 0;
}
