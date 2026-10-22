#pragma once

// The context classes of speculation (spec/context_class.h) as an engine applies them: the class
// of a context by its length and a request's depth bound in it (SpeculativeOptions::class_depth).
// Header-only and free of the registry, so the Engine families include it without the spec
// library.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace ninferno::spec {

enum class ContextClass : std::uint8_t { Short, Mid, Long, Huge };

inline constexpr std::size_t kContextClasses = 4;
// The upper bounds (exclusive) of Short, Mid and Long, in prompt tokens.
inline constexpr std::array<std::int64_t, kContextClasses - 1> kContextBounds{1024, 8192, 24576};

[[nodiscard]] constexpr ContextClass class_of(std::int64_t prompt_tokens) noexcept {
    if (prompt_tokens < kContextBounds[0]) { return ContextClass::Short; }
    if (prompt_tokens < kContextBounds[1]) { return ContextClass::Mid; }
    if (prompt_tokens < kContextBounds[2]) { return ContextClass::Long; }
    return ContextClass::Huge;
}

// short, mid, long, huge.
[[nodiscard]] constexpr const char* class_name(ContextClass c) noexcept {
    switch (c) {
    case ContextClass::Short:
        return "short";
    case ContextClass::Mid:
        return "mid";
    case ContextClass::Long:
        return "long";
    case ContextClass::Huge:
        return "huge";
    }
    return "unknown";
}

// No depth bound in a class (SpeculativeOptions::kNoClassBound).
inline constexpr std::uint32_t kNoClassBound = 0xffffffffU;

// The depth bound of a request whose context holds `tokens` tokens: bounds[class_of(tokens)].
[[nodiscard]] constexpr std::uint32_t
class_bound(const std::array<std::uint32_t, kContextClasses>& bounds,
            std::int64_t tokens) noexcept {
    return bounds[static_cast<std::size_t>(class_of(tokens))];
}

// Whether the class bounds apply to a request: the bounds are decided for the engine's default
// choice (its depth `default_depth`, the profile's pick for the server's thinking default), so
// they bound only a request that starts speculating at that depth. A request on another menu
// choice (--spec-select thinking with another depth) or plain is not bounded.
[[nodiscard]] constexpr bool class_bounds_apply(std::uint32_t start_depth,
                                                std::uint32_t default_depth) noexcept {
    return start_depth != 0 && start_depth == default_depth;
}

// A request drafting at `depth`, whose step-down depths are `steps` ({depth, floor}, deepest first,
// each below the one before), once its context holds `tokens` tokens: the depth its class allows,
// never above `depth` (a bound only lowers it: a context that grows into a longer class with a
// lower bound lowers it, any other leaves it), how many leading steps go (those at or above the
// new depth), and, when the new depth is one of them, that step's floor (the watchdog then judges
// against it). Depth 0: plain decode for the rest of the request.
struct ClassClamp {
    std::uint32_t depth = 0;
    std::size_t dropped = 0;
    std::optional<double> floor;

    [[nodiscard]] bool lowered(std::uint32_t from) const noexcept { return depth < from; }
};

[[nodiscard]] inline ClassClamp
clamp_to_class(const std::array<std::uint32_t, kContextClasses>& bounds, std::int64_t tokens,
               std::uint32_t depth, const std::vector<std::pair<std::uint32_t, double>>& steps) {
    ClassClamp out{.depth = std::min(depth, class_bound(bounds, tokens))};
    if (!out.lowered(depth)) { return out; }
    while (out.dropped < steps.size() && steps[out.dropped].first >= out.depth) {
        if (steps[out.dropped].first == out.depth) { out.floor = steps[out.dropped].second; }
        ++out.dropped;
    }
    return out;
}

} // namespace ninferno::spec
