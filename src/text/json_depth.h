#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace ninferno::text {

// Whether a JSON value nests arrays and objects deeper than `limit` levels (a scalar is level 0,
// an array or object level 1). Iterative: the values it guards are walked recursively later (chat
// templates, schema conversion, serialization), so measuring a deep one must not use the stack.
template <class Json>
[[nodiscard]] bool json_nests_deeper_than(const Json& value, std::size_t limit) {
    std::vector<std::pair<const Json*, std::size_t>> pending{{&value, 0}};
    while (!pending.empty()) {
        const auto [node, depth] = pending.back();
        pending.pop_back();
        if (!node->is_structured()) { continue; }
        if (depth + 1 > limit) { return true; }
        for (const auto& child : *node) { pending.emplace_back(&child, depth + 1); }
    }
    return false;
}

} // namespace ninferno::text
