#pragma once

// A debug aid for prefill chunk-invariance work (MaintainerOptions::debug_prefill_cuts, the
// engine's --debug-prefill-cuts).

#include <cstdint>
#include <vector>

namespace ninferno::models::qwen3_5::execution {

// Every Qwen 3.5 prefill call of the process ends at each of these absolute positions, the way a
// restored or continued context meets the cache off the chunk grid. Empty, the default: the chunk
// grid alone. Set by the engine at construction, before any prefill.
void set_debug_prefill_cuts(const std::vector<std::uint64_t>& cuts);

} // namespace ninferno::models::qwen3_5::execution
