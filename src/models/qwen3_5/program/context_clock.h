#pragma once

// The Program's wall clock, for timings it reports (prefill, transfer and host phases): telemetry
// only. Planning code (planning/) includes context_work.h, which carries no clock.

#include <chrono>
#include <cstdint>

namespace ninferno::models::qwen3_5::detail {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point started) noexcept;

} // namespace ninferno::models::qwen3_5::detail
