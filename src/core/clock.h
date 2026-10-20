#pragma once

// Time as the programs report it: UTC timestamps in ISO 8601 and elapsed seconds.

#include "core/sys.h"

#include <chrono>
#include <ctime>
#include <string>

namespace ninferno {

// "2026-10-19T16:20:00Z": UTC, to the second (sys::gmtime, safe from any thread).
[[nodiscard]] inline std::string utc_iso8601(std::chrono::system_clock::time_point time) {
    std::tm tm{};
    sys::gmtime(std::chrono::system_clock::to_time_t(time), tm);
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return text;
}

[[nodiscard]] inline std::string utc_now() { return utc_iso8601(std::chrono::system_clock::now()); }

// Seconds from `start` to now on the steady clock.
[[nodiscard]] inline double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace ninferno
