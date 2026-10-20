#pragma once

// Route catalogs keyed by a call's column count: bands [first, last] that start at one, follow
// one another without a gap and end at kAnyColumns, each naming the schedule its columns run.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninferno::ops::detail {

inline constexpr std::int32_t kAnyColumns = std::numeric_limits<std::int32_t>::max();

template <class Schedule>
struct ColumnRoute {
    std::int32_t first;
    std::int32_t last;
    Schedule schedule;

    constexpr bool contains(std::int32_t columns) const noexcept {
        return columns >= first && columns <= last;
    }
};

// Whether the bands cover every positive column count exactly once, in order.
template <class Schedule, std::size_t N>
constexpr bool column_routes_closed(const std::array<ColumnRoute<Schedule>, N>& routes) {
    std::int64_t expected = 1;
    for (const ColumnRoute<Schedule>& route : routes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyColumns) + 1;
}

// The schedule of the band holding `columns`; `uncovered` is the logic_error when none does.
template <class Schedule, std::size_t N>
Schedule column_route(const std::array<ColumnRoute<Schedule>, N>& routes, std::int32_t columns,
                      const char* uncovered) {
    for (const ColumnRoute<Schedule>& route : routes) {
        if (columns >= route.first && columns <= route.last) { return route.schedule; }
    }
    throw std::logic_error(uncovered);
}

} // namespace ninferno::ops::detail
