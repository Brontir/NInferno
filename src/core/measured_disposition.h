#pragma once

// How a measured answer applies to a board: one vocabulary for the measured gates (GateAnswer) and
// the measured value tables (TableAnswer) of core/measured_boards.h, kept in a header of its own so
// a result type can name it without the board registry.

namespace ninferno::measured {

// Measured: the board itself (distance 0, every quantity known: the board's facts are a measured
// board's, not necessarily that the very SKU was run). Carried: within kCarryDistance, every
// quantity compared. Transferred: farther, a gate's own rule carrying it (the MMQ gate's: at any
// distance; a value table never answers so). Incomplete: a quantity unknown on either side, so the
// distance can understate the difference. None: no donor (the unmeasured default).
enum class Disposition { Measured, Carried, Transferred, Incomplete, None };

[[nodiscard]] constexpr const char* disposition_name(Disposition d) noexcept {
    switch (d) {
    case Disposition::Measured:
        return "measured";
    case Disposition::Carried:
        return "carried";
    case Disposition::Transferred:
        return "transferred";
    case Disposition::Incomplete:
        return "incomplete";
    case Disposition::None:
        return "none";
    }
    return "none";
}

} // namespace ninferno::measured
