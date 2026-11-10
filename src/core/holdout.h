#pragma once

// The holdout record of a fitted constant. Decisions come from the chip's facts; a constant that is
// fitted instead is an exception, and it carries where it was fitted and on what data it was
// checked that the fit did not see. The record has three states, so a test can tell a constant
// whose holdout is declared absent from one that was forgotten:
//   Unset    nothing was written (the default: a fitted constant with it fails its test);
//   Checked  boards or models left out of the fit, a metric and the error seen on them;
//   None     the constant was fitted and checked on the same data, said plainly with the reason.
// `source` names the record (a DECISIONS entry, an issue, the comment at the constant's use).

#include <string_view>

namespace ninferno {

enum class HoldoutKind { Unset, Checked, None };

struct Holdout {
    HoldoutKind kind = HoldoutKind::Unset;
    std::string_view fitted_on;
    std::string_view held_out;
    std::string_view metric; // for None: what the record holds instead of a holdout
    std::string_view error;
    std::string_view source;
    std::string_view note;

    // Left out of the fit and checked: what the check was run on, the metric, the error seen.
    static constexpr Holdout checked(std::string_view fitted_on, std::string_view held_out,
                                     std::string_view metric, std::string_view error,
                                     std::string_view source, std::string_view note = {}) {
        return {HoldoutKind::Checked, fitted_on, held_out, metric, error, source, note};
    }

    // No held-out data: fitted and checked on the same boards or models. `why` says what the
    // record does hold instead.
    static constexpr Holdout none(std::string_view fitted_on, std::string_view why,
                                  std::string_view source, std::string_view note = {}) {
        return {HoldoutKind::None,
                fitted_on,
                "none: fitted and checked on the same data",
                why,
                {},
                source,
                note};
    }

    // A record that says enough to be read: the fields its kind requires are non-empty.
    constexpr bool complete() const {
        switch (kind) {
        case HoldoutKind::Unset:
            return false;
        case HoldoutKind::Checked:
            return !fitted_on.empty() && !held_out.empty() && !metric.empty() && !error.empty() &&
                   !source.empty();
        case HoldoutKind::None:
            return !fitted_on.empty() && !metric.empty() && !source.empty();
        }
        return false;
    }
};

} // namespace ninferno
