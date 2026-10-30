#pragma once

// A fitted calibration, not a chip fact: rows routed apart read this share of the experts
// independent routing would add. Fitted on the drafter registry's speculation floors
// (audit1018/x3); docs/maintainer/choice-notes/DECISIONS.md 53 (the round cost's B(r)). Read by the
// speculation round cost (spec/round_cost.cpp moe_read_fraction) and the configuration choice's
// step model (product/choice/choice.cpp), one definition for both. Header-only: the choice links
// no spec library.

namespace ninferno::spec {

inline constexpr double kMoeRoutedShare = 0.30;

} // namespace ninferno::spec
