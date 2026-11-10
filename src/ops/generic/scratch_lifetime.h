#pragma once
// Scratch read lifetimes of generic launches (#309): which launch touches which region of the
// shared workspace, from which event to which, and a check that no launch that starts under its
// predecessor (programmatic dependent launch) writes what that predecessor still reads, or reads
// what it still writes.
//
// The events, per launch:
//   - first touch: BeforeWait (the launch may touch the region while its predecessor still runs:
//     before griddepcontrol.wait, or in a launch the stream does not order after it) or AfterWait
//     (only once the predecessor has completed: stream order, or after the wait).
//   - last touch: AtTrigger (finished before the launch lets the next one begin) or AtExit (the
//     launch still uses it until it completes).
// PDL release is not completion: a successor that starts at the trigger sees a predecessor's
// AtExit uses as live, and a predecessor's output as not yet written. Only AfterWait touches are
// ordered behind the predecessor's completion.
//
// This is a description and a check, not a scheduler: nothing here changes what run_route or the
// segmented chain launch. The release path keeps launch_overlap's refusals (the pre-quantized R8
// form does not release because the next quantizer could take the region it still reads); this
// header states that reason as data and tests it. Host only, deterministic, no device queries.
#include "ops/generic/generic_gemm.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace ninferno::ops::detail::generic {

// A byte range of the workspace. `slot` names it for messages only; overlap is by bytes.
struct ScratchRegion {
    std::uint32_t slot = 0;
    std::size_t offset = 0;
    std::size_t bytes  = 0;
};

[[nodiscard]] constexpr bool regions_overlap(const ScratchRegion& a,
                                             const ScratchRegion& b) noexcept {
    return a.bytes != 0 && b.bytes != 0 && a.offset < b.offset + b.bytes &&
           b.offset < a.offset + a.bytes;
}

enum class ScratchAccess { Read, Write };
enum class ScratchFirst { BeforeWait, AfterWait };
enum class ScratchLast { AtTrigger, AtExit };

struct ScratchUse {
    ScratchRegion region;
    ScratchAccess access = ScratchAccess::Read;
    ScratchFirst first   = ScratchFirst::AfterWait;
    ScratchLast last     = ScratchLast::AtExit;
};

inline constexpr std::size_t kMaxScratchUses = 4;

// One launch of a chain. starts_early: it begins under its predecessor (a join or a prefetch);
// releases: it lets its successor begin at its trigger (before it completes).
struct LaunchLifetime {
    std::string_view name;
    bool starts_early = false;
    bool releases     = false;
    std::array<ScratchUse, kMaxScratchUses> uses{};
    std::size_t use_count = 0;

    // Returns false when the table is full (a description bug, caught by the caller's test).
    bool add(const ScratchUse& use) noexcept {
        if (use_count == uses.size()) return false;
        uses[use_count++] = use;
        return true;
    }
};

enum class ScratchHazard {
    WriteAfterRead,  // writes a region the predecessor reads until it exits
    ReadAfterWrite,  // reads a region the predecessor writes (its output is not yet there)
    WriteAfterWrite, // both write the region
};

[[nodiscard]] constexpr std::string_view scratch_hazard_name(ScratchHazard h) noexcept {
    switch (h) {
    case ScratchHazard::WriteAfterRead:
        return "write_after_read";
    case ScratchHazard::ReadAfterWrite:
        return "read_after_write";
    case ScratchHazard::WriteAfterWrite:
        return "write_after_write";
    }
    return "none";
}

struct ScratchViolation {
    std::size_t launch      = 0; // the launch that starts early
    std::size_t predecessor = 0; // the earlier launch still running
    ScratchHazard hazard    = ScratchHazard::WriteAfterRead;
};

// The hazard between a running predecessor use `before` and a use `now` of a launch that starts
// under it, or nothing.
[[nodiscard]] constexpr std::optional<ScratchHazard> use_hazard(const ScratchUse& before,
                                                                const ScratchUse& now) noexcept {
    if (now.first == ScratchFirst::AfterWait) return std::nullopt; // ordered behind its completion
    if (!regions_overlap(before.region, now.region)) return std::nullopt;
    if (before.access == ScratchAccess::Read) {
        if (now.access == ScratchAccess::Read) return std::nullopt;
        // A read finished at the predecessor's trigger is over before the successor may begin.
        if (before.last == ScratchLast::AtTrigger) return std::nullopt;
        return ScratchHazard::WriteAfterRead;
    }
    // A predecessor's write is a hazard whatever its last touch: writes made before the trigger
    // are visible to the dependent only after its griddepcontrol.wait (the PDL memory model), so
    // a read or write before that wait races them even where the write is AtTrigger.
    return now.access == ScratchAccess::Read ? ScratchHazard::ReadAfterWrite
                                             : ScratchHazard::WriteAfterWrite;
}

// Every launch that starts early is checked against the launch before it and, conservatively,
// against each earlier one while the launches in between also started early (a chain of early
// starts may all be running at once). A predecessor that does not release triggers implicitly at
// its exit, so a launch under it starts at its completion: no overlap, nothing to check.
// Assumption of the look-back: every early-starting kernel waits for its predecessor's full
// completion (griddepcontrol.wait) before it exits (generic_gemm.cuh, generic_r8.cuh), so a launch
// is never concurrent with the one two back unless the one between also released. This is read
// from the kernels, not derived here. Real kernels call trigger_dependents at their start, or just
// after the wait under prefetch, so AtTrigger is reachable only from constructed cases.
// Returns the first violation in launch order, or nothing.
[[nodiscard]] inline std::optional<ScratchViolation>
first_scratch_violation(const std::vector<LaunchLifetime>& launches) noexcept {
    for (std::size_t i = 1; i < launches.size(); ++i) {
        const LaunchLifetime& now = launches[i];
        if (!now.starts_early) continue;
        for (std::size_t j = i; j-- > 0;) {
            const LaunchLifetime& before = launches[j];
            if (!before.releases) {
                // The launch after it starts at its completion, so none of its reads can race;
                // its writes are visible only after the dependent's wait (the PDL memory model),
                // so the immediate predecessor's writes still meet this launch's early uses.
                if (j + 1 == i) {
                    for (std::size_t a = 0; a < before.use_count; ++a) {
                        if (before.uses[a].access != ScratchAccess::Write) continue;
                        for (std::size_t b = 0; b < now.use_count; ++b) {
                            if (const auto h = use_hazard(before.uses[a], now.uses[b])) {
                                return ScratchViolation{i, j, *h};
                            }
                        }
                    }
                }
                break;
            }
            for (std::size_t a = 0; a < before.use_count; ++a) {
                for (std::size_t b = 0; b < now.use_count; ++b) {
                    if (const auto h = use_hazard(before.uses[a], now.uses[b])) {
                        return ScratchViolation{i, j, *h};
                    }
                }
            }
            if (!before.starts_early) break; // it began after its own predecessor completed
        }
    }
    return std::nullopt;
}

// A launch under the overlap launch_overlap grants it, with the prepared-activation region it
// reads. `rule` is prepared_rule(route, weight) of the launch's route (nullopt: the kernel reads x
// as BF16 or quantizes it inside itself, so it reads no prepared scratch); the caller passes it so
// this description cannot drift from the representation rule (#308). A join or prefetch only
// starts early where the device has PDL: without it the launch falls back to stream order
// (generic_gemm_launch.cuh, generic_r8.cuh). The read is stamped AfterWait and AtExit by a
// convention, not derived from the kernels: a consumer of prepared bytes is launched in stream
// order after its quantizer and reads them to its end.
[[nodiscard]] inline LaunchLifetime
launch_lifetime(std::string_view name, const LaunchOverlap& overlap, bool pdl,
                const std::optional<PreparedRule>& rule,
                const std::optional<ScratchRegion>& prepared) noexcept {
    LaunchLifetime out;
    out.name = name;
    // `pdl` is the argument launch_overlap was given; the refusals it recorded must agree with it.
    assert(!(overlap.releases() || overlap.prefetches()) || pdl);
    assert(
        (overlap.release != OverlapRefusal::NoPdl && overlap.prefetch != OverlapRefusal::NoPdl) ||
        !pdl);
    out.starts_early = pdl && (overlap.joins() || overlap.prefetches());
    // A prefetching register MMA triggers its dependents right after its own wait whether or not
    // the release was asked (generic_gemm.cuh, the prefetch path's pdl::trigger_dependents), so
    // the launch after it may start while it still runs: it releases. An R8 prefetch triggers only
    // where its release was asked (generic_r8.cuh), which overlap.releases() already says; marking
    // every prefetch as releasing over-marks R8, the safe side (more hazards checked, none missed).
    out.releases = overlap.releases() || (pdl && overlap.prefetches());
    if (prepared && rule) {
        out.add({*prepared, ScratchAccess::Read, ScratchFirst::AfterWait, ScratchLast::AtExit});
    }
    return out;
}

// A quantizer kernel writing `region` in stream order (it runs after everything before it has
// completed), as the pre-quantized R8 form and the ggml vector kernel's call prepare activations.
[[nodiscard]] inline LaunchLifetime quantizer_lifetime(std::string_view name,
                                                       const ScratchRegion& region) noexcept {
    LaunchLifetime out;
    out.name = name;
    out.add({region, ScratchAccess::Write, ScratchFirst::AfterWait, ScratchLast::AtExit});
    return out;
}

} // namespace ninferno::ops::detail::generic
