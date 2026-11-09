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
    StartsWithoutRelease, // starts under a predecessor that never lets it begin
    WriteAfterRead,       // writes a region the predecessor reads until it exits
    ReadAfterWrite,       // reads a region the predecessor writes (its output is not yet there)
    WriteAfterWrite,      // both write the region
};

[[nodiscard]] constexpr std::string_view scratch_hazard_name(ScratchHazard h) noexcept {
    switch (h) {
    case ScratchHazard::StartsWithoutRelease:
        return "starts_without_release";
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
    ScratchHazard hazard    = ScratchHazard::StartsWithoutRelease;
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
    return now.access == ScratchAccess::Read ? ScratchHazard::ReadAfterWrite
                                             : ScratchHazard::WriteAfterWrite;
}

// Every launch that starts early is checked against the launch before it and, conservatively,
// against each earlier one while the launches in between also started early (a chain of early
// starts may all be running at once). Returns the first violation in launch order, or nothing.
[[nodiscard]] inline std::optional<ScratchViolation>
first_scratch_violation(const std::vector<LaunchLifetime>& launches) noexcept {
    for (std::size_t i = 1; i < launches.size(); ++i) {
        const LaunchLifetime& now = launches[i];
        if (!now.starts_early) continue;
        if (!launches[i - 1].releases) {
            return ScratchViolation{i, i - 1, ScratchHazard::StartsWithoutRelease};
        }
        for (std::size_t j = i; j-- > 0;) {
            const LaunchLifetime& before = launches[j];
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

// A launch of `resolved` under the overlap launch_overlap grants it, with the prepared-activation
// region it reads when it has one (the pre-quantized R8 form's quantizer output; the ggml vector
// kernel's shared q8_1 activation). Every such kernel reads the region until it exits and is
// launched in stream order after the quantizer, so its read is AfterWait.
[[nodiscard]] inline LaunchLifetime
launch_lifetime(std::string_view name, const ResolvedRoute& resolved, const LaunchOverlap& overlap,
                const std::optional<ScratchRegion>& prepared) noexcept {
    LaunchLifetime out;
    out.name         = name;
    out.starts_early = overlap.joins() || overlap.prefetches();
    out.releases     = overlap.releases();
    const bool reads_prepared =
        resolved.r8_prequantized || resolved.executed.family == Family::GgmlVec;
    if (prepared && reads_prepared) {
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
