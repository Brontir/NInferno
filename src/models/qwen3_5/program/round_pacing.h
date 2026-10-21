#pragma once

// How a decode round uses the host: how many steps an ordinary round runs back to back
// (ProgramImpl::launch_ordinary) and whether the host spins or blocks on a round's end.
//
// The step count is a fixed declared rule, never a timing: it decides which requests share each
// step's batch (one step per round admits and retires requests between steps, a run-ahead round
// keeps its batch for all its steps), so it must be the same on every machine for the same
// requests (docs/maintainer/deterministic-decisions.md). It reads machine facts, read once at
// startup: the process's CPU bandwidth quota (host_cpu_quota.h) against the logical cores; the
// quota limits the host when it is below min(4, cores) CPUs (HostCpuQuota::limits_run_ahead,
// which says why). A container's CFS limit stops every thread of the cgroup for the rest of its
// 100 ms period once the quota is used; a decode that waits for each step then leaves the GPU
// idle for the stall (30-40 % of the time on throttled lab containers). Under such a quota a round
// runs kQuotaSteps steps where the engine can run ahead, the later steps feeding on the sampled
// tokens on the device so the GPU keeps running while the host is stopped; without one it runs one
// step, which publishes each token as soon as it exists and runs nothing past a stop.
// NINFERNO_DECODE_STEPS=N replaces that default. A round runs at most allowed_steps: 1 where rows
// past a stop could not stay invisible (recurrent state, a full sliding ring, a speculative
// engine), the model's uncommitted-row headroom otherwise.
//
// The wait is the only timing-driven choice, and it cannot change a round's output bytes or the
// next round's batch: the round is fully submitted before the host waits, and the wait only moves
// when the host returns to a boundary whose contents it does not choose. The policy measures the
// host's lag in every round (its wall time beyond the round's device time); two lagging rounds
// close together mark the host as stalling until a hold period after the last one. While it
// stalls the host waits for the round's end by spinning, or by blocking (a blocking-sync event)
// where that decodes faster. Blocking spends almost none of the quota, which usually wins; but
// when many runnable threads drain the quota early in every period, a blocked host can wake too
// late to get any of it, and a spinning host does better. The policy probes both waits for
// kProbeSeconds each (the steps of their rounds over the rounds' finish-to-finish wall time),
// spinning first when a stall begins and the kept wait first afterwards, blocks for kKeepSeconds
// only if that was faster by kBlockGain, and probes again. It probes again at once when the kept
// wait's rate moves by more than kShift (the load changed). The choice outlasts a pause in the
// stall (a host that blocks well stops lagging, and its stall ends until the lag returns), so a
// returning stall resumes it at once.
//
// A speculative engine's rounds (a verify and the drafting that follows it) cannot run ahead: the
// next round's drafts and rows depend on this round's acceptance. They run one step and use only
// the wait, each round counting as one step of the probe's rate: the probe compares rounds, not
// tokens, per second. A round's device time does not depend on how many of its drafts are
// accepted, while its tokens do, and acceptance follows the text, not the wait.

#include "core/parse.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninferno::models::qwen3_5::detail {

class RoundPacing {
public:
    using Clock = std::chrono::steady_clock;

    // A round whose wall time exceeds its device time by more than this missed its wake-up.
    static constexpr double kLagSeconds = 0.002;
    // Two lagging rounds at most this far apart mark the host as stalling,
    static constexpr double kPairSeconds = 1.0;
    // until this long after the last lagging round.
    static constexpr double kHoldSeconds = 5.0;
    // While the host stalls, each wait is probed for at least this long and kProbeRounds rounds
    // (a run-ahead round can last a quarter of a second, and throttling periods make the rate of a
    // few rounds noisy), or for kProbeLimitSeconds when rounds are slower than that. Blocking is
    // kept for kKeepSeconds if its rate beat spinning's by kBlockGain, spinning otherwise. The
    // kept wait's rate is measured in windows of kShiftSeconds, and one that moves by more than
    // kShift ends the keep: a change of load moves it by far more (64 busy loops to 250 cut
    // Qwen3.6's blocked rate to a fifth), throttling periods and request boundaries by less. A
    // round submitted more than kGapSeconds after the previous one ended followed an idle engine
    // and counts for nothing.
    static constexpr double kProbeSeconds       = 1.0;
    static constexpr std::uint32_t kProbeRounds = 8;
    static constexpr double kProbeLimitSeconds  = 3.0;
    static constexpr double kKeepSeconds        = 30.0;
    static constexpr double kBlockGain          = 0.03;
    static constexpr double kShiftSeconds       = 2.0;
    static constexpr double kShift              = 0.5;
    static constexpr double kGapSeconds         = 1.0;

    enum class Wait : std::uint8_t {
        Auto,  // spin, or block while the host stalls where the probe found that faster
        Spin,  // wait in the CUDA synchronization schedule only
        Block, // block on every round's end
    };

    // Steps per round under a limiting CPU quota: a 16-step round of a 5-30 ms step lasts
    // 80-480 ms, about one CFS period (100 ms) or more, so a stopped host finds the GPU still
    // busy. The step cost is not known before the first round, so the count cannot follow it
    // without a measurement; a fast model's 5 ms steps would want min(32, 100 ms / step) = 20, a
    // slow one's 30 ms steps 4, and 16 covers both with one period's stall at most. The value
    // awaits calibration on lab containers (docs/maintainer/deterministic-decisions.md, known
    // gaps); a host whose stall it does not hide sets --decode-steps N.
    static constexpr std::uint32_t kQuotaSteps = 16;

    // The default steps per round from the facts alone: kQuotaSteps where the CPU quota limits
    // the host (HostCpuQuota::limits_run_ahead), else 1, within the allowed steps.
    [[nodiscard]] static constexpr std::uint32_t
    default_steps(bool quota_limits, std::uint32_t allowed_steps) noexcept {
        return quota_limits ? std::clamp(kQuotaSteps, 1U, std::max(allowed_steps, 1U)) : 1U;
    }

    // `steps_setting` is NINFERNO_DECODE_STEPS: null (or "auto", which warns: auto_ignored())
    // runs default_steps(); an integer N in [1, maximum_steps] runs N steps in every round that
    // can. `cpu_quota`: the process's CPU quota limits the host (HostCpuQuota::limits_run_ahead).
    // `wait_setting` is NINFERNO_DECODE_WAIT: null or "auto", "spin" or "block". A round runs at
    // most `allowed_steps` steps (1 where rows past a stop could not stay invisible).
    RoundPacing(const char* steps_setting, const char* wait_setting, std::uint32_t maximum_steps,
                std::uint32_t allowed_steps, bool cpu_quota = false)
        : allowed_steps_(std::clamp(allowed_steps, 1U, std::max(maximum_steps, 1U))),
          default_steps_(default_steps(cpu_quota, allowed_steps_)) {
        if (maximum_steps == 0) { throw std::invalid_argument("a decode round runs a step"); }
        if (steps_setting != nullptr && std::string_view(steps_setting) == "auto") {
            auto_ignored_ = true;
        } else if (steps_setting != nullptr) {
            const auto value = parse_exact<std::uint32_t>(steps_setting);
            if (!value || *value == 0 || *value > maximum_steps) {
                throw std::invalid_argument("NINFERNO_DECODE_STEPS must be an integer in [1, " +
                                            std::to_string(maximum_steps) + "]");
            }
            fixed_steps_ = *value;
        }
        if (wait_setting != nullptr) {
            const std::string_view wait(wait_setting);
            if (wait == "spin") {
                wait_ = Wait::Spin;
            } else if (wait == "block") {
                wait_ = Wait::Block;
            } else if (wait != "auto") {
                throw std::invalid_argument("NINFERNO_DECODE_WAIT must be auto, spin or block");
            }
        }
    }

    // NINFERNO_DECODE_STEPS as an integer, if it gave one.
    [[nodiscard]] std::optional<std::uint32_t> fixed_steps() const noexcept { return fixed_steps_; }

    // NINFERNO_DECODE_STEPS=auto was given: the former adaptive mode, now the default.
    [[nodiscard]] bool auto_ignored() const noexcept { return auto_ignored_; }

    // The steps every round asks for, a function of the setting and the allowed steps alone;
    // launch_ordinary may run fewer (a row's remaining limit).
    [[nodiscard]] std::uint32_t steps() const noexcept {
        return fixed_steps_ ? std::min(*fixed_steps_, allowed_steps_) : default_steps_;
    }

    [[nodiscard]] bool stalling(Clock::time_point now) const noexcept {
        return hold_until_ && now < *hold_until_;
    }

    // Whether the host blocks on the end of a round submitted at `now` rather than spinning.
    [[nodiscard]] bool block(Clock::time_point now) const noexcept {
        if (wait_ != Wait::Auto) { return wait_ == Wait::Block; }
        return stalling(now) && stall_wait_ == Wait::Block;
    }

    // One finished round of `steps` steps: host wall time from the round's timer start to the end
    // of its wait, and the device time from that start event to the stop event after the round's
    // last kernel. The start is recorded on an idle stream before the host prepares the round's
    // launches, so its submission counts as device time and the lag is the wait's late wake-up
    // alone. `follows_round`: the Program did nothing between the previous round and this one (no
    // prefill, no request started or finished), so the wall time since the previous round's end is
    // this round's alone.
    void observe(std::uint32_t steps, double wall_seconds, double device_seconds,
                 Clock::time_point now, bool follows_round = true) noexcept {
        if (steps == 0 || !(device_seconds > 0.0) || !(wall_seconds >= 0.0)) { return; }
        // A round the host submitted while stalling waited as the probe chose, unless the host
        // was idle before it (the time since the previous round's end beyond this round's own).
        const Clock::time_point submitted = now - seconds(wall_seconds);
        if (stalling(now) && probe_ != Probe::Idle && follows_round && last_finish_ &&
            submitted - *last_finish_ <= seconds(kGapSeconds)) {
            const std::size_t wait = stall_wait_ == Wait::Block ? 0 : 1;
            steps_[wait] += steps;
            span_[wait] += std::chrono::duration<double>(now - *last_finish_).count();
            rounds_[wait] += 1;
        }
        last_finish_ = now;
        if (wall_seconds - device_seconds > kLagSeconds) {
            if (stalling(now) || (last_lag_ && now - *last_lag_ <= seconds(kPairSeconds))) {
                hold_until_ = now + seconds(kHoldSeconds);
            }
            last_lag_ = now;
        }
        if (wait_ == Wait::Auto && stalling(now)) { advance_probe(now); }
    }

private:
    enum class Probe : std::uint8_t { Idle, First, Second, Keep };

    [[nodiscard]] static Clock::duration seconds(double value) noexcept {
        return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(value));
    }

    // Steps per wall second under a wait (0: block, 1: spin) in the current probe or window.
    [[nodiscard]] double rate(std::size_t wait) const noexcept {
        return span_[wait] > 0.0 ? steps_[wait] / span_[wait] : 0.0;
    }

    void clear_counts() noexcept {
        steps_  = {};
        span_   = {};
        rounds_ = {};
    }

    // Probes `first`, then the other wait.
    void start_probe(Clock::time_point now, Wait first) noexcept {
        clear_counts();
        probe_       = Probe::First;
        stall_wait_  = first;
        phase_start_ = now;
        probe_end_   = now + seconds(kProbeSeconds);
    }

    // Whether the current probe phase or keep window, ending at `end`, has measured `wait` long
    // enough.
    [[nodiscard]] bool measured(std::size_t wait, Clock::time_point now,
                                Clock::time_point end) const noexcept {
        if (now < end) { return false; }
        return rounds_[wait] >= kProbeRounds || now - phase_start_ >= seconds(kProbeLimitSeconds);
    }

    void advance_probe(Clock::time_point now) noexcept {
        const std::size_t current = stall_wait_ == Wait::Block ? 0 : 1;
        switch (probe_) {
        case Probe::Idle:
            start_probe(now, Wait::Spin);
            break;
        case Probe::First:
            if (measured(current, now, probe_end_)) {
                probe_       = Probe::Second;
                stall_wait_  = stall_wait_ == Wait::Block ? Wait::Spin : Wait::Block;
                phase_start_ = now;
                probe_end_   = now + seconds(kProbeSeconds);
            }
            break;
        case Probe::Second:
            if (measured(current, now, probe_end_)) {
                const bool block = rate(0) > rate(1) * (1.0 + kBlockGain);
                probe_           = Probe::Keep;
                stall_wait_      = block ? Wait::Block : Wait::Spin;
                kept_rate_       = rate(block ? 0 : 1);
                probe_end_       = now + seconds(kKeepSeconds);
                phase_start_     = now;
                window_end_      = now + seconds(kShiftSeconds);
                clear_counts();
            }
            break;
        case Probe::Keep: {
            if (now >= probe_end_) {
                start_probe(now, stall_wait_);
                break;
            }
            if (!measured(current, now, window_end_) || rounds_[current] == 0) { break; }
            const double moved = kept_rate_ > 0.0 ? rate(current) / kept_rate_ : 0.0;
            if (!(moved > 1.0 - kShift && moved < 1.0 + kShift)) {
                start_probe(now, stall_wait_);
                break;
            }
            phase_start_ = now;
            window_end_  = now + seconds(kShiftSeconds);
            clear_counts();
            break;
        }
        }
    }

    std::uint32_t allowed_steps_ = 1;
    std::uint32_t default_steps_ = 1;
    std::optional<std::uint32_t> fixed_steps_;
    bool auto_ignored_ = false;
    Wait wait_         = Wait::Auto;
    std::optional<Clock::time_point> last_lag_;
    std::optional<Clock::time_point> hold_until_;
    // The wait probe while the host stalls: steps, finish-to-finish wall seconds and rounds of
    // each wait (0: block, 1: spin) in the current probe, or of the kept wait in its window.
    Probe probe_     = Probe::Idle;
    Wait stall_wait_ = Wait::Spin;
    Clock::time_point probe_end_{};
    Clock::time_point phase_start_{};
    Clock::time_point window_end_{};
    double kept_rate_ = 0.0; // steps per second under the kept wait when it was chosen
    std::optional<Clock::time_point> last_finish_;
    std::array<double, 2> steps_{};
    std::array<double, 2> span_{};
    std::array<std::uint32_t, 2> rounds_{};
};

} // namespace ninferno::models::qwen3_5::detail
