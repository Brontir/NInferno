// Ordinary decode round pacing: the steps of a round are a pure function of --decode-steps, the
// CPU quota fact and the allowed steps, whatever the host's timings (one step by default); a quiet
// host spins on a round's end, and a host that lags behind the GPU waits by spinning, or by
// blocking where its probes found that faster, until the lag stops; --decode-steps and
// --decode-wait fix the policy or are refused.

#include "models/qwen3_5/program/round_pacing.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using ninferno::models::qwen3_5::detail::RoundPacing;
using Clock = RoundPacing::Clock;
using Wait  = RoundPacing::Wait;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

Clock::time_point at(double seconds) {
    return Clock::time_point{} + std::chrono::duration_cast<Clock::duration>(
                                     std::chrono::duration<double>(1000.0 + seconds));
}

bool refused(auto&& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

bool steps_refused(const char* text) {
    return refused([&] { (void)ninferno::parse_decode_steps(text); });
}

bool wait_refused(const char* text) {
    return refused([&] { (void)ninferno::parse_decode_wait(text); });
}

} // namespace

int main() {
    int failures          = 0;
    constexpr double step = 0.020; // device seconds of one step (a 31B dense model at C=1)

    {
        // A quiet host: rounds end within microseconds of their device work.
        RoundPacing pacing(0, Wait::Auto, 32, 32);
        double now = 0.0;
        for (int round = 0; round < 500; ++round) {
            failures += check(pacing.steps() == 1, "a quiet host runs more than one step");
            failures += check(!pacing.block(at(now)), "a quiet host blocks");
            pacing.observe(1, step + 0.00005, step, at(now += step));
        }
        // One late round is a hiccup, not a stalled host.
        pacing.observe(1, step + 0.030, step, at(now += step + 0.030));
        for (int round = 0; round < 60; ++round) {
            pacing.observe(1, step + 0.0001, step, at(now += step));
        }
        pacing.observe(1, step + 0.030, step, at(now += step + 0.030));
        failures += check(!pacing.stalling(at(now)) && pacing.steps() == 1,
                          "lags a second apart marked the host as stalling");
    }

    {
        // A throttled host: the process stops for tens of milliseconds in every 100 ms.
        // The stall changes only the wait, never the steps: a timing may not choose a batch.
        RoundPacing pacing(0, Wait::Auto, 32, 32);
        double now = 0.0;
        pacing.observe(1, step + 0.040, step, at(now += step + 0.040));
        failures += check(!pacing.stalling(at(now)), "a single lag marked a stall");
        pacing.observe(1, step, step, at(now += step));
        pacing.observe(1, step + 0.035, step, at(now += step + 0.035));
        failures += check(pacing.stalling(at(now)) && pacing.steps() == 1,
                          "a stalling host changed its steps per round");
        failures += check(!pacing.block(at(now)), "a stalling host does not probe spinning first");

        // Lags keep recurring while the stall lasts; each one extends the hold.
        pacing.observe(1, step + 0.040, step, at(now += 3.0));
        failures += check(pacing.stalling(at(now + 4.9)), "a recurring lag did not hold");
        failures += check(!pacing.stalling(at(now + 5.1)) && !pacing.block(at(now + 5.1)),
                          "the stall outlived its hold");
    }

    {
        // --decode-wait block blocks on a quiet host too.
        RoundPacing pacing(0, Wait::Block, 32, 32);
        pacing.observe(1, step, step, at(0.0));
        failures += check(pacing.block(at(0.0)) && pacing.steps() == 1,
                          "--decode-wait block does not block alone");
    }

    {
        // A stalling host probes spinning, then blocking, blocks only if that ran more steps per
        // wall second, and probes again after a while, the kept wait first.
        for (const bool spin_faster : {true, false}) {
            // One step of a model that cannot run ahead; the rounds each wait kept after the first
            // probes.
            RoundPacing pacing(0, Wait::Auto, 32, 1);
            constexpr double device = 0.005;
            double now              = 0.0;
            int blocked             = 0;
            int spun                = 0;
            bool probed_again       = false;
            bool kept_first         = true;
            while (now < 36.0) {
                const bool block = pacing.block(at(now));
                const double gap = device / (block == spin_faster ? 0.15 : 0.20);
                now += gap;
                pacing.observe(1, gap, device, at(now));
                if (now > 2.5 && now < 31.5) { (block ? blocked : spun) += 1; }
                // The second probe phase tries the other wait; the first keeps the kept one.
                if (now > 32.2 && now < 32.9 && block == spin_faster) { kept_first = false; }
                if (now > 33.2 && now < 33.9 && block == spin_faster) { probed_again = true; }
            }
            failures += check(spin_faster ? blocked == 0 && spun > 0 : spun == 0 && blocked > 0,
                              "a stalling host did not keep the faster wait");
            failures += check(probed_again, "a stalling host did not probe the waits again");
            failures += check(kept_first, "a probe after a keep did not start with the kept wait");
        }
    }

    {
        // Blocking is kept only for a real gain: 1% faster keeps spinning. A kept wait whose rate
        // moves by more than kShift probes again at once: here the load under which blocking was
        // faster gives way to one under which spinning is.
        // The waits of the rounds between 3 and 5 s, and after 11 s: 'b' blocked only, 's' spun
        // only, 'm' both.
        const auto waits = [&](double block_busy, double later_block_busy) {
            RoundPacing pacing(0, Wait::Auto, 32, 1);
            constexpr double device = 0.005;
            double now              = 0.0;
            std::array<std::array<int, 2>, 2> counts{};
            while (now < 14.0) {
                const bool block  = pacing.block(at(now));
                const double busy = block ? (now > 5.0 ? later_block_busy : block_busy) : 0.15;
                const double gap  = device / busy;
                now += gap;
                pacing.observe(1, gap, device, at(now));
                if (now > 3.0 && now < 5.0) { counts[0][block ? 0 : 1] += 1; }
                if (now > 11.0) { counts[1][block ? 0 : 1] += 1; }
            }
            std::string out;
            for (const auto& count : counts) {
                out += count[1] == 0 ? 'b' : count[0] == 0 ? 's' : 'm';
            }
            return out;
        };
        failures += check(waits(0.1515, 0.1515) == "ss", "a stalling host blocked for a 1% gain");
        failures += check(waits(0.20, 0.20) == "bb", "a stalling host did not keep blocking");
        failures += check(waits(0.20, 0.17) == "bb" && waits(0.20, 0.06) == "bs",
                          "a kept wait did not give way to the faster one when, and only when, its "
                          "rate fell by more than kShift");
    }

    {
        // The steps are a pure function of the setting and the allowed steps: no sequence of
        // rounds (quiet, lagging, gapped, slow) moves them.
        const auto steps_after = [&](std::uint32_t setting, std::uint32_t allowed, double lag,
                                     double device) {
            RoundPacing pacing(setting, Wait::Auto, 32, allowed);
            std::uint32_t seen = pacing.steps();
            bool constant      = true;
            double now         = 0.0;
            for (int round = 0; round < 400; ++round) {
                const std::uint32_t steps = pacing.steps();
                constant                  = constant && steps == seen;
                now += steps * device + lag;
                pacing.observe(steps, steps * device + lag, steps * device, at(now));
            }
            return constant ? seen : 0U;
        };
        for (const double lag : {0.0, 0.00005, 0.035, 0.5}) {
            for (const double device : {0.001, step, 0.2}) {
                failures += check(steps_after(0, 32, lag, device) == 1 &&
                                      steps_after(0, 1, lag, device) == 1 &&
                                      steps_after(8, 32, lag, device) == 8 &&
                                      steps_after(8, 4, lag, device) == 4 &&
                                      steps_after(32, 32, lag, device) == 32 &&
                                      steps_after(32, 1, lag, device) == 1,
                                  "a round's steps depended on the host's timings");
            }
        }
    }

    {
        // Rounds longer than a second (a throttled host that spins through 13 fixed steps) still
        // count, and a probe phase ends after kProbeLimitSeconds whatever its rounds. Spinning here
        // takes 0.5 s and 1.5 s by turns, blocking 0.9 s: blocking is faster, which only the long
        // spinning rounds show, and every round submitted after 6.5 s must block.
        RoundPacing pacing(13, Wait::Auto, 32, 32);
        constexpr double device = 13 * step;
        double now              = 0.0;
        int spun_rounds         = 0;
        int late_spins          = 0;
        while (now < 14.0) {
            const std::uint32_t steps = pacing.steps();
            const bool block          = pacing.block(at(now));
            const double round        = block ? 0.9 : (++spun_rounds % 2 == 0 ? 1.5 : 0.5);
            if (!block && now > 6.5) { ++late_spins; }
            now += round;
            pacing.observe(steps, round, device, at(now));
        }
        failures += check(late_spins == 0 && pacing.block(at(now)),
                          "rounds longer than a second kept a probe from ending or from counting");
    }

    {
        // Time the Program spent on other work (a request that ended, the next one's prefill)
        // before a round counts for neither wait: here every fifth blocked round follows 100 ms
        // of it (a request boundary that fell in the blocking probe), which would make blocking
        // look slower than spinning.
        RoundPacing pacing(0, Wait::Auto, 32, 1);
        constexpr double device = 0.005;
        double now              = 0.0;
        int round               = 0;
        while (now < 4.0) {
            const bool block         = pacing.block(at(now));
            const double gap         = device / (block ? 0.20 : 0.15);
            const bool follows_round = !block || ++round % 5 != 0;
            now += gap + (follows_round ? 0.0 : 0.1);
            pacing.observe(1, gap, device, at(now), follows_round);
        }
        failures += check(pacing.block(at(now)),
                          "other work between rounds counted against blocking");
    }

    {
        // A host that blocks well stops lagging, and its stall ends; when the lag returns, the
        // stall resumes the kept wait without probing again.
        RoundPacing pacing(0, Wait::Auto, 32, 1);
        constexpr double device = 0.005;
        double now              = 0.0;
        const auto lagging      = [&](bool blocked_better) {
            const bool block = pacing.block(at(now));
            const double gap = device / (block == blocked_better ? 0.20 : 0.15);
            now += gap;
            pacing.observe(1, gap, device, at(now));
        };
        while (now < 3.0) { lagging(true); }
        failures += check(pacing.block(at(now)), "a stalling host did not keep blocking");
        // As fast as before, but without lag.
        while (now < 10.0) {
            now += device / 0.20;
            pacing.observe(1, device, device, at(now));
        }
        failures += check(!pacing.stalling(at(now)) && !pacing.block(at(now)),
                          "a host that stopped lagging still blocks");
        lagging(true);
        lagging(true);
        failures += check(pacing.stalling(at(now)) && pacing.block(at(now)),
                          "a returning stall did not resume blocking");
    }

    {
        RoundPacing fixed(8, Wait::Auto, 32, 32);
        failures += check(fixed.steps() == 8, "--decode-steps 8 is not honoured");
        RoundPacing capped(8, Wait::Auto, 32, 4);
        failures += check(capped.steps() == 4, "fixed steps exceed the allowed steps");
        RoundPacing off(1, Wait::Spin, 32, 32);
        off.observe(1, 0.1, 0.02, at(0.0));
        off.observe(1, 0.1, 0.02, at(0.1));
        failures += check(off.steps() == 1, "--decode-steps 1 runs ahead");
        failures += check(!off.block(at(0.1)), "--decode-wait spin blocks");
        failures += check(steps_refused("") && steps_refused("0") && steps_refused("33") &&
                              steps_refused("4x") && steps_refused("-1") &&
                              steps_refused("spin") && wait_refused("") &&
                              wait_refused("sleep") && wait_refused("blocking") &&
                              refused([] { (void)RoundPacing(33, Wait::Auto, 32, 32); }),
                          "an invalid setting was accepted");
    }

    {
        // The CPU quota fact sets the default: kQuotaSteps where the engine can run ahead, else
        // one step; --decode-steps N replaces it.
        failures += check(RoundPacing(0, Wait::Auto, 32, 32, true).steps() == 16 &&
                              RoundPacing(0, Wait::Auto, 32, 4, true).steps() == 4 &&
                              RoundPacing(0, Wait::Auto, 32, 1, true).steps() == 1 &&
                              RoundPacing(0, Wait::Auto, 32, 32, false).steps() == 1 &&
                              RoundPacing(8, Wait::Auto, 32, 32, true).steps() == 8 &&
                              RoundPacing(1, Wait::Auto, 32, 32, true).steps() == 1,
                          "the CPU quota default is not the declared step count");
        // --decode-steps auto is the default: the quota fact's step count, not a fixed one.
        const RoundPacing automatic(ninferno::parse_decode_steps("auto"), Wait::Auto, 32, 32, true);
        const RoundPacing quiet(ninferno::parse_decode_steps("auto"), Wait::Auto, 32, 32, false);
        failures += check(automatic.steps() == 16 && !automatic.fixed_steps() && quiet.steps() == 1,
                          "--decode-steps auto is not the CPU quota default");
    }

    std::cout << (failures ? "FAIL" : "OK") << " round_pacing\n";
    return failures ? 1 : 0;
}
