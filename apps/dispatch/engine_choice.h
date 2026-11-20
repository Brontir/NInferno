#pragma once

// Which of a release's engines runs a GPU (apps/dispatch/main.cpp, whose top describes the rule):
// header only, so a host test checks it (tests/test_engine_choice.cpp).

#include "ops/softmax_attention/gemma4/row_codec.h"

#include <optional>
#include <string>
#include <vector>

namespace ninferno::dispatch {

struct Engine {
    std::string tag; // "120a", "89", ...
    int arch      = 0;
    bool specific = false;
    bool ptx      = false;
};

// The engine for a GPU of compute capability cc, if any (the rule: the top of
// apps/dispatch/main.cpp).
inline std::optional<Engine> choose(const std::vector<Engine>& engines, int cc) {
    const auto pick = [&](auto&& runs) -> std::optional<Engine> {
        std::optional<Engine> best;
        for (const auto& e : engines) {
            if (runs(e) && (!best || e.arch > best->arch)) { best = e; }
        }
        return best;
    };
    if (auto e = pick([&](const Engine& e) { return e.specific && e.arch == cc; })) { return e; }
    // A plain engine of the same major runs a newer minor only where both write the same KV row
    // codes (int8 on sm_75..88, E4M3 from sm_89: gemma4_rows_int8_on): the planner prices the
    // card's codec, so an 86 engine on an sm_89 card would run rows its plan does not describe
    // (#991). A release without the 89 engine then has none for Ada.
    if (auto e = pick([&](const Engine& e) {
            return !e.specific && e.arch / 10 == cc / 10 && e.arch <= cc &&
                   ninferno::ops::gemma4_rows_int8_on(e.arch) ==
                       ninferno::ops::gemma4_rows_int8_on(cc);
        })) {
        return e;
    }
    // PTX the same way: an engine compiled for the newer card writes its own tree's row codes.
    return pick([&](const Engine& e) {
        return e.ptx && e.arch <= cc &&
               ninferno::ops::gemma4_rows_int8_on(e.arch) == ninferno::ops::gemma4_rows_int8_on(cc);
    });
}

// Why choose() found no engine for cc, for the message: a plain engine of the same major and an
// older minor that writes other KV row codes (an 86 engine for an sm_89 card, #991; a 70 one for an
// sm_75 card), a GPU older than every engine, or none of the release's engines runs it.
inline std::string no_engine_reason(const std::vector<Engine>& engines, int cc) {
    bool older = !engines.empty();
    for (const auto& e : engines) {
        if (!e.specific && e.arch / 10 == cc / 10 && e.arch <= cc) {
            return "needs the sm_" + std::to_string(cc) + " engine: the release's sm_" + e.tag +
                   " engine writes other KV row codes than an sm_" + std::to_string(cc) +
                   " card's plan prices";
        }
        older = older && e.arch > cc;
    }
    return older ? "is older than this release supports" : "is not supported by this release";
}

} // namespace ninferno::dispatch
