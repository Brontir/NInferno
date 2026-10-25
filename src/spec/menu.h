#pragma once

// A speculation profile laid onto one engine that serves many requests. A tuned profile
// (spec/profile.h) chooses a default, a thinking and a non-thinking configuration, and those
// choices can differ in drafter and depth. An engine that serves requests concurrently loads one
// drafter (its backend) and captures its speculative CUDA graphs for a set of depths at start; each
// request then runs at its own depth, or plainly. build_menu() decides that engine and each
// choice's place in it:
//
//   - the engine speculates with the drafter of the primary choice (the profile's default for the
//     server's thinking default), with the primary's proposal head;
//   - every default that uses the same drafter (and head) runs at its own depth with its own
//     watchdog floor; plain decode stays plain;
//   - a choice that needs another drafter or another proposal head the engine cannot run runs
//     the primary's configuration instead, and says why (MenuChoice::note).
//
// The engine's depths (depths()) are what it must capture; a request picks its entry with
// pick(), as spec::pick() picks a profile's choice. When a request's acceptance watchdog trips,
// it steps down through the menu's shallower depths (step_down()) before it decodes plainly.

#include "spec/config.h"
#include "spec/profile.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::spec {

struct MenuChoice {
    SpecConfig config;   // what the request runs (Mode::None: plain decode)
    double floor = 0;    // its watchdog floor in tokens per round (0 when plain)
    SpecConfig profiled; // what the profile chose; differs from config when it cannot run here
    std::string note;    // why config differs from profiled ("" when they agree)
};

// A depth a tripped request steps down to, with its watchdog floor.
struct MenuStep {
    std::int32_t depth = 0;
    double floor       = 0;
};

struct SpeculationMenu {
    Mode mode          = Mode::None; // the engine's drafter (None: a plain engine)
    bool proposal_head = false;
    MenuChoice primary; // the engine's default configuration
    MenuChoice thinking, non_thinking, overall;

    // The distinct draft depths of the entries that speculate, ascending (the engine captures
    // one set of speculative graphs per depth).
    [[nodiscard]] std::vector<std::int32_t> depths() const;
    // The entry for a request: the thinking or non-thinking default when the request says, else
    // the overall default (spec::pick()'s order).
    [[nodiscard]] const MenuChoice& pick(std::optional<bool> thinking) const;
    // Where a request running `choice` goes when its watchdog trips: every menu depth below the
    // choice's, deepest first, each with the highest floor among the entries at that depth (a
    // step keeps speculating only while it pays by every measurement of it). Empty for plain.
    [[nodiscard]] std::vector<MenuStep> step_down(const MenuChoice& choice) const;
};

// `usable(choice)`: whether the choice runs here (its drafter is in the artifact and the Engine
// runs its configuration). An unusable primary gives a plain menu (every entry plain decode, noted).
[[nodiscard]] SpeculationMenu build_menu(const Profile& profile, const Choice& primary,
                                         const std::function<bool(const Choice&)>& usable);

} // namespace ninferno::spec
