#include "spec/menu.h"

#include <algorithm>

namespace ninferno::spec {

std::vector<std::int32_t> SpeculationMenu::depths() const {
    std::vector<std::int32_t> out;
    const auto add = [&](const MenuChoice& c) {
        if (c.config.enabled()) { out.push_back(c.config.k); }
    };
    add(primary);
    add(thinking);
    add(non_thinking);
    add(overall);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<MenuStep> SpeculationMenu::step_down(const MenuChoice& choice) const {
    std::vector<MenuStep> out;
    if (!choice.config.enabled()) { return out; }
    const auto add = [&](const MenuChoice& c) {
        if (!c.config.enabled() || c.config.k >= choice.config.k) { return; }
        for (auto& step : out) {
            if (step.depth == c.config.k) {
                step.floor = std::max(step.floor, c.floor);
                return;
            }
        }
        out.push_back(MenuStep{.depth = c.config.k, .floor = c.floor});
    };
    add(primary);
    add(thinking);
    add(non_thinking);
    add(overall);
    std::sort(out.begin(), out.end(),
              [](const MenuStep& a, const MenuStep& b) { return a.depth > b.depth; });
    return out;
}

const MenuChoice& SpeculationMenu::pick(std::optional<bool> thinking_request) const {
    if (thinking_request) { return *thinking_request ? thinking : non_thinking; }
    return overall;
}

SpeculationMenu build_menu(const Profile& profile, const Choice& primary,
                           const std::function<bool(const Choice&)>& usable) {
    SpeculationMenu menu;
    const bool speculative = primary.config.enabled() && primary.config.mode != Mode::Ngram && !profile.disabled &&
                             usable(primary);
    if (speculative) {
        menu.mode          = primary.config.mode;
        menu.proposal_head = primary.config.proposal_head;
    }
    const auto place = [&](const Choice& choice) {
        MenuChoice out;
        out.profiled = choice.config;
        if (!speculative) {
            if (choice.config.enabled()) {
                out.note = profile.disabled ? "the profile is disabled"
                                            : "the default's drafter cannot run in this engine";
            }
            return out;
        }
        if (!choice.config.enabled()) { return out; } // the profile chose plain decode
        if (choice.config.mode == menu.mode && choice.config.proposal_head == menu.proposal_head &&
            usable(choice)) {
            out.config = choice.config;
            out.floor  = choice.floor_tokens_per_round;
            return out;
        }
        out.config = primary.config;
        out.floor  = primary.floor_tokens_per_round;
        out.note   = choice.config.label() + " needs " +
                   (choice.config.mode != menu.mode ? std::string("the ") +
                                                          mode_name(choice.config.mode) + " drafter"
                    : choice.config.proposal_head != menu.proposal_head
                        ? std::string("another proposal head")
                        : std::string("a drafter or depth this engine cannot run")) +
                   "; the engine runs " + mode_name(menu.mode) + ", so " + primary.config.label() +
                   " runs";
        return out;
    };
    menu.primary      = place(primary);
    menu.thinking     = place(profile.thinking);
    menu.non_thinking = place(profile.non_thinking);
    menu.overall      = place(profile.overall);
    return menu;
}

} // namespace ninferno::spec
