#include "spec/load_tune.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace ninferno::spec {

using Json = nlohmann::json;

std::vector<LoadBound> load_cell_table(std::int32_t bound, std::int32_t window) {
    if (bound < 0 || bound > window) {
        throw std::invalid_argument("spec-tune --load: a cell bound beyond the engine's window");
    }
    if (bound == window) { return {}; }
    return {LoadBound{.lanes = 2, .depth = bound}};
}

std::vector<std::uint32_t> measure_load_cells(
    std::vector<std::uint32_t> wanted,
    const std::function<std::vector<std::uint32_t>(std::uint32_t,
                                                   const std::vector<std::uint32_t>&)>& measure) {
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
    std::vector<std::uint32_t> unheld;
    while (!wanted.empty()) {
        const std::uint32_t widest                = wanted.back();
        const std::vector<std::uint32_t> measured = measure(widest, wanted);
        // The widest is settled either way: measured, or held by no engine (a narrower one is
        // tried for the rest).
        if (std::find(measured.begin(), measured.end(), widest) == measured.end()) {
            unheld.push_back(widest);
        }
        std::erase_if(wanted, [&](std::uint32_t c) {
            return c == widest || std::find(measured.begin(), measured.end(), c) != measured.end();
        });
    }
    std::sort(unheld.begin(), unheld.end());
    return unheld;
}

std::string load_bound_name(std::uint32_t bound, std::uint32_t window) {
    return bound == 0 ? "plain" : bound == window ? "none" : std::to_string(bound);
}

std::uint32_t parse_load_bound_name(std::string_view name, std::uint32_t window) {
    if (name == "plain") { return 0; }
    if (name == "none") { return window; }
    std::uint32_t depth = 0;
    bool digits         = !name.empty() && name.size() <= 2;
    for (const char ch : name) {
        if (ch < '0' || ch > '9') {
            digits = false;
            break;
        }
        depth = depth * 10 + static_cast<std::uint32_t>(ch - '0');
    }
    if (!digits || depth == 0 || depth >= window) {
        throw std::invalid_argument("spec-tune --load: unknown cell bound \"" + std::string(name) +
                                    "\" (plain, none, or a depth below the window " +
                                    std::to_string(window) + ")");
    }
    return depth;
}

Json load_cells_json(const LoadCells& cells, const std::string& load,
                     const std::string& measured_on) {
    Json measured = Json::array();
    for (const LoadCell& cell : cells.cells) {
        measured.push_back({{"concurrency", cell.concurrency},
                            {"bound", load_bound_name(cell.bound, cells.window)},
                            {"tokens", cell.tokens},
                            {"seconds", cell.seconds},
                            {"tok_s", cell.tok_s()},
                            {"rounds", cell.rounds},
                            {"drafted", cell.drafted},
                            {"accepted", cell.accepted},
                            {"plain_steps", cell.plain_steps},
                            {"trips", cell.trips}});
    }
    Json unheld = Json::array();
    for (const LoadCell& cell : cells.unheld) {
        unheld.push_back({{"concurrency", cell.concurrency},
                          {"bound", load_bound_name(cell.bound, cells.window)}});
    }
    return Json{{"settings", cells.settings},  {"window", cells.window},
                {"prefill_chunk", cells.prefill_chunk}, {"load", load},
                {"measured_on", measured_on}, {"cells", measured},
                {"unheld", unheld}};
}

LoadCells load_cells_from_json(const Json& doc, const Json& expected) {
    if (!doc.is_object() || !doc.contains("settings") || !doc["settings"].is_object()) {
        throw std::runtime_error("spec-tune --load: --resume has no settings (an --out of an "
                                 "older spec-tune, or not an --out)");
    }
    const Json& settings = doc["settings"];
    for (const auto& [key, value] : expected.items()) {
        if (!settings.contains(key) || settings[key] != value) {
            throw std::runtime_error(
                "spec-tune --load: --resume was measured with another " + key + " (" +
                (settings.contains(key) ? settings[key].dump() : std::string("none")) +
                ", this run " + value.dump() + ")");
        }
    }
    for (const auto& [key, value] : settings.items()) {
        if (!expected.contains(key)) {
            throw std::runtime_error("spec-tune --load: --resume records a setting this run does "
                                     "not (" + key + ")");
        }
    }
    LoadCells out;
    out.settings      = settings;
    out.window        = doc.at("window").get<std::uint32_t>();
    out.prefill_chunk = doc.value("prefill_chunk", 0U);
    std::set<std::pair<std::uint32_t, std::uint32_t>> seen;
    for (const Json& c : doc.value("cells", Json::array())) {
        LoadCell cell;
        cell.concurrency = c.at("concurrency").get<std::uint32_t>();
        cell.bound       = parse_load_bound_name(c.at("bound").get<std::string>(), out.window);
        cell.tokens      = c.at("tokens").get<std::int64_t>();
        cell.seconds     = c.at("seconds").get<double>();
        cell.rounds      = c.at("rounds").get<std::uint64_t>();
        cell.drafted     = c.at("drafted").get<std::uint64_t>();
        cell.accepted    = c.at("accepted").get<std::uint64_t>();
        cell.plain_steps = c.at("plain_steps").get<std::uint64_t>();
        cell.trips       = c.at("trips").get<std::uint64_t>();
        if (!seen.insert({cell.concurrency, cell.bound}).second) {
            throw std::runtime_error("spec-tune --load: --resume has the cell C=" +
                                     std::to_string(cell.concurrency) + " bound " +
                                     load_bound_name(cell.bound, out.window) + " twice");
        }
        out.cells.push_back(cell);
    }
    // Unheld cells are not taken over: this run tries them again (the memory may differ).
    return out;
}

LoadDecision decide_load(const std::vector<LoadCell>& cells,
                         const std::vector<std::uint32_t>& concurrency,
                         const std::vector<std::uint32_t>& bounds, double tolerance,
                         double min_gain) {
    if (!(tolerance >= 0 && tolerance < 1) || !(min_gain >= 0)) {
        throw std::invalid_argument("decide_load: tolerance in [0,1), min_gain >= 0");
    }
    const auto find = [&](std::uint32_t c, std::uint32_t bound) -> const LoadCell* {
        for (const LoadCell& cell : cells) {
            if (cell.concurrency == c && cell.bound == bound) { return &cell; }
        }
        return nullptr;
    };
    std::vector<std::uint32_t> ascending = concurrency;
    std::sort(ascending.begin(), ascending.end());
    ascending.erase(std::unique(ascending.begin(), ascending.end()), ascending.end());
    std::vector<std::uint32_t> ordered = bounds;
    std::sort(ordered.begin(), ordered.end());
    LoadDecision out;
    for (const std::uint32_t c : ascending) {
        const LoadCell* plain = find(c, 0);
        double fastest        = 0;
        for (const std::uint32_t bound : ordered) {
            if (const LoadCell* cell = find(c, bound); bound != 0 && cell != nullptr) {
                fastest = std::max(fastest, cell->tok_s());
            }
        }
        if (plain == nullptr || fastest <= 0) {
            out.undecided.push_back(c);
            continue;
        }
        const LoadCell* best = nullptr; // ascending bounds: the first within tolerance
        for (const std::uint32_t bound : ordered) {
            const LoadCell* cell = find(c, bound);
            if (bound != 0 && cell != nullptr && cell->tok_s() >= fastest * (1.0 - tolerance)) {
                best = cell;
                break;
            }
        }
        // The fastest bound itself is within any tolerance, so `best` is set.
        const bool speculate = best->tok_s() >= plain->tok_s() * (1.0 + min_gain);
        out.choices.push_back({.concurrency = static_cast<std::int32_t>(c),
                               .depth       = static_cast<std::int32_t>(speculate ? best->bound
                                                                                  : 0U)});
    }
    return out;
}

std::vector<LoadBound> decided_load_table(const LoadDecision& decision, std::int32_t window) {
    std::vector<LoadBound> table = load_table(decision.choices, window);
    if (decision.choices.empty()) { return table; }
    const std::int32_t widest = decision.choices.back().concurrency;
    const bool wider_asked =
        std::any_of(decision.undecided.begin(), decision.undecided.end(),
                    [&](std::uint32_t c) { return static_cast<std::int32_t>(c) > widest; });
    if (!wider_asked) { return table; }
    // The shallowest decided choice (the widest concurrency's among equals), with its width.
    LoadChoice shallowest{.depth = std::numeric_limits<std::int32_t>::max()};
    for (const LoadChoice& choice : decision.choices) {
        if (choice.depth <= shallowest.depth) { shallowest = choice; }
    }
    const std::int32_t current = table.empty() ? window : table.back().depth;
    // Plain for good once a step is plain (load_table); otherwise never deeper than measured.
    if (current != 0 && shallowest.depth < current) {
        table.push_back({.lanes = widest + 1,
                         .depth = shallowest.depth,
                         .width = shallowest.depth == 0 ? DraftWidth::Default : shallowest.width});
    }
    return table;
}

} // namespace ninferno::spec
