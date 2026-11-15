#include "replay_routes.h"

#include "core/device_class.h"
#include "core/machine_facts.h"
#include "ops/generic/price_tool.h"
#include "runtime/engine/decision_fingerprint.h"
#include "runtime/engine/decision_routes.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninferno::tools {

namespace {

const char* usage() {
    return "usage: ninferno-serve replay-routes --board BOARD --log FILE\n"
           "  BOARD  a board of the facts table: its PCI id (0x2B85) or name (rtx5090)\n"
           "  FILE   a start's request log (--request-log-jsonl): its last server_start record\n"
           "Asks the start's route lines again on a CPU against BOARD's class from its registry\n"
           "row, under the start's route policy, and prints the lines that moved. Refused: a\n"
           "start whose selection a maintainer overrode, with another build, or on another board.\n"
           "Exit 0: none moved; 1: some did; 2: a usage or input error or a refusal.\n";
}

} // namespace

int replay_routes_main(int argc, char** argv) {
    std::string board, log;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if ((a == "--board" || a == "--log") && i + 1 < argc) {
            (a == "--board" ? board : log) = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::cout << usage();
            return 0;
        } else {
            std::cerr << usage();
            return 2;
        }
    }
    if (board.empty() || log.empty()) {
        std::cerr << usage();
        return 2;
    }
    const GpuFactsRow* row = price_find_board(board);
    if (row == nullptr) {
        std::cerr << "replay-routes: unknown board " << board << "\n";
        return 2;
    }
    std::ifstream in(log);
    if (!in) {
        std::cerr << "replay-routes: cannot read " << log << "\n";
        return 2;
    }
    std::optional<nlohmann::json> start;
    for (std::string line; std::getline(in, line);) {
        if (line.find("server_start") == std::string::npos) { continue; }
        nlohmann::json j = nlohmann::json::parse(line, nullptr, false);
        const auto event = j.is_object() ? j.find("event") : j.end();
        if (j.is_object() && event != j.end() && event->is_string() && *event == "server_start") {
            start = std::move(j);
        }
    }
    const auto text_of = [](const nlohmann::json& object, const char* key) {
        const auto it = object.find(key);
        return it != object.end() && it->is_string() ? it->get<std::string>() : std::string();
    };
    if (!start || !start->contains("decision") || !(*start)["decision"].is_object() ||
        text_of((*start)["decision"], "routes").empty()) {
        std::cerr << "replay-routes: no server_start record with decision.routes in " << log
                  << "\n";
        return 2;
    }
    const nlohmann::json& decision = (*start)["decision"];
    const std::string canonical    = text_of(decision, "canonical");
    const runtime::DecisionSection recorded =
        runtime::section_of_text(text_of(decision, "routes"), "routes");
    const runtime::DecisionSection options = runtime::section_of_text(canonical, "options");
    const runtime::DecisionSection taint   = runtime::section_of_text(canonical, "taint");
    const runtime::DecisionSection build   = runtime::section_of_text(canonical, "build");
    const runtime::DecisionSection facts   = runtime::section_of_text(canonical, "facts");
    if (recorded.empty()) {
        std::cerr << "replay-routes: the record holds no route lines\n";
        return 2;
    }
    // What else entered the start's selection besides the board and the shipped tables: a
    // maintainer's route table, directory, pins or tuners, or a tuner's cache (taint maintainer,
    // user_tune_cache), and the build (its selector reads the build's compile definitions). A
    // replay without them would report moves that are not the board's: refused. The other taints
    // are the release selection's own, which the replay's lookup repeats for the board's class (the
    // nearest board's table where the board has none, foreign_route_table; entries the shipped
    // tables lend; a driver without clocks; a carried context cost; a speculation fallback):
    // printed, and replayed. The start's table and lenders are printed too: a replay binary with
    // other route tables reports their change as moved lines.
    for (const auto& [key, value] : runtime::section_of_text(canonical, "sources")) {
        if (key.rfind("route_table", 0) == 0) { std::cout << "start: sources." << key << '=' << value << '\n'; }
    }
    bool overridden = false;
    for (const auto& [key, value] : taint) {
        const bool overrides = key == "maintainer" || key == "user_tune_cache";
        overridden = overridden || overrides;
        (overrides ? std::cerr : std::cout)
            << (overrides ? "replay-routes: the start's selection was overridden: taint."
                          : "note: taint.")
            << key << '=' << value << std::endl;
    }
    if (overridden) {
        std::cerr << "replay-routes: its routes are not the release selection's; not replayed"
                  << std::endl;
        return 2;
    }
    // The selector reads the build's compile definitions (its architecture, arch-specific
    // features, NVFP4: a 120a and a plain 120 build both say sm_120): every build key but the
    // version, the commit and the runtime must be this binary's.
    const runtime::DecisionSection here = runtime::build_of();
    for (const auto& [key, value] : here) {
        if (key == "version" || key == "git_commit" || key == "cudart") { continue; }
        const auto theirs = build.find(key);
        if (theirs == build.end() || theirs->second != value) {
            std::cerr << "replay-routes: the start's build." << key << " is "
                      << (theirs == build.end() ? std::string("absent") : theirs->second)
                      << ", this binary's " << value << ": the selector reads the build"
                      << std::endl;
            return 2;
        }
    }
    const auto pci = facts.find("pci_device_id");
    if (pci == facts.end() || pci->second != std::to_string(row->pci_device_id)) {
        std::cerr << "replay-routes: the start ran on PCI id "
                  << (pci == facts.end() ? std::string("?") : pci->second) << ", not " << row->name
                  << "'s " << row->pci_device_id << std::endl;
        return 2;
    }
    if (const auto commit = build.find("git_commit"); commit != build.end()) {
        std::cout << "start: build " << commit->second << ", sm_" << here.at("arch")
                  << "; replayed against " << row->name << "'s registry row" << std::endl;
    }
    // The board's class is the registry row's (its clocks the row's, not the start's driver's):
    // the replay checks that the registry gives the start's routes (#612 step 10).
    const DeviceClass device = price_board_class(*row);
    runtime::DecisionSection replayed;
    try {
        replayed = runtime::replay_routes(device, recorded, runtime::route_policy_of(options));
    } catch (const std::exception& error) {
        std::cerr << "replay-routes: " << error.what() << "\n";
        return 2;
    }
    int moved = 0;
    for (const auto& [key, value] : recorded) {
        const auto again      = replayed.find(key);
        const std::string now = again == replayed.end() ? std::string("<absent>") : again->second;
        if (now == value) { continue; }
        ++moved;
        std::cout << "moved: " << key << "\n  start:  " << value << "\n  replay: " << now << "\n";
    }
    std::printf("board %s (0x%04X): %zu route lines, %d moved\n", row->name.c_str(),
                row->pci_device_id, recorded.size(), moved);
    return moved == 0 ? 0 : 1;
}

} // namespace ninferno::tools
