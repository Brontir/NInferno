#include "replay_routes.h"

#include "core/device_class.h"
#include "core/machine_facts.h"
#include "ops/generic/price_tool.h"
#include "runtime/engine/decision_routes.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninferno::tools {

namespace {

const char* usage() {
    return "usage: ninferno-serve replay-routes --board BOARD --log FILE\n"
           "  BOARD  a board of the facts table: its PCI id (0x2B85) or name (rtx5090)\n"
           "  FILE   a start's request log (--request-log-jsonl): its last server_start record\n"
           "Asks the start's route lines again on a CPU against BOARD's class from the facts\n"
           "table, under the start's route policy, and prints the lines that moved.\n"
           "Exit 0: none moved; 1: some did; 2: a usage or input error.\n";
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
        const nlohmann::json j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_object() && j.value("event", "") == "server_start") { start = j; }
    }
    if (!start || !start->contains("decision") || !(*start)["decision"].contains("routes")) {
        std::cerr << "replay-routes: no server_start record with decision.routes in " << log
                  << "\n";
        return 2;
    }
    const nlohmann::json& decision = (*start)["decision"];
    const runtime::DecisionSection recorded =
        runtime::section_of_text(decision.value("routes", std::string()), "routes");
    const runtime::DecisionSection options =
        runtime::section_of_text(decision.value("canonical", std::string()), "options");
    if (recorded.empty()) {
        std::cerr << "replay-routes: the record holds no route lines\n";
        return 2;
    }
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
        const auto again = replayed.find(key);
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
