// The generic projection path's choices, frozen: for each board below (a substituted DeviceClass
// with the board's SMs, clocks, memory bus, L2 and per-SM limits, core/device_class.h) the route
// and the workspace it needs for every format, policy, epilogue, shape and width, under the speed,
// prefill and invariant profiles. The dump is compared with selector_golden/<board>.txt; a change
// of the selector shows there line by line, and a refactor that means none must leave every file
// as it was. Regenerate with --update (writes the files) after a change that means to move routes.
// The boards, the questions asked and the text are selector_boards.h's, which the property suite
// (test_selector_properties.cpp) digests the same way.
//
// The route table and several heuristics cache what they read of the device on first use, so one
// process runs one board: without arguments the test runs itself once per board.
#include "ops/linear/selector_boards.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

using namespace ninferno;
using namespace ninferno::test::selector;

std::filesystem::path golden_path(const TestBoard& b) {
    return std::filesystem::path(NINFERNO_SELECTOR_GOLDEN_DIR) / kBuildSet /
           (std::string(b.name) + ".txt");
}

int run_board(const TestBoard& b, bool update) {
    const ScopedDeviceClass scoped(device_class_of(b));
    const std::string now = render(evaluate(b.name));
    const auto path       = golden_path(b);
    if (update) {
        // The directory may not exist yet in a fresh checkout; a write that fails is a failure.
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path, std::ios::binary);
        file << now;
        if (!file.good()) {
            std::cerr << b.name << ": cannot write " << path.string() << '\n';
            return 1;
        }
        std::cout << b.name << ": wrote " << path.string() << '\n';
        return 0;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << b.name << ": no golden file " << path.string() << " (run with --update)\n";
        return 1;
    }
    std::stringstream was;
    was << in.rdbuf();
    if (was.str() == now) {
        std::cout << b.name << ": choices as the golden file\n";
        return 0;
    }
    // The first lines that differ, from both sides.
    std::istringstream a(was.str()), c(now);
    std::string la, lc;
    int shown = 0, line = 0;
    while (shown < 20) {
        const bool ga = static_cast<bool>(std::getline(a, la));
        const bool gc = static_cast<bool>(std::getline(c, lc));
        if (!ga && !gc) break;
        ++line;
        if (!ga) la = "<end>";
        if (!gc) lc = "<end>";
        if (la != lc) {
            std::cerr << b.name << ":" << line << "\n  golden: " << la << "\n  now:    " << lc
                      << '\n';
            ++shown;
        }
    }
    std::cerr << b.name << ": choices differ from " << path.string() << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool update = false;
        std::string board;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--update") update = true;
            if (arg == "--board" && i + 1 < argc) board = argv[++i];
        }
        if (!board.empty()) {
            for (const TestBoard& b : kGoldenBoards) {
                if (board == b.name) return run_board(b, update);
            }
            std::cerr << "unknown board " << board << '\n';
            return 2;
        }
        int failed = 0;
        for (const TestBoard& b : kGoldenBoards) {
            const std::string command =
                std::string(argv[0]) + (update ? " --update" : "") + " --board " + b.name;
            if (std::system(command.c_str()) != 0) ++failed;
        }
        if (failed != 0) {
            std::cerr << "selector golden: " << failed << " boards differ\n";
            return 1;
        }
        std::cout << "selector golden: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "selector golden: " << error.what() << '\n';
        return 1;
    }
}
