#include "ops/generic/route_analyzer_tool.h"

#include "core/machine_facts.h"
#include "ops/generic/route_analyzer.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace ninferno::tools {

namespace ra = ops::route_analyzer;
using ops::detail::generic::Epilogue;

namespace {
// A cell's PCI id ("0x2B85"); 0 (no board: the cell prints "- -") where it is not one.
std::uint32_t pci_of(const std::string& text) {
    try {
        return static_cast<std::uint32_t>(std::stoul(text, nullptr, 16));
    } catch (const std::exception&) { return 0; }
}
} // namespace

int route_analyzer_main(int argc, char** argv) {
    std::string cells;
    ra::Settings s;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cells" && i + 1 < argc) {
            cells = argv[++i];
        } else if (a == "--launch" && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m != "graph" && m != "stream") {
                std::cerr << "route-analyzer: --launch graph or stream\n";
                return 2;
            }
            s.launch = m == "graph" ? ra::LaunchMode::Graph : ra::LaunchMode::Stream;
        } else {
            std::cerr
                << "usage: ninferno-serve route-analyzer [--cells FILE] [--launch graph|stream]\n";
            return 2;
        }
    }
    std::ifstream file;
    if (!cells.empty()) {
        file.open(cells);
        if (!file) {
            std::cerr << "route-analyzer: cannot read " << cells << "\n";
            return 2;
        }
    }
    std::istream& in = cells.empty() ? std::cin : file;
    std::string line;
    char buf[64];
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream f(line);
        std::string pci, fmt, ep;
        long n = 0, k = 0, T = 0;
        if (!(f >> pci >> fmt >> ep >> n >> k >> T)) {
            std::cerr << "route-analyzer: malformed cell: " << line << "\n";
            return 2;
        }
        const std::string head = pci + ' ' + fmt + ' ' + ep + ' ' + std::to_string(n) + ' ' +
                                 std::to_string(k) + ' ' + std::to_string(T);
        const auto q           = ra::format_of_name(fmt);
        const GpuFactsRow* row = find_gpu_facts(pci_of(pci));
        const auto card        = row ? ra::card_of(*row) : std::nullopt;
        const Epilogue e       = ep == "swiglu" ? Epilogue::SwiGlu
                                 : ep == "add"  ? Epilogue::AddResidual
                                                : Epilogue::Store;
        if (!q || !card || (ep != "store" && ep != "add" && ep != "swiglu")) {
            std::cout << head << " - -\n";
            continue;
        }
        const auto ranked =
            ra::rank(*card,
                     ra::Call{*q, e, static_cast<std::int32_t>(n), static_cast<std::int32_t>(k),
                              static_cast<std::int32_t>(T)},
                     s);
        if (ranked.empty()) std::cout << head << " - -\n";
        for (const ra::Priced& p : ranked) {
            std::snprintf(buf, sizeof buf, "%.17g", p.ns);
            std::cout << head << ' ' << p.route << ' ' << buf << '\n';
        }
    }
    return 0;
}

} // namespace ninferno::tools
