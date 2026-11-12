// The cost model's price of every route of a route_tune sweep, on a substituted board (#781): reads
// a --routes-out CSV (format, compute, epilogue, n, k, T, route, ..., model_us, ...) and writes it
// back with model_us replaced by what price_routes gives that route for that call on the board
// (priced_routes_query, the selector's own pricing), empty where the model offers no price. Run on
// the host, with no GPU: the prediction for a change of the cost model, before any stand run, to
// be scored against the sweep's measured medians (tools/bench/regret_bands.py --pick model).
//
//   ninferno_route_price_sweep --board rtx2080ti --in routes_p1.csv --out priced_p1.csv
//
// The board is one of the selector golden boards (ops/generic/selector_facts.h kFactsBoards). The
// route set is this build's: run the tool from a tree of the architecture the sweep's board runs
// (an sm_75 tree for the RTX 2080 Ti), or a route the board's build does not hold is priced as if
// it did. `compute` a8 prices under AllowA8, a16 under A16Only.
#include "core/format_facts.h"
#include "ninferno/ops/linear.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_query.h"
#include "ops/generic/selector_facts.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

namespace generic = ninferno::ops::detail::generic;
using ninferno::QType;
using ninferno::ops::LinearPolicy;

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    std::istringstream in(line);
    while (std::getline(in, cell, ',')) out.push_back(cell);
    if (!line.empty() && line.back() == ',') out.emplace_back();
    return out;
}

std::string join(const std::vector<std::string>& cells) {
    std::string out;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        if (i) out += ',';
        out += cells[i];
    }
    return out;
}

std::optional<generic::Epilogue> epilogue_of(std::string_view s) {
    if (s == "store") return generic::Epilogue::Store;
    if (s == "add" || s == "residual") return generic::Epilogue::AddResidual;
    if (s == "swiglu") return generic::Epilogue::SwiGlu;
    return std::nullopt;
}

int run(int argc, char** argv) {
    std::string board_name, in_path, out_path;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string_view a = argv[i];
        if (a == "--board") board_name = argv[i + 1];
        if (a == "--in") in_path = argv[i + 1];
        if (a == "--out") out_path = argv[i + 1];
    }
    const generic::FactsBoard* board = nullptr;
    for (const auto& b : generic::kFactsBoards) {
        if (board_name == b.name) board = &b;
    }
    if (!board || in_path.empty() || out_path.empty()) {
        std::cerr
            << "usage: ninferno_route_price_sweep --board NAME --in ROUTES.csv --out OUT.csv\n";
        return 2;
    }
    const ninferno::DeviceClass d = generic::board_device_class(*board);
    std::ifstream in(in_path);
    std::ofstream out(out_path);
    std::string line;
    if (!std::getline(in, line)) return 2;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto header = split(line);
    std::map<std::string, std::size_t> col;
    for (std::size_t i = 0; i < header.size(); ++i) col[header[i]] = i;
    for (const char* need : {"format", "compute", "epilogue", "n", "k", "T", "route", "model_us"}) {
        if (!col.count(need)) {
            std::cerr << in_path << ": no column " << need << '\n';
            return 2;
        }
    }
    out << line << '\n';
    using Key = std::tuple<std::string, std::string, std::string, int, int, int>;
    std::map<Key, std::map<std::string, double>> cache;
    std::size_t rows = 0, priced = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto cells = split(line);
        if (cells.size() < header.size()) cells.resize(header.size());
        const Key key{cells[col["format"]],       cells[col["compute"]],
                      cells[col["epilogue"]],     std::stoi(cells[col["n"]]),
                      std::stoi(cells[col["k"]]), std::stoi(cells[col["T"]])};
        auto it = cache.find(key);
        if (it == cache.end()) {
            std::map<std::string, double> prices;
            const auto qtype    = ninferno::find_route(std::get<0>(key));
            const auto epilogue = epilogue_of(std::get<2>(key));
            if (qtype && epilogue) {
                const LinearPolicy policy =
                    std::get<1>(key) == "a16" ? LinearPolicy::A16Only : LinearPolicy::AllowA8;
                for (const auto& p :
                     generic::priced_routes_query(d, *qtype, policy, *epilogue, std::get<3>(key),
                                                  std::get<4>(key), std::get<5>(key))) {
                    prices[generic::route_name(p.route)] = p.seconds * 1e6;
                }
            } else {
                std::cerr << "unknown format or epilogue: " << std::get<0>(key) << " "
                          << std::get<2>(key) << '\n';
            }
            it = cache.emplace(key, std::move(prices)).first;
        }
        const auto price = it->second.find(cells[col["route"]]);
        if (price != it->second.end()) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%.6f", price->second);
            cells[col["model_us"]] = buf;
            ++priced;
        } else {
            cells[col["model_us"]].clear();
        }
        out << join(cells) << '\n';
        ++rows;
    }
    std::cout << out_path << ": " << rows << " rows, " << priced << " priced on " << board->name
              << " (" << d.id() << ")\n";
    return out.good() ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "ninferno_route_price_sweep: " << e.what() << '\n';
        return 1;
    }
}
