// The decision manifest's route lines (#304, runtime/engine/decision_routes.h): each line is the
// selector's own answer at every decode and verify width and at the prefill chunk; replaying the
// lines from their text alone on the board's class gives them back; the problems' order does not
// matter; a route table that moves a route moves its line (and the manifest, not the fingerprint);
// a line that does not parse is refused.
#include "runtime/engine/decision_routes.h"

#include "core/device_class.h"
#include "core/units.h"
#include "core/weight_view.h"
#include "ninferno/ops/route_table.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ops::detail::generic;
namespace rt      = ninferno::runtime;
using ops::LinearPolicy;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL decision routes: " << what << '\n';
        ++failures;
    }
}

generic::SeenProblem problem(QType q, LinearPolicy policy, generic::Epilogue e, std::int32_t n,
                             std::int32_t k, std::int32_t parent_rows) {
    return {generic::Problem{q, policy, n, k, e}, parent_rows, default_weight_layout(q)};
}

std::vector<generic::SeenProblem> problems() {
    using generic::Epilogue;
    return {problem(QType::GGML_Q4_K, LinearPolicy::A16Only, Epilogue::Store, 4096, 4096, 4096),
            problem(QType::GGML_Q6_K, LinearPolicy::AllowA8, Epilogue::AddResidual, 4096, 12288, 4096),
            problem(QType::BF16, LinearPolicy::A16Only, Epilogue::SwiGlu, 6144, 2048, 12288),
            // A slice of a fused parent: 1024 of its 6144 rows.
            problem(QType::Q8_G32_FP16, LinearPolicy::A16Only, Epilogue::Store, 1024, 4096, 6144)};
}

// The route a line names at width `t` (its runs), or "" when no run covers it.
std::string route_at(const std::string& line, std::int32_t t) {
    const std::string decode = line.substr(0, line.find("; prefill "));
    std::size_t at           = 0;
    while (at < decode.size()) {
        const std::size_t end  = std::min(decode.find(", ", at), decode.size());
        const std::string run  = decode.substr(at, end - at);
        const std::size_t dash = run.find('-'), space = run.find(' ');
        if (t >= std::stoi(run.substr(0, dash)) &&
            t <= std::stoi(run.substr(dash + 1, space - dash - 1))) {
            return run.substr(space + 1);
        }
        at = end + 2;
    }
    return "";
}

} // namespace

int main() {
    if (test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const auto list  = problems();
    const int chunk  = 1024;
    // A substituted board (its class: the default lookup's tables for it) and the device's own.
    DeviceClass board = synthetic_device_class(0x2684, 8, 9, 128, 72 * kMiB, 24ULL * 1024 * kMiB);
    board.sm_clock_khz     = 2520000;
    board.memory_clock_khz = 10501000;
    board.memory_bus_bits  = 384;
    board.facts            = machine_facts_of(board);
    for (const DeviceClass& d : {board, device_class()}) {
        const auto lines = rt::routes_of(d, list, chunk);
        check(lines.size() == list.size(), "one line per problem");
        // Each line is the selector's answer at every width.
        for (const auto& s : list) {
            const auto& p    = s.problem;
            const auto found = std::find_if(lines.begin(), lines.end(), [&](const auto& kv) {
                return kv.first.find(std::to_string(p.n) + 'x' + std::to_string(p.k) + " of " +
                                     std::to_string(s.parent_rows)) != std::string::npos;
            });
            check(found != lines.end(), "a line for " + std::to_string(p.n) + "x" + std::to_string(p.k));
            if (found == lines.end()) continue;
            for (std::int32_t t = 1; t <= rt::kDecisionRouteWidths; ++t) {
                const auto r = generic::route_query(d, p.qtype, p.policy, p.epilogue, p.n, p.k, t,
                                                    false, s.parent_rows, s.layout);
                check(route_at(found->second, t) == (r ? generic::route_name(*r) : "none"),
                      found->first + " T=" + std::to_string(t) + ": " + found->second);
            }
            const auto pre = generic::route_query(d, p.qtype, p.policy, p.epilogue, p.n, p.k, chunk,
                                                  true, s.parent_rows, s.layout);
            check(found->second.ends_with("; prefill 1024 " + (pre ? generic::route_name(*pre) : "none")),
                  found->first + " prefill: " + found->second);
        }
        // Replayed from the text alone.
        check(rt::replay_routes(d, lines) == lines, "the lines replay to themselves");
        // The problems' order is not an input.
        auto reversed = list;
        std::reverse(reversed.begin(), reversed.end());
        check(rt::routes_of(d, reversed, chunk) == lines, "the problems' order changes nothing");
    }
    std::cout << "lines on the device:\n";
    for (const auto& [key, value] : rt::routes_of(device_class(), list, chunk)) {
        std::cout << "  " << key << " = " << value << '\n';
    }

    // An explicit table that moves the Q8 slice's decode route moves its line: the replay of the
    // recorded lines names it, and the manifest moves while the fingerprint does not.
    const auto before = rt::routes_of(device_class(), list, chunk);
    const auto dir    = std::filesystem::temp_directory_path() / "ninferno_decision_routes";
    std::filesystem::create_directories(dir);
    const auto path = dir / "moved.json";
    std::string moved;
    for (const char* route : {"rmma.m1.n1.w2x2", "rmma.m1.n1.w1x4"}) {
        if (route_at(before.rbegin()->second, 1) != route) {
            moved = route;
            break;
        }
    }
    std::ofstream(path) << R"({"schema": "ninferno-route-table/1", "gpu": {"class": ")"
                        << ops::route_gpu_class()
                        << R"(", "name": "test"}, "entries": [{"format": "q8", "compute": "a16",
        "epilogue": "store", "n": 1024, "k": 4096, "bands": [[1, 16, ")"
                        << moved << R"("]]}]})";
    (void)ops::load_route_table(path.string());
    const auto after = rt::replay_routes(device_class(), before);
    ops::clear_route_table();
    std::size_t differ = 0;
    for (const auto& [key, value] : before) differ += after.at(key) != value ? 1 : 0;
    check(differ == 1 && route_at(after.at(before.rbegin()->first), 4) == moved,
          "the moved table moves the Q8 slice's line alone (" + std::to_string(differ) + " moved)");
    rt::DecisionInputs a;
    a.plan["route_breakpoints"] = "17";
    a.routes                    = before;
    rt::DecisionInputs b        = a;
    b.routes                    = after;
    const auto fa = rt::decision_fingerprint(a), fb = rt::decision_fingerprint(b);
    check(fa.value == fb.value && fa.manifest != fb.manifest && fa.routes != fb.routes,
          "a moved route moves the manifest, not the fingerprint");

    // Refused: a line that does not parse, or not in its canonical form.
    for (const std::string key : {"q4k a16only store 4096x4096 of 4096", "zz a16only store 1x1 of 1 layout 0",
                                  "q4k a16only store 4096x4096 of 4096 layout 0 extra",
                                  "q4k  a16only store 4096x4096 of 4096 layout 0"}) {
        bool refused = false;
        try {
            (void)rt::replay_routes(device_class(), {{key, "1-16 scalar; prefill 1024 scalar"}});
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "refused: \"" + key + "\"");
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " decision routes\n";
    return failures == 0 ? 0 : 1;
}
