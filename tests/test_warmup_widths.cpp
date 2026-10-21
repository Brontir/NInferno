// The kernel warm-up's prefill width set (src/runtime/engine/warmup_widths.h): the exact band and
// chunk, order, the loaded route table's breakpoints joining the compiled edges and leaving with
// the table, and the guard against a missed kernel class: every width 49..chunk is passed through
// the exported plan choosers (q8 linear_add with its tile variant and launcher bands, q8
// linear_pair, the BF16 GDN gating MMA) under both route profiles, and every kernel class they
// launch at or below chunk must hold a warmed width. The shipped route tables' set sizes are
// printed. CPU only: the plans resolve on the host.
#include "runtime/engine/warmup_widths.h"

#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/route_table.h"
#include "core/sha256.h"
#include "support/temp_dir.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"
#include "ops/linear_add/q8/q8_linear_add_plan.h"
#include "ops/linear_pair/q8/q8_pair_plan.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace op     = ninferno::ops::detail;
namespace warmup = ninferno::runtime::detail;
using Json       = nlohmann::json;

int check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL warmup_widths: " << what << '\n'; }
    return ok ? 0 : 1;
}

bool holds(const std::vector<std::uint32_t>& widths, std::uint32_t width) {
    return std::binary_search(widths.begin(), widths.end(), width);
}

bool holds_any(const std::vector<std::uint32_t>& widths, std::uint32_t first, std::uint32_t last) {
    const auto it = std::lower_bound(widths.begin(), widths.end(), first);
    return it != widths.end() && *it <= last;
}

// Removed when the process ends.
std::filesystem::path scratch(const std::string& name) {
    static const TempDir dir("ninferno_warmup_widths_test");
    return dir.path() / name;
}

// The width set's own contract at one chunk.
int check_shape(const std::vector<std::uint32_t>& widths, std::uint32_t chunk) {
    const std::string at = " (chunk " + std::to_string(chunk) + ")";
    int failures         = 0;
    failures += check(std::is_sorted(widths.begin(), widths.end()) &&
                          std::adjacent_find(widths.begin(), widths.end()) == widths.end(),
                      "the widths are sorted and unique" + at);
    failures += check(!widths.empty() && widths.front() >= 1 && widths.back() == chunk,
                      "the widths lie in 1..chunk and end at chunk" + at);
    bool exact = widths.size() >= warmup::kWarmUpExactWidths;
    for (std::uint32_t width = 1; exact && width <= warmup::kWarmUpExactWidths; ++width) {
        exact = widths[width - 1] == width;
    }
    failures += check(exact, "every width of the exact band 1..48 is warmed" + at);
    return failures;
}

// One exported chooser at one problem shape and profile: the kernel class of T columns.
struct Chooser {
    std::string name;
    std::function<std::string(std::int32_t)> kernel_class;
};

std::vector<Chooser> choosers() {
    std::vector<Chooser> out;
    for (const bool prefill : {false, true}) {
        const std::string profile = prefill ? " prefill" : " speed";
        for (const auto& shape : std::vector<std::pair<std::int32_t, std::int32_t>>{
                 {2048, 4096}, {2048, 6144}, {5120, 6144}, {5120, 17408}}) {
            const std::int32_t rows = shape.first;
            const std::int32_t k    = shape.second;
            out.push_back({"q8 linear_add " + std::to_string(rows) + "x" + std::to_string(k) +
                               profile,
                           [rows, k, prefill](std::int32_t cols) {
                               const ninferno::ops::ScopedPrefillRoutes scope(prefill);
                               const op::Q8LinearAddProblem problem{rows, k, k, cols};
                               const op::Q8LinearAddPlan plan =
                                   op::q8_linear_add_resolve_plan(problem);
                               const op::Q8LinearAddKernelClass c =
                                   op::q8_linear_add_kernel_class(plan, problem);
                               return std::string(op::q8_linear_add_schedule_name(c.schedule)) +
                                      (c.full ? " full" : " partial") + " band " +
                                      std::to_string(c.band);
                           }});
        }
        for (const auto& shape :
             std::vector<std::pair<std::int32_t, std::int32_t>>{{48, 5120}, {32, 2048}}) {
            const std::int32_t heads = shape.first;
            const std::int32_t rows  = shape.second;
            out.push_back({"bf16 gdn_gating " + std::to_string(heads) + "x" +
                               std::to_string(rows) + profile,
                           [heads, rows, prefill](std::int32_t cols) {
                               const ninferno::ops::ScopedPrefillRoutes scope(prefill);
                               const op::Bf16GdnGatingPlan plan =
                                   op::bf16_gdn_gating_resolve_plan({heads, rows, cols});
                               return "schedule " +
                                      std::to_string(static_cast<int>(plan.schedule)) +
                                      " variant " +
                                      std::to_string(static_cast<int>(plan.token_variant));
                           }});
        }
    }
    for (const std::int32_t k : {5120, 2048}) {
        out.push_back({"q8 linear_pair 1024x" + std::to_string(k), [k](std::int32_t cols) {
                           const op::Q8PairProblem problem{1024, k, k, cols};
                           const op::Q8PairPlan plan = op::q8_pair_resolve_plan(problem);
                           return std::string(op::q8_pair_schedule_name(plan.schedule)) +
                                  (op::q8_pair_uses_full_tile(plan, problem) ? " full"
                                                                             : " partial");
                       }});
    }
    return out;
}

// Every kernel class an exported chooser launches at 49..chunk holds a warmed width.
int check_classes(const std::vector<std::uint32_t>& widths, std::uint32_t chunk) {
    int failures = 0;
    for (const Chooser& chooser : choosers()) {
        std::map<std::string, std::int32_t> first; // class -> its first width
        std::set<std::string> warmed;
        for (std::int32_t width = warmup::kWarmUpExactWidths + 1;
             width <= static_cast<std::int32_t>(chunk); ++width) {
            const std::string kernel_class = chooser.kernel_class(width);
            first.emplace(kernel_class, width);
            if (holds(widths, static_cast<std::uint32_t>(width))) { warmed.insert(kernel_class); }
        }
        for (const auto& [kernel_class, width] : first) {
            failures += check(warmed.contains(kernel_class),
                              chooser.name + ": class '" + kernel_class + "' (first at T=" +
                                  std::to_string(width) + ") has no warmed width at chunk " +
                                  std::to_string(chunk));
        }
    }
    return failures;
}

// A route table in every class chain ("*"), with one band per dialect: generic bands and prefill
// bands, a route_as band re-asked at 700, and a generic MoE band.
std::string fixture_table() {
    return R"({"schema": "ninferno-route-table/1", "gpu": {"class": "*", "name": "test"},
  "entries": [
    {"format": "bf16", "compute": "a16", "epilogue": "store", "n": 3072, "k": 4096,
     "bands": [[100, 200, "rmma.m1.n1.w1x8"]], "prefill": "staged.128x128.w2x4.s3",
     "prefill_bands": [[300, 399, "staged.64x128.w1x4.s4"]]},
    {"format": "q8", "compute": "a16", "n": 4096, "k": 1024, "route_as": [[500, 600, 700]]}],
  "moe": [
    {"format": "mxfp4", "experts": 32, "top_k": 4, "hidden": 2880, "inter": 256,
     "bands": [[800, 900, "rmma.m1.n1.w4x1|rmma.m1.n1.w4x1"]]}]})";
}

int check_route_table() {
    int failures = 0;
    ninferno::ops::clear_route_table();
    failures += check(ninferno::ops::route_table_breakpoints(49).empty(),
                      "no route table, no route breakpoints");
    const std::vector<std::uint32_t> compiled = warmup::kernel_warm_up_widths(1024);

    const auto path = scratch("fixture.json");
    std::ofstream(path) << fixture_table();
    const auto status = ninferno::ops::load_route_table(ninferno::utf8_path(path));
    failures += check(status.loaded, "the fixture table loads: " + status.message);
    const std::vector<std::int32_t> expected{100, 201, 300, 400, 500, 601, 700, 800, 901};
    failures += check(ninferno::ops::route_table_breakpoints(49) == expected,
                      "the fixture's breakpoints are its bands' first and one-past-last widths "
                      "and the route_as width");
    const std::vector<std::uint32_t> tuned = warmup::kernel_warm_up_widths(1024);
    failures += check_shape(tuned, 1024);
    for (const std::int32_t edge : expected) {
        failures += check(holds(tuned, static_cast<std::uint32_t>(edge)),
                          "the tuned set warms the table's edge " + std::to_string(edge));
    }
    failures += check(std::includes(tuned.begin(), tuned.end(), compiled.begin(), compiled.end()),
                      "the table's edges join the compiled set");
    const std::vector<std::uint32_t> short_chunk = warmup::kernel_warm_up_widths(650);
    failures += check_shape(short_chunk, 650);
    failures += check(holds(short_chunk, 601) && !holds(short_chunk, 700) &&
                          !holds(short_chunk, 800) && !holds(short_chunk, 901),
                      "a table edge above chunk stays out of the set");

    ninferno::ops::clear_route_table();
    failures += check(warmup::kernel_warm_up_widths(1024) == compiled,
                      "after clear_route_table only the compiled set remains");
    std::filesystem::remove(path);
    return failures;
}

// A table's band widths under routes every build knows: route_table_breakpoints reads the widths
// alone, and a build without NVFP4 refuses the shipped tables' W4A4 routes.
Json widths_only(const Json& table) {
    Json out     = {{"schema", table.at("schema")}, {"gpu", table.at("gpu")}};
    Json entries = Json::array();
    Json moe     = Json::array();
    const auto bands = [](const Json& source, const char* route) {
        Json list = Json::array();
        for (const Json& band : source) { list.push_back({band.at(0), band.at(1), route}); }
        return list;
    };
    std::int32_t n = 0;
    for (const Json& entry : table.at("entries")) {
        Json e = {{"format", "bf16"}, {"compute", "a16"}, {"n", ++n}, {"k", 1}};
        if (entry.contains("bands")) { e["bands"] = bands(entry.at("bands"), "rmma.m1.n1.w1x8"); }
        if (entry.contains("prefill") || entry.contains("prefill_bands")) {
            e["prefill"] = "staged.128x128.w2x4.s3";
        }
        if (entry.contains("prefill_bands")) {
            e["prefill_bands"] = bands(entry.at("prefill_bands"), "staged.64x128.w1x4.s4");
        }
        if (entry.contains("route_as")) { e["route_as"] = entry.at("route_as"); }
        entries.push_back(e);
    }
    std::int32_t experts = 0;
    for (const Json& entry : table.value("moe", Json::array())) {
        Json e = {{"format", "mxfp4"}, {"experts", ++experts}, {"top_k", 1},
                  {"hidden", 2880},    {"inter", 256}};
        if (entry.contains("bands")) {
            e["bands"] = bands(entry.at("bands"), "rmma.m1.n1.w4x1|rmma.m1.n1.w4x1");
        }
        moe.push_back(e);
    }
    out["entries"] = entries;
    out["moe"]     = moe;
    return out;
}

// The shipped route tables, applied on any device (their class replaced by "*"): the set sizes.
int report_shipped_tables() {
    int failures   = 0;
    const auto dir = std::filesystem::path(NINFERNO_SOURCE_DIR) / "routes";
    for (const char* name : {"gb202", "gb203"}) {
        const auto source = dir / (std::string(name) + ".json");
        if (!std::filesystem::exists(source)) { continue; }
        Json table;
        std::ifstream(source) >> table;
        table["gpu"]["class"] = "*";
        const auto path       = scratch(std::string(name) + ".json");
        std::ofstream(path) << table.dump();
        std::vector<std::int32_t> table_edges;
        std::string how = "as shipped";
        try {
            (void)ninferno::ops::load_route_table(ninferno::utf8_path(path));
            table_edges = ninferno::ops::route_table_breakpoints(49);
        } catch (const std::runtime_error&) {
            how = "band widths only (this build refuses a route)";
        }
        {
            // The same band widths under routes every build knows give the same breakpoints.
            std::ofstream(path) << widths_only(table).dump();
            const auto status = ninferno::ops::load_route_table(ninferno::utf8_path(path));
            failures += check(status.loaded, std::string(name) + " loads: " + status.message);
            const std::vector<std::int32_t> edges = ninferno::ops::route_table_breakpoints(49);
            failures += check(table_edges.empty() || table_edges == edges,
                              std::string(name) + ": the band widths alone give the breakpoints");
            table_edges = edges;
        }
        for (const std::uint32_t chunk : {1024U, 2048U}) {
            const std::vector<std::uint32_t> widths = warmup::kernel_warm_up_widths(chunk);
            failures += check_shape(widths, chunk);
            failures += check_classes(widths, chunk);
            std::uint64_t tokens = 0;
            for (const std::uint32_t width : widths) { tokens += width; }
            std::cout << name << " chunk " << chunk << ": " << widths.size() << " widths ("
                      << table_edges.size() << " table breakpoints, " << how << "), " << tokens
                      << " prefill tokens\n";
        }
        ninferno::ops::clear_route_table();
        std::filesystem::remove(path);
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    try {
        ninferno::ops::clear_route_table();
        failures += check(
            [] {
                try {
                    (void)warmup::kernel_warm_up_widths(warmup::kWarmUpExactWidths);
                } catch (const std::invalid_argument&) { return true; }
                return false;
            }(),
            "a chunk inside the exact band is refused");
        for (const std::uint32_t chunk : {49U, 100U, 1000U, 1024U, 2048U, 4096U}) {
            const std::vector<std::uint32_t> widths = warmup::kernel_warm_up_widths(chunk);
            failures += check_shape(widths, chunk);
            failures += check_classes(widths, chunk);
            std::uint64_t tokens = 0;
            for (const std::uint32_t width : widths) { tokens += width; }
            std::cout << "compiled chunk " << chunk << ": " << widths.size() << " widths, "
                      << tokens << " prefill tokens\n";
        }
        const std::vector<std::uint32_t> widths = warmup::kernel_warm_up_widths(1024);
        // Regressions of the per-edge set: the q8 linear_add medium split-K band 121..125 (its
        // own tile), the MmaR64C112 full tile of the K=6144 band 961..1023 (T=1008 only), and the
        // partial R32C64 tile of the q8 n14336 k5120 band 57..64.
        failures += check(holds_any(widths, 121, 125), "a width in 121..125 is warmed");
        failures += check(holds(widths, 1008), "T=1008 is warmed");
        failures += check(holds_any(widths, 57, 63), "a width in 57..63 is warmed");
        failures += check_route_table();
        failures += report_shipped_tables();
    } catch (const std::exception& error) {
        std::cerr << "FAIL warmup_widths: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "warmup_widths: ok\n"; }
    return failures == 0 ? 0 : 1;
}
