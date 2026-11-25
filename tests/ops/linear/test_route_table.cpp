// The per-GPU-class route table (include/ninferno/ops/route_table.h): the default lookup along the
// class chain (a ':' in a key is '-' in its file name), the generic path's band lookup and prefill
// route, route_as through the same loader, the refusals (malformed file, other GPU class, a prefill
// route that is not token-invariant, a route_as width below its band, compute classes a format does
// not have, duplicates) and tolerance of keys no consumer reads. Every shipped routes/*.json must
// pass the same validation.
#include "core/board_registry.h"
#include "core/device_class.h"
#include "core/measured_boards.h"
#include "core/sha256.h"
#include "core/weight_view.h"
#include "ninferno/ops/route_table.h"
#include "ops/common/shape_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_table.h"
#include "ops/moe/moe_internal.h"
#include "ops/moe/moe_route_query.h"
#include "ops/op_tester.h"
#include "ops/route_pins.h"
#include "support/temp_dir.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <iostream>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using ninferno::utf8_path;
namespace generic = ninferno::ops::detail::generic;
namespace detail  = ninferno::ops::detail;

// Removed when the process ends (each mode runs in a process of its own).
const std::filesystem::path& scratch_dir() {
    static const TempDir dir("ninferno_route_table_test");
    return dir.path();
}

// The table's path in UTF-8, as the route-table API takes it.
std::string write_file(const std::string& name, const std::string& text) {
    const auto path = scratch_dir() / (name + ".json");
    std::ofstream(path) << text;
    return utf8_path(path);
}

std::string table(const std::string& gpu_class, const std::string& entries) {
    return "{\"schema\": \"ninferno-route-table/1\", \"gpu\": {\"class\": \"" + gpu_class +
           "\", \"name\": \"test\"}, \"entries\": [" + entries + "]}";
}

int check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL route_table: " << what << '\n'; }
    return ok ? 0 : 1;
}

std::string route_or_none(const std::optional<generic::Route>& r) {
    return r ? generic::route_name(*r) : std::string("none");
}

bool refused(const std::string& gpu, const std::string& name, const std::string& entries) {
    try {
        (void)ninferno::ops::load_route_table(write_file(name, table(gpu, entries)));
    } catch (const std::runtime_error&) { return true; }
    return false;
}

// Every shipped routes/pick/<class>.json (tools/route_pick/gen.py), read on a substitute of the
// board its gpu.pci_id names (no GPU): the default lookup of the shipped directory finds no table
// within kCarryDistance there, reads the file as this class's predicted bands, and its first
// entry answers at its first width as Predicted, with the file's route. A file the lookup would
// never read (a class some table carries to) fails here, as does one the loader refuses.
int shipped_picks() {
    namespace board_registry = ninferno::board_registry;
    int failures             = 0;
    ninferno::ops::set_route_table_dir("");
    const auto dir = ninferno::path_from_utf8(ninferno::ops::route_table_dir()) / "pick";
    const std::regex pci_re("\"pci_id\": \"(0x[0-9a-fA-F]{4})\"");
    // An entry with T 9-32 bands (after its ranked lists, where it has them), and one with ranked
    // lists (T 1-8): the problem, the first band's width and its route.
    const std::regex entry_re(R"re("format": "([a-z0-9_]+)",\s*"compute": "a16",\s*)re"
                              R"re("epilogue": "([a-z]+)",\s*"n": ([0-9]+),\s*"k": ([0-9]+),)re"
                              R"re(\s*(?:"ranked": [^}]*?,\s*)?)re"
                              R"re("bands": \[\s*\[\s*([0-9]+),\s*[0-9]+,\s*"([^"]+)")re");
    const std::regex ranked_re(
        R"re("format": "([a-z0-9_]+)",\s*"compute": "a16",\s*)re"
        R"re("epilogue": "([a-z]+)",\s*"n": ([0-9]+),\s*"k": ([0-9]+),)re"
        R"re(\s*"ranked": \[\s*\[\s*([0-9]+),\s*[0-9]+,\s*\[\s*"([^"]+)")re");
    int files = 0;
    // The file and its board for the refusals and the gap check below: of the files with two
    // entries or more, the first by name (a directory's order is the filesystem's: one stand
    // listed a one-entry file first and failed the gap check, #1123).
    std::string first_name, first_text;
    std::uint32_t first_id = 0;
    for (const auto& f : std::filesystem::directory_iterator(dir)) {
        if (f.path().extension() != ".json") continue;
        ++files;
        std::ifstream in(f.path());
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        const std::string name = f.path().filename().string();
        std::smatch m, e, r;
        // Every file holds ranked lists (every class of an analyzer architecture) but a held chip's
        // (gen.py HELD_CHIPS: AD102, #1211), which holds none; bands only the classes gen.py writes
        // them for.
        const bool held       = name.rfind("ad102-", 0) == 0;
        const bool has_ranked = std::regex_search(text, r, ranked_re);
        if (!std::regex_search(text, m, pci_re) || has_ranked == held) {
            failures += check(false, name + (held ? " names its board and has no ranked entry (a "
                                                    "held chip, gen.py HELD_CHIPS)"
                                                  : " names its board and has a ranked entry"));
            continue;
        }
        const bool has_bands = std::regex_search(text, e, entry_re);
        const auto id = static_cast<std::uint32_t>(std::stoul(m[1].str(), nullptr, 16));
        const board_registry::BoardRow* row = board_registry::find(id);
        if (row == nullptr) {
            failures += check(false, name + ": " + m[1].str() + " is a registry id");
            continue;
        }
        ninferno::DeviceClass d =
            ninferno::synthetic_device_class(id, row->major, row->minor, row->sms,
                                             static_cast<std::size_t>(row->l2_mib * (1 << 20)), 0);
        d.facts = ninferno::machine_facts_of(d);
        const ninferno::ScopedDeviceClass scoped(d);
        ninferno::ops::set_route_table_dir("");
        int entry_count = 0;
        for (auto it = std::sregex_iterator(text.begin(), text.end(), entry_re);
             it != std::sregex_iterator() && entry_count < 2; ++it) {
            ++entry_count;
        }
        // The refusals below edit a ranked entry: a held chip's file has none.
        if (!held && entry_count >= 2 && (first_name.empty() || name < first_name)) {
            first_name = name;
            first_text = text;
            first_id   = id;
        }
        try {
            const auto s = ninferno::ops::route_table_status();
            std::cout << name << " on " << ninferno::device_class().id() << ": " << s.message
                      << '\n';
            failures +=
                check(s.loaded && ninferno::path_from_utf8(s.predicted_path).filename() == name &&
                          s.predicted_entries > 0 &&
                          (held ? s.ranked_entries == 0 : s.ranked_entries > 0) &&
                          !s.predicted_sha256.empty(),
                      name + " is read as " + ninferno::device_class().id() +
                          "'s predicted bands and ranked lists: " + s.message);
            const auto problem_of = [](const std::smatch& x) {
                return generic::Problem{
                    ninferno::find_route(x[1].str()).value_or(ninferno::QType::BF16),
                    ninferno::ops::LinearPolicy::A16Only, std::stoi(x[3].str()),
                    std::stoi(x[4].str()),
                    x[2].str() == "swiglu" ? generic::Epilogue::SwiGlu
                    : x[2].str() == "add"  ? generic::Epilogue::AddResidual
                                           : generic::Epilogue::Store};
            };
            if (!held) {
                // The first ranked entry's first width lists its first route first; none under
                // the Legacy policy.
                const auto pr     = problem_of(r);
                const int width   = std::stoi(r[5].str());
                const auto routes = generic::table_ranked_routes(pr, width);
                failures +=
                    check(ninferno::find_route(r[1].str()) && !routes.empty() &&
                              generic::route_name(routes.front()) == r[6].str(),
                          name + ": its first ranked entry lists " + r[6].str() + " first, not " +
                              (routes.empty() ? std::string("none")
                                              : generic::route_name(routes.front())));
                const ninferno::ops::ScopedRoutePolicy legacy(ninferno::RoutePolicy::Legacy);
                failures += check(generic::table_ranked_routes(pr, width).empty(),
                                  name + ": under --route-policy legacy no ranked list is read");
            }
            if (!has_bands) continue;
            const auto q = ninferno::find_route(e[1].str());
            const generic::Problem p = problem_of(e);
            // A band from T 17 (#785) answers the int8 calls the table stands aside for, read by
            // table_predicted_answer whatever the board's table holds; a T 9-16 band answers in
            // the table's gaps (table_route_answer).
            const int width   = std::stoi(e[5].str());
            const auto answer = width > 16 ? generic::table_predicted_answer(p, width)
                                           : generic::table_route_answer(p, width, false);
            failures +=
                check(q && answer && answer->predicted &&
                          generic::route_name(answer->route) == e[6].str(),
                      name + ": its first entry answers " + e[6].str() + " as predicted, not " +
                          (answer ? generic::route_name(answer->route) : std::string("none")));
            {
                // Under the Legacy policy the file is not read: the table answers as before it.
                const ninferno::ops::ScopedRoutePolicy legacy(ninferno::RoutePolicy::Legacy);
                const auto old = generic::table_route_answer(p, width, false);
                failures += check(!old || !old->predicted,
                                  name + ": under --route-policy legacy no band is predicted");
                failures += check(!generic::table_predicted_answer(p, width),
                                  name + ": under --route-policy legacy no band is read alone");
            }
        } catch (const std::exception& error) {
            failures += check(false, name + " refused: " + error.what());
        }
    }
    failures += check(files > 0, "shipped predicted bands found in " + utf8_path(dir));
    // A pick file the loader refuses stops the lookup (a broken install): another class's file
    // under this class's name, and a band outside T 9-32. The tables are the shipped ones, copied.
    if (!first_name.empty()) {
        const board_registry::BoardRow* row = board_registry::find(first_id);
        ninferno::DeviceClass d =
            ninferno::synthetic_device_class(first_id, row->major, row->minor, row->sms,
                                             static_cast<std::size_t>(row->l2_mib * (1 << 20)), 0);
        d.facts = ninferno::machine_facts_of(d);
        const ninferno::ScopedDeviceClass scoped(d);
        const auto routes = dir.parent_path();
        // A directory of this process's own (two builds' ctest may share a stand), removed on
        // every way out.
        const auto tmp =
            std::filesystem::temp_directory_path() /
            ("ninferno_pick_refusals_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

        struct Removed {
            std::filesystem::path path;

            ~Removed() {
                std::error_code ignored;
                std::filesystem::remove_all(path, ignored);
            }
        } const removed{tmp};

        const auto refused_with = [&](const std::string& what, const std::string& text) {
            std::filesystem::remove_all(tmp);
            std::filesystem::create_directories(tmp / "pick");
            for (const auto& t : std::filesystem::directory_iterator(routes)) {
                if (t.path().extension() == ".json") {
                    std::filesystem::copy_file(t.path(), tmp / t.path().filename());
                }
            }
            std::ofstream(tmp / "pick" / first_name) << text;
            ninferno::ops::set_route_table_dir(utf8_path(tmp));
            bool refused = false;
            try {
                (void)ninferno::ops::route_table_status();
            } catch (const std::runtime_error& e) {
                refused = std::string(e.what()).find(first_name) != std::string::npos;
            }
            failures += check(refused, first_name + " " + what + " is refused, naming the file");
        };
        refused_with("as another class's file",
                     std::regex_replace(first_text, std::regex(R"re("class": "[^"]+")re"),
                                        R"re("class": "zz999:1sm")re",
                                        std::regex_constants::format_first_only));
        refused_with("with a band from T 5",
                     std::regex_replace(first_text, std::regex(R"re("bands": \[\s*\[\s*9,)re"),
                                        R"re("bands": [[5,)re",
                                        std::regex_constants::format_first_only));
        refused_with(
            "with a band past T 32",
            // The first entry's T 17-32 band widened to 40: no overlap, so the limit refuses it.
            std::regex_replace(first_text, std::regex(R"re(\[\s*17,\s*32,)re"), R"re([17, 40,)re",
                               std::regex_constants::format_first_only));
        refused_with(
            "with a ranked list past T 8",
            // A list at T 9 ahead of the first entry's own (no overlap: the limit refuses it).
            std::regex_replace(first_text, std::regex(R"re("ranked": \[)re"),
                               R"re("ranked": [[9, 9, ["rmma.m1.n1.w1x4"]], )re",
                               std::regex_constants::format_first_only));
        refused_with("with an empty ranked list",
                     std::regex_replace(first_text, std::regex(R"re("ranked": \[)re"),
                                        R"re("ranked": [[9, 9, []], )re",
                                        std::regex_constants::format_first_only));
        // Within kCarryDistance a predicted band fills the table's gaps only (#1068): a table of
        // this class (distance 0) holding the file's first problem answers it as measured; the
        // file's second problem, which the table lacks, is answered as predicted, also where a
        // similar chip's table lends it (Borrowed, fill_from).
        {
            std::vector<std::smatch> entries;
            for (auto it = std::sregex_iterator(first_text.begin(), first_text.end(), entry_re);
                 it != std::sregex_iterator() && entries.size() < 2; ++it) {
                entries.push_back(*it);
            }
            std::smatch cls;
            const std::regex class_re("\"class\": \"([^\"]+)\"");
            if (entries.size() < 2 || !std::regex_search(first_text, cls, class_re)) {
                failures +=
                    check(false, first_name + ": two entries and a class for the gap check");
            } else {
                const auto problem_of = [](const std::smatch& e) {
                    return generic::Problem{
                        ninferno::find_route(e[1].str()).value_or(ninferno::QType::BF16),
                        ninferno::ops::LinearPolicy::A16Only, std::stoi(e[3].str()),
                        std::stoi(e[4].str()),
                        e[2].str() == "swiglu" ? generic::Epilogue::SwiGlu
                        : e[2].str() == "add"  ? generic::Epilogue::AddResidual
                                               : generic::Epilogue::Store};
                };
                const std::string own = entries[0][6].str() == "rmma.m1.n1.w1x4"
                                            ? "rmma.m1.n2.w1x4"
                                            : "rmma.m1.n1.w1x4";
                std::filesystem::remove_all(tmp);
                std::filesystem::create_directories(tmp / "pick");
                std::ofstream(tmp / "pick" / first_name) << first_text;
                std::ofstream(tmp / first_name) << table(
                    cls[1].str(), "{\"format\": \"" + entries[0][1].str() +
                                      "\", \"compute\": \"a16\", \"epilogue\": \"" +
                                      entries[0][2].str() + "\", \"n\": " + entries[0][3].str() +
                                      ", \"k\": " + entries[0][4].str() +
                                      ", \"bands\": [[1, 16777216, \"" + own + "\"]]}");
                const auto similar     = ninferno::device_class().similar_chips();
                const std::string lent = entries[1][6].str() == "rmma.m1.n1.w1x4"
                                             ? "rmma.m1.n2.w1x4"
                                             : "rmma.m1.n1.w1x4";
                if (!similar.empty()) {
                    std::ofstream(tmp / (similar.front() + ".json"))
                        << table(similar.front(),
                                 "{\"format\": \"" + entries[1][1].str() +
                                     "\", \"compute\": \"a16\", " + "\"epilogue\": \"" +
                                     entries[1][2].str() + "\", \"n\": " + entries[1][3].str() +
                                     ", \"k\": " + entries[1][4].str() +
                                     ", \"bands\": [[1, 16777216, \"" + lent + "\"]]}");
                }
                ninferno::ops::set_route_table_dir(utf8_path(tmp));
                try {
                    const auto s  = ninferno::ops::route_table_status();
                    const auto a0 = generic::table_route_answer(
                        problem_of(entries[0]), std::stoi(entries[0][5].str()), false);
                    const auto a1 = generic::table_route_answer(
                        problem_of(entries[1]), std::stoi(entries[1][5].str()), false);
                    failures += check(
                        s.loaded && s.predicted_entries > 0,
                        first_name + ": a table of the class still reads the pick: " + s.message);
                    failures += check(a0 && !a0->predicted && generic::route_name(a0->route) == own,
                                      first_name +
                                          ": the table's own entry stands above the "
                                          "prediction (answers " +
                                          (a0 ? generic::route_name(a0->route) : "none") + ")");
                    failures += check(
                        a1 && a1->predicted &&
                            generic::route_name(a1->route) == entries[1][6].str(),
                        first_name + ": the table's gap takes the predicted band" +
                            (similar.empty()
                                 ? std::string()
                                 : ", ahead of " + similar.front() + "'s lent " + lent) +
                            " (answers " + (a1 ? generic::route_name(a1->route) : "none") + ")");
                    if (!similar.empty()) {
                        failures += check(s.borrowed_entries > 0,
                                          first_name + ": " + similar.front() +
                                              "'s table lends the gap (" + s.message + ")");
                    }
                    // A width the own entry has no band at is a gap too: the same entry measured
                    // from 17 tokens only leaves the file's band to answer at its width.
                    std::ofstream(tmp / first_name)
                        << table(cls[1].str(),
                                 "{\"format\": \"" + entries[0][1].str() +
                                     "\", \"compute\": \"a16\", \"epilogue\": \"" +
                                     entries[0][2].str() + "\", \"n\": " + entries[0][3].str() +
                                     ", \"k\": " + entries[0][4].str() +
                                     ", \"bands\": [[17, 16777216, \"" + own + "\"]]}");
                    // The same directory again: set_route_table_dir keeps a loaded table of an
                    // unchanged directory, so it is dropped first (clear_route_table marks it
                    // explicit, which the set resets to a fresh read of the directory).
                    ninferno::ops::clear_route_table();
                    ninferno::ops::set_route_table_dir(utf8_path(tmp));
                    const auto w0 = generic::table_route_answer(
                        problem_of(entries[0]), std::stoi(entries[0][5].str()), false);
                    failures += check(w0 && w0->predicted &&
                                          generic::route_name(w0->route) == entries[0][6].str(),
                                      first_name +
                                          ": an own entry with no band at the width leaves it to "
                                          "the predicted band (answers " +
                                          (w0 ? generic::route_name(w0->route) : "none") + ")");
                } catch (const std::exception& error) {
                    failures += check(false, first_name + " gap check refused: " + error.what());
                }
            }
        }
    } else if (files > 0) {
        failures +=
            check(false, "a shipped pick file with two entries for the refusals and the gap check");
    }
    // A board of a shipped class on other memory reads none of the class's bands (codex on #1066):
    // the RTX A4000 (448 GB/s) beside the RTX 3070 Ti's ga104:48sm (608), refused as another
    // board's file. The Tesla T4's class (tu104:40sm) ships no file since the SM span (gen.py
    // classes: sm_75's table boards span 68 SMs only), so it reads none either.
    for (const std::uint32_t id : {0x24B0U, 0x1EB8U}) {
        const bool shipped_class            = id == 0x24B0U;
        const board_registry::BoardRow* row = board_registry::find(id);
        if (row == nullptr) {
            failures += check(false, "the registry has board " + std::to_string(id));
            continue;
        }
        ninferno::DeviceClass d =
            ninferno::synthetic_device_class(id, row->major, row->minor, row->sms,
                                             static_cast<std::size_t>(row->l2_mib * (1 << 20)), 0);
        d.facts = ninferno::machine_facts_of(d);
        const ninferno::ScopedDeviceClass scoped(d);
        ninferno::ops::set_route_table_dir("");
        try {
            const auto s = ninferno::ops::route_table_status();
            std::cout << ninferno::device_class().id() << ": " << s.message << '\n';
            failures += check(s.predicted_path.empty() && s.predicted_entries == 0 &&
                                  (!shipped_class ||
                                   s.predicted_skipped.find("another board") != std::string::npos),
                              ninferno::device_class().id() +
                                  " reads no other board's predicted bands: " + s.message);
        } catch (const std::exception& error) {
            failures += check(false, std::string("refused: ") + error.what());
        }
    }
    ninferno::ops::set_route_table_dir("");
    std::cout << files << " shipped predicted files; " << (failures == 0 ? "OK" : "FAIL")
              << " route_table shipped picks\n";
    return failures == 0 ? 0 : 1;
}

// Every shipped routes/moe_pick/<class>.json (tools/bench/moe_expert_law.py pick, #987), read on a
// substitute of the board its gpu.pci_id names (no GPU): the default lookup reads it as the class's
// predicted MoE plans, and each entry answers each of its widths with its band's plan, for the
// banks' layouts it was priced for only, and nothing under the Legacy policy. A file the loader
// refuses (another class's, a band past T 8, a prefill plan, an unknown bank) stops the lookup
// naming the file; a file priced for another board of the class is not read.
int shipped_moe_picks() {
    namespace board_registry = ninferno::board_registry;
    namespace moe            = ninferno::ops::detail::moe;
    using Json               = nlohmann::json;
    int failures             = 0;
    ninferno::ops::set_route_table_dir("");
    const auto routes       = ninferno::path_from_utf8(ninferno::ops::route_table_dir());
    const auto dir          = routes / "moe_pick";
    const auto scoped_board = [](std::uint32_t id) {
        const board_registry::BoardRow* row = board_registry::find(id);
        ninferno::DeviceClass d =
            ninferno::synthetic_device_class(id, row->major, row->minor, row->sms,
                                             static_cast<std::size_t>(row->l2_mib * (1 << 20)), 0);
        d.facts = ninferno::machine_facts_of(d);
        return d;
    };
    int files = 0, plans = 0;
    std::string first_name, first_text;
    std::uint32_t first_id = 0;
    for (const auto& f : std::filesystem::directory_iterator(dir)) {
        if (f.path().extension() != ".json") continue;
        ++files;
        std::ifstream in(f.path());
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        const std::string name = f.path().filename().string();
        Json root;
        try {
            root = Json::parse(text);
        } catch (const std::exception& e) {
            failures += check(false, name + " parses: " + e.what());
            continue;
        }
        const auto id = static_cast<std::uint32_t>(
            std::stoul(root["gpu"]["pci_id"].get<std::string>(), nullptr, 16));
        if (board_registry::find(id) == nullptr) {
            failures += check(false, name + ": its pci_id is a registry id");
            continue;
        }
        if (first_name.empty()) {
            first_name = name;
            first_text = text;
            first_id   = id;
        }
        const ninferno::ScopedDeviceClass scoped(scoped_board(id));
        ninferno::ops::set_route_table_dir("");
        // The layers the file holds, by layout: a layer shipped in both layouts answers either, so
        // a flipped layout reads no plan only where the file holds no such entry (#1115).
        std::set<std::string> layouts;
        for (const Json& e : root["moe"]) {
            Json k = e;
            k.erase("bands");
            k["down_format"] = e.value("down_format", e["format"]);
            Json frag        = k.value("frag", Json::array());
            std::sort(frag.begin(), frag.end());
            k["frag"] = frag;
            layouts.insert(k.dump());
        }
        const auto holds_layout = [&](const Json& e, bool gate_up, bool down) {
            Json k = e;
            k.erase("bands");
            k["down_format"] = e.value("down_format", e["format"]);
            Json frag        = Json::array();
            if (down) frag.push_back("down");
            if (gate_up) frag.push_back("gate_up");
            k["frag"] = frag;
            return layouts.count(k.dump()) != 0;
        };
        try {
            const auto s = ninferno::ops::route_table_status();
            std::cout << name << " on " << ninferno::device_class().id() << ": " << s.message
                      << '\n';
            failures += check(
                s.loaded && ninferno::path_from_utf8(s.moe_predicted_path).filename() == name &&
                    s.moe_predicted_layers == root["moe"].size() && !s.moe_predicted_sha256.empty(),
                name + " is read as " + ninferno::device_class().id() +
                    "'s predicted MoE plans: " + s.message);
            for (const Json& e : root["moe"]) {
                const auto q = ninferno::find_route(e["format"].get<std::string>());
                const auto qd =
                    ninferno::find_route(e.value("down_format", e["format"].get<std::string>()));
                bool frag_gate_up = false, frag_down = false;
                for (const Json& b : e.value("frag", Json::array())) {
                    frag_gate_up = frag_gate_up || b == "gate_up";
                    frag_down    = frag_down || b == "down";
                }
                if (!q || !qd) {
                    failures += check(false, name + ": formats " + e.dump());
                    continue;
                }
                const moe::MoeRouteKey key{*q,
                                           *qd,
                                           e["experts"].get<std::int32_t>(),
                                           e["top_k"].get<std::int32_t>(),
                                           e["hidden"].get<std::int32_t>(),
                                           e["inter"].get<std::int32_t>()};
                for (const Json& b : e["bands"]) {
                    const auto want = moe::parse_plan(b[2].get<std::string>(), key.hidden);
                    for (std::int32_t t = b[0].get<std::int32_t>(); t <= b[1].get<std::int32_t>();
                         ++t) {
                        ++plans;
                        const auto got       = moe::predicted_plan(key, t, frag_gate_up, frag_down);
                        const std::string at = name + " " + e.dump() + " T " + std::to_string(t);
                        failures +=
                            check(got && want && moe::plan_name(*got) == moe::plan_name(*want),
                                  at + ": answers its band's plan");
                        failures +=
                            check((holds_layout(e, !frag_gate_up, frag_down) ||
                                   !moe::predicted_plan(key, t, !frag_gate_up, frag_down)) &&
                                      (holds_layout(e, frag_gate_up, !frag_down) ||
                                       !moe::predicted_plan(key, t, frag_gate_up, !frag_down)),
                                  at + ": a bank of a layout the file does not hold reads no plan");
                        const ninferno::ops::ScopedRoutePolicy legacy(
                            ninferno::RoutePolicy::Legacy);
                        failures +=
                            check(!moe::predicted_plan(key, t, frag_gate_up, frag_down),
                                  at + ": under --route-policy legacy no plan is predicted");
                    }
                }
            }
        } catch (const std::exception& error) {
            failures += check(false, name + " refused: " + error.what());
        }
    }
    failures += check(files > 0, "shipped predicted MoE plans found in " + utf8_path(dir));
    if (!first_name.empty()) {
        const ninferno::ScopedDeviceClass scoped(scoped_board(first_id));
        const auto tmp =
            std::filesystem::temp_directory_path() /
            ("ninferno_moe_pick_refusals_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

        struct Removed {
            std::filesystem::path path;

            ~Removed() {
                std::error_code ignored;
                std::filesystem::remove_all(path, ignored);
            }
        } const removed{tmp};

        // The shipped tables, copied, beside a changed copy of the first file.
        const auto install = [&](const std::string& text) {
            std::filesystem::remove_all(tmp);
            std::filesystem::create_directories(tmp / "moe_pick");
            for (const auto& t : std::filesystem::directory_iterator(routes)) {
                if (t.path().extension() == ".json") {
                    std::filesystem::copy_file(t.path(), tmp / t.path().filename());
                }
            }
            std::ofstream(tmp / "moe_pick" / first_name) << text;
            ninferno::ops::set_route_table_dir(utf8_path(tmp));
        };
        // The first file with its first entry changed by `edit` (the file's other fields kept).
        const auto changed = [&](const std::function<void(Json&, Json&)>& edit) {
            Json root = Json::parse(first_text);
            edit(root, root["moe"][0]);
            return root.dump(1);
        };
        const auto refused_with = [&](const std::string& what, const std::string& text) {
            install(text);
            bool refused = false;
            try {
                (void)ninferno::ops::route_table_status();
            } catch (const std::runtime_error& e) {
                refused = std::string(e.what()).find(first_name) != std::string::npos;
            }
            failures += check(refused, first_name + " " + what + " is refused, naming the file");
        };
        refused_with("as another class's file",
                     changed([](Json& root, Json&) { root["gpu"]["class"] = "zz999:1sm"; }));
        // A register-MMA band (no R8 stage, which moe_entry_of refuses past T 8 itself): the
        // pick's own limit of T 1-8.
        refused_with("with a band to T 9", changed([](Json&, Json& e) {
                         e["bands"] =
                             Json::array({Json::array({1, 9, "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4"})});
                     }));
        refused_with("with a prefill plan", changed([](Json&, Json& e) {
                         e["prefill"] = "rmma.m1.n8.w4x1|rmma.m1.n8.w4x1";
                     }));
        refused_with("with an unknown bank in frag",
                     changed([](Json&, Json& e) { e["frag"] = Json::array({"router"}); }));
        // Fragment order named on a bank whose format has none (Q5_K): never asked, refused.
        refused_with("with frag on a format without fragment order", changed([](Json&, Json& e) {
                         e["format"]      = "q5k";
                         e["down_format"] = "q5k";
                         e["frag"]        = Json::array({"gate_up"});
                     }));
        // Priced for another board (its gpu.pci_id): not read, and the status says so. The loader
        // skips a file whose board is at a nonzero board_distance from this one (a board of the
        // same facts reads it), so the other board is a registry board of the first file's arch at
        // a nonzero distance, whichever file that is (#1115: a fixed 0x2204 / 0x2206 assumed
        // GA102).
        std::string other;
        const ninferno::Board first_board = board_registry::board(first_id);
        for (const board_registry::BoardRow& row : board_registry::kRows) {
            const ninferno::Board b = board_registry::board(row.pci_device_id);
            if (row.major != first_board.major || row.minor != first_board.minor ||
                !(b.dram_gbps > 0) || ninferno::board_distance(first_board, b) == 0.0) {
                continue;
            }
            char hex[16];
            std::snprintf(hex, sizeof hex, "0x%04X", static_cast<unsigned>(row.pci_device_id));
            other = hex;
            break;
        }
        failures += check(!other.empty(), "a registry board of another facts than " + first_name);
        install(changed([&](Json& root, Json&) { root["gpu"]["pci_id"] = other; }));
        try {
            const auto s = ninferno::ops::route_table_status();
            failures +=
                check(s.moe_predicted_path.empty() && s.moe_predicted_layers == 0 &&
                          s.moe_predicted_skipped.find("another board") != std::string::npos,
                      first_name + " priced for " + other + " is not read: " + s.message);
        } catch (const std::exception& error) {
            failures += check(false, first_name + " for another board refused: " + error.what());
        }
    }
    ninferno::ops::set_route_table_dir("");
    std::cout << files << " shipped predicted MoE files, " << plans << " widths; "
              << (failures == 0 ? "OK" : "FAIL") << " route_table shipped moe picks\n";
    return failures == 0 ? 0 : 1;
}

// The pick's precedence (#1115 item 1, moe.cpp table_authority / table_entry_near): on a substitute
// of a shipped moe_pick file's board, a layer the file plans (and the board's shipped table does
// not) takes the pick; a band of this board's own table for the layer stands above the pick; the
// same band lent by a similar chip's table (fill_from) yields to it.
int moe_pick_precedence() {
    namespace board_registry = ninferno::board_registry;
    namespace moe            = ninferno::ops::detail::moe;
    using Json               = nlohmann::json;
    int failures             = 0;
    ninferno::ops::set_route_table_dir("");
    const auto routes = ninferno::path_from_utf8(ninferno::ops::route_table_dir());
    const auto read   = [](const std::filesystem::path& path) {
        std::ifstream in(path);
        return Json::parse(
            std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
    };
    const auto table_answer = [](moe::MoeAuthority a) {
        return a == moe::MoeAuthority::TableAsMeasured || a == moe::MoeAuthority::TableW4a8 ||
               a == moe::MoeAuthority::TableClassA8 || a == moe::MoeAuthority::TableR8Refused;
    };
    // In name order: the directory's own order differs by filesystem.
    std::vector<std::filesystem::path> files;
    for (const auto& f : std::filesystem::directory_iterator(routes / "moe_pick")) {
        if (f.path().extension() == ".json") files.push_back(f.path());
    }
    std::sort(files.begin(), files.end());
    for (const auto& path : files) {
        const std::filesystem::directory_entry f(path);
        const auto name = f.path().filename();
        const Json pick = read(f.path());
        const auto id   = static_cast<std::uint32_t>(
            std::stoul(pick["gpu"]["pci_id"].get<std::string>(), nullptr, 16));
        const board_registry::BoardRow* row = board_registry::find(id);
        if (row == nullptr) continue;
        ninferno::DeviceClass board =
            ninferno::synthetic_device_class(id, row->major, row->minor, row->sms,
                                             static_cast<std::size_t>(row->l2_mib * (1 << 20)), 0);
        board.facts = ninferno::machine_facts_of(board);
        const ninferno::ScopedDeviceClass scoped(board);
        // The board's own table: the first of its class chain with a shipped file ("ga102-82sm"
        // -> "ga102"), named as the lookup names it (':' is '-').
        std::filesystem::path own_name;
        for (std::string c : ninferno::device_class().class_chain()) {
            for (char& ch : c) {
                if (ch == ':') ch = '-';
            }
            if (std::filesystem::exists(routes / (c + ".json"))) {
                own_name = c + ".json";
                break;
            }
        }
        if (own_name.empty()) continue;
        for (const Json& e : pick["moe"]) {
            const auto q = ninferno::find_route(e["format"].get<std::string>());
            const auto qd =
                ninferno::find_route(e.value("down_format", e["format"].get<std::string>()));
            if (!q || !qd) continue;
            bool frag_gate_up = false, frag_down = false;
            for (const Json& b : e.value("frag", Json::array())) {
                frag_gate_up = frag_gate_up || b == "gate_up";
                frag_down    = frag_down || b == "down";
            }
            moe::MoeLayerShape layer;
            layer.gate_up        = *q;
            layer.down           = *qd;
            layer.gate_up_layout = frag_gate_up ? ninferno::QuantLayout::GgmlFrag
                                                : ninferno::default_weight_layout(*q);
            layer.down_layout =
                frag_down ? ninferno::QuantLayout::GgmlFrag : ninferno::default_weight_layout(*qd);
            layer.experts        = e["experts"].get<std::int32_t>();
            layer.top_k          = e["top_k"].get<std::int32_t>();
            layer.hidden         = e["hidden"].get<std::int32_t>();
            layer.inter          = e["inter"].get<std::int32_t>();
            const Json& band     = e["bands"][0];
            const std::int32_t t = band[0].get<std::int32_t>();
            ninferno::ops::set_route_table_dir("");
            const auto shipped = moe::moe_route_query(board, layer, t);
            if (!shipped || shipped->authority != moe::MoeAuthority::MoePick) continue;
            const std::string at = name.string() + " " + e.dump() + " T " + std::to_string(t);
            // The layer's band as a table entry: the pick's own plan, so only the authority tells.
            Json entry = {
                {"format", e["format"]}, {"experts", e["experts"]},
                {"top_k", e["top_k"]},   {"hidden", e["hidden"]},
                {"inter", e["inter"]},   {"bands", Json::array({Json::array({t, t, band[2]})})}};
            if (e.contains("down_format")) entry["down_format"] = e["down_format"];
            const auto dir = scratch_dir() / "moe_precedence";
            std::filesystem::remove_all(dir);
            std::filesystem::create_directories(dir / "moe_pick");
            std::filesystem::copy_file(f.path(), dir / "moe_pick" / name);
            // This board's own table, with the band: it stands above the pick.
            Json own   = read(routes / own_name);
            own["moe"] = own.value("moe", Json::array());
            // An own entry of the layer already (at other widths) would make a duplicate key: the
            // layer is not this case's.
            const auto same_layer = [&](const Json& m) {
                return m.value("format", Json()) == entry["format"] &&
                       m.value("down_format", m.value("format", Json())) ==
                           entry.value("down_format", entry["format"]) &&
                       m.value("experts", Json()) == entry["experts"] &&
                       m.value("top_k", Json()) == entry["top_k"] &&
                       m.value("hidden", Json()) == entry["hidden"] &&
                       m.value("inter", Json()) == entry["inter"];
            };
            if (std::any_of(own["moe"].begin(), own["moe"].end(), same_layer)) continue;
            own["moe"].push_back(entry);
            std::ofstream(dir / own_name) << own.dump(1);
            ninferno::ops::set_route_table_dir(utf8_path(dir));
            const auto near = moe::moe_route_query(board, layer, t);
            failures += check(near && table_answer(near->authority),
                              at + ": the board's own table band stands above the pick");
            // The same band in a similar chip's table, lent to this board's: it yields to the pick.
            const auto similar = ninferno::device_class().similar_chips();
            if (!similar.empty()) {
                std::ofstream(dir / own_name) << read(routes / own_name).dump(1);
                std::string lender = table(similar.front(), "");
                lender.pop_back();
                lender += ", \"moe\": [" + entry.dump() + "]}";
                std::ofstream(dir / (similar.front() + ".json")) << lender;
                // The same directory again keeps the loaded table (set_route_table_dir): reset
                // first, so the lookup reads the files as they are now.
                ninferno::ops::set_route_table_dir("");
                ninferno::ops::set_route_table_dir(utf8_path(dir));
                const auto s    = ninferno::ops::route_table_status();
                const auto lent = moe::moe_route_query(board, layer, t);
                failures += check(s.borrowed_moe_layers >= 1 && lent &&
                                      lent->authority == moe::MoeAuthority::MoePick,
                                  at + ": a band lent by " + similar.front() +
                                      " yields to the pick: " + s.message);
            } else {
                std::cout << "no similar chip of " << ninferno::device_class().id()
                          << ": the lent case is not run\n";
            }
            ninferno::ops::set_route_table_dir("");
            std::cout << "moe pick precedence on " << name.string() << ": "
                      << (failures == 0 ? "OK" : "FAIL") << '\n';
            return failures == 0 ? 0 : 1;
        }
    }
    ninferno::ops::set_route_table_dir("");
    std::cout << "SKIP: no shipped moe pick answers a layer its board's table lacks\n";
    return 77;
}

} // namespace

// A malformed default table (the device's own class file in the --route-dir directory) stops the
// lookup with an error naming the directory and why, instead of answering as untuned (another
// route, speed and, for the A8 and A4 routes, other bytes). Run in its own process: the default
// lookup happens once.
int malformed_default() {
    std::string file = ninferno::ops::route_gpu_class();
    for (char& c : file) {
        if (c == ':') c = '-';
    }
    const auto dir  = scratch_dir() / "malformed";
    const auto path = dir / (file + ".json");
    std::filesystem::create_directories(dir);
    std::ofstream(path) << "{\"schema\": \"ninferno-route-table/1\", \"gpu\": ";
    ninferno::ops::set_route_table_dir(dir);
    std::string message;
    try {
        (void)ninferno::ops::route_table_status();
    } catch (const std::exception& error) {
        message = error.what();
    }
    std::cout << "malformed default table: " << message << '\n';
    const int failures =
        check(message.find("refused") != std::string::npos &&
                  message.find(utf8_path(dir)) != std::string::npos,
              "a malformed default table stops the lookup, naming the directory and why: " + message);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " route_table malformed default\n";
    return failures == 0 ? 0 : 1;
}

// The default lookup fills the problems the device's own table lacks from its most similar chip's
// table, keeps its own entry where both have one, and says so in the status; an explicit table
// stands alone. Run in its own process: the default lookup happens once.
int similar_default() {
    const auto similar = ninferno::device_class().similar_chips();
    if (similar.empty()) {
        std::cout << "SKIP: no similar chip of this architecture\n";
        return 77;
    }
    const std::string gpu = ninferno::ops::route_gpu_class();
    std::string file      = gpu;
    for (char& c : file) {
        if (c == ':') c = '-';
    }
    const auto dir = scratch_dir() / "similar";
    std::filesystem::create_directories(dir);
    const auto entry = [](int k, const char* route) {
        return std::string(R"({"format": "q8", "compute": "a16", "epilogue": "store", "n": 4096, "k": )") +
               std::to_string(k) + R"(, "bands": [[1, 16, ")" + route + R"("]]})";
    };
    std::ofstream(dir / (file + ".json")) << table(gpu, entry(1024, "rmma.m1.n1.w1x8"));
    // The similar chip's table also holds an MoE layer the own table lacks.
    std::string lender =
        table(similar.front(), entry(1024, "rmma.m1.n1.w1x4") + ", " + entry(2048, "rmma.m1.n1.w2x2"));
    lender.pop_back();
    lender += R"(, "moe": [{"format": "mxfp4", "experts": 32, "top_k": 4, "hidden": 2880, "inter": 2880,
                "bands": [[1, 16, "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4"]]}]})";
    std::ofstream(dir / (similar.front() + ".json")) << lender;
    // A table measured on a board of another instruction set lends nothing, however near its SMs.
    const int foreign_cc = ninferno::device_class().compute_capability() == 70 ? 75 : 70;
    std::ofstream(dir / "foreign.json")
        << "{\"schema\": \"ninferno-route-table/1\", \"gpu\": {\"class\": \"zz999\", \"name\": \"test\", "
           "\"board\": {\"cc\": " + std::to_string(foreign_cc) + ", \"sms\": " +
               std::to_string(ninferno::device_class().sms) + "}}, \"entries\": [" + entry(4096, "rmma.m1.n1.w1x4") +
               "]}";
    ninferno::ops::set_route_table_dir(dir);
    const generic::Problem own{ninferno::QType::Q8_G32_FP16, ninferno::ops::LinearPolicy::A16Only,
                               4096, 1024, generic::Epilogue::Store};
    const generic::Problem lent{ninferno::QType::Q8_G32_FP16, ninferno::ops::LinearPolicy::A16Only,
                                4096, 2048, generic::Epilogue::Store};
    const auto s = ninferno::ops::route_table_status();
    std::cout << "similar default table: " << s.message << '\n';
    int failures = check(s.loaded && s.gpu_class == gpu, "the own table is the default: " + s.message);
    failures += check(route_or_none(generic::table_route(own, 4, false)) == "rmma.m1.n1.w1x8",
                      "a problem of the own table keeps its route");
    failures += check(route_or_none(generic::table_route(lent, 4, false)) == "rmma.m1.n1.w2x2",
                      "a problem the own table lacks takes the similar chip's entry");
    failures +=
        check(s.lender_sha256 ==
                  similar.front() + " " +
                      ninferno::file_sha256_hex(dir / (similar.front() + ".json")).value_or(""),
              "the status names each lending table by its content: " + s.lender_sha256);
    failures += check(s.borrowed_entries == 1 && s.borrowed_moe_layers == 1 &&
                          s.borrowed_from.find(similar.front()) == 0 &&
                          s.message.find("from the next nearest (" + similar.front() + " 2)") != std::string::npos,
                      "the status says what was borrowed from whom: " + s.message);
    const ninferno::ops::detail::moe::MoeRouteKey layer{ninferno::QType::MXFP4, ninferno::QType::MXFP4,
                                                        32, 4, 2880, 2880};
    failures += check(ninferno::ops::detail::moe::table_plan(layer, 4, false).has_value(),
                      "an MoE layer the own table lacks takes the similar chip's bands");
    failures += check(!generic::table_claims(lent, 4, false), "a borrowed entry claims nothing");
    const generic::Problem foreign{ninferno::QType::Q8_G32_FP16, ninferno::ops::LinearPolicy::A16Only,
                                   4096, 4096, generic::Epilogue::Store};
    failures += check(!generic::table_route(foreign, 4, false).has_value(),
                      "a table of another instruction set lends nothing");
    // An explicit table stands alone.
    (void)ninferno::ops::load_route_table(utf8_path(dir / (file + ".json")));
    failures += check(!generic::table_route(lent, 4, false).has_value(),
                      "an explicit table borrows nothing");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " route_table similar default\n";
    return failures == 0 ? 0 : 1;
}

// The board of every shipped table, read on a substitute of the board its gpu.pci_id names (no
// GPU): the table is that board's own, at distance 0; an id that is not its class's board is
// refused. A shipped id mistyped into another valid board, or another chip's, fails here.
int shipped_boards() {
    namespace board_registry = ninferno::board_registry;
    const auto substitute    = [](std::uint32_t id) {
        const board_registry::BoardRow* row = board_registry::find(id);
        ninferno::DeviceClass d =
            ninferno::synthetic_device_class(id, row->major, row->minor, row->sms,
                                                static_cast<std::size_t>(row->l2_mib * (1 << 20)), 0);
        d.facts = ninferno::machine_facts_of(d);
        return d;
    };
    int failures = 0;
    ninferno::ops::set_route_table_dir("");
    const std::string shipped = ninferno::ops::route_table_dir();
    const std::regex pci_re("\"pci_id\": \"(0x[0-9a-f]{4})\"");
    int files = 0;
    for (const auto& f : std::filesystem::directory_iterator(ninferno::path_from_utf8(shipped))) {
        if (f.path().extension() != ".json") continue;
        ++files;
        std::ifstream in(f.path());
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        std::smatch m;
        const std::string name = f.path().filename().string();
        if (!std::regex_search(text, m, pci_re)) {
            failures += check(false, name + " names its board by gpu.pci_id");
            continue;
        }
        const auto id = static_cast<std::uint32_t>(std::stoul(m[1].str(), nullptr, 16));
        if (board_registry::find(id) == nullptr) {
            failures += check(false, name + ": " + m[1].str() + " is a registry id");
            continue;
        }
        // The id's registry row is the card the table names (gpu.name, the driver's name, which
        // omits the suffix of a memory size, "Lite Hash Rate" or "Rev. A"): a typo into another
        // product fails, the RTX 4090 D's id for the RTX 4090 included. A table named by its chip
        // alone names no card.
        const auto normal = [](std::string s) {
            if (s.rfind("NVIDIA ", 0) == 0) s.erase(0, 7);
            std::replace(s.begin(), s.end(), '-', ' ');
            return s;
        };
        std::smatch named;
        const std::regex name_re("\"class\": \"([^\"]*)\",\\s*\"name\": \"([^\"]*)\"");
        const ninferno::DeviceClass on = substitute(id);
        if (!std::regex_search(text, named, name_re)) {
            // The check reads gpu.class and gpu.name side by side, as write_route_table writes
            // them: a table that has them apart is a failure, not a table left unchecked.
            failures += check(false, name + " has gpu.class and gpu.name side by side");
        } else if (named[2].str() != named[1].str().substr(0, named[1].str().find(':'))) {
            const std::string row = normal(on.facts.sku), card = normal(named[2].str());
            const std::regex omitted(R"( ([0-9]+GB|Lite Hash Rate|Rev\. A))");
            failures += check(row.rfind(card, 0) == 0 &&
                                  (row.size() == card.size() ||
                                   std::regex_match(row.substr(card.size()), omitted)),
                              name + ": " + m[1].str() + " (" + on.facts.sku + ") is the " +
                                  named[2].str() + " the table names");
        }
        const ninferno::ScopedDeviceClass scoped(on);
        try {
            const auto board          = generic::route_table_board(utf8_path(f.path()));
            const ninferno::Board own = ninferno::board_of(ninferno::device_class());
            const double d            = board ? ninferno::board_distance(own, *board) : -1.0;
            std::cout << name << ": " << m[1].str() << " on " << ninferno::device_class().id()
                      << ", distance " << d << '\n';
            failures += check(board && d == 0.0, name + " is measured on the board of its id");
        } catch (const std::exception& e) {
            failures += check(false, name + " refused on the board of its id: " + e.what());
        }
    }
    failures += check(files > 0, "shipped tables found in " + shipped);
    // An id that names another board than the class's is refused: another chip, or the chip's
    // board with other SMs than a chip:Nsm class names.
    const ninferno::ScopedDeviceClass scoped(substitute(0x2782));
    const auto refused_as = [&](const std::string& name, const std::string& cls, const char* id) {
        const std::string path = write_file(
            name, "{\"schema\": \"ninferno-route-table/1\", \"gpu\": {\"class\": \"" + cls +
                      "\", \"name\": \"test\", \"pci_id\": \"" + id + "\"}, \"entries\": []}");
        try {
            (void)generic::route_table_board(path);
        } catch (const std::runtime_error&) { return true; }
        return false;
    };
    failures += check(!refused_as("id_own", "ad104:60sm", "0x2782"), "the class's own board loads");
    failures += check(!refused_as("id_chip", "ad104", "0x27b8"), "a board of the bare chip loads");
    failures += check(refused_as("id_other_chip", "ad104:60sm", "0x2684"),
                      "an id of another chip is refused");
    failures += check(refused_as("id_other_sms", "ad104:60sm", "0x27b8"),
                      "an id of the chip's board with other SMs is refused");
    // The writer names a board by its id only where the board is its id's row: the 8 GB RTX 3060
    // shares the 12 GB board's id on a 128-bit bus (core/machine_facts.cpp prices it at the
    // driver's), so its table records its facts and no id, which would read back as 360 GB/s.
    {
        const auto written_on = [&](ninferno::DeviceClass d) {
            const ninferno::ScopedDeviceClass on(std::move(d));
            return generic::write_route_table(ninferno::device_class().id(), "test", {});
        };
        const std::string stock = written_on(substitute(0x2504));
        failures += check(stock.find("\"pci_id\": \"0x2504\"") != std::string::npos &&
                              stock.find("\"board\"") == std::string::npos,
                          "the 12 GB RTX 3060 is named by its id: " + stock);
        // The A100 PCIe 80GB's bandwidth differs from its row's in the last bit
        // (measured::row_board_facts): it is named by its id all the same.
        const std::string a100 = written_on(substitute(0x20B5));
        failures += check(a100.find("\"pci_id\": \"0x20b5\"") != std::string::npos &&
                              a100.find("\"board\"") == std::string::npos,
                          "the A100 PCIe 80GB is named by its id: " + a100);
        ninferno::DeviceClass narrow = substitute(0x2504);
        narrow.memory_bus_bits       = 128;
        narrow.facts                 = ninferno::machine_facts_of(narrow);
        const std::string twin       = written_on(narrow);
        failures += check(twin.find("\"pci_id\"") == std::string::npos &&
                              twin.find("\"dram_gbps\": 240") != std::string::npos,
                          "the 8 GB RTX 3060 records its own facts, not the id: " + twin);
    }
    std::cout << files << " shipped tables; " << (failures == 0 ? "OK" : "FAIL")
              << " route_table shipped boards\n";
    return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--shipped-boards") { return shipped_boards(); }
    if (argc == 2 && std::string(argv[1]) == "--shipped-picks") { return shipped_picks(); }
    if (argc == 2 && std::string(argv[1]) == "--shipped-moe-picks") { return shipped_moe_picks(); }
    if (argc == 2 && std::string(argv[1]) == "--moe-pick-precedence") {
        return moe_pick_precedence();
    }
    if (ninferno::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (argc == 2 && std::string(argv[1]) == "--malformed-default") { return malformed_default(); }
    if (argc == 2 && std::string(argv[1]) == "--similar-default") { return similar_default(); }
    int failures = 0;
    const auto& device    = ninferno::device_class();
    const std::string gpu = ninferno::ops::route_gpu_class();

    // Default lookup, before anything else touches the table: the device's variant key
    // ("gb202:170sm") is found as "gb202-170sm.json" in the --route-dir directory
    // (set_route_table_dir), and keeps its ':' inside.
    {
        std::string file = gpu;
        for (char& c : file) {
            if (c == ':') c = '-';
        }
        const auto dir = scratch_dir() / "dir";
        std::filesystem::create_directories(dir);
        std::ofstream(dir / (file + ".json")) << table(gpu, "");
        ninferno::ops::set_route_table_dir(dir);
        const auto s = ninferno::ops::route_table_status();
        failures +=
            check(s.loaded && s.gpu_class == gpu &&
                      ninferno::path_from_utf8(s.path).filename() == file + ".json",
                  "the default lookup finds " + file + ".json for " + gpu + ": " + s.message);
        failures += check(ninferno::ops::route_table_dir() == utf8_path(dir),
                          "set_route_table_dir names the dir");
        // An explicit table is dropped when the directory is named again, the same one included
        // (an engine built with no --route-table after one that had it).
        const auto explicit_path = write_file("explicit", table(gpu, ""));
        (void)ninferno::ops::load_route_table(explicit_path);
        failures += check(ninferno::ops::route_table_status().path == explicit_path,
                          "the explicit table is not active");
        ninferno::ops::set_route_table_dir(dir);
        const auto back = ninferno::ops::route_table_status();
        failures +=
            check(back.loaded && ninferno::path_from_utf8(back.path).filename() == file + ".json",
                  "an explicit table outlived set_route_table_dir: " + back.message);
    }

    const generic::Problem q8{ninferno::QType::Q8_G32_FP16, ninferno::ops::LinearPolicy::A16Only,
                              4096, 1024, generic::Epilogue::Store};
    const generic::Problem other{ninferno::QType::Q8_G32_FP16, ninferno::ops::LinearPolicy::A16Only,
                                 4096, 2048, generic::Epilogue::Store};
    const detail::ShapeRouteKey a4{ninferno::QType::NVFP4, detail::ShapeCompute::A4, 14336, 5120};
    const detail::ShapeRouteKey both{ninferno::QType::Q8_G32_FP16, detail::ShapeCompute::A16, 4096,
                                     1024};
    const detail::ShapeRouteKey a8{ninferno::QType::FP8_E4M3FN_ROW_BF16, detail::ShapeCompute::A8,
                                   5120, 6144};

    // One file, both dialects: bands + prefill for the generic path, route_as for the shape
    // tables (one entry carries both), an "a8" route_as entry, an entry without an epilogue
    // (store), and keys no consumer reads.
    const std::string good = table(
        gpu, R"({"format": "q8", "compute": "a16", "epilogue": "store", "n": 4096, "k": 1024,
                 "bands": [[1, 4, "rmma.m1.n1.w1x8"], [5, 63, "rmma.m1.n1.w1x4"],
                           [64, 16777216, "rmma.m1.n2.w2x2"]],
                 "prefill": "rmma.m1.n8.w4x1", "route_as": [[2, 3, 4]], "note": "ignored"},
                {"format": "nvfp4", "compute": "a4", "epilogue": "store", "n": 14336,
                 "k": 5120, "route_as": [[48, 96, 192]], "evidence_us": [[48, 1.0, 0.5]]},
                {"format": "fp8", "compute": "a8", "n": 5120, "k": 6144,
                 "route_as": [[20, 40, 64]]})");
    const auto good_path = write_file("good", good);
    const auto status    = ninferno::ops::load_route_table(good_path);
    failures += check(status.loaded && status.entries == 3 && status.generic_entries == 1 &&
                          status.route_as_entries == 3,
                      "a valid table loads, both dialects counted: " + status.message);
    // The decision fingerprint names the table by the bytes that were parsed.
    failures +=
        check(status.content_sha256.size() == 64 &&
                  status.content_sha256 == ninferno::file_sha256_hex(good_path).value_or("") &&
                  status.lender_sha256.empty(),
              "the status carries the SHA-256 of the file's bytes: " + status.content_sha256);
    failures += check(route_or_none(generic::table_route(q8, 1, false)) == "rmma.m1.n1.w1x8",
                      "T=1 takes the first band");
    failures += check(route_or_none(generic::table_route(q8, 16, false)) == "rmma.m1.n1.w1x4",
                      "T=16 takes the second band");
    failures += check(route_or_none(generic::table_route(q8, 100, false)) == "rmma.m1.n2.w2x2",
                      "T=100 outside the prefill profile takes the third band");
    failures += check(route_or_none(generic::table_route(q8, 16, true)) == "rmma.m1.n1.w1x4",
                      "T=16 under the prefill profile keeps the band (prefill starts at 17)");
    failures += check(route_or_none(generic::table_route(q8, 17, true)) == "rmma.m1.n8.w4x1",
                      "T=17 under the prefill profile takes the prefill route");
    failures += check(route_or_none(generic::table_route(q8, 5000, true)) == "rmma.m1.n8.w4x1",
                      "T=5000 under the prefill profile takes the prefill route");
    failures += check(!generic::table_route(other, 1, false), "an absent problem has no route");
    failures += check(detail::route_tokens(both, 2) == 4 && detail::route_tokens(both, 3) == 4 &&
                          detail::route_tokens(both, 5) == 5,
                      "the entry with bands also carries route_as for the shape table");
    failures += check(detail::route_tokens(a4, 48) == 192 && detail::route_tokens(a4, 97) == 97,
                      "an a4 route_as band");
    failures += check(detail::route_tokens(a8, 20) == 64 && detail::route_tokens(a8, 41) == 41,
                      "an a8 route_as band is keyed by the FP8 A8 table");
    failures += check(detail::route_tokens({ninferno::QType::FP8_E4M3FN_ROW_BF16,
                                            detail::ShapeCompute::A16, 5120, 6144},
                                           20) == 20,
                      "the a8 band does not steer the shape's A16 table");

    // Every key of the device's class chain is accepted: chip, architecture, "*".
    for (const std::string& key : {device.chip, device.arch(), std::string("*")}) {
        const auto s = ninferno::ops::load_route_table(write_file("key", table(key, "")));
        failures += check(s.loaded, "a table keyed " + key + " is applied: " + s.message);
    }
    (void)ninferno::ops::load_route_table(write_file("good", good));

    // Another chip (or a chip variant of another SM count): not applied, the previous table stays.
    const auto foreign = ninferno::ops::load_route_table(write_file("foreign", table("zz999", "")));
    failures += check(!ninferno::ops::load_route_table(
                           write_file("variant", table(device.chip + ":1sm", "")))
                           .loaded,
                      "a table of another SM variant of this chip is not applied");
    failures += check(!foreign.loaded, "a table of another class is not applied");
    failures += check(route_or_none(generic::table_route(q8, 1, false)) == "rmma.m1.n1.w1x8" &&
                          detail::route_tokens(a4, 48) == 192,
                      "the loaded table stays after a refused one");

    // Prefill bands: under the prefill profile a band of routes that sum as the prefill route does
    // (WarpsK == 1 register MMA, staged) narrows the prefill route by width; outside the profile
    // and between the bands the entry's own routes answer.
    {
        const generic::Problem bf16{ninferno::QType::BF16, ninferno::ops::LinearPolicy::A16Only,
                                    3072, 4096, generic::Epilogue::Store};
        const auto s = ninferno::ops::load_route_table(write_file("pbands", table(
            gpu, R"({"format": "bf16", "compute": "a16", "epilogue": "store", "n": 3072, "k": 4096,
                     "bands": [[1, 16777216, "rmma.m1.n1.w1x8"]], "prefill": "staged.128x128.w2x4.s3",
                     "prefill_bands": [[17, 1023, "staged.64x128.w1x4.s4"],
                                       [2048, 4095, "rmma.m1.n8.w4x1"]]})")));
        failures += check(s.loaded && s.generic_entries == 1, "prefill bands load: " + s.message);
        failures += check(route_or_none(generic::table_route(bf16, 16, true)) == "rmma.m1.n1.w1x8",
                          "T=16 under the prefill profile keeps the band");
        failures += check(route_or_none(generic::table_route(bf16, 17, true)) == "staged.64x128.w1x4.s4" &&
                              route_or_none(generic::table_route(bf16, 1023, true)) ==
                                  "staged.64x128.w1x4.s4",
                          "T=17..1023 under the prefill profile take the first prefill band");
        failures += check(route_or_none(generic::table_route(bf16, 1024, true)) ==
                              "staged.128x128.w2x4.s3",
                          "T=1024 (between prefill bands) takes the prefill route");
        failures += check(route_or_none(generic::table_route(bf16, 3000, true)) == "rmma.m1.n8.w4x1",
                          "T=3000 takes the second prefill band (a WarpsK == 1 register MMA)");
        failures += check(route_or_none(generic::table_route(bf16, 512, false)) == "rmma.m1.n1.w1x8",
                          "outside the prefill profile the bands answer");
        failures += check(refused(gpu, "pband_split", R"({"format": "bf16", "compute": "a16",
                                     "n": 3072, "k": 4096, "prefill": "staged.128x128.w2x4.s3",
                                     "prefill_bands": [[17, 64, "rmma.m1.n1.w1x8"]]})"),
                          "a prefill band that sums in another order (K split) is refused");
        failures += check(refused(gpu, "pband_low", R"({"format": "bf16", "compute": "a16",
                                     "n": 3072, "k": 4096, "prefill": "staged.128x128.w2x4.s3",
                                     "prefill_bands": [[16, 64, "staged.64x128.w1x4.s4"]]})"),
                          "a prefill band below T = 17 is refused");
        failures += check(refused(gpu, "pband_alone", R"({"format": "bf16", "compute": "a16",
                                     "n": 3072, "k": 4096, "bands": [],
                                     "prefill_bands": [[17, 64, "staged.64x128.w1x4.s4"]]})"),
                          "prefill bands without a prefill route are refused");
        (void)ninferno::ops::load_route_table(write_file("good", good));
    }

    // Refusals. Chunk invariance first: a prefill route with a K split (not token-invariant), and
    // a route_as band that would ask a table about fewer tokens than the band's calls have.
    failures += check(refused(gpu, "splitk", R"({"format": "q8", "compute": "a16", "epilogue": "store",
                                     "n": 4096, "k": 1024, "bands": [],
                                     "prefill": "rmma.m1.n1.w1x8"})"),
                      "a K-split prefill route is refused");
    failures += check(refused(gpu, "narrow", R"({"format": "q8", "compute": "a16", "n": 4096,
                                     "k": 1024, "route_as": [[17, 64, 32]]})"),
                      "a route_as width below its band's t_max is refused");
    failures += check(refused(gpu, "overlap", R"({"format": "q8", "compute": "a16", "epilogue": "store",
                                      "n": 4096, "k": 1024,
                                      "bands": [[1, 8, "rmma.m1.n1.w1x8"], [8, 9, "rmma.m1.n1.w1x4"]]})"),
                      "overlapping bands are refused");
    failures += check(refused(gpu, "overlap_as", R"({"format": "q8", "compute": "a16", "n": 1,
                                     "k": 1, "route_as": [[2, 8, 12], [8, 16, 24]]})"),
                      "overlapping route_as bands are refused");
    failures += check(refused(gpu, "unknown", R"({"format": "q8", "compute": "a16", "epilogue": "store",
                                      "n": 4096, "k": 1024, "bands": [[1, 8, "rmma.m9.n9.w9x9"]]})"),
                      "an unknown route name is refused");
    failures += check(refused(gpu, "a8bands", R"({"format": "fp8", "compute": "a8", "n": 5120,
                                      "k": 6144, "bands": [[1, 8, "rmma.m1.n1.w1x8"]]})"),
                      "bands on a8 (no generic A8 schedule) are refused");
    failures += check(refused(gpu, "a4q8", R"({"format": "q8", "compute": "a4", "n": 4096,
                                      "k": 1024, "route_as": [[2, 2, 3]]})"),
                      "a4 on a format without W4A4 is refused");
    failures += check(refused(gpu, "a8nvfp4", R"({"format": "nvfp4", "compute": "a8", "n": 4096,
                                      "k": 1024, "route_as": [[2, 2, 3]]})"),
                      "a8 on a format without W8A8 is refused");
    failures += check(refused(gpu, "fused", R"({"format": "q8", "compute": "a16", "epilogue": "add",
                                      "n": 4096, "k": 1024, "route_as": [[2, 8, 12]]})"),
                      "route_as on a fused epilogue is refused");
    failures += check(refused(gpu, "dup", R"({"format": "q8", "compute": "a16", "n": 4096, "k": 1024,
                                      "bands": [[1, 8, "rmma.m1.n1.w1x8"]]},
                                     {"format": "q8", "compute": "a16", "epilogue": "store",
                                      "n": 4096, "k": 1024, "route_as": [[2, 2, 3]]})"),
                      "a duplicate problem (one per dialect) is refused");
    failures += check(refused(gpu, "zero", R"({"format": "q8", "compute": "a16", "n": 1, "k": 1,
                                      "route_as": [[0, 8, 12]]})"),
                      "a band from T=0 is refused");
    failures += check(refused(gpu, "format", R"({"format": "q3", "compute": "a16", "n": 1, "k": 1,
                                      "route_as": [[2, 8, 12]]})"),
                      "an unknown format is refused");
    failures += check(route_or_none(generic::table_route(q8, 1, false)) == "rmma.m1.n1.w1x8",
                      "a refused file leaves the active table");
    bool threw = false;
    try {
        (void)ninferno::ops::load_route_table(write_file("malformed", "{\"schema\": "));
    } catch (const std::runtime_error&) { threw = true; }
    failures += check(threw, "malformed JSON is refused");
    threw = false;
    try {
        (void)ninferno::ops::inspect_route_table(write_file("schema", R"({"schema": "other"})"));
    } catch (const std::runtime_error&) { threw = true; }
    failures += check(threw, "another schema is refused by inspect as well");

    // The generic MoE Op's entries: a plan per band and a prefill plan per exact layer; a prefill
    // plan with K split across warps is refused.
    {
        namespace moe = ninferno::ops::detail::moe;
        const auto moe_table = [&](const std::string& moe_entries) {
            return "{\"schema\": \"ninferno-route-table/1\", \"gpu\": {\"class\": \"" + gpu +
                   "\"}, \"entries\": [], \"moe\": [" + moe_entries + "]}";
        };
        const std::string entry =
            R"({"format": "mxfp4", "experts": 32, "top_k": 4, "hidden": 2880, "inter": 2880,
                "bands": [[1, 1, "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4"],
                          [2, 16, "rmma.m1.n1.w4x1|rmma.m1.n4.w4x1"]],
                "prefill": "rmma.m1.n8.w4x1|rmma.m2.n8.w4x1"})";
        const auto s = ninferno::ops::load_route_table(write_file("moe", moe_table(entry)));
        failures += check(s.loaded && s.entries == 1, "a moe entry loads: " + s.message);
        const moe::MoeRouteKey key{
            ninferno::QType::MXFP4, ninferno::QType::MXFP4, 32, 4, 2880, 2880};
        const auto name = [](const std::optional<moe::MoePlan>& p) {
            return p ? moe::plan_name(*p) + " job " + std::to_string(p->job_columns)
                     : std::string("none");
        };
        failures += check(name(moe::table_plan(key, 1, false)) ==
                              "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4 job 8",
                          "moe T=1 takes the first band");
        failures += check(name(moe::table_plan(key, 9, false)) ==
                              "rmma.m1.n1.w4x1|rmma.m1.n4.w4x1 job 32",
                          "moe T=9 takes the second band; the job width is the wider schedule's");
        failures += check(name(moe::table_plan(key, 17, true)) ==
                              "rmma.m1.n8.w4x1|rmma.m2.n8.w4x1 job 64",
                          "moe T=17 under the prefill profile takes the prefill plan");
        failures += check(!moe::table_plan(key, 40, false), "moe T=40 outside every band: none");
        failures += check(!moe::table_plan({ninferno::QType::MXFP4, ninferno::QType::MXFP4, 32, 4,
                                            2880, 2816},
                                           1, false),
                          "another layer: none");
        // A mixed layer (a GGUF mixture: q5_1 gate/up over q5_0 down) has its own key, and
        // neither same-format layer of its geometry takes its bands.
        {
            const auto mixed_status = ninferno::ops::load_route_table(write_file(
                "moe_mixed",
                moe_table(R"({"format": "q51", "down_format": "q50", "experts": 128,
                    "top_k": 6, "hidden": 2688, "inter": 1856,
                    "bands": [[1, 16, "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4"]]})")));
            failures += check(mixed_status.loaded && mixed_status.entries == 1,
                              "a mixed moe entry loads: " + mixed_status.message);
            const moe::MoeRouteKey mixed{
                ninferno::QType::GGML_Q5_1, ninferno::QType::GGML_Q5_0, 128, 6, 2688, 1856};
            auto same_up       = mixed;
            same_up.down_qtype = ninferno::QType::GGML_Q5_1;
            auto same_down     = mixed;
            same_down.qtype    = ninferno::QType::GGML_Q5_0;
            failures += check(name(moe::table_plan(mixed, 4, false)) ==
                                  "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4 job 8",
                              "the mixed layer takes its band");
            failures += check(!moe::table_plan(same_up, 4, false) &&
                                  !moe::table_plan(same_down, 4, false),
                              "a same-format layer of that geometry: none");
            (void)ninferno::ops::load_route_table(write_file("moe", moe_table(entry)));
        }
        bool refused_moe = false;
        try {
            (void)ninferno::ops::load_route_table(write_file(
                "moe_bad", moe_table(R"({"format": "mxfp4", "experts": 32, "top_k": 4,
                    "hidden": 2880, "inter": 2880, "prefill": "rmma.m1.n1.w1x8|rmma.m1.n8.w4x1"})")));
        } catch (const std::runtime_error&) { refused_moe = true; }
        failures += check(refused_moe, "a moe prefill plan that splits K is refused");
        // R8 bands: only on a format the R8 kernel reads and only up to kR8Tokens = 8 (its widest
        // instance; a device or bank of a narrower r8_token_limit takes the register MMA for the
        // band's wider calls, r8_stages_admitted).
        const auto r8_entry = [](const char* format, int t_max) {
            return std::string(R"({"format": ")") + format +
                   R"(", "experts": 32, "top_k": 4, "hidden": 2816, "inter": 512, "bands": [[1, )" +
                   std::to_string(t_max) + R"(, "r8.w4r1|rmma.m1.n1.w1x4"]]})";
        };
        const auto loads = [&](const std::string& name, const std::string& entry_json) {
            try {
                const auto path = write_file(name, moe_table(entry_json));
                return ninferno::ops::load_route_table(path).loaded;
            } catch (const std::runtime_error&) { return false; }
        };
        failures +=
            check(loads("moe_r8_ok", r8_entry("iq2s", 4)), "an iq2s R8 band to T = 4 loads");
        failures += check(loads("moe_r8_eight", r8_entry("iq2s", 8)),
                          "an iq2s R8 band to T = 8 loads");
        failures += check(!loads("moe_r8_wide", r8_entry("iq2s", 9)),
                          "an R8 band beyond T = 8 is refused");
        failures += check(!loads("moe_r8_fp8", r8_entry("fp8", 4)),
                          "an R8 band on a format without the R8 kernel is refused");
        (void)ninferno::ops::load_route_table(write_file("good", good));
    }

    // The board a table was measured on is named by its PCI id (gpu.pci_id) and read from the
    // registry (#612 step 9): the writer records the id, and the facts only for a board the
    // registry lacks; the reader takes the registry's row over any facts the file spells, and a
    // file's facts where the registry has no row for its id.
    {
        const std::string entry =
            R"({"format": "q8", "compute": "a16", "epilogue": "store", "n": 4096, "k": 1024,
                "bands": [[1, 16777216, "rmma.m1.n1.w1x8"]]})";
        const auto boarded = [&](const std::string& fields) {
            return "{\"schema\": \"ninferno-route-table/1\", \"gpu\": {\"class\": \"" + gpu +
                   "\", \"name\": \"test\"" + fields + "}, \"entries\": [" + entry + "]}";
        };
        const auto distance = [&](const std::string& name, const std::string& fields) {
            (void)ninferno::ops::load_route_table(write_file(name, boarded(fields)));
            const auto answer = generic::table_route_answer(q8, 1, false);
            return answer ? answer->distance : -1.0;
        };
        const auto loads_board = [&](const std::string& name, const std::string& fields) {
            try {
                (void)ninferno::ops::load_route_table(write_file(name, boarded(fields)));
            } catch (const std::runtime_error&) { return false; }
            return true;
        };
        char hex[11];
        std::snprintf(hex, sizeof(hex), "0x%04x", static_cast<unsigned>(device.pci_device_id));
        const std::string id      = hex;
        const auto own            = ninferno::measured::find_board_of_pci(device.pci_device_id);
        const std::string written = generic::write_route_table(gpu, device.name, {});
        const std::string cc      = std::to_string(device.compute_capability());
        const std::string figures =
            ", \"board\": {\"cc\": " + cc + ", \"sms\": " + std::to_string(device.sms) + "}";
        const double figures_apart = distance("pci_figures_only", figures);
        failures += check(figures_apart >= 0, "a table with facts and no id reads them");
        // The id alone where the device is its id's registry row; the facts and no id otherwise.
        const bool row_board = own && ninferno::measured::row_board_facts(
                                          ninferno::board_of(device),
                                          ninferno::board_registry::board(device.pci_device_id));
        failures +=
            check((written.find("\"pci_id\": \"" + id + "\"") != std::string::npos) == row_board &&
                      (written.find("\"board\"") != std::string::npos) == !row_board,
                  "the writer records the id of a row board, else its facts: " + written);
        if (own) {
            const double apart = ninferno::board_distance(ninferno::board_of(device), *own);
            failures += check(distance("pci_own", ", \"pci_id\": \"" + id + "\"") == apart,
                              "a table's id reads the registry's row of it");
            failures += check(distance("pci_over_figures", ", \"pci_id\": \"" + id +
                                                               "\", \"board\": {\"cc\": " + cc +
                                                               ", \"sms\": 1}") == apart,
                              "the registry's row of the id wins over the file's facts");
        }
        failures +=
            check(!ninferno::measured::find_board_of_pci(0xFFFF), "0xffff is no registry id");
        failures +=
            check(distance("pci_unknown", ", \"pci_id\": \"0xffff\"" + figures) == figures_apart,
                  "an id the registry lacks reads the file's facts");
        int bad_files = 0;
        for (const char* bad : {"\"2782\"", "\"0x27g2\"", "\"0x278200\"", "\"\"", "10114"}) {
            failures += check(!loads_board("pci_bad_" + std::to_string(++bad_files),
                                           std::string(", \"pci_id\": ") + bad),
                              std::string("a malformed gpu.pci_id is refused: ") + bad);
        }
        (void)ninferno::ops::load_route_table(write_file("good", good));
    }

    // Every shipped class file passes the loader's validation (the default lookup would otherwise
    // drop it silently).
    ninferno::ops::set_route_table_dir("");
    const std::string shipped = ninferno::ops::route_table_dir();
    int files = 0;
    if (!shipped.empty() && std::filesystem::is_directory(ninferno::path_from_utf8(shipped))) {
        for (const auto& f :
             std::filesystem::directory_iterator(ninferno::path_from_utf8(shipped))) {
            if (f.path().extension() != ".json") continue;
            ++files;
            try {
                const auto s = ninferno::ops::inspect_route_table(utf8_path(f.path()));
                std::cout << f.path().filename().string() << ": " << s.gpu_class << ", "
                          << s.message << '\n';
                failures += check(f.path().stem().string() ==
                                      [k = s.gpu_class]() mutable {
                                          for (char& c : k) {
                                              if (c == ':') c = '-';
                                          }
                                          return k;
                                      }(),
                                  f.path().string() + " is named after its class key");
            } catch (const std::exception& e) {
                // Summation order is judged with this device's kernels (a one-tile register MMA
                // sums lane by lane without tensor mma.sync): another class's table, which this
                // device never loads, is checked on its own GPU.
                const std::string why = e.what();
                const std::string own = ninferno::ops::route_gpu_class();
                const std::string stem = f.path().stem().string();
                const bool foreign = own.substr(0, own.find(':')) != stem.substr(0, stem.find('-'));
                if (foreign && (why.find("token-invariant") != std::string::npos ||
                                why.find("sums in another order") != std::string::npos)) {
                    std::cout << f.path().filename().string() << ": not judged on " << own
                              << " (summation order is the device's): " << why << '\n';
                    continue;
                }
                failures += check(false, std::string("shipped table refused: ") + why);
            }
        }
    }
    std::cout << files << " shipped route tables in " << shipped << '\n';

    // Every shipped dense-tune entry names this build's schedule sets. The tuner looks an entry up
    // by its whole key (generic_tune.cpp, problem_key), so an entry keyed by another set matches
    // nothing and its measurements are lost without a word, as when the staged set grew from 9
    // to 13: a change of a set re-keys the entries it leaves valid or retunes them.
    if (!shipped.empty()) {
        const std::string signature = "|s" + std::to_string(generic::rmma_schedule_count()) + "." +
                                      std::to_string(generic::staged_schedule_count()) + "." +
                                      std::to_string(generic::nvfp4_tiled_schedule_count()) + "." +
                                      std::to_string(generic::r8_schedule_count()) + "." +
                                      std::to_string(generic::w4a8_schedule_count());
        std::ifstream in(ninferno::path_from_utf8(shipped) / "tuner" / "dense_tune.json");
        std::stringstream text;
        text << in.rdbuf();
        const std::string content = text.str();
        const std::regex key_pattern(R"re("([a-z0-9]+:[0-9]+sm\|[^"]+)")re");
        int keys = 0, foreign = 0;
        for (auto it = std::sregex_iterator(content.begin(), content.end(), key_pattern);
             it != std::sregex_iterator(); ++it) {
            const std::string key = (*it)[1].str();
            ++keys;
            const std::size_t at  = key.find(signature);
            const std::size_t end = at + signature.size();
            if (at == std::string::npos || (end != key.size() && key[end] != '|')) {
                if (foreign++ < 3) {
                    std::cerr << "dense tune key of other schedule sets: " << key << '\n';
                }
            }
        }
        failures += check(keys > 0 && foreign == 0,
                          "every shipped dense-tune key names this build's schedule sets (" +
                              signature.substr(1) + "): " + std::to_string(foreign) + " of " +
                              std::to_string(keys) + " do not");
    }

    // fp8_e4m3fn_row_f32 runs fp8's register MMA: without an entry of its own it takes the fp8
    // (row BF16) A16 entry of its shape; an entry of its own wins.
    {
        const std::string fp8 = table(
            gpu, R"({"format": "fp8", "compute": "a16", "n": 4096, "k": 12288,
                     "bands": [[1, 16777216, "rmma.m1.n1.w1x4"]], "prefill": "rmma.m2.n8.w4x1"},
                    {"format": "fp8", "compute": "a16", "n": 4096, "k": 4096,
                     "bands": [[1, 16777216, "rmma.m1.n1.w1x4"]]},
                    {"format": "fp8r32", "compute": "a16", "n": 4096, "k": 4096,
                     "bands": [[1, 16777216, "rmma.m1.n1.w2x2"]]})");
        const auto loaded = ninferno::ops::load_route_table(write_file("fp8", fp8));
        failures += check(loaded.loaded && loaded.generic_entries == 3,
                          "the fp8 table loads: " + loaded.message);
        const auto f32 = [](std::int32_t k, ninferno::ops::LinearPolicy policy) {
            return generic::Problem{ninferno::QType::FP8_E4M3FN_ROW_F32, policy, 4096, k,
                                    generic::Epilogue::Store};
        };
        const auto a16 = ninferno::ops::LinearPolicy::A16Only;
        failures += check(route_or_none(generic::table_route(f32(12288, a16), 1, false)) ==
                                  "rmma.m1.n1.w1x4" &&
                              route_or_none(generic::table_route(f32(12288, a16), 512, true)) ==
                                  "rmma.m2.n8.w4x1",
                          "row_f32 takes the fp8 entry of its shape");
        failures += check(route_or_none(generic::table_route(
                              f32(12288, ninferno::ops::LinearPolicy::AllowA8), 1, false)) ==
                              "rmma.m1.n1.w1x4",
                          "row_f32 stays A16 under AllowA8 and takes the fp8 A16 entry");
        failures += check(route_or_none(generic::table_route(f32(4096, a16), 1, false)) ==
                              "rmma.m1.n1.w2x2",
                          "an fp8r32 entry of its own wins");
        failures += check(!generic::table_route(f32(2048, a16), 1, false),
                          "row_f32 of a shape fp8 lacks has no route");
    }

    // "claim": a generic entry of a shape a shape-owned Linear table holds takes the call only at
    // the widths it covers (bands, or its prefill route under the prefill profile); an entry
    // without it never does; a claim on a fused epilogue is refused.
    {
        const std::string claims = table(
            gpu, R"({"format": "fp8", "compute": "a16", "n": 14336, "k": 5120, "claim": true,
                     "bands": [[64, 16777216, "rmma.m1.n4.w4x1"]], "prefill": "rmma.m1.n8.w4x1"},
                    {"format": "fp8", "compute": "a16", "n": 16384, "k": 5120,
                     "bands": [[1, 16777216, "rmma.m1.n1.w1x8"]]})");
        const auto loaded = ninferno::ops::load_route_table(write_file("claims", claims));
        failures += check(loaded.loaded && loaded.generic_entries == 2,
                          "the claim table loads: " + loaded.message);
        const auto fp8 = [](std::int32_t n, ninferno::ops::LinearPolicy policy) {
            return generic::Problem{ninferno::QType::FP8_E4M3FN_ROW_BF16, policy, n, 5120,
                                    generic::Epilogue::Store};
        };
        const auto a16 = ninferno::ops::LinearPolicy::A16Only;
        failures += check(!generic::table_claims(fp8(14336, a16), 16, false) &&
                              generic::table_claims(fp8(14336, a16), 64, false) &&
                              generic::table_claims(fp8(14336, a16), 512, false) &&
                              generic::table_claims(fp8(14336, a16), 17, true),
                          "a claiming entry takes the widths it covers");
        failures += check(!generic::table_claims(fp8(14336, ninferno::ops::LinearPolicy::AllowA8),
                                                 512, false),
                          "a claim is A16Only: a policy that allows quantized activations keeps "
                          "the shape's own table (its A8 kernels)");
        failures += check(!generic::table_claims(fp8(16384, a16), 1, false),
                          "an entry without claim does not take an owned shape");
        failures += check(refused(gpu, "claim-swiglu",
                                  R"({"format": "fp8", "compute": "a16", "epilogue": "swiglu",
                                      "n": 14336, "k": 5120, "claim": true,
                                      "bands": [[1, 16777216, "rmma.m1.n1.w1x8"]]})"),
                          "a claim on a fused epilogue is refused");
    }

    // A binary's shipped routes: <bin>/routes, else <bin>/../share/ninferno/routes, each only with
    // a table in it.
    {
        const auto pkg    = scratch_dir() / "pkg";
        const auto exe    = utf8_path(pkg / "bin" / "ninferno-serve");
        const auto beside = pkg / "bin" / "routes", share = pkg / "share" / "ninferno" / "routes";
        std::filesystem::create_directories(beside);
        std::filesystem::create_directories(share);
        failures += check(ninferno::ops::route_dir_beside(exe).empty(),
                          "routes beside the binary: none without a table");
        std::ofstream(share / "ad102.json") << "{}";
        failures += check(ninferno::ops::route_dir_beside(exe) == utf8_path(share),
                          "routes beside the binary: share/ninferno/routes");
        std::ofstream(beside / "ad102.json") << "{}";
        failures += check(ninferno::ops::route_dir_beside(exe) == utf8_path(beside),
                          "routes beside the binary: bin/routes first");
    }

    ninferno::ops::clear_route_table();
    failures +=
        check(!ninferno::ops::route_table_status().loaded, "clear_route_table drops the table");
    failures += check(!generic::table_route(q8, 1, false) && detail::route_tokens(a4, 48) == 48,
                      "no table: no tuned route, no override");

    std::cout << (failures == 0 ? "OK" : "FAIL") << " route_table\n";
    return failures == 0 ? 0 : 1;
}
