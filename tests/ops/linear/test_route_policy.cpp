// The route policy (#614 F1, --route-policy; ops/generic/route_authorities.h): the precedence the
// selector applies over evidence. Under legacy, the release's, the first authority that answers
// decides; under evidence a Borrowed or Transferred answer of the auto list (a route-table entry
// measured on another board) yields to the ranker.
//
// Asked on the nine boards of the A5 regret sweep (tools/bench/data/route_regret/, the boards by
// the PCI id of the lab's card, their gpu_facts.json row substituted as the selector goldens
// substitute a board), over that sweep's shapes (shapes.txt) and widths (route_tune's decode and
// band points, 1..512) under A16Only and AllowA8, and at the prefill chunk (1024) under the
// prefill profile:
//   - legacy: every answer carries the legacy policy and no demotion, and its text reads as before
//     the policy existed (no policy= or demoted= field);
//   - a Borrowed or Transferred answer of the auto list (the route table) yields under evidence:
//     an authority past it answers, its evidence is neither, and it names the demoted answer;
//     under legacy that entry wins;
//   - an answer of any other evidence (pin, measured, carried, ranker), and a Transferred answer
//     of the rules list (mmq-prefill-gate, which already ranks below the cost model), is the same
//     answer under evidence: measured-on-this-board is never demoted;
//   - the grid holds borrowed, transferred and measured answers (each check is exercised).
// Each board is asked twice: under the shipped tables, and under them without its own table, so
// that its nearest table is another board's (a user's card with no table: Carried or
// Transferred), as `--route-dir` to such a directory would run it. And a claiming entry
// (table_claims) of a Transferred nearest table claims under legacy and not under evidence, while
// a board's own claim is kept (check_transferred_claim).
// The routes under legacy are the goldens' (ninferno_selector_golden_test pins the legacy policy).
//
// `--write FILE` also writes the cells whose route moves, per board (the A/B list of F1, read on a
// CPU before any GPU run): tools/bench/data/route_regret/policy_evidence_moves.json. A board's
// answers depend on the routes its build compiles, so each board is written from its own build set
// (RegretBoard::own_build_set: the set its release engine is built as); a run writes the boards
// of its build set
// and keeps the others the file already holds. `--board NAME` asks one board, `--verbose` prints
// every cell's two answers.
#include "core/device_class.h"
#include "core/machine_facts.h"
#include "ninferno/option_enums.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_authorities.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"
#include "ninferno/ops/route_table.h"
#include "ops/route_pins.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;
using Kind = generic::Evidence::Kind;
using Json = nlohmann::ordered_json;

int g_failed   = 0;
bool g_verbose = false;

void fail(const std::string& what) {
    ++g_failed;
    if (g_failed <= 40) std::cerr << "FAIL: " << what << '\n';
}

// The build set this binary compiles (as ninferno_selector_golden_test names its directories).
constexpr const char* kBuildSet =
#if NINFERNO_BUILD_ARCH == 70
    "sm70";
#elif NINFERNO_BUILD_ARCH == 75
    "sm75";
#elif defined(NINFERNO_WITHOUT_NVFP4)
    "base";
#else
    "full";
#endif

// The regret sweep's boards (docs/maintainer/route-regret.md), by the PCI id of the lab's card,
// and the build set of the engine each runs (sm_120a: full; sm_75, sm_70: their own; the rest:
// base).
struct RegretBoard {
    const char* name; // the summary's file name in tools/bench/data/route_regret/
    const char* card;
    std::uint32_t pci_device_id;
    const char* own_build_set;
};

constexpr RegretBoard kRegretBoards[] = {
    {"rtx5090", "RTX 5090", 0x2B85, "full"},      {"rtx5070ti", "RTX 5070 Ti", 0x2C05, "full"},
    {"rtx4090", "RTX 4090", 0x2684, "base"},      {"rtx4070ti", "RTX 4070 Ti", 0x2782, "base"},
    {"rtx4060ti", "RTX 4060 Ti", 0x2805, "base"}, {"rtx3090", "RTX 3090", 0x2204, "base"},
    {"rtx3060", "RTX 3060", 0x2504, "base"},      {"rtx2080ti", "RTX 2080 Ti", 0x1E07, "sm75"},
    {"gv100", "Quadro GV100", 0x1DBA, "sm70"},
};

// The board as the golden boards are built: its registry row (ops/generic/selector_facts.h
// registry_facts_board), the L2 to the byte (2.25 MiB on the RTX 3060, 5.5 on the RTX 2080 Ti), so
// the board is its route table's own, at distance 0, as the card itself reports it.
DeviceClass board_class(const RegretBoard& b) {
    if (board_registry::find(b.pci_device_id) == nullptr) {
        throw std::runtime_error(std::string(b.name) + ": no gpu_facts row");
    }
    return generic::board_device_class(generic::registry_facts_board(b.name, b.pci_device_id));
}

struct SweepShape {
    QType qtype;
    generic::Epilogue epilogue;
    std::int32_t n, k;
    std::string line; // as shapes.txt spells it
};

std::vector<SweepShape> read_shapes() {
    const std::string path =
        std::string(NINFERNO_SOURCE_DIR) + "/tools/bench/data/route_regret/shapes.txt";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<SweepShape> out;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string format, compute, epilogue;
        SweepShape s{};
        fields >> format >> compute >> epilogue >> s.n >> s.k;
        if (!fields) throw std::runtime_error("shapes.txt: cannot read '" + line + "'");
        if (format == "q4k") {
            s.qtype = QType::GGML_Q4_K;
        } else if (format == "q6k") {
            s.qtype = QType::GGML_Q6_K;
        } else {
            throw std::runtime_error("shapes.txt: format " + format + " is not in this test");
        }
        if (epilogue == "store") {
            s.epilogue = generic::Epilogue::Store;
        } else if (epilogue == "swiglu") {
            s.epilogue = generic::Epilogue::SwiGlu;
        } else if (epilogue == "add") {
            s.epilogue = generic::Epilogue::AddResidual;
        } else {
            throw std::runtime_error("shapes.txt: epilogue " + epilogue);
        }
        s.line = line;
        out.push_back(s);
    }
    if (out.empty()) throw std::runtime_error(path + " holds no shape");
    return out;
}

// route_tune's decode and band points (route_tune_tool.cpp decode_points, wide_points).
constexpr std::int32_t kSpeedPoints[] = {1,  2,  3,  4,  5,  6,   7,   8,   12, 16,
                                         24, 32, 48, 64, 96, 128, 192, 256, 512};
constexpr std::int32_t kPrefillChunk  = 1024;

// The regime of selection_regret.py's report (docs/maintainer/route-regret.md, "By regime and
// authority"): decode T <= 4, verify T 5..16, T >= 17; the prefill profile's chunk apart.
const char* regime_of(std::int32_t t, bool prefill) {
    return prefill ? "prefill" : t <= 4 ? "decode T<=4" : t <= 16 ? "verify T5-16" : "T>=17";
}

struct Cell {
    const SweepShape* shape;
    LinearPolicy policy;
    std::int32_t tokens;
    bool prefill;
};

std::optional<generic::RouteAnswer> ask(const DeviceClass& d, const Cell& c, RoutePolicy policy) {
    ops::set_route_policy(policy);
    return generic::route_answer_query(d, c.shape->qtype, c.policy, c.shape->epilogue, c.shape->n,
                                       c.shape->k, c.tokens, c.prefill);
}

const char* policy_word(LinearPolicy p) { return p == LinearPolicy::AllowA8 ? "a8" : "a16"; }

std::string where(const RegretBoard& b, const Cell& c) {
    return std::string(b.name) + " " + c.shape->line + " " + policy_word(c.policy) +
           " T=" + std::to_string(c.tokens) + (c.prefill ? " prefill" : "");
}

Json answer_json(const generic::RouteAnswer& a) {
    Json out = {{"route", generic::route_name(a.route)},
                {"authority", std::string(a.evidence.source)},
                {"evidence", generic::evidence_kind_name(a.evidence.kind)}};
    if (!a.evidence.donor.empty()) out["donor"] = a.evidence.donor;
    return out;
}

// Whether `source` names an authority of the auto list, the one the policy applies to.
bool in_auto_list(std::string_view source) {
    for (const generic::NamedAuthority& a : generic::auto_authorities()) {
        if (a.name == source) return true;
    }
    return false;
}

struct BoardTally {
    int cells = 0, borrowed = 0, transferred = 0, measured = 0, moved = 0, demoted_same_route = 0;
    int kept_rules = 0; // Borrowed or Transferred answers of the rules list, kept by design
    std::map<std::string, int> moved_by_regime;
    Json moves = Json::array();
    std::set<std::string> moved_problems; // "<format>|<epilogue>|n<N>|k<K>"
};

// The grid is asked without the dense tuner: its entries are a device's own timings, which a
// substituted board never reads (tuned_dense_route), while the engine and route-tune read the
// release's routes/tuner/dense_tune.json. Under evidence the dense tuner's entry is what a demoted
// route-table entry yields to first, so a release entry for a moved problem of a board would
// answer there instead of the ranker, and the moves file would not be the A/B's list. Fails when
// the release file gains an entry of the board's class (its device id, the file's key) for a moved
// problem: then the moves must be asked with that entry.
std::vector<std::string> dense_tune_hits(const DeviceClass& d, const std::set<std::string>& moved) {
    const auto path = std::filesystem::path(ops::route_table_dir()) / "tuner" / "dense_tune.json";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("no release dense tuner file " + path.string());
    const Json file          = Json::parse(in);
    const std::string prefix = d.id() + "|";
    std::vector<std::string> hits;
    for (const auto& [key, entry] : file.at("problems").items()) {
        if (!key.starts_with(prefix)) continue;
        // "<class>|<format>|<compute>|<epilogue>|n<N>|k<K>|<schedule set>..."
        std::vector<std::string> fields;
        for (std::size_t at = 0; at <= key.size();) {
            const std::size_t end = std::min(key.find('|', at), key.size());
            fields.push_back(key.substr(at, end - at));
            at = end + 1;
        }
        if (fields.size() < 6) continue;
        const std::string problem = fields[1] + "|" + fields[3] + "|" + fields[4] + "|" + fields[5];
        if (moved.contains(problem)) hits.push_back(key);
    }
    return hits;
}

void check_dense_tune_free(const DeviceClass& d, const RegretBoard& b, const char* tables,
                           const std::set<std::string>& moved) {
    for (const std::string& key : dense_tune_hits(d, moved)) {
        fail(std::string(b.name) + " " + tables + ": the release dense tuner holds " + key +
             ", a moved problem: ask the moves with the dense tuner's entry");
    }
}

// The check sees an entry where one is: the RTX 2080 Ti's release entries hold the q4_K 4096 x 4096
// store problem (a board with no move, so check_dense_tune_free never meets it).
void check_dense_tune_reader() {
    for (const RegretBoard& b : kRegretBoards) {
        if (std::string_view(b.name) != "rtx2080ti") continue;
        if (dense_tune_hits(board_class(b), {"q4k|store|n4096|k4096"}).empty()) {
            fail("the dense tuner check finds no RTX 2080 Ti q4_K 4096x4096 entry: it reads "
                 "nothing");
        }
    }
}

// The cost model's price of `route` for the cell (priced_routes_query: the selector's own formula,
// at any width; below 17 tokens the formula's figure, which the selector does not rank by).
std::optional<double> model_seconds(const DeviceClass& d, const Cell& c, generic::Route route) {
    for (const auto& p : generic::priced_routes_query(
             d, c.shape->qtype, c.policy, c.shape->epilogue, c.shape->n, c.shape->k, c.tokens)) {
        if (p.route == route && p.seconds > 0) return p.seconds;
    }
    return std::nullopt;
}

void check_cell(const DeviceClass& d, const Cell& c, const std::string& at,
                const generic::RouteAnswer& legacy, const generic::RouteAnswer& evidence,
                BoardTally& tally) {
    const generic::Evidence& le     = legacy.evidence;
    const generic::Evidence& ee     = evidence.evidence;
    const std::string legacy_text   = generic::evidence_text(le);
    const std::string evidence_text = generic::evidence_text(ee);
    if (g_verbose) {
        std::cout << at << ": " << generic::route_name(legacy.route) << ' ' << legacy_text << " | "
                  << generic::route_name(evidence.route) << ' ' << evidence_text << '\n';
    }
    // Legacy: the policy recorded, nothing demoted, the text as before F1.
    if (le.policy != RoutePolicy::Legacy || !le.demoted_source.empty() ||
        legacy_text.find("policy=") != std::string::npos ||
        legacy_text.find("demoted=") != std::string::npos) {
        fail(at + ": legacy answer reads '" + legacy_text + "'");
    }
    if (ee.policy != RoutePolicy::Evidence ||
        evidence_text.find(" policy=evidence") == std::string::npos) {
        fail(at + ": evidence answer does not name its policy: '" + evidence_text + "'");
    }
    if (le.kind == Kind::Measured) ++tally.measured;
    const bool auto_list = in_auto_list(le.source);
    if (le.kind == Kind::Borrowed) ++(auto_list ? tally.borrowed : tally.kept_rules);
    if (le.kind == Kind::Transferred) ++(auto_list ? tally.transferred : tally.kept_rules);
    if (auto_list && generic::policy_demotes(RoutePolicy::Evidence, le.kind)) {
        // Under legacy the entry won; under evidence it yields to the ranker. It would stand only
        // where the ranker has no candidate (the scalar fallback), which no cell of this grid is.
        if (ee.kind == Kind::Borrowed || ee.kind == Kind::Transferred || in_auto_list(ee.source)) {
            fail(at + ": the " + generic::evidence_kind_name(le.kind) + " entry did not yield: '" +
                 evidence_text + "'");
        }
        if (ee.demoted_source != le.source || ee.demoted_kind != le.kind) {
            fail(at + ": the evidence answer does not name the demoted one: '" + evidence_text +
                 "'");
        }
        const std::string demoted = std::string(" demoted=") + std::string(le.source) + ":" +
                                    generic::evidence_kind_name(le.kind);
        if (evidence_text.find(demoted) == std::string::npos) {
            fail(at + ": evidence text '" + evidence_text + "' lacks" + demoted);
        }
    } else if (!(evidence.route == legacy.route) || ee.source != le.source || ee.kind != le.kind ||
               ee.donor != le.donor || !ee.demoted_source.empty()) {
        // Pin, Measured, Carried and Ranker answers are never demoted, nor a Transferred answer of
        // the rules list (mmq-prefill-gate: below the cost model already, answering where it
        // abstains).
        fail(at + ": a " + generic::evidence_kind_name(le.kind) + " answer moved: '" + legacy_text +
             "' -> '" + evidence_text + "'");
    }
    if (evidence.route == legacy.route) {
        if (!ee.demoted_source.empty()) ++tally.demoted_same_route;
        return;
    }
    ++tally.moved;
    const char* regime = regime_of(c.tokens, c.prefill);
    ++tally.moved_by_regime[regime];
    tally.moved_problems.insert(std::string(generic::qtype_name(c.shape->qtype)) + "|" +
                                generic::epilogue_name(c.shape->epilogue) + "|n" +
                                std::to_string(c.shape->n) + "|k" + std::to_string(c.shape->k));
    // The prediction before any run: the cost model's price of the evidence route over the legacy
    // route's (null where it prices either not), and the A5 sweep's measured ratio, which
    // tools/bench/policy_moves_predict.py fills from that sweep's routes CSVs on the stand.
    const auto model_legacy = model_seconds(d, c, legacy.route);
    const auto model_new    = model_seconds(d, c, evidence.route);
    Json model_ratio        = nullptr;
    if (model_legacy && model_new) model_ratio = *model_new / *model_legacy;
    tally.moves.push_back(Json{{"format", generic::qtype_name(c.shape->qtype)},
                               {"policy", policy_word(c.policy)},
                               {"epilogue", generic::epilogue_name(c.shape->epilogue)},
                               {"n", c.shape->n},
                               {"k", c.shape->k},
                               {"T", c.tokens},
                               {"profile", c.prefill ? "prefill" : "speed"},
                               {"regime", regime},
                               {"legacy", answer_json(legacy)},
                               {"evidence", answer_json(evidence)},
                               {"model_ratio", model_ratio},
                               {"a5_ratio", nullptr}});
}

BoardTally run_board(const RegretBoard& b, const std::vector<SweepShape>& shapes) {
    const DeviceClass d = board_class(b);
    BoardTally tally;
    std::vector<Cell> cells;
    for (const SweepShape& s : shapes) {
        for (const LinearPolicy p : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
            for (const std::int32_t t : kSpeedPoints) cells.push_back({&s, p, t, false});
            cells.push_back({&s, p, kPrefillChunk, true});
        }
    }
    for (const Cell& c : cells) {
        const std::string at = where(b, c);
        std::optional<generic::RouteAnswer> legacy, evidence;
        try {
            legacy   = ask(d, c, RoutePolicy::Legacy);
            evidence = ask(d, c, RoutePolicy::Evidence);
        } catch (const std::exception& error) {
            ops::set_route_policy(RoutePolicy::Legacy);
            fail(at + ": threw " + error.what());
            continue;
        }
        ops::set_route_policy(RoutePolicy::Legacy);
        if (legacy.has_value() != evidence.has_value()) {
            fail(at + ": one policy routes the call, the other does not");
            continue;
        }
        if (!legacy) continue;
        ++tally.cells;
        check_cell(d, c, at, *legacy, *evidence, tally);
    }
    return tally;
}

// The two table sets a board is asked under: the shipped tables (its own among them), and the
// same without its own table, so that its nearest table is another board's (a user's card with no
// table of its own: Carried within kCarryDistance, Transferred past it).
constexpr const char* kOwnTables  = "own";
constexpr const char* kWithoutOwn = "without_own";

// The shipped table measured on `b` (gpu.pci_id), if any, in `dir`.
std::optional<std::filesystem::path> own_table(const std::filesystem::path& dir,
                                               const RegretBoard& b) {
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;
        std::ifstream in(entry.path());
        const Json table = Json::parse(in, nullptr, false);
        if (table.is_discarded() || !table.contains("gpu")) continue;
        const std::string id = table["gpu"].value("pci_id", std::string());
        if (!id.empty() && std::stoul(id, nullptr, 16) == b.pci_device_id) return entry.path();
    }
    return std::nullopt;
}

// A directory of the shipped tables but `skip`, for the default lookup (set_route_table_dir).
std::filesystem::path tables_without(const std::filesystem::path& dir,
                                     const std::filesystem::path& skip, const RegretBoard& b) {
    const auto out = std::filesystem::temp_directory_path() /
                     (std::string("ninferno_route_policy_") + b.name + "_" + kBuildSet);
    std::filesystem::remove_all(out);
    std::filesystem::create_directories(out);
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;
        if (std::filesystem::equivalent(entry.path(), skip)) continue;
        std::filesystem::copy_file(entry.path(), out / entry.path().filename());
    }
    return out;
}

// --route-policy evidence and a claiming entry (RouteEntry::claim; table_claims): on a board whose
// nearest table is another board's past kCarryDistance, the claim moves the call off its shape's
// own kernel under legacy and does not under evidence. The board: sm_120, the RTX 5080's 84 SMs and
// 64 MiB L2 on a 64-bit bus, so the RTX 5080's table (gb203, which claims five FP8 shapes) is
// the nearest and Transferred (its bandwidth a quarter of the RTX 5080's).
void check_transferred_claim() {
    const generic::FactsBoard board{"sm120-84sm-64bit", 0, 12, 0, 84, 64, 16, 2617, 14001, 64};
    const DeviceClass d = generic::board_device_class(board);
    const generic::Problem claimed{QType::FP8_E4M3FN_ROW_BF16, LinearPolicy::A16Only, 14336, 5120,
                                   generic::Epilogue::Store};
    for (const std::int32_t t : {6, 64, 512}) {
        const std::string at = "claim fp8 14336x5120 T=" + std::to_string(t);
        ops::set_route_policy(RoutePolicy::Legacy);
        const auto answer = generic::route_answer_query(d, claimed.qtype, claimed.policy,
                                                        claimed.epilogue, claimed.n, claimed.k, t);
        if (!answer || answer->evidence.source != "route-table" ||
            answer->evidence.kind != Kind::Transferred) {
            fail(at + ": the nearest table is not a Transferred one here: " +
                 (answer ? generic::evidence_text(answer->evidence) : std::string("no answer")));
            continue;
        }
        const ScopedDeviceClass scoped(d);
        const bool legacy_claims = generic::table_claims(claimed, t, false);
        ops::set_route_policy(RoutePolicy::Evidence);
        const bool evidence_claims = generic::table_claims(claimed, t, false);
        ops::set_route_policy(RoutePolicy::Legacy);
        if (!legacy_claims) fail(at + ": the Transferred claim does not claim under legacy");
        if (evidence_claims) fail(at + ": the Transferred claim still claims under evidence");
    }
    // The board's own (measured) claims are kept under evidence: the RTX 5080 itself.
    const GpuFactsRow* r5080 = find_gpu_facts(0x2C02);
    if (r5080 == nullptr) {
        fail("claim: no gpu_facts row of the RTX 5080");
        return;
    }
    const generic::FactsBoard own{"rtx5080", 0x2C02, 12, 0, 84, 64, 16, 2617, 15001, 256};
    const ScopedDeviceClass scoped(generic::board_device_class(own));
    ops::set_route_policy(RoutePolicy::Evidence);
    const bool kept = generic::table_claims(claimed, 64, false);
    ops::set_route_policy(RoutePolicy::Legacy);
    if (!kept) fail("claim: the RTX 5080's own claim is refused under evidence");
}

Json board_json(const RegretBoard& b, const char* tables, const BoardTally& t) {
    Json regimes = Json::object();
    for (const auto& [regime, n] : t.moved_by_regime) regimes[regime] = n;
    return Json{{"board", b.name},
                {"card", b.card},
                {"tables", tables},
                {"build_set", kBuildSet},
                {"cells", t.cells},
                {"legacy_borrowed", t.borrowed},
                {"legacy_transferred", t.transferred},
                {"legacy_measured", t.measured},
                {"legacy_transferred_rules_kept", t.kept_rules},
                {"moved", t.moved},
                {"demoted_same_route", t.demoted_same_route},
                {"moved_by_regime", regimes},
                {"moves", t.moves}};
}

// A move's identity across rewrites: the cell and both routes.
std::string move_key(const Json& m) {
    return m.at("format").get<std::string>() + "|" + m.at("policy").get<std::string>() + "|" +
           m.at("epilogue").get<std::string>() + "|" + std::to_string(m.at("n").get<int>()) + "|" +
           std::to_string(m.at("k").get<int>()) + "|" + std::to_string(m.at("T").get<int>()) + "|" +
           m.at("profile").get<std::string>() + "|" +
           m.at("legacy").at("route").get<std::string>() + "|" +
           m.at("evidence").at("route").get<std::string>();
}

// The file with this build set's boards replaced and every other board kept, in kRegretBoards'
// order. A rewritten move whose cell and two routes the file already held keeps its a5_ratio (the
// measured prediction tools/bench/policy_moves_predict.py wrote); a move whose routes changed gets
// none.
void write_moves(const std::string& path, const std::map<std::string, Json>& written) {
    std::map<std::string, Json> boards;
    if (std::filesystem::exists(path)) {
        std::ifstream in(path);
        const Json old = Json::parse(in);
        for (const Json& b : old.at("boards")) {
            boards[b.at("board").get<std::string>() + "/" + b.value("tables", kOwnTables)] = b;
        }
    }
    for (const auto& [name, b] : written) {
        Json fresh = b;
        if (const auto it = boards.find(name); it != boards.end()) {
            std::map<std::string, const Json*> before;
            for (const Json& m : it->second.at("moves")) before[move_key(m)] = &m;
            for (Json& m : fresh["moves"]) {
                const auto was = before.find(move_key(m));
                if (was == before.end()) continue;
                const Json& old = *was->second;
                m["a5_ratio"]   = old.value("a5_ratio", Json());
                // The same price written by another writer (Python's repr, nlohmann's 17 digits):
                // the file keeps its text.
                const Json price = old.value("model_ratio", Json());
                if (price.is_number() && m["model_ratio"].is_number() &&
                    std::abs(price.get<double>() - m["model_ratio"].get<double>()) <=
                        1e-12 * std::abs(m["model_ratio"].get<double>())) {
                    m["model_ratio"] = price;
                }
            }
        }
        boards[name] = std::move(fresh);
    }
    Json speed = Json::array();
    for (const std::int32_t t : kSpeedPoints) speed.push_back(t);
    Json out = {{"schema", "ninferno-route-policy-moves/1"},
                {"what",
                 "the cells of the A5 regret grid whose production pick moves under "
                 "--route-policy evidence against legacy (#614 F1), per board, each from its own "
                 "build set, under the shipped tables (tables: own) and under them without the "
                 "board's own table (tables: without_own, its nearest table another board's); "
                 "written by ninferno_route_policy_test --write. model_ratio: the cost model's "
                 "price of the evidence route over the legacy one; a5_ratio: the A5 sweep's "
                 "measured ratio, filled by tools/bench/policy_moves_predict.py from the stand's "
                 "routes CSVs (null until then)"},
                {"grid",
                 {{"shapes", "tools/bench/data/route_regret/shapes.txt"},
                  {"policies", {"a16", "a8"}},
                  {"speed_T", speed},
                  {"prefill_T", {kPrefillChunk}}}},
                {"boards", Json::array()}};
    for (const RegretBoard& b : kRegretBoards) {
        for (const char* tables : {kOwnTables, kWithoutOwn}) {
            if (const auto it = boards.find(std::string(b.name) + "/" + tables); it != boards.end())
                out["boards"].push_back(it->second);
        }
    }
    std::ofstream file(path, std::ios::binary);
    file << out.dump(1) << '\n';
    if (!file.good()) throw std::runtime_error("cannot write " + path);
    std::cout << "route policy: wrote " << path << " (" << written.size() << " boards of "
              << kBuildSet << ")\n";
}

void check_policy_words() {
    if (ops::route_policy() != RoutePolicy::Legacy) fail("the default route policy");
    for (const Kind k : {Kind::Pin, Kind::Measured, Kind::Ranker, Kind::Carried, Kind::Transferred,
                         Kind::Borrowed}) {
        const bool demoted = k == Kind::Borrowed || k == Kind::Transferred;
        if (generic::policy_demotes(RoutePolicy::Legacy, k)) fail("legacy demotes an answer");
        if (generic::policy_demotes(RoutePolicy::Evidence, k) != demoted) {
            fail(std::string("evidence ") + (demoted ? "keeps" : "demotes") + " a " +
                 generic::evidence_kind_name(k) + " answer");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string write, only;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--write" && i + 1 < argc) write = argv[++i];
            if (arg == "--board" && i + 1 < argc) only = argv[++i];
            if (arg == "--verbose") g_verbose = true;
        }
        check_policy_words();
        check_transferred_claim();
        check_dense_tune_reader();
        const std::vector<SweepShape> shapes = read_shapes();
        const std::filesystem::path shipped  = ops::route_table_dir();
        if (shipped.empty() || !std::filesystem::is_directory(shipped)) {
            throw std::runtime_error("no shipped route-table directory");
        }
        std::map<std::string, Json> written;
        int borrowed = 0, transferred = 0, measured = 0;
        for (const RegretBoard& b : kRegretBoards) {
            if (!only.empty() && only != b.name) continue;
            const auto own = own_table(shipped, b);
            for (const char* tables : {kOwnTables, kWithoutOwn}) {
                const bool without = std::string_view(tables) == kWithoutOwn;
                if (without && !own) continue; // no table of its own: the first pass was this one
                std::optional<std::filesystem::path> dir;
                if (without) dir = tables_without(shipped, *own, b);
                ops::set_route_table_dir(dir ? *dir : std::filesystem::path());
                const BoardTally t = run_board(b, shapes);
                ops::set_route_table_dir({});
                if (dir) std::filesystem::remove_all(*dir);
                check_dense_tune_free(board_class(b), b, tables, t.moved_problems);
                borrowed += t.borrowed;
                transferred += t.transferred;
                measured += t.measured;
                std::cout << b.name << " " << tables << " (" << kBuildSet << "): " << t.cells
                          << " cells; legacy " << t.borrowed << " borrowed, " << t.transferred
                          << " transferred, " << t.measured << " measured, " << t.kept_rules
                          << " rules' transferred (kept); under evidence " << t.moved << " moved";
                for (const auto& [regime, n] : t.moved_by_regime) {
                    std::cout << " [" << regime << ": " << n << ']';
                }
                std::cout << ", " << t.demoted_same_route << " demoted to the same route\n";
                if (std::string_view(b.own_build_set) == kBuildSet) {
                    written[std::string(b.name) + "/" + tables] = board_json(b, tables, t);
                }
            }
        }
        if (only.empty() && borrowed == 0) fail("no borrowed answer in the grid: nothing demoted");
        if (only.empty() && transferred == 0) {
            fail("no transferred answer in the grid, even without the boards' own tables");
        }
        if (only.empty() && measured == 0) {
            fail("no measured answer in the grid: the never-demoted check is idle");
        }
        ops::set_route_policy(RoutePolicy::Legacy);
        if (!write.empty()) write_moves(write, written);
    } catch (const std::exception& error) {
        std::cerr << "route policy: " << error.what() << '\n';
        return 1;
    }
    if (g_failed != 0) {
        std::cerr << "route policy: " << g_failed << " failures\n";
        return 1;
    }
    std::cout << "route policy: ok\n";
    return 0;
}
