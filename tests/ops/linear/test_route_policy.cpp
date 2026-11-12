// The route policy (#614 F1, --route-policy; ops/generic/route_authorities.h): the precedence the
// selector applies over evidence. Under legacy, the release's, the first authority that answers
// decides; under evidence a Borrowed or Transferred answer of the auto list (a route-table entry
// measured on another board) yields to the ranker.
//
// Asked on the nine boards of the A5 regret sweep (tools/bench/data/route_regret/, the boards by
// their PCI id's row of gpu_facts.json, substituted as the selector goldens are), over that sweep's
// shapes (shapes.txt) and widths (route_tune's decode and band points, 1..512) under A16Only and
// AllowA8, and at the prefill chunk (1024) under the prefill profile:
//   - legacy: every answer carries the legacy policy and no demotion, and its text reads as before
//     the policy existed (no policy= or demoted= field);
//   - a Borrowed or Transferred route-table answer of legacy yields under evidence: another
//     authority answers, its evidence is neither, and it names the demoted answer; under legacy
//     that entry wins;
//   - an answer of any other evidence (pin, measured, carried, ranker) is the same answer under
//     evidence: measured-on-this-board is never demoted;
//   - at least one borrowed answer exists in the grid (the demotion is exercised).
// The routes under legacy are the goldens' (ninferno_selector_golden_test pins the legacy policy).
//
// `--write FILE` also writes the cells whose route moves, per board (the A/B list of F1, read on a
// CPU before any GPU run): tools/bench/data/route_regret/policy_evidence_moves.json. The answers
// depend on the build set (the routes a build compiles), which the file names.
#include "core/device_class.h"
#include "core/machine_facts.h"
#include "ninferno/option_enums.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_authorities.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"
#include "ops/route_pins.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
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

int g_failed = 0;

void fail(const std::string& what) {
    ++g_failed;
    if (g_failed <= 40) std::cerr << "FAIL: " << what << '\n';
}

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

// The regret sweep's boards (docs/maintainer/route-regret.md), by the PCI id of the lab's card.
struct RegretBoard {
    const char* name; // the summary's file name in tools/bench/data/route_regret/
    const char* card;
    std::uint32_t pci_device_id;
};

constexpr RegretBoard kRegretBoards[] = {
    {"rtx5090", "RTX 5090", 0x2B85},      {"rtx5070ti", "RTX 5070 Ti", 0x2C05},
    {"rtx4090", "RTX 4090", 0x2684},      {"rtx4070ti", "RTX 4070 Ti", 0x2782},
    {"rtx4060ti", "RTX 4060 Ti", 0x2805}, {"rtx3090", "RTX 3090", 0x2204},
    {"rtx3060", "RTX 3060", 0x2503},      {"rtx2080ti", "RTX 2080 Ti", 0x1E07},
    {"gv100", "Quadro GV100", 0x1DBA},
};

// The board as the CUDA query reports a stock one of its row (memory clock: half the data rate),
// as ninferno_decision_properties_test builds its rows.
DeviceClass board_class(const RegretBoard& b) {
    const GpuFactsRow* r = find_gpu_facts(b.pci_device_id);
    if (r == nullptr) throw std::runtime_error(std::string(b.name) + ": no gpu_facts row");
    const generic::FactsBoard board{
        b.name,
        r->pci_device_id,
        r->major,
        r->minor,
        r->sms,
        static_cast<std::size_t>(std::lround(r->l2_mib)),
        static_cast<std::size_t>(std::max(1L, std::lround(r->vram_gib))),
        r->boost_mhz,
        static_cast<int>(std::lround(r->gbps * 500.0)) + 1,
        r->bus_bits};
    return generic::board_device_class(board);
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
        if (format == "q4k")
            s.qtype = QType::GGML_Q4_K;
        else if (format == "q6k")
            s.qtype = QType::GGML_Q6_K;
        else
            throw std::runtime_error("shapes.txt: format " + format + " is not in this test");
        if (epilogue == "store")
            s.epilogue = generic::Epilogue::Store;
        else if (epilogue == "swiglu")
            s.epilogue = generic::Epilogue::SwiGlu;
        else if (epilogue == "add")
            s.epilogue = generic::Epilogue::AddResidual;
        else
            throw std::runtime_error("shapes.txt: epilogue " + epilogue);
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

const char* band_of(std::int32_t t) {
    return t <= 8     ? "decode"
           : t <= 16  ? "verify"
           : t <= 32  ? "narrow"
           : t <= 128 ? "wide"
                      : "batch129";
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

std::string where(const RegretBoard& b, const Cell& c) {
    return std::string(b.name) + " " + c.shape->line + " " +
           (c.policy == LinearPolicy::AllowA8 ? "a8" : "a16") + " T=" + std::to_string(c.tokens) +
           (c.prefill ? " prefill" : "");
}

std::string json_text(std::string_view s) {
    std::string out = "\"";
    for (const char ch : s) {
        if (ch == '"' || ch == '\\') out += '\\';
        out += ch;
    }
    return out + "\"";
}

std::string answer_json(const generic::RouteAnswer& a) {
    std::string out = "{\"route\": " + json_text(generic::route_name(a.route)) +
                      ", \"authority\": " + json_text(a.evidence.source) +
                      ", \"evidence\": " + json_text(generic::evidence_kind_name(a.evidence.kind));
    if (!a.evidence.donor.empty()) out += ", \"donor\": " + json_text(a.evidence.donor);
    return out + "}";
}

struct BoardTally {
    int cells = 0, borrowed = 0, transferred = 0, measured = 0, moved = 0, demoted_kept_route = 0;
    std::map<std::string, int> moved_by_band;
    std::vector<std::string> moves; // JSON objects
};

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
        const generic::Evidence& le = legacy->evidence;
        const generic::Evidence& ee = evidence->evidence;
        // Legacy: the policy recorded, nothing demoted, the text as before F1.
        const std::string legacy_text = generic::evidence_text(le);
        if (le.policy != RoutePolicy::Legacy || !le.demoted_source.empty() ||
            legacy_text.find("policy=") != std::string::npos ||
            legacy_text.find("demoted=") != std::string::npos) {
            fail(at + ": legacy answer reads '" + legacy_text + "'");
        }
        if (ee.policy != RoutePolicy::Evidence ||
            generic::evidence_text(ee).find(" policy=evidence") == std::string::npos) {
            fail(at + ": evidence answer does not name its policy: '" + generic::evidence_text(ee) +
                 "'");
        }
        if (le.kind == Kind::Measured) ++tally.measured;
        if (le.kind == Kind::Borrowed) ++tally.borrowed;
        if (le.kind == Kind::Transferred) ++tally.transferred;
        if (generic::policy_demotes(RoutePolicy::Evidence, le.kind)) {
            // Under legacy the entry won; under evidence it yields to the ranker.
            if (le.source != "route-table") {
                fail(at + ": a " + generic::evidence_kind_name(le.kind) + " answer of " +
                     std::string(le.source) + ", not of the route table");
            }
            if (ee.kind == Kind::Borrowed || ee.kind == Kind::Transferred ||
                ee.source == "route-table") {
                // Only where the ranker has no candidate (the scalar fallback), which no cell of
                // this grid is: these formats always have a rule's route.
                fail(at + ": the " + generic::evidence_kind_name(le.kind) +
                     " entry did not yield: '" + generic::evidence_text(ee) + "'");
            }
            if (ee.demoted_source != le.source || ee.demoted_kind != le.kind) {
                fail(at + ": the evidence answer does not name the demoted one: '" +
                     generic::evidence_text(ee) + "'");
            }
            const std::string demoted = std::string(" demoted=") + std::string(le.source) + ":" +
                                        generic::evidence_kind_name(le.kind);
            if (generic::evidence_text(ee).find(demoted) == std::string::npos) {
                fail(at + ": evidence text '" + generic::evidence_text(ee) + "' lacks" + demoted);
            }
        } else {
            // Pin, Measured, Carried and Ranker answers are never demoted: the same answer.
            if (!(evidence->route == legacy->route) || ee.source != le.source ||
                ee.kind != le.kind || ee.donor != le.donor || !ee.demoted_source.empty()) {
                fail(at + ": a " + generic::evidence_kind_name(le.kind) + " answer moved: '" +
                     generic::evidence_text(le) + "' -> '" + generic::evidence_text(ee) + "'");
            }
        }
        if (evidence->route == legacy->route) {
            if (!ee.demoted_source.empty()) ++tally.demoted_kept_route;
            continue;
        }
        ++tally.moved;
        ++tally.moved_by_band[c.prefill ? "prefill" : band_of(c.tokens)];
        std::ostringstream move;
        move << "{\"format\": " << json_text(generic::qtype_name(c.shape->qtype))
             << ", \"policy\": " << json_text(c.policy == LinearPolicy::AllowA8 ? "a8" : "a16")
             << ", \"epilogue\": " << json_text(generic::epilogue_name(c.shape->epilogue))
             << ", \"n\": " << c.shape->n << ", \"k\": " << c.shape->k << ", \"T\": " << c.tokens
             << ", \"profile\": " << json_text(c.prefill ? "prefill" : "speed")
             << ", \"band\": " << json_text(c.prefill ? "prefill" : band_of(c.tokens))
             << ", \"legacy\": " << answer_json(*legacy)
             << ", \"evidence\": " << answer_json(*evidence) << "}";
        tally.moves.push_back(move.str());
    }
    return tally;
}

void write_moves(const std::string& path,
                 const std::vector<std::pair<const RegretBoard*, BoardTally>>& boards) {
    std::ofstream out(path, std::ios::binary);
    out << "{\n \"schema\": \"ninferno-route-policy-moves/1\",\n"
        << " \"what\": \"the cells of the A5 regret grid whose production pick moves under "
           "--route-policy evidence against legacy (#614 F1), per board; written by "
           "ninferno_route_policy_test --write\",\n"
        << " \"build_set\": " << json_text(kBuildSet) << ",\n"
        << " \"grid\": {\"shapes\": \"tools/bench/data/route_regret/shapes.txt\", "
           "\"policies\": [\"a16\", \"a8\"], \"speed_T\": [";
    for (std::size_t i = 0; i < std::size(kSpeedPoints); ++i) {
        out << (i ? ", " : "") << kSpeedPoints[i];
    }
    out << "], \"prefill_T\": [" << kPrefillChunk << "]},\n \"boards\": [\n";
    for (std::size_t i = 0; i < boards.size(); ++i) {
        const auto& [b, t] = boards[i];
        out << "  {\"board\": " << json_text(b->name) << ", \"card\": " << json_text(b->card)
            << ", \"cells\": " << t.cells << ", \"legacy_borrowed\": " << t.borrowed
            << ", \"legacy_transferred\": " << t.transferred
            << ", \"legacy_measured\": " << t.measured << ", \"moved\": " << t.moved
            << ", \"demoted_same_route\": " << t.demoted_kept_route << ", \"moved_by_band\": {";
        bool first = true;
        for (const auto& [band, n] : t.moved_by_band) {
            out << (first ? "" : ", ") << json_text(band) << ": " << n;
            first = false;
        }
        out << "},\n   \"moves\": [";
        for (std::size_t m = 0; m < t.moves.size(); ++m) {
            out << (m ? ",\n    " : "\n    ") << t.moves[m];
        }
        out << (t.moves.empty() ? "]" : "\n   ]") << "}" << (i + 1 < boards.size() ? "," : "")
            << '\n';
    }
    out << " ]\n}\n";
    if (!out.good()) throw std::runtime_error("cannot write " + path);
    std::cout << "route policy: wrote " << path << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string write;
        for (int i = 1; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--write" && i + 1 < argc) write = argv[++i];
        }
        if (ops::route_policy() != RoutePolicy::Legacy) fail("the default route policy");
        if (!generic::policy_demotes(RoutePolicy::Evidence, Kind::Borrowed) ||
            !generic::policy_demotes(RoutePolicy::Evidence, Kind::Transferred)) {
            fail("evidence does not demote borrowed and transferred answers");
        }
        for (const Kind k : {Kind::Pin, Kind::Measured, Kind::Ranker, Kind::Carried,
                             Kind::Transferred, Kind::Borrowed}) {
            if (generic::policy_demotes(RoutePolicy::Legacy, k)) fail("legacy demotes an answer");
            if (k != Kind::Borrowed && k != Kind::Transferred &&
                generic::policy_demotes(RoutePolicy::Evidence, k)) {
                fail(std::string("evidence demotes a ") + generic::evidence_kind_name(k) +
                     " answer");
            }
        }
        const std::vector<SweepShape> shapes = read_shapes();
        std::vector<std::pair<const RegretBoard*, BoardTally>> boards;
        int borrowed = 0, measured = 0;
        for (const RegretBoard& b : kRegretBoards) {
            BoardTally t = run_board(b, shapes);
            borrowed += t.borrowed + t.transferred;
            measured += t.measured;
            std::cout << b.name << " (" << kBuildSet << "): " << t.cells << " cells, legacy "
                      << t.borrowed << " borrowed, " << t.transferred << " transferred, "
                      << t.measured << " measured; under evidence " << t.moved << " moved";
            for (const auto& [band, n] : t.moved_by_band) std::cout << ' ' << band << '=' << n;
            std::cout << ", " << t.demoted_kept_route << " demoted to the same route\n";
            boards.emplace_back(&b, std::move(t));
        }
        if (borrowed == 0) fail("no borrowed or transferred answer in the grid: nothing demoted");
        if (measured == 0) fail("no measured answer in the grid: the never-demoted check is idle");
        ops::set_route_policy(RoutePolicy::Legacy);
        if (!write.empty()) write_moves(write, boards);
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
