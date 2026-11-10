// The route selector's authorities (ops/generic/route_authorities.h, #614 A1): the three lists are
// well formed (distinct names, the rules end on the scalar route, which always answers), and on
// every facts board (a substituted DeviceClass, as ninferno_selector_golden_test runs them) the
// evidence of every answer names a listed authority,
// and the evidence is consistent with its kind: a Pin or a Ranker names no donor and no distance,
// a Measured answer is this board's (distance 0), a Carried answer's donor lies within
// kCarryDistance, a Borrowed or Transferred answer names its donor, and the invariant profile is
// answered by its own authority alone. The tally of answers per board and authority is printed:
// which authority decides how much of the grid on which board. Every authority is also asked
// directly (check_direct): where it answers, its evidence names it, so a label cannot be wrong or
// swapped on an authority the walk reaches rarely or never. The printed words of the evidence kinds
// are pinned as literals (check_words), and a description without workspace carries the evidence
// of the route it names.
//
// The routes themselves are the goldens' business; this test holds the evidence.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/measured_boards.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_authorities.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/common/route_probe.h"
#include "ops/generic/selector_facts.h"
#include "ops/route_pins.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace ninferno;
using namespace ninferno::test;
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

int g_failed = 0;

// What the grids reached, for the coverage check at the end (#614 A3): every authority name and
// every kind of evidence must have answered somewhere, or be listed with why it cannot.
std::set<std::string> g_reached_sources;
std::set<std::string> g_reached_kinds;
int g_described = 0, g_describe_threw = 0, g_describe_attempts = 0;
// Authorities asked directly (check_direct): each answers under its own name wherever it answers,
// whether or not an earlier authority would have answered first in the walk.
std::set<std::string> g_direct_sources;
int g_direct_answers = 0;
// Described calls without workspace whose asked route is not the selection's (the A16 heuristic's
// instead): the evidence must be that heuristic's.
int g_substituted = 0;

void fail(const std::string& what) {
    ++g_failed;
    if (g_failed <= 40) std::cerr << "FAIL: " << what << '\n';
}

constexpr QType kFormats[] = {
    QType::BF16,       QType::Q4_G64_FP16, QType::Q8_G32_FP16, QType::FP8_E4M3FN_ROW_BF16,
    QType::NVFP4,      QType::GGML_Q4_K,   QType::GGML_Q6_K,   QType::GGML_Q3_K,
    QType::GGML_IQ3_S, QType::GGML_IQ2_S,  QType::GGML_Q4_0,   QType::GGML_IQ4_XS,
};

struct Shape {
    std::int32_t n, k;
};

constexpr Shape kShapes[]        = {{256, 2048}, {1024, 4096}, {4096, 4096}, {17408, 5120}};
constexpr std::int32_t kTokens[] = {1,  2,  3,  4,  5,   8,   9,   12,  16,   17,  24,
                                    32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048};

// The names an answer may carry: every list's entries and the two an authority answers under
// when a measured entry stands in for its rule (r8_schedule's table, the dense tuner's band).
std::set<std::string_view> listed_sources() {
    std::set<std::string_view> names;
    for (const auto list : {generic::select_authorities(), generic::auto_authorities(),
                            generic::rule_authorities()}) {
        for (const generic::NamedAuthority& a : list) names.insert(a.name);
    }
    names.insert("r8-table");
    names.insert("dense-tune-prefill-band");
    return names;
}

// The tuner's --selected-out header: its old columns in their old order, then the two new ones, so
// a reader of the older files by position still finds every column it knew.
void check_selected_header() {
    const std::string header = generic::kSelectedCsvHeader;
    const std::string old    = "format,compute,epilogue,n,k,T,asked,selected,select_ns,run,device";
    if (!header.starts_with(old + ",")) fail("--selected-out header lost an old column: " + header);
    if (!header.ends_with(",authority,evidence")) fail("--selected-out header: " + header);
    if (header != old + ",authority,evidence") fail("--selected-out header has more: " + header);
}

int check_lists() {
    std::set<std::string_view> seen;
    for (const auto list : {generic::select_authorities(), generic::auto_authorities(),
                            generic::rule_authorities()}) {
        if (list.empty()) fail("an authority list is empty");
        for (const generic::NamedAuthority& a : list) {
            if (a.name.empty()) fail("an authority has no name");
            if (a.answer == nullptr) fail(std::string(a.name) + ": no function");
            if (!seen.insert(a.name).second) fail(std::string(a.name) + ": listed twice");
        }
    }
    const auto rules = generic::rule_authorities();
    if (rules.empty() || rules.back().name != "scalar-fallback") {
        fail("the rules do not end on scalar-fallback");
    }
    if (generic::select_authorities().front().name != "invariant-profile") {
        fail("the invariant profile is not the first authority");
    }
    return g_failed;
}

// The words the evidence prints, as literals (not through evidence_kind_name): a reader of a
// manifest, a description or the tuner's CSV sees these, and tools/bench/regret_collect.py groups
// by them.
void check_words() {
    using K = generic::Evidence::Kind;

    struct Word {
        K kind;
        const char* word;
    };

    constexpr Word words[] = {
        {K::Pin, "pin"},         {K::Measured, "measured"},       {K::Ranker, "ranker"},
        {K::Carried, "carried"}, {K::Transferred, "transferred"}, {K::Borrowed, "borrowed"}};
    std::set<std::string> distinct;
    for (const Word& w : words) {
        distinct.insert(w.word);
        const std::string name = generic::evidence_kind_name(w.kind);
        if (name != w.word) fail(std::string("evidence kind prints '") + name + "', not " + w.word);
        generic::Evidence e;
        e.kind = w.kind;
        e.ranker =
            w.kind == K::Ranker ? generic::Evidence::Ranker::Rule : generic::Evidence::Ranker::None;
        e.source               = "route-table";
        const std::string text = generic::evidence_text(e);
        if (text != std::string("authority=route-table evidence=") + w.word) {
            fail("evidence_text of " + std::string(w.word) + ": '" + text + "'");
        }
        const std::string columns = generic::evidence_columns(e);
        if (columns != std::string("route-table,") + w.word) {
            fail("evidence_columns of " + std::string(w.word) + ": '" + columns + "'");
        }
    }
    if (distinct.size() != 6) fail("the kinds are not six distinct words");
    // A donor and its distance: three decimals, whatever the locale.
    generic::Evidence carried;
    carried.kind     = K::Carried;
    carried.ranker   = generic::Evidence::Ranker::None;
    carried.source   = "route-table";
    carried.donor    = "tu102";
    carried.distance = 0.069;
    if (generic::evidence_text(carried) !=
        "authority=route-table evidence=carried donor=tu102 d=0.069") {
        fail("evidence_text with a donor: '" + generic::evidence_text(carried) + "'");
    }
    if (generic::evidence_columns(carried) != "route-table,carried") {
        fail("evidence_columns with a donor: '" + generic::evidence_columns(carried) + "'");
    }
    generic::Evidence borrowed = carried;
    borrowed.kind              = K::Borrowed;
    borrowed.donor             = "8.9:128sm";
    borrowed.distance          = 0.5;
    if (generic::evidence_text(borrowed) !=
        "authority=route-table evidence=borrowed donor=8.9:128sm d=0.500") {
        fail("evidence_text of a borrowed answer: '" + generic::evidence_text(borrowed) + "'");
    }
}

bool same_evidence(const generic::Evidence& a, const generic::Evidence& b) {
    return a.kind == b.kind && a.ranker == b.ranker && a.source == b.source && a.donor == b.donor &&
           a.distance == b.distance;
}

// Every authority of the three lists asked directly on one ask: where it answers, its evidence
// names it (r8-rules also answers as r8-table, where the decode table does), whatever authority
// would have answered first in the walk. The walk's tally cannot tell a wrong name from a right
// one on an authority it never reaches, nor two names swapped; this can.
void check_direct(const generic::Ask& ask, const generic::Problem& problem, const Weight& w,
                  const std::set<std::string_view>& names, const std::string& where) {
    for (const auto list : {generic::select_authorities(), generic::auto_authorities(),
                            generic::rule_authorities()}) {
        for (const generic::NamedAuthority& a : list) {
            try {
                const auto answer = a.answer(ask, problem, w);
                if (!answer) continue;
                ++g_direct_answers;
                const std::string_view source = answer->evidence.source;
                // r8-rules answers as r8-table only where the decode table gave the entry (a
                // measured or carried answer); its own rule answers are the ranker's.
                const bool right =
                    a.name == "r8-rules"
                        ? source == (answer->evidence.kind == generic::Evidence::Kind::Ranker
                                         ? "r8-rules"
                                         : "r8-table")
                        : source == a.name;
                if (!right) {
                    fail(where + ": " + std::string(a.name) + " answered as '" +
                         std::string(source) + "'");
                }
                if (names.count(source) == 0) {
                    fail(where + ": " + std::string(a.name) + " answered as an unlisted source");
                }
                g_direct_sources.insert(std::string(a.name));
            } catch (const std::exception& error) {
                fail(where + ": " + std::string(a.name) + " threw " + error.what());
            }
        }
    }
}

// A description that throws is not compared, so only the one known cause is allowed: the
// description of a ggml matrix-kernel route with workspace reads the host GPU's SM count for the
// kernel's workspace layout (ggml_mmq device_facts), and a host without a GPU has none. Any other
// throw fails, with its message.
void describe_threw(const std::string& where, generic::Route selected, bool workspace,
                    const std::exception& error) {
    ++g_describe_threw;
    const std::string message = error.what();
    if (!workspace || selected.family != generic::Family::GgmlMmq ||
        message.find("no CUDA-capable device") == std::string::npos) {
        fail(where + ": describe_call threw " + (workspace ? "with" : "without") +
             " workspace on " + generic::route_name(selected) + ": " + message);
    }
}

// The evidence the other readers print is the selection's own: route_answer_query (the decision
// manifest's route_evidence section) gives select_answer's answer, and describe_call's description
// carries its evidence (describe_text prints it, so does the route tuner's CSV through
// evidence_columns). `described`: also describe a call (the cost model prices it, which can need a
// device: a throw there is counted, not failed).
void check_readers(const DeviceClass& device, const generic::Problem& problem, const Weight& w,
                   std::int32_t t, bool prefill, const generic::RouteAnswer& a,
                   const std::string& where, bool described) {
    const auto q = generic::route_answer_query(device, problem.qtype, problem.policy,
                                               problem.epilogue, problem.n, problem.k, t, prefill);
    if (!q) {
        fail(where + ": route_answer_query gives none");
    } else if (!(q->route == a.route) || !same_evidence(q->evidence, a.evidence)) {
        fail(where + ": route_answer_query disagrees with select_answer (" +
             generic::evidence_text(q->evidence) + " vs " + generic::evidence_text(a.evidence) +
             ")");
    }
    // The tuner's two last columns: "<authority>,<kind>", no other comma in them.
    const std::string columns = generic::evidence_columns(a.evidence);
    if (columns !=
            std::string(a.evidence.source) + ',' + generic::evidence_kind_name(a.evidence.kind) ||
        std::count(columns.begin(), columns.end(), ',') != 1) {
        fail(where + ": evidence_columns '" + columns + "'");
    }
    if (!described) return;
    ++g_describe_attempts;
    try {
        const auto d =
            generic::describe_call(device, problem.qtype, problem.policy, problem.epilogue,
                                   problem.n, problem.k, t, std::size_t{1} << 30, prefill);
        ++g_described;
        if (!d || !d->evidence) {
            fail(where + ": describe_call carries no evidence");
            return;
        }
        if (!same_evidence(*d->evidence, a.evidence)) {
            fail(where + ": the description's evidence is not the selection's");
        }
        const std::string text = generic::describe_text(*d);
        const std::string tail = " " + generic::evidence_text(a.evidence);
        if (!text.ends_with(tail)) fail(where + ": describe_text does not end with " + tail);
    } catch (const std::exception& error) { describe_threw(where, a.route, true, error); }
    // Without workspace a W4A4, W4A8 or I8 selection runs the A16 heuristic's route instead
    // (workspace_free_route): the description names that route, and the evidence is the evidence
    // of the walk that gave it, never the selection's of another route.
    ++g_describe_attempts;
    try {
        const auto d =
            generic::describe_call(device, problem.qtype, problem.policy, problem.epilogue,
                                   problem.n, problem.k, t, std::nullopt, prefill);
        ++g_described;
        if (!d || !d->evidence) {
            fail(where + ": describe_call without workspace carries no evidence");
            return;
        }
        const generic::Problem a16{problem.qtype, LinearPolicy::A16Only, problem.n, problem.k,
                                   problem.epilogue};
        const generic::RouteAnswer heuristic =
            generic::heuristic_answer(generic::ask_below_profiles(t, prefill), a16, w);
        // The route the run path takes for this call (the description's asked route is that).
        const generic::Route run_route =
            generic::workspace_free_route(a.route, problem, w, t, prefill, false);
        const bool kept = run_route == a.route;
        if (!kept) ++g_substituted;
        const generic::RouteAnswer& owner = kept ? a : heuristic;
        if (!(d->resolved.asked == run_route)) {
            fail(where + ": without workspace the description asks for " +
                 generic::route_name(d->resolved.asked) + ", the run path takes " +
                 generic::route_name(run_route));
        } else if (!kept && !(run_route == heuristic.route)) {
            fail(where + ": without workspace the run path takes " +
                 generic::route_name(run_route) + ", neither the selection's " +
                 generic::route_name(a.route) + " nor the A16 heuristic's " +
                 generic::route_name(heuristic.route));
        } else if (!same_evidence(*d->evidence, owner.evidence)) {
            fail(where + ": without workspace the call asks for " +
                 generic::route_name(d->resolved.asked) + " with the evidence of " +
                 generic::route_name(a.route) + " (" + generic::evidence_text(*d->evidence) +
                 "), not of its own walk (" + generic::evidence_text(owner.evidence) + ")");
        }
        const std::string text = generic::describe_text(*d);
        if (!text.starts_with("asked=" + generic::route_name(d->resolved.asked) + " ")) {
            fail(where + ": describe_text does not start with the asked route");
        }
        if (!text.ends_with(" " + generic::evidence_text(d->evidence.value()))) {
            fail(where + ": describe_text does not end with its evidence");
        }
    } catch (const std::exception& error) { describe_threw(where, a.route, false, error); }
}

void check_evidence(const generic::RouteAnswer& a, const std::set<std::string_view>& names,
                    const std::string& where, bool invariant) {
    using K                    = generic::Evidence::Kind;
    using R                    = generic::Evidence::Ranker;
    const generic::Evidence& e = a.evidence;
    g_reached_sources.insert(std::string(e.source));
    g_reached_kinds.insert(generic::evidence_kind_name(e.kind));
    if (names.count(e.source) == 0)
        fail(where + ": unlisted source '" + std::string(e.source) + "'");
    if (invariant && e.source != "invariant-profile") {
        fail(where + ": the invariant profile answered by " + std::string(e.source));
    }
    if ((e.kind == K::Ranker) != (e.ranker != R::None)) {
        fail(where + ": kind " + generic::evidence_kind_name(e.kind) + " with ranker " +
             generic::evidence_ranker_name(e.ranker));
    }
    switch (e.kind) {
    case K::Pin:
    case K::Ranker:
    case K::Measured:
        if (!e.donor.empty() || e.distance != 0) {
            fail(where + ": " + generic::evidence_kind_name(e.kind) + " with donor '" + e.donor +
                 "' at " + std::to_string(e.distance));
        }
        break;
    case K::Carried:
        if (e.donor.empty() || e.distance <= 0 || e.distance > measured::kCarryDistance) {
            fail(where + ": carried from '" + e.donor + "' at " + std::to_string(e.distance));
        }
        break;
    case K::Transferred:
        // At distance 0 too: a quantity the driver did not report drops out of the distance, so
        // the MMQ gate's answer for such a board is a transfer even at 0 (mmq_prefill_gate).
        if (e.donor.empty() || e.distance < 0) {
            fail(where + ": transferred from '" + e.donor + "' at " + std::to_string(e.distance));
        }
        break;
    case K::Borrowed:
        if (e.donor.empty()) fail(where + ": borrowed from nobody");
        break;
    }
    // evidence_text prints the donor as one token, and the CSV column holds no comma.
    if (e.donor.find_first_of(" ,") != std::string::npos) {
        fail(where + ": donor '" + e.donor + "' has a space or a comma");
    }
    if (e.source == "cost-model" && e.ranker != R::Cost) fail(where + ": the cost model as a rule");
    // The R8 decode table's entries are measured or carried; the R8 rules' own are the ranker's.
    if (e.source == "r8-table" && e.kind != K::Measured && e.kind != K::Carried) {
        fail(where + ": r8-table answer of kind " + generic::evidence_kind_name(e.kind));
    }
    if (e.source == "r8-rules" && e.kind != K::Ranker) {
        fail(where + ": r8-rules answer of kind " + generic::evidence_kind_name(e.kind));
    }
    if (e.source == "route-pin" || e.source == "route-probe") {
        if (e.kind != K::Pin) fail(where + ": a pin's answer is not Pin evidence");
    }
}

// One answer, asked on a facts board: which authority gave it. The expected names are the current
// code's (the selector goldens hold the routes); a restatement that moved a call to another
// authority fails here, where the evidence checks above would accept any listed name.
struct KnownCase {
    std::string_view board;
    QType qtype;
    Shape shape;
    LinearPolicy policy;
    std::int32_t tokens;
    bool prefill;
    int profile; // 0 speed, 1 invariant, 2 probe (A16 forced), 3 decode pin A16, 4 prefill pin A16
    std::string_view source;
};

std::string known_source(const KnownCase& c) {
    const generic::FactsBoard* board = nullptr;
    for (const generic::FactsBoard& b : generic::kFactsBoards) {
        if (b.name == c.board) board = &b;
    }
    if (board == nullptr) return "no such board";
    const ScopedDeviceClass scoped(generic::board_device_class(*board));
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    quantized_weight::PatternedWeightOptions options;
    if (c.qtype == QType::NVFP4) options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
    const quantized_weight::PackedWeight host =
        quantized_weight::make_patterned_weight(c.qtype, c.shape.n, c.shape.k, 7U, options);
    const Weight w = host.device_weight(payload);
    const generic::Problem problem{c.qtype, c.policy, c.shape.n, c.shape.k,
                                   generic::Epilogue::Store};
    std::optional<ops::ScopedInvariantRoutes> invariant;
    if (c.profile == 1) invariant.emplace();
    std::optional<ops::detail::RouteProbe> probe;
    if (c.profile == 2) probe.emplace(0, std::int8_t{0});

    struct PinReset {
        ~PinReset() { ops::set_route_pins({}); }
    } reset;

    if (c.profile == 3) ops::set_route_pins({PrefillRoute::Auto, DecodeRoute::A16, MoePath::Auto});
    if (c.profile == 4) ops::set_route_pins({PrefillRoute::A16, DecodeRoute::Auto, MoePath::Auto});
    try {
        const generic::Evidence e =
            generic::select_answer(problem, w, c.tokens, c.prefill).evidence;
        g_reached_sources.insert(std::string(e.source));
        g_reached_kinds.insert(generic::evidence_kind_name(e.kind));
        // The same call, every authority asked on its own (the pin, the probe and the profile are
        // in force here, which the board grids never set but for the invariant profile).
        check_direct(generic::resolve_ask(c.tokens, c.prefill), problem, w, listed_sources(),
                     std::string(c.board) + " " + generic::qtype_name(c.qtype) +
                         " known case, profile " + std::to_string(c.profile) +
                         " T=" + std::to_string(c.tokens));
        return std::string(e.source);
    } catch (const std::exception& error) { return std::string("threw ") + error.what(); }
}

void check_known() {
    constexpr Shape big{4096, 4096};
    constexpr QType q8         = QType::Q8_G32_FP16;
    constexpr QType q4k        = QType::GGML_Q4_K;
    constexpr QType q6k        = QType::GGML_Q6_K;
    constexpr LinearPolicy a16 = LinearPolicy::A16Only;
    const KnownCase cases[]    = {
        // invariant profile, a pin of either band, a route probe
        {"rtx4090", q8, big, a16, 64, true, 1, "invariant-profile"},
        {"rtx4090", q8, big, a16, 4, false, 2, "route-probe"},
        {"rtx4090", q8, big, a16, 4, false, 3, "route-pin"},
        {"rtx4090", q8, big, a16, 512, true, 4, "route-pin"},
        // the route table, the cost model, the rules, the MMQ gate
        {"rtx4090", q4k, big, a16, 512, true, 0, "route-table"},
        {"rtx4090", q4k, big, a16, 1, false, 0, "route-table"},
        {"rtx4090", q8, big, a16, 64, false, 0, "cost-model"},
        {"rtx4090", q8, big, a16, 1, false, 0, "rmma-rule"},
        {"rtx4090", q6k, big, a16, 24, true, 0, "rmma-rule"},
        {"rtx4090", q6k, big, LinearPolicy::AllowA8, 24, true, 0, "mmq-prefill-gate"},
        {"rtx4090", QType::GGML_IQ3_S, big, LinearPolicy::AllowA8, 512, true, 0,
            "mmq-prefill-gate"},
    };
    for (const KnownCase& c : cases) {
        const std::string got = known_source(c);
        if (got != c.source) {
            fail(std::string(c.board) + " " + generic::qtype_name(c.qtype) + " " +
                 std::to_string(c.shape.n) + "x" + std::to_string(c.shape.k) +
                 " T=" + std::to_string(c.tokens) + " profile " + std::to_string(c.profile) +
                 ": answered by '" + got + "', expected '" + std::string(c.source) + "'");
        }
    }
}

// The MMQ prefill gate on a board whose L2 size is unknown (a card outside the facts whose driver
// reports none): the gate answers Incomplete, which is not a measurement of this board even at
// distance 0, so the evidence names a donor and is not `measured`.
void check_incomplete_gate() {
    constexpr Shape big{4096, 4096};
    for (const std::string_view name : {"rtx4090", "rtx5090", "rtx2080ti"}) {
        const generic::FactsBoard* board = nullptr;
        for (const generic::FactsBoard& b : generic::kFactsBoards) {
            if (b.name == name) board = &b;
        }
        if (board == nullptr) {
            fail("incomplete gate: no board " + std::string(name));
            continue;
        }
        for (const std::size_t l2 :
             {std::size_t{0}, generic::board_device_class(*board).l2_bytes}) {
            DeviceClass device = generic::board_device_class(*board);
            device.l2_bytes    = l2;
            const ScopedDeviceClass scoped(device);
            void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
            const quantized_weight::PackedWeight host = quantized_weight::make_patterned_weight(
                QType::GGML_Q4_K, big.n, big.k, 7U, quantized_weight::PatternedWeightOptions{});
            const Weight w = host.device_weight(payload);
            const generic::Problem problem{QType::GGML_Q4_K, LinearPolicy::AllowA8, big.n, big.k,
                                           generic::Epilogue::Store};
            const std::string where =
                std::string(name) + " q4k prefill gate, l2_bytes " + std::to_string(l2);
            std::optional<generic::RouteAnswer> answer;
            for (const generic::NamedAuthority& a : generic::rule_authorities()) {
                if (a.name == "mmq-prefill-gate") {
                    answer = a.answer(generic::resolve_ask(512, true), problem, w);
                }
            }
            if (!answer) {
                fail(where + ": the gate gives no answer");
                continue;
            }
            const generic::Evidence& e = answer->evidence;
            if (l2 == 0) {
                if (e.kind == generic::Evidence::Kind::Measured || e.donor.empty()) {
                    fail(where + ": unknown L2 printed as '" + generic::evidence_text(e) + "'");
                }
            } else if (e.kind == generic::Evidence::Kind::Transferred ||
                       (e.kind == generic::Evidence::Kind::Measured) != e.donor.empty()) {
                // The board's own facts: measured (no donor) where it is a measured board, carried
                // (a donor) where a measured board within reach answers.
                fail(where + ": the board's own facts printed as '" + generic::evidence_text(e) +
                     "'");
            }
        }
    }
}

// Authorities and kinds the grids above never reach, with why. Anything else unreached fails: a
// new authority, or a restatement that moved the last call off one, must say so here.
// The walk may leave an authority unreached when an earlier one always answers first (the staged
// MMA, the fallbacks); check_direct asks every authority on its own, so what the walk misses is
// still held by name there. Only these two cannot answer on a substituted board at all.
// (Unreached by the walk, and held by the direct asks: staged-rule, staged-fallback and
// scalar-fallback, which the grid has no weight for or always leaves to an earlier authority.)
const std::map<std::string_view, std::string_view> kUnreached = {
    {"dense-tune", "routes timed on a device are not a substituted board's "
                   "(tuned_dense_route: substituted_foreign_class) and no tune runs here"},
    {"dense-tune-prefill-band", "the same tuner's bands, the same refusal for a substituted board"},
};
const std::map<std::string_view, std::string_view> kUnreachedKinds = {};

// What this build leaves unreached on top of kUnreached, read from the build and never from an arch
// number: the W4A4 rule answers only where a W4A4 MMA kernel is compiled in (w4a4_schedule_count
// is 0 on a build without them, so route_admits refuses every W4A4 route and the rule has nothing
// to answer with). A build that has the kernels keeps the rule held: it must be reached and must
// answer on its own.
std::map<std::string_view, std::string_view> build_unreached() {
    std::map<std::string_view, std::string_view> excused;
    if (generic::w4a4_schedule_count() == 0) {
        excused.emplace("w4a4-rule", "no W4A4 MMA kernel in this build");
    }
    return excused;
}

void check_coverage(const std::set<std::string_view>& names) {
    std::map<std::string_view, std::string_view> unreached     = kUnreached;
    const std::map<std::string_view, std::string_view> excused = build_unreached();
    unreached.insert(excused.begin(), excused.end());
    std::string excused_text;
    for (const auto& [name, why] : excused) {
        excused_text += " " + std::string(name) + " (" + std::string(why) + ")";
    }
    std::string missing;
    for (const std::string_view n : names) {
        if (g_reached_sources.count(std::string(n)) == 0 &&
            g_direct_sources.count(std::string(n)) == 0 && unreached.count(n) == 0) {
            missing += " " + std::string(n);
        }
    }
    // Every listed authority answered under its own name when asked on its own.
    std::string missing_direct;
    for (const auto list : {generic::select_authorities(), generic::auto_authorities(),
                            generic::rule_authorities()}) {
        for (const generic::NamedAuthority& a : list) {
            if (g_direct_sources.count(std::string(a.name)) == 0 && unreached.count(a.name) == 0) {
                missing_direct += " " + std::string(a.name);
            }
        }
    }
    std::string missing_kinds;
    using K = generic::Evidence::Kind;
    for (const K k : {K::Pin, K::Measured, K::Ranker, K::Carried, K::Transferred, K::Borrowed}) {
        const std::string name = generic::evidence_kind_name(k);
        if (g_reached_kinds.count(name) == 0 && kUnreachedKinds.count(name) == 0) {
            missing_kinds += " " + name;
        }
    }
    std::cout << "authorities excused by the build:"
              << (excused_text.empty() ? " none" : excused_text) << '\n'
              << "unreached authorities:" << (missing.empty() ? " none" : missing) << '\n'
              << "authorities never answering when asked directly:"
              << (missing_direct.empty() ? " none" : missing_direct) << '\n'
              << "unreached evidence kinds:" << (missing_kinds.empty() ? " none" : missing_kinds)
              << '\n'
              << "direct answers compared: " << g_direct_answers << '\n'
              << "descriptions compared: " << g_described << " (threw " << g_describe_threw
              << " of " << g_describe_attempts << " asked)\n"
              << "descriptions without workspace on another route than the selection's: "
              << g_substituted << '\n';
    if (!missing.empty()) fail("authorities never reached:" + missing);
    if (!missing_direct.empty()) fail("authorities never answering on their own:" + missing_direct);
    if (g_describe_attempts == 0) fail("no call was described");
    if (g_substituted == 0) fail("no described call without workspace left the selection's route");
    if (!missing_kinds.empty()) fail("evidence kinds never reached:" + missing_kinds);
}

int run_board(const generic::FactsBoard& b, const std::set<std::string_view>& names) {
    const DeviceClass board = generic::board_device_class(b);
    const ScopedDeviceClass scoped(board);
    std::map<std::string_view, int> tally;
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const QType qtype : kFormats) {
        for (const Shape shape : kShapes) {
            quantized_weight::PatternedWeightOptions options;
            if (qtype == QType::NVFP4)
                options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
            std::optional<quantized_weight::PackedWeight> host;
            try {
                host.emplace(
                    quantized_weight::make_patterned_weight(qtype, shape.n, shape.k, 7U, options));
            } catch (const std::invalid_argument&) { continue; }
            const Weight w = host->device_weight(payload);
            if (!generic::supports(w)) continue;
            for (const LinearPolicy policy :
                 {LinearPolicy::A16Only, LinearPolicy::AllowA8, LinearPolicy::AllowA4}) {
                if (policy == LinearPolicy::AllowA4 && qtype != QType::NVFP4) continue;
                for (const generic::Epilogue epilogue :
                     {generic::Epilogue::Store, generic::Epilogue::SwiGlu}) {
                    const std::int32_t rows =
                        epilogue == generic::Epilogue::SwiGlu ? shape.n / 2 : shape.n;
                    const generic::Problem problem{qtype, policy, rows, shape.k, epilogue};
                    for (const int profile : {0, 1, 2}) { // speed, prefill, invariant
                        std::optional<ops::ScopedInvariantRoutes> invariant;
                        if (profile == 2) invariant.emplace();
                        const bool prefill = profile == 1;
                        for (const std::int32_t t : kTokens) {
                            if (prefill && t < 17) continue;
                            const std::string where =
                                std::string(b.name) + " " + generic::qtype_name(qtype) + " p" +
                                std::to_string(static_cast<int>(policy)) + " " +
                                generic::epilogue_name(epilogue) + " " + std::to_string(rows) +
                                "x" + std::to_string(shape.k) + (prefill ? " prefill" : "") +
                                (profile == 2 ? " invariant" : "") + " T=" + std::to_string(t);
                            try {
                                const generic::RouteAnswer a =
                                    generic::select_answer(problem, w, t, prefill);
                                check_evidence(a, names, where, profile == 2);
                                ++tally[a.evidence.source];
                                const bool sampled =
                                    shape.n == 4096 && epilogue == generic::Epilogue::Store &&
                                    (t == 1 || t == 4 || t == 17 || t == 64 || t == 512);
                                check_readers(board, problem, w, t, prefill, a, where, sampled);
                                // Every authority on its own at the sampled calls, for the round
                                // of one sequence and of several (the MMQ verify rule reads it).
                                if (shape.n == 4096 && epilogue == generic::Epilogue::Store &&
                                    (sampled || t == 9 || t == 16)) {
                                    const generic::Profile ask_profile =
                                        profile == 2 ? generic::Profile::Invariant
                                        : prefill    ? generic::Profile::Prefill
                                                     : generic::Profile::Speed;
                                    for (const int lanes : {1, 4}) {
                                        check_direct(
                                            generic::Ask{ask_profile, t, lanes}, problem, w, names,
                                            where + " direct lanes " + std::to_string(lanes));
                                    }
                                }
                            } catch (const std::exception& error) {
                                fail(where + ": threw " + error.what());
                            }
                        }
                    }
                }
            }
        }
    }
    // BF16 is no packed format (make_patterned_weight has none, so the grid above holds none): the
    // weight by hand, for the authorities only a BF16 weight reaches (the staged MMA).
    {
        Weight bf16{};
        bf16.payload         = payload;
        bf16.payload_bytes   = std::size_t{4096} * 4096 * 2;
        bf16.qtype           = QType::BF16;
        bf16.shape[0]        = 4096;
        bf16.shape[1]        = 4096;
        bf16.padded_shape[0] = 4096;
        bf16.padded_shape[1] = 4096;
        bf16.ndim            = 2;
        bf16.qdata           = payload;
        bf16.n               = 4096;
        bf16.k               = 4096;
        bf16.layout          = QuantLayout::Contiguous;
        const generic::Problem problem{QType::BF16, LinearPolicy::A16Only, 4096, 4096,
                                       generic::Epilogue::Store};
        if (!generic::supports(bf16))
            fail(std::string(b.name) + ": a BF16 weight is not supported");
        for (const int profile : {0, 1, 2}) { // speed, prefill, invariant
            std::optional<ops::ScopedInvariantRoutes> invariant;
            if (profile == 2) invariant.emplace();
            const bool prefill = profile == 1;
            for (const std::int32_t t : {1, 4, 9, 16, 17, 64, 512}) {
                if (prefill && t < 17) continue;
                const std::string where =
                    std::string(b.name) + " bf16 4096x4096" + (prefill ? " prefill" : "") +
                    (profile == 2 ? " invariant" : "") + " T=" + std::to_string(t);
                try {
                    const generic::RouteAnswer a =
                        generic::select_answer(problem, bf16, t, prefill);
                    check_evidence(a, names, where, profile == 2);
                    ++tally[a.evidence.source];
                    const generic::Profile ask_profile = profile == 2 ? generic::Profile::Invariant
                                                         : prefill    ? generic::Profile::Prefill
                                                                      : generic::Profile::Speed;
                    for (const int lanes : {1, 4}) {
                        check_direct(generic::Ask{ask_profile, t, lanes}, problem, bf16, names,
                                     where + " direct lanes " + std::to_string(lanes));
                    }
                } catch (const std::exception& error) { fail(where + ": threw " + error.what()); }
            }
        }
    }
    std::cout << b.name << ":";
    for (const auto& [source, count] : tally) std::cout << " " << source << "=" << count;
    std::cout << '\n';
    return 0;
}

} // namespace

int main() {
    try {
        const auto names = listed_sources();
        check_lists();
        check_words();
        check_selected_header();
        check_known();
        check_incomplete_gate();
        for (const generic::FactsBoard& b : generic::kFactsBoards) run_board(b, names);
        check_coverage(names);
    } catch (const std::exception& error) {
        std::cerr << "route authorities: " << error.what() << '\n';
        return 1;
    }
    if (g_failed != 0) {
        std::cerr << "route authorities: " << g_failed << " failures\n";
        return 1;
    }
    std::cout << "route authorities: ok\n";
    return 0;
}
