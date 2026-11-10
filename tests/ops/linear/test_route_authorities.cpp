// The route selector's authorities (ops/generic/route_authorities.h, #614 A1): the three lists are
// well formed (distinct names, the rules end on the scalar route, which always answers), and on
// every facts board (a substituted DeviceClass, as ninferno_selector_golden_test runs them) the
// answer with evidence is the route select_route gives, its evidence names a listed authority,
// and the evidence is consistent with its kind: a Pin or a Ranker names no donor and no distance,
// a Measured answer is this board's (distance 0), a Carried answer's donor lies within
// kCarryDistance, a Borrowed or Transferred answer names its donor, and the invariant profile is
// answered by its own authority alone. The tally of answers per board and authority is printed:
// which authority decides how much of the grid on which board.
//
// The routes themselves are the goldens' business; this test holds the evidence.
#include "ops/quantized_weight.h"

#include "core/device_class.h"
#include "core/measured_boards.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/route_authorities.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

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
namespace generic = ninferno::ops::detail::generic;
using ops::LinearPolicy;

int g_failed = 0;

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

void check_evidence(const generic::RouteAnswer& a, const std::set<std::string_view>& names,
                    const std::string& where, bool invariant) {
    using K                    = generic::Evidence::Kind;
    using R                    = generic::Evidence::Ranker;
    const generic::Evidence& e = a.evidence;
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
        if (e.distance <= 0) fail(where + ": transferred at " + std::to_string(e.distance));
        break;
    case K::Borrowed:
        if (e.donor.empty()) fail(where + ": borrowed from nobody");
        break;
    }
    if (e.source == "cost-model" && e.ranker != R::Cost) fail(where + ": the cost model as a rule");
    if (e.source == "route-pin" || e.source == "route-probe") {
        if (e.kind != K::Pin) fail(where + ": a pin's answer is not Pin evidence");
    }
}

int run_board(const generic::FactsBoard& b, const std::set<std::string_view>& names) {
    const ScopedDeviceClass scoped(generic::board_device_class(b));
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
                                const generic::Route r =
                                    generic::select_route(problem, w, t, prefill);
                                if (!(a.route == r)) {
                                    fail(where + ": select_answer " + generic::route_name(a.route) +
                                         " but select_route " + generic::route_name(r));
                                }
                                check_evidence(a, names, where, profile == 2);
                                ++tally[a.evidence.source];
                            } catch (const std::exception& error) {
                                fail(where + ": threw " + error.what());
                            }
                        }
                    }
                }
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
        for (const generic::FactsBoard& b : generic::kFactsBoards) run_board(b, names);
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
