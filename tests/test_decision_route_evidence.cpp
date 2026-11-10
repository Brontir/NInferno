// The decision manifest's route_evidence section (#614 A3, runtime/engine/decision_routes.h), on
// substituted boards, no GPU: same keys and count line as the routes section; at every width the
// route of a line is the routes section's and the evidence beside it is the selector's own answer
// (route_answer_query); replayed from its text it gives itself back; and the section changes no
// hash: the fingerprint, the manifest id, the routes and the observed text are the same with the
// section empty or filled.
#include "runtime/engine/decision_routes.h"

#include "core/device_class.h"
#include "core/weight_view.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/route_authorities.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ops::detail::generic;
namespace rt      = ninferno::runtime;
using ops::LinearPolicy;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL decision route evidence: " << what << '\n';
        ++failures;
    }
}

generic::SeenProblem problem(QType q, LinearPolicy policy, generic::Epilogue e, std::int32_t n,
                             std::int32_t k, std::int32_t parent_rows) {
    return {generic::Problem{q, policy, n, k, e}, parent_rows, default_weight_layout(q)};
}

std::vector<generic::SeenProblem> problems() {
    using generic::Epilogue;
    return {
        problem(QType::GGML_Q4_K, LinearPolicy::A16Only, Epilogue::Store, 4096, 4096, 4096),
        problem(QType::GGML_Q4_K, LinearPolicy::AllowA8, Epilogue::Store, 4096, 4096, 4096),
        problem(QType::GGML_Q6_K, LinearPolicy::AllowA8, Epilogue::AddResidual, 4096, 12288, 4096),
        problem(QType::BF16, LinearPolicy::A16Only, Epilogue::SwiGlu, 6144, 2048, 12288),
        problem(QType::Q8_G32_FP16, LinearPolicy::A16Only, Epilogue::Store, 1024, 4096, 6144)};
}

// A line's runs by width alone: width -> what the run names (the text after "<from>-<to> ").
std::map<std::int32_t, std::string> by_width(const std::string& line) {
    std::map<std::int32_t, std::string> out;
    const std::string decode = line.substr(0, line.find("; "));
    std::size_t at           = 0;
    while (at < decode.size()) {
        const std::size_t end  = std::min(decode.find(", ", at), decode.size());
        const std::string run  = decode.substr(at, end - at);
        const std::size_t dash = run.find('-'), space = run.find(' ');
        for (int t = std::stoi(run.substr(0, dash));
             t <= std::stoi(run.substr(dash + 1, space - dash - 1)); ++t) {
            out[t] = run.substr(space + 1);
        }
        at = end + 2;
    }
    return out;
}

const char* policy_name(LinearPolicy p) {
    switch (p) {
    case LinearPolicy::A16Only:
        return "a16only";
    case LinearPolicy::AllowA8:
        return "allowa8";
    case LinearPolicy::AllowA4:
        return "allowa4";
    }
    return "a16only";
}

std::string route_part(const std::string& named) {
    return named.substr(0, named.find(" authority="));
}

} // namespace

int main() {
    const auto list = problems();
    const int chunk = 1024;
    std::set<std::string> seen_sources;
    int boards = 0, lines_with_donor = 0;
    for (const char* name : {"rtx5090", "rtx4090", "rtx3090", "rtx2080ti", "v100", "rtx5060ti"}) {
        const auto* fb = std::find_if(
            std::begin(generic::kFactsBoards), std::end(generic::kFactsBoards),
            [&](const generic::FactsBoard& b) { return std::string_view(b.name) == name; });
        if (fb == std::end(generic::kFactsBoards)) {
            check(false, std::string("no facts board ") + name);
            continue;
        }
        ++boards;
        const DeviceClass d = generic::board_device_class(*fb);
        for (const bool invariant : {false, true}) {
            const auto routes    = rt::routes_of(d, list, chunk, invariant);
            const auto evidence  = rt::route_evidence_of(d, list, chunk, invariant);
            const std::string at = std::string(name) + (invariant ? " invariant" : "");
            check(routes.size() == evidence.size() && routes.size() == list.size() + 1,
                  at + ": one evidence line per route line and the count");
            check(evidence.at(rt::kDecisionRouteCount) == routes.at(rt::kDecisionRouteCount),
                  at + ": the count line");
            const ops::ScopedInvariantRoutes profile(invariant);
            for (const auto& [key, value] : routes) {
                if (key == rt::kDecisionRouteCount) continue;
                const auto found = evidence.find(key);
                check(found != evidence.end(), at + ": an evidence line for " + key);
                if (found == evidence.end()) continue;
                // Width by width: the route is the routes section's, the evidence the selector's.
                const auto named = by_width(found->second), plain = by_width(value);
                check(named.size() == plain.size() &&
                          named.size() == static_cast<std::size_t>(rt::kDecisionRouteWidths),
                      at + ": " + key + " covers widths 1..16");
                const auto it =
                    std::find_if(list.begin(), list.end(), [&](const generic::SeenProblem& s) {
                        return key.find(std::to_string(s.problem.n) + 'x' +
                                        std::to_string(s.problem.k) + " of " +
                                        std::to_string(s.parent_rows) + ' ') != std::string::npos &&
                               key.starts_with(std::string(generic::qtype_name(s.problem.qtype)) +
                                               ' ') &&
                               key.find(std::string(" ") + policy_name(s.problem.policy) + ' ') !=
                                   std::string::npos;
                    });
                if (it == list.end()) {
                    check(false, at + ": no problem for " + key);
                    continue;
                }
                const generic::Problem& p = it->problem;
                {
                    const ops::ScopedRoundLanes round(0); // the width-alone runs, as routes_of asks
                    for (std::int32_t t = 1; t <= rt::kDecisionRouteWidths; ++t) {
                        if (!named.count(t) || !plain.count(t)) continue;
                        check(route_part(named.at(t)) == plain.at(t),
                              at + ": " + key + " T=" + std::to_string(t) + " names " +
                                  named.at(t) + " where the routes section has " + plain.at(t));
                        const auto a =
                            generic::route_answer_query(d, p.qtype, p.policy, p.epilogue, p.n, p.k,
                                                        t, false, it->parent_rows, it->layout);
                        check(a && named.at(t) == generic::route_name(a->route) + ' ' +
                                                      generic::evidence_text(a->evidence),
                              at + ": " + key + " T=" + std::to_string(t) + ": " + named.at(t));
                        if (a) seen_sources.insert(std::string(a->evidence.source));
                    }
                }
                // The prefill chunk, evidence at the end of the line (asked outside the lanes'
                // scope, as routes_of does).
                const auto pre =
                    generic::route_answer_query(d, p.qtype, p.policy, p.epilogue, p.n, p.k, chunk,
                                                true, it->parent_rows, it->layout);
                check(pre && found->second.ends_with("; prefill 1024 " +
                                                     generic::route_name(pre->route) + ' ' +
                                                     generic::evidence_text(pre->evidence)),
                      at + ": " + key + " prefill: " + found->second);
                if (found->second.find(" donor=") != std::string::npos) ++lines_with_donor;
            }
            // Replayed from the text alone on the board's class, and the order is not an input.
            check(rt::replay_route_evidence(d, evidence) == evidence,
                  at + ": the evidence lines replay to themselves");
            auto reversed = list;
            std::reverse(reversed.begin(), reversed.end());
            check(rt::route_evidence_of(d, reversed, chunk, invariant) == evidence,
                  at + ": the problems' order changes nothing");
            // The routes section the replay reads is the one before this section existed: its text
            // carries no evidence field.
            for (const auto& [key, value] : routes) {
                check(value.find("authority=") == std::string::npos &&
                          value.find("evidence=") == std::string::npos,
                      at + ": the routes line " + key + " is unchanged: no evidence in it");
            }
        }
    }
    check(boards == 6, "all six boards ran");
    check(seen_sources.size() >= 3, "the lines name several authorities");
    std::cout << "evidence lines naming a donor: " << lines_with_donor << '\n';

    // The section is in the manifest text and in no hash: with it empty or filled, the
    // fingerprint, the manifest id, the canonical, routes and observed text are the same.
    const DeviceClass d = generic::board_device_class(generic::kFactsBoards[0]);
    rt::DecisionInputs without;
    without.plan["route_breakpoints"] = "17";
    without.facts["sms"]              = "170";
    without.routes                    = rt::routes_of(d, list, chunk, false);
    without.observed["rung_selected"] = "0";
    rt::DecisionInputs with           = without;
    with.route_evidence               = rt::route_evidence_of(d, list, chunk, false);
    const auto a = rt::decision_fingerprint(without), b = rt::decision_fingerprint(with);
    check(a.value == b.value && a.manifest == b.manifest && a.canonical == b.canonical &&
              a.routes == b.routes && a.observed == b.observed,
          "the evidence section moves no hash and no other section's text");
    check(a.route_evidence.empty() && !b.route_evidence.empty(),
          "the section's text is there only when it is filled");
    // One text line per section line, each "route_evidence.<key>\t<value>\n".
    check(static_cast<std::size_t>(std::count(b.route_evidence.begin(), b.route_evidence.end(),
                                              '\n')) == with.route_evidence.size(),
          "one text line per evidence line");
    check(a.routes.size() > 0 && b.routes.starts_with("routes."),
          "the routes text keeps its prefix");
    check(b.route_evidence.starts_with("route_evidence."), "the evidence text's prefix");

    std::cout << (failures == 0 ? "OK" : "FAIL") << " decision route evidence\n";
    return failures == 0 ? 0 : 1;
}
