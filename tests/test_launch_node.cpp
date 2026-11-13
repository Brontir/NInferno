// The launch node cost (#617 step B): one constant, ninferno::launch_node_seconds
// (core/cost_constants.h), and the launch-like numbers that stay separate facts. Pins every use
// site's value so a change is deliberate. And the per-architecture launch facts
// (core/launch_facts.h, #613 PR-8b), which no model reads yet: they are the ISA rows' launch
// entries, unknown where a row has none, and the Ada row's node + ramp is held against the
// census's 1.5 us within the event tick, a discrepancy printed (the census value stays the round
// cost's until a DECISIONS entry moves it). No GPU.
#include "core/isa_facts.h"
#include "core/launch_facts.h"
#include "ops/generic/route_cost_constants.h"
#include "spec/round_cost.h"

#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "launch node: %s\n", what.c_str());
        ++failures;
    }
}

} // namespace

int main() {
    namespace sp = ninferno::spec;
    namespace rc = ninferno::ops::detail::generic::route_cost;

    // The one constant: Ada only, every other generation unknown (0).
    expect(ninferno::launch_node_seconds("ada") == 1.5e-6, "Ada's node is 1.5e-6 s");
    for (const char* g : {"ga10x", "turing", "volta", "gb20x", "", "hopper"}) {
        expect(ninferno::launch_node_seconds(g) == 0.0, std::string("no node for ") + g);
    }
    expect(ninferno::kLaunchNode[0].cards == 3, "the row records its three Ada cards");
    expect(rc::kLaunchNodeAda.from == rc::Provenance::Fitted &&
               rc::kLaunchNodeAda.value() == ninferno::launch_node_seconds("ada"),
           "provenance is in the route cost registry; no Holdout record until #635");

    // Use site 1: the speculation round cost reads it.
    expect(sp::kLaunchTime[0].generation == "ada" && sp::kLaunchTime[0].eta_dram == 0.946,
           "kLaunchTime keeps Ada's DRAM share only");
    const auto ada = sp::card_rates_of_board(0x2782);
    expect(ada.launch_seconds == 1.5e-6 && ada.eta_dram == 0.946, "an Ada card reads 1.5e-6 s");
    expect(sp::card_rates_of_board(0x2503).launch_seconds == 0.0 &&
               sp::card_rates_of_board(0x2C05).launch_seconds == 0.0,
           "Ampere and gb20x have no node");

    // Use site 2: the route model's quantizer kernel is its own fact, not the node.
    expect(rc::kQuantLaunchSeconds.value() == 1.7e-6, "quantizer launch stays 1.7e-6");
    expect(rc::kQuantLaunchSeconds.value() != ninferno::launch_node_seconds("ada"),
           "quantizer launch and graph node are different facts");

    // The launch facts: each kind of every ISA row is its launch entry's ns as seconds, unknown
    // where the entry or the fact is absent; an architecture without a row is unknown.
    using ninferno::LaunchKind;
    const auto same = [](const std::optional<double>& s, const std::optional<double>& ns) {
        return s.has_value() == ns.has_value() && (!s || *s == *ns * 1e-9);
    };
    for (const ninferno::IsaRow& isa : ninferno::isa_facts_table()) {
        const std::string where = "sm_" + isa.arch;
        const auto row          = ninferno::launch_row(isa.major, isa.minor);
        expect(row.has_value() == isa.launch.has_value(), where + ": a launch row iff an entry");
        for (const LaunchKind k :
             {LaunchKind::GraphNode, LaunchKind::GraphNodePdl, LaunchKind::Stream}) {
            const auto s = ninferno::launch_seconds(isa.major, isa.minor, k);
            if (!isa.launch) {
                expect(!s.has_value(), where + ": no launch entry, every kind unknown");
                continue;
            }
            const auto& ns = k == LaunchKind::GraphNode      ? isa.launch->node_ns
                             : k == LaunchKind::GraphNodePdl ? isa.launch->node_pdl_ns
                                                             : isa.launch->stream_ns;
            if (k == LaunchKind::GraphNodePdl && isa.major < 9) {
                expect(!s.has_value(), where + ": no PDL node below sm_90, whatever the row holds");
                expect(same(row->node_pdl_s, ns), where + ": the row keeps the reading");
                continue;
            }
            expect(same(s, ns), where + ": a kind is its entry's fact in seconds");
        }
        if (!row) continue;
        expect(row->floor_ctas[0] == 1 && row->floor_ctas[1] == row->sms &&
                   row->floor_ctas[2] == 4 * row->sms && row->floor_ctas[3] == 1024 &&
                   row->floor_ctas[4] == 4096,
               where + ": the floors are grids of 1, SMs, 4 x SMs, 1024 and 4096 CTAs");
        expect(same(row->floor_s[0], isa.launch->floor_ns_1) &&
                   same(row->floor_s[3], isa.launch->floor_ns_1024) &&
                   same(row->floor_s[4], isa.launch->floor_ns_4096),
               where + ": the floors are the entry's");
        // The ramp (#613 PR-8a): piecewise linear between the four measured grids, each point
        // reproduced exactly (floor_s[i] - floor_s[0]), none at one CTA or fewer, between two
        // points between their ramps, and past 1024 CTAs the last segment's slope, marked
        // extrapolated. Not one figure a CTA (#870: the slopes differ by segment).
        bool all_floors = true;
        for (const auto& s : row->floor_s) all_floors = all_floors && s.has_value();
        const auto at = [&](double ctas) {
            return ninferno::ramp_seconds(isa.major, isa.minor, ctas);
        };
        if (!all_floors) continue;
        for (int i = 0; i < 4; ++i) {
            const auto r = at(row->floor_ctas[i]);
            expect(r && !r->extrapolated && r->seconds == *row->floor_s[i] - *row->floor_s[0],
                   where + ": the ramp reproduces the point of " +
                       std::to_string(row->floor_ctas[i]) + " CTAs");
        }
        expect(at(0.5) && at(0.5)->seconds == 0.0 && at(1) && at(1)->seconds == 0.0,
               where + ": one CTA has no ramp");
        for (int i = 0; i < 3; ++i) {
            const double lo = *row->floor_s[i] - *row->floor_s[0];
            const double hi = *row->floor_s[i + 1] - *row->floor_s[0];
            const auto mid  = at(0.5 * (row->floor_ctas[i] + row->floor_ctas[i + 1]));
            expect(mid && !mid->extrapolated &&
                       std::fabs(mid->seconds - 0.5 * (lo + hi)) <= 1e-9 * std::fabs(hi) + 1e-18,
                   where + ": linear between two points");
        }
        const double slope =
            (*row->floor_s[3] - *row->floor_s[2]) / (row->floor_ctas[3] - row->floor_ctas[2]);
        const auto past = at(2048);
        expect(past && past->extrapolated &&
                   std::fabs(past->seconds - (*row->floor_s[3] - *row->floor_s[0] +
                                              1024 * slope)) <= 1e-9 * past->seconds,
               where + ": past 1024 CTAs the last slope, marked extrapolated");
    }
    expect(ninferno::launch_row(8, 0) || !ninferno::ramp_seconds(8, 0, 1024),
           "no launch row, no ramp");
    for (const auto& [major, minor] : {std::pair{8, 0}, std::pair{9, 0}, std::pair{10, 0}}) {
        if (ninferno::isa_row(major, minor)) continue;
        expect(!ninferno::launch_seconds(major, minor, LaunchKind::GraphNode),
               "no ISA row, no launch fact");
    }

    // The Ada census's t0 (kLaunchNode, 1.5 us: node + ramp + tail of the decode kernels, without
    // PDL) against the Ada row: the graph node plus the dispatch ramp of an empty 1024-CTA grid
    // (a decode linear's grid, #850: floor_1024 - floor_1), within the event tick of the census.
    // Printed, not failed: the census stays the round cost's value until a DECISIONS entry moves
    // it.
    constexpr double kEventTick = 1.024e-6;
    const double census         = ninferno::launch_node_seconds("ada");
    const auto ada_launch       = ninferno::launch_row(8, 9);
    if (!ada_launch || !ada_launch->node_s || !ada_launch->floor_s[0] || !ada_launch->floor_s[3]) {
        std::printf("launch node: sm_89 has no launch row yet: the census's %.2f us is unchecked\n",
                    census * 1e6);
    } else {
        const double node  = *ada_launch->node_s,
                     ramp  = *ada_launch->floor_s[3] - *ada_launch->floor_s[0];
        const double table = node + ramp, off = table - census;
        std::printf("launch node: sm_89 (%s): node %.3f us + ramp of 1024 CTAs %.3f us = %.3f us "
                    "against the census's %.3f us: %s by %.3f us (the tick %.3f us)\n",
                    ada_launch->card.c_str(), node * 1e6, ramp * 1e6, table * 1e6, census * 1e6,
                    std::fabs(off) <= kEventTick ? "holds, off" : "DOES NOT HOLD, off", off * 1e6,
                    kEventTick * 1e6);
    }

    if (failures == 0) { std::printf("launch node: ok\n"); }
    return failures == 0 ? 0 : 1;
}
