// The launch node cost (#617 step B): one constant, ninferno::launch_node_seconds
// (core/cost_constants.h), and the launch-like numbers that stay separate facts. Pins every use
// site's value so a change is deliberate. No GPU.
#include "ops/generic/route_cost_constants.h"
#include "spec/round_cost.h"

#include <cstdio>
#include <string>

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

    if (failures == 0) { std::printf("launch node: ok\n"); }
    return failures == 0 ? 0 : 1;
}
