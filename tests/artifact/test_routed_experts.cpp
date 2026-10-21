// Which bindings name a routed expert, and of which MoE layer (artifact/routed_experts.h): the
// converter's expert-bank layouts key on it.

#include "artifact/routed_experts.h"

#include <iostream>

namespace {

using ninferno::artifact::routed_expert_layer;

int failures = 0;

void require(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    require(routed_expert_layer("text/layers/3/mlp/experts/7/gate_up") == "text/layers/3",
            "a routed expert matrix names its layer");
    require(routed_expert_layer("text/layers/12/moe/experts/gate_up_bias") == "text/layers/12",
            "a stacked expert bias names its layer");
    require(routed_expert_layer("mtp/layers/0/moe/experts/2/down") == "mtp/layers/0",
            "an MTP expert names its component's layer");
    require(!routed_expert_layer("text/layers/3/mlp/shared/gate_up"),
            "a shared expert is not routed");
    require(!routed_expert_layer("text/layers/3/moe/router"), "the router is not an expert");
    require(!routed_expert_layer("text/players/3/experts/1/up"), "layers must be a whole segment");
    require(!routed_expert_layer("text/layers/x/experts/1/up"), "a layer index is a number");
    require(!routed_expert_layer("text/layers/3"), "a layer is not an expert");
    if (failures != 0) { return 1; }
    std::cout << "routed experts: all checks passed\n";
    return 0;
}
