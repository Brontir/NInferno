#include "models/common/activation_census.h"

namespace ninferno::models {

namespace {
thread_local ActivationCensus* g_census = nullptr;
} // namespace

ActivationCensus::ActivationCensus() : previous_(g_census) { g_census = this; }

ActivationCensus::~ActivationCensus() { g_census = previous_; }

void census_width(std::string_view parameter, int bits, std::string_view reason) {
    if (g_census == nullptr) { return; }
    g_census->widths.insert_or_assign(std::string(parameter),
                                      CensusWidth{bits, std::string(reason)});
}

void census_policy(std::string_view parameter, ops::LinearPolicy policy) {
    if (g_census == nullptr) { return; }
    g_census->policies.insert_or_assign(std::string(parameter), policy);
}

void census_generic_moe(const ops::MoeWeights& weights, std::span<const std::string> experts,
                        std::span<const std::string> shared_gate_up,
                        std::span<const std::string> shared_down) {
    if (g_census == nullptr) { return; }
    const int bits = ops::moe_expert_activation_bits(weights);
    const char* reason =
        bits == 16 ? "the MoE Op has no eight-bit route for these experts under this policy"
                   : "the MoE Op has no four-bit expert route";
    for (const auto& name : experts) { census_width(name, bits, reason); }
    if (weights.shared_gate_up.qdata != nullptr) {
        const auto policy = ops::moe_shared_activation_policy(weights, weights.shared_gate_up);
        for (const auto& name : shared_gate_up) { census_policy(name, policy); }
    }
    if (weights.shared_down.qdata != nullptr) {
        const auto policy = ops::moe_shared_activation_policy(weights, weights.shared_down);
        for (const auto& name : shared_down) { census_policy(name, policy); }
    }
}

} // namespace ninferno::models
