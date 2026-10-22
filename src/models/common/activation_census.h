#pragma once

// What a load hands to an expert Op, per parameter: the input of --activations' startup resolution
// (runtime/engine/activation_precision.h) for the matrices the generic linear path does not run.
// A routed expert's width is the Op's (the generic MoE's moe_expert_activation_bits, a fused
// bank's own route), a shared expert's policy the one the MoE Op gives its projections; every
// other matrix runs on the linear path under its effective_policy, which the resolution reads
// from the artifact itself.
//
// The census is a recorder of this thread, active while an ActivationCensus lives (the engine
// opens one around a family's load); the loaders record into it, and record nothing when none is
// open (a CLI or a test that loads a model alone).

#include "ninferno/ops/linear.h"
#include "ninferno/ops/moe.h"

#include <map>
#include <span>
#include <string>
#include <string_view>

namespace ninferno::models {

struct CensusWidth {
    int bits = 16;      // 4, 8 or 16: the narrowest width any call of the matrix computes with
    std::string reason; // why, for the startup log ("" when the Op's width is the policy's)
};

class ActivationCensus {
public:
    ActivationCensus(); // this thread's census until destroyed (the previous one restored)
    ~ActivationCensus();
    ActivationCensus(const ActivationCensus&)            = delete;
    ActivationCensus& operator=(const ActivationCensus&) = delete;

    std::map<std::string, CensusWidth, std::less<>> widths;         // by parameter name
    std::map<std::string, ops::LinearPolicy, std::less<>> policies; // by parameter name

private:
    ActivationCensus* previous_;
};

// Records into this thread's census, if one is open.
void census_width(std::string_view parameter, int bits, std::string_view reason);
void census_policy(std::string_view parameter, ops::LinearPolicy policy);

// A generic MoE layer as the Op runs it: `experts` (every routed-expert matrix, both banks) at
// moe_expert_activation_bits, `shared_gate_up` and `shared_down` (the shared expert's matrices,
// either span empty when absent) under moe_shared_activation_policy.
void census_generic_moe(const ops::MoeWeights& weights, std::span<const std::string> experts,
                        std::span<const std::string> shared_gate_up,
                        std::span<const std::string> shared_down);

} // namespace ninferno::models
