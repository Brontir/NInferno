#pragma once

// A MoE call's weights under MoE expert offload (core/expert_residency.h): bind the layer's
// expert bank, then read every weight through the Lease. Only the addresses change.

#include "core/expert_residency.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninferno/ops/gemma4_moe.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/sparse_moe.h"
#include "ops/host_experts/host_experts.h"

#include <cstdint>
#include <string>

namespace ninferno::models {

inline void rebase(Weight& weight, const ExpertResidency::Lease& lease) noexcept {
    weight.payload         = lease.map(weight.payload);
    weight.qdata           = lease.map(weight.qdata);
    weight.qhigh           = lease.map(weight.qhigh);
    weight.scales          = lease.map(weight.scales);
    weight.weight_divisors = lease.map(weight.weight_divisors);
    weight.k_order         = lease.map(weight.k_order);
}

inline void rebase(Tensor& tensor, const ExpertResidency::Lease& lease) noexcept {
    tensor.data = lease.map(tensor.data);
}

[[nodiscard]] inline ops::MoeWeights bound(ops::MoeWeights w,
                                           const ExpertResidency::Lease& lease) noexcept {
    for (Weight* weight : {&w.router, &w.gate_up, &w.down, &w.shared_gate_up, &w.shared_down,
                           &w.shared_gate}) {
        rebase(*weight, lease);
    }
    for (Tensor* tensor :
         {&w.router_bias, &w.selection_bias, &w.expert_scale, &w.gate_up_bias, &w.down_bias}) {
        rebase(*tensor, lease);
    }
    return w;
}

[[nodiscard]] inline ops::Gemma4MoeWeights bound(ops::Gemma4MoeWeights w,
                                                 const ExpertResidency::Lease& lease) noexcept {
    for (Weight* weight : {&w.router, &w.gate_up, &w.down}) { rebase(*weight, lease); }
    for (Tensor* tensor : {&w.router_scale, &w.expert_scale, &w.gate_up_input_divisors,
                           &w.down_input_divisors}) {
        rebase(*tensor, lease);
    }
    return w;
}

[[nodiscard]] inline ops::SparseMoeWeights bound(ops::SparseMoeWeights w,
                                                 const ExpertResidency::Lease& lease) noexcept {
    for (Weight* weight : {&w.router_shared_gate, &w.routed_gate_up, &w.routed_down,
                           &w.shared_gate_up, &w.shared_down}) {
        rebase(*weight, lease);
    }
    return w;
}

// The Lease of a MoE call over `tokens` columns that selects `top_k` of `experts` per column,
// keyed by its gate/up bank. The call reads the whole group (a staging slot) when its columns
// select at least as many experts as the layer has: on average every expert is read then. No
// residency (offload off), no Lease.
[[nodiscard]] inline ExpertResidency::Lease bind_experts(ExpertResidency* residency,
                                                         const Weight& gate_up_bank,
                                                         std::int64_t tokens, std::int64_t top_k,
                                                         std::int64_t experts,
                                                         cudaStream_t stream) {
    if (!residency) { return {}; }
    return residency->bind(gate_up_bank.payload, tokens * top_k >= experts, stream);
}

// Whether a MoE call runs its routed experts on the CPU (MoE expert offload's decode path): a CPU
// executor is attached, the layer's group is offloaded (its bytes are at their Host addresses)
// and the call is a decode call (at most 16 tokens). Otherwise the GPU reads the bank as bound.
[[nodiscard]] inline bool host_experts_call(const ops::HostExperts* host,
                                            const ExpertResidency* residency,
                                            const Weight& gate_up_bank, std::int64_t tokens) {
    return host != nullptr && residency != nullptr && tokens <= ops::HostExperts::max_tokens() &&
           residency->in_host(gate_up_bank.payload);
}

// The device view of a CPU expert call's layer: the group read in place (zero-copy), never
// streamed into a staging slot - the GPU reads only its router and selection inputs.
[[nodiscard]] inline ExpertResidency::Lease bind_for_host(ExpertResidency* residency,
                                                          const Weight& gate_up_bank,
                                                          cudaStream_t stream) {
    return residency->bind(gate_up_bank.payload, false, stream);
}

// Whether the CPU computes a layer's expert planes; `why` names the first it does not.
[[nodiscard]] inline bool host_experts_readable(const Weight& gate_up, const Weight& down,
                                                std::string* why) {
    return ops::host::host_readable(gate_up, why) && ops::host::host_readable(down, why);
}

} // namespace ninferno::models
