#pragma once

// --activations auto|16|8|4 resolved at startup (ninferno/types.h ActivationMode). A B1 decision
// (docs/maintainer/deterministic-decisions.md): it reads the option, the artifact's directory
// (each matrix's format, shape, recorded policy, name and calibrated activation divisor), the
// device's facts (ops::ActivationDevice: discrete features and the generation row) and, after
// the load, what the loaders handed to the expert Ops (models::ActivationCensus), nothing else,
// and runs on a CPU.
//
// Every matrix use of the target ("text/...") and of its in-artifact drafter ("mtp/", "dflash/",
// "dflash2/", "proposal/") with a recorded policy counts at the narrowest width a call of it
// computes with:
// - a routed or shared expert the census names: the expert Op's width (the generic MoE's
//   moe_expert_activation_bits, a fused bank's own route) or the policy the MoE Op gives it;
// - every other matrix: the generic linear path's admission under models::effective_policy, the
//   policy the loaders give it, with the int8 prefill of A16Only G64 weights where it pays
//   (ops::linear_activation_width, the same functions the routes select by).
// Before the load (no census) a routed expert counts as the linear path would take it; the
// resolution after the load replaces those counts with the Ops' own.
//
// What a request the artifact or the card cannot take does (principle 7, the unknown is
// conservative and labelled):
// - 8 or 4 that no matrix can take to that width is refused (`refuse`, before the load): it would
//   change nothing but the label (8 on BF16 weights; 4 without calibrated NVFP4 weights on an
//   sm_120a build);
// - 8 or 4 that only some matrices can take starts; the others stay at the next wider width they
//   have, and the resolution names them by format and reason (kept_wider), which the server
//   prints as a warning.
// Auto and 16 are never refused.
//
// The notes: under Auto, whether --activations 8 runs some matrices narrower than Auto does (an
// NVFP4 checkpoint recorded for four bits without calibrated divisors keeps BF16 under Auto on
// sm_120a, where 8 takes W4A8); for 8 and 4, whether the mode's policies equal Auto's
// (same_as_auto: the same kernels).

#include "artifact/schema.h"
#include "models/common/activation_census.h"
#include "ninferno/types.h"
#include "ops/linear/activation_width.h"

namespace ninferno::runtime {

[[nodiscard]] ActivationPrecision
resolve_activation_precision(const artifact::Directory& directory, ActivationMode mode,
                             const ops::ActivationDevice& device,
                             const models::ActivationCensus* census = nullptr, bool refuse = true);

// The four modes' resolutions before a load (the UI's preflight, ninferno-serve
// --activations-preflight): each mode's ActivationPrecision, or the refusal's text. In the order
// auto, 16, 8, 4.
struct ModePreflight {
    ActivationMode mode = ActivationMode::Auto;
    bool refused        = false;
    std::string refusal;
    ActivationPrecision precision;
};

[[nodiscard]] std::vector<ModePreflight>
preflight_activation_modes(const artifact::Directory& directory,
                           const ops::ActivationDevice& device);

// Whether the directory has routed-expert matrices (names with "/experts/"), whose counts before a
// load are the linear path's and settle at the load.
[[nodiscard]] bool has_routed_experts(const artifact::Directory& directory);

} // namespace ninferno::runtime
