#pragma once

// NemotronH (model_type "nemotron_h": Mamba2 / attention / relu^2 MoE blocks) as a family of the
// one Program: attention keys and values in the paged pool, Mamba2 state in the linear state of
// each sequence's state image, prefill and decode the family's forward pass over those bindings.

#include "core/device.h"
#include "models/nemotron_h/model.h"
#include "models/qwen3_5/program/model_family.h"
#include "ninferno/types.h"

#include <cstdint>
#include <memory>

namespace ninferno::models::nemotron_h {

// The deepest MTP window: the head's autoregressive chain drafts at most this many tokens a round
// (the verify's record width is one more).
inline constexpr std::uint32_t kMaximumMtpDraftTokens = 7;

// `mtp`: the MTP head (null: no speculation); it outlives the execution.
[[nodiscard]] std::unique_ptr<const qwen3_5::detail::ModelExecution>
make_program_execution(const Model& model, const MtpWeights* mtp, DeviceContext& device,
                       std::int32_t public_tokens, ActivationMode activations,
                       EnginePurpose purpose);

// The shape of the persistent state the NemotronH `config` lays out
// (models/qwen3_5/program/planning/sequence_bytes.h, plan_persistent_layout): the attention
// layers' pool and the MTP head's, the Mamba2 states; `public_tokens` the ids a request may carry
// or sample.
[[nodiscard]] StateShape state_shape(const Config& config, std::int32_t public_tokens);

// The arenas the target's own calls size for `plan` (the execution's workspace plan without the
// MTP head): the text prefill (under the prefill route profile), the ordinary round, and under
// causal scoring the scoring tiles; their widest is the capacity. `public_tokens` the sampler's
// domain. The start's execution sizes its arena with this, the head's parts added; the plan
// without the weights calls it over a described model (Model::describe, #615). The caller sets
// the activations' route profile (ops::ScopedInvariantRoutes under exact).
[[nodiscard]] qwen3_5::detail::WorkspacePlan
target_workspace_plan(const Model& model, std::int32_t public_tokens,
                      const qwen3_5::detail::SequencePlanImpl& plan);

} // namespace ninferno::models::nemotron_h
