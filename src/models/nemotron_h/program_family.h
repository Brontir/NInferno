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
                       std::int32_t public_tokens, ActivationMode activations);

} // namespace ninferno::models::nemotron_h
