#pragma once

// The Inkling 2 model as a family of the one Program: each visit slot of the looped schedule keeps
// its keys and values in the Program's paged pool (a full visit) or in a ring inside each
// sequence's state image (a sliding visit), and its convolution history in the state image's
// linear planes; its prefill and decode are Pass over those bindings. The Program owns every
// mechanism around them (admission, pages, state images, forks, graphs, sampling state).

#include "core/device.h"
#include "models/inkling2/model.h"
#include "models/qwen3_5/program/model_family.h"
#include "ninferno/types.h"

#include <cstdint>
#include <memory>

namespace ninferno::models::inkling2 {

// The execution of `model` on `device`, over a public token domain of `public_tokens` ids (the
// tokenizer's, at most the head's rows). The model must outlive it. No speculative backend.
[[nodiscard]] std::unique_ptr<const qwen3_5::detail::ModelExecution>
make_program_execution(const Model& model, DeviceContext& device, std::int32_t public_tokens,
                       ActivationMode activations, EnginePurpose purpose);

// The shape of the persistent state `config` lays out (models/qwen3_5/program/planning/
// sequence_bytes.h, plan_persistent_layout): the full visit slots' pool, the sliding visit slots'
// rings, every visit slot's convolution history.
[[nodiscard]] StateShape state_shape(const Config& config, std::int32_t public_tokens);

// The arena the start's execution sizes for `plan` (Inkling2Execution::workspace_plan): a prefill
// chunk's pass, a decode round's pass over the plan's concurrency, each with the sampler of the
// rows it draws for, and under causal scoring the scoring tiles; their widest is the capacity.
// `public_tokens` the sampler's domain. The plan without the weights calls it over a described
// model (Model::describe, #615). The caller sets the activations' route profile
// (ops::ScopedInvariantRoutes under exact).
[[nodiscard]] qwen3_5::detail::WorkspacePlan
target_workspace_plan(const Model& model, std::int32_t public_tokens,
                      const qwen3_5::detail::SequencePlanImpl& plan);

} // namespace ninferno::models::inkling2
