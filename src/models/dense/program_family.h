#pragma once

// The dense decoder (model_type "dense_decoder": the Llama lineage) as a family of the one
// Program: its full layers keep their keys and values in the Program's paged pool, its sliding
// layers (if any) keep theirs in rings inside each sequence's state image, and its prefill and
// decode are the family's forward pass over those bindings. The Program owns every mechanism
// around them (admission, pages, state images, graphs, sampling state).
//
// With a DFlash drafter (the published block-diffusion assistant of a dense target, loaded as its
// own artifact) the family also runs the DFlash speculative backend: the target's forward taps the
// drafter's feature layers, the drafter keeps its own keys and values in fixed cyclic rings of
// each sequence's state image, and models/dense/dflash.cpp drafts a whole block per round and
// verifies it through the target.

#include "core/device.h"
#include "models/dense/model.h"
#include "models/qwen3_5/program/model_family.h"
#include "models/speculation/dflash.h"
#include "ninferno/types.h"

#include <cstdint>
#include <memory>

namespace ninferno::models::dense {

// The execution of `model` on `device`, over a public token domain of `public_tokens` ids (the
// tokenizer's, at most the embedding rows). With a `drafter` (a DFlash artifact paired with this
// target) it also runs the DFlash speculative backend. Both must outlive it.
// Under ActivationMode::Exact every prefill, decode and capture runs under the invariant route
// profile (ninferno/ops/prefill_routes.h), so a token's bytes do not depend on its batch neighbours
// or on where its prompt was cut; load the model with ActivationProfile::A16 to go with it.
[[nodiscard]] std::unique_ptr<const qwen3_5::detail::ModelExecution>
make_program_execution(const Model& model, const speculation::DFlashDrafter* drafter,
                       DeviceContext& device, std::int32_t public_tokens,
                       ActivationMode activations);

} // namespace ninferno::models::dense
