#pragma once

// The Gemma 4 target as a family of the one Program: its full layers keep their keys and values in
// the Program's paged pool, its sliding layers keep theirs in rings inside each sequence's state
// image, and its prefill and decode are the target's forward pass over those bindings. The Program
// owns every mechanism around them (admission, pages, state images, graphs, sampling state).

#include "core/device.h"
#include "models/gemma4/model.h"
#include "models/qwen3_5/program/model_family.h"
#include "ninferno/types.h"

#include <cstdint>
#include <memory>

namespace ninferno::models::gemma4 {

// The deepest MTP round the Gemma 4 Program runs: the assistant drafts one token per step, and the
// target verifies k + 1 columns at decode-width arithmetic (at most 16 columns, as the CLI's
// kMaxVerifyColumns; the expert bank's BF16 route covers T <= 16 under every profile).
inline constexpr std::uint32_t kMaximumMtpDraftTokens = 15;

// The execution of `model` on `device`, over a public token domain of `public_tokens` ids (the
// tokenizer's, at most the embedding rows). With a `drafter` (Google's MTP assistant for this
// target) it also runs the MTP speculative backend. Both must outlive it.
// Under ActivationMode::Bits16 every prefill, decode, verify and capture runs under the invariant
// route profile (ninferno/ops/prefill_routes.h), a prefill under the prefill profile too; load the
// model with ActivationProfile::A16Full to go with it.
[[nodiscard]] std::unique_ptr<const qwen3_5::detail::ModelExecution>
make_program_execution(const Model& model, const Drafter* drafter, DeviceContext& device,
                       std::int32_t public_tokens, ActivationMode activations);

} // namespace ninferno::models::gemma4
