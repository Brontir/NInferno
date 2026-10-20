#pragma once

// One text prefill call of the Program's ring and hybrid families (dense, Gemma 4, Nemotron-H):
// what it covers of the prompt, its token ids and positions, and what follows the forward at a
// split frontier and on the prompt's last call.

#include "core/device.h"
#include "core/tensor.h"
#include "models/qwen3_5/program/context.h"
#include "ninferno/ops/argmax.h"
#include "ninferno/ops/position.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/scalar.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace ninferno::models::qwen3_5::execution {

struct PrefillTextCall {
    std::uint32_t base = 0; // the prompt's first token this call carries
    std::uint32_t len  = 0; // tokens it carries
    std::int32_t t     = 0; // len as a tensor width
    bool is_last       = false;
};

// This call: at most one chunk, ending on the split frontier when it falls inside it, and never
// leaving a narrow call before the frontier when it falls beyond (call_tokens(chunk, cursor,
// frontier), the family's prefill_call_tokens). `family` names the model in the refusal of a
// chunk outside the prompt.
template <class CallTokens>
[[nodiscard]] PrefillTextCall
prefill_text_call(const PrefillContext& state, std::span<const TokenId> ids,
                  std::uint32_t nominal_length, std::optional<std::uint32_t> split_frontier,
                  bool finalize_at_end, const char* family, CallTokens call_tokens) {
    const std::uint32_t base = state.text_kv_base;
    if (nominal_length == 0 || base >= ids.size() || nominal_length > ids.size() - base ||
        static_cast<std::uint64_t>(base) + nominal_length >
            static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(family) + " prefill chunk is outside the prompt");
    }
    std::uint32_t len = std::min(nominal_length, state.execution.prefill_chunk);
    if (split_frontier && *split_frontier > base) {
        len = *split_frontier <= base + len ? *split_frontier - base
                                            : call_tokens(len, base, *split_frontier);
    }
    return {.base    = base,
            .len     = len,
            .t       = static_cast<std::int32_t>(len),
            .is_last = finalize_at_end && len == nominal_length};
}

struct PrefillTextInputs {
    Tensor ids;
    Tensor positions;
};

// The call's token ids (uploaded from the prompt) and absolute positions, in the work arena.
[[nodiscard]] inline PrefillTextInputs upload_prefill_text(ExecutionCore& core,
                                                           std::span<const TokenId> ids,
                                                           const PrefillTextCall& call) {
    const cudaStream_t s = core.device.stream;
    PrefillTextInputs out{.ids       = core.work.alloc(DType::I32, {call.t}),
                          .positions = core.work.alloc(DType::I32, {call.t})};
    CUDA_CHECK(cudaMemcpyAsync(out.ids.data, ids.data() + call.base, sizeof(TokenId) * call.len,
                               cudaMemcpyHostToDevice, s));
    ops::fill_i32_positions(out.positions, static_cast<std::int32_t>(call.base), s);
    return out;
}

// After the forward, `last` being the call's last normed hidden column: at a split frontier the
// checkpoint keeps it; on the prompt's last call head(work, last, logits) and the first draw
// (sampled, or the argmax) within `public_tokens`. The drawn token is keyed by its own absolute
// position, as every decode draw is.
template <class Head>
void finish_prefill_text(PrefillContext& state, const PrefillTextCall& call,
                         std::optional<std::uint32_t> split_frontier, const Tensor& last,
                         std::int32_t public_tokens, Head head) {
    ExecutionCore& core  = state.execution;
    const cudaStream_t s = core.device.stream;
    if (split_frontier && *split_frontier == call.base + call.len &&
        state.rewrite_checkpoint_hidden != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(state.rewrite_checkpoint_hidden->data, last.data, last.bytes(),
                                   cudaMemcpyDeviceToDevice, s));
    }
    if (!call.is_last) { return; }
    auto scope    = core.work.scope();
    Tensor logits = core.io.logits.slice(1, 0, 1);
    head(core.work, last, logits);
    const auto next = static_cast<std::int32_t>(call.base + call.len);
    ops::set_i32_scalar(core.io.pos, next, s);
    ops::set_i32_scalar(core.io.rope_pos, next, s);
    if (state.sampling != nullptr) {
        ops::sample(logits, core.io.token, public_tokens, state.sampling, core.io.pos,
                    ops::kSamplePurposePrefill, core.work, s);
    } else {
        ops::argmax(logits, core.io.token, public_tokens, s);
    }
}

} // namespace ninferno::models::qwen3_5::execution
