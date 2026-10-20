#pragma once

// The ordinary (one token a row) decode body of the Program's families: everything around the
// family's forward pass and head is the same for every family.

#include "core/device.h"
#include "core/tensor.h"
#include "models/qwen3_5/program/context.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/scalar.h"
#include "ninferno/ops/scatter.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

namespace ninferno::models::qwen3_5::execution {

// Uploads the ingress, runs forward(core, tokens, positions, table_rows, destinations) over the
// batch's first `batch_size` rows (it returns the normed hidden), keeps that hidden for the rows'
// continuations, runs head(work, hidden, logits), samples within `public_tokens` and copies the
// egress back. `family` names the family in the out-of-range refusal.
template <class Forward, class Head>
[[nodiscard]] std::function<void()>
ordinary_decode_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                     std::int32_t public_tokens, const char* family, Forward forward, Head head) {
    return [&state, batch_size, public_tokens, family, forward, head] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error(std::string(family) + " ordinary decode batch is out of range");
        }
        ExecutionCore& core              = state.execution;
        const cudaStream_t s             = core.device.stream;
        qwen3_5::OrdinaryDecodeState& io = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(io.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   s));
        const Tensor tokens       = io.tokens.slice(0, 0, batch_size);
        const Tensor positions    = io.cache_positions.slice(0, 0, batch_size);
        const Tensor table_rows   = io.text_kv_table_rows.slice(0, 0, batch_size);
        const Tensor destinations = io.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = io.hidden.slice(1, 0, batch_size);
        Tensor logits             = io.logits.slice(1, 0, batch_size);
        Tensor sampled            = io.sampled_tokens.slice(0, 0, batch_size);

        const Tensor normed = forward(core, tokens, positions, table_rows, destinations);
        CUDA_CHECK(
            cudaMemcpyAsync(hidden.data, normed.data, hidden.bytes(), cudaMemcpyDeviceToDevice, s));
        head(core.work, hidden, logits);
        ops::scatter(hidden, destinations, state.continuation_hidden_store, s);
        ops::sample(logits, sampled, public_tokens, io.sampling, positions,
                    ops::kSamplePurposeDecode, core.work, s);
        ops::copy_to_pinned_host(&state.host_egress, io.egress.data,
                                 sizeof(qwen3_5::OrdinaryDecodeEgress), s);
    };
}

} // namespace ninferno::models::qwen3_5::execution
