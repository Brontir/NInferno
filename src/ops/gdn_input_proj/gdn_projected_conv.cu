#include "ops/gdn_input_proj/gdn_projected_conv.h"

#include "core/device.h"
#include "core/device_class.h"
#include "core/pdl.cuh"
#include "ops/gdn_input_proj/gdn_conv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

// The depthwise causal convolution of the projected q/k/v rows, the channel partition read at run
// time: one thread per channel, the tokens in order. StaticWidth (0: the width argument) unrolls a
// verify-sized call's token loop.
template <int StaticWidth, class Publish>
__global__ void gdn_projected_conv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ state_read, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ initial_state_slots, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, std::int32_t width,
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows, Publish publish) {
    // The recurrence that reads q, k and v may launch now: it waits for this kernel before it
    // reads them, and loads its state tile meanwhile.
    if (threadIdx.x == 0) pdl::trigger_dependents();
    const std::int32_t channels = query_rows + key_rows + value_rows;
    const std::int32_t row      = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= channels) { return; }
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);
    if constexpr (StaticWidth != 0) { width = StaticWidth; }

    std::int32_t valid       = valid_columns == nullptr ? width : valid_columns[batch];
    valid                    = valid < 0 ? 0 : (valid > width ? width : valid);
    const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
    const std::int64_t initial_base =
        static_cast<std::int64_t>(initial_state_slots[batch]) * slot_stride;
    float s0       = __bfloat162float(state_read[initial_base + row]);
    float s1       = __bfloat162float(state_read[initial_base + channels + row]);
    float s2       = __bfloat162float(state_read[initial_base + 2LL * channels + row]);
    const float w0 = __bfloat162float(conv_weight[row]);
    const float w1 = __bfloat162float(conv_weight[channels + row]);
    const float w2 = __bfloat162float(conv_weight[2LL * channels + row]);
    const float w3 = __bfloat162float(conv_weight[3LL * channels + row]);

    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
        float result              = 0.0F;
        float p                   = 0.0F;
        if (token < valid) {
            p          = __bfloat162float(projected[column * channels + row]);
            float conv = fmaf(w0, s0, 0.0F);
            conv       = fmaf(w1, s1, conv);
            conv       = fmaf(w2, s2, conv);
            conv       = fmaf(w3, p, conv);
            result     = silu(conv);
        }
        const __nv_bfloat16 output = __float2bfloat16_rn(result);
        if (row < query_rows) {
            query[column * query_rows + row] = output;
        } else if (row < query_rows + key_rows) {
            key[column * key_rows + row - query_rows] = output;
        } else {
            value[column * value_rows + row - query_rows - key_rows] = output;
        }
        if (token >= valid) { continue; }
        publish.publish(token, batch, row, s1, s2, p);
        s0 = s1;
        s1 = s2;
        s2 = p;
    }
}

template <int StaticWidth, class Publish>
void launch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
            const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
            Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    // A grid of 256-thread blocks that does not cover the SMs once (one sequence: the channels
    // over 256) takes 64-thread blocks: four times the blocks, each thread's work unchanged. The
    // compiled Qwen3.6 35B-A3B instance this replaced took them at T = 4 (40 blocks of 256 against
    // 170 SMs, 160 of 64 measured faster); the rule extends that to every width and geometry (r64).
    constexpr int kThreads      = 256;
    constexpr int kNarrowThreads = 64;
    const std::int32_t channels = projected.ne[0];
    const std::int64_t batch    = projected.ne[2];
    const std::int64_t wide_blocks = (channels + kThreads - 1) / kThreads * batch;
    const int threads = wide_blocks < device_class().sms ? kNarrowThreads : kThreads;
    const dim3 grid((channels + threads - 1) / threads, static_cast<unsigned>(batch));
    gdn_projected_conv_kernel<StaticWidth><<<grid, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(state_read.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data), projected.ne[1], query.ne[0], key.ne[0],
        value.ne[0], publish);
    CUDA_CHECK(cudaGetLastError());
}

template <class Publish>
void dispatch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
              const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
              Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    if (query.ne[0] <= 0 || key.ne[0] <= 0 || value.ne[0] <= 0 ||
        projected.ne[0] != query.ne[0] + key.ne[0] + value.ne[0]) {
        throw std::invalid_argument("GDN projected-conv: q/k/v rows must partition the channels");
    }
    if (projected.ne[1] == 4) {
        launch<4>(projected, conv_weight, state_read, valid_columns, initial_state_slots, query,
                  key, value, publish, stream);
    } else {
        launch<0>(projected, conv_weight, state_read, valid_columns, initial_state_slots, query,
                  key, value, publish, stream);
    }
}

} // namespace

void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream) {
    dispatch(projected, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value,
             SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                    static_cast<const std::int32_t*>(snapshot_base_slots.data),
                                    projected.ne[0]},
             stream);
}

void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream) {
    dispatch(conv_record, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value, NoHistoryPublish{}, stream);
}

} // namespace ninferno::ops::detail
