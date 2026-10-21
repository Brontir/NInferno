#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "core/device_class.h"
#include "core/pdl.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninferno::ops::detail::gated_delta_net {
namespace {

static_assert(sizeof(GdnReplayFoldKernelRow) == 16);
static_assert(alignof(GdnReplayFoldKernelRow) == 16);
static_assert(sizeof(GdnReplayFoldKernelRows) == 128);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);

// Calls fn with a null pointer of the state's element type: FP32, BF16 or FP16 (GdnStateStorage).
template <class Fn>
void with_state_type(DType dtype, Fn&& fn) {
    switch (dtype) {
    case DType::FP32:
        fn(static_cast<float*>(nullptr));
        return;
    case DType::BF16:
        fn(static_cast<__nv_bfloat16*>(nullptr));
        return;
    case DType::FP16:
        fn(static_cast<__half*>(nullptr));
        return;
    default:
        throw std::invalid_argument(
            "gated_delta_net: the recurrent state must be FP32, BF16 or FP16");
    }
}

template <bool NormalizeQK, class StateT>
void launch_recurrent_direct_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& state_read, Tensor& state_write, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const auto* qp = static_cast<const __nv_bfloat16*>(q.data);
    const auto* kp = static_cast<const __nv_bfloat16*>(k.data);
    const auto* vp = static_cast<const __nv_bfloat16*>(v.data);
    const auto* gp = static_cast<const float*>(g.data);
    const auto* bp = static_cast<const float*>(beta.data);
    const auto* sr = static_cast<const StateT*>(state_read.data);
    auto* sw       = static_cast<StateT*>(state_write.data);
    auto* op       = static_cast<__nv_bfloat16*>(out.data);
    // Where the device has PDL the state loads go out under the previous kernel's tail. A call of
    // several tokens streams its inputs through shared memory; the one-token decode call does not.
    const auto kernel = q.ne[2] >= kDirectStageMinTokens
                            ? recurrent_bf16_staged_kernel<NormalizeQK, StateT>
                            : recurrent_bf16_direct_kernel<NormalizeQK, StateT>;
    CUDA_CHECK(pdl::launch({grid, block, 0, stream}, device_class().features.pdl, false, kernel,
                           qp, kp, vp, gp, bp, sr, sw, op, q.ne[2], heads, scale));
}

template <bool NormalizeInputs, class StateT>
void launch_recurrent_batch_update_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta, float scale,
                                         Tensor& ssm_states, const Tensor& source_state_slots,
                                         const Tensor& destination_state_slots, Tensor& out,
                                         cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const BatchUpdateAccess<StateT> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const StateT*>(ssm_states.data),
        static_cast<StateT*>(ssm_states.data),
        static_cast<const std::int32_t*>(source_state_slots.data),
        static_cast<const std::int32_t*>(destination_state_slots.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        state_slot_stride,
        scale,
    };
    // Where the device has PDL the state loads go out under the previous kernel's tail.
    CUDA_CHECK(pdl::launch({grid, block, 0, stream}, device_class().features.pdl, false,
                           recurrent_batch_update_kernel<NormalizeInputs, StateT>, access));
}

template <bool Masked, class StateT>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const RecordAccess<Masked, StateT> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const StateT*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    // Where the device has PDL the state loads go out under the previous kernel's tail.
    CUDA_CHECK(pdl::launch({grid, block, 0, stream}, device_class().features.pdl, false,
                           recurrent_record_kernel<Masked, StateT>, access));
}

} // namespace

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream) {
    launch_recurrent_inout(q, k, v, g, beta, scale, normalize_qk, ssm_state, ssm_state, out,
                           stream);
}

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream) {
    with_state_type(ssm_state_in.dtype, [&](auto* tag) {
        using StateT = std::remove_pointer_t<decltype(tag)>;
        if (normalize_qk) {
            launch_recurrent_direct_fixed<true, StateT>(q, k, v, g, beta, scale, ssm_state_in,
                                                        ssm_state_out, out, stream);
        } else {
            launch_recurrent_direct_fixed<false, StateT>(q, k, v, g, beta, scale, ssm_state_in,
                                                         ssm_state_out, out, stream);
        }
    });
}

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, Tensor& ssm_states,
                                   const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, Tensor& out,
                                   cudaStream_t stream) {
    with_state_type(ssm_states.dtype, [&](auto* tag) {
        using StateT = std::remove_pointer_t<decltype(tag)>;
        if (normalize_qk) {
            launch_recurrent_batch_update_fixed<true, StateT>(q, k, v, g, beta, scale, ssm_states,
                                                              source_state_slots,
                                                              destination_state_slots, out, stream);
        } else {
            launch_recurrent_batch_update_fixed<false, StateT>(
                q, k, v, g, beta, scale, ssm_states, source_state_slots, destination_state_slots,
                out, stream);
        }
    });
}

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream) {
    with_state_type(ssm_states.dtype, [&](auto* tag) {
        using StateT = std::remove_pointer_t<decltype(tag)>;
        if (valid_columns.data == nullptr) {
            launch_recurrent_record_fixed<false, StateT>(
                q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, key_record,
                value_record, gate_record, out, stream);
        } else {
            launch_recurrent_record_fixed<true, StateT>(
                q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, key_record,
                value_record, gate_record, out, stream);
        }
    });
}

template <class StateT, class Geometry>
void launch_replay_fold_with(const GdnReplayRecords& records,
                             LinearAttentionStateAllLayersView states,
                             const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                             cudaStream_t stream) {
    const FoldAccess<StateT, Geometry> access{
        static_cast<const __nv_bfloat16*>(records.key.data),
        static_cast<const __nv_bfloat16*>(records.value.data),
        reinterpret_cast<const uint2*>(records.gate.data),
        static_cast<const __nv_bfloat16*>(records.conv.data),
        static_cast<StateT*>(states.recurrent_layer0.data),
        static_cast<__nv_bfloat16*>(states.conv_layer0.data),
        states.recurrent_layer_stride_bytes / static_cast<std::int64_t>(sizeof(StateT)),
        states.conv_layer_stride_bytes / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        records.spec.record_capacity,
        records.spec.width,
        rows,
        records.spec.qk_heads,
        records.spec.value_heads,
        records.spec.conv_channels,
    };
    const dim3 grid(static_cast<unsigned>(records.spec.value_heads),
                    static_cast<unsigned>(active_rows),
                    static_cast<unsigned>(records.spec.layers * (kStateDim / kBlockDv)));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_fold_kernel<StateT, Geometry><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
bool fold_geometry_is(const GdnReplayRecords& records) {
    return records.spec.qk_heads == Geometry::kQkHeads &&
           records.spec.value_heads == Geometry::kValueHeads &&
           records.spec.conv_channels == Geometry::kConvChannels;
}

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        cudaStream_t stream) {
    // Compile-time geometries for the two Qwen 3.5 GDN shapes (27B: 16 x 48 heads, 10240 conv
    // channels; 35B-A3B: 16 x 32, 8192); any other stack folds with the runtime geometry.
    using Geometry16x48 = FoldGeometry<16, 48, 10240>;
    using Geometry16x32 = FoldGeometry<16, 32, 8192>;
    with_state_type(states.recurrent_layer0.dtype, [&](auto* tag) {
        using StateT = std::remove_pointer_t<decltype(tag)>;
        if (fold_geometry_is<Geometry16x48>(records)) {
            launch_replay_fold_with<StateT, Geometry16x48>(records, states, rows, active_rows,
                                                           stream);
        } else if (fold_geometry_is<Geometry16x32>(records)) {
            launch_replay_fold_with<StateT, Geometry16x32>(records, states, rows, active_rows,
                                                           stream);
        } else {
            launch_replay_fold_with<StateT, FoldGeometryRuntime>(records, states, rows,
                                                                 active_rows, stream);
        }
    });
}

} // namespace ninferno::ops::detail::gated_delta_net
