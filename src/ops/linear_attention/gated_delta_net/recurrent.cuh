#pragma once

#include "core/pdl.cuh"
#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear_attention/gated_delta_net/common.cuh"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninferno::ops::detail::gated_delta_net {

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / kWarpSize;

static_assert(kStateDim % kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const float* base,
                                             std::uint32_t dqk_base) {
    store_vec(reg, load_vec<float4>(base + dqk_base));
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], float* base,
                                              std::uint32_t dqk_base) {
    store_vec(base + dqk_base, load_vec<float4>(reg));
}

// A BF16 state (GdnStateStorage::BF16) widens on load and rounds once on store: the recurrence
// itself runs in FP32 registers, so the rounding does not feed back within one launch.
__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const __nv_bfloat16* base,
                                             std::uint32_t dqk_base) {
    const Bf16x4Pack pack = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo       = bf16x2_to_float2(pack.pair[0]);
    const float2 hi       = bf16x2_to_float2(pack.pair[1]);
    reg[0]                = lo.x;
    reg[1]                = lo.y;
    reg[2]                = hi.x;
    reg[3]                = hi.y;
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], __nv_bfloat16* base,
                                              std::uint32_t dqk_base) {
    Bf16x4Pack pack;
    pack.pair[0] = __floats2bfloat162_rn(reg[0], reg[1]);
    pack.pair[1] = __floats2bfloat162_rn(reg[2], reg[3]);
    store_vec(base + dqk_base, pack);
}

// An FP16 state (GdnStateStorage::FP16): as BF16, with an 11-bit significand.
__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const __half* base,
                                             std::uint32_t dqk_base) {
    const uint2 bits = load_vec<uint2>(base + dqk_base);
    const float2 lo  = __half22float2(*reinterpret_cast<const __half2*>(&bits.x));
    const float2 hi  = __half22float2(*reinterpret_cast<const __half2*>(&bits.y));
    reg[0]           = lo.x;
    reg[1]           = lo.y;
    reg[2]           = hi.x;
    reg[3]           = hi.y;
}

// FP16 saturates at its largest finite value instead of rounding to infinity: a state entry
// past 65504 (none seen: the decayed state stays within about V/sqrt(1-a^2)) loses its tail,
// not the whole head.
// NaN passes as NaN (fmaxf would turn it into -65504): a poisoned update stays visible to the
// numeric checks instead of becoming a large finite state the sequence carries on.
__device__ __forceinline__ float fp16_saturate(float v) {
    return v != v ? v : fminf(fmaxf(v, -65504.0f), 65504.0f);
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], __half* base,
                                              std::uint32_t dqk_base) {
    const __half2 lo = __floats2half2_rn(fp16_saturate(reg[0]), fp16_saturate(reg[1]));
    const __half2 hi = __floats2half2_rn(fp16_saturate(reg[2]), fp16_saturate(reg[3]));
    store_vec(base + dqk_base, make_uint2(*reinterpret_cast<const unsigned*>(&lo),
                                          *reinterpret_cast<const unsigned*>(&hi)));
}

inline constexpr float kQkL2NormEps = 1.0e-6f;

struct RawQkLane {
    Bf16x4Pack bits;
    float value[kQkPerLane];
};

struct RawValueLane {
    __nv_bfloat16 bits;
    float value;
};

struct RawGatePair {
    uint2 bits;
    float g;
    float beta;
};

__device__ __forceinline__ RawQkLane load_raw_qk_lane(const __nv_bfloat16* base,
                                                      std::uint32_t dqk_base) {
    RawQkLane out;
    out.bits        = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo = bf16x2_to_float2(out.bits.pair[0]);
    const float2 hi = bf16x2_to_float2(out.bits.pair[1]);
    out.value[0]    = lo.x;
    out.value[1]    = lo.y;
    out.value[2]    = hi.x;
    out.value[3]    = hi.y;
    return out;
}

// Sums every entry of `value` across the warp in one butterfly: the N independent shuffles of a
// step hide one another's latency, and every lane ends with the totals.
template <int N>
__device__ __forceinline__ void warp_sum_all(float (&value)[N]) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int i = 0; i < N; ++i) { value[i] += __shfl_xor_sync(kFullWarpMask, value[i], offset); }
    }
}

// The row products of the state with a raw q or k vector and, when Normalize, the vector's sum of
// squares, reduced together. The L2 normalization of q/k is applied afterwards as one scalar on
// the reduced sums (FlashInfer gdn_decode_mtp: no separate reduce-and-broadcast of the norm).
template <bool Normalize>
__device__ __forceinline__ float
reduce_rows_and_norm(const float (&state)[kDvPerWarp][kQkPerLane], const float (&vector)[kQkPerLane],
                     float (&rows)[kDvPerWarp]) {
    // The last entry is the sum of squares; without Normalize it stays zero and is not reduced.
    constexpr int kSums = kDvPerWarp + (Normalize ? 1 : 0);
    float sums[kSums];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * vector[c]; }
        sums[r] = partial;
    }
    if constexpr (Normalize) {
        float squares = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { squares += vector[c] * vector[c]; }
        sums[kDvPerWarp] = squares;
    }
    warp_sum_all(sums);
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) { rows[r] = sums[r]; }
    if constexpr (Normalize) { return rsqrtf(sums[kDvPerWarp] + kQkL2NormEps); }
    return 1.0f;
}

__device__ __forceinline__ RawValueLane load_value_lane(const __nv_bfloat16* base, int lane,
                                                        std::uint32_t dv_base) {
    RawValueLane out{__float2bfloat16(0.0f), 0.0f};
    if (lane < kDvPerWarp) {
        out.bits  = base[dv_base + lane];
        out.value = __bfloat162float(out.bits);
    }
    return out;
}

__device__ __forceinline__ RawGatePair load_source_gate(const float* g, const float* beta,
                                                        std::int64_t offset) {
    const float g_value    = g[offset];
    const float beta_value = beta[offset];
    return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value, beta_value};
}

__device__ __forceinline__ RawGatePair load_record_gate(const uint2* gate, std::int64_t offset) {
    const uint2 bits = load_vec<uint2>(gate + offset);
    return {bits, __uint_as_float(bits.x), __uint_as_float(bits.y)};
}

// `key` is raw when Normalize: the unit key is key * inv, folded into the reduced products and
// into the rank-one step.
template <bool Normalize>
__device__ __forceinline__ void apply_gdn_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                     const float (&key)[kQkPerLane], float v_local,
                                                     float g, float beta) {
    const float alpha = expf(g);
    float partial[kDvPerWarp];
    const float inv = reduce_rows_and_norm<Normalize>(state, key, partial);

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
        const float delta = beta * (v_r - alpha * (partial[r] * inv));
        const float step  = delta * inv;

#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = alpha * state[r][c] + step * key[c]; }
    }
}

template <bool Normalize>
__device__ __forceinline__ void readout_and_store(float (&state)[kDvPerWarp][kQkPerLane],
                                                  const __nv_bfloat16* query, __nv_bfloat16* output,
                                                  std::uint32_t dqk_base, std::uint32_t dv_base,
                                                  int lane, float scale) {
    const RawQkLane q = load_raw_qk_lane(query, dqk_base);
    float partial[kDvPerWarp];
    const float factor = reduce_rows_and_norm<Normalize>(state, q.value, partial) * scale;

    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        if (lane == r) { attn_val = partial[r]; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(attn_val * factor); }
}

struct RecurrentCoordinates {
    int lane;
    int warp;
    std::int32_t batch;
    std::int32_t layer;
    std::int32_t state_tile;
    std::uint32_t value_head;
    std::uint32_t qk_head;
    std::uint32_t dv_base;
    std::uint32_t dqk_base;
};

__device__ __forceinline__ RecurrentCoordinates make_coordinates(std::int32_t batch,
                                                                 std::int32_t layer,
                                                                 std::int32_t state_tile,
                                                                 head_map heads) {
    const int lane                 = threadIdx.x;
    const int warp                 = threadIdx.y;
    const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t qk_head =
        static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(value_head)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
    return {lane,    warp,       batch,
            layer,   state_tile, value_head,
            qk_head, dv_base,    static_cast<std::uint32_t>(lane * kQkPerLane)};
}

struct OutputEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access&, const RecurrentCoordinates&,
                                                       std::int32_t, const RawQkLane&) {}

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access&, const RecurrentCoordinates&, std::int32_t,
                       const RawValueLane&, const RawGatePair&) {}

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                   const RecurrentCoordinates& coord, std::int32_t token) {
        readout_and_store<NormalizeInputs>(state, access.query_ptr(coord, token),
                                           access.output_ptr(coord, token), coord.dqk_base,
                                           coord.dv_base, coord.lane, access.scale);
    }
};

struct RecordEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access& access,
                                                       const RecurrentCoordinates& coord,
                                                       std::int32_t token, const RawQkLane& key) {
        access.store_key(coord, token, key);
    }

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access& access, const RecurrentCoordinates& coord, std::int32_t token,
                       const RawValueLane& value, const RawGatePair& gate) {
        access.store_value(coord, token, value);
        access.store_gate(coord, token, gate);
    }

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                   const RecurrentCoordinates& coord, std::int32_t token) {
        OutputEffects::publish_output<NormalizeInputs>(state, access, coord, token);
    }
};

struct FoldEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access&, const RecurrentCoordinates&,
                                                       std::int32_t, const RawQkLane&) {}

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access&, const RecurrentCoordinates&, std::int32_t,
                       const RawValueLane&, const RawGatePair&) {}

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&)[kDvPerWarp][kQkPerLane], const Access&, const RecurrentCoordinates&,
                   std::int32_t) {}
};

template <class StateT>
struct DirectAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const StateT* state_read;
    StateT* state_write;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(0, 0, static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        (void)coord;
        return token;
    }

    __device__ __forceinline__ const StateT*
    state_read_base(const RecurrentCoordinates& coord) const {
        return state_read + static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ StateT* state_write_base(const RecurrentCoordinates& coord) const {
        return state_write + static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }
};

template <class StateT>
struct BatchUpdateAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const StateT* states_read;
    StateT* states_write;
    const std::int32_t* source_state_slots;
    const std::int32_t* destination_state_slots;
    __nv_bfloat16* out;
    head_map heads;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        (void)token;
        return coord.batch;
    }

    __device__ __forceinline__ const StateT*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states_read +
               static_cast<std::int64_t>(source_state_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ StateT* state_write_base(const RecurrentCoordinates& coord) const {
        return states_write +
               static_cast<std::int64_t>(destination_state_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }
};

template <bool Masked, class StateT>
struct RecordAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const StateT* states;
    const std::int32_t* valid_columns;
    const std::int32_t* initial_slots;
    __nv_bfloat16* key_record;
    __nv_bfloat16* value_record;
    uint2* gate_record;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ const StateT*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states + static_cast<std::int64_t>(initial_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ void store_key(const RecurrentCoordinates& coord, std::int32_t token,
                                              const RawQkLane& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 &&
            static_cast<int>(coord.value_head) % heads.group_size() == 0) {
            __nv_bfloat16* destination =
                key_record + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
            store_vec(destination + coord.dqk_base, raw.bits);
        }
    }

    __device__ __forceinline__ void store_value(const RecurrentCoordinates& coord,
                                                std::int32_t token, const RawValueLane& raw) const {
        if (coord.lane < kDvPerWarp) {
            __nv_bfloat16* destination =
                value_record + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
            destination[coord.dv_base + coord.lane] = raw.bits;
        }
    }

    __device__ __forceinline__ void store_gate(const RecurrentCoordinates& coord,
                                               std::int32_t token, const RawGatePair& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 && coord.lane == 0) {
            gate_record[column(coord, token) * heads.H_v + coord.value_head] = raw.bits;
        }
    }
};

// A record window's keys, queries, this CTA's value rows and gates, staged in shared memory before
// the recurrence (as ninfer-all does). run_recurrent_sequence loads each token's inputs behind the
// previous token's record and output stores, which the compiler may not reorder across, so every
// token of a verify window paid a full load latency. Wider windows (single-row copy rounds of up to
// 64 columns) read global memory as before.
inline constexpr int kRecordStageTokens = 16;

struct RecordStage {
    __align__(16) __nv_bfloat16 key[kRecordStageTokens][kStateDim];
    __align__(16) __nv_bfloat16 query[kRecordStageTokens][kStateDim];
    __align__(16) __nv_bfloat16 value[kRecordStageTokens][kBlockDv];
    float g[kRecordStageTokens];
    float beta[kRecordStageTokens];
};

// RecordAccess with the window's inputs read from the stage; records and outputs still go to
// global memory, so the sequence runs the same operations on the same values.
template <bool Masked, class StateT>
struct StagedRecordAccess : RecordAccess<Masked, StateT> {
    const RecordStage* stage;

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates&,
                                                            std::int32_t token) const {
        return stage->key[token];
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates&,
                                                              std::int32_t token) const {
        return stage->query[token];
    }

    // load_value_lane indexes the row at dv_base + lane; the stage holds this CTA's state tile.
    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return stage->value[token] - coord.state_tile * kBlockDv;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates&,
                                                     std::int32_t token) const {
        const float g_value    = stage->g[token];
        const float beta_value = stage->beta[token];
        return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value,
                beta_value};
    }
};

template <bool Masked, class StateT>
__device__ __forceinline__ void stage_record_window(RecordStage& stage,
                                                    const RecordAccess<Masked, StateT>& access,
                                                    const RecurrentCoordinates& coord,
                                                    std::int32_t count) {
    const int tid              = coord.warp * kWarpSize + coord.lane;
    constexpr int kThreads     = kWarpSize * kNumWarps;
    constexpr int kRowChunks   = kStateDim * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
    constexpr int kValueChunks = kBlockDv * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
    for (int i = tid; i < count * kRowChunks; i += kThreads) {
        const int token = i / kRowChunks;
        const int chunk = (i - token * kRowChunks) * 8;
        cp_async<16>(&stage.key[token][chunk], access.key_ptr(coord, token) + chunk);
        cp_async<16>(&stage.query[token][chunk], access.query_ptr(coord, token) + chunk);
    }
    for (int i = tid; i < count * kValueChunks; i += kThreads) {
        const int token = i / kValueChunks;
        const int chunk = (i - token * kValueChunks) * 8;
        cp_async<16>(&stage.value[token][chunk],
                     access.value_ptr(coord, token) + coord.state_tile * kBlockDv + chunk);
    }
    if (tid < count) {
        const RawGatePair gate = access.load_gate(coord, tid);
        stage.g[tid]           = gate.g;
        stage.beta[tid]        = gate.beta;
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
}

// A direct call of two tokens or more (an `exact` prefill, or a short fast-mode call outside
// prefill) streams its inputs through two shared-memory windows of
// kDirectStageTokens tokens: the next window's q, k, this CTA's value rows and the gates are copied
// with cp.async while the current one runs. run_recurrent_sequence otherwise loads each token's
// value, gates and query behind the previous token's output store, so every token paid a global
// load latency. The sequence runs the same operations on the same values, so the bytes equal the
// unstaged kernel's, decode's, verify's and replay's.
inline constexpr int kDirectStageTokens = 16;

struct DirectStage {
    __align__(16) __nv_bfloat16 key[kDirectStageTokens][kStateDim];
    __align__(16) __nv_bfloat16 query[kDirectStageTokens][kStateDim];
    __align__(16) __nv_bfloat16 value[kDirectStageTokens][kBlockDv];
    float g[kDirectStageTokens];
    float beta[kDirectStageTokens];
};

// DirectAccess over one window starting at `first`: inputs from the stage, outputs to global.
template <class StateT>
struct StagedDirectAccess : DirectAccess<StateT> {
    const DirectStage* stage;
    std::int32_t first;

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates&,
                                                            std::int32_t token) const {
        return stage->key[token];
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates&,
                                                              std::int32_t token) const {
        return stage->query[token];
    }

    // load_value_lane indexes the row at dv_base + lane; the stage holds this CTA's state tile.
    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return stage->value[token] - coord.state_tile * kBlockDv;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates&,
                                                     std::int32_t token) const {
        const float g_value    = stage->g[token];
        const float beta_value = stage->beta[token];
        return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value,
                beta_value};
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return DirectAccess<StateT>::output_ptr(coord, first + token);
    }
};

// Issues (does not wait for) the copies of tokens [first, first + count) into `stage`.
template <class StateT>
__device__ __forceinline__ void issue_direct_window(DirectStage& stage,
                                                    const DirectAccess<StateT>& access,
                                                    const RecurrentCoordinates& coord,
                                                    std::int32_t first, std::int32_t count) {
    const int tid              = coord.warp * kWarpSize + coord.lane;
    constexpr int kThreads     = kWarpSize * kNumWarps;
    constexpr int kRowChunks   = kStateDim * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
    constexpr int kValueChunks = kBlockDv * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
    for (int i = tid; i < count * kRowChunks; i += kThreads) {
        const int token = i / kRowChunks;
        const int chunk = (i - token * kRowChunks) * 8;
        cp_async<16>(&stage.key[token][chunk], access.key_ptr(coord, first + token) + chunk);
        cp_async<16>(&stage.query[token][chunk], access.query_ptr(coord, first + token) + chunk);
    }
    for (int i = tid; i < count * kValueChunks; i += kThreads) {
        const int token = i / kValueChunks;
        const int chunk = (i - token * kValueChunks) * 8;
        cp_async<16>(&stage.value[token][chunk],
                     access.value_ptr(coord, first + token) + coord.state_tile * kBlockDv + chunk);
    }
    if (tid < count) {
        const std::int64_t offset =
            access.column(coord, first + tid) * access.heads.H_v + coord.value_head;
        cp_async<4>(&stage.g[tid], access.g + offset);
        cp_async<4>(&stage.beta[tid], access.beta + offset);
    }
    cp_commit();
}

// A fold's head counts and conv channels: compile-time for the stacks registered in
// launch_replay_fold, read from FoldAccess at run time for every other one. Constant geometry
// turns the group division and the slot, record and history strides into immediates: A100, 27B
// DFlash2 C4, 846 -> ~700 us a fold (ninfer-all keeps the same split).
template <int QkHeads, int ValueHeads, int ConvChannels>
struct FoldGeometry {
    static constexpr bool kRuntime     = false;
    static constexpr int kQkHeads      = QkHeads;
    static constexpr int kValueHeads   = ValueHeads;
    static constexpr int kConvChannels = ConvChannels;
    static_assert(ValueHeads % QkHeads == 0);
    static_assert(ConvChannels % 128 == 0);
};

struct FoldGeometryRuntime {
    static constexpr bool kRuntime = true;
};

template <class StateT, class Geometry = FoldGeometryRuntime>
struct FoldAccess {
    const __nv_bfloat16* key_record;
    const __nv_bfloat16* value_record;
    const uint2* gate_record;
    const __nv_bfloat16* conv_record;
    StateT* recurrent_layer0;
    __nv_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t conv_layer_stride;
    std::int32_t record_capacity;
    std::int32_t width;
    GdnReplayFoldKernelRows rows;
    std::int32_t runtime_qk_heads;
    std::int32_t runtime_value_heads;
    std::int32_t runtime_conv_channels;

    __device__ __forceinline__ std::int32_t qk_heads() const {
        if constexpr (Geometry::kRuntime) {
            return runtime_qk_heads;
        } else {
            return Geometry::kQkHeads;
        }
    }
    __device__ __forceinline__ std::int32_t value_heads() const {
        if constexpr (Geometry::kRuntime) {
            return runtime_value_heads;
        } else {
            return Geometry::kValueHeads;
        }
    }
    __device__ __forceinline__ std::int32_t conv_channels() const {
        if constexpr (Geometry::kRuntime) {
            return runtime_conv_channels;
        } else {
            return Geometry::kConvChannels;
        }
    }

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        const std::int32_t batch       = static_cast<std::int32_t>(blockIdx.y);
        const std::int32_t layer_tile  = static_cast<std::int32_t>(blockIdx.z);
        const int lane                 = threadIdx.x;
        const int warp                 = threadIdx.y;
        const std::int32_t state_tile  = layer_tile & 7;
        const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
        const std::uint32_t group =
            static_cast<std::uint32_t>(value_heads()) / static_cast<std::uint32_t>(qk_heads());
        const std::uint32_t qk_head = value_head / group;
        const std::uint32_t dv_base =
            static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
        return {lane,
                warp,
                batch,
                layer_tile >> 3,
                state_tile,
                value_head,
                qk_head,
                dv_base,
                static_cast<std::uint32_t>(lane * kQkPerLane)};
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        return rows.row[coord.batch].commit_columns;
    }

    __device__ __forceinline__ std::int64_t record_outer(const RecurrentCoordinates& coord) const {
        return static_cast<std::int64_t>(coord.layer) * record_capacity + coord.batch;
    }

    __device__ __forceinline__ const StateT*
    state_read_base(const RecurrentCoordinates& coord) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(value_heads()) * kStateDim * kStateDim;
        return recurrent_layer0 + static_cast<std::int64_t>(coord.layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(rows.row[coord.batch].source_state_slot) * slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ StateT* state_write_base(const RecurrentCoordinates& coord) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(value_heads()) * kStateDim * kStateDim;
        return recurrent_layer0 + static_cast<std::int64_t>(coord.layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(rows.row[coord.batch].destination_state_slot) *
                   slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return key_record + (column * qk_heads() + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return value_record + (column * value_heads() + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return load_record_gate(gate_record, column * value_heads() + coord.value_head);
    }

    __device__ __forceinline__ void
    store_final_state(const RecurrentCoordinates& coord,
                      const float (&state)[kDvPerWarp][kQkPerLane]) const {
        StateT* destination = state_write_base(coord);
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            store_qk_lane(state[r],
                          destination + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                          coord.dqk_base);
        }
    }

    __device__ __forceinline__ void publish_final_conv_history(const RecurrentCoordinates& coord,
                                                               std::int32_t commit) const {
        const std::int32_t channels = conv_channels();
        const std::int32_t tile_block =
            static_cast<std::int32_t>(coord.value_head) * 8 + coord.state_tile;
        if (tile_block >= channels / 128) { return; }

        const std::int32_t tid     = coord.warp * kWarpSize + coord.lane;
        const std::int32_t channel = tile_block * 128 + tid;
        const __nv_bfloat16* source_history =
            conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[coord.batch].source_state_slot) *
                (3LL * channels) +
            channel;
        __nv_bfloat16* destination_history =
            conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[coord.batch].destination_state_slot) *
                (3LL * channels) +
            channel;
        const __nv_bfloat16* record =
            conv_record + record_outer(coord) * width * channels + channel;

        __nv_bfloat16 h0;
        __nv_bfloat16 h1;
        __nv_bfloat16 h2;
        if (commit == 1) {
            h0 = source_history[channels];
            h1 = source_history[2LL * channels];
            h2 = record[0];
        } else if (commit == 2) {
            h0 = source_history[2LL * channels];
            h1 = record[0];
            h2 = record[channels];
        } else {
            h0 = record[static_cast<std::int64_t>(commit - 3) * channels];
            h1 = record[static_cast<std::int64_t>(commit - 2) * channels];
            h2 = record[static_cast<std::int64_t>(commit - 1) * channels];
        }
        destination_history[0]                             = h0;
        destination_history[channels]       = h1;
        destination_history[2LL * channels] = h2;
    }
};

template <class StateT>
__device__ __forceinline__ void load_state_tile(float (&state)[kDvPerWarp][kQkPerLane],
                                                const StateT* base,
                                                const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                     coord.dqk_base);
    }
}

template <class StateT>
__device__ __forceinline__ void store_state_tile(const float (&state)[kDvPerWarp][kQkPerLane],
                                                 StateT* base, const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                      coord.dqk_base);
    }
}

template <bool NormalizeInputs, class Effects, class Access>
__device__ __forceinline__ void
run_recurrent_sequence(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                       const RecurrentCoordinates& coord, std::int32_t valid) {
    RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, 0), coord.dqk_base);
    Effects::observe_key(access, coord, 0, key);

    for (std::int32_t token = 0; token < valid; ++token) {
        const RawGatePair gate = access.load_gate(coord, token);
        const RawValueLane value =
            load_value_lane(access.value_ptr(coord, token), coord.lane, coord.dv_base);
        Effects::observe_value_gate(access, coord, token, value, gate);

        apply_gdn_transition<NormalizeInputs>(state, key.value, value.value, gate.g, gate.beta);

        if (token + 1 < valid) {
            key = load_raw_qk_lane(access.key_ptr(coord, token + 1), coord.dqk_base);
            Effects::observe_key(access, coord, token + 1, key);
        }

        Effects::template publish_output<NormalizeInputs>(state, access, coord, token);
    }
}

template <class Access>
__device__ __forceinline__ void zero_output_suffix(const Access& access,
                                                   const RecurrentCoordinates& coord,
                                                   std::int32_t valid, std::int32_t width) {
    if (coord.lane < kDvPerWarp) {
        for (std::int32_t token = valid; token < width; ++token) {
            access.output_ptr(coord, token)[coord.dv_base + coord.lane] = __float2bfloat16(0.0f);
        }
    }
}

template <bool NormalizeInputs, class StateT = float>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_bf16_direct_kernel(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta,
                                 const StateT* __restrict__ state_read,
                                 StateT* __restrict__ state_write, __nv_bfloat16* __restrict__ out,
                                 std::int32_t width, head_map heads, float scale) {
    const DirectAccess<StateT> access{q,           k,   v,     g,     beta, state_read,
                                      state_write, out, heads, width, scale};
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    // Launched as a programmatic dependent of the conv where the device has PDL. The state is the
    // previous token's (written two or more kernels ago); q, k, v, g and beta are this token's.
    // In a plain launch the wait returns at once.
    pdl::wait_for_dependencies();
    run_recurrent_sequence<NormalizeInputs, OutputEffects>(state, access, coord, width);
    store_state_tile(state, access.state_write_base(coord), coord);
}

// The direct recurrence over two staged windows (DirectStage). Launched for calls of
// kDirectStageMinTokens tokens or more; decode's one-token calls keep recurrent_bf16_direct_kernel.
inline constexpr int kDirectStageMinTokens = 2;

template <bool NormalizeInputs, class StateT = float>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_bf16_staged_kernel(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta,
                                 const StateT* __restrict__ state_read,
                                 StateT* __restrict__ state_write, __nv_bfloat16* __restrict__ out,
                                 std::int32_t width, head_map heads, float scale) {
    const DirectAccess<StateT> access{q,           k,   v,     g,     beta, state_read,
                                      state_write, out, heads, width, scale};
    const RecurrentCoordinates coord = access.coordinates();
    __shared__ DirectStage stage[2];
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    pdl::wait_for_dependencies();
    issue_direct_window(stage[0], access, coord, 0, min(width, kDirectStageTokens));
    for (std::int32_t first = 0, window = 0; first < width;
         first += kDirectStageTokens, ++window) {
        const std::int32_t count = min(width - first, kDirectStageTokens);
        const std::int32_t next  = first + kDirectStageTokens;
        // The buffer the next window fills was last read by the window before this one, which
        // every thread finished before the barrier that closed its iteration.
        if (next < width) {
            issue_direct_window(stage[(window + 1) & 1], access, coord, next,
                                min(width - next, kDirectStageTokens));
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const StagedDirectAccess<StateT> staged{access, &stage[window & 1], first};
        run_recurrent_sequence<NormalizeInputs, OutputEffects>(state, staged, coord, count);
        __syncthreads();
    }
    store_state_tile(state, access.state_write_base(coord), coord);
}

template <bool NormalizeInputs, class StateT = float>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_batch_update_kernel(BatchUpdateAccess<StateT> access) {
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    // Launched as a programmatic dependent of the kernel before (the decode's fused input
    // projection) where the device has PDL. The state and the slots are older; q, k, v, g and
    // beta are this token's. In a plain launch the wait returns at once.
    pdl::wait_for_dependencies();
    run_recurrent_sequence<NormalizeInputs, OutputEffects>(state, access, coord, 1);
    store_state_tile(state, access.state_write_base(coord), coord);
}

template <bool Masked, class StateT = float>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_record_kernel(RecordAccess<Masked, StateT> access) {
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    // Launched as a programmatic dependent of the conv where the device has PDL: the state tile
    // (the last fold's, and the slots, older than the conv's own stream-ordered launch) streams
    // under the conv's tail; q, k, v, g, beta and the valid columns are read after the wait.
    pdl::wait_for_dependencies();
    const std::int32_t valid = access.active_columns(coord);
    if (valid <= kRecordStageTokens) {
        __shared__ RecordStage stage;
        stage_record_window(stage, access, coord, valid);
        const StagedRecordAccess<Masked, StateT> staged{access, &stage};
        run_recurrent_sequence<true, RecordEffects>(state, staged, coord, valid);
    } else {
        run_recurrent_sequence<true, RecordEffects>(state, access, coord, valid);
    }
    zero_output_suffix(access, coord, valid, access.width);
}

template <class StateT, class Geometry>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_fold_kernel(const __grid_constant__ FoldAccess<StateT, Geometry> access) {
    const RecurrentCoordinates coord = access.coordinates();
    const std::int32_t valid         = access.active_columns(coord);
    if (valid == 0) { return; }
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<true, FoldEffects>(state, access, coord, valid);
    access.store_final_state(coord, state);
    access.publish_final_conv_history(coord, valid);
}

} // namespace ninferno::ops::detail::gated_delta_net
