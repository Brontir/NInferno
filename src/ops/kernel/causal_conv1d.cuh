#pragma once

#include "core/pdl.cuh"
// ninferno::ops - causal_conv1d kernel: depthwise causal k=4 with fused SiLU.
// SiLU is computed as x / (1 + exp(-x)) in fp32, with no polynomial approximation.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninferno::ops {

__device__ __forceinline__ void causal_conv1d_acc_pair(__nv_bfloat162 w, __nv_bfloat162 x,
                                                       float& acc0, float& acc1) {
    acc0 += __low2float(w) * __low2float(x);
    acc1 += __high2float(w) * __high2float(x);
}

inline constexpr int kCausalConvChannelTile = 32;

// Every global load below that only one arm of a condition needs has an index that is valid
// whatever the condition (clamped to row 0 or channel 0), and the condition selects the value.
// The same holds for the shared history rows (row 0 when the position is past them). CUDA 12.8's
// sm_120a code evaluated such loads in the arm not taken: the state at row t past its three, the
// input at row -1 or -2, the history row t + 3; it faulted (cudaErrorIllegalAddress) where 13.1's
// did not.

namespace detail {

// Position p's value of one channel: the shared history row p + 3 before the chunk (p < 0), else
// row p of the chunk, whose rows start at `rows` (the channel's column already added). Both
// addresses are valid whatever p is (the note above); the condition selects the value.
__device__ __forceinline__ __nv_bfloat16
causal_conv_tap(const __nv_bfloat16 (&history)[3][kCausalConvChannelTile], int lane,
                const __nv_bfloat16* rows, std::int64_t C64, std::int32_t p) {
    const bool before = p < 0;
    return before ? history[before ? p + 3 : 0][lane]
                  : rows[static_cast<std::int64_t>(before ? 0 : p) * C64];
}

// One destination column resolved for a fixed row: the address of element (row,0) and the row
// count of the destination that owns it.
template <class Element>
struct CausalConvOutputColumn {
    Element* base;
    std::int32_t rows;

    __device__ __forceinline__ Element* at(std::int32_t t) const {
        return base + static_cast<std::int64_t>(t) * rows;
    }
};

// Output address map for a single contiguous [rows,T] destination. One destination admits any
// row count, so the leading dimension stays a runtime value. The pointer is not restrict: the
// entries that use this map do not establish that their destination is disjoint from x.
template <class Element>
struct CausalConvContiguousOutput {
    Element* out;
    std::int32_t rows;

    __device__ __forceinline__ CausalConvOutputColumn<Element> column(std::int64_t row) const {
        return {out + row, rows};
    }
};

// Output address map for three destinations partitioned by row (the model's q/k/v row counts, read
// at run time). Callers writing pairs pass halved row counts, which requires every boundary to be
// even. The two compares per element cost nothing measurable in these memory-bound kernels (r64:
// against the compile-time partitions of Qwen3.5 27B and 35B-A3B it replaced).
template <class Element>
struct CausalConvSplitOutputRuntime {
    Element* __restrict__ out0;
    Element* __restrict__ out1;
    Element* __restrict__ out2;
    std::int32_t rows0;
    std::int32_t rows1;
    std::int32_t rows2;

    __device__ __forceinline__ CausalConvOutputColumn<Element> column(std::int64_t row) const {
        if (row < rows0) { return {out0 + row, rows0}; }
        if (row < rows0 + rows1) { return {out1 + (row - rows0), rows1}; }
        return {out2 + (row - rows0 - rows1), rows2};
    }
};

} // namespace detail

template <class Output>
__global__ void causal_conv1d_prefill_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                             const __nv_bfloat16* conv_state, Output out,
                                             std::int32_t C, std::int32_t T) {
    const std::int64_t C64      = static_cast<std::int64_t>(C);
    const std::int64_t c_blocks = div_up(C64, static_cast<std::int64_t>(blockDim.x));
    const std::int64_t block    = static_cast<std::int64_t>(blockIdx.x);
    const std::int32_t t        = static_cast<std::int32_t>(block / c_blocks);
    const std::int64_t c_base   = (block - static_cast<std::int64_t>(t) * c_blocks) * blockDim.x;
    const std::int64_t c64      = c_base + threadIdx.x;
    if (t >= T || c64 >= C64) { return; }

    const std::int32_t c       = static_cast<std::int32_t>(c64);
    const std::int64_t in_idx = static_cast<std::int64_t>(t) * C64 + c64;
    const __nv_bfloat16 x0     = (t >= 3)
                                     ? x[static_cast<std::int64_t>(t >= 3 ? t - 3 : 0) * C64 + c64]
                                     : conv_state[static_cast<std::int64_t>(t < 3 ? t : 0) * C64 + c64];
    const __nv_bfloat16 x1 =
        (t >= 2) ? x[static_cast<std::int64_t>(t >= 2 ? t - 2 : 0) * C64 + c64]
                 : conv_state[static_cast<std::int64_t>(t < 2 ? t + 1 : 0) * C64 + c64];
    const __nv_bfloat16 x2 =
        (t >= 1) ? x[static_cast<std::int64_t>(t >= 1 ? t - 1 : 0) * C64 + c64]
                 : conv_state[static_cast<std::int64_t>(t < 1 ? t + 2 : 0) * C64 + c64];
    const __nv_bfloat16 x3    = x[in_idx];

    float acc = 0.0f;
    acc += __bfloat162float(weight[c]) * __bfloat162float(x0);
    acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(x1);
    acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(x2);
    acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x3);
    *out.column(c64).at(t) = __float2bfloat16_rn(silu(acc));
}

template <class Output>
__global__ void causal_conv1d_prefill_pairs_kernel(const __nv_bfloat16* x,
                                                   const __nv_bfloat16* weight,
                                                   const __nv_bfloat16* conv_state, Output out,
                                                   std::int32_t C, std::int32_t T) {
    const std::int64_t C2        = static_cast<std::int64_t>(C / 2);
    const std::int64_t c_blocks  = div_up(C2, static_cast<std::int64_t>(blockDim.x));
    const std::int64_t block     = static_cast<std::int64_t>(blockIdx.x);
    const std::int32_t t         = static_cast<std::int32_t>(block / c_blocks);
    const std::int64_t pair_base = (block - static_cast<std::int64_t>(t) * c_blocks) * blockDim.x;
    const std::int64_t p         = pair_base + threadIdx.x;
    if (t >= T || p >= C2) { return; }

    const auto* x2      = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* weight2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    const auto* state2  = reinterpret_cast<const __nv_bfloat162*>(conv_state);

    // Column t - 3 + k of the sequence: the input past the state's three columns, else the state.
    // The pointer is chosen before the one load: with a load in each arm of the conditional, CUDA
    // 12.8's sm_120a code read the state at row t (past its three) for every t >= 3 and faulted
    // (cudaErrorIllegalAddress in the Qwen3.5 warm-up, RTX 5080); 13.1's did not. The row is
    // never negative in either arm: 12.8 widened a negative s = t - 3 + k without its sign and read
    // 2^32 rows past x for t < 3 (address ~4 * C2 * 2^32 bytes out, causal_conv1d test).
    const auto column = [&](std::int32_t k) {
        const bool input       = t + k >= 3;
        const std::int64_t row = static_cast<std::int64_t>(input ? t + k - 3 : t + k);
        return (input ? x2 : state2) + row * C2 + p;
    };
    const __nv_bfloat162 x0  = *column(0);
    const __nv_bfloat162 x1  = *column(1);
    const __nv_bfloat162 x2v = *column(2);
    const __nv_bfloat162 x3  = x2[static_cast<std::int64_t>(t) * C2 + p];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    causal_conv1d_acc_pair(weight2[p], x0, acc0, acc1);
    causal_conv1d_acc_pair(weight2[C2 + p], x1, acc0, acc1);
    causal_conv1d_acc_pair(weight2[2 * C2 + p], x2v, acc0, acc1);
    causal_conv1d_acc_pair(weight2[3 * C2 + p], x3, acc0, acc1);
    *out.column(p).at(t) = __floats2bfloat162_rn(silu(acc0), silu(acc1));
}

// Writes the trailing width-3 conv window after consuming the T input columns.
// The initial window is read from `conv_state_in` and the new window is written
// to `conv_state_out`; passing the same pointer for both is the in-place form.
// Separating them lets prefix-append prefill read the committed state from a
// selected GDN snapshot slot while always publishing the running state to slot 0.
// The initial values are loaded into registers before any store, so overlapping
// in/out (in == out) has no read/write hazard.
__global__ void causal_conv1d_prefill_state_kernel(const __nv_bfloat16* x,
                                                   const __nv_bfloat16* conv_state_in,
                                                   __nv_bfloat16* conv_state_out, std::int32_t C,
                                                   std::int32_t T) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64    = static_cast<std::int64_t>(C);

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c     = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 old0 = conv_state_in[c];
        const __nv_bfloat16 old1 = conv_state_in[C64 + c];
        const __nv_bfloat16 old2 = conv_state_in[2 * C64 + c];

        for (std::int32_t s = 0; s < 3; ++s) {
            const std::int32_t seq_pos = T + s;
            __nv_bfloat16 v;
            if (seq_pos == 0) {
                v = old0;
            } else if (seq_pos == 1) {
                v = old1;
            } else if (seq_pos == 2) {
                v = old2;
            } else {
                v = x[static_cast<std::int64_t>(seq_pos < 3 ? 0 : seq_pos - 3) * C64 + c];
            }
            conv_state_out[static_cast<std::int64_t>(s) * C64 + c] = v;
        }
    }
}

// Small-T ordinary form parallelized across both channels and tokens. Each CTA owns one channel
// tile for the full sequence. Loading history into shared memory before any state publication makes
// the exact-alias state form safe without a second kernel.
template <class Output>
__global__ void causal_conv1d_smallt_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                            const __nv_bfloat16* conv_state_in,
                                            __nv_bfloat16* conv_state_out, Output out,
                                            std::int32_t C, std::int32_t T, int release) {
    __shared__ __nv_bfloat16 history[3][kCausalConvChannelTile];
    __shared__ __nv_bfloat16 weights[4][kCausalConvChannelTile];
    // release: the next kernel joins this one by PDL and waits for it before it reads anything
    // this kernel writes (the GDN decode recurrence); letting it launch now puts its state loads
    // under this kernel.
    if (release != 0 && threadIdx.x == 0 && threadIdx.y == 0) pdl::trigger_dependents();

    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t t    = static_cast<std::int32_t>(threadIdx.y);
    const std::int64_t c64  = static_cast<std::int64_t>(blockIdx.x) * kCausalConvChannelTile + lane;
    const std::int64_t C64  = static_cast<std::int64_t>(C);
    const bool valid        = c64 < C64;

    if (t == 0) {
        for (std::int32_t s = 0; s < 3; ++s) {
            history[s][lane] =
                valid ? conv_state_in[static_cast<std::int64_t>(s) * C64 + (valid ? c64 : 0)]
                      : __float2bfloat16(0.0f);
        }
        for (std::int32_t r = 0; r < 4; ++r) {
            weights[r][lane] = valid
                                   ? weight[static_cast<std::int64_t>(r) * C64 + (valid ? c64 : 0)]
                                   : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    if (!valid) { return; }

    const std::int32_t p0 = t - 3;
    const std::int32_t p1 = t - 2;
    const std::int32_t p2 = t - 1;
    const __nv_bfloat16 x0   = detail::causal_conv_tap(history, lane, x + c64, C64, p0);
    const __nv_bfloat16 x1   = detail::causal_conv_tap(history, lane, x + c64, C64, p1);
    const __nv_bfloat16 x2   = detail::causal_conv_tap(history, lane, x + c64, C64, p2);
    const std::int64_t x_idx = static_cast<std::int64_t>(t) * C64 + c64;
    const __nv_bfloat16 x3   = x[x_idx];

    float acc = 0.0f;
    acc += __bfloat162float(weights[0][lane]) * __bfloat162float(x0);
    acc += __bfloat162float(weights[1][lane]) * __bfloat162float(x1);
    acc += __bfloat162float(weights[2][lane]) * __bfloat162float(x2);
    acc += __bfloat162float(weights[3][lane]) * __bfloat162float(x3);
    *out.column(c64).at(t) = __float2bfloat16_rn(silu(acc));

    if (t == 0) {
        for (std::int32_t s = 0; s < 3; ++s) {
            const std::int32_t pos = T - 3 + s;
            conv_state_out[static_cast<std::int64_t>(s) * C64 + c64] =
                detail::causal_conv_tap(history, lane, x + c64, C64, pos);
        }
    }
}

__global__ void causal_conv1d_decode_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                            __nv_bfloat16* conv_state, __nv_bfloat16* out,
                                            std::int32_t C) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64    = static_cast<std::int64_t>(C);

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c   = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 s0 = conv_state[c];
        const __nv_bfloat16 s1 = conv_state[C64 + c];
        const __nv_bfloat16 s2 = conv_state[2 * C64 + c];
        const __nv_bfloat16 x0 = x[c];

        float acc = 0.0f;
        acc += __bfloat162float(weight[c]) * __bfloat162float(s0);
        acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(s1);
        acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(s2);
        acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x0);

        out[c]                  = __float2bfloat16_rn(silu(acc));
        conv_state[c]           = s1;
        conv_state[C64 + c]     = s2;
        conv_state[2 * C64 + c] = x0;
    }
}

__global__ void causal_conv1d_decode_distinct_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ conv_state_in, __nv_bfloat16* __restrict__ conv_state_out,
    __nv_bfloat16* __restrict__ out, std::int32_t C) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64    = static_cast<std::int64_t>(C);

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c   = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 s0 = conv_state_in[c];
        const __nv_bfloat16 s1 = conv_state_in[C64 + c];
        const __nv_bfloat16 s2 = conv_state_in[2 * C64 + c];
        const __nv_bfloat16 x0 = x[c];

        float acc = 0.0f;
        acc += __bfloat162float(weight[c]) * __bfloat162float(s0);
        acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(s1);
        acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(s2);
        acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x0);

        out[c]                      = __float2bfloat16_rn(silu(acc));
        conv_state_out[c]           = s1;
        conv_state_out[C64 + c]     = s2;
        conv_state_out[2 * C64 + c] = x0;
    }
}

__global__ void causal_conv1d_snapshot_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ conv_states, const std::int32_t* __restrict__ initial_slot,
    const std::int32_t* __restrict__ snapshot_base_slot, __nv_bfloat16* __restrict__ out,
    std::int32_t C, std::int64_t slot_stride) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64    = static_cast<std::int64_t>(C);
    const __nv_bfloat16* init =
        conv_states + static_cast<std::int64_t>(*initial_slot) * slot_stride;

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c   = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 s0 = init[c];
        const __nv_bfloat16 s1 = init[C64 + c];
        const __nv_bfloat16 s2 = init[2 * C64 + c];
        const __nv_bfloat16 x0 = x[c];

        float acc = 0.0f;
        acc += __bfloat162float(weight[c]) * __bfloat162float(s0);
        acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(s1);
        acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(s2);
        acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x0);

        out[c] = __float2bfloat16_rn(silu(acc));
        __nv_bfloat16* snapshot =
            conv_states + static_cast<std::int64_t>(*snapshot_base_slot) * slot_stride;
        snapshot[c]           = s1;
        snapshot[C64 + c]     = s2;
        snapshot[2 * C64 + c] = x0;
    }
}

__global__ void
causal_conv1d_sequence_snapshot_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                       __nv_bfloat16* conv_states, const std::int32_t* initial_slot,
                                       const std::int32_t* snapshot_base_slot, __nv_bfloat16* out,
                                       std::int32_t C, std::int32_t T, std::int64_t slot_stride) {
    const std::int64_t c64 = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t C64 = static_cast<std::int64_t>(C);
    if (c64 >= C64) { return; }

    const std::int32_t slot   = *initial_slot;
    const __nv_bfloat16* init = conv_states + static_cast<std::int64_t>(slot) * slot_stride;
    __nv_bfloat16 s0          = init[c64];
    __nv_bfloat16 s1          = init[C64 + c64];
    __nv_bfloat16 s2          = init[2 * C64 + c64];
    const float w0            = __bfloat162float(weight[c64]);
    const float w1            = __bfloat162float(weight[C64 + c64]);
    const float w2            = __bfloat162float(weight[2 * C64 + c64]);
    const float w3            = __bfloat162float(weight[3 * C64 + c64]);

    for (std::int32_t t = 0; t < T; ++t) {
        const std::int64_t out_idx = static_cast<std::int64_t>(t) * C64 + c64;
        const __nv_bfloat16 x0     = x[out_idx];

        float acc = 0.0f;
        acc += w0 * __bfloat162float(s0);
        acc += w1 * __bfloat162float(s1);
        acc += w2 * __bfloat162float(s2);
        acc += w3 * __bfloat162float(x0);

        out[out_idx] = __float2bfloat16_rn(silu(acc));
        s0           = s1;
        s1           = s2;
        s2           = x0;

        __nv_bfloat16* snapshot =
            conv_states + static_cast<std::int64_t>(*snapshot_base_slot + t) * slot_stride;
        snapshot[c64]           = s0;
        snapshot[C64 + c64]     = s1;
        snapshot[2 * C64 + c64] = s2;
    }
}

// Batched serial-column form. One block row owns one request and one thread owns a channel, so
// request-local state is loaded before that row publishes any snapshot. It is used for W=1 and
// for masked B=1 widths outside the parallel small-W domain.
template <bool Masked>
__global__ void causal_conv1d_batched_sequence_snapshot_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* conv_states,
    const std::int32_t* valid_columns, const std::int32_t* initial_state_slots,
    const std::int32_t* snapshot_base_slots, __nv_bfloat16* out, std::int32_t C, std::int32_t width,
    std::int64_t slot_stride) {
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t c64   = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t C64   = static_cast<std::int64_t>(C);
    if (c64 >= C64) { return; }

    const std::int32_t valid = Masked ? valid_columns[batch] : width;
    const __nv_bfloat16* init =
        conv_states + static_cast<std::int64_t>(initial_state_slots[batch]) * slot_stride;
    __nv_bfloat16 s0            = init[c64];
    __nv_bfloat16 s1            = init[C64 + c64];
    __nv_bfloat16 s2            = init[2 * C64 + c64];
    const float w0              = __bfloat162float(weight[c64]);
    const float w1              = __bfloat162float(weight[C64 + c64]);
    const float w2              = __bfloat162float(weight[2 * C64 + c64]);
    const float w3              = __bfloat162float(weight[3 * C64 + c64]);
    const std::int64_t row_base = static_cast<std::int64_t>(batch) * width * C64;

    for (std::int32_t column = 0; column < width; ++column) {
        const std::int64_t out_idx = row_base + static_cast<std::int64_t>(column) * C64 + c64;
        if (column >= valid) {
            out[out_idx] = __float2bfloat16(0.0f);
            continue;
        }

        const __nv_bfloat16 x0 = x[out_idx];
        float acc              = 0.0f;
        acc += w0 * __bfloat162float(s0);
        acc += w1 * __bfloat162float(s1);
        acc += w2 * __bfloat162float(s2);
        acc += w3 * __bfloat162float(x0);
        out[out_idx] = __float2bfloat16_rn(silu(acc));
        s0           = s1;
        s1           = s2;
        s2           = x0;

        __nv_bfloat16* snapshot =
            conv_states +
            static_cast<std::int64_t>(snapshot_base_slots[batch] + column) * slot_stride;
        snapshot[c64]           = s0;
        snapshot[C64 + c64]     = s1;
        snapshot[2 * C64 + c64] = s2;
    }
}

// Small-T snapshot form. A CTA owns all token outputs for one channel tile. The selected history
// is cached before any snapshot slot is written, so initial_slot may name any slot, including one
// overwritten by this call.
__global__ void
causal_conv1d_snapshot_smallt_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                     __nv_bfloat16* conv_states, const std::int32_t* initial_slot,
                                     const std::int32_t* snapshot_base_slot, __nv_bfloat16* out,
                                     std::int32_t C, std::int32_t T, std::int64_t slot_stride) {
    __shared__ __nv_bfloat16 history[3][kCausalConvChannelTile];
    __shared__ __nv_bfloat16 weights[4][kCausalConvChannelTile];

    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t t    = static_cast<std::int32_t>(threadIdx.y);
    const std::int64_t c64  = static_cast<std::int64_t>(blockIdx.x) * kCausalConvChannelTile + lane;
    const std::int64_t C64  = static_cast<std::int64_t>(C);
    const bool valid        = c64 < C64;

    if (t == 0) {
        const std::int32_t slot   = *initial_slot;
        const __nv_bfloat16* init = conv_states + static_cast<std::int64_t>(slot) * slot_stride;
        for (std::int32_t s = 0; s < 3; ++s) {
            history[s][lane] = valid ? init[static_cast<std::int64_t>(s) * C64 + (valid ? c64 : 0)]
                                     : __float2bfloat16(0.0f);
        }
        for (std::int32_t r = 0; r < 4; ++r) {
            weights[r][lane] = valid
                                   ? weight[static_cast<std::int64_t>(r) * C64 + (valid ? c64 : 0)]
                                   : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    if (!valid) { return; }

    const std::int32_t p0 = t - 3;
    const std::int32_t p1 = t - 2;
    const std::int32_t p2 = t - 1;
    const __nv_bfloat16 x0     = detail::causal_conv_tap(history, lane, x + c64, C64, p0);
    const __nv_bfloat16 x1     = detail::causal_conv_tap(history, lane, x + c64, C64, p1);
    const __nv_bfloat16 x2     = detail::causal_conv_tap(history, lane, x + c64, C64, p2);
    const std::int64_t out_idx = static_cast<std::int64_t>(t) * C64 + c64;
    const __nv_bfloat16 x3     = x[out_idx];

    float acc = 0.0f;
    acc += __bfloat162float(weights[0][lane]) * __bfloat162float(x0);
    acc += __bfloat162float(weights[1][lane]) * __bfloat162float(x1);
    acc += __bfloat162float(weights[2][lane]) * __bfloat162float(x2);
    acc += __bfloat162float(weights[3][lane]) * __bfloat162float(x3);
    out[out_idx] = __float2bfloat16_rn(silu(acc));

    __nv_bfloat16* snapshot =
        conv_states + static_cast<std::int64_t>(*snapshot_base_slot + t) * slot_stride;
    for (std::int32_t s = 0; s < 3; ++s) {
        const std::int32_t pos = t - 2 + s;
        snapshot[static_cast<std::int64_t>(s) * C64 + c64] =
            detail::causal_conv_tap(history, lane, x + c64, C64, pos);
    }
}

template <bool Masked>
__global__ void causal_conv1d_batched_snapshot_smallt_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* conv_states,
    const std::int32_t* valid_columns, const std::int32_t* initial_state_slots,
    const std::int32_t* snapshot_base_slots, __nv_bfloat16* out, std::int32_t C, std::int32_t width,
    std::int64_t slot_stride) {
    __shared__ __nv_bfloat16 history[3][kCausalConvChannelTile];
    __shared__ __nv_bfloat16 weights[4][kCausalConvChannelTile];

    const std::int32_t batch  = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t lane   = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t column = static_cast<std::int32_t>(threadIdx.y);
    const std::int64_t c64 = static_cast<std::int64_t>(blockIdx.x) * kCausalConvChannelTile + lane;
    const std::int64_t C64 = static_cast<std::int64_t>(C);
    const bool channel_valid = c64 < C64;

    if (column == 0) {
        const __nv_bfloat16* init =
            conv_states + static_cast<std::int64_t>(initial_state_slots[batch]) * slot_stride;
        for (std::int32_t s = 0; s < 3; ++s) {
            history[s][lane] =
                channel_valid ? init[static_cast<std::int64_t>(s) * C64 + (channel_valid ? c64 : 0)]
                              : __float2bfloat16(0.0f);
        }
        for (std::int32_t r = 0; r < 4; ++r) {
            weights[r][lane] =
                channel_valid
                    ? weight[static_cast<std::int64_t>(r) * C64 + (channel_valid ? c64 : 0)]
                    : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    if (!channel_valid) { return; }

    const std::int32_t valid    = Masked ? valid_columns[batch] : width;
    const std::int64_t row_base = static_cast<std::int64_t>(batch) * width * C64;
    const std::int64_t out_idx  = row_base + static_cast<std::int64_t>(column) * C64 + c64;
    if (column >= valid) {
        out[out_idx] = __float2bfloat16(0.0f);
        return;
    }

    const std::int32_t p0 = column - 3;
    const std::int32_t p1 = column - 2;
    const std::int32_t p2 = column - 1;
    const __nv_bfloat16 x0 = detail::causal_conv_tap(history, lane, x + row_base + c64, C64, p0);
    const __nv_bfloat16 x1 = detail::causal_conv_tap(history, lane, x + row_base + c64, C64, p1);
    const __nv_bfloat16 x2 = detail::causal_conv_tap(history, lane, x + row_base + c64, C64, p2);
    const __nv_bfloat16 x3 = x[out_idx];

    float acc = 0.0f;
    acc += __bfloat162float(weights[0][lane]) * __bfloat162float(x0);
    acc += __bfloat162float(weights[1][lane]) * __bfloat162float(x1);
    acc += __bfloat162float(weights[2][lane]) * __bfloat162float(x2);
    acc += __bfloat162float(weights[3][lane]) * __bfloat162float(x3);
    out[out_idx] = __float2bfloat16_rn(silu(acc));

    __nv_bfloat16* snapshot =
        conv_states + static_cast<std::int64_t>(snapshot_base_slots[batch] + column) * slot_stride;
    for (std::int32_t s = 0; s < 3; ++s) {
        const std::int32_t pos = column - 2 + s;
        snapshot[static_cast<std::int64_t>(s) * C64 + c64] =
            detail::causal_conv_tap(history, lane, x + row_base + c64, C64, pos);
    }
}

} // namespace ninferno::ops
