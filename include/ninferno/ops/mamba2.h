#pragma once

// ninferno::ops - the Mamba2 mixer's Ops (NemotronH): the causal convolution, the selective state
// scan and the gated group RMSNorm, over B sequences whose state lives in a slot pool (the
// Program's linear state: a conv window [C, 3] and a recurrent state [N, P, H] per slot).
//
// A call of T columns carries B rows of L = T / B consecutive tokens each; row b's columns are
// [b*L, (b+1)*L), and its state is slot slots[b]. Every Op reads its state from the slot and
// writes the state after the row's last token back to the same slot (a fork copies the source
// slot first). Every Op is stream-ordered and CUDA-graph capturable; its launch geometry depends
// on the shapes only.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninferno::ops {

/**
 * Depthwise causal width-4 convolution with bias, then SiLU:
 *
 *   out[c,t] = silu(bias[c] + sum_{j=0..3} weight[j,c] * u[c,t-3+j])
 *
 * where u[c,-3..-1] is the row's window in its slot and u[c,t] = x[c,t]. x is BF16 [C, T] whose
 * columns may be strided (a row range of a wider projection); weight is contiguous BF16 [4, C]
 * (HF's conv1d.weight [C,1,4]); bias contiguous BF16 [C] or empty; states BF16 [C, 3, S]; slots
 * I32 [B] on the device; out contiguous BF16 [C, T]. FP32 accumulation, one BF16 rounding.
 *
 * input_record, when given (contiguous BF16 [C, T]), receives every column's input u[c,t] and the
 * slots' windows are left as they were: a speculative verify pass, whose rejected columns must not
 * reach the state (mamba2_verify_fold commits the accepted ones).
 */
void mamba2_conv(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& states,
                 const Tensor& slots, std::int32_t batch, Tensor& out, cudaStream_t stream,
                 Tensor* input_record = nullptr, bool join_previous = false);

struct Mamba2Geometry {
    std::int32_t heads    = 0; // H
    std::int32_t head_dim = 0; // P
    std::int32_t groups   = 0; // G: head h reads group h / (H / G)
    std::int32_t state    = 0; // N
    float dt_min          = 0.0F;
    float dt_max          = 0.0F; // 0: no upper clamp
};

/**
 * The selective state scan. xbc is the convolution's output, contiguous BF16 [H*P + 2*G*N, T]:
 * rows [0, H*P) are x, then G blocks of N rows of B, then G blocks of N rows of C. dt is BF16
 * [H, T] (strided columns allowed); dt_bias, a_log and d are FP32 [H]; states FP32 or BF16 [N, P, H, S] (BF16: the
 * recurrence still runs in FP32 and the state is rounded once when stored).
 * Per token t of a row, per head h (group g), in FP32:
 *
 *   dt      = clamp(softplus(dt[h,t] + dt_bias[h]), dt_min, dt_max)
 *   s[n,p]  = s[n,p] * exp(-dt * exp(a_log[h])) + dt * B[g,n,t] * x[h,p,t]
 *   y[h,p,t] = sum_n s[n,p] * C[g,n,t] + d[h] * x[h,p,t]
 *
 * y is contiguous BF16 [H*P, T], rounded once.
 *
 * A speculative verify pass passes store_state = false (the slots' states are only read) and a
 * dt_record (contiguous BF16 [H, T]), which receives dt[h,t] as read.
 */
void mamba2_scan(const Tensor& xbc, const Tensor& dt, const Tensor& dt_bias, const Tensor& a_log,
                 const Tensor& d, Tensor& states, const Tensor& slots, std::int32_t batch,
                 const Mamba2Geometry& geometry, Tensor& y, cudaStream_t stream,
                 bool store_state = true, Tensor* dt_record = nullptr);

/// One row of a verify pass to commit: its state is read from source_slot, and the state after
/// its first `commit` recorded columns is written to dest_slot.
struct Mamba2FoldRow {
    std::int32_t source_slot = 0;
    std::int32_t dest_slot   = 0;
    std::int32_t commit      = 0;
};
inline constexpr std::size_t kMamba2FoldMaxRows = 8;

/**
 * Commits a verify pass's accepted columns (mamba2_conv with input_record, then mamba2_scan with
 * store_state = false and a dt_record, over B rows of `width` columns each; row b's columns are
 * [b*width, (b+1)*width), rows[b] its commit). input_record is the conv's input_record, xbc_record
 * its out, dt_record the scan's: contiguous BF16 [C, width*B], [C, width*B], [H, width*B],
 * C = H*P + 2*G*N. For every row b:
 *
 *   conv window [dest]  = the last 3 of {window [source], u[:, b*width + j] for j < commit}
 *   state [dest]        = the scan from state [source] over columns b*width + j, j < commit
 *
 * the same bytes an in-place mamba2_conv + mamba2_scan over those columns leaves. commit 0 copies
 * the source state (nothing, when dest is the source). 1 <= B <= kMamba2FoldMaxRows, 0 <= commit
 * <= width; the rows' dest slots are pairwise distinct and none is another row's source.
 * conv_states and states are the pools mamba2_conv and mamba2_scan take. Rows are passed by
 * value: the call is stream-ordered, needs no device staging and can follow a graph replay.
 */
void mamba2_verify_fold(const Tensor& input_record, const Tensor& xbc_record,
                        const Tensor& dt_record, std::int32_t width,
                        std::span<const Mamba2FoldRow> rows, const Tensor& dt_bias,
                        const Tensor& a_log, const Tensor& d, Tensor& conv_states, Tensor& states,
                        const Mamba2Geometry& geometry, cudaStream_t stream);

/**
 * The gated group RMSNorm (transformers MambaRMSNormGated, norm after the gate), per column:
 *
 *   v   = y * silu(z)                                         (FP32)
 *   n   = v / sqrt(mean_{group}(v^2) + eps)                   (per group of `group` rows)
 *   out = weight * bf16(n)                                    (rounded to BF16)
 *
 * y contiguous BF16 [R, T]; z BF16 [R, T] (strided columns allowed); weight BF16 [R]; out
 * contiguous BF16 [R, T]; R % group == 0.
 */
void mamba2_gated_norm(const Tensor& y, const Tensor& z, const Tensor& weight, std::int32_t group,
                       float eps, Tensor& out, cudaStream_t stream);

} // namespace ninferno::ops
