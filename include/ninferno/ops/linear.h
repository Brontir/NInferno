#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops {

/**
 * @brief Permitted private activation-compute profiles for a linear projection.
 *
 * The policy constrains private route selection; it does not select a kernel or prescribe a
 * particular MMA instruction. The public activation and output tensors remain BF16 for every
 * policy.
 */
enum class LinearPolicy : std::uint8_t {
    A16Only, ///< Admit only A16 compute profiles.
    AllowA8, ///< Admit either A16 or A8 compute profiles.
    AllowA4, ///< Admit A16, A8 or A4 compute profiles.
};

[[nodiscard]] constexpr bool valid_linear_policy(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
           policy == LinearPolicy::AllowA4;
}

[[nodiscard]] constexpr bool allows_a8(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8 || policy == LinearPolicy::AllowA4;
}

[[nodiscard]] constexpr bool allows_a4(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA4;
}

/**
 * The policy of a projection in a verify pass (the dense and Gemma targets' Activations::Verify):
 * the recorded one, the policy plain decode of the same weights runs, capped at A8, so a W4A4
 * route at verify widths never accepts drafts against a coarser model than plain decode runs. It
 * was A16Only where the build has tensor mma.sync: GGUF weights then decoded on int8 activations
 * (R8) and verified on BF16 ones (the register MMA), another model, and their verify cost 1.2-1.5
 * times a decode step's linears where Qwen3.5's, which keeps the recorded policy, cost 1.0-1.1
 * (Gemma 4 12B Q4_K_M, RTX 5070 Ti, nsys: gate/up 110 us against 83, down 60 against 41).
 */
[[nodiscard]] LinearPolicy verify_linear_policy(LinearPolicy recorded) noexcept;

/**
 * The policy of a speculative drafter's projection: AllowA8 for a weight recorded A16Only whose
 * format verifies a whole eight-token block on R8 (int8 activations per 32 values, plain decode's
 * arithmetic: Q8, GGUF Q8_0, on every chip), else the recorded one. A DFlash block (dflash:7 is
 * eight tokens) of an A16Only Q8 drafter ran the BF16 register MMA, whose 16-row CTAs walk all of
 * K: at K = 12288 and 32768 (Qwen3.5-9B's drafter down and feature projections) it read 180-200
 * GB/s against R8's 440-465 on the RTX 4070 Ti (ninferno_r8_schedule_bench, T = 5 and 8), 8 % and
 * 4 % of a dflash:7 round. Q8 runs A8 under every target that records it (formats.cpp,
 * kept_block_policy); a drafter converted without a recorded policy now runs it too. The other
 * formats' R8 stops at four tokens and keeps the recorded policy. So does a Q8 shape the tuned
 * table of the op that runs it owns (`op`: plain linear, fused add or fused SwiGLU; Gemma 4 31B
 * DFlash's 5376 x 10752 through linear, a Qwen DFlash down projection 2048 x 6144 through
 * linear_add): that dispatcher runs the table's A16 kernels at 2..8 tokens and sends only a
 * one-token call to R8, so the lift would change nothing. The segmented projection (Generic)
 * reads no table.
 */
enum class DrafterOp { Linear, LinearAdd, LinearSwiGlu, Generic };
[[nodiscard]] LinearPolicy drafter_linear_policy(const Weight& weight, LinearPolicy recorded,
                                                 DrafterOp op);

/**
 * Returns the caller-owned transient capacity required by Linear for every T in the inclusive
 * `[min_tokens,max_tokens]` interval. Invalid registered profiles, policies, or intervals throw;
 * a legal route that requires no transient storage returns zero.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

/**
 * @brief Applies a bias-free matrix projection independently to every input column.
 *
 * @details The ideal mathematical result is
 *
 * @f[
 *   \mathrm{ideal}_{n,t} =
 *   \sum_{k=0}^{K-1}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * `out` stores a BF16 approximation of this ideal result under the named numerical criterion for
 * the selected private activation-compute path.
 *
 * @par Logical tensors and layout
 * `x` is contiguous, non-null, 16-byte-aligned BF16 `[K,T]`, `w` has logical shape `[N,K]`, and
 * `out` is contiguous, non-null, 16-byte-aligned BF16 `[N,T]`. Every logical extent is positive;
 * in particular, `T=0` is invalid rather than a no-op. Dimension zero is stored fastest. The Op has
 * no bias, activation, residual addition, or transpose mode.
 *
 * @par Supported execution domain
 * Registered execution uses RowSplit Q4_G64_FP16, Q5_G64_FP16, Q6_G64_FP16, or Q8_G32_FP16 weights
 * with FP16 scales, block-scaled NVFP4 weights, row-scaled FP8_E4M3FN_ROW_BF16 weights, plus
 * registered contiguous BF16 problems. Each format owns a finite registry of exact physical
 * weight problems and selects its kernel internally; a valid encoding and alignment do not imply
 * arbitrary N/K support. FP8 currently registers `[N,K]` in `{[14336,5120], [16384,5120],
 * [34816,5120], [248320,5120], [5120,6144], [5120,17408]}` at every positive T. The current NVFP4
 * problems register the five non-vocabulary FP8 geometries and accept every positive T. NVFP4 also
 * registers the Gemma 4 problems `{[8192,2816], [1024,2816], [2816,4096], [2816,8192],
 * [2816,2112], [16384,5376], [18432,5376], [2048,5376], [43008,5376], [5376,8192], [5376,16384],
 * [5376,21504]}` at every positive T. Q8 also registers `[5120,25600]` at every positive T, and
 * the Gemma 4 problems `{[4224,2816], [8192,2816], [9216,2816], [2816,4096], [2816,8192],
 * [2816,2112], [16384,5376], [18432,5376], [5376,8192], [5376,16384]}` at every positive T,
 * and the Gemma 4 MTP drafter problems `{[1024,5632], [4096,1024], [8192,1024], [1024,4096],
 * [1024,8192], [16384,1024], [2816,1024], [262144,1024], [1024,10752], [1024,16384],
 * [5376,1024], [32768,1024], [65536,1024], [98304,1024], [131072,1024]}` at every positive T,
 * and the Gemma 4 DFlash drafter problems `{[2816,16896], [10240,2816], [6144,2816],
 * [11264,2816], [2816,5632], [5376,32256], [10240,5376], [21504,5376], [5376,10752]}` at every
 * positive T. Q6 registers the Gemma 4 vocabulary problems `[262144,2816]` and `[262144,5376]` at every
 * positive T. BF16 registers `[14336,5120]`,
 * `[5120,6144]`, and `[256,5120]` at every positive T. Text and MTP packed-weight problems accept
 * every positive column extent T. Registered Vision problems accept raw-patch P in
 * `{4,8,...,131072}` or merged-token V in `[1,32768]`; a matrix column does not inherently
 * represent a text token. FP32 is unsupported.
 *
 * @par Numerical contract
 * Test fixture code materializes the persistent weight as its logical FP32 dequantized matrix.
 * The one Linear oracle accepts that matrix and the FP32 values represented by the BF16 activation,
 * evaluates every complete dot product with naive FP64 accumulation, and retains the FP64 result.
 * The BF16 output is promoted and compared against that result. Output representation,
 * accumulator precision, activation quantization, staging, reduction order, and kernel schedule
 * are private implementation effects covered by the named tolerance for the selected
 * activation-compute path; none is copied into the oracle. Kernel, schedule, template instance,
 * host launcher, and T region do not create separate criteria inside one path.
 *
 * @par Compute policy
 * `policy` specifies the permitted private activation-compute set. A permission does not require a
 * corresponding low-precision route: the resolved plan may remain A16 when that is the qualified
 * choice. Every policy permits the existing A16 implementations of BF16 and Q4/Q5/Q6/Q8.
 * FP8 accepts all three policies; AllowA8 and AllowA4 permit the private resolver to select a
 * qualified A16 or A8 route for the registered problem and T. FP8 `[248320,5120]` retains A16
 * compute under every policy at every positive T. NVFP4 uses A16 for A16Only. Under AllowA8 (A8
 * without A4) a call of 17 tokens or more inside a prefill (ops::prefill_routes_active) takes the
 * generic path's int8 W4A8 (activations quantized per 32 values) instead of the registered
 * problem's A16 kernels; every other AllowA8 call (decode, speculative verify of any width) keeps
 * them. AllowA4 permits the private resolver to select either a qualified A16 route or activation
 * quantization to NVFP4 at every positive T. The selected route depends only on the registered
 * problem, T and the route profile. A build without the native NVFP4 kernels runs every NVFP4
 * call on the generic path: A16, or under AllowA8/AllowA4 int8 activations (W4A8 from 17 tokens,
 * R8 at decode).
 *
 * @par Workspace
 * `workspace` is caller-owned call-scoped transient storage sized by
 * linear_workspace_capacity_bytes(). It must not overlap x, any weight plane, or out. Linear does
 * not allocate device memory internally.
 *
 * @param[in] x Contiguous, non-null, 16-byte-aligned BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous, non-null, 16-byte-aligned BF16 output matrix `[N,T]`. It must not
 * overlap `x` or any weight plane.
 * @param[in] policy Permitted private activation-compute profiles.
 * @param[in,out] workspace Caller-owned transient arena.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream);

/**
 * @brief Applies the A16-only form of the bias-free matrix projection.
 *
 * @details This overload admits only A16 compute and requires no transient workspace. All tensor,
 * weight, aliasing, and execution-domain requirements of the policy-bearing overload apply.
 *
 * @param[in] x Contiguous BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous BF16 output matrix `[N,T]`.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

/**
 * @brief ops::linear (policy-bearing overload) whose launch may let the next kernel start early.
 *
 * @details Same result and requirements as linear(). Returns true when the launch triggers
 * programmatic dependent launch of the next kernel on the stream (a generic register-MMA or R8
 * route on a device with PDL): that kernel must then be launched as a programmatic dependent
 * that waits (griddepcontrol.wait) before it reads `out` or anything else this call writes. On
 * false the call ran in plain stream order.
 */
bool linear_release_next(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                         WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninferno::ops
