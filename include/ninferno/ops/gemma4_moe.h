#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops {

/**
 * The stored form of one Gemma 4 26B-A4B sparse-MoE layer: 128 experts over a 2816-wide hidden
 * state, each with a 704-row GeGLU intermediate.
 *
 * - router: BF16 contiguous [128, 2816].
 * - router_scale: FP32 [2816], the router's per-channel scale.
 * - expert_scale: FP32 [128], the gain applied to a selected expert's route weight.
 * - gate_up: NVFP4 stacked [128*1408, 2816]. Expert e owns rows [e*1408, e*1408+704) as its gate
 *   and [e*1408+704, (e+1)*1408) as its up projection, and one weight divisor
 *   (weight_divisor_rows = 1408).
 * - down: NVFP4 stacked [128*2816, 704]. Expert e owns rows [e*2816, (e+1)*2816) and one weight
 *   divisor (weight_divisor_rows = 2816).
 * - gate_up_input_divisors, down_input_divisors: FP32 [128] each, every expert's activation
 *   global scale (the checkpoint's input_global_scale) for the input of its gate/up and of its
 *   down projection. They carry no semantics - the oracle is exact - and only the route that
 *   encodes activations in four bits reads them, expert by expert.
 */
struct Gemma4MoeWeights {
    Weight router;
    Tensor router_scale;
    Tensor expert_scale;
    Weight gate_up;
    Weight down;
    Tensor gate_up_input_divisors;
    Tensor down_input_divisors;
};

/**
 * How the prefill route (T > 16) of gemma4_moe represents the expert inputs.
 *
 * - Recorded: each expert's gate/up input and its GeGLU product are encoded to NVFP4 against the
 *   checkpoint's per-expert activation divisors (W4A4), as the NVFP4 checkpoint was calibrated.
 * - A16Only: the activations stay BF16 against the stored NVFP4 weights decoded exactly (W4A16).
 *
 * Decode (T = 1) and the small-T route (2 <= T <= 16) keep every activation represented under
 * either value; only calls of more than 16 tokens differ.
 */
enum class Gemma4MoeActivations : std::uint8_t { Recorded, A16Only };

/**
 * Transient capacity required by gemma4_moe for every T in the inclusive [min_tokens,max_tokens]
 * interval, under either Gemma4MoeActivations value and either route profile. Invalid intervals
 * throw.
 */
[[nodiscard]] std::size_t gemma4_moe_workspace_bytes(std::int32_t min_tokens,
                                                     std::int32_t max_tokens);

/**
 * Whether this build runs gemma4_moe (its NVFP4 expert banks are sm_120a kernels). Elsewhere a
 * Gemma 4 MoE layer takes the generic moe Op, whatever its expert format.
 */
[[nodiscard]] bool gemma4_moe_available() noexcept;

/**
 * Closed sparse-MoE Op for the exact Gemma 4 26B-A4B geometry.
 *
 * For contiguous BF16 router_input and expert_input [2816,T] and a BF16 destination out [2816,T],
 * T>0, every column t is independent (x_r = router_input[:,t], x_e = expert_input[:,t]):
 *
 *     r        = x_r / sqrt(mean(x_r^2) + eps) * router_scale * 2816^-0.5
 *     s        = router . r                          (128 logits)
 *     p        = softmax(s)
 *     top      = the 8 largest p; at an exact tie the lower expert id wins
 *     w_k      = p_k / sum_{top} p * expert_scale[id_k]
 *     out[:,t] = sum_k w_k * Down_k(gelu_tanh(Gate_k x_e) * (Up_k x_e)),  k over id_k
 *
 * with gelu_tanh(z) = 0.5 z (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3))). out is fully overwritten;
 * it is not accumulated into a residual. The oracle starts from the represented BF16 inputs and
 * the exact stored-weight decode and evaluates the formula without rounding. Intermediate
 * representation, reduction order and activation encoding are private execution choices.
 *
 * Any other geometry or storage format is rejected with std::invalid_argument before a launch.
 * The inputs (which may be one tensor), out, every weight plane and the live workspace must not
 * overlap. Execution is enqueued on stream without host synchronisation and is CUDA-graph
 * capturable; workspace is caller-owned transient storage that carries no state between calls.
 *
 * `activations` picks the prefill route's activation encoding (see Gemma4MoeActivations). Under
 * either value a column's output does not depend on how many other columns share a call of more
 * than 16 tokens. Under the invariant route profile (ninferno/ops/prefill_routes.h) every call, a
 * single token included, takes the A16 prefill route, so it does not depend on T at all.
 */
void gemma4_moe(const Tensor& router_input, const Tensor& expert_input,
                const Gemma4MoeWeights& weights, float eps, Tensor& out, WorkspaceArena& workspace,
                cudaStream_t stream,
                Gemma4MoeActivations activations = Gemma4MoeActivations::Recorded);

} // namespace ninferno::ops
