#pragma once
//
// Integer-activation route for the Qwen3.8/3.6-27B mlp/gate_up LinearSwiGLU. Ported from
// ninfer-all (Apache-2.0, see NOTICE), src/ops/linear_swiglu/q4a8, unchanged but for the namespace.
//
// Reads the artifact's native Q4_G64_FP16 row-split weight with no repack: for Q4 that layout's
// code plane is exactly row-major [n][k/2] with two codes per byte, and its scale plane exactly
// row-major [n][k/64] FP16, so Weight::qdata and Weight::scales can be indexed directly.
//
// Activations are quantised to s8 with one scale per (token, group of 64) so that an outlier
// channel can only spoil its own group, then fed to mma.m16n8k32.s32.s8.s8.s32 -- about 4.7x the
// rate of the bf16 f32-accumulate MMA the A16 route issues on this hardware.
//
// This adds activation quantisation error the A16 route does not have: about 0.9% relative L2,
// against upstream's own kA8QuantizationAllowance of 4% for A8 activation compute (see
// tests/ops/linear/linear_test_common.cpp). Ours takes it only where the prefill
// int8 tier is on (prefill_a8_policy; --activations exact keeps A16).

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail {

// True when this route can serve the call: the registered 27B gate_up profile, a row-split Q4G64
// weight, and a token count the 128-wide tiling covers exactly. Partial prefill chunks and decode
// fall through to the A16 route.
// Whether ninfer-all's int8 prefill kernels beat the generic route on this card (gate_up; the
// LinearAdds). Measured per card, 27B q4_g64 24K prefill; unlisted cards keep the generic route.
[[nodiscard]] bool q4a8_swiglu_pays();
[[nodiscard]] bool a8_add_pays();

[[nodiscard]] bool q4a8_swiglu_supported(const Weight& gate_up, std::int32_t tokens);

// Transient bytes this route needs for the quantised activation planes over [min,max] tokens.
[[nodiscard]] std::size_t q4a8_swiglu_workspace_capacity_bytes(std::int32_t min_tokens,
                                                               std::int32_t max_tokens);

// x is contiguous BF16 [k, tokens]; out is contiguous BF16 [n/2, tokens].
void q4a8_swiglu_launch(const Tensor& x, const Weight& gate_up, Tensor& out,
                        WorkspaceArena& workspace, cudaStream_t stream);

// Companion route for the Q4G64 and Q5G64 row-split LinearAdds: mlp/down [5120,17408] and the
// attention o_proj / GDN out_proj [5120,6144], which differ only in K. A Q4 code is a signed nibble
// in [-8,7]; Q5 adds an 8-byte-per-group high plane carrying each code's fifth bit, and a code
// decodes as ((low4 | hbit << 4) ^ 0x10) - 0x10 over [-16,15]. Int8 holds both exactly.
[[nodiscard]] bool a8_add_supported(const Weight& down, std::int32_t tokens);
[[nodiscard]] std::size_t a8_add_workspace_capacity_bytes(std::int32_t input_rows,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);
void a8_add_launch(const Tensor& x, const Weight& down, Tensor& residual, WorkspaceArena& workspace,
                   cudaStream_t stream);

// ninfer-all's int8 routes for the 27B Q4/Q5 G64 input projections (attention q/k + gate/v, GDN
// qk + value/z): one activation quantisation, one a8 launch per output range. Where the card is
// listed (a8_input_pays) they replace the generic int8 tier's two parent GEMMs.
[[nodiscard]] bool a8_input_pays();
[[nodiscard]] bool a8_attn_input_supported(const Weight& query_key, const Weight& gate_value,
                                           std::int32_t tokens);
[[nodiscard]] bool a8_gdn_input_supported(const Weight& qk, const Weight& value_z,
                                          std::int32_t tokens);
[[nodiscard]] std::size_t a8_input_workspace_capacity_bytes(std::int32_t min_tokens,
                                                            std::int32_t max_tokens);
void a8_attn_input_launch(const Tensor& x, const Weight& query_key, const Weight& gate_value,
                          Tensor& q, Tensor& gate, Tensor& k, Tensor& v, WorkspaceArena& workspace,
                          cudaStream_t stream);
void a8_gdn_input_launch(const Tensor& x, const Weight& qk, const Weight& value_z, Tensor& qkv,
                         Tensor& z, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninferno::ops::detail
