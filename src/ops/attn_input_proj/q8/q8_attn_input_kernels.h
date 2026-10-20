#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops::detail {

// The widest T of the small-T launchers: the split-K target projection (one kernel instance per
// T = 2..48) and the DFlash2 attention input's small-T form (T = 1..48: one instance per T through
// 16, one per 8-column tile above).
inline constexpr std::int32_t kQ8AttnInputLastTargetExactCols    = 48;
inline constexpr std::int32_t kQ8Dflash2AttnInputLastSmallTokens = 48;

void q8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                 Tensor& v, cudaStream_t stream);
void q8_attn_input_simt_r8_c4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_simt_r8_c4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                     Tensor& v, cudaStream_t stream);
void q8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                     Tensor& v, cudaStream_t stream);
void q8_attn_input_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                       Tensor& v, cudaStream_t stream);
void q8_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_mma_prefill_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& gate, Tensor& k, Tensor& v,
                                              cudaStream_t stream);
void q8_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                       Tensor& v, cudaStream_t stream);
void q8_companion_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream);
void q8_companion_attn_input_mma_r64_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream);
void q8_companion_attn_input_mma_r64_c96_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream);
void q8_companion_attn_input_mma_r128_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                 Tensor& k, Tensor& v, cudaStream_t stream);
void q8_companion_attn_input_mma_r128_c80_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                 Tensor& k, Tensor& v, cudaStream_t stream);
void q8_dflash2_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& k, Tensor& v, cudaStream_t stream);
void q8_dflash2_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream);
void q8_dflash2_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream);

void q8_dflash2_attn_input_mma_r16_c64_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t);

void q8_dflash2_attn_input_mma_r32_c32_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t);

void q8_dflash2_attn_input_mma_r32_c64_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t);

} // namespace ninferno::ops::detail
