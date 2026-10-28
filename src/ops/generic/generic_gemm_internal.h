#pragma once

// Private launch entry points of the generic projection path, one per compiled family and format.

#include "ops/generic/generic_gemm.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninferno::ops::detail::generic {

// out [K, T] BF16 = x [K, T] (leading dimension ldx) gathered by K int32 column indices.
void launch_gather_k(const void* x, int ldx, const void* order, int k, int tokens, void* out,
                     cudaStream_t stream);
void launch_rmma_bf16(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_fp8(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_nvfp4(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q4(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q5(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q6(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q3g128(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q3g64(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q8(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_mxfp4(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_fp8_row_f32(int schedule, Epilogue epilogue, const GemmArgs& args,
                             cudaStream_t stream);
void launch_rmma_fp8_b128(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_u4zp(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_u4zp_g32(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_u4zp_g64(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q4k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q5k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q6k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q3k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q2k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq4xs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq3xxs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq3s(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q40(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q41(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q50(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q51(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq4nl(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq2xxs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq2xs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq2s(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq1s(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq1m(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_tq10(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_tq20(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q10(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q20(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);

void launch_scalar(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream);

// The DRAM bytes the device's bandwidth asks of each SM per cycle (MachineFacts: bandwidth over
// SMs times the reference clock); 0 when unknown. Once per device.
[[nodiscard]] double dram_bytes_per_sm_cycle();

// The DRAM bytes per SM cycle below which the wide-load Q6_K (GgmlQ6KWideDecoder) of one token
// tile pays: there the plain kernel's twice-fetched sectors fill the load queue (ncu, RTX 4090,
// Gemma 4 12B ffn_down 3840x15360 at T = 6: lg_throttle a third of the stalls, 54 % of DRAM).
// A/B, T = 1..8, bytes per SM cycle: RTX 4090 3.1 (that add 71 -> 59 us), 5070 Ti 5.2 (every
// problem 1-8 % faster); on the 3060 (7.2), whose plain kernel streams at 84 % of DRAM, the
// extra shuffles cost 2-10 %. The boundary of rmma_latency_pays' one row set, from the other side.
inline constexpr double kRmmaWideBytes = 5.5;

// rmma_wide_pays' rule on its inputs: below kRmmaWideBytes (0: unknown, no), not before sm_75
// (Volta's register MMA is FMA-emulated, compute-bound), and only where the wide kernel holds as
// many CTAs per SM as the plain one and spills at most kRmmaWideSpillSlack bytes a thread more
// (sm_89's one-row-set tiles, held to 128 registers, keep 8 bytes of loop-invariant state in
// local memory; SwiGLU's 128-register tiles spill 160 bytes in the loop, RTX 4090 w1x16 at
// T = 6 100.3 -> 103.9 us).
inline constexpr std::size_t kRmmaWideSpillSlack = 16;
[[nodiscard]] bool rmma_wide_rule(double bytes_per_sm_cycle, int compute_capability, int wide_ctas,
                                  int plain_ctas, std::size_t wide_local_bytes,
                                  std::size_t plain_local_bytes);

// Volta's warp-per-row GEMV of the ggml Q4_K / Q5_K / Q6_K, Q8 G32 and BF16 formats at T <= 8
// (generic_volta_gemv.cu), in place of a lane-local register-MMA call.
bool volta_gemv_takes(WeightCodec codec, const GemmArgs& args);
void launch_volta_gemv(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                       cudaStream_t stream);

// Tiled A16 (generic_tiled.cu); the codec must be one tiled_supports() admits.
void launch_tiled(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                  cudaStream_t stream);

// Volta prefill on CUTLASS's sm_70 GEMM over weight rows decoded once to FP16
// (generic_volta_cutlass.cu; built for sm_70 alone). launch_volta_cutlass returns false, having
// launched nothing, when the build lacks it, the GEMM refuses the shape or the workspace left
// cannot hold one chunk; the tiled route then takes the call.
bool volta_cutlass_compiled() noexcept;
std::size_t volta_cutlass_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens,
                                          bool swiglu);
bool launch_volta_cutlass(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                          WorkspaceArena& workspace, cudaStream_t stream);

// Staged register MMA (generic_staged.cu): BF16 weights only.
void launch_staged(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);

// NVFP4 A16 in shared-memory activation tiles (generic_nvfp4_tiled.cu), summing in the order of
// the register MMA with WarpsK = classes (1, 4 or 8).
void launch_nvfp4_tiled(int classes, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);

// NVFP4 W4A4: quantize the activation into `codes`/`scales` (row-major, [T,K/2] and [T,K/16]),
// then run the schedule.
void launch_w4a4(int schedule, Epilogue epilogue, const GemmArgs& args, float input_divisor,
                 std::uint8_t* codes, std::uint8_t* scales, cudaStream_t stream);

// The W4A8 activation planes' row length: K rounded up to whole 128-value tiles (w4a8_act_ld).
inline int w4a8_ld(int k) { return (k + 127) / 128 * 128; }

// W4A8 (generic_w4a8.cu): quantize the activation into `codes` [T,ld] int8, `scales` [T,ld/32]
// (FP32 amax/127) and `sums` [T,ld/32] (the scale times the codes' sum; ld = w4a8_ld(K), the tail
// blocks zero), then run the schedule. The
// codec must be one w4a8_supports() admits. token_major: the grid order of the 128-token tiles
// (W4a8Activation::token_major).
void launch_w4a8(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                 std::int8_t* codes, float* scales, float* sums, bool token_major,
                 cudaStream_t stream);
// The I8 route: quantize x per 64 values into [padded T][K] codes (k-permuted), [K/64][padded T]
// scales and sums, then the tile (generic_i8.cu).
void launch_i8(int schedule, int bits, Epilogue epilogue, const GemmArgs& args, std::int8_t* codes,
               float* scales, int* sums, cudaStream_t stream);
// The W4A8 activation quantizer alone (generic_w4a8.cu): args.tokens x args.k values into codes,
// scales and sums, rows of `ld` values (0: args.k; the blocks past args.k zero).
void launch_w4a8_quantize(const GemmArgs& args, std::int8_t* codes, float* scales, float* sums,
                          cudaStream_t stream, int ld = 0);
// R8 (generic_r8.cu): W8A8 decode for T <= 8, the activation quantized inside the kernel, or, with
// `pre`, read from a pre-quantized activation (verify widths, r8_verify_supports formats). The
// codec must be one r8_supports() admits; the iq3/iq2/ggml4/kquant entries return false for
// other codecs.
void launch_r8(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
               cudaStream_t stream, W4a8Activation pre = {});
bool launch_r8_iq3(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream, W4a8Activation pre);
bool launch_r8_iq2(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream, W4a8Activation pre);
bool launch_r8_ggml4(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                     cudaStream_t stream, W4a8Activation pre);
bool launch_r8_kquant(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                      cudaStream_t stream, W4a8Activation pre);

// Family::GgmlMmq's epilogue: the FP32 plane [T][plane_rows] (SwiGlu: gate rows, then up rows from
// args.rows) to the call's BF16 segments.
void launch_mmq_epilogue(Epilogue epilogue, const GemmArgs& args, const float* plane,
                         int plane_rows, cudaStream_t stream);

} // namespace ninferno::ops::detail::generic
