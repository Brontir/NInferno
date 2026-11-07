#pragma once

// Helpers of the generic path's route selection (generic_gemm.cpp) that its other files share.

#include "core/device_class.h"
#include "ops/generic/generic_gemm.h"

namespace ninferno::ops::detail::generic {

// The token-major grid of the W4A8 tiles (W4a8Activation::token_major), first the 128-token
// ones on the RTX 4090 (sm_89), where the row-major grid streams a large matrix from DRAM once per
// token tile: Muse 30B
// Q4_K_M shapes (a W4A8 micro-bench, T = 1024 / 512 / 200, time against the row-major grid):
// classic 128x64 Q4_K SwiGLU 19968x6656 0.94 / 0.96 / 0.99, Store 6656x19968 0.97 / 0.97 / 0.98,
// the producer-warp tile Q6_K Store and AddResidual 6656x19968 0.92-0.96; no shape slower (Q5_K
// 6656x4096, 4096x6656 and 256x6656 within 0.5 %). Gemma 4 31B UD-IQ3_XXS (IQ3_XXS and IQ2_S on
// the producer-warp tile, the classic tile's formats besides) prefill 1865 -> 1891 t/s
// at 22.6-31.4k tokens (A B B A, every slice ahead), though per call (the micro-bench at its
// shapes) IQ2_S and IQ4_XS SwiGLU run 1.02-1.05 and IQ3_S down 1.04 against 0.93 for IQ3_XXS and
// IQ3_S SwiGLU: the product decides, every format keeps it. The order of the CTAs changes no bit.
// Every GPU, then: the RTX 5090 (gb202: Gemma 4 31B UD-Q4_K_XL prefill of a 16K prompt 2508 /
// 2484 -> 2595 / 2587 t/s, D TM TM D) gains as the 4090 did. The row-major order stays in the
// kernel for the micro-bench's comparison.
// The 64-token tiles take it too (the classic Q8 tile of q8_prefill_a8 among them): a matrix larger
// than L2 streams from DRAM once per 64 tokens in row-major order, the call DRAM-bound where the
// tile's compute is faster than those bytes. Qwen3.5 9B Q8_0, one 1819-token prompt (nsys, two
// runs each, the same master): RTX 4060 Ti 0.908 -> 0.551 s, 4070 Ti 0.460 -> 0.315, 4090 0.222 ->
// 0.164, 5070 Ti 0.299 -> 0.257, RTX 3060 0.927 -> 0.901 (its tile compute-bound at that rate);
// greedy answers byte-identical (4060 Ti, 4070 Ti, 3060).
inline constexpr bool kW4a8TokenMajor = true;

// The K multiple the register MMA takes: whole steps, except the ggml 32-value blocks, whose
// decoder reads a step's missing blocks as zero and whose groups mma_step stops at K
// (kGroupTail): whole blocks.
int k_multiple(WeightCodec codec);

// The code bits of an I8 weight.
int i8_bits(QType q);

// The current device's features; when no device answers (a host-only caller), every feature, so
// the build's own gates decide as before.
DeviceFeatures current_features();

// The formats the tiled kernel has no tile for (BF16, FP8, NVFP4, MXFP4, Q6_K, Q3_K, the IQ
// formats, ...). On Volta a call of kVoltaCutlassMinTokens or more of one takes the tiled route all
// the same: generic_volta_cutlass decodes any codec's rows to FP16 for CUTLASS (decode_element),
// where their W4A8 tiles ran on dp4a over warp shuffles (V100, Gemma 4 31B Q4_K_XL prefill: the
// Q6_K calls 14.3 ms each, 31 % of it) and BF16 on the staged or register MMA over FP32 FMAs.
// Narrower calls keep their routes.
bool volta_cutlass_only(WeightCodec codec);

// Whether a TiledMma call of `tokens` runs on the decoded copy (and so needs its workspace): a
// CUTLASS-only format's (`only`: it has no tile of its own) from the prefill band's 17 tokens, a
// tiled format's where the copy pays.
bool volta_cutlass_runs(bool only, std::int32_t tokens);

// GemmArgs::add_sums: outside the invariant and prefill profiles, whose calls keep one summation
// order at every T (a call there may change width from one chunk or round to the next), a tensor-core
// register-MMA tile of four or more token tiles adds a min-term format's group sums in FP32 where
// the MMA takes BF16 operands: a BF16 pair widens to FP32 by a shift and a mask, while Turing's f16
// operands (F16Of) each take a conversion. ABBA route sweeps, n4/n8 tiles at T 17..32 against the
// MMA with ones: RTX 3060 0.70-1.25x, 4060 Ti 0.73-1.40x, 5070 Ti 0.60-1.29x (the fastest route
// -3..-16 % on wide heads), RTX 2080 Ti 1.01-1.42x everywhere.
bool rmma_add_sums();

int spread_smem(Route route, WeightCodec codec, Epilogue epilogue, const GemmArgs& args);

// The formats the producer-warp W4A8 tile is compiled for (w4a8_ws_compiled, generic_w4a8.cuh).
bool w4a8_ws_supports(QType q);

int heuristic_rmma(const Problem& problem, std::int32_t tokens, bool prefill_band);

int heuristic_w4a8(const Problem& problem, const Weight& weight, std::int32_t tokens);

int heuristic_r8(const Problem& p, std::int32_t tokens);

// The codecs the tiled route takes on this GPU: tiled_supports, and Q8 where tensor mma.sync is
// missing (Volta): elsewhere Q8 has its own kernels, on Volta those tiles run on FMAs.
bool tiled_takes(WeightCodec codec);

// The I8 tile of a call: the one with the fewest tile-waves weighted by its tile's work over its
// share of the int8 peak (kI8Eff), of those that fit the build: the largest tile while its grid
// fills the SMs, a smaller one where it would leave them idle. Every schedule gives the same
// bytes, so the choice may vary with T. -1 where none fits.
int i8_pick(const Problem& problem, QType qtype, std::int32_t tokens);

// The route cost model (generic_cost.cpp). price_routes hands `take` every admitted route it
// prices with its time (the A16 scale applied); false where the facts have no roofline (nothing
// priced). cost_route: the cheapest of them.
using PricedRoute = std::function<void(Route, double)>;
bool price_routes(const Problem& problem, const Weight& weight, std::int32_t tokens,
                  const PricedRoute& take);
std::optional<Route> cost_route(const Problem& problem, const Weight& weight, std::int32_t tokens);

// The weight-side admission of the A8 tiles (generic_gemm.cpp; their refusals name the reason).
bool allows_w4a8(const Weight& w, LinearPolicy policy);
bool allows_i8(const Weight& w, LinearPolicy policy);

} // namespace ninferno::ops::detail::generic
