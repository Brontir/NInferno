#pragma once

// Helpers of the generic path's route selection (generic_gemm.cpp) that its other files share.

#include "core/device_class.h"
#include "core/measured_boards.h"
#include "ops/generic/generic_gemm.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>

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

// The R8 schedule of a call: at one token the nearest measured board's decode table entry
// (tuned_r8), else the R8 rules (r8_rules_route).
int heuristic_r8(const Problem& p, std::int32_t tokens);

// The R8 decode table's entry for the exact problem (r8_decode_table.h), from the table of the
// nearest measured board within kR8CarryDistance: the schedule, the board's distance to this
// device (0: the device's own) and the table's board id ("gb202:170sm"); nullopt where no table
// is near enough or none holds the problem.
struct R8TableHit {
    int schedule;
    double distance;
    std::string_view gpu;
};

std::optional<R8TableHit> tuned_r8(const Problem& p);

// The route selection's rules and fixed routes, which route_authorities.cpp asks in order
// (generic_gemm.cpp holds the bodies and their measurements).
// Whether a call of T tokens takes R8 (decode, T <= kR8DecodeTokens, and the activation fits;
// r8_call_refusal, generic_gemm.h, names why not).
bool r8_call(const Weight& w, LinearPolicy policy, std::int32_t tokens);
// The verify widths the vector kernel takes by rule: 2..4 tokens with tensor mma.sync, every
// width it reads without.
bool vec_rule(const Weight& w, LinearPolicy policy, std::int32_t tokens);
// ggml's matrix kernel for the verify band: 9..16 tokens of the codebook formats in a round of
// several sequences (`lanes`: ops::round_lanes(), which the four-argument form reads itself); on
// an emulated BF16 MMA every mmq format from 9 tokens and the codebook formats from 5.
bool mmq_verify_rule(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens);
bool mmq_verify_rule(const Weight& w, LinearPolicy policy, std::int32_t rows, std::int32_t tokens,
                     int lanes);
// Whether a tiled-codec call outside the prefill profile takes the tiled route: from
// kTiledSpeedMinTokens on a native BF16 MMA where its 128x128 tiles fill the SMs (kSpeedFill).
bool tiled_speed_pays(std::int32_t n, std::int32_t tokens);
// Outside the prefill profile the staged MMA tile of a BF16 weight from kRmmaMidMinTokens, the
// largest whose grid still fills the SMs; nullopt where none does or none admits the weight.
std::optional<Route> staged_speed_route(const Problem& problem, const Weight& weight,
                                        std::int32_t tokens);
// The invariant profile's register MMA: the decode split of K for every T.
int invariant_rmma(std::int32_t k, std::int32_t tokens);
// The W4A4 tile: the widest under the prefill band, else the widest that still gives every SM a
// CTA.
int heuristic_w4a4(std::int32_t rows, std::int32_t tokens, bool prefill_band);
// Index of the register-MMA schedule with this shape (generic_gemm_launch.cuh's list); 0 where
// none has it. A cluster split asks for the C == 1 schedule on a build without them.
int rmma_index(int warps_m, int warps_k, int m_tiles, int n_tiles, int cluster = 1);
// Whether the device's BF16 MMA is native (sm_80 and up); a device that cannot be queried counts
// as native.
bool bf16_mma_native();
// The NVFP4 tiled route in the order of the register MMA with WarpsK = `classes`, for calls of two
// token tiles or more whose tiles give every SM at least one CTA; nullopt otherwise.
std::optional<Route> nvfp4_tiled_route(const Problem& problem, const Weight& weight, int classes,
                                       std::int32_t tokens);
// Volta's HMMA route of the verify widths (1..16 tokens of Q4_K / Q5_K / Q6_K or Q8 G32 without
// tensor mma.sync, outside the vector kernel's and R8's widths); nullopt elsewhere.
std::optional<Route> volta_hmma_route(const Problem& problem, const Weight& weight,
                                      std::int32_t tokens);
// The maintainer's route pin for the call's band (ops/route_pins.h), where its family can compute
// the call; notes the pin ledger. nullopt: no pin, or the band's pin cannot compute it.
std::optional<Route> pinned_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                                  bool prefill);
// The automatic selection after the profiles, the probe and Volta's fixed routes
// (route_authorities.cpp): the route tables, the dense tuner and the rules.
Route auto_route(const Problem& problem, const Weight& weight, std::int32_t tokens, bool prefill);
// The MMQ prefill gate's answer for the current device (core/measured_boards.h
// mmq_prefill_gate), with its donor and disposition; the unmeasured default where no device
// answers.
measured::GateAnswer mmq_prefill_device_gate();

// The codecs the tiled route takes on this GPU: tiled_supports, and Q8 where tensor mma.sync is
// missing (Volta): elsewhere Q8 has its own kernels, on Volta those tiles run on FMAs.
bool tiled_takes(WeightCodec codec);

// The I8 tile of a call: the one with the fewest tile-waves weighted by its tile's work over its
// share of the int8 peak (kI8Eff), of those that fit the build: the largest tile while its grid
// fills the SMs, a smaller one where it would leave them idle. Every schedule gives the same
// bytes, so the choice may vary with T. -1 where none fits.
int i8_pick(const Problem& problem, QType qtype, std::int32_t tokens);

// The quantized activation's planes a route reads from the caller's workspace, in bytes, in the
// order run_route allocates them (generic_workspace.cpp; W4A4 has no sums plane: 0). Shared so
// that run_route (generic_gemm.cpp) allocates by the description the workspace query sizes.
struct ActivationPlanes {
    std::size_t codes = 0, scales = 0, sums = 0;
};

ActivationPlanes activation_planes(Route route, std::int32_t k, std::int32_t tokens);

// Shared by the selection and the workspace queries (generic_gemm.cpp).
std::int32_t vec_max_tokens();
bool volta_tiled_qtype(QType qtype);
bool volta_cutlass_qtype(QType qtype);
bool volta_cutlass_only_qtype(QType qtype);

// The route cost model (generic_cost.cpp). price_routes hands `take` every admitted route it
// prices with its time (the A16 scale applied); false where the facts have no roofline (nothing
// priced). cost_route: the cheapest of them.
using PricedRoute = std::function<void(Route, double)>;
bool price_routes(const Problem& problem, const Weight& weight, std::int32_t tokens,
                  const PricedRoute& take);
std::optional<Route> cost_route(const Problem& problem, const Weight& weight, std::int32_t tokens);

// The weight-side admission of the A8 and A4 tiles and of R8 (generic_gemm.cpp; their refusals
// name the reason).
bool allows_w4a8(const Weight& w, LinearPolicy policy);
bool allows_i8(const Weight& w, LinearPolicy policy);
bool allows_w4a4(const Weight& w, LinearPolicy policy);
bool allows_r8(const Weight& w, LinearPolicy policy);

} // namespace ninferno::ops::detail::generic
