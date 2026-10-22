#pragma once

// Generic projection path: every registered weight format at any legal [N,K], any T.
//
// The shape-owned tables under src/ops/linear*/ stay the fast paths and win whenever they hold the
// exact shape. Everything else lands here: one kernel family whose tile is chosen by shape class
// (and, when a route table is loaded, by a tuned entry for the exact shape), see
// generic_gemm.cuh for the arithmetic and route_table.h for the tables.

#include "core/arena.h"
#include "core/device_class.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ninferno/option_enums.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::ops::detail::generic {

inline constexpr int kMaxSegments = 4;

// The widths where the speed profile's heuristic choices change (generic_gemm.cpp, below the route
// table's bands): the register MMA's stream regime through 32 tokens (through 512 where the BF16
// MMA is emulated), its 64-row tiles and a tiled codec's or BF16's staged tiles from 33 where
// they fill the SMs (heuristic_rmma, tiled_speed_pays, staged_speed_route), the staged MMA's
// fallback from 256, and the W4A8 lookup/K-quant formats their producer-warp and 128x128 tiles
// from 128 (128x64 from 65, heuristic_w4a8). kSpeedFill: the waves of the SMs a tile's grid
// fills before it pays (1.5; 1.0..2.0 cost under 0.5 %).
inline constexpr std::int32_t kRmmaMidMinTokens     = 33;
inline constexpr std::int32_t kStreamMaxTokens      = 512;
inline constexpr std::int32_t kTiledSpeedMinTokens  = 33;
inline constexpr std::int32_t kStagedSpeedMinTokens = 256;
inline constexpr double kSpeedFill                  = 1.5;
inline constexpr std::int32_t kW4a8WideMinTokens    = 128;
// The widest call whose register-MMA launch prefetches under the previous kernel
// (GemmArgs::pdl_prefetch): decode and speculative verify.
inline constexpr std::int32_t kPdlPrefetchMaxTokens = 16;

enum class Epilogue : std::uint8_t {
    Store,       // out = W x (GemmArgs::relu2: relu(W x)^2, one rounding)
    AddResidual, // out += W x (out is the residual, updated in place)
    SwiGlu,      // out[i] = SiLU((W x)[i]) * (W x)[i + up_offset], one rounding (GemmArgs::
                 // gelu_gate: GELU's tanh form in place of SiLU, GeGLU)
};

enum class WeightCodec : std::uint8_t {
    Bf16,
    Fp8,
    Nvfp4,
    Q4,
    Q5,
    Q6,
    Q8,
    Mxfp4,
    Fp8RowF32,
    Fp8B128,
    U4Zp,
    GgmlQ4K,
    GgmlQ5K,
    GgmlQ6K,
    GgmlQ3K,
    GgmlQ2K,
    GgmlIq4Xs,
    GgmlIq3Xxs,
    GgmlIq3S,
    GgmlQ40,
    GgmlQ41,
    GgmlQ50,
    GgmlQ51,
    GgmlIq4Nl,
    GgmlIq2Xxs,
    GgmlIq2Xs,
    GgmlIq2S,
    GgmlIq1S,
    GgmlIq1M,
    GgmlTq10,
    GgmlTq20,
    GgmlQ10,
    GgmlQ20,
    U4ZpG32,
    U4ZpG64,
    Q3G128,
    Q3G64,
};

// Device-visible arguments of one generic call. Plain data, passed by value.
struct GemmArgs {
    // Activation: token t starts at x + t * ldx (BF16).
    const void* x = nullptr;
    int ldx       = 0;
    int k         = 0;
    int tokens    = 0;
    // Weight planes of the parent matrix.
    const std::uint8_t* codes  = nullptr;
    const std::uint8_t* high   = nullptr;
    const std::uint8_t* scales = nullptr;
    int groups_per_row = 0; // RowSplit: padded K / group; RowGroupZp: K / group; ggml: K / 256
    int scale_tiles            = 0;  // NVFP4: K / 64; FP8 128x128 blocks: ceil(K / 128)
    float inverse_divisor      = 1.0F;
    const float* divisors      = nullptr; // NVFP4 stacked plane: one divisor per divisor_rows rows
    int divisor_rows           = 0;
    // Rows: local row l of the call reads parent row row_begin + l; SwiGLU's up row is
    // row_begin + l + up_offset.
    int row_begin = 0;
    int rows      = 0;
    int up_offset = 0;
    // SwiGlu epilogue: the gate's GELU (tanh form) in place of its SiLU (GeGLU).
    int gelu_gate = 0;
    // Store epilogue: relu(v)^2 before the rounding (a non-gated relu^2 projection).
    int relu2 = 0;
    // Output segments: local rows [segment_end[s-1], segment_end[s]) go to out[s], BF16 with token
    // stride ld[s] elements (a segment may be a row range of a wider [ld, T] tensor).
    void* out[kMaxSegments]         = {};
    int segment_end[kMaxSegments]   = {};
    int ld[kMaxSegments]            = {};
    int segments                    = 1;
    // Register MMA, Store epilogue only: the launch is a programmatic dependent of the kernel
    // before it (Call::join_previous) and waits for it just before exiting; release_next lets the
    // next launch start once every CTA of this one has (Call::release_next).
    int join_previous = 0;
    int release_next  = 0;
    // Register MMA, decode widths (never with join_previous): the launch is a programmatic
    // dependent of the kernel before it. Its CTAs issue the weight ring's first loads (weights
    // only: complete before any kernel of the step ran), then wait for that kernel, and let the
    // next launch start once every CTA of this one is past that wait. The weight stream of the
    // call starts under the previous kernel instead of after it (residual_norm and sigmoid_mul
    // let their dependents launch at their start). Only the ring's first slot: fetching the second
    // before the wait too measured slower (Muse Glimmer 30B NVFP4 decode on an RTX 5090: 80.0 ->
    // 77.7 t/s at C=1, 74.3 -> 71.5 at C=4).
    // Audit: it is a runtime flag, so tools/pdl_sass_audit.py also follows the path of a launch
    // without it, where the wait is skipped (a plain launch: nothing is a running producer's)
    // and x is read with no wait before it. tools/pdl_prewait_loads.txt therefore allows x (0, 8,
    // 12) before the wait for every epilogue, and the audit cannot catch a read of x hoisted above
    // the wait on the pdl_prefetch path; the NVFP4 w1x8 SASS was checked by hand (every load
    // through x's c[0x0][0x380] follows ACQBULK).
    // A compile-time variant (a Schedule or kernel template flag, compiled for the decode schedules
    // only: w1x8 n1, its cluster form, w1x4 n2) would let the audit hold x to after the wait on
    // that path; it is not built, since for every schedule it would double the rmma_kernel
    // instances.
    int pdl_prefetch = 0;
    // Host side only (the kernel reads none): dynamic shared memory each CTA of the launch
    // reserves, so that an SM holds fewer of its CTAs (generic_gemm.cpp, run_route).
    int spread_smem = 0;
    // Register MMA without tensor mma.sync (sm_70): a one-tile CTA of at most four tokens sums lane
    // by lane (generic_gemm.cuh mma_step_local), not in the order of a wider call's emulated MMA.
    // run_route sets it outside the invariant profile, whose calls keep one order at every T.
    int lane_local = 0;
};

// An activation quantized to int8 per 32 values (the W4A8 quantizer's output): what the W4A8
// kernel reads, and R8 at verify widths.
struct W4a8Activation {
    const std::int8_t* codes = nullptr; // [T, K]
    const float* scales      = nullptr; // [T, K/32]
    const float* sums        = nullptr; // [T, K/32], scale times the codes' sum (min-term formats)
    // W4A8 tiles of 128 tokens: the grid runs token tiles along x, so the CTAs of a wave are the
    // token tiles of a few weight row blocks and read those rows once from DRAM (else row blocks
    // along x: every token tile streams the whole matrix). Changes no bit (w4a8_grid).
    int token_major = 0;
};

// A route is one compiled schedule of one kernel family.
enum class Family : std::uint8_t {
    RegisterMma, // generic_gemm.cuh rmma_kernel, A16
    Scalar,      // correctness fallback for K the register MMA cannot tile
    W4a4Mma,     // NVFP4 weights with NVFP4-quantized activations (AllowA4)
    TiledMma,    // generic_tiled.cuh: shared-memory BF16 tensor-core tiles, A16, prefill-sized T
    W4a8Mma,     // generic_w4a8.cuh: 2- to 8-bit weights with int8 activations (AllowA8)
    StagedMma, // generic_staged.cuh: BF16 weights, the register MMA's arithmetic from shared memory
    // generic_nvfp4_tiled.cu: NVFP4 A16 in shared-memory activation tiles, prefill-sized T, summing
    // in the order of the register MMA with WarpsK = classes (nvfp4_a16_tiled.cuh).
    Nvfp4TiledMma,
    R8Mma, // generic_r8.cuh: q8/codebook weights with int8 activations, T <= 8 (AllowA8)
    // ggml_mmq/: ggml's int8 tensor-core tiles with stream-k over K (third_party/ggml-quants), the
    // ggml block formats at speculative-verify widths (AllowA8), an FP32 plane then the epilogue.
    GgmlMmq,
    // ggml_mmq/ggml_vec.cu: ggml's dp4a vector kernel (after ninfer-all's), the ggml block
    // formats at 1..8 tokens (AllowA8), the epilogue in the kernel; the dense tuner's candidate.
    GgmlVec,
    // generic_i8.cu: Q4 G64 weights with int8 activations scaled per 64 values, the code as the
    // MMA's unsigned operand, from 17 tokens (AllowA8).
    I8Mma,
};

struct Route {
    Family family         = Family::RegisterMma;
    std::uint8_t schedule = 0; // index into the family's compiled schedule list
};

bool operator==(Route a, Route b);

// Compiled schedules. RegisterMma: (warps_m, warps_k, m_tiles, n_tiles, cluster: K split over a
// thread-block cluster of that many CTAs); W4a4Mma: (block_m, block_n) with warps fixed per entry.
struct RmmaScheduleInfo {
    int warps_m, warps_k, m_tiles, n_tiles;
    int cluster = 1;
};
struct W4a4ScheduleInfo {
    int block_m, block_n;
};
// TiledMma: block_m tokens by block_n weight rows.
struct TiledScheduleInfo {
    int block_m, block_n;
};
// W4a8Mma: block_m tokens by block_n weight rows; producers: 0 for the kernel whose every warp
// loads and multiplies, else the producer warps of the warp-specialized kernel.
struct W4a8ScheduleInfo {
    int block_m, block_n;
    int producers = 0;
};
// StagedMma: rows (weight rows staged per CTA; SwiGLU's gate and up rows together) by tokens, on a
// warps_m x warps_n warp grid, with `stages` K tiles in flight.
struct StagedScheduleInfo {
    int rows, tokens, warps_m, warps_n, stages;
};
// Nvfp4TiledMma: the register MMA order it reproduces (classes = WarpsK), block_m tokens by
// block_n weight rows.
struct Nvfp4TiledScheduleInfo {
    int classes, block_m, block_n;
};
// R8Mma: warps per CTA, rows per warp.
struct R8ScheduleInfo {
    int warps, rows;
};
[[nodiscard]] int rmma_schedule_count();
[[nodiscard]] RmmaScheduleInfo rmma_schedule_info(int index);
// Whether this build can launch register-MMA schedule `index`: a thread-block cluster split
// (cluster > 1) needs sm_90+, so the Ada build has none.
[[nodiscard]] bool rmma_schedule_launchable(int index);
[[nodiscard]] int w4a4_schedule_count();
// Every W4A4 schedule, compiled or not: the names parse_route takes on any build (a shipped
// Blackwell table names them); w4a4_schedule_count is 0 on a build without the kernels.
[[nodiscard]] int w4a4_schedule_names();
[[nodiscard]] W4a4ScheduleInfo w4a4_schedule_info(int index);
[[nodiscard]] int tiled_schedule_count();
[[nodiscard]] TiledScheduleInfo tiled_schedule_info(int index);
// The I8 tiles: tokens x weight rows per CTA, threads, cp.async stages, dynamic shared memory.
struct I8ScheduleInfo {
    int block_m;
    int block_n;
    int threads;
    int stages;
    std::size_t smem_bytes;
};
[[nodiscard]] int i8_schedule_count();
[[nodiscard]] I8ScheduleInfo i8_schedule_info(int index);
// Whether an I8 schedule fits one block's shared memory for a format with at least three stages:
// Q5 / Q6 add their high plane to every stage (Q6 at 256x128 runs three).
[[nodiscard]] bool i8_fits(int schedule, QType qtype);
// The I8 quantizer pads a call's tokens to whole 256-token tiles (zero codes, scales and sums).
[[nodiscard]] constexpr int i8_padded_tokens(int tokens) { return (tokens + 255) / 256 * 256; }
[[nodiscard]] int w4a8_schedule_count();
[[nodiscard]] W4a8ScheduleInfo w4a8_schedule_info(int index);
[[nodiscard]] int nvfp4_tiled_schedule_count();
[[nodiscard]] Nvfp4TiledScheduleInfo nvfp4_tiled_schedule_info(int index);
[[nodiscard]] int staged_schedule_count();
[[nodiscard]] StagedScheduleInfo staged_schedule_info(int index);
[[nodiscard]] int r8_schedule_count();
[[nodiscard]] R8ScheduleInfo r8_schedule_info(int index);
// Whether a weight format has a W4A8 codec: a scale (and min) per 32 values (Q6_K, Q3_K, Q2_K,
// IQ2_XS, IQ2_S: per 16) over integer codes that fit int8 (ggml Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS/
// Q4_K/Q5_K/Q6_K/Q3_K/Q2_K, MXFP4, u4_g128_zp, q8_g32, and the signed grid values of IQ3_XXS/
// IQ3_S/IQ2_XXS/IQ2_XS/IQ2_S), and NVFP4 (doubled E2M1 codes) on every build.
[[nodiscard]] bool w4a8_supports(QType qtype);
// K the W4A8 tiles take for this format (generic_gemm.cpp).
[[nodiscard]] bool w4a8_k_admits(QType qtype, std::int32_t k);
// Whether a weight format has an R8 (W8A8 decode) codec: q8_g32, MXFP4, the ggml 4/5-bit blocks
// (Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS), the K-quants Q2_K..Q6_K and the codebook i-quants: the
// formats of a published GGUF, whose A16 decode is the costly part of a decode call.
[[nodiscard]] bool r8_supports(QType qtype);
// The most tokens an R8 launch takes, the widths the heuristic gives it, and whether its quantized
// activation (36 bytes per 32 values and token slot, 40 with the block sums a codec with a min
// needs, token slots rounded up to a power of two) fits one CTA's 48 KiB. The heuristic takes R8 at
// T = 1 only: at T = 4 (speculative verify) its dp4a work grows with T while the register MMA's
// does not, and it lost on 26 of 42 decode shapes of Qwen3.5-4B/9B (RTX 5080); at T = 1 it
// won on all 42 (0.58-0.96 of the register MMA), and for the ggml 4/5-bit blocks and K-quants on
// 69 of 70 Qwen3.8-27B / Gemma 4 31B / Qwen3.6-35B-A3B shapes (0.77-0.99; Q6_K 4096x2048
// 1.05).
// Four tokens at most and no four-row schedule: the K-quant codecs' registers (a Q6_K block's codes,
// high bits and scales for every row and token slot) spilled there, up to 2 KiB of stack per
// thread (w4r4 SwiGLU at eight tokens), and no heuristic took those forms (T >= 5 is A16; w4r4
// was the best T = 1 schedule on 2 of 70 shapes, by 1 %).
inline constexpr std::int32_t kR8MaxTokens    = 4;
inline constexpr std::int32_t kR8DecodeTokens = 1;
[[nodiscard]] bool r8_fits(QType qtype, std::int32_t k, std::int32_t tokens);
// The multiple of K an R8 weight needs: a lane loads w4a8_span * w4a8_stride 32-value blocks
// together (generic_w4a8.cuh: two for Q4_K and Q5_K, four for Q6_K and Q3_K, one otherwise).
[[nodiscard]] std::int32_t r8_k_values(QType qtype);
// Speculative verify (2 <= T <= kR8VerifyTokens) takes R8 too for the formats where it is not
// slower than the register MMA there, with the activation quantized once per call into the
// workspace (the W4A8 quantizer: the same codes, scales and sums the CTA makes at T = 1) and read
// from L2 by every CTA, so a token's bytes are those of plain decode and greedy speculation
// reproduces it. RTX 5080, 27B/31B/35B dense shapes, quantize + R8 against the register MMA:
// T = 4 IQ2_S/IQ3_S/IQ3_XXS 0.73-0.91 on the large shapes (IQ2_XXS/IQ2_XS follow their
// family); IQ4_XS and Q8 0.96-1.0 in the micro-benchmark but slower in the product (Qwen3.8-27B
// UD-IQ4_XS, MTP draft 2/3: -1.6 %/-2.4 % tok/s), Q4_K/Q5_K 1.0-1.3 and Q6_K 1.15-1.77:
// those keep R8 at T = 1 only; at T = 8 the median is 1.35, so no format takes it. Q4_0 takes it
// (one scale per block, no min: the cheapest codec): serve --spec none at C = 4, the same binary
// with and without it, Gemma 4 31B QAT (all Q4_0) on the RTX 4090 135 -> 161 tok/s, Gemma 4
// 26B-A4B Q4_0 on the RTX 5090 710 -> 728 (its attention projections; MTP C = 1 unchanged); Muse
// 30B Q4_K_M (Q4_K/Q5_K/Q6_K) on the RTX 4090 lost 3 % with the K-quants taking it.
inline constexpr std::int32_t kR8VerifyTokens = 4;
[[nodiscard]] bool r8_verify_supports(QType qtype);
// The formats whose R8 codec has the verify-width form compiled: every R8 format but Q3_K.
[[nodiscard]] bool r8_verify_compiled(QType qtype);
// Whether a call of 2..kR8VerifyTokens tokens takes R8 on this device: the formats measured faster
// there (r8_verify_supports) where the MMA takes BF16 operands itself; every compiled format where
// it does not (Turing: each operand rounded to f16 and two k8 MMAs per k16 make the A16 register
// MMA the costlier path; RTX 2080 Ti).
[[nodiscard]] bool r8_verify_takes(QType qtype);
// r8_verify_takes for one matrix (Q4_0 takes R8 at the verify widths only on a large one).
[[nodiscard]] bool r8_verify_takes_shape(QType qtype, std::int64_t n, std::int64_t k);
// Whether a weight format has a TiledMma codec (then its prefill routes are the tiled ones).
[[nodiscard]] bool tiled_supports(WeightCodec codec);
// The schedule's shared tiles for the codec fit one block of this build (generic_tiled.cu).
[[nodiscard]] bool tiled_fits(int schedule, WeightCodec codec);
// The staged schedule's shared stages fit one block of this build (generic_staged.cu).
[[nodiscard]] bool staged_fits(int schedule);
[[nodiscard]] bool staged_device_fits(int schedule);
// The W4A8 schedule's buffers for the format fit one block of this build (the producer ring needs
// 99 KiB).
[[nodiscard]] bool w4a8_fits(int schedule, QType qtype);

// Stable textual names: "rmma.m<M>.n<N>.w<WM>x<WK>", "w4a4.<BM>x<BN>", "tiled.<BM>x<BN>",
// "w4a8.<BM>x<BN>", "w4a8ws.<BM>x<BN>.p<producer warps>", "r8.w<W>r<R>",
// "staged.<rows>x<tokens>.w<WM>x<WN>.s<stages>",
// "nvfp4t.c<classes>.<BM>x<BN>", "scalar".
[[nodiscard]] std::string route_name(Route route);
[[nodiscard]] std::optional<Route> parse_route(std::string_view name);

[[nodiscard]] WeightCodec codec_of(const Weight& weight);
// Whether the register-MMA tile of m_tiles x n_tiles is compiled for the codec (rmma_compiled,
// generic_gemm.cuh: not the 2 x 8 tile of a codec that reads 32-value blocks). `k` is the weight's
// K where the decoder depends on it: u4 G32/G64 at K of whole 128-value steps take the quad-run
// spread decoder, which reads 32 values per decode; 0 names the codec's other decoder.
[[nodiscard]] bool rmma_tile_compiled(WeightCodec codec, int m_tiles, int n_tiles,
                                      std::int32_t k = 0);

// The weight-plane fields of GemmArgs for one matrix (k, planes, groups_per_row, NVFP4 divisor
// and scale tiles): what decode_element needs to read any element of it.
[[nodiscard]] GemmArgs weight_args(const Weight& weight);

// Whether the generic path can compute this weight at all (format/layout pairing, NVFP4 plane
// geometry, alignment). Throws nothing. False for a ggml_frag_v1 bank.
[[nodiscard]] bool supports(const Weight& weight);

// A decode call (T <= kR8DecodeTokens) of a shape whose own kernels read A16 at that width
// takes the generic R8 when its policy allows A8 (Qwen3.5/3.6 Q8 projections: their tables
// have no A8 decode kernel). The wrappers ask this before they hand a registered shape to its
// table.
[[nodiscard]] bool owned_decode_r8(const Weight& weight, LinearPolicy policy, std::int32_t tokens);

// A call of 17 tokens or more (T >= kQ8PrefillA8Tokens) inside a prefill (prefill_routes_active)
// of an NVFP4 weight that the generic W4A8 takes: its policy allows A8 and, where the build has
// the W4A4 kernels, not A4 (nvfp4_prefill_a8_policy). The native NVFP4 plans of sm_120a compute A16
// and A4 only, so the wrappers hand such a call of a registered shape to the generic path, as
// q8_prefill_a8 hands a Q8 one: a W4A16 checkpoint under --activations int8 prefills on int8
// activations on every card. Outside a prefill (decode, speculative verify of any width, the
// drafters' rounds, Gemma 4's speed-profile prefill) the plans' routes stay.
[[nodiscard]] bool owned_prefill_w4a8(const Weight& weight, LinearPolicy policy,
                                      std::int32_t tokens);
// The policy half of it, without the weight (workspace queries, and allows_w4a8 for NVFP4): an
// NVFP4 problem whose policy allows A8 and, where this build has the W4A4 kernels, not A4. An
// AllowA4 weight stays off W4A8 whatever its divisor (A16 where W4A4 cannot run, as before).
[[nodiscard]] bool nvfp4_prefill_a8_policy(QType qtype, LinearPolicy policy);

// A Q8_G32_FP16 call of 17 tokens or more (T >= kQ8PrefillA8Tokens) inside a prefill
// (prefill_routes_active) whose policy allows A8 takes
// the generic path's int8 tiers instead of the shape tables, whose Q8 kernels read A16 at every
// width: linear, the fused GDN and attention input projections, linear_add and linear_swiglu.
// Qwen3.6-35B prefill 4K/16K: RTX 4090 +7.4/+7.1 %, RTX 5090 +4 % (linear and GDN input alone);
// decode is untouched and the perplexity of a 517 KB book at 8K windows moves by 0.0005 nats or
// less. The width is the chunk-invariance contract's (select_route): W4A8 sums each token over its
// own activation whatever the tile, so every call from 17 tokens gives a token the same bytes; a
// higher width put a 100-token cut on A16 and its 2048-token chunk on A8 (Gemma 4 26B gate_up).
// Linear, linear_add and linear_swiglu all take it. Outside a prefill (decode, speculative verify,
// the drafters' rounds) the tables stay: a DFlash2 round of 17+ columns on A8 cost Qwen3.8-27B
// --spec auto C=1 2-3 % on an RTX 5090, and those calls need no chunk invariance.
// The native Qwen formats Q4_G64, Q5_G64 and Q6_G64 take the same int8 tiers (the generic
// path's W4A8, W4a8G64): their shape tables read A16 at every width too. Qwen3.8-27B 24K prefill
// on an RTX 5090 ran 2830 tok/s against 3720 in ninfer-all, whose dense Q4/Q5 prefill is
// int8: the Q4 SwiGLU 2456 us a chunk against 1385, Q5 down 1219 against 635, the Q5 5120-row
// linear 825 against 290. They take the int8 tiers on the cards where int8 activations pay
// (half-rate fp32-accumulate HMMA: GeForce, Ada/Ampere inference parts), not on sm80/90/100.
inline constexpr std::int32_t kQ8PrefillA8Tokens = 17;
[[nodiscard]] bool prefill_a8_format(QType qtype);
// The policy a prefill-width call of this format runs under: the artifact's own, except that
// where int8 activations pay, A8 for a Q4/Q5/Q6 G64 weight the artifact records as A16Only
// (the official Qwen3.8-27B artifact does; ninfer-all computes those weights on int8 activations
// regardless). Never under the invariant route profile (--activations exact).
[[nodiscard]] LinearPolicy prefill_a8_policy(QType qtype, LinearPolicy policy);

// The same rule from the device's facts (prefill_a8_policy and --activations' startup resolution,
// ops/linear/activation_width.h, read this one): int8 activations pay at prefill on a chip with
// tensor-core mma.sync (not emulated: mma_slow) whose FP32-accumulate HMMA runs below full rate
// (GenerationRow::fp32acc_full_rate); the raise is for Q4/Q5/Q6 G64 A16Only weights and never
// under the invariant route profile.
[[nodiscard]] constexpr bool qx_prefill_a8_pays_on(const DeviceFeatures& features,
                                                   bool fp32acc_full_rate) noexcept {
    return features.mma_tensor && !features.mma_slow && !fp32acc_full_rate;
}

[[nodiscard]] constexpr bool qx_prefill_format(QType qtype) noexcept {
    return qtype == QType::Q4_G64_FP16 || qtype == QType::Q5_G64_FP16 ||
           qtype == QType::Q6_G64_FP16;
}

[[nodiscard]] constexpr LinearPolicy prefill_a8_policy_on(QType qtype, LinearPolicy policy,
                                                          bool pays, bool invariant) noexcept {
    return policy == LinearPolicy::A16Only && qx_prefill_format(qtype) && pays && !invariant
               ? LinearPolicy::AllowA8
               : policy;
}
[[nodiscard]] bool q8_prefill_a8(const Weight& weight, LinearPolicy policy, std::int32_t tokens);
[[nodiscard]] bool q8_prefill_a8_fused(const Weight& weight, LinearPolicy policy,
                                       std::int32_t tokens);

// Throws std::invalid_argument naming `op` when the weight is a ggml_frag_v1 routed-expert bank
// (core/ggml_frag.h): only the grouped MoE kernel reads that layout, never a projection path
// (register MMA, R8, W4A8, tiled or scalar).
void refuse_frag(const Weight& weight, const char* op);

// Whether a route may compute this problem: the register MMA needs K to be a whole number of its
// lane runs; W4A4 needs an NVFP4 weight, an AllowA4 policy and a positive activation divisor;
// W4A8 a format with a W4A8 codec, a K w4a8_k_admits and a policy that allows A8. Every route also
// needs the current device to have the features its kernel issues (route_requires).
[[nodiscard]] bool route_admits(Route route, const Weight& weight, LinearPolicy policy);

// The device features a route's kernel issues (core/device_class.h). A route is admitted by what
// it needs, not by an architecture number; which kernels a build carries is route_admits' other
// half.
[[nodiscard]] DeviceFeatures route_requires(Route route);

// Whether a route sums every output element in the same order as every other route that can be
// selected for T >= 17 under the prefill route profile (the chunk-invariance contract). StagedMma
// sums in the order of the WarpsK == 1 register-MMA schedules: the two give the same bytes.
[[nodiscard]] bool route_is_token_invariant(Route route);

// Whether two routes sum every output element in the same order, so give the same bytes: a route
// and itself, and any two of the WarpsK == 1 register-MMA and staged schedules. A table may pick
// among such routes by T under the prefill profile ("prefill_bands") without a token's bytes
// depending on its call's width.
[[nodiscard]] bool same_summation(Route a, Route b);

struct Problem {
    QType qtype;
    LinearPolicy policy;
    std::int32_t n; // rows of the op's output (SwiGLU: the gate rows)
    std::int32_t k;
    Epilogue epilogue;
};

// The workspace the int8 prefill tier of `problem` takes over [min_tokens, max_tokens]: 0 where no
// call takes it (a format off the int8 tiers, a policy without A8 here, or an interval below
// kQ8PrefillA8Tokens). A wrapper's query takes the larger of this and its own plan's, since the
// same widths run the plan outside a prefill. NVFP4 under nvfp4_prefill_a8_policy: the W4A8 planes
// of its calls from 17 tokens (owned_prefill_w4a8; the query cannot know whether a prefill runs).
[[nodiscard]] std::size_t prefill_a8_workspace_bytes(const Problem& problem, std::int32_t min_tokens,
                                                     std::int32_t max_tokens);

// Whether a call of 17 tokens or more of this format takes the tiled route on this GPU (Volta,
// volta_tiled_route), whatever shape plan the format has: workspace queries without a weight.
[[nodiscard]] bool volta_tiled_format(QType qtype);

// The tiled route a call takes on a GPU without tensor mma.sync (Volta), from 17 tokens on, for a
// format with a tiled codec; nothing elsewhere (generic_gemm.cpp).
[[nodiscard]] std::optional<Route> volta_tiled_route(const Weight& weight, LinearPolicy policy,
                                                     std::int32_t tokens);

// The bytes a Volta prefill call of up to max_tokens tokens decodes its weight into for CUTLASS's
// sm_70 GEMM (0 on a GPU with tensor mma.sync, below the CUTLASS threshold, or for a format
// without a tiled codec). A caller that sizes its own workspace for an owned shape takes the max
// with it, or the call falls back to the tiled kernel. `rows`: the output rows (SwiGLU's gate rows,
// decoded with the up rows).
[[nodiscard]] std::size_t volta_prefill_workspace_bytes(QType qtype, std::int32_t rows,
                                                        std::int32_t k, std::int32_t max_tokens,
                                                        bool swiglu);

// The route the engine takes for one call of T tokens: a loaded table's entry for the exact
// problem if there is one, otherwise the shape-class heuristic. Under the prefill route profile a
// call of 17 tokens or more always takes the problem's prefill route, whatever T is. Under the
// invariant route profile every call takes the register MMA with the problem's K split (or the
// scalar route where that cannot read the weight), whatever T, the tables or the policy say; an
// NVFP4 call of two token tiles or more that gives every SM a tile takes the tiled route in that
// same order. A GPU without tensor mma.sync takes volta_tiled_route first.
[[nodiscard]] Route select_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                                 bool prefill);

// The W4A8 tile of the rules a device without clocks runs (the cost model's fallback), else the
// first schedule that fits the build; nullopt where none admits the weight.
[[nodiscard]] std::optional<Route> w4a8_route(const Problem& problem, const Weight& weight,
                                              std::int32_t tokens);

// The shape-class heuristic alone (what the tuner is compared against).
[[nodiscard]] Route heuristic_route(const Problem& problem, const Weight& weight,
                                    std::int32_t tokens, bool prefill);

// The R8 schedule of the shape-class rules alone, without the per-GPU decode entries of
// r8_decode_table.h (what those entries are measured against).
[[nodiscard]] Route r8_rules_route(const Problem& problem, std::int32_t tokens);

// Transient storage a route needs for T tokens (W4A4, W4A8: the quantized activation).
[[nodiscard]] std::size_t route_workspace_bytes(Route route, std::int32_t k, std::int32_t tokens);
// Family::GgmlMmq's workspace for a call of `plane_rows` weight rows (SwiGlu: gate and up) and T
// tokens: the q8_1 activation, the FP32 plane and the stream-k fixup bound.
// Whether ggml's matrix kernel (Family::GgmlMmq) reads the format.
[[nodiscard]] bool ggml_mmq_format(QType qtype);
// Whether a verify call of T tokens (2..kR8VerifyTokens) may run R8 where the format does not take
// it by rule (r8_verify_takes): the dense tuner's candidate. R8 sums a token as at decode.
[[nodiscard]] bool r8_verify_tunable(const Weight& w, LinearPolicy policy, std::int32_t tokens);
// Whether the dense tuner may time R8 at T tokens: r8_verify_tunable, or a width where the format
// has R8 but ggml's vector kernel takes it by rule.
[[nodiscard]] bool r8_tunable(const Weight& w, LinearPolicy policy, std::int32_t tokens);
// Whether a call of `rows` rows and T tokens may take ggml's matrix kernel (the dense tuner's
// candidate at verify widths 5..16).
[[nodiscard]] bool ggml_mmq_call(const Weight& w, LinearPolicy policy, std::int32_t rows,
                                 std::int32_t tokens);
[[nodiscard]] std::size_t ggml_mmq_workspace_bytes(std::int32_t k, std::int32_t plane_rows,
                                                   std::int32_t tokens);
// Whether ggml's vector kernel (Family::GgmlVec) reads the format; whether a call of T tokens may
// take it (the dense tuner's candidate at 1..8 tokens); its workspace (the activation).
[[nodiscard]] bool ggml_vec_format(QType qtype);
[[nodiscard]] std::size_t ggml_vec_workspace_bytes(std::int32_t k, std::int32_t tokens);
// The widest call ggml's vector kernel takes (ggml_mmq::kVecMaxColumns).
inline constexpr std::int32_t kGgmlVecMaxTokens = 8;
// The widest call ggml's matrix kernel takes (its 5..16-token band).
inline constexpr std::int32_t kGgmlMmqMaxTokens = 16;
// Whether a call of `problem` on `weight` of T tokens takes ggml's vector kernel, the bytes of its
// q8_1 activation of a [K, T] x (Call::vec_activation), and that quantization of x.
[[nodiscard]] bool ggml_vec_takes(const Problem& problem, const Weight& weight,
                                  std::int32_t tokens);
[[nodiscard]] std::size_t ggml_vec_activation_bytes(std::int32_t k, std::int32_t tokens);
void ggml_vec_quantize(const Tensor& x, void* out, cudaStream_t stream);

// The SwiGLU (GeGLU: gelu) of a gate and an up weight on one x: out[r] = act(gate row r . x) *
// (up row r . x) for `rows` rows from each half's `row`, out BF16 [>= rows, T]. Where both halves
// take ggml's vector kernel at this T (the dense tuner's or the heuristic's route, as a call of
// that half alone would), x is quantized once (or `q8` carries it, ggml_vec_quantize): a pair of
// one format runs as one gated launch over both weights, a pair of two formats as the gate's
// product into an FP32 plane and the up product multiplying it in its epilogue (no separate
// activation kernel, the gate never rounded to BF16). False, with nothing launched, otherwise.
// ggml_vec_takes for a call of `rows` rows of `weight` from `row` on x, after the dense tuner has
// timed that problem (the route a call of it alone would take).
[[nodiscard]] bool ggml_vec_takes_tuned(const Tensor& x, const Weight& weight, LinearPolicy policy,
                                        std::int32_t row, std::int32_t rows, cudaStream_t stream);

// The ggml matrix kernel's route for a call of `rows` rows of `weight` from `row` on x after the
// dense tuner has timed it (ggml_vec_takes_tuned's matrix-kernel counterpart).
[[nodiscard]] bool ggml_mmq_takes_tuned(const Tensor& x, const Weight& weight, LinearPolicy policy,
                                        std::int32_t row, std::int32_t rows, cudaStream_t stream);

struct PairHalf {
    const Weight* weight = nullptr;
    LinearPolicy policy  = LinearPolicy::A16Only;
    std::int32_t row     = 0;
};

[[nodiscard]] bool ggml_vec_gated_pair(const Tensor& x, const PairHalf& gate, const PairHalf& up,
                                       std::int32_t rows, bool gelu, Tensor& out,
                                       WorkspaceArena& workspace, const void* q8,
                                       cudaStream_t stream);
// Its workspace at T tokens: the q8_1 activation, the FP32 gate plane and the arena's alignment.
[[nodiscard]] std::size_t ggml_vec_gated_pair_bytes(std::int32_t k, std::int32_t rows,
                                                    std::int32_t tokens);

// The gated pair on ggml's matrix kernel (its token band, or a prefill where both halves take it,
// ggml_mmq_slice_tokens at a time; rows whole 128-row tiles): both halves' products into one FP32
// [gate; up] plane from one quantization of x (two where the formats read different q8_1 layouts)
// and the gated product in the epilogue. False, out untouched, where a half takes another route or
// the workspace is short.
bool ggml_mmq_gated_pair(const Tensor& x, const PairHalf& gate, const PairHalf& up,
                         std::int32_t rows, bool gelu, Tensor& out, WorkspaceArena& workspace,
                         cudaStream_t stream);
// Its workspace for a call (or a prefill slice) of `tokens` tokens.
[[nodiscard]] std::size_t ggml_mmq_gated_pair_bytes(std::int32_t k, std::int32_t rows,
                                                    std::int32_t tokens);
// The widest slice ggml's matrix kernel runs in one pass for a weight of `qtype` on this device:
// a prefill slice where its prefill route holds (mmq_prefill_wins), its 16-token band otherwise.
[[nodiscard]] std::int32_t ggml_mmq_slice_tokens(QType qtype);

// Calls routed to ggml's matrix or vector kernel that ran A16 instead (a workspace short of the
// route, an unaligned x or rows off its multiple), since the process started.
[[nodiscard]] std::uint64_t ggml_mmq_fallback_count() noexcept;

// Capacity for every T in [min_tokens, max_tokens] under the active profile.
[[nodiscard]] std::size_t workspace_capacity_bytes(const Problem& problem, const Weight* weight,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens);

// A fused projection whose fixed-shape kernels take A16Only parents (the Qwen3.8-27B GDN and
// attention input pairs) and that prefill_a8_policy moves to the int8 tiers one generic call
// per parent: the policy a parent runs under, whether both parents take the tiers at this width,
// and the workspace the largest parent needs over a token interval (0 while the option leaves
// the pair on its tables).
struct FixedParent {
    QType qtype;
    std::int32_t rows;
};

[[nodiscard]] LinearPolicy fixed_parent_prefill_a8_policy(QType qtype);
[[nodiscard]] bool fixed_pair_prefill_a8(const Weight& first, const Weight& second,
                                         std::int32_t tokens);
[[nodiscard]] std::size_t
fixed_parents_prefill_a8_workspace_bytes(std::initializer_list<FixedParent> parents, std::int32_t k,
                                         std::int32_t min_tokens, std::int32_t max_tokens);

struct Call {
    const Tensor* x         = nullptr; // [K, T] BF16
    const Weight* weight    = nullptr;
    int row_begin           = 0;
    int rows                = 0;
    int up_offset           = 0;
    Epilogue epilogue       = Epilogue::Store;
    bool gelu_gate                 = false; // SwiGlu: GeGLU (GemmArgs::gelu_gate)
    bool relu2                     = false; // Store: relu(W x)^2 (GemmArgs::relu2)
    Tensor* out[kMaxSegments] = {};
    int segment_rows[kMaxSegments] = {};
    // Segment s writes rows [out_row_offset[s], out_row_offset[s] + segment_rows[s]) of out[s],
    // a contiguous BF16 [rows >= offset + segment rows, T] tensor.
    int out_row_offset[kMaxSegments] = {};
    int segments            = 1;
    // The kernel before this call on the stream is another call that read the same x (complete
    // before that call started) and writes nothing this call reads or writes. A register-MMA
    // launch then runs alongside it (programmatic dependent launch) and joins it only before
    // exiting, so the stream's next kernel still follows both. Store epilogue only; other routes
    // launch in stream order.
    bool join_previous = false;
    // The next kernel on the stream is a call with join_previous: this register-MMA launch (Store
    // epilogue) lets it start once every CTA of this one has started. Nothing else may follow a
    // call with release_next set: a programmatic dependent could then read this call's output
    // before it is written.
    bool release_next = false;
    // ggml's vector kernel's q8_1 activation of x (ggml_vec_quantize), quantized once for the
    // calls that read one x (segmented_projection); null: a vector call quantizes x itself.
    const void* vec_activation = nullptr;
};

// The dense tuner (generic_tune.cpp). tune_dense times the problem of `call` on its own weight at
// the decode widths (1..kDenseTuneMaxTokens) where the heuristic takes a workspace-free A16
// schedule, or R8 (another R8 schedule: same workspace, same bytes), and no route table covers it, once per problem and process, outside a graph capture;
// the choice persists per GPU class in a JSON file (read even inside a capture). tuned_dense_route is its route for a call of
// `tokens` tokens (nullopt: the heuristic's), dense_tune_report one line per timed problem.
inline constexpr std::int32_t kDenseTuneMaxTokens = 16;
void tune_dense(const Problem& problem, const Call& call, cudaStream_t stream);
[[nodiscard]] std::optional<Route> tuned_dense_route(const Problem& problem, std::int32_t tokens);
// Its prefill-band route for a call of `tokens` >= 17 whose heuristic route is `heuristic`
// (nullopt: the heuristic's, also where the band was timed against another route): a W4A8
// schedule for a W4A8 call, a route of the prefill profile's summation order otherwise.
[[nodiscard]] std::optional<Route> tuned_prefill_route(const Problem& problem, std::int32_t tokens,
                                                       Route heuristic);
// The routes a prefill band may take over `base` (the tests compare their bytes).
[[nodiscard]] std::vector<Route> prefill_band_candidates(Route base, const Weight& weight,
                                                         LinearPolicy policy);
[[nodiscard]] std::vector<std::string> dense_tune_report();
// Off until enabled: the engine enables it for a loaded model, tests and benches keep the heuristic.
void enable_dense_tune(bool on);
// The tuner's mode (MaintainerOptions::dense_tune, TuneMode::Off by default: nothing is timed)
// and where its file lives (MaintainerOptions::tune_cache: <dir>/dense_tune.json; empty:
// NInferno's cache directory). The engine sets both at construction.
void set_dense_tune(TuneMode mode, const std::filesystem::path& cache_dir);
void clear_dense_tune();
// The files the tuner's choices come from, for the decision fingerprint: "dense_tune.shipped",
// the release's routes/tuner/dense_tune.json, and "dense_tune.user", the user's own cache, which
// is read only while --dense-tune times ("absent" when it does not exist yet). Each value
// is "<file name> sha256:<hex of its bytes>". Empty while the tuner is off.
[[nodiscard]] std::map<std::string, std::string> dense_tune_sources();

// Runs one call on an explicit route (the tuner and tests), or on select_route's choice.
// released (when given): whether the launch lets the next kernel start early (Call::release_next
// honoured: a register-MMA or R8 Store launch without the pre-quantized R8 form, on a device with
// programmatic dependent launch).
void run_route(Route route, const Call& call, WorkspaceArena* workspace, cudaStream_t stream,
               bool* released = nullptr);
void run(const Call& call, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream,
         bool* released = nullptr);

// Convenience entry points used by the op wrappers' fallbacks.
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena* workspace, cudaStream_t stream);
// linear with Call::release_next; returns run()'s `released`.
bool linear_release_next(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                         WorkspaceArena* workspace, cudaStream_t stream);
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena* workspace, cudaStream_t stream);
void linear_swiglu(const Tensor& x, const Weight& gate_up, Tensor& out, LinearPolicy policy,
                   WorkspaceArena* workspace, cudaStream_t stream);
// linear with relu(W x)^2 stored (NemotronH's shared expert up projection).
void linear_relu2(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream);
// linear_swiglu with the gate's GELU (tanh form) in place of its SiLU.
void linear_geglu(const Tensor& x, const Weight& gate_up, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninferno::ops::detail::generic
