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
    // Register MMA on tensor cores with BF16 operands (not Turing's f16): a tile of four or more
    // token tiles takes a min-term format's group sums by FP32 adds, not an MMA with ones
    // (generic_gemm.cuh mma_step), a sum in another order than a narrower tile's (same_summation).
    // run_route sets it outside the invariant and prefill profiles, whose calls keep one order at
    // every T; narrower tiles ignore it, and Volta's build always adds. Host side only: the launch
    // takes the kernel instance that adds (rmma_kernel's AddSums), the kernel reads none.
    int add_sums = 0;
};

// An activation quantized to int8 per 32 values (the W4A8 quantizer's output): what the W4A8
// kernel reads, and R8 at verify widths.
struct W4a8Activation {
    const std::int8_t* codes = nullptr; // [T, K]
    const float* scales      = nullptr; // [T, K/32]
    const float* sums        = nullptr; // [T, K/32], scale times the codes' sum (min-term formats)
    // W4A8 tiles: the grid runs token tiles along x, so the CTAs of a wave are the
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
// Speculative verify (2 <= T <= r8_verify_widest) takes R8 too for the formats where it is not
// slower than the register MMA there, with the activation quantized once per call into the
// workspace (the W4A8 quantizer: the same codes, scales and sums the CTA makes at T = 1) and read
// from L2 by every CTA, so a token's bytes are those of plain decode and greedy speculation
// reproduces it. RTX 5080, 27B/31B/35B dense shapes, quantize + R8 against the register MMA:
// T = 4 IQ2_S/IQ3_S/IQ3_XXS 0.73-0.91 on the large shapes (IQ2_XXS/IQ2_XS follow their
// family); IQ4_XS 0.96-1.0 in the micro-benchmark but slower in the product (Qwen3.8-27B
// UD-IQ4_XS, MTP draft 2/3: -1.6 %/-2.4 % tok/s), Q4_K/Q5_K 1.0-1.3 and Q6_K 1.15-1.77:
// those keep R8 at T = 1 only; at T = 8 the median is 1.35, so no format takes it. Q4_0 takes it
// (one scale per block, no min: the cheapest codec): serve --spec none at C = 4, the same binary
// with and without it, Gemma 4 31B QAT (all Q4_0) on the RTX 4090 135 -> 161 tok/s, Gemma 4
// 26B-A4B Q4_0 on the RTX 5090 710 -> 728 (its attention projections; MTP C = 1 unchanged); Muse
// 30B Q4_K_M (Q4_K/Q5_K/Q6_K) on the RTX 4090 lost 3 % with the K-quants taking it.
// The verify form is compiled for up to eight tokens (kR8VerifyTokens); r8_verify_widest is the
// widest call a format takes it for by rule.
inline constexpr std::int32_t kR8VerifyTokens = 8;
// The narrowest call on the ggml matrix kernel's route (generic_ggml_internal.h, kMmqMaxTokens): five
// tokens, past the four R8 verifies for every format but Q8.
inline constexpr std::int32_t kMmqMinTokens = 5;
[[nodiscard]] bool r8_verify_supports(QType qtype);
// Eight tokens for Q8 (GGUF Q8_0), four for the other formats: Q8 at T = 2..8 on the RTX 5090
// (Qwen3.5-9B shapes, ninferno_r8_schedule_bench --tokens) took 0.96-1.15 of its T = 1 time on
// R8 against 0.99-1.24 on the register MMA (SwiGLU 1.03-1.06 against 1.09-1.10, the down
// projection 0.96-1.03 against 1.21-1.24), each token summed as at decode; past four tokens the
// other formats' R8 lost (Q4_K 1.48-1.89, IQ3_XXS 1.49-1.80, IQ4_XS 1.19-1.53 against
// 1.07-1.62). The rule is the format's on every chip, from the chip's own rates: Q8's codes are
// whole bytes (no unpacking), so R8 does dp4a work only, and dp4a runs four int8 products per FP32
// lane per clock on every generation from Volta; the width at which that work takes as long as
// reading the 1.0625 bytes per weight is 21 tokens on the A100 and H100, 37 on the RTX 3060, 40 on
// the GV100 and 86 on the RTX 4090, well past eight. Measured: RTX 5090 and 5070 Ti (the bench),
// RTX 4090 (Qwen3.5-9B Q8_0 spec-tune, dflash:7 229 -> 273 tok/s).
[[nodiscard]] std::int32_t r8_verify_widest(QType qtype);
// The formats whose R8 codec has the verify-width form compiled: every R8 format but Q3_K.
[[nodiscard]] bool r8_verify_compiled(QType qtype);
// Whether a call of 2..r8_verify_widest tokens takes R8 on this device: the formats measured faster
// there (r8_verify_supports) where the MMA takes BF16 operands itself; every compiled format where
// it does not (Turing: each operand rounded to f16 and two k8 MMAs per k16 make the A16 register
// MMA the costlier path; RTX 2080 Ti).
[[nodiscard]] bool r8_verify_takes(QType qtype);
// r8_verify_takes for one matrix (Q4_0 takes R8 at the verify widths only on a large one).
[[nodiscard]] bool r8_verify_takes_shape(QType qtype, std::int64_t n, std::int64_t k);
// Whether R8 takes a call of kR8VerifyTokens tokens of this weight under A8: its format verifies a
// whole eight-token block on R8 (Q8, r8_verify_widest) and the generic path reads the weight.
// A speculative drafter's A16Only projections take A8 there (ops::drafter_linear_policy).
[[nodiscard]] bool r8_takes_block(const Weight& weight);
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
// A family's name, the first field of its routes' names ("rmma", "w4a8", "ggml_vec", ...).
[[nodiscard]] std::string_view family_name(Family family) noexcept;
// The W4A8 schedule run_route executes for a W4A8 route of this weight (its producer-warp or
// classic twin where the weight takes the other one); any other route unchanged.
[[nodiscard]] Route w4a8_executed_route(Route route, const Weight& weight);
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
// the W4A4 kernels, not A4 (nvfp4_prefill_a8_policy); without tensor mma.sync the tiled route owns
// the call (volta_tiled_route). The native NVFP4 plans of sm_120a compute A16
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

// The order in which a route sums every output element. Routes of one order give the same bytes:
// the register order holds the WarpsK == 1 register-MMA schedules (but a lane-local one) and the
// staged schedules; the I8 tiles share theirs (K groups ascending, never split); every other route
// has an order of its own. Without tensor mma.sync (Volta) outside the invariant profile, a
// one-tile register-MMA schedule sums lane by lane (rmma_sums_lane_local): an order of its own.
struct SummationOrder {
    enum class Kind : std::uint8_t { Register, I8, Own };
    Kind kind = Kind::Own;
    Route own{}; // Kind::Own: the route itself
    friend bool operator==(const SummationOrder&, const SummationOrder&) = default;
};
[[nodiscard]] SummationOrder summation_order(Route route);

// Whether two routes sum every output element in the same order (summation_order), so give the
// same bytes. A table may pick among such routes by T under the prefill profile ("prefill_bands")
// without a token's bytes depending on its call's width.
[[nodiscard]] bool same_summation(Route a, Route b);

struct Problem {
    QType qtype;
    LinearPolicy policy;
    std::int32_t n; // rows of the op's output (SwiGLU: the gate rows)
    std::int32_t k;
    Epilogue epilogue;
};

// A distinct problem that reached the generic path on this thread since clear_seen_problems, with
// its weight's stored rows and layout: what route_query needs to ask the call's route again (the
// decision manifest's route lines, runtime/engine/decision_routes.h). In (format, policy,
// epilogue, N, K, stored rows, layout) order.
struct SeenProblem {
    Problem problem;
    std::int32_t parent_rows = 0;
    QuantLayout layout       = QuantLayout::RowSplit;
};
[[nodiscard]] std::vector<SeenProblem> seen_problems();
// Forgets them and collects this thread's from here on: an Engine's load starts its own record, and
// the calls of another live Engine (its own worker thread) do not enter it.
void clear_seen_problems();

// Why a schedule of a family is not a route for a weight on this build and device (#307), one
// reason each: not declared (an index past the family's list), not compiled (this build omits
// it: the thread-block cluster schedules outside the full sm_12x build), the weight's format or
// layout (supports), the tile not compiled for the codec (rmma_tile_compiled: the large tiles of
// the 32-value decoders), an instruction the device lacks (route_requires), a plane off the
// alignment its kernel reads, or K not a whole number of the codec's runs; the first that applies,
// in that order, so only Format hides whether a tile is compiled. Policy: the call's policy
// does not allow the family's activations (R8: int8; r8_refusal asks it after Format, before
// Instruction). Resource: the tile's shared memory exceeds the block's (W4A8, tiled, staged, I8).
// W4A8 asks in its own order (w4a8_refusal): Format before NotCompiled, Resource and Policy
// before Instruction; the tiled, staged and I8 families Resource before Instruction (I8 Policy
// too).
enum class ScheduleRefusal { None, NotDeclared, NotCompiled, Format, CodecTile, Instruction,
                             Alignment, Extent, Policy, Resource };
[[nodiscard]] std::string_view schedule_refusal_name(ScheduleRefusal why) noexcept;
// Register-MMA schedule `index` for `weight`: route_admits' own rule for the family (it decides
// by this), so None exactly where route_admits takes the route.
[[nodiscard]] ScheduleRefusal rmma_refusal(int index, const Weight& weight);

// The register-MMA family's coverage of a call (the pilot family of #307): every declared schedule
// with its refusal, and the counts apart. declared: the schedule list; compiled: the schedules
// this build holds for the weight's codec, whatever else refuses the weight (0 where the format
// has no codec); legal: those the
// weight and device take (None). The family has no policy gate, so admitted equals legal here.
// heuristic: the family's own pick for the call (heuristic_rmma), the tile a selection that
// reaches the register MMA tries first; heuristic_refusal: why it does not run where it is not
// legal (the selection then takes a narrower tile, the staged MMA or the scalar route). Which
// route the call selects across families is select_route's.
struct ScheduleCoverage {
    int schedule             = 0;
    ScheduleRefusal refusal  = ScheduleRefusal::None;
};
struct FamilyCoverage {
    std::vector<ScheduleCoverage> schedules;
    int declared = 0, compiled = 0, legal = 0;
    int heuristic = -1;
    ScheduleRefusal heuristic_refusal = ScheduleRefusal::None;
};
// R8 schedule `index` for `weight` under `policy`: route_admits' own rule for the family (it
// decides by this). Not declared, format (the generic path or R8 does not read it), policy (no
// A8), instruction, alignment (the ggml high plane or the codes off 16 bytes), extent (K off the
// format's R8 run).
[[nodiscard]] ScheduleRefusal r8_refusal(int index, const Weight& weight, LinearPolicy policy);
// W4A8 schedule `index` for `weight` under `policy`: route_admits' own rule for the family (it
// decides by this). Not declared, format (the generic path or W4A8 does not read it), not compiled
// (a producer-warp tile for a format it is not built for, w4a8_ws_supports), resource (the tile's
// shared memory past the block's), policy (no A8, or NVFP4 outside its prefill A8 policy),
// instruction, alignment (the codes off 16 bytes; a producer tile's ggml high plane off 16, an
// IQ3_XXS sign plane off 8), extent (K the format's W4A8 groups do not take).
[[nodiscard]] ScheduleRefusal w4a8_refusal(int index, const Weight& weight, LinearPolicy policy);

// Why a call of `tokens` tokens does not take R8 where its weight could (#307): the weight not
// admitted to R8 (allows_r8), a decode call whose activation does not fit one CTA, a verify call
// wider than the format's widest R8 verify, or a shape the verify kernel does not take. r8_call
// decides by this.
enum class R8CallRefusal { None, NotAdmitted, DecodeFit, VerifyWidth, VerifyShape };
[[nodiscard]] std::string_view r8_call_refusal_name(R8CallRefusal why) noexcept;
[[nodiscard]] R8CallRefusal r8_call_refusal(const Weight& weight, LinearPolicy policy,
                                            std::int32_t tokens);

[[nodiscard]] FamilyCoverage rmma_coverage(const Problem& problem, const Weight& weight,
                                           std::int32_t tokens, bool prefill_band);

// The R8 family's coverage of a call (#307): every schedule with its refusal (compiled: every
// schedule for a format R8 reads, none for another), the family's pick for the call
// (heuristic_r8) and its refusal, why the call itself does or does not take R8, and whether the
// format has R8's pre-quantized verify form (r8_verify_compiled; it runs where a verify call has
// the workspace).
struct R8Coverage {
    FamilyCoverage family;
    R8CallRefusal call         = R8CallRefusal::None;
    bool prequantized_compiled = false;
};
[[nodiscard]] R8Coverage r8_coverage(const Problem& problem, const Weight& weight,
                                     std::int32_t tokens);

// The W4A8 family's coverage of a call (#307): every schedule with its refusal (compiled: the
// classic tiles for a format W4A8 reads, the producer-warp tiles for the formats they are built
// for), and the family's pick for the call (heuristic_w4a8, before w4a8_route falls back to a
// schedule that admits the weight) with its refusal. The route a producer tile resolves to where
// the weight is not admitted to it is resolve_route's (Demotion::W4a8Admission).
[[nodiscard]] FamilyCoverage w4a8_coverage(const Problem& problem, const Weight& weight,
                                           std::int32_t tokens);

// Tiled schedule `index` for `weight`: route_admits' own rule for the family (it decides by this).
// Not declared; format (the generic path does not read it, or the tiled kernel has no tile for
// its codec and the device is not Volta, whose CUTLASS path decodes any codec); codec tile (a
// CUTLASS-only format runs as schedule 0 alone); resource (the tile's shared memory past the
// block's); instruction; alignment (a ggml high plane off 16 bytes; the codes off 16 where the
// tiles read them); extent (K not whole 64-value steps).
[[nodiscard]] ScheduleRefusal tiled_refusal(int index, const Weight& weight);
// The tiled family's coverage of a weight (#307): every schedule with its refusal (compiled: every
// schedule for a codec the tiled kernel takes, schedule 0 alone for a CUTLASS-only one), and the
// family's pick, the first schedule whose tiles fit the build (schedule 0 for a CUTLASS-only
// format; -1 where none does), with its refusal. The family takes no policy (f16 activations) and
// its tile does not depend on the call's width.
[[nodiscard]] FamilyCoverage tiled_coverage(const Weight& weight);

// Staged schedule `index` for `weight`: route_admits' own rule for the family (it decides by
// this). Not declared; format (the generic path does not read it, or it is not BF16); resource
// (the tile's shared memory past the build's block, staged_fits, or the device's, the opt-in
// limit an sm_80 build meets on a 99 KiB part, staged_device_fits); instruction; alignment (the
// codes off 16 bytes); extent (K not whole 64-value steps).
[[nodiscard]] ScheduleRefusal staged_refusal(int index, const Weight& weight);
// The staged family's coverage of a weight (#307): every schedule with its refusal (compiled:
// every schedule for a BF16 weight), and the family's pick under the prefill profile, the first
// schedule whose tiles fit the build and the device (-1 where none does), with its refusal.
// Outside the profile the pick depends on the call (staged_speed_route); where the BF16 MMA is
// emulated (Turing, Volta) the selection takes no staged tile at all (bf16_mma_native), whatever
// this reports. No policy (BF16 activations).
[[nodiscard]] FamilyCoverage staged_coverage(const Weight& weight);

// I8 schedule `index` for `weight` under `policy`: route_admits' own rule for the family (it
// decides by this). Not declared; format (the generic path does not read it, or it is not Q4 /
// Q5 / Q6 G64); resource (the tile's stages past the build's shared memory, i8_fits); policy (no
// A8); instruction; alignment (the codes or scales off 16 bytes, a Q5 high plane off 8, a Q6 one
// off 16); extent (K or the padded K not whole 512-value runs).
[[nodiscard]] ScheduleRefusal i8_refusal(int index, const Weight& weight, LinearPolicy policy);
// The I8 family's coverage of a call of `tokens` tokens (#307): every schedule with its refusal
// (compiled: every schedule for Q4 / Q5 / Q6 G64), and the family's pick, the tile with the
// fewest weighted waves of those that fit the build (i8_route takes it where the weight is
// admitted), with its refusal.
[[nodiscard]] FamilyCoverage i8_coverage(const Problem& problem, const Weight& weight,
                                         std::int32_t tokens);

// The remaining families' own rules in route_admits (#307), which it decides by:
// - W4A4 schedule `index`: not declared (past its names); not compiled (a build without the
//   Blackwell kernels, w4a4_schedule_count 0, still parses the names); format (not NVFP4, or no
//   recorded input scale divisor, finite and positive); policy (no A4); instruction.
// - NVFP4 tiled schedule `index`: not declared; not compiled (outside the 120a build,
//   kNativeNvfp4); format (not NVFP4); instruction; alignment (the codes off 8 bytes).
// - ggml's matrix and vector kernels (Family::GgmlMmq / GgmlVec, schedule 0): not declared (any
//   other schedule); format (a format the kernel does not read, a layout it does not take, a
//   plane it needs missing); policy (no A8); instruction (the matrix kernel below sm_75,
//   ggml_mmq::device_supported); alignment (a plane off 16 bytes); extent (K off the kernel's
//   multiple: 256 values for the matrix kernel, a block for the vector one).
// Every family adds route_admits' generic checks: the generic path reads the weight, the
// device's instructions (route_requires), a ggml high plane on 16 bytes.
[[nodiscard]] ScheduleRefusal w4a4_refusal(int index, const Weight& weight, LinearPolicy policy);
[[nodiscard]] ScheduleRefusal nvfp4_tiled_refusal(int index, const Weight& weight);
[[nodiscard]] ScheduleRefusal ggml_refusal(Family family, int index, const Weight& weight,
                                           LinearPolicy policy);
// ggml's own admission of a weight to its matrix / vector kernel (mmq_prefill_wins and vec_call
// ask it), apart from route_admits' generic checks.
[[nodiscard]] bool ggml_mmq_admits(const Weight& weight, LinearPolicy policy);
[[nodiscard]] bool ggml_vec_admits(const Weight& weight, LinearPolicy policy);
// One of these families' coverage of a weight under `policy`: every declared schedule with its
// refusal (compiled: the schedules this build holds for a format the family reads). No pick: the
// selection prices these families against the others (cost_route), and the call decides.
[[nodiscard]] FamilyCoverage other_family_coverage(Family family, const Weight& weight,
                                                   LinearPolicy policy);

// Each generic family as an alternative for a call (#306): the family's route for it, its cost
// where the selection prices it, else why the family has none.
// - route: the family's cheapest priced route, else its pick where that is legal, else its first
//   legal schedule.
// - seconds: cost_route's time for that route, where heuristic_route prices the call (17 tokens or
//   more outside the prefill profile, not NVFP4) and the route is one it prices.
// - refusal: set where no route. It is the pick's refusal, else the first schedule's.
// - call: R8's refusal of the call itself; there is then no route.
// - compiled: as the family's coverage counts it.
// Which route the call selects is select_route's; this lists, it does not choose.
struct FamilyAlternative {
    Family family = Family::RegisterMma;
    std::optional<Route> route;
    std::optional<double> seconds;
    ScheduleRefusal refusal = ScheduleRefusal::None;
    R8CallRefusal call      = R8CallRefusal::None;
    int compiled            = 0;
};
[[nodiscard]] std::vector<FamilyAlternative> family_alternatives(const Problem& problem,
                                                                 const Weight& weight,
                                                                 std::int32_t tokens, bool prefill);
// cost_route's time for `route` on a call of `tokens` tokens (the A16 scale applied). nullopt where
// it does not price the route: no roofline in the facts, a family it does not price, or a route
// route_admits refuses.
[[nodiscard]] std::optional<double> route_cost_seconds(const Problem& problem, const Weight& weight,
                                                       Route route, std::int32_t tokens);

// The compiled resources of the kernel a launch runs (#298). The register-MMA kernels' come from
// the committed kernel-facts snapshot of the SASS the device runs (ops/generic/kernel_facts.h,
// #612: what ptxas made of the release's kernels, so the price is the same on every machine); the
// other families' are read from the driver for this binary on this device (cudaFuncGetAttributes).
// The launch geometry is the call's: its grid, threads and dynamic shared memory.
struct KernelResources {
    int registers               = 0; // per thread
    std::size_t local_bytes     = 0; // per thread: register spills and stack arrays
    std::size_t static_smem     = 0; // per CTA
    std::size_t dynamic_smem    = 0; // per CTA, the launch's own
    int threads                 = 0; // per CTA, the launch's
    int max_threads_per_block   = 0; // what the kernel's registers and smem allow
    int ptx_version             = 0; // the PTX and SASS it was compiled to (major * 10 + minor)
    int binary_version          = 0;
    unsigned grid_x = 0, grid_y = 0, grid_z = 0;
    // The register-MMA kernel variant the launch took: its latency form (rmma_latency_pays), or
    // neither form (the wide decoder or the plain kernel, told apart by the registers).
    bool latency_variant = false;
};

// The compiled facts the cost model read for one register-MMA schedule of one format (#556): what
// price_routes took from rmma_resources. A test installs a sink to digest exactly these reads (the
// selector golden keys a facts-priced board's file by them); nothing is recorded otherwise.
struct RmmaFactRead {
    QType qtype;
    int schedule;
    Epilogue epilogue;
    KernelResources facts;
};

class ScopedRmmaFactsSink {
public:
    explicit ScopedRmmaFactsSink(std::vector<RmmaFactRead>& sink) noexcept;
    ~ScopedRmmaFactsSink();
    ScopedRmmaFactsSink(const ScopedRmmaFactsSink&)            = delete;
    ScopedRmmaFactsSink& operator=(const ScopedRmmaFactsSink&) = delete;

private:
    std::vector<RmmaFactRead>* previous_;
};

void note_rmma_fact_read(const RmmaFactRead& read) noexcept;

// What register-MMA schedule `schedule` runs for a `tokens`-token call of `weight` under `epilogue`
// on the current device: the same launch path as run_route takes (the codec's decoder, its wide or
// latency variant by the same rules), described instead of launched. nullopt where the schedule is
// not compiled for the weight's codec or the build (rmma_refusal's NotCompiled, CodecTile), the
// generic path does not read the weight, or K is off the codec's step (the call runs scalar). On a
// device without tensor mma.sync the HMMA and GEMV routes Volta takes are not described.
// `rows` is the call's output rows (a slice of a fused weight has fewer than the weight's stored
// ones; SwiGLU: the gate rows); 0 takes the weight's own.
[[nodiscard]] std::optional<KernelResources> rmma_resources(int schedule, const Weight& weight,
                                                            Epilogue epilogue, std::int32_t tokens,
                                                            std::int32_t rows = 0);

// The same for R8 schedule `schedule`: the R8 kernel run_route launches for the call, its
// in-kernel quantization or, at verify widths where the format has it, its pre-quantized form
// (`prequantized`: one activation quantizer runs before it, not described). nullopt where
// resolve_route takes the call off R8 (the format, K, or tokens past the kernel's fit), with the
// workspace the call needs given (a weight with a k_order also gathers x into the workspace, which
// can leave a production call short of the pre-quantized form's planes).
struct R8Resources {
    KernelResources kernel;
    bool prequantized = false;
};
[[nodiscard]] std::optional<R8Resources> r8_resources(int schedule, const Weight& weight,
                                                      Epilogue epilogue, std::int32_t tokens);

// The kernel `route` runs for the call, for every family whose kernel the generic path launches
// itself: the register MMA and R8 (rmma_resources, r8_resources) and the shared-memory tiles
// (TiledMma, StagedMma), the grouped W4A8 tile and its producer-warp form (W4a8Mma) and the I8 tile
// (I8Mma), the FP4 tiles (W4a4Mma, Nvfp4TiledMma) and ggml's matrix and vector kernels (GgmlMmq on
// the call's first prefill slice, GgmlVec on its first kVecMaxColumns tokens: a wider call's last
// part can run another instantiation of the kernel), each with the workspace the call needs given
// and its activation quantizer (and ggml's stream-K fixup and the matrix kernel's epilogue kernel)
// not described. nullopt where resolve_route takes the call off `route`, on Volta's CUTLASS band of a
// tiled call, and for the scalar route.
[[nodiscard]] std::optional<KernelResources> route_resources(Route route, const Weight& weight,
                                                             Epilogue epilogue,
                                                             std::int32_t tokens);

// The output tile one CTA of `route` computes for a `tokens`-token call (#298), as the cost model
// prices it (price_routes, a8_candidates read this): output rows (under SwiGLU each reads its gate
// and its up row), token columns, and the CTAs K is split over (a register-MMA cluster). Declared
// geometry of the schedule, not a measurement. nullopt for the families the cost model does not
// price by a declared tile (R8, the FP4 tiles, scalar) and for ggml's kernels, whose matrix
// kernel picks its tile at launch by device and format (select_launch; route_resources
// describes that launch).
struct RouteTile {
    int out_rows = 0;
    int tokens   = 0;
    int k_split  = 1;
};
[[nodiscard]] std::optional<RouteTile> route_tile(Route route, Epilogue epilogue,
                                                  std::int32_t tokens);

// The logical traffic of a call on `route` (#298): the bytes its CTAs request, before any cache
// serves them, predicted from route_tile. Every token column of CTAs reads the whole weight
// (payload_bytes over the call's n rows, as the cost model's bpw), every row block of CTAs reads the
// whole activation (the register MMA's every row warp: no shared memory holds its operands) in the route's representation (A16 bf16: 2 bytes a value; the int8 tiers one
// byte and their scales: W4A8 36 bytes a 32-value block, 40 with a min's block sums, over K in
// whole 128-value tiles; I8 an FP32 scale and an int32 sum a 64, over tokens padded to 256),
// and the output is written once in bf16 (a cluster's K split reduces in distributed shared
// memory). A prediction, not a counter: the caches serve the re-reads (RTX 5090, Qwen3.5-9B
// shapes: a 16 x 8 register-MMA tile at T = 512 requested 8 TB/s, past the L2's rate, its
// activation re-read by every row block from L1), so a hit rate turns it into DRAM bytes.
struct RouteTraffic {
    double weight_bytes     = 0;
    double activation_bytes = 0;
    double output_bytes     = 0;
};
[[nodiscard]] std::optional<RouteTraffic> route_traffic(Route route, const Weight& weight,
                                                        const Problem& problem,
                                                        std::int32_t tokens);

// What this build holds of the generic GEMM families (#307), read without a device: every declared
// schedule, whether the binary holds its kernel, and where that depends on the format, which
// formats. The refusals' NotCompiled and CodecTile read the same predicates: a schedule this build
// omits (rmma_schedule_launchable: a cluster split outside the 120a build; w4a4_schedule_count;
// kNativeNvfp4), the formats a schedule's kernel alone is compiled for (the producer-warp W4A8
// tile, w4a8_ws_supports; ggml's kernels, their own types), and the formats whose decoder omits a
// register-MMA tile (rmma_tile_compiled), or whose tile is past the build's shared-memory ceiling
// and compiles to a stub (tiled_fits, w4a8_fits, i8_fits; staged_fits for a whole schedule: the
// refusals' Resource); the I8 kernels from sm_80 on. What the device can run (instructions, its
// own shared memory, staged_device_fits) is the coverage queries' to say, not this one's.
struct OmittedFormat {
    QType qtype;
    // 0: at every K; else at K a multiple of this (the u4 G32 / G64 quad-run decoder from 128).
    std::int32_t k_multiple = 0;
};
struct ManifestSchedule {
    Route route;
    bool compiled = false;
    std::vector<QType> only_formats; // empty: every format the family reads
    std::vector<OmittedFormat> omitted_formats;
};
struct BuildManifest {
    int arch           = 0; // the architecture the tree compiles for (NINFERNO_BUILD_ARCH)
    bool arch_specific = false;
    bool native_nvfp4  = false;
    // Volta's CUTLASS prefill (generic_volta_cutlass.cu, volta_cutlass_compiled): on a build
    // without tensor mma.sync, what the tiled route's schedule 0 runs for the formats without a
    // tile of their own; without it those formats are omitted there.
    bool volta_cutlass = false;
    std::vector<ManifestSchedule> schedules; // family by family, in schedule order
    std::vector<QType> r8_verify_formats;    // R8's verify-width form (r8_verify_compiled)
};
[[nodiscard]] BuildManifest build_manifest();

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
// The route Volta's own rules give a call before any tuned or tabled one (select_route):
// volta_tiled_route, else the tensor-core route of the verify widths; none on GPUs with tensor mma.sync.
[[nodiscard]] std::optional<Route> volta_fixed_route(const Problem& problem, const Weight& weight,
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
// A route pin (ops/route_pins.h: --prefill-route, --decode-route) applies after the invariant
// profile, a route probe and Volta's tiled route (which the pin ledger notes as the a16 family's):
// a call of its band takes the pinned family where pinned_family_route gives one, else the
// automatic route; without a pin nothing changes.
[[nodiscard]] Route select_route(const Problem& problem, const Weight& weight, std::int32_t tokens,
                                 bool prefill);

// The route a pinned family gives a call of its band (prefill: 17 tokens or more; decode:
// 1..16), nullopt where the family cannot compute it or T is outside the band. A16 is the
// automatic selection under an A16Only policy; W4A8 the W4A8 heuristic's tile (with its tuned
// prefill band); I8 the I8 cost rule's tile; Mmq ggml's matrix kernel (prefill in 512-token
// slices; at decode widths its 5..16-token band); W4A4 the W4A4 heuristic's tile; R8 the R8 rules'
// schedule (1 token where the activation fits a CTA, 2..8 on the pre-quantized form); Vec ggml's
// vector kernel.
[[nodiscard]] std::optional<Route> pinned_family_route(PrefillRoute family, const Problem& problem,
                                                       const Weight& weight, std::int32_t tokens,
                                                       bool prefill);
[[nodiscard]] std::optional<Route> pinned_family_route(DecodeRoute family, const Problem& problem,
                                                       const Weight& weight, std::int32_t tokens,
                                                       bool prefill);

// Whether this build and the current device hold the kernels of a pinnable family at all (the
// startup's first refusal, before any weight is read).
[[nodiscard]] bool pin_family_on_device(PrefillRoute family);
[[nodiscard]] bool pin_family_on_device(DecodeRoute family);

// The W4A8 tile of the rules a device without clocks runs (the cost model's fallback), else the
// first schedule that fits the build; nullopt where none admits the weight.
[[nodiscard]] std::optional<Route> w4a8_route(const Problem& problem, const Weight& weight,
                                              std::int32_t tokens);
// The I8 tile of a call: i8_coverage's pick where the weight is admitted to it; nullopt where
// none admits it.
[[nodiscard]] std::optional<Route> i8_route(const Problem& problem, const Weight& weight,
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
// q8_1 activation of a [K, T] x (Call::prepared), and that quantization of x.
[[nodiscard]] bool ggml_vec_takes(const Problem& problem, const Weight& weight,
                                  std::int32_t tokens);
[[nodiscard]] std::size_t ggml_vec_activation_bytes(std::int32_t k, std::int32_t tokens);

// An activation prepared once for several calls that read one x (#308): what it was made from and
// by which rule, so a call takes it only where its own preparation would give the same bytes. The
// representation is the rule: a nominal "int8 activation" is no contract (W4A8's codes, I8's and
// ggml's q8_1 group, scale and sum differently). Non-owning; valid while its source and storage are
// (the Op call that made it), never cached by address across calls.
enum class PreparedRepresentation : std::uint8_t {
    GgmlQ8_1, // ggml's block_q8_1 of 32 values (vec_quantize_activation: per-block absmax scale,
              // rounded to nearest, the block's sum), [K, T] token-major
    // launch_w4a8_quantize: 32-value blocks, an FP32 amax/127 scale, rounded to nearest, the scale
    // times the codes' sum; [T, ld] codes, [T, ld/32] scales and sums, the blocks past K zero (the
    // W4A8 tiles at ld = w4a8_ld(K), K up to whole 128-value tiles; the R8 verify widths'
    // pre-quantized form at ld = K).
    W4a8Blocks,
    // launch_i8: 64-value groups, the codes k-permuted, int sums, [padded T][K] (generic_i8.cu).
    I8Blocks64,
    // ggml MMQ's block_q8_1_mmq tiles (ggml_mmq.cu), whose scale/sum layout is the weight format's.
    GgmlQ8_1Mmq,
    // W4A4: NVFP4 codes [T, K/2] and FP8 scales [T, K/16] under the weight's input divisor.
    Nvfp4,
};
struct PreparedActivation {
    PreparedRepresentation representation = PreparedRepresentation::GgmlQ8_1;
    const void* data   = nullptr; // the prepared bytes
    const void* source = nullptr; // the BF16 x they were made from
    std::int32_t ldx    = 0;      // its stride, BF16 values from one token to the next
    std::int32_t k      = 0;
    std::int32_t tokens = 0;
};
// Whether `prepared` is what a call of `representation` would make itself from x at `source`
// (stride `ldx`, `k` x `tokens`): the same rule, the same source region and extent. A permuted
// input (a weight stored in another column order reads x gathered) is another source.
[[nodiscard]] bool prepared_serves(const PreparedActivation* prepared,
                                   PreparedRepresentation representation, const void* source,
                                   std::int32_t ldx, std::int32_t k, std::int32_t tokens) noexcept;
// ggml's q8_1 of x into `out` (ggml_vec_activation_bytes of it), described.
PreparedActivation ggml_vec_quantize(const Tensor& x, void* out, cudaStream_t stream);

// The SwiGLU (GeGLU: gelu) of a gate and an up weight on one x: out[r] = act(gate row r . x) *
// (up row r . x) for `rows` rows from each half's `row`, out BF16 [>= rows, T]. Where both halves
// take ggml's vector kernel at this T (the dense tuner's or the heuristic's route, as a call of
// that half alone would), x is quantized once (or `prepared` carries it where it serves): a pair of
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
                                       WorkspaceArena& workspace,
                                       const PreparedActivation* prepared, cudaStream_t stream);
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
// Instrumentation (tests; one relaxed fetch_add per joined launch): launches that ran joined to the
// call before them (launch_overlap's join), since the process
// started: the segmented chain's and the tests' view of what the rule granted.
[[nodiscard]] std::uint64_t instrumentation_joined_launches() noexcept;
// Calls whose route ran another one with other bytes (any family: a workspace short of it, an
// unaligned x, a shape it cannot hold), since the process started; each pair of routes is also
// said once on stderr. A demotion that keeps the bytes (staged or NVFP4 tiled to the register MMA
// of the same order) is not one.
[[nodiscard]] std::uint64_t route_demotion_count() noexcept;

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
    // calls that read one x (segmented_projection); a vector call takes it where it serves the x it
    // reads (prepared_serves), else (null included) quantizes x itself.
    const PreparedActivation* prepared = nullptr;
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

// What a call guarantees the route it runs on, beside its weight (#298): its activation's
// alignment and stride, its tokens, rows, output segments and epilogue, and the workspace free to
// it (from the arena's next 256-byte boundary; nullopt: no workspace). A plan describes the call
// it will make (route_query's contiguous, aligned activation); run_route the call it is given.
struct CallLayout {
    bool x_aligned16  = true;
    int ldx           = 0; // BF16 values from one token to the next (the call's k: contiguous)
    int tokens        = 1;
    int rows          = 0; // the call's output rows (SwiGLU: the gate's)
    int segments      = 1;
    Epilogue epilogue = Epilogue::Store;
    std::optional<std::size_t> workspace_free;
};

// Why the route that runs is not the one asked for (the last step that changed it).
enum class Demotion : std::uint8_t {
    None,
    Alignment,     // the activation is off the 16-byte words or the stride its staging reads
    W4a8Admission, // a producer-warp W4A8 tile the weight is not admitted to: its classic tile
    R8Fit,         // R8 neither fits the tokens in a CTA nor has its pre-quantized form here
    GgmlLayout,    // ggml's kernels: rows off the matrix kernel's multiple, or too many segments
    GgmlWorkspace, // ggml's kernels: no workspace, or less free than their planes take
    Workspace,     // W4A4, W4A8 or I8 given less workspace than their activation planes take
};
[[nodiscard]] std::string_view demotion_name(Demotion demotion) noexcept;

// The route a call runs on when `asked` is selected for it: run_route's own resolution (the same
// function), with the R8 verify widths' pre-quantized form where it takes it. `executed` equals
// `asked` where nothing demotes it. Not the explicit-route refusals (a route this build lacks, a
// weight layout the generic path does not read), which run_route throws, nor the scalar route a
// register-MMA launch the device refuses falls back to at the launch.
struct ResolvedRoute {
    Route asked;
    Route executed;
    bool r8_prequantized = false;
    bool ggml_fallback   = false; // a ggml route the call could not take (ggml_mmq_fallback_count)
    Demotion reason      = Demotion::None;
};
[[nodiscard]] ResolvedRoute resolve_route(Route asked, const Weight& weight, const CallLayout& call);

// The activation a resolved route prepares for a call on `weight` (#308): the rule's
// representation, its row length where it has one (W4a8Blocks), and its variant (GgmlQ8_1Mmq: the
// weight's format, the scale/sum layout follows it; formats that share one are counted apart, so
// no reuse is claimed between them). weight_bound: the bytes depend on the weight itself (W4A4's
// input divisor) and serve no other weight. nullopt where the kernel reads x as BF16 (the A16
// families) or quantizes it inside itself (R8 at decode).
struct PreparedRule {
    PreparedRepresentation representation = PreparedRepresentation::GgmlQ8_1;
    std::int32_t ld      = 0;
    std::int32_t variant = 0;
    bool weight_bound    = false;
};
[[nodiscard]] std::optional<PreparedRule> prepared_rule(const ResolvedRoute& route,
                                                        const Weight& weight);
// Whether the bytes one rule prepares are the bytes the other's call would make itself from the
// same x (its source, stride and extent are each call's: prepared_serves): the same
// representation, row length and variant, neither bound to its weight. The same nominal activation
// dtype is no such match: W4A8's, I8's and ggml's int8 forms group, scale and sum apart. A match
// is of the rule only; a consumer that shares bytes also needs prepared_serves (K, T, stride and
// source) and a buffer that holds the whole call: MMQ quantizes kMmqPrefillSlice tokens at a time
// into one reused buffer, ggml's vector kernel 8 at a time into scoped ones, and I8 lays its
// scales out by the padded token count.
[[nodiscard]] bool prepared_rules_match(const PreparedRule& a, const PreparedRule& b) noexcept;
// The overlap a launch allows its neighbours (#309), run_route's one rule. Three events apart:
// join (this launch starts under the call before it, which it waits for before exiting:
// Call::join_previous), release (the next launch may begin once every CTA of this one has
// triggered, after its own wait under prefetch, or exited: the next call may then read this
// launch's inputs, not yet its output; release is not completion, and this launch's scratch stays
// in use until it finishes), prefetch (its weight stream starts under the previous kernel,
// GemmArgs::pdl_prefetch). Each refused names why: the call did not ask, the family or epilogue
// does not join, the pre-quantized R8 form (it neither joins, its quantizer and kernel launching in
// stream order, nor releases: the next call's quantizer could take the workspace it still reads),
// no programmatic dependent launch on the device, a call wider than the prefetch's tokens, or a
// joined call.
enum class OverlapRefusal { None, NotAsked, Family, Epilogue, PrequantizedR8, NoPdl, Wide, Joined };
[[nodiscard]] std::string_view overlap_refusal_name(OverlapRefusal why) noexcept;
struct LaunchOverlap {
    OverlapRefusal join     = OverlapRefusal::NotAsked;
    OverlapRefusal release  = OverlapRefusal::NotAsked;
    OverlapRefusal prefetch = OverlapRefusal::NotAsked;
    [[nodiscard]] bool joins() const noexcept { return join == OverlapRefusal::None; }
    [[nodiscard]] bool releases() const noexcept { return release == OverlapRefusal::None; }
    [[nodiscard]] bool prefetches() const noexcept { return prefetch == OverlapRefusal::None; }
};
[[nodiscard]] LaunchOverlap launch_overlap(const ResolvedRoute& resolved, Epilogue epilogue,
                                           int tokens, bool join_asked, bool release_asked,
                                           bool pdl);
// The route run() hands run_route for a call `selected` for: a quantized family that needs
// workspace (W4A4, W4A8, I8) takes the A16 heuristic's route on a call without one.
[[nodiscard]] Route workspace_free_route(Route selected, const Problem& problem,
                                         const Weight& weight, std::int32_t tokens, bool prefill,
                                         bool workspace);

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

// Under NINFERNO_PROJECTION_RECORD=<file>: each distinct (op, route, T, N, K) a specialized
// projection dispatch took, appended as "<op> <route> <T> <N> <K>". A debug trace of the
// width-dependent route choice (A4 or A16 by call width) that NINFERNO_ROUTE_RECORD, which holds
// generic-path problems only, does not show. Off (no file), a call costs one test.
void record_projection_route(const char* op, const char* route, std::int32_t tokens, std::int32_t n,
                             std::int32_t k);

} // namespace ninferno::ops::detail::generic
