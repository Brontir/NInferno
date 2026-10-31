#pragma once

// Private pieces of the generic MoE Op (include/ninferno/ops/moe.h): the kernel arguments, the
// compiled schedule list, the routing workspace and the launch entry points.
//
// A layer runs as six stream-ordered launches, none of which asks the host anything:
//   1. router   : logits[t, e] = router . x_t + bias             (grouped-GEMM kernel, Router)
//   2. select   : top-k experts and route weights per token      (one warp per token)
//   3. plan     : expert-major slot order and the job list        (one CTA; a call of at most
//                 kPlanWarps tokens selects in the same CTA, one launch for 2 and 3)
//   4. gate_up  : mid[slot] = act(gate . x, up . x)               (grouped-GEMM kernel, GateUp)
//   5. down     : y[slot] = down . mid[slot] + bias               (grouped-GEMM kernel, Down)
//   6. combine  : out[t] (+)= sum_k w_k y[slot(t, k)] (+ shared)  (one CTA per token)
// The grouped-GEMM kernel is the generic projection path's register-MMA body (the same weight
// decoders, so every format that path computes is an expert format here) with MoE addressing: a
// job is (expert, run of up to kJobColumns slots); its rows come from the expert's span of the
// stacked plane and its columns are gathered through the slot -> token map.

#include "core/tensor.h"
#include "ninferno/ops/moe.h"
#include "ops/generic/generic_gemm.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::ops::detail::moe {

// Up: a non-gated (Relu2) expert's first projection, one set of rows whose one sum the activation
// takes; GateUp reads a gate and an up row per output.
enum class Stage : std::uint8_t { Router, GateUp, Down, Up };

// Columns of one job: the slots of one expert, cut into runs of at most this many. A call picks
// its job width (the columns one gate/up CTA covers) up to this.
inline constexpr int kJobColumns = 64;
inline constexpr int kMaxExperts = 1024;
inline constexpr int kMaxTopK    = 32;
// Warps of the single-CTA plan kernel; its shared memory holds kPlanWarps x E counters.
inline constexpr int kPlanWarps = 16;

// Most jobs `assignments` slots can make over E experts: every expert's last run may be partial.
[[nodiscard]] constexpr std::int32_t max_jobs(std::int32_t assignments, std::int32_t experts,
                                              std::int32_t width) {
    return (assignments + width - 1) / width + (assignments < experts ? assignments : experts);
}

// Arguments of the routing kernels (moe_route.cu).
struct SelectArgs {
    const float* logits         = nullptr; // [T, E]
    const float* selection_bias = nullptr;
    const float* expert_scale   = nullptr;
    int experts                 = 0;
    int top_k                   = 0;
    int tokens                  = 0;
    int sigmoid                 = 0;
    int renormalize             = 1;
    float routed_scale          = 1.0F;
    int* ids                    = nullptr; // [T, k]
    float* weights              = nullptr; // [T, k]
};

struct PlanArgs {
    const int* ids    = nullptr; // [A]
    int assignments   = 0;
    int top_k         = 0;
    int experts       = 0;
    int job_columns   = kJobColumns;
    int* packed_token = nullptr; // [A] slot -> token
    int* slot_of      = nullptr; // [A] assignment -> slot
    int* job_expert   = nullptr; // [max_jobs]
    int* job_begin    = nullptr;
    int* job_count    = nullptr;
    int* job_total    = nullptr; // [1]
};

// The SIMT router (moe_router_simt.cu): warps per CTA, the widest call, and its dynamic shared
// memory: x staged as BF16.
inline constexpr int kRouterSimtWarps     = 4;
inline constexpr int kRouterSimtMaxTokens = 4;

[[nodiscard]] constexpr std::size_t router_simt_smem_bytes(int tokens, int k) {
    return static_cast<std::size_t>(tokens) * static_cast<std::size_t>(k) * 2;
}

struct CombineArgs {
    const void* y           = nullptr; // BF16 [A, H] in slot order
    const int* slot_of      = nullptr; // [A]
    const float* weights    = nullptr; // [T, k]
    const void* shared      = nullptr; // optional BF16 [T, H] shared-expert output
    const void* shared_gate = nullptr; // optional BF16 [T] gate logit (sigmoid applied here)
    void* out               = nullptr; // BF16 [T, H]
    int hidden              = 0;
    int top_k               = 0;
    int tokens              = 0;
    int add_residual        = 0;
    int pdl                 = 0; // host only, as GemmArgs::pdl
};

struct GemmArgs {
    generic::GemmArgs w; // the weight planes and K; its token/row/output fields are unused
    const void* x = nullptr;
    int ldx       = 0;
    int tokens    = 0; // Router: the call's T
    // Grouped stages: the job list the plan kernel wrote.
    const int* job_expert   = nullptr;
    const int* job_begin    = nullptr;
    const int* job_count    = nullptr;
    const int* job_total    = nullptr;
    const int* packed_token = nullptr; // GateUp: slot -> token; Down: none (the slot is the column)
    int sub_tiles           = 1;       // job width / columns of one CTA
    int r8_lanes            = 32;      // an R8 warp's lanes per row: 8, 16 or 32 (moe_r8.cuh)
    // An R8 gate/up stage's activation quantized once per token (codes [T, x_ld], scales and sums
    // [T, x_ld / 32], quantize_x_task's layout); nullptr: every CTA quantizes its slots.
    const std::int8_t* x_codes = nullptr;
    const float* x_scales      = nullptr;
    const float* x_sums        = nullptr;
    int x_ld                   = 0;
    // Router: its CTAs also quantize xq (T rows at stride xq_ldx, xq_k values) into xq_codes,
    // xq_scales and xq_sums at stride xq_ld, the tasks spread over the grid: the R8 gate/up's
    // x_codes, without a launch of their own.
    const void* xq        = nullptr;
    int xq_ldx            = 0;
    int xq_k              = 0;
    int xq_ld             = 0;
    std::int8_t* xq_codes = nullptr;
    float* xq_scales      = nullptr;
    float* xq_sums        = nullptr;
    // Rows: output row l of expert e reads weight row e * expert_rows + l * row_step (+ up_offset
    // for GateUp's up row).
    int rows        = 0;
    int expert_rows = 0;
    int row_step    = 1;
    int up_offset   = 0;
    // Epilogue.
    const void* bias = nullptr; // indexed like the weight row (Router: by expert)
    int bias_bf16    = 0;
    int activation   = 0; // MoeActivation
    float alpha      = 1.0F;
    float limit      = 0.0F;
    float* out_f32   = nullptr; // Router: [T, E]
    void* out_bf16   = nullptr; // GateUp / Down: [slot, ldo]
    int ldo          = 0;
    // Down (R8): the combine in the down stage (moe_r8.cuh). The last CTA of each run of output
    // rows to finish, counted on combine_tickets[blockIdx.x], adds the run's rows of every token's
    // experts into combine.out as combine_kernel does (combine.shared unset: no shared expert),
    // and the separate combine launch is skipped. nullptr: the down stage stores y only.
    int* combine_tickets = nullptr;
    CombineArgs combine;
    // Host only: launch as a programmatic dependent of the stream's previous kernel (core/pdl.cuh).
    // Every grouped kernel waits for its producer before its first global load.
    int pdl = 0;
};

// The compiled schedule list: (warps_m, warps_k, m_tiles, n_tiles) of the register MMA.
#define NINFERNO_MOE_SCHEDULES(X)                                                                  \
    X(0, 1, 8, 1, 1)                                                                               \
    X(1, 1, 4, 1, 1)                                                                               \
    X(2, 2, 2, 1, 1)                                                                               \
    X(3, 4, 1, 1, 1)                                                                               \
    X(4, 4, 1, 1, 4)                                                                               \
    X(5, 4, 1, 1, 8)                                                                               \
    X(6, 4, 1, 2, 8)                                                                               \
    X(7, 1, 8, 1, 4)                                                                               \
    X(8, 1, 4, 1, 4)

struct ScheduleInfo {
    int warps_m, warps_k, m_tiles, n_tiles;
    [[nodiscard]] int rows_per_cta() const { return warps_m * m_tiles * 16; }
    [[nodiscard]] int tokens_per_cta() const { return n_tiles * 8; }
};

[[nodiscard]] int schedule_count();
[[nodiscard]] ScheduleInfo schedule_info(int index);

// The grouped R8 kernel's schedules (moe_r8.cuh): (warps, rows per warp) of generic::R8Schedule.
#define NINFERNO_MOE_R8_SCHEDULES(X)                                                               \
    X(0, 4, 1)                                                                                     \
    X(1, 4, 2)                                                                                     \
    X(2, 8, 1)                                                                                     \
    X(3, 2, 1)

struct R8ScheduleInfo {
    int warps, rows;
    [[nodiscard]] int rows_per_cta() const { return warps * rows; }
};

[[nodiscard]] int r8_schedule_count();
[[nodiscard]] R8ScheduleInfo r8_schedule_info(int index);

// The grouped R8 route: the codebook i-quant experts (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S),
// whose register-MMA decode is the costly part of a decode call, under an A8 expert policy, for
// calls of at most kR8Tokens tokens (the dense R8's decode scope): one R8 CTA covers a whole job,
// at most T slots (a token selects an expert once), whose quantized activation must fit it
// (r8_admits). Wider calls (speculative verify, prefill) compute A16.
inline constexpr std::int32_t kR8Tokens = 4;
// The formats the R8 kernel reads: the codebook ones (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S),
// the K-quants (Q3_K, Q4_K, Q5_K, Q6_K) and the 32-value ggml blocks (Q4_0, Q4_1, Q5_0, Q5_1,
// IQ4_NL, IQ4_XS).
[[nodiscard]] bool r8_expert_format(QType format);
// Whether an R8 stage may take `bank` at `tokens` (the policy, the format, T <= kR8Tokens and the
// quantized activation within a CTA's shared memory).
[[nodiscard]] bool r8_admits(const MoeWeights& w, const Weight& bank, std::int32_t tokens);
[[nodiscard]] bool r8_expert_supports(const Weight& w);
// Lanes per row of an R8 warp on this bank (GemmArgs::r8_lanes); an R8 CTA covers
// rows_per_cta() * 32 / r8_lanes rows.
[[nodiscard]] int r8_lanes(const Weight& w);

// The route of one call: a compiled schedule per stage and the job width (the columns one job
// covers; a multiple of both grouped register-MMA schedules' columns, at most kJobColumns). A
// grouped stage with an R8 schedule (gate_up_r8 / down_r8 >= 0) runs the R8 kernel instead of its
// register-MMA schedule: one CTA covers a whole job (at most T <= kR8Tokens slots), so it adds no
// job-width constraint. a16: a route-table band measured on A16 ("|a16" in its name), which takes
// no default A8 route on top.
struct MoePlan {
    int router_schedule  = 0;
    int gate_up_schedule = 0;
    int down_schedule    = 0;
    int job_columns      = 8;
    int gate_up_r8       = -1;
    int down_r8          = -1;
    bool a16             = false;
    // Both grouped stages on the int8 tier (moe_w4a8.cuh) instead of their register-MMA
    // schedules: a wide prefill call under an A8 expert policy (w4a8_admits); job_columns stays
    // the prefill width, a multiple of the W4A8 tile's slots.
    bool w4a8 = false;
};

// The route moe takes for T tokens under the given profile, and moe on an explicit route (the
// tuner, benches and tests); the explicit route's workspace need.
[[nodiscard]] MoePlan select_plan(const MoeWeights& weights, const MoeConfig& config,
                                  std::int32_t tokens, bool prefill);
void moe_with_plan(const Tensor& x, const MoeWeights& weights, const MoeConfig& config,
                   MoeEpilogue epilogue, Tensor& out, WorkspaceArena& workspace,
                   cudaStream_t stream, const MoePlan& plan, const Tensor* router_input = nullptr);
[[nodiscard]] std::size_t plan_workspace_bytes(const MoeWeights& weights, const MoeConfig& config,
                                               std::int32_t tokens, const MoePlan& plan);
// Whether every stage's schedule is compiled for its weight's format (a 2 x 8 register-MMA tile is
// not, for the formats that read 32-value blocks); moe_with_plan refuses a plan that is not.
[[nodiscard]] bool plan_compiled(const MoeWeights& weights, const MoePlan& plan);
// Whether a grouped stage over `bank` (weights.gate_up or weights.down) may take the R8 kernel at
// T tokens: an A8 expert policy, a codebook format with 16-byte aligned planes, T <= 4 and an
// activation that fits a CTA; moe_with_plan refuses an R8 stage that is not.
[[nodiscard]] bool r8_admits(const MoeWeights& weights, const Weight& bank, std::int32_t tokens);

// The current GPU class's default A8 route (moe.cpp's per-class table): the R8 schedule of a
// codebook gate/up bank at each T = 1..kR8Tokens, for a bank in its published layout and in
// ggml_frag_v1 (-1: that call stays A16), and the register-MMA schedule an IQ3_XXS down of width
// 512..1023 of an A8 call takes instead of the streaming 1 x 4 warps (-1: the streaming one), the
// same at every T so that a token's bytes do not depend on T. nullopt: the class was not measured,
// and its default route stays A16.
struct A8Route {
    // The R8 schedule of a published 2-bit codebook gate/up bank (IQ2_XXS, IQ2_XS, IQ2_S) at each
    // T; every other R8 bank, and a ggml_frag_v1 bank, takes w4r1 (schedule 0) at every T.
    std::array<int, kR8Tokens> gate_up_2bit{};
    int down = -1;
    [[nodiscard]] int gate_up_at(std::int32_t tokens, const Weight& bank) const {
        const bool two_bit = bank.qtype == QType::GGML_IQ2_XXS ||
                             bank.qtype == QType::GGML_IQ2_XS || bank.qtype == QType::GGML_IQ2_S;
        return two_bit && bank.layout != QuantLayout::GgmlFrag
                   ? gate_up_2bit[static_cast<std::size_t>(tokens - 1)]
                   : 0;
    }
};
[[nodiscard]] std::optional<A8Route> a8_default_route();

// Whether every stage of the plan sums each dot product in one order whatever T is.
[[nodiscard]] bool plan_is_token_invariant(const MoePlan& plan);

// A plan's stable name in the route table: "<gate/up schedule>|<down schedule>", each schedule
// named as the generic path names its register-MMA routes ("rmma.m1.n1.w1x8") or its R8 routes
// ("r8.w4r1"), and "|a16" after a plan of register-MMA stages marked A16. The router keeps its own
// heuristic schedule and the job width is the wider of the two grouped schedules.
[[nodiscard]] std::string plan_name(const MoePlan& plan);
[[nodiscard]] std::optional<MoePlan> parse_plan(std::string_view name, std::int32_t hidden);

// The per-GPU-class route table's MoE entries (routes/<class>.json, "moe"): per exact layer
// (expert format, E, top-k, H, F) a plan per token band and one prefill plan, which the loader
// refuses unless it is token-invariant (src/ops/route_table.cpp). A band naming an R8 stage is
// refused unless the format is a codebook one and the band ends at T <= kR8Tokens. A band of
// register-MMA stages of a codebook format takes the class's default A8 gate/up on top at
// T <= kR8Tokens under an A8 expert policy; a band tuned to stay A16 there says so with "|a16".
// The key names both expert banks' formats: a GGUF mixture keeps gate/up and down in different
// blocks (Nemotron's q5_1 gate/up over q5_0 down), and the entry says "down_format" then.
struct MoeRouteKey {
    QType qtype = QType::BF16, down_qtype = QType::BF16;
    std::int32_t experts = 0, top_k = 0, hidden = 0, inter = 0;
};
struct MoeRouteBand {
    std::int32_t t_min = 1, t_max = 1;
    MoePlan plan;
};
struct MoeRouteEntry {
    MoeRouteKey key;
    std::vector<MoeRouteBand> bands;
    std::optional<MoePlan> prefill;
};
// The load-time tuner (moe_tune.cpp): decode widths it covers, its plan for the layer at T
// (nullopt: not tuned, or the default stays), and select_plan without it (the route table, else
// the heuristics). table_covers: a route-table band serves the layer at T (the tuner leaves it).
inline constexpr std::int32_t kTuneMaxTokens = 16;
[[nodiscard]] std::optional<MoePlan> tuned_plan(const MoeWeights& w, const MoeConfig& c,
                                                std::int32_t tokens);
[[nodiscard]] MoePlan untuned_plan(const MoeWeights& weights, const MoeConfig& config,
                                   std::int32_t tokens, bool prefill);
[[nodiscard]] bool table_covers(const MoeWeights& weights, const MoeConfig& config,
                                std::int32_t tokens);
// Forgets every tuned plan (tests).
void clear_tuned_plans();

[[nodiscard]] std::optional<MoePlan> table_plan(const MoeRouteKey& key, std::int32_t tokens,
                                                bool prefill);
[[nodiscard]] MoeRouteKey route_key(const MoeWeights& weights, const MoeConfig& config);

// One grouped-GEMM launch of `stage` on a compiled schedule over `grid`; `frag`: the weight is a
// ggml_frag_v1 bank (core/ggml_frag.h), read by the format's fragment-order decoder.
void launch_gemm(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                 const GemmArgs& args, dim3 grid, cudaStream_t stream);
// One grouped R8 launch (GateUp or Down) of a call of `tokens` <= kR8Tokens tokens; grid y is
// the job list's capacity. `frag`: a ggml_frag_v1 bank (IQ2_S, IQ3_XXS), read in fragment order.
void launch_r8_gemm(generic::WeightCodec codec, bool frag, Stage stage, int schedule, int tokens,
                    const GemmArgs& args, dim3 grid, cudaStream_t stream);
// Per-family entry points (moe_r8_<family>.cu): false for a codec (or a frag bank) of another
// family.
bool launch_r8_gemm_iq2(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                        int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream);
bool launch_r8_gemm_iq3(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                        int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream);
bool launch_r8_gemm_kquant(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                           int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream);
bool launch_r8_gemm_ggml4(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                          int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream);
bool launch_r8_gemm_native(generic::WeightCodec codec, bool frag, Stage stage, int schedule,
                           int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream);

// The grouped W4A8 (moe_w4a8.cuh): the experts' gate/up (or up) and down stages of a wide prefill
// call on int8 tensor cores, the activation quantized per 32 values (int8 codes, a scale and the
// block's sum) at a row stride of K rounded up to 128 (ld) with a zero tail, so a K of whole
// 32-value blocks (gpt-oss 2880, Nemotron 1856, Gemma 4 704) reads whole 128-value tiles.
inline constexpr int kMoeW4a8BlockM = 64; // slots per CTA (divides kJobColumns)
// The narrowest call the speed profile hands the grouped W4A8 (the prefill profile does from
// T=17): a prefill chunk of a family that runs its prefill in the speed profile (Gemma 4), a
// verify of many rows.
inline constexpr std::int32_t kMoeW4a8SpeedMinTokens = 64;
inline constexpr int kMoeW4a8BlockN = 64; // weight tile rows per CTA (32 local rows when gated)
static_assert(kJobColumns % kMoeW4a8BlockM == 0);

struct MoeW4a8Activation {
    const std::int8_t* codes = nullptr; // [row][ld]
    const float* scales      = nullptr; // [row][ld / 32]
    const float* sums        = nullptr; // [row][ld / 32]: scale times the codes' integer sum
    int ld                   = 0;
};

struct MoeW4a8Buffers {
    std::int8_t* codes = nullptr;
    float* scales      = nullptr;
    float* sums        = nullptr;
    int ld             = 0;

    [[nodiscard]] MoeW4a8Activation view() const { return {codes, scales, sums, ld}; }
};

[[nodiscard]] inline int moe_w4a8_ld(int k) { return (k + 127) / 128 * 128; }

// Rows of BF16 x ([rows][ldx], K values each) into `out` (moe_w4a8_quantize.cu).
void launch_moe_w4a8_quantize(const void* x, int ldx, int k, int rows, const MoeW4a8Buffers& out,
                              cudaStream_t stream);
// The same rows as MXFP8 for launch_moe_w4a8_mx: E4M3 codes (satfinite) and per 32 values the
// power-of-two scale 2^e, the smallest with amax / 2^e <= 448 (sums untouched).
void launch_moe_mx8_quantize(const void* x, int ldx, int k, int rows, const MoeW4a8Buffers& out,
                             cudaStream_t stream);
// One grouped W4A8 launch of `stage` (GateUp, Up or Down) over `grid` (x: blocks of
// kMoeW4a8BlockN tile rows, y: jobs x sub-tiles of kMoeW4a8BlockM slots); false for a codec of
// another family (moe_w4a8_<family>.cu).
bool launch_moe_w4a8_ggml32(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                            const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream);
// MXFP4 banks on the block-scaled MMA against MXFP8 activations (launch_moe_mx8_quantize's codes):
// sm_120a family builds only; false elsewhere or for another codec.
bool launch_moe_w4a8_mx(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                        const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream);
bool launch_moe_w4a8_superblock(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                                const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream);
bool launch_moe_w4a8_codebook(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                              const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream);
// Q4/Q5/Q6 G64 and u4 G32/G64/G128 (moe_w4a8_native.cu).
bool launch_moe_w4a8_native(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                            const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream);
// A ggml_frag_v1 bank (Q4_0, Q4_K, IQ3_XXS, IQ2_S; moe_w4a8_frag.cu).
bool launch_moe_w4a8_frag(generic::WeightCodec codec, Stage stage, const GemmArgs& args,
                          const MoeW4a8Activation& act, dim3 grid, cudaStream_t stream);

// Per-format entry points (moe_gemm_<format>.cu).
#define NINFERNO_MOE_FORMATS(X)                                                                    \
    X(bf16)                                                                                        \
    X(fp8) X(nvfp4) X(mxfp4) X(q4) X(q5) X(q6) X(q8) X(fp8_row_f32) X(fp8_b128) X(u4zp)            \
        X(u4zp_g32) X(u4zp_g64) X(ggml_q4k) X(ggml_q5k) X(ggml_q6k) X(ggml_q3k) X(ggml_q2k)        \
            X(ggml_iq4xs) X(ggml_iq3xxs) X(ggml_iq3s) X(ggml_q40) X(ggml_q41) X(ggml_q50)          \
                X(ggml_q51) X(ggml_iq4nl) X(ggml_iq2xxs) X(ggml_iq2xs) X(ggml_iq2s) X(ggml_iq1s)   \
                    X(ggml_iq1m) X(ggml_tq10) X(ggml_tq20) X(ggml_q10) X(ggml_q20)
#define NINFERNO_MOE_DECLARE(F)                                                                    \
    void launch_gemm_##F(Stage stage, int schedule, const GemmArgs& args, dim3 grid,               \
                         cudaStream_t stream);
NINFERNO_MOE_FORMATS(NINFERNO_MOE_DECLARE)
// The ggml_frag_v1 banks (moe_gemm_<format>_frag.cu).
#define NINFERNO_MOE_FRAG_FORMATS(X)                                                               \
    X(ggml_q40_frag) X(ggml_q4k_frag) X(ggml_iq3xxs_frag) X(ggml_iq2s_frag)
NINFERNO_MOE_FRAG_FORMATS(NINFERNO_MOE_DECLARE)
#undef NINFERNO_MOE_DECLARE

// Routing kernels (moe_route.cu).
void launch_select(const SelectArgs& args, cudaStream_t stream);

void launch_plan(const PlanArgs& args, cudaStream_t stream);
// Both in one launch, for a call of at most kPlanWarps tokens (the same results).
void launch_select_plan(const SelectArgs& select, const PlanArgs& plan, cudaStream_t stream);

void launch_combine(const CombineArgs& args, cudaStream_t stream);
// Run tickets of the fused combine (GemmArgs::combine_tickets): zero at load, each run's last CTA
// sets its ticket back to zero. A down grid of more runs keeps the separate combine. One such
// down stage runs at a time on a device (a MoE layer's calls are in stream order).
inline constexpr int kCombineTickets = 1024;
[[nodiscard]] int* combine_tickets();
// The BF16 router at 1..kRouterSimtMaxTokens tokens, one warp per expert (GemmArgs of the
// register-MMA router: weights, x, bias, logits and xq). With select and plan (at most 32
// assignments) its last CTA also selects and plans: launch_select_plan's results.
void launch_router_simt(const GemmArgs& args, cudaStream_t stream,
                        const SelectArgs* select = nullptr, const PlanArgs* plan = nullptr);


// Route choices the test compares with their alternatives, process-wide: the MXFP4 banks'
// block-scaled MMA against MXFP8 activations (sm_120a family builds; the int8 W4A8 otherwise) and
// the R8 down stage's fused combine. Both are on by default.
void set_mx8(bool enabled);
void set_fused_combine(bool enabled);

} // namespace ninferno::ops::detail::moe
