#pragma once

// ninferno::ops - the generic sparse Mixture-of-Experts Op: any expert count, top-k, hidden width,
// expert width, router scoring, activation and weight format, with an optional shared expert.
//
// The shape-owned MoE Ops stay the fast paths for the geometries they register (sparse_moe:
// Qwen3.5/3.6-MoE 256x512 top-8; gemma4_moe: Gemma 4 26B-A4B 128x704 top-8). Every other MoE
// geometry runs here.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ninferno/option_enums.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

namespace ninferno::ops {

/// How router logits become scores, and which scores select experts.
enum class MoeScoring : std::uint8_t {
    Softmax, ///< s = softmax(logits) over all E experts (Qwen-MoE, Mixtral, gpt-oss, Granite-MoE)
    Sigmoid, ///< s = sigmoid(logits) (DeepSeek-V3 / GLM-style routers)
};

/// The expert activation between gate/up and down, with g = gate row, u = up row (bias included).
enum class MoeActivation : std::uint8_t {
    SwiGlu,        ///< silu(g) * u
    GeGluTanh,     ///< gelu_tanh(g) * u
    ClampedSwiGlu, ///< gpt-oss: g' = min(g, limit), u' = clamp(u, -limit, limit),
                   ///< (u' + 1) * g' * sigmoid(alpha * g')
    Relu2,         ///< NemotronH: relu(u)^2 of a non-gated expert (gate_up holds up rows only:
                   ///< [E * F, H]; the shared expert is non-gated too: shared_gate_up [Fs, H])
};

/// Row order of one expert's 2F gate/up rows in the stacked gate_up plane.
enum class MoeGateUpLayout : std::uint8_t {
    Concatenated, ///< rows [0, F) gate, [F, 2F) up (Qwen, Mixtral, Gemma)
    Interleaved,  ///< row 2i gate i, row 2i+1 up i (gpt-oss)
};

enum class MoeEpilogue : std::uint8_t {
    Store,       ///< out = MoE(x)
    AddResidual, ///< out += MoE(x) (out holds the residual on entry)
};

/**
 * The stored form of one MoE layer. E experts over hidden width H, each with an F-wide
 * intermediate, stacked expert-major. Every Weight is a complete matrix in any format the generic
 * projection path computes (BF16, FP8 row, NVFP4 with one divisor per stacked source, MXFP4,
 * Q4/Q5/Q6 G64, Q8 G32).
 *
 * - router: [E, H]. router_bias: optional FP32 or BF16 [E], added to the logits.
 * - selection_bias: optional FP32 [E], added to the scores for selection only (Sigmoid routers'
 *   correction bias); it never enters a weight.
 * - expert_scale: optional FP32 [E], multiplies the route weight of a selected expert.
 * - gate_up: [E * 2F, K], expert e's rows at [e * 2F, (e + 1) * 2F) in `gate_up_layout` order.
 *   K is H, or wider when the stored bank's reduction axis is zero-padded to a block multiple (a
 *   GGUF whose H no 256-value block divides, padded so its K/IQ formats apply): the Op feeds the
 *   bank x widened with zeros, which the padded columns multiply, so the result is unchanged.
 *   gate_up_bias: optional FP32 or BF16 [E * 2F], same row order.
 * - down: [E * H, F], expert e's rows at [e * H, (e + 1) * H). down_bias: optional [E * H].
 * - shared_gate_up [2Fs, H] (concatenated), shared_down [H, Fs]: an optional always-selected
 *   shared expert with the SwiGlu activation and unit route weight; shared_gate: optional [1, H],
 *   whose sigmoid(shared_gate . x) multiplies the shared expert's output. Empty Weights (qdata
 *   null) are absent.
 */
struct MoeWeights {
    Weight router;
    Tensor router_bias;
    Tensor selection_bias;
    Tensor expert_scale;
    Weight gate_up;
    Tensor gate_up_bias;
    Weight down;
    Tensor down_bias;
    Weight shared_gate_up;
    Weight shared_down;
    Weight shared_gate;
    /// The shared expert's projections run on the generic projection path under this policy (the
    /// artifact's, where both of them allow it; moe_policy). Under AllowA8 an NVFP4 projection
    /// takes the generic int8 W4A8 from 17 tokens on every build (the generic path does not tell a
    /// prefill from a wide verify), R8 at decode only without the native NVFP4 kernels. Under
    /// AllowA4 (either projection recorded for A4) they run as under AllowA8, except an NVFP4
    /// projection where the build has the W4A4 kernels (sm_120a), which runs A16 at every width
    /// (moe.cpp shared_projection_policy).
    LinearPolicy shared_policy = LinearPolicy::A16Only;
    /// The routed experts' policy (the artifact's, where both banks allow it; moe_policy). NVFP4
    /// banks under AllowA8 (A8 without A4) take the grouped int8 W4A8 on a prefill call of 17
    /// tokens or more (the prefill route profile) on every build; on builds without the native
    /// NVFP4 kernels also on a wide call of the speed profile (from kMoeW4a8SpeedMinTokens), which
    /// sm_120a keeps on its A16 plan. AllowA4 (either bank recorded for A4) keeps NVFP4 banks on
    /// A16 at every width where the build has W4A4 kernels (sm_120a) and otherwise reads as
    /// AllowA8. Under AllowA8 a codebook bank (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) may compute
    /// a decode call (T <= 4, as the dense R8) with the activation quantized to int8 per 32 values
    /// (the generic path's R8 arithmetic), on the GPU classes whose default route lists it. A
    /// prefill call of at most four tokens (a prompt that short) is a decode width too; a longer
    /// prompt's chunking never leaves such a tail. Every other call (T >= 5, K too wide for a CTA's
    /// quantized activation: above 10922 at T = 3 or 4, a route-table band marked A16), and every
    /// call under A16Only, computes A16.
    LinearPolicy expert_policy = LinearPolicy::A16Only;
};

/// A MoE pair's policy from its two projections' (the routed banks, or the shared expert's gate/up
/// and down): A16Only unless both allow A8; then AllowA4 where either allows A4, else AllowA8.
/// Where the build has W4A4 kernels (sm_120a) an AllowA4 pair's NVFP4 projections stay A16 at every
/// width (no W4A4 on this path, no int8 W4A8); elsewhere AllowA4 reads as AllowA8.
LinearPolicy moe_policy(LinearPolicy gate_up, LinearPolicy down);

/// The shared expert's policy from its projections': moe_policy.
LinearPolicy shared_moe_policy(LinearPolicy gate_up, LinearPolicy down);

/// The narrowest activation width the Op computes this layer's routed experts with, at any T:
/// 8 where a call may take the grouped W4A8 (int8, or MXFP8 for MXFP4 banks on sm_120a) or the
/// R8 decode (expert_policy, the banks' formats and K, as the plans select them), else 16. The
/// Op has no four-bit expert route. What --activations' startup resolution counts.
[[nodiscard]] int moe_expert_activation_bits(const MoeWeights& weights);

/// The policy one shared-expert projection (shared_gate_up or shared_down) runs under on the
/// generic projection path (MoeWeights::shared_policy, AllowA4 read as the Op reads it).
[[nodiscard]] LinearPolicy moe_shared_activation_policy(const MoeWeights& weights,
                                                        const Weight& projection);

/// The routing and activation choices of a layer; everything else comes from the weights.
struct MoeConfig {
    std::int32_t top_k           = 0;
    MoeScoring scoring           = MoeScoring::Softmax;
    bool renormalize             = true; ///< w_k = s_k / sum over the selected scores
    float routed_scale           = 1.0F; ///< multiplies every routed weight
    MoeActivation activation     = MoeActivation::SwiGlu;
    float alpha                  = 1.702F; ///< ClampedSwiGlu only
    float limit                  = 7.0F;   ///< ClampedSwiGlu only
    MoeGateUpLayout gate_up_layout = MoeGateUpLayout::Concatenated;
    /// Programmatic dependent launches through the decode chain where the device and the layer
    /// take them; false keeps a pass's chain on plain launches (a drafter's: Nemotron 3.5 MTP).
    bool pdl = true;
};

/**
 * Transient capacity moe needs for every T in [min_tokens, max_tokens], under either route profile
 * and whatever plan the route table gives. Invalid layers, configs or intervals throw.
 */
[[nodiscard]] std::size_t moe_workspace_bytes(const MoeWeights& weights, const MoeConfig& config,
                                              std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: moe
 *
 * Math / indexing, per column t of x [H, T] (every column independent):
 *
 *   l        = router . x[:,t] + router_bias                       (E logits)
 *   s        = softmax(l)  |  sigmoid(l)                           (per MoeScoring)
 *   top      = the top_k experts by s + selection_bias; at an exact tie the lower id wins
 *   w_e      = (renormalize ? s_e / sum_{top} s : s_e) * routed_scale * expert_scale[e]
 *   g, u     = the gate / up rows of expert e: GateUp_e . x[:,t] + gate_up_bias_e
 *   y_e      = Down_e . act(g, u) + down_bias_e                    (act per MoeActivation)
 *   r        = sum_{e in top} w_e y_e
 *              + [shared] sigmoid(shared_gate . x[:,t]) * SharedDown . swiglu(SharedGateUp . x)
 *   out[:,t] = r   (Store)   |   out[:,t] + r   (AddResidual)
 *
 * with silu(z) = z / (1 + e^-z), gelu_tanh(z) = z/2 (1 + tanh(sqrt(2/pi)(z + 0.044715 z^3))).
 *
 * Supported domain: 1 <= top_k <= min(E, 32), E <= 1024, H and F positive with every weight's
 * format K rule (MXFP4 K % 32, NVFP4 K % 64 and N % 128, ...); x and out contiguous BF16 [H, T],
 * T >= 1, x (and router_input) 16-byte aligned. Anything else throws std::invalid_argument before
 * a launch.
 *
 * Numeric: the oracle evaluates the formula in FP64 from the represented BF16 inputs and the exact
 * stored-weight decode. Logits, scores and route weights are FP32; the gate/up product and each
 * selected expert's output are rounded to BF16 once each (as the published reference kernels
 * do); the sum over experts is FP32 and rounded once into out. A grouped stage that computes A8
 * (expert_policy) reads its activation (x for gate/up, act(g, u) for down) quantized per 32
 * values as the generic path's R8 does: the oracle takes that quantization, and the criterion is
 * the A16 one.
 *
 * Token invariance: under the prefill route profile (ops::ScopedPrefillRoutes) every call of 17
 * tokens or more sums every dot product in one fixed order and the experts in rank order, so a
 * column's bytes do not depend on T or on its position in the call.
 *
 * Effects: out is the only observable mutation. x, out, every weight plane and live workspace are
 * pairwise non-overlapping. Stream-ordered, no host synchronisation, CUDA-graph capturable; the
 * launch geometry depends only on T and the layer, never on routing.
 *
 * router_input, when given, is what the router reads instead of x (contiguous BF16 [H, T], the
 * same T): a family whose router sees a differently normalized stream (Gemma 4) passes it.
 */
void moe(const Tensor& x, const MoeWeights& weights, const MoeConfig& config, MoeEpilogue epilogue,
         Tensor& out, WorkspaceArena& workspace, cudaStream_t stream,
         const Tensor* router_input = nullptr);

/// What tune_moe did for one layer (summary: one line for the startup log).
struct MoeTuneReport {
    bool ran      = false; ///< plans are in place for the layer (timed now or read from the cache)
    bool cached   = false; ///< read from the cache (or tuned earlier in this process)
    int improved  = 0;     ///< widths whose plan the timing changed
    std::string summary;
};

/// The MoE tuner's mode (MaintainerOptions::moe_tune; TuneMode::Off by default: tune_moe times
/// nothing) and where its file lives (MaintainerOptions::tune_cache: <dir>/moe_tune.json; empty:
/// NInferno's cache directory). The engine sets both at construction.
void set_moe_tune(TuneMode mode, const std::filesystem::path& cache_dir);

/**
 * Tunes this layer's decode routes on this GPU: at T = 1..max_tokens (at most 16), every widths a
 * route table does not cover, the plan of the least time on the layer's own weights (each call
 * after an L2 flush, the median of nine), where it is at least 3 % under the default one; moe()
 * then takes it. Prefill routes (T >= 17, token-invariant) are not touched. The choices persist
 * per GPU class and layer (<tune cache>/moe_tune.json, set_moe_tune), so a restart takes the same
 * plans and gives the same bytes without timing. Off (the default) keeps the default plans;
 * Retune times again. Call it at load, outside
 * any stream capture, before the free memory is measured: it allocates its own buffers (x, out,
 * workspace, an L2 flush where it fits) and frees them on return. A layer tuned before (the same
 * key) returns at once. Invalid layers throw as moe() does; a failure while timing leaves the
 * default plans and says so in the summary.
 */
MoeTuneReport tune_moe(const MoeWeights& weights, const MoeConfig& config, std::int32_t max_tokens,
                       cudaStream_t stream);

/// The file tune_moe's plans come from, for the decision fingerprint: "moe_tune.user", the user's
/// own cache, which is read only while --moe-tune times ("<file name> sha256:<hex of its
/// bytes>", "absent" when it does not exist yet). Empty while the tuner is off.
[[nodiscard]] std::map<std::string, std::string> moe_tune_sources();

} // namespace ninferno::ops
