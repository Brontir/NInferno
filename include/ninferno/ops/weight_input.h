#pragma once

#include "core/weight_view.h"
#include "ninferno/ops/gemma4_moe.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/sparse_moe.h"

#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace ninferno::ops {

// One mathematical use of a logical matrix. The model owns the view and its backing.
struct WeightInput {
    const WeightView& weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
};

struct SingleProjectionWeight {
    Weight weight;
    LinearPolicy policy = LinearPolicy::A16Only;
};

// The workspace of linear() / linear_add() over this projection at its own policy.
[[nodiscard]] inline std::size_t linear_workspace_capacity_bytes(const SingleProjectionWeight& p,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens) {
    return linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                           min_tokens, max_tokens);
}

[[nodiscard]] inline std::size_t
linear_add_workspace_capacity_bytes(const SingleProjectionWeight& p, std::int32_t min_tokens,
                                    std::int32_t max_tokens) {
    return linear_add_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                               min_tokens, max_tokens);
}

struct PairedProjectionWeights {
    Weight first, second;
};

// One logical input of a projection as rows [row_begin, row_begin + rows) of a complete native
// parent. FP8 and NVFP4 consumers address planes with the whole parent's geometry, so the parent is
// kept whole and the rows are an offset into it; groupwise parents are handled the same way.
struct ProjectionSlice {
    Weight parent;
    std::int32_t row_begin = 0;
    std::int32_t rows      = 0;
    LinearPolicy policy    = LinearPolicy::A16Only;
};

// A projection whose geometry or format combination no fused registration owns: its inputs in
// logical row order (for example query, key, gate, value). The generic path computes it
// (ninferno/ops/segmented_projection.h), merging consecutive rows of one parent into one call.
struct SegmentedProjectionWeights {
    std::vector<ProjectionSlice> slices;

    [[nodiscard]] std::int32_t rows() const;
    [[nodiscard]] std::int32_t columns() const;
};

using ProjectionWeights =
    std::variant<SingleProjectionWeight, PairedProjectionWeights, SegmentedProjectionWeights>;

// Prepare the existing native forms; no device allocation, upload, execution or graph rewrite.
// Runtime shape/phase choices and scratch remain with the actual calling Op.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(const WeightInput& input);
// Row order is supplied by the calling implementation, for example a Vision Q/K/V bank.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows);
[[nodiscard]] SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                                     const WeightInput& key,
                                                                     const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query,
                                                                const WeightInput& key,
                                                                const WeightInput& gate,
                                                                const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query,
                                                               const WeightInput& key,
                                                               const WeightInput& value,
                                                               const WeightInput& z);
[[nodiscard]] ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a,
                                                                const WeightInput& b);
// Whether the inputs, in this order, are consecutive rows of one parent (what the packed forms
// prepared from them require).
[[nodiscard]] bool consecutive_rows_of_one_parent(std::span<const WeightInput> inputs);
// The generic form of any projection bank: each input becomes one slice of its complete parent.
[[nodiscard]] SegmentedProjectionWeights
prepare_segmented_projection_weights(std::span<const WeightInput> inputs);
[[nodiscard]] SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                                  const WeightInput& up);
// A dense FFN's gate and up: one [2I, H] parent (the fused SwiGLU forms), or, when they do not
// share one (a GGUF publishes them in different formats, e.g. IQ2_XS gate and IQ2_S up), a
// two-slice segmented bank the generic path computes as published.
[[nodiscard]] ProjectionWeights prepare_dense_gate_up_weights(const WeightInput& gate,
                                                              const WeightInput& up);
[[nodiscard]] SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down);
// A Qwen3.5-MoE layer (softmax router, shared expert behind a sigmoid gate) in the generic moe
// Op's form: every expert's gate and up rows, and every expert's down rows, must be consecutive
// rows of one parent each (as a converter packs a bank); the shared gate and up one parent. Any
// format the generic projection path computes (GGUF K-quant and i-quant banks included). The
// expert policy is AllowA8 where every expert input allows A8, A16Only otherwise.
[[nodiscard]] MoeWeights prepare_moe_weights(const WeightInput& router,
                                             const WeightInput& shared_score,
                                             std::span<const WeightInput> expert_gate_up,
                                             std::span<const WeightInput> expert_down,
                                             const WeightInput& shared_gate,
                                             const WeightInput& shared_up,
                                             const WeightInput& shared_down);
// Whether prepare_sparse_moe_weights admits the same layer: SparseMoe's registered geometry
// (256 experts of 512, hidden 2048, shared 512) and one of its codec profiles with BF16 routers.
[[nodiscard]] bool sparse_moe_admits(const MoeWeights& weights) noexcept;

// Gemma 4 26B-A4B routed experts. `expert_gate_up` holds one [1408, 2816] input per expert and
// `expert_down` one [2816, 704]; each list must be the consecutive row ranges, in expert order, of
// one complete stacked NVFP4 parent carrying one divisor per expert, so that the returned planes
// are those parents themselves. The router is one BF16 [128, 2816] input. The four Tensors are
// caller-owned device storage and are passed through; the activation divisors they must hold are
// the ones gemma4_moe_input_divisors reads off the same inputs. Every expert input must allow A4,
// because the prefill route encodes both expert inputs in four bits.
[[nodiscard]] Gemma4MoeWeights
prepare_gemma4_moe_weights(const WeightInput& router, const Tensor& router_scale,
                           const Tensor& expert_scale, std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down,
                           const Tensor& gate_up_input_divisors, const Tensor& down_input_divisors);

// Every expert's activation divisor, in expert order, for the caller to upload into the two
// Tensors prepare_gemma4_moe_weights takes. Unlike a shared-input bank, the experts may differ.
struct Gemma4MoeInputDivisors {
    std::vector<float> gate_up;
    std::vector<float> down;
};

[[nodiscard]] Gemma4MoeInputDivisors
gemma4_moe_input_divisors(std::span<const WeightInput> expert_gate_up,
                          std::span<const WeightInput> expert_down);

} // namespace ninferno::ops
