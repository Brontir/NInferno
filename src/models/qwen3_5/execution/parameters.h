#pragma once

#include "models/qwen3_5/model.h"
#include "ninferno/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninferno::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

struct DenseParameters {
    LinearParameters gate_up; // one [2I, H] parent; unused when split_gate_up is set
    LinearParameters down;
    // Gate and up in separate parents (formats differ): [gate; up] through the generic segmented
    // projection, then silu_mul.
    std::optional<ops::SegmentedProjectionWeights> split_gate_up;
};

// A MoE layer SparseMoe does not admit (a GGUF's K-/i-quant expert banks, another geometry): the
// generic moe Op.
struct GenericMoeParameters {
    ops::MoeWeights weights;
    ops::MoeConfig config;
};

using FfnParameters = std::variant<DenseParameters, ops::SparseMoeWeights, GenericMoeParameters>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters> mixer;
    FfnParameters ffn;
    ops::SparseMoeHints projection_prefetch;
};

struct TextParameters {
    Weight token_embedding;
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
};

struct MtpProjectionParameters {
    // Q, K, gate, V as one parent; absent when a published checkpoint stores them in separate
    // parents (a GGUF nextn block: Q6_K attn_q, Q8_0 attn_k/attn_v), which the dense rows serve.
    std::optional<LinearParameters> packed;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses its existing complete-parent Attention projection.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

// Whether a drafter matrix is the profile its fused kernels read (attn_input_proj's Q/K/V,
// context_kv_materialize): one RowSplit Q8_G32_FP16 parent. Anything else (BF16, Q4, NVFP4, a
// published GGUF's blocks) runs the generic projections (draft.cpp; startup sizes them).
[[nodiscard]] inline bool fused_q8_weight(const Weight& weight) {
    return weight.qtype == QType::Q8_G32_FP16 && weight.layout == QuantLayout::RowSplit;
}

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    // Query/key/value as slices of their parents: the generic projection of a drafter whose
    // matrices are not the fused Q8 profile (a published GGUF's Q4_0).
    ops::SegmentedProjectionWeights query_key_value_bank;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Weight predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head;
    std::optional<Tensor> token_ids;
    std::uint32_t rows = 0;
};

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
class Parameters {
public:
    explicit Parameters(const Model& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
};

} // namespace ninferno::models::qwen3_5::execution
