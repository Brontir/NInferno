#include "models/qwen3_5/execution/attention.h"

#include "ninferno/ops/attn_input_proj.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/rmsnorm_rope.h"
#include "ninferno/ops/rope.h"
#include "ninferno/ops/segmented_projection.h"

#include <stdexcept>

namespace ninferno::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

// The fused text form is registered for the two text head geometries with a one-dimensional
// position axis. The MRoPE path and any other geometry take the three calls it replaces.
//
// It is also bounded in width. One warp owns one head, so the fused kernel stops gaining once a
// width alone fills the machine, and past that the three separate kernels - each free to choose
// its own shape - are ahead: measured on an RTX 5090, the fused form wins by 22 to 52 % through
// 256 tokens and loses by up to 22 % at 1024. The bound sits a doubling below the crossover
// because the two geometries cross at different widths. The Op itself is valid at any width; this
// is a dispatch choice, and both branches are the same arithmetic bit for bit.
constexpr std::int32_t kFusedTextQkNormRopeMaximumTokens = 256;

// The fused kernel fixes the RMSNorm epsilon (1e-6) and the RoPE frequencies (theta 1e7) in its
// formula; a checkpoint with other constants takes the three-call route at every width.
constexpr float kFusedTextRopeTheta  = 1.0e7F;
constexpr float kFusedTextRmsEpsilon = 1.0e-6F;

bool fused_text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                             const AttentionConfig& attention, float rms_norm_eps,
                             std::int32_t tokens) {
    return !rope.yarn() && positions.ne[1] == 1 && tokens <= kFusedTextQkNormRopeMaximumTokens &&
           rope.rope_theta == kFusedTextRopeTheta && rms_norm_eps == kFusedTextRmsEpsilon &&
           attention.head_dim == 256 && rope.rotary_dim == 64 &&
           ops::rmsnorm_rope_text_heads(attention.num_attention_heads,
                                        attention.num_key_value_heads);
}

} // namespace

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& weight = single->weight;
        return ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                             single->policy, first, last);
    }
    if (const auto* segmented =
            std::get_if<ops::SegmentedProjectionWeights>(&parameters.projection)) {
        return ops::segmented_projection_workspace_capacity_bytes(*segmented, first, last);
    }
    return ops::attn_input_proj_pair_workspace_capacity_bytes(first, last);
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* segmented =
            std::get_if<ops::SegmentedProjectionWeights>(&parameters.projection)) {
        // Logical row order is query, key, gate, value (prepare_attn_input_proj_weights).
        const ops::ProjectionOutput outputs[] = {{&query, 0, query.ne[0]},
                                                 {&key, 0, key.ne[0]},
                                                 {&gate, 0, gate.ne[0]},
                                                 {&value, 0, value.ne[0]}};
        ops::segmented_projection(hidden, *segmented, outputs, workspace, stream);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(hidden, pair->first, pair->second, query, gate, key, value, workspace,
                             stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::attn_input_proj(hidden, single.weight, query, gate, key, value, single.policy,
                             workspace, stream);
    }
}

// The device frequencies of a YaRN rope (RopeConfig::yarn_frequencies) as the Op's FP32 table.
Tensor yarn_frequencies(const RopeConfig& config) {
    if (config.yarn_frequencies == nullptr) {
        throw std::logic_error("Qwen3.5 YaRN rope has no device frequencies");
    }
    return Tensor(const_cast<float*>(config.yarn_frequencies), DType::FP32,
                  {static_cast<std::int32_t>(config.rotary_dim / 2)});
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               DeviceExecutionView execution) {
    require_rope_axes(positions, config);
    if (config.yarn()) {
        ops::rope_scaled(positions, dimension(config.rotary_dim), yarn_frequencies(config),
                         config.yarn_attention_scale, query, execution);
        return;
    }
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, execution);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               DeviceExecutionView execution) {
    require_rope_axes(positions, config);
    if (config.yarn()) {
        ops::rope_scaled(positions, dimension(config.rotary_dim), yarn_frequencies(config),
                         config.yarn_attention_scale, query, key, execution);
        return;
    }
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, key, execution);
}

void text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                       const AttentionConfig& attention, float rms_norm_eps,
                       const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                       const Tensor& query, const Tensor& key, Tensor& normalized_query,
                       Tensor& normalized_key, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    require_rope_axes(positions, rope);
    if (fused_text_qk_norm_rope(positions, rope, attention, rms_norm_eps, query.ne[2])) {
        ops::rmsnorm_rope(positions, q_norm_weight, k_norm_weight, query, key, normalized_query,
                          normalized_key, stream);
        return;
    }
    ops::rmsnorm(query, q_norm_weight, rms_norm_eps, true, normalized_query, stream);
    ops::rmsnorm(key, k_norm_weight, rms_norm_eps, true, normalized_key, stream);
    text_rope(positions, rope, normalized_query, normalized_key, execution);
}

} // namespace ninferno::models::qwen3_5::execution
