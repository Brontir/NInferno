#include "models/qwen3_5/execution/gdn.h"

#include "ninferno/ops/gdn_gating_proj.h"
#include "ninferno/ops/gdn_input_proj.h"
#include "ninferno/ops/segmented_projection.h"

#include <algorithm>
#include <stdexcept>

namespace ninferno::models::qwen3_5::execution {

std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters, std::int32_t first,
                                           std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("GDN projection: invalid column interval");
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& w = single->weight;
        return ops::gdn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, single->policy,
                                                            first, last);
    }
    if (const auto* segmented =
            std::get_if<ops::SegmentedProjectionWeights>(&parameters.projection)) {
        return ops::segmented_projection_workspace_capacity_bytes(*segmented, first, last);
    }
    return ops::gdn_input_proj_pair_workspace_capacity_bytes(first, last);
}

std::size_t gdn_control_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                        std::int32_t hidden_size, std::int32_t first,
                                        std::int32_t last) {
    if (const auto* segmented = std::get_if<ops::SegmentedProjectionWeights>(&parameters.control)) {
        return ops::segmented_gdn_norm_gating_workspace_capacity_bytes(*segmented, first, last);
    }
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(
        static_cast<std::int32_t>(config.linear_num_value_heads), hidden_size, first, last);
}

std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                         std::int32_t batch, std::int32_t first_width,
                                         std::int32_t last_width) {
    if (const auto* segmented =
            std::get_if<ops::SegmentedProjectionWeights>(&parameters.projection)) {
        return std::max(std::size_t{1}, ops::segmented_gdn_conv_workspace_capacity_bytes(
                                            *segmented,
                                            static_cast<std::int32_t>(2 * config.key_width() +
                                                                      config.value_width()),
                                            batch, first_width, last_width));
    }
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    std::size_t bytes;
    if (single && single->weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single->weight;
        bytes         = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single->policy, batch, first_width, last_width);
    } else {
        bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), batch, first_width, last_width);
    }
    // The native overlap contract takes a disjoint span even for a zero-scratch fused route.
    return std::max(std::size_t{1}, bytes);
}

std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                       std::int32_t batch, std::int32_t first_width,
                                       std::int32_t last_width) {
    if (const auto* segmented =
            std::get_if<ops::SegmentedProjectionWeights>(&parameters.projection)) {
        // The record plane is caller storage; only the projection's own scratch is private.
        return std::max(std::size_t{1}, ops::segmented_projection_workspace_capacity_bytes(
                                            *segmented, first_width * batch, last_width * batch));
    }
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    std::size_t bytes;
    if (single && single->weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single->weight;
        bytes         = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single->policy, batch, first_width, last_width);
    } else {
        bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), batch, first_width, last_width);
    }
    return std::max(std::size_t{1}, bytes);
}

void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* segmented =
            std::get_if<ops::SegmentedProjectionWeights>(&parameters.projection)) {
        const ops::ProjectionOutput outputs[] = {{&qkv, 0, qkv.ne[0]}, {&z, 0, z.ne[0]}};
        ops::segmented_projection(hidden, *segmented, outputs, workspace, stream);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj(hidden, pair->first, pair->second, qkv, z, workspace, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj(hidden, single.weight, qkv, z, single.policy, workspace, stream);
    }
}

void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      WorkspaceArena& workspace, DeviceExecutionView execution) {
    if (const auto* segmented = std::get_if<ops::SegmentedProjectionWeights>(&parameters.control)) {
        ops::segmented_gdn_norm_gating_proj(residual, norm, epsilon, *segmented, parameters.a_log,
                                            parameters.dt_bias, workspace, hidden, g, beta,
                                            execution.stream);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.control)) {
        ops::gdn_norm_gating_proj(residual, norm, epsilon, pair->first, pair->second,
                                  parameters.a_log, parameters.dt_bias, workspace, hidden, g, beta,
                                  execution);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.control);
        ops::gdn_norm_gating_proj(residual, norm, epsilon, single.weight, parameters.a_log,
                                  parameters.dt_bias, workspace, hidden, g, beta, execution);
    }
}

void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             const GdnConfig& config, Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slots,
                             const Tensor& destination_slots, Tensor& query, Tensor& key,
                             Tensor& value, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    auto scope = workspace.scope();
    WorkspaceArena scratch(workspace.alloc_bytes(gdn_snapshot_workspace_bytes(
        parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (const auto* segmented =
            std::get_if<ops::SegmentedProjectionWeights>(&parameters.projection)) {
        ops::segmented_gdn_input_proj_conv_snapshot(
            hidden, *segmented, parameters.convolution, conv_states, valid_columns, initial_slots,
            destination_slots, query, key, value, z, scratch, stream);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, pair->first, pair->second, parameters.convolution,
                                          conv_states, valid_columns, initial_slots,
                                          destination_slots, query, key, value, z, scratch, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_snapshot(
            hidden, single.weight, parameters.convolution, conv_states, valid_columns,
            initial_slots, destination_slots, query, key, value, z, single.policy, scratch, stream);
    }
}

void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const GdnConfig& config, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    WorkspaceArena scratch(workspace.alloc_bytes(
        gdn_record_workspace_bytes(parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (const auto* segmented =
            std::get_if<ops::SegmentedProjectionWeights>(&parameters.projection)) {
        ops::segmented_gdn_input_proj_conv_record(hidden, *segmented, parameters.convolution,
                                                  conv_states, valid_columns, initial_slots,
                                                  conv_record, query, key, value, z, scratch,
                                                  stream);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_record(hidden, pair->first, pair->second, parameters.convolution,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, z, scratch, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_record(hidden, single.weight, parameters.convolution, conv_states,
                                        valid_columns, initial_slots, conv_record, query, key,
                                        value, z, single.policy, scratch, stream);
    }
}

} // namespace ninferno::models::qwen3_5::execution
