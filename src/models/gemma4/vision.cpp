#include "models/gemma4/vision.h"

#include "core/arena.h"
#include "core/checked.h"
#include "core/layout.h"
#include "ninferno/ops/gemma4_fused.h"
#include "ninferno/ops/gemma4_vision.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/residual_add.h"
#include "ninferno/ops/rmsnorm.h"
#include "ninferno/ops/softmax_attention.h"
#include "ninferno/ops/vision_pos_embed.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninferno::models::gemma4 {
namespace {

constexpr std::size_t kAlignment = 256;

struct Layout {
    TensorRegion patches, positions, table_indices, table_weights;
    TensorRegion x, normed, qkv, q, k, v, attended, projected, gate_up, activation, pooled;
    LayoutRegion patch_scratch, qkv_scratch, output_scratch, up_scratch, down_scratch,
        projection_scratch;
    std::size_t bytes = 0;
};

std::size_t capacity(const Projection& p, std::int32_t first, std::int32_t last) {
    return std::max<std::size_t>(1, ops::linear_workspace_capacity_bytes(
                                        p.weight.qtype, p.weight.n, p.weight.k, p.policy, first,
                                        last));
}

Layout build_layout(const VisionConfig& c, const VisionWeights& w, std::int32_t patches) {
    const std::int32_t pool = c.pooling * c.pooling;
    if (patches <= 0 || patches % pool != 0) {
        throw std::invalid_argument("gemma4 vision: the patch count must be whole pooling cells");
    }
    const std::int32_t tokens = patches / pool;
    const std::int32_t h = c.hidden_size, ff = c.intermediate_size;
    std::size_t qkv_bytes = 0, out_bytes = 0, up_bytes = 0, down_bytes = 0;
    for (const auto& layer : w.layers) {
        if (layer.qkv) {
            qkv_bytes = std::max(qkv_bytes, capacity(*layer.qkv, 1, patches));
        } else {
            for (const Projection* p : {&layer.query, &layer.key, &layer.value}) {
                qkv_bytes = std::max(qkv_bytes, capacity(*p, 1, patches));
            }
        }
        out_bytes  = std::max(out_bytes, capacity(layer.output, 1, patches));
        up_bytes   = std::max(up_bytes, capacity(layer.gate_up, 1, patches));
        down_bytes = std::max(down_bytes, capacity(layer.down, 1, patches));
    }
    LayoutBuilder b;
    const auto add = [&](DType dtype, std::initializer_list<std::int32_t> shape, const char* label) {
        return b.add_tensor(dtype, shape, kAlignment, label);
    };
    Layout out;
    out.patches       = add(DType::BF16, {c.patch_width(), patches}, "vision patches");
    out.positions     = add(DType::I32, {2, patches}, "vision positions");
    out.table_indices = add(DType::I32, {4, patches}, "vision table indices");
    out.table_weights = add(DType::FP32, {4, patches}, "vision table weights");
    out.x             = add(DType::BF16, {h, patches}, "vision residual");
    out.normed        = add(DType::BF16, {h, patches}, "vision normed");
    out.qkv           = add(DType::BF16, {3 * h, patches}, "vision qkv");
    out.q             = add(DType::BF16, {c.head_dim, c.heads, patches}, "vision q");
    out.k             = add(DType::BF16, {c.head_dim, c.heads, patches}, "vision k");
    out.v             = add(DType::BF16, {c.head_dim, c.heads, patches}, "vision v");
    out.attended      = add(DType::BF16, {c.head_dim, c.heads, patches}, "vision attended");
    out.projected     = add(DType::BF16, {h, patches}, "vision projected");
    out.gate_up       = add(DType::BF16, {2 * ff, patches}, "vision gate_up");
    out.activation    = add(DType::BF16, {ff, patches}, "vision activation");
    out.pooled        = add(DType::BF16, {h, tokens}, "vision soft tokens");
    out.patch_scratch = b.add(capacity(w.patch_embedding, 1, patches), kAlignment, "patch scratch");
    out.qkv_scratch   = b.add(qkv_bytes, kAlignment, "qkv scratch");
    out.output_scratch = b.add(out_bytes, kAlignment, "attention output scratch");
    out.up_scratch     = b.add(up_bytes, kAlignment, "gate_up scratch");
    out.down_scratch   = b.add(down_bytes, kAlignment, "down scratch");
    out.projection_scratch =
        b.add(capacity(w.projection, 1, tokens), kAlignment, "projection scratch");
    out.bytes = b.finish(kAlignment, "gemma4 vision workspace");
    return out;
}

} // namespace

VisionEncoder::VisionEncoder(const VisionConfig& config, const VisionWeights& weights)
    : config_(config), weights_(weights) {
    if (weights_.layers.size() != static_cast<std::size_t>(config_.layers)) {
        throw std::invalid_argument("gemma4 vision: weights and config disagree on the layers");
    }
}

std::size_t VisionEncoder::workspace_bytes(std::int32_t max_patches) const {
    return build_layout(config_, weights_, max_patches).bytes;
}

void VisionEncoder::encode(const VisionImage& image, Tensor& output, DeviceSpan workspace,
                           cudaStream_t stream) const {
    const VisionConfig& c   = config_;
    const std::int32_t pool = c.pooling;
    if (image.width <= 0 || image.rows <= 0 || image.width % pool != 0 || image.rows % pool != 0) {
        throw std::invalid_argument("gemma4 vision: the patch grid must be whole pooling cells");
    }
    const std::uint64_t count =
        checked_mul(std::uint64_t(image.width), std::uint64_t(image.rows), "patch count");
    if (count > std::uint64_t(c.table_size) * std::uint64_t(c.table_size) ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("gemma4 vision: the image has too many patches");
    }
    const auto patches = static_cast<std::int32_t>(count);
    if (image.patches.size() != std::size_t(patches) * std::size_t(c.patch_width())) {
        throw std::invalid_argument("gemma4 vision: the patch buffer does not match its grid");
    }
    const std::int32_t tokens = soft_tokens(patches);
    if (output.dtype != DType::BF16 || !output.is_contiguous() ||
        output.ne[0] != c.output_hidden || output.ne[1] != tokens || output.ne[2] != 1) {
        throw std::invalid_argument("gemma4 vision: output must be BF16 [text hidden, tokens]");
    }
    const Layout layout = build_layout(c, weights_, patches);
    if (workspace.data == nullptr || workspace.bytes < layout.bytes) {
        throw std::invalid_argument("gemma4 vision: the workspace is too small for this image");
    }
    const auto project = [&](const Tensor& x, const Projection& p, Tensor& out,
                             const LayoutRegion& region) {
        WorkspaceArena scratch(region.bind(workspace));
        ops::linear(x, p.weight, out, p.policy, scratch, stream);
    };

    // Patch embedding: the patches arrive as 2x - 1, the reference's own input scaling.
    Tensor patch_in = layout.patches.bind(workspace);
    Tensor x        = layout.x.bind(workspace);
    copy_from_host(image.patches.data(), patch_in, stream);
    project(patch_in, weights_.patch_embedding, x, layout.patch_scratch);

    // Positions (x, y) per patch, and the table rows they add: row x of the x table, row y of the
    // y table (the y table follows the x table's `table_size` rows).
    std::vector<std::int32_t> positions(std::size_t(2) * patches);
    std::vector<std::int32_t> indices(std::size_t(4) * patches, 0);
    std::vector<float> weights(std::size_t(4) * patches, 0.0F);
    for (std::int32_t p = 0; p < patches; ++p) {
        const std::int32_t px = p % image.width, py = p / image.width;
        if (px >= c.table_size || py >= c.table_size) {
            throw std::invalid_argument("gemma4 vision: the image is wider than the position table");
        }
        positions[2 * std::size_t(p)]     = px;
        positions[2 * std::size_t(p) + 1] = py;
        indices[std::size_t(p)]           = px;
        indices[std::size_t(patches) + p] = c.table_size + py;
        weights[std::size_t(p)]           = 1.0F;
        weights[std::size_t(patches) + p] = 1.0F;
    }
    Tensor position_ids  = layout.positions.bind(workspace);
    Tensor table_indices = layout.table_indices.bind(workspace);
    Tensor table_weights = layout.table_weights.bind(workspace);
    copy_from_host(positions.data(), position_ids, stream);
    copy_from_host(indices.data(), table_indices, stream);
    copy_from_host(weights.data(), table_weights, stream);
    const Tensor table = weights_.position_table.reshape({c.hidden_size, 2 * c.table_size});
    ops::vision_pos_embed_add(table, table_indices, table_weights, x, stream);
    // The host vectors must outlive the copies.
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("gemma4 vision: position upload failed");
    }

    const std::int32_t h  = c.hidden_size;
    const float eps       = c.rms_norm_eps;
    const float q_scale   = std::sqrt(static_cast<float>(c.head_dim)); // the tower attends at scale 1
    const float att_scale = 1.0F / q_scale;
    Tensor normed     = layout.normed.bind(workspace);
    Tensor qkv        = layout.qkv.bind(workspace);
    Tensor q          = layout.q.bind(workspace);
    Tensor k          = layout.k.bind(workspace);
    Tensor v          = layout.v.bind(workspace);
    Tensor attended   = layout.attended.bind(workspace);
    Tensor projected  = layout.projected.bind(workspace);
    Tensor gate_up    = layout.gate_up.bind(workspace);
    Tensor activation = layout.activation.bind(workspace);
    for (const VisionLayer& layer : weights_.layers) {
        ops::rmsnorm(x, layer.input_norm, eps, false, normed, stream);
        Tensor q_src, k_src, v_src;
        if (layer.qkv) {
            project(normed, *layer.qkv, qkv, layout.qkv_scratch);
            q_src = qkv.slice(0, 0, h);
            k_src = qkv.slice(0, h, h);
            v_src = qkv.slice(0, 2 * h, h);
        } else {
            Tensor planes = qkv.reshape({h, 3 * patches});
            q_src         = planes.slice(1, 0, patches);
            k_src         = planes.slice(1, patches, patches);
            v_src         = planes.slice(1, 2 * patches, patches);
            project(normed, layer.query, q_src, layout.qkv_scratch);
            project(normed, layer.key, k_src, layout.qkv_scratch);
            project(normed, layer.value, v_src, layout.qkv_scratch);
        }
        ops::gemma4_vision_qkv_prepare(q_src, k_src, v_src, c.head_dim, position_ids,
                                       layer.query_norm, layer.key_norm, eps, c.rope_theta,
                                       q_scale, q, k, v, stream);
        ops::packed_softmax_attention(q, k, v, {c.head_dim, c.heads, c.heads}, att_scale, patches,
                                      attended, stream);
        Tensor attended_flat = attended.reshape({h, patches});
        project(attended_flat, layer.output, projected, layout.output_scratch);
        ops::rmsnorm(projected, layer.post_attention_norm, eps, false, normed, stream);
        ops::residual_add(normed, x, stream);

        ops::rmsnorm(x, layer.pre_ffn_norm, eps, false, normed, stream);
        project(normed, layer.gate_up, gate_up, layout.up_scratch);
        ops::gelu_tanh_mul(gate_up, activation, stream);
        project(activation, layer.down, projected, layout.down_scratch);
        ops::rmsnorm(projected, layer.post_ffn_norm, eps, false, normed, stream);
        ops::residual_add(normed, x, stream);
    }

    Tensor pooled = layout.pooled.bind(workspace);
    ops::gemma4_vision_pool(x, image.width, pool, weights_.std_bias, weights_.std_scale,
                            std::sqrt(static_cast<float>(h)), eps, pooled, stream);
    project(pooled, weights_.projection, output, layout.projection_scratch);
}

} // namespace ninferno::models::gemma4
