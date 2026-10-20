#include "ninferno/ops/softmax_attention.h"

#include "core/layout.h"
#include "ops/softmax_attention/dense/packed/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

// The Vision encoders' heads: 64 values (Qwen3.5 0.8B to 9B) or 72 (27B, Qwen3.6 35B), any head
// count, keys and values per query head, scaled by 1/sqrt(head_dim).
bool supported_head_dim(std::int32_t head_dim) { return head_dim == 64 || head_dim == 72; }

float expected_scale(std::int32_t head_dim) {
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
}

void require_profile(AttentionHeadGeometry geometry, float scale, const char* op) {
    if (!valid_attention_head_geometry(geometry) || !supported_head_dim(geometry.head_dim) ||
        geometry.query_heads != geometry.kv_heads) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
    if (!std::isfinite(scale) || std::abs(scale - expected_scale(geometry.head_dim)) > 1.0e-7f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(head_dim)");
    }
}

std::int32_t scratch_tiles(std::int32_t tokens, std::int32_t segments) {
    if (segments == 1) { return 0; }
    constexpr std::int64_t tile_rows = 64;
    const std::int64_t tiles =
        (static_cast<std::int64_t>(tokens) + tile_rows - 1) / tile_rows + segments - 1LL;
    if (tiles > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("packed_softmax_attention: descriptor tile count exceeds int32");
    }
    return static_cast<std::int32_t>(tiles);
}

template <class Allocator>
Tensor allocate_workspace(Allocator& allocator, std::int32_t tokens, std::int32_t segments) {
    const std::int32_t tiles = scratch_tiles(tokens, segments);
    return tiles == 0 ? Tensor{} : allocator.alloc(DType::I32, {4, tiles});
}

void require_qkv(const Tensor& tensor, AttentionHeadGeometry geometry, std::int32_t tokens,
                 const char* op, const char* name) {
    const std::int64_t head_dim = geometry.head_dim;
    const std::int64_t heads    = geometry.query_heads;
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != head_dim || tensor.ne[1] != heads ||
        tensor.ne[2] != tokens || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * head_dim * heads || (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name + " strides");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

std::int32_t validate_qkv(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
                          AttentionHeadGeometry geometry, float scale, const char* op) {
    require_profile(geometry, scale, op);
    const std::int32_t tokens = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_qkv(q, geometry, tokens, op, "q");
    require_qkv(k, geometry, tokens, op, "k");
    require_qkv(v, geometry, tokens, op, "v");
    require_qkv(out, geometry, tokens, op, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": out must be contiguous");
    }
    return tokens;
}

} // namespace

std::size_t packed_softmax_attention_workspace_capacity_bytes(AttentionHeadGeometry geometry,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens,
                                                              std::int32_t min_segments,
                                                              std::int32_t max_segments) {
    require_profile(geometry, expected_scale(geometry.head_dim),
                    "packed_softmax_attention workspace");
    if (min_tokens <= 0 || max_tokens < min_tokens || min_segments <= 0 ||
        max_segments < min_segments || min_segments > max_tokens) {
        throw std::invalid_argument(
            "packed_softmax_attention workspace: invalid execution envelope");
    }
    const std::int32_t segments = std::min(max_segments, max_tokens);
    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, max_tokens, segments);
    return layout.peak_bytes(1);
}

void softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                       AttentionHeadGeometry geometry, float scale, WorkspaceArena& workspace,
                       Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens = validate_qkv(q, k, v, out, geometry, scale, "softmax_attention");
    auto scope                = workspace.scope();
    detail::packed_attention_uniform_launch(q, k, v, tokens, scale, out, stream);
}

void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale, const Tensor& cu_seqlens,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens =
        validate_qkv(q, k, v, out, geometry, scale, "packed_softmax_attention");
    const std::int32_t segments = cu_seqlens.ne[0] - 1;
    if (cu_seqlens.dtype != DType::I32 || segments <= 0 || segments > tokens ||
        cu_seqlens.ne[1] != 1 || cu_seqlens.ne[2] != 1 || cu_seqlens.ne[3] != 1 ||
        !cu_seqlens.is_contiguous() || cu_seqlens.data == nullptr) {
        throw std::invalid_argument(
            "packed_softmax_attention: cu_seqlens must be contiguous I32 [S+1]");
    }
    auto scratch_scope = workspace.scope();
    Tensor tiles       = allocate_workspace(workspace, tokens, segments);
    Tensor* tiles_ptr  = tiles.data == nullptr ? nullptr : &tiles;
    detail::packed_attention_launch(q, k, v, cu_seqlens, tiles_ptr, scale, out, stream);
}

void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale,
                              std::int32_t segment_length, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens =
        validate_qkv(q, k, v, out, geometry, scale, "packed_softmax_attention");
    if (segment_length <= 0 || tokens % segment_length != 0) {
        throw std::invalid_argument("packed_softmax_attention: invalid uniform segment length");
    }
    detail::packed_attention_uniform_launch(q, k, v, segment_length, scale, out, stream);
}

} // namespace ninferno::ops
