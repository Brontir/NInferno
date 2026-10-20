#include "ninferno/ops/gemma4_attention.h"

#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/softmax_attention/gemma4/kv_address.h"
#include "ops/softmax_attention/gemma4/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

struct Profile {
    std::int32_t head_dim;
    std::int32_t q_heads;
    std::int32_t kv_heads;
    std::int32_t window;
};

// Any Gemma-style layer: head width 64, 128, 256 or 512 (the kernels' 128-column warp slices),
// any head counts whose GQA group is 1..16, a sliding window of any length or none (full layer). The shipped 26B-A4B / 31B layers and their DFlash drafters are points of this set.
bool head_width(std::int32_t head_dim) {
    return head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512;
}

bool is_geometry(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads) {
    if (!head_width(head_dim) || q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0) {
        return false;
    }
    const std::int32_t group = q_heads / kv_heads;
    return group >= 1 && group <= detail::kGemma4MaxGroup;
}

std::optional<Profile> find_profile(std::int32_t head_dim, std::int32_t q_heads,
                                    std::int32_t kv_heads, std::int32_t window) {
    if (!is_geometry(head_dim, q_heads, kv_heads) || window < 0) { return std::nullopt; }
    return Profile{head_dim, q_heads, kv_heads, window};
}

[[noreturn]] void fail(const char* op, const std::string& message) {
    throw std::invalid_argument(std::string(op) + ": " + message);
}

// positions [T] for one batch row, [T, B] for a batched call.
void require_positions(const Tensor& positions, std::int32_t tokens, std::int32_t batch,
                       const char* op) {
    if (positions.dtype != DType::I32 || positions.ne[0] != tokens || positions.ne[1] != batch ||
        positions.ne[2] != 1 || positions.ne[3] != 1) {
        fail(op, "positions must be I32 [T] or [T, B]");
    }
    if (!positions.is_contiguous() || positions.data == nullptr) {
        fail(op, "positions must be contiguous and non-null");
    }
}

// A [n0, n1, n2] BF16 tensor, or [n0, n1, n2, B] for a batched call.
void require_bf16_batch(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                        std::int32_t batch, const char* op, const char* name) {
    if (tensor.dtype != DType::BF16) { fail(op, std::string(name) + " must be BF16"); }
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != batch) {
        fail(op, std::string("invalid shape for ") + name);
    }
    if (!tensor.is_contiguous()) { fail(op, std::string(name) + " must be contiguous"); }
    if (tensor.data == nullptr) { fail(op, std::string(name) + " data must be non-null"); }
}

[[noreturn]] void fail_window(const char* op, std::int32_t window, std::int32_t head_dim) {
    fail(op, "window " + std::to_string(window) + " is not registered for head_dim " +
                 std::to_string(head_dim));
}

void require_envelope(Gemma4AttentionEnvelope envelope, const char* op) {
    if (envelope.max_tokens < 1 || envelope.max_visible_keys == 0 ||
        envelope.max_visible_keys >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(envelope.max_tokens)) {
        fail(op, "invalid execution envelope");
    }
}

struct PartialWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

// Partials of `batch_rows` rows: each row's block is the single-row call's [D, rows, Hkv, splits].
template <class Allocator>
PartialWorkspace allocate_partials(Allocator& workspace, const Profile& profile,
                                   const detail::Gemma4DecodePlan& plan,
                                   std::int32_t batch_rows = 1) {
    const std::int32_t splits = plan.splits * batch_rows;
    return {
        workspace.alloc(DType::FP32, {profile.head_dim, plan.rows, profile.kv_heads, splits}),
        workspace.alloc(DType::FP32, {plan.rows, profile.kv_heads, splits}),
        workspace.alloc(DType::FP32, {plan.rows, profile.kv_heads, splits}),
    };
}

// A chained prefill call's workspace (with its alignment), 0 where calls run plain.
std::size_t chain_bound(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                        Gemma4AttentionEnvelope envelope, std::int32_t window) {
    const std::size_t bytes = detail::gemma4_attention_chain_bound(head_dim, q_heads, kv_heads,
                                                                   window, envelope.max_tokens);
    return bytes == 0 ? 0 : bytes + 256;
}

// Batch rows whose partials fit in the arena's free bytes, 1..batch.
std::int32_t rows_per_launch(const WorkspaceArena& workspace, const Profile& profile,
                             const detail::Gemma4DecodePlan& plan, std::int32_t batch) {
    if (batch <= 1) return 1;
    const std::size_t per_row = static_cast<std::size_t>(plan.rows) * profile.kv_heads *
                                plan.splits * (profile.head_dim + 2) * sizeof(float);
    const std::size_t slack = 3 * 256; // three allocations' alignment
    const std::size_t used  = workspace.used();
    const std::size_t cap   = workspace.capacity();
    if (used + slack >= cap) return 1;
    const std::size_t fit = (cap - used - slack) / per_row;
    return static_cast<std::int32_t>(std::clamp<std::size_t>(fit, 1, batch));
}

} // namespace

namespace detail {

Gemma4KvBinding gemma4_kv_binding(const Gemma4KvLayer& cache, std::int32_t head_dim,
                                  std::int32_t kv_heads, const char* op) {
    const Tensor& k = cache.k;
    const Tensor& v = cache.v;
    const bool fp8  = k.dtype == DType::FP8_E4M3FN;
    for (const Tensor* plane : {&k, &v}) {
        if (plane->dtype != (fp8 ? DType::FP8_E4M3FN : DType::BF16) || !plane->is_contiguous() ||
            plane->data == nullptr) {
            fail(op, "cache planes must be contiguous non-null BF16, or both FP8_E4M3FN");
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (k.ne[i] != v.ne[i]) { fail(op, "cache k and v planes differ in shape"); }
    }
    if (k.ne[0] != head_dim) { fail(op, "cache planes do not match head_dim"); }
    if (fp8) {
        for (const Tensor* scale : {&cache.k_scale, &cache.v_scale}) {
            if (scale->dtype != DType::FP32 || !scale->is_contiguous() || scale->data == nullptr ||
                scale->ne[0] != k.ne[1] || scale->ne[1] != k.ne[2] || scale->ne[2] != k.ne[3] ||
                scale->ne[3] != 1) {
                fail(op, "FP8 cache planes need contiguous FP32 row scales shaped like the planes "
                         "without D");
            }
        }
    } else if (cache.k_scale.data != nullptr || cache.v_scale.data != nullptr) {
        fail(op, "BF16 cache planes take no scales");
    }
    Gemma4KvBinding binding;
    binding.fp8 = fp8;
    if (cache.rows.data != nullptr) {
        const Tensor& rows = cache.rows;
        if (rows.dtype != DType::I32 || rows.ne[0] < 1 || rows.ne[1] != 1 || rows.ne[2] != 1 ||
            rows.ne[3] != 1 || !rows.is_contiguous()) {
            fail(op, "cache rows must be contiguous I32 [B]");
        }
        binding.batch            = rows.ne[0];
        binding.address.bindings = static_cast<const std::int32_t*>(rows.data);
    }
    if (cache.stored.data != nullptr) {
        const Tensor& stored = cache.stored;
        if (stored.dtype != DType::I32 || stored.ne[0] != binding.batch || stored.ne[1] != 1 ||
            stored.ne[2] != 1 || stored.ne[3] != 1 || !stored.is_contiguous()) {
            fail(op, "cache stored must be contiguous I32 [B], one count per batch row");
        }
        binding.address.stored = static_cast<const std::int32_t*>(stored.data);
    }
    if (cache.block_tables.data != nullptr) {
        const Tensor& tables = cache.block_tables;
        if (tables.dtype != DType::I32 || tables.ne[0] < 1 || tables.ne[1] < 1 ||
            tables.ne[2] != 1 || tables.ne[3] != 1 || !tables.is_contiguous()) {
            fail(op, "block_tables must be contiguous I32 [pages_per_row, table rows]");
        }
        if (k.ne[1] != kPagedKVPageSize || k.ne[2] < 1 || k.ne[3] != kv_heads) {
            fail(op, "a paged layer's planes must be a head-major pool [D, 64, pages, Hkv]");
        }
        const std::int64_t rows    = static_cast<std::int64_t>(k.ne[1]) * k.ne[2];
        const std::int64_t visible = static_cast<std::int64_t>(tables.ne[0]) * kPagedKVPageSize;
        if (rows > std::numeric_limits<std::int32_t>::max() ||
            visible > std::numeric_limits<std::int32_t>::max()) {
            fail(op, "paged layer exceeds int32 rows");
        }
        binding.paged                 = true;
        binding.address.capacity      = static_cast<std::int32_t>(rows);
        binding.address.tables        = static_cast<const std::int32_t*>(tables.data);
        binding.address.pages_per_row = tables.ne[0];
        binding.visible_rows          = static_cast<std::int32_t>(visible);
    } else {
        if (k.ne[1] < 1 || k.ne[2] != kv_heads || k.ne[3] < 1) {
            fail(op, "cache planes must be [D, capacity, Hkv] or [D, capacity, Hkv, images]");
        }
        if (cache.rows.data == nullptr && k.ne[3] != 1) {
            fail(op, "image storage needs rows to bind its batch rows");
        }
        binding.address.capacity       = k.ne[1];
        binding.address.image_elements = static_cast<std::int64_t>(k.ne[0]) * k.ne[1] * k.ne[2];
        binding.visible_rows           = k.ne[1];
    }
    return binding;
}

} // namespace detail

void gemma4_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                      const Gemma4KvLayer& cache, cudaStream_t stream) {
    constexpr const char* op = "gemma4_kv_append";
    if (!head_width(k.ne[0]) || k.ne[1] < 1) { fail(op, "unsupported KV geometry"); }
    if (cache.window < 0) { fail_window(op, cache.window, k.ne[0]); }
    const std::int32_t tokens = k.ne[2];
    if (tokens < 1) { fail(op, "T must be positive"); }
    const detail::Gemma4KvBinding binding = detail::gemma4_kv_binding(cache, k.ne[0], k.ne[1], op);
    require_bf16_batch(k, k.ne[0], k.ne[1], tokens, binding.batch, op, "k");
    require_bf16_batch(v, k.ne[0], k.ne[1], tokens, binding.batch, op, "v");
    require_positions(positions, tokens, binding.batch, op);
    if (tokens > binding.address.capacity) { fail(op, "T exceeds the cache capacity"); }
    detail::gemma4_kv_append_launch(k, v, positions, cache, binding, stream);
}

std::size_t gemma4_attention_workspace_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                             std::int32_t kv_heads,
                                             Gemma4AttentionEnvelope envelope,
                                             std::int32_t window) {
    constexpr const char* op = "gemma4_attention workspace";
    if (!is_geometry(head_dim, q_heads, kv_heads)) { fail(op, "unsupported geometry"); }
    if (window < 0) { fail_window(op, window, head_dim); }
    require_envelope(envelope, op);
    // The split count depends on T and the window, so the widest decode call need not be the
    // largest. The mask does not enter the plan.
    std::size_t capacity    = 0;
    const std::int32_t last = std::min(envelope.max_tokens, detail::kGemma4DecodeMaxTokens);
    const Profile profile{head_dim, q_heads, kv_heads, window};
    for (std::int32_t tokens = 1; tokens <= last; ++tokens) {
        WorkspaceLayoutBuilder layout;
        (void)allocate_partials(
            layout, profile,
            detail::gemma4_attention_decode_plan(head_dim, q_heads, kv_heads, window, tokens));
        capacity = std::max(capacity, layout.peak_bytes(1));
    }
    return std::max(capacity, chain_bound(head_dim, q_heads, kv_heads, envelope, window));
}

std::size_t gemma4_attention_workspace_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                             std::int32_t kv_heads,
                                             Gemma4AttentionEnvelope envelope) {
    constexpr const char* op = "gemma4_attention workspace";
    if (!is_geometry(head_dim, q_heads, kv_heads)) { fail(op, "unsupported geometry"); }
    require_envelope(envelope, op);
    // A window only narrows the split count (decode plan), so the full layer's plan bounds
    // every window.
    std::size_t capacity    = 0;
    const std::int32_t last = std::min(envelope.max_tokens, detail::kGemma4DecodeMaxTokens);
    const Profile profile{head_dim, q_heads, kv_heads, 0};
    for (std::int32_t tokens = 1; tokens <= last; ++tokens) {
        const auto plan =
            detail::gemma4_attention_decode_plan(head_dim, q_heads, kv_heads, 0, tokens);
        WorkspaceLayoutBuilder layout;
        (void)allocate_partials(layout, profile, plan);
        capacity = std::max(capacity, layout.peak_bytes(1));
    }
    return std::max(capacity, chain_bound(head_dim, q_heads, kv_heads, envelope, 0));
}

std::size_t gemma4_attention_batch_workspace_bytes(const WorkspaceArena& arena, std::int32_t batch,
                                                   std::int32_t head_dim, std::int32_t q_heads,
                                                   std::int32_t kv_heads,
                                                   Gemma4AttentionEnvelope envelope,
                                                   std::int32_t window) {
    const std::size_t one =
        gemma4_attention_workspace_bytes(head_dim, q_heads, kv_heads, envelope, window);
    if (batch <= 1) return one;
    // Room for the carve's own alignment.
    const std::size_t used = arena.used() + 4096;
    const std::size_t room = arena.capacity() > used ? arena.capacity() - used : 0;
    return std::max(one, std::min(one * static_cast<std::size_t>(batch), room));
}

void gemma4_attention(const Tensor& q, const Tensor& positions, const Gemma4KvLayer& cache,
                      float scale, Gemma4AttentionEnvelope envelope, WorkspaceArena& workspace,
                      Tensor& out, cudaStream_t stream, Gemma4AttentionMask mask,
                      const Tensor* sinks) {
    constexpr const char* op = "gemma4_attention";
    // A page pool keeps its pages on the third axis and its heads on the fourth.
    const std::int32_t kv_heads =
        cache.block_tables.data != nullptr ? cache.k.ne[3] : cache.k.ne[2];
    if (!is_geometry(q.ne[0], q.ne[1], kv_heads)) {
        fail(op, "unsupported geometry [D=" + std::to_string(q.ne[0]) +
                     ", Hq=" + std::to_string(q.ne[1]) + ", Hkv=" + std::to_string(kv_heads) + "]");
    }
    const auto found = find_profile(q.ne[0], q.ne[1], kv_heads, cache.window);
    if (!found) { fail_window(op, cache.window, q.ne[0]); }
    const Profile* profile = &*found;
    if (mask != Gemma4AttentionMask::Causal) {
        if (mask != Gemma4AttentionMask::Block) { fail(op, "unknown mask"); }
        // D128: the DFlash drafters' blocks; D256: a Gemma 4 target's sliding layers, where an
        // image's soft tokens see one another (the full D512 layers stay causal).
        if (profile->head_dim != 128 && profile->head_dim != 256) {
            fail(op, "the Block mask is admitted only on head_dim 128 and 256 layers");
        }
    }
    const std::int32_t tokens = q.ne[2];
    require_envelope(envelope, op);
    if (tokens < 1 || tokens > envelope.max_tokens) { fail(op, "T outside [1, max_tokens]"); }
    const detail::Gemma4KvBinding binding =
        detail::gemma4_kv_binding(cache, profile->head_dim, profile->kv_heads, op);
    require_bf16_batch(q, profile->head_dim, profile->q_heads, tokens, binding.batch, op, "q");
    require_bf16_batch(out, profile->head_dim, profile->q_heads, tokens, binding.batch, op, "out");
    require_positions(positions, tokens, binding.batch, op);
    if (profile->window > 0 && !binding.paged) {
        if (static_cast<std::int64_t>(binding.visible_rows) <
            static_cast<std::int64_t>(profile->window) + envelope.max_tokens - 1) {
            fail(op, "sliding ring capacity is below window + max_tokens - 1");
        }
    } else if (static_cast<std::uint32_t>(binding.visible_rows) < envelope.max_visible_keys) {
        fail(op, "full-layer capacity is below max_visible_keys");
    }
    if (!std::isfinite(scale) || !(scale > 0.0f)) { fail(op, "scale must be finite and positive"); }

    if (sinks != nullptr &&
        (sinks->dtype != DType::FP32 || !sinks->is_contiguous() || sinks->data == nullptr ||
         sinks->numel() != profile->q_heads)) {
        fail(op, "sinks must be contiguous FP32 [Hq]");
    }
    const detail::Gemma4AttentionLaunch launch{q,
                                               positions,
                                               cache.k,
                                               cache.v,
                                               binding.address,
                                               cache.window,
                                               scale,
                                               mask == Gemma4AttentionMask::Block,
                                               out,
                                               sinks,
                                               binding.fp8 ? &cache.k_scale : nullptr,
                                               binding.fp8 ? &cache.v_scale : nullptr};
    // The invariant route profile (prefill_routes.h): every call takes the prefill kernel, whose
    // row blocks and key tiles sit on the absolute-position grid, so a prompt cut anywhere, a
    // verify call and the decode steps of its columns give each token the same bytes. The decode
    // kernel cuts the union key range of its call (from its first row's window start to its last
    // position) into splits and folds tiles from each row block's first window start: a verify
    // column differed from a decode step once the keys passed one split, or on a sliding layer
    // (gpt-oss, 64q/8kv W128, T=8: 2478 of 32768 values).
    if (tokens > detail::kGemma4DecodeMaxTokens || invariant_routes_active()) {
        // A chained call (full causal layers) when its workspace fits what the arena has free:
        // the same bytes either way, so the arena only decides the speed.
        auto scope              = workspace.scope();
        void* chain             = nullptr;
        const std::size_t bytes = detail::gemma4_attention_chain_bytes(
            profile->head_dim, profile->q_heads, profile->kv_heads, profile->window,
            mask == Gemma4AttentionMask::Block, binding.fp8, tokens);
        if (bytes > 0 && workspace.used() + bytes + 256 <= workspace.capacity()) {
            chain = workspace.alloc_bytes(bytes).data;
        }
        detail::gemma4_attention_prefill_launch(launch, binding.batch, chain, stream);
        return;
    }
    auto scope = workspace.scope();
    const auto plan = detail::gemma4_attention_decode_plan(profile->head_dim, profile->q_heads,
                                                           profile->kv_heads, profile->window, tokens);
    // As many batch rows per launch as the arena's free bytes hold partials for (the workspace
    // bound sizes one row; the arena is the model's widest op, usually far more). Each row is
    // the single-row call whatever the count, so the count changes launches, not bytes.
    const std::int32_t rows  = rows_per_launch(workspace, *profile, plan, binding.batch);
    PartialWorkspace partial = allocate_partials(workspace, *profile, plan, rows);
    detail::gemma4_attention_decode_launch(launch, plan, binding.batch, rows, partial.acc,
                                           partial.m, partial.l, stream);
}

} // namespace ninferno::ops
