#include "ops/softmax_attention/dense/context/launch.h"
#include "core/per_device.h"

#include "core/device.h"
#include "ops/softmax_attention/dense/context/kernel.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

// The most KV splits context_attention_resolve_plan chooses (its widest split_limit); a plan
// outside [1, this] was not made by it.
constexpr std::int32_t kContextAttentionMaxSplits = 64;

template <class Geo, int Tokens, class Launch>
void dispatch_token_case(Launch&& launch) {
    launch.template operator()<Tokens, kContextQueryWarps<Geo, Tokens>>();
}

template <class Geo, class Launch>
void dispatch_tokens(std::int32_t tokens, Launch&& launch) {
    switch (tokens) {
#define NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(TOKENS)                                              \
    case TOKENS:                                                                                   \
        dispatch_token_case<Geo, TOKENS>(launch);                                                  \
        return
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(1);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(2);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(3);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(4);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(5);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(6);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(7);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(8);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(9);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(10);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(11);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(12);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(13);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(14);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(15);
        NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE(16);
#undef NINFERNO_CONTEXT_ATTENTION_TOKEN_CASE
    default:
        throw std::invalid_argument("context_softmax_attention: unsupported T");
    }
}

std::int32_t profile_group(ContextAttentionProfile profile) {
    return profile == ContextAttentionProfile::D64Q64K8 ? ContextQueryD64::Group
                                                        : ContextQueryD128::Group;
}

template <class Geo>
void launch_profile(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                    const Tensor& context_lengths, const Tensor& valid_columns,
                    const Tensor& table_rows, float scale, const PagedKVBatchLayerView& context,
                    const ContextAttentionPlan& plan, Tensor& partial_acc, Tensor& partial_m,
                    Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    dispatch_tokens<Geo>(q.ne[2], [&]<int Tokens, int Warps>() {
        const bool direct = plan.route == ContextAttentionRoute::Direct;
        if (plan.warps != Warps || plan.split_capacity < 1 ||
            plan.split_capacity > kContextAttentionMaxSplits ||
            (plan.route != ContextAttentionRoute::Direct &&
             plan.route != ContextAttentionRoute::SplitKv) ||
            (direct && plan.split_capacity != 1)) {
            throw std::invalid_argument("context_softmax_attention: inconsistent plan");
        }
        // Q staging reuses the two K/V tiles: a CTA's Warps x 16 query rows fit 2 x KeyBlock.
        const auto launch = [&]<int KeyBlock>() {
            constexpr std::size_t SmemBytes = 2u * KeyBlock * Geo::D * sizeof(__nv_bfloat16);
            const auto partial = [&]<bool Direct>() {
                const dim3 grid(Geo::Hkv, Direct ? 1 : plan.split_capacity, q.ne[3]);
                if (context.storage == KvCacheStorage::Int8Group64) {
                    // The int8 tiles' raw staging rides after the operand tiles; with 64-key tiles
                    // that passes the 48 KiB default, so the kernel opts in once.
                    constexpr std::size_t Bytes =
                        SmemBytes + kContextQueryInt8StageBytes<Geo::D, KeyBlock>;
                    const auto kernel =
                        &context_attention_split_partial_kernel<Geo, Tokens, Warps, KeyBlock,
                                                                             Direct, std::int8_t, std::int8_t>;
                    if constexpr (Bytes > 48u * 1024u) {
                        static ::ninferno::PerDevice<bool> opted;
                        ::ninferno::allow_dynamic_smem(opted, kernel, static_cast<int>(Bytes));
                    }
                    kernel<<<grid, Warps * 32, Bytes, stream>>>(
                        static_cast<const __nv_bfloat16*>(q.data),
                        static_cast<const __nv_bfloat16*>(query_k.data),
                        static_cast<const __nv_bfloat16*>(query_v.data),
                        static_cast<const std::int32_t*>(context_lengths.data),
                        static_cast<const std::int32_t*>(valid_columns.data),
                        static_cast<const std::int32_t*>(table_rows.data),
                        static_cast<const std::int8_t*>(context.k_pages.data),
                        static_cast<const std::int8_t*>(context.v_pages.data),
                        static_cast<const __half*>(context.k_scale_pages.data),
                        static_cast<const __half*>(context.v_scale_pages.data),
                        static_cast<const std::int32_t*>(context.block_tables.data),
                        context.k_pages.ne[2], context.block_tables.ne[0],
                        context.block_tables.ne[0] * kPagedKVPageSize,
                        Direct ? 1 : plan.split_capacity, scale,
                        static_cast<__nv_bfloat16*>(partial_acc.data),
                        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                        static_cast<__nv_bfloat16*>(out.data));
                    CUDA_CHECK(cudaGetLastError());
                    return;
                }
                context_attention_split_partial_kernel<Geo, Tokens, Warps, KeyBlock, Direct>
                    <<<grid, Warps * 32, SmemBytes, stream>>>(
                        static_cast<const __nv_bfloat16*>(q.data),
                        static_cast<const __nv_bfloat16*>(query_k.data),
                        static_cast<const __nv_bfloat16*>(query_v.data),
                        static_cast<const std::int32_t*>(context_lengths.data),
                        static_cast<const std::int32_t*>(valid_columns.data),
                        static_cast<const std::int32_t*>(table_rows.data),
                        static_cast<const __nv_bfloat16*>(context.k_pages.data),
                        static_cast<const __half*>(context.v_pages.data), nullptr, nullptr,
                        static_cast<const std::int32_t*>(context.block_tables.data),
                        context.k_pages.ne[2], context.block_tables.ne[0],
                        context.block_tables.ne[0] * kPagedKVPageSize,
                        Direct ? 1 : plan.split_capacity, scale,
                        static_cast<__nv_bfloat16*>(partial_acc.data),
                        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                        static_cast<__nv_bfloat16*>(out.data));
                CUDA_CHECK(cudaGetLastError());
            };
            if (direct) {
                partial.template operator()<true>();
                return;
            }
            partial.template operator()<false>();
            const dim3 reduce_grid(Geo::Hq, Tokens, q.ne[3]);
            context_attention_reduce_kernel<Geo, Tokens, KeyBlock><<<reduce_grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(context_lengths.data),
                static_cast<const std::int32_t*>(valid_columns.data),
                context.block_tables.ne[0] * kPagedKVPageSize, plan.split_capacity,
                static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        };
        constexpr int Rows = Warps * 16;
        if (plan.key_block == 32) {
            if constexpr (Rows <= 64) {
                launch.template operator()<32>();
                return;
            }
        } else if (plan.key_block == 64) {
            if constexpr (Tokens > 8) {
                launch.template operator()<64>();
                return;
            }
        }
        throw std::invalid_argument("context_softmax_attention: inconsistent plan");
    });
}

} // namespace

ContextAttentionProfile context_attention_profile(AttentionHeadGeometry geometry) {
    if (geometry.head_dim == ContextQueryD128::D && geometry.query_heads == ContextQueryD128::Hq &&
        geometry.kv_heads == ContextQueryD128::Hkv) {
        return ContextAttentionProfile::D128Q32K8;
    }
    if (geometry.head_dim == ContextQueryD64::D && geometry.query_heads == ContextQueryD64::Hq &&
        geometry.kv_heads == ContextQueryD64::Hkv) {
        return ContextAttentionProfile::D64Q64K8;
    }
    throw std::invalid_argument("context_softmax_attention: unsupported head geometry");
}

ContextAttentionPlan context_attention_resolve_plan(ContextAttentionProfile profile,
                                                    std::int32_t tokens,
                                                    ContextAttentionExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("context_softmax_attention plan: T must be 1..16");
    }
    if (envelope.min_context > envelope.max_context) {
        throw std::invalid_argument("context_softmax_attention plan: invalid envelope");
    }
    const std::int32_t group = profile_group(profile);
    const std::int32_t warps = (tokens * group + 15) / 16;
    const bool direct        = envelope.max_context == 0;
    // A CTA stages its warps x 16 query rows in the two key/value tiles: wider row blocks take
    // 64-key tiles.
    const bool wide_rows = warps * 16 > 64;
    const std::int32_t key_block =
        wide_rows ? 64 : (direct || tokens <= 8 || envelope.max_context <= 65536u ? 32 : 64);
    std::int32_t split_limit = 32;
    if (tokens <= 8) {
        split_limit =
            envelope.max_context <= 131072u ? 32 : (envelope.max_context <= 196608u ? 48 : 64);
    } else if (key_block == 64) {
        split_limit =
            envelope.max_context <= 131072u ? 32 : (envelope.max_context <= 196608u ? 38 : 40);
    }
    const std::uint32_t envelope_tiles =
        (envelope.max_context + static_cast<std::uint32_t>(key_block) - 1u) /
        static_cast<std::uint32_t>(key_block);
    const std::int32_t splits =
        direct ? 1 : std::min(split_limit, std::max(1, static_cast<std::int32_t>(envelope_tiles)));
    if (splits > kContextAttentionMaxSplits) {
        throw std::logic_error("context_softmax_attention plan: split limit above the maximum");
    }
    return {
        .profile        = profile,
        .route          = direct ? ContextAttentionRoute::Direct : ContextAttentionRoute::SplitKv,
        .tokens         = tokens,
        .warps          = warps,
        .key_block      = key_block,
        .split_capacity = splits,
    };
}

void context_attention_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& context_lengths, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale,
                              const PagedKVBatchLayerView& context,
                              const ContextAttentionPlan& plan, Tensor& partial_acc,
                              Tensor& partial_m, Tensor& partial_l, Tensor& out,
                              cudaStream_t stream) {
    switch (plan.profile) {
    case ContextAttentionProfile::D128Q32K8:
        launch_profile<ContextQueryD128>(q, query_k, query_v, context_lengths, valid_columns,
                                         table_rows, scale, context, plan, partial_acc, partial_m,
                                         partial_l, out, stream);
        return;
    case ContextAttentionProfile::D64Q64K8:
        launch_profile<ContextQueryD64>(q, query_k, query_v, context_lengths, valid_columns,
                                        table_rows, scale, context, plan, partial_acc, partial_m,
                                        partial_l, out, stream);
        return;
    }
    throw std::invalid_argument("context_softmax_attention: inconsistent plan");
}

} // namespace ninferno::ops::detail
