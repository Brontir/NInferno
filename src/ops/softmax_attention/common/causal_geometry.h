#pragma once
#include "ops/softmax_attention/common/head_mapping.cuh"
#include <cstddef>
#include <iterator>
#include <stdexcept>

namespace ninferno::ops::detail {
inline constexpr int kCausalHeadDim = 256;

template <int HeadDim, int QueryHeads, int KVHeads>
struct CausalGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static_assert(HeadDim > 0 && HeadDim % 64 == 0);
    static constexpr int kHeadDim = HeadDim;
};

// The instantiated geometries, (query heads, KV heads) at kCausalHeadDim, listed once (#312): the
// supported predicate, the dispatch and the pairs' table below all expand from it, so they cannot
// disagree (the list stays defined for them: public on purpose). The kernels read everything from
// G; a new pair is one line here (and its oracle, workspace and launch tests).
#define NINFERNO_CAUSAL_GEOMETRIES(X) \
    X(24, 4)                          \
    X(16, 2)                          \
    X(8, 2)                           \
    X(16, 4)

struct CausalHeadPair {
    int query_heads;
    int kv_heads;
};

inline constexpr CausalHeadPair kCausalHeadPairs[] = {
#define NINFERNO_CAUSAL_PAIR(Q, KV) {Q, KV},
    NINFERNO_CAUSAL_GEOMETRIES(NINFERNO_CAUSAL_PAIR)
#undef NINFERNO_CAUSAL_PAIR
};

[[nodiscard]] constexpr bool causal_geometry_supported(int query_heads, int kv_heads) {
    for (const CausalHeadPair pair : kCausalHeadPairs) {
        if (pair.query_heads == query_heads && pair.kv_heads == kv_heads) return true;
    }
    return false;
}

// Each pair once: the dispatch below takes the first match.
[[nodiscard]] constexpr bool causal_pairs_distinct() {
    for (std::size_t i = 0; i < std::size(kCausalHeadPairs); ++i) {
        for (std::size_t j = i + 1; j < std::size(kCausalHeadPairs); ++j) {
            if (kCausalHeadPairs[i].query_heads == kCausalHeadPairs[j].query_heads &&
                kCausalHeadPairs[i].kv_heads == kCausalHeadPairs[j].kv_heads)
                return false;
        }
    }
    return true;
}
static_assert(causal_pairs_distinct(), "a causal geometry is listed twice");

template <class F>
void with_causal_geometry(int query_heads, int kv_heads, F&& f) {
#define NINFERNO_CAUSAL_DISPATCH(Q, KV)                                     \
    if (query_heads == (Q) && kv_heads == (KV)) {                           \
        f.template operator()<CausalGeometry<kCausalHeadDim, (Q), (KV)>>(); \
        return;                                                             \
    }
    NINFERNO_CAUSAL_GEOMETRIES(NINFERNO_CAUSAL_DISPATCH)
#undef NINFERNO_CAUSAL_DISPATCH
    throw std::invalid_argument("causal attention: unsupported head geometry");
}
} // namespace ninferno::ops::detail
