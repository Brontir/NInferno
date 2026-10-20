#pragma once
#include "ops/softmax_attention/common/head_mapping.cuh"
#include <stdexcept>

namespace ninferno::ops::detail {
inline constexpr int kCausalHeadDim = 256;

template <int HeadDim, int QueryHeads, int KVHeads>
struct CausalGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static_assert(HeadDim > 0 && HeadDim % 64 == 0);
    static constexpr int kHeadDim = HeadDim;
};

using CausalD256H24Kv4 = CausalGeometry<256, 24, 4>;
using CausalD256H16Kv2 = CausalGeometry<256, 16, 2>;
using CausalD256H8Kv2  = CausalGeometry<256, 8, 2>;
using CausalD256H16Kv4 = CausalGeometry<256, 16, 4>;

// The instantiated (query heads, KV heads) pairs. The kernels read everything from G, so a new
// pair is one line here and one in causal_geometry_supported.
[[nodiscard]] constexpr bool causal_geometry_supported(int query_heads, int kv_heads) {
    return (query_heads == 24 && kv_heads == 4) || (query_heads == 16 && kv_heads == 2) ||
           (query_heads == 8 && kv_heads == 2) || (query_heads == 16 && kv_heads == 4);
}

template <class F>
void with_causal_geometry(int query_heads, int kv_heads, F&& f) {
    if (query_heads == 24 && kv_heads == 4)
        f.template operator()<CausalD256H24Kv4>();
    else if (query_heads == 16 && kv_heads == 2)
        f.template operator()<CausalD256H16Kv2>();
    else if (query_heads == 8 && kv_heads == 2)
        f.template operator()<CausalD256H8Kv2>();
    else if (query_heads == 16 && kv_heads == 4)
        f.template operator()<CausalD256H16Kv4>();
    else
        throw std::invalid_argument("causal attention: unsupported head geometry");
}
} // namespace ninferno::ops::detail
