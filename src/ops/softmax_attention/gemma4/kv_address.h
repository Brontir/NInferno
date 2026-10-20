#pragma once

// Where the rows of a Gemma4KvLayer live (include/ninferno/ops/gemma4_attention.h), in the form the
// kernels read: one sequence's slot storage, one ring image per batch row, or a head-major page
// pool addressed through block tables. Every kernel that reads or writes a layer's keys and values
// computes the row address here, so the three layouts cannot drift apart.

#include "ninferno/ops/gemma4_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops::detail {

inline constexpr int kGemma4PageTokensLog2 = 6; // kPagedKVPageSize = 64

struct Gemma4KvAddress {
    std::int32_t capacity        = 0; // rows of one head plane (a page pool: 64 * pages)
    std::int64_t image_elements  = 0; // slot storage: elements of one image, D * capacity * Hkv
    const std::int32_t* tables   = nullptr; // paged: I32 [pages_per_row, table rows]
    std::int32_t pages_per_row   = 0;
    const std::int32_t* bindings = nullptr; // per batch row: its table row or image; null: 0
    const std::int32_t* stored   = nullptr; // per batch row: tokens that write; null: all
};

#ifdef __CUDACC__
// The table row or image of batch row b.
__device__ __forceinline__ int gemma4_kv_binding(const Gemma4KvAddress& a, int b) {
    return a.bindings != nullptr ? a.bindings[b] : 0;
}

// Whether token j of batch row b writes its keys and values (Gemma4KvLayer::stored).
__device__ __forceinline__ bool gemma4_kv_stores(const Gemma4KvAddress& a, int b, int j) {
    return a.stored == nullptr || j < a.stored[b];
}

// Element offset of a binding's planes from the start of the layer's k (or v) tensor.
__device__ __forceinline__ std::int64_t gemma4_kv_plane_offset(const Gemma4KvAddress& a,
                                                               int binding) {
    return a.tables != nullptr ? 0 : static_cast<std::int64_t>(binding) * a.image_elements;
}

// The table row of a paged binding, or null for slot storage.
__device__ __forceinline__ const std::int32_t* gemma4_kv_table(const Gemma4KvAddress& a,
                                                               int binding) {
    return a.tables != nullptr ? a.tables + static_cast<std::int64_t>(binding) * a.pages_per_row
                               : nullptr;
}

// Pool row of absolute position p through one table row.
__device__ __forceinline__ int gemma4_kv_page_slot(const std::int32_t* table, int p) {
    return (table[p >> kGemma4PageTokensLog2] << kGemma4PageTokensLog2) +
           (p & ((1 << kGemma4PageTokensLog2) - 1));
}

// A row's scale sits at (element offset of the row) / D in the scale plane: the scale planes
// are the code planes without their D axis.

// Row of absolute position p inside one head plane of the binding.
__device__ __forceinline__ int gemma4_kv_slot(const Gemma4KvAddress& a, int binding, int p) {
    const std::int32_t* table = gemma4_kv_table(a, binding);
    return table != nullptr ? gemma4_kv_page_slot(table, p) : p % a.capacity;
}
#endif

// The shape facts of a layer binding, validated against the geometry. Throws
// std::invalid_argument naming `op` for anything outside the contract.
struct Gemma4KvBinding {
    Gemma4KvAddress address;
    std::int32_t batch        = 1; // batch rows the binding serves
    std::int32_t visible_rows = 0; // positions a binding row can hold: capacity, or 64 * pages
    bool paged                = false;
    bool fp8                  = false; // FP8_E4M3FN planes with FP32 row scales
};

[[nodiscard]] Gemma4KvBinding gemma4_kv_binding(const Gemma4KvLayer& cache, std::int32_t head_dim,
                                                std::int32_t kv_heads, const char* op);

} // namespace ninferno::ops::detail
