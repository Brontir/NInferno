#pragma once

// The KV bindings of a ring decoder (the dense family and Gemma 4 under the Program): full layers
// read the paged pool, sliding layers their ring in a state image.

#include "core/device.h"
#include "core/tensor.h"
#include "models/qwen3_5/program/context.h"
#include "ninferno/ops/gemma4_attention.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninferno::models::qwen3_5::execution {

// One binding per layer: a full layer (window_of(i) == 0) reads the paged pool through `tables`
// (row rows_full[b] of batch row b), a sliding layer its ring in state image rows_ring[b]. With
// `stored`, batch row b writes the keys and values of its first stored[b] columns only (a
// verify's padding). index[i] is layer i's place among the layers of its kind.
template <class WindowOf>
[[nodiscard]] std::vector<ops::Gemma4KvLayer>
ring_decoder_bindings(const PagedKVCache& cache, const StateImageDevicePool& images,
                      std::span<const std::uint32_t> index, WindowOf window_of,
                      const Tensor& tables, const Tensor& rows_full, const Tensor& rows_ring,
                      const Tensor& stored) {
    std::vector<ops::Gemma4KvLayer> out;
    out.reserve(index.size());
    for (std::size_t i = 0; i < index.size(); ++i) {
        const std::int32_t window = window_of(i);
        if (window == 0) {
            const auto view = cache.batch_layer_view(index[i]);
            ops::Gemma4KvLayer layer{.k            = view.k_pages,
                                     .v            = view.v_pages,
                                     .window       = 0,
                                     .block_tables = tables,
                                     .rows         = rows_full,
                                     .stored       = stored};
            if (view.k_scale_pages.data != nullptr) {
                // Scale planes [1, 64, pages, heads]: the code planes without their D axis.
                const auto& k    = view.k_pages;
                const auto rows  = static_cast<std::int32_t>(k.ne[1]);
                const auto pages = static_cast<std::int32_t>(k.ne[2]);
                const auto heads = static_cast<std::int32_t>(k.ne[3]);
                layer.k_scale    = view.k_scale_pages.view({rows, pages, heads});
                layer.v_scale    = view.v_scale_pages.view({rows, pages, heads});
            }
            out.push_back(layer);
        } else {
            const auto planes = images.ring_planes(0, index[i]);
            ops::Gemma4KvLayer layer{.k      = planes.k,
                                     .v      = planes.v,
                                     .window = window,
                                     .rows   = rows_ring,
                                     .stored = stored};
            if (planes.k_scale.data != nullptr) {
                // Scale planes [1, capacity, heads, images]: [capacity, heads, images].
                const auto& sc = planes.k_scale;
                const auto cap = static_cast<std::int32_t>(sc.ne[1]);
                const auto hkv = static_cast<std::int32_t>(sc.ne[2]);
                const auto img = static_cast<std::int32_t>(sc.ne[3]);
                layer.k_scale  = planes.k_scale.view({cap, hkv, img});
                layer.v_scale  = planes.v_scale.view({cap, hkv, img});
            }
            out.push_back(layer);
        }
    }
    return out;
}

// The one-sequence bindings of a prefill or bridge: its table as a one-row table, its rings in
// state image `slot`. The two row indices live in `rows` (I32 [2], carved by the caller);
// bind(cache, images, tables, rows_full, rows_ring) is the family's batch bindings.
template <class Bind>
[[nodiscard]] std::vector<ops::Gemma4KvLayer>
ring_decoder_sequence_bindings(PrefillContext& state, std::int32_t slot, Tensor& rows, Bind bind) {
    const std::array<std::int32_t, 2> rows_host{0, slot};
    CUDA_CHECK(cudaMemcpyAsync(rows.data, rows_host.data(), sizeof(rows_host),
                               cudaMemcpyHostToDevice, state.execution.device.stream));
    const Tensor row_table = state.text_kv.layer_view(0).block_table;
    const Tensor table(row_table.data, DType::I32, {row_table.ne[0], 1});
    return bind(state.text_cache, state.execution.state_images, table, rows.slice(0, 0, 1),
                rows.slice(0, 1, 1));
}

} // namespace ninferno::models::qwen3_5::execution
