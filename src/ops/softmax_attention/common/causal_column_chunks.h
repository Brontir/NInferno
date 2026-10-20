#pragma once

#include "core/arena.h"
#include "ops/softmax_attention/common/causal_operands.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {

// The columns one launch of a grouped kernel takes under the invariant route profile
// (prefill_routes.h): a wider call of one row runs as consecutive chunks of them after its keys
// and values are in the cache, so the grouped kernel serves every width and its partials stay
// within 64 columns x the heads x the aligned splits (64 at 262,144 keys). A column's bytes do
// not depend on the chunk that carries it: its row's splits are aligned to the keys, and the rows
// of a token tile do not mix.
inline constexpr int kCausalInvariantChunkColumns = 64;

// The columns one grouped launch of a `width`-column call takes: the whole call, or under the
// invariant route profile its chunks.
inline int causal_launch_columns(int width, bool invariant) noexcept {
    return invariant ? std::min(width, kCausalInvariantChunkColumns) : width;
}

// out[0] = clamp(valid[0] - offset, 0, width): the valid columns of the chunk at `offset`.
void causal_chunk_valid_columns(const std::int32_t* valid, int offset, int width, std::int32_t* out,
                                cudaStream_t stream);

// The workspace for_each_causal_column_chunk takes besides the launch's partials.
template <class Allocator>
void allocate_causal_chunk_metadata(Allocator& allocator) {
    (void)allocator.alloc(DType::I32, {1});
}

// Runs `run(operands, valid_columns)` over consecutive chunks of at most `columns` columns of a
// one-row call, or once over the whole call when it fits.
template <class Run>
void for_each_causal_column_chunk(const CausalAttentionOperands& p, const std::int32_t* valid,
                                  int columns, WorkspaceArena& workspace, cudaStream_t stream,
                                  Run&& run) {
    if (p.width <= columns) {
        run(p, valid);
        return;
    }
    if (p.batch != 1)
        throw std::invalid_argument("causal attention: column chunks of one row only");
    std::int32_t* chunk_valid =
        valid ? static_cast<std::int32_t*>(workspace.alloc(DType::I32, {1}).data) : nullptr;
    const std::size_t column = static_cast<std::size_t>(p.query_heads) * p.head_dim;
    for (int offset = 0; offset < p.width; offset += columns) {
        auto part  = p;
        part.width = std::min(columns, p.width - offset);
        part.q += column * offset;
        part.out += column * offset;
        part.positions += offset;
        if (part.gate) part.gate += column * offset;
        if (valid) causal_chunk_valid_columns(valid, offset, part.width, chunk_valid, stream);
        run(part, chunk_valid);
    }
}

} // namespace ninferno::ops::detail
