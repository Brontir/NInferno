#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops {

/**
 * Op: scatter
 *
 * Math / indexing:
 *   destination[:, indices[i]] = source[:, i] for 0 <= i < V.
 *
 * Logical shapes:
 *   BF16 source [D,V], I32 indices [V], BF16 destination [D,T], all contiguous.
 *
 * Numeric:
 *   Exact BF16 element copies. Callers provide indices in [0,T) with no duplicates when
 *   deterministic overwrite order is required.
 *
 * Effects:
 *   Overwrites only the selected destination columns; all other columns retain their old values.
 *   Inputs and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect beyond the selected destination writes.
 *
 * Supported routes:
 *   D divisible by eight with 16-byte-aligned source/destination uses BF16x8 copies; other aligned
 *   even D uses BF16x2; remaining contiguous dimensions use scalar copies.
 */
void scatter(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream);

/**
 * Scatter compact request-major BF16 blocks into lane-owned fixed storage.
 *
 * source is contiguous [D,W,B], lanes and valid_columns are contiguous I32 [B], and destination
 * is a possibly row-sliced [D,W,C] view with dense rows and explicit column/lane strides.
 * C is lane capacity, independent of B; a feature slice retains its parent storage strides.
 * For each b and j<valid_columns[b]:
 *
 *   destination[:,j,lanes[b]] = source[:,j,b]
 *
 * Columns j>=valid_columns[b], every unselected lane and rows outside the destination slice
 * remain unchanged. The caller supplies distinct lanes in [0,C), valid_columns in [0,W], and
 * strides describing disjoint columns/lanes. D is divisible by eight and all column starts are
 * 16-byte aligned. Inputs and destination do not alias. There is no workspace; device zero
 * counts leave the corresponding lane unchanged without reading source values.
 */
void scatter_bf16_batch(const Tensor& source, const Tensor& lanes, const Tensor& valid_columns,
                        Tensor& destination, cudaStream_t stream);

/**
 * Op: extract_bf16_columns
 *
 * Math / indexing:
 *   For source [D,T] and destination [D',T]:
 *   destination[d,t] = source[source_offset + d,t], 0<=d<D', 0<=t<T.
 *
 * Logical shapes:
 *   Contiguous BF16 source [D,T] and destination [D',T].
 *   source_offset is an offset in the fastest (ne[0]) dimension; 0<=source_offset and
 *   source_offset+D'<=D.
 *
 * Numeric:
 *   Exact BF16 element copies.
 *
 * Effects:
 *   Writes the full destination; source and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void extract_bf16_columns(const Tensor& source, std::int32_t source_offset, Tensor& destination,
                          cudaStream_t stream);

/**
 * Op: regroup_bf16_heads
 *
 * Math / indexing:
 *   For source and destination [D,H,T] (T may span ne[2] and ne[3]) with H = groups * per:
 *   destination[:, j*groups + g, t] = source[:, g*per + j, t], 0<=g<groups, 0<=j<per.
 *   (Heads grouped by key head become llama.cpp's tiled order.)
 *
 * Logical shapes:
 *   Contiguous, same-shaped BF16 source and destination; D divisible by eight, groups divides H,
 *   both 16-byte aligned.
 *
 * Numeric:
 *   Exact BF16 element copies.
 *
 * Effects:
 *   Writes the full destination; source and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void regroup_bf16_heads(const Tensor& source, std::int32_t groups, Tensor& destination,
                        cudaStream_t stream);

} // namespace ninferno::ops
