#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninferno::ops {

/**
 * Op: fill_i32_positions
 *
 * Math / indexing:
 *   positions[i] = start + i, 0 <= i < T.
 *
 * Logical shapes:
 *   positions is a contiguous I32 vector [T].
 *
 * Numeric:
 *   T is positive, start is nonnegative, and start+T must not exceed INT32_MAX. Thus every
 *   emitted value is a nonnegative I32 position.
 *
 * Effects:
 *   Writes the full positions vector.
 *
 * Workspace:
 *   None. The Op has no other state side effect.
 */
void fill_i32_positions(Tensor& positions, std::int32_t start, cudaStream_t stream);

/**
 * Op: offset_i32_positions
 *
 * Math / indexing:
 *   destination[i] = source[i] + delta[0], 0 <= i < T.
 *
 * Logical shapes:
 *   source and destination are contiguous I32 vectors [T]; delta is an I32 scalar [1].
 *
 * Numeric:
 *   Callers provide values whose sums are representable by I32.
 *
 * Effects:
 *   Writes the full destination. Source and destination may alias; delta must not alias a written
 *   destination element.
 *
 * Workspace:
 *   None. The Op has no other state side effect.
 */
void offset_i32_positions(const Tensor& source, const Tensor& delta, Tensor& destination,
                          cudaStream_t stream);

/**
 * Op: feed_sampled_tokens
 *
 * Math / indexing:
 *   For 0 <= b < B:
 *     record[b] = sampled[b], tokens[b] = sampled[b],
 *     cache_positions[b]' = cache_positions[b] + 1, rope_positions[b]' = rope_positions[b] + 1.
 *   (A decode step's sampled tokens become the next step's inputs one position later.)
 *
 * Logical shapes:
 *   Contiguous I32 vectors [B], B >= 1, all of the same length.
 *
 * Numeric:
 *   Callers keep every advanced position within I32.
 *
 * Effects:
 *   Writes record and tokens and advances both position vectors in place. The five vectors are
 *   pairwise disjoint.
 *
 * Workspace:
 *   None. The Op has no other state side effect.
 */
void feed_sampled_tokens(const Tensor& sampled, Tensor& record, Tensor& tokens,
                         Tensor& cache_positions, Tensor& rope_positions, cudaStream_t stream);

} // namespace ninferno::ops
