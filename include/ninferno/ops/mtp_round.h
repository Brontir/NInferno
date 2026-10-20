#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninferno::ops {

/**
 * Op: mtp_prepare_next_round
 *
 * Math / indexing:
 *   For T=K+1 and each row b with A=accepted[b], L=licensed_counts[b]:
 *     alignment_ids[j,b] = verify_ids[j+1,b]  for 0<=j<A
 *                          next_anchors[b]     otherwise;
 *     remaining_after = max(remaining_budgets[b]-L,0);
 *     context_after   = max(max_context-updated_frontiers[b]-1,0);
 *     next_extents[b] = min(K,max(remaining_after-1,0),context_after);
 *     For S=max(K-1,1) and 0<=s<S:
 *       ar_positions[b,s]      = min(updated_frontiers[b]+s,max_context-1);
 *       ar_rope_positions[b,s] = ar_positions[b,s]+rope_deltas[b];
 *       ar_valid_columns[b,s]  = (s+1 < next_extents[b]).
 *
 * Logical shapes / effects:
 *   verify_ids/alignment_ids are distinct contiguous I32 [K+1,B]. ar_positions,
 *   ar_rope_positions, and ar_valid_columns are I32 [B,max(K-1,1)] with contiguous rows and one
 *   shared step stride at least B; this permits an exact-B prefix of a fixed-capacity frame. All
 *   other tensors are contiguous I32 [B]. B>=1, 1<=K<=5, 0<=accepted[b]<=K,
 *   licensed_counts[b]=accepted[b]+1, updated_frontiers and remaining_budgets are non-negative,
 *   and max_context is positive. The Op writes every output slot, including safe invalid-tail
 *   values: an invalid step still runs and reads the keys up to its position, so no position
 *   leaves the context, even when the round ended on its last slot (updated_frontiers[b] =
 *   max_context). Inputs remain unchanged. No workspace or other state is used.
 */
void mtp_prepare_next_round(const Tensor& verify_ids, const Tensor& next_anchors,
                            const Tensor& accepted, const Tensor& updated_frontiers,
                            const Tensor& remaining_budgets, const Tensor& licensed_counts,
                            const Tensor& rope_deltas, Tensor& alignment_ids, Tensor& next_extents,
                            Tensor& ar_positions, Tensor& ar_rope_positions,
                            Tensor& ar_valid_columns, std::int32_t max_context,
                            cudaStream_t stream);

/**
 * Op: mtp_prepare_next_round with a draft depth D (1<=D<=K) below the verify width: the round
 * verified K+1 columns (a copy round), the drafter proposes only D for the next one. As above with
 * next_extents[b] = min(D, max(remaining_after-1,0), context_after) and S = max(D-1,1) AR steps;
 * the AR outputs are I32 [B,max(D-1,1)]. D = K is the overload above.
 */
void mtp_prepare_next_round(const Tensor& verify_ids, const Tensor& next_anchors,
                            const Tensor& accepted, const Tensor& updated_frontiers,
                            const Tensor& remaining_budgets, const Tensor& licensed_counts,
                            const Tensor& rope_deltas, Tensor& alignment_ids, Tensor& next_extents,
                            Tensor& ar_positions, Tensor& ar_rope_positions,
                            Tensor& ar_valid_columns, std::int32_t max_context,
                            std::int32_t draft_depth, cudaStream_t stream);

} // namespace ninferno::ops
