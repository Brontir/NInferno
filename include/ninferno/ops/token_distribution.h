#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninferno::ops {

/**
 * Op: Top-K token log-probabilities, and the log-probabilities of given tokens
 *
 * The two halves of a quantization-quality measurement (ninferno-perplexity --kld): a reference
 * model records each position's K most probable tokens with their log-probabilities, and the
 * measured model reads its own log-probabilities of those same tokens.
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c] and
 *   lse[c] = log(sum_{r=0..valid_rows-1} exp(l[r,c])).
 *
 *   logprob_topk: ids[i,c] for i in [0,K) are the rows of the K largest l[r,c], r < valid_rows,
 *   by value descending and row ascending among equal values; logprobs[i,c] = l[ids[i,c],c] -
 *   lse[c]. A row more than kTokenTopkWindow below the column's maximum is never listed: its
 *   slot holds id -1 and logprob -inf (its probability is below e^-30 of the most probable row's).
 *
 *   gather_logprobs: out[i,c] = l[ids[i,c],c] - lse[c] for ids[i,c] in [0,valid_rows), and -inf
 *   for id -1.
 *
 * Logical shapes:
 *   logits is [physical_rows,C] with 1 <= valid_rows <= physical_rows; ids, logprobs and out are
 *   [K,C] with 1 <= K <= kTokenTopkMax. Rows [valid_rows,physical_rows) take no part.
 *
 * Supported domain:
 *   logits is contiguous finite BF16; ids I32, logprobs and out FP32, all contiguous with their
 *   dtype's natural alignment. gather_logprobs ids are in [0,valid_rows) or -1.
 *
 * Numeric:
 *   FP32 approximations of the formulas; the order of the listed rows is exact. When more than
 *   kTokenTopkCandidates rows share the K-th largest value, which of the tied rows are listed is
 *   unspecified (their log-probabilities are equal).
 *
 * Effects:
 *   Writes every output element and preserves the inputs. Outputs must not overlap the inputs.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on stream and owns no persistent state.
 */
inline constexpr std::int32_t kTokenTopkMax        = 64;
inline constexpr std::int32_t kTokenTopkCandidates = 1024;
inline constexpr float kTokenTopkWindow            = 30.0F;

void logprob_topk(const Tensor& logits, std::int32_t valid_rows, Tensor& ids, Tensor& logprobs,
                  cudaStream_t stream);

void gather_logprobs(const Tensor& logits, std::int32_t valid_rows, const Tensor& ids, Tensor& out,
                     cudaStream_t stream);

} // namespace ninferno::ops
