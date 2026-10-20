#include "models/speculation/ngram_round.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninferno::models::speculation {
namespace {

// Opens a prompt-lookup round: the verify batch covers P..P+k, the pending token first.
__global__ void ngram_begin(const std::int32_t* position, const std::int32_t* token,
                            std::int32_t* verify_positions, std::int32_t* draft_ids,
                            std::int32_t columns) {
    const std::int32_t p = *position;
    for (std::int32_t i = threadIdx.x; i < columns; i += blockDim.x) { verify_positions[i] = p + i; }
    if (threadIdx.x == 0) { draft_ids[0] = *token; }
}

} // namespace

NgramRound::NgramRound(DeviceContext& device, std::int32_t max_drafts)
    : device_(device), max_drafts_(max_drafts),
      draft_ids_(sizeof(std::int32_t) * (max_drafts + 1)),
      staged_(sizeof(std::int32_t) * max_drafts) {
    draft_ids_.fill(0);
    std::memset(staged_.data(), 0, staged_.size());
}

Tensor NgramRound::ids(std::int32_t columns) const {
    return Tensor(draft_ids_.p, DType::I32, {columns});
}

void NgramRound::stage(std::span<const std::int32_t> drafts) {
    if (drafts.size() > static_cast<std::size_t>(max_drafts_)) {
        throw std::invalid_argument("prompt lookup: too many drafts staged");
    }
    std::memcpy(staged_.data(), drafts.data(), drafts.size_bytes());
}

void NgramRound::draft(const std::int32_t* position, const std::int32_t* token,
                       std::int32_t* verify_positions, std::int32_t k) {
    const cudaStream_t s = device_.stream;
    ngram_begin<<<1, 32, 0, s>>>(position, token, verify_positions,
                                 static_cast<std::int32_t*>(draft_ids_.p), k + 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(draft_ids_.p) + 1, staged_.data(),
                               sizeof(std::int32_t) * k, cudaMemcpyHostToDevice, s));
}

std::vector<std::int32_t> ngram_lookup(std::span<const std::int32_t> history, std::int32_t min_n,
                                       std::int32_t max_n, std::int32_t limit) {
    const auto size = static_cast<std::int64_t>(history.size());
    for (std::int32_t n = std::min<std::int64_t>(max_n, size - 1); n >= min_n; --n) {
        const auto* suffix = history.data() + (size - n);
        // The most recent earlier occurrence: its continuation is the likeliest repeat.
        for (std::int64_t start = size - n - 1; start >= 0; --start) {
            if (std::equal(suffix, suffix + n, history.data() + start)) {
                const auto from = start + n;
                const auto take = std::min<std::int64_t>(limit, size - from);
                if (take <= 0) { continue; }
                return {history.begin() + from, history.begin() + from + take};
            }
        }
    }
    return {};
}

} // namespace ninferno::models::speculation
