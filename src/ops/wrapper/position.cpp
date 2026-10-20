#include "ninferno/ops/position.h"

#include "ops/launcher/position.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

void require_i32_vector(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] <= 0 || tensor.ne[1] != 1 || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(name) +
                                    " must be a non-empty contiguous I32 vector");
    }
}

void require_i32_scalar(const Tensor& tensor, const char* name) {
    require_i32_vector(tensor, name);
    if (tensor.ne[0] != 1) {
        throw std::invalid_argument(std::string(name) + " must be an I32 scalar");
    }
}

} // namespace

void fill_i32_positions(Tensor& positions, std::int32_t start, cudaStream_t stream) {
    require_i32_vector(positions, "fill_i32_positions positions");
    if (start < 0 || start > std::numeric_limits<std::int32_t>::max() - positions.ne[0]) {
        throw std::invalid_argument("fill_i32_positions: range is outside non-negative I32");
    }
    detail::fill_i32_positions_launch(positions, start, stream);
}

void offset_i32_positions(const Tensor& source, const Tensor& delta, Tensor& destination,
                          cudaStream_t stream) {
    require_i32_vector(source, "offset_i32_positions source");
    require_i32_scalar(delta, "offset_i32_positions delta");
    require_i32_vector(destination, "offset_i32_positions destination");
    if (source.ne[0] != destination.ne[0]) {
        throw std::invalid_argument("offset_i32_positions: source and destination shapes differ");
    }
    if (delta.data == destination.data) {
        throw std::invalid_argument("offset_i32_positions: delta must not alias destination");
    }
    detail::offset_i32_positions_launch(source, delta, destination, stream);
}

void feed_sampled_tokens(const Tensor& sampled, Tensor& record, Tensor& tokens,
                         Tensor& cache_positions, Tensor& rope_positions, cudaStream_t stream) {
    require_i32_vector(sampled, "feed_sampled_tokens sampled");
    require_i32_vector(record, "feed_sampled_tokens record");
    require_i32_vector(tokens, "feed_sampled_tokens tokens");
    require_i32_vector(cache_positions, "feed_sampled_tokens cache positions");
    require_i32_vector(rope_positions, "feed_sampled_tokens RoPE positions");
    const std::array<const Tensor*, 5> vectors{&sampled, &record, &tokens, &cache_positions,
                                               &rope_positions};
    const auto bytes = static_cast<std::uintptr_t>(sampled.ne[0]) * sizeof(std::int32_t);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        if (vectors[i]->ne[0] != sampled.ne[0]) {
            throw std::invalid_argument("feed_sampled_tokens: vector lengths differ");
        }
        const auto begin = reinterpret_cast<std::uintptr_t>(vectors[i]->data);
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(vectors[j]->data);
            if (begin < other + bytes && other < begin + bytes) {
                throw std::invalid_argument("feed_sampled_tokens: vectors must not overlap");
            }
        }
    }
    detail::feed_sampled_tokens_launch(sampled, record, tokens, cache_positions, rope_positions,
                                       stream);
}

} // namespace ninferno::ops
