// ninferno::ops - logprob_topk / gather_logprobs wrapper: public contract validation and launcher
// dispatch.
#include "ninferno/ops/token_distribution.h"

#include "ops/launcher/token_distribution.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

[[noreturn]] void fail(const char* op, const std::string& message) {
    throw std::invalid_argument(std::string(op) + ": " + message);
}

void require_accessible(const char* op, const Tensor& tensor, DType dtype, std::size_t alignment,
                        const char* label) {
    if (tensor.dtype != dtype) fail(op, std::string(label) + " has the wrong dtype");
    if (!tensor.is_contiguous()) fail(op, std::string(label) + " must be contiguous");
    if (tensor.data == nullptr) fail(op, std::string(label) + " data must be non-null");
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        fail(op, std::string(label) + " data is not naturally aligned");
    }
}

// logits [physical_rows, C] with 1 <= valid_rows <= physical_rows; each of `per_column` [K, C].
std::int32_t validate(const char* op, const Tensor& logits, std::int32_t valid_rows,
                      std::initializer_list<const Tensor*> per_column) {
    require_accessible(op, logits, DType::BF16, alignof(std::uint16_t), "logits");
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1) {
        fail(op, "logits must be rank-2 with positive dimensions");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        fail(op, "valid_rows must be in [1, physical_rows]");
    }
    const std::int32_t k = (*per_column.begin())->ne[0];
    if (k < 1 || k > kTokenTopkMax) fail(op, "K must be in [1, kTokenTopkMax]");
    for (const Tensor* t : per_column) {
        if (t->ne[0] != k || t->ne[1] != logits.ne[1] || t->ne[2] != 1 || t->ne[3] != 1) {
            fail(op, "per-column tensors must have shape [K, columns]");
        }
    }
    return k;
}

} // namespace

void logprob_topk(const Tensor& logits, std::int32_t valid_rows, Tensor& ids, Tensor& logprobs,
                  cudaStream_t stream) {
    constexpr const char* kOp = "logprob_topk";
    validate(kOp, logits, valid_rows, {&ids, &logprobs});
    require_accessible(kOp, ids, DType::I32, alignof(std::int32_t), "ids");
    require_accessible(kOp, logprobs, DType::FP32, alignof(float), "logprobs");
    if (overlaps(ids, logits) || overlaps(logprobs, logits) || overlaps(ids, logprobs)) {
        fail(kOp, "outputs must not overlap the logits or each other");
    }
    detail::logprob_topk_launch(logits, valid_rows, ids, logprobs, stream);
}

void gather_logprobs(const Tensor& logits, std::int32_t valid_rows, const Tensor& ids, Tensor& out,
                     cudaStream_t stream) {
    constexpr const char* kOp = "gather_logprobs";
    validate(kOp, logits, valid_rows, {&ids, &out});
    require_accessible(kOp, ids, DType::I32, alignof(std::int32_t), "ids");
    require_accessible(kOp, out, DType::FP32, alignof(float), "out");
    if (overlaps(out, logits) || overlaps(out, ids)) {
        fail(kOp, "out must not overlap either input");
    }
    detail::gather_logprobs_launch(logits, valid_rows, ids, out, stream);
}

} // namespace ninferno::ops
