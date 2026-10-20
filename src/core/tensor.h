#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninferno {

struct Tensor {
    void* data         = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

// Whether a non-null pointer is a multiple of `alignment` (a power of two).
[[nodiscard]] inline bool aligned_to(const void* pointer, std::uintptr_t alignment) noexcept {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

// Whether two byte ranges share a byte. An empty or null range shares none.
[[nodiscard]] bool overlaps(const void* a, std::size_t a_bytes, const void* b,
                            std::size_t b_bytes) noexcept;

// Whether the bytes of two contiguous tensors, data through data + bytes(), share a byte.
[[nodiscard]] bool overlaps(const Tensor& a, const Tensor& b);

} // namespace ninferno
