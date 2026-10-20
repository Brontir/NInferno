#pragma once

// Host-side operand checks shared by the ops wrappers. Messages read "<op>: <name> ...".

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninferno::ops {

// The element count of t, 0 when an extent is 0. A negative extent is std::invalid_argument; a
// count past int64 is std::overflow_error.
inline std::int64_t numel_allow_zero(const Tensor& t, const char* op, const char* name) {
    bool empty = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string(op) + ": " + name +
                                        " dimensions must be nonnegative");
        }
        empty = empty || t.ne[d] == 0;
    }
    if (empty) { return 0; }
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error(std::string(op) + ": " + name + " size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

inline void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                          std::int32_t n3, const char* op, const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

inline void require_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

// A named byte range of an operand, for require_disjoint.
struct OperandRange {
    const void* data;
    std::size_t bytes;
    const char* name;
};

// Throws std::invalid_argument("<op>: <a> overlaps <b>") for the first pair of ranges that share
// a byte, and std::overflow_error for a range past the end of the address space. A null or empty
// range shares none. The index pairs in may_alias may share storage (read-only operands).
inline void require_disjoint(std::span<const OperandRange> ranges, const char* op,
                             std::span<const std::pair<std::size_t, std::size_t>> may_alias = {}) {
    for (const OperandRange& r : ranges) {
        if (r.data != nullptr && r.bytes > std::numeric_limits<std::uintptr_t>::max() -
                                               reinterpret_cast<std::uintptr_t>(r.data)) {
            throw std::overflow_error(std::string(op) + ": " + r.name + " address range overflows");
        }
    }
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            bool excused = false;
            for (const auto& [a, b] : may_alias) { excused = excused || (a == i && b == j); }
            if (!excused &&
                overlaps(ranges[i].data, ranges[i].bytes, ranges[j].data, ranges[j].bytes)) {
                throw std::invalid_argument(std::string(op) + ": " + ranges[i].name + " overlaps " +
                                            ranges[j].name);
            }
        }
    }
}

} // namespace ninferno::ops
