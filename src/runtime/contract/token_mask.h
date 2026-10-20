#pragma once

#include "ninferno/types.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninferno::runtime {

// Engine side of the constrained-decoding mask protocol for one Program execution unit. Rows are
// aligned with the unit's membership order (a prefill unit has one row).
//
// The Program asks constrained() before it prepares the unit, points each constrained row's
// sampling configuration at a fixed mask buffer, launches the unit, and only then calls fill() for
// every constrained row, so mask computation on the host overlaps the device forward pass (a unit
// whose masks are slow to compute, row_cost_ns(), fills and releases them before its launch). The
// sampling and acceptance kernels wait for the Program's release of the unit's mask epoch before
// reading a mask; the Program releases on every path, including exceptions.
class TokenMaskSource {
public:
    virtual ~TokenMaskSource() = default;

    [[nodiscard]] virtual bool constrained(std::size_t row) const noexcept = 0;

    // Writes drafts.size()+1 mask rows of stride_words 32-bit words each. Row j constrains the
    // token sampled after the committed output plus drafts[0..j). Must not throw: a failure is
    // recorded by the source and expressed as a mask that ends the request. The source advances
    // *progress after every row it computes, so the waiting kernels know the host is working.
    virtual void fill(std::size_t row, std::span<const TokenId> drafts, std::uint32_t* masks,
                      std::size_t stride_words, std::uint32_t* progress) noexcept = 0;

    // The slowest mask row this row's request has computed so far, in nanoseconds (0: none yet).
    // A unit whose masks would take a large part of the kernels' wait bound computes them before
    // its launch instead.
    [[nodiscard]] virtual std::uint64_t row_cost_ns(std::size_t row) const noexcept {
        (void)row;
        return 0;
    }

    // The device did not observe this row's mask release within the kernels' wait bound; the
    // unit's tokens for the row were sampled without the mask.
    virtual void fault(std::size_t row) noexcept = 0;
};

} // namespace ninferno::runtime
