#pragma once

// Size arithmetic that throws instead of wrapping. `what` names the quantity in the message.

#include <compare>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace ninferno {

[[nodiscard]] inline std::uint64_t checked_add(std::uint64_t a, std::uint64_t b,
                                               std::string_view what) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error(std::string(what) + " overflows");
    }
    return a + b;
}

[[nodiscard]] inline std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b,
                                               std::string_view what) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error(std::string(what) + " overflows");
    }
    return a * b;
}

// The full 128-bit product of two 64-bit words, without a 128-bit type (MSVC has none).
struct Wide {
    std::uint64_t high = 0;
    std::uint64_t low  = 0;
    // High word first: the members' order compares as the 128-bit numbers do.
    friend constexpr auto operator<=>(const Wide&, const Wide&) noexcept = default;
};

[[nodiscard]] inline Wide mul_wide(std::uint64_t a, std::uint64_t b) noexcept {
#if defined(__SIZEOF_INT128__)
    const unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
    return {static_cast<std::uint64_t>(product >> 64U), static_cast<std::uint64_t>(product)};
#else
    Wide out;
    out.low = _umul128(a, b, &out.high);
    return out;
#endif
}

// a + b, saturated to the largest uint64_t.
[[nodiscard]] constexpr std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return b > std::numeric_limits<std::uint64_t>::max() - a
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

// a + b of two 128-bit values, saturated to 64 bits.
[[nodiscard]] inline std::uint64_t saturating_u64(Wide a, Wide b) noexcept {
    const std::uint64_t low   = a.low + b.low;
    const std::uint64_t carry = low < a.low ? 1U : 0U;
    return a.high != 0 || b.high != 0 || carry != 0 ? std::numeric_limits<std::uint64_t>::max() : low;
}

// __builtin_mul_overflow: *out = a * b, true when the exact product does not fit in R (*out then
// holds the wrapped value on GCC and Clang, unspecified elsewhere). Integral types only.
template <class A, class B, class R>
[[nodiscard]] inline bool mul_overflow(A a, B b, R* out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, out);
#else
    static_assert(std::is_integral_v<A> && std::is_integral_v<B> && std::is_integral_v<R>);
    // |a| * |b| as 128 bits, then the sign, then whether R holds it.
    const bool negative = (a < 0) != (b < 0) && a != 0 && b != 0;
    const auto magnitude = [](auto v) {
        return v < 0 ? std::uint64_t(0) - static_cast<std::uint64_t>(v) : static_cast<std::uint64_t>(v);
    };
    std::uint64_t high = 0;
    const std::uint64_t low = _umul128(magnitude(a), magnitude(b), &high);
    *out = static_cast<R>(negative ? std::uint64_t(0) - low : low);
    if (high != 0) { return true; }
    if (negative) {
        if constexpr (std::is_unsigned_v<R>) {
            return true;
        } else {
            return low > static_cast<std::uint64_t>(std::numeric_limits<R>::max()) + 1U;
        }
    }
    return low > static_cast<std::uint64_t>(std::numeric_limits<R>::max());
#endif
}

// value rounded up to a multiple of alignment, which must be a power of two.
[[nodiscard]] inline std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment,
                                            std::string_view what) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string(what) + " alignment must be a power of two");
    }
    return checked_add(value, alignment - 1, what) & ~(alignment - 1);
}

} // namespace ninferno
