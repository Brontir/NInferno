// The proof of codec_traits.h's rows (#615 F3): the host rules (r8_fits, w4a8_fits,
// mmq_plane_bytes) read the literal rows of codec_traits(WeightCodec), and this unit, compiled
// with the kernels, holds every row equal to the constants its codec struct is compiled with
// (generic_w4a8.cuh codec_traits<C>). A change to a codec's kMin, kHalfMin, kSpan, kGridWords or
// kBytes fails the build here until the row follows; no code is emitted.
//
// It also holds the launch lists (compiled_codecs.h) to one another: every R8 and producer-warp
// entry names the codec struct of its codec's W4A8 entry, and w4a8_ws_compiled<C> (the template
// the W4A8 launch switch reads) is true for exactly the producer-warp list's codecs.
#include "ops/generic/codec_traits.h"

#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_w4a8.cuh"

#include <array>
#include <cstddef>
#include <type_traits>

namespace ninferno::ops::detail::generic {
namespace {

// Codec struct C is the codec W runs as: its constants are W's row. A mismatch names both here.
template <class C, WeightCodec W>
constexpr WeightCodec proven() {
    static_assert(codec_traits<C>() == codec_traits(W),
                  "codec_traits.h: codec W's row differs from the constants of codec struct C");
    return W;
}

// One line per codec of the W4A8 launch files (generic_w4a8_*.cu); the R8 lists name the same
// structs (below).
#define NINFERNO_PROVEN_ROW(CODEC, TYPE) proven<TYPE, WeightCodec::CODEC>(),
constexpr std::array kRows{NINFERNO_W4A8_CODECS(NINFERNO_PROVEN_ROW)};
#undef NINFERNO_PROVEN_ROW

// The codec struct each W4A8 entry instantiates, by codec.
template <WeightCodec W>
struct W4a8StructOf {};

#define NINFERNO_W4A8_STRUCT_OF(CODEC, TYPE)                                                       \
    template <>                                                                                    \
    struct W4a8StructOf<WeightCodec::CODEC> {                                                      \
        using type = TYPE;                                                                         \
    };
NINFERNO_W4A8_CODECS(NINFERNO_W4A8_STRUCT_OF)
#undef NINFERNO_W4A8_STRUCT_OF

// An R8 or producer-warp entry's struct is its codec's W4A8 struct (a codec with no W4A8 entry
// fails here: W4a8StructOf has no `type`).
#define NINFERNO_SAME_STRUCT(CODEC, TYPE, ...)                                                     \
    static_assert(std::is_same_v<TYPE, typename W4a8StructOf<WeightCodec::CODEC>::type>,           \
                  "compiled_codecs.h: " #CODEC "'s entry names another struct than its W4A8 one");
NINFERNO_R8_CODECS(NINFERNO_SAME_STRUCT)
NINFERNO_W4A8_WS_CODECS(NINFERNO_SAME_STRUCT)
#undef NINFERNO_SAME_STRUCT

// w4a8_ws_compiled<C>, the launch's test, against the list w4a8_ws_supports reads.
#define NINFERNO_WS_AGREES(CODEC, TYPE)                                                            \
    static_assert(w4a8_ws_compiled<TYPE>() == compiled(Variant::W4a8Ws, WeightCodec::CODEC),       \
                  "generic_w4a8.cuh: w4a8_ws_compiled disagrees with NINFERNO_W4A8_WS_CODECS "     \
                  "for " #CODEC);
NINFERNO_W4A8_CODECS(NINFERNO_WS_AGREES)
#undef NINFERNO_WS_AGREES

constexpr bool proven_here(WeightCodec codec) {
    for (WeightCodec c : kRows) {
        if (c == codec) return true;
    }
    return false;
}

constexpr bool each_codec_once() {
    for (std::size_t i = 0; i < kRows.size(); ++i) {
        for (std::size_t j = i + 1; j < kRows.size(); ++j) {
            if (kRows[i] == kRows[j]) return false;
        }
    }
    return true;
}

static_assert(each_codec_once());

// Every codec the generic path may run as W4A8 or R8 (compiled_codecs.h) is proven here: a codec
// added to a launch list and missed in codec_traits.h would read the default row, and r8_fits
// would size its shared memory short.
constexpr bool every_w4a8_r8_codec_proven() {
    for (WeightCodec c : kCompiledW4a8Codecs) {
        if (!proven_here(c)) return false;
    }
    for (const R8CompiledCodec& e : kCompiledR8Codecs) {
        if (!proven_here(e.codec)) return false;
    }
    return true;
}

static_assert(every_w4a8_r8_codec_proven());

// A codec with no codec struct here has the default row: codec_traits.h states nothing a kernel
// does not hold.
constexpr bool unproven_codecs_default() {
    for (const FormatFacts& f : kFormatFacts) {
        if (f.codec && !proven_here(*f.codec) && !(codec_traits(*f.codec) == CodecTraits{})) {
            return false;
        }
    }
    return true;
}

static_assert(unproven_codecs_default());

} // namespace
} // namespace ninferno::ops::detail::generic
