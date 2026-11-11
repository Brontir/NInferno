// The proof of codec_traits.h's rows (#615 F3): the host rules (r8_fits, w4a8_fits,
// mmq_plane_bytes) read the literal rows of codec_traits(WeightCodec), and this unit, compiled
// with the kernels, holds every row equal to the constants its codec struct is compiled with
// (generic_w4a8.cuh codec_traits<C>). A change to a codec's kMin, kHalfMin, kSpan, kGridWords or
// kBytes fails the build here until the row follows; no code is emitted.
#include "ops/generic/codec_traits.h"

#include "ops/generic/format_traits.h"
#include "ops/generic/generic_w4a8.cuh"

#include <array>
#include <cstddef>

namespace ninferno::ops::detail::generic {
namespace {

// Codec struct C is the codec W runs as: its constants are W's row. A mismatch names both here.
template <class C, WeightCodec W>
constexpr WeightCodec proven() {
    static_assert(codec_traits<C>() == codec_traits(W),
                  "codec_traits.h: codec W's row differs from the constants of codec struct C");
    return W;
}

// One line per codec a format runs as in the launch files (generic_w4a8_*.cu, generic_r8*.cu).
constexpr std::array kRows{
    proven<W4a8Q8, WeightCodec::Q8>(),
    proven<W4a8G64<4>, WeightCodec::Q4>(),
    proven<W4a8G64<5>, WeightCodec::Q5>(),
    proven<W4a8G64<6>, WeightCodec::Q6>(),
    proven<W4a8U4Zp, WeightCodec::U4Zp>(),
    proven<W4a8U4ZpG32, WeightCodec::U4ZpG32>(),
    proven<W4a8U4ZpG64, WeightCodec::U4ZpG64>(),
    proven<W4a8Q3<128>, WeightCodec::Q3G128>(),
    proven<W4a8Q3<64>, WeightCodec::Q3G64>(),
    proven<W4a8Ggml32<Ggml32::Q4_0>, WeightCodec::GgmlQ40>(),
    proven<W4a8Ggml32<Ggml32::Q4_1>, WeightCodec::GgmlQ41>(),
    proven<W4a8Ggml32<Ggml32::Q5_0>, WeightCodec::GgmlQ50>(),
    proven<W4a8Ggml32<Ggml32::Q5_1>, WeightCodec::GgmlQ51>(),
    proven<W4a8Ggml32<Ggml32::IQ4_NL>, WeightCodec::GgmlIq4Nl>(),
    proven<W4a8Iq4Xs, WeightCodec::GgmlIq4Xs>(),
    proven<W4a8Mxfp4, WeightCodec::Mxfp4>(),
    proven<W4a8Nvfp4, WeightCodec::Nvfp4>(),
    proven<W4a8Q45K<4>, WeightCodec::GgmlQ4K>(),
    proven<W4a8Q45K<5>, WeightCodec::GgmlQ5K>(),
    proven<W4a8Q6K, WeightCodec::GgmlQ6K>(),
    proven<W4a8Q3K, WeightCodec::GgmlQ3K>(),
    proven<W4a8Q2K, WeightCodec::GgmlQ2K>(),
    proven<W4a8Iq3Xxs, WeightCodec::GgmlIq3Xxs>(),
    proven<W4a8Iq3S, WeightCodec::GgmlIq3S>(),
    proven<W4a8Iq2Xxs, WeightCodec::GgmlIq2Xxs>(),
    proven<W4a8Iq2Xs, WeightCodec::GgmlIq2Xs>(),
    proven<W4a8Iq2S, WeightCodec::GgmlIq2S>(),
    proven<W4a8Iq1S, WeightCodec::GgmlIq1S>(),
    proven<W4a8Iq1M, WeightCodec::GgmlIq1M>(),
};

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

// Every format the generic path may run as W4A8 or R8 (format_traits.h) has its codec proven
// here: a codec added to a launch list and missed in codec_traits.h would read the default row,
// and r8_fits would size its shared memory short.
constexpr bool every_w4a8_r8_format_proven() {
    for (const FormatTraits& t : kFormatTraits) {
        if (t.w4a8 == Support::No && t.r8 == Support::No) continue;
        const auto codec = format_facts(t.q).codec;
        if (!codec || !proven_here(*codec)) return false;
    }
    return true;
}

static_assert(every_w4a8_r8_format_proven());

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
