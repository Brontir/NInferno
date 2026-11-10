// The codec constants the host rules read (codec_traits.h): one row per codec of the W4A8 and R8
// launch files, taken from the codec struct itself (generic_w4a8.cuh codec_traits<C>), so
// r8_fits, w4a8_fits and mmq_plane_bytes read what the kernel is compiled with (#615 F3). The
// static_asserts pin the literals those rules carried before.
#include "ops/generic/codec_traits.h"

#include "ops/generic/format_traits.h"
#include "ops/generic/generic_w4a8.cuh"

#include <array>
#include <cstddef>
#include <initializer_list>

namespace ninferno::ops::detail::generic {
namespace {

struct Row {
    WeightCodec codec;
    CodecTraits traits;
};

template <class C>
constexpr Row row(WeightCodec codec) {
    return {codec, codec_traits<C>()};
}

// One line per codec struct the launch files instantiate (generic_w4a8_*.cu, generic_r8*.cu).
constexpr std::array kRows{
    row<W4a8Q8>(WeightCodec::Q8),
    row<W4a8G64<4>>(WeightCodec::Q4),
    row<W4a8G64<5>>(WeightCodec::Q5),
    row<W4a8G64<6>>(WeightCodec::Q6),
    row<W4a8U4Zp>(WeightCodec::U4Zp),
    row<W4a8U4ZpG32>(WeightCodec::U4ZpG32),
    row<W4a8U4ZpG64>(WeightCodec::U4ZpG64),
    row<W4a8Q3<128>>(WeightCodec::Q3G128),
    row<W4a8Q3<64>>(WeightCodec::Q3G64),
    row<W4a8Ggml32<Ggml32::Q4_0>>(WeightCodec::GgmlQ40),
    row<W4a8Ggml32<Ggml32::Q4_1>>(WeightCodec::GgmlQ41),
    row<W4a8Ggml32<Ggml32::Q5_0>>(WeightCodec::GgmlQ50),
    row<W4a8Ggml32<Ggml32::Q5_1>>(WeightCodec::GgmlQ51),
    row<W4a8Ggml32<Ggml32::IQ4_NL>>(WeightCodec::GgmlIq4Nl),
    row<W4a8Iq4Xs>(WeightCodec::GgmlIq4Xs),
    row<W4a8Mxfp4>(WeightCodec::Mxfp4),
    row<W4a8Nvfp4>(WeightCodec::Nvfp4),
    row<W4a8Q45K<4>>(WeightCodec::GgmlQ4K),
    row<W4a8Q45K<5>>(WeightCodec::GgmlQ5K),
    row<W4a8Q6K>(WeightCodec::GgmlQ6K),
    row<W4a8Q3K>(WeightCodec::GgmlQ3K),
    row<W4a8Q2K>(WeightCodec::GgmlQ2K),
    row<W4a8Iq3Xxs>(WeightCodec::GgmlIq3Xxs),
    row<W4a8Iq3S>(WeightCodec::GgmlIq3S),
    row<W4a8Iq2Xxs>(WeightCodec::GgmlIq2Xxs),
    row<W4a8Iq2Xs>(WeightCodec::GgmlIq2Xs),
    row<W4a8Iq2S>(WeightCodec::GgmlIq2S),
    row<W4a8Iq1S>(WeightCodec::GgmlIq1S),
    row<W4a8Iq1M>(WeightCodec::GgmlIq1M),
};

constexpr bool each_codec_once() {
    for (std::size_t i = 0; i < kRows.size(); ++i) {
        for (std::size_t j = i + 1; j < kRows.size(); ++j) {
            if (kRows[i].codec == kRows[j].codec) return false;
        }
    }
    return true;
}

static_assert(each_codec_once());

// Every format the generic path may run as W4A8 or R8 (format_traits.h) has its codec here: a
// codec added to a launch list and missed here would read the default traits, and r8_fits would
// size its shared memory short.
constexpr bool every_w4a8_r8_format_listed() {
    for (const FormatTraits& t : kFormatTraits) {
        if (t.w4a8 == Support::No && t.r8 == Support::No) continue;
        const auto codec = format_facts(t.q).codec;
        if (!codec) return false;
        bool found = false;
        for (const Row& r : kRows) found = found || r.codec == *codec;
        if (!found) return false;
    }
    return true;
}

static_assert(every_w4a8_r8_format_listed());

constexpr CodecTraits traits_of(WeightCodec codec) {
    for (const Row& r : kRows) {
        if (r.codec == codec) return r.traits;
    }
    return {};
}

// The literals the host rules carried before F3, now read from the codecs: every other codec
// has the default of each field.
constexpr bool only(bool (*field)(const CodecTraits&), std::initializer_list<WeightCodec> codecs) {
    for (const Row& r : kRows) {
        bool listed = false;
        for (WeightCodec c : codecs) listed = listed || c == r.codec;
        if (field(r.traits) != listed) return false;
    }
    return true;
}

// r8_fits: the formats whose activation carries the block sums (40 bytes a slot, not 36).
static_assert(only([](const CodecTraits& t) { return t.min; },
                   {WeightCodec::GgmlQ41, WeightCodec::GgmlQ51, WeightCodec::GgmlQ4K,
                    WeightCodec::GgmlQ5K}));
// w4a8_fits: the half-min plane (block_n * 32 bytes a buffer).
static_assert(only([](const CodecTraits& t) { return t.half_min; }, {WeightCodec::GgmlQ2K}));
// mmq_plane_bytes: the whole superblock.
static_assert(only([](const CodecTraits& t) { return t.block_bytes > 0; },
                   {WeightCodec::GgmlQ4K, WeightCodec::GgmlQ5K}));
static_assert(traits_of(WeightCodec::GgmlQ4K).block_bytes == 144 &&
              traits_of(WeightCodec::GgmlQ5K).block_bytes == 176);
// r8_fits: the staged grid's bytes (sizeof(std::uint32_t) * kGridWords).
static_assert(only([](const CodecTraits& t) { return t.grid_words > 0; },
                   {WeightCodec::GgmlIq3Xxs, WeightCodec::GgmlIq3S, WeightCodec::GgmlIq2Xxs,
                    WeightCodec::GgmlIq2Xs, WeightCodec::GgmlIq2S, WeightCodec::GgmlIq1S,
                    WeightCodec::GgmlIq1M}));
static_assert(4 * traits_of(WeightCodec::GgmlIq3Xxs).grid_words == 1024 &&
              4 * traits_of(WeightCodec::GgmlIq3S).grid_words == 2048 &&
              4 * traits_of(WeightCodec::GgmlIq2Xxs).grid_words == 2048 &&
              4 * traits_of(WeightCodec::GgmlIq2Xs).grid_words == 4096 &&
              4 * traits_of(WeightCodec::GgmlIq2S).grid_words == 8192 &&
              4 * traits_of(WeightCodec::GgmlIq1S).grid_words == 8192 &&
              4 * traits_of(WeightCodec::GgmlIq1M).grid_words == 8192);
// kSpan as the codec comments state it: two blocks a load for Q4_K / Q5_K and Q6_K, four for
// Q3_K and Q2_K, one for a 32-value block format.
static_assert(traits_of(WeightCodec::GgmlQ4K).span == 2 &&
              traits_of(WeightCodec::GgmlQ5K).span == 2 &&
              traits_of(WeightCodec::GgmlQ6K).span == 2 &&
              traits_of(WeightCodec::GgmlQ3K).span == 4 &&
              traits_of(WeightCodec::GgmlQ2K).span == 4 && traits_of(WeightCodec::Q8).span == 1);

} // namespace

CodecTraits codec_traits(WeightCodec codec) { return traits_of(codec); }

} // namespace ninferno::ops::detail::generic
