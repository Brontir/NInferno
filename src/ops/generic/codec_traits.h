#pragma once

// What the host's fit and layout rules need to know of a W4A8 / R8 codec (#615 F3): one row per
// WeightCodec, read by r8_fits and mmq_plane_bytes (generic_gemm.cpp) and w4a8_fits
// (generic_routes.cpp, the host-only part of the selector). Plain host header, no CUDA: the rows
// are literals here, so ninferno_ops_host and the device library share one constexpr definition.
// generic_codec_traits.cu proves them: it static_asserts that every codec struct of the launch
// files (generic_w4a8.cuh codec_traits<C>) equals its row, so a change to a kernel's codec
// constants fails the build until the row follows.

#include "core/format_facts.h"

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::detail::generic {

struct CodecTraits {
    // kMin: the block has a min term, so the activation's block sums travel with it
    // (r8_smem_bytes, W4a8Stage::kSum).
    bool min = false;
    // kHalfMin (Q2_K): a min per 16 values in a plane of its own (W4a8Buffer::b_hmin).
    bool half_min = false;
    // kSpan: neighbouring blocks that share their bytes and load at once.
    int span = 1;
    // kGridWords: a codebook codec's grid, staged in shared memory below sm_80 (r8_grid_words).
    int grid_words = 0;
    // kBytes: the whole kept block the codec reads (Q4_K / Q5_K superblocks); 0 where it reads
    // the ggml planes (ggml_plane_bytes).
    int block_bytes = 0;

    constexpr bool operator==(const CodecTraits&) const = default;
};

// The staged grid starts at the next 8-byte boundary after the activation (whose bytes are a
// multiple of 4): at most this many bytes of padding (r8_launch_tokens, r8_fits).
inline constexpr std::size_t r8_grid_pad = 4;

// The codec's traits, as its codec struct (generic_w4a8.cuh) states them; the default row for a
// codec with no W4A8 / R8 codec struct (BF16, FP8, the ternary and 1- / 2-bit ggml blocks). Every
// enumerator is a case (no default), so a new codec is a -Wswitch warning until it has its row.
[[nodiscard]] constexpr CodecTraits codec_traits(WeightCodec codec) noexcept {
    using C = WeightCodec;
    switch (codec) {
    // The 32-value ggml formats with a min (W4a8Ggml32 kWithMin).
    case C::GgmlQ41:
    case C::GgmlQ51:
        return {.min = true};
    // The Q4_K / Q5_K superblocks, read whole, two blocks a load (W4a8Q45K).
    case C::GgmlQ4K:
        return {.min = true, .span = 2, .block_bytes = 144};
    case C::GgmlQ5K:
        return {.min = true, .span = 2, .block_bytes = 176};
    case C::GgmlQ6K:
        return {.span = 2};
    case C::GgmlQ3K:
        return {.span = 4};
    case C::GgmlQ2K:
        return {.half_min = true, .span = 4};
    // The codebook codecs' grids (kGridWords, 32-bit words).
    case C::GgmlIq3Xxs:
        return {.grid_words = 256};
    case C::GgmlIq3S:
    case C::GgmlIq2Xxs:
        return {.grid_words = 512};
    case C::GgmlIq2Xs:
        return {.grid_words = 1024};
    case C::GgmlIq2S:
    case C::GgmlIq1S:
    case C::GgmlIq1M:
        return {.grid_words = 2048};
    // Codec structs with every field at its default.
    case C::Q8:
    case C::Q4:
    case C::Q5:
    case C::Q6:
    case C::U4Zp:
    case C::U4ZpG32:
    case C::U4ZpG64:
    case C::Q3G128:
    case C::Q3G64:
    case C::GgmlQ40:
    case C::GgmlQ50:
    case C::GgmlIq4Nl:
    case C::GgmlIq4Xs:
    case C::Mxfp4:
    case C::Nvfp4:
    // No W4A8 / R8 codec struct.
    case C::Bf16:
    case C::Fp8:
    case C::Fp8RowF32:
    case C::Fp8B128:
    case C::GgmlTq10:
    case C::GgmlTq20:
    case C::GgmlQ10:
    case C::GgmlQ20:
        return {};
    }
    return {};
}

// The traits of the format's codec; the default row for a format with no codec.
[[nodiscard]] constexpr CodecTraits codec_traits(QType q) noexcept {
    const auto codec = format_facts(q).codec;
    return codec ? codec_traits(*codec) : CodecTraits{};
}

} // namespace ninferno::ops::detail::generic
