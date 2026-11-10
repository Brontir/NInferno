#pragma once

// What the host's fit and layout rules need to know of a W4A8 / R8 codec, read from the codec
// struct itself (generic_w4a8.cuh codec_traits<C>) and tabled by generic_codec_traits.cu, so
// r8_fits, w4a8_fits and mmq_plane_bytes (generic_gemm.cpp) read the constants the kernel is
// compiled with instead of a copy (#615 F3). Plain host header: no CUDA.

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

// The codec's traits; a default row for a codec with no W4A8 / R8 codec struct (BF16, FP8, the
// ternary and 1- / 2-bit ggml blocks).
[[nodiscard]] CodecTraits codec_traits(WeightCodec codec);

// The traits of the format's codec; the default row for a format with no codec.
[[nodiscard]] inline CodecTraits codec_traits(QType q) {
    const auto codec = format_facts(q).codec;
    return codec ? codec_traits(*codec) : CodecTraits{};
}

} // namespace ninferno::ops::detail::generic
