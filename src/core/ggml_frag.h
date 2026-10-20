#pragma once

// ggml_frag_v1 (QuantLayout::GgmlFrag): a routed-expert bank of a ggml block format whose code
// bytes are stored in the order the register MMA (ops/generic/generic_gemm.cuh) reads them. The
// published decoders gather a lane's bytes of a K step from four or more 4-byte words 16 bytes
// apart; in fragment order they are one aligned 8- or 16-byte vector. The map is a fixed
// permutation of each row's code bytes (docs/maintainer/storage-layouts.md, ggml_frag_v1): the
// numeric format, the geometry, the high and scale planes are the published layout's, and the
// inverse map gives back the published bytes, so the layout is lossless.
//
// Units of the map, `new[i] = old[source(i)]` within a unit:
//   Q4_0     each whole 64-byte span of four blocks of a row (4 x 4 transpose of 32-bit words:
//            new[16q + 4i + t] = old[16i + 4q + t]); a row's last K/32 % 4 blocks stay natural
//   Q4_K     each 144-byte superblock: the 16-byte head unchanged, each 64-byte half of the 128
//            nibble bytes 4 x 4-transposed by words
//   IQ3_XXS  each 64-byte qs plane block: in each 32-byte half an 8 x 4 byte transpose,
//            new[32h + 8q + m] = old[32h + 4m + q]
//   IQ2_S    each 72-byte code block: per 32-value half h, [qs even | signs even | qs odd |
//            signs odd] (8 bytes each) at 32h; the eight qh bytes unchanged

#include "core/weight.h"

#include <cstdint>

namespace ninferno {

// Whether a format has a ggml_frag_v1 map (and a fragment-order grouped MoE kernel).
[[nodiscard]] bool ggml_frag_format(QType format) noexcept;

// The layout a ggml block format is published in: GgmlBlocks (Q4_K, Q5_K) or GgmlPlanes.
[[nodiscard]] QuantLayout ggml_published_layout(QType format) noexcept;

// Bytes of one map unit (Q4_0: a four-block span) and the published byte the fragment-order
// byte `i` of a unit holds. Zero for a format without a map.
[[nodiscard]] std::uint32_t ggml_frag_unit_bytes(QType format) noexcept;
[[nodiscard]] std::uint32_t ggml_frag_source(QType format, std::uint32_t i) noexcept;

// One row of K values' code bytes (the row's span of the code plane, code_bytes_per_row bytes)
// from the published order to fragment order (to_frag) or back. `in` and `out` do not overlap.
// Throws for a format without a map or K not a whole number of its blocks.
void ggml_frag_row(QType format, std::uint64_t k, const std::uint8_t* in, std::uint8_t* out,
                   bool to_frag);

} // namespace ninferno
