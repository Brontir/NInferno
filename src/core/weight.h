#pragma once

#include "core/dtype.h"
#include "core/format_facts.h"

#include <cstdint>

namespace ninferno {

// The formats (QType) and what is true of their bytes are core/format_facts.h; this header holds
// how a weight's planes sit in memory and the handle an operator reads them through.

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // MXFP4: codes [N, K/2] (element 2i in the low nibble), then at a 256-byte boundary the E8M0
    // exponents [N, K/32] row-major, one byte each. Rows are intrinsic, so row views are legal.
    BlockScaleK32E8M0 = 4,
    // FP8 block-scaled: codes [N, K], then at a 256-byte boundary one binary32 multiplier per
    // 128x128 block, [ceil(N/128), ceil(K/128)] row-major. Row views are not legal (a block spans
    // 128 rows); a packed parent is read whole.
    BlockScale128F32 = 5,
    // Zero-point groups of G = u4_zp_group(format) values: codes [N, K/2] (element 2i in the low
    // nibble), then at a 256-byte boundary the zero points [N, K/G] (one byte each, in the high
    // plane), then at a 256-byte boundary the binary16 scales [N, K/G]. Rows are intrinsic, so row
    // views are legal.
    RowGroupZp = 6,
    // ggml superblocks unchanged: row r is its K/256 blocks (144 bytes for Q4_K, 176 for Q5_K),
    // one plane. Rows are intrinsic, so row views are legal.
    GgmlBlocks = 7,
    // ggml blocks as planes, each block's bytes unchanged: per row the code bytes of each block
    // (Q6_K ql and qh, 192; Q3_K hmask and qs, 96; Q2_K qs, 64; IQ4_XS qs, 128; IQ3_XXS qs, 64;
    // IQ3_S qs, qh and signs, 104; IQ2_XXS qs, 64; IQ2_XS qs, 64; IQ2_S qs, signs and qh, 72;
    // IQ1_S, IQ1_M qs, 32; TQ1_0 qs and qh, 52; TQ2_0 qs, 64; Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL,
    // Q1_0, Q2_0 qs, 16), then at a 256-byte boundary its high bytes (Q6_K 16 int8 scales, Q3_K
    // 12 packed 6-bit scales, Q2_K 16 4-bit pairs, IQ4_XS scales_l 4, IQ3_XXS scales_and_signs
    // 32, IQ3_S scales 4, IQ2_XS / IQ2_S scales 8, IQ1_S / IQ1_M qh 16, Q5_0 / Q5_1 qh 4; none for
    // the others), then at a 256-byte boundary its scale bytes: binary16 d (Q2_K: d, dmin;
    // IQ4_XS: d, scales_h; Q4_1 / Q5_1: d, m; IQ1_M: its 8 scale bytes, d in their top nibbles).
    // ggml_plane_bytes() gives the sizes. Rows are intrinsic, so row views are legal.
    GgmlPlanes = 8,
    // A routed-expert bank of a ggml block format in fragment order (core/ggml_frag.h): the
    // geometry and planes of the format's published layout (GgmlBlocks for Q4_K, GgmlPlanes
    // otherwise), with each row's code bytes permuted by a fixed lossless byte map so that the
    // register MMA reads a lane's bytes of a K step as one aligned vector. High and scale planes
    // are unchanged. Rows are intrinsic, so row views are legal. Only the grouped MoE kernel reads
    // it; every other consumer refuses it.
    GgmlFrag = 9,
    // RowGroupZp whose K values are stored in a published order other than the natural one
    // (GPTQ act-order: every group's columns gathered together): the three RowGroupZp planes of
    // the reordered matrix, then at a 256-byte boundary K int32 indices, order[j] being the input
    // column stored at position j. The operator gathers x[order[j]] before multiplying, so the
    // product equals the published matrix's without a value changed. Rows are intrinsic, so row
    // views are legal (they share the order).
    RowGroupZpOrdered = 40,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;

    // An NVFP4 plane assembled from several source matrices carries one divisor per source, in the
    // payload after the scales. `weight_divisors` addresses them and `weight_divisor_rows` says how
    // many consecutive rows each covers, so the divisor of row r is element r /
    // weight_divisor_rows. A plane with one source sets the rows to its own row count, which makes
    // that index zero for every row, and `weight_scale_divisor` is then the whole story. On a stack
    // `weight_scale_divisor` holds only the first source's word, so a route that reads it for any
    // other row is silently wrong by a scale factor.
    const void* weight_divisors      = nullptr;
    std::int32_t weight_divisor_rows = 0;

    // RowGroupZpOrdered: K int32 input-column indices (see the layout); null for every other.
    const void* k_order = nullptr;
};

} // namespace ninferno
