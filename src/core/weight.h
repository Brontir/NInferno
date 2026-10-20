#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninferno {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    // OCP MXFP4: E2M1 codes with one E8M0 exponent per 32 consecutive K values, no global scale.
    MXFP4 = 9,
    // FP8 E4M3FN codes with one binary32 multiplier per row (per-channel or per-tensor scales a
    // BF16 word cannot hold).
    FP8_E4M3FN_ROW_F32 = 10,
    // FP8 E4M3FN codes with one binary32 multiplier per 128x128 block (rows x K).
    FP8_E4M3FN_B128_F32 = 11,
    // Unsigned 4-bit codes with one zero point and one binary16 scale per 128 consecutive K values:
    // w = (code - zero) * scale (AWQ, GPTQ, compressed-tensors int4 g128).
    U4_G128_ZP_FP16 = 12,
    // The same with one zero point and one binary16 scale per 32 (64) consecutive K values, in the
    // same layout (RowGroupZp): AWQ, GPTQ, compressed-tensors and AutoRound int4 g32 (g64).
    U4_G32_ZP_FP16 = 40,
    U4_G64_ZP_FP16 = 41,
    // Symmetric 3-bit codes in [-4, 3] with one binary16 multiplier per 128 (64) consecutive K
    // values: w = code * scale. RowSplit: code j of a group in bits 3j..3j+2 of its 48 (24) code
    // bytes, no high plane (the GSQ 3-bit checkpoints the RTX 4080 and 5080 forks repack).
    Q3_G128_FP16 = 42,
    Q3_G64_FP16  = 43,
    // ggml's K-quants, kept as published (ggml semantics, 256-value superblocks along K):
    // Q4_K w = d*sc*q - dmin*m and Q5_K the same with a fifth code bit (6-bit sc and m per 32
    // values, binary16 d and dmin per superblock); Q6_K w = d*sc*(q - 32) (int8 sc per 16 values);
    // Q3_K w = d*(sc - 32)*q, q in [-4, 3] (6-bit sc per 16 values); Q2_K w = d*sc*q - dmin*m,
    // q in [0, 3] (4-bit sc and m per 16 values).
    GGML_Q4_K = 13,
    GGML_Q5_K = 14,
    GGML_Q6_K = 15,
    GGML_Q3_K = 16,
    GGML_Q2_K = 17,
    // ggml's i-quants, kept as published (256-value superblocks along K, binary16 d):
    // IQ4_XS w = d*(ls - 32)*kvalues_iq4nl[q] (4-bit codes into a 16-entry int8 codebook, 6-bit
    // ls per 32 values); IQ3_XXS w = d*(0.5 + ls)*0.5*grid*sign (one byte per 4 values into the
    // 256-entry iq3xxs_grid, 7-bit sign patterns per 8 values, 4-bit ls per 32 values); IQ3_S
    // w = d*(1 + 2*ls)*grid*sign (9-bit indices into the 512-entry iq3s_grid, a sign bit per
    // value, 4-bit ls per 32 values).
    GGML_IQ4_XS  = 18,
    GGML_IQ3_XXS = 19,
    GGML_IQ3_S   = 20,
    // ggml's 32-value block formats, kept as published (binary16 d per block; _1: also m):
    // Q4_0 w = d*(q - 8), Q4_1 w = d*q + m (4-bit q); Q5_0 w = d*(q - 16), Q5_1 w = d*q + m (5-bit
    // q, the fifth bits in a 32-bit qh word); IQ4_NL w = d*kvalues_iq4nl[q].
    GGML_Q4_0   = 21,
    GGML_Q4_1   = 22,
    GGML_Q5_0   = 23,
    GGML_Q5_1   = 24,
    GGML_IQ4_NL = 25,
    // ggml's 2- and 1-bit i-quants, kept as published (256-value superblocks): IQ2_XXS
    // w = d*(0.5 + ls)*0.25*grid*sign (one byte per 8 values into the 256-entry iq2xxs_grid,
    // 7-bit sign patterns, 4-bit ls per 32 values); IQ2_XS the same with 9-bit indices into
    // iq2xs_grid and a 4-bit ls per 16 values; IQ2_S 10-bit indices into iq2s_grid and a sign bit
    // per value; IQ1_S w = d*(2*ls + 1)*(grid + delta) (11-bit indices into iq1s_grid, values -1,
    // 0, 1; delta +-1/8 and a 3-bit ls per 32 values); IQ1_M the same with a 3-bit ls per 16
    // values and a delta per 8, its binary16 d spread over the top nibbles of the scales.
    GGML_IQ2_XXS = 26,
    GGML_IQ2_XS  = 27,
    GGML_IQ2_S   = 28,
    GGML_IQ1_S   = 29,
    GGML_IQ1_M   = 30,
    // ggml's ternary formats (256-value superblocks, binary16 d): TQ1_0 w = d*(t - 1), five base-3
    // digits t per byte; TQ2_0 w = d*(q - 1), four 2-bit q per byte.
    GGML_TQ1_0 = 31,
    GGML_TQ2_0 = 32,
    // ggml's Q1_0 (128-value blocks: w = +d or -d by one bit per value) and Q2_0 (64-value
    // blocks: w = d*(q - 1), 2-bit q), binary16 d per block.
    GGML_Q1_0 = 33,
    GGML_Q2_0 = 34,
};

// ggml's block formats the engine runs as published: the K-quants (GGML_Q2_K .. GGML_Q6_K), the
// i-quants (GGML_IQ4_XS, GGML_IQ3_XXS, GGML_IQ3_S, GGML_IQ2_XXS .. GGML_IQ1_M), the 32-value
// formats (GGML_Q4_0 .. GGML_IQ4_NL), the ternary TQ1_0 / TQ2_0 and Q1_0 / Q2_0.
constexpr bool is_ggml_block(QType q) noexcept {
    const auto v = static_cast<std::uint16_t>(q);
    return (v >= static_cast<std::uint16_t>(QType::GGML_Q4_K) &&
            v <= static_cast<std::uint16_t>(QType::GGML_Q2_K)) ||
           (v >= static_cast<std::uint16_t>(QType::GGML_IQ4_XS) &&
            v <= static_cast<std::uint16_t>(QType::GGML_Q2_0));
}

// Per ggml block in each ggml_planes_v1 plane (GgmlPlanes): its values along K and its code,
// high and scale bytes. Zero for a format not in that layout.
struct GgmlPlaneBytes {
    std::uint32_t values, code, high, scale;
};
constexpr GgmlPlaneBytes ggml_plane_bytes(QType q) noexcept {
    switch (q) {
    case QType::GGML_Q6_K: return {256, 192, 16, 2};
    case QType::GGML_Q3_K: return {256, 96, 12, 2};
    case QType::GGML_Q2_K: return {256, 64, 16, 4};
    case QType::GGML_IQ4_XS: return {256, 128, 4, 4};
    case QType::GGML_IQ3_XXS: return {256, 64, 32, 2};
    case QType::GGML_IQ3_S: return {256, 104, 4, 2};
    case QType::GGML_Q4_0: return {32, 16, 0, 2};
    case QType::GGML_Q4_1: return {32, 16, 0, 4};
    case QType::GGML_Q5_0: return {32, 16, 4, 2};
    case QType::GGML_Q5_1: return {32, 16, 4, 4};
    case QType::GGML_IQ4_NL: return {32, 16, 0, 2};
    case QType::GGML_IQ2_XXS: return {256, 64, 0, 2};
    case QType::GGML_IQ2_XS: return {256, 64, 8, 2};
    case QType::GGML_IQ2_S: return {256, 72, 8, 2};
    case QType::GGML_IQ1_S: return {256, 32, 16, 2};
    case QType::GGML_IQ1_M: return {256, 32, 16, 8};
    case QType::GGML_TQ1_0: return {256, 52, 0, 2};
    case QType::GGML_TQ2_0: return {256, 64, 0, 2};
    case QType::GGML_Q1_0: return {128, 16, 0, 2};
    case QType::GGML_Q2_0: return {64, 16, 0, 2};
    default: return {0, 0, 0, 0};
    }
}

// Values along K per block of a ggml block format (256 for GgmlBlocks' Q4_K / Q5_K).
constexpr std::uint32_t ggml_block_values(QType q) noexcept {
    return q == QType::GGML_Q4_K || q == QType::GGML_Q5_K ? 256 : ggml_plane_bytes(q).values;
}

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

// The K values one zero point and scale of a u4 zero-point format cover; 0 for other formats.
constexpr std::uint32_t u4_zp_group(QType q) noexcept {
    switch (q) {
    case QType::U4_G32_ZP_FP16: return 32;
    case QType::U4_G64_ZP_FP16: return 64;
    case QType::U4_G128_ZP_FP16: return 128;
    default: return 0;
    }
}

// The K values one scale of a 3-bit groupwise format covers; 0 for other formats.
constexpr std::uint32_t q3_group(QType q) noexcept {
    return q == QType::Q3_G128_FP16 ? 128 : q == QType::Q3_G64_FP16 ? 64 : 0;
}

} // namespace ninferno
