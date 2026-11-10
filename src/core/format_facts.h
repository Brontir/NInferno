#pragma once

// The stored weight formats and what is true of their bytes, in one table (architecture review
// item 3, F2). A row per QType: its two spellings (the artifact's and the route table's), the GGUF
// type id it is kept from, the generic codec that reads it, ggml's MMQ loader for it, whether its
// codes widen to int8 exactly, its ggml block geometry and the group lengths of the groupwise
// formats. Every field holds for every build and every card; which kernels a build compiles for a
// format and what a card was measured to prefer live elsewhere (ops/generic/format_traits.h).
//
// The table is host-only: lookups loop over a host constexpr array, so a device-side call would not
// compile into anything useful. A QType without a row (a value the enum does not name) reads as the
// default row kNoFormatFacts: no spelling, no codec, no geometry, and a `.q` that means nothing.
// The checks at the end hold the table to the enum and to itself.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

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

// The decoder of the generic projection path that reads a format (ops/generic). U4 G32 / G64 and
// Q3 G128 / G64 have codecs of their own; every other QType maps to one codec and back.
enum class WeightCodec : std::uint8_t {
    Bf16,
    Fp8,
    Nvfp4,
    Q4,
    Q5,
    Q6,
    Q8,
    Mxfp4,
    Fp8RowF32,
    Fp8B128,
    U4Zp,
    GgmlQ4K,
    GgmlQ5K,
    GgmlQ6K,
    GgmlQ3K,
    GgmlQ2K,
    GgmlIq4Xs,
    GgmlIq3Xxs,
    GgmlIq3S,
    GgmlQ40,
    GgmlQ41,
    GgmlQ50,
    GgmlQ51,
    GgmlIq4Nl,
    GgmlIq2Xxs,
    GgmlIq2Xs,
    GgmlIq2S,
    GgmlIq1S,
    GgmlIq1M,
    GgmlTq10,
    GgmlTq20,
    GgmlQ10,
    GgmlQ20,
    U4ZpG32,
    U4ZpG64,
    Q3G128,
    Q3G64,
};

// The loader of ggml's integer tensor-core kernel (ops/generic/ggml_mmq) that reads a ggml block
// format as published; IQ1_M has the vector kernel only (ggml_mmq::matrix_supported).
enum class MmqType : std::uint8_t {
    Q4_K,
    Q5_K,
    Q6_K,
    IQ4_XS,
    IQ3_XXS,
    IQ3_S,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    IQ4_NL,
    Q2_K,
    Q3_K,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ1_S,
    IQ1_M,
};

// Per ggml block in each ggml_planes_v1 plane (QuantLayout::GgmlPlanes): its values along K and
// its code, high and scale bytes. Zero for a format not in that layout.
struct GgmlPlaneBytes {
    std::uint32_t values, code, high, scale;
};

// Fields are in the order the rows below initialize them.
struct FormatFacts {
    QType q = QType::BF16;
    // The artifact's spelling of the format (tensor_format in an artifact's manifest).
    std::string_view name;
    // The route table's and the build manifest's shorter spelling; empty for a format the generic
    // path never routes (FP32, INT32).
    std::string_view route;
    // The GGUF type id the converter keeps the format from (ggml_type), -1 for the engine's own.
    std::int16_t ggml_type = -1;
    // The generic path's codec; none for a format it does not multiply.
    std::optional<WeightCodec> codec;
    // ggml's MMQ loader for the block format; none where ggml has no int8 matrix kernel for it.
    std::optional<MmqType> mmq;
    // Every code of the format is an int8 times a scale (and a min) per 32 or 16 values, or a
    // signed grid value, or an E2M1 code that widens to int8 exactly: a site in this format may
    // compute on int8 activations (the converter's kept_block_policy AllowA8).
    bool int8_input = false;
    // Values along K per ggml block; 0 for a format that is not a ggml block format.
    std::uint16_t block_values = 0;

    // Code, high and scale bytes of one block in each plane of ggml_planes_v1; zero for the blocks
    // kept whole (Q4_K, Q5_K: ggml_blocks_v1) and for every other format.
    struct PlaneBytes {
        std::uint8_t code = 0, high = 0, scale = 0;
    };

    PlaneBytes planes;
    // A 256-value ggml superblock whose sub-blocks carry their own scales (and, for some, minima):
    // the K-quants and the codebook i-quants. A decoder of these unpacks scales and codes per
    // sub-block before the multiply, which is what keeps its tile's registers and its issue slots
    // busy (the register MMA's cost model prices it by them). TQ1_0 and TQ2_0 are 256-value blocks
    // with one scale for the whole block, so they are plain; so are the 32-value formats and
    // Q1_0 / Q2_0.
    bool subblock_scales = false;
    // The K values one zero point and scale of a u4 zero-point format cover; 0 otherwise.
    std::uint16_t u4_zp_group = 0;
    // The K values one scale of a 3-bit groupwise format covers; 0 otherwise.
    std::uint16_t q3_group = 0;
};

namespace format_fact_rows {
using Q = QType;
using C = WeightCodec;
using M = MmqType;
// Rows in QType order. Every named QType has exactly one row (checked below).
inline constexpr std::array kFormats{
    FormatFacts{.q          = Q::Q4_G64_FP16,
                .name       = "q4_g64_fp16",
                .route      = "q4",
                .codec      = C::Q4,
                .int8_input = true},
    FormatFacts{.q          = Q::Q5_G64_FP16,
                .name       = "q5_g64_fp16",
                .route      = "q5",
                .codec      = C::Q5,
                .int8_input = true},
    FormatFacts{.q          = Q::Q6_G64_FP16,
                .name       = "q6_g64_fp16",
                .route      = "q6",
                .codec      = C::Q6,
                .int8_input = true},
    // ggml's Q8_0 is kept as this format (the same words).
    FormatFacts{.q          = Q::Q8_G32_FP16,
                .name       = "q8_g32_fp16",
                .route      = "q8",
                .ggml_type  = 8,
                .codec      = C::Q8,
                .int8_input = true},
    // ggml's F16 lands here too (format_of_ggml); it is not a QType of its own.
    FormatFacts{.q = Q::BF16, .name = "bf16", .route = "bf16", .ggml_type = 30, .codec = C::Bf16},
    FormatFacts{.q = Q::FP32, .name = "fp32", .ggml_type = 0},
    FormatFacts{.q = Q::INT32, .name = "int32", .ggml_type = 26},
    FormatFacts{
        .q = Q::NVFP4, .name = "nvfp4", .route = "nvfp4", .ggml_type = 40, .codec = C::Nvfp4},
    FormatFacts{.q     = Q::FP8_E4M3FN_ROW_BF16,
                .name  = "fp8_e4m3fn_row_bf16",
                .route = "fp8",
                .codec = C::Fp8},
    // E2M1 codes widen to int8 exactly, the UE8M0 block scale to a float (gpt-oss experts).
    FormatFacts{.q          = Q::MXFP4,
                .name       = "mxfp4",
                .route      = "mxfp4",
                .ggml_type  = 39,
                .codec      = C::Mxfp4,
                .int8_input = true},
    FormatFacts{.q     = Q::FP8_E4M3FN_ROW_F32,
                .name  = "fp8_e4m3fn_row_f32",
                .route = "fp8r32",
                .codec = C::Fp8RowF32},
    FormatFacts{.q     = Q::FP8_E4M3FN_B128_F32,
                .name  = "fp8_e4m3fn_b128_f32",
                .route = "fp8b128",
                .codec = C::Fp8B128},
    FormatFacts{.q           = Q::U4_G128_ZP_FP16,
                .name        = "u4_g128_zp_fp16",
                .route       = "u4zp",
                .codec       = C::U4Zp,
                .int8_input  = true,
                .u4_zp_group = 128},
    FormatFacts{.q               = Q::GGML_Q4_K,
                .name            = "ggml_q4_k",
                .route           = "q4k",
                .ggml_type       = 12,
                .codec           = C::GgmlQ4K,
                .mmq             = M::Q4_K,
                .int8_input      = true,
                .block_values    = 256,
                .subblock_scales = true},
    FormatFacts{.q               = Q::GGML_Q5_K,
                .name            = "ggml_q5_k",
                .route           = "q5k",
                .ggml_type       = 13,
                .codec           = C::GgmlQ5K,
                .mmq             = M::Q5_K,
                .int8_input      = true,
                .block_values    = 256,
                .subblock_scales = true},
    FormatFacts{.q               = Q::GGML_Q6_K,
                .name            = "ggml_q6_k",
                .route           = "q6k",
                .ggml_type       = 14,
                .codec           = C::GgmlQ6K,
                .mmq             = M::Q6_K,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {192, 16, 2},
                .subblock_scales = true},
    FormatFacts{.q               = Q::GGML_Q3_K,
                .name            = "ggml_q3_k",
                .route           = "q3k",
                .ggml_type       = 11,
                .codec           = C::GgmlQ3K,
                .mmq             = M::Q3_K,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {96, 12, 2},
                .subblock_scales = true},
    FormatFacts{.q               = Q::GGML_Q2_K,
                .name            = "ggml_q2_k",
                .route           = "q2k",
                .ggml_type       = 10,
                .codec           = C::GgmlQ2K,
                .mmq             = M::Q2_K,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {64, 16, 4},
                .subblock_scales = true},
    FormatFacts{.q               = Q::GGML_IQ4_XS,
                .name            = "ggml_iq4_xs",
                .route           = "iq4xs",
                .ggml_type       = 23,
                .codec           = C::GgmlIq4Xs,
                .mmq             = M::IQ4_XS,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {128, 4, 4},
                .subblock_scales = true},
    FormatFacts{.q               = Q::GGML_IQ3_XXS,
                .name            = "ggml_iq3_xxs",
                .route           = "iq3xxs",
                .ggml_type       = 18,
                .codec           = C::GgmlIq3Xxs,
                .mmq             = M::IQ3_XXS,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {64, 32, 2},
                .subblock_scales = true},
    FormatFacts{.q               = Q::GGML_IQ3_S,
                .name            = "ggml_iq3_s",
                .route           = "iq3s",
                .ggml_type       = 21,
                .codec           = C::GgmlIq3S,
                .mmq             = M::IQ3_S,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {104, 4, 2},
                .subblock_scales = true},
    FormatFacts{.q            = Q::GGML_Q4_0,
                .name         = "ggml_q4_0",
                .route        = "q40",
                .ggml_type    = 2,
                .codec        = C::GgmlQ40,
                .mmq          = M::Q4_0,
                .int8_input   = true,
                .block_values = 32,
                .planes       = {16, 0, 2}},
    FormatFacts{.q            = Q::GGML_Q4_1,
                .name         = "ggml_q4_1",
                .route        = "q41",
                .ggml_type    = 3,
                .codec        = C::GgmlQ41,
                .mmq          = M::Q4_1,
                .int8_input   = true,
                .block_values = 32,
                .planes       = {16, 0, 4}},
    FormatFacts{.q            = Q::GGML_Q5_0,
                .name         = "ggml_q5_0",
                .route        = "q50",
                .ggml_type    = 6,
                .codec        = C::GgmlQ50,
                .mmq          = M::Q5_0,
                .int8_input   = true,
                .block_values = 32,
                .planes       = {16, 4, 2}},
    FormatFacts{.q            = Q::GGML_Q5_1,
                .name         = "ggml_q5_1",
                .route        = "q51",
                .ggml_type    = 7,
                .codec        = C::GgmlQ51,
                .mmq          = M::Q5_1,
                .int8_input   = true,
                .block_values = 32,
                .planes       = {16, 4, 4}},
    FormatFacts{.q            = Q::GGML_IQ4_NL,
                .name         = "ggml_iq4_nl",
                .route        = "iq4nl",
                .ggml_type    = 20,
                .codec        = C::GgmlIq4Nl,
                .mmq          = M::IQ4_NL,
                .int8_input   = true,
                .block_values = 32,
                .planes       = {16, 0, 2}},
    FormatFacts{.q               = Q::GGML_IQ2_XXS,
                .name            = "ggml_iq2_xxs",
                .route           = "iq2xxs",
                .ggml_type       = 16,
                .codec           = C::GgmlIq2Xxs,
                .mmq             = M::IQ2_XXS,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {64, 0, 2},
                .subblock_scales = true},
    FormatFacts{.q               = Q::GGML_IQ2_XS,
                .name            = "ggml_iq2_xs",
                .route           = "iq2xs",
                .ggml_type       = 17,
                .codec           = C::GgmlIq2Xs,
                .mmq             = M::IQ2_XS,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {64, 8, 2},
                .subblock_scales = true},
    FormatFacts{.q               = Q::GGML_IQ2_S,
                .name            = "ggml_iq2_s",
                .route           = "iq2s",
                .ggml_type       = 22,
                .codec           = C::GgmlIq2S,
                .mmq             = M::IQ2_S,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {72, 8, 2},
                .subblock_scales = true},
    // IQ1_S / IQ1_M: grid -1, 0, 1 plus a delta of +-1/8 is 8g +- 1 in int8, exactly.
    FormatFacts{.q               = Q::GGML_IQ1_S,
                .name            = "ggml_iq1_s",
                .route           = "iq1s",
                .ggml_type       = 19,
                .codec           = C::GgmlIq1S,
                .mmq             = M::IQ1_S,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {32, 16, 2},
                .subblock_scales = true},
    FormatFacts{.q               = Q::GGML_IQ1_M,
                .name            = "ggml_iq1_m",
                .route           = "iq1m",
                .ggml_type       = 29,
                .codec           = C::GgmlIq1M,
                .mmq             = M::IQ1_M,
                .int8_input      = true,
                .block_values    = 256,
                .planes          = {32, 16, 8},
                .subblock_scales = true},
    // The ternary and two-bit blocks keep BF16 activations: no int8 kernel reads them.
    FormatFacts{.q            = Q::GGML_TQ1_0,
                .name         = "ggml_tq1_0",
                .route        = "tq10",
                .ggml_type    = 34,
                .codec        = C::GgmlTq10,
                .block_values = 256,
                .planes       = {52, 0, 2}},
    FormatFacts{.q            = Q::GGML_TQ2_0,
                .name         = "ggml_tq2_0",
                .route        = "tq20",
                .ggml_type    = 35,
                .codec        = C::GgmlTq20,
                .block_values = 256,
                .planes       = {64, 0, 2}},
    FormatFacts{.q            = Q::GGML_Q1_0,
                .name         = "ggml_q1_0",
                .route        = "q10",
                .ggml_type    = 41,
                .codec        = C::GgmlQ10,
                .block_values = 128,
                .planes       = {16, 0, 2}},
    FormatFacts{.q            = Q::GGML_Q2_0,
                .name         = "ggml_q2_0",
                .route        = "q20",
                .ggml_type    = 42,
                .codec        = C::GgmlQ20,
                .block_values = 64,
                .planes       = {16, 0, 2}},
    // u4 G32 / G64 read on the same int8 codec as G128 (W4a8U4ZpG): the same activation policy.
    FormatFacts{.q           = Q::U4_G32_ZP_FP16,
                .name        = "u4_g32_zp_fp16",
                .route       = "u4zp32",
                .codec       = C::U4ZpG32,
                .int8_input  = true,
                .u4_zp_group = 32},
    FormatFacts{.q           = Q::U4_G64_ZP_FP16,
                .name        = "u4_g64_zp_fp16",
                .route       = "u4zp64",
                .codec       = C::U4ZpG64,
                .int8_input  = true,
                .u4_zp_group = 64},
    FormatFacts{.q          = Q::Q3_G128_FP16,
                .name       = "q3_g128_fp16",
                .route      = "q3g128",
                .codec      = C::Q3G128,
                .int8_input = true,
                .q3_group   = 128},
    FormatFacts{.q          = Q::Q3_G64_FP16,
                .name       = "q3_g64_fp16",
                .route      = "q3g64",
                .codec      = C::Q3G64,
                .int8_input = true,
                .q3_group   = 64},
};
} // namespace format_fact_rows

inline constexpr const auto& kFormatFacts = format_fact_rows::kFormats;
// The row for a value the table does not name. Its `.q` is a placeholder, never the queried value:
// tell it apart by `.name.empty()`, not by `.q`.
inline constexpr FormatFacts kNoFormatFacts{};

// The format's row; the default row for a value the table does not name.
constexpr const FormatFacts& format_facts(QType q) noexcept {
    for (const FormatFacts& f : kFormatFacts) {
        if (f.q == q) return f;
    }
    return kNoFormatFacts;
}

// The format an artifact spells `name`, if any.
constexpr std::optional<QType> find_format(std::string_view name) noexcept {
    for (const FormatFacts& f : kFormatFacts) {
        if (f.name == name) return f.q;
    }
    return std::nullopt;
}

// The format the converter keeps a GGUF type as, if it keeps it unchanged (F16 is not here: it is
// widened to BF16 by the converter, see format_of_ggml).
constexpr std::optional<QType> ggml_format(std::uint32_t ggml_type) noexcept {
    for (const FormatFacts& f : kFormatFacts) {
        if (f.ggml_type >= 0 && static_cast<std::uint32_t>(f.ggml_type) == ggml_type) return f.q;
    }
    return std::nullopt;
}

// ggml's block formats the engine runs as published: the K-quants, the i-quants, the 32-value
// formats, the ternary TQ1_0 / TQ2_0 and Q1_0 / Q2_0.
constexpr bool is_ggml_block(QType q) noexcept { return format_facts(q).block_values != 0; }

// Values along K per block of a ggml block format (256 for GgmlBlocks' Q4_K / Q5_K).
constexpr std::uint32_t ggml_block_values(QType q) noexcept { return format_facts(q).block_values; }

// Block bytes per plane of ggml_planes_v1; zero for a format not in that layout.
constexpr GgmlPlaneBytes ggml_plane_bytes(QType q) noexcept {
    const FormatFacts& f = format_facts(q);
    if (f.planes.code == 0) return {0, 0, 0, 0};
    return {f.block_values, f.planes.code, f.planes.high, f.planes.scale};
}

constexpr bool has_subblock_scales(QType q) noexcept { return format_facts(q).subblock_scales; }

constexpr std::uint32_t u4_zp_group(QType q) noexcept { return format_facts(q).u4_zp_group; }

constexpr std::uint32_t q3_group(QType q) noexcept { return format_facts(q).q3_group; }

namespace format_fact_checks {

constexpr int rows_of(QType q) noexcept {
    int n = 0;
    for (const FormatFacts& f : kFormatFacts) n += f.q == q ? 1 : 0;
    return n;
}

// No two rows share a spelling, a GGUF id, a codec or an MMQ loader (the maps invert), and every
// artifact spelling parses back to its row.
constexpr bool rows_distinct() noexcept {
    for (std::size_t i = 0; i < kFormatFacts.size(); ++i) {
        const FormatFacts& a = kFormatFacts[i];
        if (a.name.empty() || find_format(a.name) != a.q) return false;
        for (std::size_t j = i + 1; j < kFormatFacts.size(); ++j) {
            const FormatFacts& b = kFormatFacts[j];
            if (a.name == b.name) return false;
            if (!a.route.empty() && a.route == b.route) return false;
            if (a.ggml_type >= 0 && a.ggml_type == b.ggml_type) return false;
            if (a.codec && a.codec == b.codec) return false;
            if (a.mmq && a.mmq == b.mmq) return false;
        }
    }
    return true;
}

// The geometry is of one piece: planes only on a ggml block format, an MMQ loader or an int8
// decoder only where a codec reads the format, and sub-block scales exactly on the 256-value ggml
// blocks that are not the ternary pair.
constexpr bool rows_consistent() noexcept {
    for (const FormatFacts& f : kFormatFacts) {
        if (f.planes.code != 0 && f.block_values == 0) return false;
        if (f.mmq && (!f.codec || f.block_values == 0)) return false;
        if (f.int8_input && !f.codec) return false;
        const bool formula =
            f.block_values == 256 && f.q != QType::GGML_TQ1_0 && f.q != QType::GGML_TQ2_0;
        if (f.subblock_scales != formula) return false;
    }
    return true;
}

static_assert(rows_distinct(), "format_facts.h: spellings, GGUF ids, codecs and MMQ loaders must "
                               "each name one row");
static_assert(rows_consistent(), "format_facts.h: a row's geometry fields disagree");

} // namespace format_fact_checks

// Whether the enum names the value (a compiler prints an unnamed enumerator as a cast): the rows
// are checked against the enum itself, not against a second hand-written list of it. Host
// compilers only; the device compiler's front end sees the same table in every host unit.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
template <QType Q>
constexpr bool named_qtype() noexcept {
    constexpr std::string_view where = __PRETTY_FUNCTION__;
    const std::size_t at             = where.find("Q = ");
    return at != std::string_view::npos && where[at + 4] != '(';
}

namespace format_fact_checks {

template <std::size_t... V>
constexpr bool every_named_qtype_has_one_row(std::index_sequence<V...>) noexcept {
    const bool once =
        ((!named_qtype<static_cast<QType>(V)>() || rows_of(static_cast<QType>(V)) == 1) && ...);
    const int named = ((named_qtype<static_cast<QType>(V)>() ? 1 : 0) + ... + 0);
    return once && static_cast<std::size_t>(named) == kFormatFacts.size();
}

// 0..255 covers the enum (its largest value is 43); a value past it would need this raised.
static_assert(
    every_named_qtype_has_one_row(std::make_index_sequence<256>{}),
    "format_facts.h: every named QType needs exactly one row, and every row a named QType");

} // namespace format_fact_checks
#endif

} // namespace ninferno
