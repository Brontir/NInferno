// The format facts table (core/format_facts.h) against the switches and spelling tables it
// replaced (#615 F2): for every QType value in a byte's range, the old answer (kept below in
// `legacy`, verbatim but for comments) and the new read of the table agree, so the move changed no
// value. That every named QType has exactly one row, and the rows invert, is a static_assert in the
// header. No GPU.
#include "core/format_facts.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

using ninferno::GgmlPlaneBytes;
using ninferno::MmqType;
using ninferno::QType;
using ninferno::WeightCodec;

namespace legacy {

// generic_gemm.cpp codec_of (the throw of its default is a nullopt here).
std::optional<WeightCodec> codec_of(QType q) {
    switch (q) {
    case QType::BF16:
        return WeightCodec::Bf16;
    case QType::FP8_E4M3FN_ROW_BF16:
        return WeightCodec::Fp8;
    case QType::NVFP4:
        return WeightCodec::Nvfp4;
    case QType::Q4_G64_FP16:
        return WeightCodec::Q4;
    case QType::Q5_G64_FP16:
        return WeightCodec::Q5;
    case QType::Q6_G64_FP16:
        return WeightCodec::Q6;
    case QType::Q8_G32_FP16:
        return WeightCodec::Q8;
    case QType::MXFP4:
        return WeightCodec::Mxfp4;
    case QType::FP8_E4M3FN_ROW_F32:
        return WeightCodec::Fp8RowF32;
    case QType::FP8_E4M3FN_B128_F32:
        return WeightCodec::Fp8B128;
    case QType::U4_G128_ZP_FP16:
        return WeightCodec::U4Zp;
    case QType::U4_G32_ZP_FP16:
        return WeightCodec::U4ZpG32;
    case QType::U4_G64_ZP_FP16:
        return WeightCodec::U4ZpG64;
    case QType::Q3_G128_FP16:
        return WeightCodec::Q3G128;
    case QType::Q3_G64_FP16:
        return WeightCodec::Q3G64;
    case QType::GGML_Q4_K:
        return WeightCodec::GgmlQ4K;
    case QType::GGML_Q5_K:
        return WeightCodec::GgmlQ5K;
    case QType::GGML_Q6_K:
        return WeightCodec::GgmlQ6K;
    case QType::GGML_Q3_K:
        return WeightCodec::GgmlQ3K;
    case QType::GGML_Q2_K:
        return WeightCodec::GgmlQ2K;
    case QType::GGML_IQ4_XS:
        return WeightCodec::GgmlIq4Xs;
    case QType::GGML_IQ3_XXS:
        return WeightCodec::GgmlIq3Xxs;
    case QType::GGML_IQ3_S:
        return WeightCodec::GgmlIq3S;
    case QType::GGML_Q4_0:
        return WeightCodec::GgmlQ40;
    case QType::GGML_Q4_1:
        return WeightCodec::GgmlQ41;
    case QType::GGML_Q5_0:
        return WeightCodec::GgmlQ50;
    case QType::GGML_Q5_1:
        return WeightCodec::GgmlQ51;
    case QType::GGML_IQ4_NL:
        return WeightCodec::GgmlIq4Nl;
    case QType::GGML_IQ2_XXS:
        return WeightCodec::GgmlIq2Xxs;
    case QType::GGML_IQ2_XS:
        return WeightCodec::GgmlIq2Xs;
    case QType::GGML_IQ2_S:
        return WeightCodec::GgmlIq2S;
    case QType::GGML_IQ1_S:
        return WeightCodec::GgmlIq1S;
    case QType::GGML_IQ1_M:
        return WeightCodec::GgmlIq1M;
    case QType::GGML_TQ1_0:
        return WeightCodec::GgmlTq10;
    case QType::GGML_TQ2_0:
        return WeightCodec::GgmlTq20;
    case QType::GGML_Q1_0:
        return WeightCodec::GgmlQ10;
    case QType::GGML_Q2_0:
        return WeightCodec::GgmlQ20;
    default:
        return std::nullopt;
    }
}

// generic_gemm.cpp mmq_type.
std::optional<MmqType> mmq_type(QType q) {
    switch (q) {
    case QType::GGML_Q4_K:
        return MmqType::Q4_K;
    case QType::GGML_Q5_K:
        return MmqType::Q5_K;
    case QType::GGML_Q6_K:
        return MmqType::Q6_K;
    case QType::GGML_IQ4_XS:
        return MmqType::IQ4_XS;
    case QType::GGML_IQ3_XXS:
        return MmqType::IQ3_XXS;
    case QType::GGML_IQ3_S:
        return MmqType::IQ3_S;
    case QType::GGML_Q4_0:
        return MmqType::Q4_0;
    case QType::GGML_Q4_1:
        return MmqType::Q4_1;
    case QType::GGML_Q5_0:
        return MmqType::Q5_0;
    case QType::GGML_Q5_1:
        return MmqType::Q5_1;
    case QType::GGML_IQ4_NL:
        return MmqType::IQ4_NL;
    case QType::GGML_Q2_K:
        return MmqType::Q2_K;
    case QType::GGML_Q3_K:
        return MmqType::Q3_K;
    case QType::GGML_IQ2_XXS:
        return MmqType::IQ2_XXS;
    case QType::GGML_IQ2_XS:
        return MmqType::IQ2_XS;
    case QType::GGML_IQ2_S:
        return MmqType::IQ2_S;
    case QType::GGML_IQ1_S:
        return MmqType::IQ1_S;
    case QType::GGML_IQ1_M:
        return MmqType::IQ1_M;
    default:
        return std::nullopt;
    }
}

// artifact/formats.cpp kFormats.
constexpr std::pair<QType, std::string_view> kArtifactNames[] = {
    {QType::BF16, "bf16"},
    {QType::FP32, "fp32"},
    {QType::INT32, "int32"},
    {QType::Q4_G64_FP16, "q4_g64_fp16"},
    {QType::Q5_G64_FP16, "q5_g64_fp16"},
    {QType::Q6_G64_FP16, "q6_g64_fp16"},
    {QType::Q3_G128_FP16, "q3_g128_fp16"},
    {QType::Q3_G64_FP16, "q3_g64_fp16"},
    {QType::Q8_G32_FP16, "q8_g32_fp16"},
    {QType::NVFP4, "nvfp4"},
    {QType::FP8_E4M3FN_ROW_BF16, "fp8_e4m3fn_row_bf16"},
    {QType::MXFP4, "mxfp4"},
    {QType::FP8_E4M3FN_ROW_F32, "fp8_e4m3fn_row_f32"},
    {QType::FP8_E4M3FN_B128_F32, "fp8_e4m3fn_b128_f32"},
    {QType::U4_G128_ZP_FP16, "u4_g128_zp_fp16"},
    {QType::U4_G32_ZP_FP16, "u4_g32_zp_fp16"},
    {QType::U4_G64_ZP_FP16, "u4_g64_zp_fp16"},
    {QType::GGML_Q4_K, "ggml_q4_k"},
    {QType::GGML_Q5_K, "ggml_q5_k"},
    {QType::GGML_Q6_K, "ggml_q6_k"},
    {QType::GGML_Q3_K, "ggml_q3_k"},
    {QType::GGML_Q2_K, "ggml_q2_k"},
    {QType::GGML_IQ4_XS, "ggml_iq4_xs"},
    {QType::GGML_IQ3_XXS, "ggml_iq3_xxs"},
    {QType::GGML_IQ3_S, "ggml_iq3_s"},
    {QType::GGML_Q4_0, "ggml_q4_0"},
    {QType::GGML_Q4_1, "ggml_q4_1"},
    {QType::GGML_Q5_0, "ggml_q5_0"},
    {QType::GGML_Q5_1, "ggml_q5_1"},
    {QType::GGML_IQ4_NL, "ggml_iq4_nl"},
    {QType::GGML_IQ2_XXS, "ggml_iq2_xxs"},
    {QType::GGML_IQ2_XS, "ggml_iq2_xs"},
    {QType::GGML_IQ2_S, "ggml_iq2_s"},
    {QType::GGML_IQ1_S, "ggml_iq1_s"},
    {QType::GGML_IQ1_M, "ggml_iq1_m"},
    {QType::GGML_TQ1_0, "ggml_tq1_0"},
    {QType::GGML_TQ2_0, "ggml_tq2_0"},
    {QType::GGML_Q1_0, "ggml_q1_0"},
    {QType::GGML_Q2_0, "ggml_q2_0"},
};

std::string_view format_name(QType q) {
    for (const auto& [value, spelling] : kArtifactNames) {
        if (value == q) return spelling;
    }
    return {};
}

std::optional<QType> parse_format(std::string_view name) {
    for (const auto& [value, spelling] : kArtifactNames) {
        if (spelling == name) return value;
    }
    return std::nullopt;
}

// route_table.cpp qtype_name.
const char* qtype_name(QType q) {
    switch (q) {
    case QType::BF16:
        return "bf16";
    case QType::FP8_E4M3FN_ROW_BF16:
        return "fp8";
    case QType::NVFP4:
        return "nvfp4";
    case QType::Q4_G64_FP16:
        return "q4";
    case QType::Q5_G64_FP16:
        return "q5";
    case QType::Q6_G64_FP16:
        return "q6";
    case QType::Q8_G32_FP16:
        return "q8";
    case QType::MXFP4:
        return "mxfp4";
    case QType::FP8_E4M3FN_ROW_F32:
        return "fp8r32";
    case QType::FP8_E4M3FN_B128_F32:
        return "fp8b128";
    case QType::U4_G128_ZP_FP16:
        return "u4zp";
    case QType::U4_G32_ZP_FP16:
        return "u4zp32";
    case QType::U4_G64_ZP_FP16:
        return "u4zp64";
    case QType::Q3_G128_FP16:
        return "q3g128";
    case QType::Q3_G64_FP16:
        return "q3g64";
    case QType::GGML_Q4_K:
        return "q4k";
    case QType::GGML_Q5_K:
        return "q5k";
    case QType::GGML_Q6_K:
        return "q6k";
    case QType::GGML_Q3_K:
        return "q3k";
    case QType::GGML_Q2_K:
        return "q2k";
    case QType::GGML_IQ4_XS:
        return "iq4xs";
    case QType::GGML_IQ3_XXS:
        return "iq3xxs";
    case QType::GGML_IQ3_S:
        return "iq3s";
    case QType::GGML_Q4_0:
        return "q40";
    case QType::GGML_Q4_1:
        return "q41";
    case QType::GGML_Q5_0:
        return "q50";
    case QType::GGML_Q5_1:
        return "q51";
    case QType::GGML_IQ4_NL:
        return "iq4nl";
    case QType::GGML_IQ2_XXS:
        return "iq2xxs";
    case QType::GGML_IQ2_XS:
        return "iq2xs";
    case QType::GGML_IQ2_S:
        return "iq2s";
    case QType::GGML_IQ1_S:
        return "iq1s";
    case QType::GGML_IQ1_M:
        return "iq1m";
    case QType::GGML_TQ1_0:
        return "tq10";
    case QType::GGML_TQ2_0:
        return "tq20";
    case QType::GGML_Q1_0:
        return "q10";
    case QType::GGML_Q2_0:
        return "q20";
    default:
        return "unknown";
    }
}

// convert/formats.cpp kept_block_policy == AllowA8.
bool int8_input(QType q) {
    switch (q) {
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ4_XS:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
    case QType::Q8_G32_FP16:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    case QType::MXFP4:
        return true;
    default:
        return false;
    }
}

// core/weight.h is_ggml_block, ggml_plane_bytes, ggml_block_values, has_subblock_scales,
// u4_zp_group, q3_group.
bool is_ggml_block(QType q) {
    const auto v = static_cast<std::uint16_t>(q);
    return (v >= static_cast<std::uint16_t>(QType::GGML_Q4_K) &&
            v <= static_cast<std::uint16_t>(QType::GGML_Q2_K)) ||
           (v >= static_cast<std::uint16_t>(QType::GGML_IQ4_XS) &&
            v <= static_cast<std::uint16_t>(QType::GGML_Q2_0));
}

GgmlPlaneBytes ggml_plane_bytes(QType q) {
    switch (q) {
    case QType::GGML_Q6_K:
        return {256, 192, 16, 2};
    case QType::GGML_Q3_K:
        return {256, 96, 12, 2};
    case QType::GGML_Q2_K:
        return {256, 64, 16, 4};
    case QType::GGML_IQ4_XS:
        return {256, 128, 4, 4};
    case QType::GGML_IQ3_XXS:
        return {256, 64, 32, 2};
    case QType::GGML_IQ3_S:
        return {256, 104, 4, 2};
    case QType::GGML_Q4_0:
        return {32, 16, 0, 2};
    case QType::GGML_Q4_1:
        return {32, 16, 0, 4};
    case QType::GGML_Q5_0:
        return {32, 16, 4, 2};
    case QType::GGML_Q5_1:
        return {32, 16, 4, 4};
    case QType::GGML_IQ4_NL:
        return {32, 16, 0, 2};
    case QType::GGML_IQ2_XXS:
        return {256, 64, 0, 2};
    case QType::GGML_IQ2_XS:
        return {256, 64, 8, 2};
    case QType::GGML_IQ2_S:
        return {256, 72, 8, 2};
    case QType::GGML_IQ1_S:
        return {256, 32, 16, 2};
    case QType::GGML_IQ1_M:
        return {256, 32, 16, 8};
    case QType::GGML_TQ1_0:
        return {256, 52, 0, 2};
    case QType::GGML_TQ2_0:
        return {256, 64, 0, 2};
    case QType::GGML_Q1_0:
        return {128, 16, 0, 2};
    case QType::GGML_Q2_0:
        return {64, 16, 0, 2};
    default:
        return {0, 0, 0, 0};
    }
}

// (Calls qualified: the enum's namespace would otherwise offer the new functions by ADL.)
std::uint32_t ggml_block_values(QType q) {
    return q == QType::GGML_Q4_K || q == QType::GGML_Q5_K ? 256
                                                          : legacy::ggml_plane_bytes(q).values;
}

bool has_subblock_scales(QType q) {
    return legacy::is_ggml_block(q) && legacy::ggml_block_values(q) == 256 &&
           q != QType::GGML_TQ1_0 && q != QType::GGML_TQ2_0;
}

std::uint32_t u4_zp_group(QType q) {
    switch (q) {
    case QType::U4_G32_ZP_FP16:
        return 32;
    case QType::U4_G64_ZP_FP16:
        return 64;
    case QType::U4_G128_ZP_FP16:
        return 128;
    default:
        return 0;
    }
}

std::uint32_t q3_group(QType q) {
    return q == QType::Q3_G128_FP16 ? 128 : q == QType::Q3_G64_FP16 ? 64 : 0;
}

// plan_facts.cpp format_of_ggml for every GGUF id the converter keeps unchanged (convert::ggml
// ids; F16 is widened to bf16 and is no row).
constexpr std::pair<std::uint32_t, std::string_view> kGgmlSpellings[] = {
    {0, "fp32"},         {30, "bf16"},         {8, "q8_g32_fp16"},  {39, "mxfp4"},
    {40, "nvfp4"},       {26, "int32"},        {2, "ggml_q4_0"},    {3, "ggml_q4_1"},
    {6, "ggml_q5_0"},    {7, "ggml_q5_1"},     {10, "ggml_q2_k"},   {11, "ggml_q3_k"},
    {12, "ggml_q4_k"},   {13, "ggml_q5_k"},    {14, "ggml_q6_k"},   {16, "ggml_iq2_xxs"},
    {17, "ggml_iq2_xs"}, {18, "ggml_iq3_xxs"}, {19, "ggml_iq1_s"},  {20, "ggml_iq4_nl"},
    {21, "ggml_iq3_s"},  {22, "ggml_iq2_s"},   {23, "ggml_iq4_xs"}, {29, "ggml_iq1_m"},
    {34, "ggml_tq1_0"},  {35, "ggml_tq2_0"},   {41, "ggml_q1_0"},   {42, "ggml_q2_0"},
};

std::optional<std::string_view> ggml_spelling(std::uint32_t type) {
    for (const auto& [id, spelling] : kGgmlSpellings) {
        if (id == type) return spelling;
    }
    return std::nullopt;
}

// route_table.cpp parse_qtype before the table.
std::optional<QType> parse_qtype(const std::string& s) {
    if (s == "bf16") return QType::BF16;
    if (s == "fp8") return QType::FP8_E4M3FN_ROW_BF16;
    if (s == "nvfp4") return QType::NVFP4;
    if (s == "q4") return QType::Q4_G64_FP16;
    if (s == "q5") return QType::Q5_G64_FP16;
    if (s == "q6") return QType::Q6_G64_FP16;
    if (s == "q8") return QType::Q8_G32_FP16;
    if (s == "mxfp4") return QType::MXFP4;
    if (s == "fp8r32") return QType::FP8_E4M3FN_ROW_F32;
    if (s == "fp8b128") return QType::FP8_E4M3FN_B128_F32;
    if (s == "u4zp") return QType::U4_G128_ZP_FP16;
    if (s == "u4zp32") return QType::U4_G32_ZP_FP16;
    if (s == "u4zp64") return QType::U4_G64_ZP_FP16;
    if (s == "q3g128") return QType::Q3_G128_FP16;
    if (s == "q3g64") return QType::Q3_G64_FP16;
    if (s == "q4k") return QType::GGML_Q4_K;
    if (s == "q5k") return QType::GGML_Q5_K;
    if (s == "q6k") return QType::GGML_Q6_K;
    if (s == "q3k") return QType::GGML_Q3_K;
    if (s == "q2k") return QType::GGML_Q2_K;
    if (s == "iq4xs") return QType::GGML_IQ4_XS;
    if (s == "iq3xxs") return QType::GGML_IQ3_XXS;
    if (s == "iq3s") return QType::GGML_IQ3_S;
    if (s == "q40") return QType::GGML_Q4_0;
    if (s == "q41") return QType::GGML_Q4_1;
    if (s == "q50") return QType::GGML_Q5_0;
    if (s == "q51") return QType::GGML_Q5_1;
    if (s == "iq4nl") return QType::GGML_IQ4_NL;
    if (s == "iq2xxs") return QType::GGML_IQ2_XXS;
    if (s == "iq2xs") return QType::GGML_IQ2_XS;
    if (s == "iq2s") return QType::GGML_IQ2_S;
    if (s == "iq1s") return QType::GGML_IQ1_S;
    if (s == "iq1m") return QType::GGML_IQ1_M;
    if (s == "tq10") return QType::GGML_TQ1_0;
    if (s == "tq20") return QType::GGML_TQ2_0;
    if (s == "q10") return QType::GGML_Q1_0;
    if (s == "q20") return QType::GGML_Q2_0;
    return std::nullopt;
}

// gguf_model.cpp's kept-block switch (ggml:: ids written as their numbers: ggml_quants.h is the
// converter's, the values are the GGUF type ids).
std::optional<QType> kept_block(std::uint32_t type) {
    std::optional<QType> kept;
    switch (type) {
    case 12:
        kept = QType::GGML_Q4_K;
        break;
    case 13:
        kept = QType::GGML_Q5_K;
        break;
    case 14:
        kept = QType::GGML_Q6_K;
        break;
    case 11:
        kept = QType::GGML_Q3_K;
        break;
    case 10:
        kept = QType::GGML_Q2_K;
        break;
    case 23:
        kept = QType::GGML_IQ4_XS;
        break;
    case 18:
        kept = QType::GGML_IQ3_XXS;
        break;
    case 21:
        kept = QType::GGML_IQ3_S;
        break;
    case 2:
        kept = QType::GGML_Q4_0;
        break;
    case 3:
        kept = QType::GGML_Q4_1;
        break;
    case 6:
        kept = QType::GGML_Q5_0;
        break;
    case 7:
        kept = QType::GGML_Q5_1;
        break;
    case 20:
        kept = QType::GGML_IQ4_NL;
        break;
    case 16:
        kept = QType::GGML_IQ2_XXS;
        break;
    case 17:
        kept = QType::GGML_IQ2_XS;
        break;
    case 22:
        kept = QType::GGML_IQ2_S;
        break;
    case 19:
        kept = QType::GGML_IQ1_S;
        break;
    case 29:
        kept = QType::GGML_IQ1_M;
        break;
    case 34:
        kept = QType::GGML_TQ1_0;
        break;
    case 35:
        kept = QType::GGML_TQ2_0;
        break;
    case 41:
        kept = QType::GGML_Q1_0;
        break;
    case 42:
        kept = QType::GGML_Q2_0;
        break;
    default:
        break;
    }
    return kept;
}

} // namespace legacy

namespace {

int failures    = 0;
int comparisons = 0;

template <class T>
void agree(const T& old_value, const T& new_value, const char* what, int v) {
    ++comparisons;
    if (old_value == new_value) return;
    ++failures;
    std::cerr << "format facts: " << what << " disagrees at QType " << v << "\n";
}

bool same(const GgmlPlaneBytes& a, const GgmlPlaneBytes& b) {
    return a.values == b.values && a.code == b.code && a.high == b.high && a.scale == b.scale;
}

} // namespace

int main() {
    using ninferno::format_facts;

    for (int v = 0; v < 256; ++v) {
        const QType q = static_cast<QType>(v);
        const auto& f = format_facts(q);
        agree(legacy::codec_of(q), f.codec, "codec", v);
        agree(legacy::mmq_type(q), f.mmq, "mmq type", v);
        agree(legacy::format_name(q), f.name, "artifact spelling", v);
        agree(std::string_view(legacy::qtype_name(q)),
              f.route.empty() ? std::string_view("unknown") : f.route, "route spelling", v);
        agree(legacy::int8_input(q), f.int8_input, "int8 input (kept_block_policy)", v);
        agree(legacy::is_ggml_block(q), ninferno::is_ggml_block(q), "is_ggml_block", v);
        agree(same(legacy::ggml_plane_bytes(q), ninferno::ggml_plane_bytes(q)), true,
              "ggml_plane_bytes", v);
        agree(legacy::ggml_block_values(q), ninferno::ggml_block_values(q), "ggml_block_values", v);
        agree(legacy::has_subblock_scales(q), ninferno::has_subblock_scales(q),
              "has_subblock_scales", v);
        agree(legacy::u4_zp_group(q), ninferno::u4_zp_group(q), "u4_zp_group", v);
        agree(legacy::q3_group(q), ninferno::q3_group(q), "q3_group", v);
        // Both spellings round-trip through the table as the old tables did.
        if (!f.name.empty())
            agree(legacy::parse_format(f.name), ninferno::find_format(f.name), "parse", v);
    }
    // No spelling the table does not know parses, as before.
    for (const char* bad : {"", "q4_k", "GGML_Q4_K", "ggml_q8_0", "unknown"}) {
        agree(legacy::parse_format(bad), ninferno::find_format(bad), bad, -1);
    }
    // Every GGUF id the converter keeps unchanged spells the same format; every other id has no
    // row.
    for (std::uint32_t id = 0; id < 64; ++id) {
        const auto old_spelling = legacy::ggml_spelling(id);
        const auto q            = ninferno::ggml_format(id);
        std::optional<std::string_view> spelling;
        if (q) spelling = format_facts(*q).name;
        agree(old_spelling, spelling, "format_of_ggml", static_cast<int>(id));
    }

    // The route spellings: the new loop over the rows (route_table.cpp parse_qtype) against the old
    // chain, for every spelling in the table and for junk; a format with no route spelling (FP32,
    // INT32) and the artifact spellings that are not route spellings must not parse.
    // route_table.cpp's parse_qtype is find_route; the test calls the shipped lookup.
    const auto parse_qtype_new = [](const std::string& str) { return ninferno::find_route(str); };
    static_assert(!ninferno::find_route("").has_value());
    for (const auto& row : ninferno::kFormatFacts) {
        if (row.route.empty()) {
            agree(parse_qtype_new(std::string(row.name)),
                  legacy::parse_qtype(std::string(row.name)), "unrouted artifact spelling parse",
                  static_cast<int>(row.q));
            continue;
        }
        const std::string spelling(row.route);
        agree(parse_qtype_new(spelling), std::optional<QType>(row.q), "route round trip",
              static_cast<int>(row.q));
        agree(parse_qtype_new(spelling), legacy::parse_qtype(spelling), "parse_qtype",
              static_cast<int>(row.q));
    }
    for (const char* junk : {"", "Q4", "q4 ", " q4", "q4k ", "Q4K", "ggml_q4_k", "fp32", "int32",
                             "bf16x", "b", "q", "q4kk", "unknown", "q8_g32_fp16"}) {
        agree(parse_qtype_new(junk), legacy::parse_qtype(junk), junk, -1);
    }
    agree(parse_qtype_new("fp32").has_value(), false, "fp32 must not parse", -1);
    agree(parse_qtype_new("int32").has_value(), false, "int32 must not parse", -1);

    // gguf_model.cpp kept-block mapping: the old 22-case switch against the table lookup
    // (ggml_format filtered by is_ggml_block), over ids past the last.
    for (std::uint32_t id = 0; id < 64; ++id) {
        agree(legacy::kept_block(id), ninferno::kept_ggml_block(id), "kept block",
              static_cast<int>(id));
    }

    if (failures != 0) {
        std::cerr << failures << " disagreements in " << comparisons << " comparisons\n";
        return 1;
    }
    std::cout << "format facts: " << comparisons << " comparisons agree\n";
    return 0;
}
