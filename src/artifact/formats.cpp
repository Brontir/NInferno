#include "artifact/formats.h"

#include "artifact/schema.h"

#include <array>

namespace ninferno::artifact {
namespace {

constexpr std::array kFormats = {
    std::pair{QType::BF16, std::string_view{"bf16"}},
    std::pair{QType::FP32, std::string_view{"fp32"}},
    std::pair{QType::INT32, std::string_view{"int32"}},
    std::pair{QType::Q4_G64_FP16, std::string_view{"q4_g64_fp16"}},
    std::pair{QType::Q5_G64_FP16, std::string_view{"q5_g64_fp16"}},
    std::pair{QType::Q6_G64_FP16, std::string_view{"q6_g64_fp16"}},
    std::pair{QType::Q3_G128_FP16, std::string_view{"q3_g128_fp16"}},
    std::pair{QType::Q3_G64_FP16, std::string_view{"q3_g64_fp16"}},
    std::pair{QType::Q8_G32_FP16, std::string_view{"q8_g32_fp16"}},
    std::pair{QType::NVFP4, std::string_view{"nvfp4"}},
    std::pair{QType::FP8_E4M3FN_ROW_BF16, std::string_view{"fp8_e4m3fn_row_bf16"}},
    std::pair{QType::MXFP4, std::string_view{"mxfp4"}},
    std::pair{QType::FP8_E4M3FN_ROW_F32, std::string_view{"fp8_e4m3fn_row_f32"}},
    std::pair{QType::FP8_E4M3FN_B128_F32, std::string_view{"fp8_e4m3fn_b128_f32"}},
    std::pair{QType::U4_G128_ZP_FP16, std::string_view{"u4_g128_zp_fp16"}},
    std::pair{QType::U4_G32_ZP_FP16, std::string_view{"u4_g32_zp_fp16"}},
    std::pair{QType::U4_G64_ZP_FP16, std::string_view{"u4_g64_zp_fp16"}},
    std::pair{QType::GGML_Q4_K, std::string_view{"ggml_q4_k"}},
    std::pair{QType::GGML_Q5_K, std::string_view{"ggml_q5_k"}},
    std::pair{QType::GGML_Q6_K, std::string_view{"ggml_q6_k"}},
    std::pair{QType::GGML_Q3_K, std::string_view{"ggml_q3_k"}},
    std::pair{QType::GGML_Q2_K, std::string_view{"ggml_q2_k"}},
    std::pair{QType::GGML_IQ4_XS, std::string_view{"ggml_iq4_xs"}},
    std::pair{QType::GGML_IQ3_XXS, std::string_view{"ggml_iq3_xxs"}},
    std::pair{QType::GGML_IQ3_S, std::string_view{"ggml_iq3_s"}},
    std::pair{QType::GGML_Q4_0, std::string_view{"ggml_q4_0"}},
    std::pair{QType::GGML_Q4_1, std::string_view{"ggml_q4_1"}},
    std::pair{QType::GGML_Q5_0, std::string_view{"ggml_q5_0"}},
    std::pair{QType::GGML_Q5_1, std::string_view{"ggml_q5_1"}},
    std::pair{QType::GGML_IQ4_NL, std::string_view{"ggml_iq4_nl"}},
    std::pair{QType::GGML_IQ2_XXS, std::string_view{"ggml_iq2_xxs"}},
    std::pair{QType::GGML_IQ2_XS, std::string_view{"ggml_iq2_xs"}},
    std::pair{QType::GGML_IQ2_S, std::string_view{"ggml_iq2_s"}},
    std::pair{QType::GGML_IQ1_S, std::string_view{"ggml_iq1_s"}},
    std::pair{QType::GGML_IQ1_M, std::string_view{"ggml_iq1_m"}},
    std::pair{QType::GGML_TQ1_0, std::string_view{"ggml_tq1_0"}},
    std::pair{QType::GGML_TQ2_0, std::string_view{"ggml_tq2_0"}},
    std::pair{QType::GGML_Q1_0, std::string_view{"ggml_q1_0"}},
    std::pair{QType::GGML_Q2_0, std::string_view{"ggml_q2_0"}},
};
constexpr std::array kLayouts = {
    std::pair{QuantLayout::Contiguous, std::string_view{"contiguous_le_v1"}},
    std::pair{QuantLayout::RowSplit, std::string_view{"row_split_k128_v1"}},
    std::pair{QuantLayout::RowScale, std::string_view{"row_scale_v1"}},
    std::pair{QuantLayout::BlockScaleK16M128x4, std::string_view{"block_scale_k16_m128x4_v1"}},
    std::pair{QuantLayout::BlockScaleK32E8M0, std::string_view{"block_scale_k32_e8m0_v1"}},
    std::pair{QuantLayout::BlockScale128F32, std::string_view{"block_scale_128x128_f32_v1"}},
    std::pair{QuantLayout::RowGroupZp, std::string_view{"row_group_zp_v1"}},
    std::pair{QuantLayout::RowGroupZpOrdered, std::string_view{"row_group_zp_ordered_v1"}},
    std::pair{QuantLayout::GgmlBlocks, std::string_view{"ggml_blocks_v1"}},
    std::pair{QuantLayout::GgmlPlanes, std::string_view{"ggml_planes_v1"}},
    std::pair{QuantLayout::GgmlFrag, std::string_view{"ggml_frag_v1"}},
};

} // namespace

QType parse_format(std::string_view name) {
    for (const auto& [format, spelling] : kFormats) {
        if (spelling == name) { return format; }
    }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

QuantLayout parse_layout(std::string_view name) {
    for (const auto& [layout, spelling] : kLayouts) {
        if (spelling == name) { return layout; }
    }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

std::string_view format_name(QType format) noexcept {
    for (const auto& [value, spelling] : kFormats) {
        if (value == format) { return spelling; }
    }
    return {};
}

std::string_view layout_name(QuantLayout layout) noexcept {
    for (const auto& [value, spelling] : kLayouts) {
        if (value == layout) { return spelling; }
    }
    return {};
}

} // namespace ninferno::artifact
