#include "artifact/formats.h"

#include "artifact/schema.h"
#include "core/format_facts.h"

#include <array>

namespace ninferno::artifact {
namespace {

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
    if (const auto format = find_format(name)) { return *format; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

QuantLayout parse_layout(std::string_view name) {
    for (const auto& [layout, spelling] : kLayouts) {
        if (spelling == name) { return layout; }
    }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

std::string_view format_name(QType format) noexcept { return format_facts(format).name; }

std::string_view layout_name(QuantLayout layout) noexcept {
    for (const auto& [value, spelling] : kLayouts) {
        if (value == layout) { return spelling; }
    }
    return {};
}

} // namespace ninferno::artifact
