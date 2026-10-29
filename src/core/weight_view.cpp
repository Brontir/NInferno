#include "core/weight_view.h"

#include "core/ggml_frag.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninferno {
namespace {

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("weight geometry addition overflows u64");
    }
    return a + b;
}

std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("weight geometry multiplication overflows u64");
    }
    return a * b;
}

std::uint64_t aligned(std::uint64_t n, std::uint64_t alignment) {
    return add(n, alignment - 1) / alignment * alignment;
}

std::int32_t dimension(std::uint64_t n) {
    if (!n || n > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("weight dimension is outside the native i32 domain");
    }
    return static_cast<std::int32_t>(n);
}

std::uint64_t word_bytes(QType format) {
    switch (format) {
    case QType::BF16:
        return 2;
    case QType::FP32:
    case QType::INT32:
        return 4;
    default:
        throw std::invalid_argument("direct weight requires BF16, FP32 or INT32");
    }
}

DType direct_dtype(QType format) {
    switch (format) {
    case QType::BF16:
        return DType::BF16;
    case QType::FP32:
        return DType::FP32;
    case QType::INT32:
        return DType::I32;
    default:
        throw std::invalid_argument("quantized weight cannot be bound as a direct tensor");
    }
}

void validate_region(const WeightRegion& region) {
    if (!region.parent || region.begin >= region.end ||
        region.end > region.parent->geometry.elements) {
        throw std::invalid_argument("weight region exceeds its parent element domain");
    }
}

} // namespace

std::uint64_t weight_element_count(std::span<const std::uint64_t> shape) {
    if (shape.size() > 16) { throw std::invalid_argument("weight rank exceeds 16"); }
    std::uint64_t count = 1;
    for (const auto dim : shape) {
        if (!dim) { throw std::invalid_argument("weight dimensions must be positive"); }
        count = mul(count, dim);
    }
    return count;
}

WeightGeometry weight_geometry(QType format, QuantLayout layout,
                               std::span<const std::uint64_t> shape, std::uint64_t divisors) {
    if (divisors == 0 || (divisors != 1 && format != QType::NVFP4)) {
        throw std::invalid_argument("only an NVFP4 plane carries more than one divisor");
    }
    WeightGeometry out;
    out.format        = format;
    out.divisor_count = divisors;
    out.layout        = layout;
    out.shape.assign(shape.begin(), shape.end());
    out.elements = weight_element_count(shape);
    if (layout == QuantLayout::Contiguous) {
        out.bytes = out.code_bytes = mul(out.elements, word_bytes(format));
        if (shape.size() == 2) {
            out.padded_columns     = shape[1];
            out.code_bytes_per_row = mul(shape[1], word_bytes(format));
        }
        return out;
    }
    if (shape.size() != 2) { throw std::invalid_argument("quantized weight must be a matrix"); }
    if (layout == QuantLayout::GgmlFrag) {
        // The published layout's geometry: only the order of each row's code bytes differs.
        if (!ggml_frag_format(format)) {
            throw std::invalid_argument("ggml_frag_v1 requires a format with a fragment-order map "
                                        "(ggml_q4_0, ggml_q4_k, ggml_iq3_xxs, ggml_iq2_s)");
        }
        auto out   = weight_geometry(format, ggml_published_layout(format), shape, divisors);
        out.layout = QuantLayout::GgmlFrag;
        return out;
    }
    const auto n       = shape[0];
    const auto k       = shape[1];
    out.padded_columns = k;
    if (layout == QuantLayout::RowSplit) {
        std::uint64_t high_per_group = 0;
        std::uint64_t code_per_group = 32;
        switch (format) {
        case QType::Q4_G64_FP16:
            out.group_size = 64;
            break;
        case QType::Q5_G64_FP16:
            out.group_size = 64;
            high_per_group = 8;
            break;
        case QType::Q6_G64_FP16:
            out.group_size = 64;
            high_per_group = 16;
            break;
        case QType::Q8_G32_FP16:
            out.group_size = 32;
            break;
        case QType::Q3_G128_FP16:
            out.group_size = 128;
            code_per_group = 48;
            break;
        case QType::Q3_G64_FP16:
            out.group_size = 64;
            code_per_group = 24;
            break;
        default:
            throw std::invalid_argument("RowSplit requires a grouped integer format");
        }
        out.padded_columns      = aligned(k, 128);
        const auto groups       = out.padded_columns / out.group_size;
        out.code_bytes_per_row  = mul(groups, code_per_group);
        out.high_bytes_per_row  = mul(groups, high_per_group);
        out.scale_bytes_per_row = mul(groups, 2);
        out.code_bytes          = mul(n, out.code_bytes_per_row);
        out.high_offset         = aligned(out.code_bytes, 256);
        out.high_bytes          = mul(n, out.high_bytes_per_row);
        out.scale_offset        = add(out.high_offset, aligned(out.high_bytes, 256));
    } else if (layout == QuantLayout::RowScale) {
        if (format != QType::FP8_E4M3FN_ROW_BF16 && format != QType::FP8_E4M3FN_ROW_F32) {
            throw std::invalid_argument("RowScale requires row-scaled FP8");
        }
        out.group_size          = k;
        out.code_bytes_per_row  = k;
        out.scale_bytes_per_row = format == QType::FP8_E4M3FN_ROW_F32 ? 4 : 2;
        out.code_bytes          = out.elements;
        out.scale_offset        = aligned(out.code_bytes, 256);
    } else if (layout == QuantLayout::RowGroupZp || layout == QuantLayout::RowGroupZpOrdered) {
        const auto group = u4_zp_group(format);
        if (!group || k % group) {
            throw std::invalid_argument(
                "RowGroupZp requires a u4 zero-point format and K in whole groups");
        }
        out.group_size          = group;
        out.code_bytes_per_row  = k / 2;
        out.high_bytes_per_row  = k / group;
        out.scale_bytes_per_row = k / group * 2;
        out.code_bytes          = mul(n, out.code_bytes_per_row);
        out.high_offset         = aligned(out.code_bytes, 256);
        out.high_bytes          = mul(n, out.high_bytes_per_row);
        out.scale_offset        = add(out.high_offset, aligned(out.high_bytes, 256));
    } else if (layout == QuantLayout::GgmlBlocks) {
        if ((format != QType::GGML_Q4_K && format != QType::GGML_Q5_K) || k % 256) {
            throw std::invalid_argument("GgmlBlocks requires ggml_q4_k or ggml_q5_k and K%256=0");
        }
        out.group_size         = 256;
        out.code_bytes_per_row = k / 256 * (format == QType::GGML_Q4_K ? 144 : 176);
        out.code_bytes         = mul(n, out.code_bytes_per_row);
        out.scale_offset       = out.code_bytes;
        out.bytes              = out.code_bytes;
        return out;
    } else if (layout == QuantLayout::GgmlPlanes) {
        const auto plane = ggml_plane_bytes(format);
        if (plane.values == 0 || k % plane.values) {
            throw std::invalid_argument("GgmlPlanes requires a ggml planes format (Q6_K, Q3_K, "
                                        "Q2_K, the i-quants, Q4_0 .. Q5_1, IQ4_NL, TQ1_0, TQ2_0, "
                                        "Q1_0, Q2_0) and K a whole number of its blocks");
        }
        const auto blocks       = k / plane.values;
        out.group_size          = plane.values;
        out.code_bytes_per_row  = blocks * plane.code;
        out.high_bytes_per_row  = blocks * plane.high;
        out.scale_bytes_per_row = blocks * plane.scale;
        out.code_bytes          = mul(n, out.code_bytes_per_row);
        out.high_offset         = aligned(out.code_bytes, 256);
        out.high_bytes          = mul(n, out.high_bytes_per_row);
        out.scale_offset        = add(out.high_offset, aligned(out.high_bytes, 256));
    } else if (layout == QuantLayout::BlockScale128F32) {
        if (format != QType::FP8_E4M3FN_B128_F32) {
            throw std::invalid_argument("BlockScale128F32 requires block-scaled FP8");
        }
        out.group_size         = 128;
        out.code_bytes_per_row = k;
        out.code_bytes         = out.elements;
        out.scale_offset       = aligned(out.code_bytes, 256);
        out.scale_bytes        = mul(mul((n + 127) / 128, (k + 127) / 128), 4);
        out.bytes              = add(out.scale_offset, out.scale_bytes);
        return out;
    } else if (layout == QuantLayout::BlockScaleK16M128x4) {
        if (format != QType::NVFP4 || k % 64) {
            throw std::invalid_argument("NVFP4 BlockScale requires K%64=0");
        }
        // The scale plane is addressed in whole 128-row tiles: a last tile N does not fill (a
        // vocabulary of 202048 rows) keeps its full extent, the words of the rows past N zero.
        out.group_size          = 16;
        out.code_bytes_per_row  = k / 2;
        out.scale_bytes_per_row = k / 16;
        out.code_bytes          = out.elements / 2;
        out.scale_offset        = aligned(out.code_bytes, 256);
        out.scale_bytes         = mul(aligned(n, 128), out.scale_bytes_per_row);
    } else if (layout == QuantLayout::BlockScaleK32E8M0) {
        if (format != QType::MXFP4 || k % 32) {
            throw std::invalid_argument("MXFP4 BlockScaleK32E8M0 requires K%32=0");
        }
        out.group_size          = 32;
        out.code_bytes_per_row  = k / 2;
        out.scale_bytes_per_row = k / 32;
        out.code_bytes          = out.elements / 2;
        out.scale_offset        = aligned(out.code_bytes, 256);
    } else {
        throw std::invalid_argument("unknown quantized weight layout");
    }
    if (layout != QuantLayout::BlockScaleK16M128x4) {
        out.scale_bytes = mul(n, out.scale_bytes_per_row);
    }
    out.bytes = add(out.scale_offset, out.scale_bytes);
    if (layout == QuantLayout::RowGroupZpOrdered) {
        out.order_offset = aligned(out.bytes, 256);
        out.bytes        = add(out.order_offset, mul(k, 4));
    }
    if (format == QType::NVFP4) {
        if (n % divisors) {
            throw std::invalid_argument("divisor count must divide the plane's rows");
        }
        // A divisor may end inside a 128-row scale tile: the scale words are per row, and every
        // consumer divides a row's product by its own word (divisors[row / divisor_rows]).
        out.divisor_offset = out.bytes;
        out.bytes          = add(out.bytes, mul(divisors, 4));
    }
    return out;
}

WeightRegion contiguous_weight_region(const WeightView& view) {
    if (view.parts.empty()) { throw std::invalid_argument("weight view has no regions"); }
    WeightRegion merged = view.parts.front();
    validate_region(merged);
    for (std::size_t i = 1; i < view.parts.size(); ++i) {
        const auto& next = view.parts[i];
        validate_region(next);
        if (next.parent != merged.parent || next.begin != merged.end) {
            throw std::invalid_argument("native input requires one contiguous parent region");
        }
        merged.end = next.end;
    }
    if (merged.end - merged.begin != weight_element_count(view.shape)) {
        throw std::invalid_argument("weight view shape differs from its regions");
    }
    return merged;
}

bool is_complete_weight(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    if (!parent || view.shape != parent->geometry.shape) { return false; }
    std::uint64_t end = 0;
    for (const auto& part : view.parts) {
        if (part.parent != parent || part.begin != end || part.end <= part.begin) { return false; }
        end = part.end;
    }
    return end == parent->geometry.elements;
}

std::uint64_t weight_scale_offset(const WeightGeometry& geometry, std::uint64_t row,
                                  std::uint64_t group) {
    if (geometry.shape.size() != 2 || !geometry.group_size || row >= geometry.shape[0] ||
        group >= (geometry.padded_columns + geometry.group_size - 1) / geometry.group_size) {
        throw std::invalid_argument("weight scale coordinate exceeds its parent");
    }
    if (geometry.layout == QuantLayout::BlockScaleK16M128x4) {
        const auto inner = row % 128;
        return geometry.scale_offset + (row / 128 * (geometry.shape[1] / 64) + group / 4) * 512 +
               inner % 32 * 16 + inner / 32 * 4 + group % 4;
    }
    if (geometry.layout == QuantLayout::BlockScaleK32E8M0) {
        return geometry.scale_offset + row * geometry.scale_bytes_per_row + group;
    }
    if (geometry.layout == QuantLayout::BlockScale128F32) {
        return geometry.scale_offset + (row / 128 * ((geometry.shape[1] + 127) / 128) + group) * 4;
    }
    if (geometry.layout == QuantLayout::RowScale) {
        return geometry.scale_offset + row * geometry.scale_bytes_per_row;
    }
    return geometry.scale_offset + row * geometry.scale_bytes_per_row + group * 2;
}

WeightRowPlanes weight_row_planes(const WeightRegion& region) {
    validate_region(region);
    const auto& parent = *region.parent;
    const auto& g      = parent.geometry;
    if (!parent.data || g.shape.size() != 2 || region.begin % g.shape[1] ||
        region.end % g.shape[1]) {
        throw std::invalid_argument("row view requires resident complete logical rows");
    }
    WeightRowPlanes out;
    out.row_begin       = region.begin / g.shape[1];
    out.row_count       = (region.end - region.begin) / g.shape[1];
    out.code_row_bytes  = g.code_bytes_per_row;
    out.high_row_bytes  = g.high_bytes_per_row;
    out.scale_row_bytes = g.scale_bytes_per_row;
    out.codes           = parent.data + out.row_begin * out.code_row_bytes;
    if (g.high_bytes) {
        out.high = parent.data + g.high_offset + out.row_begin * out.high_row_bytes;
    }
    // NVFP4's swizzled tiles and FP8's 128-row blocks are addressed from the plane's start.
    out.swizzled_scales = g.layout == QuantLayout::BlockScaleK16M128x4 ||
                          g.layout == QuantLayout::BlockScale128F32;
    if (g.scale_bytes) {
        out.scales = parent.data + g.scale_offset +
                     (out.swizzled_scales ? 0 : out.row_begin * out.scale_row_bytes);
    }
    return out;
}

std::optional<WeightView> weight_row_prefix(const WeightView& view, std::uint64_t rows) {
    if (view.shape.size() != 2 || rows == 0 || rows >= view.shape[0] || view.parts.empty()) {
        return std::nullopt;
    }
    for (const auto& part : view.parts) {
        validate_region(part);
        const auto layout = part.parent->geometry.layout;
        if (layout == QuantLayout::RowScale || layout == QuantLayout::BlockScaleK16M128x4 ||
            layout == QuantLayout::BlockScale128F32) {
            return std::nullopt;
        }
    }
    WeightView out{{rows, view.shape[1]}, {}};
    std::uint64_t left = mul(rows, view.shape[1]);
    for (const auto& part : view.parts) {
        if (left == 0) { break; }
        const auto take = std::min(left, part.end - part.begin);
        out.parts.push_back(WeightRegion{part.parent, part.begin, part.begin + take});
        left -= take;
    }
    return out;
}

Tensor weight_tensor(const WeightView& view, std::initializer_list<std::int32_t> internal_shape) {
    const auto region = contiguous_weight_region(view);
    const auto& g     = region.parent->geometry;
    if (g.layout != QuantLayout::Contiguous || !region.parent->data) {
        throw std::invalid_argument("direct tensor requires a resident contiguous weight");
    }
    const auto offset = mul(region.begin, word_bytes(g.format));
    Tensor out(const_cast<std::byte*>(region.parent->data + offset), direct_dtype(g.format),
               internal_shape);
    if (static_cast<std::uint64_t>(out.numel()) != region.end - region.begin) {
        throw std::invalid_argument("native tensor shape differs from its logical weight");
    }
    return out;
}

QuantLayout default_weight_layout(QType format) noexcept {
    switch (format) {
    case QType::BF16:
    case QType::FP32:
    case QType::INT32:
        return QuantLayout::Contiguous;
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::FP8_E4M3FN_ROW_F32:
        return QuantLayout::RowScale;
    case QType::FP8_E4M3FN_B128_F32:
        return QuantLayout::BlockScale128F32;
    case QType::NVFP4:
        return QuantLayout::BlockScaleK16M128x4;
    case QType::MXFP4:
        return QuantLayout::BlockScaleK32E8M0;
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
        return QuantLayout::RowGroupZp;
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
        return QuantLayout::GgmlBlocks;
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    case QType::GGML_TQ1_0:
    case QType::GGML_TQ2_0:
    case QType::GGML_Q1_0:
    case QType::GGML_Q2_0:
        return QuantLayout::GgmlPlanes;
    default:
        return QuantLayout::RowSplit;
    }
}

Weight native_weight(const WeightView& view, float input_divisor) {
    const auto region = contiguous_weight_region(view);
    const auto& g     = region.parent->geometry;
    if (view.shape.size() != 2 || !region.parent->data) {
        throw std::invalid_argument("native Weight requires a resident logical matrix");
    }
    Weight out;
    out.payload          = region.parent->data;
    out.payload_bytes    = g.bytes;
    out.high_plane_bytes = g.high_bytes;
    out.qtype            = g.format;
    out.layout           = g.layout;
    out.ndim             = 2;
    out.n = out.shape[0] = out.padded_shape[0] = dimension(view.shape[0]);
    out.k = out.shape[1] = out.padded_shape[1] = dimension(view.shape[1]);
    if (g.layout == QuantLayout::Contiguous) {
        out.qdata = region.parent->data + mul(region.begin, word_bytes(g.format));
        return out;
    }
    if (g.shape.size() != 2 || view.shape[1] != g.shape[1]) {
        throw std::invalid_argument("quantized native Weight requires unchanged parent K");
    }
    const auto planes = weight_row_planes(region);
    if ((g.layout == QuantLayout::RowScale || g.layout == QuantLayout::BlockScaleK16M128x4 ||
         g.layout == QuantLayout::BlockScale128F32) &&
        !is_complete_weight(view)) {
        throw std::invalid_argument(
            "this native Weight input requires a complete FP8/NVFP4 parent");
    }
    out.padded_shape[1]      = dimension(g.padded_columns);
    out.qdata                = planes.codes;
    out.qhigh                = planes.high;
    out.scales               = planes.scales;
    out.group_size           = static_cast<std::uint32_t>(g.group_size);
    out.group                = g.group_size ? dimension(g.group_size) : 0;
    out.weight_scale_divisor = region.parent->weight_scale_divisor;
    out.input_scale_divisor  = input_divisor;
    if (g.layout == QuantLayout::RowGroupZpOrdered) {
        out.k_order = region.parent->data + g.order_offset;
    }
    if (g.layout == QuantLayout::RowSplit || g.layout == QuantLayout::RowGroupZp ||
        g.layout == QuantLayout::RowGroupZpOrdered || g.layout == QuantLayout::GgmlPlanes ||
        (g.layout == QuantLayout::GgmlFrag &&
         ggml_published_layout(g.format) == QuantLayout::GgmlPlanes)) {
        out.scale_dtype = DType::FP16;
        out.scale_ne[0] = dimension(g.scale_bytes_per_row / 2);
        out.scale_ne[1] = out.n;
        out.scale_nb[0] = 2;
        out.scale_nb[1] = static_cast<std::int64_t>(g.scale_bytes_per_row);
        out.scale_nb[2] = out.scale_nb[3] = out.scale_nb[1] * out.n;
    } else if (g.layout == QuantLayout::RowScale) {
        const auto word = static_cast<std::int64_t>(g.scale_bytes_per_row);
        out.scale_dtype = word == 4 ? DType::FP32 : DType::BF16;
        out.scale_ne[0] = out.n;
        out.scale_nb[0] = word;
        out.scale_nb[1] = out.scale_nb[2] = out.scale_nb[3] =
            static_cast<std::int64_t>(out.n) * word;
    } else if (g.layout == QuantLayout::BlockScale128F32) {
        out.scale_dtype = DType::FP32;
        out.scale_ne[0] = dimension((g.shape[1] + 127) / 128);
        out.scale_ne[1] = dimension((g.shape[0] + 127) / 128);
        out.scale_nb[0] = 4;
        out.scale_nb[1] = static_cast<std::int64_t>(out.scale_ne[0]) * 4;
        out.scale_nb[2] = out.scale_nb[3] = out.scale_nb[1] * out.scale_ne[1];
    } else if (g.layout == QuantLayout::BlockScaleK16M128x4) {
        out.scale_dtype         = DType::FP8_E4M3FN;
        out.weight_divisors     = region.parent->data + g.divisor_offset;
        out.weight_divisor_rows = dimension(g.shape[0] / g.divisor_count);
    } else if (g.layout == QuantLayout::BlockScaleK32E8M0) {
        out.scale_dtype = DType::U8; // E8M0 exponents, value 2^(e - 127)
        out.scale_ne[0] = dimension(g.shape[1] / 32);
        out.scale_ne[1] = out.n;
        out.scale_nb[0] = 1;
        out.scale_nb[1] = static_cast<std::int64_t>(g.scale_bytes_per_row);
        out.scale_nb[2] = out.scale_nb[3] = out.scale_nb[1] * out.n;
    }
    return out;
}

} // namespace ninferno
