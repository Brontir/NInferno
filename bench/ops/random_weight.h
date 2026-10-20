#pragma once

// A format-correct random weight of any stored format, for the benches that time the generic
// projection path's routes (ninferno_route_tune, ninferno_r8_schedule_bench): random codes, finite
// positive scales, stored in the layout the converter writes for that format.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "core/weight_view.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace ninferno::bench {

inline QuantLayout layout_of(QType q) {
    switch (q) {
    case QType::BF16:
        return QuantLayout::Contiguous;
    case QType::FP8_E4M3FN_ROW_BF16:
        return QuantLayout::RowScale;
    case QType::NVFP4:
        return QuantLayout::BlockScaleK16M128x4;
    case QType::MXFP4:
        return QuantLayout::BlockScaleK32E8M0;
    case QType::FP8_E4M3FN_ROW_F32:
        return QuantLayout::RowScale;
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
        return QuantLayout::RowGroupZp;
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
        return QuantLayout::GgmlBlocks;
    case QType::Q3_G128_FP16: // numbered after the ggml formats, stored RowSplit like Q4..Q6 G64
    case QType::Q3_G64_FP16:
        return QuantLayout::RowSplit;
    default: // the converter's layouts (src/convert/planner.cpp): the other ggml formats as planes
        return static_cast<int>(q) >= static_cast<int>(QType::GGML_Q6_K) ? QuantLayout::GgmlPlanes
                                                                       : QuantLayout::RowSplit;
    }
}

// A random parent of [rows, k] and its native Weight.
struct RandomWeight {
    DeviceBuffer buffer;
    WeightParent parent;
    Weight weight;
};

inline RandomWeight random_weight(QType q, std::int32_t rows, std::int32_t k, std::uint32_t seed) {
    const std::uint64_t shape[2] = {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(k)};
    RandomWeight out;
    out.parent.geometry = weight_geometry(q, layout_of(q), shape);
    const auto& g       = out.parent.geometry;
    std::vector<std::uint8_t> host(g.bytes);
    std::mt19937 rng(seed);
    for (auto& b : host) b = static_cast<std::uint8_t>(rng());
    const auto fill16 = [&](std::uint64_t offset, std::uint64_t bytes, std::uint16_t word) {
        for (std::uint64_t i = offset; i + 1 < offset + bytes; i += 2) {
            host[i]     = static_cast<std::uint8_t>(word & 0xff);
            host[i + 1] = static_cast<std::uint8_t>(word >> 8);
        }
    };
    switch (q) {
    case QType::BF16:
        for (std::uint64_t i = 1; i < host.size(); i += 2) host[i] = 0x3c ^ (host[i] & 0x81);
        break;
    case QType::FP8_E4M3FN_ROW_BF16:
        for (std::uint64_t i = 0; i < g.code_bytes; ++i) host[i] &= 0xb7; // finite E4M3
        fill16(g.scale_offset, g.scale_bytes, 0x3c00);                     // BF16 0.0078
        break;
    case QType::NVFP4:
        for (std::uint64_t i = g.scale_offset; i < g.scale_offset + g.scale_bytes; ++i) {
            host[i] = 0x30; // E4M3 0.5
        }
        {
            const float one = 1.0F;
            std::memcpy(host.data() + g.divisor_offset, &one, sizeof(one));
        }
        break;
    default:
        fill16(g.scale_offset, g.scale_bytes, 0x2000); // FP16 2^-7
        break;
    }
    out.buffer = DeviceBuffer(host.size());
    out.buffer.copy_from_host(host.data(), host.size());
    out.parent.data                 = static_cast<const std::byte*>(out.buffer.p);
    out.parent.weight_scale_divisor = 1.0F;
    WeightView view{g.shape, {WeightRegion{&out.parent, 0, g.elements}}};
    out.weight = native_weight(view, 4.0F);
    return out;
}

// The same bytes at another device address (a benchmark's copies that defeat the L2).
inline RandomWeight clone_weight(const RandomWeight& source) {
    RandomWeight out;
    out.parent    = source.parent;
    const auto& g = out.parent.geometry;
    out.buffer    = DeviceBuffer(source.buffer.bytes);
    CUDA_CHECK(cudaMemcpy(out.buffer.p, source.buffer.p, source.buffer.bytes,
                          cudaMemcpyDeviceToDevice));
    out.parent.data = static_cast<const std::byte*>(out.buffer.p);
    WeightView view{g.shape, {WeightRegion{&out.parent, 0, g.elements}}};
    out.weight = native_weight(view, 4.0F);
    return out;
}

} // namespace ninferno::bench
