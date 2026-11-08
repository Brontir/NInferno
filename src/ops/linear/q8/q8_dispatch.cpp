#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/q8/q8_shapes.h"
#include "ops/common/shape_routes.h"

#include <algorithm>
#include <array>
#include <optional>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::ops::detail {
namespace {
using ShapeSelector = Q8Launch (*)(std::int32_t);

struct ShapeEntry {
    std::int32_t n;
    std::int32_t k;
    ShapeSelector select;
};

template <class Geometry>
constexpr ShapeEntry shape(ShapeSelector select) {
    return {Geometry::kOutputRows, Geometry::kInputRows, select};
}

constexpr std::array kShapes{
    shape<Q8N1024K2048>(select_q8_n1024_k2048),     shape<Q8N1024K5120>(select_q8_n1024_k5120),
    shape<Q8N2048K4096>(select_q8_n2048_k4096),     shape<Q8N2048K4608>(select_q8_n2048_k4608),
    shape<Q8N2048K16384>(select_q8_n2048_k16384),   shape<Q8N4608K4608>(select_q8_n4608_k4608),
    shape<Q8N5120K4608>(select_q8_n5120_k4608),     shape<Q8N5120K6144>(select_q8_n5120_k6144),
    shape<Q8N5120K10240>(select_q8_n5120_k10240),   shape<Q8N5120K17408>(select_q8_n5120_k17408),
    shape<Q8N5120K25600>(select_q8_n5120_k25600),   shape<Q8N6144K5120>(select_q8_n6144_k5120),
    shape<Q8N9216K2048>(select_q8_n9216_k2048),     shape<Q8N12288K2048>(select_q8_n12288_k2048),
    shape<Q8N14336K5120>(select_q8_n14336_k5120),   shape<Q8N34816K5120>(select_q8_n34816_k5120),
    shape<Q8N248320K5120>(select_q8_n248320_k5120), shape<Q8N4224K2816>(select_q8_n4224_k2816),
    shape<Q8N8192K2816>(select_q8_n8192_k2816),     shape<Q8N9216K2816>(select_q8_n9216_k2816),
    shape<Q8N2816K4096>(select_q8_n2816_k4096),     shape<Q8N2816K8192>(select_q8_n2816_k8192),
    shape<Q8N2816K2112>(select_q8_n2816_k2112),     shape<Q8N16384K5376>(select_q8_n16384_k5376),
    shape<Q8N18432K5376>(select_q8_n18432_k5376),   shape<Q8N5376K8192>(select_q8_n5376_k8192),
    shape<Q8N5376K16384>(select_q8_n5376_k16384),   shape<Q8N1024K5632>(select_q8_n1024_k5632),
    shape<Q8N4096K1024>(select_q8_n4096_k1024),     shape<Q8N8192K1024>(select_q8_n8192_k1024),
    shape<Q8N1024K4096>(select_q8_n1024_k4096),     shape<Q8N1024K8192>(select_q8_n1024_k8192),
    shape<Q8N16384K1024>(select_q8_n16384_k1024),   shape<Q8N2816K1024>(select_q8_n2816_k1024),
    shape<Q8N262144K1024>(select_q8_n262144_k1024), shape<Q8N1024K10752>(select_q8_n1024_k10752),
    shape<Q8N1024K16384>(select_q8_n1024_k16384),   shape<Q8N5376K1024>(select_q8_n5376_k1024),
    shape<Q8N32768K1024>(select_q8_n32768_k1024),   shape<Q8N65536K1024>(select_q8_n65536_k1024),
    shape<Q8N98304K1024>(select_q8_n98304_k1024),   shape<Q8N131072K1024>(select_q8_n131072_k1024),
    shape<Q8N5376K10752>(select_q8_n5376_k10752),   shape<Q8N21504K5376>(select_q8_n21504_k5376),
    shape<Q8N10240K5376>(select_q8_n10240_k5376),   shape<Q8N5376K32256>(select_q8_n5376_k32256),
    shape<Q8N2816K5632>(select_q8_n2816_k5632),     shape<Q8N11264K2816>(select_q8_n11264_k2816),
    shape<Q8N6144K2816>(select_q8_n6144_k2816),     shape<Q8N10240K2816>(select_q8_n10240_k2816),
    shape<Q8N2816K16896>(select_q8_n2816_k16896),
};
} // namespace

bool q8_linear_registered(std::int32_t n, std::int32_t k) {
    return find_shape(kShapes, n, k) != nullptr;
}

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q8 linear: T must be positive");
    if (const auto* entry = find_shape(kShapes, n, k)) {
        return entry->select(route_tokens({QType::Q8_G32_FP16, ShapeCompute::A16, n, k}, t));
    }
    throw std::invalid_argument("q8 linear: unsupported shape");
}

Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q8 linear: unsupported policy");
    return select_q8_a16_launch(n, k, t);
}

std::optional<Q8LaunchTile> q8_launch_tile(Q8Launch launch) {
    // The launches' CTA tiles, read off their names (r<rows>_t<token columns>); a launch whose name
    // does not say (the gemv, the K-sliced MMA tiles with a warp layout in the rows) has none.
    static const std::array<std::pair<Q8Launch, Q8LaunchTile>, 13> kTiles{{
        {launch_q8_a16_simt_r4_t4_w2_g16_s2, {4, 4}},
        {launch_q8_a16_simt_r8_t4, {8, 4}},
        {launch_q8_a16_simt_r8_t8, {8, 8}},
        {launch_q8_a16_sliced_r16_t16_w8_s2, {16, 16}},
        {launch_q8_a16_sliced_r16_t16_w4_s2, {16, 16}},
        {launch_q8_a16_sliced_r16_t32_w4_s2, {16, 32}},
        {launch_q8_a16_mma_r32_t64, {32, 64}},
        {launch_q8_a16_mma_r32_t96, {32, 96}},
        {launch_q8_a16_mma_r32_t128, {32, 128}},
        {launch_q8_a16_mma_r48_t64, {48, 64}},
        {launch_q8_a16_mma_r64_t96, {64, 96}},
        {launch_q8_a16_mma_r64_t128, {64, 128}},
        {launch_q8_a16_mma_r128_t64, {128, 64}},
    }};
    for (const auto& [fn, tile] : kTiles) {
        if (fn == launch) { return tile; }
    }
    return std::nullopt;
}

std::string q8_launch_id(std::int32_t n, std::int32_t k, std::int32_t t) {
    const Q8Launch at = select_q8_a16_launch(n, k, t);
    // Most launches are template instances inside each shape's translation unit, with no name
    // here: the shape's launches are numbered in the order the widths first select them.
    std::vector<Q8Launch> seen;
    for (std::int32_t width = 1; width <= t; ++width) {
        const Q8Launch launch = select_q8_a16_launch(n, k, width);
        if (std::find(seen.begin(), seen.end(), launch) == seen.end()) { seen.push_back(launch); }
        if (launch == at) { break; }
    }
    const auto index = std::find(seen.begin(), seen.end(), at) - seen.begin();
    return "n" + std::to_string(n) + "_k" + std::to_string(k) + ".launch" + std::to_string(index);
}

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q8_launch(w.n, w.k, x.ne[1], policy)(x, w, out, stream);
}

} // namespace ninferno::ops::detail
