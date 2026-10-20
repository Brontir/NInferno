#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail {
template <std::int32_t OutputRows, std::int32_t InputRows>
struct Nvfp4Geometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kOutputRows       = OutputRows;
    static constexpr std::int32_t kInputRows        = InputRows;
    static constexpr std::int32_t kGroupsPerRow     = InputRows / 16;
    static constexpr std::int32_t kScaleTilesPerRow = InputRows / 64;
    static constexpr std::int32_t kCodeBytesPerRow  = InputRows / 2;
};

template <std::int32_t InputRows>
struct Nvfp4ActivationGeometry {
    static_assert(InputRows > 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kInputRows       = InputRows;
    static constexpr std::int32_t kGroupsPerRow    = InputRows / 16;
    static constexpr std::int32_t kCodeBytesPerRow = InputRows / 2;
};

using Nvfp4N14336K5120 = Nvfp4Geometry<14336, 5120>;
using Nvfp4N16384K5120 = Nvfp4Geometry<16384, 5120>;
using Nvfp4N34816K5120 = Nvfp4Geometry<34816, 5120>;
using Nvfp4N5120K6144  = Nvfp4Geometry<5120, 6144>;
using Nvfp4N5120K17408 = Nvfp4Geometry<5120, 17408>;
// Gemma 4 26B-A4B dense projections.
using Nvfp4N8192K2816 = Nvfp4Geometry<8192, 2816>;
using Nvfp4N1024K2816 = Nvfp4Geometry<1024, 2816>;
using Nvfp4N2816K4096 = Nvfp4Geometry<2816, 4096>;
using Nvfp4N2816K8192 = Nvfp4Geometry<2816, 8192>;
using Nvfp4N2816K2112 = Nvfp4Geometry<2816, 2112>;
// Gemma 4 31B dense projections.
using Nvfp4N16384K5376 = Nvfp4Geometry<16384, 5376>;
using Nvfp4N18432K5376 = Nvfp4Geometry<18432, 5376>;
using Nvfp4N2048K5376  = Nvfp4Geometry<2048, 5376>;
using Nvfp4N43008K5376 = Nvfp4Geometry<43008, 5376>;
using Nvfp4N5376K8192  = Nvfp4Geometry<5376, 8192>;
using Nvfp4N5376K16384 = Nvfp4Geometry<5376, 16384>;
using Nvfp4N5376K21504 = Nvfp4Geometry<5376, 21504>;

using Nvfp4Activation5120Geometry  = Nvfp4ActivationGeometry<5120>;
using Nvfp4Activation6144Geometry  = Nvfp4ActivationGeometry<6144>;
using Nvfp4Activation17408Geometry = Nvfp4ActivationGeometry<17408>;
using Nvfp4Activation2112Geometry  = Nvfp4ActivationGeometry<2112>;
using Nvfp4Activation2816Geometry  = Nvfp4ActivationGeometry<2816>;
using Nvfp4Activation4096Geometry  = Nvfp4ActivationGeometry<4096>;
using Nvfp4Activation5376Geometry  = Nvfp4ActivationGeometry<5376>;
using Nvfp4Activation8192Geometry  = Nvfp4ActivationGeometry<8192>;
using Nvfp4Activation16384Geometry = Nvfp4ActivationGeometry<16384>;
using Nvfp4Activation21504Geometry = Nvfp4ActivationGeometry<21504>;

// Problems the W4A4 TMA route is built for. The route reads activation scales in tiles of sixteen
// groups, so a K that is not a multiple of 256 has no entry: [2816,2112] runs its W4A4 GEMM on
// the cp.async MMA route at every T.
enum class Nvfp4GeometryId : std::uint8_t {
    N14336K5120,
    N16384K5120,
    N34816K5120,
    N5120K6144,
    N5120K17408,
    N8192K2816,
    N2816K2112,
    N1024K2816,
    N2816K4096,
    N2816K8192,
    N16384K5376,
    N18432K5376,
    N2048K5376,
    N43008K5376,
    N5376K8192,
    N5376K16384,
    N5376K21504,
};

inline Nvfp4GeometryId resolve_nvfp4_geometry(std::int32_t output_rows, std::int32_t input_rows) {
    struct Entry {
        std::int32_t n;
        std::int32_t k;
        Nvfp4GeometryId id;
    };

    constexpr Entry kEntries[] = {
        {Nvfp4N14336K5120::kOutputRows, Nvfp4N14336K5120::kInputRows, Nvfp4GeometryId::N14336K5120},
        {Nvfp4N16384K5120::kOutputRows, Nvfp4N16384K5120::kInputRows, Nvfp4GeometryId::N16384K5120},
        {Nvfp4N34816K5120::kOutputRows, Nvfp4N34816K5120::kInputRows, Nvfp4GeometryId::N34816K5120},
        {Nvfp4N5120K6144::kOutputRows, Nvfp4N5120K6144::kInputRows, Nvfp4GeometryId::N5120K6144},
        {Nvfp4N5120K17408::kOutputRows, Nvfp4N5120K17408::kInputRows, Nvfp4GeometryId::N5120K17408},
        {Nvfp4N8192K2816::kOutputRows, Nvfp4N8192K2816::kInputRows, Nvfp4GeometryId::N8192K2816},
        {Nvfp4N2816K2112::kOutputRows, Nvfp4N2816K2112::kInputRows, Nvfp4GeometryId::N2816K2112},
        {Nvfp4N1024K2816::kOutputRows, Nvfp4N1024K2816::kInputRows, Nvfp4GeometryId::N1024K2816},
        {Nvfp4N2816K4096::kOutputRows, Nvfp4N2816K4096::kInputRows, Nvfp4GeometryId::N2816K4096},
        {Nvfp4N2816K8192::kOutputRows, Nvfp4N2816K8192::kInputRows, Nvfp4GeometryId::N2816K8192},
        {Nvfp4N16384K5376::kOutputRows, Nvfp4N16384K5376::kInputRows, Nvfp4GeometryId::N16384K5376},
        {Nvfp4N18432K5376::kOutputRows, Nvfp4N18432K5376::kInputRows, Nvfp4GeometryId::N18432K5376},
        {Nvfp4N2048K5376::kOutputRows, Nvfp4N2048K5376::kInputRows, Nvfp4GeometryId::N2048K5376},
        {Nvfp4N43008K5376::kOutputRows, Nvfp4N43008K5376::kInputRows, Nvfp4GeometryId::N43008K5376},
        {Nvfp4N5376K8192::kOutputRows, Nvfp4N5376K8192::kInputRows, Nvfp4GeometryId::N5376K8192},
        {Nvfp4N5376K16384::kOutputRows, Nvfp4N5376K16384::kInputRows, Nvfp4GeometryId::N5376K16384},
        {Nvfp4N5376K21504::kOutputRows, Nvfp4N5376K21504::kInputRows, Nvfp4GeometryId::N5376K21504},
    };
    for (const Entry& entry : kEntries) {
        if (entry.n == output_rows && entry.k == input_rows) { return entry.id; }
    }
    throw std::invalid_argument("unsupported NVFP4 problem");
}

} // namespace ninferno::ops::detail
