#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"

#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"

namespace ninferno::ops::detail {
namespace {
template <int K, CUtensorMapL2promotion Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE>
void launch(const Nvfp4A4Operands& p, __nv_bfloat16* output, cudaStream_t stream) {
    const LinearBf16Output y{output, p.rows};
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch_nvfp4_a4_tma_mma<
            Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 4, 1, Promotion>, K>>(
            p, y, LinearIdentityEpilogue{}, stream);
    else
        launch_nvfp4_a4_tma_mma<
            Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1, Promotion>, K>>(
            p, y, LinearIdentityEpilogue{}, stream);
}
} // namespace

void launch_nvfp4_a4_tma_linear(Nvfp4GeometryId problem, const Nvfp4A4Operands& p,
                                __nv_bfloat16* output, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4GeometryId::N14336K5120:
        launch<5120>(p, output, stream);
        return;
    case Nvfp4GeometryId::N16384K5120:
        launch<5120>(p, output, stream);
        return;
    case Nvfp4GeometryId::N34816K5120:
        launch<5120, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5120K6144:
        launch<6144>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch<17408>(p, output, stream);
        return;
    // Gemma 4 projections (K128 rows read 64 code bytes: the 128B promotion prefetches the next
    // K tile's half-line where it was measured faster).
    case Nvfp4GeometryId::N8192K2816:
        launch<2816>(p, output, stream);
        return;
    case Nvfp4GeometryId::N1024K2816:
        launch<2816, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N2816K4096:
        launch<4096>(p, output, stream);
        return;
    case Nvfp4GeometryId::N2816K8192:
        launch<8192>(p, output, stream);
        return;
    case Nvfp4GeometryId::N16384K5376:
        launch<5376, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N18432K5376:
        launch<5376, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N2048K5376:
        launch<5376, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N43008K5376:
        launch<5376, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5376K8192:
        launch<8192, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5376K16384:
        launch<16384, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5376K21504:
        launch<21504, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N2816K2112: // K is no whole number of TMA tiles
        break;
    }
    throw std::invalid_argument("NVFP4 TMA linear: unsupported geometry");
}
} // namespace ninferno::ops::detail
