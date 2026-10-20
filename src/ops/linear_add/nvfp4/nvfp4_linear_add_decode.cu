#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/common/epilogue.cuh"

namespace ninferno::ops::detail {
namespace {

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n},
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}}, stream);
}

} // namespace

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream) {
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        launch<Nvfp4N5120K6144>(x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch<Nvfp4N5120K17408>(x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
    case Nvfp4GeometryId::N8192K2816:
    case Nvfp4GeometryId::N1024K2816:
    case Nvfp4GeometryId::N2816K4096:
    case Nvfp4GeometryId::N2816K8192:
    case Nvfp4GeometryId::N16384K5376:
    case Nvfp4GeometryId::N18432K5376:
    case Nvfp4GeometryId::N2048K5376:
    case Nvfp4GeometryId::N43008K5376:
    case Nvfp4GeometryId::N5376K8192:
    case Nvfp4GeometryId::N5376K16384:
    case Nvfp4GeometryId::N5376K21504:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninferno::ops::detail
