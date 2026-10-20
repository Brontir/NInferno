#include "nvfp4_a16_mma_sweep.cuh"

namespace ninferno::bench::a16_sweep {
void sweep_n34816_k5120(const Options& options) {
    sweep<ops::detail::Nvfp4Geometry<34816, 5120>, NINFERNO_A16_SWEEP_FAMILIES>(options,
                                                                                "n34816_k5120");
}
} // namespace ninferno::bench::a16_sweep
