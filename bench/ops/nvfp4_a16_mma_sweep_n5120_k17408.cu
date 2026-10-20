#include "nvfp4_a16_mma_sweep.cuh"

namespace ninferno::bench::a16_sweep {
void sweep_n5120_k17408(const Options& options) {
    sweep<ops::detail::Nvfp4Geometry<5120, 17408>, NINFERNO_A16_SWEEP_FAMILIES>(options,
                                                                                "n5120_k17408");
}
} // namespace ninferno::bench::a16_sweep
