#include "nvfp4_a16_mma_sweep.cuh"

namespace ninferno::bench::a16_sweep {
void sweep_n14336_k5120(const Options& options) {
    sweep<ops::detail::Nvfp4Geometry<14336, 5120>, NINFERNO_A16_SWEEP_FAMILIES>(options,
                                                                                "n14336_k5120");
}
} // namespace ninferno::bench::a16_sweep
