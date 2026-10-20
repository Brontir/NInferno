#include "nvfp4_a16_mma_sweep.cuh"

namespace ninferno::bench::a16_sweep {
void sweep_n5120_k6144(const Options& options) {
    sweep<ops::detail::Nvfp4Geometry<5120, 6144>, NINFERNO_A16_SWEEP_FAMILIES>(options,
                                                                               "n5120_k6144");
}
} // namespace ninferno::bench::a16_sweep
