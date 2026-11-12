#pragma once

// The launches tools/bench/mmq_launch_microbench.cu times: ggml's matrix kernel through the
// production launch template (ops/generic/ggml_mmq/ggml_mmq_internal.cuh detail::launch), with the
// column tile J and the CTA count forced instead of chosen. The ggml headers define CUDA_CHECK and
// friends, so only the per-format units (mmq_launch_microbench_<format>.cu, through
// mmq_launch_microbench_launch.cuh) include them; this header and the main unit see raw pointers.
// BENCH ONLY: no production code calls these.

#include "ops/generic/ggml_mmq/ggml_mmq.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninferno::bench::mmq_launch {

namespace ggml_mmq = ::ninferno::ops::ggml_mmq;

// ggml's config of one column tile on this device (mmq.cuh ggml_cuda_mmq_get_config, the shared
// bytes of mmq_get_nbytes_shared): `ok` false where its table has no such tile or the device's
// opt-in shared memory cannot hold it.
struct TileConfig {
    bool ok            = false;
    int rows           = 0; // I: the tile's weight rows
    int threads        = 0;
    std::size_t shared = 0;
    int k_vram         = 0; // a stream-k CTA's unit along K
    bool stream_k      = false;
};

// One forced launch: a rows x columns product over k with column tile j on `blocks` CTAs. ggml's
// stream-k kernel takes [b W / B, (b + 1) W / B) of the W = tiles x k / k_vram units: `blocks`
// equal to the tiles is a tile a CTA, any other count (at most one CTA a SM) splits the tiles'
// units, and where the CTAs do not divide the tiles the fixup kernel runs.
struct Forced {
    int rows    = 0;
    int k       = 0;
    int columns = 0;
    int j       = 0;
    int blocks  = 0;
};

struct FormatOps {
    const char* name = "";
    ggml_mmq::Type type{};
    // ggml's tile config at width j on this device.
    TileConfig (*config)(int j) = nullptr;
    // One product with the forced launch: out[c * rows + r], the fixup plane `fixup` (blocks x j x
    // the tile's rows FP32) where the launch has one. Throws std::invalid_argument for a width
    // the device has no tile of.
    void (*product)(const ggml_mmq::Planes& weight, int blocks_per_row, const Forced& launch,
                    const void* activation, float* out, void* fixup, cudaStream_t stream) = nullptr;
    // The stream-k fixup kernel of the forced launch alone (it adds the partial tiles of `fixup`
    // into `out` again: the timing does not depend on the values).
    void (*fixup_only)(const Forced& launch, float* out, void* fixup,
                       cudaStream_t stream) = nullptr;
    // CTAs of the forced launch's tile kernel one SM holds at once (cudaOccupancy...): the launch
    // law assumes one. Valid after a product at that width (which raises the shared-memory limit).
    int (*resident_ctas)(int j) = nullptr;
};

extern const FormatOps kQ4K;
extern const FormatOps kQ5K;
extern const FormatOps kQ6K;

} // namespace ninferno::bench::mmq_launch
