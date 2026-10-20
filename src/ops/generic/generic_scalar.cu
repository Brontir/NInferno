// The generic scalar projection: one warp per output element, every codec. It takes what the
// register MMA cannot read (a K that is no whole number of the codec's MMA steps).
#include "core/device.h"
#include "ops/generic/generic_gemm.cuh"
#include "ops/generic/generic_gemm_internal.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

template <Epilogue E>
void launch_scalar_epilogue(WeightCodec codec, const GemmArgs& args, cudaStream_t stream) {
    const std::int64_t warps  = static_cast<std::int64_t>(args.rows) * args.tokens;
    const std::int64_t blocks = (warps + 7) / 8;
    if (blocks > 0x7fffffffLL) throw std::invalid_argument("generic scalar: problem too large");
    scalar_kernel<E><<<static_cast<unsigned>(blocks), 256, 0, stream>>>(args, codec);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_scalar(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream) {
    switch (epilogue) {
    case Epilogue::Store:
        launch_scalar_epilogue<Epilogue::Store>(codec, args, stream);
        return;
    case Epilogue::AddResidual:
        launch_scalar_epilogue<Epilogue::AddResidual>(codec, args, stream);
        return;
    case Epilogue::SwiGlu:
        launch_scalar_epilogue<Epilogue::SwiGlu>(codec, args, stream);
        return;
    }
    throw std::invalid_argument("generic scalar: unknown epilogue");
}

} // namespace ninferno::ops::detail::generic
