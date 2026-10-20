// Generic NVFP4 W4A4 schedules.
#include "ops/generic/generic_w4a4.cuh"

#include "core/device.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/common/arch_limits.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

// Keep in the order of kW4a4Schedules in generic_gemm.cpp.
#define NINFERNO_GENERIC_W4A4_SCHEDULES(X)                                                         \
    X(0, 32, 64, 2, 2, 4)                                                                          \
    X(1, 32, 128, 2, 4, 4)                                                                         \
    X(2, 64, 128, 4, 2, 4)                                                                         \
    X(3, 128, 128, 4, 2, 3)

template <class S, Epilogue E>
void launch_schedule(const GemmArgs& args, W4a4Activation act, cudaStream_t stream) {
    constexpr int kRowsPerBlock = E == Epilogue::SwiGlu ? S::kBlockN / 2 : S::kBlockN;
    const dim3 grid((args.rows + kRowsPerBlock - 1) / kRowsPerBlock,
                    (args.tokens + S::kBlockM - 1) / S::kBlockM);
    if (grid.y > kMaxGridYZ)
        throw std::invalid_argument("generic W4A4: token extent exceeds the grid");
    w4a4_kernel<S, E><<<grid, S::kThreads, 0, stream>>>(args, act);
    CUDA_CHECK(cudaGetLastError());
}

template <Epilogue E>
void launch_epilogue(int schedule, const GemmArgs& args, W4a4Activation act, cudaStream_t stream) {
    switch (schedule) {
#define NINFERNO_GENERIC_CASE(I, BM, BN, WM, WN, ST)                                               \
    case I:                                                                                        \
        launch_schedule<W4a4Schedule<BM, BN, WM, WN, ST>, E>(args, act, stream);                   \
        return;
        NINFERNO_GENERIC_W4A4_SCHEDULES(NINFERNO_GENERIC_CASE)
#undef NINFERNO_GENERIC_CASE
    default:
        throw std::invalid_argument("generic W4A4: unknown schedule");
    }
}

} // namespace

void launch_w4a4(int schedule, Epilogue epilogue, const GemmArgs& args, float input_divisor,
                 std::uint8_t* codes, std::uint8_t* scales, cudaStream_t stream) {
    const std::int64_t tasks = static_cast<std::int64_t>(args.tokens) * (args.k / 16);
    w4a4_quantize_kernel<<<static_cast<unsigned>((tasks + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(args.x), args.ldx, args.k, args.tokens, input_divisor,
        codes, scales);
    CUDA_CHECK(cudaGetLastError());
    const W4a4Activation act{codes, scales, input_divisor};
    switch (epilogue) {
    case Epilogue::Store:
        launch_epilogue<Epilogue::Store>(schedule, args, act, stream);
        return;
    case Epilogue::AddResidual:
        launch_epilogue<Epilogue::AddResidual>(schedule, args, act, stream);
        return;
    case Epilogue::SwiGlu:
        launch_epilogue<Epilogue::SwiGlu>(schedule, args, act, stream);
        return;
    }
    throw std::invalid_argument("generic W4A4: unknown epilogue");
}

} // namespace ninferno::ops::detail::generic
