// Generic staged register-MMA schedules (generic_staged.cuh): BF16 weights.
#include "ops/generic/generic_staged.cuh"

#include "core/per_device.h"
#include "core/device.h"
#include "ops/common/arch_limits.h"
#include "ops/generic/generic_gemm_internal.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

// Keep in the order of kStagedSchedules in generic_gemm.cpp: rows (weight rows staged, SwiGLU gate
// and up together), tokens, warps_m, warps_n, stages. 9 (64 KiB) runs two CTAs per SM where an SM
// holds 128 KiB; 10-12 (144-160 KiB) exist only where one block holds them (GA100 and Orin).
#define NINFERNO_GENERIC_STAGED_SCHEDULES(X)                                                       \
    X(0, 128, 128, 2, 4, 3)                                                                        \
    X(1, 128, 64, 2, 2, 4)                                                                         \
    X(2, 64, 128, 1, 4, 4)                                                                         \
    X(3, 128, 128, 2, 2, 3)                                                                        \
    X(4, 256, 128, 4, 2, 2)                                                                        \
    X(5, 64, 64, 2, 2, 4)                                                                          \
    X(6, 128, 64, 2, 2, 2)                                                                         \
    X(7, 64, 128, 1, 4, 2)                                                                         \
    X(8, 64, 64, 2, 2, 3)                                                                          \
    X(9, 128, 128, 2, 4, 2)                                                                        \
    X(10, 128, 128, 2, 4, 5)                                                                       \
    X(11, 256, 128, 4, 2, 3)                                                                       \
    X(12, 128, 256, 2, 4, 3)

template <class S, Epilogue E>
void launch_schedule(const GemmArgs& args, cudaStream_t stream) {
    constexpr int kRowsPerBlock = E == Epilogue::SwiGlu ? S::kRows / 2 : S::kRows;
    constexpr int kShared       = static_cast<int>(sizeof(StagedShared<S, S::kStages>));
    if constexpr (kShared > kMaxSharedBytesPerBlock) {
        // route_admits never names it (staged_fits); a direct call is refused.
        throw std::invalid_argument("generic staged: schedule exceeds this build's shared memory");
    } else {
        static_assert(StagedStages<S>::value == S::kStages, "the route name states the stages it runs");
        static ::ninferno::PerDevice<bool> configured;
        ::ninferno::allow_dynamic_smem(configured, staged_kernel<S, E>, kShared);
        const dim3 grid((args.rows + kRowsPerBlock - 1) / kRowsPerBlock,
                        (args.tokens + S::kTokens - 1) / S::kTokens);
        if (grid.y > kMaxGridYZ)
            throw std::invalid_argument("generic staged: token extent exceeds the grid");
        staged_kernel<S, E><<<grid, S::kThreads, kShared, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <Epilogue E>
void launch_epilogue(int schedule, const GemmArgs& args, cudaStream_t stream) {
    switch (schedule) {
#define NINFERNO_GENERIC_CASE(I, R, T, WM, WN, ST)                                                 \
    case I:                                                                                        \
        launch_schedule<StagedSchedule<R, T, WM, WN, ST>, E>(args, stream);                        \
        return;
        NINFERNO_GENERIC_STAGED_SCHEDULES(NINFERNO_GENERIC_CASE)
#undef NINFERNO_GENERIC_CASE
    default:
        throw std::invalid_argument("generic staged: unknown schedule");
    }
}

} // namespace

bool staged_fits(int schedule) {
    switch (schedule) {
#define NINFERNO_GENERIC_FITS(I, R, T, WM, WN, ST)                                                 \
    case I:                                                                                        \
        return sizeof(StagedShared<StagedSchedule<R, T, WM, WN, ST>, ST>) <= kMaxSharedBytesPerBlock;
        NINFERNO_GENERIC_STAGED_SCHEDULES(NINFERNO_GENERIC_FITS)
#undef NINFERNO_GENERIC_FITS
    default:
        return false;
    }
}

void launch_staged(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    switch (epilogue) {
    case Epilogue::Store:
        launch_epilogue<Epilogue::Store>(schedule, args, stream);
        return;
    case Epilogue::AddResidual:
        launch_epilogue<Epilogue::AddResidual>(schedule, args, stream);
        return;
    case Epilogue::SwiGlu:
        launch_epilogue<Epilogue::SwiGlu>(schedule, args, stream);
        return;
    }
    throw std::invalid_argument("generic staged: unknown epilogue");
}

} // namespace ninferno::ops::detail::generic
