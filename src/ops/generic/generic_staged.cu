// Generic staged register-MMA schedules (generic_staged.cuh): BF16 weights.
#include "ops/generic/generic_staged.cuh"

#include "core/per_device.h"
#include "core/device.h"
#include "ops/common/arch_limits.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/generic_shared_bytes.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

// The schedules: rows (weight rows staged, SwiGLU gate
// and up together), tokens, warps_m, warps_n, stages. 9 (64 KiB) runs two CTAs per SM where an SM
// holds 128 KiB; 10-12 (144-160 KiB) exist only where one block holds them (GA100 and Orin).
// NINFERNO_GENERIC_STAGED_SCHEDULES: ops/generic/generic_schedules.h (one list for the
// instantiations here and the host table in generic_schedule_tables.h).

// generic_shared_bytes.h's table of the staged layouts (staged_fits reads it with no CUDA header),
// held equal to the layouts' own sizes, and those to the family's record (family_facts.h) the
// price reads: a stage of stage_k BF16 values of every row and token.
static_assert(kStagedMmaFacts.stage_row_bytes == kStagedMmaFacts.stage_k * 2 && // BF16
              kStagedMmaFacts.stage_token_bytes == kStagedMmaFacts.stage_row_bytes);
#define NINFERNO_GENERIC_HOLDS(I, R, T, WM, WN, ST)                                                \
    static_assert(kStagedSharedBytes[I] ==                                                         \
                  static_cast<int>(sizeof(StagedShared<StagedSchedule<R, T, WM, WN, ST>, ST>)));   \
    static_assert(kStagedSharedBytes[I] == ST * (R + T) * kStagedMmaFacts.stage_row_bytes);
NINFERNO_GENERIC_STAGED_SCHEDULES(NINFERNO_GENERIC_HOLDS)
#undef NINFERNO_GENERIC_HOLDS

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
        if (KernelResources* described = kernel_describe_target()) {
            cudaFuncAttributes attributes{};
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, (staged_kernel<S, E>)));
            describe_kernel(*described, attributes, grid, S::kThreads, kShared, false);
            return;
        }
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
