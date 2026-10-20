#pragma once

// Compiled schedule lists and their launchers. Each format's translation unit instantiates
// launch_rmma<Decoder> once (generic_gemm_<format>.cu), so the whole list is compiled per format.

#include "core/device.h"
#include "ops/generic/generic_gemm.cuh"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/common/arch_limits.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {

// Keep in the order of kRmmaSchedules in generic_gemm.cpp: index, warps_m, warps_k, m_tiles,
// n_tiles, cluster (K split over a thread-block cluster of that many CTAs, generic_gemm.cuh).
#define NINFERNO_GENERIC_RMMA_SCHEDULES(X)                                                         \
    X(0, 1, 4, 1, 1, 1)                                                                            \
    X(1, 1, 8, 1, 1, 1)                                                                            \
    X(2, 2, 2, 1, 1, 1)                                                                            \
    X(3, 4, 1, 1, 1, 1)                                                                            \
    X(4, 4, 1, 2, 1, 1)                                                                            \
    X(5, 2, 2, 1, 2, 1)                                                                            \
    X(6, 4, 1, 1, 2, 1)                                                                            \
    X(7, 4, 1, 2, 2, 1)                                                                            \
    X(8, 4, 1, 1, 4, 1)                                                                            \
    X(9, 4, 1, 2, 4, 1)                                                                            \
    X(10, 4, 1, 1, 8, 1)                                                                           \
    X(11, 4, 1, 2, 8, 1)                                                                           \
    X(12, 2, 1, 2, 8, 1)                                                                           \
    X(13, 8, 1, 1, 8, 1)                                                                           \
    X(14, 1, 8, 1, 4, 1)                                                                           \
    X(15, 1, 4, 1, 4, 1)                                                                           \
    X(16, 1, 8, 1, 1, 8)                                                                           \
    X(17, 1, 4, 1, 2, 1)                                                                           \
    X(18, 1, 8, 1, 1, 2)                                                                           \
    X(19, 1, 16, 1, 1, 1)

template <class Decoder, class Schedule, Epilogue E>
void launch_rmma_schedule(const GemmArgs& args, cudaStream_t stream) {
    const dim3 grid((args.rows + Schedule::kRowsPerCta - 1) / Schedule::kRowsPerCta,
                    (args.tokens + Schedule::kTokensPerCta - 1) / Schedule::kTokensPerCta);
    if (grid.y > kMaxGridYZ)
        throw std::invalid_argument("generic linear: token extent exceeds the grid");
    const auto smem = static_cast<std::size_t>(args.spread_smem);
    if (Schedule::kCluster == 1 &&
        ((args.join_previous == 0 && args.pdl_prefetch == 0) || !pdl::supported())) {
        rmma_kernel<Decoder, Schedule, E><<<grid, Schedule::kThreads, smem, stream>>>(args);
    } else {
        cudaLaunchAttribute attributes[2]{};
        unsigned count = 0;
        if (Schedule::kCluster > 1) {
            attributes[count].id               = cudaLaunchAttributeClusterDimension;
            attributes[count].val.clusterDim.x = 1;
            attributes[count].val.clusterDim.y = 1;
            attributes[count].val.clusterDim.z = Schedule::kCluster;
            ++count;
        }
        if ((args.join_previous != 0 || args.pdl_prefetch != 0) && pdl::supported()) {
            attributes[count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attributes[count].val.programmaticStreamSerializationAllowed = 1;
            ++count;
        }
        cudaLaunchConfig_t config{};
        config.gridDim  = dim3(grid.x, grid.y, Schedule::kCluster);
        config.blockDim = dim3(Schedule::kThreads);
        config.dynamicSmemBytes = smem;
        config.stream   = stream;
        config.attrs    = attributes;
        config.numAttrs = count;
        CUDA_CHECK(cudaLaunchKernelEx(&config, rmma_kernel<Decoder, Schedule, E>, args));
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Decoder, Epilogue E>
void launch_rmma_epilogue(int schedule, const GemmArgs& args, cudaStream_t stream) {
    switch (schedule) {
#define NINFERNO_GENERIC_CASE(I, WM, WK, MT, NT, C)                                                \
    case I:                                                                                        \
        if constexpr (rmma_compiled<Decoder, MT, NT>) {                                            \
            launch_rmma_schedule<Decoder, RmmaSchedule<WM, WK, MT, NT, C>, E>(args, stream);       \
            return;                                                                                \
        }                                                                                          \
        break;
        // Thread-block clusters are sm_90+; on the Ada build the C>1 schedules' cluster epilogue is
        // carved out in the preprocessor (generic_gemm.cuh) and rmma_index() clamps its cluster
        // argument to 1 there, so the selector never offers a cluster split that build cannot
        // launch.
        NINFERNO_GENERIC_RMMA_SCHEDULES(NINFERNO_GENERIC_CASE)
#undef NINFERNO_GENERIC_CASE
    default:
        throw std::invalid_argument("generic linear: unknown register-MMA schedule");
    }
    throw std::invalid_argument("generic linear: register-MMA schedule not compiled for this format");
}

template <class Decoder>
void launch_rmma(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    switch (epilogue) {
    case Epilogue::Store:
        launch_rmma_epilogue<Decoder, Epilogue::Store>(schedule, args, stream);
        return;
    case Epilogue::AddResidual:
        launch_rmma_epilogue<Decoder, Epilogue::AddResidual>(schedule, args, stream);
        return;
    case Epilogue::SwiGlu:
        launch_rmma_epilogue<Decoder, Epilogue::SwiGlu>(schedule, args, stream);
        return;
    }
    throw std::invalid_argument("generic linear: unknown epilogue");
}

} // namespace ninferno::ops::detail::generic
