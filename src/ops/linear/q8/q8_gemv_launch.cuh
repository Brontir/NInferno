#pragma once
#include "core/device.h"
#include "core/device_class.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_a16_gemv.cuh"

namespace ninferno::ops::detail {
// The K=2048 decode GEMV of BlockRows rows a CTA: 16 codes a lane (one 16-byte load, four phases)
// instead of 8 (the Qwen3.6-35B GDN and attention input projections ran at ~78 % of the bandwidth
// with 8-byte loads).
template <int BlockRows, class Output, class Epilogue>
void launch_q8_a16_gemv_k2048(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                              cudaStream_t stream);
template <class Schedule, class Output, class Epilogue>
void launch_q8_a16_gemv(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                        cudaStream_t stream) {
    validate_q8_operands(operands);
    if (operands.tokens > Schedule::kBlockTokens || operands.k % Schedule::kValuesPerLane != 0)
        throw std::invalid_argument(
            "Q8 GEMV requires T within the schedule's capacity and K aligned to its lane width");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument(
                "Q8 GEMV static K requires matching, padding-free operands");
    }
    // Where the device has PDL the kernel prefetches its first codes under the previous kernel.
    const bool dependent = device_class().features.pdl;
    const auto launch    = [&]<bool FullRows>() {
        const unsigned blocks = div_up(operands.rows, Schedule::kBlockRows);
        CUDA_CHECK(pdl::launch({dim3(blocks), dim3(Schedule::kThreads), 0, stream}, dependent,
                               false, q8_a16_gemv_kernel<Schedule, FullRows, Output, Epilogue>,
                               operands, output, epilogue));
    };
    if (operands.rows % Schedule::kBlockRows == 0)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

template <int BlockRows, class Output, class Epilogue>
void launch_q8_a16_gemv_k2048(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                              cudaStream_t stream) {
    launch_q8_a16_gemv<Q8A16GemvSchedule<BlockRows, 1, 2, 2048, 4, 1, 16>>(operands, output,
                                                                           epilogue, stream);
}
} // namespace ninferno::ops::detail
