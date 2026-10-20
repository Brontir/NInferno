#pragma once
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_a16_grouped_sliced_k_mma.cuh"

namespace ninferno::ops::detail {
constexpr int largest_proper_divisor(int n) {
    for (int d = n / 2; d > 1; --d) {
        if (n % d == 0) return d;
    }
    return 1;
}

template <class Schedule, class Output, class Epilogue>
void launch_q8_a16_grouped_sliced_k_mma(const Q8LinearOperands& operands, Output output,
                                        Epilogue epilogue, cudaStream_t stream) {
    // NINFERNO-SPECIAL(arch:sm75): a tile over the build's shared memory (Turing's 64 KiB) runs
    // single-staged where that fits, else over fewer token groups: the same K split and sum order
    // for every column either way.
    if constexpr (!Schedule::kFitsSharedLimit && Schedule::kStages == 2 &&
                  Schedule::template with_stages<1>::kFitsSharedLimit) {
        launch_q8_a16_grouped_sliced_k_mma<typename Schedule::template with_stages<1>>(
            operands, output, epilogue, stream);
    } else if constexpr (!Schedule::kFitsSharedLimit && Schedule::kTokenGroups > 1) {
        using Narrower = typename Schedule::template with_token_groups<largest_proper_divisor(
            Schedule::kTokenGroups)>;
        launch_q8_a16_grouped_sliced_k_mma<Narrower>(operands, output, epilogue, stream);
    } else {
        validate_q8_operands(operands);
        if (operands.k % 8 != 0)
            throw std::invalid_argument("Q8 grouped sliced-K MMA requires K aligned to 8");
        if constexpr (Schedule::kStaticK > 0) {
            if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
                throw std::invalid_argument(
                    "Q8 grouped sliced-K static K requires matching, padding-free operands");
        }
        if (!Schedule::kTiledTokens && operands.tokens > Schedule::kBlockTokens)
            throw std::invalid_argument(
                "Q8 grouped sliced-K token count exceeds the schedule capacity");
        for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const dim3 grid(div_up(operands.rows, 16), div_up(count, Schedule::kBlockTokens));
            const auto launch = [&]<bool Full>() {
                static constexpr auto kernel =
                    q8_a16_grouped_sliced_k_mma_kernel<Schedule, Full, Output, Epilogue>;
                const int shared = q8_prepare_shared<Schedule::kSharedBytes, kernel>();
                kernel<<<grid, Schedule::kThreads, shared, stream>>>(operands, output, epilogue,
                                                                     offset, count);
                CUDA_CHECK(cudaGetLastError());
            };
            if (operands.rows % 16 == 0 && operands.k == operands.padded_k &&
                operands.k % Schedule::kBlockK == 0)
                launch.template operator()<true>();
            else
                launch.template operator()<false>();
        });
    }
}
} // namespace ninferno::ops::detail
