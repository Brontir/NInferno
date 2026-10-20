#pragma once

#include "core/device.h"
#include "core/device_class.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"
#include "ops/linear/fp8/fp8_a16_mma.cuh"
#include "ops/linear/fp8/fp8_a16_sliced_k_mma.cuh"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_tma_mma.cuh"

namespace ninferno::ops::detail {
template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_fp8_a16_gemv(const Fp8A16Operands& p, Output output, Epilogue epilogue,
                         cudaStream_t stream, Rows rows = {}) {
    validate_fp8_operands<Schedule>(p);
    if (p.tokens != 1 || p.rows % Schedule::kBlockRows || p.k % (32 * Schedule::kValuesPerLane))
        throw std::invalid_argument("FP8 GEMV requires T=1 and complete row/K tiles");
    // Where the device has PDL the kernel streams its first codes under the previous kernel.
    CUDA_CHECK(pdl::launch(
        {dim3(p.rows / Schedule::kBlockRows), dim3(Schedule::kThreads), 0, stream},
        device_class().features.pdl, false, fp8_a16_gemv_kernel<Schedule, Output, Epilogue, Rows>,
        p.x, p.codes, p.scales, output, epilogue, rows, p.rows, p.k));
}

template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_fp8_a16_simt(const Fp8A16Operands& p, Output output, Epilogue epilogue,
                         cudaStream_t stream, Rows rows = {}) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % (32 * Schedule::kValuesPerLane) ||
        (Schedule::kTokenCapacity && p.tokens > Schedule::kTokenCapacity) ||
        (Schedule::kExactTokens && p.tokens != Schedule::kTokenCapacity))
        throw std::invalid_argument(
            "FP8 SIMT requires complete row/K tiles and matching token capacity");
    if constexpr (requires { Epilogue::kRowTokens; }) {
        if (p.tokens != Epilogue::kRowTokens)
            throw std::invalid_argument("FP8 row epilogue requires its complete token interval");
    }
    const int capacity = Schedule::kTokenCapacity ? Schedule::kTokenCapacity : p.tokens;
    const int blocks   = p.rows / Schedule::kBlockRows * div_up(capacity, Schedule::kBlockTokens);
    fp8_a16_simt_kernel<Schedule>
        <<<blocks, Schedule::kThreads, 0, stream>>>(p, output, epilogue, rows);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_fp8_a16_mma(const Fp8A16Operands& p, Output output, Epilogue epilogue,
                        cudaStream_t stream, Rows rows = {}) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("FP8 A16 MMA requires complete row/K tiles");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        const auto launch = [&]<bool Full>() {
            static constexpr auto kernel = fp8_a16_mma_kernel<Schedule, Full, Output, Epilogue, Rows>;
            const int bytes =
                fp8_prepare_shared<fp8_mma_shared_bytes<Schedule, Epilogue>, kernel, true>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(
                p.x, p.codes, p.scales, output, epilogue, rows, p.rows, p.k, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

// Whether a two-stage tile over the build's shared memory fits single-staged (a schedule without a
// staging depth, a GEMV one, has no single-staged form).
template <class Schedule>
constexpr bool fp8_single_stage_fallback() {
    if constexpr (!Schedule::kFitsSharedLimit && Schedule::kStages == 2)
        return Schedule::template with_stages<1>::kFitsSharedLimit;
    else
        return false;
}

template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_fp8_a16_sliced_k_mma(const Fp8A16Operands& p, Output output, Epilogue epilogue,
                                 cudaStream_t stream, Rows rows = {}) {
    // NINFERNO-SPECIAL(arch:sm75): a two-stage tile over the build's shared memory (Turing's 64
    // KiB) runs single-staged where that fits: the same K split and sum order for every column.
    if constexpr (fp8_single_stage_fallback<Schedule>()) {
        launch_fp8_a16_sliced_k_mma<typename Schedule::template with_stages<1>>(p, output, epilogue,
                                                                                stream, rows);
    } else {
        validate_fp8_operands<Schedule>(p);
        constexpr int capacity =
            Schedule::kTokenCapacity ? Schedule::kTokenCapacity : Schedule::kBlockTokens;
        static_assert(capacity > 0 && capacity <= Schedule::kBlockTokens);
        if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
            (Schedule::kExactTokens && p.tokens != capacity))
            throw std::invalid_argument(
                "FP8 sliced-K requires complete row/K tiles and matching tokens");
        static constexpr auto kernel = fp8_a16_sliced_k_mma_kernel<Schedule, Output, Epilogue, Rows>;
        const int bytes       = fp8_prepare_shared<Schedule::kSharedBytes, kernel>();
        for_each_token_slice(p.tokens, capacity, [&](int offset, int count) {
            const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, capacity));
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p, output, epilogue, rows, offset);
            CUDA_CHECK(cudaGetLastError());
        });
    }
}

template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_fp8_a8_mma(const Fp8A8Operands& p, Output output, Epilogue epilogue,
                       cudaStream_t stream, Rows rows = {}) {
    // A tile over the build's shared memory is a build without the E4M3 mma (sm_75's 64 KiB), where
    // no route admits FP8 A8: refused without instantiating its kernel.
    if constexpr (fp8_mma_shared_bytes<Schedule, Epilogue> > kMaxSharedBytesPerBlock) {
        throw std::invalid_argument("FP8 A8: schedule exceeds this build's shared memory");
    } else {
        validate_fp8_operands<Schedule>(p);
        if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
            p.k / Schedule::kBlockK < Schedule::kStages)
            throw std::invalid_argument(
                "FP8 A8 MMA requires complete row/K tiles and enough K stages");
        for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const int blocks =
                p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
            const auto launch = [&]<bool Full>() {
                static constexpr auto kernel = fp8_a8_mma_kernel<Schedule, Full, Epilogue, Output, Rows>;
                const int bytes =
                    fp8_prepare_shared<fp8_mma_shared_bytes<Schedule, Epilogue>, kernel>();
                kernel<<<blocks, Schedule::kThreads, bytes, stream>>>(p, output, epilogue, rows,
                                                                      offset, count);
                CUDA_CHECK(cudaGetLastError());
            };
            if (count % Schedule::kBlockTokens == 0)
                launch.template operator()<true>();
            else
                launch.template operator()<false>();
        });
    }
}

// A TMA schedule on a device without TMA (sm_89) runs as the cp.async tile it extends, with the
// same static K; a split-K schedule runs its whole K loop there. NINFERNO-SPECIAL(arch:sm89)
template <class Schedule, class Output, class Epilogue, class RowPolicy = Fp8IdentityRows>
void launch_fp8_a8_tma_mma(const Fp8A8Operands& p, Output output, Epilogue epilogue,
                           cudaStream_t stream, float* partials = nullptr,
                           RowPolicy row_policy = {}) {
    // A TMA tile over the build's shared memory is a build for a device without TMA.
    if constexpr (Schedule::kFitsSharedLimit) {
        if (device_class().features.tma) {
            launch_fp8_a8_tma_mma_native<Schedule>(p, output, epilogue, stream, partials,
                                                   row_policy);
            return;
        }
    }
    using Base = typename Schedule::Base;
    if constexpr (Schedule::kStaticK > 0)
        launch_fp8_a8_mma<Fp8ScheduleInstance<Base, Schedule::kStaticK>>(p, output, epilogue,
                                                                         stream, row_policy);
    else
        launch_fp8_a8_mma<Base>(p, output, epilogue, stream, row_policy);
}
} // namespace ninferno::ops::detail
