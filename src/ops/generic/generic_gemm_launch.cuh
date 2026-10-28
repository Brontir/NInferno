#pragma once

// Compiled schedule lists and their launchers. Each format's translation unit instantiates
// launch_rmma<Decoder> once (generic_gemm_<format>.cu), so the whole list is compiled per format.

#include "core/device.h"
#include "core/device_class.h"
#include "core/per_device.h"
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

template <class Decoder, class Schedule, Epilogue E, bool Latency>
void launch_rmma_kernel(const GemmArgs& args, cudaStream_t stream);

// The schedules and decoders with a latency variant (rmma_kernel's Latency): one token tile of a
// K-quant whose heads unpack late (UnpackOf), SwiGLU of one that stages its activations, or one
// row set of one with the activation ring (kRmmaAheadForm: Q3_K and Q6_K, which have no load_raw;
// Q4_K / Q5_K's one-row-set variant takes the ring too). The one-row-set variant runs from
// kRmmaLatencyOneSetBytes and Q6_K's wide variant below kRmmaWideBytes: one board takes one.
static_assert(kRmmaWideBytes <= kRmmaLatencyOneSetBytes,
              "the Q6_K latency (ring) kernel would shadow the wide one");
template <class Decoder, class Schedule, Epilogue E>
inline constexpr bool kRmmaLatencyVariant =
    Schedule::kNTiles == 1 &&
    (UnpackOf<Decoder>::value || (StageActsOf<Decoder>::value && E == Epilogue::SwiGlu) ||
     (E != Epilogue::SwiGlu && kRmmaAheadForm<Decoder, Schedule, E>));

// The DRAM bytes per SM cycle (dram_bytes_per_sm_cycle: the board's facts) from which the plain
// K-quant kernel of one token tile is latency-bound. It cuts a row's scales out as it loads them,
// so a warp's prefetch issues row after row, each waiting for its own head; the latency variant
// issues them back to back. A/B on five boards, T = 5..8, Qwen3.5-9B shapes (bytes per SM cycle:
// RTX 4090 3.1, 4070 Ti 3.2, 5070 Ti 5.2, 2080 Ti 5.9, 3060 7.2):
// - SwiGLU (two row sets, the activations staged): the 4090 and 4070 Ti, whose plain kernel
//   streamed at 86-91 % of DRAM, lost 8-34 % to the variant (ncu, Q4_K 12288x4096 on the 4090:
//   lg_throttle 0.02 -> 6.9 stalls per issue, the queued loads); the 5070 Ti and 3060, at
//   64-74 %, gained 2-15 % and 11-25 %.
// - One row set (store, add; nothing staged): the 3060 gained 8-21 %, the 2080 Ti 0-4 %, and
//   the 5070 Ti between +14 % (Q3_K) and -3 % (Q4_K add 4096x12288 w1x4, 40.7 -> 41.7 us), so
//   its threshold lies above that board.
// Both thresholds lie between the boards measured on either side of them, not on one.
// (kRmmaLatencySwiGluBytes, kRmmaLatencyOneSetBytes: generic_gemm_internal.h.)

// Whether this device launches the latency variant: where the board is latency-bound for the
// epilogue (above), where SwiGLU's activations are staged with cp.async (sm_80+) or ride the
// activation ring (kRmmaAheadForm; on the RTX 2080 Ti the variant with neither left Q4_K SwiGLU
// 7-8 % slower, with the ring w1x4 took 168.9 -> 122.3 us at T = 8), and only where the variant
// holds as many CTAs per SM as the plain kernel (its registers: Q5_K's w1x4 at 129 against 127 on
// sm_120 dropped a quarter of the warps and lost 12-30 %, RTX 5070 Ti).
template <class Decoder, class Schedule, Epilogue E>
bool rmma_latency_pays() {
    static PerDevice<bool> pays;
    return pays.get([](int) {
        const bool swiglu = E == Epilogue::SwiGlu;
        if (dram_bytes_per_sm_cycle() <
            (swiglu ? kRmmaLatencySwiGluBytes : kRmmaLatencyOneSetBytes)) {
            return false;
        }
        const auto& d = device_class();
        // Volta's register MMA is emulated on FMA (no m16n8k16): its kernel is compute-bound,
        // and the variant only cost there (GV100, T 5..8: 0-5 % slower, never picked).
        if (d.major * 10 + d.minor < 75) return false;
        // Without cp.async (sm_75) SwiGLU's variant is the activation ring's form, or none.
        if (swiglu && d.major < 8 && !kRmmaAheadForm<Decoder, Schedule, E>) return false;
        const auto ctas = [](const void* kernel) {
            return kernel_resident_ctas_per_sm(kernel, Schedule::kThreads, 0);
        };
        return ctas(reinterpret_cast<const void*>(&rmma_kernel<Decoder, Schedule, E, true>)) >=
               ctas(reinterpret_cast<const void*>(&rmma_kernel<Decoder, Schedule, E, false>));
    });
}

// The schedules and decoders with a wide variant (WideOf: the wide-load Q6_K), one token tile.
template <class Decoder, class Schedule, Epilogue E>
inline constexpr bool kRmmaWideVariant =
    Schedule::kNTiles == 1 && !std::is_void_v<typename WideOf<Decoder>::type>;

// Whether this device launches the wide variant (rmma_wide_rule, generic_gemm_internal.h): the
// board's DRAM bytes per SM cycle and compute capability, and the two kernels' resident CTAs per
// SM and local memory (spills), read once per device.
template <class Decoder, class Schedule, Epilogue E>
bool rmma_wide_pays() {
    static PerDevice<bool> pays;
    return pays.get([](int) {
        using Wide        = typename WideOf<Decoder>::type;
        const auto& d     = device_class();
        const void* plain = reinterpret_cast<const void*>(&rmma_kernel<Decoder, Schedule, E>);
        const void* wide  = reinterpret_cast<const void*>(&rmma_kernel<Wide, Schedule, E>);
        const auto local  = [](const void* kernel) {
            cudaFuncAttributes attr{};
            CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
            return attr.localSizeBytes;
        };
        return rmma_wide_rule(dram_bytes_per_sm_cycle(), d.major * 10 + d.minor,
                              kernel_resident_ctas_per_sm(wide, Schedule::kThreads, 0),
                              kernel_resident_ctas_per_sm(plain, Schedule::kThreads, 0),
                              local(wide), local(plain));
    });
}

template <class Decoder, class Schedule, Epilogue E>
void launch_rmma_schedule(const GemmArgs& args, cudaStream_t stream) {
    if constexpr (kRmmaLatencyVariant<Decoder, Schedule, E>) {
        // The choice reads the device's facts and occupancy; where either fails this launch
        // takes the plain kernel (master's), and the next one asks again.
        bool pays = false;
        try {
            pays = rmma_latency_pays<Decoder, Schedule, E>();
        } catch (const std::exception&) {
        }
        if (pays) {
            launch_rmma_kernel<Decoder, Schedule, E, true>(args, stream);
            return;
        }
    }
    if constexpr (kRmmaWideVariant<Decoder, Schedule, E>) {
        // Where the latency variant does not run (it pays on the other side of the bandwidth per
        // SM); a failed read takes the plain kernel, as above.
        bool pays = false;
        try {
            pays = rmma_wide_pays<Decoder, Schedule, E>();
        } catch (const std::exception&) {
        }
        if (pays) {
            launch_rmma_kernel<typename WideOf<Decoder>::type, Schedule, E, false>(args, stream);
            return;
        }
    }
    launch_rmma_kernel<Decoder, Schedule, E, false>(args, stream);
}

template <class Decoder, class Schedule, Epilogue E, bool Latency>
void launch_rmma_kernel(const GemmArgs& args, cudaStream_t stream) {
    const dim3 grid((args.rows + Schedule::kRowsPerCta - 1) / Schedule::kRowsPerCta,
                    (args.tokens + Schedule::kTokensPerCta - 1) / Schedule::kTokensPerCta);
    if (grid.y > kMaxGridYZ)
        throw std::invalid_argument("generic linear: token extent exceeds the grid");
    const auto smem = static_cast<std::size_t>(args.spread_smem);
    if (Schedule::kCluster == 1 &&
        ((args.join_previous == 0 && args.pdl_prefetch == 0) || !pdl::supported())) {
        rmma_kernel<Decoder, Schedule, E, Latency>
            <<<grid, Schedule::kThreads, smem, stream>>>(args);
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
        CUDA_CHECK(cudaLaunchKernelEx(&config, rmma_kernel<Decoder, Schedule, E, Latency>, args));
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
