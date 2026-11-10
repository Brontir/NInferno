#pragma once

// Compiled schedule lists and their launchers. Each format's translation unit instantiates
// launch_rmma<Decoder> once (generic_gemm_<format>.cu), so the whole list is compiled per format.

#include "core/device.h"
#include "core/device_class.h"
#include "core/kernel_occupancy.h"
#include "core/per_device.h"
#include "ops/generic/generic_gemm.cuh"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/common/arch_limits.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/kernel_facts.h"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace ninferno::ops::detail::generic {

// The schedules: index, warps_m, warps_k, m_tiles,
// n_tiles, cluster (K split over a thread-block cluster of that many CTAs, generic_gemm.cuh).
// NINFERNO_GENERIC_RMMA_SCHEDULES: ops/generic/generic_schedules.h (one list for the
// instantiations here and the host table in generic_gemm.cpp).

template <class Decoder, class Schedule, Epilogue E, bool Latency, bool AddSums = false>
void launch_rmma_kernel(const GemmArgs& args, cudaStream_t stream);

// The schedules and decoders with a latency variant (rmma_kernel's Latency): one token tile of a
// K-quant whose heads unpack late (UnpackOf), or SwiGLU of one that stages its activations. The
// variant carries the activation ring where it has its form (kRmmaAhead). One row set of Q3_K /
// Q6_K (no load_raw) gets no ring-only variant: on the RTX 2080 Ti the Q6_K output head (248320 x
// 4096, store) at T = 1, whose plain kernel streams at 94 % of DRAM, took 1.52 -> 1.71 ms with it.
template <class Decoder, class Schedule, Epilogue E>
inline constexpr bool kRmmaLatencyVariant =
    Schedule::kNTiles == 1 &&
    (UnpackOf<Decoder>::value || (StageActsOf<Decoder>::value && E == Epilogue::SwiGlu));

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

// The CTAs a call of this schedule launches (grid x, y and the cluster's z).
template <class Schedule>
std::int64_t rmma_grid_ctas(const GemmArgs& args) {
    const std::int64_t x = (args.rows + Schedule::kRowsPerCta - 1) / Schedule::kRowsPerCta;
    const std::int64_t y = (args.tokens + Schedule::kTokensPerCta - 1) / Schedule::kTokensPerCta;
    return x * y * Schedule::kCluster;
}

// The key of a kernel instance in the kernel-facts snapshots (ops/generic/kernel_facts.h): the
// template arguments as the symbol spells them.
template <class Decoder, class Schedule, Epilogue E, bool Latency, bool AddSums>
kernel_facts::RmmaKernelKey rmma_kernel_key() {
    return {kernel_facts::decoder_name<Decoder>(),
            Schedule::kWarpsM,
            Schedule::kWarpsK,
            Schedule::kMTiles,
            Schedule::kNTiles,
            Schedule::kCluster,
            static_cast<int>(E),
            Latency,
            AddSums};
}

// The committed facts of a kernel instance (registers, stack, static shared memory: #612) for the
// kernels this thread's device runs (kernel_facts::sass_arch: this build's SASS, or the release
// engine of a substituted board), read once per instance and device (a static per template
// instantiation, bounded by the instances; PerDevice keeps a substitute's answer apart by its
// content). nullopt where no facts are known: the device runs PTX the driver compiled, there is
// no device, or the snapshot lacks the instance. The host GPU's own kernels are never asked
// (cudaFuncGetAttributes) except by the live check (kernel_facts::live_check_target).
template <class Decoder, class Schedule, Epilogue E, bool Latency, bool AddSums>
const std::optional<kernel_facts::KernelFacts>& rmma_kernel_facts() {
    static PerDevice<std::optional<kernel_facts::KernelFacts>> kept;
    return kept.get([](int) -> std::optional<kernel_facts::KernelFacts> {
        const auto arch = kernel_facts::sass_arch();
        if (!arch) return std::nullopt;
        return kernel_facts::rmma_facts(*arch,
                                        rmma_kernel_key<Decoder, Schedule, E, Latency, AddSums>());
    });
}

// The maintainer's live check of one instance (ninferno-serve selector-facts,
// ninferno_kernel_facts_live_test): the running GPU's attributes and occupancy of the kernel
// beside the snapshot's, recorded for the caller to compare.
template <class Decoder, class Schedule, Epilogue E, bool Latency, bool AddSums>
void rmma_kernel_facts_live(kernel_facts::LiveCheckSink& sink,
                            const std::optional<kernel_facts::KernelFacts>& snapshot,
                            std::size_t dynamic_smem) {
    const void* kernel =
        reinterpret_cast<const void*>(&rmma_kernel<Decoder, Schedule, E, Latency, AddSums>);
    cudaFuncAttributes a{};
    CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
    kernel_facts::LiveRead read;
    read.key                      = rmma_kernel_key<Decoder, Schedule, E, Latency, AddSums>();
    read.arch                     = kernel_facts::sass_arch().value_or("");
    read.snapshot                 = snapshot;
    read.live.registers           = a.numRegs;
    read.live.stack_bytes         = a.localSizeBytes;
    read.live.static_shared_bytes = a.sharedSizeBytes;
    read.threads                  = Schedule::kThreads;
    read.dynamic_shared_bytes     = dynamic_smem;
    read.live_ctas = kernel_resident_ctas_per_sm(kernel, Schedule::kThreads, dynamic_smem);
    read.snapshot_ctas =
        snapshot
            ? resident_ctas_per_sm(device_class(),
                                   KernelShape{snapshot->registers, snapshot->static_shared_bytes,
                                               Schedule::kThreads, dynamic_smem})
            : 0;
    sink.push_back(read);
}

// A variant kernel's and the plain kernel's resident CTAs per SM on the call's dynamic shared
// memory, from their committed facts and the device's limits (resident_ctas_per_sm, no GPU
// asked): the caller's slot holds them for none (every K-quant call), once per device. A kernel
// without facts holds no CTA (0: the variant never pays, rmma_variant_occupancy_rule).
template <class Schedule>
bool rmma_occupancy_kept(PerDevice<std::array<int, 2>>& slot,
                         const std::optional<kernel_facts::KernelFacts>& variant,
                         const std::optional<kernel_facts::KernelFacts>& plain,
                         const GemmArgs& args) {
    const auto smem     = static_cast<std::size_t>(args.spread_smem);
    const auto resident = [&](std::size_t bytes) {
        const auto& d   = device_class();
        const auto ctas = [&](const std::optional<kernel_facts::KernelFacts>& f) {
            if (!f) return 0;
            return resident_ctas_per_sm(
                d, KernelShape{f->registers, f->static_shared_bytes, Schedule::kThreads, bytes});
        };
        return std::array<int, 2>{ctas(variant), ctas(plain)};
    };
    const std::array<int, 2> ctas =
        smem == 0 ? slot.get([&](int) { return resident(0); }) : resident(smem);
    return rmma_variant_occupancy_rule(rmma_grid_ctas<Schedule>(args), device_class().sms,
                                       ctas[0], ctas[1]);
}

// Whether this device launches the latency variant: where the board is latency-bound for the
// epilogue (above), where SwiGLU's activations are staged with cp.async (sm_80+) or ride the
// activation ring (kRmmaAheadForm; on the RTX 2080 Ti the variant with neither left Q4_K SwiGLU
// 7-8 % slower, with the ring w1x4 took 168.9 -> 122.3 us at T = 8), and only where the call's
// grid takes no more waves at the variant's resident CTAs per SM than at the plain kernel's
// (rmma_variant_occupancy_rule: the registers a decoder edit moves change the choice only where
// they change the waves).
template <class Decoder, class Schedule, Epilogue E>
bool rmma_latency_board() {
    static PerDevice<bool> board;
    return board.get([](int) {
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
        return !(swiglu && d.major < 8 && !kRmmaAheadForm<Decoder, Schedule, E>);
    });
}

template <class Decoder, class Schedule, Epilogue E>
bool rmma_latency_pays(const GemmArgs& args) {
    static PerDevice<std::array<int, 2>> ctas;
    return rmma_latency_board<Decoder, Schedule, E>() &&
           rmma_occupancy_kept<Schedule>(
               ctas, rmma_kernel_facts<Decoder, Schedule, E, true, false>(),
               rmma_kernel_facts<Decoder, Schedule, E, false, false>(), args);
}

// The schedules and decoders with a wide variant (WideOf: the wide-load Q6_K), one token tile.
template <class Decoder, class Schedule, Epilogue E>
inline constexpr bool kRmmaWideVariant =
    Schedule::kNTiles == 1 && !std::is_void_v<typename WideOf<Decoder>::type>;

// Whether this device launches the wide variant (rmma_wide_rule, generic_gemm_internal.h): the
// board's DRAM bytes per SM cycle and compute capability and the two kernels' stack bytes
// (spills) from their committed facts, and the call's waves at either kernel's resident CTAs per
// SM. A kernel without facts never pays.
template <class Decoder, class Schedule, Epilogue E>
bool rmma_wide_pays(const GemmArgs& args) {
    using Wide        = typename WideOf<Decoder>::type;
    const auto& plain = rmma_kernel_facts<Decoder, Schedule, E, false, false>();
    const auto& wide  = rmma_kernel_facts<Wide, Schedule, E, false, false>();
    if (!plain || !wide) return false;
    const auto& d = device_class();
    // The board's facts first: the occupancy only where they admit the wide kernel.
    if (!rmma_wide_rule(dram_bytes_per_sm_cycle(), d.major * 10 + d.minor, true,
                        wide->spill_bytes(), plain->spill_bytes())) {
        return false;
    }
    static PerDevice<std::array<int, 2>> ctas;
    return rmma_occupancy_kept<Schedule>(ctas, wide, plain, args);
}

template <class Decoder, class Schedule, Epilogue E>
void launch_rmma_schedule(const GemmArgs& args, cudaStream_t stream) {
    if constexpr (kRmmaAddVariant<Decoder, Schedule>) {
        // The call's own choice (run_route: rmma_add_sums), not the board's.
        if (args.add_sums != 0) {
            launch_rmma_kernel<Decoder, Schedule, E, false, true>(args, stream);
            return;
        }
    }
    if constexpr (kRmmaLatencyVariant<Decoder, Schedule, E>) {
        // The choice reads the device's facts and occupancy; where either fails this launch
        // takes the plain kernel (master's), and the next one asks again.
        bool pays = false;
        try {
            pays = rmma_latency_pays<Decoder, Schedule, E>(args);
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
            pays = rmma_wide_pays<Decoder, Schedule, E>(args);
        } catch (const std::exception&) {
        }
        if (pays) {
            launch_rmma_kernel<typename WideOf<Decoder>::type, Schedule, E, false>(args, stream);
            return;
        }
    }
    launch_rmma_kernel<Decoder, Schedule, E, false>(args, stream);
}

template <class Decoder, class Schedule, Epilogue E, bool Latency, bool AddSums>
void launch_rmma_kernel(const GemmArgs& args, cudaStream_t stream) {
    const dim3 grid((args.rows + Schedule::kRowsPerCta - 1) / Schedule::kRowsPerCta,
                    (args.tokens + Schedule::kTokensPerCta - 1) / Schedule::kTokensPerCta);
    if (grid.y > kMaxGridYZ)
        throw std::invalid_argument("generic linear: token extent exceeds the grid");
    const auto smem = static_cast<std::size_t>(args.spread_smem);
    if (KernelResources* described = kernel_describe_target()) {
        // The kernel's committed facts (the cost model describes up to 20 schedules a call: one
        // snapshot read per instance and device). Without them nothing is described: the caller
        // prices by threads alone.
        const auto& facts = rmma_kernel_facts<Decoder, Schedule, E, Latency, AddSums>();
        if (kernel_facts::LiveCheckSink* live = kernel_facts::live_check_target()) {
            rmma_kernel_facts_live<Decoder, Schedule, E, Latency, AddSums>(*live, facts, smem);
        }
        if (!facts) return;
        describe_kernel(*described, *facts, dim3(grid.x, grid.y, Schedule::kCluster),
                        Schedule::kThreads, smem, Latency);
        return;
    }
    if (Schedule::kCluster == 1 &&
        ((args.join_previous == 0 && args.pdl_prefetch == 0) || !pdl::supported())) {
        rmma_kernel<Decoder, Schedule, E, Latency, AddSums>
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
        CUDA_CHECK(
            cudaLaunchKernelEx(&config, rmma_kernel<Decoder, Schedule, E, Latency, AddSums>, args));
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
