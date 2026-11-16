// Generic tiled A16 schedules (generic_tiled.cuh): u4_g{32,64,128}_zp_fp16, ggml_q4_k, ggml_q5_k,
// q4/q5/q6_g64 and q3_g128/q3_g64; q8_g32 on Volta (NINFERNO_TILED_CODECS and
// NINFERNO_TILED_VOLTA_CODECS, compiled_codecs.h).
#include "ops/generic/generic_tiled.cuh"

#include "core/per_device.h"
#include "core/device.h"
#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/common/arch_limits.h"
#include "ops/generic/generic_schedules.h"
#include "ops/generic/generic_shared_bytes.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

// The schedules: block_m (tokens), block_n (rows),
// warps_m, warps_n, stages (lowered where a codec's tiles would not fit).
// NINFERNO_GENERIC_TILED_SCHEDULES: ops/generic/generic_schedules.h (one list for the
// instantiations here and the host table in generic_schedule_tables.h).

template <class S, class C, Epilogue E>
void launch_schedule(const GemmArgs& args, cudaStream_t stream) {
    constexpr int kRowsPerBlock = E == Epilogue::SwiGlu ? S::kBlockN / 2 : S::kBlockN;
    constexpr int kShared       = tiled_shared_bytes<S, C>();
    if constexpr (!tiled_schedule_fits<S, C>()) {
        // route_admits never names it (tiled_fits); a direct call is refused.
        throw std::invalid_argument("generic tiled: schedule exceeds this build's shared memory");
    } else {
        static ::ninferno::PerDevice<bool> configured;
        ::ninferno::allow_dynamic_smem(configured, tiled_kernel<S, C, E>, kShared);
        const dim3 grid((args.rows + kRowsPerBlock - 1) / kRowsPerBlock,
                        (args.tokens + S::kBlockM - 1) / S::kBlockM);
        if (grid.y > kMaxGridYZ)
            throw std::invalid_argument("generic tiled: token extent exceeds the grid");
        if (KernelResources* described = kernel_describe_target()) {
            cudaFuncAttributes attributes{};
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, (tiled_kernel<S, C, E>)));
            describe_kernel(*described, attributes, grid, S::kThreads, kShared, false);
            return;
        }
        tiled_kernel<S, C, E><<<grid, S::kThreads, kShared, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
    }
}

// generic_shared_bytes.h's table of this build's tiled layouts (tiled_fits reads it with no CUDA
// header), held equal to the layouts' own sizes, and their activation stages to the family's
// record (tiled_stage_counted).
template <class C>
constexpr bool tiled_table_holds(int codec_index) {
    bool holds = true;
#define NINFERNO_GENERIC_HOLDS(I, BM, BN, WM, WN, ST)                                              \
    holds = holds && tiled_shared_bytes<TiledSchedule<BM, BN, WM, WN, ST>, C>() ==                 \
                         tiled_shared_table(codec_index, I);                                       \
    holds = holds && tiled_stage_counted<TiledSchedule<BM, BN, WM, WN, ST>, C>();
    NINFERNO_GENERIC_TILED_SCHEDULES(NINFERNO_GENERIC_HOLDS)
#undef NINFERNO_GENERIC_HOLDS
    return holds;
}

// Every codec of the tiled lists (compiled_codecs.h).
#define NINFERNO_TILED_TABLE_HOLDS(CODEC, TYPE)                                                    \
    static_assert(tiled_table_holds<TYPE>(tiled_codec_index(WeightCodec::CODEC)));
NINFERNO_TILED_CODECS(NINFERNO_TILED_TABLE_HOLDS)
NINFERNO_TILED_VOLTA_CODECS(NINFERNO_TILED_TABLE_HOLDS)
#undef NINFERNO_TILED_TABLE_HOLDS

template <class C, Epilogue E>
void launch_epilogue(int schedule, const GemmArgs& args, cudaStream_t stream) {
    switch (schedule) {
#define NINFERNO_GENERIC_CASE(I, BM, BN, WM, WN, ST)                                               \
    case I:                                                                                        \
        launch_schedule<TiledSchedule<BM, BN, WM, WN, ST>, C, E>(args, stream);                    \
        return;
        NINFERNO_GENERIC_TILED_SCHEDULES(NINFERNO_GENERIC_CASE)
#undef NINFERNO_GENERIC_CASE
    default:
        throw std::invalid_argument("generic tiled: unknown schedule");
    }
}

template <class C>
void launch_codec(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    switch (epilogue) {
    case Epilogue::Store:
        launch_epilogue<C, Epilogue::Store>(schedule, args, stream);
        return;
    case Epilogue::AddResidual:
        launch_epilogue<C, Epilogue::AddResidual>(schedule, args, stream);
        return;
    case Epilogue::SwiGlu:
        launch_epilogue<C, Epilogue::SwiGlu>(schedule, args, stream);
        return;
    }
    throw std::invalid_argument("generic tiled: unknown epilogue");
}

} // namespace


void launch_tiled(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                  cudaStream_t stream) {
    switch (codec) {
#define NINFERNO_TILED_LAUNCH_CASE(CODEC, TYPE)                                                    \
    case WeightCodec::CODEC:                                                                       \
        launch_codec<TYPE>(schedule, epilogue, args, stream);                                      \
        return;
        NINFERNO_TILED_CODECS(NINFERNO_TILED_LAUNCH_CASE)
        NINFERNO_TILED_VOLTA_CODECS(NINFERNO_TILED_LAUNCH_CASE) // Volta only (volta_tiled_route)
#undef NINFERNO_TILED_LAUNCH_CASE
    default:
        throw std::invalid_argument("generic tiled: weight format has no tiled codec");
    }
}

} // namespace ninferno::ops::detail::generic
