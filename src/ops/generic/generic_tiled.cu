// Generic tiled A16 schedules (generic_tiled.cuh): u4_g{32,64,128}_zp_fp16, ggml_q4_k, ggml_q5_k,
// q4/q5/q6_g64 and q3_g128/q3_g64; q8_g32 on Volta.
#include "ops/generic/generic_tiled.cuh"

#include "core/per_device.h"
#include "core/device.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/common/arch_limits.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

// Keep in the order of kTiledSchedules in generic_gemm.cpp: block_m (tokens), block_n (rows),
// warps_m, warps_n, stages (lowered where a codec's tiles would not fit).
#define NINFERNO_GENERIC_TILED_SCHEDULES(X)                                                        \
    X(0, 128, 128, 2, 4, 3)                                                                        \
    X(1, 64, 128, 2, 4, 3)                                                                         \
    X(2, 128, 64, 2, 2, 3)

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
        tiled_kernel<S, C, E><<<grid, S::kThreads, kShared, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class C>
bool codec_fits(int schedule) {
    switch (schedule) {
#define NINFERNO_GENERIC_FITS(I, BM, BN, WM, WN, ST)                                               \
    case I:                                                                                        \
        return tiled_schedule_fits<TiledSchedule<BM, BN, WM, WN, ST>, C>();
        NINFERNO_GENERIC_TILED_SCHEDULES(NINFERNO_GENERIC_FITS)
#undef NINFERNO_GENERIC_FITS
    default:
        return false;
    }
}

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

bool tiled_supports(WeightCodec codec) {
    switch (codec) {
    case WeightCodec::U4Zp:
    case WeightCodec::U4ZpG32:
    case WeightCodec::U4ZpG64:
    case WeightCodec::GgmlQ4K:
    case WeightCodec::GgmlQ5K:
    case WeightCodec::Q4:
    case WeightCodec::Q5:
    case WeightCodec::Q6:
    case WeightCodec::Q3G128:
    case WeightCodec::Q3G64:
        return true;
    default:
        return false;
    }
}

bool tiled_fits(int schedule, WeightCodec codec) {
    switch (codec) {
    case WeightCodec::U4Zp:
        return codec_fits<U4ZpTile>(schedule);
    case WeightCodec::U4ZpG32:
        return codec_fits<U4ZpG32Tile>(schedule);
    case WeightCodec::U4ZpG64:
        return codec_fits<U4ZpG64Tile>(schedule);
    case WeightCodec::GgmlQ4K:
        return codec_fits<GgmlQ45KTile<4>>(schedule);
    case WeightCodec::GgmlQ5K:
        return codec_fits<GgmlQ45KTile<5>>(schedule);
    case WeightCodec::Q4:
        return codec_fits<GroupwiseG64Tile<4>>(schedule);
    case WeightCodec::Q5:
        return codec_fits<GroupwiseG64Tile<5>>(schedule);
    case WeightCodec::Q6:
        return codec_fits<GroupwiseG64Tile<6>>(schedule);
    case WeightCodec::Q3G128:
        return codec_fits<GroupwiseQ3Tile<128>>(schedule);
    case WeightCodec::Q3G64:
        return codec_fits<GroupwiseQ3Tile<64>>(schedule);
    case WeightCodec::Q8:
        return codec_fits<Q8Tile>(schedule);
    default:
        return false;
    }
}

void launch_tiled(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                  cudaStream_t stream) {
    switch (codec) {
    case WeightCodec::U4Zp:
        launch_codec<U4ZpTile>(schedule, epilogue, args, stream);
        return;
    case WeightCodec::U4ZpG32:
        launch_codec<U4ZpG32Tile>(schedule, epilogue, args, stream);
        return;
    case WeightCodec::U4ZpG64:
        launch_codec<U4ZpG64Tile>(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ4K:
        launch_codec<GgmlQ45KTile<4>>(schedule, epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ5K:
        launch_codec<GgmlQ45KTile<5>>(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q4:
        launch_codec<GroupwiseG64Tile<4>>(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q5:
        launch_codec<GroupwiseG64Tile<5>>(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q6:
        launch_codec<GroupwiseG64Tile<6>>(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q3G128:
        launch_codec<GroupwiseQ3Tile<128>>(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q3G64:
        launch_codec<GroupwiseQ3Tile<64>>(schedule, epilogue, args, stream);
        return;
    case WeightCodec::Q8: // Volta only (volta_tiled_route)
        launch_codec<Q8Tile>(schedule, epilogue, args, stream);
        return;
    default:
        throw std::invalid_argument("generic tiled: weight format has no tiled codec");
    }
}

} // namespace ninferno::ops::detail::generic
