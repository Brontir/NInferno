// Generic NVFP4 A16 in shared-memory activation tiles: nvfp4_a16_tiled.cuh on a runtime problem,
// Bytes = 8 (the register MMA's 16-value NVFP4 run) and Classes = the register MMA's WarpsK, so a
// call's bytes are those of rmma_kernel<Nvfp4Decoder> with that split (generic_gemm.cuh): the
// same K steps, the same k slots, the same meeting of partials and the same epilogue arithmetic.
#include "ops/generic/generic_gemm.cuh"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/linear/nvfp4/nvfp4_a16_tiled.cuh"
#include "ops/common/arch_limits.h"

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

// Keep in the order of kNvfp4TiledSchedules in generic_gemm.cpp: four warps of two m16 tiles by
// 64 tokens, three activation steps in flight.
template <int Classes>
using Nvfp4TiledSchedule = Nvfp4A16TiledSchedule<4, 1, 2, 8, 3, 1, Classes, 8>;

// The register MMA's epilogue on the tiled kernel's sums (it passes them through unscaled): the
// row's weight divisor, SwiGLU's SiLU(gate) * up, then store_output.
template <Epilogue E>
struct GenericNvfp4Output {
    GemmArgs a;
    __device__ __forceinline__ void store(int row, int token, float value) const {
        store_output<E>(a, row, token, value * Nvfp4Decoder::row_scale(a, a.row_begin + row));
    }
    __device__ __forceinline__ void store_pair(int row, int token, float gate, float up) const {
        const float scale0 = Nvfp4Decoder::row_scale(a, a.row_begin + row);
        const float scale1 = Nvfp4Decoder::row_scale(a, a.row_begin + row + a.up_offset);
        store_output<E>(a, row, token, generic_gate(a, gate * scale0) * (up * scale1));
    }
};

template <int Classes, Epilogue E>
void launch_classes(const GemmArgs& args, cudaStream_t stream) {
    constexpr bool kPaired = E == Epilogue::SwiGlu;
    const Nvfp4A16RuntimeProblem<kPaired> problem{args.k, args.ldx, args.rows, args.row_begin,
                                                  args.up_offset};
    if ((args.tokens + Nvfp4TiledSchedule<Classes>::kTokens - 1) /
            Nvfp4TiledSchedule<Classes>::kTokens >
        kMaxGridYZ) {
        throw std::invalid_argument("generic nvfp4 tiled: token extent exceeds the grid");
    }
    launch_nvfp4_a16_tiled_problem<Nvfp4A16RuntimeProblem<kPaired>, Nvfp4TiledSchedule<Classes>>(
        problem, static_cast<const __nv_bfloat16*>(args.x), args.codes, args.scales, 1.0F,
        LinearIdentityEpilogue{}, GenericNvfp4Output<E>{args}, args.tokens, stream);
}

template <int Classes>
void launch_epilogue(Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    switch (epilogue) {
    case Epilogue::Store:
        launch_classes<Classes, Epilogue::Store>(args, stream);
        return;
    case Epilogue::AddResidual:
        launch_classes<Classes, Epilogue::AddResidual>(args, stream);
        return;
    case Epilogue::SwiGlu:
        launch_classes<Classes, Epilogue::SwiGlu>(args, stream);
        return;
    }
    throw std::invalid_argument("generic nvfp4 tiled: unknown epilogue");
}

} // namespace

void launch_nvfp4_tiled(int classes, Epilogue epilogue, const GemmArgs& args,
                        cudaStream_t stream) {
    switch (classes) {
    case 1:
        launch_epilogue<1>(epilogue, args, stream);
        return;
    case 4:
        launch_epilogue<4>(epilogue, args, stream);
        return;
    case 8:
        launch_epilogue<8>(epilogue, args, stream);
        return;
    default:
        throw std::invalid_argument("generic nvfp4 tiled: classes must be 1, 4 or 8");
    }
}

} // namespace ninferno::ops::detail::generic
