// Generic R8 (W8A8 decode) for the K-quants Q4_K and Q5_K (a scale and a min per 32 values, two
// blocks per load), Q6_K (a scale per 16 values, blocks u and u + 2 per load), Q3_K (a scale per
// 16 values) and Q2_K (a scale and a min per 16 values), the last two the four blocks of a half
// superblock per load (NINFERNO_R8_KQUANT_CODECS, compiled_codecs.h).
#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_r8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_r8_kquant(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                      cudaStream_t stream, W4a8Activation pre) {
    switch (codec) {
        NINFERNO_R8_KQUANT_CODECS(NINFERNO_R8_LAUNCH_CASE)
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
