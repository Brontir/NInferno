// Generic W4A8 for the formats with a scale per 32 (Q6_K, Q3_K, Q2_K: 16) values inside larger
// blocks: IQ4_XS, Q4_K, Q5_K, Q6_K, Q3_K, Q2_K (256-value superblocks), u4_g128_zp (128-value
// groups), q4/q5/q6_g64 (64-value groups) and q3_g128 / q3_g64 (128- and 64-value groups)
// (NINFERNO_W4A8_SUPERBLOCK_CODECS, compiled_codecs.h).
#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_w4a8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_w4a8_superblock(int schedule, WeightCodec codec, Epilogue epilogue,
                            const GemmArgs& args, W4a8Activation act, cudaStream_t stream) {
    switch (codec) {
        NINFERNO_W4A8_SUPERBLOCK_CODECS(NINFERNO_W4A8_LAUNCH_CASE)
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
