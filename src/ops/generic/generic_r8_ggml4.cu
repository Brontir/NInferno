// Generic R8 (W8A8 decode) for the 4- and 5-bit ggml blocks: Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL
// (32-value blocks) and IQ4_XS (32-value sub-blocks of a superblock, one scale each), MXFP4, and
// NVFP4 on a build without the native NVFP4 kernels (NINFERNO_R8_GGML4_CODECS, compiled_codecs.h).
#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_r8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_r8_ggml4(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                     cudaStream_t stream, W4a8Activation pre) {
    switch (codec) {
        NINFERNO_R8_GGML4_CODECS(NINFERNO_R8_LAUNCH_CASE)
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
