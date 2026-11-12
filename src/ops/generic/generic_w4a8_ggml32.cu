// Generic W4A8 for the ggml 32-value block formats: Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL,
// and MXFP4 (32-value blocks with an E8M0 exponent); NVFP4 (doubled E2M1 codes) where W4A4 does
// not take the weight (NINFERNO_W4A8_GGML32_CODECS, compiled_codecs.h).
#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_w4a8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_w4a8_ggml32(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                        W4a8Activation act, cudaStream_t stream) {
    switch (codec) {
        NINFERNO_W4A8_GGML32_CODECS(NINFERNO_W4A8_LAUNCH_CASE)
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
