// Generic W4A8 for the formats whose int8 code is not a bit field: q8_g32 (GGUF Q8_0) and the
// codebook i-quants IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M (signed grid values)
// (NINFERNO_W4A8_CODEBOOK_CODECS, compiled_codecs.h).
#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_w4a8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_w4a8_codebook(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                          W4a8Activation act, cudaStream_t stream) {
    switch (codec) {
        NINFERNO_W4A8_CODEBOOK_CODECS(NINFERNO_W4A8_LAUNCH_CASE)
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
