// Generic R8 (W8A8 decode) for IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M
// (NINFERNO_R8_IQ2_CODECS, compiled_codecs.h).
#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_r8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_r8_iq2(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream, W4a8Activation pre) {
    switch (codec) {
        NINFERNO_R8_IQ2_CODECS(NINFERNO_R8_LAUNCH_CASE)
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
