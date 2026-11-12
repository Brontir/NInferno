// Generic R8 (W8A8 decode) for IQ3_XXS and IQ3_S (NINFERNO_R8_IQ3_CODECS, compiled_codecs.h).
#include "ops/generic/compiled_codecs.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_r8.cuh"

namespace ninferno::ops::detail::generic {

bool launch_r8_iq3(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream, W4a8Activation pre) {
    switch (codec) {
        NINFERNO_R8_IQ3_CODECS(NINFERNO_R8_LAUNCH_CASE)
    default:
        return false;
    }
}

} // namespace ninferno::ops::detail::generic
