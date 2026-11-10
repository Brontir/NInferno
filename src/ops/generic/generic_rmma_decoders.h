#pragma once

// The register-MMA decoders of the generic path as the host needs them (#620): per weight codec,
// the decoder's canonical name in the kernel-facts snapshots (kernel_facts.h decoder_name) and the
// traits that decide which kernel instances a launch may take (generic_gemm.cuh: UnpackOf,
// StageActsOf, SpreadOf, FinishOf, GroupTailOf, MinTermOf, WideOf). The launch reads the types; the
// host reads this table, so rmma_resources (generic_rmma_describe.cpp) describes a launch with no
// CUDA. The table is the types' truth as of a probe of the decoders (a program printing
// decoder_name and the traits of each launch_rmma<Decoder> of generic_gemm_<format>.cu);
// ninferno_rmma_describe_test compares the host's description with the launch's own for every
// codec, schedule and epilogue, so a decoder that changes a trait fails it.

#include "ops/generic/generic_gemm.h"

#include <cstdint>

namespace ninferno::ops::detail::generic {

struct RmmaDecoder {
    const char* name; // the decoder's key in the kernel-facts snapshots
    const char* wide; // the wide-load decoder's key; "" where none (WideOf)
    bool unpack;      // UnpackOf: load_raw() and unpack(): the latency variant's form
    bool stage_acts;  // StageActsOf (kStageActs)
    bool spread;      // SpreadOf: K split over lanes or a min term
    bool finish;      // FinishOf (kFinish)
    bool group_tail;  // GroupTailOf (kGroupTail)
    bool min_term;    // MinTermOf (kMinTerm): the add-sums variant's form
};

// The decoder launch_rmma_<format> instantiates for a call of K values; name == nullptr where the
// codec has no register-MMA launch. U4Zp's G32 and G64 take their quad-run decoder at K a multiple
// of 128.
inline RmmaDecoder rmma_decoder_of(WeightCodec codec, std::int32_t k) {
    switch (codec) {
    case WeightCodec::Bf16:
        return {"Bf16Decoder", "", false, false, false, false, false, false};
    case WeightCodec::Fp8:
        return {"Fp8Decoder", "", false, false, false, false, false, false};
    case WeightCodec::Nvfp4:
        return {"Nvfp4Decoder", "", false, false, false, false, false, false};
    case WeightCodec::Q4:
        return {"GroupwiseG64Decoder<4>", "", false, false, false, false, false, false};
    case WeightCodec::Q5:
        return {"GroupwiseG64Decoder<5>", "", false, false, false, false, false, false};
    case WeightCodec::Q6:
        return {"GroupwiseG64Decoder<6>", "", false, false, false, false, false, false};
    case WeightCodec::Q3G128:
        return {"GroupwiseQ3Decoder<128>", "", false, false, false, false, false, false};
    case WeightCodec::Q3G64:
        return {"GroupwiseQ3Decoder<64>", "", false, false, false, false, false, false};
    case WeightCodec::Q8:
        return {"Q8Decoder", "", false, false, false, false, false, false};
    case WeightCodec::Fp8RowF32:
        return {"Fp8RowF32Decoder", "", false, false, false, false, false, false};
    case WeightCodec::Fp8B128:
        return {"Fp8B128Decoder", "", false, false, false, false, false, false};
    case WeightCodec::U4Zp:
        return {"U4ZpDecoder", "", false, false, false, false, false, false};
    case WeightCodec::GgmlQ4K:
        return {"GgmlQ45KQuadDecoder<4>", "", true, true, true, false, false, true};
    case WeightCodec::GgmlQ5K:
        return {"GgmlQ45KQuadDecoder<5>", "", true, true, true, false, false, true};
    case WeightCodec::GgmlQ6K:
        return {"GgmlQ6KDecoder", "GgmlQ6KWideDecoder", false, true, true, false, false, false};
    case WeightCodec::GgmlQ3K:
        return {"GgmlQ3KDecoder", "", true, true, true, false, false, false};
    case WeightCodec::GgmlQ2K:
        return {"GgmlQ2KDecoder", "", false, false, true, false, false, true};
    case WeightCodec::GgmlIq4Xs:
        return {"GgmlIq4XsDecoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlIq3Xxs:
        return {"GgmlIq3XxsDecoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlIq3S:
        return {"GgmlIq3SDecoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlQ40:
        return {"Ggml32Decoder<0>", "", false, false, true, false, true, false};
    case WeightCodec::GgmlQ41:
        return {"Ggml32Decoder<1>", "", false, false, true, false, true, true};
    case WeightCodec::GgmlQ50:
        return {"Ggml32Decoder<2>", "", false, false, true, false, true, false};
    case WeightCodec::GgmlQ51:
        return {"Ggml32Decoder<3>", "", false, false, true, false, true, true};
    case WeightCodec::GgmlIq4Nl:
        return {"Ggml32Decoder<4>", "", false, false, true, false, true, false};
    case WeightCodec::GgmlIq2Xxs:
        return {"GgmlIq2XxsDecoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlIq2Xs:
        return {"GgmlIq2XsDecoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlIq2S:
        return {"GgmlIq2SDecoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlIq1S:
        return {"GgmlIq1SDecoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlIq1M:
        return {"GgmlIq1MDecoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlTq10:
        return {"GgmlTq10Decoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlTq20:
        return {"GgmlTq20Decoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlQ10:
        return {"GgmlQ10Decoder", "", false, false, true, false, false, false};
    case WeightCodec::GgmlQ20:
        return {"GgmlQ20Decoder", "", false, false, true, false, false, false};
    case WeightCodec::Mxfp4:
        return {"Mxfp4SpreadDecoder", "", false, false, true, false, true, false};
    case WeightCodec::U4ZpG32:
        return k % 128 == 0
                   ? RmmaDecoder{"U4ZpQuadDecoder<32>", "", false, false, true, false, false, false}
                   : RmmaDecoder{
                         "U4ZpSmallDecoder<32>", "", false, false, false, false, false, false};
    case WeightCodec::U4ZpG64:
        return k % 128 == 0
                   ? RmmaDecoder{"U4ZpQuadDecoder<64>", "", false, false, true, false, false, false}
                   : RmmaDecoder{
                         "U4ZpSmallDecoder<64>", "", false, false, false, false, false, false};
    default:
        return {nullptr, "", false, false, false, false, false, false};
    }
}

} // namespace ninferno::ops::detail::generic
