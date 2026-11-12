#pragma once

// What this binary compiles for each codec (#615 F4): one X-macro list per launch file, the one
// place a codec's kernels are named. Each launch file's codec switch is generated from its list
// (the case labels and the template arguments they instantiate), and the host reads the same lists
// as constexpr arrays (compiled): the admission predicates (r8_supports, w4a8_supports,
// r8_verify_compiled, w4a8_ws_supports, tiled_supports, volta_hmma_codec, volta_gemv_takes) and the
// build manifest (build_manifest). A codec added to a list is a kernel and an admission at once; a
// codec the host admits without a kernel cannot be written.
//
// An entry is X(CODEC, TYPE) with CODEC a WeightCodec enumerator and TYPE the codec struct (or
// decoder) the launch file instantiates for it; the R8 lists add PREQUANT, whether the codec also
// has R8's pre-quantized verify form (r8_kernel's kPre, generic_r8.cuh). TYPE is spelled only where
// a launch file expands the list (its own includes name it); host code reads CODEC and PREQUANT.
//
// Not listed: generic_volta_cutlass.cu decodes every codec (launch_decode_rows over the whole
// enum); which formats take it is a device rule (volta_cutlass_only), not a list.

#include "core/format_facts.h"

#include <array>
#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------------------------
// R8 (W8A8 decode), X(CODEC, TYPE, PREQUANT).
// ---------------------------------------------------------------------------------------------

// generic_r8.cu: q8_g32, q4/q5/q6_g64, u4_g{128,32,64}_zp, q3_g128 / q3_g64. The u4 zero-point
// formats decode as in W4A8 (W4a8U4ZpG): before, a u4 checkpoint's decode stayed on the A16
// register MMA wherever its other formats took R8. U4 G32 is the G128 codec with its group index.
#define NINFERNO_R8_BASE_CODECS(X)                                                                 \
    X(Q8, W4a8Q8, true)                                                                            \
    X(Q4, W4a8G64<4>, true)                                                                        \
    X(Q5, W4a8G64<5>, true)                                                                        \
    X(Q6, W4a8G64<6>, true)                                                                        \
    X(U4Zp, W4a8U4Zp, true)                                                                        \
    X(U4ZpG32, W4a8U4ZpG32, true)                                                                  \
    X(U4ZpG64, W4a8U4ZpG64, true)                                                                  \
    X(Q3G128, W4a8Q3<128>, true)                                                                   \
    X(Q3G64, W4a8Q3<64>, true)

// generic_r8_iq3.cu: IQ3_XXS, IQ3_S.
#define NINFERNO_R8_IQ3_CODECS(X)                                                                  \
    X(GgmlIq3Xxs, W4a8Iq3Xxs, true)                                                                \
    X(GgmlIq3S, W4a8Iq3S, true)

// generic_r8_iq2.cu: IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M. IQ1_S / IQ1_M decode as in W4A8
// (W4a8Iq1S / W4a8Iq1M): before, their decode stayed on the A16 register MMA (27B UD-IQ2_XXS on
// the RTX 2080 Ti, nsys: the IQ1_S rmma calls 7.2 + 4.0 + 2.6 ms of a 45 ms token).
#define NINFERNO_R8_IQ2_CODECS(X)                                                                  \
    X(GgmlIq2Xxs, W4a8Iq2Xxs, true)                                                                \
    X(GgmlIq2Xs, W4a8Iq2Xs, true)                                                                  \
    X(GgmlIq2S, W4a8Iq2S, true)                                                                    \
    X(GgmlIq1S, W4a8Iq1S, true)                                                                    \
    X(GgmlIq1M, W4a8Iq1M, true)

// NINFERNO-SPECIAL(arch:no-fp4-mma): without the native NVFP4 kernels (kNativeNvfp4 false,
// ops/common/native_nvfp4.h) decode multiplies the doubled E2M1 codes on dp4a (W4a8Nvfp4, as the
// W4A8 prefill) instead of the BF16 register MMA, so R8 has NVFP4 only there; sm_120a keeps its
// native decode. The int8 tier multiplies the doubled E2M1 codes exactly on every build
// (NINFERNO_W4A8_GGML32_CODECS): with the native kernels W4A4 keeps the weights it takes
// (allows_w4a8), and W4A8 runs the prefill of those whose policy allows A8 but not A4.
#ifdef NINFERNO_WITHOUT_NVFP4
#    define NINFERNO_R8_NVFP4_CODECS(X) X(Nvfp4, W4a8Nvfp4, true)
#else
#    define NINFERNO_R8_NVFP4_CODECS(X)
#endif

// generic_r8_ggml4.cu: Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, IQ4_XS, MXFP4, and NVFP4 without the native
// NVFP4 kernels.
#define NINFERNO_R8_GGML4_CODECS(X)                                                                \
    X(GgmlQ40, W4a8Ggml32<Ggml32::Q4_0>, true)                                                     \
    X(GgmlQ41, W4a8Ggml32<Ggml32::Q4_1>, true)                                                     \
    X(GgmlQ50, W4a8Ggml32<Ggml32::Q5_0>, true)                                                     \
    X(GgmlQ51, W4a8Ggml32<Ggml32::Q5_1>, true)                                                     \
    X(GgmlIq4Nl, W4a8Ggml32<Ggml32::IQ4_NL>, true)                                                 \
    X(GgmlIq4Xs, W4a8Iq4Xs, true)                                                                  \
    X(Mxfp4, W4a8Mxfp4, true)                                                                      \
    NINFERNO_R8_NVFP4_CODECS(X)

// generic_r8_kquant.cu: Q4_K, Q5_K, Q6_K; Q3_K and Q2_K without the pre-quantized form (their
// verify calls take the register MMA).
#define NINFERNO_R8_KQUANT_CODECS(X)                                                               \
    X(GgmlQ4K, W4a8Q45K<4>, true)                                                                  \
    X(GgmlQ5K, W4a8Q45K<5>, true)                                                                  \
    X(GgmlQ6K, W4a8Q6K, true)                                                                      \
    X(GgmlQ3K, W4a8Q3K, false)                                                                     \
    X(GgmlQ2K, W4a8Q2K, false)

// Every R8 list, in launch_r8's order.
#define NINFERNO_R8_CODECS(X)                                                                      \
    NINFERNO_R8_BASE_CODECS(X)                                                                     \
    NINFERNO_R8_IQ3_CODECS(X)                                                                      \
    NINFERNO_R8_IQ2_CODECS(X)                                                                      \
    NINFERNO_R8_GGML4_CODECS(X)                                                                    \
    NINFERNO_R8_KQUANT_CODECS(X)

// ---------------------------------------------------------------------------------------------
// W4A8 (the int8 prefill tiles), X(CODEC, TYPE).
// ---------------------------------------------------------------------------------------------

// generic_w4a8_ggml32.cu: Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4; NVFP4 on every build (on sm_120a
// the prefill of a weight W4A4 does not take, allows_w4a8).
#define NINFERNO_W4A8_GGML32_CODECS(X)                                                             \
    X(GgmlQ40, W4a8Ggml32<Ggml32::Q4_0>)                                                           \
    X(GgmlQ41, W4a8Ggml32<Ggml32::Q4_1>)                                                           \
    X(GgmlQ50, W4a8Ggml32<Ggml32::Q5_0>)                                                           \
    X(GgmlQ51, W4a8Ggml32<Ggml32::Q5_1>)                                                           \
    X(GgmlIq4Nl, W4a8Ggml32<Ggml32::IQ4_NL>)                                                       \
    X(Mxfp4, W4a8Mxfp4)                                                                            \
    X(Nvfp4, W4a8Nvfp4)

// generic_w4a8_superblock.cu: IQ4_XS and the K-quants (256-value superblocks), u4_g{128,32,64}_zp,
// q4/q5/q6_g64, q3_g128 / q3_g64.
#define NINFERNO_W4A8_SUPERBLOCK_CODECS(X)                                                         \
    X(GgmlIq4Xs, W4a8Iq4Xs)                                                                        \
    X(GgmlQ4K, W4a8Q45K<4>)                                                                        \
    X(GgmlQ5K, W4a8Q45K<5>)                                                                        \
    X(GgmlQ6K, W4a8Q6K)                                                                            \
    X(GgmlQ3K, W4a8Q3K)                                                                            \
    X(GgmlQ2K, W4a8Q2K)                                                                            \
    X(U4Zp, W4a8U4Zp)                                                                              \
    X(U4ZpG32, W4a8U4ZpG32)                                                                        \
    X(U4ZpG64, W4a8U4ZpG64)                                                                        \
    X(Q4, W4a8G64<4>)                                                                              \
    X(Q5, W4a8G64<5>)                                                                              \
    X(Q6, W4a8G64<6>)                                                                              \
    X(Q3G128, W4a8Q3<128>)                                                                         \
    X(Q3G64, W4a8Q3<64>)

// generic_w4a8_codebook.cu: q8_g32 and the codebook i-quants (signed grid values).
#define NINFERNO_W4A8_CODEBOOK_CODECS(X)                                                           \
    X(Q8, W4a8Q8)                                                                                  \
    X(GgmlIq3Xxs, W4a8Iq3Xxs)                                                                      \
    X(GgmlIq3S, W4a8Iq3S)                                                                          \
    X(GgmlIq2Xxs, W4a8Iq2Xxs)                                                                      \
    X(GgmlIq2Xs, W4a8Iq2Xs)                                                                        \
    X(GgmlIq2S, W4a8Iq2S)                                                                          \
    X(GgmlIq1S, W4a8Iq1S)                                                                          \
    X(GgmlIq1M, W4a8Iq1M)

// Every W4A8 list, in launch_w4a8's order.
#define NINFERNO_W4A8_CODECS(X)                                                                    \
    NINFERNO_W4A8_GGML32_CODECS(X)                                                                 \
    NINFERNO_W4A8_SUPERBLOCK_CODECS(X)                                                             \
    NINFERNO_W4A8_CODEBOOK_CODECS(X)

// The codecs the producer-warp tile (w4a8_ws_kernel, generic_w4a8.cuh) is compiled for, in the
// launch files above: w4a8_ws_compiled<C> reads this list, and generic_codec_traits.cu holds each
// entry's TYPE to the W4A8 list's TYPE of its codec.
#define NINFERNO_W4A8_WS_CODECS(X)                                                                 \
    X(GgmlIq3Xxs, W4a8Iq3Xxs)                                                                      \
    X(GgmlIq2S, W4a8Iq2S)                                                                          \
    X(GgmlQ4K, W4a8Q45K<4>)                                                                        \
    X(GgmlQ6K, W4a8Q6K)                                                                            \
    X(GgmlQ40, W4a8Ggml32<Ggml32::Q4_0>)                                                           \
    X(GgmlIq4Xs, W4a8Iq4Xs)                                                                        \
    X(GgmlIq2Xs, W4a8Iq2Xs)                                                                        \
    X(GgmlIq2Xxs, W4a8Iq2Xxs)                                                                      \
    X(GgmlIq3S, W4a8Iq3S)

// ---------------------------------------------------------------------------------------------
// The A16 shared-memory tiles and Volta's kernels, X(CODEC, TYPE).
// ---------------------------------------------------------------------------------------------

// generic_tiled.cu: the codecs of the tiled route on every build (tiled_supports).
#define NINFERNO_TILED_CODECS(X)                                                                   \
    X(U4Zp, U4ZpTile)                                                                              \
    X(U4ZpG32, U4ZpG32Tile)                                                                        \
    X(U4ZpG64, U4ZpG64Tile)                                                                        \
    X(GgmlQ4K, GgmlQ45KTile<4>)                                                                    \
    X(GgmlQ5K, GgmlQ45KTile<5>)                                                                    \
    X(Q4, GroupwiseG64Tile<4>)                                                                     \
    X(Q5, GroupwiseG64Tile<5>)                                                                     \
    X(Q6, GroupwiseG64Tile<6>)                                                                     \
    X(Q3G128, GroupwiseQ3Tile<128>)                                                                \
    X(Q3G64, GroupwiseQ3Tile<64>)

// generic_tiled.cu: the codecs it compiles for Volta's tiled route alone (tiled_takes: where
// tensor mma.sync is missing; elsewhere they have kernels of their own).
#define NINFERNO_TILED_VOLTA_CODECS(X) X(Q8, Q8Tile)

// generic_volta_hmma.cu: Volta's tensor-core route of the verify widths (volta_hmma_codec).
#define NINFERNO_VOLTA_HMMA_CODECS(X)                                                              \
    X(GgmlQ4K, HK45<4>)                                                                            \
    X(GgmlQ5K, HK45<5>)                                                                            \
    X(GgmlQ6K, HK6)                                                                                \
    X(Q8, HQ8)

// generic_volta_gemv.cu: Volta's warp-per-row GEMV (volta_gemv_takes).
#define NINFERNO_VOLTA_GEMV_CODECS(X)                                                              \
    X(GgmlQ4K, K45<4>)                                                                             \
    X(GgmlQ5K, K45<5>)                                                                             \
    X(GgmlQ6K, K6)                                                                                 \
    X(Q8, Q8G32)                                                                                   \
    X(Bf16, Bf16Rows)

namespace ninferno::ops::detail::generic {

// A kernel variant a list above compiles a codec for.
enum class Variant : std::uint8_t {
    R8,         // the R8 lists
    R8Prequant, // the R8 entries with PREQUANT (r8_verify_compiled)
    W4a8,       // the W4A8 lists
    W4a8Ws,     // NINFERNO_W4A8_WS_CODECS
    Tiled,      // NINFERNO_TILED_CODECS
    TiledVolta, // NINFERNO_TILED_VOLTA_CODECS
    VoltaHmma,  // NINFERNO_VOLTA_HMMA_CODECS
    VoltaGemv,  // NINFERNO_VOLTA_GEMV_CODECS
};

struct R8CompiledCodec {
    WeightCodec codec;
    bool prequant;
};

namespace compiled_lists {
#define NINFERNO_COMPILED_R8_ROW(CODEC, TYPE, PREQUANT)                                            \
    R8CompiledCodec{WeightCodec::CODEC, PREQUANT},
#define NINFERNO_COMPILED_ROW(CODEC, TYPE) WeightCodec::CODEC,
inline constexpr std::array kR8 =
    std::to_array<R8CompiledCodec>({NINFERNO_R8_CODECS(NINFERNO_COMPILED_R8_ROW)});
inline constexpr std::array kW4a8 =
    std::to_array<WeightCodec>({NINFERNO_W4A8_CODECS(NINFERNO_COMPILED_ROW)});
inline constexpr std::array kW4a8Ws =
    std::to_array<WeightCodec>({NINFERNO_W4A8_WS_CODECS(NINFERNO_COMPILED_ROW)});
inline constexpr std::array kTiled =
    std::to_array<WeightCodec>({NINFERNO_TILED_CODECS(NINFERNO_COMPILED_ROW)});
inline constexpr std::array kTiledVolta =
    std::to_array<WeightCodec>({NINFERNO_TILED_VOLTA_CODECS(NINFERNO_COMPILED_ROW)});
inline constexpr std::array kVoltaHmma =
    std::to_array<WeightCodec>({NINFERNO_VOLTA_HMMA_CODECS(NINFERNO_COMPILED_ROW)});
inline constexpr std::array kVoltaGemv =
    std::to_array<WeightCodec>({NINFERNO_VOLTA_GEMV_CODECS(NINFERNO_COMPILED_ROW)});
#undef NINFERNO_COMPILED_ROW
#undef NINFERNO_COMPILED_R8_ROW

constexpr WeightCodec codec_of_entry(WeightCodec codec) noexcept { return codec; }

constexpr WeightCodec codec_of_entry(const R8CompiledCodec& entry) noexcept { return entry.codec; }

template <class List>
constexpr bool holds(const List& list, WeightCodec codec) noexcept {
    for (const auto& entry : list) {
        if (codec_of_entry(entry) == codec) return true;
    }
    return false;
}

// Each codec once in each variant's lists (a codec in two launch files of one variant would leave
// the second one's case unreachable).
template <class List>
constexpr bool each_once(const List& list) noexcept {
    for (std::size_t i = 0; i < list.size(); ++i) {
        for (std::size_t j = i + 1; j < list.size(); ++j) {
            if (codec_of_entry(list[i]) == codec_of_entry(list[j])) return false;
        }
    }
    return true;
}

static_assert(each_once(kR8) && each_once(kW4a8) && each_once(kW4a8Ws) && each_once(kTiled) &&
                  each_once(kTiledVolta) && each_once(kVoltaHmma) && each_once(kVoltaGemv),
              "compiled_codecs.h: a codec listed twice for one variant");

// The producer-warp tile reads the classic tile's codec; the Volta tiles are not the others.
constexpr bool ws_within_w4a8() noexcept {
    for (WeightCodec c : kW4a8Ws) {
        if (!holds(kW4a8, c)) return false;
    }
    for (WeightCodec c : kTiledVolta) {
        if (holds(kTiled, c)) return false;
    }
    return true;
}

static_assert(ws_within_w4a8(), "compiled_codecs.h: a producer-warp codec without its W4A8 entry, "
                                "or a Volta tiled codec the tiled list already holds");
} // namespace compiled_lists

// The arrays the host reads (build_manifest and the predicates).
inline constexpr const auto& kCompiledR8Codecs         = compiled_lists::kR8;
inline constexpr const auto& kCompiledW4a8Codecs       = compiled_lists::kW4a8;
inline constexpr const auto& kCompiledW4a8WsCodecs     = compiled_lists::kW4a8Ws;
inline constexpr const auto& kCompiledTiledCodecs      = compiled_lists::kTiled;
inline constexpr const auto& kCompiledTiledVoltaCodecs = compiled_lists::kTiledVolta;
inline constexpr const auto& kCompiledVoltaHmmaCodecs  = compiled_lists::kVoltaHmma;
inline constexpr const auto& kCompiledVoltaGemvCodecs  = compiled_lists::kVoltaGemv;

// Whether this binary compiles the variant's kernels for the codec.
constexpr bool compiled(Variant variant, WeightCodec codec) noexcept {
    using namespace compiled_lists;
    switch (variant) {
    case Variant::R8:
        return holds(kR8, codec);
    case Variant::R8Prequant:
        for (const R8CompiledCodec& e : kR8) {
            if (e.codec == codec) return e.prequant;
        }
        return false;
    case Variant::W4a8:
        return holds(kW4a8, codec);
    case Variant::W4a8Ws:
        return holds(kW4a8Ws, codec);
    case Variant::Tiled:
        return holds(kTiled, codec);
    case Variant::TiledVolta:
        return holds(kTiledVolta, codec);
    case Variant::VoltaHmma:
        return holds(kVoltaHmma, codec);
    case Variant::VoltaGemv:
        return holds(kVoltaGemv, codec);
    }
    return false;
}

// The same for a stored format, through its codec (core/format_facts.h); false for a format the
// generic path has no codec for.
constexpr bool compiled(Variant variant, QType qtype) noexcept {
    const auto codec = format_facts(qtype).codec;
    return codec.has_value() && compiled(variant, *codec);
}

} // namespace ninferno::ops::detail::generic
