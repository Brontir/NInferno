#pragma once

// Quantized checkpoint encodings beyond the NVFP4 publishers the Python converter reads, detected
// from tensor names and dtypes (with config.json's quantization_config where the layout alone is
// ambiguous). Each decoder hands its codes to an engine format unchanged where one fits (keeps())
// and can dequantize rows to binary32; a conversion refuses a dequantization of published
// weights unless --maintainer-quantize asks for one (convert.cpp).
//
//   FP8 E4M3 ("weight" F8_E4M3):
//     per-tensor or per-channel scale  -> kept as fp8_e4m3fn_row_bf16 when every scale is a BF16
//                                         value, else as fp8_e4m3fn_row_f32 (always lossless)
//     128x128 block scales (weight_scale_inv) -> kept as fp8_e4m3fn_b128_f32 (lossless); other
//                                         block shapes are dequantized
//   compressed-tensors pack-quantized int4/int8 (weight_packed I32, weight_scale, optional
//     weight_zero_point and weight_g_idx)                       -> kept (below), else dequantized
//   compressed-tensors int-quantized int8 (weight I8, weight_scale, optional zero point)
//                                                               -> kept, else dequantized
//   AutoAWQ GEMM (qweight [K, N/8], qzeros, scales)             -> kept, else dequantized
//   GPTQ (qweight [K*b/32, N], qzeros, scales, optional g_idx)  -> kept, else dequantized
//     (kept: 4-bit codes as u4_g<G>_zp_fp16, G the largest of 128, 64 and 32 dividing K and
//     the checkpoint's group; 8-bit as q8_g32_fp16; act-order only where its order maps)
//   compressed-tensors MXFP4 (weight_packed U8, weight_scale U8 E8M0)
//                                    -> NVFP4 exactly when every block exponent fits E4M3's
//                                       2^-9..2^8 window under one power-of-two divisor (blocks
//                                       below it must be all zero), else dequantized
//   gpt-oss MXFP4 (<name>_blocks U8 [..., k/32, 16], <name>_scales U8 E8M0 [..., k/32]; element
//     2i in the low nibble; leading dimensions (experts, rows) flatten into rows)
//                                                                -> as compressed-tensors MXFP4
//   a GGUF tensor (a computed tensor_view.h view)                -> NVFP4 kept (ggml NVFP4, and
//                                       MXFP4 under the rule above), Q8_0 kept as q8_g32_fp16
//                                       (the same groups of 32 codes and binary16 scales), else
//                                       dequantized from its ggml blocks
//
// Dequantized values: f32(q - zero) * f32(scale), or f32(e4m3/e2m1 value) * f32(scale), each one
// IEEE binary32 multiply. Unsigned packed words follow compressed-tensors (value + 2^(bits-1),
// element i of an int32 at bits i*bits), AutoAWQ (nibble order 0,2,4,6,1,3,5,7 along N) and
// GPTQ (sequential along K; stored zeros are zero - 1 unless checkpoint_format is gptq_v2).

#include "convert/model.h"
#include "convert/sources.h"
#include "core/weight.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::convert {

class QuantizedMatrix {
public:
    virtual ~QuantizedMatrix() = default;

    // Human-readable encoding, e.g. "awq int4 g128 zero-point".
    [[nodiscard]] virtual std::string format() const = 0;

    // The engine format these words map onto without dequantization, if any.
    [[nodiscard]] virtual std::optional<QType> keeps() const { return std::nullopt; }

    // Whether keeping them is exact.
    [[nodiscard]] virtual bool keeps_losslessly() const { return false; }

    // The column order append_encoded() yields (sources.h EncodedRows), when not the natural one.
    [[nodiscard]] virtual KOrder k_order() const { return nullptr; }

    // Whether a reader of the original checkpoint (tools/convert) understands this encoding.
    [[nodiscard]] virtual bool python_readable() const { return false; }

    // Rows [first, last) as binary32, row-major.
    virtual void dequantize(std::uint64_t first, std::uint64_t last, float* out,
                            ReadLog* log) const = 0;
    // Engine words of rows [first, last), appended (keeps() must be set).
    virtual void append_encoded(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                                ReadLog* log) const;
    [[nodiscard]] virtual std::uint32_t weight_divisor() const;
    // Whether the words are MXFP4 that the engine's mxfp4 format keeps unchanged (always
    // lossless), and those words: natural E2M1 codes [rows, k/2] and E8M0 exponents [rows, k/32].
    [[nodiscard]] virtual bool keeps_mxfp4() const { return false; }
    virtual void append_mxfp4(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                              ReadLog* log) const;
    // A calibrated activation divisor of a kept NVFP4 matrix, when the checkpoint has one.
    [[nodiscard]] virtual std::optional<std::uint32_t> input_divisor() const {
        return std::nullopt;
    }

    virtual void tensors(std::vector<const TensorInfo*>& out) const = 0;
    // Bytes rows [first, last) read, for read-ahead and release; empty when rows are not
    // contiguous in the checkpoint (transposed packings), which are then kept until the end.
    virtual void ranges(std::uint64_t first, std::uint64_t last, ReadLog& out) const = 0;
};

// The decoder for the matrix at `prefix` (name without ".weight") of logical shape [n, k], or
// null when the tensors there are plain values or a Python-read NVFP4 layout.
[[nodiscard]] std::unique_ptr<QuantizedMatrix> probe_quantized(const Checkpoint& checkpoint,
                                                               const std::string& prefix,
                                                               std::uint64_t n, std::uint64_t k);

// An engine format a checkpoint matrix keeps without dequantization: the Python-read NVFP4
// layouts (with a calibrated activation divisor), or a formats.h encoding that maps exactly or
// nearly exactly (MXFP4 as NVFP4 without an activation divisor, FP8 with row/tensor scales).
struct KeptEncoding {
    QType format            = QType::NVFP4;
    bool activation_divisor = true;
};

[[nodiscard]] std::optional<KeptEncoding> kept_encoding(const Checkpoint& checkpoint,
                                                        const std::string& prefix, std::uint64_t n,
                                                        std::uint64_t k);

// How a recipe asks for a kept site's words (Parameter::encoded_source) and which activations the
// site admits: NVFP4 with its activation divisor may take FP4 inputs, FP8 may take FP8 inputs,
// the integer formats with a scale per 32 or 16 values and MXFP4 (kept_block_policy) int8 inputs,
// every other kept encoding computes with BF16 activations.
[[nodiscard]] MatrixFormat encoded_matrix_format(QType format) noexcept;

// The activation policy of a kept integer encoding whose every 32 values are int8 codes times a
// scale (and a min; Q6_K, IQ2_XS, IQ2_S a scale per 16): the generic W4A8 route computes it on int8
// tensor cores against int8 activations with a scale per 32 values (ggml Q4_0, Q4_1, Q5_0, Q5_1,
// IQ4_NL, IQ4_XS, Q4_K, Q5_K, Q6_K, u4_g128_zp, and Q8_0 and the codebook i-quants IQ3_XXS, IQ3_S,
// IQ2_XXS, IQ2_XS, IQ2_S, whose signed grid values are the codes; these last also decode through
// the R8 dp4a route), MXFP4 (E2M1 codes widen to int8 exactly) and Q3_K (R8 decode only; from 17
// tokens it runs A16): AllowA8. Under --activations exact the engine runs every one of them with
// BF16 activations. Every other format: A16Only (keep_published then leaves the recipe's policy).
[[nodiscard]] ActivationPolicy kept_block_policy(QType format) noexcept;

// The GGUF encodings (Q8_0 and the K-quants) that are no recipe's encoded site: they take no
// activation policy and no packing rule, and convert() keeps them wherever a recipe would decode
// them (keep_published).
[[nodiscard]] bool kept_where_decoded(QType format) noexcept;
// --round-tiny-scales: keep an integer site whose only inexact binary16 scales lie below binary16's
// normal range, those scales rounded (they change the weights; off by default).
void set_round_tiny_scales(bool on) noexcept;
[[nodiscard]] ActivationPolicy kept_policy(const KeptEncoding& site) noexcept;
// A kept output head's policy: an NVFP4 or FP8 head keeps BF16 activations (A16Only) whatever its
// site calibrated, so its logits take one route at every token count; any other head kept_policy.
[[nodiscard]] ActivationPolicy head_policy(const KeptEncoding& site) noexcept;

// The E4M3 word of 2^power for power in [-9, 8] (0 when the power is below the range).
[[nodiscard]] std::uint8_t e4m3_power_of_two(int power) noexcept;

} // namespace ninferno::convert
