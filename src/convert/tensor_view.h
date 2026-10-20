#pragma once

// A checkpoint tensor whose values are computed from other bytes instead of being stored as its
// own contiguous words: a GGUF tensor seen under its Hugging Face name, sliced, reordered,
// transformed back from what llama.cpp's converter wrote, and decoded from its ggml blocks.
//
// A TensorInfo carrying a view has no data pointer; sources ask the view for flat values in the
// Hugging Face (row-major) order, and for the checkpoint bytes those values come from.

#include "convert/sources.h"
#include "core/weight.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::convert {

class TensorView {
public:
    virtual ~TensorView() = default;

    // Type of the words values() writes: the stored words when they are passed through (BF16,
    // F16, F32), else F32.
    [[nodiscard]] virtual SourceDType dtype() const = 0;
    // Flat values [begin, end) as dtype() words.
    virtual void values(std::uint64_t begin, std::uint64_t end, void* out, ReadLog* log) const = 0;
    // The stored encoding, e.g. "gguf q4_k" or "gguf f32 (1+w stored)".
    [[nodiscard]] virtual std::string encoding() const = 0;
    // Whether values() decodes a quantized encoding (a second quantization is lossy).
    [[nodiscard]] virtual bool dequantizes() const = 0;
    // The stored tensors values() reads.
    virtual void tensors(std::vector<const TensorInfo*>& out) const = 0;
    // The stored bytes values [begin, end) read.
    virtual void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const = 0;

    // A 2-D view (rows of the last dimension) can keep its words as an engine format.
    [[nodiscard]] virtual std::optional<QType> keeps() const { return std::nullopt; }
    [[nodiscard]] virtual bool keeps_losslessly() const { return false; }
    virtual void append_encoded(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                                ReadLog* log) const;
    [[nodiscard]] virtual std::uint32_t weight_divisor() const;
    [[nodiscard]] virtual std::optional<std::uint32_t> input_divisor() const {
        return std::nullopt;
    }
    // Whether the words are MXFP4 blocks the engine's mxfp4 format keeps as they are (natural
    // E2M1 codes [rows, k/2], one E8M0 exponent per 32 values), and those words.
    [[nodiscard]] virtual bool keeps_mxfp4() const { return false; }
    virtual void append_mxfp4(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                              ReadLog* log) const;
};

} // namespace ninferno::convert
