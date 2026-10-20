#pragma once

// Logical sources of the native converter, one class per kind the Python adapters build
// (tools/convert/sources/*.py and the architecture factories). Every source knows its logical
// shape, its report label, how to produce values, and - for NVFP4 - its encoded rows and divisors.
//
// Sources that probe the checkpoint (MatrixSource) decide their encoding in resolve(), once, even
// when several selections reach one source and resolve it on several threads; after that every
// method is const and thread-safe.

#include "convert/safetensors.h"
#include "core/weight.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::convert {

using Shape = std::vector<std::uint64_t>;

[[nodiscard]] std::uint64_t shape_elements(const Shape& shape) noexcept;

// The torch dtype a source's values() would carry, which cast_direct converts from.
enum class ValueType { BF16, F16, F32, F64, I32, I64, I8, U8 };
[[nodiscard]] std::size_t value_bytes(ValueType type) noexcept;

// Checkpoint bytes a task has touched, released when the task finishes.
struct ReadLog {
    struct Range {
        const TensorInfo* tensor;
        const std::byte* begin;
        std::size_t bytes;
    };

    std::vector<Range> ranges;

    void add(const TensorInfo& tensor, const std::byte* begin, std::size_t bytes) {
        ranges.push_back({&tensor, begin, bytes});
    }
};

// Encoded rows of one plane. NVFP4: packed codes [rows, k/2], natural E4M3 scales [rows, k/16]
// and the weight divisor. Row-scaled FP8: codes [rows, k] and BF16 row scales (2 bytes a row).
// U4 G128 with zero points: packed codes [rows, k/2] (element 2i in the low nibble), one zero
// point byte per group in `zeros` [rows, k/128] and binary16 group scales [rows, k/128].
//
// A u4 matrix published with its K values in another order (GPTQ act-order) yields its columns in
// that order and sets `order` (order[j]: the input column at position j), which the writer checks
// against the order the plan stores with the plane.
struct EncodedRows {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> scales;
    std::vector<std::uint8_t> zeros;
    std::uint32_t divisor = 0;
    std::shared_ptr<const std::vector<std::uint32_t>> order;
};

using KOrder = std::shared_ptr<const std::vector<std::uint32_t>>;

// Whether two column orders are the same (both absent, or equal indices).
[[nodiscard]] inline bool same_order(const KOrder& a, const KOrder& b) {
    return a == b || (a && b && *a == *b);
}

class Source {
public:
    Source(Shape shape, std::string label) : shape_(std::move(shape)), label_(std::move(label)) {}

    virtual ~Source()                = default;
    Source(const Source&)            = delete;
    Source& operator=(const Source&) = delete;

    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }

    [[nodiscard]] const std::string& label() const noexcept { return label_; }

    [[nodiscard]] std::uint64_t elements() const noexcept { return shape_elements(shape_); }

    // Plan-time decisions (checkpoint probes); idempotent, single-threaded.
    virtual void resolve() const {}

    // Flat C-order values [begin, end) as binary32.
    virtual void values_f32(std::uint64_t begin, std::uint64_t end, float* out,
                            ReadLog* log) const        = 0;
    [[nodiscard]] virtual ValueType value_type() const = 0;
    // Flat values [begin, end) as value_type() words.
    virtual void values_raw(std::uint64_t begin, std::uint64_t end, void* out,
                            ReadLog* log) const = 0;

    // NVFP4 access. `provides_divisors` mirrors Python's "weight_divisor is not None": whether
    // the source can be asked, not whether the answer exists.
    [[nodiscard]] virtual bool provides_divisors() const { return false; }

    [[nodiscard]] virtual bool provides_input_divisor() const { return provides_divisors(); }

    // False when input_divisor() is a stand-in for an activation scale calibration never
    // measured (0, NaN or infinite in the checkpoint): a shared plane then takes its calibrated
    // peers' divisor instead.
    [[nodiscard]] virtual bool calibrated_input() const { return true; }

    [[nodiscard]] virtual bool is_nvfp4() const { return false; }

    // The column order encoded_rows() yields its K values in, when not the natural one.
    [[nodiscard]] virtual KOrder k_order() const { return nullptr; }

    // The engine format encoded_rows() yields (NVFP4 or row-scaled FP8), if any.
    [[nodiscard]] virtual std::optional<QType> encoded_format() const {
        return is_nvfp4() ? std::optional<QType>(QType::NVFP4) : std::nullopt;
    }

    // The checkpoint encoding values come from, e.g. "bf16" or "awq int4 g128 zero-point".
    [[nodiscard]] virtual std::string encoding() const = 0;

    // Whether values() dequantizes a quantized encoding (a second quantization is lossy). A source
    // that wraps another says what the wrapped one says; decodes_quantized() below backs this up
    // from the stored tensors, so a wrapper that forgets cannot hide a decode.
    [[nodiscard]] virtual bool dequantizes() const { return false; }

    // Why values() decodes a quantized encoding when this source's own transform needs the values
    // (a constant folded into them, a permutation of the axes): the conversion then reports the
    // decode with this reason instead of refusing it. Empty: no reason, the decode is refused
    // unless the user forces the conversion.
    [[nodiscard]] virtual std::string decode_reason() const { return {}; }

    // Whether encoded_rows() reproduces the checkpoint's values exactly.
    [[nodiscard]] virtual bool keeps_losslessly() const { return true; }

    // Whether tools/convert reads this encoding (the report then stays the Python one).
    [[nodiscard]] virtual bool python_readable() const { return true; }

    [[nodiscard]] virtual std::uint32_t weight_divisor() const;
    [[nodiscard]] virtual std::uint32_t input_divisor() const;
    virtual void encoded_rows(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                              ReadLog* log) const;

    // Checkpoint tensors this source reads, for page-cache policy.
    virtual void tensors(const Checkpoint& checkpoint,
                         std::vector<const TensorInfo*>& out) const = 0;
    // Checkpoint bytes a read of flat elements [begin, end) touches (values or, for NVFP4,
    // the encoded rows covering them), for read-ahead.
    virtual void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const = 0;

private:
    Shape shape_;
    std::string label_;
};

using SourcePtr = std::shared_ptr<const Source>;

// Whether any value `source` gives is decoded from a quantized stored encoding: its own
// dequantizes(), or a tensor it reads whose view decodes blocks, or a stored GGUF tensor of a block
// type (Q4_0, Q6_K, IQ4_XS, ...) read as values. A tensor of any name counts, with or without the
// .weight suffix. Only for sources read as values: an encoded import reads its blocks as words.
[[nodiscard]] bool decodes_quantized(const Checkpoint& checkpoint, const Source& source);

// tensor_source: a checkpoint tensor read as values, optionally at an element offset.
[[nodiscard]] SourcePtr tensor_source(const Checkpoint& checkpoint, const std::string& name,
                                      Shape shape);
// nvfp4_rounded_source: a plain matrix as round-to-nearest NVFP4 (no activation divisor).
// Every member of `divisor_group` (the source itself among them) shares the matrix divisor.
[[nodiscard]] SourcePtr nvfp4_rounded_source(SourcePtr source,
                                           std::vector<SourcePtr> divisor_group = {});
// _scaled_source: values of `source` times a binary32 constant.
[[nodiscard]] SourcePtr scaled_source(SourcePtr source, float factor);

// Mxfp4: the checkpoint's MXFP4 words kept unchanged as the engine's mxfp4 format.
enum class MatrixFormat { Auto, Nvfp4, Fp8Row, Mxfp4 };
enum class Nvfp4Fields { CompressedTensors, ModelOpt };

// Which publisher's NVFP4 fields `prefix` uses, if any (compressed_tensors.nvfp4_layout).
[[nodiscard]] std::optional<Nvfp4Fields> nvfp4_layout(const Checkpoint& checkpoint,
                                                      const std::string& prefix);

// matrix_source: a matrix whose encoding (plain, NVFP4, or a formats.h encoding) is probed at
// resolve().
[[nodiscard]] std::shared_ptr<const Source> matrix_source(const Checkpoint& checkpoint,
                                                          const std::string& name, Shape shape,
                                                          MatrixFormat format);
// stacked_matrix_source(adopt_first=True): NVFP4 matrices stacked by rows under the first
// matrix's divisor, codes and block scales untouched.
[[nodiscard]] SourcePtr stacked_source(std::vector<SourcePtr> parts, const std::string& label);
// _concat_values: plain matrices concatenated by rows.
[[nodiscard]] SourcePtr concat_source(std::vector<SourcePtr> parts, Shape shape,
                                      const std::string& label);
// divisor_scaled_source with factor d_up/d_gate: the matrix read against divisor*factor.
[[nodiscard]] SourcePtr divisor_scaled_source(SourcePtr matrix, SourcePtr gate, SourcePtr up);
// tensor_source with an element offset into a tensor of shape `source_shape` (fused tensors).
[[nodiscard]] SourcePtr tensor_source(const Checkpoint& checkpoint, const std::string& name,
                                      Shape shape, std::uint64_t offset, Shape source_shape);
// compressed_matrix_source: an encoded matrix at `prefix`, labelled "path:prefix (format)".
[[nodiscard]] SourcePtr compressed_source(const Checkpoint& checkpoint, const std::string& prefix,
                                          Shape shape, MatrixFormat format);
// transpose_source: the whole tensor permuted by `axes`, reshaped to `shape`.
[[nodiscard]] SourcePtr transpose_source(SourcePtr source, std::vector<std::size_t> axes,
                                         Shape shape);
// select_rows: row ranges [first, last) of a matrix, in order.
[[nodiscard]] SourcePtr select_rows(SourcePtr matrix,
                                    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans);
// gather_source: the rows `ids` of a matrix, in order.
[[nodiscard]] SourcePtr gather_source(SourcePtr matrix, std::vector<std::int64_t> ids);
// array_source over int32 ids.
[[nodiscard]] SourcePtr int32_source(std::vector<std::int32_t> values, std::string label);

} // namespace ninferno::convert
