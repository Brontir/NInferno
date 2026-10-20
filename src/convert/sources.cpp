#include "convert/sources.h"
#include "core/ascii.h"

#include "convert/formats.h"
#include "convert/ggml_quants.h"
#include "convert/numerics.h"
#include "convert/tensor_view.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace ninferno::convert {
namespace {

std::string shape_text(const Shape& shape) {
    std::string out = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) { out += ", "; }
        out += std::to_string(shape[i]);
    }
    if (shape.size() == 1) { out += ","; }
    return out + ")";
}

ValueType value_type_of(SourceDType dtype, const std::string& name) {
    switch (dtype) {
    case SourceDType::BF16:
        return ValueType::BF16;
    case SourceDType::F16:
        return ValueType::F16;
    case SourceDType::F32:
        return ValueType::F32;
    case SourceDType::F64:
        return ValueType::F64;
    case SourceDType::I32:
        return ValueType::I32;
    case SourceDType::I64:
        return ValueType::I64;
    case SourceDType::I8:
        return ValueType::I8;
    case SourceDType::U8:
        return ValueType::U8;
    default:
        throw std::runtime_error(name + ": FP8 values are not read by the native converter");
    }
}

void words_to_f32(ValueType type, const std::byte* in, std::uint64_t count, float* out) {
    switch (type) {
    case ValueType::BF16:
        for (std::uint64_t i = 0; i < count; ++i) {
            std::uint16_t w;
            std::memcpy(&w, in + 2 * i, 2);
            out[i] = bf16_to_f32(w);
        }
        return;
    case ValueType::F16:
        for (std::uint64_t i = 0; i < count; ++i) {
            std::uint16_t w;
            std::memcpy(&w, in + 2 * i, 2);
            out[i] = f16_to_f32(w);
        }
        return;
    case ValueType::F32:
        std::memcpy(out, in, count * 4);
        return;
    case ValueType::F64:
        for (std::uint64_t i = 0; i < count; ++i) {
            double v;
            std::memcpy(&v, in + 8 * i, 8);
            out[i] = static_cast<float>(v);
        }
        return;
    case ValueType::I32:
        for (std::uint64_t i = 0; i < count; ++i) {
            std::int32_t v;
            std::memcpy(&v, in + 4 * i, 4);
            out[i] = static_cast<float>(v);
        }
        return;
    case ValueType::I64:
        for (std::uint64_t i = 0; i < count; ++i) {
            std::int64_t v;
            std::memcpy(&v, in + 8 * i, 8);
            out[i] = static_cast<float>(v);
        }
        return;
    case ValueType::I8:
        for (std::uint64_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(static_cast<std::int8_t>(in[i]));
        }
        return;
    case ValueType::U8:
        for (std::uint64_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(static_cast<std::uint8_t>(in[i]));
        }
        return;
    }
}

// ---------------------------------------------------------------------------------------------

class TensorSource final : public Source {
public:
    TensorSource(const Checkpoint& checkpoint, std::string name, Shape shape,
                 std::uint64_t offset = 0, std::optional<Shape> source_shape = std::nullopt)
        : Source(shape, checkpoint.path_string() + ":" + name + "[" + std::to_string(offset) + ":" +
                            std::to_string(offset + shape_elements(shape)) + "]"),
          checkpoint_(checkpoint), name_(std::move(name)), offset_(offset),
          source_shape_(source_shape ? *source_shape : shape) {}

    void resolve() const override { info(); }

    [[nodiscard]] ValueType value_type() const override {
        return value_type_of(info().dtype, name_);
    }

    [[nodiscard]] std::string encoding() const override {
        if (const auto& view = info().view) { return view->encoding(); }
        return ascii_lower(info().dtype_name);
    }

    [[nodiscard]] bool dequantizes() const override {
        const auto& view = info().view;
        return view && view->dequantizes();
    }

    [[nodiscard]] bool python_readable() const override { return !info().view; }

    // A row block of a computed tensor that keeps its words (an expert's rows of a GGUF expert
    // bank) keeps them too: its encoded rows are the view's rows at the block's offset.
    [[nodiscard]] std::optional<QType> encoded_format() const override {
        const auto* view = row_view();
        return view ? view->keeps() : std::nullopt;
    }

    [[nodiscard]] bool keeps_losslessly() const override {
        const auto* view = row_view();
        return view ? view->keeps_losslessly() : true;
    }

    void encoded_rows(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                      ReadLog* log) const override {
        const auto* view = row_view();
        if (!view) {
            Source::encoded_rows(first, last, out, log);
            return;
        }
        const auto row = offset_ / shape()[1];
        view->append_encoded(row + first, row + last, out, log);
    }

    void values_raw(std::uint64_t begin, std::uint64_t end, void* out,
                    ReadLog* log) const override {
        const auto& tensor = info();
        if (tensor.view) {
            tensor.view->values(offset_ + begin, offset_ + end, out, log);
            return;
        }
        const auto word  = dtype_bytes(tensor.dtype);
        const auto* data = tensor.data + (offset_ + begin) * word;
        std::memcpy(out, data, (end - begin) * word);
        if (log) { log->add(tensor, data, (end - begin) * word); }
    }

    void values_f32(std::uint64_t begin, std::uint64_t end, float* out,
                    ReadLog* log) const override {
        const auto& tensor = info();
        const auto word    = dtype_bytes(tensor.dtype);
        if (tensor.view) {
            if (tensor.dtype == SourceDType::F32) {
                tensor.view->values(offset_ + begin, offset_ + end, out, log);
                return;
            }
            std::vector<std::byte> words((end - begin) * word);
            tensor.view->values(offset_ + begin, offset_ + end, words.data(), log);
            words_to_f32(value_type(), words.data(), end - begin, out);
            return;
        }
        const auto* data = tensor.data + (offset_ + begin) * word;
        words_to_f32(value_type(), data, end - begin, out);
        if (log) { log->add(tensor, data, (end - begin) * word); }
    }

    void tensors(const Checkpoint&, std::vector<const TensorInfo*>& out) const override {
        const auto& tensor = info();
        if (tensor.view) {
            tensor.view->tensors(out);
            return;
        }
        out.push_back(&tensor);
    }

    void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const override {
        const auto& tensor = info();
        if (tensor.view) {
            tensor.view->ranges(offset_ + begin, offset_ + end, out);
            return;
        }
        const auto word = dtype_bytes(tensor.dtype);
        out.add(tensor, tensor.data + (offset_ + begin) * word, (end - begin) * word);
    }

private:
    // The view this 2-D block reads whole rows of, when that view keeps its words.
    const TensorView* row_view() const {
        const auto& tensor = info();
        if (!tensor.view || shape().size() != 2 || shape()[1] == 0 || tensor.shape.empty() ||
            tensor.shape.back() != shape()[1] || offset_ % shape()[1] != 0 ||
            !tensor.view->keeps()) {
            return nullptr;
        }
        return tensor.view.get();
    }

    const TensorInfo& info() const {
        const auto& tensor = checkpoint_.readable(name_);
        if (tensor.shape != source_shape_) {
            throw std::runtime_error(name_ + ": expected source shape " +
                                     shape_text(source_shape_) + ", got " +
                                     shape_text(tensor.shape));
        }
        return tensor;
    }

    const Checkpoint& checkpoint_;
    std::string name_;
    std::uint64_t offset_ = 0;
    Shape source_shape_;
};

class ScaledSource final : public Source {
public:
    ScaledSource(SourcePtr source, float factor)
        : Source(source->shape(), "scale(" + source->label() + "," + py_format_g(factor, 9) + ")"),
          source_(std::move(source)), factor_(factor) {}

    void resolve() const override { source_->resolve(); }

    [[nodiscard]] ValueType value_type() const override { return ValueType::F32; }

    [[nodiscard]] std::string encoding() const override { return source_->encoding(); }

    [[nodiscard]] bool dequantizes() const override { return source_->dequantizes(); }

    [[nodiscard]] std::string decode_reason() const override {
        return "to fold the constant " + py_format_g(factor_, 9) + " into its values";
    }

    void values_f32(std::uint64_t begin, std::uint64_t end, float* out,
                    ReadLog* log) const override {
        source_->values_f32(begin, end, out, log);
        for (std::uint64_t i = 0; i < end - begin; ++i) { out[i] = out[i] * factor_; }
    }

    void values_raw(std::uint64_t begin, std::uint64_t end, void* out,
                    ReadLog* log) const override {
        values_f32(begin, end, static_cast<float*>(out), log);
    }

    void tensors(const Checkpoint& checkpoint, std::vector<const TensorInfo*>& out) const override {
        source_->tensors(checkpoint, out);
    }

    void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const override {
        source_->ranges(begin, end, out);
    }

private:
    SourcePtr source_;
    float factor_;
};

// ---------------------------------------------------------------------------------------------

class MatrixSource final : public Source {
public:
    MatrixSource(const Checkpoint& checkpoint, std::string name, Shape shape, MatrixFormat format,
                 std::optional<std::string> label = std::nullopt)
        : Source(shape, label ? *label : checkpoint.path_string() + ":" + name),
          checkpoint_(checkpoint), name_(std::move(name)), format_(format) {
        prefix_ = name_;
        if (prefix_.ends_with(".weight")) { prefix_.resize(prefix_.size() - 7); }
    }

    void resolve() const override {
        if (resolved_.load(std::memory_order_acquire)) { return; }
        std::lock_guard lock(resolve_mutex_);
        if (resolved_.load(std::memory_order_relaxed)) { return; }
        if (nvfp4_layout(checkpoint_, prefix_)) {
            if (format_ == MatrixFormat::Fp8Row || format_ == MatrixFormat::Mxfp4) {
                throw std::runtime_error(prefix_ + ": no FP8 representation in this checkpoint");
            }
            resolve_nvfp4();
        } else if ((decoder_ = probe_quantized(checkpoint_, prefix_, shape()[0], shape()[1]))) {
            const auto kept = decoder_->keeps();
            if (format_ == MatrixFormat::Nvfp4 && kept != QType::NVFP4) {
                throw std::runtime_error(prefix_ + ": no NVFP4 representation in this checkpoint");
            }
            if (format_ == MatrixFormat::Fp8Row && kept != QType::FP8_E4M3FN_ROW_BF16 &&
                kept != QType::FP8_E4M3FN_ROW_F32 && kept != QType::FP8_E4M3FN_B128_F32) {
                throw std::runtime_error(prefix_ + ": no FP8 representation in this checkpoint");
            }
            if (format_ == MatrixFormat::Mxfp4 && !decoder_->keeps_mxfp4()) {
                throw std::runtime_error(prefix_ + ": no MXFP4 representation in this checkpoint");
            }
        } else if (format_ != MatrixFormat::Auto) {
            throw std::runtime_error(prefix_ + ": no " +
                                     (format_ == MatrixFormat::Nvfp4   ? "NVFP4"
                                      : format_ == MatrixFormat::Mxfp4 ? "MXFP4"
                                                                       : "FP8") +
                                     " representation in this checkpoint");
        } else {
            plain_ = tensor_source(checkpoint_, name_, shape());
            plain_->resolve();
        }
        resolved_.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool is_nvfp4() const override {
        resolve();
        if (format_ == MatrixFormat::Mxfp4) { return false; }
        return nvfp4_ || (decoder_ && decoder_->keeps() == QType::NVFP4);
    }

    [[nodiscard]] KOrder k_order() const override {
        resolve();
        return decoder_ && decoder_->keeps() ? decoder_->k_order() : nullptr;
    }

    [[nodiscard]] std::optional<QType> encoded_format() const override {
        resolve();
        if (format_ == MatrixFormat::Mxfp4) { return QType::MXFP4; }
        if (nvfp4_) { return QType::NVFP4; }
        return decoder_ ? decoder_->keeps() : std::nullopt;
    }

    [[nodiscard]] std::string encoding() const override {
        resolve();
        if (nvfp4_) {
            std::string text =
                fields_ == Nvfp4Fields::ModelOpt ? "nvfp4 (modelopt" : "nvfp4 (compressed-tensors";
            for (const auto* note : {&dead_note_, &input_note_}) {
                if (!note->empty()) { text += "; " + *note; }
            }
            return text + ")";
        }
        return decoder_ ? decoder_->format() : plain_->encoding();
    }

    [[nodiscard]] bool dequantizes() const override {
        resolve();
        return nvfp4_ || decoder_;
    }

    [[nodiscard]] bool keeps_losslessly() const override {
        resolve();
        if (format_ == MatrixFormat::Mxfp4) { return true; }
        return !decoder_ || decoder_->keeps_losslessly();
    }

    [[nodiscard]] bool python_readable() const override {
        resolve();
        // tools/convert refuses a dead matrix's scales; the native notes record the handling.
        if (!dead_note_.empty() || !input_note_.empty()) { return false; }
        return !decoder_ || decoder_->python_readable();
    }

    [[nodiscard]] bool provides_input_divisor() const override {
        resolve();
        if (decoder_) { return decoder_->input_divisor().has_value(); }
        // A published NVFP4 matrix carries an activation divisor only when its checkpoint
        // calibrated the input (.input_scale / .input_global_scale); a W4A16 matrix (ModelOpt
        // MIXED MLPs) has none, and a fused plane of such matrices keeps BF16 activations.
        std::lock_guard lock(resolve_mutex_);
        return input_divisor_.has_value();
    }

    [[nodiscard]] bool calibrated_input() const override {
        resolve();
        return input_note_.empty();
    }

    [[nodiscard]] bool provides_divisors() const override { return true; }

    [[nodiscard]] ValueType value_type() const override {
        resolve();
        return plain_ ? plain_->value_type() : ValueType::F32;
    }

    void values_f32(std::uint64_t begin, std::uint64_t end, float* out,
                    ReadLog* log) const override {
        resolve();
        if (plain_) {
            plain_->values_f32(begin, end, out, log);
            return;
        }
        if (decoder_) {
            decode(begin, end, out, log);
            return;
        }
        dequantize(begin, end, divisor_value_, out, log);
    }

    void values_raw(std::uint64_t begin, std::uint64_t end, void* out,
                    ReadLog* log) const override {
        resolve();
        if (plain_) {
            plain_->values_raw(begin, end, out, log);
            return;
        }
        values_f32(begin, end, static_cast<float*>(out), log);
    }

    [[nodiscard]] std::uint32_t weight_divisor() const override {
        resolve();
        if (decoder_ && decoder_->keeps() == QType::NVFP4) { return decoder_->weight_divisor(); }
        if (!nvfp4_) {
            throw std::runtime_error(name_ + ": selected source does not provide weight_divisor");
        }
        return weight_divisor_;
    }

    [[nodiscard]] std::uint32_t input_divisor() const override {
        resolve();
        if (decoder_) {
            if (const auto divisor = decoder_->input_divisor()) { return *divisor; }
        }
        if (!nvfp4_) {
            throw std::runtime_error(name_ + ": selected source does not provide input_divisor");
        }
        std::lock_guard lock(resolve_mutex_);
        if (!input_divisor_) {
            // Absent: the scalar's own error names it.
            input_divisor_ = fields_ == Nvfp4Fields::ModelOpt
                                 ? reciprocal_word(prefix_ + ".input_scale")
                                 : divisor_word(prefix_ + ".input_global_scale");
        }
        return *input_divisor_;
    }

    void encoded_rows(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                      ReadLog* log) const override {
        resolve();
        if (format_ == MatrixFormat::Mxfp4) {
            decoder_->append_mxfp4(first, last, out, log);
            return;
        }
        if (decoder_ && decoder_->keeps()) {
            decoder_->append_encoded(first, last, out, log);
            return;
        }
        if (!nvfp4_) {
            throw std::runtime_error(name_ + ": selected source does not provide encoded rows");
        }
        append_rows(first, last, out, log);
        out.divisor = weight_divisor_;
    }

    void tensors(const Checkpoint& checkpoint, std::vector<const TensorInfo*>& out) const override {
        resolve();
        if (plain_) {
            plain_->tensors(checkpoint, out);
            return;
        }
        if (decoder_) {
            decoder_->tensors(out);
            return;
        }
        out.push_back(packed_);
        out.push_back(scales_);
    }

    void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const override {
        resolve();
        if (plain_) {
            plain_->ranges(begin, end, out);
            return;
        }
        const auto k = shape()[1];
        if (decoder_) {
            decoder_->ranges(begin / k, (end + k - 1) / k, out);
            return;
        }
        const auto first = begin / k;
        const auto last  = (end + k - 1) / k;
        out.add(*packed_, packed_->data + first * (k / 2), (last - first) * (k / 2));
        out.add(*scales_, scales_->data + first * (k / 16), (last - first) * (k / 16));
    }

    // Stacked and divisor-scaled sources read this matrix's words directly.
    void append_rows(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                     ReadLog* log) const {
        const auto k        = shape()[1];
        const auto rows     = last - first;
        const auto* codes   = packed_->data + first * (k / 2);
        const auto* scales  = scales_->data + first * (k / 16);
        const auto code_at  = out.codes.size();
        const auto scale_at = out.scales.size();
        out.codes.resize(code_at + rows * (k / 2));
        out.scales.resize(scale_at + rows * (k / 16));
        std::memcpy(out.codes.data() + code_at, codes, rows * (k / 2));
        std::memcpy(out.scales.data() + scale_at, scales, rows * (k / 16));
        if (zero_scales_) {
            std::fill_n(out.scales.data() + scale_at, rows * (k / 16), std::uint8_t{0});
        }
        for (std::uint64_t i = 0; i < rows * (k / 16); ++i) {
            if (out.scales[scale_at + i] > 0x7e) {
                throw std::runtime_error(
                    prefix_ + ".weight_scale: expected nonnegative finite E4M3FN scales");
            }
        }
        if (log) {
            log->add(*packed_, codes, rows * (k / 2));
            log->add(*scales_, scales, rows * (k / 16));
        }
    }

    void dequantize(std::uint64_t begin, std::uint64_t end, float divisor, float* out,
                    ReadLog* log) const {
        const auto k = shape()[1];
        std::vector<float> row(k);
        const std::vector<std::uint8_t> zeros(zero_scales_ ? k / 16 : 0);
        std::uint64_t cursor = begin;
        while (cursor < end) {
            const auto r     = cursor / k;
            const auto low   = cursor - r * k;
            const auto high  = std::min<std::uint64_t>(k, end - r * k);
            const auto* code = reinterpret_cast<const std::uint8_t*>(packed_->data) + r * (k / 2);
            const auto* sc =
                zero_scales_ ? zeros.data()
                             : reinterpret_cast<const std::uint8_t*>(scales_->data) + r * (k / 16);
            for (std::uint64_t i = 0; i < k / 16; ++i) {
                if (sc[i] > 0x7e) {
                    throw std::runtime_error(
                        prefix_ + ".weight_scale: expected nonnegative finite E4M3FN scales");
                }
            }
            if (low == 0 && high == k) {
                dequantize_nvfp4_row(code, sc, divisor, k, out + (cursor - begin));
            } else {
                dequantize_nvfp4_row(code, sc, divisor, k, row.data());
                std::memcpy(out + (cursor - begin), row.data() + low, (high - low) * 4);
            }
            if (log) {
                log->add(*packed_, packed_->data + r * (k / 2), k / 2);
                log->add(*scales_, scales_->data + r * (k / 16), k / 16);
            }
            cursor = r * k + high;
        }
    }

private:
    // Rows of a formats.h decoder, sliced to the flat range [begin, end).
    void decode(std::uint64_t begin, std::uint64_t end, float* out, ReadLog* log) const {
        const auto k     = shape()[1];
        const auto first = begin / k;
        const auto last  = (end + k - 1) / k;
        if (begin == first * k && end == last * k) {
            decoder_->dequantize(first, last, out, log);
            return;
        }
        std::vector<float> rows((last - first) * k);
        decoder_->dequantize(first, last, rows.data(), log);
        std::memcpy(out, rows.data() + (begin - first * k), (end - begin) * 4);
    }

    std::uint32_t divisor_word(const std::string& name) const {
        const auto& info = checkpoint_.describe(name);
        if (info.dtype_name != "F32" || info.elements() != 1 || info.bytes != 4) {
            throw std::runtime_error(name + ": expected a source FP32 scalar");
        }
        std::uint32_t word = 0;
        checkpoint_.read(info, &word);
        if (!valid_positive_f32_word(word)) {
            throw std::runtime_error(name + ": divisor must be finite and positive");
        }
        return word;
    }

    std::uint32_t reciprocal_word(const std::string& name) const {
        const float multiplier = f32_from_bits(divisor_word(name));
        const auto word        = f32_bits(1.0F / multiplier);
        if (!valid_positive_f32_word(word)) {
            throw std::runtime_error(name + ": reciprocal divisor is not finite and positive");
        }
        return word;
    }

    // The stored FP32 scalar of `name`, unvalidated.
    float scalar(const std::string& name) const {
        const auto& info = checkpoint_.describe(name);
        if (info.dtype_name != "F32" || info.elements() != 1 || info.bytes != 4) {
            throw std::runtime_error(name + ": expected a source FP32 scalar");
        }
        std::uint32_t word = 0;
        checkpoint_.read(info, &word);
        return f32_from_bits(word);
    }

    bool block_scales_zero() const {
        const auto* words = reinterpret_cast<const std::uint8_t*>(scales_->data);
        return std::all_of(words, words + scales_->bytes, [](std::uint8_t w) { return w == 0; });
    }

    // A dead matrix (an expert calibration never saw, amax 0): its scalar scale is 0, NaN or
    // infinite, which no divisor represents. Its values are still exact when they are zero:
    // every block scale word 0 (any divisor gives zeros), or a zero multiplier (ModelOpt
    // weight_scale_2 = 0; compressed-tensors weight_global_scale = +inf). Such a matrix is kept
    // as zero block scales under divisor 1, and the log and report say so. Anything else
    // (a NaN multiplier over live block scales) is refused by name.
    std::uint32_t weight_divisor_word() const {
        const auto name     = prefix_ + (fields_ == Nvfp4Fields::ModelOpt ? ".weight_scale_2"
                                                                          : ".weight_global_scale");
        const float stored  = scalar(name);
        const float divisor = fields_ == Nvfp4Fields::ModelOpt ? 1.0F / stored : stored;
        if (std::isfinite(stored) && stored > 0 && valid_positive_f32_word(f32_bits(divisor))) {
            return f32_bits(divisor);
        }
        const bool zero_multiplier = fields_ == Nvfp4Fields::ModelOpt
                                         ? (stored == 0 || (std::isfinite(stored) && stored > 0))
                                         : (std::isinf(stored) && stored > 0);
        if (block_scales_zero() || zero_multiplier) {
            zero_scales_ = true;
            dead_note_   = "weight scale " + py_format_g(stored, 9) + " on a zero matrix";
            return f32_bits(1.0F);
        }
        return fields_ == Nvfp4Fields::ModelOpt ? reciprocal_word(name) : divisor_word(name);
    }

    // An activation divisor calibration never measured (0, NaN or infinite) carries no range:
    // the activation route takes divisor 1 (block scales in E4M3's normal range for block
    // maxima from 2^-6 * 6 to 2688), and the log and report say so.
    std::uint32_t input_divisor_word() const {
        const auto name =
            prefix_ + (fields_ == Nvfp4Fields::ModelOpt ? ".input_scale" : ".input_global_scale");
        const float stored  = scalar(name);
        const float divisor = fields_ == Nvfp4Fields::ModelOpt ? 1.0F / stored : stored;
        if (std::isfinite(stored) && stored > 0 && valid_positive_f32_word(f32_bits(divisor))) {
            return f32_bits(divisor);
        }
        input_note_ = "uncalibrated activation scale " + py_format_g(stored, 9) + " taken as 1";
        return f32_bits(1.0F);
    }

    void signature(const TensorInfo& info, const Shape& expected, const char* dtype) const {
        if (info.shape != expected || info.dtype_name != dtype) {
            throw std::runtime_error(info.name + ": expected " + dtype + shape_text(expected) +
                                     ", got " + info.dtype_name + shape_text(info.shape));
        }
    }

    void resolve_nvfp4() const {
        const auto n = shape()[0];
        const auto k = shape()[1];
        if (k % 16) {
            throw std::runtime_error(prefix_ + ": NVFP4 source K must be divisible by 16");
        }
        const auto fields = nvfp4_layout(checkpoint_, prefix_);
        if (!fields) {
            throw std::runtime_error(prefix_ + ": no NVFP4 representation in this checkpoint");
        }
        fields_ = *fields;
        packed_ = &checkpoint_.describe(
            prefix_ + (fields_ == Nvfp4Fields::CompressedTensors ? ".weight_packed" : ".weight"));
        scales_ = &checkpoint_.describe(prefix_ + ".weight_scale");
        signature(*packed_, {n, k / 2}, "U8");
        signature(*scales_, {n, k / 16}, "F8_E4M3");
        weight_divisor_ = weight_divisor_word();
        divisor_value_  = f32_from_bits(weight_divisor_);
        if (checkpoint_.has(prefix_ + (fields_ == Nvfp4Fields::ModelOpt ? ".input_scale"
                                                                        : ".input_global_scale"))) {
            input_divisor_ = input_divisor_word();
        }
        nvfp4_ = true;
    }

    const Checkpoint& checkpoint_;
    std::string name_;
    std::string prefix_;
    MatrixFormat format_;
    mutable std::mutex resolve_mutex_;
    mutable std::atomic<bool> resolved_{false};
    mutable bool nvfp4_                   = false;
    mutable Nvfp4Fields fields_           = Nvfp4Fields::CompressedTensors;
    mutable const TensorInfo* packed_     = nullptr;
    mutable const TensorInfo* scales_     = nullptr;
    mutable std::uint32_t weight_divisor_ = 0;
    mutable float divisor_value_          = 0.0F;
    mutable std::optional<std::uint32_t> input_divisor_;
    mutable bool zero_scales_ = false; // a dead matrix: kept as zero block scales, divisor 1
    mutable std::string dead_note_;
    mutable std::string input_note_;
    mutable SourcePtr plain_;
    mutable std::shared_ptr<const QuantizedMatrix> decoder_;
};

// Flat values [begin, end) of an NVFP4 source read through its encoded rows against `divisor`:
// f32(e2m1 * e4m3) / divisor, as the Python stacked and divisor-scaled sources compute them.
void nvfp4_values(const Source& source, float divisor, std::uint64_t begin, std::uint64_t end,
                  float* out, ReadLog* log) {
    const auto k     = source.shape()[1];
    const auto first = begin / k;
    const auto last  = (end + k - 1) / k;
    EncodedRows words;
    source.encoded_rows(first, last, words, log);
    std::vector<float> row(k);
    for (auto r = first; r < last; ++r) {
        const auto low  = r == first ? begin - first * k : 0;
        const auto high = std::min<std::uint64_t>(k, end - r * k);
        dequantize_nvfp4_row(words.codes.data() + (r - first) * (k / 2),
                             words.scales.data() + (r - first) * (k / 16), divisor, k, row.data());
        std::memcpy(out + (r * k + low - begin), row.data() + low, (high - low) * 4);
    }
}

// ---------------------------------------------------------------------------------------------

class StackedSource final : public Source {
public:
    StackedSource(std::vector<SourcePtr> parts, Shape shape, std::string label)
        : Source(std::move(shape), std::move(label)), parts_(std::move(parts)) {}

    void resolve() const override {
        for (const auto& part : parts_) {
            part->resolve();
            if (!part->is_nvfp4()) {
                throw std::runtime_error(this->label() + ": stacked sources must all be NVFP4");
            }
        }
    }

    [[nodiscard]] bool is_nvfp4() const override { return true; }

    [[nodiscard]] bool provides_divisors() const override { return true; }

    [[nodiscard]] bool provides_input_divisor() const override {
        return parts_.front()->provides_input_divisor();
    }

    [[nodiscard]] bool calibrated_input() const override {
        return parts_.front()->calibrated_input();
    }

    [[nodiscard]] std::string encoding() const override { return parts_.front()->encoding(); }

    [[nodiscard]] bool dequantizes() const override { return true; }

    [[nodiscard]] bool keeps_losslessly() const override {
        return std::all_of(parts_.begin(), parts_.end(),
                           [](const SourcePtr& part) { return part->keeps_losslessly(); });
    }

    [[nodiscard]] bool python_readable() const override {
        return std::all_of(parts_.begin(), parts_.end(),
                           [](const SourcePtr& part) { return part->python_readable(); });
    }

    [[nodiscard]] std::uint32_t weight_divisor() const override {
        return parts_.front()->weight_divisor();
    }

    [[nodiscard]] std::uint32_t input_divisor() const override {
        return parts_.front()->input_divisor();
    }

    [[nodiscard]] ValueType value_type() const override { return ValueType::F32; }

    void encoded_rows(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                      ReadLog* log) const override {
        std::uint64_t cursor = 0;
        for (const auto& part : parts_) {
            const auto rows = part->shape()[0];
            const auto low  = std::max(first, cursor);
            const auto high = std::min(last, cursor + rows);
            if (low < high) { part->encoded_rows(low - cursor, high - cursor, out, log); }
            cursor += rows;
        }
        out.divisor = weight_divisor();
    }

    void values_f32(std::uint64_t begin, std::uint64_t end, float* out,
                    ReadLog* log) const override {
        nvfp4_values(*this, f32_from_bits(weight_divisor()), begin, end, out, log);
    }

    void values_raw(std::uint64_t begin, std::uint64_t end, void* out,
                    ReadLog* log) const override {
        values_f32(begin, end, static_cast<float*>(out), log);
    }

    void tensors(const Checkpoint& checkpoint, std::vector<const TensorInfo*>& out) const override {
        for (const auto& part : parts_) { part->tensors(checkpoint, out); }
    }

    void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const override {
        std::uint64_t cursor = 0;
        for (const auto& part : parts_) {
            const auto count = part->elements();
            const auto low   = std::max(begin, cursor);
            const auto high  = std::min(end, cursor + count);
            if (low < high) { part->ranges(low - cursor, high - cursor, out); }
            cursor += count;
        }
    }

private:
    std::vector<SourcePtr> parts_;
};

class ConcatSource final : public Source {
public:
    ConcatSource(std::vector<SourcePtr> parts, Shape shape, std::string label)
        : Source(std::move(shape), std::move(label)), parts_(std::move(parts)) {}

    void resolve() const override {
        for (const auto& part : parts_) { part->resolve(); }
        for (const auto& part : parts_) {
            if (part->value_type() != parts_.front()->value_type()) {
                throw std::runtime_error(label() + ": concatenated parts differ in dtype");
            }
        }
    }

    [[nodiscard]] ValueType value_type() const override { return parts_.front()->value_type(); }

    [[nodiscard]] std::string encoding() const override { return parts_.front()->encoding(); }

    [[nodiscard]] bool dequantizes() const override {
        return std::any_of(parts_.begin(), parts_.end(),
                           [](const SourcePtr& part) { return part->dequantizes(); });
    }

    [[nodiscard]] bool keeps_losslessly() const override {
        return std::all_of(parts_.begin(), parts_.end(),
                           [](const SourcePtr& part) { return part->keeps_losslessly(); });
    }

    [[nodiscard]] bool python_readable() const override {
        return std::all_of(parts_.begin(), parts_.end(),
                           [](const SourcePtr& part) { return part->python_readable(); });
    }

    // Row-scaled FP8, groupwise Q8 and U4 matrices stack by rows without any shared value: codes,
    // zero points and scales simply follow each other (the NVFP4 stack needs its divisor rule and
    // has its own class).
    [[nodiscard]] std::optional<QType> encoded_format() const override {
        const auto format = parts_.front()->encoded_format();
        if (format != QType::FP8_E4M3FN_ROW_BF16 && format != QType::FP8_E4M3FN_ROW_F32 &&
            format != QType::FP8_E4M3FN_B128_F32 && format != QType::Q8_G32_FP16 &&
            !(format && (u4_zp_group(*format) || is_ggml_block(*format)))) {
            return std::nullopt;
        }
        for (std::size_t i = 0; i < parts_.size(); ++i) {
            if (parts_[i]->encoded_format() != format) { return std::nullopt; }
            // One plane stores one column order.
            if (!same_order(parts_[i]->k_order(), parts_.front()->k_order())) {
                return std::nullopt;
            }
            // 128x128 FP8 blocks restart at every part: all but the last need whole blocks.
            if (format == QType::FP8_E4M3FN_B128_F32 && i + 1 < parts_.size() &&
                parts_[i]->shape()[0] % 128) {
                return std::nullopt;
            }
        }
        return format;
    }

    [[nodiscard]] KOrder k_order() const override {
        return encoded_format() ? parts_.front()->k_order() : nullptr;
    }

    void encoded_rows(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                      ReadLog* log) const override {
        if (!encoded_format()) {
            throw std::runtime_error(label() + ": encoded rows are unavailable");
        }
        std::uint64_t cursor = 0;
        for (const auto& part : parts_) {
            const auto rows = part->shape()[0];
            const auto low  = std::max(first, cursor);
            const auto high = std::min(last, cursor + rows);
            if (low < high) { part->encoded_rows(low - cursor, high - cursor, out, log); }
            cursor += rows;
        }
    }

    void values_f32(std::uint64_t begin, std::uint64_t end, float* out,
                    ReadLog* log) const override {
        each(begin, end,
             [&](const Source& part, std::uint64_t low, std::uint64_t high, std::uint64_t at) {
                 part.values_f32(low, high, out + at, log);
             });
    }

    void values_raw(std::uint64_t begin, std::uint64_t end, void* out,
                    ReadLog* log) const override {
        const auto word = value_bytes(value_type());
        each(begin, end,
             [&](const Source& part, std::uint64_t low, std::uint64_t high, std::uint64_t at) {
                 part.values_raw(low, high, static_cast<std::byte*>(out) + at * word, log);
             });
    }

    void tensors(const Checkpoint& checkpoint, std::vector<const TensorInfo*>& out) const override {
        for (const auto& part : parts_) { part->tensors(checkpoint, out); }
    }

    void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const override {
        each(begin, end,
             [&](const Source& part, std::uint64_t low, std::uint64_t high, std::uint64_t) {
                 part.ranges(low, high, out);
             });
    }

private:
    template <class F>
    void each(std::uint64_t begin, std::uint64_t end, F&& visit) const {
        std::uint64_t cursor = 0;
        for (const auto& part : parts_) {
            const auto count = part->elements();
            const auto low   = std::max(begin, cursor);
            const auto high  = std::min(end, cursor + count);
            if (low < high) { visit(*part, low - cursor, high - cursor, low - begin); }
            cursor += count;
        }
    }

    std::vector<SourcePtr> parts_;
};

class DivisorScaledSource final : public Source {
public:
    DivisorScaledSource(SourcePtr matrix, SourcePtr gate, SourcePtr up)
        : Source(matrix->shape(), "divisor\xc3\x97"
                                  "factor(" +
                                      matrix->label() + ")"),
          matrix_(std::move(matrix)), gate_(std::move(gate)), up_(std::move(up)) {}

    void resolve() const override {
        matrix_->resolve();
        gate_->resolve();
        up_->resolve();
        if (!matrix_->is_nvfp4()) {
            throw std::runtime_error(label() +
                                     ": only an encoded matrix can take a scaled divisor");
        }
    }

    [[nodiscard]] bool is_nvfp4() const override { return true; }

    [[nodiscard]] bool provides_divisors() const override { return true; }

    [[nodiscard]] bool provides_input_divisor() const override {
        return matrix_->provides_input_divisor();
    }

    [[nodiscard]] bool calibrated_input() const override { return matrix_->calibrated_input(); }

    [[nodiscard]] std::string encoding() const override { return matrix_->encoding(); }

    [[nodiscard]] bool dequantizes() const override { return true; }

    [[nodiscard]] bool keeps_losslessly() const override { return matrix_->keeps_losslessly(); }

    [[nodiscard]] bool python_readable() const override { return matrix_->python_readable(); }

    [[nodiscard]] std::uint32_t weight_divisor() const override {
        return scaled(matrix_->weight_divisor(), true);
    }

    [[nodiscard]] std::uint32_t input_divisor() const override {
        return scaled(matrix_->input_divisor(), false);
    }

    [[nodiscard]] ValueType value_type() const override { return ValueType::F32; }

    void encoded_rows(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                      ReadLog* log) const override {
        matrix_->encoded_rows(first, last, out, log);
        out.divisor = weight_divisor();
    }

    void values_f32(std::uint64_t begin, std::uint64_t end, float* out,
                    ReadLog* log) const override {
        nvfp4_values(*this, f32_from_bits(weight_divisor()), begin, end, out, log);
    }

    void values_raw(std::uint64_t begin, std::uint64_t end, void* out,
                    ReadLog* log) const override {
        values_f32(begin, end, static_cast<float*>(out), log);
    }

    void tensors(const Checkpoint& checkpoint, std::vector<const TensorInfo*>& out) const override {
        matrix_->tensors(checkpoint, out);
    }

    void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const override {
        matrix_->ranges(begin, end, out);
    }

private:
    // Python: struct.pack("<f", value * f) with value and f binary64, f = d_up / d_gate.
    std::uint32_t scaled(std::uint32_t word, bool multiply) const {
        const double value  = f32_from_bits(word);
        const double factor = static_cast<double>(f32_from_bits(up_->weight_divisor())) /
                              static_cast<double>(f32_from_bits(gate_->weight_divisor()));
        const auto result =
            f32_bits(static_cast<float>(multiply ? value * factor : value / factor));
        if (!valid_positive_f32_word(result)) {
            throw std::runtime_error(label() + ": scaled divisor is not finite and positive");
        }
        return result;
    }

    SourcePtr matrix_;
    SourcePtr gate_;
    SourcePtr up_;
};

// select_rows: row ranges of a matrix, in order. Values and (for an encoded matrix) encoded rows
// pass through; every span shares the matrix's divisors.
class SelectRowsSource final : public Source {
public:
    SelectRowsSource(SourcePtr matrix, std::vector<std::pair<std::uint64_t, std::uint64_t>> spans,
                     std::uint64_t rows)
        : Source({rows, matrix->shape()[1]}, "rows(" + matrix->label() + ", " +
                                                 std::to_string(spans.size()) + " spans, " +
                                                 std::to_string(rows) + " rows)"),
          matrix_(std::move(matrix)), spans_(std::move(spans)) {
        std::uint64_t cursor = 0;
        for (const auto& [first, last] : spans_) {
            starts_.push_back(cursor);
            cursor += last - first;
        }
    }

    void resolve() const override { matrix_->resolve(); }

    [[nodiscard]] bool provides_divisors() const override { return matrix_->provides_divisors(); }

    [[nodiscard]] bool provides_input_divisor() const override {
        return matrix_->provides_input_divisor();
    }

    [[nodiscard]] bool calibrated_input() const override { return matrix_->calibrated_input(); }

    [[nodiscard]] bool is_nvfp4() const override { return matrix_->is_nvfp4(); }

    [[nodiscard]] std::optional<QType> encoded_format() const override {
        return matrix_->encoded_format();
    }

    [[nodiscard]] std::uint32_t weight_divisor() const override {
        return matrix_->weight_divisor();
    }

    [[nodiscard]] std::uint32_t input_divisor() const override { return matrix_->input_divisor(); }

    [[nodiscard]] KOrder k_order() const override { return matrix_->k_order(); }

    [[nodiscard]] ValueType value_type() const override { return matrix_->value_type(); }

    [[nodiscard]] std::string encoding() const override { return matrix_->encoding(); }

    [[nodiscard]] bool dequantizes() const override { return matrix_->dequantizes(); }

    [[nodiscard]] bool keeps_losslessly() const override { return matrix_->keeps_losslessly(); }

    [[nodiscard]] bool python_readable() const override { return matrix_->python_readable(); }

    void values_f32(std::uint64_t begin, std::uint64_t end, float* out,
                    ReadLog* log) const override {
        each(begin, end, [&](std::uint64_t low, std::uint64_t high, std::uint64_t at) {
            matrix_->values_f32(low, high, out + at, log);
        });
    }

    void values_raw(std::uint64_t begin, std::uint64_t end, void* out,
                    ReadLog* log) const override {
        const auto word = value_bytes(value_type());
        each(begin, end, [&](std::uint64_t low, std::uint64_t high, std::uint64_t at) {
            matrix_->values_raw(low, high, static_cast<std::byte*>(out) + at * word, log);
        });
    }

    void encoded_rows(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                      ReadLog* log) const override {
        const auto k = shape()[1];
        std::optional<std::uint32_t> divisor;
        each(first * k, last * k, [&](std::uint64_t low, std::uint64_t high, std::uint64_t) {
            matrix_->encoded_rows(low / k, high / k, out, log);
            if (divisor && *divisor != out.divisor) {
                throw std::runtime_error(label() + ": incompatible encoded row groups");
            }
            divisor = out.divisor;
        });
    }

    void tensors(const Checkpoint& checkpoint, std::vector<const TensorInfo*>& out) const override {
        matrix_->tensors(checkpoint, out);
    }

    void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const override {
        each(begin, end, [&](std::uint64_t low, std::uint64_t high, std::uint64_t) {
            matrix_->ranges(low, high, out);
        });
    }

private:
    // Visit the source element ranges covering selected elements [begin, end).
    template <class F>
    void each(std::uint64_t begin, std::uint64_t end, F&& visit) const {
        const auto k      = shape()[1];
        const auto origin = begin;
        while (begin < end) {
            const auto row  = begin / k;
            const auto span = static_cast<std::size_t>(
                std::upper_bound(starts_.begin(), starts_.end(), row) - starts_.begin() - 1);
            const auto span_end = (starts_[span] + (spans_[span].second - spans_[span].first)) * k;
            const auto stop     = std::min(end, span_end);
            const auto source   = (spans_[span].first - starts_[span]) * k;
            visit(source + begin, source + stop, begin - origin);
            begin = stop;
        }
    }

    SourcePtr matrix_;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans_;
    std::vector<std::uint64_t> starts_;
};

// transpose_source: a small tensor read whole and permuted (Qwen's GDN convolution kernel).
class TransposeSource final : public Source {
public:
    TransposeSource(SourcePtr source, std::vector<std::size_t> axes, Shape shape)
        : Source(std::move(shape), "transpose(" + source->label() + "," + axes_text(axes) + ")"),
          source_(std::move(source)), axes_(std::move(axes)) {}

    void resolve() const override { source_->resolve(); }

    [[nodiscard]] ValueType value_type() const override { return source_->value_type(); }

    [[nodiscard]] std::string encoding() const override { return source_->encoding(); }

    [[nodiscard]] bool dequantizes() const override { return source_->dequantizes(); }

    [[nodiscard]] std::string decode_reason() const override {
        return "to permute its axes " + axes_text(axes_);
    }

    void values_f32(std::uint64_t begin, std::uint64_t end, float* out,
                    ReadLog* log) const override {
        const auto all = permuted(log);
        std::vector<float> values(all.size() / value_bytes(value_type()));
        words_to_f32(value_type(), all.data(), values.size(), values.data());
        std::memcpy(out, values.data() + begin, (end - begin) * 4);
    }

    void values_raw(std::uint64_t begin, std::uint64_t end, void* out,
                    ReadLog* log) const override {
        const auto all  = permuted(log);
        const auto word = value_bytes(value_type());
        std::memcpy(out, all.data() + begin * word, (end - begin) * word);
    }

    void tensors(const Checkpoint& checkpoint, std::vector<const TensorInfo*>& out) const override {
        source_->tensors(checkpoint, out);
    }

    void ranges(std::uint64_t, std::uint64_t, ReadLog& out) const override {
        source_->ranges(0, source_->elements(), out);
    }

private:
    static std::string axes_text(const std::vector<std::size_t>& axes) {
        std::string out = "(";
        for (std::size_t i = 0; i < axes.size(); ++i) {
            if (i) { out += ", "; }
            out += std::to_string(axes[i]);
        }
        if (axes.size() == 1) { out += ","; }
        return out + ")";
    }

    // The whole tensor, permuted: out[j] with j over the output (permuted) shape in C order.
    std::vector<std::byte> permuted(ReadLog* log) const {
        const auto word  = value_bytes(value_type());
        const auto count = source_->elements();
        std::vector<std::byte> in(count * word);
        source_->values_raw(0, count, in.data(), log);
        const auto& from = source_->shape();
        std::vector<std::uint64_t> stride(from.size(), 1);
        for (auto d = from.size(); d-- > 1;) { stride[d - 1] = stride[d] * from[d]; }
        std::vector<std::byte> out(count * word);
        std::vector<std::uint64_t> index(axes_.size(), 0);
        for (std::uint64_t j = 0; j < count; ++j) {
            std::uint64_t offset = 0;
            for (std::size_t d = 0; d < axes_.size(); ++d) {
                offset += index[d] * stride[axes_[d]];
            }
            std::memcpy(out.data() + j * word, in.data() + offset * word, word);
            for (auto d = axes_.size(); d-- > 0;) {
                if (++index[d] < from[axes_[d]]) { break; }
                index[d] = 0;
            }
        }
        return out;
    }

    SourcePtr source_;
    std::vector<std::size_t> axes_;
};

class Int32Source final : public Source {
public:
    Int32Source(std::vector<std::int32_t> values, std::string label)
        : Source({values.size()}, std::move(label)), values_(std::move(values)) {}

    [[nodiscard]] ValueType value_type() const override { return ValueType::I32; }

    [[nodiscard]] std::string encoding() const override { return "int32"; }

    void values_f32(std::uint64_t begin, std::uint64_t end, float* out, ReadLog*) const override {
        for (auto i = begin; i < end; ++i) { out[i - begin] = static_cast<float>(values_[i]); }
    }

    void values_raw(std::uint64_t begin, std::uint64_t end, void* out, ReadLog*) const override {
        std::memcpy(out, values_.data() + begin, (end - begin) * 4);
    }

    void tensors(const Checkpoint&, std::vector<const TensorInfo*>&) const override {}

    void ranges(std::uint64_t, std::uint64_t, ReadLog&) const override {}

private:
    std::vector<std::int32_t> values_;
};

} // namespace

std::uint64_t shape_elements(const Shape& shape) noexcept {
    std::uint64_t out = 1;
    for (const auto dim : shape) { out *= dim; }
    return out;
}

std::size_t value_bytes(ValueType type) noexcept {
    switch (type) {
    case ValueType::F64:
    case ValueType::I64:
        return 8;
    case ValueType::F32:
    case ValueType::I32:
        return 4;
    case ValueType::BF16:
    case ValueType::F16:
        return 2;
    default:
        return 1;
    }
}

std::uint32_t Source::weight_divisor() const {
    throw std::runtime_error(label() + ": source does not provide a weight divisor");
}

std::uint32_t Source::input_divisor() const {
    throw std::runtime_error(label() + ": source does not provide an input divisor");
}

void Source::encoded_rows(std::uint64_t, std::uint64_t, EncodedRows&, ReadLog*) const {
    throw std::runtime_error(label() + ": encoded rows are unavailable");
}

std::optional<Nvfp4Fields> nvfp4_layout(const Checkpoint& checkpoint, const std::string& prefix) {
    // compressed-tensors also packs integer (I32) and MXFP4 (U8 with E8M0 scales) weights into
    // weight_packed; NVFP4 is the U8 packing with E4M3 block scales.
    if (checkpoint.has(prefix + ".weight_packed") &&
        checkpoint.describe(prefix + ".weight_packed").dtype_name == "U8" &&
        checkpoint.has(prefix + ".weight_scale") &&
        checkpoint.describe(prefix + ".weight_scale").dtype_name == "F8_E4M3") {
        return Nvfp4Fields::CompressedTensors;
    }
    if (checkpoint.has(prefix + ".weight_scale_2") &&
        checkpoint.describe(prefix + ".weight").dtype_name == "U8") {
        return Nvfp4Fields::ModelOpt;
    }
    return std::nullopt;
}

SourcePtr tensor_source(const Checkpoint& checkpoint, const std::string& name, Shape shape) {
    return std::make_shared<TensorSource>(checkpoint, name, std::move(shape));
}

SourcePtr tensor_source(const Checkpoint& checkpoint, const std::string& name, Shape shape,
                        std::uint64_t offset, Shape source_shape) {
    return std::make_shared<TensorSource>(checkpoint, name, std::move(shape), offset,
                                          std::move(source_shape));
}

SourcePtr compressed_source(const Checkpoint& checkpoint, const std::string& prefix, Shape shape,
                            MatrixFormat format) {
    const auto label = checkpoint.path_string() + ":" + prefix + " (" +
                       (format == MatrixFormat::Fp8Row  ? "fp8_e4m3fn_row_bf16"
                        : format == MatrixFormat::Mxfp4 ? "mxfp4"
                                                        : "nvfp4") +
                       ")";
    return std::make_shared<MatrixSource>(checkpoint, prefix + ".weight", std::move(shape), format,
                                          label);
}

SourcePtr transpose_source(SourcePtr source, std::vector<std::size_t> axes, Shape shape) {
    return std::make_shared<TransposeSource>(std::move(source), std::move(axes), std::move(shape));
}

SourcePtr select_rows(SourcePtr matrix,
                      std::vector<std::pair<std::uint64_t, std::uint64_t>> spans) {
    if (matrix->shape().size() != 2 || spans.empty()) {
        throw std::runtime_error("row selection requires a matrix and nonempty ranges");
    }
    std::uint64_t rows = 0;
    for (const auto& [first, last] : spans) {
        if (!(first < last && last <= matrix->shape()[0])) {
            throw std::runtime_error(matrix->label() + ": invalid selected rows [" +
                                     std::to_string(first) + "," + std::to_string(last) + ")");
        }
        rows += last - first;
    }
    return std::make_shared<SelectRowsSource>(std::move(matrix), std::move(spans), rows);
}

namespace {

// A plain matrix quantized to NVFP4 by round to nearest (numerics.h quantize_nvfp4_rows). The
// matrix divisor needs the whole matrix's max|W|, read once at resolve(); rows are then encoded on
// demand. There is no calibrated activation divisor: the weight runs with BF16 activations.
class Nvfp4RoundedSource final : public Source {
public:
    Nvfp4RoundedSource(SourcePtr source, std::vector<SourcePtr> divisor_group)
        : Source(source->shape(), "nvfp4_rtn(" + source->label() + ")"), source_(std::move(source)),
          group_(std::move(divisor_group)) {
        if (shape().size() != 2 || shape()[1] % 16 != 0) {
            throw std::runtime_error(label() + ": NVFP4 needs a matrix with K % 16 == 0");
        }
    }

    void resolve() const override {
        source_->resolve();
        if (resolved_) { return; }
        // One divisor for the matrices a fused parent will hold together (q/k/v, gate/up).
        float max_abs = 0.0F;
        std::vector<SourcePtr> members = group_;
        if (members.empty()) { members.push_back(source_); }
        for (const auto& member : members) {
            member->resolve();
            const auto rows = member->shape()[0], k = member->shape()[1];
            std::vector<float> chunk;
            for (std::uint64_t r = 0; r < rows; r += 256) {
                const auto n = std::min<std::uint64_t>(256, rows - r);
                chunk.resize(n * k);
                member->values_f32(r * k, (r + n) * k, chunk.data(), nullptr);
                for (const float v : chunk) { max_abs = std::max(max_abs, std::fabs(v)); }
            }
        }
        divisor_  = nvfp4_divisor(max_abs);
        resolved_ = true;
    }

    [[nodiscard]] ValueType value_type() const override { return ValueType::F32; }
    [[nodiscard]] bool is_nvfp4() const override { return true; }
    [[nodiscard]] bool provides_divisors() const override { return true; }
    [[nodiscard]] bool provides_input_divisor() const override { return false; }
    [[nodiscard]] bool keeps_losslessly() const override { return false; }
    [[nodiscard]] bool python_readable() const override { return false; }

    [[nodiscard]] std::string encoding() const override { return source_->encoding(); }

    [[nodiscard]] std::uint32_t weight_divisor() const override {
        resolve();
        return f32_bits(divisor_);
    }

    void encoded_rows(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                      ReadLog* log) const override {
        resolve();
        const auto k = shape()[1], rows = last - first;
        std::vector<float> values(rows * k);
        source_->values_f32(first * k, last * k, values.data(), log);
        const auto code_at  = out.codes.size();
        const auto scale_at = out.scales.size();
        out.codes.resize(code_at + rows * (k / 2));
        out.scales.resize(scale_at + rows * (k / 16));
        quantize_nvfp4_rows(values.data(), rows, k, divisor_, out.codes.data() + code_at,
                            out.scales.data() + scale_at);
        out.divisor = f32_bits(divisor_);
    }

    // The represented values: the encoding decoded again.
    void values_f32(std::uint64_t begin, std::uint64_t end, float* out,
                    ReadLog* log) const override {
        const auto k = shape()[1];
        for (std::uint64_t r = begin / k; r * k < end; ++r) {
            EncodedRows words;
            encoded_rows(r, r + 1, words, log);
            std::vector<float> row(k);
            dequantize_nvfp4_row(words.codes.data(), words.scales.data(), divisor_, k, row.data());
            const auto from = std::max(begin, r * k), to = std::min(end, (r + 1) * k);
            std::copy(row.begin() + (from - r * k), row.begin() + (to - r * k), out + (from - begin));
        }
    }

    void values_raw(std::uint64_t begin, std::uint64_t end, void* out,
                    ReadLog* log) const override {
        values_f32(begin, end, static_cast<float*>(out), log);
    }

    void tensors(const Checkpoint& checkpoint, std::vector<const TensorInfo*>& out) const override {
        source_->tensors(checkpoint, out);
    }

    void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const override {
        source_->ranges(begin, end, out);
    }

private:
    SourcePtr source_;
    std::vector<SourcePtr> group_;
    mutable bool resolved_ = false;
    mutable float divisor_ = 1.0F;
};

} // namespace

SourcePtr nvfp4_rounded_source(SourcePtr source, std::vector<SourcePtr> divisor_group) {
    return std::make_shared<Nvfp4RoundedSource>(std::move(source), std::move(divisor_group));
}

SourcePtr scaled_source(SourcePtr source, float factor) {
    if (factor == 1.0F) { return source; }
    return std::make_shared<ScaledSource>(std::move(source), factor);
}

std::shared_ptr<const Source> matrix_source(const Checkpoint& checkpoint, const std::string& name,
                                            Shape shape, MatrixFormat format) {
    return std::make_shared<MatrixSource>(checkpoint, name, std::move(shape), format);
}

SourcePtr stacked_source(std::vector<SourcePtr> parts, const std::string& label) {
    if (parts.empty()) { throw std::runtime_error("a stacked source needs at least one matrix"); }
    const auto k       = parts.front()->shape()[1];
    std::uint64_t rows = 0;
    for (const auto& part : parts) {
        if (part->shape().size() != 2 || part->shape()[1] != k) {
            throw std::runtime_error(label + ": stacked matrices must share their column count");
        }
        rows += part->shape()[0];
    }
    const auto count = parts.size();
    return std::make_shared<StackedSource>(std::move(parts), Shape{rows, k},
                                           "stack(" + label + ", " + std::to_string(count) +
                                               " matrices, " + std::to_string(rows) + " rows)");
}

SourcePtr concat_source(std::vector<SourcePtr> parts, Shape shape, const std::string& label) {
    const auto count = parts.size();
    return std::make_shared<ConcatSource>(std::move(parts), std::move(shape),
                                          "concat(" + label + ", " + std::to_string(count) +
                                              " matrices)");
}

SourcePtr divisor_scaled_source(SourcePtr matrix, SourcePtr gate, SourcePtr up) {
    return std::make_shared<DivisorScaledSource>(std::move(matrix), std::move(gate), std::move(up));
}

SourcePtr gather_source(SourcePtr matrix, std::vector<std::int64_t> ids) {
    if (matrix->shape().size() != 2) { throw std::runtime_error("gather source must be a matrix"); }
    if (ids.empty()) { throw std::runtime_error("a gather source needs at least one row id"); }
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::uint64_t>(id) >= matrix->shape()[0]) {
            throw std::runtime_error("gather row id " + std::to_string(id) + " is outside the " +
                                     std::to_string(matrix->shape()[0]) + " source rows");
        }
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    spans.reserve(ids.size());
    for (const auto id : ids) {
        spans.emplace_back(static_cast<std::uint64_t>(id), static_cast<std::uint64_t>(id) + 1);
    }
    return select_rows(std::move(matrix), std::move(spans));
}

SourcePtr int32_source(std::vector<std::int32_t> values, std::string label) {
    return std::make_shared<Int32Source>(std::move(values), std::move(label));
}

bool decodes_quantized(const Checkpoint& checkpoint, const Source& source) {
    if (source.dequantizes()) { return true; }
    std::vector<const TensorInfo*> tensors;
    source.tensors(checkpoint, tensors);
    return std::any_of(tensors.begin(), tensors.end(), [](const TensorInfo* tensor) {
        return tensor->view ? tensor->view->dequantizes() : !ggml::is_plain(tensor->ggml_type);
    });
}

} // namespace ninferno::convert
