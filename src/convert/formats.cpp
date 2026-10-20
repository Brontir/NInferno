#include "convert/formats.h"
#include "core/ascii.h"

#include "convert/numerics.h"
#include "convert/tensor_view.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>

namespace ninferno::convert {

namespace {
// --round-tiny-scales (ConvertOptions::round_tiny_scales): set by convert() for the run.
std::atomic<bool> g_round_tiny_scales{false};

bool round_tiny_scales() noexcept { return g_round_tiny_scales.load(std::memory_order_relaxed); }
} // namespace

void set_round_tiny_scales(bool on) noexcept {
    g_round_tiny_scales.store(on, std::memory_order_relaxed);
}
namespace {

const OJson* member(const OJson& object, const char* key) {
    if (!object.is_object()) { return nullptr; }
    const auto it = object.find(key);
    return it == object.end() || it->is_null() ? nullptr : &*it;
}

// config.json's quantization_config, and the weight fields of its first config group
// (compressed-tensors) or its top level (AWQ, GPTQ, FP8 block quantizers).
struct QuantConfig {
    std::string method; // quant_method, lowercased
    std::string format; // compressed-tensors "format"
    int bits              = 0;
    std::uint64_t group   = 0;
    bool gptq_v2          = false;
    bool awq_packing      = false; // AutoRound exported with AutoAWQ's packing
    bool sym              = false; // GPTQ/AutoRound "sym": true (zero points must be 2^(bits-1))
    std::uint64_t block_n = 0, block_k = 0;
    std::string refused; // why the checkpoint's quantization has no native execution
};

QuantConfig quant_config(const OJson& config) {
    QuantConfig out;
    const auto* q = member(config, "quantization_config");
    if (!q) {
        if (const auto* text = member(config, "text_config")) {
            q = member(*text, "quantization_config");
        }
    }
    if (!q) { return out; }
    if (const auto* v = member(*q, "quant_method"); v && v->is_string()) {
        out.method = ascii_lower(v->get<std::string>());
    }
    if (const auto* v = member(*q, "format"); v && v->is_string()) {
        out.format = v->get<std::string>();
    }
    if (const auto* v = member(*q, "bits"); v && v->is_number_integer()) {
        out.bits = v->get<int>();
    }
    if (const auto* v = member(*q, "w_bit"); v && v->is_number_integer()) {
        out.bits = v->get<int>();
    }
    if (const auto* v = member(*q, "group_size");
        v && v->is_number_integer() && v->get<long>() > 0) {
        out.group = v->get<std::uint64_t>();
    }
    if (const auto* v = member(*q, "q_group_size"); v && v->is_number_integer()) {
        out.group = v->get<std::uint64_t>();
    }
    if (const auto* v = member(*q, "checkpoint_format"); v && v->is_string()) {
        out.gptq_v2 = v->get<std::string>() == "gptq_v2";
    }
    if (out.method == "gptq" && out.format == "gptq_v2") { out.gptq_v2 = true; }
    if (const auto* v = member(*q, "sym"); v && v->is_boolean()) { out.sym = v->get<bool>(); }
    // AutoRound (quant_method "auto-round"): its packing_format names the layout it exported.
    // auto_round:auto_gptq / auto_round:gptq store GPTQ's words with zero - 1 (GPTQ v1);
    // auto_round and auto_round:gptqmodel store the zero point itself (as gptq_v2);
    // auto_round:auto_awq stores AutoAWQ's words.
    if (out.method == "auto-round" || out.method == "intel/auto-round") {
        std::string packing = "auto_round:auto_gptq";
        if (const auto* v = member(*q, "packing_format"); v && v->is_string()) {
            packing = v->get<std::string>();
        }
        out.awq_packing = packing.find("awq") != std::string::npos;
        out.gptq_v2     = packing.find("gptqmodel") != std::string::npos ||
                      packing.find("gptq") == std::string::npos;
        out.method      = out.awq_packing ? "awq" : "gptq";
    }
    // compressed-tensors transforms (QuIP/SpinQuant Hadamard rotations) and sparsity (2:4
    // packings) change what the stored words mean; no engine format executes them.
    if (const auto* v = member(*q, "transform_config"); v && v->is_object() && !v->empty()) {
        out.refused = "compressed-tensors transform_config (Hadamard/rotation transforms of "
                      "weights and activations)";
    }
    if (const auto* v = member(*q, "sparsity_config"); v && v->is_object() && !v->empty()) {
        const auto* format = member(*v, "format");
        if (!format || !format->is_string() || format->get<std::string>() != "dense") {
            out.refused = "compressed-tensors sparsity_config (sparse packings)";
        }
    }
    if (const auto* v = member(*q, "weight_block_size"); v && v->is_array() && v->size() == 2) {
        out.block_n = (*v)[0].get<std::uint64_t>();
        out.block_k = (*v)[1].get<std::uint64_t>();
    }
    if (const auto* groups = member(*q, "config_groups"); groups && groups->is_object()) {
        for (const auto& [name, group] : groups->items()) {
            const auto* weights = member(group, "weights");
            if (!weights) { continue; }
            if (const auto* v = member(*weights, "num_bits"); v && v->is_number_integer()) {
                out.bits = v->get<int>();
            }
            if (const auto* v = member(*weights, "group_size"); v && v->is_number_integer()) {
                out.group = v->get<std::uint64_t>();
            }
            if (const auto* v = member(*weights, "block_structure");
                v && v->is_array() && v->size() == 2) {
                out.block_n = (*v)[0].get<std::uint64_t>();
                out.block_k = (*v)[1].get<std::uint64_t>();
            }
            break;
        }
    }
    return out;
}

std::string shape_text(const std::vector<std::uint64_t>& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        out += (i ? ", " : "") + std::to_string(shape[i]);
    }
    return out + "]";
}

[[noreturn]] void bad(const TensorInfo& tensor, const std::string& expected) {
    throw std::runtime_error(tensor.name + ": expected " + expected + ", got " + tensor.dtype_name +
                             shape_text(tensor.shape));
}

// A floating-point scale word of any of the checkpoint float dtypes, as binary32.
float scale_value(const TensorInfo& tensor, std::uint64_t index) {
    const auto* data = tensor.data;
    if (tensor.dtype_name == "F32") {
        float v;
        std::memcpy(&v, data + 4 * index, 4);
        return v;
    }
    std::uint16_t w;
    std::memcpy(&w, data + 2 * index, 2);
    if (tensor.dtype_name == "BF16") { return bf16_to_f32(w); }
    if (tensor.dtype_name == "F16") { return f16_to_f32(w); }
    throw std::runtime_error(tensor.name + ": unsupported scale dtype " + tensor.dtype_name);
}

bool float_dtype(const TensorInfo& tensor) {
    return tensor.dtype_name == "F32" || tensor.dtype_name == "BF16" || tensor.dtype_name == "F16";
}

std::uint32_t load_i32(const std::byte* data, std::uint64_t index) {
    std::uint32_t v;
    std::memcpy(&v, data + 4 * index, 4);
    return v;
}

// Row-major tensors: rows [first, last) of `row_bytes` each.
void add_rows(ReadLog& log, const TensorInfo& tensor, std::uint64_t first, std::uint64_t last,
              std::uint64_t row_bytes) {
    log.add(tensor, tensor.data + first * row_bytes, (last - first) * row_bytes);
}

// ---------------------------------------------------------------------------------------------
// FP8 E4M3 with per-tensor, per-channel or block scales.

class Fp8Matrix final : public QuantizedMatrix {
public:
    enum class Scale { Tensor, Channel, Block };

    Fp8Matrix(const Checkpoint& checkpoint, const TensorInfo& weight, const TensorInfo& scale,
              std::uint64_t n, std::uint64_t k, const QuantConfig& config)
        : weight_(weight), scale_(scale), n_(n), k_(k) {
        if (weight.shape != std::vector<std::uint64_t>{n, k}) {
            bad(weight, "F8_E4M3" + shape_text({n, k}));
        }
        if (!float_dtype(scale)) { bad(scale, "a floating-point scale"); }
        const auto count = scale.elements();
        if (count == 1) {
            kind_ = Scale::Tensor;
        } else if (count == n && (scale.shape.size() == 1 || scale.shape.back() == 1)) {
            kind_ = Scale::Channel;
        } else if (scale.shape.size() == 2) {
            kind_    = Scale::Block;
            block_n_ = config.block_n ? config.block_n : 128;
            block_k_ = config.block_k ? config.block_k : 128;
            if (scale.shape[0] != (n + block_n_ - 1) / block_n_ ||
                scale.shape[1] != (k + block_k_ - 1) / block_k_) {
                bad(scale, "block scales of " + std::to_string(block_n_) + "x" +
                               std::to_string(block_k_) + " blocks");
            }
        } else {
            bad(scale, "a per-tensor, per-channel or block scale");
        }
        // Scales are read while planning (row scales decide exactness) and pinned.
        std::vector<std::byte> raw(scale.bytes);
        checkpoint.read(scale, raw.data());
        TensorInfo copy = scale;
        copy.data       = raw.data();
        if (kind_ == Scale::Block) {
            block_scale_.resize(scale.elements());
            for (std::uint64_t i = 0; i < block_scale_.size(); ++i) {
                const float value = scale_value(copy, i);
                if (!std::isfinite(value) || value < 0.0F || std::signbit(value)) {
                    throw std::runtime_error(scale.name +
                                             ": FP8 scales must be nonnegative and finite");
                }
                block_scale_[i] = value;
            }
            return;
        }
        row_scale_.resize(n);
        row_word_.resize(n);
        exact_ = true;
        for (std::uint64_t r = 0; r < n; ++r) {
            const float value = scale_value(copy, kind_ == Scale::Tensor ? 0 : r);
            if (!std::isfinite(value) || value < 0.0F || std::signbit(value)) {
                throw std::runtime_error(scale.name +
                                         ": FP8 scales must be nonnegative and finite");
            }
            row_scale_[r] = value;
            row_word_[r]  = f32_to_bf16(value);
            exact_        = exact_ && bf16_to_f32(row_word_[r]) == value;
        }
        python_ = kind_ == Scale::Channel && scale.dtype_name == "BF16";
    }

    std::string format() const override {
        const auto dtype = ascii_lower(scale_.dtype_name);
        switch (kind_) {
        case Scale::Tensor:
            return "fp8 e4m3 per-tensor " + dtype + " scale";
        case Scale::Channel:
            return "fp8 e4m3 per-channel " + dtype + " scale";
        case Scale::Block:
            return "fp8 e4m3 " + std::to_string(block_n_) + "x" + std::to_string(block_k_) +
                   " block " + dtype + " scale";
        }
        return {};
    }

    // Per-tensor and per-channel scales: row_bf16 when every one is a BF16 value, else row_f32
    // (every F32, F16 and BF16 scale is a binary32 value). 128x128 blocks: b128_f32. Other block
    // shapes have no engine format.
    std::optional<QType> keeps() const override {
        if (kind_ == Scale::Block) {
            return block_n_ == 128 && block_k_ == 128
                       ? std::optional<QType>(QType::FP8_E4M3FN_B128_F32)
                       : std::nullopt;
        }
        return exact_ ? QType::FP8_E4M3FN_ROW_BF16 : QType::FP8_E4M3FN_ROW_F32;
    }

    bool keeps_losslessly() const override { return keeps().has_value(); }

    bool python_readable() const override { return python_; }

    void dequantize(std::uint64_t first, std::uint64_t last, float* out,
                    ReadLog* log) const override {
        const auto* codes = reinterpret_cast<const std::uint8_t*>(weight_.data);
        for (auto r = first; r < last; ++r) {
            const auto* row = codes + r * k_;
            float* dst      = out + (r - first) * k_;
            for (std::uint64_t c = 0; c < k_; ++c) {
                if ((row[c] & 0x7F) == 0x7F) {
                    throw std::runtime_error(weight_.name + ": FP8 NaN code");
                }
                const float scale =
                    kind_ == Scale::Block
                        ? block_scale_[(r / block_n_) * scale_.shape[1] + c / block_k_]
                        : row_scale_[r];
                dst[c] = kE4m3Values[row[c]] * scale;
            }
        }
        if (log) {
            add_rows(*log, weight_, first, last, k_);
            (void)0; // scales are pinned
        }
    }

    // Codes [rows, k] and the scales of the kept format: BF16 or binary32 words per row, or for
    // 128x128 blocks the binary32 words of every block row the rows touch ([ceil, K/128] each;
    // `first` is then a multiple of 128).
    void append_encoded(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                        ReadLog* log) const override {
        const auto kept   = keeps();
        const auto* codes = reinterpret_cast<const std::uint8_t*>(weight_.data);
        const auto at     = out.codes.size();
        out.codes.insert(out.codes.end(), codes + first * k_, codes + last * k_);
        const auto push_f32 = [&](float value) {
            const auto bits = f32_bits(value);
            for (int b = 0; b < 4; ++b) {
                out.scales.push_back(static_cast<std::uint8_t>(bits >> (8 * b)));
            }
        };
        for (auto r = first; r < last; ++r) {
            const float scale =
                kept == QType::FP8_E4M3FN_B128_F32 ? 1.0F : row_scale_[r];
            for (std::uint64_t c = 0; c < k_; ++c) {
                const auto code = out.codes[at + (r - first) * k_ + c];
                if ((code & 0x7F) == 0x7F) {
                    throw std::runtime_error(weight_.name + ": FP8 NaN code");
                }
                const float block_value =
                    kept == QType::FP8_E4M3FN_B128_F32
                        ? block_scale_[(r / 128) * scale_.shape[1] + c / 128]
                        : scale;
                if (block_value == 0.0F && (code & 0x7F) != 0) {
                    throw std::runtime_error(weight_.name +
                                             ": a zero scale requires only zero FP8 codes");
                }
            }
            if (kept == QType::FP8_E4M3FN_ROW_BF16) {
                const auto word = row_word_[r];
                out.scales.push_back(static_cast<std::uint8_t>(word & 0xFF));
                out.scales.push_back(static_cast<std::uint8_t>(word >> 8));
            } else if (kept == QType::FP8_E4M3FN_ROW_F32) {
                push_f32(row_scale_[r]);
            }
        }
        if (kept == QType::FP8_E4M3FN_B128_F32) {
            if (first % 128) {
                throw std::logic_error(weight_.name + ": FP8 block rows must start at a block");
            }
            const auto columns = scale_.shape[1];
            for (auto b = first / 128; b < (last + 127) / 128; ++b) {
                for (std::uint64_t c = 0; c < columns; ++c) {
                    push_f32(block_scale_[b * columns + c]);
                }
            }
        }
        if (log) { add_rows(*log, weight_, first, last, k_); }
    }

    void tensors(std::vector<const TensorInfo*>& out) const override {
        out.push_back(&weight_);
        out.push_back(&scale_);
    }

    void ranges(std::uint64_t first, std::uint64_t last, ReadLog& out) const override {
        add_rows(out, weight_, first, last, k_);
    }

private:
    const TensorInfo& weight_;
    const TensorInfo& scale_;
    std::uint64_t n_, k_;
    Scale kind_            = Scale::Tensor;
    std::uint64_t block_n_ = 0, block_k_ = 0;
    std::vector<float> row_scale_;
    std::vector<float> block_scale_;
    std::vector<std::uint16_t> row_word_;
    bool exact_  = false;
    bool python_ = false;
};

// ---------------------------------------------------------------------------------------------
// Integer weights: compressed-tensors pack-quantized / int-quantized, AutoAWQ, GPTQ.

class IntMatrix final : public QuantizedMatrix {
public:
    enum class Layout { CompressedPacked, CompressedInt8, Awq, Gptq };

    struct Parts {
        const TensorInfo* weight = nullptr; // packed or I8 codes
        const TensorInfo* scale  = nullptr;
        const TensorInfo* zero   = nullptr; // optional
        const TensorInfo* g_idx  = nullptr; // optional
    };

    // `symmetric`: the config declares symmetric quantization, so every zero point must decode to
    // 2^(bits-1); a checkpoint whose words say otherwise is refused rather than read with a
    // zero-point convention it does not use (GPTQ v1 stores zero - 1, v2 and AutoRound's own
    // packing the zero itself).
    IntMatrix(Layout layout, Parts parts, std::uint64_t n, std::uint64_t k, int bits,
              std::uint64_t group, bool gptq_v2, bool symmetric = false)
        : layout_(layout), p_(parts), n_(n), k_(k), bits_(bits), group_(group), gptq_v2_(gptq_v2) {
        if (bits_ != 2 && bits_ != 3 && bits_ != 4 && bits_ != 8) {
            throw std::runtime_error(p_.weight->name + ": " + std::to_string(bits_) +
                                     "-bit integer weights are not supported");
        }
        if (bits_ == 3 && layout_ != Layout::CompressedPacked) {
            throw std::runtime_error(
                p_.weight->name + ": 3-bit integer weights are read from compressed-tensors only");
        }
        pack_         = 32 / static_cast<std::uint64_t>(bits_);
        stream_       = 32 % bits_ != 0;
        mask_         = (1U << bits_) - 1U;
        offset_       = 1 << (bits_ - 1);
        const auto& w = *p_.weight;
        const auto& s = *p_.scale;
        if (!float_dtype(s)) { bad(s, "a floating-point scale"); }
        switch (layout_) {
        case Layout::CompressedPacked: {
            const auto words = stream_ ? (k * bits_ + 31) / 32 : (k + pack_ - 1) / pack_;
            if (w.dtype_name != "I32" || w.shape != std::vector<std::uint64_t>{n, words}) {
                bad(w, "I32" + shape_text({n, words}));
            }
            if (stream_ && p_.zero) {
                bad(*p_.zero, "no zero point (3-bit codes are read as symmetric only)");
            }
            groups_ = s.elements() / n;
            break;
        }
        case Layout::CompressedInt8:
            if (w.dtype_name != "I8" || w.shape != std::vector<std::uint64_t>{n, k}) {
                bad(w, "I8" + shape_text({n, k}));
            }
            groups_ = s.elements() / n;
            break;
        case Layout::Awq:
            if (bits_ != 4 || w.dtype_name != "I32" ||
                w.shape != std::vector<std::uint64_t>{k, n / 8} || n % 8) {
                bad(w, "AWQ I32" + shape_text({k, n / 8}));
            }
            groups_ = s.shape.at(0);
            break;
        case Layout::Gptq:
            if (w.dtype_name != "I32" ||
                w.shape != std::vector<std::uint64_t>{(k + pack_ - 1) / pack_, n}) {
                bad(w, "GPTQ I32" + shape_text({(k + pack_ - 1) / pack_, n}));
            }
            groups_ = s.shape.at(0);
            break;
        }
        if (!groups_ || s.elements() != groups_ * n) {
            bad(s, "one scale per output row and group");
        }
        // The configured group, unless the scales say otherwise (a per-channel scale, or a
        // compressed-tensors checkpoint whose config groups quantize matrices differently).
        if (!group_ || (!p_.g_idx && (k + group_ - 1) / group_ != groups_)) {
            group_ = (k + groups_ - 1) / groups_;
        }
        if (!p_.g_idx && (k + group_ - 1) / group_ != groups_) {
            bad(s,
                std::to_string((k + group_ - 1) / group_) + " groups of " + std::to_string(group_));
        }
        if (p_.g_idx &&
            (p_.g_idx->dtype_name != "I32" || p_.g_idx->shape != std::vector<std::uint64_t>{k})) {
            bad(*p_.g_idx, "I32" + shape_text({k}));
        }
        // A g_idx that maps every column to its own consecutive group is not act-order.
        if (p_.g_idx) {
            for (std::uint64_t c = 0; c < k_ && !act_order_; ++c) {
                act_order_ = load_i32(p_.g_idx->data, c) != c / group_;
            }
        }
        if (symmetric && p_.zero && (layout_ == Layout::Gptq || layout_ == Layout::Awq)) {
            for (std::uint64_t r = 0; r < n_; ++r) {
                for (std::uint64_t g = 0; g < groups_; ++g) {
                    if (zero(r, g) != offset_) {
                        throw std::runtime_error(
                            p_.zero->name + ": the config declares symmetric quantization but "
                                            "a zero point decodes to " +
                            std::to_string(zero(r, g)) + ", not " + std::to_string(offset_) +
                            (gptq_v2_ ? " (read as gptq_v2: stored zero)"
                                      : " (read as GPTQ v1: stored zero + 1)"));
                    }
                }
            }
        }
        plan_kept();
    }

    // The engine format that holds these words exactly, if any (plan_kept()).
    std::optional<QType> keeps() const override { return kept_; }

    bool keeps_losslessly() const override { return kept_.has_value(); }

    // Act-order: the columns sorted by group (k_order()), so every group is G consecutive values.
    KOrder k_order() const override { return kept_ ? order_ : nullptr; }

    // u4 formats: unsigned codes (element 2i in the low nibble), then per kept group of G values
    // its unsigned zero point and binary16 scale, the columns in k_order() when one is set.
    // q8_g32_fp16: the int8 code - zero per value, then per 32 values the binary16 scale.
    void append_encoded(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                        ReadLog* log) const override {
        if (!kept_) { throw std::logic_error(p_.weight->name + ": no engine format keeps it"); }
        const auto push_scale = [&](float value) {
            const auto word = f32_to_f16(value);
            out.scales.push_back(static_cast<std::uint8_t>(word & 0xFF));
            out.scales.push_back(static_cast<std::uint8_t>(word >> 8));
        };
        if (q3_group(*kept_)) {
            const std::uint64_t padded = (k_ + 127) / 128 * 128;
            for (auto r = first; r < last; ++r) {
                const auto start = out.codes.size();
                out.codes.resize(start + padded * 3 / 8, std::uint8_t{0});
                for (std::uint64_t c = 0; c < k_; ++c) {
                    const auto u   = static_cast<unsigned>(code(r, c) - zero(r, group_of(c))) & 7U;
                    const auto bit = 3 * c;
                    out.codes[start + bit / 8] |= static_cast<std::uint8_t>(u << (bit % 8));
                    if (bit % 8 > 5) {
                        out.codes[start + bit / 8 + 1] |=
                            static_cast<std::uint8_t>(u >> (8 - bit % 8));
                    }
                }
                for (std::uint64_t j = 0; j < padded / kept_group_; ++j) {
                    push_scale(kept_group_ * j < k_ ? scale(r, kept_group_ * j / group_) : 0.0F);
                }
            }
        } else if (*kept_ == QType::Q8_G32_FP16) {
            for (auto r = first; r < last; ++r) {
                for (std::uint64_t c = 0; c < k_; ++c) {
                    const int q = code(r, c) - zero(r, group_of(c));
                    out.codes.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(q)));
                }
                for (std::uint64_t j = 0; j < k_ / 32; ++j) { push_scale(scale(r, 32 * j / group_)); }
            }
        } else {
            const int bias = code_bias();
            const auto col = [&](std::uint64_t j) { return order_ ? (*order_)[j] : j; };
            for (auto r = first; r < last; ++r) {
                for (std::uint64_t c = 0; c < k_; c += 2) {
                    const auto lo = static_cast<unsigned>(code(r, col(c)) + bias);
                    const auto hi = static_cast<unsigned>(code(r, col(c + 1)) + bias);
                    out.codes.push_back(static_cast<std::uint8_t>(lo | (hi << 4)));
                }
                for (std::uint64_t j = 0; j < k_ / kept_group_; ++j) {
                    const auto g = kept_group_ * j / group_;
                    out.zeros.push_back(static_cast<std::uint8_t>(zero(r, g) + bias));
                    push_scale(scale(r, g));
                }
            }
            out.order = order_;
        }
        if (log) { ranges(first, last, *log); }
    }

    std::string format() const override {
        std::string name;
        switch (layout_) {
        case Layout::CompressedPacked:
            name = "compressed-tensors int" + std::to_string(bits_) + " pack-quantized";
            break;
        case Layout::CompressedInt8:
            name = "compressed-tensors int8";
            break;
        case Layout::Awq:
            name = "awq int4";
            break;
        case Layout::Gptq:
            name = std::string(gptq_v2_ ? "gptq_v2" : "gptq") + " int" + std::to_string(bits_);
            break;
        }
        name += groups_ == 1 ? " per-channel" : " g" + std::to_string(group_);
        name += p_.zero ? " zero-point" : " symmetric";
        if (act_order_) { name += " act-order"; }
        if (!kept_ && !not_kept_.empty()) { name += " (not kept: " + not_kept_ + ")"; }
        if (kept_ && rounded_) {
            name += " (" + std::to_string(rounded_) +
                    " scales below binary16's normal range rounded: --round-tiny-scales)";
        }
        return name;
    }

    void dequantize(std::uint64_t first, std::uint64_t last, float* out,
                    ReadLog* log) const override {
        for (auto r = first; r < last; ++r) {
            float* dst = out + (r - first) * k_;
            for (std::uint64_t c = 0; c < k_; ++c) {
                const auto g = group_of(c);
                const int q  = code(r, c);
                const int z  = zero(r, g);
                dst[c]       = static_cast<float>(q - z) * scale(r, g);
            }
        }
        if (log && (layout_ == Layout::CompressedPacked || layout_ == Layout::CompressedInt8)) {
            add_rows(*log, *p_.weight, first, last, p_.weight->bytes / n_);
            add_rows(*log, *p_.scale, first, last, p_.scale->bytes / n_);
        }
    }

    void tensors(std::vector<const TensorInfo*>& out) const override {
        for (const auto* t : {p_.weight, p_.scale, p_.zero, p_.g_idx}) {
            if (t) { out.push_back(t); }
        }
    }

    void ranges(std::uint64_t first, std::uint64_t last, ReadLog& out) const override {
        if (layout_ == Layout::CompressedPacked || layout_ == Layout::CompressedInt8) {
            add_rows(out, *p_.weight, first, last, p_.weight->bytes / n_);
            add_rows(out, *p_.scale, first, last, p_.scale->bytes / n_);
        }
    }

private:
    // compressed-tensors stores signed codes and zero points (offset by 2^(bits-1)); the engine
    // format stores them unsigned. The other layouts store unsigned words already.
    int code_bias() const { return layout_ == Layout::CompressedPacked ? offset_ : 0; }

    // Which engine format holds (code - zero) * scale of every value exactly:
    //  * 4-bit codes: u4_g<G>_zp_fp16 with G the largest of 128, 64 and 32 that divides both K and
    //    the published group (a per-channel scale, or a group of 256, repeats its zero point and
    //    scale over the G-value groups it covers), when every scale is exact in binary16 and every
    //    unsigned zero point is 0..127 (128 + zero is exact in BF16, which the kernels' byte-code
    //    conversion needs; a 4-bit zero point is at most 16);
    //  * 8-bit codes: q8_g32_fp16 (code - zero as int8, the scale repeated per 32 values) when K is
    //    whole 32-value groups, the group is per-channel or a multiple of 32, every zero point is
    //    the neutral one (so code - zero is an int8 value) and every scale is exact in binary16.
    // Act-order 4-bit codes (a g_idx other than c / group) are kept the same way with their
    // columns sorted by group, when every group has exactly `group` columns: the plane then stores
    // that column order (RowGroupZpOrdered) and the operator gathers its input in it, so no value
    // changes. Not kept: act-order 8-bit codes (q8_g32_fp16 has no column order), 2-bit codes,
    // anything else.
    // A value that stays unkept names why in not_kept_ (format() reports it with the refusal).
    void plan_kept() {
        if (act_order_ && (bits_ != 4 || !sort_by_group())) {
            not_kept_ = bits_ != 4 ? "act-order " + std::to_string(bits_) + "-bit codes"
                                   : "act-order groups of unequal size";
            return;
        }
        std::uint64_t target = 0;
        if (bits_ == 4) {
            for (const std::uint64_t g : {128, 64, 32}) {
                if (k_ % g == 0 && group_ % g == 0) {
                    target = g;
                    break;
                }
            }
        } else if (bits_ == 8 && k_ % 32 == 0 && (groups_ == 1 || group_ % 32 == 0)) {
            target = 32;
        } else if (bits_ == 3) {
            // q3_g<G>_fp16 pads K to 128: a group must not straddle the logical end.
            for (const std::uint64_t g : {128, 64}) {
                if (k_ % g == 0 && group_ % g == 0) {
                    target = g;
                    break;
                }
            }
        }
        if (!target) {
            not_kept_ = std::to_string(bits_) + "-bit codes in groups of " + std::to_string(group_) +
                        " over K=" + std::to_string(k_);
            return;
        }
        const int bias    = code_bias();
        const int neutral = layout_ == Layout::Gptq ? offset_ : 0;
        std::uint64_t inexact = 0, zeros = 0, tiny = 0;
        float smallest = 0.0f;
        for (std::uint64_t r = 0; r < n_; ++r) {
            for (std::uint64_t g = 0; g < groups_; ++g) {
                const float s = scale(r, g);
                if (!std::isfinite(s) || f16_to_f32(f32_to_f16(s)) != s) {
                    smallest = inexact++ == 0 ? std::fabs(s) : std::min(smallest, std::fabs(s));
                    // Below binary16's normal range (2^-14): the subnormal keeps fewer bits.
                    tiny += std::isfinite(s) && std::fabs(s) < 6.103515625e-05F ? 1 : 0;
                }
                const int z = zero(r, g);
                zeros += bits_ == 8 || bits_ == 3 ? z != neutral : (z + bias < 0 || z + bias > 127);
            }
        }
        if (inexact && inexact == tiny && !zeros && round_tiny_scales()) {
            rounded_ = inexact; // the user chose rounding them over decoding the matrix
            inexact  = 0;
        }
        if (inexact || zeros) {
            char text[200];
            std::snprintf(text, sizeof(text),
                          inexact ? "%llu of %llu scales not exact in binary16 (smallest |s| %g)"
                                  : "%llu of %llu zero points outside the kept range",
                          static_cast<unsigned long long>(inexact ? inexact : zeros),
                          static_cast<unsigned long long>(n_ * groups_),
                          static_cast<double>(smallest));
            not_kept_ = text;
            if (inexact && inexact == tiny && !zeros) {
                not_kept_ += "; --round-tiny-scales keeps it with those scales rounded";
            }
            return;
        }
        kept_group_ = target;
        if (bits_ == 8) {
            kept_ = QType::Q8_G32_FP16;
        } else if (bits_ == 3) {
            kept_ = target == 128 ? QType::Q3_G128_FP16 : QType::Q3_G64_FP16;
        } else {
            kept_ = target == 128  ? QType::U4_G128_ZP_FP16
                    : target == 64 ? QType::U4_G64_ZP_FP16
                                   : QType::U4_G32_ZP_FP16;
        }
    }

    // order_: the columns sorted by g_idx (stable), when that makes every group `group_`
    // consecutive columns.
    bool sort_by_group() {
        if (k_ % group_ || k_ / group_ != groups_) { return false; }
        std::vector<std::uint64_t> next(groups_, 0);
        for (std::uint64_t c = 0; c < k_; ++c) {
            const auto g = group_of(c);
            if (next[g] == group_) { return false; }
            ++next[g];
        }
        auto order = std::make_shared<std::vector<std::uint32_t>>(k_);
        for (std::uint64_t g = 0; g < groups_; ++g) { next[g] = g * group_; }
        for (std::uint64_t c = 0; c < k_; ++c) {
            (*order)[next[group_of(c)]++] = static_cast<std::uint32_t>(c);
        }
        order_ = std::move(order);
        return true;
    }

    std::uint64_t group_of(std::uint64_t c) const {
        if (p_.g_idx) {
            const auto g = load_i32(p_.g_idx->data, c);
            if (g >= groups_) { throw std::runtime_error(p_.g_idx->name + ": group out of range"); }
            return g;
        }
        return c / group_;
    }

    // Signed code of row r (output channel), column c (input channel).
    int code(std::uint64_t r, std::uint64_t c) const {
        switch (layout_) {
        case Layout::CompressedPacked: {
            const auto row = r * p_.weight->shape[1];
            if (stream_) {
                const auto bit     = c * static_cast<std::uint64_t>(bits_);
                std::uint64_t word = load_i32(p_.weight->data, row + bit / 32);
                if (bit % 32 + bits_ > 32) {
                    word |=
                        static_cast<std::uint64_t>(load_i32(p_.weight->data, row + bit / 32 + 1))
                        << 32;
                }
                return static_cast<int>((word >> (bit % 32)) & mask_) - offset_;
            }
            const auto word = load_i32(p_.weight->data, row + c / pack_);
            return static_cast<int>((word >> ((c % pack_) * bits_)) & mask_) - offset_;
        }
        case Layout::CompressedInt8:
            return static_cast<std::int8_t>(
                std::to_integer<std::uint8_t>(p_.weight->data[r * k_ + c]));
        case Layout::Awq: {
            static constexpr int kInverse[8] = {0, 4, 1, 5, 2, 6, 3, 7};
            const auto word                  = load_i32(p_.weight->data, c * (n_ / 8) + r / 8);
            return static_cast<int>((word >> (4 * kInverse[r % 8])) & 0xFU);
        }
        case Layout::Gptq: {
            const auto word = load_i32(p_.weight->data, (c / pack_) * n_ + r);
            return static_cast<int>((word >> ((c % pack_) * bits_)) & mask_);
        }
        }
        return 0;
    }

    int zero(std::uint64_t r, std::uint64_t g) const {
        switch (layout_) {
        case Layout::CompressedPacked:
        case Layout::CompressedInt8: {
            if (!p_.zero) { return 0; }
            const auto& z = *p_.zero;
            if (z.dtype_name == "I8") {
                return static_cast<std::int8_t>(
                    std::to_integer<std::uint8_t>(z.data[r * groups_ + g]));
            }
            // Packed along rows: [ceil(n / pack), groups].
            const auto word = load_i32(z.data, (r / pack_) * groups_ + g);
            return static_cast<int>((word >> ((r % pack_) * bits_)) & mask_) - offset_;
        }
        case Layout::Awq: {
            if (!p_.zero) { return 8; }
            static constexpr int kInverse[8] = {0, 4, 1, 5, 2, 6, 3, 7};
            const auto word                  = load_i32(p_.zero->data, g * (n_ / 8) + r / 8);
            return static_cast<int>((word >> (4 * kInverse[r % 8])) & 0xFU);
        }
        case Layout::Gptq: {
            if (!p_.zero) { return offset_; }
            const auto per_row = p_.zero->shape[1];
            const auto word    = load_i32(p_.zero->data, g * per_row + r / pack_);
            const int stored   = static_cast<int>((word >> ((r % pack_) * bits_)) & mask_);
            return gptq_v2_ ? stored : stored + 1;
        }
        }
        return 0;
    }

    float scale(std::uint64_t r, std::uint64_t g) const {
        if (layout_ == Layout::Awq || layout_ == Layout::Gptq) {
            return scale_value(*p_.scale, g * n_ + r);
        }
        return scale_value(*p_.scale, r * groups_ + g);
    }

    Layout layout_;
    Parts p_;
    std::uint64_t n_, k_;
    int bits_;
    std::uint64_t group_;
    bool gptq_v2_;
    std::uint64_t pack_       = 8;
    bool stream_              = false; // codes as one bit stream per row (3-bit)
    std::uint32_t mask_       = 15;
    int offset_               = 8;
    std::uint64_t groups_     = 1;
    bool act_order_           = false;
    std::optional<QType> kept_;
    std::uint64_t kept_group_ = 0;
    KOrder order_;
    std::string not_kept_; // why plan_kept() found no engine format, for format()
    std::uint64_t rounded_ = 0; // scales below binary16's normal range kept rounded
};

// ---------------------------------------------------------------------------------------------
// MXFP4: E2M1 codes with one E8M0 exponent per 32 elements.

class MxMatrix final : public QuantizedMatrix {
public:
    enum class Layout {
        CompressedTensors, // weight_packed [n, k/2], weight_scale [n, k/32]
        GptOss,            // <name>_blocks [..., k/32, 16], <name>_scales [..., k/32]; the
                           // leading dimensions (experts, rows) flatten into the n rows
    };

    MxMatrix(const Checkpoint& checkpoint, const TensorInfo& packed, const TensorInfo& scale,
             std::uint64_t n, std::uint64_t k, Layout layout = Layout::CompressedTensors)
        : packed_(packed), scale_(scale), n_(n), k_(k), layout_(layout) {
        if (layout == Layout::CompressedTensors) {
            if (k % 32 || packed.dtype_name != "U8" ||
                packed.shape != std::vector<std::uint64_t>{n, k / 2}) {
                bad(packed, "U8" + shape_text({n, k / 2}));
            }
            if (scale.dtype_name != "U8" || scale.shape != std::vector<std::uint64_t>{n, k / 32}) {
                bad(scale, "U8 E8M0" + shape_text({n, k / 32}));
            }
        } else {
            const auto& shape = packed.shape;
            std::uint64_t rows = 1;
            for (std::size_t d = 0; d + 2 < shape.size(); ++d) { rows *= shape[d]; }
            if (k % 32 || packed.dtype_name != "U8" || shape.size() < 3 || shape.back() != 16 ||
                shape[shape.size() - 2] != k / 32 || rows != n) {
                bad(packed, "U8 [..., " + std::to_string(k / 32) + ", 16] blocks of " +
                                std::to_string(n) + " rows");
            }
            const std::vector<std::uint64_t> expected(shape.begin(), shape.end() - 1);
            if (scale.dtype_name != "U8" || scale.shape != expected) {
                bad(scale, "U8 E8M0" + shape_text(expected));
            }
        }
        exponents_.resize(scale.bytes);
        checkpoint.read(scale, exponents_.data());
        int emax = -1;
        for (const auto e : exponents_) {
            if (e == 0xFF) { throw std::runtime_error(scale.name + ": E8M0 NaN exponent"); }
            emax = std::max(emax, static_cast<int>(e));
        }
        // One power-of-two divisor 2^t puts the largest exponent at E4M3's 2^8; blocks more than
        // 17 binades below it cannot be represented, unless every code of the block is zero.
        mappable_ = k % 64 == 0 && emax >= 8 && emax <= 261;
        if (mappable_) {
            emax_ = emax;
            std::vector<std::uint8_t> block(16);
            for (std::uint64_t i = 0; i < exponents_.size() && mappable_; ++i) {
                if (static_cast<int>(exponents_[i]) - emax_ + 8 >= -9) { continue; }
                checkpoint.read_range(packed, i * 16, 16, block.data());
                for (const auto byte : block) {
                    if ((byte & 0x77) != 0) { mappable_ = false; }
                }
            }
        }
        if (mappable_) { divisor_ = static_cast<std::uint32_t>(135 - emax_ + 127) << 23; }
    }

    std::string format() const override {
        return layout_ == Layout::GptOss ? "mxfp4 (gpt-oss blocks)" : "mxfp4 (compressed-tensors)";
    }

    std::optional<QType> keeps() const override {
        return mappable_ ? std::optional<QType>(QType::NVFP4) : std::nullopt;
    }

    bool keeps_losslessly() const override { return mappable_; }

    std::uint32_t weight_divisor() const override { return divisor_; }

    void dequantize(std::uint64_t first, std::uint64_t last, float* out,
                    ReadLog* log) const override {
        const auto* codes = reinterpret_cast<const std::uint8_t*>(packed_.data);
        for (auto r = first; r < last; ++r) {
            float* dst = out + (r - first) * k_;
            for (std::uint64_t j = 0; j < k_ / 2; ++j) {
                const int e     = exponents_[r * (k_ / 32) + (2 * j) / 32];
                const auto byte = codes[r * (k_ / 2) + j];
                dst[2 * j]      = std::ldexp(kE2m1Values[byte & 15U], e - 127);
                dst[2 * j + 1]  = std::ldexp(kE2m1Values[byte >> 4], e - 127);
            }
        }
        if (log) { add_rows(*log, packed_, first, last, k_ / 2); }
    }

    void append_encoded(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                        ReadLog* log) const override {
        const auto* codes = reinterpret_cast<const std::uint8_t*>(packed_.data);
        out.codes.insert(out.codes.end(), codes + first * (k_ / 2), codes + last * (k_ / 2));
        for (auto r = first; r < last; ++r) {
            for (std::uint64_t b = 0; b < k_ / 32; ++b) {
                const auto word =
                    e4m3_power_of_two(static_cast<int>(exponents_[r * (k_ / 32) + b]) - emax_ + 8);
                out.scales.push_back(word);
                out.scales.push_back(word);
            }
        }
        out.divisor = divisor_;
        if (log) { add_rows(*log, packed_, first, last, k_ / 2); }
    }

    bool keeps_mxfp4() const override { return true; }

    // The checkpoint's own words: codes as stored (element 2i in the low nibble, the engine's
    // order) and one E8M0 exponent per 32 values, row-major. No word is changed.
    void append_mxfp4(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                      ReadLog* log) const override {
        const auto* codes = reinterpret_cast<const std::uint8_t*>(packed_.data);
        out.codes.insert(out.codes.end(), codes + first * (k_ / 2), codes + last * (k_ / 2));
        out.scales.insert(out.scales.end(), exponents_.begin() + first * (k_ / 32),
                          exponents_.begin() + last * (k_ / 32));
        out.divisor = 0;
        if (log) { add_rows(*log, packed_, first, last, k_ / 2); }
    }

    void tensors(std::vector<const TensorInfo*>& out) const override {
        out.push_back(&packed_);
        out.push_back(&scale_);
    }

    void ranges(std::uint64_t first, std::uint64_t last, ReadLog& out) const override {
        add_rows(out, packed_, first, last, k_ / 2);
    }

private:
    const TensorInfo& packed_;
    const TensorInfo& scale_;
    std::uint64_t n_, k_;
    Layout layout_;
    std::vector<std::uint8_t> exponents_;
    bool mappable_         = false;
    int emax_              = 0;
    std::uint32_t divisor_ = 0;
};

// ---------------------------------------------------------------------------------------------
// A computed tensor (GGUF): its view decodes the rows and may keep NVFP4 words.

class ViewMatrix final : public QuantizedMatrix {
public:
    ViewMatrix(const TensorInfo& tensor, std::uint64_t n, std::uint64_t k)
        : view_(*tensor.view), k_(k) {
        if (tensor.shape != std::vector<std::uint64_t>{n, k}) {
            bad(tensor, "a matrix" + shape_text({n, k}));
        }
    }

    std::string format() const override { return view_.encoding(); }

    std::optional<QType> keeps() const override { return view_.keeps(); }

    bool keeps_losslessly() const override { return view_.keeps_losslessly(); }

    void dequantize(std::uint64_t first, std::uint64_t last, float* out,
                    ReadLog* log) const override {
        if (view_.dtype() == SourceDType::F32) {
            view_.values(first * k_, last * k_, out, log);
            return;
        }
        // BF16 / F16 words passed through (a reordered plain matrix).
        std::vector<std::uint16_t> words((last - first) * k_);
        view_.values(first * k_, last * k_, words.data(), log);
        for (std::size_t i = 0; i < words.size(); ++i) {
            out[i] = view_.dtype() == SourceDType::BF16 ? bf16_to_f32(words[i])
                                                        : f16_to_f32(words[i]);
        }
    }

    void append_encoded(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                        ReadLog* log) const override {
        view_.append_encoded(first, last, out, log);
    }

    std::uint32_t weight_divisor() const override { return view_.weight_divisor(); }

    std::optional<std::uint32_t> input_divisor() const override { return view_.input_divisor(); }

    bool keeps_mxfp4() const override { return view_.keeps_mxfp4(); }

    void append_mxfp4(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                      ReadLog* log) const override {
        view_.append_mxfp4(first, last, out, log);
    }

    void tensors(std::vector<const TensorInfo*>& out) const override { view_.tensors(out); }

    void ranges(std::uint64_t first, std::uint64_t last, ReadLog& out) const override {
        view_.ranges(first * k_, last * k_, out);
    }

private:
    const TensorView& view_;
    std::uint64_t k_;
};

} // namespace

std::uint8_t e4m3_power_of_two(int power) noexcept {
    if (power < -9 || power > 8) { return 0; }
    if (power >= -6) { return static_cast<std::uint8_t>((power + 7) << 3); }
    return static_cast<std::uint8_t>(1 << (power + 9)); // subnormal 2^-9, 2^-8, 2^-7
}

void QuantizedMatrix::append_encoded(std::uint64_t, std::uint64_t, EncodedRows&, ReadLog*) const {
    throw std::logic_error(format() + ": no engine format keeps these words");
}

void QuantizedMatrix::append_mxfp4(std::uint64_t, std::uint64_t, EncodedRows&, ReadLog*) const {
    throw std::logic_error(format() + ": not MXFP4");
}

std::uint32_t QuantizedMatrix::weight_divisor() const {
    throw std::logic_error(format() + ": no weight divisor");
}

std::optional<KeptEncoding> kept_encoding(const Checkpoint& checkpoint, const std::string& prefix,
                                          std::uint64_t n, std::uint64_t k) {
    if (const auto fields = nvfp4_layout(checkpoint, prefix)) {
        // W4A4 needs the checkpoint's calibrated activation divisor; a weight-only (W4A16)
        // NVFP4 matrix has none and keeps BF16 activations.
        const bool divisor = checkpoint.has(prefix + (*fields == Nvfp4Fields::ModelOpt
                                                          ? ".input_scale"
                                                          : ".input_global_scale"));
        return KeptEncoding{QType::NVFP4, divisor};
    }
    const auto decoder = probe_quantized(checkpoint, prefix, n, k);
    // GGUF Q8_0 and K-quant words are not a recipe's encoded site (kept_where_decoded).
    if (!decoder || !decoder->keeps() || kept_where_decoded(*decoder->keeps())) {
        return std::nullopt;
    }
    return KeptEncoding{*decoder->keeps(), decoder->input_divisor().has_value()};
}

bool kept_where_decoded(QType format) noexcept {
    return format == QType::Q8_G32_FP16 || is_ggml_block(format) || q3_group(format) != 0 ||
           u4_zp_group(format) != 0;
}

MatrixFormat encoded_matrix_format(QType format) noexcept {
    switch (format) {
    case QType::NVFP4:
        return MatrixFormat::Nvfp4;
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::FP8_E4M3FN_ROW_F32:
    case QType::FP8_E4M3FN_B128_F32:
        return MatrixFormat::Fp8Row;
    case QType::MXFP4:
        return MatrixFormat::Mxfp4;
    default:
        return MatrixFormat::Auto;
    }
}

ActivationPolicy kept_block_policy(QType format) noexcept {
    switch (format) {
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ4_XS:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::U4_G128_ZP_FP16:
    // u4 G32 / G64 read on the same int8 codec as G128 (W4a8U4ZpG): the same policy.
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
    case QType::Q8_G32_FP16:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    // IQ1_S / IQ1_M: grid -1, 0, 1 plus a delta of +-1/8 is 8g +- 1 in int8, exactly.
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    // MXFP4 (gpt-oss experts): E2M1 codes widen to int8 exactly, the UE8M0 block scale to a float;
    // the generic and MoE W4A8 take it (MXFP4 codec, test_moe gpt-oss-like W4A8 cases).
    case QType::MXFP4:
        return ActivationPolicy::AllowA8;
    default:
        return ActivationPolicy::A16Only;
    }
}

ActivationPolicy kept_policy(const KeptEncoding& site) noexcept {
    switch (encoded_matrix_format(site.format)) {
    case MatrixFormat::Nvfp4:
        return site.activation_divisor ? ActivationPolicy::AllowA4 : ActivationPolicy::A16Only;
    case MatrixFormat::Fp8Row:
        return ActivationPolicy::AllowA8;
    default:
        return kept_block_policy(site.format);
    }
}

ActivationPolicy head_policy(const KeptEncoding& site) noexcept {
    const auto format = encoded_matrix_format(site.format);
    return format == MatrixFormat::Nvfp4 || format == MatrixFormat::Fp8Row
               ? ActivationPolicy::A16Only
               : kept_policy(site);
}

void refuse_split_nvfp4_tiles(const Model& model, const Recipe& recipe,
                              const std::vector<std::size_t>& group) {
    const auto& first = recipe.selections[group.front()];
    const auto rows   = model.parameters[group.front()].shape[0];
    const bool ragged = std::any_of(group.begin(), group.end(), [&](std::size_t i) {
        return model.parameters[i].shape[0] % 128 != 0;
    });
    if (first.format != QType::NVFP4 || first.method != Method::ImportEncoded || !ragged) {
        return;
    }
    std::set<std::uint32_t> divisors;
    for (const auto i : group) {
        const auto& selection = recipe.selections[i];
        if (selection.format != QType::NVFP4 || !selection.source ||
            !selection.source->provides_divisors()) {
            return;
        }
        divisors.insert(selection.source->weight_divisor());
    }
    if (divisors.size() < 2) { return; }
    throw std::runtime_error(
        model.parameters[group.front()].name + " .. " + model.parameters[group.back()].name +
        ": NVFP4 matrices of " + std::to_string(rows) +
        " rows (not whole 128-row scale tiles) quantized against " +
        std::to_string(divisors.size()) +
        " different weight divisors cannot share one stacked plane, whose scale tiles would span "
        "two divisors. NInferno keeps them only as published and has no other lossless form of "
        "them: use a checkpoint whose stacked matrices (each expert bank) share one weight "
        "divisor, or the model's BF16 original.");
}

std::unique_ptr<QuantizedMatrix> probe_quantized(const Checkpoint& checkpoint,
                                                 const std::string& prefix, std::uint64_t n,
                                                 std::uint64_t k) {
    if (checkpoint.has(prefix + ".weight")) {
        const auto& weight = checkpoint.describe(prefix + ".weight");
        if (weight.view) {
            // A plain view (unquantized GGUF values) reads as plain values.
            if (!weight.view->dequantizes() && !weight.view->keeps()) { return nullptr; }
            return std::make_unique<ViewMatrix>(weight, n, k);
        }
    } else if (checkpoint.has(prefix) && checkpoint.describe(prefix).view) {
        // A mapped GGUF tensor named without .weight (DFlash2's selector codebooks): the same
        // view, kept or decoded as a .weight one would be, never read as plain values.
        const auto& tensor = checkpoint.describe(prefix);
        if (!tensor.view->dequantizes() && !tensor.view->keeps()) { return nullptr; }
        return std::make_unique<ViewMatrix>(tensor, n, k);
    }
    const auto config = quant_config(checkpoint.config());
    if (!config.refused.empty()) {
        throw std::runtime_error(prefix + ": " + config.refused +
                                 " has no native execution in NInferno (it would have to be "
                                 "dequantized, which NInferno does not do to a published model)");
    }
    const auto has    = [&](const char* suffix) { return checkpoint.has(prefix + suffix); };
    const auto get    = [&](const char* suffix) -> const TensorInfo& {
        return checkpoint.describe(prefix + suffix);
    };
    const auto optional = [&](const char* suffix) -> const TensorInfo* {
        return has(suffix) ? &get(suffix) : nullptr;
    };
    // gpt-oss: <name>_blocks and <name>_scales beside <name>_bias, with no .weight.
    if (has("_blocks") && has("_scales")) {
        return std::make_unique<MxMatrix>(checkpoint, get("_blocks"), get("_scales"), n, k,
                                          MxMatrix::Layout::GptOss);
    }
    if (has(".weight_packed")) {
        const auto& packed = get(".weight_packed");
        if (!has(".weight_scale")) { return nullptr; }
        const auto& scale = get(".weight_scale");
        if (packed.dtype_name == "U8" && scale.dtype_name == "U8") {
            return std::make_unique<MxMatrix>(checkpoint, packed, scale, n, k);
        }
        if (packed.dtype_name == "I32") {
            // The code width is the packing's: config groups may quantize matrices differently.
            auto bits = config.bits;
            for (const int b : {2, 3, 4, 8}) {
                if (packed.shape.size() == 2 && packed.shape[1] == (k * b + 31) / 32) { bits = b; }
            }
            return std::make_unique<IntMatrix>(IntMatrix::Layout::CompressedPacked,
                                               IntMatrix::Parts{&packed, &scale,
                                                                optional(".weight_zero_point"),
                                                                optional(".weight_g_idx")},
                                               n, k, bits, config.group, false);
        }
        return nullptr;
    }
    if (has(".weight")) {
        const auto& weight = get(".weight");
        if (weight.dtype_name == "F8_E4M3") {
            const TensorInfo* scale = optional(".weight_scale_inv");
            if (!scale) { scale = optional(".weight_scale"); }
            if (!scale) {
                throw std::runtime_error(weight.name + ": FP8 weight without a scale tensor");
            }
            return std::make_unique<Fp8Matrix>(checkpoint, weight, *scale, n, k, config);
        }
        if (weight.dtype_name == "I8" && has(".weight_scale")) {
            return std::make_unique<IntMatrix>(IntMatrix::Layout::CompressedInt8,
                                               IntMatrix::Parts{&weight, &get(".weight_scale"),
                                                                optional(".weight_zero_point"),
                                                                nullptr},
                                               n, k, 8, config.group, false);
        }
        return nullptr;
    }
    if (has(".qweight") && has(".scales")) {
        const auto& qweight = get(".qweight");
        bool awq            = config.method == "awq";
        if (config.method != "awq" && config.method != "gptq") {
            // No quantization_config: AWQ packs along N ([K, N/8]), GPTQ along K ([K/8, N]).
            awq = qweight.shape.size() == 2 && qweight.shape[0] == k;
        }
        // GPTQ packs codes along K ([K * bits / 32, N]): the width is the packing's when the
        // config does not give one that fits.
        auto bits = config.bits ? config.bits : 4;
        for (const int b : {2, 3, 4, 8}) {
            if (!awq && qweight.shape.size() == 2 && qweight.shape[0] != (k * bits + 31) / 32 &&
                qweight.shape[0] == (k * b + 31) / 32) {
                bits = b;
            }
        }
        return std::make_unique<IntMatrix>(
            awq ? IntMatrix::Layout::Awq : IntMatrix::Layout::Gptq,
            IntMatrix::Parts{&qweight, &get(".scales"), optional(".qzeros"), optional(".g_idx")}, n,
            k, bits, config.group, config.gptq_v2, config.sym);
    }
    return nullptr;
}

} // namespace ninferno::convert
