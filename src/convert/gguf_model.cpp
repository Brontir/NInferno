#include "convert/gguf_model.h"

#include "convert/formats.h"
#include "convert/ggml_quants.h"
#include "convert/numerics.h"
#include "convert/tensor_view.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>

namespace ninferno::convert {

void TensorView::append_encoded(std::uint64_t, std::uint64_t, EncodedRows&, ReadLog*) const {
    throw std::logic_error(encoding() + ": no engine format keeps these words");
}

std::uint32_t TensorView::weight_divisor() const {
    throw std::logic_error(encoding() + ": no weight divisor");
}

void TensorView::append_mxfp4(std::uint64_t, std::uint64_t, EncodedRows&, ReadLog*) const {
    throw std::logic_error(encoding() + ": not MXFP4");
}

namespace {

// ---------------------------------------------------------------------------------------------
// Views over GGUF tensors.

enum class Transform { None, Subtract1, NegExpLog };

struct Part {
    const TensorInfo* tensor = nullptr;
    std::uint64_t cols       = 0; // values per stored row (ne[0])
    std::uint64_t row_bytes  = 0;
    // NVFP4 global scales (<name>.scale) and activation scales (<name>.input_scale): one per
    // expert of a bank, one for a matrix; each covers rows_per_scale stored rows.
    std::vector<float> weight_scales;
    std::vector<float> input_scales;
    std::uint64_t rows_per_scale = 1;

    [[nodiscard]] float weight_scale(std::uint64_t row) const {
        return weight_scales.empty() ? 1.0F : weight_scales[row / rows_per_scale];
    }

    [[nodiscard]] std::optional<float> input_scale(std::uint64_t row) const {
        if (input_scales.empty()) { return std::nullopt; }
        return input_scales[row / rows_per_scale];
    }
};

struct Spec {
    std::vector<Part> parts;
    // Stored (part, row) of every logical row, in order.
    std::vector<std::pair<std::uint32_t, std::uint64_t>> rows;
    std::uint64_t cols = 0;
    // Column chunks: logical chunk j reads stored chunk chunk_of[j] (empty: identity).
    std::uint64_t chunk = 0;
    std::vector<std::uint64_t> chunk_of;
    Transform transform = Transform::None;
    std::string note;
};

bool float_type(std::uint32_t type) {
    return type == ggml::F32 || type == ggml::F16 || type == ggml::BF16;
}

SourceDType dtype_of(std::uint32_t type) {
    switch (type) {
    case ggml::F16:
        return SourceDType::F16;
    case ggml::BF16:
        return SourceDType::BF16;
    default:
        return SourceDType::F32;
    }
}

const char* dtype_spelling(SourceDType dtype) {
    switch (dtype) {
    case SourceDType::F16:
        return "F16";
    case SourceDType::BF16:
        return "BF16";
    default:
        return "F32";
    }
}

class GgufView final : public TensorView {
public:
    explicit GgufView(Spec spec) : s_(std::move(spec)) {
        const auto type = s_.parts.front().tensor->ggml_type;
        bool same       = true;
        for (const auto& part : s_.parts) {
            same       = same && part.tensor->ggml_type == type;
            quantized_ = quantized_ || !ggml::is_plain(part.tensor->ggml_type);
        }
        passthrough_ = same && float_type(type) && s_.transform == Transform::None;
        dtype_       = passthrough_ ? dtype_of(type) : SourceDType::F32;
        std::set<std::string> names;
        std::string joined;
        for (const auto& part : s_.parts) {
            const auto name = ggml::type_name(part.tensor->ggml_type);
            if (names.insert(name).second) { joined += (joined.empty() ? "" : "+") + name; }
        }
        encoding_ = "gguf " + joined + (s_.note.empty() ? "" : " (" + s_.note + ")");
        if (same && s_.transform == Transform::None &&
            (type == ggml::NVFP4 || type == ggml::MXFP4) && s_.cols % 64 == 0 &&
            (s_.chunk_of.empty() || s_.chunk % 64 == 0)) {
            // One engine plane carries one divisor: every row must share its scales.
            const auto& [p0, r0] = s_.rows.front();
            weight_scale_        = s_.parts[p0].weight_scale(r0);
            input_scale_         = s_.parts[p0].input_scale(r0);
            bool uniform         = true;
            for (const auto& [p, r] : s_.rows) {
                uniform = uniform && s_.parts[p].weight_scale(r) == weight_scale_ &&
                          s_.parts[p].input_scale(r) == input_scale_;
            }
            encodable_ = uniform;
        }
        // MXFP4 blocks (an E8M0 exponent, then 16 bytes holding values j and j + 16 in the low
        // and high nibble of byte j) are the engine's mxfp4 words once each block's nibbles are
        // put in natural order (value 2i low, 2i + 1 high): the same codes and exponents.
        mxfp4_ = same && s_.transform == Transform::None && type == ggml::MXFP4 &&
                 s_.cols % 32 == 0 && (s_.chunk_of.empty() || s_.chunk % 32 == 0);
        // Q8_0 blocks (a binary16 scale, then 32 int8 codes) are q8_g32_fp16's groups: the same
        // codes times the same scale, so the words move over unchanged.
        q8_ = same && s_.transform == Transform::None && type == ggml::Q8_0 && s_.cols % 32 == 0 &&
              (s_.chunk_of.empty() || s_.chunk % 32 == 0);
        // K-quant, i-quant and ternary superblocks (256 values along K) and the blocks of Q4_0,
        // Q4_1, Q5_0, Q5_1, IQ4_NL (32 values), Q1_0 (128) and Q2_0 (64) are the engine's ggml_*
        // words unchanged.
        std::optional<QType> kept;
        switch (type) {
        case ggml::Q4_K: kept = QType::GGML_Q4_K; break;
        case ggml::Q5_K: kept = QType::GGML_Q5_K; break;
        case ggml::Q6_K: kept = QType::GGML_Q6_K; break;
        case ggml::Q3_K: kept = QType::GGML_Q3_K; break;
        case ggml::Q2_K: kept = QType::GGML_Q2_K; break;
        case ggml::IQ4_XS: kept = QType::GGML_IQ4_XS; break;
        case ggml::IQ3_XXS: kept = QType::GGML_IQ3_XXS; break;
        case ggml::IQ3_S: kept = QType::GGML_IQ3_S; break;
        case ggml::Q4_0: kept = QType::GGML_Q4_0; break;
        case ggml::Q4_1: kept = QType::GGML_Q4_1; break;
        case ggml::Q5_0: kept = QType::GGML_Q5_0; break;
        case ggml::Q5_1: kept = QType::GGML_Q5_1; break;
        case ggml::IQ4_NL: kept = QType::GGML_IQ4_NL; break;
        case ggml::IQ2_XXS: kept = QType::GGML_IQ2_XXS; break;
        case ggml::IQ2_XS: kept = QType::GGML_IQ2_XS; break;
        case ggml::IQ2_S: kept = QType::GGML_IQ2_S; break;
        case ggml::IQ1_S: kept = QType::GGML_IQ1_S; break;
        case ggml::IQ1_M: kept = QType::GGML_IQ1_M; break;
        case ggml::TQ1_0: kept = QType::GGML_TQ1_0; break;
        case ggml::TQ2_0: kept = QType::GGML_TQ2_0; break;
        case ggml::Q1_0: kept = QType::GGML_Q1_0; break;
        case ggml::Q2_0: kept = QType::GGML_Q2_0; break;
        default: break;
        }
        if (kept) {
            const auto values = ggml_block_values(*kept);
            if (same && s_.transform == Transform::None && s_.cols % values == 0 &&
                (s_.chunk_of.empty() || s_.chunk % values == 0)) {
                kquant_ = kept;
            }
        }
    }

    [[nodiscard]] SourceDType dtype() const override { return dtype_; }

    [[nodiscard]] std::string encoding() const override { return encoding_; }

    [[nodiscard]] bool dequantizes() const override { return quantized_; }

    void tensors(std::vector<const TensorInfo*>& out) const override {
        for (const auto& part : s_.parts) {
            if (std::find(out.begin(), out.end(), part.tensor) == out.end()) {
                out.push_back(part.tensor);
            }
        }
    }

    void ranges(std::uint64_t begin, std::uint64_t end, ReadLog& out) const override {
        if (begin >= end) { return; }
        for (auto r = begin / s_.cols; r < (end - 1) / s_.cols + 1; ++r) {
            const auto& [p, row] = s_.rows[r];
            const auto& part     = s_.parts[p];
            out.add(*part.tensor, part.tensor->data + row * part.row_bytes, part.row_bytes);
        }
    }

    void values(std::uint64_t begin, std::uint64_t end, void* out, ReadLog* log) const override {
        if (begin >= end) { return; }
        const auto word = dtype_bytes(dtype_);
        auto* dst       = static_cast<std::byte*>(out);
        std::vector<float> row(passthrough_ ? 0 : s_.cols);
        std::vector<std::uint8_t> codes, scales;
        for (auto r = begin / s_.cols; r < (end - 1) / s_.cols + 1; ++r) {
            const auto low       = r == begin / s_.cols ? begin - r * s_.cols : 0;
            const auto high      = std::min<std::uint64_t>(s_.cols, end - r * s_.cols);
            const auto& [p, idx] = s_.rows[r];
            const auto& part     = s_.parts[p];
            const auto* src      = part.tensor->data + idx * part.row_bytes;
            if (log) { log->add(*part.tensor, src, part.row_bytes); }
            auto* target = dst + (r * s_.cols + low - begin) * word;
            if (passthrough_) {
                if (s_.chunk_of.empty()) {
                    std::memcpy(target, src + low * word, (high - low) * word);
                } else {
                    for (auto c = low; c < high; ++c) {
                        std::memcpy(target + (c - low) * word, src + stored_col(c) * word, word);
                    }
                }
                continue;
            }
            if (part.tensor->ggml_type == ggml::NVFP4) {
                // As the engine and a safetensors NVFP4 source read it: f32(e2m1 * e4m3) divided
                // by the divisor 1/scale (ggml multiplies by the scale at run time instead).
                const auto* blocks = reinterpret_cast<const std::uint8_t*>(src);
                codes.resize(s_.cols / 2);
                scales.resize(s_.cols / 16);
                for (std::uint64_t b = 0; b < s_.cols / 64; ++b) {
                    ggml::nvfp4_block_natural(blocks + 36 * b, codes.data() + 32 * b,
                                              scales.data() + 4 * b);
                }
                dequantize_nvfp4_row(codes.data(), scales.data(),
                                     f32_from_bits(divisor_word(part.weight_scale(idx))), s_.cols,
                                     row.data());
            } else {
                ggml::dequantize(part.tensor->ggml_type, src, s_.cols, row.data());
            }
            auto* values = reinterpret_cast<float*>(target);
            for (auto c = low; c < high; ++c) {
                values[c - low] = transform(row[s_.chunk_of.empty() ? c : stored_col(c)]);
            }
        }
    }

    [[nodiscard]] std::optional<QType> keeps() const override {
        if (q8_) { return QType::Q8_G32_FP16; }
        if (kquant_) { return kquant_; }
        if (!encodable_) { return std::nullopt; }
        if (s_.parts.front().tensor->ggml_type == ggml::NVFP4) { return QType::NVFP4; }
        probe_mx();
        return mx_mappable_ ? std::optional<QType>(QType::NVFP4) : std::nullopt;
    }

    // MXFP4 words move over to mxfp4 unchanged (append_mxfp4), whether or not NVFP4 holds them.
    [[nodiscard]] bool keeps_losslessly() const override { return keeps().has_value() || mxfp4_; }

    void append_encoded(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                        ReadLog* log) const override {
        if (!keeps()) { TensorView::append_encoded(first, last, out, log); }
        if (q8_) {
            append_q8(first, last, out, log);
            return;
        }
        if (kquant_) {
            append_kquant(first, last, out, log);
            return;
        }
        const bool nv = s_.parts.front().tensor->ggml_type == ggml::NVFP4;
        const auto k  = s_.cols;
        std::vector<std::uint8_t> codes(k / 2);
        std::vector<std::uint8_t> scales(k / 16);
        for (auto r = first; r < last; ++r) {
            const auto& [p, idx] = s_.rows[r];
            const auto& part     = s_.parts[p];
            const auto* src =
                reinterpret_cast<const std::uint8_t*>(part.tensor->data + idx * part.row_bytes);
            if (log) { log->add(*part.tensor, part.tensor->data + idx * part.row_bytes, part.row_bytes); }
            if (nv) {
                for (std::uint64_t b = 0; b < k / 64; ++b) {
                    ggml::nvfp4_block_natural(src + 36 * b, codes.data() + 32 * b,
                                              scales.data() + 4 * b);
                }
            } else {
                for (std::uint64_t b = 0; b < k / 32; ++b) {
                    const int e = ggml::mxfp4_block_natural(src + 17 * b, codes.data() + 16 * b);
                    const auto word = e4m3_power_of_two(e - mx_emax_ + 8);
                    scales[2 * b]     = word;
                    scales[2 * b + 1] = word;
                }
            }
            if (s_.chunk_of.empty()) {
                out.codes.insert(out.codes.end(), codes.begin(), codes.end());
                out.scales.insert(out.scales.end(), scales.begin(), scales.end());
            } else {
                for (std::uint64_t j = 0; j < k / s_.chunk; ++j) {
                    const auto from = s_.chunk_of[j];
                    out.codes.insert(out.codes.end(), codes.begin() + from * (s_.chunk / 2),
                                     codes.begin() + (from + 1) * (s_.chunk / 2));
                    out.scales.insert(out.scales.end(), scales.begin() + from * (s_.chunk / 16),
                                      scales.begin() + (from + 1) * (s_.chunk / 16));
                }
            }
        }
        out.divisor = weight_divisor();
    }

    [[nodiscard]] std::uint32_t weight_divisor() const override {
        if (!keeps() || q8_ || kquant_) { return TensorView::weight_divisor(); }
        if (s_.parts.front().tensor->ggml_type == ggml::MXFP4) { return mx_divisor_; }
        return divisor_word(weight_scale_);
    }

    [[nodiscard]] std::optional<std::uint32_t> input_divisor() const override {
        if (!keeps() || !input_scale_) { return std::nullopt; }
        return divisor_word(*input_scale_);
    }

    [[nodiscard]] bool keeps_mxfp4() const override { return mxfp4_; }

    void append_mxfp4(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                      ReadLog* log) const override {
        if (!mxfp4_) { TensorView::append_mxfp4(first, last, out, log); }
        const auto k = s_.cols;
        std::vector<std::uint8_t> codes(k / 2);
        std::vector<std::uint8_t> exponents(k / 32);
        for (auto r = first; r < last; ++r) {
            const auto& [p, idx] = s_.rows[r];
            const auto& part     = s_.parts[p];
            const auto* src =
                reinterpret_cast<const std::uint8_t*>(part.tensor->data + idx * part.row_bytes);
            if (log) {
                log->add(*part.tensor, part.tensor->data + idx * part.row_bytes, part.row_bytes);
            }
            for (std::uint64_t b = 0; b < k / 32; ++b) {
                exponents[b] = ggml::mxfp4_block_natural(src + 17 * b, codes.data() + 16 * b);
                if (exponents[b] == 0xFF) {
                    throw std::runtime_error(part.tensor->name + ": an MXFP4 block exponent is " +
                                             "NaN (E8M0 0xFF); mxfp4 cannot keep it");
                }
            }
            if (s_.chunk_of.empty()) {
                out.codes.insert(out.codes.end(), codes.begin(), codes.end());
                out.scales.insert(out.scales.end(), exponents.begin(), exponents.end());
                continue;
            }
            for (std::uint64_t j = 0; j < k / s_.chunk; ++j) {
                const auto from = s_.chunk_of[j];
                out.codes.insert(out.codes.end(), codes.begin() + from * (s_.chunk / 2),
                                 codes.begin() + (from + 1) * (s_.chunk / 2));
                out.scales.insert(out.scales.end(), exponents.begin() + from * (s_.chunk / 32),
                                  exponents.begin() + (from + 1) * (s_.chunk / 32));
            }
        }
        out.divisor = 0;
    }

private:
    // Q8_0 rows as q8_g32_fp16 words: int8 codes [rows, k] and binary16 scales [rows, k/32]
    // (little-endian bytes), in the logical column order. The engine format admits codes
    // -127..127 and nonnegative finite scales; ggml's quantizer writes nothing else, and any
    // other block stops the conversion instead of being changed.
    void append_q8(std::uint64_t first, std::uint64_t last, EncodedRows& out, ReadLog* log) const {
        const auto k = s_.cols;
        std::vector<std::uint8_t> codes(k);
        std::vector<std::uint8_t> scales(k / 16);
        for (auto r = first; r < last; ++r) {
            const auto& [p, idx] = s_.rows[r];
            const auto& part     = s_.parts[p];
            const auto* src =
                reinterpret_cast<const std::uint8_t*>(part.tensor->data + idx * part.row_bytes);
            if (log) {
                log->add(*part.tensor, part.tensor->data + idx * part.row_bytes, part.row_bytes);
            }
            for (std::uint64_t b = 0; b < k / 32; ++b) {
                const auto* block = src + 34 * b;
                auto word = static_cast<std::uint16_t>(block[0] | (block[1] << 8));
                if (word == 0x8000) { word = 0; } // -0 scales the same zeros
                if ((word & 0x8000) || (word & 0x7C00) == 0x7C00) {
                    throw std::runtime_error(part.tensor->name + ": a Q8_0 block scale is " +
                                             "negative or not finite; q8_g32_fp16 cannot keep it");
                }
                for (int j = 0; j < 32; ++j) {
                    if (block[2 + j] == 0x80) {
                        throw std::runtime_error(part.tensor->name + ": a Q8_0 block holds code " +
                                                 "-128, which q8_g32_fp16 cannot keep");
                    }
                }
                std::memcpy(codes.data() + 32 * b, block + 2, 32);
                scales[2 * b]     = static_cast<std::uint8_t>(word & 0xFF);
                scales[2 * b + 1] = static_cast<std::uint8_t>(word >> 8);
            }
            if (s_.chunk_of.empty()) {
                out.codes.insert(out.codes.end(), codes.begin(), codes.end());
                out.scales.insert(out.scales.end(), scales.begin(), scales.end());
                continue;
            }
            for (std::uint64_t j = 0; j < k / s_.chunk; ++j) {
                const auto from = s_.chunk_of[j];
                out.codes.insert(out.codes.end(), codes.begin() + from * s_.chunk,
                                 codes.begin() + (from + 1) * s_.chunk);
                out.scales.insert(out.scales.end(), scales.begin() + from * (s_.chunk / 16),
                                  scales.begin() + (from + 1) * (s_.chunk / 16));
            }
        }
        out.divisor = 0;
    }

    // Where each engine plane's bytes sit in a ggml K-quant block (ggml_planes_v1), and where
    // its binary16 d (and dmin) are.
    struct KquantBlock {
        std::uint64_t size;
        std::uint64_t code_begin, code_end; // codes
        std::uint64_t high_begin, high_end; // scale bytes (the high plane, carried as zeros)
        std::uint64_t d_begin, d_end;       // d (and dmin; IQ4_XS: d, scales_h): the scale plane
        std::uint64_t halves_end;           // d_begin .. halves_end: binary16 words, finite
    };

    [[nodiscard]] KquantBlock kquant_block() const {
        switch (*kquant_) {
        case QType::GGML_Q4_K:
            return {144, 0, 144, 0, 0, 0, 4, 4};
        case QType::GGML_Q5_K:
            return {176, 0, 176, 0, 0, 0, 4, 4};
        case QType::GGML_Q6_K: // ql[128] qh[64] scales[16] d
            return {210, 0, 192, 192, 208, 208, 210, 210};
        case QType::GGML_Q3_K: // hmask[32] qs[64] scales[12] d
            return {110, 0, 96, 96, 108, 108, 110, 110};
        case QType::GGML_IQ4_XS: // d scales_h[2] scales_l[4] qs[128]
            return {136, 8, 136, 4, 8, 0, 4, 2};
        case QType::GGML_IQ3_XXS: // d qs[64] scales_and_signs[32]
            return {98, 2, 66, 66, 98, 0, 2, 2};
        case QType::GGML_IQ3_S: // d qs[64] qh[8] signs[32] scales[4]
            return {110, 2, 106, 106, 110, 0, 2, 2};
        case QType::GGML_Q4_0: // d qs[16]
        case QType::GGML_IQ4_NL:
            return {18, 2, 18, 0, 0, 0, 2, 2};
        case QType::GGML_Q4_1: // d m qs[16]
            return {20, 4, 20, 0, 0, 0, 4, 4};
        case QType::GGML_Q5_0: // d qh[4] qs[16]
            return {22, 6, 22, 2, 6, 0, 2, 2};
        case QType::GGML_Q5_1: // d m qh[4] qs[16]
            return {24, 8, 24, 4, 8, 0, 4, 4};
        case QType::GGML_IQ2_XXS: // d qs[64]
            return {66, 2, 66, 0, 0, 0, 2, 2};
        case QType::GGML_IQ2_XS: // d qs[64] scales[8]
            return {74, 2, 66, 66, 74, 0, 2, 2};
        case QType::GGML_IQ2_S: // d qs[32] signs[32] qh[8] scales[8]
            return {82, 2, 74, 74, 82, 0, 2, 2};
        case QType::GGML_IQ1_S: // d qs[32] qh[16]
            return {50, 2, 34, 34, 50, 0, 2, 2};
        case QType::GGML_IQ1_M: // qs[32] qh[16] scales[8] (d in their top nibbles)
            return {56, 0, 32, 32, 48, 48, 56, 48};
        case QType::GGML_TQ1_0: // qs[48] qh[4] d
            return {54, 0, 52, 0, 0, 52, 54, 54};
        case QType::GGML_TQ2_0: // qs[64] d
            return {66, 0, 64, 0, 0, 64, 66, 66};
        case QType::GGML_Q1_0: // d qs[16]
            return {18, 2, 18, 0, 0, 0, 2, 2};
        case QType::GGML_Q2_0: // d qs[16]
            return {18, 2, 18, 0, 0, 0, 2, 2};
        default: // Q2_K: scales[16] qs[64] d dmin
            return {84, 16, 80, 0, 16, 80, 84, 84};
        }
    }

    // K-quant and i-quant rows as the engine's words, in the logical column order (whole
    // superblocks move). ggml_q4_k / ggml_q5_k: the blocks unchanged (codes). The ggml_planes_v1
    // formats: per block its code bytes (codes), scale bytes (zeros: the high plane) and d/dmin
    // (scales; IQ4_XS d and scales_h), each unchanged. A block whose d or dmin is not finite
    // stops the conversion instead of being changed.
    void append_kquant(std::uint64_t first, std::uint64_t last, EncodedRows& out,
                       ReadLog* log) const {
        const auto layout        = kquant_block();
        const bool whole         = kquant_ == QType::GGML_Q4_K || kquant_ == QType::GGML_Q5_K;
        const std::uint64_t size = layout.size;
        const auto values        = ggml_block_values(*kquant_);
        const auto blocks        = s_.cols / values;
        const auto finite        = [&](const std::uint8_t* p, const TensorInfo& t) {
            if (((p[0] | (p[1] << 8)) & 0x7C00) == 0x7C00) {
                throw std::runtime_error(t.name + ": a " + ggml::type_name(t.ggml_type) +
                                         " block scale is not finite; it cannot be kept");
            }
        };
        for (auto r = first; r < last; ++r) {
            const auto& [p, idx] = s_.rows[r];
            const auto& part     = s_.parts[p];
            const auto* src =
                reinterpret_cast<const std::uint8_t*>(part.tensor->data + idx * part.row_bytes);
            if (log) {
                log->add(*part.tensor, part.tensor->data + idx * part.row_bytes, part.row_bytes);
            }
            for (std::uint64_t b = 0; b < blocks; ++b) {
                const auto stored = s_.chunk_of.empty()
                                        ? b
                                        : s_.chunk_of[b / (s_.chunk / values)] *
                                                  (s_.chunk / values) +
                                              b % (s_.chunk / values);
                const auto* block = src + stored * size;
                for (auto at = layout.d_begin; at < layout.halves_end; at += 2) {
                    finite(block + at, *part.tensor);
                }
                if (kquant_ == QType::GGML_IQ1_M) {
                    // d's four nibbles sit on top of the four 16-bit scale words.
                    const auto* sc = block + layout.d_begin;
                    const std::uint8_t d[2] = {
                        static_cast<std::uint8_t>((sc[1] >> 4) | (sc[3] & 0xF0)),
                        static_cast<std::uint8_t>((sc[5] >> 4) | (sc[7] & 0xF0))};
                    finite(d, *part.tensor);
                }
                if (whole) {
                    out.codes.insert(out.codes.end(), block, block + size);
                    continue;
                }
                out.codes.insert(out.codes.end(), block + layout.code_begin,
                                 block + layout.code_end);
                out.zeros.insert(out.zeros.end(), block + layout.high_begin,
                                 block + layout.high_end);
                out.scales.insert(out.scales.end(), block + layout.d_begin,
                                  block + layout.d_end);
            }
        }
        out.divisor = 0;
    }

    // A scale multiplier as the artifact's divisor word: f32(1 / scale), as the Model Optimizer
    // reader takes weight_scale_2 and input_scale.
    std::uint32_t divisor_word(float scale) const {
        const auto word = f32_bits(1.0F / scale);
        if (!valid_positive_f32_word(word)) {
            throw std::runtime_error(encoding_ + ": NVFP4 scale is not finite and positive");
        }
        return word;
    }

    std::uint64_t stored_col(std::uint64_t c) const {
        return s_.chunk_of[c / s_.chunk] * s_.chunk + c % s_.chunk;
    }

    float transform(float value) const {
        switch (s_.transform) {
        case Transform::None:
            return value;
        case Transform::Subtract1:
            return value - 1.0F;
        case Transform::NegExpLog:
            return static_cast<float>(std::log(-static_cast<double>(value)));
        }
        return value;
    }

    // MXFP4 maps onto NVFP4 when one power-of-two divisor puts every block exponent within
    // E4M3's 2^-9..2^8 (blocks below it must hold only zeros), as formats.h's MXFP4 decoder.
    void probe_mx() const {
        std::call_once(mx_once_, [this] {
            const auto k = s_.cols;
            int emax     = -1;
            for (const auto& [p, idx] : s_.rows) {
                const auto* src = reinterpret_cast<const std::uint8_t*>(
                    s_.parts[p].tensor->data + idx * s_.parts[p].row_bytes);
                for (std::uint64_t b = 0; b < k / 32; ++b) {
                    emax = std::max(emax, static_cast<int>(src[17 * b]));
                }
            }
            bool ok = emax >= 8 && emax <= 261 && emax != 0xFF;
            for (std::size_t r = 0; ok && r < s_.rows.size(); ++r) {
                const auto& [p, idx] = s_.rows[r];
                const auto* src      = reinterpret_cast<const std::uint8_t*>(
                    s_.parts[p].tensor->data + idx * s_.parts[p].row_bytes);
                for (std::uint64_t b = 0; ok && b < k / 32; ++b) {
                    if (src[17 * b] == 0xFF) { ok = false; }
                    if (static_cast<int>(src[17 * b]) - emax + 8 >= -9) { continue; }
                    for (int j = 1; j <= 16; ++j) {
                        if ((src[17 * b + j] & 0x77) != 0) { ok = false; }
                    }
                }
            }
            mx_mappable_ = ok;
            if (ok) {
                mx_emax_    = emax;
                mx_divisor_ = static_cast<std::uint32_t>(135 - emax + 127) << 23;
            }
        });
    }

    Spec s_;
    bool passthrough_ = false;
    bool quantized_   = false;
    bool encodable_   = false;
    bool q8_          = false;
    bool mxfp4_       = false;
    std::optional<QType> kquant_;
    float weight_scale_ = 1.0F;
    std::optional<float> input_scale_;
    SourceDType dtype_ = SourceDType::F32;
    std::string encoding_;
    mutable std::once_flag mx_once_;
    mutable bool mx_mappable_         = false;
    mutable int mx_emax_              = 0;
    mutable std::uint32_t mx_divisor_ = 0;
};

// A constant F32 vector (Gemma 4's per-expert scale once folded into the NVFP4 experts).
class ConstantView final : public TensorView {
public:
    ConstantView(float value, std::string note) : value_(value), note_(std::move(note)) {}

    [[nodiscard]] SourceDType dtype() const override { return SourceDType::F32; }

    void values(std::uint64_t begin, std::uint64_t end, void* out, ReadLog*) const override {
        std::fill_n(static_cast<float*>(out), end - begin, value_);
    }

    [[nodiscard]] std::string encoding() const override { return "gguf constant (" + note_ + ")"; }

    [[nodiscard]] bool dequantizes() const override { return false; }

    void tensors(std::vector<const TensorInfo*>&) const override {}

    void ranges(std::uint64_t, std::uint64_t, ReadLog&) const override {}

private:
    float value_;
    std::string note_;
};

// ---------------------------------------------------------------------------------------------
// Mapping helpers.

using Json = nlohmann::json;

class Mapper {
public:
    Mapper(const GgufFile& file, const StoredLookup& stored, const StoredRead& read,
           std::string label)
        : file_(file), stored_(stored), read_(read), label_(std::move(label)) {
        const auto* arch = file.find("general.architecture");
        if (!arch || !arch->is_string()) {
            throw std::runtime_error(label_ + ": GGUF without general.architecture");
        }
        arch_ = arch->get<std::string>();
    }

    [[nodiscard]] const std::string& arch() const noexcept { return arch_; }

    [[nodiscard]] const GgufFile& file() const noexcept { return file_; }

    [[nodiscard]] const TensorInfo* find(const std::string& name) const { return stored_(name); }

    [[nodiscard]] const TensorInfo& need(const std::string& name) const {
        const auto* tensor = stored_(name);
        if (!tensor) {
            throw std::runtime_error(label_ + ": " + arch_ + " GGUF lacks tensor '" + name + "'");
        }
        return *tensor;
    }

    // Metadata of this architecture ("{arch}.key").
    [[nodiscard]] const Json* meta(const std::string& key) const {
        return file_.find(arch_ + "." + key);
    }

    [[nodiscard]] std::uint64_t count(const std::string& key) const {
        const auto* value = meta(key);
        if (!value) { throw std::runtime_error(label_ + ": missing " + arch_ + "." + key); }
        return as_count(*value, key);
    }

    [[nodiscard]] std::uint64_t count_or(const std::string& key, std::uint64_t fallback) const {
        const auto* value = meta(key);
        return value ? as_count(*value, key) : fallback;
    }

    // A per-layer array or one value for every layer.
    [[nodiscard]] std::uint64_t per_layer(const std::string& key, std::uint64_t layer) const {
        const auto* value = meta(key);
        if (!value) { throw std::runtime_error(label_ + ": missing " + arch_ + "." + key); }
        if (value->is_array()) {
            if (layer >= value->size()) {
                throw std::runtime_error(label_ + ": " + arch_ + "." + key + " is too short");
            }
            return as_count((*value)[layer], key);
        }
        return as_count(*value, key);
    }

    [[nodiscard]] double real(const std::string& key) const {
        const auto* value = meta(key);
        if (!value || !value->is_number()) {
            throw std::runtime_error(label_ + ": missing " + arch_ + "." + key);
        }
        return value->get<double>();
    }

    [[nodiscard]] double real_or(const std::string& key, double fallback) const {
        const auto* value = meta(key);
        return value && value->is_number() ? value->get<double>() : fallback;
    }

    // A stored tensor's rows, with its NVFP4 scales (<stem>.scale / <stem>.input_scale: one value
    // for a matrix, one per expert for a bank) when it is NVFP4.
    [[nodiscard]] Part part(const TensorInfo& tensor) const {
        Part out;
        out.tensor    = &tensor;
        out.cols      = tensor.shape.back();
        out.row_bytes = ggml::bytes_for(tensor.ggml_type, out.cols);
        if (tensor.ggml_type != ggml::NVFP4 || !tensor.name.ends_with(".weight")) { return out; }
        const auto stem = tensor.name.substr(0, tensor.name.size() - 7);
        const auto rows = rows_of(tensor);
        std::optional<std::uint64_t> grouping;
        const auto load = [&](const std::string& name, std::vector<float>& into) {
            const auto* scale = find(name);
            if (!scale) { return; }
            const auto count = scale->elements();
            if (scale->ggml_type != ggml::F32 || !count || rows % count ||
                (grouping && *grouping != rows / count)) {
                throw std::runtime_error(label_ + ": " + name + ": expected F32 scales, one per " +
                                         "matrix or expert of " + tensor.name);
            }
            grouping = rows / count;
            into.resize(count);
            read_(*scale, 0, 4 * count, into.data());
        };
        load(stem + ".scale", out.weight_scales);
        load(stem + ".input_scale", out.input_scales);
        out.rows_per_scale = grouping.value_or(rows);
        return out;
    }

    [[nodiscard]] static std::uint64_t rows_of(const TensorInfo& tensor) {
        return tensor.elements() / tensor.shape.back();
    }

    // Rows [first, first + count) of a stored tensor.
    [[nodiscard]] Spec slice(const TensorInfo& tensor, std::uint64_t first,
                             std::uint64_t count) const {
        Spec spec;
        spec.parts.push_back(part(tensor));
        spec.cols = spec.parts[0].cols;
        if (first + count > rows_of(tensor)) {
            throw std::runtime_error(label_ + ": " + tensor.name + " has too few rows");
        }
        for (std::uint64_t r = 0; r < count; ++r) { spec.rows.emplace_back(0, first + r); }
        return spec;
    }

    [[nodiscard]] Spec whole(const TensorInfo& tensor) const {
        return slice(tensor, 0, rows_of(tensor));
    }


    [[nodiscard]] float element(const TensorInfo& tensor, std::uint64_t index) const {
        if (tensor.ggml_type != ggml::F32 || index >= tensor.elements()) {
            throw std::runtime_error(label_ + ": " + tensor.name + ": expected an F32 scale with " +
                                     std::to_string(index + 1) + "+ values");
        }
        float value = 0.0F;
        read_(tensor, 4 * index, 4, &value);
        return value;
    }

    void add(const std::string& name, Shape shape, Spec spec) {
        std::uint64_t elements = 1;
        for (const auto dim : shape) { elements *= dim; }
        if (spec.rows.size() * spec.cols != elements) {
            throw std::runtime_error(label_ + ": " + name + " maps to " +
                                     std::to_string(spec.rows.size() * spec.cols) +
                                     " stored values, expected " + std::to_string(elements));
        }
        for (const auto& part : spec.parts) {
            if (part.cols != spec.cols) {
                throw std::runtime_error(label_ + ": " + name + ": stored rows differ in length");
            }
        }
        add_view(name, std::move(shape), std::make_shared<GgufView>(std::move(spec)));
    }

    void add_view(const std::string& name, Shape shape, std::shared_ptr<const TensorView> view) {
        TensorInfo info;
        info.name        = name;
        info.shape       = std::move(shape);
        info.dtype       = view->dtype();
        info.dtype_name  = dtype_spelling(info.dtype);
        info.known_dtype = true;
        info.bytes       = info.elements() * dtype_bytes(info.dtype);
        info.view        = std::move(view);
        out.views.push_back(std::move(info));
    }

    void note(const std::string& kind) { ++notes_[kind]; }

    GgufMapping finish() {
        for (const auto& [kind, n] : notes_) {
            out.notes.push_back(arch_ + ": " + std::to_string(n) + " x " + kind);
        }
        return std::move(out);
    }

    GgufMapping out;

private:
    std::uint64_t as_count(const Json& value, const std::string& key) const {
        if (!value.is_number_integer() ||
            (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)) {
            throw std::runtime_error(label_ + ": " + arch_ + "." + key +
                                     " must be a nonnegative integer");
        }
        return value.get<std::uint64_t>();
    }

    const GgufFile& file_;
    const StoredLookup& stored_;
    const StoredRead& read_;
    std::string label_;
    std::string arch_;
    std::map<std::string, std::uint64_t> notes_;
};

// A GGUF flag array element: BOOL, or an integer 0/1 some writers emit instead.
bool flag(const Json& value) {
    if (value.is_boolean()) { return value.get<bool>(); }
    if (value.is_number_integer()) { return value.get<std::int64_t>() != 0; }
    throw std::runtime_error("expected a boolean flag, got " + value.dump());
}

Spec subtract1(Spec spec) {
    spec.transform = Transform::Subtract1;
    spec.note      = "1+w stored";
    return spec;
}

// llama.cpp's _reorder_v_heads (grouped by key head -> tiled) inverted: logical (grouped) index
// (g * per + j) * dim + d reads stored (tiled) index (j * groups + g) * dim + d.
std::vector<std::uint64_t> grouped_from_tiled(std::uint64_t groups, std::uint64_t per) {
    std::vector<std::uint64_t> out(groups * per);
    for (std::uint64_t g = 0; g < groups; ++g) {
        for (std::uint64_t j = 0; j < per; ++j) { out[g * per + j] = j * groups + g; }
    }
    return out;
}

// Reorder rows [first, first + groups*per*dim) of a spec by value heads of `dim` rows.
void reorder_rows(Spec& spec, std::uint64_t first, std::uint64_t groups, std::uint64_t per,
                  std::uint64_t dim) {
    if (per == 1) { return; }
    const auto order = grouped_from_tiled(groups, per);
    const auto rows  = spec.rows;
    for (std::uint64_t head = 0; head < groups * per; ++head) {
        for (std::uint64_t d = 0; d < dim; ++d) {
            spec.rows[first + head * dim + d] = rows[first + order[head] * dim + d];
        }
    }
}

void reorder_columns(Spec& spec, std::uint64_t groups, std::uint64_t per, std::uint64_t dim) {
    if (per == 1) { return; }
    spec.chunk    = dim;
    spec.chunk_of = grouped_from_tiled(groups, per);
}

// The trunk depth of a GGUF that may carry its MTP (NextN) blocks after the trunk: block_count
// counts them too, nextn_predict_layers says how many. The first of them must carry the NextN
// fusion (blk.<trunk>.nextn.eh_proj.weight), so a stray count cannot cut trunk layers off.
std::uint64_t trunk_blocks(Mapper& m, const std::string& arch) {
    const auto blocks = m.count("block_count");
    const auto nextn  = m.count_or("nextn_predict_layers", 0);
    if (nextn >= blocks) {
        throw std::runtime_error(arch + ": nextn_predict_layers leaves no trunk block");
    }
    const auto first = "blk." + std::to_string(blocks - nextn) + ".nextn.eh_proj.weight";
    if (nextn != 0 && m.find(first) == nullptr) {
        throw std::runtime_error(arch + ": nextn_predict_layers " + std::to_string(nextn) +
                                 ", but " + first + " is absent: the MTP block is not there");
    }
    return blocks - nextn;
}

// ---------------------------------------------------------------------------------------------
// qwen35 / qwen35moe

void map_qwen35(Mapper& m) {
    const bool moe       = m.arch() == "qwen35moe";
    const auto& embd     = m.need("token_embd.weight");
    const auto vocab     = embd.shape[0];
    const auto hidden    = m.count("embedding_length");
    const auto mtp       = m.count_or("nextn_predict_layers", 0);
    const auto layers    = trunk_blocks(m, "qwen35");
    const auto head_dim  = m.count("attention.key_length");
    const auto nk        = m.count("ssm.group_count");
    const auto dk        = m.count("ssm.state_size");
    const auto nv        = m.count("ssm.time_step_rank");
    const auto inner     = m.count("ssm.inner_size");
    if (!nv || !nk || inner % nv || nv % nk) {
        throw std::runtime_error("qwen35: invalid linear-attention geometry");
    }
    const auto dv  = inner / nv;
    const auto per = nv / nk;

    std::vector<bool> recurrent(layers);
    if (const auto* flags = m.meta("attention.recurrent_layers"); flags && flags->is_array()) {
        if (flags->size() < layers) {
            throw std::runtime_error("qwen35: attention.recurrent_layers is too short");
        }
        for (std::uint64_t i = 0; i < layers; ++i) { recurrent[i] = flag((*flags)[i]); }
    } else {
        const auto interval = m.count_or("full_attention_interval", 4);
        for (std::uint64_t i = 0; i < layers; ++i) { recurrent[i] = (i + 1) % interval != 0; }
    }
    // Attention head counts may be per-layer arrays (zero on recurrent layers): read them at the
    // first full-attention layer (the MTP block when every text layer is recurrent).
    std::uint64_t attention_layer = layers;
    for (std::uint64_t i = 0; i < layers; ++i) {
        if (!recurrent[i]) {
            attention_layer = i;
            break;
        }
    }
    if (attention_layer == layers && !mtp) { attention_layer = 0; }
    const auto heads    = m.per_layer("attention.head_count", attention_layer);
    const auto kv_heads = m.per_layer("attention.head_count_kv", attention_layer);

    OJson config            = OJson::object();
    config["architectures"] = OJson::array({moe ? "Qwen3_5MoeForCausalLM" : "Qwen3_5ForCausalLM"});
    config["model_type"]    = moe ? "qwen3_5_moe_text" : "qwen3_5_text";
    config["hidden_size"]   = hidden;
    config["vocab_size"]    = vocab;
    config["num_hidden_layers"]       = layers;
    config["max_position_embeddings"] = m.count("context_length");
    config["tie_word_embeddings"]     = m.find("output.weight") == nullptr;
    config["rms_norm_eps"]            = m.real("attention.layer_norm_rms_epsilon");
    OJson types                       = OJson::array();
    for (std::uint64_t i = 0; i < layers; ++i) {
        types.push_back(recurrent[i] ? "linear_attention" : "full_attention");
    }
    config["layer_types"]         = types;
    config["num_attention_heads"] = heads;
    config["num_key_value_heads"] = kv_heads;
    config["head_dim"]            = head_dim;
    const auto rotary             = m.count("rope.dimension_count");
    OJson sections                = OJson::array();
    if (const auto* given = m.meta("rope.dimension_sections"); given && given->is_array()) {
        for (std::size_t i = 0; i < 3 && i < given->size(); ++i) {
            sections.push_back((*given)[i].get<std::int64_t>());
        }
    }
    if (sections.size() != 3) { sections = OJson::array({11, 11, 10}); }
    config["rope_parameters"] = OJson{{"rope_type", "default"},
                                      {"rope_theta", m.real("rope.freq_base")},
                                      {"partial_rotary_factor",
                                       static_cast<double>(rotary) / static_cast<double>(head_dim)},
                                      {"mrope_section", sections},
                                      {"mrope_interleaved", true}};
    config["linear_num_key_heads"]   = nk;
    config["linear_key_head_dim"]    = dk;
    config["linear_num_value_heads"] = nv;
    config["linear_value_head_dim"]  = dv;
    config["linear_conv_kernel_dim"] = m.count("ssm.conv_kernel");
    if (moe) {
        config["num_experts"]                     = m.count("expert_count");
        config["num_experts_per_tok"]             = m.count("expert_used_count");
        config["moe_intermediate_size"]           = m.count("expert_feed_forward_length");
        config["shared_expert_intermediate_size"] = m.count("expert_shared_feed_forward_length");
    } else {
        config["intermediate_size"] = m.per_layer("feed_forward_length", 0);
    }
    config["hidden_act"]      = "silu";
    config["attention_bias"]  = false;
    config["attn_output_gate"] = true;
    config["mamba_ssm_dtype"] = "float32";
    if (mtp) { config["mtp_num_hidden_layers"] = mtp; }
    // GDN out_proj columns come in llama.cpp's tiled value-head order. Moving them back is exact
    // for float, Q8_0 and 32-value block chunks, but a K-quant or i-quant superblock (256 values)
    // holds value heads that are not neighbours in grouped order under one d (and dmin): no
    // same-format repack is lossless.
    // Such a model keeps every out_proj as stored and says so in its config; the engine hands
    // out_proj its input in that order (value heads regrouped, bytes exact).
    bool tiled_output = false;
    for (std::uint64_t i = 0; i < layers && per > 1 && dv % 256 != 0; ++i) {
        const auto* out = recurrent[i] ? m.find("blk." + std::to_string(i) + ".ssm_out.weight") : nullptr;
        if (out == nullptr) { continue; }
        const auto t = out->ggml_type;
        tiled_output = tiled_output || t == ggml::Q2_K || t == ggml::Q3_K || t == ggml::Q4_K ||
                       t == ggml::Q5_K || t == ggml::Q6_K || t == ggml::IQ4_XS ||
                       t == ggml::IQ3_XXS || t == ggml::IQ3_S || t == ggml::IQ2_XXS ||
                       t == ggml::IQ2_XS || t == ggml::IQ2_S || t == ggml::IQ1_S ||
                       t == ggml::IQ1_M || t == ggml::TQ1_0 || t == ggml::TQ2_0;
    }
    if (tiled_output) { config["linear_output_head_order"] = "tiled"; }
    m.out.config = std::move(config);

    m.add("model.embed_tokens.weight", {vocab, hidden}, m.whole(embd));
    if (const auto* output = m.find("output.weight")) {
        m.add("lm_head.weight", {vocab, hidden}, m.whole(*output));
    }
    m.add("model.norm.weight", {hidden}, subtract1(m.whole(m.need("output_norm.weight"))));
    m.note("RMSNorm weight stored as 1+w, read as stored-1");

    const auto kg = nk * dk;
    const auto vg = nv * dv;
    const auto block = [&](const std::string& s, const std::string& p, bool linear) {
        m.add(p + "input_layernorm.weight", {hidden},
              subtract1(m.whole(m.need(s + "attn_norm.weight"))));
        m.add(p + "post_attention_layernorm.weight", {hidden},
              subtract1(m.whole(m.need(s + "post_attention_norm.weight"))));
        m.note("RMSNorm weight stored as 1+w, read as stored-1");
        m.note("RMSNorm weight stored as 1+w, read as stored-1");
        if (!linear) {
            const auto q = heads * head_dim;
            m.add(p + "self_attn.q_proj.weight", {2 * q, hidden}, m.whole(m.need(s + "attn_q.weight")));
            m.add(p + "self_attn.k_proj.weight", {kv_heads * head_dim, hidden},
                  m.whole(m.need(s + "attn_k.weight")));
            m.add(p + "self_attn.v_proj.weight", {kv_heads * head_dim, hidden},
                  m.whole(m.need(s + "attn_v.weight")));
            m.add(p + "self_attn.o_proj.weight", {hidden, q},
                  m.whole(m.need(s + "attn_output.weight")));
            m.add(p + "self_attn.q_norm.weight", {head_dim},
                  subtract1(m.whole(m.need(s + "attn_q_norm.weight"))));
            m.add(p + "self_attn.k_norm.weight", {head_dim},
                  subtract1(m.whole(m.need(s + "attn_k_norm.weight"))));
            m.note("RMSNorm weight stored as 1+w, read as stored-1");
            m.note("RMSNorm weight stored as 1+w, read as stored-1");
        } else {
            const auto l = p + "linear_attn.";
            auto qkv     = m.whole(m.need(s + "attn_qkv.weight"));
            reorder_rows(qkv, 2 * kg, nk, per, dv);
            m.add(l + "in_proj_qkv.weight", {2 * kg + vg, hidden}, std::move(qkv));
            auto z = m.whole(m.need(s + "attn_gate.weight"));
            reorder_rows(z, 0, nk, per, dv);
            m.add(l + "in_proj_z.weight", {vg, hidden}, std::move(z));
            for (const auto& [from, to] : {std::pair{"ssm_alpha.weight", "in_proj_a.weight"},
                                           std::pair{"ssm_beta.weight", "in_proj_b.weight"}}) {
                auto spec = m.whole(m.need(s + from));
                reorder_rows(spec, 0, nk, per, 1);
                m.add(l + to, {nv, hidden}, std::move(spec));
            }
            auto out = m.whole(m.need(s + "ssm_out.weight"));
            if (tiled_output) {
                m.note("GDN out_proj kept in tiled value-head order (K-quant superblocks span "
                       "value heads); its input is regrouped at run time");
            } else {
                reorder_columns(out, nk, per, dv);
            }
            m.add(l + "out_proj.weight", {hidden, vg}, std::move(out));
            m.add(l + "norm.weight", {dv}, m.whole(m.need(s + "ssm_norm.weight")));
            const auto& conv_tensor = m.need(s + "ssm_conv1d.weight");
            auto conv               = m.whole(conv_tensor);
            reorder_rows(conv, 2 * kg, nk, per, dv);
            m.add(l + "conv1d.weight", {2 * kg + vg, 1, conv_tensor.shape.back()}, std::move(conv));
            auto a = m.whole(m.need(s + "ssm_a"));
            reorder_columns(a, nk, per, 1);
            a.transform = Transform::NegExpLog;
            a.note      = "-exp(A_log) stored";
            m.add(l + "A_log", {nv}, std::move(a));
            m.note("A_log stored as -exp(A_log), read as log(-stored)");
            auto dt = m.whole(m.need(s + "ssm_dt.bias"));
            reorder_columns(dt, nk, per, 1);
            m.add(l + "dt_bias", {nv}, std::move(dt));
            if (per > 1) { m.note("GDN layer with value heads re-tiled back to grouped order"); }
        }
        if (!moe) {
            const auto ff = m.per_layer("feed_forward_length", 0);
            m.add(p + "mlp.gate_proj.weight", {ff, hidden}, m.whole(m.need(s + "ffn_gate.weight")));
            m.add(p + "mlp.up_proj.weight", {ff, hidden}, m.whole(m.need(s + "ffn_up.weight")));
            m.add(p + "mlp.down_proj.weight", {hidden, ff}, m.whole(m.need(s + "ffn_down.weight")));
            return;
        }
        const auto experts = m.count("expert_count");
        const auto ir      = m.count("expert_feed_forward_length");
        const auto shared  = m.count("expert_shared_feed_forward_length");
        m.add(p + "mlp.gate.weight", {experts, hidden}, m.whole(m.need(s + "ffn_gate_inp.weight")));
        m.add(p + "mlp.shared_expert_gate.weight", {1, hidden},
              m.whole(m.need(s + "ffn_gate_inp_shexp.weight")));
        for (const auto& [from, to, rows, cols] :
             {std::tuple{"ffn_gate_shexp.weight", "gate_proj", shared, hidden},
              std::tuple{"ffn_up_shexp.weight", "up_proj", shared, hidden},
              std::tuple{"ffn_down_shexp.weight", "down_proj", hidden, shared}}) {
            auto spec = m.whole(m.need(s + from));
            m.add(p + "mlp.shared_expert." + to + ".weight", {rows, cols}, std::move(spec));
        }
        // Banks: one [E, 2*ir, h] gate_up and one [E, h, ir] down, and per-expert matrices.
        const auto* fused = m.find(s + "ffn_gate_up_exps.weight");
        const auto* gate  = fused ? nullptr : &m.need(s + "ffn_gate_exps.weight");
        const auto* up    = fused ? nullptr : &m.need(s + "ffn_up_exps.weight");
        const auto& down  = m.need(s + "ffn_down_exps.weight");
        Spec bank;
        if (fused) {
            bank = m.whole(*fused);
        } else {
            bank.parts = {m.part(*gate), m.part(*up)};
            bank.cols  = hidden;
            for (std::uint64_t e = 0; e < experts; ++e) {
                for (std::uint32_t half = 0; half < 2; ++half) {
                    for (std::uint64_t r = 0; r < ir; ++r) { bank.rows.emplace_back(half, e * ir + r); }
                }
            }
        }
        m.add(p + "mlp.experts.gate_up_proj", {experts, 2 * ir, hidden}, std::move(bank));
        m.add(p + "mlp.experts.down_proj", {experts, hidden, ir}, m.whole(down));
        for (std::uint64_t e = 0; e < experts; ++e) {
            const auto ep = p + "mlp.experts." + std::to_string(e) + ".";
            for (std::uint32_t half = 0; half < 2; ++half) {
                const auto* tensor = fused ? fused : (half ? up : gate);
                const auto first   = fused ? e * 2 * ir + half * ir : e * ir;
                auto spec          = m.slice(*tensor, first, ir);
                m.add(ep + (half ? "up_proj.weight" : "gate_proj.weight"), {ir, hidden},
                      std::move(spec));
            }
            auto spec = m.slice(down, e * hidden, hidden);
            m.add(ep + "down_proj.weight", {hidden, ir}, std::move(spec));
        }
        m.note("expert bank exposed as a Hugging Face bank and per-expert matrices");
    };
    for (std::uint64_t i = 0; i < layers; ++i) {
        block("blk." + std::to_string(i) + ".", "model.layers." + std::to_string(i) + ".",
              recurrent[i]);
    }
    for (std::uint64_t j = 0; j < mtp; ++j) {
        const auto s = "blk." + std::to_string(layers + j) + ".";
        block(s, "mtp.layers." + std::to_string(j) + ".", false);
        if (j) { continue; }
        m.add("mtp.fc.weight", {hidden, 2 * hidden}, m.whole(m.need(s + "nextn.eh_proj.weight")));
        for (const auto& [from, to] : {std::pair{"nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight"},
                                       std::pair{"nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight"},
                                       std::pair{"nextn.shared_head_norm.weight", "mtp.norm.weight"}}) {
            m.add(to, {hidden}, subtract1(m.whole(m.need(s + from))));
            m.note("RMSNorm weight stored as 1+w, read as stored-1");
        }
        m.note("MTP block moved back from nextn layer to mtp.*");
    }
}

// ---------------------------------------------------------------------------------------------
// gemma4

// The full-attention layers' RoPE. Proportional RoPE: rope_freqs holds 1 for rotated pairs and
// 1e30 for the rest.
OJson gemma4_full_rope(Mapper& m, std::uint64_t global_dim) {
    if (const auto* freqs = m.find("rope_freqs.weight")) {
        std::vector<float> values(freqs->elements());
        for (std::size_t i = 0; i < values.size(); ++i) { values[i] = m.element(*freqs, i); }
        const auto rotated =
            static_cast<std::uint64_t>(std::count(values.begin(), values.end(), 1.0F));
        return OJson{{"rope_type", "proportional"},
                     {"partial_rotary_factor",
                      static_cast<double>(2 * rotated) / static_cast<double>(global_dim)},
                     {"rope_theta", m.real("rope.freq_base")}};
    }
    return OJson{{"rope_type", "default"}, {"rope_theta", m.real("rope.freq_base")}};
}

void map_gemma4(Mapper& m) {
    const auto& embd   = m.need("token_embd.weight");
    const auto vocab   = embd.shape[0];
    const auto hidden  = m.count("embedding_length");
    const auto layers  = m.count("block_count");
    const auto* swa    = m.meta("attention.sliding_window_pattern");
    if (!swa || !swa->is_array() || swa->size() < layers) {
        throw std::runtime_error("gemma4: attention.sliding_window_pattern must cover every layer");
    }
    // Gemma 4 E2B/E4B: per-layer embeddings (PLE) and the last shared_kv_layers layers reading
    // the keys and values of an earlier layer (llama.cpp gemma4.cpp: n_layer_kv_from_start).
    const auto ple_width = m.count_or("embedding_length_per_layer_input", 0);
    const auto shared    = m.count_or("attention.shared_kv_layers", 0);
    if (shared >= layers) {
        throw std::runtime_error("gemma4: attention.shared_kv_layers leaves no layer computing "
                                 "its own keys and values");
    }
    const auto first_shared = layers - shared;
    std::optional<std::uint64_t> first_full, first_sliding;
    OJson types = OJson::array();
    for (std::uint64_t i = 0; i < layers; ++i) {
        const bool sliding = flag((*swa)[i]);
        types.push_back(sliding ? "sliding_attention" : "full_attention");
        auto& first = sliding ? first_sliding : first_full;
        if (!first) { first = i; }
    }
    const auto heads = m.per_layer("attention.head_count", 0);
    const auto s0    = [](std::uint64_t layer) { return "blk." + std::to_string(layer) + "."; };
    const bool k_eq_v =
        first_full && !m.find(s0(*first_full) + "attn_v.weight") &&
        !m.find(s0(*first_full) + "attn_qkv.weight");
    OJson config            = OJson::object();
    config["architectures"] = OJson::array({"Gemma4ForCausalLM"});
    config["model_type"]    = "gemma4_text";
    config["hidden_size"]   = hidden;
    config["vocab_size"]    = vocab;
    config["num_hidden_layers"]       = layers;
    config["max_position_embeddings"] = m.count("context_length");
    config["tie_word_embeddings"]     = m.find("output.weight") == nullptr;
    config["rms_norm_eps"]            = m.real("attention.layer_norm_rms_epsilon");
    config["layer_types"]             = types;
    config["num_attention_heads"]     = heads;
    config["attention_k_eq_v"]        = k_eq_v;
    OJson rope                        = OJson::object();
    if (first_sliding) {
        config["sliding_window"]      = m.count("attention.sliding_window");
        config["head_dim"]            = m.count("attention.key_length_swa");
        config["num_key_value_heads"] = m.per_layer("attention.head_count_kv", *first_sliding);
        rope["sliding_attention"]     = OJson{{"rope_type", "default"},
                                              {"rope_theta", m.real_or("rope.freq_base_swa", 10000.0)}};
    }
    if (first_full) {
        const auto global_dim        = m.count("attention.key_length");
        config["global_head_dim"]    = global_dim;
        config["num_global_key_value_heads"] =
            m.per_layer("attention.head_count_kv", *first_full);
        if (!first_sliding) {
            config["num_key_value_heads"] = config["num_global_key_value_heads"];
        }
        rope["full_attention"] = gemma4_full_rope(m, global_dim);
    }
    config["rope_parameters"] = rope;
    if (const auto* cap = m.meta("final_logit_softcapping"); cap && cap->is_number()) {
        config["final_logit_softcapping"] = cap->get<double>();
    }
    config["intermediate_size"] = m.per_layer("feed_forward_length", 0);
    const auto experts          = m.count_or("expert_count", 0);
    config["enable_moe_block"]  = experts > 0;
    if (experts) {
        config["num_experts"]           = experts;
        config["top_k_experts"]         = m.count("expert_used_count");
        config["moe_intermediate_size"] = m.per_layer("expert_feed_forward_length", 0);
    }
    config["hidden_activation"]           = "gelu_pytorch_tanh";
    config["attention_bias"]              = false;
    config["hidden_size_per_layer_input"] = ple_width;
    config["num_kv_shared_layers"]        = shared;
    // transformers' use_double_wide_mlp: the shared layers' MLP is twice layer 0's. The GGUF
    // stores each layer's width; any other per-layer pattern has no Hugging Face config.
    const auto ff0 = m.per_layer("feed_forward_length", 0);
    bool uniform = true, double_wide = shared > 0;
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto ff = m.per_layer("feed_forward_length", i);
        uniform       = uniform && ff == ff0;
        double_wide   = double_wide && ff == (i < first_shared ? ff0 : 2 * ff0);
    }
    if (!uniform && !double_wide) {
        throw std::runtime_error("gemma4: feed_forward_length differs between layers other than "
                                 "by doubling on the shared-KV layers");
    }
    config["use_double_wide_mlp"] = !uniform;
    const TensorInfo* ple_table   = nullptr;
    if (ple_width) {
        ple_table = &m.need("per_layer_token_embd.weight");
        if (ple_table->shape.size() != 2 || ple_table->shape[1] != layers * ple_width) {
            throw std::runtime_error("gemma4: per_layer_token_embd is not [vocab, layers * " +
                                     std::to_string(ple_width) + "]");
        }
        config["vocab_size_per_layer_input"] = ple_table->shape[0];
    }
    // The image tokens of the prompt (a vision projector file names none): <|image> opens an
    // image, <|image|> is each soft token, <image|> closes it.
    if (const auto* tokens = m.file().find("tokenizer.ggml.tokens"); tokens && tokens->is_array()) {
        for (const auto& [key, text] : {std::pair{"boi_token_id", "<|image>"},
                                        std::pair{"image_token_id", "<|image|>"},
                                        std::pair{"eoi_token_id", "<image|>"}}) {
            for (std::size_t id = 0; id < tokens->size(); ++id) {
                if ((*tokens)[id].is_string() && (*tokens)[id].get<std::string>() == text) {
                    config[key] = id;
                    break;
                }
            }
        }
    }
    m.out.config = std::move(config);

    m.add("model.embed_tokens.weight", {vocab, hidden}, m.whole(embd));
    if (ple_table) {
        const auto rows = layers * ple_width;
        m.add("model.embed_tokens_per_layer.weight", {ple_table->shape[0], rows},
              m.whole(*ple_table));
        m.add("model.per_layer_model_projection.weight", {rows, hidden},
              m.whole(m.need("per_layer_model_proj.weight")));
        m.add("model.per_layer_projection_norm.weight", {ple_width},
              m.whole(m.need("per_layer_proj_norm.weight")));
    }
    if (const auto* output = m.find("output.weight")) {
        m.add("lm_head.weight", {vocab, hidden}, m.whole(*output));
    }
    m.add("model.norm.weight", {hidden}, m.whole(m.need("output_norm.weight")));
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto s       = s0(i);
        const auto p       = "model.layers." + std::to_string(i) + ".";
        const bool sliding = flag((*swa)[i]);
        const auto d = sliding ? m.count("attention.key_length_swa") : m.count("attention.key_length");
        const auto kv = m.per_layer("attention.head_count_kv", i);
        const auto vector = [&](const std::string& from, const std::string& to, std::uint64_t n) {
            m.add(p + to, {n}, m.whole(m.need(s + from)));
        };
        vector("attn_norm.weight", "input_layernorm.weight", hidden);
        const bool owns_kv = i < first_shared;
        if (const auto* qkv = m.find(s + "attn_qkv.weight")) {
            const bool has_v = !(k_eq_v && !sliding);
            m.add(p + "self_attn.q_proj.weight", {heads * d, hidden}, m.slice(*qkv, 0, heads * d));
            if (owns_kv) {
                m.add(p + "self_attn.k_proj.weight", {kv * d, hidden},
                      m.slice(*qkv, heads * d, kv * d));
            }
            if (owns_kv && has_v) {
                m.add(p + "self_attn.v_proj.weight", {kv * d, hidden},
                      m.slice(*qkv, heads * d + kv * d, kv * d));
            }
            m.note("fused attn_qkv split back into q/k/v projections");
        } else {
            for (const auto& [from, to, rows] :
                 {std::tuple{"attn_q.weight", "q_proj", heads * d},
                  std::tuple{"attn_k.weight", "k_proj", kv * d},
                  std::tuple{"attn_v.weight", "v_proj", kv * d}}) {
                // A shared-KV layer computes its query only; the GGUF may still carry k/v.
                if (!owns_kv && std::string(to) != "q_proj") { continue; }
                const auto* tensor = m.find(s + from);
                if (!tensor) {
                    if (std::string(to) == "v_proj" && k_eq_v && !sliding) { continue; }
                    (void)m.need(s + from);
                }
                auto spec = m.whole(*tensor);
                m.add(p + "self_attn." + to + ".weight", {rows, hidden}, std::move(spec));
            }
        }
        auto o = m.whole(m.need(s + "attn_output.weight"));
        m.add(p + "self_attn.o_proj.weight", {hidden, heads * d}, std::move(o));
        vector("attn_q_norm.weight", "self_attn.q_norm.weight", d);
        if (owns_kv) { vector("attn_k_norm.weight", "self_attn.k_norm.weight", d); }
        vector("post_attention_norm.weight", "post_attention_layernorm.weight", hidden);
        vector("ffn_norm.weight", "pre_feedforward_layernorm.weight", hidden);
        vector("post_ffw_norm.weight", "post_feedforward_layernorm.weight", hidden);
        const auto ff = m.per_layer("feed_forward_length", i);
        for (const auto& [from, to, rows, cols] :
             {std::tuple{"ffn_gate.weight", "gate_proj", ff, hidden},
              std::tuple{"ffn_up.weight", "up_proj", ff, hidden},
              std::tuple{"ffn_down.weight", "down_proj", hidden, ff}}) {
            auto spec = m.whole(m.need(s + from));
            m.add(p + "mlp." + to + ".weight", {rows, cols}, std::move(spec));
        }
        if (const auto* scalar = m.find(s + "layer_output_scale.weight")) {
            m.add(p + "layer_scalar", {1}, m.whole(*scalar));
        }
        if (ple_width) {
            m.add(p + "per_layer_input_gate.weight", {ple_width, hidden},
                  m.whole(m.need(s + "inp_gate.weight")));
            m.add(p + "per_layer_projection.weight", {hidden, ple_width},
                  m.whole(m.need(s + "proj.weight")));
            vector("post_norm.weight", "post_per_layer_input_norm.weight", hidden);
        }
        if (!experts) { continue; }
        vector("pre_ffw_norm_2.weight", "pre_feedforward_layernorm_2.weight", hidden);
        vector("post_ffw_norm_1.weight", "post_feedforward_layernorm_1.weight", hidden);
        vector("post_ffw_norm_2.weight", "post_feedforward_layernorm_2.weight", hidden);
        m.add(p + "router.proj.weight", {experts, hidden}, m.whole(m.need(s + "ffn_gate_inp.weight")));
        m.add(p + "router.scale", {hidden}, m.whole(m.need(s + "ffn_gate_inp.scale")));
        const auto ir     = m.per_layer("expert_feed_forward_length", i);
        const auto* fused = m.find(s + "ffn_gate_up_exps.weight");
        const auto* gate  = fused ? nullptr : &m.need(s + "ffn_gate_exps.weight");
        const auto* up    = fused ? nullptr : &m.need(s + "ffn_up_exps.weight");
        const auto& down  = m.need(s + "ffn_down_exps.weight");
        const bool folded = down.ggml_type == ggml::NVFP4;
        for (std::uint64_t e = 0; e < experts; ++e) {
            const auto ep = p + "experts." + std::to_string(e) + ".";
            for (std::uint32_t half = 0; half < 2; ++half) {
                const auto* tensor = fused ? fused : (half ? up : gate);
                const auto first   = fused ? e * 2 * ir + half * ir : e * ir;
                auto spec          = m.slice(*tensor, first, ir);
                m.add(ep + (half ? "up_proj.weight" : "gate_proj.weight"), {ir, hidden},
                      std::move(spec));
            }
            auto spec = m.slice(down, e * hidden, hidden);
            m.add(ep + "down_proj.weight", {hidden, ir}, std::move(spec));
        }
        const auto* scale = m.find(s + "ffn_down_exps.scale");
        if (folded || !scale) {
            m.add_view(p + "router.per_expert_scale", {experts},
                       std::make_shared<ConstantView>(
                           1.0F, folded ? "per-expert scale folded into the NVFP4 down experts"
                                        : "per-expert scale of one"));
            m.note(folded ? "per-expert scale kept folded in the NVFP4 down-expert divisors"
                          : "per-expert scale absent (all one)");
        } else {
            m.add(p + "router.per_expert_scale", {experts}, m.whole(*scale));
        }
        m.note("expert bank split into per-expert matrices");
    }
}

// ---------------------------------------------------------------------------------------------
// gemma4-assistant: Gemma 4's multi-token-prediction drafter (llama.cpp's gemma4-assistant, e.g.
// unsloth's mtp-gemma-4-31B-it.gguf). Every layer computes queries only and reads the target's
// keys and values; the drafter projects [token embedding, target hidden] down to its own width
// (nextn.pre_projection) and its final hidden back up (nextn.post_projection); its head is its
// own token table. Mapped to transformers' Gemma4AssistantForCausalLM.

void map_gemma4_assistant(Mapper& m) {
    const auto& embd    = m.need("token_embd.weight");
    const auto vocab    = embd.shape[0];
    const auto hidden   = m.count("embedding_length");
    const auto backbone = m.count("embedding_length_out");
    const auto layers   = m.count("block_count");
    if (m.count("attention.shared_kv_layers") != layers) {
        throw std::runtime_error("gemma4-assistant: attention.shared_kv_layers must equal "
                                 "block_count (every layer reads the target's keys and values)");
    }
    if (m.count_or("embedding_length_per_layer_input", 0)) {
        throw std::runtime_error("gemma4-assistant: per-layer inputs are not a drafter feature");
    }
    if (m.find("output.weight")) {
        throw std::runtime_error("gemma4-assistant: a separate output.weight is not the tied "
                                 "head of Gemma4AssistantForCausalLM");
    }
    const auto* swa = m.meta("attention.sliding_window_pattern");
    if (!swa || !swa->is_array() || swa->size() < layers) {
        throw std::runtime_error(
            "gemma4-assistant: attention.sliding_window_pattern must cover every layer");
    }
    std::optional<std::uint64_t> first_full, first_sliding;
    OJson types = OJson::array();
    for (std::uint64_t i = 0; i < layers; ++i) {
        const bool sliding = flag((*swa)[i]);
        types.push_back(sliding ? "sliding_attention" : "full_attention");
        auto& first = sliding ? first_sliding : first_full;
        if (!first) { first = i; }
    }
    const auto heads = m.per_layer("attention.head_count", 0);
    for (std::uint64_t i = 0; i < layers; ++i) {
        if (m.per_layer("attention.head_count", i) != heads) {
            throw std::runtime_error("gemma4-assistant: attention.head_count differs by layer");
        }
    }
    // The key/value geometry is the target's: the drafter's own attention computes queries of it.
    OJson text                   = OJson::object();
    text["model_type"]           = "gemma4_text";
    text["hidden_activation"]    = "gelu_pytorch_tanh";
    text["enable_moe_block"]     = false;
    text["hidden_size_per_layer_input"] = 0;
    text["hidden_size"]          = hidden;
    text["vocab_size"]           = vocab;
    text["num_hidden_layers"]    = layers;
    text["num_kv_shared_layers"] = layers;
    text["layer_types"]          = types;
    text["num_attention_heads"]  = heads;
    text["max_position_embeddings"] = m.count("context_length");
    const auto ff                = m.per_layer("feed_forward_length", 0);
    text["intermediate_size"]    = ff;
    text["rms_norm_eps"]         = m.real("attention.layer_norm_rms_epsilon");
    text["tie_word_embeddings"]  = true;
    OJson rope                   = OJson::object();
    // assistant_config reads both geometries; a drafter of one layer kind still names both.
    text["sliding_window"] = m.count("attention.sliding_window");
    text["head_dim"]       = m.count("attention.key_length_swa");
    text["num_key_value_heads"] =
        m.per_layer("attention.head_count_kv", first_sliding.value_or(0));
    rope["sliding_attention"] = OJson{{"rope_type", "default"},
                                      {"rope_theta", m.real_or("rope.freq_base_swa", 10000.0)}};
    const auto global_dim        = m.count("attention.key_length");
    text["global_head_dim"]      = global_dim;
    text["num_global_key_value_heads"] =
        m.per_layer("attention.head_count_kv", first_full.value_or(0));
    rope["full_attention"] = gemma4_full_rope(m, global_dim);
    text["rope_parameters"] = rope;
    for (std::uint64_t i = 0; i < layers; ++i) {
        if (m.per_layer("feed_forward_length", i) != ff) {
            throw std::runtime_error("gemma4-assistant: feed_forward_length differs by layer");
        }
    }
    OJson config                    = OJson::object();
    config["architectures"]         = OJson::array({"Gemma4AssistantForCausalLM"});
    config["model_type"]            = "gemma4_assistant";
    config["backbone_hidden_size"]  = backbone;
    config["use_ordered_embeddings"] = false;
    config["tie_word_embeddings"]   = true;
    config["text_config"]           = std::move(text);
    m.out.config                    = std::move(config);

    m.add("model.embed_tokens.weight", {vocab, hidden}, m.whole(embd));
    m.add("model.norm.weight", {hidden}, m.whole(m.need("output_norm.weight")));
    m.add("pre_projection.weight", {hidden, 2 * backbone},
          m.whole(m.need("nextn.pre_projection.weight")));
    m.add("post_projection.weight", {backbone, hidden},
          m.whole(m.need("nextn.post_projection.weight")));
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto s       = "blk." + std::to_string(i) + ".";
        const auto p       = "model.layers." + std::to_string(i) + ".";
        const bool sliding = flag((*swa)[i]);
        const auto d = sliding ? m.count("attention.key_length_swa") : global_dim;
        for (const char* own : {"attn_k.weight", "attn_v.weight", "attn_qkv.weight"}) {
            if (m.find(s + own)) {
                throw std::runtime_error("gemma4-assistant: " + s + own +
                                         " present; the drafter reads the target's keys and "
                                         "values");
            }
        }
        for (const auto& [from, to, n] :
             {std::tuple{"attn_norm.weight", "input_layernorm.weight", hidden},
              std::tuple{"post_attention_norm.weight", "post_attention_layernorm.weight", hidden},
              std::tuple{"ffn_norm.weight", "pre_feedforward_layernorm.weight", hidden},
              std::tuple{"post_ffw_norm.weight", "post_feedforward_layernorm.weight", hidden},
              std::tuple{"attn_q_norm.weight", "self_attn.q_norm.weight", d}}) {
            m.add(p + to, {n}, m.whole(m.need(s + from)));
        }
        m.add(p + "self_attn.q_proj.weight", {heads * d, hidden},
              m.whole(m.need(s + "attn_q.weight")));
        m.add(p + "self_attn.o_proj.weight", {hidden, heads * d},
              m.whole(m.need(s + "attn_output.weight")));
        for (const auto& [from, to, rows, cols] :
             {std::tuple{"ffn_gate.weight", "gate_proj", ff, hidden},
              std::tuple{"ffn_up.weight", "up_proj", ff, hidden},
              std::tuple{"ffn_down.weight", "down_proj", hidden, ff}}) {
            m.add(p + "mlp." + to + ".weight", {rows, cols}, m.whole(m.need(s + from)));
        }
        m.add(p + "layer_scalar", {1}, m.whole(m.need(s + "layer_output_scale.weight")));
    }
}

// ---------------------------------------------------------------------------------------------
// Dense decoders (llama, qwen2, qwen3, granite, olmo2, smollm3, mistral3, phi3), run by the
// dense_decoder recipes: the GGUF is read back into the Hugging Face config and tensor names
// llama.cpp's converter wrote it from (convert_hf_to_gguf.py's conversion/ modules).

enum class QkNorm : std::uint8_t { None, Head, Projection };

struct DenseArch {
    const char* gguf;
    const char* architecture;
    const char* model_type;
    // llama.cpp's LlamaModel.permute: each head's Q/K rows interleaved for ggml's NORM rope.
    bool interleaved_qk;
    bool qkv_bias;   // Qwen2's q/k/v biases
    QkNorm qk_norm;  // Qwen3: RMSNorm per head; OLMo 2: over the whole projection
    bool post_norms; // OLMo 2: RMSNorm after attention and after the MLP, none before them
};

constexpr DenseArch kDenseArchs[] = {
    {"llama", "LlamaForCausalLM", "llama", true, false, QkNorm::None, false},
    {"qwen2", "Qwen2ForCausalLM", "qwen2", false, true, QkNorm::None, false},
    {"qwen3", "Qwen3ForCausalLM", "qwen3", false, false, QkNorm::Head, false},
    // GraniteModel(LlamaModel): the muP multipliers as attention.scale, embedding_scale,
    // residual_scale and logit_scale.
    {"granite", "GraniteForCausalLM", "granite", true, false, QkNorm::None, false},
    // Olmo2Model(TextModel), also for Olmo3ForCausalLM: NEOX rope, Q/K stored as published.
    {"olmo2", "Olmo2ForCausalLM", "olmo2", false, false, QkNorm::Projection, true},
    // SmolLM3Model(LlamaModel): no rope on every 4th layer (llama.cpp's n_no_rope_layer_step).
    {"smollm3", "SmolLM3ForCausalLM", "smollm3", true, false, QkNorm::None, false},
    // Mistral3Model.Ministral3Model(LlamaModel): yarn and the llama 4 attention temperature.
    {"mistral3", "Ministral3ForCausalLM", "ministral3", true, false, QkNorm::None, false},
    // Phi3MiniModel(TextModel): fused attn_qkv and gate/up ffn_up, partial rotary, LongRoPE
    // factors as rope_factors_short / rope_factors_long; NEOX rope, Q/K stored as published.
    {"phi3", "Phi3ForCausalLM", "phi3", false, false, QkNorm::None, false},
    // MuseGlimmerModel(TextModel): Gemma 2's sandwich of (1 + w) norms stored with the 1 added
    // (the final norm plain), Q/K rows un-permuted to ggml's NORM rope, the weightless q/k norms
    // written as attn_q_norm = qk_scale_factor and attn_k_norm = 1, the attention output gate as
    // attn_gate, output_multiplier as logit_scale, NoPE on the full-attention layers.
    {"muse-glimmer", "MuseGlimmerForConditionalGeneration", "muse_glimmer_text", true, false,
     QkNorm::None, false},
};

const DenseArch* dense_arch(const std::string& name) {
    for (const auto& arch : kDenseArchs) {
        if (name == arch.gguf) { return &arch; }
    }
    return nullptr;
}

// LlamaModel.permute inverted: row (h * dim + half * dim / 2 + j) of a head reads the stored row
// (h * dim + 2 * j + half).
void deinterleave_heads(Spec& spec, std::uint64_t heads, std::uint64_t dim) {
    const auto rows = spec.rows;
    for (std::uint64_t h = 0; h < heads; ++h) {
        for (std::uint64_t half = 0; half < 2; ++half) {
            for (std::uint64_t j = 0; j < dim / 2; ++j) {
                spec.rows[h * dim + half * (dim / 2) + j] = rows[h * dim + 2 * j + half];
            }
        }
    }
}

// Which layers slide, as llama.cpp's load_swa_pattern reads attention.sliding_window_pattern: one
// flag per layer, or a period n (every n-th layer full, all sliding for 0; n_pattern when absent).
std::vector<bool> sliding_layers(Mapper& m, const std::string& arch, std::uint64_t layers,
                                 std::uint64_t n_pattern) {
    std::vector<bool> out(layers);
    const auto* pattern = m.meta("attention.sliding_window_pattern");
    if (pattern && pattern->is_array()) {
        if (pattern->size() != layers) {
            throw std::runtime_error(arch + ": attention.sliding_window_pattern must cover "
                                            "every layer");
        }
        for (std::uint64_t i = 0; i < layers; ++i) { out[i] = flag((*pattern)[i]); }
        return out;
    }
    const auto n = pattern ? m.count("attention.sliding_window_pattern") : n_pattern;
    for (std::uint64_t i = 0; i < layers; ++i) { out[i] = n == 0 || i % n < n - 1; }
    return out;
}

// The rope of a dense GGUF as transformers' rope_parameters: rope.scaling.* back into the keys
// TextModel.set_gguf_parameters read them from (linear, yarn), rope_freqs as per-pair frequency
// factors (Llama 3.x), and Ministral 3's attention temperature.
OJson dense_rope(Mapper& m, const std::string& arch, std::uint64_t dim, double theta) {
    OJson rope = OJson{{"rope_type", "default"}, {"rope_theta", theta}};
    const auto* scaling = m.meta("rope.scaling.type");
    const auto type = scaling && scaling->is_string() ? scaling->get<std::string>() : "none";
    const auto real = [&](const char* key, const char* to) {
        if (const auto* v = m.meta(key); v && v->is_number()) { rope[to] = v->get<double>(); }
    };
    if (type == "linear") {
        rope["rope_type"] = "linear";
        rope["factor"]    = m.real("rope.scaling.factor");
    } else if (type == "yarn") {
        rope["rope_type"] = "yarn";
        rope["factor"]    = m.real("rope.scaling.factor");
        rope["original_max_position_embeddings"] = m.count("rope.scaling.original_context_length");
        real("rope.scaling.yarn_beta_fast", "beta_fast");
        real("rope.scaling.yarn_beta_slow", "beta_slow");
        // attention_factor: rope.scaling.yarn_attn_factor since llama.cpp split the key from
        // the attn_factor older converters wrote (OLMo 3 GGUFs carry that one).
        real("rope.scaling.attn_factor", "attention_factor");
        real("rope.scaling.yarn_attn_factor", "attention_factor");
        // Written from mscale_all_dim (Ministral 3); transformers divides mscale's by it, and
        // the published configs carry the two equal.
        real("rope.scaling.yarn_log_multiplier", "mscale_all_dim");
        real("rope.scaling.yarn_log_multiplier", "mscale");
        if (m.real_or("rope.scaling.yarn_ext_factor", 1.0) != 1.0) {
            throw std::runtime_error(arch + ": rope.scaling.yarn_ext_factor other than 1 is not "
                                            "mapped");
        }
    } else if (type != "none") {
        throw std::runtime_error(arch + ": rope.scaling.type '" + type + "' is not mapped");
    }
    if (const auto* beta = m.meta("attention.temperature_scale")) {
        if (arch != "mistral3" || type != "yarn" || !beta->is_number()) {
            throw std::runtime_error(arch + ": attention.temperature_scale is mapped for "
                                            "mistral3 with yarn only");
        }
        if (beta->get<double>() != 0.0) { rope["llama_4_scaling_beta"] = beta->get<double>(); }
    }
    const auto* long_factors  = m.find("rope_factors_long.weight");
    const auto* short_factors = m.find("rope_factors_short.weight");
    if (long_factors || short_factors) {
        // Phi-3's LongRoPE: per-pair short and long factors, original_max_position_embeddings
        // from rope.scaling.original_context_length. The attn_factor Phi3MiniModel writes is
        // transformers' own default for it, which the dense adapter derives again.
        if (arch != "phi3" || !long_factors || !short_factors || type != "none" ||
            long_factors->elements() != dim / 2 || short_factors->elements() != dim / 2) {
            throw std::runtime_error(arch + ": rope_factors_long / rope_factors_short must hold "
                                            "one factor per rotated pair (phi3, no rope.scaling.type)");
        }
        const auto factors = [&](const TensorInfo& tensor) {
            OJson out = OJson::array();
            for (std::uint64_t i = 0; i < dim / 2; ++i) {
                out.push_back(static_cast<double>(m.element(tensor, i)));
            }
            return out;
        };
        rope = OJson{{"rope_type", "longrope"},
                     {"rope_theta", theta},
                     {"original_max_position_embeddings",
                      m.count("rope.scaling.original_context_length")},
                     {"short_factor", factors(*short_factors)},
                     {"long_factor", factors(*long_factors)}};
        m.note("LongRoPE factors read from rope_factors_short / rope_factors_long");
    }
    if (const auto* freqs = m.find("rope_freqs.weight")) {
        // Llama 3.x's rope scaling, as llama.cpp writes it: pair i turns at base_i / factor_i.
        if (type != "none" || freqs->elements() != dim / 2) {
            throw std::runtime_error(arch + ": rope_freqs must hold one factor per rotated pair "
                                            "of an unscaled rope");
        }
        OJson factors = OJson::array();
        for (std::uint64_t i = 0; i < dim / 2; ++i) {
            factors.push_back(static_cast<double>(m.element(*freqs, i)));
        }
        rope = OJson{{"rope_type", "frequency_factors"}, {"rope_theta", theta},
                     {"factors", std::move(factors)}};
        m.note("rope_freqs read as per-pair frequency factors");
    }
    return rope;
}

void map_dense(Mapper& m, const DenseArch& a) {
    const std::string arch = a.gguf;
    const auto& embd       = m.need("token_embd.weight");
    const auto vocab       = embd.shape[0];
    const auto hidden      = m.count("embedding_length");
    const auto layers      = m.count("block_count");
    const auto heads       = m.count("attention.head_count");
    const auto kv          = m.count_or("attention.head_count_kv", heads);
    const auto dim         = m.count_or("attention.key_length", hidden / heads);
    if (!heads || !kv || heads % kv || !dim || dim % 2) {
        throw std::runtime_error(arch + ": invalid attention geometry");
    }
    if (m.count_or("attention.value_length", dim) != dim) {
        throw std::runtime_error(arch + ": value heads wider or narrower than key heads");
    }
    // Phi-3 rotates the leading rope.dimension_count of each head (partial_rotary_factor).
    const auto rot = m.count_or("rope.dimension_count", dim);
    if (rot != dim && !(arch == "phi3" && rot && rot < dim && rot % 2 == 0)) {
        throw std::runtime_error(arch + ": partial rotary (rope.dimension_count below the head "
                                        "width) is not mapped");
    }
    OJson config            = OJson::object();
    if (rot != dim) {
        config["partial_rotary_factor"] = static_cast<double>(rot) / static_cast<double>(dim);
    }
    config["architectures"] = OJson::array({a.architecture});
    config["model_type"]    = a.model_type;
    config["hidden_size"]   = hidden;
    config["vocab_size"]    = vocab;
    config["num_hidden_layers"]       = layers;
    config["max_position_embeddings"] = m.count("context_length");
    config["num_attention_heads"]     = heads;
    config["num_key_value_heads"]     = kv;
    config["head_dim"]                = dim;
    config["intermediate_size"]       = m.count("feed_forward_length");
    config["rms_norm_eps"]            = m.real("attention.layer_norm_rms_epsilon");
    config["hidden_act"]              = "silu";
    config["tie_word_embeddings"]     = m.find("output.weight") == nullptr;
    config["attention_bias"]          = a.qkv_bias;
    const double theta = m.real_or("rope.freq_base", 10000.0);
    auto rope          = dense_rope(m, arch, rot, theta);

    if (arch == "granite") {
        // GraniteModel writes each multiplier only when the config's is truthy.
        const auto multiplier = [&](const char* key, const char* to) {
            if (const auto* v = m.meta(key); v && v->is_number()) { config[to] = v->get<double>(); }
        };
        multiplier("attention.scale", "attention_multiplier");
        multiplier("embedding_scale", "embedding_multiplier");
        multiplier("residual_scale", "residual_multiplier");
        multiplier("logit_scale", "logits_scaling");
        m.note("muP multipliers read from attention.scale, embedding_scale, residual_scale and "
               "logit_scale");
    }
    if (arch == "smollm3") {
        // llama.cpp hard-codes n_no_rope_layer_step = 4: layer i rotates unless (i + 1) % 4 == 0,
        // as SmolLM3Config's no_rope_layers with no_rope_layer_interval 4.
        OJson rotated = OJson::array();
        for (std::uint64_t i = 0; i < layers; ++i) { rotated.push_back((i + 1) % 4 != 0 ? 1 : 0); }
        config["no_rope_layers"] = std::move(rotated);
        m.note("no rope on every 4th layer (llama.cpp's n_no_rope_layer_step)");
    }
    const auto window = m.count_or("attention.sliding_window", 0);
    if (window && arch == "olmo2") {
        // OLMo 3: sliding window on the layers sliding_window_pattern marks, the rope scaling
        // on full-attention layers only (the sliding ones rotate at freq_base_swa, unscaled).
        OJson kinds = OJson::array();
        for (const bool slides : sliding_layers(m, arch, layers, 4)) { // load_swa_pattern(ml, 4)
            kinds.push_back(slides ? "sliding_attention" : "full_attention");
        }
        config["architectures"] = OJson::array({"Olmo3ForCausalLM"});
        config["model_type"]    = "olmo3";
        config["layer_types"]   = std::move(kinds);
        config["sliding_window"] = window;
        rope = OJson{{"full_attention", std::move(rope)},
                     {"sliding_attention",
                      OJson{{"rope_type", "default"},
                            {"rope_theta", m.real_or("rope.freq_base_swa", theta)}}}};
        m.note("OLMo 3 layer pattern read from attention.sliding_window_pattern");
    } else if (arch == "muse-glimmer") {
        // MuseGlimmerModel writes sliding_window and the layer pattern; llama.cpp rotates the
        // sliding layers only (at freq_base_swa, else freq_base) and leaves the full ones
        // position-free: Muse Glimmer's layer_rope_theta, 0 on a NoPE layer. With no rope on
        // the full layers, one rope group carries the sliding layers' base.
        if (!window) {
            throw std::runtime_error(arch + ": attention.sliding_window is required");
        }
        OJson kinds  = OJson::array();
        OJson thetas = OJson::array();
        const double swa_theta = m.real_or("rope.freq_base_swa", theta);
        for (const bool slides : sliding_layers(m, arch, layers, 4)) { // load_swa_pattern(ml, 4)
            kinds.push_back(slides ? "sliding_attention" : "full_attention");
            thetas.push_back(slides ? swa_theta : 0.0);
        }
        config["layer_types"]      = std::move(kinds);
        config["layer_rope_theta"] = std::move(thetas);
        config["sliding_window"]   = window;
        rope = dense_rope(m, arch, rot, swa_theta);
        // logit_scale is output_multiplier (applied before the soft cap); llama.cpp's graph
        // hard-codes the post-norm epsilon 1e-8 the converter does not write.
        config["output_multiplier"] = m.real("logit_scale");
        if (const double cap = m.real_or("final_logit_softcapping", 0.0); cap != 0.0) {
            config["final_logit_softcapping"] = cap;
        }
        config["post_norm_eps"] = 1e-8;
        // The weightless q/k norms: attn_q_norm holds qk_scale_factor in every element of every
        // layer, attn_k_norm ones; anything else is not the model the converter wrote.
        std::optional<float> factor;
        for (std::uint64_t i = 0; i < layers; ++i) {
            const auto s   = "blk." + std::to_string(i) + ".";
            const auto& qn = m.need(s + "attn_q_norm.weight");
            const auto& kn = m.need(s + "attn_k_norm.weight");
            if (qn.elements() != dim || kn.elements() != dim) {
                throw std::runtime_error(arch + ": q/k norms must hold one value per head dim");
            }
            for (std::uint64_t j = 0; j < dim; ++j) {
                const float q = m.element(qn, j);
                if (!factor) { factor = q; }
                if (q != *factor || m.element(kn, j) != 1.0F) {
                    throw std::runtime_error(arch + ": attn_q_norm / attn_k_norm are not the "
                                                    "constant qk_scale_factor / ones the "
                                                    "converter writes (layer " +
                                             std::to_string(i) + ")");
                }
            }
        }
        if (!factor || !(*factor > 0.0F)) {
            throw std::runtime_error(arch + ": qk_scale_factor must be positive");
        }
        config["qk_scale_factor"] = f32_shortest(*factor);
        m.note("qk_scale_factor read from the constant attn_q_norm (attn_k_norm all ones)");
        m.note("layer pattern and NoPE full layers read from attention.sliding_window_pattern");
    } else if (window && arch == "phi3") {
        // Phi3MiniModel writes the config's sliding_window (0 for none); the dense adapter reads
        // a window past every position as full attention.
        config["sliding_window"] = window;
    } else if (window) {
        throw std::runtime_error(arch + ": attention.sliding_window is not mapped for this "
                                        "architecture");
    }
    config["rope_parameters"] = std::move(rope);
    m.out.config              = std::move(config);

    m.add("model.embed_tokens.weight", {vocab, hidden}, m.whole(embd));
    if (const auto* output = m.find("output.weight")) {
        m.add("lm_head.weight", {vocab, hidden}, m.whole(*output));
    }
    m.add("model.norm.weight", {hidden}, m.whole(m.need("output_norm.weight")));
    const auto ff = m.count("feed_forward_length");
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto s      = "blk." + std::to_string(i) + ".";
        const auto p      = "model.layers." + std::to_string(i) + ".";
        const auto vector = [&](const std::string& from, const std::string& to, std::uint64_t n) {
            m.add(p + to, {n}, m.whole(m.need(s + from)));
        };
        if (arch == "muse-glimmer") {
            // The sandwich: every layer norm stored as 1 + w (MuseGlimmerModel.norm_shift).
            for (const auto& [from, to] :
                 {std::pair{"attn_norm.weight", "input_layernorm.weight"},
                  std::pair{"post_attention_norm.weight", "post_attention_layernorm.weight"},
                  std::pair{"ffn_norm.weight", "pre_feedforward_layernorm.weight"},
                  std::pair{"post_ffw_norm.weight", "post_feedforward_layernorm.weight"}}) {
                m.add(p + to, {hidden}, subtract1(m.whole(m.need(s + from))));
            }
            m.add(p + "self_attn.gate_proj.weight", {heads * dim, hidden},
                  m.whole(m.need(s + "attn_gate.weight")));
            m.note("RMSNorm weight stored as 1+w, read as stored-1");
        } else if (a.post_norms) {
            vector("post_attention_norm.weight", "post_attention_layernorm.weight", hidden);
            vector("post_ffw_norm.weight", "post_feedforward_layernorm.weight", hidden);
        } else {
            vector("attn_norm.weight", "input_layernorm.weight", hidden);
            vector("ffn_norm.weight", "post_attention_layernorm.weight", hidden);
        }
        // Phi-3 publishes attn_qkv (rows q, k, v) and ffn_up holding gate then up (the Hugging
        // Face qkv_proj and gate_up_proj): each part is a row range of the stored matrix.
        const auto* qkv = m.find(s + "attn_qkv.weight");
        std::uint64_t first = 0;
        for (const auto& [from, to, n] : {std::tuple{"attn_q", "q_proj", heads},
                                          std::tuple{"attn_k", "k_proj", kv},
                                          std::tuple{"attn_v", "v_proj", kv}}) {
            auto spec = qkv ? m.slice(*qkv, first, n * dim) : m.whole(m.need(s + from + ".weight"));
            first += n * dim;
            if (a.interleaved_qk && std::string(to) != "v_proj") {
                deinterleave_heads(spec, n, dim);
            }
            m.add(p + "self_attn." + to + ".weight", {n * dim, hidden}, std::move(spec));
            const auto* bias = m.find(s + from + ".bias");
            if (a.qkv_bias) {
                vector(std::string(from) + ".bias", "self_attn." + std::string(to) + ".bias",
                       n * dim);
            } else if (bias) {
                throw std::runtime_error(arch + ": " + bias->name + " (attention biases) is not "
                                                "mapped for this architecture");
            }
        }
        if (const auto* bias = m.find(s + "attn_output.bias")) {
            throw std::runtime_error(arch + ": " + bias->name + " (attention biases) is not "
                                            "mapped for this architecture");
        }
        m.add(p + "self_attn.o_proj.weight", {hidden, heads * dim},
              m.whole(m.need(s + "attn_output.weight")));
        if (a.qk_norm == QkNorm::Head) {
            vector("attn_q_norm.weight", "self_attn.q_norm.weight", dim);
            vector("attn_k_norm.weight", "self_attn.k_norm.weight", dim);
        } else if (a.qk_norm == QkNorm::Projection) {
            vector("attn_q_norm.weight", "self_attn.q_norm.weight", heads * dim);
            vector("attn_k_norm.weight", "self_attn.k_norm.weight", kv * dim);
        }
        if (qkv && first != Mapper::rows_of(*qkv)) {
            throw std::runtime_error(arch + ": " + qkv->name + " is not q, k and v rows");
        }
        const auto& up = m.need(s + "ffn_up.weight");
        const bool gate_up = !m.find(s + "ffn_gate.weight") && Mapper::rows_of(up) == 2 * ff;
        m.add(p + "mlp.gate_proj.weight", {ff, hidden},
              gate_up ? m.slice(up, 0, ff) : m.whole(m.need(s + "ffn_gate.weight")));
        m.add(p + "mlp.up_proj.weight", {ff, hidden}, gate_up ? m.slice(up, ff, ff) : m.whole(up));
        m.add(p + "mlp.down_proj.weight", {hidden, ff}, m.whole(m.need(s + "ffn_down.weight")));
        if (qkv) { m.note("attn_qkv split into q, k and v rows"); }
        if (gate_up) { m.note("ffn_up split into gate and up rows"); }
    }
    if (a.interleaved_qk) { m.note("Q/K rows of every head de-interleaved (LlamaModel.permute)"); }
}

// ---------------------------------------------------------------------------------------------
// gpt-oss, read back into transformers' GptOssForCausalLM (convert_hf_to_gguf.py GptOssModel):
// the experts' interleaved gate/up bank (gate on even rows) from ffn_gate_exps / ffn_up_exps and
// their biases, the router (ffn_gate_inp), attention sinks (attn_sinks), q/k/v/o with biases, and
// the layer pattern llama.cpp hard-codes (sliding window on even layers, set_swa_pattern(2)).
// MXFP4 experts keep their words (the nibbles of each block put in natural order), Q8_0 matrices
// keep theirs.

// Elements of F32 vectors, one logical row per element: rows[i] = (part, element).
Spec elements(const std::vector<const TensorInfo*>& tensors,
              std::vector<std::pair<std::uint32_t, std::uint64_t>> rows) {
    Spec spec;
    for (const auto* tensor : tensors) {
        if (tensor->ggml_type != ggml::F32) {
            throw std::runtime_error(tensor->name + ": expected an F32 vector");
        }
        Part part;
        part.tensor    = tensor;
        part.cols      = 1;
        part.row_bytes = 4;
        spec.parts.push_back(part);
    }
    spec.cols = 1;
    spec.rows = std::move(rows);
    return spec;
}

void map_gpt_oss(Mapper& m) {
    const auto& embd   = m.need("token_embd.weight");
    const auto vocab   = embd.shape[0];
    const auto hidden  = m.count("embedding_length");
    const auto layers  = m.count("block_count");
    const auto heads   = m.count("attention.head_count");
    const auto kv      = m.count("attention.head_count_kv");
    const auto dim     = m.count("attention.key_length");
    const auto experts = m.count("expert_count");
    const auto used    = m.count("expert_used_count");
    const auto ff      = m.count("expert_feed_forward_length");
    const auto window  = m.count("attention.sliding_window");
    if (!heads || !kv || heads % kv || !dim || dim % 2 || !experts || !used || !ff || !window) {
        throw std::runtime_error("gpt-oss: invalid geometry");
    }
    if (m.count_or("attention.value_length", dim) != dim) {
        throw std::runtime_error("gpt-oss: value heads wider or narrower than key heads");
    }
    if (m.count_or("rope.dimension_count", dim) != dim) {
        throw std::runtime_error("gpt-oss: partial rotary is not mapped");
    }
    const auto* scaling = m.meta("rope.scaling.type");
    if (!scaling || !scaling->is_string() || scaling->get<std::string>() != "yarn") {
        throw std::runtime_error("gpt-oss: expected rope.scaling.type yarn");
    }
    OJson config                      = OJson::object();
    config["architectures"]           = OJson::array({"GptOssForCausalLM"});
    config["model_type"]              = "gpt_oss";
    config["hidden_size"]             = hidden;
    config["vocab_size"]              = vocab;
    config["num_hidden_layers"]       = layers;
    config["max_position_embeddings"] = m.count("context_length");
    config["num_attention_heads"]     = heads;
    config["num_key_value_heads"]     = kv;
    config["head_dim"]                = dim;
    config["intermediate_size"]       = ff;
    config["num_local_experts"]       = experts;
    config["num_experts_per_tok"]     = used;
    config["rms_norm_eps"]            = m.real("attention.layer_norm_rms_epsilon");
    config["hidden_act"]              = "silu";
    config["attention_bias"]          = true;
    config["tie_word_embeddings"]     = m.find("output.weight") == nullptr;
    // Not in the GGUF: llama.cpp's build_openai_moe hard-codes the clamp (7.0), as GptOssConfig's
    // swiglu_limit; alpha 1.702 is a constant of both modeling codes.
    config["swiglu_limit"]   = 7.0;
    config["sliding_window"] = window;
    OJson kinds              = OJson::array();
    for (std::uint64_t i = 0; i < layers; ++i) {
        kinds.push_back(i % 2 == 0 ? "sliding_attention" : "full_attention");
    }
    config["layer_types"] = std::move(kinds);
    // GptOssConfig's rope_scaling. Its truncate (false: the YaRN correction range is not rounded)
    // is not written to the GGUF and is every gpt-oss checkpoint's.
    config["rope_parameters"] = OJson{
        {"rope_type", "yarn"},
        {"rope_theta", m.real_or("rope.freq_base", 10000.0)},
        {"factor", m.real("rope.scaling.factor")},
        {"original_max_position_embeddings", m.count("rope.scaling.original_context_length")},
        {"beta_fast", m.real_or("rope.scaling.yarn_beta_fast", 32.0)},
        {"beta_slow", m.real_or("rope.scaling.yarn_beta_slow", 1.0)},
        {"truncate", false}};
    m.out.config = std::move(config);
    m.note("layer pattern: sliding window on even layers (llama.cpp's set_swa_pattern(2))");
    m.note("swiglu_limit 7.0 and rope truncate false (not stored in the GGUF)");

    m.add("model.embed_tokens.weight", {vocab, hidden}, m.whole(embd));
    if (const auto* output = m.find("output.weight")) {
        m.add("lm_head.weight", {vocab, hidden}, m.whole(*output));
    }
    m.add("model.norm.weight", {hidden}, m.whole(m.need("output_norm.weight")));
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto s      = "blk." + std::to_string(i) + ".";
        const auto p      = "model.layers." + std::to_string(i) + ".";
        const auto vector = [&](const std::string& from, const std::string& to, std::uint64_t n) {
            m.add(p + to, {n}, m.whole(m.need(s + from)));
        };
        vector("attn_norm.weight", "input_layernorm.weight", hidden);
        vector("post_attention_norm.weight", "post_attention_layernorm.weight", hidden);
        for (const auto& [from, to, n] : {std::tuple{"attn_q", "q_proj", heads},
                                          std::tuple{"attn_k", "k_proj", kv},
                                          std::tuple{"attn_v", "v_proj", kv}}) {
            m.add(p + "self_attn." + to + ".weight", {n * dim, hidden},
                  m.whole(m.need(s + from + ".weight")));
            vector(std::string(from) + ".bias", "self_attn." + std::string(to) + ".bias", n * dim);
        }
        m.add(p + "self_attn.o_proj.weight", {hidden, heads * dim},
              m.whole(m.need(s + "attn_output.weight")));
        vector("attn_output.bias", "self_attn.o_proj.bias", hidden);
        vector("attn_sinks.weight", "self_attn.sinks", heads);
        m.add(p + "mlp.router.weight", {experts, hidden},
              m.whole(m.need(s + "ffn_gate_inp.weight")));
        vector("ffn_gate_inp.bias", "mlp.router.bias", experts);

        // The gate/up bank with gate and up rows interleaved, expert by expert.
        const auto& gate = m.need(s + "ffn_gate_exps.weight");
        const auto& up   = m.need(s + "ffn_up_exps.weight");
        if (Mapper::rows_of(gate) != experts * ff || Mapper::rows_of(up) != experts * ff) {
            throw std::runtime_error("gpt-oss: " + gate.name + " / " + up.name +
                                     " are not experts x expert_feed_forward_length rows");
        }
        Spec gate_up;
        gate_up.parts = {m.part(gate), m.part(up)};
        gate_up.cols  = gate_up.parts[0].cols;
        std::vector<std::pair<std::uint32_t, std::uint64_t>> bias_rows;
        for (std::uint64_t r = 0; r < experts * ff; ++r) {
            gate_up.rows.emplace_back(0, r);
            gate_up.rows.emplace_back(1, r);
            bias_rows.emplace_back(0, r);
            bias_rows.emplace_back(1, r);
        }
        m.add(p + "mlp.experts.gate_up_proj.weight", {experts * 2 * ff, hidden},
              std::move(gate_up));
        m.add(p + "mlp.experts.gate_up_proj_bias", {experts, 2 * ff},
              elements({&m.need(s + "ffn_gate_exps.bias"), &m.need(s + "ffn_up_exps.bias")},
                       std::move(bias_rows)));
        const auto& down = m.need(s + "ffn_down_exps.weight");
        if (Mapper::rows_of(down) != experts * hidden) {
            throw std::runtime_error("gpt-oss: " + down.name + " is not experts x hidden rows");
        }
        m.add(p + "mlp.experts.down_proj.weight", {experts * hidden, ff}, m.whole(down));
        m.add(p + "mlp.experts.down_proj_bias", {experts, hidden},
              m.whole(m.need(s + "ffn_down_exps.bias")));
    }
    m.note("expert gate/up banks interleaved back from ffn_gate_exps / ffn_up_exps");
}

// What a DFlash and a DFlash2 GGUF share: the attention geometry (full rotary, no rope scaling),
// the target layers (stored one past the transformers target_layer_ids), the mask token, the
// vocabulary and the per-layer attention kinds, with no token table or head of the drafter's own
// (it uses the target's). llama.cpp reads the per-layer window flags only with a window; without
// one every layer is full.
struct DFlashGeometry {
    std::uint64_t hidden = 0, layers = 0, heads = 0, kv = 0, dim = 0, window = 0;
    std::size_t targets = 0;
    std::size_t vocab   = 0;
    std::int64_t mask   = 0;
    OJson taps          = OJson::array();
    OJson kinds         = OJson::array();
};

DFlashGeometry read_dflash_geometry(Mapper& m) {
    const std::string arch = m.arch();
    DFlashGeometry g;
    g.hidden = m.count("embedding_length");
    g.layers = m.count("block_count");
    g.heads  = m.count("attention.head_count");
    g.kv     = m.count_or("attention.head_count_kv", g.heads);
    g.dim    = m.count_or("attention.key_length", g.hidden / g.heads);
    if (!g.heads || !g.kv || g.heads % g.kv || !g.dim ||
        m.count_or("attention.value_length", g.dim) != g.dim) {
        throw std::runtime_error(arch + ": invalid attention geometry");
    }
    if (m.count_or("rope.dimension_count", g.dim) != g.dim) {
        throw std::runtime_error(arch + ": partial rotary is not mapped");
    }
    if (const auto* scaling = m.meta("rope.scaling.type");
        scaling && !(scaling->is_string() && scaling->get<std::string>() == "none")) {
        throw std::runtime_error(arch + ": rope scaling is not mapped");
    }
    const auto* targets = m.meta("target_layers");
    if (!targets || !targets->is_array() || targets->empty()) {
        throw std::runtime_error(arch + ": target_layers is required");
    }
    for (const auto& layer : *targets) {
        if (!layer.is_number_integer() || layer.get<std::int64_t>() < 1) {
            throw std::runtime_error(arch + ": target_layers must name layers past the embedding");
        }
        g.taps.push_back(layer.get<std::int64_t>() - 1);
    }
    g.targets        = targets->size();
    const auto* mask = m.file().find("tokenizer.ggml.mask_token_id");
    if (!mask || !mask->is_number_integer()) {
        throw std::runtime_error(arch + ": tokenizer.ggml.mask_token_id is required");
    }
    g.mask             = mask->get<std::int64_t>();
    const auto* tokens = m.file().find("tokenizer.ggml.tokens");
    if (!tokens || !tokens->is_array()) {
        throw std::runtime_error(arch + ": tokenizer.ggml.tokens is required");
    }
    g.vocab             = tokens->size();
    g.window            = m.count_or("attention.sliding_window", 0);
    const auto* pattern = m.meta("attention.sliding_window_pattern");
    if (g.window && (!pattern || !pattern->is_array() || pattern->size() != g.layers)) {
        throw std::runtime_error(arch + ": attention.sliding_window_pattern must flag every layer");
    }
    for (std::uint64_t i = 0; i < g.layers; ++i) {
        g.kinds.push_back(g.window && flag((*pattern)[i]) ? "sliding_attention" : "full_attention");
    }
    for (const char* name : {"token_embd.weight", "output.weight"}) {
        if (m.find(name)) {
            throw std::runtime_error(arch + ": " + name +
                                     " present; a DFlash drafter uses the target's");
        }
    }
    return g;
}

// llama.cpp's DFlashModel written from z-lab's DFlash2DraftModel (Qwen3-style layers, dynamic
// grouped convolutions after attention and MLP, a candidate selector with two token codebooks):
// read back into DFlash2DraftModel's config and names. As in DFlash, no token table and no head
// (the target's), target_layers one past the transformers target_layer_ids, norms plain. Every
// stored matrix is referenced as published (llama.cpp's Q4_0 / Q8_0 words, codebooks included);
// the convolution base kernels are F32 [2, kernel, hidden] as transformers stores them.
void map_dflash2(Mapper& m) {
    const std::string arch = m.arch();
    DFlashGeometry g       = read_dflash_geometry(m);
    const auto hidden = g.hidden, layers = g.layers, heads = g.heads, kv = g.kv, dim = g.dim;
    const auto ff = m.count("feed_forward_length");
    if (dim % 2) { throw std::runtime_error(arch + ": invalid attention geometry"); }
    // An M-RoPE target: llama.cpp writes degenerate sections [dim/2, 0, 0, 0], the draft rotating
    // every pair by the text position, which is the plain rope DFlash2DraftModel applies.
    if (const auto* sections = m.meta("rope.dimension_sections")) {
        bool plain = sections->is_array() && !sections->empty();
        for (std::size_t i = 0; plain && i < sections->size(); ++i) {
            const auto& value = (*sections)[i];
            plain             = value.is_number_integer() &&
                    value.get<std::int64_t>() == (i == 0 ? static_cast<std::int64_t>(dim / 2) : 0);
        }
        if (!plain) {
            throw std::runtime_error(arch + ": rope.dimension_sections other than [head_dim/2, 0, "
                                            "0, 0] is not mapped");
        }
    }
    // The published graph is bidirectional inside the block (is_causal false).
    if (const auto* causal = m.meta("attention.causal"); causal && flag(*causal)) {
        throw std::runtime_error(arch + ": a causal DFlash2 drafter is not mapped");
    }
    const auto kernel = m.count("conv_kernel_size");
    const auto group  = m.count("conv_group_size");
    const auto rank   = m.count("selector_rank");
    if (!kernel || !group || hidden % group || !rank || !m.count("selector_top_k")) {
        throw std::runtime_error(arch + ": invalid DFlash2 convolution or selector geometry");
    }
    OJson draft = OJson{{"target_layer_ids", std::move(g.taps)},
                        {"mask_token_id", g.mask},
                        {"block_size", m.count("block_size")},
                        {"conv_kernel_size", kernel},
                        {"conv_group_size", group},
                        {"selector_rank", rank},
                        {"selector_top_k", m.count("selector_top_k")}};
    // What DFlashModel writes only when the source set it; the converter's DFlash2 config check
    // refuses anything but the identity (output_multiplier 1, no soft cap, embedding scale 1).
    if (const auto* scale = m.meta("logit_scale")) { draft["output_multiplier"] = *scale; }
    if (const auto* cap = m.meta("final_logit_softcapping")) {
        draft["final_logit_softcapping"] = *cap;
    }
    if (const auto* scale = m.meta("embedding_scale")) { draft["input_embedding_scale"] = *scale; }

    OJson config                      = OJson::object();
    config["architectures"]           = OJson::array({"DFlash2DraftModel"});
    config["model_type"]              = "qwen3";
    config["hidden_size"]             = hidden;
    config["vocab_size"]              = g.vocab;
    config["num_hidden_layers"]       = layers;
    config["num_attention_heads"]     = heads;
    config["num_key_value_heads"]     = kv;
    config["head_dim"]                = dim;
    config["intermediate_size"]       = ff;
    config["max_position_embeddings"] = m.count("context_length");
    config["rms_norm_eps"]            = m.real("attention.layer_norm_rms_epsilon");
    config["hidden_act"]              = "silu";
    config["attention_bias"]          = false;
    config["is_causal"]               = false;
    config["layer_types"]             = std::move(g.kinds);
    config["use_sliding_window"]      = g.window != 0;
    if (g.window) { config["sliding_window"] = g.window; }
    config["rope_parameters"] =
        OJson{{"rope_type", "default"}, {"rope_theta", m.real_or("rope.freq_base", 10000.0)}};
    config["dflash_config"] = std::move(draft);
    m.out.config            = std::move(config);

    const auto vector = [&](const std::string& from, const std::string& to, std::uint64_t n) {
        m.add(to, {n}, m.whole(m.need(from)));
    };
    const auto matrix = [&](const std::string& from, const std::string& to, std::uint64_t rows,
                            std::uint64_t cols) {
        const auto& stored = m.need(from);
        if (Mapper::rows_of(stored) != rows || stored.shape.back() != cols) {
            throw std::runtime_error(arch + ": " + from + " is not [" + std::to_string(rows) +
                                     ", " + std::to_string(cols) + "]");
        }
        m.add(to, {rows, cols}, m.whole(stored));
    };
    const auto vocab = static_cast<std::uint64_t>(g.vocab);
    matrix("fc.weight", "fc.weight", hidden, g.targets * hidden);
    vector("enc.output_norm.weight", "hidden_norm.weight", hidden);
    vector("output_norm.weight", "norm.weight", hidden);
    matrix("selector_hidden.weight", "candidate_selector.hidden_projection.weight", rank, hidden);
    matrix("selector_predecessor.weight", "candidate_selector.predecessor_codebook", vocab, rank);
    matrix("selector_successor.weight", "candidate_selector.successor_codebook", vocab, rank);
    const auto conv_rows = 2 * kernel * (hidden / group);
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto s = "blk." + std::to_string(i) + ".";
        const auto p = "layers." + std::to_string(i) + ".";
        vector(s + "attn_norm.weight", p + "input_layernorm.weight", hidden);
        vector(s + "ffn_norm.weight", p + "post_attention_layernorm.weight", hidden);
        vector(s + "attn_q_norm.weight", p + "self_attn.q_norm.weight", dim);
        vector(s + "attn_k_norm.weight", p + "self_attn.k_norm.weight", dim);
        matrix(s + "attn_q.weight", p + "self_attn.q_proj.weight", heads * dim, hidden);
        matrix(s + "attn_k.weight", p + "self_attn.k_proj.weight", kv * dim, hidden);
        matrix(s + "attn_v.weight", p + "self_attn.v_proj.weight", kv * dim, hidden);
        matrix(s + "attn_output.weight", p + "self_attn.o_proj.weight", hidden, heads * dim);
        matrix(s + "ffn_gate.weight", p + "mlp.gate_proj.weight", ff, hidden);
        matrix(s + "ffn_up.weight", p + "mlp.up_proj.weight", ff, hidden);
        matrix(s + "ffn_down.weight", p + "mlp.down_proj.weight", hidden, ff);
        for (const auto& [from, to] :
             {std::pair{"attn_conv", "attention_conv"}, std::pair{"ffn_conv", "mlp_conv"}}) {
            // ggml [hidden, kernel, 2] is transformers' [2, kernel, hidden] in the same order.
            const auto& base = m.need(s + from + "_base");
            if (base.ggml_type != ggml::F32 || base.elements() != 2 * kernel * hidden ||
                base.shape.back() != hidden) {
                throw std::runtime_error(arch + ": " + base.name + " must be F32 [2, " +
                                         std::to_string(kernel) + ", hidden]");
            }
            m.add(p + to + ".base_kernel", {2, kernel, hidden}, m.whole(base));
            matrix(s + from + "_proj.weight", p + to + ".kernel_projection.weight", conv_rows,
                   hidden);
        }
    }
    m.note("DFlash2 read back as DFlash2DraftModel: target_layers one past as target_layer_ids, "
           "mask token from tokenizer metadata, codebooks kept as published");
}

// llama.cpp's DFlash drafter (LLM_ARCH_DFLASH, written from Muse Glimmer's assistant): Qwen3-style
// layers stored as Hugging Face lays them out (plain norms, no Q/K permutation), no token table and
// no head (the target's). target_layers name the target layers whose inputs feed the context, one
// past the transformers target_layer_ids (the outputs recorded). llama.cpp's graph runs every
// layer non-causal: the block mask. A file with DFlash2's convolutions and candidate selector is
// read by map_dflash2.
void map_dflash(Mapper& m) {
    const std::string arch = m.arch();
    for (const char* key :
         {"conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"}) {
        if (m.meta(key)) {
            map_dflash2(m);
            return;
        }
    }
    DFlashGeometry g  = read_dflash_geometry(m);
    const auto hidden = g.hidden, layers = g.layers, heads = g.heads, kv = g.kv, dim = g.dim;
    OJson config                  = OJson::object();
    config["architectures"]       = OJson::array({"DFlashDraftModel"});
    config["model_type"]          = "dflash";
    config["hidden_size"]         = hidden;
    config["vocab_size"]          = g.vocab;
    config["num_hidden_layers"]   = layers;
    config["num_attention_heads"] = heads;
    config["num_key_value_heads"] = kv;
    config["head_dim"]            = dim;
    config["intermediate_size"]   = m.count("feed_forward_length");
    config["rms_norm_eps"]        = m.real("attention.layer_norm_rms_epsilon");
    config["hidden_act"]          = "silu";
    config["attention_bias"]      = false;
    config["layer_types"]         = std::move(g.kinds);
    config["sliding_window"]      = g.window ? g.window : m.count("context_length");
    config["rope_theta"]          = m.real_or("rope.freq_base", 10000.0);
    config["block_size"]          = m.count("block_size");
    config["dflash_config"] =
        OJson{{"target_layer_ids", std::move(g.taps)}, {"mask_token_id", g.mask}};
    m.out.config = std::move(config);

    const auto vector = [&](const std::string& from, const std::string& to, std::uint64_t n) {
        m.add(to, {n}, m.whole(m.need(from)));
    };
    const auto& fc = m.need("fc.weight");
    if (Mapper::rows_of(fc) != hidden) {
        throw std::runtime_error(arch + ": fc.weight must project to the hidden width");
    }
    // Under the names llama.cpp's converter read them from (MuseGlimmerAssistantModel's): the
    // GGUF's own fc.weight would shadow a view of the same name.
    m.add("encoder.fc.weight", {hidden, fc.shape.back()}, m.whole(fc));
    vector("enc.output_norm.weight", "encoder.output_norm_enc.weight", hidden);
    vector("output_norm.weight", "norm.weight", hidden);
    const auto ff = m.count("feed_forward_length");
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto s = "blk." + std::to_string(i) + ".";
        const auto p = "layers." + std::to_string(i) + ".";
        vector(s + "attn_norm.weight", p + "input_layernorm.weight", hidden);
        vector(s + "ffn_norm.weight", p + "post_attention_layernorm.weight", hidden);
        vector(s + "attn_q_norm.weight", p + "self_attn.q_norm.weight", dim);
        vector(s + "attn_k_norm.weight", p + "self_attn.k_norm.weight", dim);
        for (const auto& [from, to, n] :
             {std::tuple{"attn_q", "q_proj", heads}, std::tuple{"attn_k", "k_proj", kv},
              std::tuple{"attn_v", "v_proj", kv}}) {
            m.add(p + "self_attn." + to + ".weight", {n * dim, hidden},
                  m.whole(m.need(s + from + ".weight")));
        }
        m.add(p + "self_attn.o_proj.weight", {hidden, heads * dim},
              m.whole(m.need(s + "attn_output.weight")));
        m.add(p + "mlp.gate_proj.weight", {ff, hidden}, m.whole(m.need(s + "ffn_gate.weight")));
        m.add(p + "mlp.up_proj.weight", {ff, hidden}, m.whole(m.need(s + "ffn_up.weight")));
        m.add(p + "mlp.down_proj.weight", {hidden, ff}, m.whole(m.need(s + "ffn_down.weight")));
    }
    m.note("target_layers read one past as target_layer_ids; the block mask bidirectional on "
           "every layer (llama.cpp's DFlash graph)");
}

} // namespace


// llama.cpp's NemotronHModel (LLM_ARCH_NEMOTRON_H_MOE): read back into NemotronHForCausalLM's
// config and names. A layer is attention where attention.head_count_kv is nonzero, MoE where
// feed_forward_length is, Mamba2 otherwise. ssm_a is stored as -exp(A_log); ssm_norm as
// [groups, inner / groups]; the routed experts as one stacked bank per projection (expert e's
// rows [e * F, (e + 1) * F) of ffn_up_exps, [e * H, (e + 1) * H) of ffn_down_exps). NoPE: the
// Q/K row order does not change a score, so the attention matrices are read as stored.
// A trunk GGUF may carry the MTP (NextN) blocks too (unsloth's does: block_count 53 with the
// attention+MoE MTP block at blk.52 and nextn_predict_layers 1): block_count counts them after
// the trunk, and llama.cpp's graph stops before them, so the trunk is the first
// block_count - nextn_predict_layers blocks and the rest is not mapped.
void map_nemotron_h(Mapper& m) {
    const auto& embd    = m.need("token_embd.weight");
    const auto vocab    = embd.shape[0];
    const auto hidden   = m.count("embedding_length");
    const auto nextn    = m.count_or("nextn_predict_layers", 0);
    const auto layers   = trunk_blocks(m, "nemotron_h_moe");
    const auto heads    = m.count("attention.head_count");
    const auto dim      = m.count("attention.key_length");
    const auto experts  = m.count("expert_count");
    const auto used     = m.count("expert_used_count");
    const auto ff       = m.count("expert_feed_forward_length");
    const auto shared   = m.count("expert_shared_feed_forward_length");
    // A GGUF whose expert widths no 256-value block divides may zero-pad them so its K/IQ formats
    // apply (expert_feed_forward_length and the shared width then read padded, relu2(0) = 0); the
    // routed up bank's reduction axis is padded past the hidden width under this key.
    const auto up_k     = m.count_or("moe_expert_embedding_length_padded", hidden);
    const auto inner    = m.count("ssm.inner_size");
    const auto m_heads  = m.count("ssm.time_step_rank");
    const auto groups   = m.count("ssm.group_count");
    const auto state    = m.count("ssm.state_size");
    if (!heads || !dim || !experts || !used || !ff || !shared || up_k < hidden || !m_heads ||
        inner % m_heads || !groups || m_heads % groups || !state) {
        throw std::runtime_error("nemotron_h_moe: invalid geometry");
    }
    if (m.count_or("expert_shared_count", 1) != 1 || m.count_or("expert_group_count", 1) != 1) {
        throw std::runtime_error("nemotron_h_moe: one shared expert and one expert group only");
    }
    OJson kinds = OJson::array();
    std::uint64_t kv = 0;
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto layer_kv = m.per_layer("attention.head_count_kv", i);
        if (layer_kv != 0) {
            if (kv != 0 && kv != layer_kv) {
                throw std::runtime_error("nemotron_h_moe: attention layers differ in kv heads");
            }
            kv = layer_kv;
            kinds.push_back("attention");
        } else if (m.per_layer("feed_forward_length", i) != 0) {
            kinds.push_back("moe");
        } else {
            kinds.push_back("mamba");
        }
    }
    if (kv == 0) { throw std::runtime_error("nemotron_h_moe: no attention layer"); }
    const auto* norm_topk = m.meta("expert_weights_norm");
    OJson config                                  = OJson::object();
    config["architectures"]                       = OJson::array({"NemotronHForCausalLM"});
    config["model_type"]                          = "nemotron_h";
    config["hidden_size"]                         = hidden;
    config["vocab_size"]                          = vocab;
    config["num_hidden_layers"]                   = layers;
    config["layers_block_type"]                   = std::move(kinds);
    config["max_position_embeddings"]             = m.count("context_length");
    config["num_attention_heads"]                 = heads;
    config["num_key_value_heads"]                 = kv;
    config["head_dim"]                            = dim;
    config["layer_norm_epsilon"]                  = m.real("attention.layer_norm_rms_epsilon");
    config["mamba_num_heads"]                     = m_heads;
    config["mamba_head_dim"]                      = inner / m_heads;
    config["n_groups"]                            = groups;
    config["ssm_state_size"]                      = state;
    config["conv_kernel"]                         = m.count("ssm.conv_kernel");
    config["use_conv_bias"]                       = true;
    config["n_routed_experts"]                    = experts;
    config["num_experts_per_tok"]                 = used;
    config["moe_intermediate_size"]               = ff;
    config["moe_shared_expert_intermediate_size"] = shared;
    config["n_shared_experts"]                    = 1;
    if (up_k != hidden) { config["moe_expert_input_size"] = up_k; }
    config["norm_topk_prob"]         = norm_topk == nullptr || flag(*norm_topk);
    config["routed_scaling_factor"]  = m.real_or("expert_weights_scale", 1.0);
    config["mlp_hidden_act"]         = "relu2";
    config["tie_word_embeddings"]    = m.find("output.weight") == nullptr;
    m.out.config = std::move(config);
    // 2: the trunk leaves out the MTP blocks block_count counts (version 1 read them as layers).
    m.out.mapper = "nemotron_h_moe/2";
    m.note("layer kinds from attention.head_count_kv and feed_forward_length per layer");
    m.note("time_step_limit (0, inf) is not stored: no dt clamp, as llama.cpp's ssm_scan");
    if (nextn != 0) {
        m.note("the " + std::to_string(nextn) + " MTP block(s) after the " +
               std::to_string(layers) + " trunk blocks are not mapped (not a trunk layer)");
    }

    const std::string root = "backbone.";
    m.add(root + "embedding.weight", {vocab, hidden}, m.whole(embd));
    if (const auto* output = m.find("output.weight")) {
        m.add("lm_head.weight", {vocab, hidden}, m.whole(*output));
    }
    m.add(root + "norm_f.weight", {hidden}, m.whole(m.need("output_norm.weight")));
    const auto conv = inner + 2 * groups * state;
    for (std::uint64_t i = 0; i < layers; ++i) {
        const auto s    = "blk." + std::to_string(i) + ".";
        const auto l    = root + "layers." + std::to_string(i) + ".";
        const auto x    = l + "mixer.";
        const auto kind = m.out.config["layers_block_type"][i].get<std::string>();
        m.add(l + "norm.weight", {hidden}, m.whole(m.need(s + "attn_norm.weight")));
        if (kind == "mamba") {
            m.add(x + "in_proj.weight", {inner + conv + m_heads, hidden},
                  m.whole(m.need(s + "ssm_in.weight")));
            m.add(x + "conv1d.weight", {conv, 1, 4}, m.whole(m.need(s + "ssm_conv1d.weight")));
            m.add(x + "conv1d.bias", {conv}, m.whole(m.need(s + "ssm_conv1d.bias")));
            auto a      = m.whole(m.need(s + "ssm_a"));
            a.transform = Transform::NegExpLog;
            a.note      = "-exp(A_log) stored";
            m.add(x + "A_log", {m_heads}, std::move(a));
            m.add(x + "D", {m_heads}, m.whole(m.need(s + "ssm_d")));
            m.add(x + "dt_bias", {m_heads}, m.whole(m.need(s + "ssm_dt.bias")));
            m.add(x + "norm.weight", {inner}, m.whole(m.need(s + "ssm_norm.weight")));
            m.add(x + "out_proj.weight", {hidden, inner}, m.whole(m.need(s + "ssm_out.weight")));
        } else if (kind == "attention") {
            m.add(x + "q_proj.weight", {heads * dim, hidden}, m.whole(m.need(s + "attn_q.weight")));
            m.add(x + "k_proj.weight", {kv * dim, hidden}, m.whole(m.need(s + "attn_k.weight")));
            m.add(x + "v_proj.weight", {kv * dim, hidden}, m.whole(m.need(s + "attn_v.weight")));
            m.add(x + "o_proj.weight", {hidden, heads * dim},
                  m.whole(m.need(s + "attn_output.weight")));
        } else {
            m.add(x + "gate.weight", {experts, hidden}, m.whole(m.need(s + "ffn_gate_inp.weight")));
            m.add(x + "gate.e_score_correction_bias", {experts},
                  m.whole(m.need(s + "exp_probs_b.bias")));
            const auto& up   = m.need(s + "ffn_up_exps.weight");
            const auto& down = m.need(s + "ffn_down_exps.weight");
            if (Mapper::rows_of(up) != experts * ff || Mapper::rows_of(down) != experts * hidden) {
                throw std::runtime_error("nemotron_h_moe: " + up.name + " / " + down.name +
                                         " are not stacked expert banks");
            }
            for (std::uint64_t e = 0; e < experts; ++e) {
                const auto ep = x + "experts." + std::to_string(e) + ".";
                m.add(ep + "up_proj.weight", {ff, up_k}, m.slice(up, e * ff, ff));
                m.add(ep + "down_proj.weight", {hidden, ff}, m.slice(down, e * hidden, hidden));
            }
            m.add(x + "shared_experts.up_proj.weight", {shared, hidden},
                  m.whole(m.need(s + "ffn_up_shexp.weight")));
            m.add(x + "shared_experts.down_proj.weight", {hidden, shared},
                  m.whole(m.need(s + "ffn_down_shexp.weight")));
        }
    }
    m.note("expert banks split per expert from ffn_up_exps / ffn_down_exps");
}

// Whether a nemotron_h_moe GGUF is only the MTP head: llama.cpp folds the one NextN block at
// index block_count - 1 and marks the trunk tensors not required, so a sidecar carries
// nextn_predict_layers and no blk.0.attn_norm.weight.
bool nemotron_h_mtp_only(Mapper& m) {
    return m.count_or("nextn_predict_layers", 0) != 0 &&
           m.find("blk.0.attn_norm.weight") == nullptr;
}

// ---------------------------------------------------------------------------------------------
// nemotron-h-mtp: the MTP (NextN) drafter of a NemotronH MoE target, from llama.cpp's mtp-*.gguf
// sidecar (e.g. mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf) or from a target GGUF that
// keeps its NextN block (GgufRequest::Mtp): one folded block at index block_count - 1 of a NoPE
// attention sub-layer and a relu^2 MoE sub-layer over the fusion of the next token's embedding
// and the previous hidden (nextn.enorm / hnorm / eh_proj), the pre-MoE norm and the final
// LayerNorm weight (nextn.shared_head_norm). token_embd / output / output_norm (and a target
// file's trunk blocks) are the target's and are not mapped. A GGUF that zero-pads the routed up
// bank's K (moe_expert_embedding_length_padded, as map_nemotron_h reads it) pads the NextN block's
// bank too. Mapped to the HF mtp.* names the checkpoint's converter wrote it from.

void map_nemotron_h_mtp(Mapper& m) {
    const auto& embd   = m.need("token_embd.weight");
    const auto vocab   = embd.shape[0];
    const auto hidden  = m.count("embedding_length");
    const auto layers  = m.count("block_count");
    const auto heads   = m.count("attention.head_count");
    const auto dim     = m.count_or("attention.key_length", 0);
    const auto experts = m.count("expert_count");
    const auto used    = m.count("expert_used_count");
    const auto ff      = m.count("expert_feed_forward_length");
    const auto shared  = m.count("expert_shared_feed_forward_length");
    const auto inner   = m.count_or("ssm.inner_size", 0);
    const auto m_heads = m.count_or("ssm.time_step_rank", 0);
    const auto up_k    = m.count_or("moe_expert_embedding_length_padded", hidden);
    const bool sidecar = nemotron_h_mtp_only(m);
    const auto nextn   = m.count_or("nextn_predict_layers", 0);
    if (nextn == 0 ||
        m.find("blk." + std::to_string(layers - 1) + ".nextn.eh_proj.weight") == nullptr) {
        throw std::runtime_error("nemotron_h_moe: the GGUF keeps no NextN block to draft with");
    }
    if (nextn != 1) {
        throw std::runtime_error("nemotron_h_moe: the MTP head is one nextn_predict_layers block");
    }
    // The block both attends and routes. A padded GGUF (pirola's IQ3_XXS: 1856 -> 2048) widens
    // expert_feed_forward_length and keeps the per-layer width it was trained with, so the layer's
    // own width is nonzero and at most the expert width, not equal to it.
    const auto kv       = m.per_layer("attention.head_count_kv", layers - 1);
    const auto layer_ff = m.per_layer("feed_forward_length", layers - 1);
    if (!kv || !ff || !layer_ff || layer_ff > ff) {
        throw std::runtime_error("nemotron_h_moe: block " + std::to_string(layers - 1) +
                                 " is not the attention+MoE MTP block");
    }
    if (!heads || !dim || heads % kv || !experts || !used || !ff || !shared || !m_heads ||
        inner % m_heads || up_k < hidden) {
        throw std::runtime_error("nemotron_h_moe: invalid geometry");
    }
    if (m.count_or("expert_shared_count", 1) != 1 || m.count_or("expert_group_count", 1) != 1) {
        throw std::runtime_error("nemotron_h_moe: one shared expert and one expert group only");
    }
    if (sidecar && m.find("blk." + std::to_string(layers - 2) + ".attn_norm.weight")) {
        throw std::runtime_error("nemotron_h_moe: trunk tensors beside the MTP block");
    }

    OJson config                            = OJson::object();
    config["architectures"]                 = OJson::array({"NemotronHMtpForCausalLM"});
    config["model_type"]                    = "nemotron_h_mtp";
    config["hidden_size"]                   = hidden;
    config["vocab_size"]                    = vocab;
    config["num_hidden_layers"]             = 1;
    config["max_position_embeddings"]       = m.count("context_length");
    config["rms_norm_eps"]                  = m.real("attention.layer_norm_rms_epsilon");
    config["layer_norm_epsilon"]            = m.real_or("attention.layer_norm_epsilon",
                                             config["rms_norm_eps"].get<double>());
    config["num_attention_heads"]           = heads;
    config["num_key_value_heads"]           = kv;
    config["head_dim"]                      = dim;
    config["n_routed_experts"]              = experts;
    config["num_experts_per_tok"]           = used;
    config["moe_intermediate_size"]         = ff;
    config["moe_shared_expert_intermediate_size"] = shared;
    config["n_shared_experts"]              = 1;
    const auto* norm_topk                   = m.meta("expert_weights_norm");
    config["norm_topk_prob"]                = norm_topk == nullptr || flag(*norm_topk);
    config["routed_scaling_factor"]         = m.real_or("expert_weights_scale", 1.0);
    config["mlp_hidden_act"]                = "relu2";
    config["tie_word_embeddings"]           = false;
    if (up_k != hidden) { config["moe_expert_input_size"] = up_k; }
    m.out.config                            = std::move(config);
    m.note("layer kinds from attention.head_count_kv and feed_forward_length per layer");
    m.note(sidecar ? "the sidecar's token_embd / output / output_norm are the target's; not mapped"
                   : "the target GGUF's NextN block; its trunk blocks and head are the target's, "
                     "not mapped");

    const std::string s  = "blk." + std::to_string(layers - 1) + ".";
    const std::string x0 = "mtp.layers.0.mixer.";
    const std::string x1 = "mtp.layers.1.mixer.";
    m.add("mtp.layers.0.enorm.weight", {hidden},
          m.whole(m.need(s + "nextn.enorm.weight")));
    m.add("mtp.layers.0.hnorm.weight", {hidden},
          m.whole(m.need(s + "nextn.hnorm.weight")));
    // eh_proj reads [norm(e); norm(h)]; the GGUF row order is that concatenation, kept as stored.
    m.add(x0 + "eh_proj.weight", {hidden, 2 * hidden}, m.whole(m.need(s + "nextn.eh_proj.weight")));
    m.add("mtp.layers.0.norm.weight", {hidden}, m.whole(m.need(s + "attn_norm.weight")));
    m.add(x0 + "q_proj.weight", {heads * dim, hidden}, m.whole(m.need(s + "attn_q.weight")));
    m.add(x0 + "k_proj.weight", {kv * dim, hidden}, m.whole(m.need(s + "attn_k.weight")));
    m.add(x0 + "v_proj.weight", {kv * dim, hidden}, m.whole(m.need(s + "attn_v.weight")));
    m.add(x0 + "o_proj.weight", {hidden, heads * dim}, m.whole(m.need(s + "attn_output.weight")));
    m.add("mtp.layers.1.norm.weight", {hidden},
          m.whole(m.need(s + "post_attention_norm.weight")));
    m.add(x1 + "gate.weight", {experts, hidden}, m.whole(m.need(s + "ffn_gate_inp.weight")));
    m.add(x1 + "gate.e_score_correction_bias", {experts},
          m.whole(m.need(s + "exp_probs_b.bias")));
    const auto& up   = m.need(s + "ffn_up_exps.weight");
    const auto& down = m.need(s + "ffn_down_exps.weight");
    if (Mapper::rows_of(up) != experts * ff || Mapper::rows_of(down) != experts * hidden) {
        throw std::runtime_error("nemotron_h_moe: " + up.name + " / " + down.name +
                                 " are not stacked expert banks");
    }
    for (std::uint64_t e = 0; e < experts; ++e) {
        const auto ep = x1 + "experts." + std::to_string(e) + ".";
        m.add(ep + "up_proj.weight", {ff, up_k}, m.slice(up, e * ff, ff));
        m.add(ep + "down_proj.weight", {hidden, ff}, m.slice(down, e * hidden, hidden));
    }
    m.add(x1 + "shared_experts.up_proj.weight", {shared, hidden},
          m.whole(m.need(s + "ffn_up_shexp.weight")));
    m.add(x1 + "shared_experts.down_proj.weight", {hidden, shared},
          m.whole(m.need(s + "ffn_down_shexp.weight")));
    m.add("mtp.layers.1.final_layernorm.weight", {hidden},
          m.whole(m.need(s + "nextn.shared_head_norm.weight")));
    m.note("expert banks split per expert from ffn_up_exps / ffn_down_exps");
}

// ---------------------------------------------------------------------------------------------
// clip: a Gemma 4 vision projector (mmproj-*.gguf, projector type gemma4v): every tensor is
// transformers' under a llama.cpp name, the patch kernel stored as a [c, y, x] convolution.

void map_clip_gemma4(Mapper& m) {
    const auto h     = m.count("vision.embedding_length");
    const auto ff    = m.count("vision.feed_forward_length");
    const auto depth = m.count("vision.block_count");
    const auto heads = m.count("vision.attention.head_count");
    const auto ps    = m.count("vision.patch_size");
    const auto out   = m.count("vision.projection_dim");
    const auto* eps_value = m.meta("vision.attention.layer_norm_epsilon");
    const double eps = eps_value && eps_value->is_number() ? eps_value->get<double>() : 1.0e-6;
    if (h % heads != 0 || (h / heads) % 4 != 0) {
        throw std::runtime_error("clip: gemma4v head geometry is not an axial 2-D rope's");
    }
    const auto& pos = m.need("v.position_embd.weight");
    if (pos.elements() % (2 * h) != 0) {
        throw std::runtime_error("clip: v.position_embd.weight is not [2, table, hidden]");
    }
    const auto table = pos.elements() / (2 * h);

    // llama.cpp keeps the patch embedding as a convolution kernel [h, c, y, x]; transformers'
    // input_proj reads each patch as (y, x, c). One stored row per output feature, its columns
    // gathered in that order.
    const auto& patch = m.need("v.patch_embd.weight");
    const auto width  = 3 * ps * ps;
    if (patch.elements() != h * width) {
        throw std::runtime_error("clip: v.patch_embd.weight is not [" + std::to_string(h) +
                                 ", 3, " + std::to_string(ps) + ", " + std::to_string(ps) + "]");
    }
    std::uint64_t value_bytes = 0;
    if (patch.ggml_type == ggml::F32) { value_bytes = 4; }
    if (patch.ggml_type == ggml::F16 || patch.ggml_type == ggml::BF16) { value_bytes = 2; }
    if (value_bytes == 0) {
        throw std::runtime_error("clip: v.patch_embd.weight must be F32, F16 or BF16");
    }
    Spec embed;
    embed.parts = {m.part(patch)};
    embed.parts[0].cols      = width;
    embed.parts[0].row_bytes = width * value_bytes;
    embed.cols               = width;
    embed.chunk              = 1;
    for (std::uint64_t o = 0; o < h; ++o) { embed.rows.emplace_back(0, o); }
    for (std::uint64_t y = 0; y < ps; ++y) {
        for (std::uint64_t x = 0; x < ps; ++x) {
            for (std::uint64_t c = 0; c < 3; ++c) {
                embed.chunk_of.push_back((c * ps + y) * ps + x);
            }
        }
    }
    embed.note = "convolution kernel [c, y, x] read as (y, x, c)";
    const std::string t = "model.vision_tower.";
    m.add(t + "patch_embedder.input_proj.weight", {h, width}, std::move(embed));
    m.add(t + "patch_embedder.position_embedding_table", {2, table, h}, m.whole(pos));
    m.add(t + "std_bias", {h}, m.whole(m.need("v.std_bias")));
    m.add(t + "std_scale", {h}, m.whole(m.need("v.std_scale")));
    for (std::uint64_t i = 0; i < depth; ++i) {
        const auto s  = "v.blk." + std::to_string(i) + ".";
        const auto hf = t + "encoder.layers." + std::to_string(i) + ".";
        for (const auto& [from, to] :
             {std::pair{"ln1", "input_layernorm"}, std::pair{"attn_post_norm", "post_attention_layernorm"},
              std::pair{"ln2", "pre_feedforward_layernorm"},
              std::pair{"ffn_post_norm", "post_feedforward_layernorm"}}) {
            m.add(hf + to + ".weight", {h}, m.whole(m.need(s + from + ".weight")));
        }
        for (const auto& [from, to] : {std::pair{"attn_q", "q_proj"}, std::pair{"attn_k", "k_proj"},
                                       std::pair{"attn_v", "v_proj"}, std::pair{"attn_out", "o_proj"}}) {
            m.add(hf + "self_attn." + to + ".linear.weight", {h, h},
                  m.whole(m.need(s + from + ".weight")));
        }
        m.add(hf + "self_attn.q_norm.weight", {h / heads}, m.whole(m.need(s + "attn_q_norm.weight")));
        m.add(hf + "self_attn.k_norm.weight", {h / heads}, m.whole(m.need(s + "attn_k_norm.weight")));
        m.add(hf + "mlp.gate_proj.linear.weight", {ff, h}, m.whole(m.need(s + "ffn_gate.weight")));
        m.add(hf + "mlp.up_proj.linear.weight", {ff, h}, m.whole(m.need(s + "ffn_up.weight")));
        m.add(hf + "mlp.down_proj.linear.weight", {h, ff}, m.whole(m.need(s + "ffn_down.weight")));
    }
    m.add("model.embed_vision.embedding_projection.weight", {out, h},
          m.whole(m.need("mm.input_projection.weight")));
    m.note("vision tensors renamed v.* / mm.* -> model.vision_tower.* / model.embed_vision.*");

    OJson vision                     = OJson::object();
    vision["model_type"]             = "gemma4_vision";
    vision["hidden_activation"]      = "gelu_pytorch_tanh";
    vision["attention_bias"]         = false;
    vision["standardize"]            = true;
    vision["use_clipped_linears"]    = false;
    vision["hidden_size"]            = h;
    vision["num_hidden_layers"]      = depth;
    vision["num_attention_heads"]    = heads;
    vision["num_key_value_heads"]    = heads;
    vision["head_dim"]               = h / heads;
    vision["intermediate_size"]      = ff;
    vision["patch_size"]             = ps;
    vision["pooling_kernel_size"]    = 3; // llama.cpp's gemma4v pools 3 x 3 (clip.cpp)
    vision["position_embedding_size"] = table;
    vision["default_output_length"]  = 280;
    vision["rope_parameters"]        = OJson{{"rope_type", "axial"}, {"rope_theta", 100.0}};
    vision["rms_norm_eps"]           = eps;
    OJson config                     = OJson::object();
    config["model_type"]             = "gemma4_vision_projector";
    config["vision_config"]          = std::move(vision);
    m.out.config                     = std::move(config);
}

// ---------------------------------------------------------------------------------------------
// clip: a Qwen3.5 / 3.6 / 3.8 vision projector (mmproj-*.gguf, projector type qwen3vl_merger)

void map_clip(Mapper& m) {
    // Older llama.cpp files name it clip.projector_type; since the audio encoders, the vision
    // projector is clip.vision.projector_type (Gemma 4's mmproj has only that key).
    const auto* projector = m.file().find("clip.vision.projector_type");
    if (!projector) { projector = m.file().find("clip.projector_type"); }
    const auto type = projector && projector->is_string() ? projector->get<std::string>() : "";
    if (type == "gemma4v") {
        map_clip_gemma4(m);
        return;
    }
    if (type != "qwen3vl_merger") {
        throw std::runtime_error("clip: projector type '" + type +
                                 "' has no converter mapping; supported: qwen3vl_merger (the "
                                 "Qwen3.5 / 3.6 / 3.8 vision projector), gemma4v (Gemma 4's)");
    }
    if (const auto* deep = m.meta("vision.is_deepstack_layers"); deep && deep->is_array()) {
        for (const auto& layer : *deep) {
            if (flag(layer)) { throw std::runtime_error("clip: deepstack layers are not supported"); }
        }
    }
    for (const char* key : {"vision.image_mean", "vision.image_std"}) {
        const auto* values = m.meta(key);
        if (!values || !values->is_array() || values->size() != 3 ||
            std::any_of(values->begin(), values->end(),
                        [](const Json& v) { return !v.is_number() || v.get<double>() != 0.5; })) {
            throw std::runtime_error(std::string("clip: ") + key +
                                     " must be [0.5, 0.5, 0.5], the Qwen vision encoder's");
        }
    }
    const auto h      = m.count("vision.embedding_length");
    const auto ff     = m.count("vision.feed_forward_length");
    const auto depth  = m.count("vision.block_count");
    const auto heads  = m.count("vision.attention.head_count");
    const auto ps     = m.count("vision.patch_size");
    const auto merge  = m.count("vision.spatial_merge_size");
    const auto out    = m.count("vision.projection_dim");
    const auto& pos   = m.need("v.position_embd.weight");
    const auto merger = merge * merge * h;

    // llama.cpp splits the Conv3D patch kernel [h, 3, 2, ps, ps] along time into two Conv2D
    // kernels (.weight: t = 0, .weight.1: t = 1): rows of ps values interleave them again.
    const auto& t0 = m.need("v.patch_embd.weight");
    const auto* t1 = m.find("v.patch_embd.weight.1");
    if (!t1) {
        throw std::runtime_error("clip: v.patch_embd.weight.1 is absent (a temporal patch of 2 "
                                 "frames is what the Qwen encoder reads)");
    }
    if (t0.elements() != h * 3 * ps * ps || t1->elements() != t0.elements() ||
        t0.shape.back() != ps) {
        throw std::runtime_error("clip: v.patch_embd.weight is not [" + std::to_string(h) +
                                 ", 3, " + std::to_string(ps) + ", " + std::to_string(ps) + "]");
    }
    Spec patch;
    patch.parts = {m.part(t0), m.part(*t1)};
    patch.cols  = ps;
    for (std::uint64_t o = 0; o < h; ++o) {
        for (std::uint64_t c = 0; c < 3; ++c) {
            for (std::uint32_t t = 0; t < 2; ++t) {
                for (std::uint64_t y = 0; y < ps; ++y) {
                    patch.rows.emplace_back(t, (o * 3 + c) * ps + y);
                }
            }
        }
    }
    patch.note = "temporal halves interleaved";
    m.add("model.visual.patch_embed.proj.weight", {h, 3, 2, ps, ps}, std::move(patch));
    m.note("patch embedding rebuilt from its two temporal Conv2D kernels");
    m.add("model.visual.patch_embed.proj.bias", {h}, m.whole(m.need("v.patch_embd.bias")));
    m.add("model.visual.pos_embed.weight", {Mapper::rows_of(pos), h}, m.whole(pos));
    for (std::uint64_t i = 0; i < depth; ++i) {
        const auto s  = "v.blk." + std::to_string(i) + ".";
        const auto hf = "model.visual.blocks." + std::to_string(i) + ".";
        for (const auto& [from, to] : {std::pair{"ln1", "norm1"}, std::pair{"ln2", "norm2"}}) {
            m.add(hf + to + ".weight", {h}, m.whole(m.need(s + from + ".weight")));
            m.add(hf + to + ".bias", {h}, m.whole(m.need(s + from + ".bias")));
        }
        m.add(hf + "attn.qkv.weight", {3 * h, h}, m.whole(m.need(s + "attn_qkv.weight")));
        m.add(hf + "attn.qkv.bias", {3 * h}, m.whole(m.need(s + "attn_qkv.bias")));
        m.add(hf + "attn.proj.weight", {h, h}, m.whole(m.need(s + "attn_out.weight")));
        m.add(hf + "attn.proj.bias", {h}, m.whole(m.need(s + "attn_out.bias")));
        m.add(hf + "mlp.linear_fc1.weight", {ff, h}, m.whole(m.need(s + "ffn_up.weight")));
        m.add(hf + "mlp.linear_fc1.bias", {ff}, m.whole(m.need(s + "ffn_up.bias")));
        m.add(hf + "mlp.linear_fc2.weight", {h, ff}, m.whole(m.need(s + "ffn_down.weight")));
        m.add(hf + "mlp.linear_fc2.bias", {h}, m.whole(m.need(s + "ffn_down.bias")));
    }
    m.add("model.visual.merger.norm.weight", {h}, m.whole(m.need("v.post_ln.weight")));
    m.add("model.visual.merger.norm.bias", {h}, m.whole(m.need("v.post_ln.bias")));
    m.add("model.visual.merger.linear_fc1.weight", {merger, merger}, m.whole(m.need("mm.0.weight")));
    m.add("model.visual.merger.linear_fc1.bias", {merger}, m.whole(m.need("mm.0.bias")));
    m.add("model.visual.merger.linear_fc2.weight", {out, merger}, m.whole(m.need("mm.2.weight")));
    m.add("model.visual.merger.linear_fc2.bias", {out}, m.whole(m.need("mm.2.bias")));
    m.note("vision tensors renamed v.* / mm.* -> model.visual.*");

    OJson vision                       = OJson::object();
    vision["in_channels"]              = 3;
    vision["hidden_act"]               = "gelu_pytorch_tanh";
    vision["deepstack_visual_indexes"] = OJson::array();
    vision["out_hidden_size"]          = out;
    vision["depth"]                    = depth;
    vision["hidden_size"]              = h;
    vision["intermediate_size"]        = ff;
    vision["num_heads"]                = heads;
    vision["patch_size"]               = ps;
    vision["temporal_patch_size"]      = 2;
    vision["spatial_merge_size"]       = merge;
    vision["num_position_embeddings"]  = Mapper::rows_of(pos);
    OJson config                       = OJson::object();
    config["model_type"]               = "qwen3_vl_projector";
    config["vision_config"]            = std::move(vision);
    m.out.config                       = std::move(config);
}

const std::vector<std::string>& supported_gguf_architectures() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> out = {"qwen35", "qwen35moe", "gemma4", "gpt-oss",
                                        "gemma4-assistant", "dflash", "nemotron_h_moe",
                                        "clip"};
        for (const auto& arch : kDenseArchs) { out.emplace_back(arch.gguf); }
        return out;
    }();
    return names;
}

GgufMapping map_gguf_model(const GgufFile& file, const StoredLookup& stored, const StoredRead& read,
                           const std::string& label, GgufRequest request) {
    Mapper m(file, stored, read, label);
    if (m.arch() == "qwen35" || m.arch() == "qwen35moe") {
        map_qwen35(m);
    } else if (m.arch() == "gemma4") {
        map_gemma4(m);
    } else if (m.arch() == "gpt-oss") {
        map_gpt_oss(m);
    } else if (m.arch() == "gemma4-assistant") {
        map_gemma4_assistant(m);
    } else if (m.arch() == "dflash") {
        map_dflash(m);
    } else if (m.arch() == "clip") {
        map_clip(m);
    } else if (m.arch() == "nemotron_h_moe") {
        // llama.cpp writes the sidecar under the same architecture string as the trunk: the
        // nextn_predict_layers count and the absent trunk block tell the two apart. A trunk file
        // asked for its MTP head alone reads its NextN block the same way.
        if (nemotron_h_mtp_only(m) || request == GgufRequest::Mtp) {
            map_nemotron_h_mtp(m);
        } else {
            map_nemotron_h(m);
        }
    } else if (const auto* dense = dense_arch(m.arch())) {
        map_dense(m, *dense);
    } else {
        std::string names;
        for (const auto& name : supported_gguf_architectures()) {
            names += (names.empty() ? "" : ", ") + name;
        }
        throw std::runtime_error(label + ": GGUF architecture '" + m.arch() +
                                 "' has no converter mapping; supported architectures: " + names +
                                 " (GGUF)");
    }
    return m.finish();
}

} // namespace ninferno::convert
