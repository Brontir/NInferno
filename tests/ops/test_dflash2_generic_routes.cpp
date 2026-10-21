// The generic routes a DFlash2 drafter published outside the fused Q8/BF16 profile runs on (the
// dflash GGUF's Q4_0 matrices, a K-quant or NVFP4 target head):
//  - linear_topk over a head no fused kernel owns: the stable top sixteen of Linear's BF16 logits;
//  - rmsnorm_dynamic_grouped_conv_prepare / linear_dynamic_grouped_conv_add with Q4_0
//    projections, against the fused BF16 / Q8 kernels given the same represented weights;
//  - the DFlash (v1) draft's Q/K/V of a drafter outside the fused Q8 profile (a BF16 artifact,
//    a published GGUF's Q4_0): segmented_projection over its bank, against the fused Q8
//    attn_input_proj given the same represented weights (draft.cpp, fused_q8).
#include "ninferno/ops/attn_input_proj.h"
#include "ninferno/ops/dynamic_grouped_conv.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_topk.h"
#include "ninferno/ops/segmented_projection.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <utility>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;

namespace {

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// Q4_0 values (q - 8) * d with d = 2^-6: every value is exact in BF16 and in Q8_G32 (code q - 8,
// scale d), so the three encodings below hold one represented matrix.
struct Matrix {
    std::int32_t n = 0, k = 0;
    std::vector<std::int8_t> codes; // q - 8, row-major [n, k]
    static constexpr std::uint16_t kScaleF16 = 0x2400; // 2^-6
    static constexpr float kScale            = 1.0F / 64.0F;

    float value(std::int32_t row, std::int32_t col) const {
        return static_cast<float>(codes[static_cast<std::size_t>(row) * k + col]) * kScale;
    }
};

Matrix make_matrix(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    Matrix m;
    m.n = n;
    m.k = k;
    m.codes.resize(static_cast<std::size_t>(n) * k);
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> d(-8, 7);
    for (auto& c : m.codes) c = static_cast<std::int8_t>(d(g));
    return m;
}

Weight base_weight(const Matrix& m, void* data, std::uint64_t bytes) {
    Weight w;
    w.payload       = data;
    w.payload_bytes = bytes;
    w.ndim          = 2;
    w.n = w.shape[0] = w.padded_shape[0] = m.n;
    w.k = w.shape[1] = w.padded_shape[1] = m.k;
    w.qdata                              = data;
    return w;
}

// ggml Q4_0 planes: per 32-value block 16 qs bytes (value j low nibble of qs[j], j + 16 high),
// then one binary16 d per block.
DeviceBuffer q4_0_payload(const Matrix& m, std::size_t& scale_offset) {
    const std::size_t blocks = static_cast<std::size_t>(m.n) * (m.k / 32);
    scale_offset             = align256(blocks * 16);
    std::vector<std::uint8_t> bytes(scale_offset + blocks * 2, 0);
    for (std::int32_t r = 0; r < m.n; ++r) {
        for (std::int32_t c = 0; c < m.k; ++c) {
            const std::size_t block = static_cast<std::size_t>(r) * (m.k / 32) + c / 32;
            const int v             = c % 32;
            const auto q = static_cast<std::uint8_t>(m.codes[static_cast<std::size_t>(r) * m.k + c] + 8);
            bytes[block * 16 + (v & 15)] |= v < 16 ? q : static_cast<std::uint8_t>(q << 4);
        }
    }
    for (std::size_t b = 0; b < blocks; ++b) {
        bytes[scale_offset + 2 * b]     = Matrix::kScaleF16 & 0xff;
        bytes[scale_offset + 2 * b + 1] = Matrix::kScaleF16 >> 8;
    }
    return to_device(bytes);
}

Weight q4_0_weight(const Matrix& m, const DeviceBuffer& payload, std::size_t scale_offset) {
    Weight w       = base_weight(m, payload.p, payload.bytes);
    w.qtype        = QType::GGML_Q4_0;
    w.layout       = QuantLayout::GgmlPlanes;
    w.scales       = static_cast<std::uint8_t*>(payload.p) + scale_offset;
    w.scale_dtype  = DType::FP16;
    w.group_size   = 32;
    w.group        = 32;
    return w;
}

DeviceBuffer q8_payload(const Matrix& m, std::size_t& scale_offset) {
    const std::size_t codes = static_cast<std::size_t>(m.n) * m.k;
    scale_offset            = align256(codes);
    std::vector<std::uint8_t> bytes(scale_offset + codes / 32 * 2, 0);
    for (std::size_t i = 0; i < codes; ++i) bytes[i] = static_cast<std::uint8_t>(m.codes[i]);
    for (std::size_t g = 0; g < codes / 32; ++g) {
        bytes[scale_offset + 2 * g]     = Matrix::kScaleF16 & 0xff;
        bytes[scale_offset + 2 * g + 1] = Matrix::kScaleF16 >> 8;
    }
    return to_device(bytes);
}

Weight q8_weight(const Matrix& m, const DeviceBuffer& payload, std::size_t scale_offset) {
    Weight w      = base_weight(m, payload.p, payload.bytes);
    w.qtype       = QType::Q8_G32_FP16;
    w.layout      = QuantLayout::RowSplit;
    w.scales      = static_cast<std::uint8_t*>(payload.p) + scale_offset;
    w.scale_dtype = DType::FP16;
    w.group_size  = 32;
    w.group       = 32;
    return w;
}

DeviceBuffer bf16_payload(const Matrix& m) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(m.n) * m.k);
    for (std::int32_t r = 0; r < m.n; ++r)
        for (std::int32_t c = 0; c < m.k; ++c)
            bits[static_cast<std::size_t>(r) * m.k + c] = f32_to_bf16(m.value(r, c));
    return to_device(bits);
}

Weight bf16_weight(const Matrix& m, const DeviceBuffer& payload) {
    Weight w  = base_weight(m, payload.p, payload.bytes);
    w.qtype   = QType::BF16;
    w.layout  = QuantLayout::Contiguous;
    return w;
}

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// max |a - b| over max |b|: both sides approximate one ideal value in BF16.
double relative_gap(const std::vector<double>& a, const std::vector<double>& b) {
    double gap = 0, scale = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        gap   = std::max(gap, std::abs(a[i] - b[i]));
        scale = std::max(scale, std::abs(b[i]));
    }
    return scale > 0 ? gap / scale : gap;
}

int check(bool ok, const std::string& what) {
    if (!ok) std::cerr << "FAIL " << what << '\n';
    return ok ? 0 : 1;
}

int test_projected_topk(std::int32_t rows, std::int32_t valid_rows, std::int32_t columns) {
    constexpr std::int32_t kK = 512, kTop = 16;
    const Matrix m = make_matrix(rows, kK, 17U + static_cast<std::uint32_t>(rows));
    std::size_t scale_offset = 0;
    DeviceBuffer payload     = q4_0_payload(m, scale_offset);
    const Weight head        = q4_0_weight(m, payload, scale_offset);
    DeviceBuffer hidden      = to_device(random_bf16(static_cast<std::size_t>(kK) * columns, 5U,
                                                     -1.0F, 1.0F));
    Tensor x(hidden.p, DType::BF16, {kK, columns});

    // Reference: the same Linear's BF16 logits, ranked on the host.
    DeviceBuffer logits_device(static_cast<std::size_t>(rows) * columns * 2);
    Tensor logits(logits_device.p, DType::BF16, {rows, columns});
    {
        const auto bytes = ops::linear_workspace_capacity_bytes(head.qtype, rows, kK,
                                                           ops::LinearPolicy::A16Only, columns,
                                                           columns);
        DeviceBuffer scratch(std::max<std::size_t>(bytes, 1));
        WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
        ops::linear(x, head, logits, ops::LinearPolicy::A16Only, workspace, nullptr);
        cuda_synchronize();
    }
    const auto host_logits = from_device_bf16(logits_device, static_cast<std::size_t>(rows) * columns);

    DeviceBuffer ids_device(static_cast<std::size_t>(kTop) * columns * 4);
    DeviceBuffer scores_device(static_cast<std::size_t>(kTop) * columns * 4);
    Tensor ids(ids_device.p, DType::I32, {kTop, columns});
    Tensor scores(scores_device.p, DType::FP32, {kTop, columns});
    const auto capacity =
        ops::linear_topk_workspace_capacity_bytes(head.qtype, rows, kK, columns, columns);
    DeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
    ops::linear_topk(x, head, valid_rows, ids, scores, workspace, nullptr);
    cuda_synchronize();
    const auto got_ids    = from_device<int>(ids_device, static_cast<std::size_t>(kTop) * columns);
    const auto got_scores = from_device<float>(scores_device, static_cast<std::size_t>(kTop) * columns);

    int failures = 0;
    for (std::int32_t t = 0; t < columns; ++t) {
        std::vector<std::int32_t> order(static_cast<std::size_t>(valid_rows));
        std::iota(order.begin(), order.end(), 0);
        const auto score = [&](std::int32_t r) {
            return host_logits[static_cast<std::size_t>(t) * rows + r];
        };
        std::partial_sort(order.begin(), order.begin() + kTop, order.end(),
                          [&](std::int32_t a, std::int32_t b) {
                              return score(a) > score(b) || (score(a) == score(b) && a < b);
                          });
        for (std::int32_t r = 0; r < kTop; ++r) {
            const std::size_t at = static_cast<std::size_t>(t) * kTop + r;
            if (got_ids[at] != order[r] || got_scores[at] != static_cast<float>(score(order[r]))) {
                ++failures;
            }
        }
    }
    return check(failures == 0, "linear_topk projected rows=" + std::to_string(rows) +
                                    " valid=" + std::to_string(valid_rows) +
                                    " U=" + std::to_string(columns));
}

int test_conv_prepare(std::int32_t width, std::int32_t batch) {
    constexpr std::int32_t kH = 5120, kRows = 1280, kGroups = 320;
    const std::int32_t columns = width * batch;
    const Matrix m             = make_matrix(kRows, kH, 23U);
    std::size_t scale_offset   = 0;
    DeviceBuffer q4            = q4_0_payload(m, scale_offset);
    DeviceBuffer bf16          = bf16_payload(m);
    DeviceBuffer residual      = to_device(random_bf16(static_cast<std::size_t>(kH) * columns, 1U, -2.0F, 2.0F));
    DeviceBuffer norm          = to_device(random_bf16(kH, 2U, 0.5F, 1.5F));
    DeviceBuffer base          = to_device(random_bf16(static_cast<std::size_t>(kH) * 4, 3U, -0.5F, 0.5F));
    Tensor r(residual.p, DType::BF16, {kH, width, batch});
    Tensor nw(norm.p, DType::BF16, {kH});
    Tensor bk(base.p, DType::BF16, {kH, 2, 2});
    std::vector<std::vector<double>> prepared(2), delta(2);
    for (int route = 0; route < 2; ++route) {
        const Weight w = route == 0 ? bf16_weight(m, bf16) : q4_0_weight(m, q4, scale_offset);
        DeviceBuffer out(static_cast<std::size_t>(kH) * columns * 2);
        DeviceBuffer fd(static_cast<std::size_t>(kGroups) * 2 * columns * 2);
        Tensor p(out.p, DType::BF16, {kH, width, batch});
        Tensor f(fd.p, DType::BF16, {kGroups, 2, width, batch});
        const auto bytes = ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
            w.qtype, width, width, batch, batch);
        DeviceBuffer scratch(std::max<std::size_t>(bytes, 1));
        WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
        ops::rmsnorm_dynamic_grouped_conv_prepare(r, nw, 1e-6F, bk, w, p, f, workspace, nullptr);
        cuda_synchronize();
        prepared[route] = from_device_bf16(out, static_cast<std::size_t>(kH) * columns);
        delta[route]    = from_device_bf16(fd, static_cast<std::size_t>(kGroups) * 2 * columns);
    }
    const double gp = relative_gap(prepared[1], prepared[0]);
    const double gd = relative_gap(delta[1], delta[0]);
    std::cout << "conv prepare W=" << width << " B=" << batch << " Q4_0 vs BF16: prepared " << gp
              << ", finish_delta " << gd << '\n';
    return check(gp < 2e-2 && gd < 2e-2, "conv prepare generic W=" + std::to_string(width));
}

int test_conv_add(std::int32_t input_rows, std::int32_t width, std::int32_t batch) {
    constexpr std::int32_t kH = 5120, kGroups = 320;
    const std::int32_t columns = width * batch;
    const Matrix m             = make_matrix(kH, input_rows, 29U + static_cast<std::uint32_t>(input_rows));
    std::size_t q4_scales = 0, q8_scales = 0;
    DeviceBuffer q4       = q4_0_payload(m, q4_scales);
    DeviceBuffer q8       = q8_payload(m, q8_scales);
    DeviceBuffer x        = to_device(random_bf16(static_cast<std::size_t>(input_rows) * columns, 4U, -1.0F, 1.0F));
    DeviceBuffer base     = to_device(random_bf16(static_cast<std::size_t>(kH) * 4, 6U, -0.5F, 0.5F));
    DeviceBuffer fd       = to_device(random_bf16(static_cast<std::size_t>(kGroups) * 2 * columns, 7U, -0.25F, 0.25F));
    const auto start      = random_bf16(static_cast<std::size_t>(kH) * columns, 8U, -1.0F, 1.0F);
    Tensor tx(x.p, DType::BF16, {input_rows, width, batch});
    Tensor bk(base.p, DType::BF16, {kH, 2, 2});
    Tensor tf(fd.p, DType::BF16, {kGroups, 2, width, batch});
    std::vector<std::vector<double>> result(2);
    for (int route = 0; route < 2; ++route) {
        const Weight w = route == 0 ? q8_weight(m, q8, q8_scales) : q4_0_weight(m, q4, q4_scales);
        DeviceBuffer residual = to_device(start);
        Tensor tr(residual.p, DType::BF16, {kH, width, batch});
        const auto bytes = ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
            w.qtype, input_rows, width, width, batch, batch);
        DeviceBuffer scratch(std::max<std::size_t>(bytes, 1));
        WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
        ops::linear_dynamic_grouped_conv_add(tx, w, bk, tf, tr, workspace, nullptr);
        cuda_synchronize();
        result[route] = from_device_bf16(residual, static_cast<std::size_t>(kH) * columns);
    }
    const double gap = relative_gap(result[1], result[0]);
    std::cout << "conv add C=" << input_rows << " W=" << width << " B=" << batch
              << " Q4_0 vs Q8: " << gap << '\n';
    return check(gap < 2e-2, "conv add generic C=" + std::to_string(input_rows));
}

// The host decode of each payload the packers above write, independent of the kernels: ggml Q4_0
// (low nibble = value j, high = j + 16, binary16 d per block), Q8_G32_FP16 RowSplit (int8 codes,
// then binary16 scales per 32 values), BF16 contiguous. Row-major [n, k] floats.
float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000U) << 16, exponent = (h >> 10) & 0x1FU,
                        mantissa = h & 0x3FFU;
    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa ? std::nanf("") : INFINITY;
    } else {
        value = std::ldexp(static_cast<float>(mantissa | 0x400U), static_cast<int>(exponent) - 25);
    }
    return sign ? -value : value;
}

std::vector<float> decode_q4_0(const DeviceBuffer& payload, std::size_t scale_offset,
                               std::int32_t n, std::int32_t k) {
    const auto bytes = from_device<std::uint8_t>(payload, payload.bytes);
    std::vector<float> out(static_cast<std::size_t>(n) * k);
    for (std::size_t block = 0; block < out.size() / 32; ++block) {
        const float d = f16_to_f32(static_cast<std::uint16_t>(
            bytes[scale_offset + 2 * block] | (bytes[scale_offset + 2 * block + 1] << 8)));
        for (int j = 0; j < 32; ++j) {
            const std::uint8_t byte = bytes[block * 16 + (j & 15)];
            const int q             = j < 16 ? (byte & 15) : (byte >> 4);
            out[block * 32 + j]     = static_cast<float>(q - 8) * d;
        }
    }
    return out;
}

std::vector<float> decode_q8(const DeviceBuffer& payload, std::size_t scale_offset, std::int32_t n,
                             std::int32_t k) {
    const auto bytes = from_device<std::uint8_t>(payload, payload.bytes);
    std::vector<float> out(static_cast<std::size_t>(n) * k);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::size_t group = i / 32;
        const float scale       = f16_to_f32(static_cast<std::uint16_t>(
            bytes[scale_offset + 2 * group] | (bytes[scale_offset + 2 * group + 1] << 8)));
        out[i]                  = static_cast<float>(static_cast<std::int8_t>(bytes[i])) * scale;
    }
    return out;
}

std::vector<float> decode_bf16(const DeviceBuffer& payload, std::int32_t n, std::int32_t k) {
    const auto bits = from_device<std::uint16_t>(payload, static_cast<std::size_t>(n) * k);
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bf16_to_f32(bits[i]);
    return out;
}

// Q/K/V of a 5120-wide DFlash layer (32 x 128 query, 8 x 128 key and value rows): the fused Q8
// kernel and the segmented bank over one BF16 or one Q4_0 parent, rows sliced as the drafter's
// bank slices them, on the same represented values. Each route is checked against an FP64 product
// of the host-decoded weights and the BF16 input (the oracle: no kernel, no shared layout code),
// then against the fused route.
int test_qkv_bank(std::int32_t columns) {
    constexpr std::int32_t kH = 5120, kQ = 4096, kKv = 1024, kRows = kQ + 2 * kKv;
    const Matrix m        = make_matrix(kRows, kH, 41U);
    std::size_t q4_scales = 0, q8_scales = 0;
    DeviceBuffer q4   = q4_0_payload(m, q4_scales);
    DeviceBuffer q8   = q8_payload(m, q8_scales);
    DeviceBuffer bf16 = bf16_payload(m);
    DeviceBuffer x =
        to_device(random_bf16(static_cast<std::size_t>(kH) * columns, 9U, -1.0F, 1.0F));
    Tensor tx(x.p, DType::BF16, {kH, columns});
    int failures = 0;

    // The payloads decode to the represented matrix, exactly, in every format.
    const auto w_q8             = decode_q8(q8, q8_scales, kRows, kH);
    const auto w_q4             = decode_q4_0(q4, q4_scales, kRows, kH);
    const auto w_bf16           = decode_bf16(bf16, kRows, kH);
    std::size_t decode_mismatch = 0;
    for (std::int32_t r = 0; r < kRows; ++r) {
        for (std::int32_t c = 0; c < kH; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * kH + c;
            const float want    = m.value(r, c);
            decode_mismatch += (w_q8[i] != want) + (w_q4[i] != want) + (w_bf16[i] != want);
        }
    }
    failures += check(decode_mismatch == 0, "qkv bank payload decode (" +
                                                std::to_string(decode_mismatch) + " mismatches)");

    // The oracle in FP64, laid out as the routes' outputs: q [4096, T], k [1024, T], v [1024, T]
    // (rows fastest), concatenated.
    const auto x_bits = from_device<std::uint16_t>(x, static_cast<std::size_t>(kH) * columns);
    std::vector<double> xd(x_bits.size());
    for (std::size_t i = 0; i < x_bits.size(); ++i) xd[i] = bf16_to_f32(x_bits[i]);
    std::vector<double> oracle;
    oracle.reserve(static_cast<std::size_t>(kRows) * columns);
    for (const auto [begin, rows] :
         {std::pair{0, kQ}, std::pair{kQ, kKv}, std::pair{kQ + kKv, kKv}}) {
        for (std::int32_t t = 0; t < columns; ++t) {
            for (std::int32_t r = begin; r < begin + rows; ++r) {
                double sum         = 0.0;
                const float* row   = w_bf16.data() + static_cast<std::size_t>(r) * kH;
                const double* xcol = xd.data() + static_cast<std::size_t>(t) * kH;
                for (std::int32_t c = 0; c < kH; ++c) sum += static_cast<double>(row[c]) * xcol[c];
                oracle.push_back(sum);
            }
        }
    }
    double scale = 0.0;
    for (const double value : oracle) scale = std::max(scale, std::abs(value));

    std::vector<std::vector<double>> result(3);
    for (int route = 0; route < 3; ++route) {
        DeviceBuffer qd(static_cast<std::size_t>(kQ) * columns * 2);
        DeviceBuffer kd(static_cast<std::size_t>(kKv) * columns * 2);
        DeviceBuffer vd(static_cast<std::size_t>(kKv) * columns * 2);
        Tensor q(qd.p, DType::BF16, {kQ, columns});
        Tensor k(kd.p, DType::BF16, {kKv, columns});
        Tensor v(vd.p, DType::BF16, {kKv, columns});
        if (route == 0) {
            ops::attn_input_proj(tx, q8_weight(m, q8, q8_scales), q, k, v, nullptr);
        } else {
            const Weight parent = route == 1 ? bf16_weight(m, bf16) : q4_0_weight(m, q4, q4_scales);
            ops::SegmentedProjectionWeights bank;
            bank.slices = {{parent, 0, kQ}, {parent, kQ, kKv}, {parent, kQ + kKv, kKv}};
            const std::array<ops::ProjectionOutput, 3> outputs{
                {{&q, 0, kQ}, {&k, 0, kKv}, {&v, 0, kKv}}};
            const auto bytes =
                ops::segmented_projection_workspace_capacity_bytes(bank, columns, columns);
            DeviceBuffer scratch(std::max<std::size_t>(bytes, 1));
            WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
            ops::segmented_projection(tx, bank, outputs, workspace, nullptr);
        }
        cuda_synchronize();
        for (DeviceBuffer* part : {&qd, &kd, &vd}) {
            const auto values = from_device_bf16(*part, part->bytes / 2);
            result[route].insert(result[route].end(), values.begin(), values.end());
        }
    }
    // Against the oracle, element by element: the BF16 output's rounding (half an ulp: at most
    // 2^-8 of the value, just above a power of two) plus an accumulation allowance relative to the
    // largest output: 1e-3 for the A16 routes (FP32 accumulation of exact products: fused Q8,
    // segmented BF16), 1e-2 for Q4_0, whose generic kernel may take the activations to 8-bit blocks
    // (ggml's q8_1 vector path).
    const char* names[]    = {"fused Q8", "segmented BF16", "segmented Q4_0"};
    const double allowed[] = {1e-3, 1e-3, 1e-2};
    for (int route = 0; route < 3; ++route) {
        double worst        = 0.0; // max |got - oracle| / scale
        double used         = 0.0; // max |got - oracle| / its tolerance
        std::size_t outside = 0;
        for (std::size_t i = 0; i < oracle.size(); ++i) {
            const double gap = std::abs(result[route][i] - oracle[i]);
            worst            = std::max(worst, gap / scale);
            used = std::max(used, gap / (std::abs(oracle[i]) * 0x1p-8 + allowed[route] * scale));
            outside += gap > std::abs(oracle[i]) * 0x1p-8 + allowed[route] * scale;
        }
        std::cout << "dflash qkv bank T=" << columns << " " << names[route]
                  << " vs FP64 oracle: max error " << worst << " of the largest output (" << scale
                  << "), " << used << " of its tolerance at most, " << outside << " outside\n";
        failures += check(outside == 0, "dflash qkv bank T=" + std::to_string(columns) + " " +
                                            names[route] + " vs oracle");
    }
    const double bf16_gap = relative_gap(result[1], result[0]);
    const double q4_gap   = relative_gap(result[2], result[0]);
    std::cout << "dflash qkv bank T=" << columns << " vs fused Q8: BF16 " << bf16_gap << ", Q4_0 "
              << q4_gap << '\n';
    failures +=
        check(bf16_gap < 2e-2 && q4_gap < 2e-2, "dflash qkv bank T=" + std::to_string(columns));
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        int failures = 0;
        failures += test_projected_topk(4096, 4096, 9);
        failures += test_projected_topk(5000, 4777, 18);
        failures += test_projected_topk(1024, 999, 130); // two column chunks
        failures += test_conv_prepare(9, 1);
        failures += test_conv_prepare(9, 3);
        failures += test_conv_add(4096, 9, 2);
        failures += test_conv_add(17408, 9, 1);
        failures += test_qkv_bank(8);  // one lane, depth 7
        failures += test_qkv_bank(32); // four lanes
        std::cout << (failures == 0 ? "OK" : "FAIL") << " dflash / dflash2 generic routes\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "dflash2 generic routes test: " << error.what() << '\n';
        return 1;
    }
}
