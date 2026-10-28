#include "models/speculation/dflash.h"

#include "artifact/reader.h"
#include "models/common/config_json.h"
#include "models/common/loading.h"
#include "core/weight_view.h"
#include "ninferno/ops/linear.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninferno::models::speculation {

using loading::Demands;
using loading::Id;
using loading::open_text_component;
using loading::Prepare;

namespace {

std::int32_t positive(const artifact::Json& json, const char* key) {
    const auto value = json_dimension(json, key);
    if (value > static_cast<std::uint32_t>(INT32_MAX)) {
        throw artifact::ArtifactError(std::string("dflash config: ") + key + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

} // namespace

DFlashConfig parse_dflash_config(const artifact::Json& c) {
    const auto model_type = c.value("model_type", std::string());
    if (model_type != "gemma4_dflash" && model_type != "dflash") {
        throw std::invalid_argument("dflash config: model_type must be gemma4_dflash or dflash");
    }
    DFlashConfig out;
    out.hidden_size       = positive(c, "hidden_size");
    // A drafter without its own table may leave both unrecorded (0): load() pairs it with the
    // target's.
    out.vocab_size        = c.contains("vocab_size") ? positive(c, "vocab_size") : 0;
    out.layers            = positive(c, "num_hidden_layers");
    out.target_layers     = c.contains("num_target_layers") ? positive(c, "num_target_layers") : 0;
    out.query_heads       = positive(c, "num_attention_heads");
    out.kv_heads          = positive(c, "num_key_value_heads");
    out.head_dim          = positive(c, "head_dim");
    out.intermediate_size = positive(c, "intermediate_size");
    // A drafter without a sliding layer (gpt-oss's) records no window.
    out.sliding_window    = c.contains("sliding_window") && !c.at("sliding_window").is_null()
                                ? positive(c, "sliding_window")
                                : 0;
    out.block_size        = positive(c, "block_size");
    out.max_positions     = c.contains("max_position_embeddings")
                                ? static_cast<std::uint32_t>(positive(c, "max_position_embeddings"))
                                : 0U;
    out.rope_theta        = json_positive_float(c, "rope_theta");
    out.rms_norm_eps      = json_positive_float(c, "rms_norm_eps");
    if ((out.head_dim != 128 && out.head_dim != 64) || out.query_heads % out.kv_heads != 0) {
        throw std::invalid_argument("dflash config: layers must be D128 or D64 grouped-query");
    }
    out.attention_bias = c.value("attention_bias", false);
    // The rotary table: the artifact's resolved one (a scaled rope, YaRN), else the plain
    // frequencies of rope_theta over the whole head, in binary32 as transformers computes them.
    if (c.contains("rope")) {
        const auto& rope = c.at("rope");
        const auto& freq = rope.at("inv_freq");
        if (json_dimension(rope, "rotary_dim") != static_cast<std::uint32_t>(out.head_dim) ||
            !freq.is_array() || static_cast<std::int32_t>(freq.size()) != out.head_dim / 2) {
            throw artifact::ArtifactError(
                "dflash config: rope must rotate the whole head (head_dim/2 frequencies)");
        }
        for (const auto& value : freq) {
            const double f = value.get<double>();
            if (!std::isfinite(f) || f <= 0) {
                throw artifact::ArtifactError("dflash config: rope frequencies must be positive");
            }
            out.rope_inv_freq.push_back(static_cast<float>(f));
        }
        out.rope_attention_scaling = json_positive_float(rope, "attention_scaling");
        out.rope_scaled            = true;
    } else {
        for (std::int32_t i = 0; i < out.head_dim; i += 2) {
            const float exponent = static_cast<float>(i) / static_cast<float>(out.head_dim);
            out.rope_inv_freq.push_back(1.0F / std::pow(out.rope_theta, exponent));
        }
    }
    const auto& kinds = c.at("layer_types");
    if (!kinds.is_array() || static_cast<std::int32_t>(kinds.size()) != out.layers) {
        throw std::invalid_argument("dflash config: layer_types must cover every layer");
    }
    for (const auto& kind : kinds) {
        const auto name = kind.get<std::string>();
        if (name != "full_attention" && name != "sliding_attention") {
            throw std::invalid_argument("dflash config: unknown layer type " + name);
        }
        if (name == "sliding_attention" && out.sliding_window == 0) {
            throw artifact::ArtifactError("dflash config: a sliding layer needs sliding_window");
        }
        out.windows.push_back(name == "full_attention" ? 0 : out.sliding_window);
    }
    const auto& draft = c.at("dflash_config");
    const auto& taps  = draft.at("target_layer_ids");
    // Without a recorded depth the taps are bounded here by the largest (distinct, in range of a
    // layer index), and by the target's depth in load().
    std::uint32_t bound = static_cast<std::uint32_t>(out.target_layers);
    if (bound == 0 && taps.is_array()) {
        for (const auto& tap : taps) {
            const auto index = json_u32(tap, "target_layer_ids", false);
            if (index >= static_cast<std::uint32_t>(INT32_MAX)) {
                throw artifact::ArtifactError("dflash config: target layer index exceeds int32");
            }
            bound = std::max(bound, index + 1);
        }
    }
    for (const auto tap : json_layer_indices(taps, bound, "target_layer_ids")) {
        out.taps.push_back(static_cast<std::int32_t>(tap));
    }
    out.mask_token =
        static_cast<std::int32_t>(json_u32(draft.at("mask_token_id"), "mask token", false));
    if (out.vocab_size && out.mask_token >= out.vocab_size) {
        throw artifact::ArtifactError("dflash config: mask token outside the vocabulary");
    }
    if (model_type == "dflash") {
        const auto mask = draft.value("sliding_attention_mask", std::string());
        if (mask != "block" && mask != "causal") {
            throw artifact::ArtifactError(
                "dflash config: dflash_config.sliding_attention_mask must be block or causal");
        }
        out.sliding_block = mask == "block";
    }
    return out;
}

DFlashDrafter::~DFlashDrafter() = default;

std::uint64_t DFlashDrafter::device_bytes() const noexcept {
    return backing_ ? backing_->materialized.stats().device_capacity_bytes : 0;
}

namespace {

// A transient device allocation, freed however the split ends.
struct DeviceFree {
    void operator()(void* p) const noexcept { (void)cudaFree(p); }
};

} // namespace

std::vector<Projection> split_projection_columns(Projection& full_projection, std::uint64_t parts,
                                                 cudaStream_t stream, std::string* why) {
    const auto refuse = [why](const std::string& reason) {
        if (why != nullptr) { *why = reason; }
        return std::vector<Projection>{};
    };
    const Weight& w            = full_projection.weight;
    const auto n               = static_cast<std::uint64_t>(w.n);
    const auto k               = static_cast<std::uint64_t>(w.k);
    const std::uint64_t part_k = parts != 0 ? k / parts : 0;
    if (parts < 2 || w.payload == nullptr || part_k * parts != k || part_k == 0) {
        return refuse("not a matrix of whole column blocks");
    }
    const bool bf16 = w.layout == QuantLayout::Contiguous && w.qtype == QType::BF16;
    if (!bf16 && w.layout != QuantLayout::GgmlBlocks && w.layout != QuantLayout::GgmlPlanes) {
        return refuse("its format is neither BF16 nor a ggml block layout");
    }
    const std::array<std::uint64_t, 2> full_shape{n, k};
    const std::array<std::uint64_t, 2> part_shape{n, part_k};
    WeightGeometry full, part;
    try {
        full = weight_geometry(w.qtype, w.layout, full_shape);
        part = weight_geometry(w.qtype, w.layout, part_shape);
    } catch (const std::invalid_argument&) {
        return refuse("a column block is not a whole number of quantization blocks");
    }
    // Each plane: its offset in the full matrix and in a slice, its bytes per full and slice row.
    struct Plane {
        std::uint64_t full_offset, part_offset, full_row, part_row;
    };
    std::vector<Plane> planes{{0, 0, full.code_bytes_per_row, part.code_bytes_per_row}};
    if (w.layout == QuantLayout::GgmlPlanes) {
        planes.push_back({full.high_offset, part.high_offset, full.high_bytes_per_row,
                          part.high_bytes_per_row});
        planes.push_back({full.scale_offset, part.scale_offset, full.scale_bytes_per_row,
                          part.scale_bytes_per_row});
    }
    for (const Plane& plane : planes) {
        if (plane.full_row != parts * plane.part_row) {
            return refuse("a plane's rows do not split by column block");
        }
    }
    const std::uint64_t stride = (part.bytes + 255) / 256 * 256;
    // The matrix is its parent, whole (not a part of a packed one).
    if (w.qdata != w.payload || full.bytes != w.payload_bytes || parts * stride > full.bytes) {
        return refuse("the matrix shares its storage");
    }

    // Slice i is columns [i*K/parts, (i+1)*K/parts) of every row, each plane's rows now
    // contiguous: copied from a transient copy of the whole matrix into the matrix's own bytes. The
    // copy is not the stream pool's (nothing stays reserved), and a failed allocation leaves the
    // matrix as it was.
    auto* base = static_cast<std::byte*>(const_cast<void*>(w.payload));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    void* raw = nullptr;
    if (cudaMalloc(&raw, full.bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        return refuse("no device memory for its transient copy (" +
                      std::to_string((full.bytes + (1U << 20) - 1) >> 20) + " MiB)");
    }
    const std::unique_ptr<void, DeviceFree> copy(raw);
    CUDA_CHECK(cudaMemcpyAsync(copy.get(), base, full.bytes, cudaMemcpyDeviceToDevice, stream));
    for (std::uint64_t i = 0; i < parts; ++i) {
        for (const Plane& plane : planes) {
            if (plane.part_row == 0) { continue; }
            CUDA_CHECK(cudaMemcpy2DAsync(
                base + i * stride + plane.part_offset, plane.part_row,
                static_cast<const std::byte*>(copy.get()) + plane.full_offset + i * plane.part_row,
                plane.full_row, plane.part_row, n, cudaMemcpyDeviceToDevice, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<Projection> slices;
    slices.reserve(parts);
    for (std::uint64_t i = 0; i < parts; ++i) {
        const WeightParent parent{.geometry = part, .data = base + i * stride};
        const WeightView view{.shape = {n, part_k}, .parts = {{&parent, 0, n * part_k}}};
        slices.push_back(
            Projection{.weight = native_weight(view), .policy = full_projection.policy});
    }
    // Only its shape remains: its bytes are the slices'.
    full_projection.weight.payload = nullptr;
    full_projection.weight.qdata   = nullptr;
    full_projection.weight.qhigh   = nullptr;
    full_projection.weight.scales  = nullptr;
    return slices;
}

bool DFlashDrafter::split_feature_projection(DeviceContext& device, std::string* why) {
    if (!weights_.feature_slices.empty()) { return true; }
    const Weight& w = weights_.feature_projection.weight;
    const auto taps = static_cast<std::uint64_t>(config_.taps.size());
    const auto h    = static_cast<std::uint64_t>(config_.hidden_size);
    if (taps < 2 || static_cast<std::uint64_t>(w.n) != h ||
        static_cast<std::uint64_t>(w.k) != taps * h) {
        if (why != nullptr) { *why = "not an [H, taps*H] matrix over two taps or more"; }
        return false;
    }
    weights_.feature_slices =
        split_projection_columns(weights_.feature_projection, taps, device.stream, why);
    // A slice has its own geometry: the full six-tap matrix may be one a tuned table owns
    // (Gemma 4 31B's 5376 x 32256, kept A16) while its [H, H] slices are not, so each slice takes
    // the drafter policy of its own shape (ops::drafter_linear_policy).
    for (auto& slice : weights_.feature_slices) {
        if (slice.policy == ops::LinearPolicy::A16Only || slice.policy == ops::LinearPolicy::AllowA8) {
            slice.policy = ops::drafter_linear_policy(slice.weight, ops::LinearPolicy::A16Only);
        }
    }
    return !weights_.feature_slices.empty();
}

std::unique_ptr<DFlashDrafter> DFlashDrafter::load(const std::filesystem::path& path,
                                                   DeviceContext& device,
                                                   const TargetShape& target) {
    artifact::Reader reader(path);
    const auto& text = open_text_component(reader, "dflash");
    auto drafter     = std::unique_ptr<DFlashDrafter>(new DFlashDrafter());
    drafter->config_ = parse_dflash_config(text);
    auto& c          = drafter->config_;
    if (c.vocab_size == 0) { c.vocab_size = target.vocab_size; }
    if (c.target_layers == 0) { c.target_layers = target.layers; }
    // The drafter embeds with the target's table, scores with its head and reads its residual
    // stream after the tapped layers: each check is a pairing that would otherwise fail silently.
    if (c.hidden_size != target.hidden_size || c.vocab_size != target.vocab_size ||
        c.target_layers != target.layers) {
        throw artifact::ArtifactError("dflash: does not pair with this target");
    }
    for (const auto tap : c.taps) {
        if (tap < 0 || tap >= target.layers) {
            throw artifact::ArtifactError("dflash: tapped layer outside the target");
        }
    }
    if (c.mask_token >= c.vocab_size) {
        throw artifact::ArtifactError("dflash: mask token outside the target's vocabulary");
    }
    const auto h       = static_cast<std::uint64_t>(c.hidden_size);
    const auto d       = static_cast<std::uint64_t>(c.head_dim);
    const auto q_rows  = std::uint64_t(c.query_heads) * d;
    const auto kv_rows = std::uint64_t(c.kv_heads) * d;
    const auto ir      = static_cast<std::uint64_t>(c.intermediate_size);

    Demands demands(reader);
    const Id features =
        demands.matrix("text/feature_projection", {h, c.taps.size() * h}, "text/target_features");
    const Id ctx_norm   = demands.direct("text/context_norm", {h}, QType::BF16);
    const Id final_norm = demands.direct("text/final_norm", {h}, QType::BF16);

    struct Ids {
        Id input_norm, post_attention_norm, query_norm, key_norm;
        Id query, key, value, context_key, context_value, output, gate, up, down;
        std::optional<Id> query_bias, key_bias, value_bias, output_bias;
    };

    std::vector<Ids> ids(c.layers);
    std::vector<Id> context;
    for (std::int32_t i = 0; i < c.layers; ++i) {
        const std::string p   = "text/layers/" + std::to_string(i) + "/";
        auto& l               = ids[i];
        l.input_norm          = demands.direct(p + "input_norm", {h}, QType::BF16);
        l.post_attention_norm = demands.direct(p + "post_attention_norm", {h}, QType::BF16);
        l.query_norm          = demands.direct(p + "attention/query_norm", {d}, QType::BF16);
        l.key_norm            = demands.direct(p + "attention/key_norm", {d}, QType::BF16);
        l.query = demands.matrix(p + "attention/query", {q_rows, h}, p + "mixer_input");
        l.key   = demands.matrix(p + "attention/key", {kv_rows, h}, p + "mixer_input");
        l.value = demands.matrix(p + "attention/value", {kv_rows, h}, p + "mixer_input");
        l.context_key =
            demands.matrix(p + "attention/context_key", {kv_rows, h}, "text/context_input");
        l.context_value =
            demands.matrix(p + "attention/context_value", {kv_rows, h}, "text/context_input");
        context.push_back(l.context_key);
        context.push_back(l.context_value);
        l.output = demands.matrix(p + "attention/output", {h, q_rows}, p + "attention/context");
        l.gate   = demands.matrix(p + "mlp/gate", {ir, h}, p + "ffn_input");
        l.up     = demands.matrix(p + "mlp/up", {ir, h}, p + "ffn_input");
        l.down   = demands.matrix(p + "mlp/down", {h, ir}, p + "mlp/product");
        if (c.attention_bias) {
            l.query_bias  = demands.direct(p + "attention/query_bias", {q_rows}, QType::BF16);
            l.key_bias    = demands.direct(p + "attention/key_bias", {kv_rows}, QType::BF16);
            l.value_bias  = demands.direct(p + "attention/value_bias", {kv_rows}, QType::BF16);
            l.output_bias = demands.direct(p + "attention/output_bias", {h}, QType::BF16);
        }
    }
    drafter->backing_ = demands.materialize(device);
    // BF16 activations, but A8 where the format verifies a whole draft block on R8 (Q8:
    // ops::drafter_linear_policy). Under --activations exact the invariant routes run BF16
    // whatever the policy.
    const Prepare prepare(demands, *drafter->backing_, /*recorded=*/false);
    const auto lift = [](Projection p) {
        p.policy = ops::drafter_linear_policy(p.weight, p.policy);
        return p;
    };
    const auto lift_packed = [&](std::optional<Projection> p) {
        if (p) { *p = lift(*p); }
        return p;
    };
    auto& w              = drafter->weights_;
    w.feature_projection = lift(prepare.projection(features));
    w.context_kv         = lift_packed(prepare.packed(context));
    w.context_norm       = prepare.tensor(ctx_norm);
    w.final_norm         = prepare.tensor(final_norm);
    {
        auto& buffer = drafter->backing_->derived.emplace_back(c.rope_inv_freq.size() *
                                                               sizeof(float));
        buffer.copy_from_host(c.rope_inv_freq.data(), buffer.bytes);
        w.rope_inv_freq = Tensor(buffer.p, DType::FP32,
                                 {static_cast<std::int32_t>(c.rope_inv_freq.size())});
    }
    w.layers.resize(c.layers);
    for (std::int32_t i = 0; i < c.layers; ++i) {
        auto& layer               = w.layers[i];
        const auto& l             = ids[i];
        layer.window              = c.windows[i];
        layer.input_norm          = prepare.tensor(l.input_norm);
        layer.post_attention_norm = prepare.tensor(l.post_attention_norm);
        layer.query_norm          = prepare.tensor(l.query_norm);
        layer.key_norm            = prepare.tensor(l.key_norm);
        if (c.attention_bias) {
            layer.query_bias  = prepare.tensor(*l.query_bias);
            layer.key_bias    = prepare.tensor(*l.key_bias);
            layer.value_bias  = prepare.tensor(*l.value_bias);
            layer.output_bias = prepare.tensor(*l.output_bias);
        }
        const std::array qkv{l.query, l.key, l.value};
        layer.qkv = lift_packed(prepare.packed(qkv));
        if (!layer.qkv) {
            layer.query = lift(prepare.projection(l.query));
            layer.key   = lift(prepare.projection(l.key));
            layer.value = lift(prepare.projection(l.value));
        }
        if (!w.context_kv) {
            layer.context_key   = lift(prepare.projection(l.context_key));
            layer.context_value = lift(prepare.projection(l.context_value));
        }
        layer.output = lift(prepare.projection(l.output));
        const std::array gate_up{l.gate, l.up};
        layer.gate_up = lift_packed(prepare.packed(gate_up));
        if (!layer.gate_up) {
            layer.gate = lift(prepare.projection(l.gate));
            layer.up   = lift(prepare.projection(l.up));
        }
        layer.down = lift(prepare.projection(l.down));
    }
    return drafter;
}

} // namespace ninferno::models::speculation
