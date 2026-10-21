#include "ninferno/ops/weight_input.h"

#include "ops/common/native_nvfp4.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninferno::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

const std::vector<std::uint64_t>& matrix(const WeightInput& input) {
    require(input.weight.shape.size() == 2, "projection weight must be a matrix");
    const auto elements = weight_element_count(input.weight.shape);
    require(!input.weight.parts.empty(), "projection weight has no parent regions");
    std::uint64_t covered = 0;
    for (const auto& part : input.weight.parts) {
        require(part.parent && part.begin < part.end &&
                    part.end <= part.parent->geometry.elements &&
                    part.end - part.begin <= elements - covered,
                "projection region exceeds its parent or logical matrix");
        covered += part.end - part.begin;
    }
    require(covered == elements, "projection regions do not cover its logical matrix");
    return input.weight.shape;
}

WeightView concatenate_rows(std::span<const WeightInput> inputs) {
    require(!inputs.empty(), "projection bank is empty");
    const auto columns = matrix(inputs.front())[1];
    WeightView out{{0, columns}, {}};
    for (const auto& input : inputs) {
        const auto& shape = matrix(input);
        require(shape[1] == columns &&
                    shape[0] <= std::numeric_limits<std::uint64_t>::max() - out.shape[0],
                "projection bank has inconsistent columns or overflowing rows");
        out.shape[0] += shape[0];
        out.parts.insert(out.parts.end(), input.weight.parts.begin(), input.weight.parts.end());
    }
    return out;
}

bool contiguous(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    auto end           = view.parts.front().begin;
    for (const auto& part : view.parts) {
        if (!parent || part.parent != parent || part.begin != end || part.end <= part.begin) {
            return false;
        }
        end = part.end;
    }
    return true;
}

LinearPolicy common_policy(std::span<const WeightInput> inputs) {
    auto result = LinearPolicy::AllowA4;
    for (const auto& input : inputs) {
        switch (input.policy) {
        case LinearPolicy::A16Only:
            result = LinearPolicy::A16Only;
            break;
        case LinearPolicy::AllowA8:
            if (result == LinearPolicy::AllowA4) { result = LinearPolicy::AllowA8; }
            break;
        case LinearPolicy::AllowA4:
            break;
        default:
            throw std::invalid_argument("invalid projection activation policy");
        }
    }
    return result;
}

SingleProjectionWeight single(std::span<const WeightInput> inputs) {
    auto view         = concatenate_rows(inputs);
    const auto region = contiguous_weight_region(view);
    const auto policy = common_policy(inputs);
    float divisor     = 0;
    if (region.parent->geometry.format == QType::NVFP4) {
        require(std::isfinite(region.parent->weight_scale_divisor) &&
                    region.parent->weight_scale_divisor > 0,
                "NVFP4 parent requires a positive weight divisor");
        for (const auto& input : inputs) {
            if (!input.activation_input_divisor) {
                require(!allows_a4(policy), "NVFP4 A4 native input requires an activation divisor");
                continue;
            }
            require(std::isfinite(*input.activation_input_divisor) &&
                        *input.activation_input_divisor > 0,
                    "NVFP4 native input requires a positive activation divisor");
            if (divisor == 0) {
                // A16 does not read this auxiliary; retain a stored positive value for the ABI.
                divisor = *input.activation_input_divisor;
            } else if (allows_a4(policy)) {
                // AllowA8 runs A16 or the generic W4A8, which quantizes per 32 values and reads
                // no divisor. Only the A4 route quantizes the shared activation with it and
                // therefore requires a common divisor.
                require(std::bit_cast<std::uint32_t>(divisor) ==
                            std::bit_cast<std::uint32_t>(*input.activation_input_divisor),
                        "shared NVFP4 native input requires identical activation divisors");
            }
        }
        // The legacy native ABI validates this field even on A16, where it is never read.
        // Keep that ABI detail out of the artifact's optional Use auxiliaries.
        if (divisor == 0) { divisor = 1.0F; }
    }
    return {native_weight(view, divisor), policy};
}

ProjectionSlice projection_slice(const WeightInput& input) {
    const auto& shape  = matrix(input);
    const auto region  = contiguous_weight_region(input.weight);
    const auto& parent = *region.parent;
    const auto& g      = parent.geometry;
    require(g.shape.size() == 2 && g.shape[1] == shape[1] && g.shape[1] != 0 &&
                region.begin % g.shape[1] == 0,
            "projection slice must be whole rows of a matrix parent");
    WeightView full{g.shape, {WeightRegion{&parent, 0, g.elements}}};
    LinearPolicy policy = input.policy;
    float divisor       = 0.0F;
    if (g.format == QType::NVFP4) {
        require(std::isfinite(parent.weight_scale_divisor) && parent.weight_scale_divisor > 0,
                "NVFP4 parent requires a positive weight divisor");
        if (input.activation_input_divisor && std::isfinite(*input.activation_input_divisor) &&
            *input.activation_input_divisor > 0) {
            divisor = *input.activation_input_divisor;
        } else {
            // No calibrated activation divisor: the generic path computes this slice at A16.
            divisor = 1.0F;
            if (allows_a4(policy)) policy = LinearPolicy::AllowA8;
        }
    }
    ProjectionSlice out;
    out.parent    = native_weight(full, divisor);
    out.row_begin = static_cast<std::int32_t>(region.begin / g.shape[1]);
    out.rows      = static_cast<std::int32_t>(shape[0]);
    out.policy    = policy;
    return out;
}

ProjectionWeights input_projection(std::span<const WeightInput, 4> inputs, bool attention) {
    const auto& q      = matrix(inputs[0]);
    const auto& k      = matrix(inputs[1]);
    const auto& third  = matrix(inputs[2]);
    const auto& fourth = matrix(inputs[3]);
    const bool dense =
        attention ? q == std::vector<std::uint64_t>{6144, 5120} &&
                        k == std::vector<std::uint64_t>{1024, 5120} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 5120} && k == q &&
                        third == std::vector<std::uint64_t>{6144, 5120} && fourth == third;
    const bool moe =
        attention ? q == std::vector<std::uint64_t>{4096, 2048} &&
                        k == std::vector<std::uint64_t>{512, 2048} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 2048} && k == q &&
                        third == std::vector<std::uint64_t>{4096, 2048} && fourth == third;
    // A geometry or format combination no fused registration owns takes the generic path.
    // NINFERNO-SPECIAL(model:qwen35-27b,qwen35-35b-a3b): the fused input projections are compiled
    // for these two geometries; a runtime-geometry fused projection would remove the check.
    if (!(dense || moe)) return prepare_segmented_projection_weights(inputs);
    const auto joined = concatenate_rows(inputs);
    if (contiguous(joined)) {
        const auto format = contiguous_weight_region(joined).parent->geometry.format;
        const bool supported = (moe && format == QType::Q8_G32_FP16) ||
                               (dense && ((format == QType::NVFP4 && detail::kNativeNvfp4) ||
                                          format == QType::FP8_E4M3FN_ROW_BF16 ||
                                          (attention && format == QType::BF16)));
        if (!supported) return prepare_segmented_projection_weights(inputs);
        return single(inputs);
    }
    if (!dense) return prepare_segmented_projection_weights(inputs);
    const auto first_joined  = concatenate_rows(inputs.first<2>());
    const auto second_joined = concatenate_rows(inputs.last<2>());
    if (!contiguous(first_joined) || !contiguous(second_joined) ||
        contiguous_weight_region(first_joined).parent->geometry.format != QType::Q4_G64_FP16 ||
        contiguous_weight_region(second_joined).parent->geometry.format != QType::Q5_G64_FP16) {
        return prepare_segmented_projection_weights(inputs);
    }
    const auto first  = single(inputs.first<2>());
    const auto second = single(inputs.last<2>());
    return PairedProjectionWeights{first.weight, second.weight};
}

} // namespace

SingleProjectionWeight prepare_linear_weight(const WeightInput& input) {
    return single({&input, 1});
}

SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows) {
    return single(rows);
}

SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                       const WeightInput& key,
                                                       const WeightInput& value) {
    const auto& q = matrix(query);
    require((q == std::vector<std::uint64_t>{4096, 2048} ||
             q == std::vector<std::uint64_t>{4096, 5120}) &&
                matrix(key) == std::vector<std::uint64_t>{1024, q[1]} &&
                matrix(value) == matrix(key),
            "QKV input projection: unsupported logical geometry");
    const std::array inputs{query, key, value};
    auto result = single(inputs);
    require(result.weight.qtype == QType::Q8_G32_FP16,
            "QKV input projection: native form requires Q8");
    return result;
}

ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query, const WeightInput& key,
                                                  const WeightInput& gate,
                                                  const WeightInput& value) {
    const std::array inputs{query, key, gate, value};
    return input_projection(inputs, true);
}

ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query, const WeightInput& key,
                                                 const WeightInput& value, const WeightInput& z) {
    const std::array inputs{query, key, value, z};
    return input_projection(inputs, false);
}

ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a, const WeightInput& b) {
    const auto& shape = matrix(a);
    require(shape == matrix(b), "GDN control: A and B geometry differ");
    const std::array inputs{a, b};
    // NINFERNO-SPECIAL(model:qwen35-27b,qwen35-35b-a3b): the fused control projection is compiled
    // for these head counts; others take the segmented generic path.
    const bool registered = shape == std::vector<std::uint64_t>{48, 5120} ||
                            shape == std::vector<std::uint64_t>{32, 2048};
    if (!registered) return prepare_segmented_projection_weights(inputs);
    // The fused control kernel reads BF16; published quantized control weights (GGUF Q8_0) run
    // through the segmented generic path as stored.
    if (contiguous(concatenate_rows(inputs))) {
        auto result = single(inputs);
        if (result.weight.qtype != QType::BF16) return prepare_segmented_projection_weights(inputs);
        return result;
    }
    // Separate parents (a published recipe keeps A and B apart): the 27B pairs its two BF16
    // planes; the 35B geometry has no paired kernel and takes the segmented generic path.
    if (shape[0] != 48) return prepare_segmented_projection_weights(inputs);
    const auto first  = prepare_linear_weight(a);
    const auto second = prepare_linear_weight(b);
    if (first.weight.qtype != QType::BF16 || second.weight.qtype != QType::BF16) {
        return prepare_segmented_projection_weights(inputs);
    }
    return PairedProjectionWeights{first.weight, second.weight};
}

std::int32_t SegmentedProjectionWeights::rows() const {
    std::int32_t total = 0;
    for (const auto& slice : slices) total += slice.rows;
    return total;
}

std::int32_t SegmentedProjectionWeights::columns() const {
    return slices.empty() ? 0 : slices.front().parent.k;
}

bool consecutive_rows_of_one_parent(std::span<const WeightInput> inputs) {
    return contiguous(concatenate_rows(inputs));
}

SegmentedProjectionWeights prepare_segmented_projection_weights(std::span<const WeightInput> inputs) {
    require(!inputs.empty(), "projection bank is empty");
    SegmentedProjectionWeights out;
    out.slices.reserve(inputs.size());
    for (const auto& input : inputs) {
        out.slices.push_back(projection_slice(input));
        require(out.slices.back().parent.k == out.slices.front().parent.k,
                "projection bank has inconsistent columns");
    }
    return out;
}

SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                    const WeightInput& up) {
    require(matrix(gate) == matrix(up), "SwiGLU gate and up geometry differs");
    const std::array inputs{gate, up};
    return single(inputs);
}

ProjectionWeights prepare_dense_gate_up_weights(const WeightInput& gate, const WeightInput& up) {
    require(matrix(gate) == matrix(up), "SwiGLU gate and up geometry differs");
    const std::array inputs{gate, up};
    if (contiguous(concatenate_rows(inputs))) { return single(inputs); }
    return prepare_segmented_projection_weights(inputs);
}

MoeWeights prepare_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                               std::span<const WeightInput> expert_gate_up,
                               std::span<const WeightInput> expert_down,
                               const WeightInput& shared_gate, const WeightInput& shared_up,
                               const WeightInput& shared_down) {
    require(!expert_gate_up.empty() && expert_gate_up.size() == 2 * expert_down.size(),
            "MoE: one gate and one up input per expert down input");
    MoeWeights out;
    out.router         = prepare_linear_weight(router).weight;
    out.shared_gate    = prepare_linear_weight(shared_score).weight;
    const auto gate_up = single(expert_gate_up);
    const auto down    = single(expert_down);
    out.gate_up        = gate_up.weight;
    out.down           = down.weight;
    // The routed experts take A8 only where every expert projection allows it, and keep A4 where
    // one allows that (moe_policy: the grouped W4A8 leaves A4 NVFP4 banks on A16 where W4A4
    // kernels exist, moe.cpp w4a8_banks_admit).
    out.expert_policy         = moe_policy(gate_up.policy, down.policy);
    const auto shared_gate_up = prepare_linear_swiglu_weight(shared_gate, shared_up);
    const auto shared_down_w  = prepare_linear_weight(shared_down);
    out.shared_gate_up        = shared_gate_up.weight;
    out.shared_down           = shared_down_w.weight;
    out.shared_policy         = shared_moe_policy(shared_gate_up.policy, shared_down_w.policy);
    return out;
}

bool sparse_moe_admits(const MoeWeights& w) noexcept {
    const bool geometry = w.router.n == 256 && w.router.k == 2048 && w.shared_gate.n == 1 &&
                          w.shared_gate.k == 2048 && w.gate_up.n == 256 * 1024 &&
                          w.gate_up.k == 2048 && w.down.n == 256 * 2048 && w.down.k == 512 &&
                          w.shared_gate_up.n == 1024 && w.shared_gate_up.k == 2048 &&
                          w.shared_down.n == 2048 && w.shared_down.k == 512;
    const bool routers  = w.router.qtype == QType::BF16 && w.shared_gate.qtype == QType::BF16;
    const auto gate = w.gate_up.qtype, down = w.down.qtype;
    const auto shared = w.shared_gate_up.qtype, shared_down = w.shared_down.qtype;
    const bool groupwise =
        shared == QType::Q8_G32_FP16 && shared_down == QType::Q8_G32_FP16 &&
        ((gate == QType::Q4_G64_FP16 &&
          (down == QType::Q5_G64_FP16 || down == QType::Q6_G64_FP16)) ||
         (gate == QType::Q8_G32_FP16 && down == QType::Q8_G32_FP16));
    const bool nvfp4 = detail::kNativeNvfp4 && gate == QType::NVFP4 && down == QType::NVFP4 &&
                       shared == QType::NVFP4 && shared_down == QType::NVFP4;
    return geometry && routers && (groupwise || nvfp4);
}

SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down) {
    require(expert_gate_up.size() == 512 && expert_down.size() == 256,
            "SparseMoe native input requires 256 experts");
    require(matrix(router) == std::vector<std::uint64_t>{256, 2048} &&
                matrix(shared_score) == std::vector<std::uint64_t>{1, 2048},
            "SparseMoe router geometry differs");
    for (const auto& input : expert_gate_up) {
        require(matrix(input) == std::vector<std::uint64_t>{512, 2048},
                "SparseMoe gate/up geometry differs");
    }
    for (const auto& input : expert_down) {
        require(matrix(input) == std::vector<std::uint64_t>{2048, 512},
                "SparseMoe down geometry differs");
    }
    require(matrix(shared_gate) == std::vector<std::uint64_t>{512, 2048} &&
                matrix(shared_up) == matrix(shared_gate) &&
                matrix(shared_down) == std::vector<std::uint64_t>{2048, 512},
            "SparseMoe shared expert geometry differs");
    const std::array router_inputs{router, shared_score};
    const auto router_bank        = single(router_inputs);
    const auto gate_up_bank       = single(expert_gate_up);
    const auto down_bank          = single(expert_down);
    const auto shared             = prepare_linear_swiglu_weight(shared_gate, shared_up);
    const auto down               = prepare_linear_weight(shared_down);
    const auto gate_format        = gate_up_bank.weight.qtype;
    const auto down_format        = down_bank.weight.qtype;
    const auto shared_format      = shared.weight.qtype;
    const auto shared_down_format = down.weight.qtype;
    const bool groupwise =
        shared_format == QType::Q8_G32_FP16 && shared_down_format == QType::Q8_G32_FP16 &&
        ((gate_format == QType::Q4_G64_FP16 &&
          (down_format == QType::Q5_G64_FP16 || down_format == QType::Q6_G64_FP16)) ||
         (gate_format == QType::Q8_G32_FP16 && down_format == QType::Q8_G32_FP16));
    // The four expert matrices are one profile rather than two independent choices: the prefill
    // route quantises the activation as well as the weight and its gate/up epilogue emits the
    // encoded intermediate `down` consumes, so a mixture with a groupwise codec has no route
    // behind it at any token count.
    const bool nvfp4 = gate_format == QType::NVFP4 && down_format == QType::NVFP4 &&
                       shared_format == QType::NVFP4 && shared_down_format == QType::NVFP4;
    require(router_bank.weight.qtype == QType::BF16 && (groupwise || nvfp4),
            "SparseMoe native bank formats are unsupported");
    if (nvfp4) {
        // The prefill route of this profile quantises the hidden state to four bits, which is what
        // AllowA4 grants. The groupwise profiles consume it represented and need no permission.
        const auto permits = [](const WeightInput& input) { return allows_a4(input.policy); };
        require(std::all_of(expert_gate_up.begin(), expert_gate_up.end(), permits) &&
                    std::all_of(expert_down.begin(), expert_down.end(), permits) &&
                    permits(shared_gate) && permits(shared_up) && permits(shared_down),
                "SparseMoe NVFP4 requires AllowA4 on every expert input: its prefill route "
                "encodes activations in four bits and has no BF16-activation form "
                "(--activations exact is unavailable for this model)");
    }
    return {router_bank.weight, gate_up_bank.weight, down_bank.weight, shared.weight, down.weight};
}

namespace {

constexpr std::uint64_t kGemma4Experts      = 128;
constexpr std::uint64_t kGemma4Hidden       = 2816;
constexpr std::uint64_t kGemma4GateUpRows   = 1408;
constexpr std::uint64_t kGemma4Intermediate = 704;

// One stacked plane of experts: the inputs are its consecutive row ranges in expert order and
// cover the whole parent, which carries one divisor per expert. The shared-input path (`single`)
// cannot build it, because A4 there requires one activation divisor for the whole bank and Gemma's
// experts each carry their own; the op reads those from its own Tensors instead.
Weight gemma4_expert_plane(std::span<const WeightInput> inputs, std::uint64_t rows,
                           std::uint64_t columns, const char* name) {
    const std::string label = std::string("Gemma4Moe ") + name;
    require(inputs.size() == kGemma4Experts, (label + " requires 128 experts").c_str());
    for (const auto& input : inputs) {
        require(matrix(input) == std::vector<std::uint64_t>{rows, columns},
                (label + " geometry differs").c_str());
        require(allows_a4(input.policy),
                (label + " requires AllowA4: its prefill route encodes activations in four bits")
                    .c_str());
    }
    const auto view = concatenate_rows(inputs);
    require(contiguous(view), (label + " experts must be consecutive rows of one parent").c_str());
    const auto region = contiguous_weight_region(view);
    require(region.parent->geometry.format == QType::NVFP4 &&
                region.parent->geometry.layout == QuantLayout::BlockScaleK16M128x4,
            (label + " must be NVFP4").c_str());
    require(std::isfinite(region.parent->weight_scale_divisor) &&
                region.parent->weight_scale_divisor > 0,
            (label + " parent requires a positive weight divisor").c_str());
    // The native ABI validates this field on every NVFP4 weight; the op never reads it.
    const Weight weight = native_weight(view, 1.0F);
    require(weight.weight_divisor_rows == static_cast<std::int32_t>(rows),
            (label + " parent must carry one weight divisor per expert").c_str());
    return weight;
}

float gemma4_input_divisor(const WeightInput& input, const char* name) {
    require(input.activation_input_divisor.has_value() &&
                std::isfinite(*input.activation_input_divisor) &&
                *input.activation_input_divisor > 0,
            (std::string("Gemma4Moe ") + name + " requires a positive activation divisor").c_str());
    return *input.activation_input_divisor;
}

} // namespace

Gemma4MoeWeights prepare_gemma4_moe_weights(const WeightInput& router, const Tensor& router_scale,
                                            const Tensor& expert_scale,
                                            std::span<const WeightInput> expert_gate_up,
                                            std::span<const WeightInput> expert_down,
                                            const Tensor& gate_up_input_divisors,
                                            const Tensor& down_input_divisors) {
    require(matrix(router) == std::vector<std::uint64_t>{kGemma4Experts, kGemma4Hidden},
            "Gemma4Moe router geometry differs");
    const auto router_weight = prepare_linear_weight(router);
    require(router_weight.weight.qtype == QType::BF16, "Gemma4Moe router must be BF16");
    Gemma4MoeWeights out;
    out.router       = router_weight.weight;
    out.router_scale = router_scale;
    out.expert_scale = expert_scale;
    out.gate_up = gemma4_expert_plane(expert_gate_up, kGemma4GateUpRows, kGemma4Hidden, "gate_up");
    out.down    = gemma4_expert_plane(expert_down, kGemma4Hidden, kGemma4Intermediate, "down");
    out.gate_up_input_divisors = gate_up_input_divisors;
    out.down_input_divisors    = down_input_divisors;
    return out;
}

Gemma4MoeInputDivisors gemma4_moe_input_divisors(std::span<const WeightInput> expert_gate_up,
                                                 std::span<const WeightInput> expert_down) {
    require(expert_gate_up.size() == kGemma4Experts && expert_down.size() == kGemma4Experts,
            "Gemma4Moe requires 128 experts");
    Gemma4MoeInputDivisors out;
    out.gate_up.reserve(kGemma4Experts);
    out.down.reserve(kGemma4Experts);
    for (const auto& input : expert_gate_up) {
        out.gate_up.push_back(gemma4_input_divisor(input, "gate_up"));
    }
    for (const auto& input : expert_down) {
        out.down.push_back(gemma4_input_divisor(input, "down"));
    }
    return out;
}

} // namespace ninferno::ops
