#include "runtime/engine/activation_precision.h"

#include "artifact/formats.h"
#include "core/fnv.h"
#include "models/common/weight_use.h"
#include "product/activation_options.h"

#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

namespace ninferno::runtime {

namespace {

ops::LinearPolicy linear_policy(artifact::ActivationPolicy policy) {
    switch (policy) {
    case artifact::ActivationPolicy::A16Only:
        return ops::LinearPolicy::A16Only;
    case artifact::ActivationPolicy::AllowA8:
        return ops::LinearPolicy::AllowA8;
    case artifact::ActivationPolicy::AllowA4:
        return ops::LinearPolicy::AllowA4;
    }
    return ops::LinearPolicy::A16Only;
}

// The stored format and the [N, K] of a parameter's binding: its first part's tensor (a stacked
// expert bank's last two dimensions); N or K 0 when the tensor is not a matrix.
struct Stored {
    std::string format;
    QType qtype    = QType::BF16;
    std::int64_t n = 0;
    std::int64_t k = 0;
    bool known     = false;
};

Stored stored_of(const artifact::Directory& directory, const std::string& parameter) {
    Stored out;
    const auto binding = directory.bindings.find(parameter);
    if (binding == directory.bindings.end() || binding->second.parts.empty()) { return out; }
    const auto* tensor = std::get_if<artifact::TensorObject>(
        &directory.object(binding->second.parts.front().object));
    if (tensor == nullptr) { return out; }
    out.format = tensor->format;
    try {
        out.qtype = artifact::parse_format(tensor->format);
    } catch (const std::exception&) {
        return out; // a format this build does not know: the loader refuses it
    }
    if (tensor->shape.size() >= 2) {
        out.n = static_cast<std::int64_t>(tensor->shape[tensor->shape.size() - 2]);
        out.k = static_cast<std::int64_t>(tensor->shape.back());
    }
    out.known = true;
    return out;
}

// The in-artifact drafter components: their matrices follow the mode as the target's do.
constexpr std::array<std::string_view, 4> kDrafterPrefixes{"mtp/", "dflash/", "dflash2/",
                                                           "proposal/"};

bool drafter_name(std::string_view name) {
    for (const auto prefix : kDrafterPrefixes) {
        if (name.starts_with(prefix)) { return true; }
    }
    return false;
}

enum class Role { Target, Drafter, Other };

// A use belongs to the drafter when its matrix or its input is the drafter's (the Qwen MTP head
// is "text/output_head" read from "mtp/final_hidden"), to the target when its matrix is "text/".
Role role_of(const std::string& parameter, const std::string& input) {
    if (drafter_name(parameter) || drafter_name(input)) { return Role::Drafter; }
    return parameter.starts_with("text/") ? Role::Target : Role::Other;
}

// One matrix use as the mode resolves it.
struct Resolved {
    int bits                 = 16;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    // The policy as the routes read it (what the hash folds): A16Only for a matrix with no
    // narrower route under any policy on this device (BF16 weights), whose kernels it never moves.
    ops::LinearPolicy kernels = ops::LinearPolicy::A16Only;
    // An NVFP4 routed or shared expert under --activations 4: which Op takes it (a fused four-bit
    // bank, or the generic MoE at eight bits) is the load's choice, so the hash keeps 4 apart from
    // Auto on it rather than claim the same kernels.
    bool four_bit_expert = false;
    // Its width is the census's, from the mode the model was loaded under (not this resolution's
    // own when it resolves another mode for a comparison).
    bool from_census = false;
    Role role        = Role::Other;
    std::string format;
    std::string reason;
};

using Key = std::pair<std::string, std::string>;

std::map<Key, Resolved> resolve_uses(const artifact::Directory& directory, ActivationMode mode,
                                     const ops::ActivationDevice& device,
                                     const models::ActivationCensus* census) {
    std::map<Key, Resolved> out;
    const bool forced    = mode == ActivationMode::Bits8 || mode == ActivationMode::Bits4;
    const bool invariant = mode == ActivationMode::Bits16;
    for (const auto& [key, use] : directory.uses) {
        const auto& [parameter, input] = key;
        const Role role                = role_of(parameter, input);
        if (!use.activation_policy || role == Role::Other) { continue; }
        const Stored stored = stored_of(directory, parameter);
        if (!stored.known) { continue; }
        const bool divisor = use.auxiliaries.contains("activation_input_divisor");
        const models::MatrixUse matrix{
            .recorded              = linear_policy(*use.activation_policy),
            .logits_or_embedding   = models::logits_or_embedding(parameter),
            .nvfp4_without_divisor = stored.qtype == QType::NVFP4 && !divisor,
        };
        Resolved r;
        r.role            = role;
        r.format          = stored.format;
        r.policy          = models::effective_policy(matrix, mode);
        r.four_bit_expert = mode == ActivationMode::Bits4 && stored.qtype == QType::NVFP4 &&
                            (parameter.find("/experts/") != std::string::npos ||
                             parameter.find("/shared") != std::string::npos);
        if (census != nullptr) {
            if (const auto it = census->widths.find(parameter); it != census->widths.end()) {
                r.bits        = it->second.bits;
                r.reason      = it->second.reason;
                r.kernels     = r.policy;
                r.from_census = true;
                out.emplace(key, std::move(r));
                continue;
            }
            if (const auto it = census->policies.find(parameter); it != census->policies.end()) {
                r.policy      = it->second;
                r.from_census = true;
            }
        }
        const auto width =
            ops::linear_activation_width(stored.qtype, r.policy, stored.n, stored.k, divisor,
                                         matrix.logits_or_embedding, invariant, device);
        r.bits    = width.bits;
        r.reason  = std::string(width.reason);
        r.kernels = ops::narrowest_activation_width(stored.qtype, ops::LinearPolicy::AllowA4,
                                                    stored.n, stored.k, divisor, device.features)
                                .bits == 16
                        ? ops::LinearPolicy::A16Only
                        : r.policy;
        if (forced && r.policy == ops::LinearPolicy::A16Only) {
            r.reason = "a head or embedding keeps its recorded 16-bit activations";
        } else if (forced && matrix.nvfp4_without_divisor && mode == ActivationMode::Bits4 &&
                   r.bits == 8) {
            r.reason = "no calibrated activation divisor for the four-bit route";
        }
        out.emplace(key, std::move(r));
    }
    return out;
}

std::uint64_t hash_of(const std::map<Key, Resolved>& uses) {
    std::uint64_t h = fnv::kOffset;
    for (const auto& [key, r] : uses) {
        h = fnv::mix(fnv::bytes(key.first, h), 0);
        h = fnv::mix(fnv::bytes(key.second, h), 0);
        h = fnv::mix(h, static_cast<std::uint64_t>(r.kernels));
        h = fnv::mix(h, static_cast<std::uint64_t>(r.bits));
        h = fnv::mix(h, r.four_bit_expert ? 1 : 0);
    }
    return h;
}

std::string matrices(std::uint64_t count) {
    return std::to_string(count) + (count == 1 ? " matrix" : " matrices");
}

ActivationPrecision summarize(const std::map<Key, Resolved>& uses, ActivationMode mode) {
    ActivationPrecision out;
    out.requested       = mode;
    out.policy_hash     = hash_of(uses);
    const int requested = product::activation_mode_bits(mode);
    const bool forced   = mode == ActivationMode::Bits8 || mode == ActivationMode::Bits4;
    // (format, bits, reason) -> count and the first parameter, in name order.
    std::map<std::tuple<std::string, int, std::string>, std::pair<std::uint64_t, std::string>>
        wider;
    for (const auto& [key, r] : uses) {
        if (r.role == Role::Drafter) {
            (r.bits == 4   ? out.drafter_bits4
             : r.bits == 8 ? out.drafter_bits8
                           : out.drafter_bits16) += 1;
            continue;
        }
        (r.bits == 4 ? out.bits4 : r.bits == 8 ? out.bits8 : out.bits16) += 1;
        if (forced && r.bits > requested) {
            auto& group = wider[{r.format, r.bits, r.reason}];
            if (group.first++ == 0) { group.second = key.first; }
        }
    }
    for (const auto& [key, group] : wider) {
        const auto& [format, bits, reason] = key;
        out.kept_wider.push_back(
            matrices(group.first) + " " + format + " at " + std::to_string(bits) +
            "-bit: " + (reason.empty() ? std::string("no narrower route") : reason) + " (e.g. " +
            group.second + ")");
    }
    return out;
}

std::uint64_t taken(const ActivationPrecision& out, ActivationMode mode) {
    return mode == ActivationMode::Bits4   ? out.bits4
           : mode == ActivationMode::Bits8 ? out.bits4 + out.bits8
                                           : 1;
}

// `weights`: whether the weights alone decide it (no device this build runs takes the width on
// them), else the GPU does.
void refuse_untaken(const ActivationPrecision& out, ActivationMode mode, bool weights) {
    const int requested = product::activation_mode_bits(mode);
    if (taken(out, mode) != 0) { return; }
    std::string why;
    for (const auto& line : out.kept_wider) { why += (why.empty() ? "" : "; ") + line; }
    const std::string bits = std::to_string(requested) + "-bit activations";
    throw std::invalid_argument(
        std::string("--activations ") + product::activation_mode_name(mode) + ": " +
        (weights ? "the weights of this model have no matrix that can compute with " + bits
                 : "this GPU, with this build, runs no matrix of this model with " + bits) +
        (why.empty() ? std::string() : " (" + why + ")") +
        (mode == ActivationMode::Bits4 && out.bits8 > 0
             ? "; --activations 8 runs eight bits on " + matrices(out.bits8)
             : std::string()) +
        "; --activations auto runs what the artifact recorded");
}

} // namespace

ActivationPrecision resolve_activation_precision(const artifact::Directory& directory,
                                                 ActivationMode mode,
                                                 const ops::ActivationDevice& device,
                                                 const models::ActivationCensus* census,
                                                 bool refuse) {
    const auto uses         = resolve_uses(directory, mode, device, census);
    ActivationPrecision out = summarize(uses, mode);
    if (refuse && (mode == ActivationMode::Bits8 || mode == ActivationMode::Bits4) &&
        taken(out, mode) == 0) {
        // The same weights on an RTX 50 of an sm_120a build (FP8 and FP4 MMA, int8 prefill
        // paying): whether any card could take the width at all.
        ops::ActivationDevice widest = device;
        widest.features              = device_features_for(12, 0);
        widest.features.mma_mxf4nvf4 = true;
        widest.features.mma_slow     = false;
        widest.fp32acc_full_rate     = false;
        const auto best = summarize(resolve_uses(directory, mode, widest, census), mode);
        refuse_untaken(out, mode, taken(best, mode) == 0);
    }
    if (!refuse && (mode == ActivationMode::Bits8 || mode == ActivationMode::Bits4) &&
        taken(out, mode) == 0) {
        // Admitted before the load (counting each expert as the linear path would take it), but
        // the expert Ops the model loaded have no such route: say so rather than stop a model
        // that was promised to start.
        out.notes.push_back(std::string("--activations ") + product::activation_mode_name(mode) +
                            ": no matrix of this model computes with " +
                            std::to_string(product::activation_mode_bits(mode)) +
                            "-bit activations after the load (its expert Ops have no such route)");
    }
    if (mode == ActivationMode::Auto) {
        // Whether 8 computes some matrices narrower than Auto does here (and Auto some narrower
        // than 8): an NVFP4 checkpoint recorded for four bits without calibrated divisors keeps
        // BF16 under Auto where the build has the W4A4 kernels, and 8 takes W4A8 there.
        const auto eight     = resolve_uses(directory, ActivationMode::Bits8, device, census);
        std::uint64_t faster = 0, slower = 0;
        std::string example;
        for (const auto& [key, r] : uses) {
            const auto other = eight.find(key);
            // A census width is the loaded mode's for both resolutions: not a comparison.
            if (other == eight.end() || r.role != Role::Target || r.from_census ||
                other->second.from_census) {
                continue;
            }
            if (other->second.bits < r.bits) {
                if (faster++ == 0) { example = key.first + " (" + r.format + ")"; }
            } else if (other->second.bits > r.bits) {
                ++slower;
            }
        }
        if (faster > 0) {
            out.faster_than_auto = "8";
            out.notes.push_back("--activations 8 computes " + matrices(faster) +
                                " on eight-bit activations that auto keeps at 16-bit here (e.g. " +
                                example + "), so auto is slower than 8 on this GPU" +
                                (slower > 0 ? "; auto runs " + matrices(slower) + " narrower than 8"
                                            : std::string()));
        }
    } else if (mode == ActivationMode::Bits8 || mode == ActivationMode::Bits4) {
        const auto automatic = resolve_uses(directory, ActivationMode::Auto, device, census);
        out.same_as_auto     = hash_of(automatic) == out.policy_hash;
        if (out.same_as_auto) {
            out.notes.push_back(std::string("--activations ") +
                                product::activation_mode_name(mode) +
                                " resolves to the same kernels as auto on this model and GPU");
        }
    }
    return out;
}

std::vector<ModePreflight> preflight_activation_modes(const artifact::Directory& directory,
                                                      const ops::ActivationDevice& device) {
    std::vector<ModePreflight> out;
    for (const auto mode : {ActivationMode::Auto, ActivationMode::Bits16, ActivationMode::Bits8,
                            ActivationMode::Bits4}) {
        ModePreflight row;
        row.mode = mode;
        try {
            row.precision = resolve_activation_precision(directory, mode, device);
        } catch (const std::invalid_argument& error) {
            row.refused = true;
            row.refusal = error.what();
        }
        out.push_back(std::move(row));
    }
    return out;
}

bool has_routed_experts(const artifact::Directory& directory) {
    for (const auto& [key, use] : directory.uses) {
        if (key.first.find("/experts/") != std::string::npos) { return true; }
    }
    return false;
}

} // namespace ninferno::runtime
