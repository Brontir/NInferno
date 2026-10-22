#include "runtime/engine/served_bytes.h"

#include "artifact/reader.h"
#include "core/device_class.h"
#include "models/common/weight_use.h"
#include "ops/generic/generic_gemm.h"
#include "runtime/engine/family_registry.h"

#include <exception>

namespace ninferno::runtime {

KvCacheStorage resolved_kv_cache(const EngineOptions& options) {
    if (!options.kv_cache_automatic) { return options.kv_cache; }
    // The device's compute capability decides FP8 rows (automatic_kv_cache); a host without the
    // device keeps the family's storage, which the load then reports or refuses.
    int compute_capability = 1000;
    try {
        compute_capability = device_class(options.device).compute_capability();
    } catch (const std::exception&) {}
    return automatic_kv_cache(artifact_model_type(options.artifact_path), compute_capability);
}

std::string_view activation_bits(ActivationMode mode, bool allow_a4_recorded,
                                 bool w4a4_device) noexcept {
    if (mode == ActivationMode::Exact) { return "exact"; }
    const bool a4 =
        allow_a4_recorded &&
        models::effective_policy(ops::LinearPolicy::AllowA4, mode) == ops::LinearPolicy::AllowA4 &&
        w4a4_device;
    return a4 ? "fast-a4" : "fast-a8";
}

std::string served_activations(const EngineOptions& options) {
    bool recorded = false;
    bool device   = false;
    try {
        const artifact::Reader reader(options.artifact_path);
        for (const auto& [key, use] : reader.directory().uses) {
            // W4A4 needs an AllowA4 use with its calibrated activation divisor (route_admits).
            if (use.activation_policy == artifact::ActivationPolicy::AllowA4 &&
                use.auxiliaries.contains("activation_input_divisor")) {
                recorded = true;
                break;
            }
        }
    } catch (const std::exception&) {}
    try {
        namespace generic = ops::detail::generic;
        // The device's features and this build's W4A4 kernels (none in a build without them:
        // an sm_120 build without the family-specific "120a" target, say).
        device = generic::w4a4_schedule_count() > 0 &&
                 has_features(
                     device_class(options.device).features,
                     generic::route_requires(generic::Route{.family = generic::Family::W4a4Mma}));
    } catch (const std::exception&) {}
    return std::string(activation_bits(options.activations, recorded, device));
}

} // namespace ninferno::runtime
