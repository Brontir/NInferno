#include "runtime/contract/sampling.h"

#include <algorithm>

namespace ninferno::runtime {
ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                            SamplingMode mode, const SamplingOverrides& overrides) {
    const SamplingPreset& preset = defaults.for_mode(mode);
    ResolvedSamplingParameters resolved{
        .temperature       = overrides.temperature.value_or(preset.temperature),
        .top_k             = overrides.top_k.value_or(preset.top_k),
        .top_p             = overrides.top_p.value_or(preset.top_p),
        .min_p             = overrides.min_p.value_or(preset.min_p),
        .presence_penalty  = overrides.presence_penalty.value_or(preset.presence_penalty),
        .frequency_penalty = overrides.frequency_penalty.value_or(preset.frequency_penalty),
        .repetition_penalty = overrides.repetition_penalty.value_or(1.0F),
        .seed              = overrides.seed.value_or(0),
        .logit_bias        = overrides.logit_bias,
        .logprobs          = overrides.logprobs ? static_cast<std::int32_t>(std::min<std::uint32_t>(
                                                      *overrides.logprobs, 1U << 30))
                                                : -1,
    };
    // The registered sampling pipeline has an exact top-64 candidate domain. Preserve the
    // existing public meaning of zero (use the target cap), then expose only concrete values to
    // runtimes.
    if (resolved.top_k == 0) { resolved.top_k = kMaximumTopK; }
    check_resolved_sampling(resolved);
    return resolved;
}

} // namespace ninferno::runtime
